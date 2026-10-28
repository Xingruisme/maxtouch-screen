#!/usr/bin/env python3
"""
Step 2 of the audit: independent scoring. The ONLY place in the blind path where the payload exists.

Payload: regenerated here from the recurrence o[n+7] = o[n] XOR o[n+1], o[0..6] = 1 (= Fibonacci LFSR x^7 + x^6 + 1,
seed all ones, output = last stage), cyclically shifted by 120 (the documented start offset). Cross-checked against
a plain LFSR and against the payload the frozen decoder saved, but not derived from either.

Frame matching (no decoder output is trusted for this): a decoded frame starting at s0 is assigned to transmitted
frame k = round((s0 - t_first) / P'), P' = frame period on the decoder's clock; residual = s0 - t_first - k P'.
  true frame       : 0 <= k < sent and the residual within 0..25 ms (the physical delay; measured 11-12 ms)
  misaligned       : 0 <= k < sent but residual outside that window (a frame decoded off the bit grid)
  false positive   : k outside the transmission (accepted where nothing was sent)
  duplicate        : a second accepted frame on the same k
BER = errors / (127 x true frames). Missing frames are counted separately, never as zero errors; the conservative BER
counts every missing frame as 50 % errors and every false-positive or misaligned frame with its scored errors.
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.dirname(HERE)
DELAY_WIN_S = (0.0, 0.025)


def prbs7_recurrence():
    o = [1] * 7
    while len(o) < 127:
        n = len(o) - 7
        o.append(o[n] ^ o[n + 1])
    return np.array(o, int)


def prbs7_lfsr():
    s, out = [1] * 7, []
    for _ in range(127):
        out.append(s[6]); s = [s[6] ^ s[5]] + s[:6]
    return np.array(out, int)


def payload():
    p = prbs7_recurrence()
    assert np.array_equal(p, prbs7_lfsr()), "PRBS formulations disagree"
    assert p.sum() == 64 and all(np.roll(p, k).tolist() != p.tolist() for k in range(1, 127)), "not an m-sequence"
    return np.roll(p, 120)


def match(frames, t_first, sent, Tb, eps):
    P = 170 * Tb * (1 + eps)
    rows, seen = [], set()
    for f in frames:
        if not f["valid"]:
            continue
        k = int(round((f["s0"] - t_first) / P)); res = f["s0"] - t_first - k * P
        if not 0 <= k < sent:
            kind = "false_positive"
        elif not DELAY_WIN_S[0] <= res <= DELAY_WIN_S[1]:
            kind = "misaligned"
        elif k in seen:
            kind = "duplicate"
        else:
            kind = "true"; seen.add(k)
        rows.append(dict(m=f["m"], k=k, residual_ms=res * 1e3, kind=kind, r=f.get("r")))
    return rows


def score(frames, t_first, sent, Tb, eps, pl):
    rows = match(frames, t_first, sent, Tb, eps)
    acc = [f for f in frames if f["valid"]]
    for row, f in zip(rows, acc):
        g = np.array(f["got"], int)
        row["errors"] = int(np.sum(g != pl)); row["bits"] = len(g)
        row["err_pos"] = np.where(g != pl)[0].tolist()
    true = [r for r in rows if r["kind"] == "true"]
    bits = sum(r["bits"] for r in true); errs = sum(r["errors"] for r in true)
    other = [r for r in rows if r["kind"] != "true"]
    missing = sent - len(true)
    cons_e = errs + sum(r["errors"] for r in other) + 0.5 * 127 * missing
    cons_b = bits + sum(r["bits"] for r in other) + 127 * missing
    return dict(frames_sent=sent, candidates=len(frames), accepted=len(acc), true_frames=len(true),
                missing_frames=missing, false_positive=sum(r["kind"] == "false_positive" for r in rows),
                misaligned=sum(r["kind"] == "misaligned" for r in rows),
                duplicate=sum(r["kind"] == "duplicate" for r in rows),
                payload_bits=bits, errors=errs, ber=errs / bits if bits else None,
                upper95=(1 - 0.05 ** (1 / bits)) if bits and errs == 0 else None,
                ber_conservative=cons_e / cons_b if cons_b else None,
                residual_ms=[round(r["residual_ms"], 3) for r in true], rows=rows)


def run_meta(run):
    m = json.load(open(os.path.join(DATA, run, "meta.json")))
    return m["t_first"], int(m["tx_frames"]), 1.0 / float(m["args"]["bps"])


def main():
    pl = payload()
    out = dict(payload_definition="PRBS7 o[n+7]=o[n]^o[n+1], seed 1111111, cyclic shift 120", runs={})
    files = sorted(f for f in os.listdir(os.path.join(HERE, "blind_out")) if f.endswith(".json"))
    for fn in files:
        b = json.load(open(os.path.join(HERE, "blind_out", fn)))
        run, mode = b["run"], b["mode"]
        t_first, sent, Tb = run_meta(run)
        e = out["runs"].setdefault(run, {})
        e[mode] = score(b["frames"], t_first, sent, Tb, b["eps_ppm"] * 1e-6, pl)
        if "frozen_v4" not in e:                      # the frozen decoder's saved output, scored the same way
            d = json.load(open(os.path.join(DATA, run, "decoded_v4.json")))
            r4 = json.load(open(os.path.join(DATA, run, "result_v4.json")))
            e["saved_payload_equals_regenerated"] = bool(np.array_equal(np.array(d["sent_payload"]), pl))
            e["frozen_v4"] = score(d["blind_frames"], t_first, sent, Tb, r4["combined_blind"]["clock_ppm"] * 1e-6, pl)
            e["frozen_v4_reported"] = dict(errors=r4["combined_blind"]["errors"], bits=r4["combined_blind"]["bits"],
                                           frames_valid=r4["combined_blind"]["frames_valid"])
            # same rows as v4 but with the payload-BER half of the acceptance rule removed (header r >= 0.3 only)
            hdr_only = [dict(fr, valid=bool(fr["r"] >= 0.3)) for fr in d["blind_frames"]]
            e["frozen_v4_header_only_acceptance"] = score(hdr_only, t_first, sent, Tb,
                                                          r4["combined_blind"]["clock_ppm"] * 1e-6, pl)
            e["frozen_v4_frames_rejected_only_by_payload_rule"] = [
                dict(m=fr["m"], r=round(fr["r"], 3), ber=round(fr["ber"], 3)) for fr in d["blind_frames"]
                if fr["r"] >= 0.3 and not fr["valid"]]
            e["frozen_v4_rejected_candidates"] = [dict(m=fr["m"], r=round(fr["r"], 3), ber=round(fr["ber"], 3))
                                                  for fr in d["blind_frames"] if not fr["valid"]]
    # bit-for-bit agreement between the blind re-implementation and the frozen decoder's saved bits
    for run, e in out["runs"].items():
        d = json.load(open(os.path.join(DATA, run, "decoded_v4.json")))
        v4 = {fr["m"]: fr for fr in d["blind_frames"]}
        for mode in ("v4rules_blind", "strict_blind"):
            b = json.load(open(os.path.join(HERE, "blind_out", f"{run}.{mode}.json")))
            same = diff = 0; dn = []
            for fr in b["frames"]:
                if fr["m"] in v4:
                    g1, g2 = np.array(fr["got"]), np.array(v4[fr["m"]]["got"])
                    same += int(np.sum(g1 == g2)); diff += int(np.sum(g1 != g2))
                    dn.append(abs(np.array(fr["norm"]) - np.array(v4[fr["m"]]["norm"])).max())
            e[mode]["bits_vs_frozen_v4"] = dict(same=same, different=diff, max_abs_norm_diff=float(max(dn)) if dn else None)
    json.dump(out, open(os.path.join(HERE, "score_results.json"), "w"), indent=1)
    print(f"{'run':42s} {'decoder':32s} {'acc':>4} {'true':>4} {'miss':>4} {'FP':>3} {'mis':>3} {'errors/bits':>13} {'cons BER':>9}  vs v4 bits")
    for run, e in out["runs"].items():
        for k in ("frozen_v4", "frozen_v4_header_only_acceptance", "v4rules_blind", "strict_blind"):
            s = e[k]
            ag = s.get("bits_vs_frozen_v4")
            print(f"{run[10:52]:42s} {k:32s} {s['accepted']:4d} {s['true_frames']:4d} {s['missing_frames']:4d} "
                  f"{s['false_positive']:3d} {s['misaligned']:3d} {s['errors']:6d}/{s['payload_bits']:<6d} "
                  f"{s['ber_conservative']:9.5f}  " + (f"{ag['same']} same / {ag['different']} diff" if ag else ""))
        print(f"{'':42s} reported in result_v4.json: {e['frozen_v4_reported']}; payload saved == regenerated: "
              f"{e['saved_payload_equals_regenerated']}; rejected only by payload rule: "
              f"{e['frozen_v4_frames_rejected_only_by_payload_rule']}")


if __name__ == "__main__":
    main()
