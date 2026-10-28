#!/usr/bin/env python3
"""
Decoder v2 for the stent -> maXTouch BER runs, applied offline to the saved raw data (capture.npz + meta.json).
The original decoder (maxtouch_stent_rx.py analyse_ber -> result.json / decoded.json) stays the baseline and is not
touched; v2 writes result_v2.json and decoded_v2.json next to it.

   python decode_v2.py data_maxtouch/stent_ber_c156_20bps_<time>     # one run
   python decode_v2.py all                                           # every stent_ber_* run + comparison table

Rules (fixed 2026-10-22 before the 34/45 bps and 159.6/163.6 kHz runs, applied unchanged to every run):
  1. Same sample grid (maxtouch_stent_rx.regular) and node ranking as the baseline: energy increase after START TX
     vs the lead (decided before decoding). The top 3 nodes are used.
  2. One feature for every node: logenergy = log10(1 + (x[n]-x[n-1])^2 / noise). The carrier shows as huge zero-mean
     random-sign swings, so an energy feature; the baseline picked a feature per node after decoding ('auto') and
     used absdev for the combined decision.
  3. Combined decision = per sample the MAX over the 3 nodes of z-scored logenergy. The fades (a 1 read as 0) hit
     Y2/Y3/Y4 at the same bits in the 20 bps run, so averaging cannot hide them; the max keeps the strongest node.
  4. Single node: the top-ranked node (chosen before decoding = the honest single-node number) and, for reference
     only, the best of the 3 after decoding (optimistic).
  5. Known timing: the transmitter's known first-bit time plus ONE offset for the whole run, chosen from the
     preamble+SFD (the known header every frame carries) within +-3 samples - samples arrive a few ms after their
     scan, so the raw host time put the bit grid slightly off (known timing scored worse than blind in the baseline).
  6. Accounting: BER over decoded (valid) frames as in the baseline, plus frames decoded / sent and a conservative
     BER that counts every undecoded frame's payload as 50 % errors.
  7. Health: share of 2 s windows during transmission in which the top node's energy is >= 10 dB over the lead.
     Below 95 % the run is marked 'signal lost' (the transmitter or its wiring failed) and its BER is not a link
     measurement.
Rules 2 and 3 were chosen after looking at the one good 20 bps run (stent_ber_c156_20bps_20261022_180358), so that
run flatters v2 a little; later runs are the honest test.
"""
from __future__ import annotations

import contextlib
import csv
import glob
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import maxtouch_stent_rx as R                   # also loads cap_ook_rx (RX) with stent_frames as the TX module

RX, TX = R.RX, R.TX
N_NODES = 3
FEATURE = "logenergy"
HEALTH_WIN_S, HEALTH_DB, HEALTH_MIN = 2.0, 10.0, 0.95
SHIFT_SAMPLES = 3.0


def frames_summary(frames, n_sent, payload_bits):
    valid = [d for d in frames if d["valid"]]
    bits = sum(d["bits"] for d in valid)
    errs = sum(d["errors"] for d in valid)
    missing = max(n_sent - len(valid), 0) * payload_bits if n_sent else 0
    return dict(frames_valid=len(valid), frames_sent=n_sent, bits=bits, errors=errs,
                ber=errs / bits if bits else None, ber_upper95=(3.0 / bits if bits and errs == 0 else None),
                ber_conservative=((errs + 0.5 * missing) / (bits + missing) if bits + missing else None))


def blind(f, k, plan, truth, n_sent):
    RX.set_guard(k)
    fr = RX.find_frames(f, k, plan["frame_bits"], truth)["frames"]
    return frames_summary(fr, n_sent, len(truth)), fr


def known(f, k, plan, truth, s0g, n_sent):
    """known start + one header-chosen offset for the whole run"""
    RX.set_guard(k)
    P = plan["frame_bits"] * k
    starts = [s0g + m * P for m in range(n_sent) if s0g + m * P + (len(RX.HDR) + len(truth)) * k < len(f)]
    if not starts:
        return frames_summary([], n_sent, len(truth)), [], 0.0
    shifts = np.arange(-SHIFT_SAMPLES, SHIFT_SAMPLES + 1e-9, 0.25)
    score = [np.sum(RX.header_corr(f, k, np.array(starts) + d)) for d in shifts]
    d0 = float(shifts[int(np.argmax(np.abs(score)))])
    fr = []
    for m, s in enumerate(starts):
        x = RX.decode_frame(f, k, s + d0, truth)
        x.update(m=m, s0=float(s + d0), valid=True)            # timing is given: every sent frame counts
        fr.append(x)
    return frames_summary(fr, n_sent, len(truth)), fr, d0


def health(G, T, t0, meta, node, total_s):
    e = np.concatenate([[0.0], np.diff(G[:, node]) ** 2])
    tt = t0 + np.arange(len(G)) * T
    lead = e[tt < meta["t_prompt"]].mean() + 1e-9
    w0, ok, n = meta["t_first"], 0, 0
    while w0 + HEALTH_WIN_S <= meta["t_first"] + total_s:
        m = (tt >= w0) & (tt < w0 + HEALTH_WIN_S)
        if m.any():
            n += 1
            ok += 10 * np.log10(e[m].mean() / lead + 1e-12) >= HEALTH_DB
        w0 += HEALTH_WIN_S
    frac = ok / n if n else 0.0
    return dict(windows=n, signal_fraction=frac, signal_lost=bool(frac < HEALTH_MIN))


def decode_run(d):
    meta = json.load(open(os.path.join(d, "meta.json")))
    if meta.get("mode") != "ber":
        return None
    z = np.load(os.path.join(d, "capture.npz"))
    with contextlib.redirect_stdout(io.StringIO()):
        G, T, t0, jit, _ = R.regular(z["t"], z["X"])
    a = R.argparse.Namespace(**meta["args"])
    link = R.link_from(a, False)
    _, plan = TX.build(link)
    _, truth = TX.frame_bits(link.payload, link.prbs)
    k = (1.0 / plan["bps"]) / T
    names = R.node_names(meta)
    n_sent = meta.get("tx_frames", 0) or 0
    score, _ = R.node_scores(G, T, t0, meta.get("t_prompt"))
    top = [int(i) for i in np.argsort(-score)[:N_NODES]]
    feats = {i: RX.features(G[:, i])[FEATURE] for i in top}
    single = []
    for i in top:
        s, _ = blind(feats[i], k, plan, truth, n_sent)
        single.append(dict(node=names[i], score_db=float(score[i]), **s))
    comb_f = np.max([R.zscore(feats[i]) for i in top], axis=0)
    comb, comb_frames = blind(comb_f, k, plan, truth, n_sent)
    res = dict(decoder="v2", rules=f"top {N_NODES} nodes by energy increase, {FEATURE}, max of z, header-aligned known timing",
               run=os.path.basename(d), bps=plan["bps"], carrier_hz=a.carrier, samples_per_bit=k,
               scan_period_ms=T * 1e3, nodes=[names[i] for i in top],
               single_top_ranked=single[0], single_best_of_3_after_decoding=min(single, key=lambda s: (s["ber"] is None, s["ber"] or 0)),
               single_all=single, combined_blind=comb)
    kn_frames = []
    if meta.get("t_first") and n_sent:
        kn, kn_frames, d0 = known(comb_f, k, plan, truth, (meta["t_first"] - t0) / T, n_sent)
        res["combined_known_timing"] = dict(kn, header_offset_samples=d0)
        res["health"] = health(G, T, t0, meta, top[0], n_sent * plan["frame_s"])
    json.dump(res, open(os.path.join(d, "result_v2.json"), "w"), indent=1)
    keep = ("m", "s0", "valid", "bits", "errors", "ber", "q", "thr", "inverted", "got", "norm")
    pack = lambda frs: [{kk: v for kk, v in fr.items() if kk in keep} for fr in frs]
    json.dump(dict(decoder="v2", sent_payload=[int(b) for b in truth], combined_nodes=res["nodes"],
                   blind_frames=pack(comb_frames), known_timing_frames=pack(kn_frames)),
              open(os.path.join(d, "decoded_v2.json"), "w"))
    return res


def fmt(s):
    if not s or s.get("bits") is None:
        return "-"
    if not s["bits"]:
        return f"no frame (0/{s.get('frames_sent', 0)})"
    return f"{s['errors']}/{s['bits']} = {s['ber']:.2%} ({s['frames_valid']}/{s['frames_sent']} fr)"


def report(res):
    h = res.get("health", {})
    print(f"\n[v2] {res['run']}: {res['bps']:g} bps, {res['carrier_hz'] or 0:g} Hz, {res['samples_per_bit']:.1f} samples/bit, "
          f"nodes {', '.join(res['nodes'])}" + (f", signal present {h['signal_fraction']:.0%}" if h else "")
          + ("   ** SIGNAL LOST - not a link measurement **" if h.get("signal_lost") else ""))
    print(f"  single, top-ranked {res['single_top_ranked']['node']}:  {fmt(res['single_top_ranked'])}")
    b = res["single_best_of_3_after_decoding"]
    print(f"  single, best of 3 (after decoding, optimistic) {b['node']}:  {fmt(b)}")
    print(f"  combined blind:        {fmt(res['combined_blind'])}   conservative {res['combined_blind']['ber_conservative'] or 0:.2%}")
    kt = res.get("combined_known_timing")
    if kt:
        print(f"  combined known timing: {fmt(kt)}   (header offset {kt['header_offset_samples']:+.2f} samples)")


def baseline_row(d):
    p = os.path.join(d, "result.json")
    if not os.path.exists(p):
        return {}
    r = json.load(open(p))
    if r.get("mode") != "ber":
        return {}
    c, g = r["combined"], r.get("known_timing") or {}
    best = min((p for p in r["per_node"] if p["bits"]), key=lambda p: p["errors"] / p["bits"], default=None)
    return dict(base_comb=f"{c['errors']}/{c['bits']}" if c["bits"] else "no frame",
                base_comb_ber=c["ber"], base_known=f"{g.get('errors')}/{g.get('bits')}" if g.get("bits") else "-",
                base_known_ber=g.get("ber"),
                base_best_node=f"{best['node']} {best['errors']}/{best['bits']}" if best else "-")


def compare_all():
    rows = []
    for d in sorted(glob.glob(os.path.join(HERE, "data_maxtouch", "stent_ber_*"))):
        if not os.path.exists(os.path.join(d, "capture.npz")):
            continue
        res = decode_run(d)
        if res is None:
            continue
        report(res)
        b = baseline_row(d)
        kt = res.get("combined_known_timing") or {}
        rows.append(dict(run=res["run"], bps=res["bps"], carrier_hz=res["carrier_hz"],
                         samples_per_bit=round(res["samples_per_bit"], 2),
                         signal_present=(res.get("health") or {}).get("signal_fraction"),
                         signal_lost=(res.get("health") or {}).get("signal_lost"),
                         **b,
                         v2_single_top=fmt(res["single_top_ranked"]), v2_single_top_ber=res["single_top_ranked"]["ber"],
                         v2_single_best=fmt(res["single_best_of_3_after_decoding"]),
                         v2_comb_blind=fmt(res["combined_blind"]), v2_comb_blind_ber=res["combined_blind"]["ber"],
                         v2_comb_blind_conservative=res["combined_blind"]["ber_conservative"],
                         v2_comb_known=fmt(kt) if kt else "-", v2_comb_known_ber=kt.get("ber")))
    out = os.path.join(HERE, "data_maxtouch", "ber_compare_v2.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"\n{'run':48s} {'sig':>5} | {'BASELINE comb blind':>20} {'known':>11} | {'v2 top node':>26} {'v2 comb blind':>26} {'v2 comb known':>26}")
    for r in rows:
        sig = "LOST" if r["signal_lost"] else f"{r['signal_present']:.0%}" if r["signal_present"] is not None else "-"
        print(f"{r['run'][:48]:48s} {sig:>5} | {r.get('base_comb', '-'):>20} {r.get('base_known', '-'):>11} | "
              f"{r['v2_single_top']:>26} {r['v2_comb_blind']:>26} {r['v2_comb_known']:>26}")
    print(f"\ntable: {os.path.relpath(out, HERE)}  (baseline columns = result.json, v2 = result_v2.json)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    if sys.argv[1] == "all":
        compare_all()
    else:
        report(decode_run(sys.argv[1]))
