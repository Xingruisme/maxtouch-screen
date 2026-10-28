#!/usr/bin/env python3
"""
Independent audit of the frozen v4 results for the three 155.7 kHz, 80 bps validation runs. Read-only: it opens
meta.json, capture.npz, result_v4.json and decoded_v4.json and writes only audit_results.json next to itself.

Independent of the decoder:
  * the transmitted payload is regenerated here from its specification (PRBS7, x^7 + x^6 + 1, Fibonacci LFSR,
    seed all ones, 127 bits) and the frame (16-bit 1010 preamble, 11-bit Barker SFD, payload, 16 zero gap bits),
    without stent_frames.py; the transmitter's schedule checksum is recomputed from these bits;
  * bit errors are recounted from the decoded payload bits saved in decoded_v4.json;
  * every accepted frame is matched in time to exactly one transmitted frame;
  * the edge guard is re-evaluated per payload bit from the raw sample timestamps;
  * a decoder-free raw check: per payload bit, the largest |delta - lead median| of nodes Y2-Y4 in the bit.
Only bit_index-style arithmetic and the frame timing (tau, clock offset) come from the decoder's saved output.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.dirname(DATA)
RUNS = ["stent_ber_val4_80bps_20261022_192702", "stent_ber_val4b_80bps_20261022_195008", "stent_ber_val4b_80bps_20261022_195109"]
LOGGED_CHECKSUM = {"stent_ber_val4b_80bps_20261022_195008": 3089373205,   # printed by the transmitter upload at run time
                   "stent_ber_val4b_80bps_20261022_195109": 3089373205}   # (terminal log); 192702's line was not captured
GUARD_S, R_LOCAL, BER_INVALID = 0.002, 0.3, 0.35


def prbs7_lfsr():
    s, out = [1] * 7, []
    for _ in range(127):
        out.append(s[6])
        s = [s[6] ^ s[5]] + s[:6]
    return np.array(out)


def frame(payload):
    return np.concatenate([[1, 0] * 8, [1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0], payload, np.zeros(16, int)]).astype(int)


def tx_checksum(bits, bps, carrier, frames):
    sys.path.insert(0, ROOT)
    import stent_tx as STX                      # transmitter-side code: word and checksum arithmetic only
    allbits = np.tile(bits, frames)
    bit_us = int(round(1e6 / bps))
    segs, cur = [], None
    for b in allbits:
        hz = carrier if b else 0.0
        if segs and segs[-1][0] == hz:
            segs[-1][1] += bit_us
        else:
            segs.append([hz, bit_us])
    return STX.checksum(STX.to_schedule([tuple(s) for s in segs]))


def main():
    ref = prbs7_lfsr()
    out = {"prbs7_ones": int(ref.sum()), "prbs7_length": len(ref)}
    report = []
    for run in RUNS:
        d = os.path.join(DATA, run)
        meta = json.load(open(os.path.join(d, "meta.json")))
        res = json.load(open(os.path.join(d, "result_v4.json")))
        dec = json.load(open(os.path.join(d, "decoded_v4.json")))
        z = np.load(os.path.join(d, "capture.npz")); t = z["t"].astype(float); X = z["X"].astype(float)
        a = meta["args"]; bps = float(a["bps"]); Tb = 1 / bps; sent = int(meta["tx_frames"])
        saved_pl = np.array(dec["sent_payload"])
        shifts = [k for k in range(127) if np.array_equal(np.roll(ref, k), saved_pl)]
        payload = np.roll(ref, shifts[0]) if shifts else None
        fb = frame(payload); nb = len(fb); P = nb * Tb
        cs = tx_checksum(fb, bps, float(a["carrier"]), sent)
        eps = res["combined_blind"]["clock_ppm"] * 1e-6
        frames = dec["blind_frames"]
        acc = [f for f in frames if f["valid"]]
        # 1. recount errors from the saved decoded bits vs the independently generated payload
        per = [int(np.sum(np.array(f["got"]) != payload)) for f in acc]
        bits_cmp = sum(len(f["got"]) for f in acc)
        # 2. acceptance: which candidate frames were rejected, and by which criterion
        delay = res["combined_known_timing"]["delay_ms"] / 1e-3 * 1e-6 if False else res["combined_known_timing"]["delay_ms"] / 1e3
        def tx_index(s0):
            k = (s0 - (meta["t_first"] + delay)) / (P * (1 + eps))
            return int(round(k)), (k - round(k)) * P * (1 + eps) * 1e3
        rejected = []
        for f in frames:
            if f["valid"]:
                continue
            k, resid = tx_index(f["s0"])
            rejected.append(dict(m=f["m"], tx_frame=k, inside_transmission=bool(0 <= k < sent), r=round(f["r"], 3),
                                 ber=round(f["ber"], 3), rejected_by_header=bool(f["r"] < R_LOCAL),
                                 rejected_only_by_payload_ber=bool(f["r"] >= R_LOCAL and f["ber"] >= BER_INVALID)))
        # 3. every accepted frame <-> exactly one transmitted frame
        match = [tx_index(f["s0"]) for f in acc]
        ks = [k for k, _ in match]
        # 4. edge guard per payload bit, from raw timestamps and the decoder's frame timing
        tau = acc[0]["s0"] - acc[0]["m"] * nb * Tb * (1 + eps)
        u = (t - tau) / (Tb * (1 + eps)); mf = np.floor(u / nb).astype(int); jb = u - mf * nb; j = np.floor(jb).astype(int); pos = jb - j
        g = GUARD_S / (Tb * (1 + eps)); inner = (pos > g) & (pos < 1 - g)
        nodes = [int(n[1:]) for n in res["nodes"]]
        pre = t < meta["t_prompt"]
        dev = np.max([np.abs(X[:, i] - np.median(X[pre, i])) for i in nodes], axis=0)
        sig = max(np.median([1.4826 * np.median(np.abs(X[pre, i] - np.median(X[pre, i]))) for i in nodes]), 1.0)
        cats = dict(inner=0, fallback_edge_only=0, no_sample=0)
        err_by_cat = dict(inner=0, fallback_edge_only=0)
        raw1, raw0 = [], []
        for f, e in zip(acc, per):
            got = np.array(f["got"])
            for b in range(127):
                w = (mf == f["m"]) & (j == 27 + b)
                if not w.any():
                    cats["no_sample"] += 1; continue
                c = "inner" if (w & inner).any() else "fallback_edge_only"
                cats[c] += 1
                err_by_cat[c] += int(got[b] != payload[b])
                val = dev[w & inner].max() if (w & inner).any() else dev[w].max()
                (raw1 if payload[b] else raw0).append(val / sig)
        raw1, raw0 = np.array(raw1), np.array(raw0)
        # decision margin from the saved normalised decision values: > 0 = correct side, ~0.5 = typical level
        nrm = np.concatenate([np.array(f["norm"]) for f in acc]); snt = np.tile(payload, len(acc))
        marg = np.where(snt == 1, nrm, -nrm)
        margins = dict(min=float(marg.min()), median=float(np.median(marg)),
                       **{f"below_{x:g}": int((marg < x).sum()) for x in (0.0, 0.05, 0.1, 0.25)})
        r_entry = dict(run=run, role=res["role"], rules_sha=res["rules_sha"], carrier_hz=float(a["carrier"]), bps=bps,
                       payload_shift_vs_lfsr=shifts, payload_matches_independent_prbs7=bool(shifts),
                       transmitter_checksum_recomputed=int(cs), transmitter_checksum_logged=LOGGED_CHECKSUM.get(run),
                       frames_transmitted=sent, candidate_frames_in_recording=len(frames), frames_accepted=len(acc),
                       accepted_frames_matched_to_tx=sorted(ks), accepted_tx_frames_unique=len(set(ks)) == len(ks),
                       all_tx_frames_recovered=sorted(ks) == list(range(sent)),
                       match_residual_ms_max=float(max(abs(r) for _, r in match)),
                       payload_bits_compared=bits_cmp, errors_recounted=int(sum(per)),
                       errors_in_result_v4=res["combined_blind"]["errors"], bits_in_result_v4=res["combined_blind"]["bits"],
                       rejected_candidates=rejected,
                       rejected_inside_transmission=[x for x in rejected if x["inside_transmission"]],
                       rejected_only_by_payload_ber=[x for x in rejected if x["rejected_only_by_payload_ber"]],
                       edge_guard_bits=cats, errors_by_guard_category=err_by_cat, decision_margin=margins,
                       raw_check=dict(lead_sigma_counts=float(sig), sent1_min_dev_over_sigma=float(raw1.min()),
                                      sent0_max_dev_over_sigma=float(raw0.max()), sent1_bits=len(raw1), sent0_bits=len(raw0),
                                      sent1_below_10sigma=int((raw1 < 10).sum()), sent0_above_10sigma=int((raw0 > 10).sum())))
        report.append(r_entry)
    out["runs"] = report
    tot_b = sum(r["payload_bits_compared"] for r in report); tot_e = sum(r["errors_recounted"] for r in report)
    out["pooled"] = dict(payload_bits=tot_b, errors=tot_e, upper95_one_sided=1 - 0.05 ** (1 / tot_b) if tot_e == 0 else None)
    json.dump(out, open(os.path.join(HERE, "audit_results.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
