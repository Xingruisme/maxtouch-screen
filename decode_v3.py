#!/usr/bin/env python3
"""
Decoder v3 for the stent -> maXTouch BER runs (FROZEN 2026-10-22 19:01, before any held-out run was decoded).
Applied offline to the saved raw data; writes result_v3.json / decoded_v3.json. The original decoder (result.json,
decoded.json) and v2 (result_v2.json, decoded_v2.json, decode_v2.py) are not touched.

   python decode_v3.py data_maxtouch/stent_ber_<label>_<bps>bps_<time>    # one run
   python decode_v3.py all                                                # every run + baseline / v2 / v3 table

Rules (fixed, the same for every run):
  1. Sample grid and node ranking as the baseline and v2: maxtouch_stent_rx.regular, energy increase after START TX
     vs the lead (decided before decoding); the top 3 nodes are used.
  2. Feature by samples per bit k (from the measured scan period): logenergy if k >= 4, absdev if k < 4. logenergy
     differences consecutive samples, which needs several samples inside a bit; absdev judges each sample alone.
     The threshold 4 is the one cap_ook_rx already uses for its per-bit guard sample (set_guard).
  3. Combined decision = per sample the MAX over the 3 nodes of the z-scored feature.
  4. Single node: the top-ranked node (chosen before decoding); best of 3 after decoding for reference only.
  5. Known timing, accounting and health check exactly as v2 (decode_v2.known / frames_summary / health).
  6. Error types per run (descriptive, for the fading comparison): sent 1 decided 0 (fade) vs sent 0 decided 1, and
     how many errors fall in bursts (an error within 2 bits of another).
Run roles: 'exploratory' = runs looked at while developing v2/v3 (DEVELOPMENT_RUNS below); every other run is
'validation' (held out: decoded for the first time with these frozen rules).
"""
from __future__ import annotations

import contextlib
import csv
import glob
import hashlib
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import decode_v2 as V2                          # helpers only (blind, known, health, frames_summary, fmt, baseline_row)

R, RX, TX = V2.R, V2.RX, V2.TX
VERSION = "v3"
FROZEN_AT = "2026-10-22 19:01"
N_NODES = 3
K_SWITCH = 4.0                                  # samples per bit: logenergy at or above, absdev below
DEVELOPMENT_RUNS = {                            # looked at while designing v2 / v3 -> exploratory
    "stent_ber_c156_20bps_20261022_160629", "stent_ber_c156_20bps_20261022_175518",
    "stent_ber_c156_20bps_20261022_180016", "stent_ber_c156_20bps_20261022_180358",
    "stent_ber_c156_34bps_20261022_180957", "stent_ber_c156_45bps_20261022_181139",
}
RULES_SHA = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:12]


def feature_for(k):
    return "logenergy" if k >= K_SWITCH else "absdev"


def error_types(frames, truth):
    """sent 1 -> 0 (fade) / sent 0 -> 1, and errors within 2 bits of another error (bursts), over valid frames"""
    t = np.array(truth, int)
    f10 = f01 = burst = 0
    for fr in frames:
        if not fr.get("valid") or "got" not in fr:
            continue
        g = np.array(fr["got"], int)
        bad = np.where(g != t)[0]
        f10 += int(np.sum(t[bad] == 1)); f01 += int(np.sum(t[bad] == 0))
        burst += int(sum(1 for b in bad if np.any((np.abs(bad - b) <= 2) & (bad != b))))
    return dict(sent1_read0=f10, sent0_read1=f01, in_bursts=burst)


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
    feat = feature_for(k)
    names = R.node_names(meta)
    n_sent = meta.get("tx_frames", 0) or 0
    score, _ = R.node_scores(G, T, t0, meta.get("t_prompt"))
    top = [int(i) for i in np.argsort(-score)[:N_NODES]]
    feats = {i: RX.features(G[:, i])[feat] for i in top}
    single = []
    for i in top:
        s, fr = V2.blind(feats[i], k, plan, truth, n_sent)
        single.append(dict(node=names[i], score_db=float(score[i]), **s, errors_by_type=error_types(fr, truth)))
    comb_f = np.max([R.zscore(feats[i]) for i in top], axis=0)
    comb, comb_frames = V2.blind(comb_f, k, plan, truth, n_sent)
    run = os.path.basename(os.path.normpath(d))
    res = dict(decoder=VERSION, frozen_at=FROZEN_AT, rules_sha=RULES_SHA,
               role="exploratory" if run in DEVELOPMENT_RUNS else "validation",
               rules=f"top {N_NODES} nodes by energy increase; feature logenergy if k >= {K_SWITCH:g} else absdev; "
                     "max of z; header-aligned known timing",
               run=run, bps=plan["bps"], carrier_hz=a.carrier, samples_per_bit=k, feature=feat,
               scan_period_ms=T * 1e3, nodes=[names[i] for i in top],
               single_top_ranked=single[0],
               single_best_of_3_after_decoding=min(single, key=lambda s: (s["ber"] is None, s["ber"] or 0)),
               single_all=single, combined_blind=dict(comb, errors_by_type=error_types(comb_frames, truth)))
    kn_frames = []
    if meta.get("t_first") and n_sent:
        kn, kn_frames, d0 = V2.known(comb_f, k, plan, truth, (meta["t_first"] - t0) / T, n_sent)
        res["combined_known_timing"] = dict(kn, header_offset_samples=d0, errors_by_type=error_types(kn_frames, truth))
        res["health"] = V2.health(G, T, t0, meta, top[0], n_sent * plan["frame_s"])
    json.dump(res, open(os.path.join(d, "result_v3.json"), "w"), indent=1)
    keep = ("m", "s0", "valid", "bits", "errors", "ber", "q", "thr", "inverted", "got", "norm")
    pack = lambda frs: [{kk: v for kk, v in fr.items() if kk in keep} for fr in frs]
    json.dump(dict(decoder=VERSION, rules_sha=RULES_SHA, sent_payload=[int(b) for b in truth], combined_nodes=res["nodes"],
                   feature=feat, blind_frames=pack(comb_frames), known_timing_frames=pack(kn_frames)),
              open(os.path.join(d, "decoded_v3.json"), "w"))
    return res


def report(res):
    h = res.get("health", {})
    print(f"\n[v3 {res['role'].upper()}] {res['run']}: {res['bps']:g} bps, {res['carrier_hz'] or 0:g} Hz, "
          f"{res['samples_per_bit']:.1f} samples/bit -> {res['feature']}, nodes {', '.join(res['nodes'])}"
          + (f", signal present {h['signal_fraction']:.0%}" if h else "")
          + ("   ** SIGNAL LOST - not a link measurement **" if h.get("signal_lost") else ""))
    print(f"  single, top-ranked {res['single_top_ranked']['node']}:  {V2.fmt(res['single_top_ranked'])}")
    b = res["single_best_of_3_after_decoding"]
    print(f"  single, best of 3 (after decoding, optimistic) {b['node']}:  {V2.fmt(b)}")
    c = res["combined_blind"]
    print(f"  combined blind:        {V2.fmt(c)}   conservative {c['ber_conservative'] or 0:.2%}   errors {c['errors_by_type']}")
    kt = res.get("combined_known_timing")
    if kt:
        print(f"  combined known timing: {V2.fmt(kt)}   (header offset {kt['header_offset_samples']:+.2f} samples)")


def compare_all():
    rows = []
    for d in sorted(glob.glob(os.path.join(HERE, "data_maxtouch", "stent_ber_*"))):
        if not os.path.exists(os.path.join(d, "capture.npz")):
            continue
        res = decode_run(d)
        if res is None:
            continue
        report(res)
        b = V2.baseline_row(d)
        p2 = os.path.join(d, "result_v2.json")           # v2 as saved, not recomputed
        v2 = json.load(open(p2)) if os.path.exists(p2) else None
        h = res.get("health") or {}
        rows.append(dict(run=res["run"], role=res["role"], carrier_hz=res["carrier_hz"], bps=res["bps"],
                         samples_per_bit=round(res["samples_per_bit"], 2), feature_v3=res["feature"],
                         signal_present=h.get("signal_fraction"), signal_lost=h.get("signal_lost"), **b,
                         v2_comb_blind=V2.fmt(v2["combined_blind"]) if v2 else "not run",
                         v2_comb_blind_ber=v2["combined_blind"]["ber"] if v2 else None,
                         v3_single_top=V2.fmt(res["single_top_ranked"]), v3_single_top_ber=res["single_top_ranked"]["ber"],
                         v3_comb_blind=V2.fmt(res["combined_blind"]), v3_comb_blind_ber=res["combined_blind"]["ber"],
                         v3_comb_blind_conservative=res["combined_blind"]["ber_conservative"],
                         v3_comb_known=V2.fmt(res.get("combined_known_timing")),
                         v3_sent1_read0=res["combined_blind"]["errors_by_type"]["sent1_read0"],
                         v3_sent0_read1=res["combined_blind"]["errors_by_type"]["sent0_read1"],
                         v3_errors_in_bursts=res["combined_blind"]["errors_by_type"]["in_bursts"],
                         rules_sha=res["rules_sha"]))
    out = os.path.join(HERE, "data_maxtouch", "ber_compare_v3.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"\nv3 rules sha {RULES_SHA}, frozen {FROZEN_AT}")
    print(f"{'run':40s} {'role':>11} {'sig':>5} | {'BASELINE comb':>13} | {'v2 comb':>24} | {'v3 top node':>24} {'v3 comb blind':>24}")
    for r in rows:
        sig = "LOST" if r["signal_lost"] else f"{r['signal_present']:.0%}" if r["signal_present"] is not None else "-"
        print(f"{r['run'][10:50]:40s} {r['role']:>11} {sig:>5} | {r.get('base_comb', '-'):>13} | {r['v2_comb_blind']:>24} | "
              f"{r['v3_single_top']:>24} {r['v3_comb_blind']:>24}")
    print(f"\ntable: {os.path.relpath(out, HERE)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    if sys.argv[1] == "all":
        compare_all()
    else:
        report(decode_run(sys.argv[1]))
