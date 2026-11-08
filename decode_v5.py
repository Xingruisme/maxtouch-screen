#!/usr/bin/env python3
"""
Decoder v5 for the stent -> maXTouch OOK / FSK runs: decode_v4 with a baseline that follows the finger.
Writes result_v5.json / decoded_v5.json next to the run; decode_v4.py and decode_fska.py are imported, never changed,
and no v4 / FSK-A file is touched. Comparison and reports: v5_compare.py (separate, so this file's sha only covers the
algorithm).

   python decode_v5.py data_maxtouch/stent_mod_<label>_<mod>_<bps>bps_<time>

Why: when the finger moves or presses, each node's resting level shifts by hundreds to ~1500 counts. v4 measures
every sample against the OFF level of the lead (before START TX), so a shifted resting level reads as carrier
(sent 0 read 1) or hides it. v5 measures each sample against the node's own running median, which follows the shift.
The carrier does not move that median: each scan sees the tone at a fresh random phase, so its deltas are symmetric
about the resting level.

Rules (fixed at FROZEN_AT; nothing below changes after that, and no run recorded after it was looked at before):
  1. Nodes: v4 rule 1 unchanged (the 3 with the largest scan-to-scan energy increase after START TX vs the lead).
  2. Feature per raw sample and node: log10(1 + (x - B)^2 / sigma^2), B = median of the node's raw samples within
     +-BASE_HALF_S of that sample (centred window), sigma = OFF noise of the lead (1.4826 x MAD, at least 1 count,
     as v4); z-scored over the run (median / IQR, as v4); combined = per-sample max over the 3 nodes (as v4).
  3-6. Timing search, guard, per-frame threshold, frame validity and accounting: v4 rules 3-6, by calling
     decode_v4's own search / decode_frames / frames_summary (sha of decode_v4.py stored with every result).

Offline: the centred window needs BASE_HALF_S of samples after the one being decoded (0.25 s look-ahead). v4 and v5
are also offline for reasons that are not the window: z-scoring over the whole run, the known-timing and blind
searches over all frames (clock drift), and the per-frame threshold, which uses the trailing gap bits after the
payload. v5 is the offline reference; a real-time version needs the changes listed in v5_compare.py --realtime-notes.

History (development, before FROZEN_AT): variants were tried on the 8 hand159 still/swipe runs of 2026-10-26 and then
on the 96 ankle-to-finger runs of 2026-10-27..11-01 (20/40/50/60 bps), known-timing errors:
  hand159 (8 x 1016 bits): v4 612 | per-scan node tracking 579 | drift baseline 227 | tracking + drift 285
  ankle-to-finger still (48768 bits): v4 203 | tracking 199 | drift 144 | both 144
  ankle-to-finger moving (50800 bits): v4 392 | tracking 495 | drift 273 | both 356
Node tracking was dropped (it follows nodes whose energy comes from the finger moving). Every run recorded before
FROZEN_AT is development data; only runs recorded after it are validation.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import decode_v2 as V2                          # frames_summary
import decode_v3 as V3                          # error_types
import decode_v4 as V4                          # frozen: search, decode_frames, bit timing, guard

R, RX, TX = V4.R, V4.RX, V4.TX
VERSION = "v5"
BASE_HALF_S = 0.25
FROZEN_AT = "2026-11-02 11:00:00"
RULES_SHA = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:12]
V4_SHA = V4.RULES_SHA


def recorded_at(d, meta):
    m = re.search(r"_(\d{8}_\d{6})$", os.path.basename(os.path.normpath(d)))
    if m:
        return datetime.datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
    return datetime.datetime.fromtimestamp(meta["t_first"])


def role(d, meta):
    frozen = datetime.datetime.strptime(FROZEN_AT, "%Y-%m-%d %H:%M:%S")
    return "validation" if recorded_at(d, meta) > frozen else "development"


def drift_absdev(x, t, lead):
    """log10(1 + (x - running median over +-BASE_HALF_S)^2 / OFF noise^2)"""
    med0 = float(np.median(x[lead]))
    sig = max(1.4826 * float(np.median(np.abs(x[lead] - med0))), 1.0)
    lo, hi = np.searchsorted(t, t - BASE_HALF_S), np.searchsorted(t, t + BASE_HALF_S, side="right")
    b = np.array([np.median(x[a:z]) for a, z in zip(lo, hi)])
    return np.log10(1 + (x - b) ** 2 / sig ** 2)


def nodes(X, t_prompt, t):
    """v4 rule 1"""
    pre = t < t_prompt
    e = np.diff(X, axis=0) ** 2
    score = 10 * np.log10((e[~pre[1:]].mean(axis=0) + 1e-9) / (e[pre[1:]].mean(axis=0) + 1e-9))
    return [int(i) for i in np.argsort(-score)[:V4.N_NODES]]


def decode_run(d, write=True):
    meta = json.load(open(os.path.join(d, "meta.json")))
    z = np.load(os.path.join(d, "capture.npz"))
    t, X = z["t"].astype(float), z["X"].astype(float)
    a = R.argparse.Namespace(**meta["args"])
    link = R.link_from(a, False)
    _, plan = TX.build(link)
    frame_bits, truth = TX.frame_bits(link.payload, link.prbs)
    Tb, nb = 1.0 / plan["bps"], len(frame_bits)
    P = nb * Tb
    n_sent = meta.get("tx_frames", 0) or 0
    pre = t < meta["t_prompt"]
    top = nodes(X, meta["t_prompt"], t)
    comb = np.max([R.zscore(drift_absdev(X[:, i], t, pre)) for i in top], axis=0)

    def all_frames(tau, eps):                    # as decode_v4.decode_run
        last = int((t[-1] - tau) / (P * (1 + eps)))
        return np.arange(max(0, int(np.ceil((t[0] - tau) / (P * (1 + eps))))), last + 1)

    tau_b, eps_b = V4.search(comb, t, Tb, nb, t[0], t[0] + P, all_frames)
    bfr = V4.decode_frames(comb, t, tau_b, Tb, eps_b, frame_bits, truth, all_frames(tau_b, eps_b), False)
    tau_k, eps_k = V4.search(comb, t, Tb, nb, meta["t_first"], meta["t_first"] + 0.025, lambda tau, eps: np.arange(n_sent))
    kfr = V4.decode_frames(comb, t, tau_k, Tb, eps_k, frame_bits, truth, np.arange(n_sent), True)
    names = R.node_names(meta)
    run = os.path.basename(os.path.normpath(d))
    res = dict(decoder=VERSION, rules_sha=RULES_SHA, frozen_at=FROZEN_AT, v4_rules_sha=V4_SHA, base_half_s=BASE_HALF_S,
               role=role(d, meta), run=run, modulation=meta.get("modulation"), bps=plan["bps"],
               nodes=[names[i] for i in top],
               combined_blind=dict(V2.frames_summary(bfr, n_sent, len(truth)), clock_ppm=eps_b * 1e6,
                                   errors_by_type=V3.error_types(bfr, truth)),
               combined_known_timing=dict(V2.frames_summary(kfr, n_sent, len(truth)),
                                          delay_ms=(tau_k - meta["t_first"]) * 1e3, clock_ppm=eps_k * 1e6,
                                          errors_by_type=V3.error_types(kfr, truth)))
    if write:
        json.dump(res, open(os.path.join(d, "result_v5.json"), "w"), indent=1)
        keep = ("m", "s0", "r", "valid", "bits", "errors", "ber", "thr", "got", "norm")
        pack = lambda frs: [{k: v for k, v in f.items() if k in keep} for f in frs]
        json.dump(dict(decoder=VERSION, rules_sha=RULES_SHA, sent_payload=[int(b) for b in truth], combined_nodes=res["nodes"],
                       blind_frames=pack(bfr), known_timing_frames=pack(kfr)), open(os.path.join(d, "decoded_v5.json"), "w"))
    return res


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    r = decode_run(sys.argv[1])
    print(f"[v5 {r['role']}] {r['run']}  nodes {', '.join(r['nodes'])}\n  blind: {V2.fmt(r['combined_blind'])}\n"
          f"  known timing: {V2.fmt(r['combined_known_timing'])}   rules sha {RULES_SHA}")
