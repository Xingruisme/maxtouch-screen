#!/usr/bin/env python3
"""
Decoder v7 (EXPERIMENTAL, not frozen): localisation-guided node selection. v5's drift baseline, timing search and
bit decisions, but the decoding nodes follow the finger instead of being fixed for the whole run.
decode_v4.py and decode_v5.py (both frozen) are imported, never changed; nothing is written into run folders.

   python decode_v7.py data_maxtouch/stent_mod_<label>_<mod>_<bps>bps_<time>      # one run, v5 vs v7
   python v7_compare.py ...                                                        # runs -> tables + figures

Why (development data, 2026-11-02 13:39 swipe pair, touch_vs_carrier_swipe60_1339): while the finger swipes, the
carrier follows it over Y0-Y9, but v4/v5 decode three nodes fixed for the run, which held the carrier peak in only
28-33 % of 0.25 s windows; the strongest touch node was the strongest carrier node in 75 % of windows, within one
node in 98 %. Second failure found on the same runs: v5's noise sigma is the lead MAD around the lead median, and the
finger was already swiping in the lead, so sigma on the swiped nodes read 264-400 counts instead of 6-12 (signal
squashed by ~30 dB).

Rules (the bits are never used; timing and decisions are v5's):
  1. Slow profile B_j(t) = running median of node j over +-BASE_HALF_S (0.25 s, v5's baseline, same windows).
  2. Finger position (bit-free): if max_j B_j > VIS (100 counts): centroid of the B_j >= FRAC (25 %) of that max.
     Otherwise the finger is not visible, which after a finger-on calibration means it is back where it was at
     calibration: that place is the node where B dips below -NEG (300 counts: where the finger left) - the median
     argmin over the samples showing such a dip, if they are >= 10 % of the run. Without such a dip (finger-off
     calibration, or the finger never left): hold the last visible position.
  3. Node selection with hysteresis: centre node c; c moves to round(position) only when |position - c| > 0.5 + HYST
     (0.3 node), so the set does not flicker when the finger sits between two nodes. Nodes = c-1, c, c+1 (N = 3,
     shifted inwards at the row ends).
  4. Noise sigma per node = 1.4826 MAD of (x - B) over the lead: the deviation from the slow profile, so a finger
     moving during the lead does not count as noise.
  5. Feature per node = log10(1 + (x - B)^2 / sigma^2) (v5's drift absdev with rule 4's sigma), in units of each
     node's own noise, so no whole-run z-score: a node used for only part of the run has a whole-run median / IQR
     that mixes carrier and no-carrier time. Combined = per-sample max over that sample's 3 nodes.
  6. v4's header timing search, guard, per-frame threshold and frame validity (via decode_v4, as v5), known start
     and blind.
Options (ablations for the comparison only): nodes = 'track' (rules 2-3) | 'v4' (v5's fixed nodes) | 'oracle'
(carrier peak node from the bits per 0.25 s window, +-1: a diagnostic ceiling, not a decoder); sigma = 'runmed'
(rule 4) | 'lead' (v5's); zscore = False (rule 5) | True (v5's).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import decode_v2 as V2
import decode_v3 as V3
import decode_v4 as V4                          # frozen
import decode_v5 as V5                          # frozen

R, TX = V4.R, V4.TX
VERSION = "v7-experimental"
FROZEN_AT = None                                # not frozen: every run is development data until it is
BASE_HALF_S = V5.BASE_HALF_S
VIS, NEG, NEG_MIN_FRAC, FRAC, HYST, N = 100.0, 300.0, 0.10, 0.25, 0.3, V4.N_NODES


def running_median(X, t, half=BASE_HALF_S):
    lo, hi = np.searchsorted(t, t - half), np.searchsorted(t, t + half, side="right")
    return np.array([np.median(X[a:z], axis=0) for a, z in zip(lo, hi)])


def finger_track(B):
    """rule 2: position per sample and where it came from (1 touch, 2 calibration place, 0 held)"""
    n = B.shape[1]
    neg = B.min(axis=1) < -NEG
    cal = float(np.median(B[neg].argmin(axis=1))) if neg.mean() >= NEG_MIN_FRAC else None
    pos, src = np.empty(len(B)), np.zeros(len(B), int)
    last = cal if cal is not None else float(np.argmax(B.max(axis=0)))
    for i, b in enumerate(B):
        mx = b.max()
        if mx > VIS:
            p = np.where(b >= FRAC * mx, b, 0)
            last, src[i] = float((p * np.arange(n)).sum() / p.sum()), 1
        elif cal is not None:
            last, src[i] = cal, 2
        pos[i] = last
    return pos, src, cal


def centres(pos, n):
    """rule 3: hysteresis on the centre node"""
    c = int(np.clip(round(pos[0]), 1, n - 2))
    out = np.empty(len(pos), int)
    for i, p in enumerate(pos):
        if abs(p - c) > 0.5 + HYST:
            c = int(np.clip(round(p), 1, n - 2))
        out[i] = c
    return out


def decode_run(d, nodes="track", sigma="runmed", zscore=False, oracle_centre=None):
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
    lead = t < meta["t_prompt"]
    n = X.shape[1]
    B = running_median(X, t)
    if sigma == "runmed":
        sig = np.maximum(1.4826 * np.median(np.abs(X[lead] - B[lead]), axis=0), 1.0)
    else:                                       # v5: MAD around the lead median
        sig = np.maximum(1.4826 * np.median(np.abs(X[lead] - np.median(X[lead], axis=0)), axis=0), 1.0)
    F = np.log10(1 + (X - B) ** 2 / sig ** 2)
    if zscore:
        F = np.column_stack([R.zscore(F[:, j]) for j in range(n)])
    pos, src, cal = finger_track(B)
    fixed = V5.nodes(X, meta["t_prompt"], t)
    if nodes == "track":
        cen = centres(pos, n)
    elif nodes == "oracle":
        cen = np.clip(np.asarray(oracle_centre, int), 1, n - 2)
    if nodes == "v4":
        sel = np.tile(fixed, (len(t), 1))
    else:
        sel = cen[:, None] + np.arange(-(N // 2), N - N // 2)[None, :]
    comb = np.take_along_axis(F, sel, axis=1).max(axis=1)

    def all_frames(tau, eps):                    # as decode_v4 / decode_v5
        last = int((t[-1] - tau) / (P * (1 + eps)))
        return np.arange(max(0, int(np.ceil((t[0] - tau) / (P * (1 + eps))))), last + 1)

    tau_b, eps_b = V4.search(comb, t, Tb, nb, t[0], t[0] + P, all_frames)
    bfr = V4.decode_frames(comb, t, tau_b, Tb, eps_b, frame_bits, truth, all_frames(tau_b, eps_b), False)
    tau_k, eps_k = V4.search(comb, t, Tb, nb, meta["t_first"], meta["t_first"] + 0.025, lambda tau, eps: np.arange(n_sent))
    kfr = V4.decode_frames(comb, t, tau_k, Tb, eps_k, frame_bits, truth, np.arange(n_sent), True)
    switches = int(np.sum(np.any(np.diff(sel, axis=0) != 0, axis=1)))
    tx = (t >= meta["t_first"]) & (t <= meta["t_first"] + n_sent * P)
    return dict(decoder=VERSION, nodes_rule=nodes, sigma=sigma, zscore=zscore, v4_rules_sha=V4.RULES_SHA,
                v5_rules_sha=V5.RULES_SHA, run=os.path.basename(os.path.normpath(d)), modulation=meta.get("modulation"),
                bps=plan["bps"], calibration=meta.get("calibration", "finger-on"),
                calibration_place=None if cal is None else f"Y{cal:g}", v5_nodes=[f"Y{i}" for i in fixed],
                node_switches=switches, switches_per_s=switches / max(t[tx][-1] - t[tx][0], 1e-9) if tx.any() else None,
                position_from=dict(touch=float(np.mean(src[tx] == 1)), calibration_place=float(np.mean(src[tx] == 2)),
                                   held=float(np.mean(src[tx] == 0))) if tx.any() else None,
                sigma_counts=[round(float(s), 1) for s in sig],
                known=dict(V2.frames_summary(kfr, n_sent, len(truth)), errors_by_type=V3.error_types(kfr, truth)),
                blind=dict(V2.frames_summary(bfr, n_sent, len(truth)), errors_by_type=V3.error_types(bfr, truth))), \
        dict(t=t, pos=pos, src=src, sel=sel, B=B, known_frames=kfr, truth=truth, Tb=Tb * (1 + eps_k), hdr=len(V4.HDR))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    d = sys.argv[1]
    v5 = V5.decode_run(d, write=False)
    r, _ = decode_run(d)
    print(f"  v5 known {V2.fmt(v5['combined_known_timing'])}  blind {V2.fmt(v5['combined_blind'])}  nodes {v5['nodes']}")
    print(f"  v7 known {V2.fmt(r['known'])}  blind {V2.fmt(r['blind'])}  switches {r['node_switches']} "
          f"({r['switches_per_s']:.2f}/s), position from {r['position_from']}")
