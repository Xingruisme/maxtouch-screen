#!/usr/bin/env python3
"""
Decoder v6 (EXPERIMENTAL, not frozen): v5's drift baseline with nodes chosen around the finger, located from the touch
signal of the transmitter's OFF bits. decode_v4.py and decode_v5.py (both frozen) are imported, never changed.
Writes nothing into run folders; compare with v6_compare.py.

   python decode_v6.py data_maxtouch/stent_mod_<label>_<mod>_<bps>bps_<time>

What the panel gives (looked at 2026-11-02 on existing runs, all development data):
  * one row only: X line 0, nodes Y0-Y27, so the finger can be placed along Y and nowhere else;
  * each node's delta is relative to the chip's calibration at setup, which happens with the finger already on the
    panel: a finger that has not moved since then reads ~0 everywhere (invisible); where it moved to reads positive
    (touch = positive delta, up to ~+1500 counts), where it left reads negative. So the touch signal locates CHANGES
    of finger position: visible in 28 % (still) to 56 % (moving) of 0.25 s windows of the ankle-to-finger runs;
  * with the carrier on, the median of ON-bit samples moves the estimated centre by ~0.4-0.7 nodes (median; 0.06
    without a carrier); OFF-bit samples do not carry the carrier, so the finger is located from OFF bits only (needs
    bit timing -> two passes);
  * the physical finger centre and the strongest-carrier node are 0.4-0.9 nodes apart (median): not the same thing.

Development result (2026-11-02, v6_compare.py --label dev, 104 existing runs; known-timing errors v5 -> v6):
  hand swipe 221 -> 345, hand still 6 -> 9, ankle-to-finger moving 273 -> 500, still 144 -> 232; blind frames equal
  or fewer. Better only where the finger left v4's nodes (hand swipe OOK 10 bps 136 -> 102). Why: the located centre
  is the centre of the CHANGE since calibration, not of the finger, and is noisy when the change is small (still
  runs: 22 % of window-to-window jumps > 0.5 node), so the node set switches when nothing moved. Not promoted.

Rules:
  1. Pass 1 = v5 (v4 nodes, drift baseline): known-timing and blind bit timing.
  2. Finger centre, every WIN_S during the frames of pass 1: touch profile T_j = median of node j's guarded OFF-bit
     samples in the window; if max_j T_j > VIS_COUNTS the centre = centroid of max(T, 0) over nodes, else the previous
     centre is held. Start: the mean index of v4's nodes. Outside the found frames: held.
  3. Nodes per window: of the nodes within +-REGION of the rounded centre, the N (3) with the largest carrier score
     (v4 rule 1's run-level energy increase after START TX). The physical centre picks the region; the carrier score
     picks the nodes inside it.
  4. Feature: v5's drift absdev per node, z-scored over the run; combined = per-sample max over the window's nodes.
  5. Pass 2: v4's timing search, guard, threshold and frame validity on the new combined feature (known and blind).
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
VERSION = "v6-experimental"
WIN_S, VIS_COUNTS, REGION, N = 0.25, 100.0, 2, V4.N_NODES
GUARD_FRAC = 0.12                               # OFF samples for the touch profile: away from the bit edges


def carrier_score(X, t, t_prompt):
    pre = t < t_prompt
    e = np.diff(X, axis=0) ** 2
    return 10 * np.log10((e[~pre[1:]].mean(axis=0) + 1e-9) / (e[pre[1:]].mean(axis=0) + 1e-9))


def off_mask(t, tau, Tb, eps, frame_bits, frames):
    m, j, pos = V4.bit_index(t, tau, Tb, eps, len(frame_bits))
    fb = np.asarray(frame_bits)
    ok = np.isin(m, frames) & (j >= 0) & (j < len(fb)) & (pos > GUARD_FRAC) & (pos < 1 - GUARD_FRAC)
    return ok & (fb[np.clip(j, 0, len(fb) - 1)] == 0)


def locate(X, t, off, t0, t1, start):
    """finger centre per window (held when the finger is not visible); returns window starts, centres, visible"""
    w0 = np.arange(t0, t1, WIN_S)
    cen, vis, c = [], [], start
    for a in w0:
        k = off & (t >= a) & (t < a + WIN_S)
        v = False
        if k.sum() >= 3:
            T = np.median(X[k], axis=0)
            if T.max() > VIS_COUNTS:
                p = np.clip(T, 0, None)
                c, v = float((p * np.arange(len(T))).sum() / p.sum()), True
        cen.append(c); vis.append(v)
    return w0, np.array(cen), np.array(vis)


def pick_nodes(cen, score):
    out = []
    for c in cen:
        lo, hi = max(0, int(round(c)) - REGION), min(len(score) - 1, int(round(c)) + REGION)
        cand = np.arange(lo, hi + 1)
        out.append(tuple(sorted(cand[np.argsort(-score[cand])[:N]])))
    return out


def combined(Zcache, X, t, pre, w0, sets, fallback):
    """per-sample max of z features over the nodes of the sample's window (fallback outside the windows)"""
    def z(i):
        if i not in Zcache:
            Zcache[i] = R.zscore(V5.drift_absdev(X[:, i], t, pre))
        return Zcache[i]
    idx = np.searchsorted(w0, t, side="right") - 1
    comb = np.max([z(i) for i in fallback], axis=0)
    inside = (idx >= 0) & (t < w0[-1] + WIN_S)
    for s in set(sets):
        k = inside & np.isin(idx, [n for n, ss in enumerate(sets) if ss == s])
        if k.any():
            comb[k] = np.max([z(i)[k] for i in s], axis=0)
    return comb


def decode_run(d):
    meta = json.load(open(os.path.join(d, "meta.json")))
    zf = np.load(os.path.join(d, "capture.npz"))
    t, X = zf["t"].astype(float), zf["X"].astype(float)
    a = R.argparse.Namespace(**meta["args"])
    link = R.link_from(a, False)
    _, plan = TX.build(link)
    frame_bits, truth = TX.frame_bits(link.payload, link.prbs)
    Tb, nb = 1.0 / plan["bps"], len(frame_bits)
    P = nb * Tb
    n_sent = meta.get("tx_frames", 0) or 0
    pre = t < meta["t_prompt"]
    score = carrier_score(X, t, meta["t_prompt"])
    top = V5.nodes(X, meta["t_prompt"], t)
    Zc = {}
    comb5 = np.max([Zc.setdefault(i, R.zscore(V5.drift_absdev(X[:, i], t, pre))) for i in top], axis=0)

    def all_frames(tau, eps):
        last = int((t[-1] - tau) / (P * (1 + eps)))
        return np.arange(max(0, int(np.ceil((t[0] - tau) / (P * (1 + eps))))), last + 1)

    known_frames = lambda tau, eps: np.arange(n_sent)
    out = dict(decoder=VERSION, v5_rules_sha=V5.RULES_SHA, v4_rules_sha=V4.RULES_SHA, run=os.path.basename(os.path.normpath(d)),
               modulation=meta.get("modulation"), bps=plan["bps"], v4_nodes=[f"Y{i}" for i in top])
    for mode in ("known", "blind"):
        if mode == "known":
            lo, hi, frames_of = meta["t_first"], meta["t_first"] + 0.025, known_frames
        else:
            lo, hi, frames_of = t[0], t[0] + P, all_frames
        tau1, eps1 = V4.search(comb5, t, Tb, nb, lo, hi, frames_of)
        fr1 = V4.decode_frames(comb5, t, tau1, Tb, eps1, frame_bits, truth, frames_of(tau1, eps1), mode == "known")
        found = [f["m"] for f in fr1 if f["valid"]]
        off = off_mask(t, tau1, Tb, eps1, frame_bits, found)
        t0 = tau1 + (min(found) if found else 0) * P * (1 + eps1)
        t1 = tau1 + ((max(found) + 1) if found else 0) * P * (1 + eps1)
        if t1 <= t0:
            t0, t1 = meta["t_first"], meta["t_first"] + n_sent * P
        w0, cen, vis = locate(X, t, off, t0, t1, float(np.mean(top)))
        sets = pick_nodes(cen, score)
        comb6 = combined(Zc, X, t, pre, w0, sets, top)
        tau2, eps2 = V4.search(comb6, t, Tb, nb, lo, hi, frames_of)
        fr2 = V4.decode_frames(comb6, t, tau2, Tb, eps2, frame_bits, truth, frames_of(tau2, eps2), mode == "known")
        for name, fr in (("v5", fr1), ("v6", fr2)):
            out[f"{name}_{mode}"] = dict(V2.frames_summary(fr, n_sent, len(truth)), errors_by_type=V3.error_types(fr, truth),
                                         frame_errors={int(f["m"]): int(f["errors"]) for f in fr if f["valid"]})
        if mode == "known":
            used = {}
            for s in sets:
                for i in s:
                    used[f"Y{i}"] = used.get(f"Y{i}", 0) + 1
            fs = tau1 + np.arange(n_sent + 1) * P * (1 + eps1)          # frame boundaries (pass-1 timing)
            moved = np.r_[False, np.abs(np.diff(cen)) > 0.5]             # centre moved > 0.5 node since the window before
            out.update(windows=len(w0), finger_visible=float(vis.mean()) if len(vis) else None,
                       centre_range=[float(np.min(cen)), float(np.max(cen))] if len(cen) else None,
                       window_sets_outside_v4=float(np.mean([any(i not in top for i in s) for s in sets])) if sets else None,
                       nodes_used={k: round(v / max(len(sets), 1), 3) for k, v in sorted(used.items(), key=lambda kv: -kv[1])},
                       frame_moving={int(m): bool(moved[(w0 >= fs[m]) & (w0 < fs[m + 1])].any()) for m in range(n_sent)})
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    r = decode_run(sys.argv[1])
    for k in ("v5_known", "v6_known", "v5_blind", "v6_blind"):
        print(f"  {k:9s} {V2.fmt(r[k])}  {r[k]['errors_by_type']}")
    print(f"  finger visible {r['finger_visible']:.0%}, centre Y{r['centre_range'][0]:.1f}-Y{r['centre_range'][1]:.1f}, "
          f"windows using a node outside v4's {r['window_sets_outside_v4']:.0%}, nodes {r['nodes_used']}")
