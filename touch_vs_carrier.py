#!/usr/bin/env python3
"""
Offline: where is the finger (touch response) and where is the carrier (communication response) on the recorded row,
and are they the same nodes? No decoding: v4 / v5 (frozen) are imported read-only for the bit timing and v4's nodes.
Writes into a new folder only.

   python touch_vs_carrier.py still=<runs or sessions ...> moving=<runs or sessions ...> --label <name>
   python touch_vs_carrier.py still=modcomp_a2f_offcal_still_<time> moving=modcomp_a2f_offcal_swipe_<time> --label offcal1

Groups: NAME=<folder> starts a group, the following bare folders join it; a folder is a run or a session
(modcomp_<label>_<time>: its OOK / FSK runs), '+' also joins folders (as v5_compare.py). The condition is given
here, not read from meta.json (the sit-shaking runs of 2026-11-01 were recorded with --condition still).

Meant for runs recorded with stent_modcomp.py --calibrate finger-off (keyed: OOK / FSK). On finger-on runs (all runs
before 2026-11-02) the 'touch' profile is only the change of the finger since calibration (a still finger reads ~0);
they are analysed too but labelled as such and kept apart in the summary.

Per run, bit timing = v4's known-timing search on v5's combined feature (as decode_v6.py pass 1), samples within
v4's 2 ms guard of a bit edge left out. Per 0.25 s window during the transmission:
  touch level    T_j = median of node j's sent-0 samples (no carrier; FSK's '0' tone arrives ~32 dB down)
  touch node     argmax T (only when max T > 100 counts: 'visible'); touch centroid = centroid of T >= 25 % of max
  carrier power  C_j = mean over sent-1 samples of (x - b_j(t))^2 - (sent-0 residual variance), b_j(t) = the touch
                 level interpolated in time between sent-0 samples (so finger motion within the window is not
                 counted as carrier); carrier node = argmax C, carrier centroid = centroid of C >= 25 % of max
  timing-free touch estimates (a receiver that does not know the bits yet):
                 'runmed'  centroid of the per-node median of ALL samples in the window (v5's baseline idea)
                 'on'      centroid of the median of the sent-1 samples (how much the carrier moves the level)
                 compared with the sent-0 centroid (reference)
Reported per run / group: touch visible %, touch node == carrier node %, |carrier - touch node| median / p90, signed
offset, carrier node within touch node +-1 / +-2 %, all 3 strongest carrier nodes within +-2 %, carrier node among
v4's 3 nodes %, timing-free localisation error, and the lead (before START TX, no carrier) touch node.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import decode_v4 as V4                          # frozen, read only
import decode_v5 as V5                          # frozen, read only
from v5_compare import expand

R, TX = V4.R, V4.TX
DATADIR = R.DATADIR
WIN_S, VIS, FRAC = 0.25, 100.0, 0.25
KEYED = ("ook", "fsk")


def centroid(p):
    p = np.where(p >= FRAC * p.max(), p, 0) if p.max() > 0 else p
    return float((p * np.arange(len(p))).sum() / p.sum()) if p.sum() > 0 else np.nan


def timing(meta, t, X):
    a = R.argparse.Namespace(**meta["args"])
    link = R.link_from(a, False)
    _, plan = TX.build(link)
    frame_bits, _ = TX.frame_bits(link.payload, link.prbs)
    Tb, nb, n_sent = 1.0 / plan["bps"], len(frame_bits), meta.get("tx_frames", 0) or 0
    pre = t < meta["t_prompt"]
    top = V5.nodes(X, meta["t_prompt"], t)
    comb = np.max([R.zscore(V5.drift_absdev(X[:, i], t, pre)) for i in top], axis=0)
    frames = np.arange(n_sent)
    tau, eps = V4.search(comb, t, Tb, nb, meta["t_first"], meta["t_first"] + 0.025, lambda tau, eps: frames)
    m, j, pos = V4.bit_index(t, tau, Tb, eps, nb)
    g = V4.GUARD_MS / 1e3 / Tb
    inside = np.isin(m, frames) & (j >= 0) & (j < nb) & (pos > g) & (pos < 1 - g)
    fb = np.asarray(frame_bits)[np.clip(j, 0, nb - 1)]
    t0, t1 = tau, tau + n_sent * nb * Tb * (1 + eps)
    return dict(on=inside & (fb == 1), off=inside & (fb == 0), t0=t0, t1=t1, top=top, bps=plan["bps"])


def groups_of(tokens):
    out, cur = {}, None
    for tok in tokens:
        if "=" in tok and not os.path.exists(tok):
            cur, tok = tok.split("=", 1)
            out.setdefault(cur, [])
        if cur is None:
            sys.exit("start with NAME=<folder> (e.g. still=modcomp_...)")
        out[cur] += expand(tok)
    return out


def analyse_run(d):
    meta = json.load(open(os.path.join(d, "meta.json")))
    if meta.get("modulation") not in KEYED:
        return None, None
    z = np.load(os.path.join(d, "capture.npz"))
    t, X = z["t"].astype(float), z["X"].astype(float)
    tm = timing(meta, t, X)
    on, off = tm["on"], tm["off"]
    lead = t < meta["t_prompt"]
    Tlead = np.median(X[lead], axis=0)
    # touch level in time from sent-0 samples (5-sample running median), per node
    io = np.flatnonzero(off)
    from scipy.ndimage import median_filter
    Xo = median_filter(X[io], size=(5, 1), mode="nearest")
    base = np.column_stack([np.interp(t, t[io], Xo[:, k]) for k in range(X.shape[1])])
    wins = []
    for w0 in np.arange(tm["t0"], tm["t1"] - WIN_S / 2, WIN_S):
        k = (t >= w0) & (t < w0 + WIN_S)
        ko, kn = k & off, k & on
        if ko.sum() < 3 or kn.sum() < 3:
            continue
        T = np.median(X[ko], axis=0)
        resid = ((X[ko] - base[ko]) ** 2).mean(axis=0)
        C = ((X[kn] - base[kn]) ** 2).mean(axis=0) - resid
        vis = T.max() > VIS
        tn, cn = int(np.argmax(T)), int(np.argmax(C))
        c3 = np.argsort(-C)[:3]
        r = dict(t=float(w0 - tm["t0"]), visible=bool(vis), touch_node=tn if vis else None, touch_level=float(T.max()),
                 touch_centroid=centroid(np.clip(T, 0, None)) if vis else np.nan,
                 carrier_node=cn, carrier_rms=float(np.sqrt(max(C.max(), 0))),
                 carrier_centroid=centroid(np.clip(C, 0, None)), carrier_top3=[int(i) for i in c3],
                 carrier_in_v4=bool(cn in tm["top"]))
        if vis:
            r.update(offset=cn - tn, top3_within2=bool(np.all(np.abs(c3 - tn) <= 2)),
                     runmed_centroid=centroid(np.clip(np.median(X[k], axis=0), 0, None)),
                     on_centroid=centroid(np.clip(np.median(X[kn], axis=0), 0, None)))
        wins.append(r)
    names = [f"Y{i}" for i in range(X.shape[1])]
    info = dict(run=os.path.basename(os.path.normpath(d)), modulation=meta["modulation"], bps=tm["bps"],
                calibration=meta.get("calibration", "finger-on"),
                lead_touch_node=names[int(np.argmax(Tlead))], lead_touch_counts=float(Tlead.max()),
                v4_nodes=[names[i] for i in tm["top"]])
    return info, wins


def summarise(wins):
    v = [w for w in wins if w["visible"]]
    s = dict(windows=len(wins), touch_visible=len(v) / max(len(wins), 1),
             carrier_in_v4=float(np.mean([w["carrier_in_v4"] for w in wins])) if wins else None)
    if v:
        off = np.array([w["offset"] for w in v])
        rm = np.array([abs(w["runmed_centroid"] - w["touch_centroid"]) for w in v])
        onc = np.array([abs(w["on_centroid"] - w["touch_centroid"]) for w in v])
        cc = np.array([w["carrier_centroid"] - w["touch_centroid"] for w in v])
        s.update(same_node=float(np.mean(off == 0)), within1=float(np.mean(np.abs(off) <= 1)),
                 within2=float(np.mean(np.abs(off) <= 2)), abs_offset_median=float(np.median(np.abs(off))),
                 abs_offset_p90=float(np.percentile(np.abs(off), 90)), signed_offset_median=float(np.median(off)),
                 centroid_offset_median=float(np.nanmedian(cc)), top3_within2=float(np.mean([w["top3_within2"] for w in v])),
                 runmed_err_median=float(np.nanmedian(rm)), runmed_err_p90=float(np.nanpercentile(rm, 90)),
                 on_err_median=float(np.nanmedian(onc)), on_err_p90=float(np.nanpercentile(onc, 90)),
                 touch_nodes=sorted({f"Y{w['touch_node']}" for w in v}), offsets=[int(o) for o in off])
    return s


def plot_run(info, wins, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tw = np.array([w["t"] for w in wins])
    fig, ax = plt.subplots(1, 1, figsize=(11, 3.8))
    vis = np.array([w["visible"] for w in wins])
    ax.plot(tw[vis], [w["touch_centroid"] for w in np.array(wins)[vis]], "-", color="#1f5fbf", lw=1.2,
            label="touch centroid (sent-0 samples)")
    ax.plot(tw[vis], [w["touch_node"] for w in np.array(wins)[vis]], "s", ms=3, color="#1f5fbf", label="strongest touch node")
    ax.plot(tw, [w["carrier_centroid"] for w in wins], "-", color="#d62728", lw=1.2, label="carrier centroid (sent-1 power)")
    ax.plot(tw, [w["carrier_node"] for w in wins], "o", ms=3, mfc="none", color="#d62728", label="strongest carrier node")
    for n in info["v4_nodes"]:
        ax.axhline(int(n[1:]), color="#999", lw=0.6, ls=":")
    ax.set_xlabel("time from first frame (s)"); ax.set_ylabel("Y node (X line 0)")
    ax.set_title(f"{info['run']} ({info['calibration']} calibration; dotted: v4 nodes {', '.join(info['v4_nodes'])})", fontsize=9)
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("groups", nargs="+", help="NAME=<run or session folder> [more folders] ...")
    p.add_argument("--label", required=True)
    a = p.parse_args()
    out = os.path.join(DATADIR, f"touch_vs_carrier_{a.label}")
    if os.path.exists(out):                     # never overwrite an earlier analysis
        out += f"_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    os.makedirs(out)
    runs, groups, allw = [], {}, []
    for cond, dirs in groups_of(a.groups).items():
        for d in sorted(dirs):
            info, wins = analyse_run(d)
            if info is None:
                print(f"  skip {os.path.basename(os.path.normpath(d))} (not OOK / FSK)")
                continue
            info["condition"] = cond
            s = summarise(wins)
            runs.append(dict(info, **{k: v for k, v in s.items() if k != "offsets"}))
            groups.setdefault(f"{info['calibration']} | {cond}", []).extend(wins)
            allw += [dict(run=info["run"], condition=cond, **{k: v for k, v in w.items() if k != "carrier_top3"})
                     for w in wins]
            plot_run(info, wins, os.path.join(out, f"{info['run']}.png"))
    summary = {k: summarise(w) for k, w in groups.items()}
    json.dump(dict(groups=summary, runs=runs, window_s=WIN_S, visible_counts=VIS, v4_rules_sha=V4.RULES_SHA,
                   v5_rules_sha=V5.RULES_SHA), open(os.path.join(out, "summary.json"), "w"), indent=1)
    with open(os.path.join(out, "per_window.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in allw for k in r}))
        w.writeheader(); w.writerows(allw)
    cols = ["run", "calibration", "condition", "modulation", "bps", "lead_touch_node", "lead_touch_counts", "v4_nodes",
            "windows", "touch_visible", "same_node", "within1", "within2", "abs_offset_median", "signed_offset_median",
            "top3_within2", "carrier_in_v4", "runmed_err_median", "on_err_median"]
    with open(os.path.join(out, "per_run.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(runs)
    # summary figure: signed offset (carrier node - touch node) per group
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = [k for k in summary if summary[k].get("offsets")]
    if keys:
        fig, ax = plt.subplots(1, len(keys), figsize=(3.6 * len(keys), 3.2), squeeze=False, sharey=True)
        for a_, k in zip(ax[0], keys):
            o = np.array(summary[k]["offsets"])
            b = np.arange(-6, 7)
            a_.bar(b, [np.mean(o == v) for v in b], color="#1f5fbf" if k.startswith("finger-off") else "#999999", width=0.8)
            a_.set_title(f"{k}\nsame node {summary[k]['same_node']:.0%}, within 1 {summary[k]['within1']:.0%}", fontsize=8)
            a_.set_xlabel("carrier node - touch node")
        ax[0][0].set_ylabel("fraction of 0.25 s windows")
        fig.tight_layout(); fig.savefig(os.path.join(out, "offsets.png"), dpi=140); plt.close(fig)
    print(f"{'group':34s} {'win':>5s} {'vis':>5s} {'same':>5s} {'<=1':>5s} {'<=2':>5s} {'|off| p50/p90':>13s} "
          f"{'signed':>6s} {'top3<=2':>7s} {'in v4':>6s} {'runmed err':>10s} {'on err':>7s}")
    for k, s in summary.items():
        if "same_node" not in s:
            print(f"{k:34s} {s['windows']:5d} {s['touch_visible']:5.0%}  (touch not visible)")
            continue
        print(f"{k:34s} {s['windows']:5d} {s['touch_visible']:5.0%} {s['same_node']:5.0%} {s['within1']:5.0%} "
              f"{s['within2']:5.0%} {s['abs_offset_median']:6.1f}/{s['abs_offset_p90']:<6.1f} {s['signed_offset_median']:+6.1f} "
              f"{s['top3_within2']:7.0%} {s['carrier_in_v4']:6.0%} {s['runmed_err_median']:10.2f} {s['on_err_median']:7.2f}")
    print(f"saved {os.path.relpath(out, HERE)}/ (summary.json, per_run.csv, per_window.csv, offsets.png, one png per run)")


if __name__ == "__main__":
    main()
