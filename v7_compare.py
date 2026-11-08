#!/usr/bin/env python3
"""
Frozen v5 vs experimental v7 (decode_v7.py) on exactly the same recordings, plus ablations and a bit-informed ceiling.
Writes only into data_maxtouch/v7_<role>_<label>/ (never into run folders, never over an earlier comparison).

   python v7_compare.py swipe=<run or session folders ...> [still=...] --label <name>

Groups as touch_vs_carrier.py (NAME=<folder> starts a group, bare folders / '+' join it). v7 is not frozen
(decode_v7.FROZEN_AT = None), so every run is DEVELOPMENT data and the folder says so.

Decoders per run (known start and blind timing each):
  v5               frozen decode_v5.decode_run(write=False) (also re-checked against decode_v7 with v5's settings)
  v7               decode_v7 rules: finger-tracked nodes, sigma around the running median, no z-score
  v7 nodes=v5      ablation: v5's fixed nodes, v7's sigma / feature (what the noise fix alone does)
  v7 sigma=lead    ablation: tracked nodes, v5's lead sigma (what the node tracking alone does, with v7's feature)
  v7 z-score       ablation: v7 with v5's whole-run z-score per node
  oracle nodes     NOT a decoder: nodes = carrier peak node (from the bits, per 0.25 s window) +-1, v7's feature:
                   the ceiling for any node choice with this feature and timing
Figure per run: touch profile (slow running median), estimated finger position, v7's nodes, the carrier peak node
(from the bits, diagnostic), v5's fixed nodes, and bit errors per 0.25 s of v5 and v7 (known timing).
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import sys

import numpy as np
from scipy.stats import beta

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import decode_v2 as V2
import decode_v5 as V5                          # frozen
import decode_v7 as V7
import touch_vs_carrier as TVC

DATADIR = V5.R.DATADIR
WIN = 0.25
VARIANTS = [                                    # name, decode_v7 options
    ("v7", dict()),
    ("v7 nodes=v5", dict(nodes="v4")),
    ("v7 sigma=lead", dict(sigma="lead")),
    ("v7 z-score", dict(zscore=True)),
]


def cp(e, n):
    if not n:
        return None, None
    return (float(beta.ppf(0.025, e, n - e + 1)) if e else 0.0, float(beta.ppf(0.975, e + 1, n - e)) if e < n else 1.0)


def role(d, meta):
    if V7.FROZEN_AT is None:
        return "development"
    frozen = datetime.datetime.strptime(V7.FROZEN_AT, "%Y-%m-%d %H:%M:%S")
    return "validation" if V5.recorded_at(d, meta) > frozen else "development"


def oracle_centres(d, t):
    """per sample: the carrier peak node of its 0.25 s window (bits used), nearest window outside them"""
    meta = json.load(open(os.path.join(d, "meta.json")))
    _, wins = TVC.analyse_run(d)
    tm = TVC.timing(meta, t, np.load(os.path.join(d, "capture.npz"))["X"].astype(float))
    wt = np.array([w["t"] for w in wins]) + tm["t0"]
    cn = np.array([w["carrier_node"] for w in wins])
    i = np.clip(np.searchsorted(wt, t, side="right") - 1, 0, len(wt) - 1)
    return cn[i], wt, cn


def error_times(tr):
    """times of the payload bit errors of the known-timing frames"""
    out = []
    truth = np.array(tr["truth"])
    for f in tr["known_frames"]:
        bad = np.flatnonzero(np.array(f["got"]) != truth)
        out += list(f["s0"] + (tr["hdr"] + bad + 0.5) * tr["Tb"])
    return np.array(out)


def row(cond, d, rl, name, known, blind, extra=None):
    return dict(condition=cond, run=os.path.basename(os.path.normpath(d)), role=rl, decoder=name,
                errors=known["errors"], bits=known["bits"], frames_known=known["frames_valid"],
                frames_sent=known["frames_sent"], blind_frames=blind["frames_valid"], blind_errors=blind["errors"],
                blind_bits=blind["bits"], **{k: known["errors_by_type"][k] for k in ("sent1_read0", "sent0_read1", "in_bursts")},
                **(extra or {}))


def plot_run(d, rl, v5res, tr7, et5, et7, oc, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    meta = json.load(open(os.path.join(d, "meta.json")))
    t = tr7["t"]; t0 = meta["t_first"]
    _, wt, cn = oc
    top = int(max(np.max(tr7["sel"]), cn.max(), max(int(n[1:]) for n in v5res["nodes"]))) + 3
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw=dict(height_ratios=[2.3, 1]))
    B = tr7["B"][:, :top]
    lim = max(300, np.percentile(np.abs(B), 99.5))
    ax[0].pcolormesh(t - t0, np.arange(top), B.T, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="nearest", rasterized=True)
    sel = tr7["sel"]
    ax[0].fill_between(t - t0, sel.min(axis=1) - 0.5, sel.max(axis=1) + 0.5, step="mid", color="#2ca02c", alpha=0.28,
                       lw=0, label="v7 decoding nodes")
    ax[0].plot(t - t0, tr7["pos"], color="black", lw=1.1, label="finger position (bit-free estimate)")
    ax[0].plot(wt - t0, cn, "o", ms=4, mfc="none", mec="#d62728", mew=1.2, label="carrier peak node (from the bits)")
    for i, n in enumerate(v5res["nodes"]):
        ax[0].axhline(int(n[1:]), color="#ff9f1c", lw=1.3, ls="--", label="v5 nodes (fixed)" if i == 0 else None)
    ax[0].set_ylim(-0.5, top - 0.5); ax[0].set_ylabel("Y node (X line 0)")
    ax[0].legend(fontsize=7.5, loc="upper right", ncol=2, framealpha=0.9)
    ax[0].set_title(f"{os.path.basename(os.path.normpath(d))} - {rl.upper()} data, {meta.get('calibration', 'finger-on')} "
                    f"calibration (colour: slow touch profile, running median +-0.25 s)", fontsize=9)
    end = max(t[-1], (et5.max() if len(et5) else 0), (et7.max() if len(et7) else 0))
    edges = np.arange(t0, end + WIN, WIN)
    for et, name, c in ((et5, f"v5 ({len(et5)} errors)", "#ff9f1c"), (et7, f"v7 ({len(et7)} errors)", "#2ca02c")):
        h, _ = np.histogram(et, edges)
        ax[1].step(edges[:-1] - t0, h, where="post", color=c, lw=1.3, label=name)
    ax[1].set_ylabel(f"bit errors per {WIN:g} s"); ax[1].set_xlabel("time from first frame (s)")
    ax[1].legend(fontsize=8, loc="upper right")
    ax[1].set_xlim(-0.5, edges[-1] - t0)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("groups", nargs="+")
    p.add_argument("--label", required=True)
    a = p.parse_args()
    groups = TVC.groups_of(a.groups)
    roles = {d: role(d, json.load(open(os.path.join(d, "meta.json")))) for ds in groups.values() for d in ds}
    tag = "dev" if any(r == "development" for r in roles.values()) else "val"
    out = os.path.join(DATADIR, f"v7_{tag}_{a.label}")
    if os.path.exists(out):
        out += f"_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    os.makedirs(out)
    rows, runinfo = [], []
    for cond, dirs in groups.items():
        for d in sorted(dirs):
            meta = json.load(open(os.path.join(d, "meta.json")))
            if meta.get("modulation") not in ("ook", "fsk"):
                continue
            rl = roles[d]
            name = os.path.basename(os.path.normpath(d))
            v5 = V5.decode_run(d, write=False)
            chk, _ = V7.decode_run(d, nodes="v4", sigma="lead", zscore=True)
            assert chk["known"]["errors"] == v5["combined_known_timing"]["errors"], "v7 with v5 settings != v5"
            rows.append(row(cond, d, rl, "v5", dict(v5["combined_known_timing"]), dict(v5["combined_blind"]),
                            dict(nodes=" ".join(v5["nodes"]))))
            tr7 = None
            for vname, kw in VARIANTS:
                r, tr = V7.decode_run(d, **kw)
                if vname == "v7":
                    tr7, r7 = tr, r
                rows.append(row(cond, d, rl, vname, r["known"], r["blind"],
                                dict(node_switches=r["node_switches"], switches_per_s=r["switches_per_s"])))
            oc = oracle_centres(d, tr7["t"])
            r, _ = V7.decode_run(d, nodes="oracle", oracle_centre=oc[0])
            rows.append(row(cond, d, rl, "oracle nodes (uses bits)", r["known"], r["blind"],
                            dict(node_switches=r["node_switches"], switches_per_s=r["switches_per_s"])))
            # v5's per-bit errors for the error-time panel: same frames as v5 (decode_v7 with v5's settings)
            _, tr5 = V7.decode_run(d, nodes="v4", sigma="lead", zscore=True)
            et5, et7 = error_times(tr5), error_times(tr7)
            # how often v7's / v5's nodes contain the carrier peak (per 0.25 s window, bits used for the check only)
            _, wt, cn = oc
            k = np.clip(np.searchsorted(tr7["t"], wt + WIN / 2), 0, len(tr7["t"]) - 1)
            in7 = float(np.mean([c in s for c, s in zip(cn, tr7["sel"][k])]))
            in5 = float(np.mean([f"Y{c}" in v5["nodes"] for c in cn]))
            runinfo.append(dict(condition=cond, run=name, role=rl, modulation=meta["modulation"], bps=v5["bps"],
                                calibration=r7["calibration"], calibration_place=r7["calibration_place"],
                                v5_nodes=" ".join(v5["nodes"]), carrier_peak_in_v5_nodes=in5, carrier_peak_in_v7_nodes=in7,
                                v7_switches_per_s=r7["switches_per_s"], position_from=r7["position_from"]))
            plot_run(d, rl, v5, tr7, et5, et7, oc, os.path.join(out, f"{name}.png"))
            print(f"  {name[10:]:50s} [{rl}] carrier peak in nodes: v5 {in5:.0%}, v7 {in7:.0%}")
    with open(os.path.join(out, "per_run.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader(); w.writerows(rows)
    summ = []
    for cond in groups:
        for dec in dict.fromkeys(r["decoder"] for r in rows):
            rr = [r for r in rows if r["condition"] == cond and r["decoder"] == dec]
            if not rr:
                continue
            e, n = sum(r["errors"] for r in rr), sum(r["bits"] for r in rr)
            be, bn = sum(r["blind_errors"] for r in rr), sum(r["blind_bits"] for r in rr)
            lo, hi = cp(e, n)
            summ.append(dict(condition=cond, decoder=dec, runs=len(rr), roles="+".join(sorted({r["role"] for r in rr})),
                             errors=e, bits=n, ber=e / n if n else None, ber_lo95=lo, ber_hi95=hi,
                             sent1_read0=sum(r["sent1_read0"] for r in rr), sent0_read1=sum(r["sent0_read1"] for r in rr),
                             blind_frames=sum(r["blind_frames"] for r in rr), frames_sent=sum(r["frames_sent"] for r in rr),
                             blind_errors=be, blind_bits=bn, blind_ber=be / bn if bn else None))
    with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0])); w.writeheader(); w.writerows(summ)
    json.dump(dict(runs=runinfo, summary=summ, v7_rules=dict(VIS=V7.VIS, NEG=V7.NEG, FRAC=V7.FRAC, HYST=V7.HYST, N=V7.N,
                                                             base_half_s=V7.BASE_HALF_S),
                   v5_rules_sha=V5.RULES_SHA, v4_rules_sha=V5.V4_SHA, v7_frozen_at=V7.FROZEN_AT),
              open(os.path.join(out, "summary.json"), "w"), indent=1, default=str)
    print(f"\n{'condition':8s} {'decoder':26s} {'role':12s} {'errors/bits':>11s} {'BER (95 % CI)':>24s} "
          f"{'1->0':>5s} {'0->1':>5s} {'blind fr':>9s} {'blind BER':>9s}")
    for s in summ:
        print(f"{s['condition']:8s} {s['decoder']:26s} {s['roles']:12s} {s['errors']:5d}/{s['bits']:<5d} "
              f"{100 * s['ber']:6.2f} % ({100 * s['ber_lo95']:5.2f}-{100 * s['ber_hi95']:5.2f}) {s['sent1_read0']:5d} "
              f"{s['sent0_read1']:5d} {s['blind_frames']:4d}/{s['frames_sent']:<4d} "
              + (f"{100 * s['blind_ber']:8.2f} %" if s["blind_ber"] is not None else "        -"))
    print(f"saved {os.path.relpath(out, HERE)}/ (summary.csv, per_run.csv, summary.json, one figure per run)")


if __name__ == "__main__":
    main()
