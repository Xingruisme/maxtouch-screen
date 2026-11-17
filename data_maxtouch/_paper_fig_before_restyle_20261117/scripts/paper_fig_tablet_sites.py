#!/usr/bin/env python3
"""
Paper figure: tablet (maXTouch) through-body BER per TX electrode site at one bit rate (default 40 bps), OOK and FSK,
sitting still, 1 mA, 159.6 kHz carrier, drawn at print size for one column, in the style of fig_tablet_states_40bps.

Data (nothing re-decoded; all uncoded, no FEC):
  Shoulder data_maxtouch/trxber_shoulder2finger_1mA_20261116_150313/summary.csv  (2026-11-16 15:03, coded session of
         stent_trx_ber.py): its uncoded frames ('none_full': 96 bits at one bit per symbol, no FEC) only, 12 frames =
         1152 bits per rate and modulation, decode_trx_tablet.py
  Back   data_maxtouch/modcomp_back2finger_sit_still_20261115_{025506,030045,030352}/summary.json
         3 runs per modulation (2026-11-15 02:55, 03:00, 03:03)
  Knee   data_maxtouch/modcomp_knee2finger_sit_still_20261115_{025153,030750,031056}/summary.json
         3 runs per modulation (2026-11-15 02:51, 03:07, 03:10), pooled with the uncoded frames of
         data_maxtouch/trxber_knee2finger_1mA_20261116_152934 (2026-11-16 15:29, 8 frames = 768 bits)
  Ankle  data_maxtouch/modcomp_conditions_a2f_stand_vs_sit_still_20to60/conditions.csv, 'sitting still' row
         3 runs per modulation (2026-10-30)
  stent_modcomp runs: errors / bits from the stent_modcomp analyses, frozen decode_v4, known timing.
The back session of 2026-11-15 02:53 (modcomp_back2finger_sit_still_20261115_025352) is left out: its only run stopped
after 6 s and decoded no bits. Sites ordered by body path length to the finger, as paper_fig_body_sites.py.
Bar = pooled errors / bits, value printed above it; error bar = lowest to highest per-run BER (paper_runs.py; a
trxber session counts per recording chunk). The csv keeps the bit counts and the Clopper-Pearson 95 % interval.

   python paper_fig_tablet_sites.py [--bps 40]

Also the same counts as a line chart vs distance: BER (log) against the assumed body path to the finger (back 1 m,
knee 1.5 m, ankle 2 m; round figures, not measured), one line per modulation. Sites without a distance (shoulder)
are left out of it.

Output: data_maxtouch/paper_fig_tablet_ber/ (fig_tablet_sites_40bps.{pdf,png,svg}, fig_tablet_sites_40bps.csv,
        fig_tablet_distance_40bps.{pdf,png,svg})
   \\includegraphics[width=\\columnwidth]{fig_tablet_distance_40bps.pdf}   (3.5 in wide, fonts ~9 pt at 100 %)
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import paper_runs as PR
import matplotlib.pyplot as plt
import numpy as np

from paper_fig_tablet_ber import HERE, RC, STYLE, cp
from paper_fig_tablet_snr import FS, FS_TICK, INK

D = os.path.join(HERE, "data_maxtouch")
# (axis label, source): a list of modcomp session / trxber summary folders, or (conditions table, condition prefix)
SITES = [("Shoulder", ["trxber_shoulder2finger_1mA_20261116_150313"]),
         ("Back", [f"modcomp_back2finger_sit_still_20261115_{t}" for t in ("025506", "030045", "030352")]),
         ("Knee", [f"modcomp_knee2finger_sit_still_20261115_{t}" for t in ("025153", "030750", "031056")]
          + ["trxber_knee2finger_1mA_20261116_152934"]),
         ("Ankle", ("modcomp_conditions_a2f_stand_vs_sit_still_20to60", "sitting still"))]
# body path from the TX electrode to the finger on the panel: assumed round figures, not measured
DISTANCE_M = {"Back": 1.0, "Knee": 1.5, "Ankle": 2.0}


def counts(src, m, bps):
    """(errors, bits, runs) for one modulation at one rate"""
    if isinstance(src, tuple):
        table, prefix = src
        hit = [r for r in csv.DictReader(open(os.path.join(D, table, "conditions.csv")))
               if r["condition"].startswith(prefix) and r["modulation"] == m and round(float(r["bps"])) == bps]
        assert len(hit) == 1, (src, m, bps, len(hit))
        return int(hit[0]["errors"]), int(hit[0]["bits"]), len(hit[0]["run"].split("+"))
    k = n = runs = 0
    for s in src:
        if s.startswith("trxber_"):                 # coded stent_trx_ber session: its uncoded frames only
            rows = [r for r in csv.DictReader(open(os.path.join(D, s, "summary.csv")))
                    if r["decoder"] == "none_full" and r["modulation"] == m and round(float(r["baud"])) == bps]
            assert len(rows) == 1, (s, m, bps, len(rows))
            k += int(rows[0]["pre_fec_errors"]); n += int(rows[0]["pre_fec_bits"]); runs += 1
            continue
        rows = [r for r in json.load(open(os.path.join(D, s, "summary.json")))["rows"]
                if r["modulation"] == m and round(r["bps"]) == bps and r.get("bits")]
        assert rows, (s, m, bps)
        k += sum(r["errors"] for r in rows); n += sum(r["bits"] for r in rows); runs += len(rows)
    return k, n, runs


def run_list(src, m, bps):
    """[(errors, bits) per run] behind counts(), for the error bars"""
    if isinstance(src, tuple):
        table, prefix = src
        row = next(r for r in csv.DictReader(open(os.path.join(D, table, "conditions.csv")))
                   if r["condition"].startswith(prefix) and r["modulation"] == m and round(float(r["bps"])) == bps)
        return PR.conditions_row(row)
    out = []
    for s in src:
        out += PR.trx(s, "none_full", m, bps) if s.startswith("trxber_") else PR.sessions([s], m, bps)
    return out


def line_vs_distance(data, bps, out, rr):
    """the same counts as a line chart: BER (log) vs assumed body path to the finger, one line per modulation"""
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(3.5, 2.4), layout="constrained")
        fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02)
        sites = sorted((s for s in SITES if s[0] in DISTANCE_M), key=lambda s: DISTANCE_M[s[0]])
        x = np.array([DISTANCE_M[lab] for lab, _ in sites])
        for m in ("ook", "fsk"):
            st = STYLE[m]
            ber = [data[lab, m][0] / data[lab, m][1] for lab, _ in sites]
            ax.errorbar(x + (-0.01 if m == "ook" else 0.01), ber,
                        yerr=PR.yerr(ber, [PR.spread(rr[lab, m]) for lab, _ in sites]), color=st["color"],
                        marker=st["marker"], ms=4.5, lw=1.3, label=st["label"], zorder=3, **PR.EB)
        ax.set_yscale("log"); ax.set_ylim(1e-4, 1e-1)
        ax.set_xticks(x, [f"{d:g}\n{lab}" for d, (lab, _) in zip(x, sites)])
        ax.set_xlim(x[0] - 0.15, x[-1] + 0.15)
        ax.set_xlabel("Body transmission distance (m)", fontsize=FS, color=INK, labelpad=2)
        ax.set_ylabel(f"Bit error rate, {bps} bps", fontsize=FS, color=INK, labelpad=2)
        ax.tick_params(labelsize=FS_TICK, colors=INK, width=0.6, length=3, pad=2)
        ax.tick_params(which="minor", width=0.5, length=1.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_linewidth(0.6)
        ax.legend(loc="upper left", frameon=False, prop={"size": FS_TICK}, handlelength=1.8, handletextpad=0.4,
                  borderaxespad=0.2)
        for ext in ("pdf", "svg", "png"):
            fig.savefig(f"{out}.{ext}", dpi=600 if ext == "png" else None)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bps", type=int, default=40)
    ap.add_argument("--out", default=os.path.join(D, "paper_fig_tablet_ber"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data = {(lab, m): counts(src, m, a.bps) for lab, src in SITES for m in ("ook", "fsk")}
    rr = {(lab, m): run_list(src, m, a.bps) for lab, src in SITES for m in ("ook", "fsk")}
    for key, rl in rr.items():                          # the per-run lists must add up to the pooled counts
        assert (sum(e for e, _ in rl), sum(n for _, n in rl)) == data[key][:2], key
    name = f"fig_tablet_sites_{a.bps}bps"
    with open(os.path.join(a.out, name + ".csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["site", "modulation", "bps", "runs", "errors", "bits", "ber", "ci95_lo", "ci95_hi"])
        for lab, _ in SITES:
            for m in ("ook", "fsk"):
                k, n, runs = data[lab, m]
                w.writerow([lab.lower(), m, a.bps, runs, k, n, f"{k / n:.3e}", *(f"{x:.3e}" for x in cp(k, n))])

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(3.5, 2.4), layout="constrained")
        fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02)
        x = np.arange(len(SITES))
        wd = 0.34
        top = 0.0
        for j, m in enumerate(("ook", "fsk")):
            st = STYLE[m]
            pct = np.array([100 * data[lab, m][0] / data[lab, m][1] for lab, _ in SITES])
            sp = [PR.spread(rr[lab, m], 100) for lab, _ in SITES]
            hi = np.array([v if q is None else max(v, q[1]) for v, q in zip(pct, sp)])
            top = max(top, hi.max())
            xb = x + (j - 0.5) * (wd + 0.04)
            ax.bar(xb, pct, width=wd, color=st["color"], label=st["label"], zorder=3)
            ax.errorbar(xb, pct, yerr=PR.yerr(pct, sp), fmt="none", ecolor=INK, zorder=4, **PR.EB)
            for xi, v, h in zip(xb, pct, hi):         # value above the bar and its error bar
                ax.text(xi, h + 0.03 * top, f"{v:.2f}", ha="center", va="bottom", fontsize=FS_TICK - 2, color=INK)
        ax.set_xticks(x, [lab for lab, _ in SITES])
        ax.set_ylim(0, top * 1.18)
        ax.set_xlabel("TX electrode location on the body", fontsize=FS, color=INK, labelpad=2)
        ax.set_ylabel(f"Bit error rate (%), {a.bps} bps", fontsize=FS, color=INK, labelpad=2)
        ax.tick_params(labelsize=FS_TICK, colors=INK, width=0.6, length=3, pad=2)
        ax.tick_params(axis="x", length=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_linewidth(0.6)
        ax.legend(loc="upper right", frameon=False, prop={"size": FS_TICK}, handlelength=1.2, handletextpad=0.4,
                  borderaxespad=0.2)
        for ext in ("pdf", "svg", "png"):
            fig.savefig(os.path.join(a.out, f"{name}.{ext}"), dpi=600 if ext == "png" else None)
        plt.close(fig)
    line_vs_distance(data, a.bps, os.path.join(a.out, f"fig_tablet_distance_{a.bps}bps"), rr)
    for lab, _ in SITES:
        print(f"{lab:6s} " + "  ".join(f"{m.upper()} {data[lab, m][0]}/{data[lab, m][1]} = "
                                      f"{data[lab, m][0] / data[lab, m][1]:.2%} ({data[lab, m][2]} run(s))"
                                      for m in ("ook", "fsk")))
    print("->", a.out)


if __name__ == "__main__":
    main()
