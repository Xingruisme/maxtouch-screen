#!/usr/bin/env python3
"""
Paper figures: tablet (maXTouch) through-body link vs bit rate, sitting still vs standing still, OOK and FSK, drawn at
print size for one column (the layout of the slide 'Comparison of gestures of static condition').

Data = the standing-still vs sitting-still table made by `stent_modcomp.py conditions`
(data_maxtouch/modcomp_conditions_a2f_stand_vs_sit_still_20to60/conditions.csv): per posture the latest 3 runs per
rate and modulation (20/50 bps 2026-10-29, 40/60 bps 2026-10-30), ankle to finger, 1 mA, 159.6 kHz carrier, 3048 bits
per point; frozen decode_v4, known timing. Nothing is re-decoded.
  SNR figure: symbol SNR per recording from the sessions' summary.json, marker = median of the 3 recordings, error
              bar = lowest to highest recording.
  BER figure: pooled errors / bits of the 3 recordings, error bar = lowest to highest per-run BER (paper_runs.py),
              down to the axis where a run had no error (the csv keeps the Clopper-Pearson 95 % interval).
Colour = posture, line style and marker = modulation; the four series are drawn 0.6 bps apart so none hides another.
A point with 0 errors has no BER on a log axis: it is drawn as a hollow downward triangle at its 95 % upper bound
(about 3/3048), not joined to the line.

   python paper_fig_tablet_static.py

Output: data_maxtouch/paper_fig_tablet_ber/ (fig_tablet_snr_static.*, fig_tablet_ber_static.*, fig_tablet_static.csv)
   \\includegraphics[width=\\columnwidth]{fig_tablet_ber_static.pdf}   (3.5 in wide, fonts ~9 pt at 100 %)
"""
from __future__ import annotations

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import paper_runs as PR
import matplotlib.pyplot as plt
import numpy as np

from paper_fig_tablet_ber import HERE, RC, cp, per_run_snr
from paper_fig_tablet_snr import FS, FS_TICK, INK

SRC = os.path.join(HERE, "data_maxtouch", "modcomp_conditions_a2f_stand_vs_sit_still_20to60", "conditions.csv")
POSTURE = {"sitting": dict(prefix="sitting still", color="#1f5fbf", label="sitting"),
           "standing": dict(prefix="standing still", color="#D55E00", label="standing")}
MOD = {"ook": dict(ls="-", marker="s", mfc=None, label="OOK"),
       "fsk": dict(ls=(0, (4, 2)), marker="^", mfc="white", label="FSK")}
SERIES = [(p, m) for p in POSTURE for m in MOD]
OFF = dict(zip(SERIES, (-0.9, -0.3, 0.3, 0.9)))


def style(p, m):
    c = POSTURE[p]["color"]
    return dict(color=c, ls=MOD[m]["ls"], marker=MOD[m]["marker"], mfc=MOD[m]["mfc"] or c, mec=c, mew=0.9, ms=4,
                lw=1.2, label=f"{MOD[m]['label']}, {POSTURE[p]['label']}")


def axes(ax, rates, ylabel):
    ax.set_xticks(rates); ax.set_xlim(rates[0] - 3, rates[-1] + 3)
    ax.set_xlabel("Bit rate (bps)", fontsize=FS, color=INK, labelpad=2)
    ax.set_ylabel(ylabel, fontsize=FS, color=INK, labelpad=2)
    ax.tick_params(labelsize=FS_TICK, colors=INK, width=0.6, length=3, pad=2)
    ax.tick_params(which="minor", width=0.5, length=1.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.6)


def figure(out, draw, ylabel):
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(3.5, 2.6), layout="constrained")
        fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02)
        rates = draw(ax)
        axes(ax, rates, ylabel)
        fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncol=2, frameon=False,
                   prop={"size": FS_TICK}, handlelength=2.4, columnspacing=1.2, handletextpad=0.4, borderaxespad=0.1)
        for ext in ("pdf", "svg", "png"):
            fig.savefig(f"{out}.{ext}", dpi=600 if ext == "png" else None)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=os.path.join(HERE, "data_maxtouch", "paper_fig_tablet_ber"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = list(csv.DictReader(open(a.src)))
    ber, snr, runs = {}, {}, {}
    for p, st in POSTURE.items():
        mine = [r for r in rows if r["condition"].startswith(st["prefix"])]
        assert mine, f"no '{st['prefix']}' rows in {a.src}"
        for r in mine:
            ber[p, r["modulation"], round(float(r["bps"]))] = (int(r["errors"]), int(r["bits"]))
            runs[p, r["modulation"], round(float(r["bps"]))] = PR.conditions_row(r)
        for (m, b), v in per_run_snr(mine).items():
            snr[p, m, b] = v
    rates = sorted({b for _, _, b in ber})
    with open(os.path.join(a.out, "fig_tablet_static.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["posture", "modulation", "bps", "errors", "bits", "ber", "ci95_lo", "ci95_hi", "snr_db_median",
                    "snr_db_values"])
        for (p, m, b), (k, n) in sorted(ber.items()):
            w.writerow([p, m, b, k, n, f"{k / n:.3e}", *(f"{x:.3e}" for x in cp(k, n)),
                        f"{np.median(snr[p, m, b]):.2f}", " ".join(f"{x:.2f}" for x in snr[p, m, b])])

    def draw_snr(ax):
        for p, m in SERIES:
            y = [np.median(snr[p, m, b]) for b in rates]
            ax.errorbar(np.array(rates, float) + OFF[p, m], y,
                        yerr=PR.yerr(y, [(min(snr[p, m, b]), max(snr[p, m, b])) for b in rates]), zorder=3,
                        **style(p, m), **PR.EB)
        ax.set_ylim(30, 70); ax.set_yticks(range(30, 71, 10))
        return rates

    def draw_ber(ax):
        for p, m in SERIES:
            k = np.array([ber[p, m, b][0] for b in rates]); n = np.array([ber[p, m, b][1] for b in rates])
            lo, hi = np.array([cp(*ber[p, m, b]) for b in rates]).T
            x = np.array(rates, float) + OFF[p, m]
            # 0 errors cannot sit on a log axis: no point and no line there, a downward marker at the 95 % upper bound
            zero = k == 0
            y = np.where(zero, np.nan, k / n)
            sp = [None if z else PR.spread(runs[p, m, b]) for z, b in zip(zero, rates)]
            ax.errorbar(x, y, yerr=PR.yerr(np.nan_to_num(y), sp), zorder=3, **style(p, m), **PR.EB)
            if zero.any():
                ax.plot(x[zero], hi[zero], ls="none", marker="v", ms=4.5, color=POSTURE[p]["color"], mfc="white",
                        mew=0.9, zorder=4)
        # key for the 0-error marker inside the plot; the four series go in the figure legend
        if any(v[0] == 0 for v in ber.values()):
            zero_key = [plt.Line2D([], [], ls="none", marker="v", ms=4.5, color="#52514e", mfc="white", mew=0.9)]
            ax.add_artist(ax.legend(zero_key, ["0 errors (drawn at 95 % upper bound)"], loc="upper left",
                                    frameon=False, prop={"size": FS_TICK - 1}, handletextpad=0.3, borderaxespad=0.2))
        ax.set_yscale("log"); ax.set_ylim(1e-4, 1e-1)
        return rates

    figure(os.path.join(a.out, "fig_tablet_snr_static"), draw_snr, "Symbol SNR (dB)")
    figure(os.path.join(a.out, "fig_tablet_ber_static"), draw_ber, "Bit error rate")
    for (p, m, b), (k, n) in sorted(ber.items()):
        print(f"{p:8s} {m} {b} bps: {k}/{n} = {k / n:.2%}   SNR median {np.median(snr[p, m, b]):.1f} dB "
              f"({' '.join(f'{x:.1f}' for x in snr[p, m, b])})")
    print("->", a.out)


if __name__ == "__main__":
    main()
