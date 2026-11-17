#!/usr/bin/env python3
"""
Paper figure: tablet (maXTouch) through-body SNR vs bit rate, OOK and FSK, drawn at print size for one column.

Data = static sessions only: the 'static' rows of the static-vs-moving table made by `stent_modcomp.py conditions`
(sitting still 6 + standing still 6 sessions, ankle to finger, 1 mA; 6 recordings per rate and modulation, 6096 bits);
moving sessions (standing walking, sitting shaking) left out. SNR per recording read from the sessions' summary.json
(nothing re-decoded):
  SNR = symbol Es/N0 = k P1 / (2 sigma^2) (stent_modcomp.es_n0_db): k = panel samples per bit, P1 = carrier power of the
  sent-1 samples over the OFF baseline, sigma = OFF noise of the top node.
Markers = median of the 6 recordings, error bar = lowest to highest recording (the csv keeps quartiles, min, max).
Note: 20 and 50 bps were recorded on 2026-10-29, 40 and 60 bps on 2026-10-30, so the zig-zag between neighbouring
rates is mostly the day, not the rate.

   python paper_fig_tablet_snr.py [--src data_maxtouch/modcomp_conditions_a2f_static_vs_moving_20to60/conditions.csv]
                                  [--condition static]   (rows whose condition starts with this)

Output: data_maxtouch/paper_fig_tablet_ber/ (fig_tablet_snr_column.{pdf,png,svg}, fig_tablet_snr_column.csv)
   \\includegraphics[width=\\columnwidth]{fig_tablet_snr_column.pdf}   (3.5 in wide, fonts ~9 pt at 100 %)
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

from paper_fig_tablet_ber import HERE, RC, STYLE, per_run_snr

INK = "#1a1a1a"
FS, FS_TICK = 9, 8                              # pt at print size
SRC = os.path.join(HERE, "data_maxtouch", "modcomp_conditions_a2f_static_vs_moving_20to60", "conditions.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--condition", default="static", help="keep the rows whose condition starts with this")
    ap.add_argument("--out", default=os.path.join(HERE, "data_maxtouch", "paper_fig_tablet_ber"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = [r for r in csv.DictReader(open(a.src)) if r["condition"].startswith(a.condition)]
    assert rows, f"no '{a.condition}' rows in {a.src}"
    per = per_run_snr(rows)
    rates = sorted({b for _, b in per})
    q = {k: np.percentile(v, [25, 50, 75]) for k, v in per.items()}            # quartiles + median
    with open(os.path.join(a.out, "fig_tablet_snr_column.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["modulation", "bps", "n_recordings", "snr_db_median", "q25", "q75", "min", "max",
                                       "values"])
        for (m, b), v in sorted(per.items()):
            w.writerow([m, b, len(v), *(f"{x:.2f}" for x in (q[m, b][1], q[m, b][0], q[m, b][2], min(v), max(v))),
                        " ".join(f"{x:.2f}" for x in v)])

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(3.5, 2.4), layout="constrained")
        fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02)
        x = np.array(rates, float)
        for m in ("ook", "fsk"):
            st = STYLE[m]
            med = [q[m, b][1] for b in rates]
            ax.errorbar(x + (-0.5 if m == "ook" else 0.5), med,
                        yerr=PR.yerr(med, [(min(per[m, b]), max(per[m, b])) for b in rates]), color=st["color"],
                        marker=st["marker"], ms=4.5, lw=1.3, label=st["label"], zorder=3, **PR.EB)
        ax.set_xticks(rates); ax.set_xlim(rates[0] - 3, rates[-1] + 3)
        ax.set_ylim(30, 70); ax.set_yticks(range(30, 71, 10))
        ax.set_xlabel("Bit rate (bps)", fontsize=FS, color=INK, labelpad=2)
        ax.set_ylabel("Symbol SNR (dB)", fontsize=FS, color=INK, labelpad=2)
        ax.tick_params(labelsize=FS_TICK, colors=INK, width=0.6, length=3, pad=2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_linewidth(0.6)
        ax.legend(loc="upper right", frameon=False, prop={"size": FS_TICK}, handlelength=1.8, handletextpad=0.4,
                  borderaxespad=0.2)
        for ext in ("pdf", "svg", "png"):
            fig.savefig(os.path.join(a.out, f"fig_tablet_snr_column.{ext}"), dpi=600 if ext == "png" else None)
        plt.close(fig)
    for (m, b), v in sorted(per.items()):
        print(f"{m} {b} bps: median {np.median(v):.1f} dB ({' '.join(f'{x:.1f}' for x in v)})")
    print("->", a.out)


if __name__ == "__main__":
    main()
