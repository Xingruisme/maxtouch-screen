#!/usr/bin/env python3
"""
Paper figure: tablet (maXTouch) through-body BER vs bit rate, OOK and FSK, drawn at print size for one column, in the
style of fig_tablet_snr_column (paper_fig_tablet_snr.py).

Data = the same static sessions as that figure: the 'static' rows of the static-vs-moving table made by
`stent_modcomp.py conditions` (sitting still 6 + standing still 6 sessions, ankle to finger, 1 mA, 159.6 kHz carrier;
6 recordings = 6096 bits per rate and modulation; frozen decode_v4, known timing). Nothing is re-decoded.
BER per point = pooled errors / pooled bits of the 6 recordings (sitting and standing still merged); error bar =
lowest to highest per-run BER (paper_runs.py), down to the axis where a run had no error. The csv keeps the
Clopper-Pearson 95 % interval of the pooled count.
OOK and FSK drawn 0.5 bps apart: their BERs are identical at 50 and 60 bps.

   python paper_fig_tablet_ber_column.py

Output: data_maxtouch/paper_fig_tablet_ber/ (fig_tablet_ber_column.{pdf,png,svg}, fig_tablet_ber_column.csv)
   \\includegraphics[width=\\columnwidth]{fig_tablet_ber_column.pdf}   (3.5 in wide, fonts ~9 pt at 100 %)
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

from paper_fig_tablet_ber import HERE, RC, STYLE, cp
from paper_fig_tablet_snr import FS, FS_TICK, INK, SRC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--condition", default="static", help="keep the rows whose condition starts with this")
    ap.add_argument("--out", default=os.path.join(HERE, "data_maxtouch", "paper_fig_tablet_ber"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = [r for r in csv.DictReader(open(a.src)) if r["condition"].startswith(a.condition)]
    assert rows, f"no '{a.condition}' rows in {a.src}"
    data = {(r["modulation"], round(float(r["bps"]))): (int(r["errors"]), int(r["bits"])) for r in rows}
    runs = {(r["modulation"], round(float(r["bps"]))): PR.conditions_row(r) for r in rows}
    rates = sorted({b for _, b in data})
    with open(os.path.join(a.out, "fig_tablet_ber_column.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["modulation", "bps", "errors", "bits", "ber", "ci95_lo", "ci95_hi"])
        for (m, b), (k, n) in sorted(data.items()):
            w.writerow([m, b, k, n, f"{k / n:.3e}", *(f"{x:.3e}" for x in cp(k, n))])

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(3.5, 2.4), layout="constrained")
        fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02)
        off = {"ook": -0.5, "fsk": 0.5}          # OOK and FSK are equal at 50 and 60 bps: keep both markers visible
        for m in ("ook", "fsk"):
            st = STYLE[m]
            ber = [data[m, b][0] / data[m, b][1] for b in rates]
            ax.errorbar(np.array(rates, float) + off[m], ber, yerr=PR.yerr(ber, [PR.spread(runs[m, b]) for b in rates]),
                        color=st["color"], marker=st["marker"], ms=4.5, lw=1.3, label=st["label"], zorder=3, **PR.EB)
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1e-1)
        ax.set_xticks(rates); ax.set_xlim(rates[0] - 3, rates[-1] + 3)
        ax.set_xlabel("Bit rate (bps)", fontsize=FS, color=INK, labelpad=2)
        ax.set_ylabel("Bit error rate", fontsize=FS, color=INK, labelpad=2)
        ax.tick_params(labelsize=FS_TICK, colors=INK, width=0.6, length=3, pad=2)
        ax.tick_params(which="minor", width=0.5, length=1.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_linewidth(0.6)
        ax.legend(loc="upper left", frameon=False, prop={"size": FS_TICK}, handlelength=1.8, handletextpad=0.4,
                  borderaxespad=0.2)
        for ext in ("pdf", "svg", "png"):
            fig.savefig(os.path.join(a.out, f"fig_tablet_ber_column.{ext}"), dpi=600 if ext == "png" else None)
        plt.close(fig)
    for (m, b), (k, n) in sorted(data.items()):
        print(f"{m} {b} bps: {k}/{n} = {k / n:.2%}")
    print("->", a.out)


if __name__ == "__main__":
    main()
