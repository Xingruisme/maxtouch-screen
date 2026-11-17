#!/usr/bin/env python3
"""
Paper figure: tablet (maXTouch) through-body link vs bit rate, static sessions (sitting + standing still), OOK and FSK,
as one single-column figure with the two panels side by side (3.5 x 2.1 in, the layout of the paper's Figure 13:
bold a / b at each panel's top left, legend inside panel a) at print size:
  a  bit error rate   - as fig_tablet_ber_column (paper_fig_tablet_ber_column.py): pooled errors / bits of the 6 runs
                        per point (6096 bits); error bar = lowest to highest per-run BER (paper_runs.py), down to the
                        axis where a run had no error
  b  symbol SNR       - as fig_tablet_snr_column (paper_fig_tablet_snr.py): median of the 6 recordings; error bar =
                        lowest to highest recording
OOK and FSK drawn 1 bps apart in both panels (equal or close at several rates).
Same data and processing as those two scripts (nothing re-decoded): the 'static' rows of
data_maxtouch/modcomp_conditions_a2f_static_vs_moving_20to60/conditions.csv, ankle to finger, 1 mA, 159.6 kHz.

   python paper_fig_tablet_ber_snr.py

Output: data_maxtouch/paper_fig_tablet_ber/ (fig_tablet_ber_snr.{pdf,png,svg})
   \\includegraphics[width=\\columnwidth]{fig_tablet_ber_snr.pdf}   (3.5 in wide, fonts 7-8 pt at 100 %)
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
from paper_fig_tablet_snr import INK, SRC

MODS = ("ook", "fsk")
FS, FS_TICK = 8, 7                              # pt at print size: two panels share one 3.5 in column


def tidy(ax, rates, ylabel, letter):
    ax.set_xticks(rates); ax.set_xlim(rates[0] - 3, rates[-1] + 3)
    ax.set_xlabel("Bit rate (bps)", fontsize=FS, color=INK, labelpad=1)
    ax.set_ylabel(ylabel, fontsize=FS, color=INK, labelpad=1)
    # panel letter in bold at the top left, outside the axes (as the paper's other figures)
    ax.text(-0.30, 1.04, letter, transform=ax.transAxes, fontsize=FS + 2, fontweight="bold", color=INK,
            ha="left", va="bottom")
    ax.tick_params(labelsize=FS_TICK, colors=INK, width=0.6, length=2.5, pad=1.5)
    ax.tick_params(which="minor", width=0.5, length=1.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--condition", default="static", help="keep the rows whose condition starts with this")
    ap.add_argument("--out", default=os.path.join(HERE, "data_maxtouch", "paper_fig_tablet_ber"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = [r for r in csv.DictReader(open(a.src)) if r["condition"].startswith(a.condition)]
    assert rows, f"no '{a.condition}' rows in {a.src}"
    ber = {(r["modulation"], round(float(r["bps"]))): (int(r["errors"]), int(r["bits"])) for r in rows}
    runs = {(r["modulation"], round(float(r["bps"]))): PR.conditions_row(r) for r in rows}
    snr = per_run_snr(rows)
    rates = sorted({b for _, b in ber})

    with plt.rc_context(RC):
        fig, (ax_b, ax_s) = plt.subplots(1, 2, figsize=(3.5, 2.1), layout="constrained")
        fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.08)
        off = {"ook": -0.5, "fsk": 0.5}          # OOK and FSK BERs are equal at 50 and 60 bps: keep both markers visible
        for m in MODS:
            st = STYLE[m]
            kw = dict(color=st["color"], marker=st["marker"], ms=3.5, lw=1.1, label=st["label"], zorder=3, **PR.EB)
            x = np.array(rates, float) + off[m]
            y = [ber[m, b][0] / ber[m, b][1] for b in rates]
            ax_b.errorbar(x, y, yerr=PR.yerr(y, [PR.spread(runs[m, b]) for b in rates]), **kw)
            kw.pop("label")
            ys = [np.median(snr[m, b]) for b in rates]
            ax_s.errorbar(x, ys, yerr=PR.yerr(ys, [(min(snr[m, b]), max(snr[m, b])) for b in rates]), **kw)
        ax_b.set_yscale("log"); ax_b.set_ylim(1e-4, 1e-1)
        ax_s.set_ylim(30, 70); ax_s.set_yticks(range(30, 71, 10))
        tidy(ax_b, rates, "Bit error rate", "a")
        tidy(ax_s, rates, "Symbol SNR (dB)", "b")
        ax_b.legend(loc="upper left", frameon=False, prop={"size": FS_TICK}, handlelength=1.6, handletextpad=0.4,
                    borderaxespad=0.2, labelspacing=0.3)
        for ext in ("pdf", "svg", "png"):
            fig.savefig(os.path.join(a.out, f"fig_tablet_ber_snr.{ext}"), dpi=600 if ext == "png" else None)
        plt.close(fig)
    for m in MODS:
        print(m, "  ".join(f"{b} bps: BER {ber[m, b][0]}/{ber[m, b][1]}, SNR {np.median(snr[m, b]):.1f} dB" for b in rates))
    print("->", os.path.join(a.out, "fig_tablet_ber_snr.pdf"))


if __name__ == "__main__":
    main()
