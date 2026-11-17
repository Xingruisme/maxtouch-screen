#!/usr/bin/env python3
"""
Paper figure: tablet (maXTouch) through-body link vs bit rate, static sessions (sitting + standing still), OOK and FSK,
as one single-column figure with the two panels side by side (3.5 x 2.1 in, the layout of the paper's Figure 13:
bold a / b at each panel's top left, legend inside panel a) at print size:
  a  bit error rate   - as fig_tablet_ber_column (paper_fig_tablet_ber_column.py): pooled errors / bits of the 6 runs
                        per point (6096 bits); error bar = lowest to highest per-run BER (paper_runs.py), down to the
                        axis where a run had no error
  b  symbol SNR       - as fig_tablet_snr_column (paper_fig_tablet_snr.py): median of the 6 recordings; error bar =
                        lowest to highest recording; small open dots = the 6 recordings
OOK and FSK drawn 1 bps apart in both panels (equal or close at several rates).
Same data and processing as those two scripts (nothing re-decoded): the 'static' rows of
data_maxtouch/modcomp_conditions_a2f_static_vs_moving_20to60/conditions.csv, ankle to finger, 1 mA, 159.6 kHz.

   python paper_fig_tablet_ber_snr.py

Output: data_maxtouch/paper_fig_tablet_ber/ (fig_tablet_ber_snr.{pdf,png,svg})
   \\includegraphics[width=\\columnwidth]{fig_tablet_ber_snr.pdf}   (3.5 in wide, fonts 6-7 pt at 100 %)
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

import paper_style as PS
from paper_fig_tablet_ber import HERE, RC, STYLE, per_run_snr
from paper_fig_tablet_snr import SRC
from paper_style import FS, FS_TICK, INK        # pt at print size; the two-panel figure scripts take them from here

MODS = ("ook", "fsk")


def tidy(ax, rates, ylabel, letter=None):
    """bit-rate axis, labels, open axes and the panel's bold letter above its top left corner"""
    ax.set_xticks(rates); ax.set_xlim(rates[0] - 4, rates[-1] + 4)
    ax.set_xlabel("Bit rate (bps)")
    ax.set_ylabel(ylabel)
    PS.finish(ax)
    if letter:
        PS.panel(ax, letter)


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
        fig, (ax_b, ax_s) = PS.figure(3.5, 2.1, 1, 2, wspace=0.08)
        for m, side in (("ook", -1), ("fsk", 1)):    # OOK and FSK BERs are equal at 50 and 60 bps: drawn apart
            st = STYLE[m]
            kw = PS.marks(st["color"], st["marker"], ms=PS.MS - 0.4)
            x = np.array(rates, float) + 0.8 * side
            PS.series(ax_b, x, [ber[m, b][0] / ber[m, b][1] for b in rates], [PR.spread(runs[m, b]) for b in rates],
                      label=st["label"], **kw)
            PS.series(ax_s, x, [np.median(snr[m, b]) for b in rates],
                      [(min(snr[m, b]), max(snr[m, b])) for b in rates], **kw)
            for xi, b in zip(x, rates):                 # the recordings behind each median, beside its range bar
                PS.run_dots(ax_s, xi + 1.9 * side, snr[m, b], st["color"], ms=1.6)
        ax_b.set_yscale("log"); ax_b.set_ylim(1e-4, 1e-1)
        ax_s.set_ylim(30, 70); ax_s.set_yticks(range(30, 71, 10))
        tidy(ax_b, rates, "Bit error rate", "a")
        tidy(ax_s, rates, "Symbol SNR (dB)", "b")
        ax_b.legend(loc="upper left")
        PS.save(fig, os.path.join(a.out, "fig_tablet_ber_snr"))
    for m in MODS:
        print(m, "  ".join(f"{b} bps: BER {ber[m, b][0]}/{ber[m, b][1]}, SNR {np.median(snr[m, b]):.1f} dB" for b in rates))
    print("->", os.path.join(a.out, "fig_tablet_ber_snr.pdf"))


if __name__ == "__main__":
    main()
