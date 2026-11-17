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
   \\includegraphics[width=\\columnwidth]{fig_tablet_ber_column.pdf}   (3.5 in wide, fonts 6-7 pt at 100 %)
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
from paper_fig_tablet_ber import HERE, RC, STYLE, cp
from paper_fig_tablet_snr import SRC


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
        fig, ax = PS.figure(3.5, 2.4)
        for m, side in (("ook", -1), ("fsk", 1)):   # OOK and FSK are equal at 50 and 60 bps: drawn 1.2 bps apart
            st = STYLE[m]
            PS.series(ax, np.array(rates, float) + 0.6 * side, [data[m, b][0] / data[m, b][1] for b in rates],
                      [PR.spread(runs[m, b]) for b in rates], **PS.marks(st["color"], st["marker"], label=st["label"]))
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1e-1)
        ax.set_xticks(rates); ax.set_xlim(rates[0] - 3, rates[-1] + 3)
        ax.set_xlabel("Bit rate (bps)"); ax.set_ylabel("Bit error rate")
        PS.finish(ax)
        ax.legend(loc="upper left")
        PS.save(fig, os.path.join(a.out, "fig_tablet_ber_column"))
    for (m, b), (k, n) in sorted(data.items()):
        print(f"{m} {b} bps: {k}/{n} = {k / n:.2%}")
    print("->", a.out)


if __name__ == "__main__":
    main()
