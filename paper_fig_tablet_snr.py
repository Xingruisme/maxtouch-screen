#!/usr/bin/env python3
"""
Paper figure: tablet (maXTouch) through-body SNR vs bit rate, OOK and FSK, drawn at print size for one column.

Data = static sessions only: the 'static' rows of the static-vs-moving table made by `stent_modcomp.py conditions`
(sitting still 6 + standing still 6 sessions, ankle to finger, 1 mA; 6 recordings per rate and modulation, 6096 bits);
moving sessions (standing walking, sitting shaking) left out. SNR per recording read from the sessions' summary.json
(nothing re-decoded):
  SNR = symbol Es/N0 = k P1 / (2 sigma^2) (stent_modcomp.es_n0_db): k = panel samples per bit, P1 = carrier power of the
  sent-1 samples over the OFF baseline, sigma = OFF noise of the top node.
Markers = median of the 6 recordings, error bar = lowest to highest recording, small open dots = the 6 recordings
(the csv keeps quartiles, min, max).
Note: 20 and 50 bps were recorded on 2026-10-29, 40 and 60 bps on 2026-10-30, so the zig-zag between neighbouring
rates is mostly the day, not the rate.

   python paper_fig_tablet_snr.py [--src data_maxtouch/modcomp_conditions_a2f_static_vs_moving_20to60/conditions.csv]
                                  [--condition static]   (rows whose condition starts with this)

Output: data_maxtouch/paper_fig_tablet_ber/ (fig_tablet_snr_column.{pdf,png,svg}, fig_tablet_snr_column.csv)
   \\includegraphics[width=\\columnwidth]{fig_tablet_snr_column.pdf}   (3.5 in wide, fonts 6-7 pt at 100 %)
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
from paper_style import FS, FS_TICK, INK        # pt at print size; the other figure scripts take them from here
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
        fig, ax = PS.figure(3.5, 2.4)
        for m, side in (("ook", -1), ("fsk", 1)):
            st = STYLE[m]
            x = np.array(rates, float) + 0.6 * side
            PS.series(ax, x, [q[m, b][1] for b in rates], [(min(per[m, b]), max(per[m, b])) for b in rates],
                      **PS.marks(st["color"], st["marker"], label=st["label"]))
            for xi, b in zip(x, rates):                 # the recordings behind each median, beside its range bar
                PS.run_dots(ax, xi + 1.0 * side, per[m, b], st["color"])
        ax.set_xticks(rates); ax.set_xlim(rates[0] - 3, rates[-1] + 3)
        ax.set_ylim(30, 70); ax.set_yticks(range(30, 71, 10))
        ax.set_xlabel("Bit rate (bps)"); ax.set_ylabel("Symbol SNR (dB)")
        PS.finish(ax)
        ax.legend(loc="upper right")
        PS.save(fig, os.path.join(a.out, "fig_tablet_snr_column"))
    for (m, b), v in sorted(per.items()):
        print(f"{m} {b} bps: median {np.median(v):.1f} dB ({' '.join(f'{x:.1f}' for x in v)})")
    print("->", a.out)


if __name__ == "__main__":
    main()
