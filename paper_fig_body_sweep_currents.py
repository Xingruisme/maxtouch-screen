#!/usr/bin/env python3
"""
Paper figure: through-body frequency sweep at the tablet (maXTouch), ankle to finger, at chosen set currents (default
1 / 2.5 / 4 mA), one panel at print size for one column. A single-site companion of paper_fig_body_sweeps.py, with
its colours, markers and metric.

Data = the 2026-10-26 ankle-to-finger sweeps in data_maxtouch/body_sweep_currents_20261026.json (runs and 'carrier
detected' verdicts from body_current_sweeps.py), re-analysed with maxtouch_stent_rx.analyse_probe (nothing
re-recorded). Sweeps with no carrier detected (4 mA: TX configuration fault) are dashed.
The 2026-10-24 sweeps at the same currents are not used: they fall in the window when the transmitter output was
~40 dB low (fsk_tone_survey.TX_FAULT).
Metric per carrier (cap_ook_rx.analyse_probe 'energy_db'): 10 log10 of the mean squared scan-to-scan change of the
node's delta in that carrier's ON blocks over its interleaved OFF blocks; best node of Y0-Y3.
Dashed rule = 159.585 kHz, the carrier of every BER run.

   python paper_fig_body_sweep_currents.py [--currents 1,2.5,4]

Output: data_maxtouch/paper_fig_body_sweeps/ (fig_body_sweep_ankle_currents.{pdf,png,svg}, ...csv)
   \\includegraphics[width=\\columnwidth]{fig_body_sweep_ankle_currents.pdf}   (3.5 in wide, fonts 6-7 pt at 100 %)
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import paper_style as PS
from paper_fig_body_sweeps import CARRIER, CUR, D, cur_key, sweep
SRC = os.path.join(D, "body_sweep_currents_20261026.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--currents", default="1,2.5,4", help="set currents [mA] to draw")
    ap.add_argument("--out", default=os.path.join(D, "paper_fig_body_sweeps"))
    a = ap.parse_args()
    want = [round(float(c), 2) for c in a.currents.split(",")]
    os.makedirs(a.out, exist_ok=True)
    rows = {round(float(r["current_ma"]), 2): r for r in json.load(open(SRC))}
    missing = [c for c in want if c not in rows]
    assert not missing, f"no ankle sweep at {missing} mA in {SRC}"
    data = [(c, rows[c]["run"], rows[c]["detected"], *sweep(rows[c]["run"])) for c in want]
    tag = "_".join(f"{c:g}".replace(".", "p") for c in want)
    name = "fig_body_sweep_ankle_currents" if a.currents == "1,2.5,4" else f"fig_body_sweep_ankle_{tag}mA"
    with open(os.path.join(a.out, name + ".csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["current_ma", "run", "carrier_detected", "carrier_hz", "snr_db"])
        for c, run, det, car, y in data:
            for ci, yi in zip(car, y):
                w.writerow([c, run, det, f"{ci:.1f}", f"{yi:.2f}"])

    car = data[0][3]
    with plt.rc_context(PS.RC):
        fig, ax = PS.figure(3.5, 2.4)
        PS.vmark(ax, CARRIER / 1e3, "159.6 kHz")
        for k, (c, run, det, x, y) in enumerate(sorted(data, key=lambda r: r[2])):   # no-carrier sweeps underneath
            PS.trace(ax, x / 1e3, y, CUR[c]["color"], CUR[c]["marker"], (2 * k, 6), solid=det)
        PS.khz_axis(ax, car)
        ax.set_ylim(-10, 70); ax.set_yticks(range(0, 70, 20))
        ax.set_xlabel("Carrier frequency (kHz)"); ax.set_ylabel("SNR (dB)")
        PS.finish(ax)
        PS.top_legend(fig, [cur_key(c, det) for c, _, det, _, _ in data],                      # in --currents order
                      [CUR[c]["label"] + ("" if det else " (no carrier)") for c, _, det, _, _ in data], handlelength=2.2)
        PS.save(fig, os.path.join(a.out, name))
    j = int(np.argmin(abs(car - CARRIER)))
    for c, run, det, x, y in data:
        print(f"{c:g} mA ({run[-15:]}): {y[j]:+.1f} dB at 159.6 kHz, median {np.median(y):+.1f} dB"
              + ("" if det else "  [no carrier detected]"))
    print("->", a.out)


if __name__ == "__main__":
    main()
