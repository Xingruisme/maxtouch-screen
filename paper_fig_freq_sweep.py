#!/usr/bin/env python3
"""
Paper figure: tablet (maXTouch) frequency response, direct contact vs through body from three TX electrode sites,
all at 1 mA, one column at print size (3.5 in), colours and markers of paper_style.PATH.

Data = four existing 94-carrier probe sweeps (log 50-500 kHz, 2.5 % steps, 0.5 s on / off blocks x 2), re-analysed with
maxtouch_stent_rx.analyse_probe as body_current_sweeps.py does (nothing re-recorded):
  direct contact  data_maxtouch/stent_probe_frequency_sweep_new_20261026_113719   (2026-10-26 11:37, 1 mA)
  back to finger  data_maxtouch/stent_probe_body_back2finger_1mA_20261026_175820 (2026-10-26 17:58, 1 mA)
  knee to finger  data_maxtouch/stent_probe_body_knee2finger_1mA_20261026_171930 (2026-10-26 17:19, 1 mA)
  ankle to finger data_maxtouch/stent_probe_body_1mA_20261026_155003             (2026-10-26 15:50, 1 mA)
Metric per carrier (cap_ook_rx.analyse_probe 'energy_db'): 10 log10 of the mean squared scan-to-scan change of the
node's delta, (x[n] - x[n-1])^2, in that carrier's ON blocks over its interleaved OFF blocks (2 x 0.5 s each, edges
trimmed); best node of Y0-Y3 (where the finger sits). Dashed line = 159.585 kHz, the carrier of every BER run.

   python paper_fig_freq_sweep.py

Output: data_maxtouch/paper_fig_freq_sweep/ (fig_freq_sweep.{png,pdf,svg}, fig_freq_sweep.csv)
   \\includegraphics[width=\\columnwidth]{fig_freq_sweep.pdf}   (3.5 in wide, fonts 6-7 pt at 100 %)
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import maxtouch_stent_rx as R
import paper_style as PS

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data_maxtouch")
SWEEPS = [("direct", "stent_probe_frequency_sweep_new_20261026_113719"),
          ("back", "stent_probe_body_back2finger_1mA_20261026_175820"),
          ("knee", "stent_probe_body_knee2finger_1mA_20261026_171930"),
          ("ankle", "stent_probe_body_1mA_20261026_155003")]
NODES = 4                                       # Y0-Y3, where the finger sits
CARRIER = 159584.9
LABEL = {"direct": "Direct contact", "back": "Back to finger", "knee": "Knee to finger", "ankle": "Ankle to finger"}


def load(d):
    meta = json.load(open(os.path.join(d, "meta.json")))
    z = np.load(os.path.join(d, "capture.npz"))
    with contextlib.redirect_stdout(io.StringIO()):
        res, aux = R.analyse_probe(z["t"], z["X"], meta)
    return np.array(aux["car"]), aux["E"][:NODES].max(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(D, "paper_fig_freq_sweep"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    curves = {k: load(os.path.join(D, d)) for k, d in SWEEPS}
    car = curves["direct"][0]
    assert all(np.allclose(car, v[0]) for v in curves.values()), "the sweeps must share the carrier grid"
    with open(os.path.join(a.out, "fig_freq_sweep.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["carrier_hz", *(f"{k}_db" for k, _ in SWEEPS)])
        for j, c in enumerate(car):
            w.writerow([f"{c:.1f}", *(f"{curves[k][1][j]:.2f}" for k, _ in SWEEPS)])

    with plt.rc_context(PS.RC):
        fig, ax = PS.figure(3.5, 2.3)
        PS.vmark(ax, CARRIER / 1e3, "159.6 kHz")
        for i, (k, _) in enumerate(SWEEPS):
            x, y = curves[k]
            # every 8th carrier marked, staggered per line, so four dense lines stay readable
            PS.trace(ax, x / 1e3, y, PS.PATH[k]["color"], PS.PATH[k]["marker"], (2 * i, 8), label=LABEL[k])
        PS.khz_axis(ax, car)
        ax.set_ylim(0, 65); ax.set_yticks(range(0, 61, 20))
        ax.set_xlabel("Carrier frequency (kHz)"); ax.set_ylabel("SNR (dB)")
        PS.finish(ax)
        PS.top_legend(fig, *ax.get_legend_handles_labels(), ncol=2, handlelength=2.2)
        PS.save(fig, os.path.join(a.out, "fig_freq_sweep"))
    j = int(np.argmin(abs(car - CARRIER)))
    for k, _ in SWEEPS:
        y = curves[k][1]
        print(f"{k}: {y[j]:+.1f} dB at {CARRIER / 1e3:.1f} kHz, peak {y.max():+.1f} dB at {car[y.argmax()] / 1e3:.1f} kHz, "
              f"median {np.median(y):+.1f} dB")
    print("->", a.out)


if __name__ == "__main__":
    main()
