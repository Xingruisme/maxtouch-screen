#!/usr/bin/env python3
"""
Paper figure: maXTouch frequency response under different set-ups, two panels at print size (7.16 in, two columns):
  a  agar set-up vs on-body set-up at different TX locations (all ground line removed, 1 mA, 2026-11-22)
  b  effect of the ground connection: through body, ankle to finger, with (2026-10-26) and without (2026-11-22) the
     ground line to the screen
(c, the effect of the device - tablets vs smartphone - waits for sweeps recorded on the same footing, see below.)

Data = existing 94-carrier probe sweeps (log 50-500 kHz, 2.5 % steps, 0.5 s on / off blocks x 2), analysed as in
paper_fig_freq_sweep.py (maxtouch_stent_rx.analyse_probe 'energy_db' = SNR in dB of the node's scan-to-scan change,
best node of Y0-Y3; nothing re-recorded):
  agar     stent_probe_withinagar_distance15cm_nognd_1mA_20261122_130100, ..._distance30cm_..._123127
  on body  stent_probe_onbody_{ankle,knee,waist,middleback,shoulder}2finger_nognd_1mA_20261122_*  (waist: the 19:33 sweep is an incomplete first attempt and is not used; the 19:38 one is)
  ground   stent_probe_body_1mA_20261026_155003 (ankle, with ground: the Oct 26 sweeps carry no 'nognd' label, so they are
           taken to have the ground line connected - to be confirmed) vs ..._onbody_ankle2finger_nognd_1mA_20261122_192723
Dashed line = 159.585 kHz, the carrier of every BER run.

   python paper_fig_freq_settings.py

Output: data_maxtouch/paper_fig_freq_sweep/ (fig_freq_settings.{pdf,png,svg}, fig_freq_settings.csv)
   \\\\includegraphics[width=\\\\textwidth]{fig_freq_settings.pdf}   (7.16 in wide, fonts 7-8 pt at 100 %)
"""
from __future__ import annotations

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import paper_style as PS
from paper_fig_freq_sweep import CARRIER, D, load

A = [("Agar, 15 cm", "stent_probe_withinagar_distance15cm_nognd_1mA_20261122_130100", "#222222", "D", "-"),
     ("Agar, 30 cm", "stent_probe_withinagar_distance30cm_nognd_1mA_20261122_123127", "#9a9a9a", "D", "-"),
     ("Ankle", "stent_probe_onbody_ankle2finger_nognd_1mA_20261122_192723", "#6a4fb5", "o", "-"),
     ("Knee", "stent_probe_onbody_knee2finger_nognd_1mA_20261122_193132", "#009E73", "s", "-"),
     ("Waist", "stent_probe_onbody_waist2finger_nognd_1mA_20261122_193821", "#0072B2", "^", "-"),
     ("Middle back", "stent_probe_onbody_middleback2finger_nognd_1mA_20261122_194414", "#D55E00", "v", "-"),
     ("Shoulder", "stent_probe_onbody_shoulder2finger_nognd_1mA_20261122_195031", "#CC79A7", "P", "-")]
B = [("With ground (Oct 26)", "stent_probe_body_1mA_20261026_155003", "#555555", "D"),
     ("Without ground (Nov 22)", "stent_probe_onbody_ankle2finger_nognd_1mA_20261122_192723", "#6a4fb5", "o")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(D, "paper_fig_freq_sweep"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    curves = {n: load(os.path.join(D, d)) for n, d, *_ in A}
    for n, d, *_ in B:
        curves[n] = load(os.path.join(D, d))
    car = curves["Ankle"][0]
    assert all(np.allclose(car, v[0]) for v in curves.values()), "the sweeps must share the carrier grid"
    with open(os.path.join(a.out, "fig_freq_settings.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["carrier_hz", *(f"{n}_db" for n in curves)])
        for j, c in enumerate(car):
            w.writerow([f"{c:.1f}", *(f"{curves[n][1][j]:.2f}" for n in curves)])
    with plt.rc_context(PS.RC):
        fig, (ax1, ax2) = PS.figure(7.16, 3.1, 1, 2, wspace=0.08, width_ratios=(1.25, 1), sharey=True)
        for ax in (ax1, ax2):
            PS.vmark(ax, CARRIER / 1e3, "159.6 kHz")
        for i, (n, _, col, mk, ls) in enumerate(A):
            x, y = curves[n]
            PS.trace(ax1, x / 1e3, y, col, mk, (2 * i, 8), label=n)
        for i, (n, _, col, mk) in enumerate(B):
            x, y = curves[n]
            PS.trace(ax2, x / 1e3, y, col, mk, (2 * i, 8), label=n)
        for ax, letter in ((ax1, "a"), (ax2, "b")):
            PS.khz_axis(ax, car)
            ax.set_ylim(0, 65); ax.set_yticks(range(0, 61, 20))
            ax.set_xlabel("Carrier frequency (kHz)")
            PS.panel(ax, letter, dx=-27 if letter == "a" else -6, dy=24)
            PS.finish(ax)
            ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=4 if letter == "a" else 2, handlelength=1.5, columnspacing=0.9)
        ax1.set_ylabel("SNR (dB)")
        PS.save(fig, os.path.join(a.out, "fig_freq_settings"))
    j = int(np.argmin(abs(car - CARRIER)))
    for n, (x, y) in curves.items():
        print(f"{n:26s} {y[j]:+5.1f} dB at 159.6 kHz, peak {y.max():+5.1f} dB at {car[y.argmax()] / 1e3:6.1f} kHz, median {np.median(y):+5.1f}")
    print("->", a.out)


if __name__ == "__main__":
    main()
