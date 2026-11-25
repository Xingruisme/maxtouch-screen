#!/usr/bin/env python3
"""
Paper figure: frequency response of the screens (x = carrier frequency, y = response), three panels one above the other
at print size for one column (3.5 in):
  a  experimental set-up: within agar vs on body (maXTouch tablet, 1 mA, ground line removed)
  b  effect of the ground connection: through body, ankle to finger, screen grounded vs ungrounded
  c  effect of the device: the two tablets (maXTouch tablet, OnePlus Pad), through body, in one panel
     (the Pixel 4 and Pixel 6 curves were taken out on 2026-11-24; their data stay in data_body_sweep_all)

Nothing is re-recorded. a and b = existing 94-carrier probe sweeps (log 50-500 kHz, 2.5 % steps, 0.5 s on / off blocks
x 2), analysed as in paper_fig_freq_sweep.py (maxtouch_stent_rx.analyse_probe 'energy_db' = SNR in dB of the node's
scan-to-scan change, carrier on over carrier off, best node of Y0-Y3):
  direct   stent_probe_directcontact_nognd_1mA_20261124_185535 (direct contact, no body in the path). This sweep is on
           its own grid, 252 carriers 1-492 kHz in 2.5 % steps (nearest carriers to 159.6 kHz: 157.9 and 161.9 kHz); the
           panel shows its part from 50 kHz up, the whole sweep is in fig_freq_summary_a_direct.csv
  agar     stent_probe_withinagar_distance{15,30}cm_nognd_1mA_20261122_* (both distances, one curve each)
  belly    stent_probe_onbody_belly2finger_nognd_1mA_20261123_204916
  heart    stent_probe_onbody_heart2finger_nognd_1mA_20261124_135555, the latest of the three heart sweeps of that day
           (13:43, 13:50, 13:55) and the only one with a response at the carrier (recorded after the TeensyTRX ask1 firmware was
           flashed; 31 dB under the belly sweep at the carrier - transmitter output not verified, see the run notes)
  ground   stent_probe_body_1mA_20261026_155003 (ankle; the Oct 26 sweeps carry no 'nognd' label, so they are taken to have
           the ground line connected - to be confirmed) vs stent_probe_onbody_ankle2finger_nognd_1mA_20261122_192723
c = the OnePlus Pad's per-carrier results saved with the 2026-11-14 device sweeps (screen-bloodflow-core-main/
data_body_sweep_all, hand to finger, 0.1 mA, 1-500 kHz in 10 % steps, 1 on / off block), next to the maXTouch
direct-contact sweep of panel a, whole (1-492 kHz; since 2026-11-24 - the ankle sweep of panel b, 50 kHz up, before).
NOT on one footing: the maXTouch sweep is direct contact at 1 mA, the Pad sweep is through the body (hand to finger) at
0.1 mA, and each device answers in its own way.
To put them on the dB axis of a and b, each device's own measure is taken over its level with nothing transmitted:
  maXTouch tablet   SNR of the scan-to-scan change, carrier on over off (as a, b)
  OnePlus Pad       map activity while the carrier is on, frame-to-frame rms (the map is frozen without a carrier)
                    as 20 log10 (measure / rms of the same measure over the carriers of the control run: finger on,
                    nothing transmitted), not below 0 dB
So the heights in c compare how far each device rises above its own quiet level, not the same physical quantity.

   python paper_fig_freq_summary.py

Output: data_maxtouch/paper_fig_freq_sweep/ (fig_freq_summary.{pdf,png,svg}, fig_freq_summary_ab.csv, ..._c.csv)
   \\includegraphics[width=\\columnwidth]{fig_freq_summary.pdf}   (3.5 in wide, fonts 9-10 pt at 100 %)
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
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter

import paper_style as PS
from paper_fig_freq_sweep import CARRIER, D, load

DEV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "screen-bloodflow-core-main",
                   "data_body_sweep_all")
SCALE = 1.25                                    # type 25 % over the paper look: labels 10 pt, ticks and keys 8.75 pt
H = 5.2                                         # in, the three panels together (6.9 before 2026-11-24)
A_DIRECT = ("Direct contact", "stent_probe_directcontact_nognd_1mA_20261124_185535", "#CC79A7", "o")
A = [("Within agar, 15 cm", "stent_probe_withinagar_distance15cm_nognd_1mA_20261122_130100", "#222222", "D"),
     ("Within agar, 30 cm", "stent_probe_withinagar_distance30cm_nognd_1mA_20261122_123127", "#9a9a9a", "D"),
     ("On body, belly", "stent_probe_onbody_belly2finger_nognd_1mA_20261123_204916", "#D55E00", "^"),
     ("On body, heart", "stent_probe_onbody_heart2finger_nognd_1mA_20261124_135555", "#009E73", "s")]
B = [("Grounded", "stent_probe_body_1mA_20261026_155003", "#555555", "D"),                          # 2026-10-26
     ("Ungrounded", "stent_probe_onbody_ankle2finger_nognd_1mA_20261122_192723", "#6a4fb5", "o")]    # 2026-11-22


def over_control(y, yc):
    """a device's own measure -> dB over its rms in the control run (nothing transmitted), not below 0"""
    floor = float(np.sqrt(np.mean(np.square(yc))))
    return 20 * np.log10(np.maximum(np.abs(y), floor) / floor), floor


def devices():
    """[(name, colour, marker, f_hz, dB, measure, control rms)] - see the header for each device's measure"""
    f, y = load(os.path.join(D, A_DIRECT[1]))               # direct contact, 1-492 kHz
    pad = json.load(open(os.path.join(DEV, "pad_activity_rms.json")))
    arr = lambda rows, k: np.array([r[k] for r in rows], float)
    hp, cp = pad["hand_1k_500k_20261114_211114"], pad["ctl_20261114_210534"]
    out = [("maXTouch tablet", B[1][2], B[1][3], f, y, "scan-to-scan change, on over off, direct contact", None)]
    for name, col, mk, fx, yy, yc, meas in (
            ("OnePlus Pad", "#0072B2", "^", arr(hp, "f"), arr(hp, "on"), arr(cp, "on"), "map activity [counts]"),):
        db, floor = over_control(yy, yc)
        out.append((name, col, mk, fx, db, meas, floor))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(D, "paper_fig_freq_sweep"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    curves = {n: load(os.path.join(D, d)) for n, d, *_ in A + B}
    car = curves[B[1][0]][0]
    assert all(np.allclose(car, v[0]) for v in curves.values()), "the sweeps must share the carrier grid"
    dev = devices()
    fd, yd = load(os.path.join(D, A_DIRECT[1]))            # own carrier grid (1-492 kHz)
    with open(os.path.join(a.out, "fig_freq_summary_a_direct.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["carrier_hz", f"{A_DIRECT[0]}_db"])
        w.writerows([f"{x_:.1f}", f"{y_:.2f}"] for x_, y_ in zip(fd, yd))
    with open(os.path.join(a.out, "fig_freq_summary_ab.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["carrier_hz", *(f"{n}_db" for n in curves)])
        for j, c in enumerate(car):
            w.writerow([f"{c:.1f}", *(f"{curves[n][1][j]:.2f}" for n in curves)])
    with open(os.path.join(a.out, "fig_freq_summary_c.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["device", "measure", "control_rms", "carrier_hz", "db"])
        for name, _, _, fx, db, meas, floor in dev:
            for x_, y_ in zip(fx, db):
                w.writerow([name, meas, "" if floor is None else f"{floor:.4f}", f"{x_:.1f}", f"{y_:.2f}"])

    with plt.rc_context(PS.rc(SCALE)):
        fig, (ax1, ax2, ax3) = PS.figure(3.5, H, 3, 1, hspace=0.04)
        shown = fd >= car[0] * 0.999                       # the direct sweep from the panel's first carrier up
        PS.trace(ax1, fd[shown] / 1e3, yd[shown], A_DIRECT[2], A_DIRECT[3], (6, 8), label=A_DIRECT[0])
        for ax, rows, letter in ((ax1, A, "a"), (ax2, B, "b")):
            for i, (n, _, col, mk) in enumerate(rows):
                x, y = curves[n]
                PS.trace(ax, x / 1e3, y, col, mk, (2 * i, 8), label=n)
            PS.khz_axis(ax, car)
        for i, (name, col, mk, fx, db, _, _) in enumerate(dev):
            PS.trace(ax3, fx / 1e3, db, col, mk, (2 * i, 8 if len(fx) < 150 else 20), label=name)
        ticks = [1, 10, 100, 500]
        ax3.set_xscale("log")
        ax3.xaxis.set_major_locator(FixedLocator(ticks)); ax3.xaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
        ax3.xaxis.set_minor_formatter(NullFormatter())
        ax3.set_xlim(0.93, 520)
        for ax, letter in ((ax1, "a"), (ax2, "b"), (ax3, "c")):
            PS.vmark(ax, CARRIER / 1e3, "159.6 kHz", scale=SCALE)
            ax.set_ylim(0, 65); ax.set_yticks(range(0, 61, 20))
            ax.set_xlabel("Carrier frequency (kHz)"); ax.set_ylabel("SNR (dB)")
            PS.panel(ax, letter, dx=-27, dy=21.5 if letter == "a" else 3, scale=SCALE)   # a: beside the key's top row
            PS.finish(ax)
            # the key starts beside the panel letter, left of the axis; a: three rows, filled by column
            ax.legend(loc="lower left", bbox_to_anchor=(-0.105, 1.0), ncol=2, handlelength=1.2, columnspacing=0.7,
                      handletextpad=0.3)
        PS.save(fig, os.path.join(a.out, "fig_freq_summary"))

    j = int(np.argmin(abs(car - CARRIER)))
    k = int(np.argmin(abs(fd - CARRIER))); lo = fd < car[0] * 0.999
    print(f"{A_DIRECT[0]:26s} {yd[k]:+5.1f} dB at {fd[k] / 1e3:.1f} kHz, peak {yd.max():+5.1f} dB at {fd[yd.argmax()] / 1e3:6.1f} kHz, "
          f"median {np.median(yd[~lo]):+5.1f} (50 kHz up); below 50 kHz (not drawn): median {np.median(yd[lo]):+5.1f}, "
          f"max {yd[lo].max():+5.1f} dB at {fd[lo][yd[lo].argmax()] / 1e3:.1f} kHz")
    for n, (x, y) in curves.items():
        print(f"{n:26s} {y[j]:+5.1f} dB at 159.6 kHz, peak {y.max():+5.1f} dB at {car[y.argmax()] / 1e3:6.1f} kHz, "
              f"median {np.median(y):+5.1f}")
    for name, _, _, fx, db, meas, floor in dev:
        top = np.argsort(-db)[:3]
        print(f"{name:16s} max {db.max():5.1f} dB at {fx[db.argmax()] / 1e3:6.1f} kHz; top 3 at "
              + ", ".join(f"{fx[k] / 1e3:.1f}" for k in sorted(top)) + f" kHz; median {np.median(db):4.1f} dB"
              + (f"; {meas}, control rms {floor:.3f}" if floor is not None else f"; {meas}"))
    print("->", a.out)


if __name__ == "__main__":
    main()
