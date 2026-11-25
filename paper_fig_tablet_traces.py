#!/usr/bin/env python3
"""
Paper figure: time traces of the signal the tablet (maXTouch panel) receives in the four body states, OOK at 40 bps,
TX electrode on the ankle, finger on the panel, 1 mA, one column at print size (3.5 in), four panels on a shared axis.

Raw panel delta counts of the strongest receiving node (v4's pick) over the first 1.5 s (60 bits) of frame 0 of one run
per state, time from the frame start (the ~12 ms panel latency is part of the saved decode); counts are relative to the
median before the prompt. Recorded runs, nothing re-decoded; loaders from paper_fig1_rx_signals.py unchanged.
  Sitting still     stent_mod_ankle2finger_sit_still_ook_40bps_20261030_121747      (session modcomp_..._sit_still_20261030_121747)
  Standing still    stent_mod_ankle2finger_sit_still_ook_40bps_20261030_123044      (label on disk says sit_still; session
                                                                                    modcomp_..._stand_still_20261030_123044)
  Sitting shaking   stent_mod_ankle2finger_sit-shaking_ook_40bps_20261101_164447
  Standing walking  stent_mod_ankle2finger_stand-walking_ook_40bps_20261101_172732

   python paper_fig_tablet_traces.py [--bits 60]

Output: data_maxtouch/paper_fig_tablet_ber/ (fig_tablet_traces.{pdf,png,svg}, fig_tablet_traces.csv = run, node, errors)
   \\includegraphics[width=\\columnwidth]{fig_tablet_traces.pdf}   (3.5 in wide, fonts 7-8 pt at 100 %)
"""
from __future__ import annotations

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import paper_fig1_rx_signals as F
import paper_style as PS
from paper_fig_tablet_ber import HERE, RC

STATES = [("Sitting still", "stent_mod_ankle2finger_sit_still_ook_40bps_20261030_121747"),
          ("Standing still", "stent_mod_ankle2finger_sit_still_ook_40bps_20261030_123044"),
          ("Sitting shaking", "stent_mod_ankle2finger_sit-shaking_ook_40bps_20261101_164447"),
          ("Standing walking", "stent_mod_ankle2finger_stand-walking_ook_40bps_20261101_172732")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=60, help="bits of frame 0 shown (40 bps: 60 bits = 1.5 s)")
    ap.add_argument("--out", default=os.path.join(HERE, "data_maxtouch", "paper_fig_tablet_ber"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sigs = [(lab, F.load(run, 0, a.bits, 0.0)) for lab, run in STATES]
    lim = max(np.percentile(np.abs(s["y"]), 99.5) for _, s in sigs)
    with open(os.path.join(a.out, "fig_tablet_traces.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["state", "run", "node", "frame0_errors", "frame_bits", "run_errors", "run_bits"])
        for lab, s in sigs:
            w.writerow([lab, s["run"], s["node"], s["frame_errors"], s["frame_bits_n"], s["run_errors"], s["run_bits"]])
    with plt.rc_context(RC):
        fig, axes = PS.figure(3.5, 4.4, 4, 1, hspace=0.04, sharex=True, sharey=True)
        for ax, (lab, s), letter in zip(axes, sigs, "abcd"):
            ax.plot(s["t"], s["y"], lw=0.5, color=PS.STYLE["ook"]["color"])
            ax.axhline(0, color=PS.RULE, lw=0.4, zorder=0)
            ax.set_ylim(-1.1 * lim, 1.5 * lim)
            PS.note(ax, lab, x=0.015, y=0.97)
            PS.panel(ax, letter, dx=-27)
            ax.set_ylabel("Counts")
            PS.finish(ax)
        for ax in axes[:-1]:
            ax.tick_params(axis="x", length=0)
        axes[-1].set_xlabel("Time (s)")
        axes[-1].set_xlim(0, sigs[0][1]["Tb"] * a.bits)
        PS.save(fig, os.path.join(a.out, "fig_tablet_traces"))
    for lab, s in sigs:
        print(f"{lab:17s} node {s['node']}  frame 0 errors {s['frame_errors']}/{s['frame_bits_n']}  run {s['run_errors']}/{s['run_bits']}"
              f"  peak |counts| {np.max(np.abs(s['y'])):.0f}")
    print("->", os.path.join(a.out, "fig_tablet_traces.pdf"))


if __name__ == "__main__":
    main()
