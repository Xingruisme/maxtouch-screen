#!/usr/bin/env python3
"""
Paper figure: tablet (maXTouch) frequency response, direct contact vs through body from three TX electrode sites,
all at 1 mA.

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
from matplotlib.ticker import FixedLocator, NullFormatter
import numpy as np

import maxtouch_stent_rx as R

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data_maxtouch")
SWEEPS = [("direct", "stent_probe_frequency_sweep_new_20261026_113719"),
          ("back", "stent_probe_body_back2finger_1mA_20261026_175820"),
          ("knee", "stent_probe_body_knee2finger_1mA_20261026_171930"),
          ("ankle", "stent_probe_body_1mA_20261026_155003")]
NODES = 4                                       # Y0-Y3, where the finger sits
CARRIER = 159584.9
RC = {"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
      "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}
INK, INK2, GRID = "#1a1a1a", "#52514e", "#d9d9d9"
STYLE = {"direct": dict(color="#1f5fbf", marker="o", ls="-", label="Direct contact"),
         "back": dict(color="#D55E00", marker="^", ls="-", label="Back to finger"),
         "knee": dict(color="#882255", marker="D", ls="-", label="Knee to finger"),
         "ankle": dict(color="#00897b", marker="s", ls="-", label="Ankle to finger")}
FS = 20


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

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(9.6, 4.8))
        fig.subplots_adjust(left=0.12, right=0.98, top=0.86, bottom=0.17)
        ax.axvline(CARRIER / 1e3, color=INK2, ls="--", lw=1.2, zorder=1)
        for i, (k, _) in enumerate(SWEEPS):
            x, y = curves[k]
            st = STYLE[k]
            # every 4th carrier marked, staggered per line, so four dense lines stay readable
            ax.plot(x / 1e3, y, color=st["color"], marker=st["marker"], ls=st["ls"], ms=6, markevery=(i, 4),
                    mec="white", mew=0.6, lw=1.8, label=st["label"], zorder=3)
        ax.set_xscale("log")
        ticks = [50, 100, 200, 500]
        ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.set_xticklabels([str(t) for t in ticks])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(car[0] / 1e3 * 0.97, car[-1] / 1e3 * 1.03)
        ax.set_ylim(0, 65)
        ax.grid(True, which="major", color=GRID, lw=0.8, zorder=0)
        ax.set_xlabel("Carrier frequency (kHz)", fontsize=FS, fontweight="bold", color=INK)
        ax.set_ylabel("SNR (dB)", fontsize=FS, fontweight="bold", color=INK)
        ax.tick_params(labelsize=FS, colors=INK, width=1.0, length=5)
        ax.tick_params(which="minor", length=2.5)
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_fontweight("bold")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_linewidth(1.0)
        ax.text(CARRIER / 1e3 * 1.02, 63, "159.6 kHz", fontsize=FS - 4, color=INK2, va="top", fontweight="bold")
        fig.legend(*ax.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.54, 1.0), ncol=len(SWEEPS),
                   frameon=False, prop={"size": FS - 4, "weight": "bold"}, handlelength=1.6, columnspacing=0.9)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(os.path.join(a.out, f"fig_freq_sweep.{ext}"), dpi=600 if ext == "png" else None)
        plt.close(fig)
    j = int(np.argmin(abs(car - CARRIER)))
    for k, _ in SWEEPS:
        y = curves[k][1]
        print(f"{k}: {y[j]:+.1f} dB at {CARRIER / 1e3:.1f} kHz, peak {y.max():+.1f} dB at {car[y.argmax()] / 1e3:.1f} kHz, "
              f"median {np.median(y):+.1f} dB")
    print("->", a.out)


if __name__ == "__main__":
    main()
