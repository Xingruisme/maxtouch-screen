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
Dotted line = 159.585 kHz, the carrier of every BER run.

   python paper_fig_body_sweep_currents.py [--currents 1,2.5,4]

Output: data_maxtouch/paper_fig_body_sweeps/ (fig_body_sweep_ankle_currents.{pdf,png,svg}, ...csv)
   \\includegraphics[width=\\columnwidth]{fig_body_sweep_ankle_currents.pdf}   (3.5 in wide, fonts ~9 pt at 100 %)
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter
import numpy as np

from paper_fig_body_sweeps import CARRIER, CUR, D, GRID, INK, INK2, NO_CARRIER, RC, sweep

FS, FS_TICK = 9, 8                              # pt at print size
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

    ticks = [50, 100, 200, 500]
    car = data[0][3]
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(3.5, 2.4), layout="constrained")
        fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02)
        ax.axvline(CARRIER / 1e3, color=INK2, ls=":", lw=1.0, zorder=1)
        for k, (c, run, det, x, y) in enumerate(sorted(data, key=lambda r: r[2])):   # no-carrier sweeps underneath
            st = CUR[c]
            ax.plot(x / 1e3, y, color=st["color"], lw=1.1 if det else 0.8, ls="-" if det else NO_CARRIER,
                    marker=st["marker"], ms=3.0, markevery=(2 * k, 6), mec="white", mew=0.3,
                    label=st["label"] + ("" if det else " (no carrier)"), zorder=3 if det else 2)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.set_xticklabels([str(t) for t in ticks])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(car[0] / 1e3 * 0.97, car[-1] / 1e3 * 1.03)
        ax.set_ylim(-10, 70); ax.set_yticks(range(0, 70, 20))
        ax.grid(True, which="major", color=GRID, lw=0.5, zorder=0)
        ax.set_xlabel("Carrier frequency (kHz)", fontsize=FS, fontweight="bold", color=INK, labelpad=2)
        ax.set_ylabel("SNR (dB)", fontsize=FS, fontweight="bold", color=INK, labelpad=2)
        ax.tick_params(labelsize=FS_TICK, colors=INK, width=0.6, length=3, pad=2)
        ax.tick_params(which="minor", width=0.5, length=1.5)
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_fontweight("bold")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_linewidth(0.6)
        ax.text(CARRIER / 1e3 * 1.04, 67, "159.6 kHz", fontsize=FS_TICK - 1, color=INK2, va="top")
        by_label = dict(zip(*ax.get_legend_handles_labels()[::-1]))           # label -> handle
        labels = [CUR[c]["label"] + ("" if det else " (no carrier)") for c, _, det, _, _ in data]   # in --currents order
        fig.legend([by_label[l] for l in labels], labels, loc="outside upper center", ncol=len(labels),
                   frameon=False, prop={"size": FS_TICK, "weight": "bold"}, handlelength=2.0, columnspacing=1.2,
                   handletextpad=0.5, borderaxespad=0.1)
        for ext in ("pdf", "svg", "png"):
            fig.savefig(os.path.join(a.out, f"{name}.{ext}"), dpi=600 if ext == "png" else None)
        plt.close(fig)
    j = int(np.argmin(abs(car - CARRIER)))
    for c, run, det, x, y in data:
        print(f"{c:g} mA ({run[-15:]}): {y[j]:+.1f} dB at 159.6 kHz, median {np.median(y):+.1f} dB"
              + ("" if det else "  [no carrier detected]"))
    print("->", a.out)


if __name__ == "__main__":
    main()
