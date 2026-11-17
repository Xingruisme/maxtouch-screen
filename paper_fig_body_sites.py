#!/usr/bin/env python3
"""
Paper figure: through-body carrier at the tablet vs where the TX electrode sits (finger on the panel's top edge).

Data = the 2026-10-26 through-body frequency sweeps already summarised by body_current_sweeps.py (nothing re-run):
  back2finger  data_maxtouch/body_sweep_back2finger_20261026.json   (17:50-18:11)
  knee2finger  data_maxtouch/body_sweep_knee2finger_20261026.json   (17:16-17:42)
  ankle2finger data_maxtouch/body_sweep_currents_20261026.json      (15:16-16:29, electrode on the ankle, still)
Sites ordered by body path length to the finger (not measured in cm). Per site, the sweeps at the set currents where a
carrier was detected (0.75 / 1 / 2.5 mA; 0.5 and 4 mA delivered no carrier at every site = TX configuration fault).
Metric at the 159.6 kHz carrier, best node of Y0-Y3:
  SNR (dB)  = energy increase of the carrier blocks over the no-carrier lead (analyse_probe)
  amplitude = scan-to-scan rms in the carrier ON blocks (counts)

   python paper_fig_body_sites.py
Output: data_maxtouch/paper_fig_body_sites/ (fig_body_sites_snr.*, fig_body_sites_counts.*, fig_body_sites.csv)
"""
from __future__ import annotations

import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data_maxtouch")
SITES = [("Back", "body_sweep_back2finger_20261026.json"),
         ("Knee", "body_sweep_knee2finger_20261026.json"),
         ("Ankle", "body_sweep_currents_20261026.json")]
CARRIER_IDX = 1                                   # PEAKS[1] = 159.585 kHz, the carrier used in every BER run
RC = {"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
      "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}
INK, GRID = "#1a1a1a", "#d9d9d9"
CUR = {0.75: dict(color="#7fa6e0", marker="^", label="0.75 mA"),
       1.0: dict(color="#1f5fbf", marker="o", label="1 mA"),
       2.5: dict(color="#0b2e66", marker="s", label="2.5 mA")}
FS = 20


def load():
    rows = []
    for site, f in SITES:
        for r in json.load(open(os.path.join(D, f))):
            if not r["detected"]:
                continue
            cur = round(float(r["current_ma"]), 2)
            rows.append(dict(site=site, current_ma=cur, run=r["run"], snr_db=r["energy_db"][CARRIER_IDX],
                             on_counts=r["on_rms"][CARRIER_IDX], off_counts=r["off_rms"][CARRIER_IDX]))
    return rows


def plot(rows, key, ylabel, ylim, out):
    sites = [s for s, _ in SITES]
    x = np.arange(len(sites))
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        fig.subplots_adjust(left=0.19, right=0.97, top=0.95, bottom=0.17)
        for i, (cur, st) in enumerate(CUR.items()):
            y = [next((r[key] for r in rows if r["site"] == s and r["current_ma"] == cur), np.nan) for s in sites]
            ax.plot(x + (i - 1) * 0.08, y, color=st["color"], marker=st["marker"], ms=10, mec="white", mew=1.0, lw=2.2,
                    label=st["label"], zorder=3)
        ax.set_xticks(x, sites); ax.set_xlim(-0.4, len(sites) - 0.6)
        ax.set_ylim(*ylim)
        ax.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
        ax.set_xlabel("TX electrode site", fontsize=FS, fontweight="bold", color=INK)
        ax.set_ylabel(ylabel, fontsize=FS, fontweight="bold", color=INK)
        ax.tick_params(labelsize=FS, colors=INK, width=1.0, length=5)
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_fontweight("bold")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_linewidth(1.0)
        ax.legend(loc="lower right", frameon=False, prop={"size": FS - 2, "weight": "bold"}, handlelength=1.6, ncol=3,
                  columnspacing=0.8, handletextpad=0.4)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(f"{out}.{ext}", dpi=600 if ext == "png" else None)
        plt.close(fig)


def main():
    out = os.path.join(D, "paper_fig_body_sites")
    os.makedirs(out, exist_ok=True)
    rows = load()
    with open(os.path.join(out, "fig_body_sites.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    plot(rows, "snr_db", "SNR (dB)", (30, 70), os.path.join(out, "fig_body_sites_snr"))
    plot(rows, "on_counts", "Carrier amplitude (counts)", (0, 2500), os.path.join(out, "fig_body_sites_counts"))
    for s, _ in SITES:
        rr = [r for r in rows if r["site"] == s]
        print(f"{s:6s} " + "  ".join(f"{r['current_ma']} mA: {r['snr_db']:.1f} dB {r['on_counts']:.0f} counts" for r in rr)
              + f"   mean {np.mean([r['snr_db'] for r in rr]):.1f} dB")
    print("->", out)


if __name__ == "__main__":
    main()
