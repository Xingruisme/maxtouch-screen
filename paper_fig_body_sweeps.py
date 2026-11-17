#!/usr/bin/env python3
"""
Paper figure: through-body frequency sweeps at the tablet (maXTouch) per TX electrode site, one line per set current.

Data = the 2026-10-26 through-body probe sweeps (94 carriers, log 50-500 kHz, 2.5 % steps), re-analysed with
maxtouch_stent_rx.analyse_probe as body_current_sweeps.py does (nothing re-recorded). The runs and their
'carrier detected' verdicts come from that script's summaries:
  Back          data_maxtouch/body_sweep_back2finger_20261026.json   (17:50-18:11)
  Knee          data_maxtouch/body_sweep_knee2finger_20261026.json   (17:16-17:42)
  Ankle         data_maxtouch/body_sweep_currents_20261026.json      (15:16-16:29, still)
Panels ordered by body path length to the finger, as paper_fig_body_sites.py. Every set current is drawn; sweeps where
no carrier was detected (body_current_sweeps.py rule) are dashed: 0.5 and 4 mA at every site (TX configuration fault).
Ankle 0.5 mA = the 15:19 sweep of the summary (the 16:29 repeat: no carrier either).
Metric per carrier (cap_ook_rx.analyse_probe 'energy_db'): 10 log10 of the mean squared scan-to-scan change of the
node's delta, (x[n] - x[n-1])^2, in that carrier's ON blocks over its interleaved OFF blocks (2 x 0.5 s each, edges
trimmed); best node of Y0-Y3.
Dotted line = 159.585 kHz, the carrier of every BER run.

   python paper_fig_body_sweeps.py

Output: data_maxtouch/paper_fig_body_sweeps/, drawn at print size (fonts ~9 pt at 100 %, so include without scaling):
  fig_body_sweeps_wide.{pdf,png,svg}    1 x 3, 7.16 in wide   \\begin{figure*} ... \\includegraphics[width=\\textwidth]{...}
  fig_body_sweeps_column.{pdf,png,svg}  3 x 1, 3.5 in wide    \\begin{figure}  ... \\includegraphics[width=\\columnwidth]{...}
  fig_body_sweeps_1mA.{pdf,png,svg}     the three sites at 1 mA in one panel, 3.5 in wide (\\columnwidth)
  fig_body_sweeps.csv
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
SITES = [("Back", "body_sweep_back2finger_20261026.json"),
         ("Knee", "body_sweep_knee2finger_20261026.json"),
         ("Ankle", "body_sweep_currents_20261026.json")]
NODES = 4                                       # Y0-Y3, where the finger sits
CARRIER = 159584.9
RC = {"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
      "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}
INK, INK2, GRID = "#1a1a1a", "#52514e", "#d9d9d9"
CUR = {0.5: dict(color="#CC79A7", marker="v", label="0.5 mA"),       # Okabe-Ito: distinct hues, colour-blind safe
       0.75: dict(color="#009E73", marker="^", label="0.75 mA"),
       1.0: dict(color="#0072B2", marker="o", label="1 mA"),
       2.5: dict(color="#D55E00", marker="s", label="2.5 mA"),
       4.0: dict(color="#000000", marker="D", label="4 mA")}
NO_CARRIER = (0, (3, 1.5))
FS, FS_TICK = 9, 8                              # pt at print size
# IEEE two-column: \textwidth 7.16 in, \columnwidth 3.5 in
LAYOUTS = {"wide": dict(size=(7.16, 2.45), grid=(1, 3), legend_cols=6),
           "column": dict(size=(3.5, 5.4), grid=(3, 1), legend_cols=3)}
SITE_STYLE = {"Back": dict(color="#CC6677", marker="^"),   # Tol muted: hues not used for the currents above
              "Knee": dict(color="#117733", marker="s"),
              "Ankle": dict(color="#332288", marker="o")}


def sweep(run):
    d = os.path.join(D, run)
    meta = json.load(open(os.path.join(d, "meta.json")))
    z = np.load(os.path.join(d, "capture.npz"))
    with contextlib.redirect_stdout(io.StringIO()):
        res, aux = R.analyse_probe(z["t"], z["X"], meta)
    return np.array(aux["car"]), aux["E"][:NODES].max(axis=0)


def load():
    """{site: [(current_ma, run, detected, carriers, best-of-Y0-Y3 dB)]}, every set current, by current"""
    out = {}
    for site, f in SITES:
        rows = json.load(open(os.path.join(D, f)))
        out[site] = [(round(float(r["current_ma"]), 2), r["run"], r["detected"], *sweep(r["run"]))
                     for r in sorted(rows, key=lambda r: r["current_ma"])]
    return out


def plot(data, car, layout, out):
    rows, cols = layout["grid"]
    ticks = [50, 100, 200, 500]
    with plt.rc_context(RC):
        fig, axs = plt.subplots(rows, cols, figsize=layout["size"], sharex=True, sharey=True, layout="constrained")
        fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.03, hspace=0.04)
        for k, (ax, (site, _)) in enumerate(zip(axs.flat, SITES)):
            ax.axvline(CARRIER / 1e3, color=INK2, ls=":", lw=1.0, zorder=1)
            for cur, run, det, c, y in sorted(data[site], key=lambda r: r[2]):     # no-carrier sweeps underneath
                st = CUR[cur]
                ax.plot(c / 1e3, y, color=st["color"], lw=1.1 if det else 0.8, ls="-" if det else NO_CARRIER,
                        marker=st["marker"], ms=3.0, markevery=(k % 3, 6), mec="white", mew=0.3,
                        zorder=3 if det else 2)
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.set_xticklabels([str(t) for t in ticks])
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xlim(car[0] / 1e3 * 0.97, car[-1] / 1e3 * 1.03)
            ax.set_ylim(-10, 70); ax.set_yticks(range(0, 70, 20))
            ax.set_title(f"({'abc'[k]}) {site}", fontsize=FS, fontweight="bold", color=INK, loc="left", pad=3)
            ax.grid(True, which="major", color=GRID, lw=0.5, zorder=0)
            ax.tick_params(labelsize=FS_TICK, colors=INK, width=0.6, length=3, pad=2)
            ax.tick_params(which="minor", width=0.5, length=1.5)
            for lab in ax.get_xticklabels() + ax.get_yticklabels():
                lab.set_fontweight("bold")
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_linewidth(0.6)
            if k == 0:
                ax.text(CARRIER / 1e3 * 1.04, 67, "159.6 kHz", fontsize=FS_TICK - 1, color=INK2, va="top")
        fig.supxlabel("Carrier frequency (kHz)", fontsize=FS, fontweight="bold", color=INK)
        fig.supylabel("SNR (dB)", fontsize=FS, fontweight="bold", color=INK)
        never = {cur for cur in CUR if not any(r[2] for rows in data.values() for r in rows if r[0] == cur)}
        handles = [plt.Line2D([], [], color=st["color"], marker=st["marker"], ms=3.5, mec="white", mew=0.3, lw=1.3,
                              ls=NO_CARRIER if cur in never else "-", label=st["label"]) for cur, st in CUR.items()]
        handles.append(plt.Line2D([], [], color=INK2, ls=NO_CARRIER, lw=0.9, label="no carrier"))
        fig.legend(handles=handles, loc="outside upper center", ncol=layout["legend_cols"], frameon=False,
                   prop={"size": FS_TICK, "weight": "bold"}, handlelength=2.0, columnspacing=1.2, handletextpad=0.5,
                   borderaxespad=0.1)
        for ext in ("pdf", "svg", "png"):
            fig.savefig(f"{out}.{ext}", dpi=600 if ext == "png" else None)
        plt.close(fig)


def plot_sites(data, car, cur, out):
    """one panel: every site at one set current, single column"""
    ticks = [50, 100, 200, 500]
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(3.5, 2.5), layout="constrained")
        fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02)
        ax.axvline(CARRIER / 1e3, color=INK2, ls=":", lw=1.0, zorder=1)
        for k, (site, _) in enumerate(SITES):
            _, run, det, c, y = next(r for r in data[site] if r[0] == cur)
            assert det, f"{run}: no carrier at {cur:g} mA"
            st = SITE_STYLE[site]
            ax.plot(c / 1e3, y, color=st["color"], lw=1.1, marker=st["marker"], ms=3.0, markevery=(2 * k, 6),
                    mec="white", mew=0.3, label=site, zorder=3)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.set_xticklabels([str(t) for t in ticks])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(car[0] / 1e3 * 0.97, car[-1] / 1e3 * 1.03)
        ax.set_ylim(0, 70); ax.set_yticks(range(0, 70, 20))
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
        ax.text(CARRIER / 1e3 * 1.04, 69, "159.6 kHz", fontsize=FS_TICK - 1, color=INK2, va="top")
        fig.legend(loc="outside upper center", ncol=len(SITES), frameon=False, prop={"size": FS_TICK, "weight": "bold"},
                   handlelength=2.0, columnspacing=1.4, handletextpad=0.5, borderaxespad=0.1,
                   title=f"TX electrode site, {cur:g} mA", title_fontproperties={"size": FS_TICK, "weight": "bold"})
        for ext in ("pdf", "svg", "png"):
            fig.savefig(f"{out}.{ext}", dpi=600 if ext == "png" else None)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(D, "paper_fig_body_sweeps"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data = load()
    car = next(iter(data.values()))[0][3]
    with open(os.path.join(a.out, "fig_body_sweeps.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["site", "current_ma", "run", "carrier_detected", "carrier_hz", "snr_db"])
        for site, rows in data.items():
            for cur, run, det, c, y in rows:
                assert np.allclose(c, car), f"{run}: carrier grid differs"
                for ci, yi in zip(c, y):
                    w.writerow([site, cur, run, det, f"{ci:.1f}", f"{yi:.2f}"])

    for name, layout in LAYOUTS.items():
        plot(data, car, layout, os.path.join(a.out, f"fig_body_sweeps_{name}"))
    plot_sites(data, car, 1.0, os.path.join(a.out, "fig_body_sweeps_1mA"))
    j = int(np.argmin(abs(car - CARRIER)))
    for site, rows in data.items():
        print(f"{site}: " + "  ".join(f"{cur:g} mA {y[j]:+.1f} dB (median {np.median(y):+.1f})" + ("" if det else " [no carrier]")
                                      for cur, _, det, _, y in rows))
    print("->", a.out)


if __name__ == "__main__":
    main()
