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
Dashed rule = 159.585 kHz, the carrier of every BER run.

   python paper_fig_body_sweeps.py

Output: data_maxtouch/paper_fig_body_sweeps/, drawn at print size (fonts 6-7 pt at 100 %, so include without scaling):
  fig_body_sweeps_wide.{pdf,png,svg}    1 x 3, 7.16 in wide   \\begin{figure*} ... \\includegraphics[width=\\textwidth]{...}
  fig_body_sweeps_column.{pdf,png,svg}  3 x 1, 3.5 in wide    \\begin{figure}  ... \\includegraphics[width=\\columnwidth]{...}
  fig_body_sweeps_1mA.{pdf,png,svg}     the three sites at 1 mA in one panel, 3.5 in wide (\\columnwidth), in the
                                        colours of fig_freq_sweep (paper_style.PATH)
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
import numpy as np

import maxtouch_stent_rx as R
import paper_style as PS

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data_maxtouch")
SITES = [("Back", "body_sweep_back2finger_20261026.json"),
         ("Knee", "body_sweep_knee2finger_20261026.json"),
         ("Ankle", "body_sweep_currents_20261026.json")]
NODES = 4                                       # Y0-Y3, where the finger sits
CARRIER = 159584.9
CUR = {0.5: dict(color="#CC79A7", marker="v", label="0.5 mA"),       # Okabe-Ito: distinct hues, colour-blind safe
       0.75: dict(color="#009E73", marker="^", label="0.75 mA"),
       1.0: dict(color="#0072B2", marker="o", label="1 mA"),
       2.5: dict(color="#D55E00", marker="s", label="2.5 mA"),
       4.0: dict(color="#000000", marker="D", label="4 mA")}
# IEEE two-column: \textwidth 7.16 in, \columnwidth 3.5 in
LAYOUTS = {"wide": dict(size=(7.16, 2.3), grid=(1, 3), legend_cols=6),
           "column": dict(size=(3.5, 5.0), grid=(3, 1), legend_cols=3)}


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


def cur_key(cur, solid=True):
    st = CUR[cur]
    return plt.Line2D([], [], color=st["color"], marker=st["marker"], ms=3.0, mec="white", mew=0.4, lw=PS.LW,
                      ls="-" if solid else PS.DASH, label=st["label"])


def plot(data, car, layout, out):
    rows, cols = layout["grid"]
    with plt.rc_context(PS.RC):
        fig, axs = PS.figure(*layout["size"], rows, cols, sharex=True, sharey=True, wspace=0.03, hspace=0.04)
        for k, (ax, (site, _)) in enumerate(zip(axs.flat, SITES)):
            PS.vmark(ax, CARRIER / 1e3, "159.6 kHz" if k == 0 else None)
            for cur, run, det, c, y in sorted(data[site], key=lambda r: r[2]):     # no-carrier sweeps underneath
                PS.trace(ax, c / 1e3, y, CUR[cur]["color"], CUR[cur]["marker"], (k % 3, 6), solid=det)
            PS.khz_axis(ax, car)
            ax.set_ylim(-10, 70); ax.set_yticks(range(0, 70, 20))
            PS.finish(ax)
            PS.note(ax, site)
            PS.panel(ax, "abc"[k], dx=-6 if cols > 1 and k else -20)
        fig.supxlabel("Carrier frequency (kHz)", fontsize=PS.FS); fig.supylabel("SNR (dB)", fontsize=PS.FS)
        never = {cur for cur in CUR if not any(r[2] for rows in data.values() for r in rows if r[0] == cur)}
        handles = [cur_key(cur, cur not in never) for cur in CUR]
        handles.append(plt.Line2D([], [], color=PS.INK2, ls=PS.DASH, lw=0.7, label="no carrier"))
        PS.top_legend(fig, handles, ncol=layout["legend_cols"], handlelength=2.2)
        PS.save(fig, out)


def plot_sites(data, car, cur, out):
    """one panel: every site at one set current, single column"""
    with plt.rc_context(PS.RC):
        fig, ax = PS.figure(3.5, 2.4)
        PS.vmark(ax, CARRIER / 1e3, "159.6 kHz")
        for k, (site, _) in enumerate(SITES):
            _, run, det, c, y = next(r for r in data[site] if r[0] == cur)
            assert det, f"{run}: no carrier at {cur:g} mA"
            st = PS.PATH[site.lower()]
            PS.trace(ax, c / 1e3, y, st["color"], st["marker"], (2 * k, 6), label=site)
        PS.khz_axis(ax, car)
        ax.set_ylim(0, 70); ax.set_yticks(range(0, 70, 20))
        ax.set_xlabel("Carrier frequency (kHz)"); ax.set_ylabel("SNR (dB)")
        PS.finish(ax)
        PS.top_legend(fig, *ax.get_legend_handles_labels(), handlelength=2.2,
                      title=f"TX electrode location, {cur:g} mA", title_fontproperties={"size": PS.FS_TICK})
        PS.save(fig, out)


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
