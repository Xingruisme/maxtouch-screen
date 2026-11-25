#!/usr/bin/env python3
"""
Paper figure: 2-FSK vs 4-FSK through the body to the maXTouch tablet at the same bit rate (40 bps), uncoded frames,
coded frames before FEC (code bits) and after FEC (info bits), one column at print size (3.5 in).

Data = data_maxtouch/fsk4ber_fsk2v4_body_nognd_1mA_20261120_205801 (fsk4_trx_ber.py analyse; through body, ground line
removed, 1 mA; 4 frames of 96 info bits per bar; nothing re-decoded). 2-FSK = the M-FSK decoder with M = 2 at 40 baud;
4-FSK = the M-FSK decoder with M = 4 at 20 baud (2 bits per symbol). The frozen-tablet-decoder 2-FSK result of the same
session is left out (only the new 2-FSK result is kept). Bar = errors / bits of the 4 frames, dots = the 4 frames,
error bar = lowest to highest frame (paper_runs-style). Linear axis.

   python paper_fig_tablet_fsk2v4.py

Output: data_maxtouch/paper_fig_tablet_ber/ (fig_tablet_fsk2v4.{pdf,png,svg}, fig_tablet_fsk2v4.csv)
   \\\\includegraphics[width=\\\\columnwidth]{fig_tablet_fsk2v4.pdf}   (3.5 in wide, fonts 7-8 pt at 100 %)
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
from paper_fig_tablet_ber import HERE, RC, cp

SESSION = os.path.join(HERE, "data_maxtouch", "fsk4ber_fsk2v4_body_nognd_1mA_20261120_205801")
SCHEMES = [("2-FSK", "2-FSK, M-FSK decoder (M=2)"), ("4-FSK", "4-FSK, M-FSK decoder (M=4)")]
TEAL = PS.STYLE["fsk"]["color"]
STAGES = [("Uncoded", "none_full", "info", "#9a9a9a"), ("Coded, before FEC", "cc4", "pre", "#99d0c8"),
          ("Coded, after FEC", "cc4", "info", TEAL)]


def frames(scheme, frame, kind):
    """[(errors, bits)] per frame"""
    ke, kn = ("info_err", "info_bits") if kind == "info" else ("pre_err", "code_bits")
    return [(int(r[ke]), int(r[kn])) for r in csv.DictReader(open(os.path.join(SESSION, "frames.csv")))
            if r["scheme"] == scheme and r["frame"] == frame]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "data_maxtouch", "paper_fig_tablet_ber"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data = {(s, st): frames(full, fr, kind) for s, full in SCHEMES for st, fr, kind, _ in STAGES}
    with open(os.path.join(a.out, "fig_tablet_fsk2v4.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scheme", "stage", "frames", "errors", "bits", "ber", "ci95_hi", "frame_bers"])
        for (s, st), d in data.items():
            k, n = sum(e for e, _ in d), sum(b for _, b in d)
            w.writerow([s, st, len(d), k, n, f"{k / n:.3e}", f"{cp(k, n)[1]:.3e}", " ".join(f"{e / b:.3f}" for e, b in d)])
    with plt.rc_context(RC):
        fig, ax = PS.figure(3.5, 2.3)
        x = np.arange(len(SCHEMES)); wd = 0.26; top = 0.0
        for j, (st, _, _, col) in enumerate(STAGES):
            y = [100 * sum(e for e, _ in data[s, st]) / sum(b for _, b in data[s, st]) for s, _ in SCHEMES]
            runs = [[100 * e / b for e, b in data[s, st]] for s, _ in SCHEMES]
            top = max(top, max(max(r) for r in runs))
            PS.bars(ax, x + (j - 1) * (wd + 0.03), y, runs, col, wd, st)
        ax.set_xticks(x, [s for s, _ in SCHEMES])
        PS.pct_axis(ax, top)
        ax.set_xlabel("Modulation (40 bps)"); ax.set_ylabel("Bit error rate (%)")
        PS.finish(ax, categorical=True)
        h, l = ax.get_legend_handles_labels()
        PS.top_legend(fig, h, l, ncol=3, handlelength=1.0, columnspacing=0.8)
        PS.save(fig, os.path.join(a.out, "fig_tablet_fsk2v4"))
    for (s, st), d in data.items():
        print(f"{s:6s} {st:18s} {sum(e for e, _ in d)}/{sum(b for _, b in d)}  per frame {[e for e, _ in d]}")
    print("->", os.path.join(a.out, "fig_tablet_fsk2v4.pdf"))


if __name__ == "__main__":
    main()
