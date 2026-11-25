#!/usr/bin/env python3
"""
Paper figure: tablet (maXTouch) BER of the three most recent coded recordings of 2026-11-19 (the 'o2i' experiment),
OOK at 40 bps, 1 mA, before and after error correction (FEC), one column at print size (3.5 in).

Data = three conditions, each 3 recordings of one frame (96 info bits) = 3 frames, recorded with stent_trx_ber.py (the
stent link's frames, convolutional code K = 4, rate 1/2, soft decision - 'cc4'); summaries made by
`stent_trx_ber.py analyse --current-ma 1` (nothing re-decoded):
  phantom_tx1  data_maxtouch/trxber_phantom_tx1_1mA_20261119_183310   (named 'phantom' on disk; the run's meta.json
                                                                      records the correction: THROUGH BODY, finger on the screen)
  dtap_check   data_maxtouch/trxber_dtap_check_1mA_20261119_200117
  dtap_tx1     data_maxtouch/trxber_dtap_tx1_1mA_20261119_200331
Before FEC = BER of the 192 code bits per frame; after FEC = info-bit BER after Viterbi decoding (96 info bits per
frame). Each bar = errors / bits of the 3 frames; error bar = lowest to highest per-frame BER, dots = the 3 frames
(paper_runs.trx_frames). Linear axis.

   python paper_fig_tablet_o2i.py

Output: data_maxtouch/paper_fig_tablet_ber/ (fig_tablet_o2i.{pdf,png,svg}, fig_tablet_o2i.csv)
   \\includegraphics[width=\\columnwidth]{fig_tablet_o2i.pdf}   (3.5 in wide, fonts 7-8 pt at 100 %)
"""
from __future__ import annotations

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import paper_runs as PR
import paper_style as PS
from paper_fig_tablet_ber import HERE, RC, cp

CODE = "cc4"
RATE = 40
CONDS = [("phantom_tx1", "trxber_phantom_tx1_1mA_20261119_183310"),
         ("dtap_check", "trxber_dtap_check_1mA_20261119_200117"),
         ("dtap_tx1", "trxber_dtap_tx1_1mA_20261119_200331")]
OOK = PS.STYLE["ook"]["color"]
LIGHT = "#a9c0ea"                                   # OOK blue, before FEC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "data_maxtouch", "paper_fig_tablet_ber"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pre = {c: PR.trx_frames(f, CODE, "ook", RATE) for c, f in CONDS}
    post = {c: PR.trx_frames(f, CODE, "ook", RATE, after_fec=True) for c, f in CONDS}
    with open(os.path.join(a.out, "fig_tablet_o2i.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["condition", "stage", "frames", "errors", "bits", "ber", "ci95_hi", "frame_bers"])
        for c, _ in CONDS:
            for stage, d in (("before_fec", pre), ("after_fec", post)):
                k = sum(e for e, _ in d[c]); n = sum(b for _, b in d[c])
                w.writerow([c, stage, len(d[c]), k, n, f"{k / n:.3e}", f"{cp(k, n)[1]:.3e}",
                            " ".join(f"{e / b:.3f}" for e, b in d[c])])
    with plt.rc_context(RC):
        fig, ax = PS.figure(3.5, 2.3)
        x = np.arange(len(CONDS)); wd = 0.36
        top = 0.0
        for j, (stage, d, col, lab) in enumerate((("before", pre, LIGHT, "Before FEC"), ("after", post, OOK, "After FEC"))):
            y = [100 * sum(e for e, _ in d[c]) / sum(b for _, b in d[c]) for c, _ in CONDS]
            runs = [[100 * e / b for e, b in d[c]] for c, _ in CONDS]
            top = max(top, max(max(r) for r in runs))
            PS.bars(ax, x + (j - 0.5) * (wd + 0.05), y, runs, col, wd, lab)
        ax.set_xticks(x, [c.replace("_", " ") for c, _ in CONDS])
        PS.pct_axis(ax, top)
        ax.set_xlabel("Recording"); ax.set_ylabel("Bit error rate (%)")
        PS.finish(ax, categorical=True)
        h, l = ax.get_legend_handles_labels()
        PS.top_legend(fig, h, l)
        PS.save(fig, os.path.join(a.out, "fig_tablet_o2i"))
    for c, _ in CONDS:
        print(c.ljust(12), "before", f"{sum(e for e, _ in pre[c])}/{sum(b for _, b in pre[c])}",
              "after", f"{sum(e for e, _ in post[c])}/{sum(b for _, b in post[c])}",
              "per-frame after:", [e for e, _ in post[c]])
    print("->", os.path.join(a.out, "fig_tablet_o2i.pdf"))


if __name__ == "__main__":
    main()
