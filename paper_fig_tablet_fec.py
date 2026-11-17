#!/usr/bin/env python3
"""
Paper figure: tablet (maXTouch) through-body BER before and after error correction (FEC), OOK and FSK, one column with
the two panels side by side (3.5 x 2.1 in, the layout of fig_tablet_ber_snr / the paper's Figure 13).

Data = the 2026-11-14 coded runs of stent_trx_ber.py (the stent link's own frames: 96 info bits = 80 data + CRC-16,
convolutional code K = 4, rate 1/2, soft decision - 'cc4', Bluetooth LE Coded), TX electrode on the hand, finger on
the panel, still, 1 mA, 12 frames per rate; summaries made by `stent_trx_ber.py analyse` (nothing re-decoded):
  a  OOK, carrier 168 kHz                   data_maxtouch/trxber_hand_trx_1mA_20261114_133911, modulation ook
  b  FSK, tones 159 / 168 kHz               data_maxtouch/trxber_hand_trx_fsk159_168_1mA_20to60
The FSK runs of the first session (tones 168 / 171 kHz, 3 kHz apart) are left out: 20-37 % of the info bits wrong
even after FEC - the panel cannot tell the two tones apart.
Before FEC = code-bit BER of the cc4 frames as the decoder sees them (2304 code bits per rate); after FEC = info-bit
BER of the same frames after Viterbi decoding (1152 info bits per rate). A point with 0 errors after FEC is drawn as a
hollow downward triangle at its 95 % upper bound (3 / 1152). Error bar = lowest to highest BER of the 2 recording
chunks behind each point (paper_runs.py). x = symbol rate on the air; the info
rate is 0.375 x the symbol rate (rate-1/2 code, frame overhead).

   python paper_fig_tablet_fec.py

Output: data_maxtouch/paper_fig_tablet_ber/ (fig_tablet_fec.{pdf,png,svg}, fig_tablet_fec.csv)
   \\includegraphics[width=\\columnwidth]{fig_tablet_fec.pdf}   (3.5 in wide, fonts 6-8 pt at 100 %)
"""
from __future__ import annotations

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import paper_runs as PR
import matplotlib.pyplot as plt
import numpy as np

import paper_style as PS
from paper_fig_tablet_ber import HERE, RC, STYLE, cp
from paper_fig_tablet_ber_snr import tidy

D = os.path.join(HERE, "data_maxtouch")
PANELS = [("a", "OOK, 168 kHz", "ook", "trxber_hand_trx_1mA_20261114_133911"),
          ("b", "FSK, 159 / 168 kHz", "fsk", "trxber_hand_trx_fsk159_168_1mA_20to60")]
CODE = "cc4"


def load(folder, mod):
    """{baud: (pre_err, pre_bits, post_err, post_bits, info_bps)} for the cc4 frames"""
    out = {}
    for r in csv.DictReader(open(os.path.join(D, folder, "summary.csv"))):
        if r["modulation"] == mod and r["decoder"] == CODE:
            out[round(float(r["baud"]))] = (int(r["pre_fec_errors"]), int(r["pre_fec_bits"]),
                                            int(r["post_fec_errors"]), int(r["post_fec_bits"]), float(r["info_bps"]))
    assert out, (folder, mod)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(D, "paper_fig_tablet_ber"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data = {p[0]: load(p[3], p[2]) for p in PANELS}
    with open(os.path.join(a.out, "fig_tablet_fec.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["panel", "modulation", "baud", "info_bps", "pre_fec_errors", "pre_fec_bits", "pre_fec_ber",
                    "post_fec_errors", "post_fec_bits", "post_fec_ber", "post_ci95_hi"])
        for letter, title, mod, _ in PANELS:
            for baud, (pe, pn, qe, qn, ib) in sorted(data[letter].items()):
                w.writerow([letter, title, baud, f"{ib:g}", pe, pn, f"{pe / pn:.3e}", qe, qn, f"{qe / qn:.3e}",
                            f"{cp(qe, qn)[1]:.3e}"])

    with plt.rc_context(RC):
        fig, axs = PS.figure(3.5, 2.1, 1, 2, sharey=True)
        for ax, (letter, title, mod, folder) in zip(axs, PANELS):
            c = STYLE[mod]["color"]
            rates = sorted(data[letter])
            x = np.array(rates, float)
            pre = [data[letter][b][0] / data[letter][b][1] for b in rates]
            k = np.array([data[letter][b][2] for b in rates]); n = np.array([data[letter][b][3] for b in rates])
            zero = k == 0
            post = np.where(zero, np.nan, k / np.maximum(n, 1))          # no point and no line where 0 errors
            PS.series(ax, x - 0.6, pre, [PR.spread(PR.trx(folder, CODE, mod, b)) for b in rates],
                      **PS.marks(c, "o", hollow=True, ls=PS.DASH))
            PS.series(ax, x + 0.6, post, [None if z else PR.spread(PR.trx(folder, CODE, mod, b, after_fec=True))
                                          for z, b in zip(zero, rates)], **PS.marks(c, "o", ms=PS.MS - 0.4))
            if zero.any():
                ax.plot(x[zero] + 0.6, [cp(0, int(m))[1] for m in n[zero]], zorder=4,
                        **PS.marks(c, "v", hollow=True, ls="none"))
            ax.set_yscale("log"); ax.set_ylim(1e-3, 1)
            tidy(ax, rates, "Bit error rate" if letter == "a" else "")
            ax.set_xlabel("Symbol rate (baud)")
            PS.note(ax, title)
            PS.panel(ax, letter, dx=-22 if letter == "a" else -6)
        # one neutral key above both panels (colour = modulation, given by each panel's title)
        PS.top_legend(fig, [PS.key(hollow=True, ls=PS.DASH), PS.key(ms=PS.MS - 0.4), PS.key(marker="v", hollow=True, ls="none")],
                      ["Before FEC", "After FEC", "0 errors (95% bound)"], columnspacing=1.0, handletextpad=0.4)
        PS.save(fig, os.path.join(a.out, "fig_tablet_fec"))
    for letter, title, mod, _ in PANELS:
        print(title + ": " + "  ".join(f"{b} baud {pe}/{pn} -> {qe}/{qn}" for b, (pe, pn, qe, qn, _) in
                                       sorted(data[letter].items())))
    print("->", os.path.join(a.out, "fig_tablet_fec.pdf"))


if __name__ == "__main__":
    main()
