#!/usr/bin/env python3
"""
Paper figure: tablet (maXTouch) BER after error correction (FEC) per body state - sitting still, standing still,
sitting shaking, standing walking, ankle to finger - at 40 and 60 bps, OOK and FSK, one column at print size (3.5 in),
same layout as fig_tablet_fec_states / fig_tablet_states_line_40_60bps (the two moving states on a faint band).

Data = four coded sessions recorded with stent_trx_ber.py (the stent link's frames: 96 info bits = 80 data + CRC-16,
convolutional code K = 4, rate 1/2, soft decision - 'cc4'), TX electrode on the ankle, finger on the panel, 1 mA,
OOK carrier 159.6 kHz, FSK tones 159.6 / 180.6 kHz; summaries made by `stent_trx_ber.py analyse` (nothing re-decoded).
The sessions were run in this order (as told by the experimenter):
  Sitting still     data_maxtouch/trxber_ankle2finger_1mA_20261117_160723   10 frames (960 info bits) per point
  Sitting shaking   data_maxtouch/trxber_ankle2finger_1mA_20261118_132517   8 frames (768 info bits)
  Standing still    data_maxtouch/trxber_ankle2finger_1mA_20261118_135344   8 frames (768 info bits)
  Standing walking  data_maxtouch/trxber_ankle2finger_1mA_20261118_142352   8 frames (768 info bits)
After FEC = info-bit BER after Viterbi decoding. x = bit rate on air; the info rate is 0.375 x that.
Log BER axis; a point with 0 errors is a hollow downward triangle at its 95 % upper bound (about 3/n), not joined to the line. Error bar = lowest to highest BER of
the recording chunks behind a point (paper_runs.py); a point is one recording, so the bar spans its frames
(96 info bits each; 8 or 10 per point, paper_runs.trx_frames): lowest to highest per-frame BER, down to the axis where a frame was
error-free. The pooled value is errors / bits of all the frames.

   python paper_fig_tablet_fec_states.py

Output: data_maxtouch/paper_fig_tablet_ber/ (fig_tablet_fec_states.{pdf,png,svg}, fig_tablet_fec_states.csv)
   \\includegraphics[width=\\columnwidth]{fig_tablet_fec_states.pdf}   (3.5 in wide, fonts 7-8 pt at 100 %)
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

import paper_fig_tablet_states as ST
import paper_style as PS
from paper_fig_tablet_ber import HERE, RC, STYLE, cp

D = os.path.join(HERE, "data_maxtouch")
SITES = [("Sitting\nstill", "trxber_ankle2finger_1mA_20261117_160723"),
         ("Standing\nstill", "trxber_ankle2finger_1mA_20261118_135344"),
         ("Sitting\nshaking", "trxber_ankle2finger_1mA_20261118_132517"),
         ("Standing\nwalking", "trxber_ankle2finger_1mA_20261118_142352")]
RATES = (40, 60)
RATE_LS = {40: "-", 60: PS.DASH}
CODE = "cc4"


def load(folder):
    """{(modulation, rate): (post_err, post_bits, frames)} for the cc4 frames"""
    out = {}
    for r in csv.DictReader(open(os.path.join(D, folder, "summary.csv"))):
        if r["decoder"] == CODE:
            out[r["modulation"], round(float(r["baud"]))] = (int(r["post_fec_errors"]), int(r["post_fec_bits"]),
                                                             int(r["frames"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(D, "paper_fig_tablet_ber"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data = {lab: load(f) for lab, f in SITES}
    for lab, d in data.items():
        missing = [(m, b) for m in ("ook", "fsk") for b in RATES if (m, b) not in d]
        assert not missing, (lab, missing)
    with open(os.path.join(a.out, "fig_tablet_fec_states.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["location", "modulation", "bit_rate_on_air_bps", "frames", "post_fec_errors", "post_fec_bits",
                    "post_fec_ber", "ci95_hi"])
        for lab, folder in SITES:
            for m in ("ook", "fsk"):
                for b in RATES:
                    k, n, fr = data[lab][m, b]
                    w.writerow([lab.replace("\n", " "), m, b, fr, k, n, f"{k / n:.3e}", f"{cp(k, n)[1]:.3e}"])

    with plt.rc_context(RC):
        fig, ax = PS.figure(3.5, 2.5)
        x = np.arange(len(SITES))
        for i, b in enumerate(RATES):
            for m in ("ook", "fsk"):
                st = STYLE[m]
                k = np.array([data[lab][m, b][0] for lab, _ in SITES]); n = np.array([data[lab][m, b][1] for lab, _ in SITES])
                hi = np.array([cp(*data[lab][m, b][:2])[1] for lab, _ in SITES])
                dx = (-0.04 if m == "ook" else 0.04) + (-0.015 if i == 0 else 0.015)   # zeros coincide: keep all visible
                zero = k == 0
                sp = [None if z else PR.spread(PR.trx_frames(fo, CODE, m, b, after_fec=True)) for z, (_, fo) in zip(zero, SITES)]
                PS.series(ax, x + dx, np.where(zero, np.nan, k / n), sp, clip_on=False,
                          **PS.marks(st["color"], st["marker"], hollow=i > 0, ls=RATE_LS[b],
                                     label=f"{st['label']}, {b} bps"))
                if zero.any():      # 0 errors has no place on a log axis: hollow triangle at the 95 % upper bound
                    ax.plot(x[zero] + dx, hi[zero], zorder=4, clip_on=False,
                            **PS.marks(st["color"], "v", hollow=True, ls="none", ms=PS.MS - 1.0))
        ST.state_axis(ax)
        ax.set_yscale("log"); ax.set_ylim(1e-4, 1)
        ax.set_xlabel("Body state"); ax.set_ylabel("Bit error rate after FEC")
        h, l = ax.get_legend_handles_labels()
        PS.top_legend(fig, h, l, ncol=2, handlelength=2.6)
        PS.save(fig, os.path.join(a.out, "fig_tablet_fec_states"))
    for lab, _ in SITES:
        print(lab.replace("\n", " ").ljust(15), "  ".join(f"{m.upper()} {b}: {data[lab][m, b][0]}/{data[lab][m, b][1]}"
                                                          for b in RATES for m in ("ook", "fsk")))
    print("->", os.path.join(a.out, "fig_tablet_fec_states.pdf"))


if __name__ == "__main__":
    main()
