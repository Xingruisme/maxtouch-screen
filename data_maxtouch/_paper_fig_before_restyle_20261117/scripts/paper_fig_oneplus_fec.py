#!/usr/bin/env python3
"""
Paper figure: OnePlus Pad (OPD2203) through-body BER after error correction (FEC) vs bit rate, OOK and FSK, one column
at print size (3.5 in), in the style of the tablet (maXTouch) figures.

Data = the coded through-body session of 2026-11-14 23:20-00:03 in
../screen-bloodflow-core-main/data_oneplus/trx_ber/trxber_ankle_full_1mA_20261114_232032/summary.json (nothing
re-decoded): TX electrode on the ankle, finger on the pad, still, 1 mA; OOK carrier 37.4 kHz, FSK tones 37.4 / 445 kHz;
the stent link's frames (96 info bits = 80 data + CRC-16), convolutional code K = 4, rate 1/2, soft decision ('cc4');
3 frames = 288 info bits per point. After FEC = info-bit BER after Viterbi decoding.
x = bit rate on air (code bits per second, as the session's own ber_vs_bitrate_pad figure); the info rate is
0.375 x that (rate-1/2 code, frame overhead).
Linear BER axis: 5 of the 6 points have no error after FEC, and a log axis cannot show 0. Error bar = lowest to
highest BER of the 3 recording chunks (one frame each) behind each point (paper_runs.py).
Not used: the 1-frame OOK 2 bps re-run of 2026-11-15 01:23 (trxber_ankle_full_1mA_20261115_012316: 50 / 96 wrong,
the frame was lost).

   python paper_fig_oneplus_fec.py

Output: data_maxtouch/paper_fig_oneplus/ (fig_oneplus_fec.{pdf,png,svg}, fig_oneplus_fec.csv)
   \\includegraphics[width=\\columnwidth]{fig_oneplus_fec.pdf}   (3.5 in wide, fonts ~9 pt at 100 %)
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import paper_runs as PR
import matplotlib.pyplot as plt
import numpy as np

from paper_fig_tablet_ber import HERE, RC, STYLE, cp
from paper_fig_tablet_snr import FS, FS_TICK, INK

SRC = os.path.join(HERE, "..", "screen-bloodflow-core-main", "data_oneplus", "trx_ber",
                   "trxber_ankle_full_1mA_20261114_232032", "summary.json")
CODE = "cc4"
MODS = ("ook", "fsk")
DX = {"ook": -0.03, "fsk": 0.03}               # OOK and FSK drawn 0.06 bps apart: both 0 at 3 and 4 bps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=os.path.join(HERE, "data_maxtouch", "paper_fig_oneplus"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = [r for r in json.load(open(a.src))["summary"] if r["decoder"] == CODE and r["modulation"] in MODS]
    data = {(r["modulation"], round(r["baud"])): (r["post_fec_errors"], r["post_fec_bits"], r["info_bps"]) for r in rows}
    rates = sorted({b for _, b in data})
    with open(os.path.join(a.out, "fig_oneplus_fec.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["modulation", "bit_rate_on_air_bps", "info_bps", "post_fec_errors", "post_fec_bits", "post_fec_ber",
                    "ci95_lo", "ci95_hi"])
        for (m, b), (k, n, ib) in sorted(data.items()):
            w.writerow([m, b, ib, k, n, f"{k / n:.3e}", *(f"{x:.3e}" for x in cp(k, n))])

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(3.5, 2.4), layout="constrained")
        fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02)
        for m in MODS:
            st = STYLE[m]
            x = np.array(rates, float) + DX[m]
            pct = [100 * data[m, b][0] / data[m, b][1] for b in rates]
            sp = [PR.spread(PR.trx(os.path.basename(os.path.dirname(a.src)), CODE, m, b, after_fec=True,
                                   base=os.path.dirname(os.path.dirname(a.src))), 100) for b in rates]
            ax.errorbar(x, pct, yerr=PR.yerr(pct, sp), color=st["color"], marker=st["marker"], ms=4.5, lw=1.3,
                        label=st["label"], zorder=3, clip_on=False, **PR.EB)
        ax.set_xticks(rates); ax.set_xlim(rates[0] - 0.3, rates[-1] + 0.3)
        ax.set_ylim(0, 25)
        ax.set_xlabel("Bit rate (bps)", fontsize=FS, color=INK, labelpad=2)
        ax.set_ylabel("Bit error rate after FEC (%)", fontsize=FS, color=INK, labelpad=2)
        ax.tick_params(labelsize=FS_TICK, colors=INK, width=0.6, length=3, pad=2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_linewidth(0.6)
        ax.legend(loc="upper right", frameon=False, prop={"size": FS_TICK}, handlelength=1.8, handletextpad=0.4,
                  borderaxespad=0.2)
        for ext in ("pdf", "svg", "png"):
            fig.savefig(os.path.join(a.out, f"fig_oneplus_fec.{ext}"), dpi=600 if ext == "png" else None)
        plt.close(fig)
    for (m, b), (k, n, ib) in sorted(data.items()):
        print(f"{m} {b} bps on air ({ib:g} info bps): {k}/{n} after FEC")
    print("->", os.path.join(a.out, "fig_oneplus_fec.pdf"))


if __name__ == "__main__":
    main()
