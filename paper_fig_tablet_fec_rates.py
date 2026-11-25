#!/usr/bin/env python3
"""
Paper figure: tablet (maXTouch) BER after error correction (FEC) vs bit rate for the four TX electrode locations
(shoulder, back, knee, ankle to finger), one line per location, OOK (a) and FSK (b) side by side, one column at print
size (3.5 in).

Data = the four coded sessions of 2026-11-17 (stent_trx_ber.py, K = 4 rate-1/2 convolutional code, soft decision,
'cc4'; 1 mA, still; OOK carrier 159.6 kHz, FSK tones 159.6 / 180.6 kHz), 10 frames (960 info bits) per point,
20-60 bps; the sessions are listed in paper_fig_tablet_fec_locations.py, whose loader is used unchanged (nothing
re-decoded). After FEC = info-bit BER after Viterbi decoding; x = bit rate on air (info rate = 0.375 x that).
Log BER axis; a point with 0 errors is a hollow downward triangle at its 95 % upper bound (about 3/n), not joined to the line. Error bar = lowest to highest BER of the recording chunks behind a point
(paper_runs.py); a point from one recording has no bar. Back at 50 bps pools three sessions (10 + 4 + 4 frames).

   python paper_fig_tablet_fec_rates.py

Output: data_maxtouch/paper_fig_tablet_ber/ (fig_tablet_fec_rates.{pdf,png,svg}, fig_tablet_fec_rates.csv)
   \\includegraphics[width=\\columnwidth]{fig_tablet_fec_rates.pdf}   (3.5 in wide, fonts 7-8 pt at 100 %)
"""
from __future__ import annotations

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import paper_fig_tablet_fec_locations as FL
import paper_runs as PR
import paper_style as PS
from paper_fig_tablet_ber import HERE, RC, cp

RATES = (20, 30, 40, 50, 60)
MODS = {"ook": "OOK", "fsk": "FSK"}
# extra sessions pooled into one point (same TX location, 1 mA, still): two 50 bps back runs of 2026-11-17 16:38 / 16:40
# (4 frames = 384 info bits each) added to the 10 frames of the 14:45 session
EXTRA = {("Back", 50): ["trxber_back2finger_1mA_20261117_163845", "trxber_back2finger_1mA_20261117_164056"]}
LOC = {"Shoulder": dict(color="#b8860b", marker="p"), "Back": PS.PATH["back"], "Knee": PS.PATH["knee"],
       "Ankle": PS.PATH["ankle"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "data_maxtouch", "paper_fig_tablet_ber"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    FL.RATES = RATES
    data = {lab: FL.load(f) for lab, f in FL.SITES}
    for lab, d in data.items():
        missing = [(m, b) for m in MODS for b in RATES if (m, b) not in d]
        assert not missing, (lab, missing)
    # counts per point: the location's session, plus any extra sessions pooled into it
    cnt, runs = {}, {}
    for lab, folder in FL.SITES:
        for m in MODS:
            for b in RATES:
                k, n, fr = data[lab][m, b]
                rr = PR.trx(folder, FL.CODE, m, b, after_fec=True)
                for ex in EXTRA.get((lab, b), []):
                    ek, en, efr = FL.load(ex)[m, b]
                    k += ek; n += en; fr += efr
                    rr = rr + PR.trx(ex, FL.CODE, m, b, after_fec=True)
                cnt[lab, m, b] = (k, n, fr); runs[lab, m, b] = rr
    with open(os.path.join(a.out, "fig_tablet_fec_rates.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["location", "modulation", "bit_rate_on_air_bps", "frames", "post_fec_errors", "post_fec_bits",
                    "post_fec_ber", "ci95_hi"])
        for lab, _ in FL.SITES:
            for m in MODS:
                for b in RATES:
                    k, n, fr = cnt[lab, m, b]
                    w.writerow([lab, m, b, fr, k, n, f"{k / n:.3e}", f"{cp(k, n)[1]:.3e}"])

    with plt.rc_context(RC):
        fig, axes = PS.figure(3.5, 2.0, 1, 2, wspace=0.08, sharey=True)
        for ax, (m, name), letter in zip(axes, MODS.items(), "ab"):
            for j, (lab, folder) in enumerate(FL.SITES):
                k = np.array([cnt[lab, m, b][0] for b in RATES]); n = np.array([cnt[lab, m, b][1] for b in RATES])
                hi = np.array([cp(*cnt[lab, m, b][:2])[1] for b in RATES])
                x = np.array(RATES, float) + (j - 1.5) * 0.9
                zero = k == 0
                sp = [None if z else PR.spread(runs[lab, m, b]) for z, b in zip(zero, RATES)]
                PS.series(ax, x, np.where(zero, np.nan, k / n), sp, clip_on=False,
                          **PS.marks(LOC[lab]["color"], LOC[lab]["marker"], ms=PS.MS - 0.4, label=lab))
                if zero.any():          # 0 errors has no place on a log axis: hollow triangle at the 95 % upper bound
                    ax.plot(x[zero], hi[zero], zorder=4, clip_on=False,
                            **PS.marks(LOC[lab]["color"], "v", hollow=True, ls="none", ms=PS.MS - 1.0))
            ax.set_xticks(RATES); ax.set_xlim(RATES[0] - 4, RATES[-1] + 4)
            ax.set_xlabel("Bit rate (bps)")
            PS.note(ax, name, x=0.5, ha="center")
            PS.panel(ax, letter, dx=-27 if letter == "a" else -6)
            PS.finish(ax)
        axes[0].set_yscale("log"); axes[0].set_ylim(1e-4, 1)
        axes[0].set_ylabel("Bit error rate after FEC")
        h, l = axes[0].get_legend_handles_labels()
        PS.top_legend(fig, h, l, ncol=4)
        PS.save(fig, os.path.join(a.out, "fig_tablet_fec_rates"))
    for lab, _ in FL.SITES:
        print(lab.ljust(9), "  ".join(f"{m.upper()} " + " ".join(f"{cnt[lab, m, b][0]:>2}/{cnt[lab, m, b][1]}" for b in RATES) for m in MODS))
    print("->", os.path.join(a.out, "fig_tablet_fec_rates.pdf"))


if __name__ == "__main__":
    main()
