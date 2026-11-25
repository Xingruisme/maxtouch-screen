#!/usr/bin/env python3
"""
Paper figure: tablet (maXTouch) BER after error correction (FEC) vs bit rate for the four body states (sitting still,
standing still, sitting shaking, standing walking; ankle to finger), one line per state, OOK (a) and FSK (b) side by
side, one column at print size (3.5 in). The two moving states are dashed.

Data = the four coded sessions listed in paper_fig_tablet_fec_states.py (stent_trx_ber.py, K = 4 rate-1/2
convolutional code, soft decision, 'cc4'; 1 mA; OOK carrier 159.6 kHz, FSK tones 159.6 / 180.6 kHz), 20-60 bps,
960 (sitting still) or 768 (the others) info bits per point; its loader is used unchanged (nothing re-decoded).
After FEC = info-bit BER after Viterbi decoding; x = bit rate on air (info rate = 0.375 x that).
Log BER axis; a point with 0 errors is a hollow downward triangle at its 95 % upper bound (about 3/n), not joined to the line. Error bar = lowest to highest BER of the recording chunks behind a point
(paper_runs.py); a point is one recording, so the bar spans its frames
(96 info bits each; 8 or 10 per point, paper_runs.trx_frames): lowest to highest per-frame BER, down to the axis where a frame was
error-free. The pooled value is errors / bits of all the frames.

   python paper_fig_tablet_fec_states_rates.py

Output: data_maxtouch/paper_fig_tablet_ber/ (fig_tablet_fec_states_rates.{pdf,png,svg}, fig_tablet_fec_states_rates.csv)
   \\includegraphics[width=\\columnwidth]{fig_tablet_fec_states_rates.pdf}   (3.5 in wide, fonts 7-8 pt at 100 %)
"""
from __future__ import annotations

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import paper_fig_tablet_fec_states as FL
import paper_runs as PR
import paper_style as PS
from paper_fig_tablet_ber import HERE, RC, cp

RATES = (20, 30, 40, 50, 60)
MODS = {"ook": "OOK", "fsk": "FSK"}
EXTRA = {}
LOC = {"Sitting\nstill": dict(color="#7b3294", marker="o", ls="-"),
       "Standing\nstill": dict(color="#D55E00", marker="s", ls="-"),
       "Sitting\nshaking": dict(color="#009E73", marker="^", ls=PS.DASH),
       "Standing\nwalking": dict(color="#555555", marker="D", ls=PS.DASH)}


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
                rr = PR.trx_frames(folder, FL.CODE, m, b, after_fec=True)
                for ex in EXTRA.get((lab, b), []):
                    ek, en, efr = FL.load(ex)[m, b]
                    k += ek; n += en; fr += efr
                    rr = rr + PR.trx(ex, FL.CODE, m, b, after_fec=True)
                cnt[lab, m, b] = (k, n, fr); runs[lab, m, b] = rr
    with open(os.path.join(a.out, "fig_tablet_fec_states_rates.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["state", "modulation", "bit_rate_on_air_bps", "frames", "post_fec_errors", "post_fec_bits",
                    "post_fec_ber", "ci95_hi"])
        for lab, _ in FL.SITES:
            for m in MODS:
                for b in RATES:
                    k, n, fr = cnt[lab, m, b]
                    w.writerow([lab.replace("\n", " "), m, b, fr, k, n, f"{k / n:.3e}", f"{cp(k, n)[1]:.3e}"])

    with plt.rc_context(RC):
        fig, axes = PS.figure(3.5, 2.0, 1, 2, wspace=0.08, sharey=True)
        top = 0.0
        for ax, (m, name), letter in zip(axes, MODS.items(), "ab"):
            for j, (lab, folder) in enumerate(FL.SITES):
                k = np.array([cnt[lab, m, b][0] for b in RATES]); n = np.array([cnt[lab, m, b][1] for b in RATES])
                hi = np.array([cp(*cnt[lab, m, b][:2])[1] for b in RATES])
                x = np.array(RATES, float) + (j - 1.5) * 0.9
                zero = k == 0
                sp = [None if z else PR.spread(runs[lab, m, b]) for z, b in zip(zero, RATES)]
                PS.series(ax, x, np.where(zero, np.nan, k / n), sp, clip_on=False,
                          **PS.marks(LOC[lab]["color"], LOC[lab]["marker"], ms=PS.MS - 0.4, ls=LOC[lab]["ls"],
                                     label=lab.replace("\n", " ")))
                if zero.any():          # 0 errors has no place on a log axis: hollow triangle at the 95 % upper bound
                    ax.plot(x[zero], hi[zero], zorder=4, clip_on=False,
                            **PS.marks(LOC[lab]["color"], "v", hollow=True, ls="none", ms=PS.MS - 1.0))
            ax.set_xticks(RATES); ax.set_xlim(RATES[0] - 4, RATES[-1] + 4)
            ax.set_xlabel("Bit rate (bps)")
            PS.note(ax, name, x=0.05, y=0.97)
            PS.panel(ax, letter, dx=-27 if letter == "a" else -6)
            PS.finish(ax)
        axes[0].set_yscale("log"); axes[0].set_ylim(1e-4, 1)
        axes[0].set_ylabel("Bit error rate after FEC")
        h, l = axes[0].get_legend_handles_labels()
        PS.top_legend(fig, h, l, ncol=2, handlelength=2.6)
        PS.save(fig, os.path.join(a.out, "fig_tablet_fec_states_rates"))
    for lab, _ in FL.SITES:
        print(lab.replace("\n", " ").ljust(17), "  ".join(f"{m.upper()} " + " ".join(f"{cnt[lab, m, b][0]:>2}/{cnt[lab, m, b][1]}" for b in RATES) for m in MODS))
    print("->", os.path.join(a.out, "fig_tablet_fec_states_rates.pdf"))


if __name__ == "__main__":
    main()
