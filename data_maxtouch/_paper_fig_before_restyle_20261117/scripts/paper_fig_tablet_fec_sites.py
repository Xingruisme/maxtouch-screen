#!/usr/bin/env python3
"""
Paper figure: tablet (maXTouch) BER after error correction (FEC) per TX electrode location, 40 and 60 bps, OOK and
FSK, one column at print size (3.5 in), in the style of fig_tablet_states_line_40_60bps.

Data = the coded sessions of 2026-11-16 recorded with stent_trx_ber.py (the stent link's frames: 96 info bits = 80
data + CRC-16, convolutional code K = 4, rate 1/2, soft decision - 'cc4'), 1 mA, still, finger-on calibration,
OOK carrier 159.6 kHz, FSK tones 159.6 / 180.6 kHz; summaries made by `stent_trx_ber.py analyse` (nothing re-decoded):
  Direct contact   data_maxtouch/trxber_electrode_direct_1mA_20261116_140938   12 frames (1152 info bits) per point
  Shoulder         data_maxtouch/trxber_shoulder2finger_1mA_20261116_150313   12 frames (1152 info bits)
  Knee             data_maxtouch/trxber_knee2finger_1mA_20261116_152934       8 frames (768 info bits)
After FEC = info-bit BER after Viterbi decoding. x = bit rate on air; the info rate is 0.375 x that.
Linear BER axis: most points have no error after FEC, and a log axis cannot show 0. Error bar = lowest to highest
BER of the recording chunks behind each point (paper_runs.py: direct contact and shoulder 2 chunks, knee 1 - no bar).

   python paper_fig_tablet_fec_sites.py

Output: data_maxtouch/paper_fig_tablet_ber/ (fig_tablet_fec_sites.{pdf,png,svg}, fig_tablet_fec_sites.csv)
   \\includegraphics[width=\\columnwidth]{fig_tablet_fec_sites.pdf}   (3.5 in wide, fonts ~9 pt at 100 %)
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

from paper_fig_tablet_ber import HERE, RC, STYLE, cp
from paper_fig_tablet_snr import FS, FS_TICK, INK

D = os.path.join(HERE, "data_maxtouch")
SITES = [("Direct\ncontact", "trxber_electrode_direct_1mA_20261116_140938"),
         ("Shoulder", "trxber_shoulder2finger_1mA_20261116_150313"),
         ("Knee", "trxber_knee2finger_1mA_20261116_152934")]
RATES = (40, 60)
RATE_LS = {40: "-", 60: (0, (3, 1.5))}
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
    with open(os.path.join(a.out, "fig_tablet_fec_sites.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["location", "modulation", "bit_rate_on_air_bps", "frames", "post_fec_errors", "post_fec_bits",
                    "post_fec_ber", "ci95_hi"])
        for lab, folder in SITES:
            for m in ("ook", "fsk"):
                for b in RATES:
                    k, n, fr = data[lab][m, b]
                    w.writerow([lab.replace("\n", " "), m, b, fr, k, n, f"{k / n:.3e}", f"{cp(k, n)[1]:.3e}"])

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(3.5, 2.5), layout="constrained")
        fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02)
        x = np.arange(len(SITES))
        for i, b in enumerate(RATES):
            for m in ("ook", "fsk"):
                st = STYLE[m]
                pct = [100 * data[lab][m, b][0] / data[lab][m, b][1] for lab, _ in SITES]
                dx = (-0.03 if m == "ook" else 0.03) + (-0.01 if i == 0 else 0.01)   # zeros coincide: keep all visible
                sp = [PR.spread(PR.trx(f, CODE, m, b, after_fec=True), 100) for _, f in SITES]
                ax.errorbar(x + dx, pct, yerr=PR.yerr(pct, sp), color=st["color"], marker=st["marker"], ms=4, lw=1.2,
                            ls=RATE_LS[b], mfc=st["color"] if i == 0 else "white", mew=0.9,
                            label=f"{st['label']}, {b} bps", zorder=3, clip_on=False, **PR.EB)
        ax.set_xticks(x, [lab for lab, _ in SITES]); ax.set_xlim(-0.35, len(SITES) - 0.65)
        ax.set_ylim(0, 2)
        ax.set_xlabel("TX electrode location", fontsize=FS, color=INK, labelpad=2)
        ax.set_ylabel("Bit error rate after FEC (%)", fontsize=FS, color=INK, labelpad=2)
        ax.tick_params(labelsize=FS_TICK, colors=INK, width=0.6, length=3, pad=2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_linewidth(0.6)
        h, l = ax.get_legend_handles_labels()
        fig.legend(h, l, loc="outside upper center", ncol=2, frameon=False, prop={"size": FS_TICK},
                   handlelength=2.2, handletextpad=0.4, columnspacing=1.2, borderaxespad=0.1)
        for ext in ("pdf", "svg", "png"):
            fig.savefig(os.path.join(a.out, f"fig_tablet_fec_sites.{ext}"), dpi=600 if ext == "png" else None)
        plt.close(fig)
    for lab, _ in SITES:
        print(lab.replace("\n", " ").ljust(15), "  ".join(f"{m.upper()} {b}: {data[lab][m, b][0]}/{data[lab][m, b][1]}"
                                                          for b in RATES for m in ("ook", "fsk")))
    print("->", os.path.join(a.out, "fig_tablet_fec_sites.pdf"))


if __name__ == "__main__":
    main()
