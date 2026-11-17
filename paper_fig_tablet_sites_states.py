#!/usr/bin/env python3
"""
Paper figure: tablet (maXTouch) BER at one bit rate (default 60 bps), OOK and FSK, one column with two panels side by
side (3.5 x 2.3 in, the layout of fig_tablet_ber_snr / the paper's Figure 13):
  a  BER (%) per TX electrode location, sitting still - fig_tablet_sites_<bps>bps (paper_fig_tablet_sites.py)
  b  BER (log) per body state, ankle to finger        - fig_tablet_states_line_<bps>bps (paper_fig_tablet_states.py)
Same data and counting as those two scripts (their loaders are called unchanged; nothing re-decoded), all without
error correction.

   python paper_fig_tablet_sites_states.py [--bps 60]

Output: data_maxtouch/paper_fig_tablet_ber/ (fig_tablet_sites_states_<bps>bps.{pdf,png,svg})
   \\includegraphics[width=\\columnwidth]{fig_tablet_sites_states_60bps.pdf}   (3.5 in wide, fonts 6-8 pt at 100 %)
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import paper_fig_tablet_sites as SI
import paper_fig_tablet_states as ST
from paper_fig_tablet_ber import HERE, RC, STYLE
from paper_fig_tablet_ber_snr import FS, FS_TICK, INK, tidy

MODS = ("ook", "fsk")


def style_axes(ax, letter, ylabel):
    ax.set_ylabel(ylabel, fontsize=FS, color=INK, labelpad=1)
    ax.text(-0.30, 1.04, letter, transform=ax.transAxes, fontsize=FS + 2, fontweight="bold", color=INK,
            ha="left", va="bottom")
    ax.tick_params(labelsize=FS_TICK - 1, colors=INK, width=0.6, length=2.5, pad=1.5)
    ax.tick_params(which="minor", width=0.5, length=1.5)
    ax.tick_params(axis="x", length=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bps", type=int, default=60)
    ap.add_argument("--out", default=os.path.join(HERE, "data_maxtouch", "paper_fig_tablet_ber"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sites = {(lab, m): SI.counts(src, m, a.bps) for lab, src in SI.SITES for m in MODS}
    states = ST.load(a.bps)

    with plt.rc_context(RC):
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(3.5, 2.3), layout="constrained")
        fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.08)
        # a: bars per location, linear %
        x = np.arange(len(SI.SITES)); wd = 0.36; top = 0.0
        for j, m in enumerate(MODS):
            pct = np.array([100 * sites[lab, m][0] / sites[lab, m][1] for lab, _ in SI.SITES])
            top = max(top, pct.max())
            xb = x + (j - 0.5) * (wd + 0.04)
            ax_a.bar(xb, pct, width=wd, color=STYLE[m]["color"], label=STYLE[m]["label"], zorder=3)
            for xi, v in zip(xb, pct):              # a zero-height bar would read as missing data: mark it
                if v == 0:
                    ax_a.text(xi, 0, "0", ha="center", va="bottom", fontsize=FS_TICK - 1, color=STYLE[m]["color"])
        ax_a.set_xticks(x, [lab for lab, _ in SI.SITES])
        ax_a.set_ylim(0, top * 1.15)
        ax_a.set_xlabel("TX electrode location", fontsize=FS, color=INK, labelpad=1)
        style_axes(ax_a, "a", "Bit error rate (%)")
        # b: lines per body state, log
        xs = np.arange(len(ST.STATES))
        for m in MODS:
            st = STYLE[m]
            ax_b.plot(xs, [states[p, m][0] / states[p, m][1] for _, p in ST.STATES], color=st["color"],
                      marker=st["marker"], ms=3.5, lw=1.1, zorder=3)
        ax_b.set_yscale("log"); ax_b.set_ylim(1e-4, 1e-1)
        ax_b.set_xticks(xs, [lab for lab, _ in ST.STATES]); ax_b.set_xlim(-0.4, len(ST.STATES) - 0.6)
        ax_b.set_xlabel("Body state", fontsize=FS, color=INK, labelpad=1)
        style_axes(ax_b, "b", "Bit error rate")
        fig.legend(*ax_a.get_legend_handles_labels(), loc="outside upper center", ncol=2, frameon=False,
                   prop={"size": FS_TICK}, handlelength=1.4, handletextpad=0.4, columnspacing=1.6, borderaxespad=0.1)
        name = f"fig_tablet_sites_states_{a.bps}bps"
        for ext in ("pdf", "svg", "png"):
            fig.savefig(os.path.join(a.out, f"{name}.{ext}"), dpi=600 if ext == "png" else None)
        plt.close(fig)
    print("->", os.path.join(a.out, f"{name}.pdf"))


if __name__ == "__main__":
    main()
