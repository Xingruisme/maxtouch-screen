#!/usr/bin/env python3
"""
Paper figure: tablet (maXTouch) BER at one bit rate (default 60 bps), OOK and FSK, one column with two panels side by
side (3.5 x 1.9 in, the layout of fig_tablet_ber_snr / the paper's Figure 13):
  a  BER (%) per TX electrode location, sitting still - fig_tablet_sites_<bps>bps (paper_fig_tablet_sites.py)
  b  BER (log) per body state, ankle to finger        - fig_tablet_states_line_<bps>bps (paper_fig_tablet_states.py)
Same data and counting as those two scripts (their loaders are called unchanged; nothing re-decoded), all without
error correction. Error bar = lowest to highest per-run BER (paper_runs.py).

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
import paper_runs as PR
import paper_fig_tablet_states as ST
import paper_style as PS
from paper_fig_tablet_ber import HERE, RC, STYLE

MODS = ("ook", "fsk")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bps", type=int, default=60)
    ap.add_argument("--out", default=os.path.join(HERE, "data_maxtouch", "paper_fig_tablet_ber"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sites = {(lab, m): SI.counts(src, m, a.bps) for lab, src in SI.SITES for m in MODS}
    states = ST.load(a.bps)
    sites_rr = {(lab, m): SI.run_list(src, m, a.bps) for lab, src in SI.SITES for m in MODS}
    states_rr = ST.run_counts(a.bps)

    with plt.rc_context(RC):
        fig, (ax_a, ax_b) = PS.figure(3.5, 1.9, 1, 2, wspace=0.08, width_ratios=(1.0, 1))
        # a: bars per location, linear %
        x = np.arange(len(SI.SITES)); wd = 0.36; top = 0.0
        for j, m in enumerate(MODS):
            pct = [100 * sites[lab, m][0] / sites[lab, m][1] for lab, _ in SI.SITES]
            runs = [[100 * e / n for e, n in sites_rr[lab, m]] for lab, _ in SI.SITES]
            top = max(top, max(max(r) for r in runs))
            PS.bars(ax_a, x + (j - 0.5) * (wd + 0.05), pct, runs, STYLE[m]["color"], wd, STYLE[m]["label"])
        ax_a.set_xticks(x, [lab for lab, _ in SI.SITES])
        PS.pct_axis(ax_a, top)
        ax_a.set_xlabel("TX electrode location"); ax_a.set_ylabel("Bit error rate (%)")
        PS.finish(ax_a, categorical=True)
        PS.panel(ax_a, "a", dx=-22)
        # b: lines per body state, log
        xs = np.arange(len(ST.STATES))
        for m in MODS:
            st = STYLE[m]
            y = [states[p, m][0] / states[p, m][1] for _, p in ST.STATES]
            PS.series(ax_b, xs + (-0.06 if m == "ook" else 0.06), y, [PR.spread(states_rr[p, m]) for _, p in ST.STATES],
                      **PS.marks(st["color"], st["marker"], ms=PS.MS - 0.4))
        ax_b.set_yscale("log"); ax_b.set_ylim(1e-4, 1e-1)
        ax_b.set_xlabel("Body state"); ax_b.set_ylabel("Bit error rate")
        ST.state_axis(ax_b)
        PS.panel(ax_b, "b")
        for ax in (ax_a, ax_b):                     # eight category names across one column
            ax.tick_params(axis="x", labelsize=PS.FS_TICK - 0.5)
        PS.top_legend(fig, [PS.key(STYLE[m]["color"], STYLE[m]["marker"], label=STYLE[m]["label"]) for m in MODS])
        name = f"fig_tablet_sites_states_{a.bps}bps"
        PS.save(fig, os.path.join(a.out, name))
    print("->", os.path.join(a.out, f"{name}.pdf"))


if __name__ == "__main__":
    main()
