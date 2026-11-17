#!/usr/bin/env python3
"""
Paper figure: tablet (maXTouch) through-body BER per body state at one bit rate (default 40 bps), OOK and FSK, drawn
at print size for one column, in the style of fig_tablet_ber_column (paper_fig_tablet_ber_column.py).

Data = the two condition tables made by `stent_modcomp.py conditions` (nothing re-decoded):
  data_maxtouch/modcomp_conditions_a2f_stand_vs_sit_still_20to60/conditions.csv        sitting still, standing still
  data_maxtouch/modcomp_conditions_a2f_standwalking_vs_sitshaking_20to60/conditions.csv sitting shaking, standing walking
At 40 bps each state is its own latest 3 runs (sitting / standing still 2026-10-30, sitting shaking / standing walking
2026-11-01), ankle to finger, 1 mA, 159.6 kHz carrier, 3048 bits per bar; frozen decode_v4, known timing.
Bar = pooled errors / bits of the 3 runs, value printed above it; error bar = lowest to highest per-run BER
(paper_runs.py). The csv keeps the Clopper-Pearson 95 % interval of the pooled count.

   python paper_fig_tablet_states.py [--bps 40]

Also the same counts as a line chart: BER (log) per state, one line per modulation (fig_tablet_states_line_*),
and 40 and 60 bps together in one line chart (fig_tablet_states_line_40_60bps: solid filled 40, dashed hollow 60).

Output: data_maxtouch/paper_fig_tablet_ber/ (fig_tablet_states_40bps.{pdf,png,svg}, fig_tablet_states_40bps.csv,
        fig_tablet_states_line_40bps.{pdf,png,svg})
   \\includegraphics[width=\\columnwidth]{fig_tablet_states_line_40bps.pdf}   (3.5 in wide, fonts 6-7 pt at 100 %)
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

D = os.path.join(HERE, "data_maxtouch")
SRCS = [os.path.join(D, "modcomp_conditions_a2f_stand_vs_sit_still_20to60", "conditions.csv"),
        os.path.join(D, "modcomp_conditions_a2f_standwalking_vs_sitshaking_20to60", "conditions.csv")]
# static first, then moving; (label on the axis, condition prefix in the tables)
STATES = [("Sitting\nstill", "sitting still"), ("Standing\nstill", "standing still"),
          ("Sitting\nshaking", "sitting shaking"), ("Standing\nwalking", "standing walking")]


def load(bps):
    """{(condition prefix, modulation): (errors, bits, runs)} at one bit rate"""
    rows = [r for f in SRCS for r in csv.DictReader(open(f)) if round(float(r["bps"])) == bps]
    data = {}
    for lab, prefix in STATES:
        for m in ("ook", "fsk"):
            hit = [r for r in rows if r["condition"].startswith(prefix) and r["modulation"] == m]
            assert len(hit) == 1, (prefix, m, bps, len(hit))
            data[prefix, m] = (int(hit[0]["errors"]), int(hit[0]["bits"]), len(hit[0]["run"].split("+")))
    return data


def run_counts(bps):
    """{(condition prefix, modulation): [(errors, bits) per run]} at one bit rate, for the error bars"""
    rows = [r for f in SRCS for r in csv.DictReader(open(f)) if round(float(r["bps"])) == bps]
    return {(prefix, m): PR.conditions_row(next(r for r in rows if r["condition"].startswith(prefix)
                                                 and r["modulation"] == m))
            for _, prefix in STATES for m in ("ook", "fsk")}


RATE_LS = {40: "-", 60: PS.DASH}               # line style per bit rate in the two-rate chart


def state_axis(ax):
    """the four body states along x; the two with motion stand on a faint band"""
    ax.set_xticks(np.arange(len(STATES)), [lab for lab, _ in STATES]); ax.set_xlim(-0.4, len(STATES) - 0.6)
    PS.band(ax, 1.5, len(STATES) - 0.6, "in motion")
    PS.finish(ax, categorical=True)


def line_chart_rates(rates, out):
    """BER (log) per body state at several bit rates: colour = modulation, line style / fill = bit rate"""
    with plt.rc_context(RC):
        fig, ax = PS.figure(3.5, 2.5)
        x = np.arange(len(STATES))
        for i, bps in enumerate(rates):
            data, rr = load(bps), run_counts(bps)
            for m in ("ook", "fsk"):
                st = STYLE[m]
                ber = [data[p, m][0] / data[p, m][1] for _, p in STATES]
                dx = (-0.05 if m == "ook" else 0.05) + (-0.02 if i == 0 else 0.02)   # four series, bars apart
                PS.series(ax, x + dx, ber, [PR.spread(rr[p, m]) for _, p in STATES],
                          **PS.marks(st["color"], st["marker"], hollow=i > 0, ls=RATE_LS.get(bps, "-"),
                                     label=f"{st['label']}, {bps} bps"))
        ax.set_yscale("log"); ax.set_ylim(1e-4, 1e-1)
        ax.set_ylabel("Bit error rate")
        state_axis(ax)
        h, l = ax.get_legend_handles_labels()
        order = [i for m in ("OOK", "FSK") for i, lab in enumerate(l) if lab.startswith(m)]   # rows: OOK, FSK
        PS.top_legend(fig, [h[i] for i in order], [l[i] for i in order], ncol=len(rates), handlelength=2.6)
        PS.save(fig, out)


def line_chart(data, bps, out, rr):
    """the same counts as a line chart: BER (log) per body state, one line per modulation"""
    with plt.rc_context(RC):
        fig, ax = PS.figure(3.5, 2.4)
        x = np.arange(len(STATES))
        for m in ("ook", "fsk"):
            st = STYLE[m]
            ber = [data[p, m][0] / data[p, m][1] for _, p in STATES]
            PS.series(ax, x + (-0.04 if m == "ook" else 0.04), ber, [PR.spread(rr[p, m]) for _, p in STATES],
                      **PS.marks(st["color"], st["marker"], label=st["label"]))
        ax.set_yscale("log"); ax.set_ylim(1e-4, 1e-1)
        ax.set_ylabel(f"Bit error rate, {bps} bps")
        state_axis(ax)
        ax.legend(loc="upper left")
        PS.save(fig, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bps", type=int, default=40)
    ap.add_argument("--out", default=os.path.join(D, "paper_fig_tablet_ber"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data, rr = load(a.bps), run_counts(a.bps)
    name = f"fig_tablet_states_{a.bps}bps"
    with open(os.path.join(a.out, name + ".csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["state", "modulation", "bps", "runs", "errors", "bits", "ber", "ci95_lo", "ci95_hi"])
        for _, prefix in STATES:
            for m in ("ook", "fsk"):
                k, n, runs = data[prefix, m]
                w.writerow([prefix, m, a.bps, runs, k, n, f"{k / n:.3e}", *(f"{x:.3e}" for x in cp(k, n))])

    with plt.rc_context(RC):
        fig, ax = PS.figure(3.5, 2.4)
        x = np.arange(len(STATES)); wd = 0.34; top = 0.0
        for j, m in enumerate(("ook", "fsk")):
            st = STYLE[m]
            pct = [100 * data[p, m][0] / data[p, m][1] for _, p in STATES]
            runs = [[100 * e / n for e, n in rr[p, m]] for _, p in STATES]
            top = max(top, max(max(r) for r in runs))
            PS.bars(ax, x + (j - 0.5) * (wd + 0.05), pct, runs, st["color"], wd, st["label"])
        ax.set_xticks(x, [lab for lab, _ in STATES])
        PS.pct_axis(ax, top)
        ax.set_ylabel(f"Bit error rate (%), {a.bps} bps")
        PS.finish(ax, categorical=True)
        ax.legend(loc="upper left", handlelength=1.0)
        PS.save(fig, os.path.join(a.out, name))
    line_chart(data, a.bps, os.path.join(a.out, f"fig_tablet_states_line_{a.bps}bps"), rr)
    line_chart_rates([40, 60], os.path.join(a.out, "fig_tablet_states_line_40_60bps"))
    for _, prefix in STATES:
        print(f"{prefix:16s} " + "  ".join(f"{m.upper()} {data[prefix, m][0]}/{data[prefix, m][1]} = "
                                          f"{data[prefix, m][0] / data[prefix, m][1]:.2%} ({data[prefix, m][2]} runs)"
                                          for m in ("ook", "fsk")))
    print("->", a.out)


if __name__ == "__main__":
    main()
