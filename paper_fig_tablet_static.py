#!/usr/bin/env python3
"""
Paper figures: tablet (maXTouch) through-body link vs bit rate, sitting still vs standing still, OOK and FSK, drawn at
print size for one column (the layout of the slide 'Comparison of gestures of static condition').

Data = the standing-still vs sitting-still table made by `stent_modcomp.py conditions`
(data_maxtouch/modcomp_conditions_a2f_stand_vs_sit_still_20to60/conditions.csv): per posture the latest 3 runs per
rate and modulation (20/50 bps 2026-10-29, 40/60 bps 2026-10-30), ankle to finger, 1 mA, 159.6 kHz carrier, 3048 bits
per point; frozen decode_v4, known timing. Nothing is re-decoded.
  SNR figure: symbol SNR per recording from the sessions' summary.json, marker = median of the 3 recordings, error
              bar = lowest to highest recording.
  BER figure: pooled errors / bits of the 3 recordings, error bar = lowest to highest per-run BER (paper_runs.py),
              down to the axis where a run had no error (the csv keeps the Clopper-Pearson 95 % interval).
Colour = posture, line style and marker = modulation (OOK solid, filled circle; FSK dashed, hollow square); the four
series are drawn 0.6 bps apart so none hides another.
A point with 0 errors has no BER on a log axis: it is drawn as a hollow downward triangle at its 95 % upper bound
(about 3/3048), not joined to the line.

   python paper_fig_tablet_static.py

Output: data_maxtouch/paper_fig_tablet_ber/ (fig_tablet_snr_static.*, fig_tablet_ber_static.*, fig_tablet_static.csv)
   \\includegraphics[width=\\columnwidth]{fig_tablet_ber_static.pdf}   (3.5 in wide, fonts 6-7 pt at 100 %)
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
from paper_fig_tablet_ber import HERE, RC, cp, per_run_snr

SRC = os.path.join(HERE, "data_maxtouch", "modcomp_conditions_a2f_stand_vs_sit_still_20to60", "conditions.csv")
# colour = posture here (hues that are not the OOK / FSK colours of the other figures); the modulation is told by
# line style and marker: OOK solid, filled circle - FSK dashed, hollow square
POSTURE = {"sitting": dict(prefix="sitting still", color="#7b3294", label="sitting"),
           "standing": dict(prefix="standing still", color="#D55E00", label="standing")}
MOD = {"ook": dict(ls="-", marker="o", hollow=False, label="OOK"),
       "fsk": dict(ls=PS.DASH, marker="s", hollow=True, label="FSK")}
SERIES = [(p, m) for p in POSTURE for m in MOD]
OFF = dict(zip(SERIES, (-0.9, -0.3, 0.3, 0.9)))


def style(p, m):
    return PS.marks(POSTURE[p]["color"], MOD[m]["marker"], MOD[m]["hollow"], ls=MOD[m]["ls"],
                    label=f"{MOD[m]['label']}, {POSTURE[p]['label']}")


def figure(out, draw, ylabel):
    with plt.rc_context(RC):
        fig, ax = PS.figure(3.5, 2.6)
        rates = draw(ax)
        ax.set_xticks(rates); ax.set_xlim(rates[0] - 3, rates[-1] + 3)
        ax.set_xlabel("Bit rate (bps)"); ax.set_ylabel(ylabel)
        PS.finish(ax)
        h, l = ax.get_legend_handles_labels()
        PS.top_legend(fig, h, l, ncol=2, handlelength=2.6)
        PS.save(fig, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=os.path.join(HERE, "data_maxtouch", "paper_fig_tablet_ber"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = list(csv.DictReader(open(a.src)))
    ber, snr, runs = {}, {}, {}
    for p, st in POSTURE.items():
        mine = [r for r in rows if r["condition"].startswith(st["prefix"])]
        assert mine, f"no '{st['prefix']}' rows in {a.src}"
        for r in mine:
            ber[p, r["modulation"], round(float(r["bps"]))] = (int(r["errors"]), int(r["bits"]))
            runs[p, r["modulation"], round(float(r["bps"]))] = PR.conditions_row(r)
        for (m, b), v in per_run_snr(mine).items():
            snr[p, m, b] = v
    rates = sorted({b for _, _, b in ber})
    with open(os.path.join(a.out, "fig_tablet_static.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["posture", "modulation", "bps", "errors", "bits", "ber", "ci95_lo", "ci95_hi", "snr_db_median",
                    "snr_db_values"])
        for (p, m, b), (k, n) in sorted(ber.items()):
            w.writerow([p, m, b, k, n, f"{k / n:.3e}", *(f"{x:.3e}" for x in cp(k, n)),
                        f"{np.median(snr[p, m, b]):.2f}", " ".join(f"{x:.2f}" for x in snr[p, m, b])])

    def draw_snr(ax):
        for p, m in SERIES:
            PS.series(ax, np.array(rates, float) + OFF[p, m], [np.median(snr[p, m, b]) for b in rates],
                      [(min(snr[p, m, b]), max(snr[p, m, b])) for b in rates], **style(p, m))
        ax.set_ylim(30, 70); ax.set_yticks(range(30, 71, 10))
        return rates

    def draw_ber(ax):
        for p, m in SERIES:
            k = np.array([ber[p, m, b][0] for b in rates]); n = np.array([ber[p, m, b][1] for b in rates])
            hi = np.array([cp(*ber[p, m, b])[1] for b in rates])
            x = np.array(rates, float) + OFF[p, m]
            # 0 errors cannot sit on a log axis: no point and no line there, a downward marker at the 95 % upper bound
            zero = k == 0
            PS.series(ax, x, np.where(zero, np.nan, k / n),
                      [None if z else PR.spread(runs[p, m, b]) for z, b in zip(zero, rates)], **style(p, m))
            if zero.any():
                ax.plot(x[zero], hi[zero], zorder=4, **PS.marks(POSTURE[p]["color"], "v", hollow=True, ls="none"))
        # key for the 0-error marker inside the plot; the four series go in the figure legend
        if any(v[0] == 0 for v in ber.values()):
            ax.add_artist(ax.legend([PS.key(marker="v", hollow=True, ls="none")], ["0 errors (95% upper bound)"],
                                    loc="upper left", handletextpad=0.2))
        ax.set_yscale("log"); ax.set_ylim(1e-4, 1e-1)
        return rates

    figure(os.path.join(a.out, "fig_tablet_snr_static"), draw_snr, "Symbol SNR (dB)")
    figure(os.path.join(a.out, "fig_tablet_ber_static"), draw_ber, "Bit error rate")
    for (p, m, b), (k, n) in sorted(ber.items()):
        print(f"{p:8s} {m} {b} bps: {k}/{n} = {k / n:.2%}   SNR median {np.median(snr[p, m, b]):.1f} dB "
              f"({' '.join(f'{x:.1f}' for x in snr[p, m, b])})")
    print("->", a.out)


if __name__ == "__main__":
    main()
