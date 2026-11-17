#!/usr/bin/env python3
"""
Paper figure: tablet (maXTouch) BER vs bit rate, benchmarks (direct contact, agar phantom) next to through body,
one column at print size (3.5 x 2.2 in). All BERs before error correction (uncoded 127-bit PRBS7 frames, frozen decode_v4,
known timing); nothing re-decoded.
  direct contact  data_maxtouch/trxber_electrode_direct_1mA_20261116_140938/summary.csv (2026-11-16 14:09-14:37,
                  coded session of stent_trx_ber.py, electrode directly on the panel): its uncoded frames only
                  ('none_full': 96 bits at one bit per symbol, no FEC), 12 frames = 1152 bits per point, OOK + FSK,
                  159.6 / 180.6 kHz, 20-60 bps (decode_trx_tablet.py). The Oct 22 direct-contact runs (155.7 kHz,
                  OOK only) are no longer used.
  agar phantom    data_maxtouch/modcomp_mc1_1mA/summary.json  (2026-10-23 16:35-17:26, the session after the agar
                  phantom sweeps; before the 17:30 transmitter fault): 159.6 kHz, OOK + FSK (fsk 0 = 180.6 kHz),
                  10-100 bps, 5080 bits per point. DBPSK of that session is left out: ~50 % BER at every rate (the
                  decoder never locked). The 2.5 / 5 mA mc1 sessions fall in the fault window and are not used.
  Setups as identified by the user; the run metadata does not record the placement. Not used: the two agar_5cm runs
  of 2026-10-23 15:53 / 16:00 (80 bps decoded no frame; 20 bps a single run, 17 / 3048).
  through body    'static' rows of modcomp_conditions_a2f_static_vs_moving_20to60 (sitting + standing still, ankle
                  to finger, 1 mA), 20-60 bps, 6096 bits per point (as fig_tablet_ber_column)
Two panels on one BER axis: a OOK, b FSK. Colour and marker = medium (the three drawn 1 bps apart).
Only 20-60 bps (the through-body rates) are drawn. Linear BER axis (%): most direct-contact points have no error,
and 0 has no place on a log axis. Error bar = lowest to highest per-run BER (paper_runs.py): direct contact over
its 2 recording chunks, through body over its 6 runs; the agar points are single runs and have none.

   python paper_fig_tablet_bench.py

Output: data_maxtouch/paper_fig_tablet_ber/ (fig_tablet_bench.{pdf,png,svg}, fig_tablet_bench.csv)
   \\includegraphics[width=\\columnwidth]{fig_tablet_bench.pdf}   (3.5 in wide, fonts 6-8 pt at 100 %)
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

import paper_style as PS
from paper_fig_tablet_ber import HERE, RC, cp

D = os.path.join(HERE, "data_maxtouch")
MEDIA = {"direct": dict(color="#555555", label="Direct contact"),
         "agar": dict(color="#D55E00", label="Agar phantom"),
         "body": dict(color="#7b3294", label="Through body")}
MOD = {"ook": "OOK", "fsk": "FSK"}                   # one panel each
MARK = {"direct": "D", "agar": "s", "body": "o"}
MED_DX = {"direct": -1.0, "agar": 0.0, "body": 1.0}  # media drawn 1 bps apart, so their range bars do not overlap


DIRECT = "trxber_electrode_direct_1mA_20261116_140938"
RATES = (20, 40, 50, 60)


def direct():
    """uncoded frames ('none_full') of the 2026-11-16 direct-contact session"""
    return {(r["modulation"], round(float(r["baud"]))): (int(r["pre_fec_errors"]), int(r["pre_fec_bits"]))
            for r in csv.DictReader(open(os.path.join(D, DIRECT, "summary.csv")))
            if r["decoder"] == "none_full" and r["modulation"] in MOD}


def agar():
    rows = json.load(open(os.path.join(D, "modcomp_mc1_1mA", "summary.json")))["rows"]
    return {(r["modulation"], round(r["bps"])): (r["errors"], r["bits"]) for r in rows if r["modulation"] in MOD}


def per_run():
    """{medium: {(modulation, rate): [(errors, bits) per run]}} behind the pooled points"""
    src = os.path.join(D, "modcomp_conditions_a2f_static_vs_moving_20to60", "conditions.csv")
    return {"direct": {(m, b): PR.trx(DIRECT, "none_full", m, b) for m in MOD for b in RATES},
            "agar": {(m, b): PR.sessions(["modcomp_mc1_1mA"], m, b) for m in MOD for b in RATES
                     if PR.sessions(["modcomp_mc1_1mA"], m, b)},
            "body": {(r["modulation"], round(float(r["bps"]))): PR.conditions_row(r)
                     for r in csv.DictReader(open(src)) if r["condition"].startswith("static")}}


def body():
    src = os.path.join(D, "modcomp_conditions_a2f_static_vs_moving_20to60", "conditions.csv")
    return {(r["modulation"], round(float(r["bps"]))): (int(r["errors"]), int(r["bits"]))
            for r in csv.DictReader(open(src)) if r["condition"].startswith("static")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(D, "paper_fig_tablet_ber"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data = {med: {k: v for k, v in d.items() if k[1] in RATES}
            for med, d in (("direct", direct()), ("agar", agar()), ("body", body()))}
    rr = per_run()
    for med, d in data.items():                         # the per-run lists must add up to the pooled counts
        for key, v in d.items():
            assert (sum(e for e, _ in rr[med][key]), sum(n for _, n in rr[med][key])) == v, (med, key)
    with open(os.path.join(a.out, "fig_tablet_bench.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["medium", "modulation", "bps", "errors", "bits", "ber", "ci95_lo", "ci95_hi"])
        for med, d in data.items():
            for (m, b), (k, n) in sorted(d.items()):
                w.writerow([med, m, b, k, n, f"{k / n:.3e}", *(f"{x:.3e}" for x in cp(k, n))])

    with plt.rc_context(RC):
        fig, axs = PS.figure(3.5, 2.2, 1, 2, sharey=True)
        for ax, m in zip(axs, MOD):
            for med, d in data.items():
                pts = sorted((b, k, n) for (mm, b), (k, n) in d.items() if mm == m)
                x = np.array([p[0] for p in pts], float) + MED_DX[med]
                PS.series(ax, x, [100 * p[1] / p[2] for p in pts], [PR.spread(rr[med][m, p[0]], 100) for p in pts],
                          clip_on=False, **PS.marks(MEDIA[med]["color"], MARK[med], ms=PS.MS - 0.4))
            ax.set_xticks(RATES); ax.set_xlim(RATES[0] - 4, RATES[-1] + 4)
            ax.set_ylim(0, 3.0)
            ax.set_xlabel("Bit rate (bps)")
            PS.finish(ax)
            PS.note(ax, MOD[m])
        axs[0].set_ylabel("Bit error rate (%)")
        PS.panel(axs[0], "a", dx=-22); PS.panel(axs[1], "b", dx=-6)
        PS.top_legend(fig, [PS.key(MEDIA[med]["color"], MARK[med], label=MEDIA[med]["label"]) for med in MEDIA],
                      columnspacing=1.2)
        PS.save(fig, os.path.join(a.out, "fig_tablet_bench"))
    for med, d in data.items():
        print(f"{med:6s} " + "  ".join(f"{m} {b}: {k}/{n}" for (m, b), (k, n) in sorted(d.items())))
    print("->", os.path.join(a.out, "fig_tablet_bench.pdf"))


if __name__ == "__main__":
    main()
