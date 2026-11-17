#!/usr/bin/env python3
"""
Paper figure: tablet (maXTouch) BER vs bit rate, benchmarks (direct contact, agar phantom) next to through body,
one column at print size (3.5 in). All BERs before error correction (uncoded 127-bit PRBS7 frames, frozen decode_v4,
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
Colour = medium, line style / fill = modulation (OOK solid filled, FSK dashed hollow), OOK and FSK 1.2 bps apart.
Only 20-60 bps (the through-body rates) are drawn. Linear BER axis (%): most direct-contact points have no error,
and 0 has no place on a log axis. No error bars.

   python paper_fig_tablet_bench.py

Output: data_maxtouch/paper_fig_tablet_ber/ (fig_tablet_bench.{pdf,png,svg}, fig_tablet_bench.csv)
   \\includegraphics[width=\\columnwidth]{fig_tablet_bench.pdf}   (3.5 in wide, fonts 7-8 pt at 100 %)
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paper_fig_tablet_ber import HERE, RC, cp
from paper_fig_tablet_ber_snr import FS, FS_TICK, INK

D = os.path.join(HERE, "data_maxtouch")
MEDIA = {"direct": dict(color="#1a1a1a", label="Direct contact"),
         "agar": dict(color="#D55E00", label="Agar phantom"),
         "body": dict(color="#1f5fbf", label="Through body")}
MOD = {"ook": dict(ls="-", filled=True, label="OOK", dx=-0.6),          # dx: OOK and FSK drawn 1.2 bps apart -
       "fsk": dict(ls=(0, (3, 1.5)), filled=False, label="FSK", dx=0.6)}  # equal at several rates
MARK = {"direct": "D", "agar": "s", "body": "o"}


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
    with open(os.path.join(a.out, "fig_tablet_bench.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["medium", "modulation", "bps", "errors", "bits", "ber", "ci95_lo", "ci95_hi"])
        for med, d in data.items():
            for (m, b), (k, n) in sorted(d.items()):
                w.writerow([med, m, b, k, n, f"{k / n:.3e}", *(f"{x:.3e}" for x in cp(k, n))])

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(3.5, 2.5), layout="constrained")
        fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02)
        for med, d in data.items():
            c = MEDIA[med]["color"]
            for m, st in MOD.items():
                pts = sorted((b, k, n) for (mm, b), (k, n) in d.items() if mm == m)
                if not pts:
                    continue
                x = np.array([p[0] for p in pts], float) + st["dx"]
                k = np.array([p[1] for p in pts]); n = np.array([p[2] for p in pts])
                y = 100 * k / n
                ax.plot(x, y, color=c, ls=st["ls"], marker=MARK[med], ms=3.5, mfc=c if st["filled"] else "white",
                        mec=c, mew=0.9, lw=1.1, zorder=3, clip_on=False)
        ax.set_xticks(RATES); ax.set_xlim(RATES[0] - 3, RATES[-1] + 3)
        ax.set_ylim(0, 1.4)
        ax.set_xlabel("Bit rate (bps)", fontsize=FS, color=INK, labelpad=1)
        ax.set_ylabel("Bit error rate (%)", fontsize=FS, color=INK, labelpad=1)
        ax.tick_params(labelsize=FS_TICK, colors=INK, width=0.6, length=2.5, pad=1.5)
        ax.tick_params(which="minor", width=0.5, length=1.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_linewidth(0.6)
        g = "#52514e"
        media = [plt.Line2D([], [], color=MEDIA[med]["color"], marker=MARK[med], ms=3.5, lw=1.1,
                            label=MEDIA[med]["label"]) for med in MEDIA]
        styles = [plt.Line2D([], [], color=g, ls=MOD["ook"]["ls"], marker="o", ms=3.5, lw=1.1, label="OOK"),
                  plt.Line2D([], [], color=g, ls=MOD["fsk"]["ls"], marker="o", ms=3.5, mfc="white", mew=0.9, lw=1.1,
                             label="FSK")]
        key = [media[0], styles[0], media[1], styles[1], media[2]]   # fills by column: row 1 media, row 2 OOK / FSK
        fig.legend(handles=key, loc="outside upper center", ncol=3, frameon=False, prop={"size": FS_TICK - 1},
                   handlelength=1.8, handletextpad=0.3, columnspacing=0.8, borderaxespad=0.1)
        for ext in ("pdf", "svg", "png"):
            fig.savefig(os.path.join(a.out, f"fig_tablet_bench.{ext}"), dpi=600 if ext == "png" else None)
        plt.close(fig)
    for med, d in data.items():
        print(f"{med:6s} " + "  ".join(f"{m} {b}: {k}/{n}" for (m, b), (k, n) in sorted(d.items())))
    print("->", os.path.join(a.out, "fig_tablet_bench.pdf"))


if __name__ == "__main__":
    main()
