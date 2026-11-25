#!/usr/bin/env python3
"""
Paper figure: tablet (maXTouch) link vs bit rate for the TX electrode locations measured on 2026-11-20 (ankle, gut,
heart, shoulder, knee, centre back waist, bottom side waist; all to the finger, sitting still, ground line removed,
1 mA), OOK and FSK: BER after error correction (a, c) and symbol SNR (b, d). Two-column figure at print size (7.16 in).

Data = the coded sessions `trxber_<location>2finger_sitting_still_nognd_1mA_20261120_*` made with stent_trx_ber.py
(K = 4 rate-1/2 convolutional code, soft decision, 'cc4'); 4 frames = 384 info bits per point and session; summaries
from `stent_trx_ber.py analyse` (nothing re-decoded). Heart was recorded three times and shoulder twice.
  BER   after FEC: errors / bits pooled over a location's sessions; error bar = lowest to highest per-session BER
        (locations with one session: no bar). A point with 0 errors is a hollow downward triangle at its 95 % upper
        bound (3/n), not joined to the line (--log only; the default is a linear % axis).
  SNR   symbol SNR of the decision statistic the decoder uses (decode_trx_tablet.sym_values at the saved timing, known
        sent states): 20 log10(|mean ON - mean OFF| / pooled std within the states), per recording. Marker = median of a
        location's sessions, error bar = lowest to highest session. NOT the Es/N0 of the earlier uncoded figures
        (that one needs the uncoded frame layout), so the dB values are not comparable with those figures.

   python paper_fig_tablet_locations_trx.py [--log]

Output: data_maxtouch/paper_fig_tablet_ber/ (fig_tablet_locations_trx.{pdf,png,svg}, fig_tablet_locations_trx.csv)
   \\includegraphics[width=\\textwidth]{fig_tablet_locations_trx.pdf}   (7.16 in wide, fonts 7-8 pt at 100 %)
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

import decode_trx_tablet as DT
import paper_style as PS
from paper_fig_tablet_ber import HERE, RC, cp

D = os.path.join(HERE, "data_maxtouch")
T0 = "20261120"
LOC = [("Ankle", ["141211"], "#0072B2", "o"), ("Gut", ["142416"], "#E69F00", "s"),
       ("Heart", ["144131", "153536", "154512"], "#D55E00", "^"), ("Shoulder", ["145142", "155853"], "#009E73", "D"),
       ("Knee", ["152459"], "#CC79A7", "v"), ("Centre back waist", ["161051"], "#56B4E9", "P"),
       ("Bottom side waist", ["162423"], "#444444", "X")]
FOLDER = {"Ankle": "ankle", "Gut": "gut", "Heart": "heart", "Shoulder": "shoulder", "Knee": "knee",
          "Centre back waist": "centerbackwaist", "Bottom side waist": "bottomsidewaist"}
RATES = (20, 30, 40, 50, 60)
MODS = (("ook", "OOK"), ("fsk", "FSK"))
CODE = "cc4"


def folder(loc, stamp):
    return f"trxber_{FOLDER[loc]}2finger_sitting_still_nognd_1mA_{T0}_{stamp}"


def ber_counts(fd):
    """{(mod, rate): (errors, bits)} after FEC, cc4 frames of one trxber summary"""
    out = {}
    for r in csv.DictReader(open(os.path.join(D, fd, "summary.csv"))):
        if r["decoder"] == CODE:
            out[r["modulation"], round(float(r["baud"]))] = (int(r["post_fec_errors"]), int(r["post_fec_bits"]))
    return out


def run_snr(run):
    """symbol SNR (dB) of one recording at its saved timing"""
    d = os.path.join(D, run)
    meta = json.load(open(os.path.join(d, "meta.json"))); res = json.load(open(os.path.join(d, "result_trx.json")))
    z = np.load(os.path.join(d, "capture.npz")); t, X = z["t"].astype(float), z["X"].astype(float)
    sq = np.load(os.path.join(d, "tx_seq.npz"))
    states = np.unpackbits(sq["phase"])[:int(sq["n_phase"])].astype(np.uint8)
    Ts = meta["plan"]["sym_us"] / 1e6
    _, y, _ = DT.front(meta, t, X)
    v = DT.sym_values(t, y, meta["t_first"] + res["delay_ms"] / 1e3, Ts, res["clock_ppm"] * 1e-6, len(states))
    ok = ~np.isnan(v)
    hi, lo = v[ok & (states == 1)], v[ok & (states == 0)]
    sd = np.sqrt(0.5 * (hi.var() + lo.var()))
    return float(20 * np.log10(abs(hi.mean() - lo.mean()) / max(sd, 1e-12)))


def runs_of(fd):
    return {(r["modulation"], round(r["baud"])): r["run"]
            for r in json.load(open(os.path.join(D, fd, "summary.json")))["runs"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", action="store_true", help="log BER axis (default: linear, in percent)")
    ap.add_argument("--out", default=os.path.join(D, "paper_fig_tablet_ber"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data = {}                       # (loc, mod, rate) -> dict(errs=[..], bits=[..], snr=[..]) one entry per session
    for loc, stamps, _, _ in LOC:
        for stamp in stamps:
            fd = folder(loc, stamp)
            counts, runs = ber_counts(fd), runs_of(fd)
            for (m, b), (k, n) in counts.items():
                e = data.setdefault((loc, m, b), dict(errs=[], bits=[], snr=[]))
                e["errs"].append(k); e["bits"].append(n); e["snr"].append(run_snr(runs[m, b]))
    with open(os.path.join(a.out, "fig_tablet_locations_trx.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["location", "modulation", "bit_rate_on_air_bps", "sessions", "errors", "bits", "ber_after_fec",
                    "ci95_hi", "snr_db_median", "snr_db_per_session"])
        for loc, _, _, _ in LOC:
            for m, _ in MODS:
                for b in RATES:
                    e = data[loc, m, b]; k, n = sum(e["errs"]), sum(e["bits"])
                    w.writerow([loc, m, b, len(e["errs"]), k, n, f"{k / n:.3e}", f"{cp(k, n)[1]:.3e}",
                                f"{np.median(e['snr']):.2f}", " ".join(f"{x:.2f}" for x in e["snr"])])
    with plt.rc_context(RC):
        fig, axes = PS.figure(7.16, 4.6, 2, 2, wspace=0.06, hspace=0.07, sharex=True)
        for r, (m, mname) in enumerate(MODS):
            axb, axs = axes[r]
            for j, (loc, _, col, mk) in enumerate(LOC):
                x = np.array(RATES, float) + (j - 3) * 0.7
                kw = PS.marks(col, mk, ms=PS.MS - 0.5, label=loc)
                k = np.array([sum(data[loc, m, b]["errs"]) for b in RATES]); n = np.array([sum(data[loc, m, b]["bits"]) for b in RATES])
                hi = np.array([cp(int(k_), int(n_))[1] for k_, n_ in zip(k, n)])
                zero = k == 0
                if not a.log:
                    y = 100.0 * k / n
                    sp = [None if len(data[loc, m, b]["errs"]) < 2 else
                          (100 * min(e / nn for e, nn in zip(data[loc, m, b]["errs"], data[loc, m, b]["bits"])),
                           100 * max(e / nn for e, nn in zip(data[loc, m, b]["errs"], data[loc, m, b]["bits"]))) for b in RATES]
                    PS.series(axb, x, y, sp, **kw)
                else:
                    sp = [None if (len(data[loc, m, b]["errs"]) < 2 or z) else
                          (min(e / nn for e, nn in zip(data[loc, m, b]["errs"], data[loc, m, b]["bits"])),
                           max(e / nn for e, nn in zip(data[loc, m, b]["errs"], data[loc, m, b]["bits"])))
                          for z, b in zip(zero, RATES)]
                    PS.series(axb, x, np.where(zero, np.nan, k / n), sp, **kw)
                    if zero.any():
                        axb.plot(x[zero], hi[zero], zorder=4, **PS.marks(col, "v", hollow=True, ls="none", ms=PS.MS - 1.2))
                med = [float(np.median(data[loc, m, b]["snr"])) for b in RATES]
                sps = [None if len(data[loc, m, b]["snr"]) < 2 else (min(data[loc, m, b]["snr"]), max(data[loc, m, b]["snr"]))
                       for b in RATES]
                PS.series(axs, x, med, sps, **kw)
            if a.log:
                axb.set_yscale("log"); axb.set_ylim(1e-3, 1)
            else:
                axb.set_ylim(bottom=0)
            axb.set_ylabel("Bit error rate after FEC" + ("" if a.log else " (%)"))
            axs.set_ylabel("Symbol SNR (dB)")
            for ax, letter in ((axb, "ac"[r]), (axs, "bd"[r])):
                ax.set_xticks(RATES); ax.set_xlim(RATES[0] - 4, RATES[-1] + 4)
                if ax is axb:
                    PS.note(ax, mname, x=0.015, y=0.97)
                PS.panel(ax, letter, dx=-27 if ax is axb else -22)
                PS.finish(ax)
        for ax in axes[1]:
            ax.set_xlabel("Bit rate (bps)")
        h, l = axes[0][0].get_legend_handles_labels()
        PS.top_legend(fig, h, l, ncol=7, handlelength=1.4, columnspacing=0.9)
        PS.save(fig, os.path.join(a.out, "fig_tablet_locations_trx"))
    for loc, _, _, _ in LOC:
        print(loc.ljust(18), " ".join(f"{m.upper()}:" + ",".join(f"{sum(data[loc, m, b]['errs'])}/{sum(data[loc, m, b]['bits'])}" for b in RATES)
                                       for m, _ in MODS))
    print("->", os.path.join(a.out, "fig_tablet_locations_trx.pdf"))


if __name__ == "__main__":
    main()
