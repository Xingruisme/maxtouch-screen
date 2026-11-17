#!/usr/bin/env python3
"""
Paper figure: tablet (maXTouch) BER vs bit rate, OOK and FSK.

Data = the pooled sitting-still ankle-to-finger table already made by `stent_modcomp.py conditions`
(6 sessions; 3 recordings per rate, 3048 bits per cell; frozen decode_v4, known timing). Nothing is re-decoded.
Error bars = Clopper-Pearson 95 % interval.

   python paper_fig_tablet_ber.py [--src data_maxtouch/modcomp_conditions_a2f_sitstill_20to60_ber/conditions.csv]

Output: data_maxtouch/paper_fig_tablet_ber/ (fig_tablet_ber.{png,pdf,svg}, fig_tablet_ber.csv)
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
from scipy.stats import beta

from paper_style import RC, STYLE              # the look of the print-size figures, which import both from here

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "data_maxtouch", "modcomp_conditions_a2f_sitstill_20to60_ber", "conditions.csv")
# this script's own two large-format figures (fig_tablet_ber, fig_tablet_snr) keep their earlier look
RC_BIG = {"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
          "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}
INK, GRID = "#1a1a1a", "#d9d9d9"
FS = 20


def cp(k, n):
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=os.path.join(HERE, "data_maxtouch", "paper_fig_tablet_ber"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = list(csv.DictReader(open(a.src)))
    data = {}
    for r in rows:
        k, n = int(r["errors"]), int(r["bits"])
        data.setdefault(r["modulation"], []).append((round(float(r["bps"])), k, n, k / n, *cp(k, n)))
    with open(os.path.join(a.out, "fig_tablet_ber.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["modulation", "bps", "errors", "bits", "ber", "ci95_lo", "ci95_hi"])
        for m in ("ook", "fsk"):
            for p in sorted(data[m]):
                w.writerow([m, *p[:3], f"{p[3]:.3e}", f"{p[4]:.3e}", f"{p[5]:.3e}"])

    with plt.rc_context(RC_BIG):
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        fig.subplots_adjust(left=0.19, right=0.97, top=0.95, bottom=0.17)
        off = {"ook": -0.6, "fsk": 0.6}                     # small x offset so the error bars don't overlap
        for m in ("ook", "fsk"):
            p = np.array(sorted(data[m]), dtype=float)
            x, ber, lo, hi = p[:, 0], p[:, 3], p[:, 4], p[:, 5]
            st = STYLE[m]
            ax.errorbar(x + off[m], ber, yerr=[ber - lo, hi - ber], color=st["color"], marker=st["marker"], ms=9,
                        mec="white", mew=1.0, lw=2.4, elinewidth=1.6, capsize=5, capthick=1.6, label=st["label"],
                        zorder=3)
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1e-1)
        rates = sorted({p[0] for v in data.values() for p in v})
        ax.set_xticks(rates); ax.set_xlim(rates[0] - 5, rates[-1] + 5)
        ax.grid(True, which="major", color=GRID, lw=0.8, zorder=0)
        ax.set_xlabel("Bit rate (bps)", fontsize=FS, fontweight="bold", color=INK)
        ax.set_ylabel("BER", fontsize=FS, fontweight="bold", color=INK)
        ax.tick_params(labelsize=FS, colors=INK, width=1.0, length=5)
        ax.tick_params(which="minor", length=2.5)
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_fontweight("bold")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_linewidth(1.0)
        leg = ax.legend(loc="upper left", frameon=False, prop={"size": FS, "weight": "bold"}, handlelength=1.8)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(os.path.join(a.out, f"fig_tablet_ber.{ext}"), dpi=600 if ext == "png" else None)
        plt.close(fig)
    for m in ("ook", "fsk"):
        print(m, "  ".join(f"{p[0]}:{p[1]}/{p[2]}={p[3]:.2%}" for p in sorted(data[m])))
    snr_figure(rows, a.out)
    print("->", a.out)


def per_run_snr(rows):
    """Es/N0 (dB) of every recording behind each pooled cell, read from the sessions' summary.json (no re-decode).
    Es/N0 = k * P1 / (2 sigma^2) as in stent_modcomp.es_n0_db (k = panel samples per bit)."""
    base = os.path.join(HERE, "data_maxtouch")
    per = {}
    for r in rows:
        want = set(r["run"].split("+"))
        for sess in r["session"].split("+"):
            for s in json.load(open(os.path.join(base, sess, "summary.json")))["rows"]:
                if s["run"] in want and s.get("es_n0_db") is not None:
                    per.setdefault((r["modulation"], round(float(r["bps"]))), {})[s["run"]] = s["es_n0_db"]
    return {k: sorted(v.values()) for k, v in per.items()}


def snr_figure(rows, out):
    per = per_run_snr(rows)
    with open(os.path.join(out, "fig_tablet_snr.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["modulation", "bps", "n_recordings", "es_n0_db_mean", "min", "max", "values"])
        for (m, b), v in sorted(per.items()):
            w.writerow([m, b, len(v), f"{np.mean(v):.2f}", f"{min(v):.2f}", f"{max(v):.2f}",
                        " ".join(f"{x:.2f}" for x in v)])
    with plt.rc_context(RC_BIG):
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        fig.subplots_adjust(left=0.17, right=0.97, top=0.95, bottom=0.17)
        off = {"ook": -0.6, "fsk": 0.6}
        for m in ("ook", "fsk"):
            ks = sorted(b for mm, b in per if mm == m)
            x = np.array(ks, float)
            mu = np.array([np.mean(per[m, b]) for b in ks])
            lo = np.array([min(per[m, b]) for b in ks]); hi = np.array([max(per[m, b]) for b in ks])
            st = STYLE[m]
            ax.errorbar(x + off[m], mu, yerr=[mu - lo, hi - mu], color=st["color"], marker=st["marker"], ms=9,
                        mec="white", mew=1.0, lw=2.4, elinewidth=1.6, capsize=5, capthick=1.6, label=st["label"],
                        zorder=3)
        rates = sorted({b for _, b in per})
        ax.set_xticks(rates); ax.set_xlim(rates[0] - 5, rates[-1] + 5)
        ax.set_ylim(30, 70)
        ax.grid(True, which="major", color=GRID, lw=0.8, zorder=0)
        ax.set_xlabel("Bit rate (bps)", fontsize=FS, fontweight="bold", color=INK)
        ax.set_ylabel("SNR (dB)", fontsize=FS, fontweight="bold", color=INK)
        ax.tick_params(labelsize=FS, colors=INK, width=1.0, length=5)
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_fontweight("bold")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_linewidth(1.0)
        ax.legend(loc="lower left", frameon=False, prop={"size": FS, "weight": "bold"}, handlelength=1.8)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(os.path.join(out, f"fig_tablet_snr.{ext}"), dpi=600 if ext == "png" else None)
        plt.close(fig)
    for (m, b), v in sorted(per.items()):
        print(f"SNR {m} {b} bps: mean {np.mean(v):.1f} dB  ({' '.join(f'{x:.1f}' for x in v)})")


if __name__ == "__main__":
    main()
