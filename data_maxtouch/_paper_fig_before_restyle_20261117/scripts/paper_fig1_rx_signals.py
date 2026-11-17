#!/usr/bin/env python3
"""
Paper figure 1 call-out: the OOK and FSK signal as the tablet (maXTouch panel) receives it.

Raw panel delta counts of the strongest receiving node vs time over the start of one frame (preamble + SFD + first
payload bits), with the transmitted bits on top. Data = recorded runs, nothing re-decoded; bit timing (offset incl.
the ~12.8 ms panel latency, clock drift) and decoded bits come from the saved v4 known-timing result.

   ~/.venvs/galvanic/bin/python paper_fig1_rx_signals.py                 # default runs below
   ~/.venvs/galvanic/bin/python paper_fig1_rx_signals.py --ook RUN --fsk RUN --frame 0 --bits 40

Output: data_maxtouch/paper_fig1_rx_signals/  (fig1_rx_ook_fsk.{png,pdf,svg}, one file per modulation, data CSVs,
source.json with run names, node, frame errors).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import decode_v4 as D

DATA = os.path.join(HERE, "data_maxtouch")
# clean sitting-still session 2026-10-29 16:13 (TX electrode on ankle, finger on the panel, 1 mA, 20 bps)
OOK_RUN = "stent_mod_ankle2finger_sit_still_ook_20bps_20261029_161342"
FSK_RUN = "stent_mod_ankle2finger_sit_still_fsk_20bps_20261029_161459"

INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#d9d9d9"
TRACE = "#1f5fbf"
SHADE = "#1f5fbf"


def load(run, frame, n_bits, pre_s):
    d = os.path.join(DATA, run)
    meta = json.load(open(os.path.join(d, "meta.json")))
    res = json.load(open(os.path.join(d, "result_v4.json")))
    dec = json.load(open(os.path.join(d, "decoded_v4.json")))
    z = np.load(os.path.join(d, "capture.npz"))
    t, X = z["t"].astype(float), z["X"].astype(float)
    a = D.R.argparse.Namespace(**meta["args"])
    link = D.R.link_from(a, False)
    frame_bits, truth = D.TX.frame_bits(link.payload, link.prbs)
    Tb = 1.0 / meta["plan"]["bps"]
    eps = res["combined_known_timing"]["clock_ppm"] * 1e-6
    fr = next(f for f in dec["known_timing_frames"] if f["m"] == frame)
    s0 = fr["s0"]
    names = D.R.node_names(meta)
    # strongest of v4's 3 nodes = largest ON/OFF energy score (v4 rule 1)
    best = max(res["single_all"], key=lambda s: s["score_db"])
    i = names.index(best["node"])
    lead = t < meta["t_prompt"]
    base = float(np.median(X[lead, i]))
    Tbe = Tb * (1 + eps)
    w = (t >= s0 - pre_s) & (t <= s0 + n_bits * Tbe)
    nh = len(D.HDR)
    got = list(frame_bits[:nh]) + fr["got"] + list(frame_bits[nh + len(truth):])
    return dict(run=run, mod=meta["modulation"], bps=meta["plan"]["bps"], carrier=meta["plan"]["carrier_hz"],
                zero_hz=meta["args"]["fsk_zero"] if meta["modulation"] == "fsk" else 0.0,
                node=best["node"], t=t[w] - s0, y=X[w, i] - base, base=base, Tb=Tbe, bits=frame_bits[:n_bits],
                got=got[:n_bits], hdr=nh, frame=frame, frame_errors=fr["errors"], frame_bits_n=len(truth),
                run_errors=res["combined_known_timing"]["errors"], run_bits=res["combined_known_timing"]["bits"],
                delay_ms=res["combined_known_timing"]["delay_ms"], current_ma=meta["args"].get("current_ma"))


def draw(ax_b, ax_s, s, ylim, letter=None):
    T, bits = s["Tb"], s["bits"]
    n = len(bits)
    # transmitted bits (digital) on top
    xs = np.repeat(np.arange(n + 1) * T, 2)[1:-1]
    ys = np.repeat(bits, 2)
    ax_b.plot(xs, ys, color=INK, lw=1.2, solid_joinstyle="miter")
    ax_b.set_ylim(-0.75, 1.6); ax_b.set_yticks([])
    for k, b in enumerate(bits):
        ax_b.text((k + 0.5) * T, 1.18, str(b), ha="center", va="bottom", fontsize=6.5, color=MUTED)
    ax_b.axvline(s["hdr"] * T, color=MUTED, lw=0.6, ls=(0, (2, 2)))
    ax_b.text(s["hdr"] * T / 2, -0.7, "preamble + SFD", ha="center", va="bottom", fontsize=7, color=MUTED)
    ax_b.text(s["hdr"] * T + (n - s["hdr"]) * T / 2, -0.7, "payload", ha="center", va="bottom", fontsize=7,
              color=MUTED)
    for sp in ("top", "right", "left", "bottom"):
        ax_b.spines[sp].set_visible(False)
    if s["mod"] == "ook":
        title = f"OOK, {s['bps']:g} bps   1 = {s['carrier']/1e3:.1f} kHz on, 0 = off"
    else:
        title = f"FSK, {s['bps']:g} bps   1 = {s['carrier']/1e3:.1f} kHz, 0 = {s['zero_hz']/1e3:.1f} kHz"
    ax_b.set_title(("(" + letter + ")  " if letter else "") + title, loc="left", fontsize=8.5, color=INK, pad=4)
    ax_b.set_ylabel("sent", fontsize=7.5, color=MUTED, rotation=0, ha="right", va="center")
    # received: raw panel counts
    for k, b in enumerate(bits):
        if b:
            ax_s.axvspan(k * T, (k + 1) * T, color=SHADE, alpha=0.08, lw=0)
    ax_s.axhline(0, color=GRID, lw=0.8, zorder=0)
    ax_s.plot(s["t"], s["y"], color=TRACE, lw=0.9, zorder=2)
    ax_s.plot(s["t"], s["y"], "o", ms=2.2, color=TRACE, mec="white", mew=0.3, zorder=3)
    ax_s.set_ylim(*ylim)
    ax_s.set_xlim(min(0.0, s["t"][0]), n * T)
    ax_s.set_xticks(np.arange(0, n * T + 1e-9, 0.25))
    ax_s.set_ylabel(f"panel signal, node {s['node']}\n(counts)", fontsize=7.5, color=INK)
    ax_s.set_xlabel("time from frame start (s)", fontsize=7.5, color=INK)
    ax_s.tick_params(labelsize=7, colors=INK)
    for sp in ("top", "right"):
        ax_s.spines[sp].set_visible(False)
    ax_b.set_xlim(ax_s.get_xlim())
    ok = sum(int(a != b) for a, b in zip(s["bits"], s["got"]))
    ax_b.text(1.0, 1.0, f"received: {ok} bit errors in the {n} bits shown", transform=ax_b.transAxes,
              ha="right", va="bottom", fontsize=6.5, color=MUTED)


def figure(sigs, out, size):
    ylim = max(np.max(np.abs(s["y"])) for s in sigs) * 1.08
    fig = plt.figure(figsize=size)
    gs = fig.add_gridspec(2 * len(sigs), 1, height_ratios=[1, 3.2] * len(sigs), hspace=0.18)
    for j, s in enumerate(sigs):
        ax_b = fig.add_subplot(gs[2 * j]); ax_s = fig.add_subplot(gs[2 * j + 1], sharex=ax_b)
        plt.setp(ax_b.get_xticklabels(), visible=False); ax_b.tick_params(axis="x", length=0)
        draw(ax_b, ax_s, s, (-ylim, ylim), "ab"[j] if len(sigs) > 1 else None)
        if j < len(sigs) - 1:
            ax_s.set_xlabel("")
    multi = len(sigs) > 1
    fig.subplots_adjust(left=0.13, right=0.98, top=0.95 if multi else 0.90, bottom=0.13 if multi else 0.17)
    if len(sigs) > 1:
        # extra gap between the OOK and FSK blocks
        for ax in fig.axes[2:]:
            p = ax.get_position(); ax.set_position([p.x0, p.y0 - 0.035, p.width, p.height])
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{out}.{ext}", dpi=600 if ext == "png" else None)
    plt.close(fig)


CALLOUT_RC = {"font.family": "serif",          # Times New Roman when installed, else its metric twin
              "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
              "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}
FS_YLABEL, FS_YTICK = 20, 20                   # same as the x axis
FS_AX, FS_BITS, FS_NAME = 20, 18, 22         # axis labels + tick numbers / bit labels / OOK-FSK names (pt)


def callout(sigs, b0, nbits, out, size=(7.2, 5.4), border=False):
    with plt.rc_context(CALLOUT_RC):
        _callout(sigs, b0, nbits, out, size, border)


def _callout(sigs, b0, nbits, out, size, border):
    """compact call-out: received panel samples only; row 1 OOK, row 2 FSK, same bit window, sent bits labelled
    on top, 1 bits shaded, normalized amplitude; one shared y label, bold axis text"""
    cols = {"ook": "#1f5fbf", "fsk": "#00897b"}
    fig, axes = plt.subplots(len(sigs), 1, figsize=size, sharex=True)
    top, bottom = 0.86, 0.15
    fig.subplots_adjust(left=0.16, right=0.98, top=top, bottom=bottom, hspace=0.65)
    fig.text(0.045, (top + bottom) / 2, "Normalized Value", rotation=90, ha="center", va="center", fontsize=FS_YLABEL,
             fontweight="bold", color=INK)
    T = sigs[0]["Tb"]
    t0 = b0 * T
    for ax, s in zip(axes, sigs):
        w = (s["t"] >= t0 - 1e-9) & (s["t"] <= t0 + nbits * s["Tb"] + 1e-9)
        tt = (s["t"][w] - t0) * 1e3
        y = s["y"][w] / s["norm"]
        bits = s["bits"][b0:b0 + nbits]
        for k, b in enumerate(bits):
            if b:
                ax.axvspan(k * s["Tb"] * 1e3, (k + 1) * s["Tb"] * 1e3, color=cols[s["mod"]], alpha=0.10, lw=0)
            ax.text((k + 0.5) * s["Tb"] * 1e3, 1.2, str(b), ha="center", va="bottom", fontsize=FS_BITS,
                    fontweight="bold" if b else "normal", color=INK if b else MUTED)
        ax.text(0.0, 1.24, s["mod"].upper(), transform=ax.transAxes, ha="left", va="bottom", fontsize=FS_NAME,
                fontweight="bold", color=cols[s["mod"]])
        ax.axhline(0, color=GRID, lw=1.0, zorder=0)
        ax.plot(tt, y, color=cols[s["mod"]], lw=1.8, zorder=2)
        ax.plot(tt, y, "o", ms=4, color=cols[s["mod"]], mec="white", mew=0.5, zorder=3)
        ax.set_ylim(-1.15, 1.15); ax.set_yticks([-1, 0, 1])
        ax.set_xlim(0, nbits * s["Tb"] * 1e3)
        ax.set_xticks(np.arange(0, nbits * s["Tb"] * 1e3 + 1e-6, 100))
        ax.tick_params(labelsize=FS_AX, colors=INK, length=4, width=1.0)
        ax.tick_params(axis="y", labelsize=FS_YTICK)
        if ax is not axes[-1]:
            ax.tick_params(labelbottom=False)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_linewidth(1.0)
        for lab in ax.get_xticklabels() + ax.get_yticklabels():
            lab.set_fontweight("bold")
    axes[-1].set_xlabel("Time (ms)", fontsize=FS_AX, color=INK, labelpad=3, fontweight="bold")
    if border:
        fig.patches.append(matplotlib.patches.FancyBboxPatch(
            (0.008, 0.012), 0.984, 0.976, boxstyle="round,pad=0,rounding_size=0.025", transform=fig.transFigure,
            fill=False, ec="#7b5cd6", lw=1.0))
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{out}.{ext}", dpi=600 if ext == "png" else None)
    plt.close(fig)


def raw_figure(sigs, b0, nbits, out, zoom=60, size=(6.8, 3.2)):
    """raw panel delta counts (as the chip reports them, no baseline subtraction or scaling) on the strongest node;
    left column full scale, right column the same samples zoomed to +-zoom counts to show the '0' bits"""
    cols = {"ook": "#1f5fbf", "fsk": "#00897b"}
    fig, axes = plt.subplots(len(sigs), 2, figsize=size, sharex=True)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.86, bottom=0.14, hspace=0.6, wspace=0.22)
    t0 = b0 * sigs[0]["Tb"]
    for i, (row, s) in enumerate(zip(axes, sigs)):
        w = (s["t"] >= t0 - 1e-9) & (s["t"] <= t0 + nbits * s["Tb"] + 1e-9)
        tt = (s["t"][w] - t0) * 1e3
        y = s["y"][w] + s["base"]                                # raw counts
        bits = s["bits"][b0:b0 + nbits]
        Tm = s["Tb"] * 1e3
        for j, ax in enumerate(row):
            for k, b in enumerate(bits):
                if b:
                    ax.axvspan(k * Tm, (k + 1) * Tm, color=cols[s["mod"]], alpha=0.10, lw=0)
            ax.axhline(0, color=GRID, lw=0.6, zorder=0)
            ax.plot(tt, y, color=cols[s["mod"]], lw=0.8, zorder=2, clip_on=True)
            ax.plot(tt, y, "o", ms=1.8, color=cols[s["mod"]], mec="white", mew=0.25, zorder=3, clip_on=True)
            lim = 3000 if j == 0 else zoom
            ax.set_ylim(-lim, lim)
            ax.set_xlim(0, nbits * Tm)
            ax.tick_params(labelsize=6, colors=INK, length=2, width=0.5)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_linewidth(0.5)
            for k, b in enumerate(bits):
                ax.text((k + 0.5) * Tm, lim * 1.04, str(b), ha="center", va="bottom", fontsize=6,
                        fontweight="bold" if b else "normal", color=INK if b else MUTED)
            ax.set_ylabel(f"node {s['node']} (counts)", fontsize=6, color=INK, labelpad=1)
            if i == len(sigs) - 1:
                ax.set_xlabel("Time (ms)", fontsize=6.5, color=INK, labelpad=1.5)
            ttl = s["mod"].upper() + ("" if j == 0 else f"  (zoom ±{zoom} counts)")
            ax.text(0.0, 1.17, ttl, transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5,
                    fontweight="bold", color=cols[s["mod"]])
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{out}.{ext}", dpi=600 if ext == "png" else None)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ook", default=OOK_RUN)
    ap.add_argument("--fsk", default=FSK_RUN)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--bits", type=int, default=40)
    ap.add_argument("--pre", type=float, default=0.0, help="seconds before the frame start to include")
    ap.add_argument("--callout-bit", type=int, default=14, help="first frame bit shown in the call-out")
    ap.add_argument("--callout-bits", type=int, default=12, help="number of bits in the call-out")
    ap.add_argument("--out",default=os.path.join(DATA, "paper_fig1_rx_signals"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sigs = [load(a.ook, a.frame, a.bits, a.pre), load(a.fsk, a.frame, a.bits, a.pre)]
    figure(sigs, os.path.join(a.out, "fig1_rx_ook_fsk"), (6.0, 5.4))
    for s in sigs:      # a.u. = counts / 99th percentile of |counts| over the whole run on that node
        d = os.path.join(DATA, s["run"]); m = json.load(open(os.path.join(d, "meta.json")))
        z = np.load(os.path.join(d, "capture.npz")); i = D.R.node_names(m).index(s["node"])
        s["norm"] = float(np.percentile(np.abs(z["X"][z["t"] >= m["t_first"], i] - s["base"]), 99))
    callout(sigs, a.callout_bit, a.callout_bits, os.path.join(a.out, "fig1_callout"))
    raw_figure(sigs, a.callout_bit, a.callout_bits, os.path.join(a.out, "fig1_raw_counts"))
    for s in sigs:
        figure([s], os.path.join(a.out, f"fig1_rx_{s['mod']}"), (6.0, 2.7))
        with open(os.path.join(a.out, f"fig1_rx_{s['mod']}_samples.csv"), "w", newline="") as f:
            wr = csv.writer(f); wr.writerow(["t_from_frame_start_s", f"counts_{s['node']}_minus_lead_median"])
            wr.writerows(zip(np.round(s["t"], 6), s["y"].astype(int)))
        with open(os.path.join(a.out, f"fig1_rx_{s['mod']}_bits.csv"), "w", newline="") as f:
            wr = csv.writer(f); wr.writerow(["bit", "t_start_s", "t_end_s", "sent", "decoded_v4"])
            for k, (b, g) in enumerate(zip(s["bits"], s["got"])):
                wr.writerow([k, round(k * s["Tb"], 6), round((k + 1) * s["Tb"], 6), b, g])
    json.dump([{k: v for k, v in s.items() if k not in ("t", "y", "bits", "got")} for s in sigs],
              open(os.path.join(a.out, "source.json"), "w"), indent=1)
    for s in sigs:
        print(f"{s['mod']}: {s['run']}  node {s['node']}  frame {s['frame']} errors {s['frame_errors']}/"
              f"{s['frame_bits_n']}  run {s['run_errors']}/{s['run_bits']}  delay {s['delay_ms']:.1f} ms  "
              f"peak |counts| {np.max(np.abs(s['y'])):.0f}")
    print("->", a.out)


if __name__ == "__main__":
    main()
