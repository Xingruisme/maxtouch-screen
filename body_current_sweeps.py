#!/usr/bin/env python3
"""
Through-body frequency sweeps at several set currents -> one figure (slide style of 2026-10-24):
  a  energy increase vs carrier, best of Y0-Y3, one line per current
  b  carrier amplitude on the panel vs set current: scan-to-scan rms during the ON blocks of 155.7 / 159.6 / 163.6 kHz
     (the carrier, random phase every scan), the same during the OFF blocks (noise + finger), and the line expected if
     the amplitude were proportional to the current, through the reference current's mean

   python body_current_sweeps.py 0.5=data_maxtouch/stent_probe_body_0p5mA_20261026_151929 \
                                 1=data_maxtouch/stent_probe_body_1mA_20261026_155003 ... --out data_maxtouch/body_sweep_currents_20261026

A sweep counts as 'carrier not detected' unless, at one of the three carriers, the best Y0-Y3 node clears that sweep's
own response threshold (median + 6 robust sigma of its node x carrier matrix, as maxtouch_stent_rx.analyse_probe) AND
swings at least 3x more in the ON blocks than in the OFF blocks (a finger jump alone can clear the threshold).
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import io
import json
import os

import numpy as np

import maxtouch_stent_rx as R

PEAKS = (155692.5, 159584.9, 163574.5)
NODES = 4                                       # Y0-Y3, where the finger sits
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # categorical slots 1-5
# a current keeps its colour when other currents are added (slide of 2026-10-26: 0.5 / 1 / 2.5 / 4 mA = slots 1-4)
CURRENT_SLOT = {0.5: 0, 1.0: 1, 2.5: 2, 4.0: 3, 0.75: 4}


def colour(cur, used):
    if cur in CURRENT_SLOT:
        return COLORS[CURRENT_SLOT[cur]]
    free = [c for c in COLORS if c not in used]
    return free[0] if free else COLORS[len(used) % len(COLORS)]
INK2, GRID = "#52514e", "#e4e3df"


def load(d):
    meta = json.load(open(os.path.join(d, "meta.json")))
    z = np.load(os.path.join(d, "capture.npz"))
    with contextlib.redirect_stdout(io.StringIO()):
        res, aux = R.analyse_probe(z["t"], z["X"], meta)
    return meta, res, aux


def block_rms(aux, meta, f):
    """(ON, OFF) scan-to-scan rms [counts] of the best Y0-Y3 node at carrier f"""
    E, car, G, T, tau = aux["E"], list(aux["car"]), aux["G"], aux["T"], aux["tau"]
    a = meta["args"]
    j = car.index(f)
    n = int(np.argmax(E[:NODES, j]))
    b = int(round(a["block"] / T))
    s0 = tau + int(round((2 * 3.0 + j * 2 * a["reps"] * a["block"]) / T))     # after the 3 s on / 3 s off marker
    on = np.concatenate([np.arange(s0 + 2 * k * b + 3, s0 + (2 * k + 1) * b - 1) for k in range(a["reps"])])
    off = np.concatenate([np.arange(s0 + (2 * k + 1) * b + 3, s0 + (2 * k + 2) * b - 1) for k in range(a["reps"])])
    r = lambda idx: float(np.diff(G[idx, n]).std() / np.sqrt(2))
    return r(on), r(off), float(E[n, j])


def main():
    ap = argparse.ArgumentParser(description="through-body sweeps vs set current -> figure")
    ap.add_argument("runs", nargs="+", help="CURRENT_mA=probe_folder")
    ap.add_argument("--ref", type=float, default=1.0, help="current [mA] the 'proportional' line goes through")
    ap.add_argument("--title", default="Through-body sweep, finger on panel at Y0–Y3", help="panel a title")
    ap.add_argument("--out", required=True, help="output path without extension (.png and .pdf are written)")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    runs = sorted(((float(c), d) for c, d in (r.split("=", 1) for r in a.runs)), key=lambda x: x[0])
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.6), gridspec_kw=dict(width_ratios=[1.7, 1]))
    rows = []
    for k, (cur, d) in enumerate(runs):
        meta, res, aux = load(d)
        E, car = aux["E"], np.array(aux["car"])
        thr = res["noise_floor_db"]["threshold"]
        amp = [block_rms(aux, meta, f) for f in PEAKS]
        # a real carrier both clears the threshold and swings well above the OFF blocks: a touch jump alone can lift
        # the energy over the threshold (back2finger 4 mA, 163.6 kHz: +13.5 dB with ON 6 vs OFF 3.5 counts)
        detected = any(e >= thr and on >= 3 * off for on, off, e in amp)
        when = datetime.datetime.fromtimestamp(meta["t_first"]).strftime("%H:%M")
        c = colour(cur, [r["colour"] for r in rows])
        ax[0].semilogx(car, E[:NODES].max(axis=0), "-o", ms=3, lw=1.6, color=c,
                       label=f"{cur:g} mA" + ("" if detected else " - carrier not detected"))
        rows.append(dict(colour=c, current_ma=cur, run=os.path.basename(os.path.normpath(d)), time=when, threshold_db=thr,
                         detected=detected, peak_db=float(E[:NODES].max()),
                         on_rms=[x[0] for x in amp], off_rms=[x[1] for x in amp], energy_db=[x[2] for x in amp]))
    ax[0].axhline(6, color=INK2, ls="--", lw=0.8)
    ax[0].set_xlabel("carrier (Hz)"); ax[0].set_ylabel("energy increase, best of Y0–Y3 (dB)")
    ax[0].set_title(a.title, fontsize=10, loc="left")
    cur = np.array([r["current_ma"] for r in rows])
    for r in rows:
        c = r["colour"]
        ax[1].plot([r["current_ma"]] * len(PEAKS), r["on_rms"], "o", ms=8, color=c,
                   mfc=c if r["detected"] else "white", mew=1.8, zorder=3)
    ax[1].plot(cur, [np.mean(r["on_rms"]) for r in rows], "-", color=INK2, lw=1.5,
               label=f"carrier ON (mean of {', '.join(f'{f / 1e3:.1f}' for f in PEAKS)} kHz)")
    ax[1].plot(cur, [np.mean(r["off_rms"]) for r in rows], ":", color=INK2, lw=1.5, label="carrier OFF (noise + finger)")
    ref = [r for r in rows if abs(r["current_ma"] - a.ref) < 1e-9 and r["detected"]]
    if ref:
        g = np.linspace(cur.min(), cur.max(), 50)
        ax[1].plot(g, np.mean(ref[0]["on_rms"]) * g / a.ref, "--", color=INK2, lw=1.2,
                   label=f"expected if proportional to current (through {a.ref:g} mA)")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("set current (mA)"); ax[1].set_ylabel("carrier amplitude on panel (counts rms)")
    ax[1].set_xticks(cur); ax[1].set_xticklabels([f"{c:g}" for c in cur])
    ax[1].set_title("Received amplitude vs set current", fontsize=10, loc="left")
    if not all(r["detected"] for r in rows):
        ax[1].plot([], [], "o", ms=8, color=INK2, mfc="white", mew=1.8, label="hollow = carrier not detected")
    for a_, letter in zip(ax, "ab"):
        a_.grid(color=GRID, lw=0.6, which="both"); a_.set_axisbelow(True)
        for s in ("top", "right"):
            a_.spines[s].set_visible(False)
        a_.text(-0.08 if a_ is ax[0] else -0.16, 1.04, letter, transform=a_.transAxes, fontsize=14, fontweight="bold")
        if a_ is ax[0]:
            a_.legend(fontsize=8, frameon=False, loc="upper left")
        else:                                   # below the panel: inside it would cover the points
            a_.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1)
    fig.tight_layout()
    fig.savefig(a.out + ".png", dpi=150); fig.savefig(a.out + ".pdf")
    json.dump(rows, open(a.out + ".json", "w"), indent=1)
    for r in rows:
        print(f"{r['current_ma']:4g} mA ({r['time']}): peak {r['peak_db']:+5.1f} dB (threshold {r['threshold_db']:.1f}), "
              f"ON {np.round(r['on_rms']).astype(int)} / OFF {np.round(r['off_rms'], 1)} counts"
              + ("" if r["detected"] else "   -> carrier not detected"))
    print(f"figure: {a.out}.png / .pdf, numbers: {a.out}.json")


if __name__ == "__main__":
    main()
