#!/usr/bin/env python3
"""
Time trace of the OnePlus Pad's touch drive picked up by the TeensyTRX receiver (A0), DIRECT CONTACT (RX electrode
held in the touching hand, no body path), 2026-11-24. Finger touch (left axis, blue, 'Screen') and stylus touch
(right axis, orange, 'Stylus') overlaid on one time axis, like the External-screen figure.

Recording (receive only, the AD9833 stays off; --no-evk because the OnePlus is not the maXTouch EVK):
   python touch_drive_rx.py run --label oneplus_finger --site directcontact --free 12 --no-evk \
          --condition "OnePlus Pad, direct contact, finger held on the screen"
   python touch_drive_rx.py run --label oneplus_stylus --site directcontact --free 12 --no-evk \
          --condition "OnePlus Pad, direct contact, stylus held on the screen"
Plot:
   python oneplus_drive_trace.py data_touchdrive/td_oneplus_finger_<time> data_touchdrive/td_oneplus_stylus_<time>
          [--win 30] [--pre 3] [--after 2]

Raw A0 samples (12-bit, 3.3 V full scale), mean removed, no filtering. Each trace starts --pre ms before its first
burst after --after s: first 0.1 ms block whose RMS exceeds 30 % of the run's 99.9th-percentile block RMS.
Output: data_touchdrive/oneplus_drive/drive_trace.{png,pdf} + drive_trace.csv (time, finger mV, stylus mV).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "data_touchdrive" / "oneplus_drive"
MV = 3300 / 4096                                    # mV per ADC count
BLUE, ORANGE = "#1f77b4", "#d9681e"


def load(run: Path, win_ms: float, pre_ms: float, after_s: float):
    sps = json.loads((run / "meta.json").read_text()).get("sps", 1_000_000)
    x = np.load(run / "rx.npy", mmap_mode="r")
    blk = int(sps * 1e-4)                           # 0.1 ms blocks
    i0 = int(after_s * sps)
    v = np.asarray(x[i0:], float)
    v = v[:len(v) // blk * blk]
    env = (v.reshape(-1, blk).std(1)) * MV
    k = int(np.argmax(env > 0.3 * np.percentile(env, 99.9)))
    start = max(0, i0 + k * blk - int(pre_ms * 1e-3 * sps))
    seg = np.asarray(x[start:start + int(win_ms * 1e-3 * sps)], float) * MV
    return np.arange(len(seg)) / sps * 1e3, seg - seg.mean(), start / sps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("finger", help="run folder, finger touch")
    ap.add_argument("stylus", help="run folder, stylus touch")
    ap.add_argument("--win", type=float, default=30.0, help="window length [ms]")
    ap.add_argument("--pre", type=float, default=3.0, help="time before the first burst [ms]")
    ap.add_argument("--after", type=float, default=2.0, help="skip this much of each recording [s]")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    tf, vf, sf = load(Path(a.finger), a.win, a.pre, a.after)
    ts, vs, ss = load(Path(a.stylus), a.win, a.pre, a.after)
    print(f"finger: window from {sf:.4f} s, p-p {np.ptp(vf):.1f} mV;  stylus: from {ss:.4f} s, p-p {np.ptp(vs):.1f} mV")

    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ax2 = ax.twinx()
    ax.plot(tf, vf, lw=0.6, color=BLUE)
    ax2.plot(ts, vs, lw=0.6, color=ORANGE)
    ax.set_ylabel("Screen (mV)", color=BLUE)
    ax2.set_ylabel("Stylus (mV)", color=ORANGE)
    ax.tick_params(axis="y", colors=BLUE)
    ax2.tick_params(axis="y", colors=ORANGE)
    ax.spines["left"].set_color(BLUE)
    ax2.spines["right"].set_color(ORANGE)
    for s in ("top",):
        ax.spines[s].set_visible(False)
        ax2.spines[s].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Time (ms)")
    ax.set_title("OnePlus", fontsize=11)
    # zero lines of both axes at the same height
    for axx, v in ((ax, vf), (ax2, vs)):
        m = 1.1 * np.abs(v).max()
        axx.set_ylim(-m, m)
    fig.tight_layout()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"drive_trace.{ext}", dpi=300)
    n = min(len(vf), len(vs))
    np.savetxt(out / "drive_trace.csv", np.c_[tf[:n], vf[:n], vs[:n]], delimiter=",", fmt="%.4f",
               header=f"t_ms,finger_mV,stylus_mV  finger={Path(a.finger).name} stylus={Path(a.stylus).name}",
               comments="")
    print(f"-> {out}/drive_trace.png")


if __name__ == "__main__":
    main()
