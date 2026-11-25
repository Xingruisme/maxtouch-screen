"""Time-domain view of the tablet drive picked up by the stent receiver (TeensyTRX A0, RX electrodes on the wrist),
like a scope capture: finger off vs finger on the screen, plus a zoom. Raw A0 samples (1 MS/s, 12-bit, 3.3 V full
scale), mean removed, no filtering.
Run: td_ctl_bezel_20261118_165746 - despite its label a SCREEN-touch trial (user correction 2026-11-18), wrist.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE / "td_ctl_bezel_20261118_165746"
SPS = 1e6
MV = 3300 / 4096                                    # mV per ADC count
T_OFF, T_ON = 54.0, 62.0                            # s from the recording start: finger off (rest 4) / on (hold 4)
WIN, ZOOM = 0.028, 0.0015

x = np.load(RUN / "rx.npy", mmap_mode="r")


def seg(t0, w):
    i = int(t0 * SPS)
    v = np.asarray(x[i:i + int(w * SPS)], float) * MV
    return np.arange(len(v)) / SPS * 1e3, v - v.mean()


# start the touched window 3 ms before a burst: first 0.1 ms block whose RMS is above 30 mV after T_ON + 5 ms
_, v = seg(T_ON, 0.05)
env = np.sqrt((v[:len(v) // 100 * 100].reshape(-1, 100) ** 2).mean(1))
k = 50 + int(np.argmax(env[50:] > 30))
t_burst = T_ON + k * 100 / SPS
t_on = t_burst - 0.003

C = "#1f5fbf"
fig, ax = plt.subplots(3, 1, figsize=(10, 8.2))
panels = [(T_OFF, WIN, "finger OFF the screen - 28 ms"), (t_on, WIN, "finger ON the screen - 28 ms"),
          (t_burst - 0.0002, ZOOM, "finger ON the screen - one burst, 1.5 ms (zoom)")]
lim = None
for a, (t0, w, title) in zip(ax, panels):
    tt, vv = seg(t0, w)
    a.plot(tt, vv, lw=0.5, color=C)
    a.set_title(title, loc="left", fontsize=10)
    a.set_ylabel("A0 voltage, AC (mV)")
    a.grid(color="#eee")
    a.spines[["top", "right"]].set_visible(False)
    lim = lim or max(abs(np.percentile(np.r_[seg(t_on, WIN)[1]], [0.05, 99.95])))
    a.set_ylim(-1.15 * lim, 1.15 * lim)
ax[-1].set_xlabel("Time (ms)")
fig.suptitle("Tablet drive picked up by the stent receiver (TeensyTRX A0, wrist electrodes, 1 MS/s)", fontsize=12)
fig.tight_layout()
out = HERE / "time_trace_touch_vs_off"
for e in ("png", "pdf"):
    fig.savefig(f"{out}.{e}", dpi=150)
print(out)
