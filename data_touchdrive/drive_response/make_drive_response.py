"""Tablet (maXTouch EVK) drive signal as picked up by the stent receiver (TeensyTRX A0, wrist electrodes):
time domain and frequency domain, finger OFF vs finger ON the screen. Raw A0 (1 MS/s, 12 bit, 3.3 V full scale),
mean removed, no filtering. Recording: td_touch2_20261118_171250 (receive only, transmitter off).
touch_ber.py runs do not store the A0 stream, so the 2026-11-23 long-press check itself has no raw trace.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

HERE = Path(__file__).resolve().parent
RUN = HERE.parent / "td_touch2_20261118_171250"
SPS = 1e6
MV = 3300 / 4096
T_OFF, T_ON, SPAN = 4.0, 44.0, 2.0                 # s: lead (hands off) / hold 3 (finger on); span used for spectra
x = np.load(RUN / "rx.npy", mmap_mode="r")


def seg(t0, w):
    i = int(t0 * SPS)
    v = np.asarray(x[i:i + int(w * SPS)], float) * MV
    return v - v.mean()


def burst_start(t0):
    v = seg(t0, 0.05)
    env = np.sqrt((v[:len(v) // 100 * 100].reshape(-1, 100) ** 2).mean(1))
    thr = 0.5 * env.max()
    k = 50 + int(np.argmax((env[50:] > thr) & (env[45:-5] < 0.5 * thr)))
    assert env[k] > thr, "no burst found"
    return t0 + k * 100 / SPS


tb = burst_start(T_ON)
OFFC, ONC = "#8a8f98", "#1f5fbf"
fig = plt.figure(figsize=(11, 10.5))
gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1.25])
lim = 1.15 * np.percentile(np.abs(seg(tb - 0.003, 0.03)), 99.95)
rows = [(T_OFF, 0.03, OFFC, "finger OFF the screen - 30 ms"),
        (tb - 0.003, 0.03, ONC, "finger ON the screen - 30 ms (bursts = one scan each)"),
        (tb - 0.0002, 0.0015, ONC, "finger ON - one burst, 1.5 ms")]
for r, (t0, w, c, title) in enumerate(rows):
    a = fig.add_subplot(gs[r, :] if r < 2 else gs[r, 0])
    v = seg(t0, w)
    a.plot(np.arange(len(v)) / SPS * 1e3, v, lw=0.5, color=c)
    a.set_title(title, loc="left", fontsize=10)
    a.set_ylabel("A0, AC (mV)"); a.set_ylim(-lim, lim); a.set_xlabel("Time (ms)")
    a.grid(color="#eee"); a.spines[["top", "right"]].set_visible(False)
a = fig.add_subplot(gs[2, 1])
v = seg(tb + 0.00005, 0.00012)
a.plot(np.arange(len(v)) / SPS * 1e6, v, lw=1, marker="o", ms=3, color=ONC)
a.set_title("finger ON - 120 us inside the burst (samples)", loc="left", fontsize=10)
a.set_xlabel("Time (us)"); a.set_ylabel("A0, AC (mV)"); a.set_ylim(-lim, lim)
a.grid(color="#eee"); a.spines[["top", "right"]].set_visible(False)

a = fig.add_subplot(gs[3, :])
res = {}
for t0, c, lab in ((T_OFF, OFFC, "finger OFF"), (T_ON, ONC, "finger ON")):
    f, p = welch(seg(t0, SPAN), SPS, nperseg=8192, noverlap=4096, window="hann", scaling="spectrum")
    db = 10 * np.log10(p / 1e6 + 1e-30)            # dBV rms per line (mV^2 -> V^2)
    a.plot(f / 1e3, db, lw=0.8, color=c, label=lab)
    res[lab] = (f, db)
f, on = res["finger ON"]; off = res["finger OFF"][1]
peaks = {}
for h in (1, 3, 5):
    m = (f > h * 70.6e3 - 3e3) & (f < h * 70.6e3 + 3e3)
    k = np.flatnonzero(m)[np.argmax(on[m])]
    peaks[h] = dict(f_khz=round(f[k] / 1e3, 2), on_db=round(on[k], 1), off_db=round(off[k], 1))
    a.annotate(f"{h}x  {f[k]/1e3:.1f} kHz\n{on[k]-off[k]:+.0f} dB", (f[k] / 1e3, on[k]), xytext=(0, 8),
               textcoords="offset points", ha="center", fontsize=9)
a.set_xlim(0, 500); a.set_ylim(top=on.max() + 14)
a.set_xlabel("Frequency (kHz)"); a.set_ylabel("Level (dBV rms, 122 Hz bins)")
a.set_title(f"Spectrum, {SPAN:.0f} s each (Welch, Hann 8192) - no anti-alias filter: lines above 500 kHz fold back",
            loc="left", fontsize=10)
a.legend(frameon=False, loc="upper right"); a.grid(color="#eee"); a.spines[["top", "right"]].set_visible(False)
fig.suptitle("maXTouch drive signal at the stent receiver (TeensyTRX A0, wrist electrodes, 1 MS/s, TX off)\n"
             "recording td_touch2, 2026-11-18", fontsize=12)
fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(HERE / f"drive_response_time_freq.{e}", dpi=150)
rms = {k: round(float(np.sqrt((seg(t, SPAN) ** 2).mean())), 2) for k, t in (("off_mV", T_OFF), ("on_mV", T_ON))}
json.dump(dict(run=RUN.name, t_off=T_OFF, t_on=T_ON, burst_t=tb, peaks=peaks, rms=rms),
          open(HERE / "drive_response.json", "w"), indent=1)
print(peaks, rms)
