"""Drive signal at the stent receiver (TeensyTRX A0, 1 MS/s, 12 bit, 3.3 V FS): 28 ms time trace in the style of the
reference figure + spectrum. Raw samples, median removed, NO filtering.
   python make_drive_fig.py <run folder name> <out name> [t0 seconds]      (t0 omitted: strongest second is used)"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

HERE = Path(__file__).resolve().parent
RUN, OUT = HERE.parent / sys.argv[1], sys.argv[2]
WIN, SPS, MV = 0.028, 1e6, 3300 / 4096
x = np.load(RUN / "rx.npy", mmap_mode="r")
n = len(x) // 1000000
sd = [float(np.asarray(x[s * 1000000:(s + 1) * 1000000:7], float).std()) * MV for s in range(n)]
T0 = float(sys.argv[3]) if len(sys.argv) > 3 else float(np.argsort(sd)[len(sd) // 2])     # a typical (median) second
sec = np.asarray(x[int(T0 * SPS):int((T0 + 1) * SPS)], float) * MV
sec -= np.median(sec)
env = np.sqrt((sec[:len(sec) // 100 * 100].reshape(-1, 100) ** 2).mean(1))               # 0.1 ms blocks
hi = env > 0.5 * np.percentile(env, 99.5)
on = np.flatnonzero(hi[5:-300] & ~np.array([hi[j:j + 5].any() for j in range(len(hi) - 305)])) + 5
k = int(on[len(on) // 2]) if len(on) else 20
s = max(0, k * 100 - int(0.0017 * SPS))
v = sec[s:s + int(WIN * SPS)]
C = "#1f6fb2"
plt.rcParams.update({"font.size": 13})
fig, a = plt.subplots(figsize=(7.2, 3.0))
a.plot(np.arange(len(v)) / SPS * 1e3, v, lw=0.9, color=C)
a.set_xlim(0, 28); a.set_xlabel("Time (ms)"); a.set_ylabel("Voltage (mV)")
a.grid(color="#e6e6e6"); a.set_axisbelow(True); a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(HERE / f"{OUT}_time.{e}", dpi=200)

fig, ax = plt.subplots(3, 1, figsize=(7.6, 8.6), height_ratios=[1, 1, 1.15])
ax[0].plot(np.arange(len(v)) / SPS * 1e3, v, lw=0.9, color=C); ax[0].set_xlim(0, 28)
e2 = np.sqrt((v[:len(v) // 100 * 100].reshape(-1, 100) ** 2).mean(1)); big = np.flatnonzero(e2 > 0.5 * e2.max())
z0 = max(0, int(big[big < 260][0]) * 100 - 300)                    # first main burst in the window
z = v[z0:z0 + 2000]
ax[1].plot((np.arange(len(z)) + z0) / SPS * 1e3, z, lw=0.9, color=C); ax[1].set_xlim(z0 / 1e3, z0 / 1e3 + 2)
for a in ax[:2]:
    a.set_xlabel("Time (ms)"); a.set_ylabel("Voltage (mV)")
f, p = welch(sec, SPS, nperseg=8192, noverlap=4096, window="hann", scaling="spectrum")
db = 10 * np.log10(p / 1e6 + 1e-30)
ax[2].plot(f / 1e3, db, lw=0.9, color=C); ax[2].set_xlim(0, 500)
top = []
for kk in np.argsort(db)[::-1]:
    if f[kk] > 5e3 and all(abs(f[kk] - f[j]) > 8e3 for j in top):
        top.append(kk)
    if len(top) == 5:
        break
for kk in top:
    ax[2].annotate(f"{f[kk]/1e3:.1f}", (f[kk] / 1e3, db[kk]), xytext=(0, 4), textcoords="offset points",
                   ha="center", fontsize=10)
ax[2].set_ylim(top=db[top[0]] + 9)
ax[2].set_xlabel("Frequency (kHz)"); ax[2].set_ylabel("Level (dBV rms)")
for a in ax:
    a.grid(color="#e6e6e6"); a.set_axisbelow(True); a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(HERE / f"{OUT}_time_freq.{e}", dpi=200)
print("t0", T0, "window start", round(T0 + s / SPS, 4), "rms/s mV", np.round(sd, 1),
      "peak mV", round(float(np.abs(v).max()), 1), "lines kHz/dBV", [(round(f[j] / 1e3, 1), round(db[j], 1)) for j in top],
      "bursts in 1 s", len(on), "period ms", np.round(np.median(np.diff(on)) / 10, 2) if len(on) > 2 else None)
