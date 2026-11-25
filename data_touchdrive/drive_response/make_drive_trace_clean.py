"""Single-panel time trace of the maXTouch drive at the stent receiver, in the style of the direct-contact
reference figure (28 ms, Voltage vs Time). Raw A0 samples (1 MS/s, 12 bit, 3.3 V FS), mean removed, NO filtering.
Recording: td_touch_20261118_165506 (finger on the screen, receive only, TX off), steady hold 4 (t = 47 s); the window
with the strongest scan bursts in that second is shown."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RUN, T0, WIN, SPS, MV = HERE.parent / "td_touch_20261118_165506", 47.0, 0.028, 1e6, 3300 / 4096
x = np.load(RUN / "rx.npy", mmap_mode="r")
i = int(T0 * SPS)
sec = np.asarray(x[i:i + int(1.2 * SPS)], float) * MV
sec -= np.median(sec)
env = np.sqrt((sec[:len(sec) // 100 * 100].reshape(-1, 100) ** 2).mean(1))   # 0.1 ms blocks
hi = env > 0.5 * np.percentile(env, 99.5)
starts = np.flatnonzero(hi[5:] & ~np.array([hi[j:j + 5].any() for j in range(len(hi) - 5)])) + 5   # burst onsets
starts = starts[(starts > 20) & (starts < len(env) - 300)]
starts = [j for j in starts if hi[j:j + 280].mean() < 0.25]                  # scan bursts only, not the 40 ms continuous events
k = max(starts, key=lambda j: np.percentile(env[j:j + 280], 95))             # onset followed by the strongest 28 ms
s = k * 100 - int(0.0017 * SPS)                      # first burst at ~1.7 ms, like the reference
v = sec[s:s + int(WIN * SPS)]
plt.rcParams.update({"font.size": 13})
fig, a = plt.subplots(figsize=(7.2, 3.0))
a.plot(np.arange(len(v)) / SPS * 1e3, v, lw=0.9, color="#1f6fb2")
a.set_xlim(0, 28); a.set_xlabel("Time (ms)"); a.set_ylabel("Voltage (mV)")
a.grid(axis="both", color="#e6e6e6"); a.set_axisbelow(True)
a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(HERE / f"drive_trace_finger_on.{e}", dpi=200)
print(HERE / "drive_trace_finger_on.png", round(float(np.abs(v).max()), 1))
