#!/usr/bin/env python3
"""
Is the signal the stent receiver picks up really the maXTouch drive? Waveform analysis of the existing receive-only
recordings (touch_drive_rx.py, data_touchdrive/td_*), 2026-11-18. Read-only on the data; nothing is transmitted.

   python touch_drive_waveform.py                 # all 2026-11-18 runs -> data_touchdrive/waveform/
   python touch_drive_waveform.py --scope DIR     # also measure scope CSVs (screen1.csv / screen2.csv) the same way

Per run:
  W1 segments: 20 ms band power of the 70.6 kHz line (touch_drive_rx.block_spectra, +-2 bins) -> ON while 6 dB over
     the run's baseline (20th percentile), OFF below +3 dB, 100 ms debounce. These are RECEIVER-DEFINED segments,
     used only to describe the waveform; they cannot measure detection accuracy (the tablet's touch messages are
     needed for that). Agreement with the prompts is reported alongside.
  W2 bursts (raw A0, mean removed, 40-450 kHz band-pass): envelope = 10 us moving RMS; threshold = 5 x the median
     envelope of the run's OFF segments; envelope gaps < 20 us are merged into one SUB-BURST, sub-bursts < 1 ms apart
     into one BURST. Per segment: burst count, duration, repetition period, sub-bursts per burst, peak-to-peak.
  W3 carrier: per sub-burst, rising zero crossings of the 55-90 kHz band-passed signal -> frequency.
  W4 spectrum of each ON segment (Welch, 61 Hz bins): harmonics 2f..7f relative to f (a 50 % square wave has only
     odd ones: 3f -9.5 dB, 5f -14.0, 7f -16.9), and the 1 MS/s aliases of 9f..15f (A0 has no anti-alias filter that
     we know of).
Outputs (data_touchdrive/waveform/): cycles.csv (one row per segment), runs.csv (one row per run), waveform_<run>.png
for the main screen-touch run, compare_runs.png, and scope.csv / scope.png with --scope.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import touch_drive_rx as T                                          # noqa: E402  (block_spectra, sample_times)

D = HERE / "data_touchdrive"
OUT = D / "waveform"
MV = 3300 / 4096
F0_GUESS = 70.6e3
# condition of each run as CORRECTED by the user (labels on disk are kept)
RUNS = {
    "td_ctl_evk_unplugged_20261118_163339": "EVK unplugged",
    "td_ctl_notouch_20261118_163617": "EVK on, no touch",
    "td_touch_20261118_165506": "screen touch (run A)",
    "td_ctl_bezel_20261118_165746": "screen touch (run B, label 'ctl_bezel')",
    "td_ctl_bezel_20261118_170426": "bezel touch",
    "td_truthcheck_20261118_171223": "screen taps (free)",
    "td_touch2_20261118_171250": "screen touch (run C)",
    "td_ctl_hover_20261118_171454": "hover 1 cm",
}
MAIN = "td_touch2_20261118_171250"


def bp(x, lo, hi, fs, order=4):
    return sosfiltfilt(butter(order, [lo, hi], btype="band", fs=fs, output="sos"), x)


def movrms(x, n):
    return np.sqrt(np.convolve(x * x, np.ones(n) / n, mode="same"))


def runs_of(mask):
    """[(start, stop)] of True runs"""
    m = np.r_[False, mask, False].astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def segments(bdb, on_db=6.0, off_db=3.0, hold=5):
    """W1: hysteresis + debounce on band power [dB] -> list of (i0, i1, state) in blocks"""
    base = np.percentile(bdb, 20)
    st, cnt, out, i0 = False, 0, [], 0
    for i, v in enumerate(bdb):
        want = (v > base + on_db) if not st else not (v < base + off_db)
        if want != st:
            cnt += 1
            if cnt >= hold:
                out.append((i0, i - hold + 1, st))
                st, cnt, i0 = want, 0, i - hold + 1
        else:
            cnt = 0
    out.append((i0, len(bdb), st))
    return [s for s in out if s[1] > s[0]], base


def bursts(xb, fs, thr):
    """W2 on a band-passed segment -> sub-bursts and bursts (sample indices)"""
    env = movrms(xb, int(10e-6 * fs))
    sub = runs_of(env > thr)
    merged = []
    for a, b in sub:
        if merged and a - merged[-1][1] < 20e-6 * fs:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    merged = [(a, b) for a, b in merged if b - a >= 20e-6 * fs]          # at least ~1.5 cycles
    groups = []
    for a, b in merged:
        if groups and a - groups[-1][1] < 1e-3 * fs:
            groups[-1] = (groups[-1][0], b, groups[-1][2] + 1)
        else:
            groups.append((a, b, 1))
    return env, merged, groups


def carrier(xf, fs, a, b):
    """W3: frequency from rising zero crossings (linear interpolation) in [a, b)"""
    s = xf[a:b]
    k = np.flatnonzero((s[:-1] < 0) & (s[1:] >= 0))
    if len(k) < 4:
        return np.nan
    tz = (k + (-s[k]) / (s[k + 1] - s[k])) / fs
    return (len(tz) - 1) / (tz[-1] - tz[0])


def harmonics(x, fs, f0):
    """W4"""
    f, P = welch(x, fs, nperseg=1 << 14)
    def at(fr):
        fr = abs(fr - round(fr / fs) * fs)                                # fold into 0..fs/2
        k = (f > fr - 300) & (f < fr + 300)
        return float(10 * np.log10(P[k].max() + 1e-30)) if k.any() else np.nan
    h1 = at(f0)
    floor = float(10 * np.log10(np.median(P[(f > 10e3) & (f < 490e3)])))
    out = dict(h1_over_floor_db=h1 - floor)
    for n in (2, 3, 4, 5, 7):
        out[f"h{n}_db"] = at(n * f0) - h1
    for n in (9, 11, 13, 15):
        out[f"alias{n}_khz"] = abs(n * f0 - round(n * f0 / fs) * fs) / 1e3
        out[f"alias{n}_db"] = at(n * f0) - h1
    return out


def prompt_overlap(t_seg, prompts):
    """share of the segment inside prompted HOLD periods"""
    iv, cur = [], None
    for p in prompts:
        if p["state"] == 1:
            cur = p["t"]
        elif cur is not None:
            iv.append((cur, p["t"]))
            cur = None
    a, b = t_seg
    inside = sum(max(0.0, min(b, e) - max(a, s)) for s, e in iv)
    return inside / max(b - a, 1e-9)


def analyse_run(name, cond):
    d = D / name
    meta = json.load(open(d / "meta.json"))
    truth = json.load(open(d / "truth.json"))
    fs = float(meta["sps"])
    raw = np.load(d / "rx.npy")
    _, tsec = T.sample_times(np.load(d / "clock.npy"), fs, len(raw))
    f, P, L = T.block_spectra(raw.astype(np.float32), fs)
    k0 = int(np.argmin(abs(f - F0_GUESS)))
    bdb = 10 * np.log10(P[:, k0 - 2:k0 + 3].sum(1) + 1e-30)
    segs, base = segments(bdb)
    # noise floor of the envelope from OFF segments (or the whole run)
    off_idx = [s for s in segs if not s[2]] or [(0, len(bdb), False)]
    samp = []
    for i0, i1, _ in off_idx:
        a, b = i0 * L, min(i1 * L, i0 * L + 2_000_000)
        samp.append(bp(raw[a:b].astype(np.float64) * MV, 40e3, 450e3, fs))
    env_floor = float(np.median(np.concatenate([movrms(s, 10) for s in samp]))) if samp else 1.0
    thr = 5 * env_floor
    rows, on_x, f0s = [], [], []
    for n, (i0, i1, st) in enumerate(segs):
        a, b = i0 * L, i1 * L
        x = raw[a:b].astype(np.float64) * MV
        x -= x.mean()
        xb = bp(x, 40e3, 450e3, fs)
        xf = bp(x, 55e3, 90e3, fs)
        env, sub, grp = bursts(xb, fs, thr)
        fc = [carrier(xf, fs, s0, s1) for s0, s1 in sub]
        fc = [v for v in fc if np.isfinite(v)]
        on_g = [g for g in grp]
        per = np.diff([g[0] for g in on_g]) / fs * 1e3
        dur = np.array([(g[1] - g[0]) / fs * 1e3 for g in on_g])
        pp = np.array([np.ptp(xb[g[0]:g[1]]) for g in on_g]) if on_g else np.array([])
        t0, t1 = tsec[0] + i0 * L / fs, tsec[0] + i1 * L / fs
        rows.append(dict(run=name, condition=cond, segment=n, state="ON" if st else "OFF",
                         t_start_s=round(i0 * L / fs, 2), t_end_s=round(i1 * L / fs, 2),
                         dur_s=round((i1 - i0) * L / fs, 2),
                         in_prompted_hold=round(prompt_overlap((t0, t1), truth.get("prompts", [])), 2),
                         band_db_over_base=round(float(np.median(bdb[i0:i1]) - base), 1),
                         rms_bp_mv=round(float(xb.std()), 2),
                         bursts=len(on_g), bursts_per_s=round(len(on_g) / max((i1 - i0) * L / fs, 1e-9), 1),
                         burst_ms_med=round(float(np.median(dur)), 3) if len(dur) else None,
                         period_ms_med=round(float(np.median(per)), 3) if len(per) else None,
                         period_ms_iqr=round(float(np.subtract(*np.percentile(per, [75, 25]))), 3) if len(per) > 3 else None,
                         subbursts_per_burst=round(float(np.median([g[2] for g in on_g])), 1) if on_g else None,
                         burst_pp_mv_med=round(float(np.median(pp)), 1) if len(pp) else None,
                         carrier_khz_med=round(float(np.median(fc)) / 1e3, 2) if fc else None,
                         carrier_khz_p10=round(float(np.percentile(fc, 10)) / 1e3, 2) if fc else None,
                         carrier_khz_p90=round(float(np.percentile(fc, 90)) / 1e3, 2) if fc else None))
        if st:
            on_x.append(x)
            f0s += fc
    f0 = float(np.median(f0s)) if f0s else F0_GUESS
    harm = harmonics(np.concatenate(on_x), fs, f0) if on_x else {}
    on = [r for r in rows if r["state"] == "ON"]
    off = [r for r in rows if r["state"] == "OFF"]
    def med(rs, k):
        v = [r[k] for r in rs if r[k] is not None]
        return round(float(np.median(v)), 3) if v else None
    summ = dict(run=name, condition=cond, base_db=round(float(base), 1), env_floor_mv=round(env_floor, 3),
                burst_threshold_mv=round(thr, 2), on_segments=len(on), off_segments=len(off),
                on_band_db_over_base=med(on, "band_db_over_base"), off_band_db_over_base=med(off, "band_db_over_base"),
                on_bursts_per_s=med(on, "bursts_per_s"), off_bursts_per_s=med(off, "bursts_per_s"),
                on_period_ms=med(on, "period_ms_med"), on_burst_ms=med(on, "burst_ms_med"),
                on_subbursts=med(on, "subbursts_per_burst"), on_burst_pp_mv=med(on, "burst_pp_mv_med"),
                off_burst_pp_mv=med(off, "burst_pp_mv_med"), carrier_khz=round(f0 / 1e3, 2),
                on_segments_in_prompted_hold=med(on, "in_prompted_hold"), **{k: round(v, 1) for k, v in harm.items()})
    return summ, rows, dict(raw=raw, fs=fs, bdb=bdb, base=base, segs=segs, L=L, thr=thr, f0=f0)


def fig_main(name, cond, A, summ):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    raw, fs, L = A["raw"], A["fs"], A["L"]
    on = [s for s in A["segs"] if s[2]]
    off = [s for s in A["segs"] if not s[2] and s[1] - s[0] > 100]
    fig = plt.figure(figsize=(12, 11))
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1.1])
    C = "#1f5fbf"

    def seg(i_block, ms):
        a = i_block * L + 50 * L
        x = raw[a:a + int(ms * 1e-3 * fs)].astype(float) * MV
        return np.arange(len(x)) / fs * 1e3, x - x.mean()

    ax = fig.add_subplot(gs[0, :])
    t = np.arange(len(A["bdb"])) * L / fs
    ax.plot(t, A["bdb"] - A["base"], lw=0.6, color=C)
    for i0, i1, st in A["segs"]:
        if st:
            ax.axvspan(i0 * L / fs, i1 * L / fs, color="#2e8b57", alpha=0.15, lw=0)
    ax.axhline(6, color="#555", ls="--", lw=0.7)
    ax.set_ylabel("70.6 kHz band\nover baseline [dB]")
    ax.set_xlabel("time [s]")
    ax.set_title(f"{cond}: envelope over the run (green = receiver-defined ON segments, W1)", loc="left", fontsize=10)
    lim = None
    for j, (lab, s) in enumerate((("finger OFF", off[len(off) // 2] if off else None), ("finger ON", on[len(on) // 2] if on else None))):
        a = fig.add_subplot(gs[1, j])
        if s:
            tt, xx = seg(s[0], 28)
            a.plot(tt, xx, lw=0.4, color=C)
            lim = lim or max(abs(xx).max(), 1)
        a.set_title(f"{lab}: 28 ms raw A0 (mean removed)", loc="left", fontsize=9)
        a.set_xlabel("ms")
        a.set_ylabel("mV")
    for a in fig.axes[1:3]:
        a.set_ylim(-1.1 * lim, 1.1 * lim)
    # zoom on one burst of an ON segment
    a = fig.add_subplot(gs[2, 0])
    b2 = fig.add_subplot(gs[2, 1])
    if on:
        s = on[len(on) // 2]
        x = raw[s[0] * L + 50 * L: s[0] * L + 50 * L + 30000].astype(float) * MV
        x -= x.mean()
        xb = bp(x, 40e3, 450e3, fs)
        env, sub, grp = bursts(xb, fs, A["thr"])
        if grp:
            g0 = grp[min(1, len(grp) - 1)]
            i = max(0, g0[0] - 200)
            seg_x = x[i:i + 1500]
            a.plot(np.arange(len(seg_x)) / fs * 1e3, seg_x, lw=0.6, color=C)
            a.set_title("one burst, 1.5 ms", loc="left", fontsize=9)
            sx = x[g0[0] + 50:g0[0] + 150]
            b2.plot(np.arange(len(sx)) / fs * 1e6, sx, ".-", ms=3, lw=0.6, color=C)
            b2.set_title(f"100 us inside the burst (1 MS/s samples; carrier {summ['carrier_khz']:.2f} kHz)",
                         loc="left", fontsize=9)
        a.set_xlabel("ms")
        b2.set_xlabel("us")
    # spectra ON vs OFF with harmonic markers
    a = fig.add_subplot(gs[3, :])
    for segs_, lab, col in ((on, "ON segments", "#2e8b57"), (off, "OFF segments", "#999")):
        if not segs_:
            continue
        xs = np.concatenate([raw[i0 * L:min(i1 * L, i0 * L + 3_000_000)].astype(float) * MV for i0, i1, _ in segs_[:4]])
        f, P = welch(xs - xs.mean(), fs, nperseg=1 << 14)
        k = (f > 10e3) & (f < 495e3)
        a.plot(f[k] / 1e3, 10 * np.log10(P[k]), lw=0.6, color=col, label=lab)
    f0 = A["f0"]
    for n in (1, 3, 5, 7):
        a.axvline(n * f0 / 1e3, color="#c0602a", lw=0.6, ls=":")
    for n in (9, 11, 13):
        al = abs(n * f0 - round(n * f0 / fs) * fs) / 1e3
        a.axvline(al, color="#8a4fbf", lw=0.6, ls=":")
    a.set_xlabel("frequency [kHz]  (orange: f0 and odd harmonics; purple: 1 MS/s aliases of 9f0, 11f0, 13f0)")
    a.set_ylabel("PSD [dB mV^2/Hz]")
    a.legend(fontsize=8, frameon=False)
    for a in fig.axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Received maXTouch drive? waveform analysis - {name}", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / f"waveform_{name}.png", dpi=130)
    plt.close(fig)


def fig_compare(rows, summs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [s["run"] for s in summs]
    lab = [s["condition"] for s in summs]
    fig, ax = plt.subplots(1, 5, figsize=(17, 4.6), sharey=True)
    y = np.arange(len(names))
    spec = (("band_db_over_base", "70.6 kHz band over baseline [dB]", ("OFF", "ON"), None),
            ("rms_bp_mv", "RMS, 40-450 kHz [mV]", ("OFF", "ON"), None),
            ("bursts_per_s", "bursts per second (W2)", ("OFF", "ON"), None),
            ("period_ms_med", "burst period, ON only [ms]", ("ON",), (8, 11)),
            ("carrier_khz_med", "carrier, ON only [kHz]", ("ON",), (70, 71.2)))
    for j, (k, title, states, xl) in enumerate(spec):
        a = ax[j]
        for i, n in enumerate(names):
            for st, col, dy in (("OFF", "#999", -0.15), ("ON", "#2e8b57", 0.15)):
                if st not in states:
                    continue
                v = [r[k] for r in rows if r["run"] == n and r["state"] == st and r[k] is not None
                     and (st == "OFF" or r["dur_s"] >= 1.0)]
                a.plot(v, np.full(len(v), i + dy), "o", ms=4, color=col, alpha=0.8,
                       label=st if (i == 0 and j == 0) else None)
        if xl:
            a.set_xlim(*xl)
        a.set_title(title, fontsize=9)
        a.grid(axis="x", color="#eee")
        a.spines[["top", "right"]].set_visible(False)
    ax[0].set_yticks(y)
    ax[0].set_yticklabels(lab, fontsize=8)
    ax[0].invert_yaxis()
    ax[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Every segment of every run (receiver-defined W1 segments; each dot = one ON or OFF stretch; ON stretches < 1 s left out)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "compare_runs.png", dpi=130)
    plt.close(fig)


def scope(dirp):
    """the same W2-W4 measurements on Teledyne T3DSO CSVs (CH1)"""
    out = []
    for p in sorted(Path(dirp).glob("screen*.csv")):
        lines = open(p).read().splitlines()
        dt = float([l for l in lines if l.startswith("Sample Interval")][0].split(":")[1].split(",")[0])
        i = next(j for j, l in enumerate(lines) if l.startswith("Second"))
        v = np.array([float(l.split(",")[1]) for l in lines[i + 1:] if l.strip()]) * 1e3
        fs = 1 / dt
        x = v - v.mean()
        xb = bp(x, 40e3, min(450e3, 0.45 * fs), fs)
        floor = np.percentile(movrms(xb, max(2, int(10e-6 * fs))), 20)
        env, sub, grp = bursts(xb, fs, 5 * floor)
        xf = bp(x, 55e3, 130e3, fs)
        fc = [carrier(xf, fs, a, b) for a, b in sub]
        fc = [c for c in fc if np.isfinite(c)]
        per = np.diff([g[0] for g in grp]) / fs * 1e3
        out.append(dict(file=p.name, fs_hz=fs, duration_ms=len(x) / fs * 1e3, bursts=len(grp),
                        burst_ms=[round((g[1] - g[0]) / fs * 1e3, 3) for g in grp],
                        subbursts=[g[2] for g in grp], period_ms=[round(v, 3) for v in per],
                        carrier_khz_med=round(float(np.median(fc)) / 1e3, 2) if fc else None,
                        carrier_khz_each=[round(c / 1e3, 1) for c in fc][:20],
                        pp_mv=round(float(np.ptp(xb)), 1)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", help="run folder names (default: the 2026-11-18 runs)")
    ap.add_argument("--scope", help="folder with scope CSVs screen1.csv / screen2.csv")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    names = a.runs or list(RUNS)
    summs, rows = [], []
    for n in names:
        print(f"... {n}", flush=True)
        s, r, A = analyse_run(n, RUNS.get(n, n))
        summs.append(s)
        rows += r
        if n == MAIN:
            fig_main(n, RUNS[n], A, s)
        del A
    for fn, data in (("runs.csv", summs), ("cycles.csv", rows)):
        keys = list(dict.fromkeys(k for d in data for k in d))
        with open(OUT / fn, "w", newline="") as fh:
            w = csv.DictWriter(fh, keys)
            w.writeheader()
            w.writerows(data)
    fig_compare(rows, summs)
    if a.scope:
        sc = scope(a.scope)
        json.dump(sc, open(OUT / "scope.json", "w"), indent=1)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
