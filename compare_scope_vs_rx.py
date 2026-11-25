#!/usr/bin/env python3
"""
Siqi's oscilloscope recordings of a touchscreen drive (oscilloscope.zip: screen1/2.csv, SDS00007-09/14-17.csv,
Teledyne T3DSO1302A, Jul-Aug 2026, screen device not recorded in the files) vs the signal our stent receiver gets
through the body from the maXTouch EVK (TeensyTRX A0, 1 MS/s, data_touchdrive/td_*), 2026-11-18.

The two were NOT recorded on the same touchscreen, and they are measured at different points:
  scope  = a pickup at the screen (close to the drive, before any body path)
  ours   = after finger -> body -> wrist electrodes -> Ra/Rb + DC block -> Teensy ADC (1 MS/s, no known anti-alias filter)

   python compare_scope_vs_rx.py --scope <folder with the scope CSVs>
Output data_touchdrive/waveform/scope_vs_rx/: scope_vs_rx.png/.pdf (figure to share), metrics.json.

Measurements (the same burst rules as touch_drive_waveform.py W2):
  frame     bursts in the 28 ms captures (scope 500 kS/s / ours 1 MS/s): 10 us moving RMS of the 40-450 kHz band
            over 5 x its 20th percentile (scope) / 5 x the OFF-segment floor (ours); sub-bursts < 1 ms apart = burst
  pulses    scope SDS00014-17 (10-200 MS/s): two-level digitising (25 us running max/min mid-level, 25 % hysteresis)
            -> HIGH/LOW widths; ours: in-burst period from rising zero crossings of the 55-90 kHz band (W3)
  spectrum  inside bursts, Welch, both normalised to their own peak
  channel   illustrative only: the scope burst (SDS00014, 10 MS/s) through a 1st-order high-pass (fc 200 kHz, the
            differentiating coupling suggested by our harmonics) and then every 10th sample (1 MS/s, no filter, like
            the Teensy ADC) - can our receive path turn her waveform into ours?
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from scipy.signal import butter, find_peaks, lfilter, sosfiltfilt, welch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import touch_drive_waveform as W                                    # noqa: E402  (bp, movrms, bursts, carrier)
import touch_drive_rx as T                                          # noqa: E402

OUT = HERE / "data_touchdrive" / "waveform" / "scope_vs_rx"
RUN = "td_touch2_20261118_171250"            # screen touch, run C (wrist); finger-on stretch at T_ON
T_ON = 80.0
MV = 3300 / 4096
C_S, C_R, C_M = "#c0602a", "#1f5fbf", "#8a4fbf"


def load_scope(path, ch=1):
    L = open(path).read().splitlines()
    dt = float([l for l in L if l.startswith("Sample Interval")][0].split(":")[1].split(",")[0])
    i = next(j for j, l in enumerate(L) if l.startswith("Second"))
    a = np.array([[float(v) for v in l.split(",")[:ch + 1]] for l in L[i + 1:] if l.strip()])
    return a[:, ch] * 1e3, 1 / dt


def digitise(x, fs, win=25e-6, frac=0.25):
    n = int(win * fs) | 1
    mx, mn = maximum_filter1d(x, n), minimum_filter1d(x, n)
    mid, h = (mx + mn) / 2, frac * (mx - mn) / 2
    st, out = x[0] > mid[0], np.empty(len(x), bool)
    for i in range(len(x)):
        if st and x[i] < mid[i] - h[i]:
            st = False
        elif not st and x[i] > mid[i] + h[i]:
            st = True
        out[i] = st
    return out


def pulse_widths(x, fs):
    hi = digitise(x, fs)
    ed = np.flatnonzero(np.diff(hi.astype(int)) != 0)
    lv = hi[ed + 1]
    w = np.diff(ed) / fs * 1e6
    return w, lv[:-1], int(np.sum(np.diff(hi.astype(int)) == 1)) / (len(x) / fs)


def frame_bursts(x, fs, thr=None):
    x = x - x.mean()
    xb = W.bp(x, 40e3, min(450e3, 0.45 * fs), fs)
    if thr is None:
        thr = 5 * np.percentile(W.movrms(xb, max(2, int(10e-6 * fs))), 20)
    _, sub, grp = W.bursts(xb, fs, thr)
    return xb, sub, grp


def spec(x, fs, nfft=1 << 18):
    """10 kHz high-pass (removes the scope's baseline droop), Hann, zero-padded periodogram, dB re its own peak"""
    x = sosfiltfilt(butter(2, 10e3, btype="high", fs=fs, output="sos"), x - x.mean())
    P = np.abs(np.fft.rfft(x * np.hanning(len(x)), nfft)) ** 2
    f = np.fft.rfftfreq(nfft, 1 / fs)
    k = (f > 10e3) & (f < min(490e3, 0.49 * fs))
    return f, 10 * np.log10(P / P[k].max() + 1e-30)


def top_lines(f, P, n=8, lo=10e3, hi=490e3):
    k = np.flatnonzero((f > lo) & (f < hi))
    pk, _ = find_peaks(P[k], distance=max(1, int(12e3 / (f[1] - f[0]))))
    pk = sorted(pk, key=lambda i: -P[k][i])[:n]
    return [(round(float(f[k][i]) / 1e3, 1), round(float(P[k][i]), 1)) for i in sorted(pk)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scope", required=True)
    a = ap.parse_args()
    S = Path(a.scope)
    OUT.mkdir(parents=True, exist_ok=True)
    M = dict(scope={}, ours={})

    # ---------------- scope: frames
    frames = {}
    for name in ("screen1.csv", "SDS00007.csv", "SDS00008.csv", "SDS00009.csv", "screen2.csv"):
        x, fs = load_scope(S / name)
        _, sub, _ = frame_bursts(x, fs)
        d = [(b - a_) / fs * 1e3 for a_, b in sub]
        on = [a_ / fs * 1e3 for a_, _ in sub]
        long_on = [o for o, dd in zip(on, d) if dd > 0.3]
        frames[name] = dict(fs=fs, subburst_ms=[round(v, 3) for v in d], onsets_ms=[round(v, 2) for v in on],
                            long_burst_period_ms=[round(v, 3) for v in np.diff(long_on)], pp_mv=round(float(np.ptp(x)), 1))
    M["scope"]["frames"] = frames
    # ---------------- scope: pulses
    pul = {}
    for name, t0, t1 in (("SDS00014.csv", 152e-6, 626e-6), ("SDS00015.csv", 0, 1), ("SDS00016.csv", 0, 1),
                         ("SDS00017.csv", 0, 1)):
        x, fs = load_scope(S / name)
        t = np.arange(len(x)) / fs
        k = (t >= t0) & (t <= t1)
        w, lv, rate = pulse_widths(x[k], fs)
        q = np.round(w)
        pul[name] = dict(fs=fs, rising_edges_per_s=round(rate), widths_us={int(v): int(c) for v, c in zip(*np.unique(q, return_counts=True))},
                         sequence=" ".join(("H" if l else "L") + f"{v:.0f}" for v, l in zip(w[:28], lv[:28])))
    M["scope"]["pulses"] = pul
    xs15, fs15 = load_scope(S / "SDS00015.csv")              # 280 us, all inside one burst, 26 us code
    fS, PS = spec(xs15, fs15)
    M["scope"]["lines_SDS00015_khz_db"] = top_lines(fS, PS)
    xs14, fs14 = load_scope(S / "SDS00014.csv")
    t14 = np.arange(len(xs14)) / fs14
    f14, P14 = spec(xs14[(t14 > 152e-6) & (t14 < 626e-6)], fs14)
    M["scope"]["lines_SDS00014_khz_db"] = top_lines(f14, P14)
    x1, f1 = load_scope(S / "screen1.csv")
    fa, Pa = spec(x1, f1)
    M["scope"]["screen1_strongest_khz_aliased"] = top_lines(fa, Pa, 1, 10e3, 245e3)[0][0]
    # channel model: AC coupling (1st-order high-pass) then every 50th sample = 1 MS/s, no anti-alias filter
    sims = {}
    for fc in (20e3, 200e3):
        bh, ah = butter(1, fc, btype="high", fs=fs15)
        fM_, PM_ = spec(lfilter(bh, ah, xs15)[::int(round(fs15 / 1e6))], 1e6, 1 << 16)
        sims[fc] = (fM_, PM_)
        M["scope"][f"through_our_path_fc{int(fc / 1e3)}k_lines"] = top_lines(fM_, PM_, 6)
        M["scope"][f"through_our_path_fc{int(fc / 1e3)}k_db_at_70.6k"] = round(float(PM_[np.argmin(abs(fM_ - 70.6e3))]), 1)
    fM, PM = sims[20e3]

    # ---------------- ours
    d = HERE / "data_touchdrive" / RUN
    raw = np.load(d / "rx.npy", mmap_mode="r")
    fs = 1e6
    seg = np.asarray(raw[int(T_ON * fs):int(T_ON * fs) + 2_000_000], float) * MV
    seg -= seg.mean()
    off = np.asarray(raw[int(40.0 * fs):int(40.0 * fs) + 1_000_000], float) * MV      # finger-off stretch (rest 3)
    off -= off.mean()
    runs_ = {r["run"]: r for r in __import__("csv").DictReader(open(HERE / "data_touchdrive" / "waveform" / "runs.csv"))}
    thr = float(runs_[RUN]["burst_threshold_mv"])         # the same W2 threshold as touch_drive_waveform.py
    xb, sub, grp = frame_bursts(seg, fs, thr)
    xf = W.bp(seg, 55e3, 90e3, fs)
    fc = [W.carrier(xf, fs, s0, s1) for s0, s1 in sub]
    fc = np.array([v for v in fc if np.isfinite(v)])
    per = np.diff([g[0] for g in grp]) / fs * 1e3
    runs = {r["run"]: r for r in __import__("csv").DictReader(open(HERE / "data_touchdrive" / "waveform" / "runs.csv"))}
    M["ours"] = dict(run=RUN, carrier_khz_median=round(float(np.median(fc)) / 1e3, 2),
                     carrier_khz_p10_p90=[round(float(np.percentile(fc, 10)) / 1e3, 2), round(float(np.percentile(fc, 90)) / 1e3, 2)],
                     burst_ms_median=round(float(np.median([(g[1] - g[0]) / fs * 1e3 for g in grp])), 3),
                     subbursts_median=float(np.median([g[2] for g in grp])),
                     period_ms_median=round(float(np.median(per)), 3), period_ms_iqr=round(float(np.subtract(*np.percentile(per, [75, 25]))), 3),
                     burst_pp_mv_median=round(float(np.median([np.ptp(xb[g[0]:g[1]]) for g in grp])), 1),
                     period_ms_other_runs={v["condition"]: v["on_period_ms"] for v in runs.values() if v["on_period_ms"]})
    fR, PR = spec(np.concatenate([seg[g[0]:g[1]] for g in grp]), fs, 1 << 16)
    # for the figure: Welch-averaged (1 kHz bins) - a single zero-padded periodogram of ~200 bursts is too noisy to read
    fRw, PRw = welch(sosfiltfilt(butter(2, 10e3, btype="high", fs=fs, output="sos"),
                                 np.concatenate([seg[g[0]:g[1]] for g in grp])), fs, nperseg=1000)
    kw = (fRw > 10e3) & (fRw < 490e3)
    PRw = 10 * np.log10(PRw / PRw[kw].max())
    M["ours"]["lines_khz_db"] = top_lines(fR, PR, 6)
    json.dump(M, open(OUT / "metrics.json", "w"), indent=1)

    # ---------------- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(13, 15.5))
    gs = fig.add_gridspec(5, 2, height_ratios=[1, 1, 1, 1.1, 0.95], hspace=0.75, wspace=0.18)

    def style(ax_, title, xl, yl="mV"):
        ax_.set_title(title, loc="left", fontsize=9.5)
        ax_.set_xlabel(xl, fontsize=9)
        ax_.set_ylabel(yl, fontsize=9)
        ax_.spines[["top", "right"]].set_visible(False)
        ax_.grid(color="#eee", lw=0.6)
        ax_.tick_params(labelsize=8)

    # row 1: 28 ms
    x1, f1 = load_scope(S / "screen1.csv")
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(np.arange(len(x1)) / f1 * 1e3, x1 - np.median(x1), lw=0.5, color=C_S)
    style(ax, "SIQI - scope at the screen (screen1.csv, 500 kS/s): 28 ms\n"
              "long burst every 8.33 ms + short bursts; same frame in all 5 captures", "time [ms]")
    i0 = grp[0][0] - 3000 if grp and grp[0][0] > 3000 else 0
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(np.arange(28000) / fs * 1e3, seg[i0:i0 + 28000], lw=0.4, color=C_R)
    style(ax, f"OURS - Teensy A0 through the body (wrist, finger ON, 1 MS/s): 28 ms\n"
              f"one ~{M['ours']['burst_ms_median']:.1f} ms burst every {M['ours']['period_ms_median']:.2f} ms (this run)", "time [ms]")
    # row 2: one burst, 1.5 ms
    x2, f2 = load_scope(S / "screen2.csv")
    _, sub2, _ = frame_bursts(x2, f2)
    lb = next((s for s in sub2 if (s[1] - s[0]) / f2 > 0.3e-3), sub2[0])
    j0 = max(0, lb[0] - int(0.2e-3 * f2))
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(np.arange(int(1.5e-3 * f2)) / f2 * 1e3, x2[j0:j0 + int(1.5e-3 * f2)] - np.median(x2), lw=0.5, color=C_S)
    style(ax, "SIQI - one long burst (screen2.csv, 2 MS/s): 1.5 ms\none continuous ~0.52 ms burst", "time [ms]")
    g = grp[min(1, len(grp) - 1)]
    k0 = max(0, g[0] - 200)
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(np.arange(1500) / fs * 1e3, seg[k0:k0 + 1500], lw=0.5, color=C_R)
    style(ax, f"OURS - one burst: 1.5 ms\n~{M['ours']['burst_ms_median']:.2f} ms made of short sub-bursts "
              f"(median {M['ours']['subbursts_median']:.0f})", "time [ms]")
    # row 3: 70 us
    x17, f17 = load_scope(S / "SDS00017.csv")
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(np.arange(len(x17)) / f17 * 1e6, x17, lw=0.6, color=C_S)
    style(ax, "SIQI - inside a burst (SDS00017.csv, 200 MS/s): 70 us\n"
              "coded pulses: 1 us chips, H1 L1 H1 L1 H2 L2 H5 (13 chips), then inverted", "time [us]")
    ax = fig.add_subplot(gs[2, 1])
    k1 = g[0] + 100
    ax.plot(np.arange(70), seg[k1:k1 + 70], ".-", ms=3, lw=0.6, color=C_R)
    for m in np.arange(0, 70, 1e3 / M["ours"]["carrier_khz_median"]):
        ax.axvline(m, color="#bbb", lw=0.5, ls=":")
    style(ax, f"OURS - inside a burst: 70 us (dots = 1 MS/s samples)\n"
              f"regular period {1e3 / M['ours']['carrier_khz_median']:.2f} us (dotted); edges show as spikes", "time [us]")
    # row 4: spectra
    ax = fig.add_subplot(gs[3, :])
    k = (fS > 10e3) & (fS < 490e3)
    ax.plot(fS[k] / 1e3, PS[k], color=C_S, lw=0.9, label="SIQI at the screen (SDS00015, 50 MS/s): comb of lines every 38.5 kHz, no single carrier")
    k = (fM > 10e3) & (fM < 490e3)
    ax.plot(fM[k] / 1e3, PM[k], color=C_M, lw=0.9, ls="--",
            label="SIQI through a model of OUR path (AC coupling + 1 MS/s sampling): still on the 38.5 kHz grid")
    ax.plot(fRw[kw] / 1e3, PRw[kw], color=C_R, lw=1.3, label=f"OURS inside bursts (Teensy A0, 1 MS/s, Welch 1 kHz bins): {M['ours']['carrier_khz_median']:.1f} kHz + odd harmonics")
    for n in (1, 3, 5, 7):
        ax.axvline(n * M["ours"]["carrier_khz_median"], color=C_R, lw=0.6, ls=":")
    for n in range(1, 13):
        ax.plot([n * 38.4615] * 2, [-44, -40], color=C_S, lw=1.2)
    ax.set_ylim(-45, 3)
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    style(ax, "Spectra inside the bursts, each normalised to its own peak  (orange ticks: multiples of 38.46 kHz; blue dotted: 70.6 kHz and odd harmonics)",
          "frequency [kHz]", "dB re peak")
    # row 5: table
    ax = fig.add_subplot(gs[4, :])
    ax.axis("off")
    sp = pul["SDS00016.csv"]
    rows = [
        ["Measured at", "pickup at the screen (scope T3DSO1302A)", "Teensy A0, after finger + body + wrist electrodes"],
        ["Device", "not recorded in the files (Jul-Aug 2026)", "maXTouch EVK, ATMXT2952TD"],
        ["Waveform inside a burst", "coded pulses, 1 us chips (H1 L1 H1 L1 H2 L2 H5),\nsign alternates every 13 us (26 us repeat)",
         f"regular square-wave period {1e3 / M['ours']['carrier_khz_median']:.2f} us\n(edges appear as spikes: capacitive path)"],
        ["Spectrum", "no single carrier: line comb every 38.5 kHz (26 us code), lines\nwithin ~4 dB; '95-97 kHz' is an alias of the 0.5/2 MS/s captures",
         f"{M['ours']['carrier_khz_median']:.2f} kHz line (p10-p90 {M['ours']['carrier_khz_p10_p90'][0]:.2f}-{M['ours']['carrier_khz_p10_p90'][1]:.2f}) + odd harmonics"],
        ["Burst length", "long 0.52-0.54 ms; short 0.05-0.21 ms", f"{M['ours']['burst_ms_median']:.2f} ms here (1.0-1.09 ms in all runs), 3-8 sub-bursts"],
        ["Repetition (frame)", "8.33 ms (120 Hz) in all 5 captures", "8.64 / 9.10 / 9.57 ms (constant within a session)"],
        ["Amplitude", "about 90-100 mV p-p at the pickup", f"~{M['ours']['burst_pp_mv_median']:.0f} mV p-p, only while the finger touches"],
    ]
    tb = ax.table(cellText=rows, colLabels=["", "SIQI - scope at the screen", "OURS - through the body"],
                  cellLoc="left", colLoc="left", loc="upper center", colWidths=[0.16, 0.42, 0.42])
    tb.auto_set_font_size(False)
    tb.set_fontsize(8)
    tb.scale(1, 1.9)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor("#ddd")
        if r == 0:
            cell.set_text_props(weight="bold", color=[None, C_S, C_R][c] or "#222")
    fig.suptitle("Touchscreen drive: Siqi's scope recordings vs the maXTouch drive received through the body\n"
                 "(different screens and different measurement points - not the same signal path)", fontsize=12)
    for e in ("png", "pdf"):
        fig.savefig(OUT / f"scope_vs_rx.{e}", dpi=140, bbox_inches="tight")
    print(json.dumps(M, indent=1))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
