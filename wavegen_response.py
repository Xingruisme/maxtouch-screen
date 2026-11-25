#!/usr/bin/env python3
"""
Response of the maXTouch screen to a wave generator at different amplitudes.

The generator is set by hand; this program only records the screen. It asks for one amplitude at a time, records
--seconds of scans at each, and saves the raw scans, a table and a figure of response against amplitude.

   python wavegen_response.py run --label agar --freq 168e3 --amps 0.05,0.1,0.2,0.5,1,2,5
   python wavegen_response.py run --label agar --freq 168e3            # no list: type each amplitude at the prompt
   python wavegen_response.py analyse data_maxtouch/wavegen_agar_168kHz_<time>
   python wavegen_response.py summary                                  # one figure over all runs (or the folders given)
   python wavegen_response.py run --simulate --amps 0.1,0.3,1,3        # no hardware: synthetic panel, to try the program

Order of a run: generator output OFF -> Enter (the chip is calibrated and the off level recorded) -> for every amplitude:
set it on the generator, output ON, Enter -> at the end output OFF again, Enter (second off level = drift check).
At each prompt:  Enter = record the amplitude shown | a number = record that amplitude instead | off = record the off
level | r = throw away the last recording and do it again | q = finish.  Ctrl+C also finishes; what was recorded is kept.

What is recorded: the same nodes as the BER tools (X line 0, Y --y0 .. --y0 + --ny - 1, T37 mutual deltas, every sample a
fresh scan, ~130 scans/s with <= 28 nodes), with the same chip set-up as maxtouch_stent_rx (noise suppression and touch
reports off while recording, restored at the end).

Measure: the carrier phase differs from scan to scan, so a carrier shows up as scatter of a node's delta, not as a
steady level. Per node and amplitude:  rms = standard deviation of the delta over the step [counts],  mean = its mean,
peak = 99th percentile of |delta - median|.  Headline = rms over the --top nodes that respond most at the largest
amplitude; 'net' = the off level removed in power, sqrt(rms^2 - off^2).  slope = fit of log(net) against log(amplitude)
over the steps at least 6 dB over the off level: 1.0 means the screen follows the amplitude linearly.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import glob
import json
import math
import os
import signal
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DATADIR = os.path.join(HERE, "data_maxtouch")
MIN_OVER_OFF = 2.0                              # a step counts for the slope from 6 dB over the off level


# ================================================================ synthetic panel (--simulate)
class FakeChip:
    """stands in for maxtouch_stent_rx.Chip: carrier with a random phase per scan on a few nodes, compressing at high
    drive, plus noise. Only for trying the program without the screen."""

    def __init__(self, a):
        self.y0, self.ny, self.patch = a.y0, a.ny, None
        self.names = [f"Y{a.y0 + i}" for i in range(a.ny)]
        self.rng = np.random.default_rng(1)
        self.profile = np.exp(-0.5 * ((np.arange(a.ny) - 3.0) / 1.5) ** 2)
        self.drive = 0.0

    def setup(self):
        return {}

    def restore(self):
        pass

    def sample(self):
        time.sleep(0.0015)
        amp = 300.0 * self.drive / (1 + self.drive / 4.0)
        v = amp * self.profile * math.cos(self.rng.uniform(0, 2 * math.pi)) + self.rng.normal(0, 1.5, self.ny)
        return np.round(v).astype(np.int16)


# ================================================================ run
def fmt(v):
    return f"{v:g}"


def record_step(chip, seconds):
    t, x, errors = [], [], 0
    end = time.time() + seconds
    while time.time() < end:
        try:
            v = chip.sample()
        except Exception as e:                  # a dropped HID transfer: count it and carry on
            errors += 1
            if errors > 50:
                raise RuntimeError(f"maXTouch link lost ({e})")
            continue
        t.append(time.time())
        x.append(v)
    return np.array(t), np.array(x, np.int16).reshape(len(t), -1), errors


def run(a):
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))       # kill -> the finally below still restores the chip
    planned = [None] + [float(v) for v in a.amps.split(",") if v.strip()] + [None] if a.amps else [None]
    queue = list(planned)                       # None = generator output off
    u = a.unit
    print(f"Wave generator response, {a.freq / 1e3:g} kHz {a.wave}, {a.seconds:g} s per amplitude"
          + (f", planned: off, {', '.join(fmt(v) for v in planned[1:-1])} {u}, off" if a.amps else
             f", amplitudes typed at the prompt [{u}]"))
    print("  at a prompt: Enter = record what is shown | number = that amplitude instead | off | r = redo last | q = finish")
    input("\nGenerator output OFF now (the chip is calibrated on it) - press Enter ")
    chip, changed = None, {}
    steps, T, X, S = [], [], [], []
    off_rms = None
    try:
        if a.simulate:
            chip = FakeChip(a)
        else:
            import maxtouch_stent_rx as R
            chip = R.Chip(a)
        changed = chip.setup()
        print(f"Chip set up and calibrated (original CTRL bytes {changed}, restored at the end); "
              f"recording {chip.ny} nodes {chip.names[0]}-{chip.names[-1]}")
        first = True
        while True:
            nxt = queue[0] if queue else "none"
            if first:
                ans, first = "", False          # the off level right after the calibration, no second prompt
            else:
                what = ("type an amplitude (or off), Enter = finish" if nxt == "none" else
                        "next: generator output OFF" if nxt is None else f"next: {fmt(nxt)} {u}, output ON")
                ans = input(f"[{len([s for s in steps if not s['discarded']]) + 1}] {what} > ").strip().lower()
            if ans in ("q", "quit") or (ans == "" and nxt == "none"):
                break
            if ans == "r":
                last = next((s for s in reversed(steps) if not s["discarded"]), None)
                if last is None:
                    print("    nothing recorded yet"); continue
                last["discarded"] = True
                queue.insert(0, last["amp"])
                print(f"    last recording ({'off' if last['amp'] is None else fmt(last['amp']) + ' ' + u}) thrown away")
                continue
            if ans == "":
                amp = queue.pop(0)
            elif ans == "off":
                amp = None
                if queue and queue[0] is None:
                    queue.pop(0)
            else:
                try:
                    amp = float(ans)
                except ValueError:
                    print("    not understood: Enter, a number, off, r or q"); continue
                if amp <= 0:
                    print("    the amplitude must be above 0 (type off for the off level)"); continue
                if queue and queue[0] == amp:
                    queue.pop(0)
            if a.simulate:
                chip.drive = amp or 0.0
            time.sleep(a.settle)
            t, x, errors = record_step(chip, a.seconds)
            if len(t) < 20:
                print(f"    only {len(t)} samples - not kept, try again"); queue.insert(0, amp); continue
            k = len(steps)
            steps.append(dict(k=k, amp=amp, t0=float(t[0]), t1=float(t[-1]), n=int(len(t)), hid_errors=errors,
                              discarded=False))
            T.append(t); X.append(x); S.append(np.full(len(t), k))
            rms = x.astype(float).std(axis=0)
            j = int(np.argmax(rms))
            rate = len(t) / max(t[-1] - t[0], 1e-9)
            if amp is None:
                off_rms = rms if off_rms is None else np.sqrt((off_rms ** 2 + rms ** 2) / 2)
                print(f"    off: {len(t)} samples ({rate:.0f}/s)   rms median node {np.median(rms):.2f}, "
                      f"highest {chip.names[j]} {rms[j]:.2f} counts")
            else:
                over = 20 * math.log10(max(rms[j], 1e-9) / max(off_rms[j], 1e-9))
                print(f"    {fmt(amp)} {u}: {len(t)} samples ({rate:.0f}/s)   strongest node {chip.names[j]} rms {rms[j]:.2f} "
                      f"counts ({over:+.1f} dB re its off level {off_rms[j]:.2f})")
    except (KeyboardInterrupt, EOFError):
        print("\n  stopped - saving what was recorded")
    finally:
        if chip:
            chip.restore()
    kept = [s for s in steps if not s["discarded"]]
    if not kept:
        print("nothing recorded"); return
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    d = os.path.join(a.datadir or DATADIR, f"wavegen_{a.label}_{a.freq / 1e3:g}kHz_{stamp}")
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, "capture.npz"), t=np.concatenate(T), X=np.concatenate(X), step=np.concatenate(S))
    json.dump(dict(mode="wavegen_response", created=stamp, freq_hz=a.freq, wave=a.wave, unit=u, seconds=a.seconds,
                   condition=a.condition, note=a.note, simulated=bool(a.simulate), names=chip.names, y0=chip.y0, ny=chip.ny,
                   patch=chip.patch, chip_ctrl_orig={k: int(v) for k, v in changed.items()}, steps=steps, args=vars(a)),
              open(os.path.join(d, "meta.json"), "w"), indent=1)
    analyse(d, a.top)


# ================================================================ analysis
def analyse(d, top=None):
    meta = json.load(open(os.path.join(d, "meta.json")))
    z = np.load(os.path.join(d, "capture.npz"))
    X, step = z["X"].astype(float), z["step"]
    names, u = meta["names"], meta["unit"]
    top = top or meta["args"].get("top") or 4
    rows = []
    for s in meta["steps"]:
        if s["discarded"]:
            continue
        x = X[step == s["k"]]
        rows.append(dict(k=s["k"], amp=s["amp"], n=len(x), rate=len(x) / max(s["t1"] - s["t0"], 1e-9),
                         mean=x.mean(axis=0), rms=x.std(axis=0),
                         peak=np.percentile(np.abs(x - np.median(x, axis=0)), 99, axis=0)))
    off = [r for r in rows if r["amp"] is None]
    on = sorted((r for r in rows if r["amp"] is not None), key=lambda r: (r["amp"], r["k"]))
    if not off:
        print("  ! no off level recorded: the off level is taken as 0, so 'net' = rms")
    off_rms = np.sqrt(np.mean([r["rms"] ** 2 for r in off], axis=0)) if off else np.zeros(len(names))
    off_mean = np.mean([r["mean"] for r in off], axis=0) if off else np.zeros(len(names))
    ref = on[-1] if on else None                # largest amplitude: its strongest nodes are the ones reported
    nodes = list(np.argsort(-ref["rms"])[:top]) if ref else list(np.argsort(-off_rms)[:top])
    head = lambda v: float(np.sqrt(np.mean(np.asarray(v)[nodes] ** 2)))
    F = head(off_rms)
    for r in on:
        r["R"] = head(r["rms"])
        r["net"] = math.sqrt(max(r["R"] ** 2 - F ** 2, 0.0))
        r["shift"] = float(np.mean((r["mean"] - off_mean)[nodes]))
    good = [r for r in on if F == 0 or r["R"] >= MIN_OVER_OFF * F]
    slope = icpt = None
    if len({r["amp"] for r in good}) >= 2:
        slope, icpt = np.polyfit(np.log10([r["amp"] for r in good]), np.log10([r["net"] for r in good]), 1)

    print(f"\n{meta['freq_hz'] / 1e3:g} kHz {meta['wave']}" + (f", {meta['condition']}" if meta.get("condition") else "")
          + ("   ** SIMULATED, not a measurement **" if meta.get("simulated") else ""))
    print(f"Nodes reported (strongest at the largest amplitude): {', '.join(names[i] for i in nodes)}")
    print(f"Off level: {F:.2f} counts rms on these nodes ({len(off)} recording{'s' if len(off) != 1 else ''})")
    if len(off) >= 2:
        f0, f1 = head(off[0]["rms"]), head(off[-1]["rms"])
        dm = float(np.mean((off[-1]["mean"] - off[0]["mean"])[nodes]))
        print(f"  first / last off recording: {f0:.2f} / {f1:.2f} counts rms, mean moved {dm:+.2f} counts"
              + ("   ** the off level changed during the run **" if max(f0, f1) > 1.5 * max(min(f0, f1), 0.3) else ""))
    if on:
        a_max, n_max = on[-1]["amp"], on[-1]["net"]
        print(f"\n{'amplitude':>12} {'rms':>8} {'net':>8} {'re off':>8} {'peak':>8} {'mean':>8} "
              f"{'drive':>9} {'response':>9}  strongest node")
        print(f"{'[' + u + ']':>12} {'[counts]':>8} {'[counts]':>8} {'[dB]':>8} {'[counts]':>8} {'shift':>8} "
              f"{'re max':>9} {'re max':>9}")
        for r in on:
            j = int(np.argmax(r["rms"]))
            over = f"{20 * math.log10(r['R'] / F):+8.1f}" if F > 0 and r["R"] > 0 else f"{'-':>8}"
            resp = f"{20 * math.log10(r['net'] / n_max):+7.1f} dB" if r["net"] > 0 and n_max > 0 else f"{'at off':>10}"
            print(f"{fmt(r['amp']):>12} {r['R']:8.2f} {r['net']:8.2f} {over} {head(r['peak']):8.1f} {r['shift']:+8.2f} "
                  f"{20 * math.log10(r['amp'] / a_max):+6.1f} dB {resp}  {names[j]} {r['rms'][j]:.2f}")
        if slope is not None:
            print(f"\nSlope of log(net) against log(amplitude), {len(good)} steps at least 6 dB over the off level: "
                  f"{slope:.2f}  (1.00 = linear)")
        else:
            print(f"\nNo slope: fewer than 2 amplitudes at least 6 dB over the off level ({len(good)} found).")

    with open(os.path.join(d, "response.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", f"amplitude_{u}", "samples", "samples_per_s", "rms_counts", "net_counts", "db_over_off",
                    "peak_counts", "mean_shift_counts"])
        for r in sorted(rows, key=lambda r: r["k"]):
            if r["amp"] is None:
                w.writerow([r["k"], "off", r["n"], f"{r['rate']:.1f}", f"{head(r['rms']):.3f}", "", "",
                            f"{head(r['peak']):.2f}", ""])
            else:
                w.writerow([r["k"], fmt(r["amp"]), r["n"], f"{r['rate']:.1f}", f"{r['R']:.3f}", f"{r['net']:.3f}",
                            f"{20 * math.log10(r['R'] / F):.2f}" if F > 0 and r["R"] > 0 else "", f"{head(r['peak']):.2f}",
                            f"{r['shift']:.3f}"])
    with open(os.path.join(d, "nodes.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", f"amplitude_{u}", "quantity"] + names)
        for r in sorted(rows, key=lambda r: r["k"]):
            for q in ("rms", "mean", "peak"):
                w.writerow([r["k"], "off" if r["amp"] is None else fmt(r["amp"]), q] + [f"{v:.3f}" for v in r[q]])
    plot(d, meta, X, step, rows, on, names, nodes, F, off_rms, slope, icpt)
    print(f"Saved {os.path.relpath(d, HERE)}  (capture.npz, meta.json, response.csv, nodes.csv, result.png)")


def plot(d, meta, X, step, rows, on, names, nodes, F, off_rms, slope, icpt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    blue, ink, ink2, grid, band = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df", "#f0efec"
    u = meta["unit"]
    fig = plt.figure(figsize=(11, 8.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.25])
    ax_t, ax_r, ax_n = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    # (1) what was recorded: the strongest node, the steps one after the other (the waits at the prompt are not recorded)
    j, x0 = nodes[0], 0
    seq = sorted(rows, key=lambda r: r["k"])
    for i, r in enumerate(seq):
        x = X[step == r["k"], j]
        if i % 2:
            ax_t.axvspan(x0, x0 + len(x), color=band, zorder=0)
        ax_t.plot(np.arange(x0, x0 + len(x)), x, lw=0.6, color=blue)
        ax_t.text(x0 + len(x) / 2, 1.01, "off" if r["amp"] is None else fmt(r["amp"]), transform=ax_t.get_xaxis_transform(),
                  ha="center", va="bottom", fontsize=8, color=ink2, rotation=90 if len(seq) > 14 else 0)
        x0 += len(x)
    ax_t.set_xlim(0, x0)
    ax_t.set_xlabel("scan (recordings one after the other)", color=ink2)
    ax_t.set_ylabel("delta [counts]", color=ink2)
    ax_t.set_title(f"Node {names[j]}, every scan; amplitude [{u}] above each recording", fontsize=10, loc="left",
                   color=ink, pad=34 if len(seq) > 14 else 20)

    # (2) response against amplitude
    if on:
        amps = np.array([r["amp"] for r in on])
        ax_r.plot(amps, [max(r["R"], 1e-3) for r in on], "o-", color=blue, lw=2, ms=6, mec="white", mew=1,
                  label=f"rms, nodes {', '.join(names[i] for i in nodes)}")
        if F > 0:
            ax_r.axhline(F, color=ink2, lw=1, ls="--", label=f"off level {F:.2f} counts")
        clear = [r for r in on if F == 0 or r["R"] >= MIN_OVER_OFF * F]
        if len(clear) >= 2:                     # what a linear screen would give, from the lowest clear step
            xs = np.array([clear[0]["amp"], amps.max()])
            ax_r.plot(xs, np.sqrt((clear[0]["net"] * xs / xs[0]) ** 2 + F ** 2), color=ink2, lw=1,
                      label=f"linear from {fmt(clear[0]['amp'])} {u}" + (f" (fitted slope {slope:.2f})" if slope is not None else ""))
        ax_r.set_xscale("log"); ax_r.set_yscale("log")
        ax_r.legend(fontsize=8, frameon=False, loc="upper left")
    ax_r.set_xlabel(f"generator amplitude [{u}]", color=ink2)
    ax_r.set_ylabel("rms [counts]", color=ink2)
    ax_r.set_title("Response against amplitude", fontsize=10, loc="left", color=ink)

    # (3) which nodes respond: one line per amplitude, light = small, dark = large
    xs = np.arange(len(names))
    ax_n.plot(xs, np.maximum(off_rms, 1e-3), color=ink2, lw=1, ls="--", label="off")
    lvl = sorted({r["amp"] for r in on})
    cmap = plt.get_cmap("Blues")
    for r in on:
        c = cmap(0.35 + 0.6 * (lvl.index(r["amp"]) / max(len(lvl) - 1, 1)))
        show = len(lvl) <= 8 or lvl.index(r["amp"]) in (0, len(lvl) - 1)
        ax_n.plot(xs, np.maximum(r["rms"], 1e-3), color=c, lw=1.6, label=f"{fmt(r['amp'])} {u}" if show else None)
    ax_n.set_yscale("log")
    nstep = max(1, len(names) // 14)
    ax_n.set_xticks(xs[::nstep]); ax_n.set_xticklabels(names[::nstep], fontsize=8)
    ax_n.set_xlabel("node", color=ink2)
    ax_n.set_ylabel("rms [counts]", color=ink2)
    ax_n.legend(fontsize=8, frameon=False, ncol=2, loc="upper right")
    ax_n.set_title("Per node, one line per amplitude (light = small, dark = large)", fontsize=10, loc="left", color=ink)
    for a_ in (ax_t, ax_r, ax_n):
        a_.grid(color=grid, lw=0.6, which="major"); a_.set_axisbelow(True)
        a_.tick_params(colors=ink2, labelsize=8)
        for s in ("top", "right"):
            a_.spines[s].set_visible(False)
    fig.suptitle(f"maXTouch response to the wave generator, {meta['freq_hz'] / 1e3:g} kHz {meta['wave']}"
                 + (f", {meta['condition']}" if meta.get("condition") else "")
                 + ("  [SIMULATED]" if meta.get("simulated") else ""), fontsize=11, x=0.01, ha="left", color=ink)
    fig.tight_layout()
    fig.savefig(os.path.join(d, "result.png"), dpi=120)
    plt.close(fig)


# ================================================================ summary over several runs
def summary(dirs, out=None):
    """one figure, response against amplitude: the steps of all the runs given as one series, each dot with its value"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    blue = "#2a78d6"
    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
    runs = []
    for d in dirs:
        f = os.path.join(d, "response.csv")
        if not os.path.exists(f):
            continue
        meta = json.load(open(os.path.join(d, "meta.json")))
        r = list(csv.DictReader(open(f)))
        ak = next(k for k in r[0] if k.startswith("amplitude_"))
        off = [float(x["rms_counts"]) for x in r if x[ak] == "off"]
        on = sorted((float(x[ak]), float(x["rms_counts"])) for x in r if x[ak] != "off")
        if on and not meta.get("simulated"):
            runs.append(dict(d=d, meta=meta, on=on, F=math.sqrt(np.mean(np.square(off))) if off else 0.0))
    if not runs:
        raise SystemExit("summary: no run with an amplitude step found")
    u, m0 = runs[0]["meta"]["unit"], runs[0]["meta"]
    pts = [(a, R, q["F"]) for q in runs for a, R in q["on"]]
    clear = [(a, math.sqrt(R * R - F * F)) for a, R, F in pts if F == 0 or R >= MIN_OVER_OFF * F]
    gain = float(np.exp(np.mean([math.log(n / a) for a, n in clear]))) if clear else None
    F = float(np.sqrt(np.mean([q["F"] ** 2 for q in runs])))
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    amps = np.array([p[0] for p in pts])
    xs = np.geomspace(amps.min() / 1.3, amps.max() * 1.7, 200)
    if gain:
        ax.plot(xs, gain * xs, color=ink2, lw=1, label=f"linear, {gain:.0f} counts per {u}")
        ax.plot(xs, np.sqrt((gain * xs) ** 2 + F ** 2), color=ink2, lw=1, ls=":", label="linear on top of the off level")
    if F > 0:
        ax.axhline(F, color=ink2, lw=1, ls="--", label=f"off level, {F:.0f} counts")
    pts.sort()                                  # all runs as one series; every dot carries its measured value
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "o", color=blue, ms=7, mec="white", mew=1, zorder=3,
            label=f"measured, {len(pts)} amplitudes (value = rms counts)")
    for i, (a_, R, _) in enumerate(pts):        # below right; the first of two close neighbours goes above left
        up = i + 1 < len(pts) and pts[i + 1][0] < 1.3 * a_
        ax.annotate(f"{R:.0f}", (a_, R), xytext=(-6, 4) if up else (6, -4), textcoords="offset points",
                    ha="right" if up else "left", va="bottom" if up else "top", fontsize=8, color=ink, zorder=2.5,
                    bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(xs[0], xs[-1])
    ax.set_xticks(sorted(set(amps)))            # the amplitudes that were actually set
    ax.set_xticks([], minor=True)
    ax.set_xticklabels([fmt(v) for v in sorted(set(amps))], rotation=90)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel(f"generator amplitude [{u}]", color=ink2)
    ax.set_ylabel("screen response, rms [counts]", color=ink2)
    ax.set_title(f"maXTouch response to the wave generator, {m0['freq_hz'] / 1e3:g} kHz {m0['wave']}"
                 + (f", {m0['condition']}" if m0.get("condition") else ""), fontsize=11, loc="left", color=ink)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.grid(color=grid, lw=0.6, which="major"); ax.set_axisbelow(True)
    ax.tick_params(colors=ink2, labelsize=9)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    out = out or os.path.join(os.path.dirname(os.path.abspath(runs[0]["d"])),
                              f"wavegen_summary_{m0['args']['label']}_{m0['freq_hz'] / 1e3:g}kHz.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"{len(runs)} runs, {len(pts)} amplitudes" + (f", {gain:.0f} counts per {u} over the {len(clear)} steps at least "
          f"6 dB over the off level" if gain else "") + f"\nSaved {os.path.relpath(out, HERE)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("run", "analyse", "summary"))
    ap.add_argument("what", nargs="*", help="analyse: wavegen_<label>_<freq>_<time> folders (globs ok)")
    ap.add_argument("--label", default="gen")
    ap.add_argument("--freq", type=float, default=168000.0, help="generator frequency [Hz], for the record (default 168e3)")
    ap.add_argument("--amps", help="amplitudes to step through, e.g. 0.05,0.1,0.2,0.5,1,2,5 (default: typed at the prompt)")
    ap.add_argument("--unit", default="Vpp", help="unit of the amplitudes, for the record (default Vpp)")
    ap.add_argument("--wave", default="sine", help="generator waveform, for the record (default sine)")
    ap.add_argument("--seconds", type=float, default=5.0, help="recording per amplitude [s] (default 5)")
    ap.add_argument("--settle", type=float, default=0.3, help="wait after Enter before recording [s] (default 0.3)")
    ap.add_argument("--top", type=int, help="nodes in the headline number (default 4)")
    ap.add_argument("--condition", help="set-up for the record, e.g. within_agar")
    ap.add_argument("--note", help="anything else for the record, e.g. generator model, series resistor")
    ap.add_argument("--y0", type=int, default=0)
    ap.add_argument("--ny", type=int, default=28)
    ap.add_argument("--keep-noise-suppression", action="store_true")
    ap.add_argument("--keep-touch", action="store_true")
    ap.add_argument("--simulate", action="store_true", help="no hardware: synthetic panel, to try the program")
    ap.add_argument("--datadir", help="output folder (default data_maxtouch)")
    ap.add_argument("--out", help="summary: figure file (default data_maxtouch/wavegen_summary_<label>_<freq>.png)")
    a = ap.parse_args()
    a.patch = None
    if a.cmd == "run":
        run(a)
    elif a.cmd == "summary":
        summary([d for w in (a.what or [os.path.join(DATADIR, "wavegen_*")]) for d in sorted(glob.glob(w)) if os.path.isdir(d)],
                a.out)
    else:
        ds = [d for w in a.what for d in sorted(glob.glob(w))]
        if not ds:
            raise SystemExit("analyse: give a wavegen_<label>_<freq>_<time> folder")
        for d in ds:
            analyse(d, a.top)


if __name__ == "__main__":
    main()
