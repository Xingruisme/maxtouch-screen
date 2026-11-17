#!/usr/bin/env python3
"""Shared through-body carrier sweep for phones whose touch map is streamed over root adb (OnePlus 8 Pro, Pixel 4,
Pixel 6, ...): TeensyTRX (AD9833) -> body -> finger -> phone touch panel. Each phone has a thin script in its own
folder that fills in a Phone (its MatrixStream reader, grid size, default block) and calls main(phone).

Experiment and analysis = the OnePlus Pad sweep (oneplus_body_sweep.py, imported unchanged): per carrier `--reps` x
(`--block` s ON, `--block` s OFF) while the delta map is streamed, then per node a level z and a frame-to-frame noise
dB score. Additions here:
  * person-connected runs need --current-ma, checked at every carrier against the IEC 60601-1 type-BF patient-current
    limit (0.1 mA up to 1 kHz, x f/1 kHz above, 10 mA from 100 kHz); --phantom skips it when no person is connected.
    With the 1 mA network only carriers above 10 kHz pass; 100 Hz-10 kHz needs <= 0.1 mA (or a separate low-band run).
  * the phone model is checked before anything is transmitted (the right phone, root, screen on).
  * `response RUN [--control CTL]`: response = max frame-to-frame noise rise over nodes [dB] vs carrier, the control's
    curve in grey, only carriers with both ON blocks recorded (OnePlus 8 Pro 2026-11-14: the level score flagged
    finger-pressure drift as responses in a no-TX control; the noise score did not).
"""

from __future__ import annotations

import argparse
import datetime
import json
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent / "maxtouch-screen-main"):
    if str(p) not in sys.path:
        sys.path.append(str(p))
import oneplus_body_sweep as SW                         # noqa: E402  schedule + clock + analysis (unchanged)

DEFAULT_CARRIERS = "log:100:500e3:10%"


@dataclass
class Phone:
    name: str                                           # "Pixel 6"
    panel: str                                          # "STM FTM5, 16 x 34"
    rows: int
    cols: int
    stream: Callable                                    # stream(adb) -> object with read(), last_device_ts_ms, close()
    read_once: Callable                                 # read_once(adb) -> (rows, cols) array; root/screen check
    resolve_serial: Callable                            # resolve_serial(adb_exe, requested) -> serial
    model_ok: Callable[[str], bool]                     # model_ok("getprop model + device") -> True for this phone
    out_dir: Path
    fps_note: str                                       # "~3.1 frames/s"
    default_block: float = 2.0
    default_min_blob: int = 2
    min_blob_note: str = ""
    script: str = ""
    to_screen: Callable | None = None                   # (r, c) on the stored grid -> (row, col, rows, cols) as held
    baseline_frames: int = 0                            # > 0: the map is raw (STM FTS stm_fts_cmd 13) -> subtract a
                                                        # finger-off baseline (median of this many frames), as the
                                                        # phone folder's own monitor does


def iec_limit_ma(f_hz: float) -> float:
    """IEC 60601-1 patient (auxiliary) current limit, type BF, normal condition: 0.1 mA up to 1 kHz, x f/1 kHz above,
    at most 10 mA (reached at 100 kHz)"""
    return min(10.0, 0.1 * max(1.0, f_hz / 1e3))


def check_current(carriers_hz, current_ma):
    """refuse a person-connected run if any carrier's limit is below the declared (nominal) current"""
    bad = [f for f in carriers_hz if current_ma > iec_limit_ma(f) + 1e-12]
    if bad:
        f_ok = current_ma / 0.1 * 1e3
        raise SystemExit(
            f"--current-ma {current_ma:g} exceeds the IEC 60601-1 patient-current limit (type BF) at "
            f"{len(bad)} carrier(s), {min(bad):.2f}-{max(bad):.2f} Hz (limit {iec_limit_ma(min(bad)):.4g} mA at "
            f"{min(bad):.2f} Hz). Either start the sweep above {f_ok:.0f} Hz (AD9833 frequencies land slightly off) "
            f"with this setting, or lower the current network to <= {min(iec_limit_ma(f) for f in carriers_hz):.2g} mA "
            "for the full range. Nothing was transmitted.")


def take_baseline(phone: Phone, adb, n: int) -> np.ndarray:
    """median of n frames read with nothing on the screen (the raw map's per-node offsets)"""
    stream = phone.stream(adb)
    try:
        B = np.array([stream.read(timeout=8.0) for _ in range(n)], float)
    finally:
        stream.close()
    return np.median(B, axis=0)


def record(phone: Phone, adb, lead_s, total_s, tail_s, start_tx, schedule_at, baseline=None):
    """lead with the transmitter off, start it, read total_s + tail_s (as oneplus_body_sweep.record); with a baseline
    the stored frames are raw - baseline (raw frames returned too)"""
    stream = phone.stream(adb)
    frames, t_dev, t_host, raws = [], [], [], []
    t0 = time.time()
    t_first = None
    last, n_last = t0, 0
    try:
        while True:
            now = time.time()
            if t_first is None and now - t0 >= lead_s:
                t_first = start_tx()
                print(f"\n  >>> schedule started (first block at +{t_first - now:.3f} s), "
                      f"{total_s:.0f} s + {tail_s:.0f} s tail\n", flush=True)
            if t_first is not None and now >= t_first + total_s + tail_s:
                break
            try:
                m = stream.read(timeout=8.0)
            except (TimeoutError, RuntimeError) as e:
                print(f"\n  ! {phone.name} stream stopped ({e}) - keeping the frames so far (screen off?)")
                break
            if baseline is not None:
                raws.append(m.astype(np.int16))
                m = np.round(m - baseline)
            frames.append(m.astype(np.int16))
            t_host.append(time.time())
            t_dev.append(stream.last_device_ts_ms if stream.last_device_ts_ms is not None else -1)
            if now - last >= 1.0 and len(frames) > n_last + 2:
                F = np.array(frames[n_last:], float)
                d = np.abs(np.diff(F, axis=0)).max(axis=0)
                r, c = np.unravel_index(int(np.argmax(d)), d.shape)
                rate = (len(frames) - n_last) / (now - last)
                what = schedule_at(now - t_first) if t_first else "lead (TX off)"
                tr, tc = np.unravel_index(int(np.argmax(np.abs(F[-1]))), F[-1].shape)
                where = f"r{tr}c{tc}"
                if phone.to_screen:
                    sr, sc, nr, nc = phone.to_screen(tr, tc)
                    where = f"screen row {sr}/{nr} col {sc}/{nc}"
                print(f"  {now - t0:6.1f} s  {rate:4.1f} fps  touch max {np.abs(F[-1]).max():5.0f} at {where}   "
                      f"max |step| {d.max():5.0f} at r{r}c{c}   TX: {what}", flush=True)
                last, n_last = now, len(frames)
    except KeyboardInterrupt:
        print("  stopped (Ctrl+C)")
    finally:
        stream.close()
    return (np.array(frames, np.int16).reshape(-1, phone.rows, phone.cols), np.array(t_dev, float) / 1e3,
            np.array(t_host, float), t_first, stream.skipped_frames,
            np.array(raws, np.int16).reshape(-1, phone.rows, phone.cols) if baseline is not None else None)


# ================================================================ response figure
def response_curve(d: Path):
    """carriers with both ON blocks recorded -> (Hz, max noise dB over nodes, largest connected blob >= 6 dB)"""
    s = np.load(d / "scores.npz")
    res = json.loads((d / "result.json").read_text())
    reps = json.loads((d / "meta.json").read_text())["args"]["reps"]
    keep = [i for i, r in enumerate(res["carriers"]) if r.get("reps_used") == reps]
    N = s["NDB"]
    return (s["carriers"][keep], N[keep].reshape(len(keep), -1).max(1),
            [SW.largest_blob(N[i] >= 6) for i in keep], len(res["carriers"]) - len(keep))


def response_plot(d: Path, control: Path | None, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f, y, blob, missing = response_curve(d)
    meta = json.loads((d / "meta.json").read_text())
    ink, muted, blue, grey = "#1f2328", "#5f6670", "#2a78d6", "#9aa0a6"
    fig, ax = plt.subplots(figsize=(10, 5.2))
    lo = min(f.min(), 100.0) if len(f) else 100.0
    hi = max(f.max(), 500e3) if len(f) else 500e3
    ymax = max(12.0, float(np.max(y)) + 2 if len(y) else 12.0)
    if control is not None:
        fc, yc, _, _ = response_curve(control)
        cmax = float(yc.max())
        ax.axhline(cmax, color=muted, lw=1, ls="--", zorder=1)
        ax.text(lo * 1.05, cmax + 0.2, f"control maximum {cmax:.1f} dB", fontsize=9, color=muted, va="bottom")
        ax.plot(fc, yc, color=grey, lw=1.5, marker="o", ms=4, zorder=2,
                label=f"control: finger on, nothing transmitted ({control.name})")
        ymax = max(ymax, cmax + 2)
    cur = meta["args"].get("current_ma")
    lab = f"{d.name}: {'nothing transmitted' if meta['args'].get('no_tx') else 'through body'}" + \
          (f", {cur:g} mA" if cur else "")
    ax.plot(f, y, color=blue, lw=2, marker="o", ms=7, mec="white", mew=1.2, zorder=3, label=lab)
    if len(y):
        k = int(np.argmax(y))
        ax.annotate(f"{f[k] / 1e3:.3g} kHz: {y[k]:.1f} dB, {blob[k]} connected nodes >= 6 dB", (f[k], y[k]),
                    xytext=(0, 12), textcoords="offset points", ha="center", fontsize=9, color=ink)
    ax.set_xscale("log"); SW.hz_axis(ax, lo, hi); ax.set_xlim(lo / 1.2, hi * 1.2); ax.set_ylim(0, ymax)
    ax.set_xlabel("carrier frequency [Hz]", color=ink)
    ax.set_ylabel("response: max frame-to-frame noise rise,\nON vs OFF, over nodes [dB]", color=ink)
    ax.grid(axis="y", color="#e3e6ea", lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.set_title(title + (f"  ({missing} carrier(s) not fully recorded, left out)" if missing else ""),
                 fontsize=10, color=ink)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(d / f"response_vs_frequency.{ext}", dpi=160)
    plt.close(fig)
    print(f"saved {d / 'response_vs_frequency.png'}")


# ================================================================ main
def main(phone: Phone) -> int:
    ap = argparse.ArgumentParser(description=f"TeensyTRX -> body -> {phone.name} carrier sweep")
    ap.add_argument("mode", choices=("run", "analyse", "plan", "response"))
    ap.add_argument("dir", nargs="?", help="analyse / response: the run folder")
    ap.add_argument("--control", help="response: a no-TX control run folder drawn in grey")
    ap.add_argument("--carriers", default=DEFAULT_CARRIERS,
                    help="log:F0:F1:STEP%% | lin:F0:F1:STEP | f1,f2,... (default log:100:500e3:10%%)")
    ap.add_argument("--block", type=float, default=phone.default_block,
                    help=f"ON and OFF block length [s] (default {phone.default_block:g}; {phone.name} {phone.fps_note})")
    ap.add_argument("--reps", type=int, help="ON/OFF pairs per carrier (default 2; 1 with --quick)")
    ap.add_argument("--quick", action="store_true",
                    help="basic sweep, half the time: 1 ON/OFF pair per carrier, same 10 %% steps and range (simulation: "
                         "no false carriers; weak 12-count responses found 4-5/6 instead of 6/6)")
    ap.add_argument("--lead", type=float, default=10.0, help="TX-off recording before the schedule [s]")
    ap.add_argument("--tail", type=float, default=4.0, help="recording after the schedule [s]")
    ap.add_argument("--wave", default="square", choices=("square", "sine"))
    ap.add_argument("--port", help="TeensyTRX serial port (default: first Teensy)")
    ap.add_argument("--serial", help=f"adb serial of the {phone.name} (default: auto)")
    ap.add_argument("--adb", help="adb path")
    ap.add_argument("--no-tx", action="store_true", help="control run: same timing, nothing transmitted")
    ap.add_argument("--label", default="sweep")
    ap.add_argument("--site", default="", help="e.g. hand2finger (recorded)")
    ap.add_argument("--current", default="", help="current / network setting as set on the bench (recorded)")
    ap.add_argument("--current-ma", type=float, help="nominal current of the network [mA]; required for a person-"
                    "connected run, checked against the IEC 60601-1 limit at every carrier")
    ap.add_argument("--phantom", action="store_true", help="no person connected (saline/agar phantom): no current check")
    ap.add_argument("--note", default="")
    ap.add_argument("--jump", type=float, default=50.0, help="frame-to-frame step [counts] counted as a touch jump")
    ap.add_argument("--min-blob", type=int, default=phone.default_min_blob,
                    help=f"connected nodes a response must cover (default {phone.default_min_blob}{phone.min_blob_note})")
    ap.add_argument("--yes", action="store_true", help="skip the Enter prompt")
    a = ap.parse_args()

    if a.reps is None:
        a.reps = 1 if a.quick else 2
    if a.mode in ("analyse", "response"):
        if not a.dir:
            ap.error(f"{a.mode} needs the run folder")
        if a.mode == "analyse":
            SW.analyse(Path(a.dir), a.jump)
        else:
            response_plot(Path(a.dir), Path(a.control) if a.control else None, f"{phone.name} through-body sweep")
        return 0

    carriers = SW.parse_carriers(a.carriers)
    actual = [SW.ad9833_hz(f) for f in carriers]
    segs, blocks = SW.build_schedule(actual, a.block, a.reps)
    total = blocks[-1]["off"] + a.block if blocks else 0.0
    print(f"{len(carriers)} carriers {actual[0] / 1e3:.2f}-{actual[-1] / 1e3:.2f} kHz, {a.reps} x ({a.block:g} s ON, "
          f"{a.block:g} s OFF) each: schedule {total:.0f} s, run {a.lead + total + a.tail:.0f} s "
          f"({(a.lead + total + a.tail) / 60:.1f} min); {phone.name} map {phone.rows} x {phone.cols}, {phone.fps_note}")
    import stent_tx as STX
    STX.to_schedule(segs)                       # raises if the TeensyTRX cannot hold it
    lim = min(iec_limit_ma(f) for f in actual)
    print(f"IEC 60601-1 (type BF) patient-current limit over this range: >= {lim:.2g} mA "
          f"(lowest carrier {actual[0]:.0f} Hz)")
    if a.mode == "plan":
        if a.current_ma is not None:
            check_current(actual, a.current_ma)
            print(f"  {a.current_ma:g} mA is within the limit at every carrier")
        return 0
    if not a.no_tx and not a.phantom:
        if a.current_ma is None:
            raise SystemExit("person-connected run: give --current-ma (the network's nominal current) so it can be "
                             "checked against the limit at every carrier; --phantom if no person is connected")
        check_current(actual, a.current_ma)

    def schedule_at(s):
        if s < 0 or s >= total:
            return "off (schedule done)" if s >= total else "off"
        bl = blocks[min(int(s // (2 * a.block)), len(blocks) - 1)]
        return f"{actual[bl['carrier']] / 1e3:8.2f} kHz {'ON ' if (s - bl['on']) < a.block else 'off'} (rep {bl['rep'] + 1})"

    exe = SW.find_adb(a.adb)
    serial = phone.resolve_serial(exe, a.serial)
    if not serial:
        raise SystemExit(f"no adb device online - plug in the {phone.name}, unlock it, accept USB debugging")
    adb = SW.Adb(exe, serial)
    model = adb.shell("getprop ro.product.model").strip()
    device = adb.shell("getprop ro.product.device").strip()
    if not phone.model_ok(f"{model} {device}"):
        raise SystemExit(f"{serial} is a {model} ({device}), not a {phone.name} - pass --serial. Nothing was transmitted.")
    grid = phone.read_once(adb)                 # root + screen-on check before anything is transmitted
    print(f"{phone.name} ({model}, {serial}): delta map {grid.shape[0]} x {grid.shape[1]} readable (root ok, screen on)")
    tx = tx_info = None
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
    try:
        if a.no_tx:
            start_tx = lambda: time.time() + STX.GO_DELAY_MS / 1e3
            print("CONTROL run: nothing is transmitted.")
        else:
            tx, tx_info = STX.open_configured(a)
            print(f"Transmitter: {tx_info['line']}  ({tx.port}, wave {a.wave})")
            print(f"  uploaded: {tx.upload(segs)}")
            start_tx = lambda: tx.go()[0]
        baseline = None
        if phone.baseline_frames:
            if not a.yes:
                input(f"BASELINE: lift your finger OFF the {phone.name} screen (screen on, nothing touching it), "
                      "then press Enter ")
            baseline = take_baseline(phone, adb, phone.baseline_frames)
            print(f"  baseline = median of {phone.baseline_frames} finger-off frames (per-node offsets "
                  f"{baseline.min():.0f}..{baseline.max():.0f}); frames are stored as raw - baseline")
        print(f"Finger on the {phone.name} screen and keep it still until the run ends (screen stays ON); the "
              f"transmitter is OFF during the {a.lead:g} s lead.")
        if not a.yes:
            input("Finger ON now, then press Enter to start recording (Ctrl+C to cancel) ")
        off0 = SW.clock_offset(adb)
        F, t_dev, t_host, t_first, skipped, F_raw = record(phone, adb, a.lead, total, a.tail, start_tx, schedule_at,
                                                           baseline)
        off1 = SW.clock_offset(adb)
    finally:
        if tx:
            tx.close()                          # 'off'
    if len(F) < 20:
        print("too few frames recorded")
        return 1
    off = 0.5 * (off0["offset_s"] + off1["offset_s"])
    good = t_dev > 0
    t = np.where(good, t_dev - off, t_host)     # device frame stamps on the host clock
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    d = phone.out_dir / f"{a.label}_{stamp}"
    d.mkdir(parents=True, exist_ok=True)
    extra = dict(F_raw=F_raw, baseline=baseline) if baseline is not None else {}
    np.savez_compressed(d / "frames.npz", F=F, t=t, t_host=t_host, t_dev=t_dev, **extra)
    meta = dict(args=vars(a), pad_model=model, device=f"{phone.name} ({phone.panel})", product_device=device,
                map=("raw stm_fts_cmd 13 frame minus a finger-off baseline (median of "
                     f"{phone.baseline_frames} frames; F_raw / baseline in frames.npz)") if baseline is not None
                else "controller delta map",
                adb_serial=serial, carriers_hz_requested=carriers, carriers_hz_actual=actual, blocks=blocks,
                schedule_s=total, t_first=t_first, clock_offset_start=off0, clock_offset_end=off1,
                clock_offset_used_s=off, frames_without_device_stamp=int((~good).sum()), skipped_frames=skipped,
                tx=tx_info, guard_s=dict(start=SW.GUARD_START_S, end=SW.GUARD_END_S), source="delta",
                recorded_at=stamp, script=phone.script)
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"saved {d}")
    SW.analyse(d)
    if t_first is None or t[-1] < t_first + total:
        print(f"  ! the recording ended {max(0.0, t_first + total - t[-1]) if t_first else total:.0f} s before the "
              "schedule did - carriers after that have no data (response figure leaves them out)")
    response_plot(d, None, f"{phone.name} through-body sweep")
    return 0
