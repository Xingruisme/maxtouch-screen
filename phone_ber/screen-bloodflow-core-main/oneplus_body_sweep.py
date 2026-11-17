#!/usr/bin/env python3
"""Through-body carrier sweep: TeensyTRX (AD9833) -> body -> finger -> OnePlus Pad touch panel.

The TeensyTRX plays a timed schedule: for every carrier, `--reps` x (`--block` s ON, `--block` s OFF). At the same
time the Pad's Novatek touch controller is read over root adb (``/proc/touchpanel/debug_info/delta``, the full
38 x 54 map, ~6-8 frames/s; same reader as oneplus_monitor.py). Every carrier is then scored per node:

  level z   median delta ON minus the mean of the OFF-block medians on either side (cancels slow finger drift), in units of
            the node's block-to-block noise measured on all OFF blocks of the run
  noise dB  frame-to-frame change inside ON blocks vs inside OFF blocks (a carrier the controller sees as noise),
            each step clipped at DIFF_CLIP counts

A carrier "responds" if >= min_blob connected nodes (args.min_blob, default 1; Nexus 5: 3) clear max(5, median + 6
robust sigma) in |level z| or max(6 dB, median + 6 robust sigma) in noise dB over the whole node x carrier matrix. Big frame-to-frame jumps (> --jump counts) are counted
per carrier as touch artifacts (finger moved/pressed), like the maXTouch body sweeps.

Usage (galvanic venv):
    python oneplus_body_sweep.py run --label hand --site hand2finger            # default 20 kHz-1 MHz, 10 % steps
    python oneplus_body_sweep.py run --label ctl --no-tx                        # control: same timing, nothing sent
    python oneplus_body_sweep.py run --carriers 100e3,159585,250e3 --block 2 --reps 3
    python oneplus_body_sweep.py analyse data_oneplus/body_sweep/<run folder>
    python oneplus_body_sweep.py plan                                           # schedule length, no hardware

The transmitter driver is maxtouch-screen-main/stent_tx.py (unchanged); `python stent_tx.py selftest 160000` there
checks that a carrier really comes out. The output is turned off on exit, error or Ctrl+C.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import signal
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TX_DIR = HERE.parent / "maxtouch-screen-main"            # stent_tx.py + stent_frames.py (TeensyTRX driver)
if str(TX_DIR) not in sys.path:
    sys.path.append(str(TX_DIR))

from oneplus_capacitance import COLS, ROWS, Adb, device_epoch_ms, find_adb, resolve_adb_serial  # noqa: E402

OUT_DIR = HERE / "data_oneplus" / "body_sweep"
DEFAULT_CARRIERS = "log:20e3:1e6:10%"
GUARD_START_S = 0.35      # skip after each ON/OFF edge: controller latency + adb frame timing (+-~25 ms)
GUARD_END_S = 0.10
SIGMA_FLOOR = 0.5         # counts: idle delta is integer and mostly 0, so the measured noise can be ~0
DIFF_FLOOR = 1.0          # counts^2 added to both noise terms: noise under ~1 count rms is ignored
DIFF_CLIP = 5.0           # counts: each frame-to-frame step is clipped here for the noise score, so one rare big
                          # step (Nexus 5 glitch, a twitch) cannot fake '+18 dB' on a node that otherwise reads 0


# ================================================================ schedule
def parse_carriers(s: str) -> list[float]:
    """'log:F0:F1:STEP%' | 'lin:F0:F1:STEP' | 'f1,f2,...' (Hz)"""
    if s.startswith("log:"):
        _, f0, f1, st = s.split(":")
        f0, f1 = float(f0), float(f1)
        r = 1 + float(st.rstrip("%")) / 100 if st.endswith("%") else float(st)
        n = int(math.floor(math.log(f1 / f0) / math.log(r) + 1e-9)) + 1
        return [round(f0 * r ** k, 1) for k in range(n)]
    if s.startswith("lin:"):
        _, f0, f1, st = s.split(":")
        return [round(float(v), 1) for v in np.arange(float(f0), float(f1) + float(st) / 2, float(st))]
    return [float(v) for v in s.split(",")]


def ad9833_hz(f: float) -> float:
    import stent_tx as STX
    return STX.fword(f) * STX.MCLK / 268435456.0


def build_schedule(carriers: list[float], block_s: float, reps: int) -> tuple[list[tuple[float, int]], list[dict]]:
    """run-length segments [(hz, us)] for stent_tx, and the ON/OFF blocks relative to the schedule start"""
    b = int(round(block_s * 1e6))
    segs, blocks, t = [], [], 0.0
    for ci, f in enumerate(carriers):
        for r in range(reps):
            segs += [(float(f), b), (0.0, b)]
            blocks.append(dict(carrier=ci, rep=r, on=t, off=t + b / 1e6))
            t += 2 * b / 1e6
    return segs, blocks


# ================================================================ Pad
def clock_offset(adb: Adb, n: int = 7) -> dict:
    """device clock minus host clock [s], from the query with the shortest round trip"""
    best = None
    for _ in range(n):
        t0 = time.time()
        dev = device_epoch_ms(adb) / 1e3
        t1 = time.time()
        if best is None or t1 - t0 < best[1]:
            best = (dev - 0.5 * (t0 + t1), t1 - t0)
    return dict(offset_s=best[0], rtt_s=best[1])


def record(adb: Adb, lead_s: float, total_s: float, tail_s: float, start_tx, schedule_at):
    """Read delta frames: `lead_s` with the transmitter off, start it, keep reading `total_s + tail_s`."""
    from oneplus_monitor import MatrixStream

    stream = MatrixStream(adb, "delta")
    frames, t_dev, t_host = [], [], []
    t0 = time.time()
    t_first = None
    last = t0
    n_last = 0
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
                print(f"\n  ! Pad stream stopped ({e}) - keeping the frames so far")
                break
            frames.append(m.astype(np.int16))
            t_host.append(time.time())
            t_dev.append(stream.last_device_ts_ms if stream.last_device_ts_ms is not None else -1)
            if now - last >= 1.0 and len(frames) > n_last + 2:
                F = np.array(frames[n_last:], float)
                d = np.abs(np.diff(F, axis=0)).max(axis=0)
                r, c = np.unravel_index(int(np.argmax(d)), d.shape)
                rate = (len(frames) - n_last) / (now - last)
                what = schedule_at(now - t_first) if t_first else "lead (TX off)"
                print(f"  {now - t0:6.1f} s  {rate:4.1f} fps  max |step| {d.max():5.0f} at r{r}c{c}  "
                      f"touch max {np.abs(F[-1]).max():5.0f}   TX: {what}", flush=True)
                last, n_last = now, len(frames)
    except KeyboardInterrupt:
        print("  stopped (Ctrl+C)")
    finally:
        stream.close()
    return (np.array(frames, np.int16).reshape(-1, ROWS, COLS), np.array(t_dev, float) / 1e3,
            np.array(t_host, float), t_first, stream.skipped_frames)


# ================================================================ analysis
def robust_sigma(v: np.ndarray) -> float:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return float("nan")
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


TRIM_FRAC = 0.05


def trimmed_rms(v: np.ndarray) -> float:
    v = np.abs(np.asarray(v, float))
    v = np.sort(v[np.isfinite(v)])
    if len(v) < 3:
        return float("nan")
    keep = v[: max(2, int(np.ceil(len(v) * (1 - TRIM_FRAC))))]
    return float(np.sqrt(np.mean(keep ** 2)))


def largest_blob(mask: np.ndarray) -> int:
    from scipy.ndimage import label
    lab, k = label(mask, structure=np.ones((3, 3), bool))
    return int(np.bincount(lab.ravel())[1:].max()) if k else 0


def analyse(d: Path, jump: float | None = None, quiet: bool = False) -> dict:
    meta = json.loads((d / "meta.json").read_text())
    z = np.load(d / "frames.npz")
    F = z["F"].astype(float)
    t = z["t"]                                  # frame times on the host clock [s]
    jump = jump if jump is not None else meta["args"].get("jump", 50.0)
    min_blob = int(meta["args"].get("min_blob", 1))   # nodes a response must cover (8-connected); Nexus 5: 3
    car = meta["carriers_hz_actual"]
    blocks = meta["blocks"]
    b = meta["args"]["block"]
    reps = meta["args"]["reps"]
    t_first = meta["t_first"]
    N, R, C = F.shape
    X = F.reshape(N, -1)
    dX = np.diff(X, axis=0)                     # frame-to-frame change, assigned to the later frame
    # touch jumps on a 5-frame running median: sustained moves/presses, not single-frame dips (the Nexus 5 finger
    # nodes flip between ~260 and ~165 every few frames)
    from scipy.ndimage import median_filter
    jumps_all = np.abs(np.diff(median_filter(X, size=(5, 1), mode="nearest"), axis=0)).max(axis=1) > jump

    def win(a, e):
        return np.where((t >= a) & (t < e))[0]

    # every OFF window in time order (the lead counts as OFF too), for the node noise and the OFF neighbours
    off_wins = []
    lead_end = t_first - GUARD_END_S
    k = lead_end
    while k - b >= t[0] + GUARD_START_S:
        off_wins.append((k - b + GUARD_START_S, k))
        k -= b
    off_wins = off_wins[::-1]
    lead_last = len(off_wins) - 1
    on_idx, off_after = [], []
    for bl in blocks:
        on = t_first + bl["on"]
        on_idx.append(win(on + GUARD_START_S, on + b - GUARD_END_S))
        off_after.append(len(off_wins))
        off_wins.append((on + b + GUARD_START_S, on + 2 * b - GUARD_END_S))
    off_idx = [win(a, e) for a, e in off_wins]
    # block level = MEDIAN of its frames (robust to dips/spikes); node noise = RMS of the change between
    # consecutive OFF levels (/sqrt 2 = one level's noise), the largest TRIM_FRAC of changes left out (a finger
    # landing in the lead). Not a MAD: nodes that sit still and jump now and then got a MAD ~0 and z ~70 on noise.
    off_mean = np.array([np.median(X[i], axis=0) if len(i) else np.full(R * C, np.nan) for i in off_idx])
    dm = np.diff(off_mean, axis=0)
    sig = np.array([trimmed_rms(dm[:, j]) for j in range(R * C)]) / math.sqrt(2)
    sig = np.maximum(np.nan_to_num(sig, nan=SIGMA_FLOOR), SIGMA_FLOOR)

    nc = len(car)
    Z = np.zeros((nc, R * C))
    NDB = np.zeros((nc, R * C))
    rows_out = []
    for ci in range(nc):
        contr, d_on, d_off, n_on, n_jump = [], [], [], 0, 0
        for bi, bl in enumerate(blocks):
            if bl["carrier"] != ci:
                continue
            ion = on_idx[bi]
            ia = off_after[bi]
            prev = ia - 1                       # the OFF window just before this ON block
            if len(ion) == 0 or prev < 0 or not len(off_idx[prev]) or not len(off_idx[ia]):
                continue
            contr.append(np.median(X[ion], axis=0) - 0.5 * (off_mean[prev] + off_mean[ia]))
            n_on += len(ion)
            for idx, acc in ((ion, d_on), (off_idx[ia], d_off)):
                j = idx[np.isin(idx - 1, idx)] - 1  # dX[j] = X[j+1] - X[j], both frames inside the window
                if len(j):
                    acc.append(np.minimum(dX[j] ** 2, DIFF_CLIP ** 2))
            span = (t >= t_first + bl["on"]) & (t < t_first + bl["on"] + 2 * b)
            n_jump += int(jumps_all[np.where(span[1:])[0]].sum())
        if not contr:
            rows_out.append(dict(carrier_hz=car[ci], error="no frames in its blocks"))
            continue
        cm = np.mean(contr, axis=0)
        Z[ci] = cm / (sig * math.sqrt(1.5 / len(contr)))
        e_on = np.concatenate(d_on).mean(axis=0) if d_on else np.zeros(R * C)
        e_off = np.concatenate(d_off).mean(axis=0) if d_off else np.zeros(R * C)
        NDB[ci] = 10 * np.log10((e_on + DIFF_FLOOR) / (e_off + DIFF_FLOOR))
        jz, jn = int(np.argmax(np.abs(Z[ci]))), int(np.argmax(NDB[ci]))
        rows_out.append(dict(carrier_hz=car[ci], reps_used=len(contr), frames_on=n_on, touch_jumps=n_jump,
                             level=dict(node=[jz // C, jz % C], z=float(Z[ci, jz]), counts=float(cm[jz])),
                             noise=dict(node=[jn // C, jn % C], db=float(NDB[ci, jn]),
                                        on_rms=float(math.sqrt(e_on[jn])))))

    aZ = np.abs(Z)
    thr_z = max(5.0, float(np.median(aZ)) + 6 * robust_sigma(aZ.ravel()))
    thr_db = max(6.0, float(np.median(NDB)) + 6 * robust_sigma(NDB.ravel()))
    for ci, row in enumerate(rows_out):
        if "error" in row:
            continue
        row["nodes_over_z"] = int((aZ[ci] >= thr_z).sum())
        row["nodes_over_db"] = int((NDB[ci] >= thr_db).sum())
        row["blob_z"] = largest_blob(aZ[ci].reshape(R, C) >= thr_z)
        row["blob_db"] = largest_blob(NDB[ci].reshape(R, C) >= thr_db)
        row["responds"] = bool(row["blob_z"] >= min_blob or row["blob_db"] >= min_blob)
        row["touch_artifact_risk"] = bool(row["touch_jumps"] > 0)
    score = np.maximum(aZ.max(axis=1) / thr_z, NDB.max(axis=1) / thr_db)
    order = [int(i) for i in np.argsort(-score)]
    touch = np.nanmean(off_mean[: lead_last + 1], axis=0) if lead_last >= 0 else np.nanmean(off_mean, axis=0)
    dt = np.diff(t)
    res = dict(run=d.name, frames=N, fps=float(1 / np.median(dt)) if len(dt) else None,
               frame_gap_p95_ms=float(np.percentile(dt, 95) * 1e3) if len(dt) else None,
               transmitted=not meta["args"].get("no_tx", False),
               thresholds=dict(level_z=thr_z, noise_db=thr_db, min_blob_nodes=min_blob), jump_counts=jump,
               node_noise_counts=dict(median=float(np.median(sig)), floor=SIGMA_FLOOR),
               lead_touch=dict(max=float(np.nanmax(np.abs(touch))),
                               node=[int(np.nanargmax(np.abs(touch))) // C, int(np.nanargmax(np.abs(touch))) % C]),
               responding=[rows_out[i]["carrier_hz"] for i in order if rows_out[i].get("responds")],
               top=[rows_out[i] for i in order[:8]], carriers=rows_out)
    (d / "result.json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(d / "scores.npz", Z=Z.reshape(nc, R, C), NDB=NDB.reshape(nc, R, C), carriers=np.array(car),
                        sigma=sig.reshape(R, C), touch=touch.reshape(R, C))
    plot(d, res, Z.reshape(nc, R, C), NDB.reshape(nc, R, C), touch.reshape(R, C), X, t, meta, order[0])
    if not quiet:
        report(res)
    return res


def report(res: dict) -> None:
    print(f"\n{res['run']}: {res['frames']} frames, {res['fps']:.1f} fps (95 % of gaps <= "
          f"{res['frame_gap_p95_ms']:.0f} ms), {'TX on' if res['transmitted'] else 'CONTROL (nothing transmitted)'}")
    th = res["thresholds"]
    print(f"  thresholds: |level z| >= {th['level_z']:.1f}, noise >= {th['noise_db']:.1f} dB, on >= "
          f"{th.get('min_blob_nodes', 1)} connected node(s);  "
          f"lead touch max {res['lead_touch']['max']:.0f} counts at r{res['lead_touch']['node'][0]}"
          f"c{res['lead_touch']['node'][1]}" + ("  (no finger on the panel?)" if res['lead_touch']['max'] < 100 else ""))
    print(f"  {'carrier':>11} {'level z':>8} {'counts':>7} {'node':>8} {'noise dB':>9} {'node':>8} {'jumps':>6}")
    for r in res["top"]:
        if "error" in r:
            continue
        flag = " RESPONDS" if r["responds"] else ""
        flag += "  (touch jumps)" if r["touch_jumps"] else ""
        print(f"  {r['carrier_hz'] / 1e3:9.2f}k {r['level']['z']:8.1f} {r['level']['counts']:7.1f} "
              f"r{r['level']['node'][0]:02d}c{r['level']['node'][1]:02d} {r['noise']['db']:9.1f} "
              f"r{r['noise']['node'][0]:02d}c{r['noise']['node'][1]:02d} {r['touch_jumps']:6d}{flag}")
    if not res["responding"]:
        print("  no carrier clears the thresholds")
    else:
        print(f"  responding: {', '.join(f'{f / 1e3:.1f}k' for f in res['responding'])}")
    print(f"  figure: result.png, numbers: result.json / scores.npz")


def hz_axis(ax, fmin: float, fmax: float) -> None:
    """log frequency axis ticked at 1, 2, 3 x 10^n Hz, labelled 100, 200, 300, 1k, 2k, 3k, ... 1M"""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    ticks = [m * 10.0 ** e for e in range(0, 8) for m in (1, 2, 3) if fmin / 1.3 <= m * 10.0 ** e <= fmax * 1.3]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v / 1e6:g}M" if v >= 1e6 else f"{v / 1e3:g}k" if v >= 1e3 else f"{v:g}"))


def plot(d, res, Z, NDB, touch, X, t, meta, best):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    car = np.array(meta["carriers_hz_actual"]) / 1e3
    th = res["thresholds"]
    fig, ax = plt.subplots(3, 2, figsize=(13, 12))
    a = ax[0, 0]
    a.semilogx(car * 1e3, np.abs(Z).reshape(len(car), -1).max(axis=1), "o-", color="#1f5fbf", ms=4)
    a.axhline(th["level_z"], color="0.5", ls="--", lw=1)
    a.set_ylabel("max |level z| over nodes"); a.set_title("a  level shift ON vs OFF")
    a = ax[0, 1]
    a.semilogx(car * 1e3, NDB.reshape(len(car), -1).max(axis=1), "o-", color="#e66100", ms=4)
    a.axhline(th["noise_db"], color="0.5", ls="--", lw=1)
    a.set_ylabel("max noise increase [dB]"); a.set_title("b  frame-to-frame noise ON vs OFF")
    for a in ax[0]:
        a.set_xlabel("carrier [Hz]"); a.grid(alpha=0.3, which="both")
        hz_axis(a, car[0] * 1e3, car[-1] * 1e3)
        jumps = [r.get("touch_jumps", 0) for r in res["carriers"]]
        for f, j in zip(car, jumps):
            if j:
                a.axvline(f * 1e3, color="#d62728", alpha=0.15, lw=3)
    lim = max(np.abs(Z[best]).max(), 1)
    im = ax[1, 0].imshow(Z[best], cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax[1, 0].set_title(f"c  level z per node at {car[best]:.1f} kHz (best)"); fig.colorbar(im, ax=ax[1, 0])
    im = ax[1, 1].imshow(NDB[best], cmap="magma", vmin=0, vmax=max(NDB[best].max(), 6), aspect="auto")
    ax[1, 1].set_title(f"d  noise dB per node at {car[best]:.1f} kHz"); fig.colorbar(im, ax=ax[1, 1])
    tl = max(np.abs(touch).max(), 1)
    im = ax[2, 0].imshow(touch, cmap="RdBu_r", vmin=-tl, vmax=tl, aspect="auto")
    ax[2, 0].set_title("e  mean delta during the lead (TX off) = finger"); fig.colorbar(im, ax=ax[2, 0])
    for a in (ax[1, 0], ax[1, 1], ax[2, 0]):
        a.set_xlabel("col"); a.set_ylabel("row")
    # f: best node over time, ON blocks of the best carrier shaded
    j = int(np.argmax(np.maximum(np.abs(Z[best]).ravel() / th["level_z"], NDB[best].ravel() / th["noise_db"])))
    a = ax[2, 1]
    tt = t - meta["t_first"]
    a.plot(tt, X[:, j], lw=0.7, color="0.2")
    for bl in meta["blocks"]:
        if bl["carrier"] == best:
            a.axvspan(bl["on"], bl["off"], color="#1f5fbf", alpha=0.2)
    on0 = min(bl["on"] for bl in meta["blocks"] if bl["carrier"] == best)
    a.set_xlim(on0 - 4 * meta["args"]["block"], on0 + (2 * meta["args"]["reps"] + 4) * meta["args"]["block"])
    a.set_xlabel("time from schedule start [s]"); a.set_ylabel("delta [counts]")
    a.set_title(f"f  node r{j // Z.shape[2]}c{j % Z.shape[2]}, shaded = {car[best]:.1f} kHz ON")
    fig.suptitle(f"{res['run']}  ({'TX' if res['transmitted'] else 'CONTROL, no TX'})", fontsize=11)
    fig.tight_layout()
    fig.savefig(d / "result.png", dpi=120)
    plt.close(fig)


# ================================================================ main
def main() -> int:
    ap = argparse.ArgumentParser(description="TeensyTRX -> body -> OnePlus Pad carrier sweep")
    ap.add_argument("mode", choices=("run", "analyse", "plan"))
    ap.add_argument("dir", nargs="?", help="analyse: the run folder")
    ap.add_argument("--carriers", default=DEFAULT_CARRIERS, help="log:F0:F1:STEP%% | lin:F0:F1:STEP | f1,f2,... (default " + DEFAULT_CARRIERS.replace("%", "%%") + ")")
    ap.add_argument("--block", type=float, default=2.0, help="ON and OFF block length [s] (default 2: ~12 frames)")
    ap.add_argument("--reps", type=int, default=2, help="ON/OFF pairs per carrier (default 2)")
    ap.add_argument("--lead", type=float, default=10.0, help="TX-off recording before the schedule [s]")
    ap.add_argument("--tail", type=float, default=4.0, help="recording after the schedule [s]")
    ap.add_argument("--wave", default="square", choices=("square", "sine"),
                    help="AD9833 output (square as in the maXTouch sweeps; has odd harmonics)")
    ap.add_argument("--port", help="TeensyTRX serial port (default: first Teensy)")
    ap.add_argument("--serial", help="adb serial of the Pad (default: auto)")
    ap.add_argument("--adb", help="adb path")
    ap.add_argument("--no-tx", action="store_true", help="control run: same timing, nothing transmitted")
    ap.add_argument("--label", default="sweep")
    ap.add_argument("--site", default="", help="e.g. hand2finger, ankle2finger (recorded)")
    ap.add_argument("--current", default="", help="current / network setting as set on the bench (recorded)")
    ap.add_argument("--note", default="")
    ap.add_argument("--jump", type=float, default=50.0, help="frame-to-frame step [counts] counted as a touch jump")
    ap.add_argument("--yes", action="store_true", help="skip the Enter prompt")
    a = ap.parse_args()

    if a.mode == "analyse":
        if not a.dir:
            ap.error("analyse needs the run folder")
        analyse(Path(a.dir), a.jump)
        return 0

    carriers = parse_carriers(a.carriers)
    actual = [ad9833_hz(f) for f in carriers]
    segs, blocks = build_schedule(actual, a.block, a.reps)
    total = blocks[-1]["off"] + a.block if blocks else 0.0
    print(f"{len(carriers)} carriers {actual[0] / 1e3:.2f}-{actual[-1] / 1e3:.2f} kHz, {a.reps} x ({a.block:g} s ON, "
          f"{a.block:g} s OFF) each: schedule {total:.0f} s, run {a.lead + total + a.tail:.0f} s "
          f"({(a.lead + total + a.tail) / 60:.1f} min)")
    import stent_tx as STX
    STX.to_schedule(segs)                       # raises if the TeensyTRX cannot hold it
    if a.mode == "plan":
        return 0

    def schedule_at(s):
        if s < 0 or s >= total:
            return "off (schedule done)" if s >= total else "off"
        k = int(s // (2 * a.block))
        bl = blocks[min(k, len(blocks) - 1)]
        on = (s - bl["on"]) < a.block
        return f"{actual[bl['carrier']] / 1e3:8.2f} kHz {'ON ' if on else 'off'} (rep {bl['rep'] + 1})"

    exe = find_adb(a.adb)
    adb = Adb(exe, resolve_adb_serial(exe, a.serial))
    pad = adb.shell("getprop ro.product.model").strip()
    if "uid=0" not in adb.shell("id", root=True):
        raise SystemExit("no root shell on the Pad (su) - the delta map needs root")
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
        print(f"Pad: {pad}. Finger on the panel and keep it still; the transmitter is OFF during the "
              f"{a.lead:g} s lead.")
        if not a.yes:
            input("Press Enter to start recording (Ctrl+C to cancel) ")
        off0 = clock_offset(adb)
        F, t_dev, t_host, t_first, skipped = record(adb, a.lead, total, a.tail, start_tx, schedule_at)
        off1 = clock_offset(adb)
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
    d = OUT_DIR / f"{a.label}_{stamp}"
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "frames.npz", F=F, t=t, t_host=t_host, t_dev=t_dev)
    meta = dict(args=vars(a), pad_model=pad, adb_serial=adb.base[-1] if len(adb.base) > 1 else None,
                carriers_hz_requested=carriers, carriers_hz_actual=actual, blocks=blocks, schedule_s=total,
                t_first=t_first, clock_offset_start=off0, clock_offset_end=off1, clock_offset_used_s=off,
                frames_without_device_stamp=int((~good).sum()), skipped_frames=skipped,
                tx=tx_info, guard_s=dict(start=GUARD_START_S, end=GUARD_END_S), source="delta",
                recorded_at=stamp)
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"saved {d}")
    analyse(d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
