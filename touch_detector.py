#!/usr/bin/env python3
"""
Real-time touch detector for the maXTouch EVK drive picked up by the stent receiver (TeensyTRX A0), 2026-11-18.
RECEIVE ONLY. The trigger is a DRY RUN: it logs "would trigger AD9833" and never sends a transmit command.

   python touch_detector.py replay                       # all 2026-11-18 recordings -> data_touchdrive/detector/
   python touch_detector.py replay data_touchdrive/td_X  # one or more runs
   python touch_detector.py live --label bench1           # Teensy stream, terminal display, dry-run trigger log
   python touch_detector.py live --source data_touchdrive/td_touch2_20261118_171250 --speed 1
                                                          # the live loop fed from a recording (no hardware)

Detector (D1-D5, chosen 2026-11-18 from the OFFLINE analysis of the same recordings it is replayed on - see
"Limitations"):
  D1 narrowband energy: per 10 ms block, x (mV, block mean removed) times exp(-j 2 pi f0 n / fs) with a running
     sample index (phase continuous), summed in 250-sample pieces (0.25 ms boxcar = +-4 kHz first null around f0,
     wide enough for the sub-burst side lines at +-3 kHz) -> block energy = mean |piece mean|^2.
     f0 = 70.6 kHz (median drive frequency in all screen-touch runs, 70.53-70.66 kHz).
  D2 level = 10 log10 of the MEDIAN block energy of the last 3 blocks (30 ms window, updated every 10 ms; the EVK
     frame is 8.6-9.6 ms, so every window holds >= 3 drive bursts). Changed from "mean of 2 blocks" after the first
     replay: a single impulse in the hover run passed the mean-of-2 window twice (2 false touches of 30 ms); the
     median of 3 ignores one-block spikes. Because this was decided on the same data, its 0 hover false positives
     are not an independent test.
  D3 adaptive no-touch baseline: running median of the last 3 s of levels taken while the state is NO_TOUCH and the
     level is below baseline + OFF_DB (so a rising touch does not drag it up); first 1 s = initialisation (no
     decisions; the screen must not be touched then). Frozen while TOUCH.
  D4 hysteresis: TOUCH when level > baseline + 6 dB, NO_TOUCH when level < baseline + 3 dB (the offline W1 rule).
  D5 debounce: 2 consecutive blocks above ON (20 ms) to enter TOUCH, 3 below OFF (30 ms) to leave it.
  Dry-run trigger: on each TOUCH event, if >= 1 s since the last trigger, log {"would_trigger": "AD9833", "sent": false}.

Live mode: StentTX opened, 'off' confirmed before and after; the serial port is wrapped so only 'x', 'off', '?',
'mono <sps>' can be written (anything else raises). Reads of 16 kB (8 ms of samples) to keep the read delay short.
If the EVK is connected, the tablet's own touch messages (touch_drive_rx.T5Reader, read-only) are logged as ground
truth; latency / misses are computed only from those. Output data_touchdrive/live_<label>_<time>/: events.jsonl,
trace.csv (every block), meta.json, truth.json, and with --save-raw rx.npy + clock.npy for later replay.

Limitations (see the report printed by `replay`): no tablet touch times in any recording so far -> no latency or
miss rate from timing; thresholds tuned on the same data they are replayed on; 1 MS/s with no anti-alias filter (the
detector band also receives 1 MHz +- 70.6 kHz); the drive frequency is a property of this EVK configuration.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import json
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import touch_drive_rx as T                                          # noqa: E402  (parse_pairs, sample_times, T5Reader, t5_intervals)

D = HERE / "data_touchdrive"
MV = 3300 / 4096
VERSION = "touchdet-1"
RUNS = {                                                            # condition as corrected by the user
    "td_ctl_evk_unplugged_20261118_163339": ("EVK unplugged", "none"),
    "td_ctl_notouch_20261118_163617": ("EVK on, no touch", "none"),
    "td_ctl_hover_20261118_171454": ("hover 1 cm (6 prompted)", "none"),
    "td_ctl_bezel_20261118_170426": ("bezel touch (6 prompted)", "none"),
    "td_touch_20261118_165506": ("screen touch run A (6 prompted)", "screen6"),
    "td_ctl_bezel_20261118_165746": ("screen touch run B (6 prompted; label 'ctl_bezel')", "screen6"),
    "td_touch2_20261118_171250": ("screen touch run C (6 prompted)", "screen6"),
    "td_truthcheck_20261118_171223": ("screen taps (user: 3-4)", "taps3-4"),
}


# ------------------------------------------------------------------------------------------------ detector
class Detector:
    def __init__(self, fs=1e6, f0=70.6e3, block_s=0.010, win_blocks=3, piece=250, on_db=6.0, off_db=3.0,
                 on_blocks=2, off_blocks=3, base_s=3.0, init_s=1.0):
        self.fs, self.f0, self.N, self.piece = fs, f0, int(round(block_s * fs)), piece
        self.on_db, self.off_db, self.on_blocks, self.off_blocks = on_db, off_db, on_blocks, off_blocks
        self.win = collections.deque(maxlen=win_blocks)
        self.base_hist = collections.deque(maxlen=int(round(base_s / block_s)))
        self.init_blocks = int(round(init_s / block_s))
        self.buf = np.zeros(0)
        self.n = 0                                      # sample index of buf[0]
        self.k = 0                                      # blocks done
        self.state = "INIT"
        self.baseline = np.nan
        self.cnt = 0
        self.run_start = None
        self.params = dict(fs=fs, f0=f0, block_s=block_s, win_blocks=win_blocks, piece=piece, on_db=on_db,
                           off_db=off_db, on_blocks=on_blocks, off_blocks=off_blocks, base_s=base_s, init_s=init_s)

    def block_energy(self, x, n0):
        x = x - x.mean()
        z = x * np.exp(-2j * np.pi * self.f0 * (n0 + np.arange(len(x))) / self.fs)
        p = z[:len(z) // self.piece * self.piece].reshape(-1, self.piece).mean(1)
        return float(np.mean(np.abs(p) ** 2))

    def feed(self, x_mv):
        """x_mv: next samples (mV). Returns a list of per-block dicts (s = sample index at the block end)."""
        self.buf = np.concatenate([self.buf, x_mv]) if len(self.buf) else np.asarray(x_mv, float)
        out = []
        while len(self.buf) >= self.N:
            blk, self.buf = self.buf[:self.N], self.buf[self.N:]
            e = self.block_energy(blk, self.n)
            self.n += self.N
            self.k += 1
            self.win.append(e)
            raw_db = 10 * np.log10(e + 1e-30)
            lvl = 10 * np.log10(np.median(self.win) + 1e-30)
            ev = None
            if self.state == "INIT":
                self.base_hist.append(lvl)
                if len(self.base_hist) >= self.init_blocks:
                    self.baseline = float(np.median(self.base_hist))
                    self.state = "NO_TOUCH"
            else:
                m = lvl - self.baseline
                if self.state == "NO_TOUCH":
                    if m > self.on_db:
                        if self.cnt == 0:
                            self.run_start = self.n - self.N
                        self.cnt += 1
                        if self.cnt >= self.on_blocks:
                            self.state, self.cnt, ev = "TOUCH", 0, "TOUCH"
                    else:
                        self.cnt = 0
                        if m < self.off_db:
                            self.base_hist.append(lvl)
                            self.baseline = float(np.median(self.base_hist))
                else:
                    if m < self.off_db:
                        if self.cnt == 0:
                            self.run_start = self.n - self.N
                        self.cnt += 1
                        if self.cnt >= self.off_blocks:
                            self.state, self.cnt, ev = "NO_TOUCH", 0, "RELEASE"
                    else:
                        self.cnt = 0
            out.append(dict(s=self.n, raw_db=raw_db, level_db=lvl, baseline_db=self.baseline, state=self.state,
                            event=ev, onset_s=self.run_start if ev else None))
        return out


class DryRunTrigger:
    """logs what a real trigger would do; never touches the transmitter"""

    def __init__(self, log, refractory_s=1.0):
        self.log, self.refr, self.last = log, refractory_s, -1e9
        self.fired = []

    def on_event(self, ev, t_host, extra):
        rec = dict(t=t_host, event=ev, **extra)
        if ev == "TOUCH":
            ok = t_host - self.last >= self.refr
            rec.update(would_trigger="AD9833 frame" if ok else None, sent=False,
                       note="dry run - no transmit command exists in this tool" if ok else "refractory")
            if ok:
                self.last = t_host
                self.fired.append(t_host)
        if self.log:
            self.log.write(json.dumps(rec) + "\n")
            self.log.flush()
        return rec


# ------------------------------------------------------------------------------------------------ replay
def replay_run(d, det_kw=None, chunk=8192):
    d = Path(d)
    meta = json.load(open(d / "meta.json"))
    truth = json.load(open(d / "truth.json"))
    fs = float(meta["sps"])
    raw = np.load(d / "rx.npy", mmap_mode="r")
    _, tsec = T.sample_times(np.load(d / "clock.npy"), fs, 2)
    t0 = tsec[0]
    det = Detector(fs=fs, **(det_kw or {}))
    rows = []
    for i in range(0, len(raw), chunk):
        rows += det.feed(np.asarray(raw[i:i + chunk], float) * MV)
    t = np.array([r["s"] / fs for r in rows])
    ev = [dict(event=r["event"], t=r["s"] / fs, onset=r["onset_s"] / fs, level_db=r["level_db"], baseline_db=r["baseline_db"])
          for r in rows if r["event"]]
    trig = DryRunTrigger(None)
    for e in ev:
        trig.on_event(e["event"], e["t"], {})
    # ground truth that exists: tablet controller messages (none so far), else nothing timed
    tiv = []
    if truth.get("t5"):
        tiv = [(s - t0, e - t0) for s, e in T.t5_intervals(truth["t5"], truth.get("t5_touch_ids", []), tsec[0] + len(raw) / fs)]
    return dict(run=d.name, rows=rows, t=t, events=ev, triggers=trig.fired, truth_intervals=tiv,
                duration_s=len(raw) / fs, params=det.params,
                prompts=[(p["t"] - t0, p["state"]) for p in truth.get("prompts", [])])


def rise_delays(R, which, look_s=0.3):
    """algorithmic delay: from the first single 10 ms block whose own energy crosses the ON (TOUCH) / OFF (RELEASE)
    line, within look_s before the event and contiguous up to it, to the detection. Not a touch latency."""
    rows, out = R["rows"], []
    th = R["params"]["on_db"] if which == "TOUCH" else R["params"]["off_db"]
    idx = {r["s"]: i for i, r in enumerate(rows)}
    for e in R["events"]:
        if e["event"] != which:
            continue
        i = idx[int(round(e["t"] * R["params"]["fs"]))]
        j = i
        while j > 0 and (i - j) * R["params"]["block_s"] < look_s:
            m = rows[j - 1]["raw_db"] - rows[j - 1]["baseline_db"]
            if (m > th) if which == "TOUCH" else (m < th):
                j -= 1
            else:
                break
        out.append(round((i - j + 1) * R["params"]["block_s"] * 1e3))
    return out[:12]


def score(R, kind):
    """only what the available ground truth supports"""
    tou = [e for e in R["events"] if e["event"] == "TOUCH"]
    rel = [e for e in R["events"] if e["event"] == "RELEASE"]
    st = np.array([r["state"] for r in R["rows"]])
    lv = np.array([r["level_db"] - r["baseline_db"] if np.isfinite(r["baseline_db"]) else np.nan for r in R["rows"]])
    out = dict(run=R["run"], duration_s=round(R["duration_s"], 1), touch_events=len(tou), release_events=len(rel),
               dry_run_triggers=len(R["triggers"]), share_time_touch=round(float(np.mean(st == "TOUCH")), 3),
               margin_db_p50_notouch=round(float(np.nanmedian(lv[st == "NO_TOUCH"])), 1) if (st == "NO_TOUCH").any() else None,
               margin_db_p50_touch=round(float(np.nanmedian(lv[st == "TOUCH"])), 1) if (st == "TOUCH").any() else None,
               detect_delay_ms=rise_delays(R, "TOUCH"), release_delay_ms=rise_delays(R, "RELEASE"))
    if R["truth_intervals"]:
        out["truth"] = "tablet controller messages"                   # (none of the 2026-11-18 runs has them)
    elif kind == "none":
        mins = R["duration_s"] / 60
        out.update(truth="no SCREEN touch during the whole run (protocol)", false_positive_touches=len(tou),
                   false_positives_per_min=round(len(tou) / mins, 2), latency="n/a", missed="n/a")
    elif kind == "screen6":
        out.update(truth="6 screen holds instructed; touch TIMES unknown (no tablet messages, prompts unreliable)",
                   expected_touches=6, latency="not measurable", missed="count only")
    elif kind == "taps3-4":
        out.update(truth="3-4 taps (user's count); times unknown", expected_touches="3-4", latency="not measurable",
                   missed="count only")
    return out


def replay_main(paths):
    out = D / "detector"
    out.mkdir(exist_ok=True)
    names = [Path(p).name for p in paths] if paths else list(RUNS)
    res, scores = [], []
    for n in names:
        R = replay_run(D / n)
        kind = RUNS.get(n, (n, "unknown"))[1]
        s = score(R, kind)
        s["condition"] = RUNS.get(n, (n,))[0]
        res.append(R)
        scores.append(s)
        with open(out / f"events_{n}.jsonl", "w") as fh:
            for e in R["events"]:
                fh.write(json.dumps(e) + "\n")
        print(f"{s['condition']:50s} TOUCH {s['touch_events']:2d}  RELEASE {s['release_events']:2d}  "
              f"touch time {100 * s['share_time_touch']:5.1f} %  margin no-touch/touch "
              f"{s['margin_db_p50_notouch']}/{s['margin_db_p50_touch']} dB"
              + (f"  FALSE POSITIVES {s['false_positive_touches']} ({s['false_positives_per_min']}/min)"
                 if "false_positive_touches" in s else f"  expected {s.get('expected_touches')}"))
    json.dump(dict(version=VERSION, params=res[0]["params"], runs=scores), open(out / "replay_summary.json", "w"), indent=1)
    figure(res, scores, out / "replay_all.png")
    dl = [v for s in scores for v in s["detect_delay_ms"]]
    rl = [v for s in scores for v in s["release_delay_ms"]]
    if dl:
        print(f"algorithmic delay (first 10 ms block over the line -> event): touch median {np.median(dl):.0f} ms "
              f"(range {min(dl)}-{max(dl)}), release median {np.median(rl):.0f} ms (range {min(rl)}-{max(rl)})")
    print(f"-> {out}")
    return scores


def figure(res, scores, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(len(res), 1, figsize=(12, 1.9 * len(res)), sharex=True)
    ax = np.atleast_1d(ax)
    for a, R, s in zip(ax, res, scores):
        t = R["t"]
        lv = np.array([r["level_db"] for r in R["rows"]])
        bl = np.array([r["baseline_db"] for r in R["rows"]])
        st = np.array([r["state"] for r in R["rows"]])
        for i0, i1 in T_runs(st == "TOUCH"):
            a.axvspan(t[i0], t[min(i1, len(t) - 1)], color="#2e8b57", alpha=0.18, lw=0)
        a.plot(t, lv, lw=0.5, color="#1f5fbf", label="level (median of 3 x 10 ms)")
        a.plot(t, bl, lw=1.0, color="#555", label="baseline")
        a.plot(t, bl + 6, lw=0.7, color="#c0602a", ls="--", label="ON +6 dB")
        a.plot(t, bl + 3, lw=0.7, color="#c0602a", ls=":", label="OFF +3 dB")
        for tp, stp in R["prompts"]:
            if stp == 1:
                a.axvline(tp, color="#999", lw=0.5, ls=":")
        fp = f"   false positives {s['false_positive_touches']}" if "false_positive_touches" in s else \
             f"   TOUCH events {s['touch_events']} (expected {s.get('expected_touches')})"
        a.set_title(f"{s['condition']}{fp}", loc="left", fontsize=9)
        a.set_ylabel("dB", fontsize=8)
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(labelsize=7)
    ax[0].legend(fontsize=7, ncol=4, frameon=False, loc="upper right")
    ax[-1].set_xlabel("time [s] (green = detector state TOUCH; grey dotted = prompt to touch, informational only)")
    fig.suptitle("Real-time 70.6 kHz touch detector replayed on the 2026-11-18 recordings (causal, sample by sample)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def T_runs(mask):
    m = np.r_[False, mask, False].astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


# ------------------------------------------------------------------------------------------------ live
SAFE = re.compile(r"^(x|off|\?|mono \d+)$")


class SafeTeensy:
    """TeensyTRX via stent_tx.StentTX, with every write checked against a receive-only whitelist"""

    def __init__(self, port):
        import stent_tx as STX
        self.tx = STX.StentTX(port)
        raw_write = self.tx.dev.write

        def guarded(data):
            for line in data.decode(errors="replace").splitlines():
                if line.strip() and not SAFE.match(line.strip()):
                    raise RuntimeError(f"blocked non-receive command {line!r} (touch_detector is receive only)")
            return raw_write(data)
        self.tx.dev.write = guarded
        self.off_before = self.tx.off()
        self.info = self.tx.info()["line"]

    def start(self, sps):
        self.tx.dev.reset_input_buffer()
        self.tx.dev.write(f"mono {int(sps)}\n".encode())

    def read(self, n):
        return self.tx.dev.read(n)

    def stop(self):
        try:
            self.tx.dev.write(b"x\n")
            time.sleep(0.4)
            tail = self.tx.dev.read(1 << 16).decode(errors="replace")
            self.tx.dev.reset_input_buffer()
            sl = re.findall(r"slips\s+(\d+)", tail)
            off = self.tx.off()
        finally:
            self.tx.dev.close()
        return dict(slips=int(sl[-1]) if sl else None, off_after=off)


class FileSource:
    """feeds a recording through the same byte path at --speed x real time (no hardware)"""

    def __init__(self, d, speed):
        self.x = np.load(Path(d) / "rx.npy", mmap_mode="r")
        self.i, self.speed, self.t0 = 0, speed, None
        self.info, self.off_before = f"replay of {Path(d).name}", "n/a"

    def start(self, sps):
        self.sps, self.t0 = sps, time.time()

    def read(self, n):
        due = int((time.time() - self.t0) * self.sps * self.speed)
        k = min(n // 2, max(0, due - self.i), len(self.x) - self.i)
        if k <= 0:
            time.sleep(0.002)
            return b"" if self.i < len(self.x) else None
        v = (np.asarray(self.x[self.i:self.i + k], np.uint16) << 2)
        self.i += k
        b = np.empty(2 * k, np.uint8)
        b[0::2] = 0x80 | (v >> 7) & 0x7F
        b[1::2] = v & 0x7F
        return b.tobytes()

    def stop(self):
        return dict(slips=None, off_after="n/a")


def bar(m, lo=-3.0, hi=15.0, w=30):
    f = int(round((min(max(m, lo), hi) - lo) / (hi - lo) * w))
    return "#" * f + "-" * (w - f)


FROZEN_DETECTOR_SHA = "87fc749001d6"   # Detector source + default params, frozen 2026-11-18 19:00 before the bench test


def detector_fingerprint():
    import hashlib
    import inspect
    src = inspect.getsource(Detector) + json.dumps(Detector().params, sort_keys=True)
    return hashlib.sha256(src.encode()).hexdigest()[:12]


# ------------------------------------------------------------------------------------------------ bench protocol
PROTOCOLS = {
    # (phase, contacts, on_s, off_s, what to do while ON) ; contacts 0 = one stretch of off_s without contact
    "bench_v1": [("init", 0, 0, 10, None),
                 ("A_nocontact", 0, 0, 60, None),
                 ("B_taps", 20, 2, 4, "PLACE the artificial finger on the SCREEN"),
                 ("C_holds", 10, 6, 6, "PLACE the artificial finger on the SCREEN and hold"),
                 ("D_bezel", 10, 4, 6, "PLACE the artificial finger on the BEZEL (not the screen)"),
                 ("E_hover", 10, 4, 6, "PLACE the artificial finger on the 1 cm SPACER (not touching the screen)"),
                 ("F_nocontact", 0, 0, 60, None)],
    "smoke": [("init", 0, 0, 5, None), ("B_taps", 3, 2, 3, "PLACE the artificial finger on the SCREEN"),
              ("F_nocontact", 0, 0, 5, None)],
}


def protocol_schedule(name):
    ev, t = [], 0.0
    for ph, n, on, off, what in PROTOCOLS[name]:
        ev.append((t, ph, "PHASE", f"--- phase {ph} ---"))
        if n == 0:
            ev.append((t, ph, "OFF", "NO CONTACT - keep everything away from the tablet"))
            t += off
        for k in range(n):
            ev.append((t, ph, "ON", f"[{ph} {k + 1}/{n}] {what}"))
            t += on
            ev.append((t, ph, "OFF", f"[{ph} {k + 1}/{n}] LIFT - away from the tablet"))
            t += off
    ev.append((t, "end", "END", "END"))
    return ev, t


# ------------------------------------------------------------------------------------------------ live
def live(a):
    fp = detector_fingerprint()
    if FROZEN_DETECTOR_SHA and fp != FROZEN_DETECTOR_SHA:
        print(f"!!! detector code / parameters CHANGED since the freeze ({fp} != {FROZEN_DETECTOR_SHA}): "
              f"this session is not an independent test of the frozen detector")
    require_truth = a.require_truth or bool(a.protocol)
    t5, chip = None, {}
    if not a.source and not a.no_tablet_truth:
        chip = T.chip_info()
        if chip.get("touch_reports_on") is False and require_truth:
            raise SystemExit("tablet touch reports are OFF (T100 RPTEN cleared) - unplug/replug the EVK, then retry")
        try:
            t5 = T.T5Reader()
            t5.start()
        except Exception as e:
            if require_truth:
                raise SystemExit(f"tablet touch messages not available ({e}) - the bench test needs them")
            print(f"  (tablet touch messages not available: {e} - no ground truth this session)")
    src = FileSource(a.source, a.speed) if a.source else SafeTeensy(a.port)
    d = D / f"live_{a.label}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    d.mkdir(parents=True)
    sched, total = protocol_schedule(a.protocol) if a.protocol else ([], None)
    seconds = total + 2 if total else a.seconds
    det = Detector(fs=a.sps, f0=a.f0)
    log = open(d / "events.jsonl", "w")
    trig = DryRunTrigger(log, a.refractory)
    trace = open(d / "trace.csv", "w")
    trace.write("sample,t_wall,level_db,baseline_db,state,event\n")
    raw_parts, clock, prompts, clockcheck = [], [], [], []
    print(f"{src.info}\nRECEIVE ONLY - transmitter output: {src.off_before}. Dry-run trigger: nothing is ever sent.\n"
          f"detector {VERSION} fingerprint {fp}   tablet touch messages: {'ON' if t5 else 'off'}"
          f"{'   protocol ' + a.protocol + f' ({total / 60:.1f} min)' if a.protocol else ''}\n"
          f"Keep the tablet untouched for the first {det.params['init_s']:.0f} s (baseline). Ctrl+C to stop.\n")
    if a.protocol and not a.yes:
        input("Artificial finger away from the tablet, phantom and RX electrodes in place. Press Enter to start ")
    src.start(a.sps)
    t_start, n, carry, off_t, last_draw, k_ev, last_cc = time.time(), 0, b"", np.inf, 0.0, 0, 0.0
    stats = {}
    try:
        while seconds is None or time.time() - t_start < seconds:
            now = time.time()
            while k_ev < len(sched) and now - t_start >= sched[k_ev][0]:
                tt, ph, act, txt = sched[k_ev]
                prompts.append(dict(t=t_start + tt, t_shown=now, phase=ph, action=act, text=txt))
                print(f"\r{' ' * 110}\r{'' if act == 'PHASE' else chr(7)}  {now - t_start:8.2f} s  ### {txt}", flush=True)
                k_ev += 1
            if now - last_cc >= 1.0:
                clockcheck.append((now, time.monotonic()))
                last_cc = now
            b = src.read(a.read_bytes)
            if b is None:
                break
            if not b:
                continue
            raw = carry + b
            c, used = T.parse_pairs(np.frombuffer(raw, np.uint8))
            carry = raw[used:]
            if not len(c):
                continue
            now = time.time()
            n += len(c)
            clock.append((now, n))
            off_t = min(off_t, now - n / a.sps)                     # live display only; scoring refits the clock
            if a.save_raw:
                raw_parts.append(c.copy())
            for r in det.feed(c.astype(float) * MV):
                tw = time.time()
                trace.write(f"{r['s']},{tw:.4f},{r['level_db']:.2f},{r['baseline_db']:.2f},{r['state']},{r['event'] or ''}\n")
                if r["event"]:
                    th = off_t + r["s"] / a.sps
                    rec = trig.on_event(r["event"], th, dict(sample=r["s"], t_wall=tw, level_db=round(r["level_db"], 1),
                                                             baseline_db=round(r["baseline_db"], 1)))
                    msg = "TOUCH DETECTED" if r["event"] == "TOUCH" else "RELEASE DETECTED"
                    extra = (" -> would trigger AD9833 (dry run, NOT sent)" if rec.get("would_trigger") else
                             " (refractory, no trigger)") if r["event"] == "TOUCH" else ""
                    print(f"\r{' ' * 110}\r  {th - t_start:8.2f} s  >>> {msg}{extra}", flush=True)
                if now - last_draw > 0.1:
                    if np.isfinite(r["baseline_db"]):
                        m = r["level_db"] - r["baseline_db"]
                        print(f"\r  {now - t_start:8.2f} s  level {r['level_db']:6.1f} dB  baseline {r['baseline_db']:6.1f}  "
                              f"margin {m:+5.1f} [{bar(m)}] {r['state']:9s}", end="", flush=True)
                    else:
                        print(f"\r  {now - t_start:8.2f} s  measuring baseline - do not touch ...", end="", flush=True)
                    last_draw = now
    except KeyboardInterrupt:
        print("\n  stopped (Ctrl+C)")
    finally:
        stats = src.stop()
        if t5 is not None:
            t5.stop.set()
            t5.join(2.0)
        log.close()
        trace.close()
    print()
    np.save(d / "clock.npy", np.array(clock, float))                   # always: the scoring refits the sample clock
    json.dump(prompts, open(d / "prompts.json", "w"), indent=0)
    json.dump(dict(t5=t5.msgs if t5 else [], t5_touch_ids=t5.touch_ids if t5 else [], t5_error=t5.err if t5 else None,
                   t5_poll=t5.poll_stats() if t5 else None, stamp="host time at the start of the T44 poll"),
              open(d / "truth.json", "w"))
    if a.save_raw and raw_parts:
        np.save(d / "rx.npy", np.concatenate(raw_parts))
    meta = dict(version=VERSION, detector_fingerprint=fp, frozen_fingerprint=FROZEN_DETECTOR_SHA, args=vars(a),
                params=det.params, source=src.info, samples=n, t_start=t_start, host_s=time.time() - t_start,
                tx_off_before=src.off_before, **stats, chip=chip, clockcheck=clockcheck,
                dry_run_triggers=len(trig.fired), sent_anything=False)
    json.dump(meta, open(d / "meta.json", "w"), indent=1)
    print(f"saved {d}  (samples {n}, slips {stats.get('slips')}, transmitter after: {stats.get('off_after')}, "
          f"dry-run triggers {len(trig.fired)}, sent nothing)")
    if t5 is not None:
        score_session(d)


# ------------------------------------------------------------------------------------------------ scoring
SCORE_RULES = """
Scoring rules (written 2026-11-18 19:00, before the first bench session; not to be changed after seeing its data):
 S0 validity: firmware slips == 0; tablet poll gap p99 <= 10 ms; time.time() - time.monotonic() drift <= 2 ms over
    the session; detector fingerprint == frozen; >= 1 tablet touch. Any failure is printed and marks the session.
 S1 receiver sample time: lower envelope of (USB arrival host time, samples received) fitted as t = a + n / fs_eff
    (iterative: least squares, keep points on or below the line, repeat; then shift to the lowest point). The
    sample clock rate (ppm) and arrival-lag p50 / p99 are reported. Event time = sample index at the block end.
 S2 tablet touch: touched while any T100 touch report ID has DETECT set (touch_drive_rx.t5_intervals), intervals
    closer than 0.10 s merged, shorter than 0.05 s dropped. Tablet time = start of the poll that found the message.
 S3 a tablet touch [s, e] is DETECTED by the first unused detector TOUCH in [s - 0.5 s, min(e, s + 1.0 s)];
    latency = t_event - s (sample time; negative = the receiver saw it before the tablet reported it) and
    wall latency = time the decision was made - s. RELEASE: first RELEASE in [e - 0.5 s, e + 1.0 s].
 S4 MISS = tablet touch without a detector TOUCH (S3). Hit rate with a 95 % Clopper-Pearson interval.
 S5 FALSE POSITIVE = detector TOUCH not used in S3 and outside every [s - 0.5, e + 1.0]; counted per protocol phase;
    rate per minute over A_nocontact + F_nocontact with a 95 % Poisson interval.
 S6 FRAGMENT = detector TOUCH not used in S3 but inside a tablet touch window (a split detection).
 S7 BEZEL / HOVER: per prompted contact (ON prompt t_on, OFF prompt t_off), triggered if a detector TOUCH falls in
    [t_on - 0.5, t_off + 1.0]; rate = triggered / prompted, 95 % CP interval. Tablet touches during D / E are listed
    (a tablet touch there means the finger reached the active area: that contact is excluded from the rate).
 S8 with --save-raw: the detector is re-run offline on the saved samples; its events must equal the live ones.
"""


def clock_fit(clock, fs):
    t, n = clock[:, 0], clock[:, 1]
    keep = np.ones(len(t), bool)
    for _ in range(20):
        A = np.c_[np.ones(keep.sum()), n[keep]]
        a, b = np.linalg.lstsq(A, t[keep], rcond=None)[0]
        res = t - (a + b * n)
        new = res <= 0
        if new.sum() < 10 or (new == keep).all():
            break
        keep = new
    a += res.min()
    lag = t - (a + b * n)
    return a, 1 / b, dict(ppm=round((1 / b / fs - 1) * 1e6, 1), arrival_lag_ms_p50=round(float(np.median(lag)) * 1e3, 2),
                          arrival_lag_ms_p99=round(float(np.percentile(lag, 99)) * 1e3, 2))


def cp(k, n):
    from scipy.stats import beta
    if n == 0:
        return [None, None]
    return [round(float(beta.ppf(0.025, k, n - k + 1)) if k else 0.0, 3),
            round(float(beta.ppf(0.975, k + 1, n - k)) if k < n else 1.0, 3)]


def poisson_ci(k, minutes):
    from scipy.stats import chi2
    lo = chi2.ppf(0.025, 2 * k) / 2 if k else 0.0
    hi = chi2.ppf(0.975, 2 * k + 2) / 2
    return [round(lo / minutes, 3), round(hi / minutes, 3)]


def score_session(d):
    d = Path(d)
    meta = json.load(open(d / "meta.json"))
    truth = json.load(open(d / "truth.json"))
    prompts = json.load(open(d / "prompts.json")) if (d / "prompts.json").exists() else []
    fs = float(meta["params"]["fs"])
    a0, fs_eff, cst = clock_fit(np.load(d / "clock.npy"), fs)
    tr = [l.rstrip("\n").split(",") for l in open(d / "trace.csv")][1:]
    ev = [dict(event=r[5], t=a0 + int(r[0]) / fs_eff, t_wall=float(r[1])) for r in tr if r[5]]
    tou = [e for e in ev if e["event"] == "TOUCH"]
    rel = [e for e in ev if e["event"] == "RELEASE"]
    raw_iv = T.t5_intervals(truth.get("t5", []), truth.get("t5_touch_ids", []), meta["t_start"] + meta["host_s"])
    iv = []
    for s_, e_ in raw_iv:
        if iv and s_ - iv[-1][1] < 0.10:
            iv[-1] = (iv[-1][0], e_)
        else:
            iv.append((s_, e_))
    iv = [(s_, e_) for s_, e_ in iv if e_ - s_ >= 0.05]
    # phases
    ph_iv = []
    cur = None
    for p in prompts:
        if p["action"] in ("PHASE", "END"):
            if cur:
                ph_iv.append((cur[0], cur[1], p["t"]))
            cur = (p["phase"], p["t"]) if p["action"] == "PHASE" else None
    if cur:                                                         # session stopped before END: close at its end
        ph_iv.append((cur[0], cur[1], meta["t_start"] + meta["host_s"]))

    def phase_of(t):
        for ph, s_, e_ in ph_iv:
            if s_ <= t < e_:
                return ph
        return "outside"
    used, hits, lat, wlat, rlat, missed = set(), [], [], [], [], []
    for s_, e_ in iv:
        j = next((i for i, x in enumerate(tou) if i not in used and s_ - 0.5 <= x["t"] <= min(e_, s_ + 1.0)), None)
        if j is None:
            missed.append(dict(t=round(s_ - meta["t_start"], 2), phase=phase_of(s_), dur_s=round(e_ - s_, 2)))
        else:
            used.add(j)
            lat.append(tou[j]["t"] - s_)
            wlat.append(tou[j]["t_wall"] - s_)
        r = next((x for x in rel if e_ - 0.5 <= x["t"] <= e_ + 1.0), None)
        if r is not None:
            rlat.append(r["t"] - e_)
    fp, frag = [], []
    for i, x in enumerate(tou):
        if i in used:
            continue
        inside = any(s_ - 0.5 <= x["t"] <= e_ + 1.0 for s_, e_ in iv)
        (frag if inside else fp).append(dict(t=round(x["t"] - meta["t_start"], 2), phase=phase_of(x["t"])))
    nc_min = sum(e_ - s_ for ph, s_, e_ in ph_iv if ph in ("A_nocontact", "F_nocontact")) / 60
    nc_fp = sum(1 for f in fp if f["phase"] in ("A_nocontact", "F_nocontact"))

    def contact_rate(phase):
        ons = [p for p in prompts if p["phase"] == phase and p["action"] in ("ON", "OFF")]
        pairs = [(ons[i]["t"], ons[i + 1]["t"]) for i in range(0, len(ons) - 1, 2) if ons[i]["action"] == "ON"]
        tab = [(s_, e_) for s_, e_ in iv if any(on - 0.5 <= s_ <= off + 1.0 for on, off in pairs)]
        valid = [(on, off) for on, off in pairs if not any(on - 0.5 <= s_ <= off + 1.0 for s_, _ in tab)]
        trig_ = sum(1 for on, off in valid if any(on - 0.5 <= x["t"] <= off + 1.0 for x in tou))
        return dict(prompted=len(pairs), excluded_tablet_touch=len(pairs) - len(valid), triggered=trig_,
                    rate=round(trig_ / len(valid), 3) if valid else None, ci95=cp(trig_, len(valid)))
    q = lambda v: [round(float(x), 1) for x in np.percentile(np.array(v) * 1e3, [5, 50, 95])] if v else None
    cc = np.array(meta.get("clockcheck", []))
    drift = float(np.ptp(cc[:, 0] - cc[:, 1]) * 1e3) if len(cc) > 1 else None
    poll = truth.get("t5_poll") or {}
    valid = dict(slips_zero=meta.get("slips") in (0, None), poll_gap_p99_ok=(poll.get("gap_ms_p99") or 99) <= 10,
                 clock_jump_ok=drift is not None and drift <= 2.0,
                 detector_frozen=meta.get("frozen_fingerprint") in (None, meta.get("detector_fingerprint")),
                 tablet_touches_present=len(iv) > 0)
    S = dict(session=d.name, validity=valid, clock=cst, time_minus_monotonic_drift_ms=None if drift is None else round(drift, 2),
             tablet_poll=poll, tablet_touches=len(iv), detected=len(lat), missed=len(missed),
             hit_rate=round(len(lat) / len(iv), 3) if iv else None, hit_ci95=cp(len(lat), len(iv)),
             touch_latency_ms_p5_p50_p95=q(lat), touch_wall_latency_ms_p5_p50_p95=q(wlat),
             release_latency_ms_p5_p50_p95=q(rlat), false_positives=len(fp), false_positives_by_phase=collections.Counter(f["phase"] for f in fp),
             nocontact_minutes=round(nc_min, 2), nocontact_fp=nc_fp,
             nocontact_fp_per_min=round(nc_fp / nc_min, 3) if nc_min else None,
             nocontact_fp_per_min_ci95=poisson_ci(nc_fp, nc_min) if nc_min else None, fragments=len(frag),
             bezel=contact_rate("D_bezel"), hover=contact_rate("E_hover"), missed_list=missed, fp_list=fp[:30])
    if (d / "rx.npy").exists():
        det = Detector(**meta["params"])
        x = np.load(d / "rx.npy", mmap_mode="r")
        off_ev = []
        for i in range(0, len(x), 8192):
            off_ev += [(r["s"], r["event"]) for r in det.feed(np.asarray(x[i:i + 8192], float) * MV) if r["event"]]
        live_ev = [(int(r[0]), r[5]) for r in tr if r[5]]
        S["live_equals_offline_replay"] = off_ev == live_ev
    json.dump(S, open(d / "score.json", "w"), indent=1, default=str)
    print(f"\n== score {d.name}")
    for k_, v in S.items():
        if k_ not in ("missed_list", "fp_list"):
            print(f"  {k_}: {v}")
    bad = [k_ for k_, v in valid.items() if not v]
    if bad:
        print(f"  !!! VALIDITY PROBLEMS: {bad}")
    return S


def main():
    ap = argparse.ArgumentParser(description=__doc__ + SCORE_RULES, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("replay", "live", "score", "fingerprint", "protocol"))
    ap.add_argument("what", nargs="*")
    ap.add_argument("--label", default="live")
    ap.add_argument("--port")
    ap.add_argument("--sps", type=int, default=1_000_000)
    ap.add_argument("--f0", type=float, default=70.6e3)
    ap.add_argument("--seconds", type=float)
    ap.add_argument("--read-bytes", type=int, default=16384, help="serial read size (16 kB = 8 ms of samples)")
    ap.add_argument("--refractory", type=float, default=1.0, help="min time between dry-run triggers [s]")
    ap.add_argument("--save-raw", action="store_true", help="also save rx.npy for the S8 live = replay check")
    ap.add_argument("--no-tablet-truth", action="store_true", help="do not poll the tablet's touch messages")
    ap.add_argument("--require-truth", action="store_true", help="stop if the tablet touch messages are unavailable")
    ap.add_argument("--protocol", choices=sorted(PROTOCOLS), help="prompted bench protocol (implies --require-truth)")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--source", help="live: feed a recording folder instead of the Teensy (no hardware)")
    ap.add_argument("--speed", type=float, default=1.0)
    a = ap.parse_args()
    if a.cmd == "replay":
        replay_main(a.what)
    elif a.cmd == "live":
        live(a)
    elif a.cmd == "score":
        for p_ in a.what:
            score_session(p_)
    elif a.cmd == "fingerprint":
        print(detector_fingerprint(), "(frozen:", FROZEN_DETECTOR_SHA, ")")
    else:
        for name in (a.what or sorted(PROTOCOLS)):
            ev, tot = protocol_schedule(name)
            print(f"{name}: {tot:.0f} s = {tot / 60:.1f} min, "
                  f"{sum(1 for e in ev if e[2] == 'ON')} prompted contacts")
            for t_, ph, act, txt in ev:
                if act in ("PHASE", "END"):
                    print(f"   {t_:6.0f} s  {txt}")


if __name__ == "__main__":
    main()
