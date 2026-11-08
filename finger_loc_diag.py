#!/usr/bin/env python3
"""
Receiver-only finger localisation diagnostic for the maXTouch EVK (ATMXT2952, 41 x 71 nodes). The transmitter is never
opened or started. Decoders and earlier recordings are not touched; output goes to new folders only.

   python finger_loc_diag.py bench                       # sampling rates of each acquisition mode, no finger needed
   python finger_loc_diag.py record --label <name>       # guided finger test (still / known positions / slide / lift)
   python finger_loc_diag.py ratecheck --label <name>    # finger held still; native reporting on/off in 5 s blocks
   python finger_loc_diag.py analyse data_maxtouch/fingerloc_<label>_<time>

Three localisation sources, recorded together with the communication capture (X line 0, Y0-27, the stent runs' nodes):
  native   the controller's own touch coordinates (T100 messages), by one of two routes (--native):
           t5      (default) over the control link, as maxtouch_map.py does (982 messages on 2026-10-21): T44 message
                   count + T5 message reads, done while each row read waits for its scan, so the queue is emptied
                   every few ms. Costs control-link time; whether that slows the row is what ratecheck measures.
                   Coordinates -> nodes with the chip's own T100 origin / size / range / orientation (maxtouch_map).
           hidraw  the EVK's HID digitizer interface 1 (/dev/hidraw6) in a second thread. Recorded in every mode,
                   but gave 0 reports in fingerloc_nat1_20261102_115156 (T100 reports on, nothing drained).
  hold detection: analyse finds the stationary holds in the data (finger down, strongest node unchanged >= 1 s),
           because a person lags timed prompts by seconds and lifts between positions (nat1: 4-5 s late).
  row      the X0 / Y0-27 deltas themselves (what the decoders see)
  patch    optional 2D T37 deltas, X0..X(w-1) x Y0..Y(h-1), read every --patch-every row samples. T37 pages are X-major
           (71 nodes per X line, 64 per page) and every page step waits for a new scan, so a patch costs ~one scan
           per X line and leaves a gap in the row capture.

Chip settings while recording (RAM only, restored at the end): T72 noise suppression off (as in the stent runs);
T100 touch reports stay ON (the stent runs turn them off) because the native coordinates need them. The chip is
calibrated once at the start with the finger LIFTED, so the deltas show the whole touch, not its change since
calibration (stent runs calibrate with the finger on the panel).

Digitizer report (descriptor of interface 1, report ID 1): per contact 1 byte (bit 0 tip switch, bits 3-7 contact
id), X uint16 (0-1023 over 345 mm), 2 pad, Y uint16 (0-1023 over 195 mm), 2 pad, 2 bytes; contacts back to back.
"""
from __future__ import annotations

import argparse
import glob
import datetime
import json
import os
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from maxtouch_hid import DELTAS_MODE, PAGE_UP, MaxTouchHid

# T100 config offsets / bits, as maxtouch_map.py (not imported: it loads tkinter and the TkAgg backend)
T100_CFG1, T100_XORIGIN, T100_XSIZE, T100_XRANGE = 1, 8, 9, 13
T100_YORIGIN, T100_YSIZE, T100_YRANGE = 19, 20, 24
T100_SWITCHXY, T100_INVERTY, T100_INVERTX = 0x20, 0x40, 0x80
T100_DETECT = 0x80

DATADIR = os.path.join(HERE, "data_maxtouch")
VID, PID, DIGI_IFACE = 0x03EB, 0x214E, 1
T6_CALIBRATE, T6_DIAGNOSTIC, T100_RPTEN = 2, 5, 0x02
ROW_Y0, ROW_NY = 0, 28
CONTACT_BYTES = 11
VIS, HOLD_MIN_S, HOLD_SMOOTH = 100.0, 1.0, 25    # touch visible above 100 counts; hold = >= 1 s on one node
RATE_MODES = [                                  # ratecheck blocks: (name, T100 reports on, drain T5 over the link)
    ("t5", True, True),                         # native coordinates over the control link (maxtouch_map route)
    ("rpt_nodrain", True, False),               # reports on, nothing drained (as fingerloc_nat1)
    ("rpt_off", False, False),                  # reports off: the stent runs' setting
]
PROTOCOL = [                                    # (seconds, instruction); positions along the top edge (X line 0)
    (6, "LIFT the finger off the panel (calibration with nothing on it)"),
    (10, "finger DOWN on position A (the corner where it sits in the stent runs), keep STILL"),
    (6, "move to position B (~2 cm further along the top edge) and keep STILL"),
    (6, "move to position C (~2 cm further again) and keep STILL"),
    (6, "move back to position A and keep STILL"),
    (12, "SLIDE slowly back and forth along the top edge between A and C, always touching"),
    (20, "rest the finger on position A and keep STILL (long hold: drift / recalibration check)"),
    (5, "LIFT the finger"),
]


# ---------------------------------------------------------------- chip access
class Link:
    def __init__(self):
        self.m = MaxTouchHid()
        for _ in range(4):
            while self.m._h.read(64):
                pass
            try:
                self.m.load_info(); break
            except Exception:
                time.sleep(0.2)
        i = self.m.id_info
        self.nx, self.ny = i.matrix_x, i.matrix_y
        self.t6, self.t37 = self.m.object_address(6), self.m.object_address(37)
        self.t72, self.t100 = self.m.object_address(72), self.m.object_address(100)
        self.t44, self.t5, self.t5_size = self.m.object_address(44), self.m.object_address(5), self.m.object_size(5)
        self.page = self.m.object_size(37) - 2
        self.orig = {}
        # T100 report IDs (as maxtouch_map): first = screen status, second = reserved, then one per touch slot
        rid, self.t100_rid, self.t100_nrid = 0, None, 0
        for o in self.m.objects:
            if o.type_id == 100 and self.t100_rid is None and o.num_report_ids:
                self.t100_rid, self.t100_nrid = rid + 1, o.num_report_ids
            rid += o.instances * o.num_report_ids
        c = self.m.read_register(self.t100, 26)
        self.geom = dict(cfg1=c[T100_CFG1], xorg=c[T100_XORIGIN], xsize=c[T100_XSIZE] or self.nx,
                         xrange=(c[T100_XRANGE] | c[T100_XRANGE + 1] << 8) or 1023,
                         yorg=c[T100_YORIGIN], ysize=c[T100_YSIZE] or self.ny,
                         yrange=(c[T100_YRANGE] | c[T100_YRANGE + 1] << 8) or 1023)
        self.drain_on = False
        self.msgs = []                          # (time, slot, status, x, y, peak) of every T100 touch message
        self.n_drained = 0                      # every message read from T5 (any object)

    def setup(self, calibrate=True, reports=True):
        self.orig["T72"] = self.m.read_register(self.t72, 1)[0]
        self.m.write_register(self.t72, bytes([0]))
        self.orig["T100"] = self.m.read_register(self.t100, 1)[0]
        self.set_reports(reports)
        if calibrate:
            self.calibrate()

    def set_reports(self, on):
        """T100 RPTEN only (ENABLE must stay set or acquisition stops)"""
        v = self.orig["T100"] | T100_RPTEN if on else self.orig["T100"] & ~T100_RPTEN & 0xFF
        self.m.write_register(self.t100, bytes([v]))

    def drain(self):
        """T44 message count, then that many T5 reads (maxtouch_map._drain); keeps the T100 touch messages"""
        n = self.m.read_register(self.t44, 1)[0]
        for _ in range(n):
            msg = self.m.read_register(self.t5, self.t5_size)
            if msg[0] == 0xFF:                  # queue empty
                break
            self.n_drained += 1
            if self.t100_rid is not None and self.t100_rid + 2 <= msg[0] < self.t100_rid + self.t100_nrid:
                self.msgs.append((time.time(), msg[0] - self.t100_rid - 2, msg[1], msg[2] | msg[3] << 8,
                                  msg[4] | msg[5] << 8, msg[6]))

    def calibrate(self):
        self.m.write_register(self.t6 + T6_CALIBRATE, bytes([1]))
        time.sleep(1.5)

    def restore(self):
        for name, addr in (("T100", self.t100), ("T72", self.t72)):
            if name in self.orig:
                try:
                    self.m.write_register(addr, bytes([self.orig[name]]))
                except Exception as e:
                    print(f"  ! could not restore {name}={self.orig[name]}: {e} (unplug/replug the EVK)")
        self.m.close()

    def step(self, cmd):
        """T6 diagnostic command, wait for the next scan; with drain_on the wait drains touch messages instead of
        only polling (maxtouch_map._step_page)"""
        diag = self.t6 + T6_DIAGNOSTIC
        self.m.write_register(diag, bytes([cmd]))
        deadline = time.perf_counter() + 0.5
        while self.m.read_register(diag, 1)[0] != 0:
            if time.perf_counter() > deadline:
                raise TimeoutError("T6 diagnostic command timed out")
            if self.drain_on:
                self.drain()

    def row(self):
        """X line 0, Y0-27: one fresh scan, one HID transfer (as the stent runs)"""
        self.step(DELTAS_MODE)
        return np.frombuffer(self.m.read_register(self.t37 + 2 + 2 * ROW_Y0, 2 * ROW_NY), "<i2").copy()

    def patch(self, w, h):
        """X0..w-1 x Y0..h-1 deltas, page by page from page 0 (each page = one more scan)"""
        need = {}
        for x in range(w):
            for y in range(h):
                off = (x * self.ny + y) * 2
                need.setdefault(off // self.page, []).append((off % self.page, x, y))
        out = np.zeros((w, h), np.int16)
        for p in range(max(need) + 1):
            self.step(DELTAS_MODE if p == 0 else PAGE_UP)
            if p in need:
                last = max(o for o, _, _ in need[p]) + 2
                raw = self.m.read_register(self.t37, 2 + last)
                if raw[0] != DELTAS_MODE or raw[1] != p:
                    raise RuntimeError(f"unexpected T37 page header {bytes(raw[:2]).hex()} (page {p})")
                for o, x, y in need[p]:
                    out[x, y] = int.from_bytes(raw[2 + o:4 + o], "little", signed=True)
        return out


class Digitizer(threading.Thread):
    """reads the EVK's digitizer interface; keeps every contact report with its arrival time"""

    def __init__(self):
        super().__init__(daemon=True)
        nodes = glob.glob(f"/sys/bus/usb/devices/*:1.{DIGI_IFACE}/*:{VID:04X}:{PID:04X}.*/hidraw/hidraw*")
        if not nodes:
            raise RuntimeError("digitizer interface (hidraw) not found")
        self.fd = os.open("/dev/" + os.path.basename(nodes[0]), os.O_RDONLY | os.O_NONBLOCK)
        self.rows, self.stop, self.n_reports = [], threading.Event(), 0

    def run(self):
        while not self.stop.is_set():
            try:
                r = os.read(self.fd, 64)
            except BlockingIOError:
                r = b""
            if not r:
                time.sleep(0.0005); continue
            now = time.time(); self.n_reports += 1
            if r[0] != 1:
                continue
            body = r[1:]
            for k in range(0, len(body) - CONTACT_BYTES + 1, CONTACT_BYTES):
                c = body[k:k + CONTACT_BYTES]
                tip, cid = c[0] & 1, c[0] >> 3
                x, y = c[1] | c[2] << 8, c[5] | c[6] << 8
                if tip or x or y:
                    self.rows.append((now, cid, tip, x, y))

    def close(self):
        self.stop.set(); self.join(1.0); os.close(self.fd)


# ---------------------------------------------------------------- bench
def bench(a):
    L = Link(); L.setup(calibrate=False)
    res = {}
    try:
        def rate(fn, secs):
            n, t0 = 0, time.perf_counter()
            while time.perf_counter() - t0 < secs:
                fn(); n += 1
            return n / (time.perf_counter() - t0)
        res["row 28 nodes"] = rate(L.row, a.seconds)
        D = Digitizer(); D.start()
        res["row 28 nodes + digitizer thread open"] = rate(L.row, a.seconds)
        D.close()
        L.drain_on = True
        res["row 28 nodes + T5 drain in the scan wait (no finger: T44 polls only)"] = rate(L.row, a.seconds)
        L.drain_on = False
        for w, h in ((2, 8), (4, 12), (6, 28)):
            res[f"patch X0-{w - 1} x Y0-{h - 1} only (frames/s)"] = rate(lambda: L.patch(w, h), a.seconds)
        for every in (25, 125):
            gaps, ts, k = [], [], 0
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < a.seconds:
                if k and k % every == 0:
                    g = time.perf_counter(); L.patch(4, 12); gaps.append(time.perf_counter() - g)
                L.row(); ts.append(time.perf_counter()); k += 1
            res[f"row + 4x12 patch every {every} rows: row samples/s"] = len(ts) / (ts[-1] - t0)
            res[f"row + 4x12 patch every {every} rows: gap per patch (ms)"] = 1e3 * float(np.median(gaps))
    finally:
        L.restore()
    for k, v in res.items():
        print(f"  {k:55s} {v:7.1f}")
    out = os.path.join(DATADIR, f"fingerloc_bench_{datetime.datetime.now():%Y%m%d_%H%M%S}.json")
    json.dump(res, open(out, "w"), indent=1)
    print(f"saved {os.path.relpath(out, HERE)}")


# ---------------------------------------------------------------- record
def record(a):
    total = sum(s for s, _ in PROTOCOL)
    print(f"Receiver-only finger test, ~{total} s. The transmitter is not used: leave it OFF / disconnected.\n"
          f"Follow the prompts; each step is timed. Row capture X0 Y0-27"
          + (f", 2D patch X0-{a.patch_w - 1} x Y0-{a.patch_h - 1} every {a.patch_every} rows" if a.patch_every else "")
          + f", native touch coordinates ({a.native}) in parallel.")
    input("Lift your finger off the panel and press Enter to calibrate ")
    L = Link(); L.setup(calibrate=True)
    L.drain_on = a.native == "t5"
    D = Digitizer(); D.start()
    t, X, P, phases = [], [], [], []
    try:
        t_start = time.time()
        for secs, text in PROTOCOL:
            print(f"  [{time.time() - t_start:5.1f} s] {text} ({secs} s)")
            phases.append((time.time(), secs, text))
            t_end, k = time.time() + secs, 0
            while time.time() < t_end:
                if a.patch_every and k and k % a.patch_every == 0:
                    P.append((time.time(), L.patch(a.patch_w, a.patch_h)))
                X.append(L.row()); t.append(time.time()); k += 1
    except KeyboardInterrupt:
        print("  stopped early")
    finally:
        D.close(); L.restore()
    d = os.path.join(DATADIR, f"fingerloc_{a.label}_{datetime.datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(d)
    np.savez_compressed(os.path.join(d, "capture.npz"), t=np.array(t), X=np.array(X, np.int16),
                        touch=np.array(D.rows, float).reshape(-1, 5), t5=np.array(L.msgs, float).reshape(-1, 6),
                        patch_t=np.array([p[0] for p in P]), patch=np.array([p[1] for p in P], np.int16).reshape(len(P), a.patch_w, a.patch_h) if P else np.zeros((0, 1, 1)))
    json.dump(dict(args=vars(a), matrix=[L.nx, L.ny], row=dict(x=0, y0=ROW_Y0, ny=ROW_NY), phases=phases,
                   digitizer_reports=D.n_reports, t5_messages=L.n_drained, t100_geometry=L.geom, chip_orig=L.orig),
              open(os.path.join(d, "meta.json"), "w"), indent=1)
    print(f"saved {os.path.relpath(d, HERE)}: {len(t)} row samples ({len(t) / (t[-1] - t[0]):.1f}/s), "
          f"{len(L.msgs)} T5 touch messages, {len(D.rows)} digitizer contacts, {len(P)} patch frames")
    analyse(d)


# ---------------------------------------------------------------- ratecheck
def ratecheck(a):
    """does reading native coordinates slow the row? Finger held still on A the whole time; the reporting mode
    cycles in --block s blocks through RATE_MODES, --cycles times, so slow drift hits every mode alike"""
    total = len(RATE_MODES) * a.cycles * a.block
    print(f"Receiver-only rate check, ~{total:.0f} s. The transmitter is not used: leave it OFF / disconnected.\n"
          f"Modes, {a.block:g} s each, {a.cycles} cycles: " + ", ".join(m for m, _, _ in RATE_MODES))
    input("Lift your finger off the panel and press Enter to calibrate ")
    L = Link(); L.setup(calibrate=True)
    D = Digitizer(); D.start()
    t, X, mode, blocks = [], [], [], []
    try:
        input("Put the finger on position A and keep it STILL until the end; press Enter to start ")
        for c in range(a.cycles):
            for i, (name, rpt, drain) in enumerate(RATE_MODES):
                L.set_reports(rpt); L.drain_on = drain
                n5, nd = len(L.msgs), D.n_reports
                t0 = time.time()
                print(f"  [{c + 1}/{a.cycles}] {name:12s}", end="", flush=True)
                while time.time() < t0 + a.block:
                    X.append(L.row()); t.append(time.time()); mode.append(i)
                blocks.append(dict(cycle=c, mode=name, t0=t0, t1=time.time(), t5_touch_messages=len(L.msgs) - n5,
                                   digitizer_reports=D.n_reports - nd))
                print(f" {np.sum(np.array(t) >= t0) / a.block:6.1f} rows/s, {len(L.msgs) - n5:4d} T5 touch msgs, "
                      f"{D.n_reports - nd:4d} digitizer reports")
    except KeyboardInterrupt:
        print("  stopped early")
    finally:
        L.drain_on = False
        D.close(); L.restore()
    d = os.path.join(DATADIR, f"fingerloc_ratecheck_{a.label}_{datetime.datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(d)
    np.savez_compressed(os.path.join(d, "capture.npz"), t=np.array(t), X=np.array(X, np.int16), mode=np.array(mode),
                        touch=np.array(D.rows, float).reshape(-1, 5), t5=np.array(L.msgs, float).reshape(-1, 6))
    json.dump(dict(args=vars(a), kind="ratecheck", modes=RATE_MODES, blocks=blocks, matrix=[L.nx, L.ny],
                   row=dict(x=0, y0=ROW_Y0, ny=ROW_NY), digitizer_reports=D.n_reports, t5_messages=L.n_drained,
                   t100_geometry=L.geom, chip_orig=L.orig), open(os.path.join(d, "meta.json"), "w"), indent=1)
    print(f"saved {os.path.relpath(d, HERE)}")
    analyse_ratecheck(d)


def analyse_ratecheck(d, settle_s=0.5):
    """per mode: row rate (first settle_s of each block left out: a full queue is drained there), gaps, native rate"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    meta = json.load(open(os.path.join(d, "meta.json")))
    z = np.load(os.path.join(d, "capture.npz"))
    t, X, T5 = z["t"], z["X"].astype(float), z["t5"]
    out = {}
    for name, _, _ in meta["modes"]:
        r, dt, n5, nd, vis, secs = [], [], 0, 0, [], 0.0
        for b in (b for b in meta["blocks"] if b["mode"] == name):
            k = (t >= b["t0"] + settle_s) & (t < b["t1"])
            tb = t[k]
            if len(tb) < 3:
                continue
            secs += tb[-1] - tb[0]
            r.append((len(tb) - 1) / (tb[-1] - tb[0])); dt.append(np.diff(tb))
            n5 += int(np.sum((T5[:, 0] >= b["t0"] + settle_s) & (T5[:, 0] < b["t1"]))) if len(T5) else 0
            nd += b["digitizer_reports"]; vis.append(np.mean(X[k].max(axis=1) > VIS))
        if not r:
            continue
        dt = np.concatenate(dt)
        out[name] = dict(blocks=len(r), row_rate_per_block=[round(v, 2) for v in r], row_rate=float(np.mean(r)),
                         median_interval_ms=1e3 * float(np.median(dt)), p99_interval_ms=1e3 * float(np.percentile(dt, 99)),
                         intervals_over_12ms=int(np.sum(dt > 0.012)), touch_visible=float(np.mean(vis)),
                         t5_touch_msgs_per_s=n5 / secs, digitizer_reports=nd)
    ref = out.get("rpt_off", {}).get("row_rate")
    json.dump(dict(modes=out, settle_s=settle_s), open(os.path.join(d, "ratecheck.json"), "w"), indent=1)
    print(f"\n{'mode':12s} {'rows/s':>7s} {'vs off':>7s} {'median ms':>9s} {'p99 ms':>7s} {'>12 ms':>6s} "
          f"{'T5 msg/s':>8s} {'digitizer':>9s} {'touch vis':>9s}")
    for name, r in out.items():
        print(f"{name:12s} {r['row_rate']:7.1f} {(f'{100 * (r['row_rate'] / ref - 1):+6.1f}%' if ref else '-'):>7s} "
              f"{r['median_interval_ms']:9.2f} {r['p99_interval_ms']:7.2f} {r['intervals_over_12ms']:6d} "
              f"{r['t5_touch_msgs_per_s']:8.1f} {r['digitizer_reports']:9d} {r['touch_visible']:9.0%}")
    colors = {"t5": "#1f5fbf", "rpt_nodrain": "#e66100", "rpt_off": "#555555"}
    fig, ax = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
    for b in meta["blocks"]:
        k = (t >= b["t0"]) & (t < b["t1"])
        tb = t[k]
        if len(tb) > 2:
            ax[0].plot(tb[1:] - t[0], 1 / np.diff(tb), lw=0.5, color=colors.get(b["mode"], "k"),
                       label=b["mode"] if b["cycle"] == 0 else None)
        ax[1].axvspan(b["t0"] - t[0], b["t1"] - t[0], color=colors.get(b["mode"], "k"), alpha=0.12, lw=0)
    if len(T5):
        ax[1].plot(T5[:, 0] - t[0], [touch_node(r[3], r[4], meta["t100_geometry"])[1] for r in T5], ".", ms=2,
                   color="#00a000", label="native (T5), Y line")
    pk = X.max(axis=1)
    w_ = np.where(X >= 0.25 * pk[:, None], X, 0)
    ax[1].plot(t - t[0], np.where(pk > VIS, (w_ * np.arange(X.shape[1])).sum(1) / np.maximum(w_.sum(1), 1e-9), np.nan),
               lw=0.8, color="black", label="row delta centroid")
    ax[0].set_ylabel("row samples/s (per interval)"); ax[0].set_ylim(0, 160); ax[0].legend(fontsize=8, loc="lower right")
    ax[1].set_ylabel("Y node"); ax[1].set_xlabel("time (s)"); ax[1].legend(fontsize=8, loc="upper right")
    ax[0].set_title(f"{os.path.basename(d)}: row rate per reporting mode (finger still)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(d, "ratecheck.png"), dpi=140); plt.close(fig)
    print(f"figure {os.path.relpath(os.path.join(d, 'ratecheck.png'), HERE)}")


# ---------------------------------------------------------------- analyse
EVK_T100 = dict(cfg1=0xF1, xorg=0, xsize=41, xrange=1023, yorg=0, ysize=71, yrange=1023)   # read 2026-11-02


def touch_node(x, y, g=None):
    """T100 coordinate -> (X line, Y line) with the chip's T100 origin / size / range / orientation, as
    maxtouch_map.touch_to_node. g = the geometry saved with the recording (EVK_T100 for recordings made before it was
    saved). Used for both native routes; that the digitizer interface carries the same coordinates is assumed."""
    g = g or EVK_T100
    if g["cfg1"] & T100_SWITCHXY:
        x, y = y, x
    fx, fy = x / g["xrange"], y / g["yrange"]
    if g["cfg1"] & T100_INVERTX:
        fx = 1 - fx
    if g["cfg1"] & T100_INVERTY:
        fy = 1 - fy
    return g["xorg"] + fx * (g["xsize"] - 1), g["yorg"] + fy * (g["ysize"] - 1)


def row_estimates(X):
    """per sample: strongest node, its delta, centroid of the deltas >= 25 % of that peak (nan if no touch)"""
    pk = X.max(axis=1)
    w = np.where(X >= 0.25 * pk[:, None], X, 0) * (pk[:, None] > VIS)
    rc = np.where(pk > VIS, (w * np.arange(X.shape[1])).sum(1) / np.maximum(w.sum(1), 1e-9), np.nan)
    return np.where(pk > VIS, X.argmax(axis=1), -1), pk, rc


def find_holds(t, rpeak):
    """stationary holds from the data: finger down and the strongest node (median-filtered over HOLD_SMOOTH samples,
    ~0.2 s) unchanged for >= HOLD_MIN_S. Returns [(i0, i1)] sample index ranges."""
    from scipy.ndimage import median_filter
    sm = median_filter(rpeak, HOLD_SMOOTH, mode="nearest")
    out, i0 = [], 0
    for i in range(1, len(sm) + 1):
        if i == len(sm) or sm[i] != sm[i0]:
            if sm[i0] >= 0 and t[i - 1] - t[i0] >= HOLD_MIN_S:
                out.append((i0, i))
            i0 = i
    return out


def native_points(z, meta):
    """native contacts in touch: (source, times, (X line, Y line)); T5 messages preferred, else the digitizer"""
    g = meta.get("t100_geometry")
    T5 = z["t5"] if "t5" in z.files else np.zeros((0, 6))
    if len(T5):
        k = (T5[:, 2].astype(int) & T100_DETECT) > 0
        return "T5", T5[k, 0], np.array([touch_node(x, y, g) for x, y in T5[k][:, 3:5]]).reshape(-1, 2)
    T = z["touch"]
    k = T[:, 2] == 1 if len(T) else np.zeros(0, bool)
    if k.any():
        return "digitizer", T[k, 0], np.array([touch_node(x, y, g) for x, y in T[k][:, 3:5]]).reshape(-1, 2)
    return None, np.zeros(0), np.zeros((0, 2))


def analyse(d):
    """holds found in the data, per-hold stability / fade / native agreement -> localisation_holds.json / .png
    (new names: the prompt-phase localisation.json / .png of earlier analyses are left as they are)"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    meta = json.load(open(os.path.join(d, "meta.json")))
    z = np.load(os.path.join(d, "capture.npz"))
    t, X = z["t"], z["X"].astype(float)
    t0 = t[0]
    rpeak, pk, rc = row_estimates(X)
    src, nt, nat = native_points(z, meta)
    phase_of = lambda tm: next((txt.split(" (")[0][:50] for ps, secs, txt in reversed(meta.get("phases", [])) if tm >= ps), "")
    rows = []
    for i0, i1 in find_holds(t, rpeak):
        tt, c, p = t[i0:i1], rc[i0:i1], pk[i0:i1]
        ok = ~np.isnan(c)
        wins = np.arange(tt[0], tt[-1], 0.25)
        wm = np.array([np.nanmean(c[(tt >= a) & (tt < a + 0.25)]) for a in wins
                       if np.any(ok & (tt >= a) & (tt < a + 0.25))])
        q = max(1, len(c) // 4)
        r = dict(start_s=round(float(tt[0] - t0), 2), end_s=round(float(tt[-1] - t0), 2),
                 seconds=round(float(tt[-1] - tt[0]), 2), prompt=phase_of((tt[0] + tt[-1]) / 2),
                 strongest_touch_node=f"Y{int(np.median(rpeak[i0:i1]))}",
                 centroid_mean=float(np.nanmean(c)), centroid_sd=float(np.nanstd(c)),
                 max_jump_025s=float(np.max(np.abs(np.diff(wm)))) if len(wm) > 1 else None,
                 false_moves_025s=int(np.sum(np.abs(np.diff(wm)) > 0.5)) if len(wm) > 1 else None,
                 drift=float(np.nanmedian(c[-q:]) - np.nanmedian(c[:q])),
                 peak_first=float(np.median(p[:q])), peak_last=float(np.median(p[-q:])),
                 row_rate=float((i1 - i0 - 1) / (tt[-1] - tt[0])))
        r["fade_pct"] = 100 * (r["peak_last"] / r["peak_first"] - 1)
        kn = (nt >= tt[0]) & (nt <= tt[-1])
        if kn.sum() >= 2:
            off = nat[kn, 1] - np.interp(nt[kn], tt[ok], c[ok])
            r.update(native=src, native_n=int(kn.sum()), native_rate=float(kn.sum() / r["seconds"]),
                     native_y_mean=float(nat[kn, 1].mean()), native_y_sd=float(nat[kn, 1].std()),
                     native_x_mean=float(nat[kn, 0].mean()), native_x_sd=float(nat[kn, 0].std()),
                     native_minus_row_mean=float(off.mean()), native_minus_row_sd=float(off.std()))
        rows.append(r)
    # native vs row over every touching moment (holds and slides): offset, slope, residual
    agree = None
    if len(nt) >= 10:
        ok = ~np.isnan(rc)
        tk = t[ok]
        rci = np.interp(nt, tk, rc[ok])
        j = np.clip(np.searchsorted(tk, nt), 1, len(tk) - 1)
        dt_ok = np.minimum(np.abs(tk[j] - nt), np.abs(tk[j - 1] - nt)) < 0.02     # a touching row sample within 20 ms
        if dt_ok.sum() >= 10:
            A = np.c_[rci[dt_ok], np.ones(dt_ok.sum())]
            (slope, icpt), *_ = np.linalg.lstsq(A, nat[dt_ok, 1], rcond=None)
            res = nat[dt_ok, 1] - A @ [slope, icpt]
            agree = dict(source=src, n=int(dt_ok.sum()), offset_median=float(np.median(nat[dt_ok, 1] - rci[dt_ok])),
                         fit_slope=float(slope), fit_intercept=float(icpt), fit_residual_sd=float(res.std()),
                         abs_diff_p50=float(np.median(np.abs(nat[dt_ok, 1] - rci[dt_ok]))),
                         abs_diff_p90=float(np.percentile(np.abs(nat[dt_ok, 1] - rci[dt_ok]), 90)),
                         native_x_line_median=float(np.median(nat[dt_ok, 0])))
    lifted = pk <= VIS
    summary = dict(holds=rows, native_source=src, native_contacts=int(len(nt)), native_vs_row=agree,
                   digitizer_reports=meta.get("digitizer_reports"), t5_messages=meta.get("t5_messages"),
                   row_rate=float(len(t) / (t[-1] - t0)),
                   row_rate_touching_median=float(1 / np.median(np.diff(t)[~lifted[1:]])) if (~lifted[1:]).any() else None,
                   row_rate_lifted_median=float(1 / np.median(np.diff(t)[lifted[1:]])) if lifted[1:].any() else None,
                   intervals_over_12ms=int(np.sum(np.diff(t) > 0.012)),
                   hold_rule=f"strongest node, median over {HOLD_SMOOTH} samples, unchanged >= {HOLD_MIN_S} s, peak > {VIS:g}")
    json.dump(summary, open(os.path.join(d, "localisation_holds.json"), "w"), indent=1)
    # figure: heatmap + estimates + holds, native - row, row rate
    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1.1, 1]))
    lim = max(200, np.percentile(np.abs(X), 99.5))
    ax[0].pcolormesh(t - t0, np.arange(X.shape[1]), X.T, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="nearest",
                     rasterized=True)             # real sample times: the row rate is not constant
    ax[0].plot(t - t0, rc, color="black", lw=1, label="row delta centroid")
    if len(nt):
        ax[0].plot(nt - t0, nat[:, 1], ".", ms=3, color="#00b000", label=f"native ({src}), Y line")
    for r in rows:
        for a_ in ax[1:]:
            a_.axvspan(r["start_s"], r["end_s"], color="#f0c000", alpha=0.18, lw=0)
        ax[0].hlines(-0.4, r["start_s"], r["end_s"], color="#c09000", lw=4)
        ax[0].text((r["start_s"] + r["end_s"]) / 2, X.shape[1] - 1.5, f"{r['strongest_touch_node']}\nsd {r['centroid_sd']:.2f}",
                   ha="center", va="top", fontsize=7)
    ax[0].set_ylabel("Y node (X line 0)"); ax[0].legend(loc="center right", fontsize=8)
    ax[0].set_title(f"{os.path.basename(d)}: X0 deltas (colour), finger estimates, holds found in the data (bars / shaded)",
                    fontsize=10)
    if len(nt):
        ok = ~np.isnan(rc)
        ax[1].plot(nt - t0, nat[:, 1] - np.interp(nt, t[ok], rc[ok]), ".", ms=2, color="#00b000",
                   label="native Y line - row centroid")
        ax[1].axhline(0, color="#999", lw=0.6)
        ax[1].set_ylabel("nodes")
    else:
        ax[1].plot(t - t0, pk, lw=0.7, color="#1f5fbf", label="strongest row delta (touch level)")
        ax[1].set_ylabel("counts")
    ax[1].legend(loc="upper right", fontsize=8)
    ax[2].plot(t[1:] - t0, 1 / np.diff(t), lw=0.6, color="#555")
    ax[2].set_ylabel("row samples/s"); ax[2].set_xlabel("time (s)"); ax[2].set_ylim(0, 160)
    fig.tight_layout(); fig.savefig(os.path.join(d, "localisation_holds.png"), dpi=140); plt.close(fig)
    pt, PP = z["patch_t"], z["patch"].astype(float)
    if len(pt) and rows:                        # mean 2D patch per hold, native touch marked
        fig, ax = plt.subplots(1, len(rows), figsize=(3.2 * len(rows), 3.6), squeeze=False)
        for a_, r in zip(ax[0], rows):
            k = (pt >= t0 + r["start_s"]) & (pt <= t0 + r["end_s"])
            if not k.any():
                continue
            a_.imshow(PP[k].mean(axis=0), origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
            if "native_y_mean" in r:
                a_.plot(r["native_y_mean"], r["native_x_mean"], "x", color="#00b000", ms=10, mew=2)
            a_.set_title(f"{r['start_s']:.0f}-{r['end_s']:.0f} s", fontsize=8); a_.set_xlabel("Y node"); a_.set_ylabel("X line")
        fig.tight_layout(); fig.savefig(os.path.join(d, "patch_holds.png"), dpi=140); plt.close(fig)
    print(f"\nrow rate {summary['row_rate']:.1f}/s (touching median {summary['row_rate_touching_median'] or 0:.1f}, "
          f"lifted {summary['row_rate_lifted_median'] or 0:.1f}), intervals > 12 ms: {summary['intervals_over_12ms']}; "
          f"native: {src or 'none'} {len(nt)} contacts (T5 messages {meta.get('t5_messages', '-')}, "
          f"digitizer reports {meta.get('digitizer_reports')})")
    print(f"{'hold (s)':>13s} {'prompt':32s} {'node':>4s} {'centroid':>8s} {'sd':>5s} {'jump':>5s} {'drift':>6s} "
          f"{'fade':>6s} {'rows/s':>6s} {'nat Y':>6s} {'nat-row':>7s}")
    for r in rows:
        f = lambda k, fmt: (fmt.format(r[k]) if r.get(k) is not None else "-")
        print(f"{r['start_s']:6.1f}-{r['end_s']:5.1f}  {r['prompt'][:32]:32s} {r['strongest_touch_node']:>4s} "
              f"{r['centroid_mean']:8.2f} {r['centroid_sd']:5.2f} {f('max_jump_025s', '{:5.2f}'):>5s} {r['drift']:6.2f} "
              f"{r['fade_pct']:5.0f}% {r['row_rate']:6.1f} {f('native_y_mean', '{:6.2f}'):>6s} "
              f"{f('native_minus_row_mean', '{:7.2f}'):>7s}")
    if agree:
        print(f"native vs row ({agree['n']} contacts): native = {agree['fit_slope']:.2f} x row + {agree['fit_intercept']:.2f}, "
              f"residual sd {agree['fit_residual_sd']:.2f} node; |native - row| median {agree['abs_diff_p50']:.2f}, "
              f"90 % {agree['abs_diff_p90']:.2f}; native X line median {agree['native_x_line_median']:.2f}")
    print(f"figure {os.path.relpath(os.path.join(d, 'localisation_holds.png'), HERE)}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=("bench", "record", "ratecheck", "analyse"))
    p.add_argument("what", nargs="?")
    p.add_argument("--label", default="test")
    p.add_argument("--seconds", type=float, default=5.0, help="bench: seconds per mode")
    p.add_argument("--patch-every", type=int, default=0, help="record: one 2D patch every N row samples (0 = none)")
    p.add_argument("--patch-w", type=int, default=4)
    p.add_argument("--patch-h", type=int, default=12)
    p.add_argument("--native", choices=("t5", "hidraw"), default="t5",
                   help="record: native coordinates over the control link (t5, default) or the digitizer only (hidraw)")
    p.add_argument("--block", type=float, default=5.0, help="ratecheck: seconds per mode block")
    p.add_argument("--cycles", type=int, default=3, help="ratecheck: cycles through the modes")
    a = p.parse_args()
    if a.cmd == "analyse":
        m = json.load(open(os.path.join(a.what, "meta.json")))
        (analyse_ratecheck if m.get("kind") == "ratecheck" else analyse)(a.what)
        return
    {"bench": bench, "record": record, "ratecheck": ratecheck}[a.cmd](a)


if __name__ == "__main__":
    main()
