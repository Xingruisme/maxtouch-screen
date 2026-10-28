#!/usr/bin/env python3
"""
Live capacitance map for the Microchip maXTouch EVK (ATMXT2952Tx), in the style of captivate/bswp_map.py.

    python maxtouch_map.py

Left:   full mutual-capacitance delta map (T37 deltas, every X/Y node). Touches reported by T100 are drawn on top.
Right:  last 10 s of the peak map delta, and of the touch message rate (= scan rate while a finger is down).
Bottom: map refresh rate, scan rate, current touches, chip / T7 scan config / T100 geometry, errors.

Everything goes through the EVK's HID control interface (maxtouch_hid.MaxTouchHid) from one reader thread:
  map    one full T37 delta frame = matrix_x * matrix_y * 2 bytes read page by page, a few fps over full-speed HID
  touch  T44 message count + T5 messages, drained after every map page so the chip's message queue does not overflow
While this runs it consumes the touch messages, so Windows touch input may stop until it exits.
At start it turns on the T100 peak-delta byte (TCHAUX) and restores the old value on exit, like maxtouch_hid.py --mode peaks.
Untick 'Map on' to poll only touch messages: the scan rate is then measured with nothing else on the link.
Tick 'Patch mode' to read only a small square (size from the menu) centred on the strongest touch instead of the whole
map, which is much faster. The square follows the finger but only moves once the finger is a quarter of the square
off centre, so it stays put for steady recordings. The rest of the map keeps its last full frame.
T37 pages can only be stepped through from page 0 (PAGE_DOWN from 0 wraps to page 255, not the last page) and each
page step waits for the next acquisition (~12 ms measured, unpolled steps get dropped), so every page comes from a
different scan. The data is X-major (~1.1 pages per X line here): a 6x6 patch on X lines 0-5 runs ~11 fps,
on X lines 20-25 ~2.5 fps. Keep the finger near X line 0 for the fastest patch.
[Record] writes data_maxtouch/map_<time>_touches.csv (every T100 message), _frames.csv (every full map frame, X-major)
and _patch.csv (every patch frame: origin, size, then the w*h values X-major).
"""

from __future__ import annotations

import csv
import datetime
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import patches

from maxtouch_hid import (
    DEBUG_DIAGNOSTIC_T37,
    DELTAS_MODE,
    GEN_COMMANDPROCESSOR_T6,
    GEN_MESSAGEPROCESSOR_T5,
    PAGE_UP,
    SPT_MESSAGECOUNT_T44,
    T6_DIAGNOSTIC_OFFSET,
    TOUCH_MULTITOUCHSCREEN_T100,
    MaxTouchHid,
)

GEN_POWERCONFIG_T7 = 7
# T100 config byte offsets / bits (same as the Linux atmel_mxt_ts driver)
T100_CFG1, T100_XORIGIN, T100_XSIZE, T100_XRANGE = 1, 8, 9, 13
T100_YORIGIN, T100_YSIZE, T100_YRANGE = 19, 20, 24
T100_SWITCHXY, T100_INVERTY, T100_INVERTX = 0x20, 0x40, 0x80
T100_DETECT = 0x80

DATADIR = Path(__file__).resolve().parent / "data_maxtouch"
HISTORY_S = 10.0
RATE_WINDOW_S = 2.0                     # touch message rate = messages per finger over this window
TOUCH_STALE_S = 2.0                     # drop a touch if it sends nothing for this long (missed release)
REFRESH_MS = 100                        # GUI redraw interval
NICE = [10, 20, 30, 50, 80, 100, 150, 200, 300, 500, 800, 1000, 1500, 2000, 3000, 5000, 10000, 20000, 32767]


def nice_up(x):
    return next((v for v in NICE if v >= x), NICE[-1])


def describe_acqint(v):
    return "off" if v == 0 else ("free-run" if v == 255 else f"{v} ms = {1000 / v:.0f} Hz")


# ---------------------------------------------------------------- reader thread
class Reader(threading.Thread):
    """Owns the HID link. Reads T37 map frames and drains T100 touch messages; the GUI only reads the shared state."""

    def __init__(self, mxt: MaxTouchHid):
        super().__init__(daemon=True)
        self.mxt = mxt
        idn = mxt.id_info
        self.nx, self.ny = idn.matrix_x, idn.matrix_y
        addr, size = mxt.object_address, mxt.object_size
        self.t6 = addr(GEN_COMMANDPROCESSOR_T6)
        self.t37, self.t37_size = addr(DEBUG_DIAGNOSTIC_T37), size(DEBUG_DIAGNOSTIC_T37)
        self.t44 = addr(SPT_MESSAGECOUNT_T44)
        self.t5, self.t5_size = addr(GEN_MESSAGEPROCESSOR_T5), size(GEN_MESSAGEPROCESSOR_T5)
        if None in (self.t6, self.t37, self.t44, self.t5):
            raise RuntimeError("T6/T37/T44/T5 object missing from the object table")

        # T100 report IDs: first = screen status, second = reserved, then one per touch slot
        rid, self.t100_rid, self.t100_nrid = 0, None, 0
        for o in mxt.objects:
            if o.type_id == TOUCH_MULTITOUCHSCREEN_T100 and self.t100_rid is None and o.num_report_ids:
                self.t100_rid, self.t100_nrid = rid + 1, o.num_report_ids
            rid += o.instances * o.num_report_ids

        t7 = addr(GEN_POWERCONFIG_T7)
        self.t7 = mxt.read_register(t7, 2) if t7 is not None else None
        t100 = addr(TOUCH_MULTITOUCHSCREEN_T100)
        self.geom = None
        if t100 is not None and size(TOUCH_MULTITOUCHSCREEN_T100) >= 26:
            c = mxt.read_register(t100, 26)
            self.geom = dict(cfg1=c[T100_CFG1], xorg=c[T100_XORIGIN], xsize=c[T100_XSIZE] or self.nx,
                             xrange=(c[T100_XRANGE] | c[T100_XRANGE + 1] << 8) or 1023,
                             yorg=c[T100_YORIGIN], ysize=c[T100_YSIZE] or self.ny,
                             yrange=(c[T100_YRANGE] | c[T100_YRANGE + 1] << 8) or 1023)

        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.map_on = True
        self.patch_on = False
        self.patch_size = 6
        self.patch = None               # (x0, y0, values (w, h)) of the latest patch frame
        self.grid = None                # latest frame, shape (nx, ny), indexed [x line, y line]
        self.frame_id = 0
        self.map_times = deque(maxlen=10)
        self.frame_ms = deque(maxlen=10)
        self.peak_hist = deque()        # (t, max delta of the frame)
        self.rate_hist = deque()        # (t, touch message rate)
        self.touches = {}               # slot -> dict(x, y, peak, t)
        self.msg_times = {}             # slot -> deque of message times
        self.n_msgs = 0
        self.errors = 0
        self.last_error = ""
        self.t_rate = 0.0
        self.rate = 0.0
        self.touch_writer = self.frame_writer = self.patch_writer = None
        self.n_rows = 0

    def touch_to_node(self, x, y):
        """T100 touch coordinate -> (X line, Y line) on the matrix, using the T100 origin/size/range/orientation."""
        g = self.geom
        if g is None:
            return None
        if g["cfg1"] & T100_SWITCHXY:
            x, y = y, x
        fx, fy = x / g["xrange"], y / g["yrange"]
        if g["cfg1"] & T100_INVERTX:
            fx = 1 - fx
        if g["cfg1"] & T100_INVERTY:
            fy = 1 - fy
        return g["xorg"] + fx * (g["xsize"] - 1), g["yorg"] + fy * (g["ysize"] - 1)

    # ------------------------------------------------------------ touch messages
    def _drain(self):
        n = self.mxt.read_register(self.t44, 1)[0]
        for _ in range(n):
            msg = self.mxt.read_register(self.t5, self.t5_size)
            if msg[0] == 0xFF:          # queue empty
                break
            self._message(msg, time.time())

    def _message(self, msg, now):
        if self.t100_rid is None or not (self.t100_rid + 2 <= msg[0] < self.t100_rid + self.t100_nrid):
            return
        slot = msg[0] - self.t100_rid - 2
        status, x, y, peak = msg[1], msg[2] | msg[3] << 8, msg[4] | msg[5] << 8, msg[6]
        with self.lock:
            self.n_msgs += 1
            self.msg_times.setdefault(slot, deque()).append(now)
            if status & T100_DETECT:
                self.touches[slot] = dict(x=x, y=y, peak=peak, t=now)
            else:
                self.touches.pop(slot, None)
            if self.touch_writer:
                self.touch_writer.writerow([f"{now:.4f}", slot, status, x, y, peak])

    def _update_rate(self, now):
        with self.lock:
            best = 0.0
            for slot, ts in list(self.msg_times.items()):
                while ts and now - ts[0] > RATE_WINDOW_S:
                    ts.popleft()
                if len(ts) >= 3 and ts[-1] > ts[0]:
                    best = max(best, (len(ts) - 1) / (ts[-1] - ts[0]))
            for slot in [s for s, t in self.touches.items() if now - t["t"] > TOUCH_STALE_S]:
                del self.touches[slot]
            self.rate = best
            if now - self.t_rate > 0.1:
                self.t_rate = now
                self.rate_hist.append((now, best))
                while self.rate_hist and now - self.rate_hist[0][0] > HISTORY_S:
                    self.rate_hist.popleft()

    # ------------------------------------------------------------ map
    def _step_page(self, cmd):
        """Send a T6 diagnostic command and wait until the chip has done it (~one acquisition, ~12 ms).
        The wait is spent draining touch messages instead of sleeping, so the message queue is emptied about every
        2-4 ms whatever is being read. Draining only between pages lost messages and made the scan rate read low."""
        diag = self.t6 + T6_DIAGNOSTIC_OFFSET
        self.mxt.write_register(diag, bytes([cmd]))
        deadline = time.perf_counter() + 0.5
        while self.mxt.read_register(diag, 1)[0] != 0:
            if time.perf_counter() > deadline:
                raise TimeoutError("T6 diagnostic command timed out")
            self._drain()

    def _read_map(self):
        page_size = self.t37_size - 2
        nbytes = self.nx * self.ny * 2
        data = bytearray()
        t0 = time.perf_counter()
        for page in range((nbytes + page_size - 1) // page_size):
            if not self.map_on or self.stop.is_set():
                return None
            self._step_page(DELTAS_MODE if page == 0 else PAGE_UP)
            raw = self.mxt.read_register(self.t37, self.t37_size)
            if raw[0] != DELTAS_MODE or raw[1] != page:
                raise RuntimeError(f"unexpected T37 header mode=0x{raw[0]:02X} page={raw[1]} (expected {page})")
            data += raw[2:]
        grid = np.frombuffer(bytes(data[:nbytes]), "<i2").reshape(self.nx, self.ny).copy()   # T37 is X-major
        return grid, (time.perf_counter() - t0) * 1e3

    # ------------------------------------------------------------ patch
    def _pick_patch(self, size):
        """Patch origin (x0, y0): centred on the strongest touch, moved only when the touch drifts size/4 off centre.
        No touch: keep the current patch, or start on the strongest node of the last full map (else the middle)."""
        with self.lock:
            touches = list(self.touches.values())
            cur, grid = self.patch, self.grid
        w, h = min(size, self.nx), min(size, self.ny)
        node = None
        if touches:
            t = max(touches, key=lambda t: t["peak"])
            node = self.touch_to_node(t["x"], t["y"])
        if cur is not None and cur[2].shape == (w, h):
            x0, y0 = cur[0], cur[1]
            if node is None or (abs(node[0] - (x0 + (w - 1) / 2)) < size / 4
                                and abs(node[1] - (y0 + (h - 1) / 2)) < size / 4):
                return x0, y0, w, h
        if node is None:
            node = np.unravel_index(int(grid.argmax()), grid.shape) if grid is not None else (self.nx / 2, self.ny / 2)
        x0 = int(np.clip(round(node[0] - (w - 1) / 2), 0, self.nx - w))
        y0 = int(np.clip(round(node[1] - (h - 1) / 2), 0, self.ny - h))
        return x0, y0, w, h

    def _read_patch(self, x0, y0, w, h):
        """Read only the T37 bytes of a w x h patch. Pages still have to be stepped through from 0."""
        page_size = self.t37_size - 2
        wanted = {}                     # page -> [(offset in page, i, j)]
        for i in range(w):
            for j in range(h):
                off = ((x0 + i) * self.ny + y0 + j) * 2
                wanted.setdefault(off // page_size, []).append((off % page_size, i, j))
        vals = np.zeros((w, h), np.int16)
        t0 = time.perf_counter()
        for page in range(max(wanted) + 1):
            if not (self.map_on and self.patch_on) or self.stop.is_set():
                return None
            self._step_page(DELTAS_MODE if page == 0 else PAGE_UP)
            if page in wanted:
                last = max(o for o, _, _ in wanted[page]) + 2
                raw = self.mxt.read_register(self.t37, 2 + last)      # header + data up to the last wanted node
                if raw[0] != DELTAS_MODE or raw[1] != page:
                    raise RuntimeError(f"unexpected T37 header mode=0x{raw[0]:02X} page={raw[1]} (expected {page})")
                for o, i, j in wanted[page]:
                    vals[i, j] = int.from_bytes(raw[2 + o:4 + o], "little", signed=True)
        return vals, (time.perf_counter() - t0) * 1e3

    def _frame_done(self, now, grid, ms, peak):
        self.grid = grid
        self.frame_id += 1
        self.map_times.append(now)
        self.frame_ms.append(ms)
        self.peak_hist.append((now, peak))
        while self.peak_hist and now - self.peak_hist[0][0] > HISTORY_S:
            self.peak_hist.popleft()

    def run(self):
        while not self.stop.is_set():
            try:
                if self.map_on and self.patch_on:
                    x0, y0, w, h = self._pick_patch(self.patch_size)
                    got = self._read_patch(x0, y0, w, h)
                    now = time.time()
                    if got is not None:
                        vals, ms = got
                        with self.lock:
                            grid = self.grid.copy() if self.grid is not None else np.zeros((self.nx, self.ny), np.int16)
                            grid[x0:x0 + w, y0:y0 + h] = vals
                            self.patch = (x0, y0, vals)
                            self._frame_done(now, grid, ms, int(vals.max()))
                            if self.patch_writer:
                                self.patch_writer.writerow([f"{now:.4f}", x0, y0, w, h] + vals.ravel().tolist())
                                self.n_rows += 1
                elif self.map_on:
                    got = self._read_map()
                    now = time.time()
                    if got is not None:
                        grid, ms = got
                        with self.lock:
                            self._frame_done(now, grid, ms, int(grid.max()))
                            if self.frame_writer:
                                self.frame_writer.writerow([f"{now:.4f}"] + grid.ravel().tolist())
                                self.n_rows += 1
                else:
                    self._drain()
                    now = time.time()
                self._update_rate(now)
            except Exception as e:      # a dropped HID transfer should not kill the reader
                with self.lock:
                    self.errors += 1
                    self.last_error = f"{type(e).__name__}: {e}"
                time.sleep(0.05)

    # ------------------------------------------------------------ recording
    def start_record(self, stem: Path):
        with self.lock:
            self.fh_t = open(f"{stem}_touches.csv", "w", newline="")
            self.fh_f = open(f"{stem}_frames.csv", "w", newline="")
            self.touch_writer = csv.writer(self.fh_t)
            self.touch_writer.writerow(["t_unix", "slot", "status", "x", "y", "peak"])
            self.frame_writer = csv.writer(self.fh_f)
            self.frame_writer.writerow(["t_unix"] + [f"x{x}_y{y}" for x in range(self.nx) for y in range(self.ny)])
            self.fh_p = open(f"{stem}_patch.csv", "w", newline="")
            self.patch_writer = csv.writer(self.fh_p)
            self.patch_writer.writerow(["t_unix", "x0", "y0", "w", "h", "values (X-major: x0,y0 x0,y0+1 ...)"])
            self.n_rows = 0

    def stop_record(self):
        with self.lock:
            if self.touch_writer:
                self.fh_t.close()
                self.fh_f.close()
                self.fh_p.close()
            self.touch_writer = self.frame_writer = self.patch_writer = None


# ---------------------------------------------------------------- GUI
class App:
    def __init__(self, root, mxt: MaxTouchHid):
        self.root, self.mxt = root, mxt
        self.prev_tchaux = mxt.enable_t100_peak_deltas()
        self.rx = Reader(mxt)
        root.title("maXTouch Capacitance Map")

        top = ttk.Frame(root, padding=(8, 6))
        top.pack(fill="x")
        ttk.Label(top, text="Color range [delta]").pack(side="left")
        self.scale_var = tk.StringVar(value="Auto")
        ttk.Combobox(top, textvariable=self.scale_var, width=7, state="readonly",
                     values=["Auto", "50", "100", "200", "500", "1000", "2000", "5000"]).pack(side="left", padx=(4, 16))
        self.swap_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Landscape (Y lines across)", variable=self.swap_var,
                        command=self._new_orientation).pack(side="left")
        self.map_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="Map on", variable=self.map_var,
                        command=lambda: setattr(self.rx, "map_on", self.map_var.get())).pack(side="left", padx=(16, 0))
        self.patch_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="Patch mode", variable=self.patch_var,
                        command=lambda: setattr(self.rx, "patch_on", self.patch_var.get())).pack(side="left", padx=(16, 0))
        self.patch_size_var = tk.StringVar(value=str(self.rx.patch_size))
        size_box = ttk.Combobox(top, textvariable=self.patch_size_var, width=4, state="readonly",
                                values=["4", "6", "8", "10", "12"])
        size_box.pack(side="left", padx=(4, 0))
        size_box.bind("<<ComboboxSelected>>", lambda e: setattr(self.rx, "patch_size", int(self.patch_size_var.get())))
        ttk.Label(top, text="nodes").pack(side="left", padx=(4, 0))
        self.rec_btn = ttk.Button(top, text="Record", width=9, command=self.toggle_record)
        self.rec_btn.pack(side="right")
        self.rec_lbl = ttk.Label(top, text="", width=44, anchor="e")
        self.rec_lbl.pack(side="right", padx=8)

        self.fig = Figure(figsize=(12, 6.2), dpi=100)
        gs = self.fig.add_gridspec(2, 2, width_ratios=[1.45, 1], hspace=0.38, wspace=0.18,
                                   left=0.04, right=0.98, top=0.93, bottom=0.08)
        self.ax_map = self.fig.add_subplot(gs[:, 0])
        self.ax_peak = self.fig.add_subplot(gs[0, 1])
        self.ax_rate = self.fig.add_subplot(gs[1, 1])
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.status = tk.Label(root, anchor="nw", justify="left", font=("Consolas", 10), padx=8, pady=6,
                               height=5, width=150)
        self.status.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.peaks = deque(maxlen=30)   # recent frame |max| for the auto color range
        self.last_frame = -1
        self._draw_axes()

        self.rx.start()
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.tick()

    # ------------------------------------------------------------ figure skeleton
    def _draw_axes(self):
        ax = self.ax_map
        ax.clear()
        swap = self.swap_var.get()
        shape = (self.rx.nx, self.rx.ny) if swap else (self.rx.ny, self.rx.nx)
        self.im = ax.imshow(np.zeros(shape), cmap="RdBu_r", vmin=-100, vmax=100, interpolation="nearest",
                            aspect="equal")
        ax.set_xlabel("Y line" if swap else "X line", fontsize=8)
        ax.set_ylabel("X line" if swap else "Y line", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title(f"Mutual-capacitance delta, {self.rx.nx} X × {self.rx.ny} Y nodes (T37)", fontsize=10, loc="left")
        self.touch_dots, = ax.plot([], [], "o", ms=14, mfc="none", mec="#16a34a", mew=2.2)
        self.patch_box = patches.Rectangle((0, 0), 1, 1, fill=False, ec="#7c3aed", lw=2, ls="--", visible=False)
        ax.add_patch(self.patch_box)
        self.touch_txt = []
        if not hasattr(self, "cbar"):
            self.cbar = self.fig.colorbar(self.im, ax=ax, orientation="horizontal", fraction=0.04, pad=0.08, aspect=40)
            self.cbar.ax.tick_params(labelsize=8)
            self.cbar.set_label("delta [counts]  (touch → positive)", fontsize=8)
        else:
            self.cbar.update_normal(self.im)

        for a, title, ylabel in ((self.ax_peak, "Peak map delta", "delta [counts]"),
                                 (self.ax_rate, "Scan rate (T100 messages per finger)", "rate [Hz]")):
            a.clear()
            a.set_title(title, fontsize=9, loc="left")
            a.set_ylabel(ylabel, fontsize=8)
            a.set_xlim(-HISTORY_S, 0)
            a.tick_params(labelsize=8)
            a.grid(alpha=0.3)
        self.ax_rate.set_xlabel("Time [s] (0 = now)", fontsize=8)
        self.peak_line, = self.ax_peak.plot([], [], "-", lw=1.3, color="#dc2626")
        self.rate_line, = self.ax_rate.plot([], [], "-", lw=1.3, color="#2563eb")
        self.last_frame = -1

    def _new_orientation(self):
        self._draw_axes()
        self.canvas.draw_idle()

    # ------------------------------------------------------------ refresh
    def tick(self):
        try:
            self._update()
        finally:
            self.root.after(REFRESH_MS, self.tick)

    def _update(self):
        rx = self.rx
        with rx.lock:
            grid, fid = rx.grid, rx.frame_id
            touches = {k: dict(v) for k, v in rx.touches.items()}
            peak_hist, rate_hist = list(rx.peak_hist), list(rx.rate_hist)
            map_times, frame_ms = list(rx.map_times), list(rx.frame_ms)
            rate, n_msgs, errors, last_error = rx.rate, rx.n_msgs, rx.errors, rx.last_error
            n_rows = rx.n_rows if rx.frame_writer else None
            patch = rx.patch
        now = time.time()
        swap = self.swap_var.get()
        patch_mode = self.patch_var.get() and self.map_var.get()

        # patch outline: X lines x0..x0+w-1, Y lines y0..y0+h-1 -> image (col, row), pixel edges at +-0.5
        if patch_mode and patch is not None:
            x0, y0, vals = patch
            w, h = vals.shape
            c0, r0, cw, rh = (y0, x0, h, w) if swap else (x0, y0, w, h)
            self.patch_box.set_bounds(c0 - 0.5, r0 - 0.5, cw, rh)
            self.patch_box.set_visible(True)
        else:
            self.patch_box.set_visible(False)

        if grid is not None and fid != self.last_frame:
            self.last_frame = fid
            self.peaks.append(int(np.abs(grid).max()))
            vmax = nice_up(1.1 * max(self.peaks)) if self.scale_var.get() == "Auto" else float(self.scale_var.get())
            self.im.set_data(grid if swap else grid.T)
            self.im.set_clim(-vmax, vmax)

        # touches: (X line, Y line) -> image (col, row)
        for t in self.touch_txt:
            t.remove()
        self.touch_txt = []
        cols, rows = [], []
        for slot, t in sorted(touches.items()):
            node = rx.touch_to_node(t["x"], t["y"])
            if node is None:
                continue
            xl, yl = node
            c, r = (yl, xl) if swap else (xl, yl)
            cols.append(c); rows.append(r)
            self.touch_txt.append(self.ax_map.text(c, r - 2.2, str(slot), color="#16a34a", fontsize=9,
                                                   ha="center", va="bottom", fontweight="bold"))
        self.touch_dots.set_data(cols, rows)

        for line, ax, hist in ((self.peak_line, self.ax_peak, peak_hist), (self.rate_line, self.ax_rate, rate_hist)):
            if hist:
                t = np.array([h[0] for h in hist]) - now
                v = np.array([h[1] for h in hist], float)
                line.set_data(t, v)
                lo, hi = min(0.0, v.min()), max(10.0, v.max())
                ax.set_ylim(lo - 0.05 * (hi - lo), hi * 1.15)

        if n_rows is not None:
            self.rec_lbl.config(text=f"● {self.rec_stem.name}  {n_rows} frames", foreground="#dc2626")

        map_fps = (len(map_times) - 1) / (map_times[-1] - map_times[0]) if len(map_times) > 1 else 0.0
        ms = np.mean(frame_ms) if frame_ms else 0
        if not self.map_var.get():
            line1 = "Map paused - polling touch messages only"
        elif patch_mode and patch is not None:
            x0, y0, vals = patch
            w, h = vals.shape
            line1 = (f"Patch {w}x{h} at X {x0}-{x0 + w - 1}, Y {y0}-{y0 + h - 1}: {map_fps:5.1f} fps "
                     f"({ms:.0f} ms per patch)   patch peak {vals.max()} mean {vals.mean():.0f}")
        else:
            line1 = f"Map {map_fps:4.1f} fps ({ms:.0f} ms per full frame)"
        line1 += f"   Touch messages {n_msgs}   HID errors {errors}"
        line2 = (f"Scan rate: {rate:.1f} Hz (T100 messages per finger, last {RATE_WINDOW_S:.0f} s)" if rate > 0 else
                 "Scan rate: - (put a finger on the panel; the controller only reports while touched)")
        line3 = "Touches: " + ("   ".join(f"#{s} x={t['x']} y={t['y']} peak={t['peak']}" for s, t in sorted(touches.items()))
                               or "none")
        idn = self.mxt.id_info
        t7 = (f"T7 ACQINT idle {describe_acqint(rx.t7[0])} / active {describe_acqint(rx.t7[1])}"
              if rx.t7 is not None else "T7 ?")
        g = rx.geom
        geo = (f"T100 X lines {g['xorg']}-{g['xorg'] + g['xsize'] - 1}, Y lines {g['yorg']}-{g['yorg'] + g['ysize'] - 1}, "
               f"range {g['xrange']}x{g['yrange']}, cfg1 0x{g['cfg1']:02X}" if g else "T100 ?")
        line4 = (f"Chip 0x{idn.family:02X}/0x{idn.variant:02X} fw {idn.version}.{idn.build:02X}  "
                 f"matrix {idn.matrix_x}x{idn.matrix_y}   {t7}   {geo}")
        line5 = f"Last HID error: {last_error}" if last_error else ""
        self.status.config(text="\n".join([line1, line2, line3, line4, line5]),
                           fg="#b45309" if last_error and errors > 5 else "black")
        self.canvas.draw_idle()

    # ------------------------------------------------------------ record / close
    def toggle_record(self):
        if self.rx.frame_writer:
            self.rx.stop_record()
            self.rec_btn.config(text="Record")
            self.rec_lbl.config(text=f"Saved: data_maxtouch/{self.rec_stem.name}_*.csv", foreground="black")
        else:
            DATADIR.mkdir(parents=True, exist_ok=True)
            self.rec_stem = DATADIR / datetime.datetime.now().strftime("map_%Y%m%d_%H%M%S")
            self.rx.start_record(self.rec_stem)
            self.rec_btn.config(text="Stop")

    def close(self):
        self.rx.stop_record()
        self.rx.stop.set()
        self.rx.join(timeout=2.0)
        try:
            self.mxt.restore_t100_tchaux(self.prev_tchaux)
        except Exception:
            pass
        self.mxt.close()
        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        mxt = MaxTouchHid()
        mxt.load_info()
        app = App(root, mxt)
    except Exception as e:
        root.withdraw()
        messagebox.showerror("maXTouch", f"{e}")
        return
    root.mainloop()


if __name__ == "__main__":
    main()
