#!/usr/bin/env python3
"""OnePlus Pad mutual-capacitance diff heatmap via adb.

The first frame becomes the baseline. Every later frame is shown as
(current − baseline): zero is white, negative is green, positive is red.

Data is read from ``/proc/touchpanel/debug_info/`` (Novatek NT36532). Root
is required. Default source is ``delta`` (~8 Hz full 38×54 frame). Source
``raw`` (``raw_data`` in the baseline proc) is true ADC capacitance but
only ~0.8 Hz because that node dumps both baseline and raw sections (~29 KB).

Keyboard shortcuts:
  R  — toggle manual Start / Stop recording
  T  — start a timed recording (uses the duration field)
  Q  — quit

Usage:
    python3 oneplus_monitor.py --adb ./adb
    python3 oneplus_monitor.py --adb ./adb --source raw   # ~0.8 Hz true raw_data
    python3 oneplus_monitor.py --adb ./adb --snapshot out.png

Note: ``--source delta`` reaches ~7–8 Hz; ``--source raw`` is ~0.8 Hz. If touch
or adb stalls, try ``--touch-safe-sleep-ms 30``.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import select
import subprocess
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import ListedColormap

from oneplus_capacitance import (
    CAP_LOG_DIR,
    COLS,
    MATRIX_SOURCES,
    ROWS,
    Adb,
    CapWideWriter,
    find_adb,
    parse_section_from_lines,
    read_controller,
)

DEFAULT_INTERVAL_MS = 0
DEFAULT_STREAM_RATE_HZ = 0.0
DEFAULT_TOUCH_SAFE_SLEEP_MS = 0  # delta ~8 Hz; 30 ms caps delta at ~5 Hz
# Recordings are written in the wide capacitance layout (one row per frame) via
# oneplus_capacitance.CapWideWriter. See load_capacitance_session to read them.
COLOR_SCALE_ABS_MAX = 1000.0
LIVE_SOURCES = tuple(MATRIX_SOURCES)


@dataclass(frozen=True)
class MonitorConfig:
    window_title: str
    signal_name: str
    chart_title: str
    colorbar_label: str
    snapshot_title: str
    default_source: str
    live_sources: tuple[str, ...]
    default_color_max: float
    log_dir: Path
    csv_filename: str = "capacitance.csv"
    doc: str = ""


CAPACITANCE_CONFIG = MonitorConfig(
    window_title="OnePlus Pad Capacitance Monitor",
    signal_name="capacitance",
    chart_title="Mutual-sense capacitance diff",
    colorbar_label="Δ capacitance vs first frame (controller units)",
    snapshot_title="OnePlus Pad Capacitance Diff Snapshot",
    default_source="delta",
    live_sources=LIVE_SOURCES,
    default_color_max=0.0,
    log_dir=CAP_LOG_DIR,
    doc=__doc__,
)

C = {
    "bg": "#ffffff",
    "panel": "#f2f2f2",
    "card": "#f8f8f8",
    "accent": "#dddddd",
    "green": "#1a7a4a",
    "yellow": "#b36b00",
    "red": "#c0392b",
    "text": "#111111",
    "subtext": "#555555",
}


def build_rwgb_colormap() -> ListedColormap:
    colors = []
    center = 128
    for i in range(256):
        if i < center:
            ratio = i / center
            curved = ratio**0.45
            intensity = int(curved * 255)
            colors.append((intensity / 255, 1.0, intensity / 255))
        else:
            ratio = (i - center) / (255 - center)
            intensity = int((1 - ratio) * 255)
            colors.append((1.0, intensity / 255, intensity / 255))
    return ListedColormap(colors)


def read_matrix(adb: Adb, kind: str) -> np.ndarray:
    rows, _ = read_controller(adb, kind)
    matrix = np.asarray(rows, dtype=float)
    if matrix.shape != (ROWS, COLS):
        raise ValueError(f"{kind} returned {matrix.shape}; expected ({ROWS}, {COLS})")
    return matrix


def parse_stream_matrix(lines: list[str], source: str) -> np.ndarray:
    _, section = MATRIX_SOURCES[source]
    rows = parse_section_from_lines(lines, section)
    matrix = np.asarray(rows, dtype=float)
    if matrix.shape != (ROWS, COLS):
        raise ValueError(
            f"stream frame from {source} has shape {matrix.shape}; "
            f"expected ({ROWS}, {COLS})"
        )
    return matrix


class MatrixStream:
    """Read matrices from one persistent root shell."""

    def __init__(
        self,
        adb: Adb,
        source: str,
        *,
        touch_safe_sleep_ms: int = DEFAULT_TOUCH_SAFE_SLEEP_MS,
    ) -> None:
        self.adb = adb
        self.source = source
        self.touch_safe_sleep_ms = touch_safe_sleep_ms
        source_path = MATRIX_SOURCES[source][0]
        sleep_s = max(touch_safe_sleep_ms, 0) / 1000.0
        sleep_clause = f" sleep {sleep_s:.3f};" if sleep_s > 0 else ""
        remote_loop = (
            "while true; do "
            "echo FRAME_START; "
            f"cat {source_path}; "
            "echo FRAME_END $(date +%s%3N);"
            f"{sleep_clause} "
            "done"
        )
        command = f"su -c '{remote_loop}'"
        self.process = subprocess.Popen(
            [*adb.base, "shell", command],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        if self.process.stdout is None:
            raise RuntimeError("failed to open adb matrix stream")
        self._frame_lines: list[str] = []
        self._in_frame = False
        self.skipped_frames = 0
        self.last_device_ts_ms: int | None = None

    def alive(self) -> bool:
        return self.process.poll() is None

    def read(self, timeout: float = 8.0) -> np.ndarray:
        if not self.alive():
            raise RuntimeError("adb matrix stream ended")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            readable, _, _ = select.select(
                [self.process.stdout], [], [], min(0.25, max(0.0, remaining))
            )
            if not readable:
                continue
            line = self.process.stdout.readline()
            if not line:
                if self.process.poll() is not None:
                    raise RuntimeError("adb matrix stream ended")
                continue
            stripped = line.strip()
            if stripped == "FRAME_START":
                self._frame_lines = []
                self._in_frame = True
                continue
            if stripped.startswith("FRAME_END"):
                if not self._in_frame:
                    continue
                self._in_frame = False
                parts = stripped.split()
                try:
                    self.last_device_ts_ms = int(parts[1]) if len(parts) >= 2 else None
                except ValueError:
                    self.last_device_ts_ms = None
                frame_lines = self._frame_lines
                self._frame_lines = []
                try:
                    matrix = parse_stream_matrix(frame_lines, self.source)
                except ValueError:
                    self.skipped_frames += 1
                    continue
                return matrix
            if self._in_frame:
                self._frame_lines.append(line)
        raise TimeoutError(f"no matrix frame within {timeout:.1f}s")

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()


def symmetric_limits(values: np.ndarray, color_max: float | None) -> tuple[float, float]:
    if color_max is not None:
        bound = min(COLOR_SCALE_ABS_MAX, color_max)
        return -bound, bound
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        bound = 1.0
    else:
        bound = min(
            COLOR_SCALE_ABS_MAX,
            max(float(np.max(np.abs(finite))), 1e-6),
        )
    return -bound, bound


def adb_device_ready(adb: Adb) -> bool:
    try:
        devices = adb.run("devices", timeout=8)
    except (RuntimeError, subprocess.TimeoutExpired):
        return False
    for line in devices.splitlines():
        if not line.strip() or line.startswith("List of"):
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device":
            return True
    return False


class OnePlusLiveMapMonitor:
    def __init__(
        self,
        root: tk.Tk,
        *,
        config: MonitorConfig,
        adb: Adb,
        source: str,
        read_mode: str,
        interval_ms: int,
        stream_rate_hz: float,
        touch_safe_sleep_ms: int,
        color_max: float | None,
        data_dir: Path,
    ) -> None:
        self.config = config
        self.root = root
        self.adb = adb
        self.source = source
        self.read_mode = read_mode
        self.interval_ms = interval_ms
        self.stream_rate_hz = stream_rate_hz
        self.touch_safe_sleep_ms = touch_safe_sleep_ms
        self.color_max = color_max
        self.data_dir = data_dir
        self._stream: MatrixStream | None = None
        self._stream_restarts = 0
        self._stream_failures = 0

        self.baseline: np.ndarray | None = None
        self.running = False
        self._rx_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._latest_raw: np.ndarray | None = None
        self._latest_delta: np.ndarray | None = None
        self._frame_count = 0
        self._last_hz = 0.0
        self._status_text = "Connecting…"
        self._read_error = ""

        self.recording = False
        self._rec_rows: list[dict] = []
        self._rec_start = 0.0
        self._timer: threading.Timer | None = None

        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.root.title(config.window_title)
        self.root.geometry("920x920")
        self.root.minsize(720, 720)
        self.root.configure(bg=C["bg"])

        self._cmap = build_rwgb_colormap()
        self._build_ui()
        self._setup_shortcuts()

        self.running = True
        self._rx_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._rx_thread.start()

        self._anim = FuncAnimation(
            self._fig,
            self._animate,
            interval=100,
            blit=False,
            cache_frame_data=False,
        )
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        bar = tk.Frame(self.root, bg=C["panel"], pady=8)

        tk.Label(
            bar,
            text=(
                f"● OnePlus Pad {self.config.signal_name}  |  {ROWS}×{COLS} grid  |  "
                f"Δ from first frame  |  {self.source}  |  {self.read_mode} read"
            ),
            bg=C["panel"],
            fg=C["green"],
            font=("Helvetica", 10, "bold"),
        ).pack(side="left", padx=12)

        self._status_lbl = tk.Label(
            bar,
            text="Waiting…",
            bg=C["panel"],
            fg=C["subtext"],
            font=("Helvetica", 10),
        )
        self._status_lbl.pack(side="left", padx=8)

        self._rec_badge = tk.Label(
            bar,
            text="",
            bg=C["panel"],
            fg=C["red"],
            font=("Helvetica", 10, "bold"),
        )
        self._rec_badge.pack(side="right", padx=18)

        self._chart_host = tk.Frame(self.root, bg=C["bg"])

        fig_h = 6.2
        fig_w = fig_h * (COLS / ROWS) + 0.85
        self._fig = plt.Figure(figsize=(fig_w, fig_h), facecolor=C["bg"])
        self._ax = self._fig.add_subplot(111)
        vmin, vmax = symmetric_limits(np.zeros((ROWS, COLS)), self.color_max)
        self._im = self._ax.imshow(
            np.zeros((ROWS, COLS)),
            cmap=self._cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
            origin="upper",
            interpolation="nearest",
        )
        self._ax.set_title(self.config.chart_title)
        self._ax.set_xlabel(f"Mutual-sense X channel ({COLS} columns)")
        self._ax.set_ylabel(f"Mutual-sense Y channel ({ROWS} rows)")
        self._cbar = self._fig.colorbar(self._im, ax=self._ax, fraction=0.05, pad=0.03)
        self._cbar.set_label(self.config.colorbar_label)
        self._fig.subplots_adjust(left=0.08, right=0.86, top=0.95, bottom=0.06)

        canvas = FigureCanvasTkAgg(self._fig, master=self._chart_host)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self._canvas = canvas
        self._chart_host.bind("<Configure>", self._on_chart_resize)

        ctrl = tk.Frame(self.root, bg=C["panel"])

        mf = self._card(ctrl, "Manual Record")
        mf.pack(side="left", padx=10, pady=8, fill="y")
        self._rec_btn = tk.Button(
            mf,
            text="▶  Start Recording   [R]",
            command=self._toggle_record,
            bg=C["green"],
            fg=C["text"],
            font=("Helvetica", 10, "bold"),
            relief="flat",
            padx=12,
            pady=6,
            width=22,
            cursor="hand2",
        )
        self._rec_btn.pack(pady=4)

        tf = self._card(ctrl, "Timed Record")
        tf.pack(side="left", padx=10, pady=8, fill="y")
        dur_row = tk.Frame(tf, bg=C["card"])
        dur_row.pack()
        tk.Label(
            dur_row,
            text="Duration (s):",
            bg=C["card"],
            fg=C["subtext"],
            font=("Helvetica", 10),
        ).pack(side="left")
        self._dur_var = tk.StringVar(value="30")
        tk.Entry(
            dur_row,
            textvariable=self._dur_var,
            width=7,
            bg=C["accent"],
            fg=C["text"],
            insertbackground=C["text"],
            relief="flat",
            font=("Helvetica", 10),
        ).pack(side="left", padx=6)
        self._timed_btn = tk.Button(
            tf,
            text="⏱  Start Timed Record   [T]",
            command=self._start_timed,
            bg=C["yellow"],
            fg=C["text"],
            font=("Helvetica", 10, "bold"),
            relief="flat",
            padx=12,
            pady=6,
            cursor="hand2",
        )
        self._timed_btn.pack(pady=(6, 2))
        self._countdown_lbl = tk.Label(
            tf,
            text="",
            bg=C["card"],
            fg=C["subtext"],
            font=("Helvetica", 10),
        )
        self._countdown_lbl.pack()

        rf = self._card(ctrl, "Last Recording")
        rf.pack(side="left", padx=10, pady=8, fill="both", expand=True)
        self._result_lbl = tk.Label(
            rf,
            text="No recording yet.",
            bg=C["card"],
            fg=C["subtext"],
            font=("Helvetica", 10),
            justify="left",
            anchor="nw",
        )
        self._result_lbl.pack(fill="both", expand=True)

        ctrl.pack(side="bottom", fill="x", padx=10, pady=(0, 12))
        bar.pack(side="top", fill="x", padx=10, pady=(10, 0))
        self._chart_host.pack(side="top", fill="both", expand=True, padx=10, pady=8)

    def _on_chart_resize(self, event: tk.Event) -> None:
        if event.widget is not self._chart_host:
            return
        width, height = event.width, event.height
        if width < 80 or height < 80:
            return
        dpi = self._fig.get_dpi()
        self._fig.set_size_inches(width / dpi, height / dpi, forward=True)
        self._fig.subplots_adjust(left=0.08, right=0.86, top=0.95, bottom=0.06)
        self._canvas.draw_idle()

    def _card(self, parent: tk.Misc, title: str) -> tk.LabelFrame:
        return tk.LabelFrame(
            parent,
            text=f"  {title}  ",
            bg=C["card"],
            fg=C["subtext"],
            font=("Helvetica", 9),
            padx=10,
            pady=6,
            relief="flat",
            bd=1,
            highlightbackground=C["accent"],
        )

    def _setup_shortcuts(self) -> None:
        def _guard(fn):
            def _inner(_event):
                if isinstance(self.root.focus_get(), tk.Entry):
                    return
                fn()

            return _inner

        for key in ("<r>", "<R>"):
            self.root.bind(key, _guard(self._toggle_record))
        for key in ("<t>", "<T>"):
            self.root.bind(key, _guard(self._start_timed))
        for key in ("<q>", "<Q>"):
            self.root.bind(key, _guard(self._on_close))

    def _open_stream(self) -> None:
        if self._stream is not None:
            self._stream.close()
        self._stream = MatrixStream(
            self.adb,
            self.source,
            touch_safe_sleep_ms=self.touch_safe_sleep_ms,
        )

    def _append_recording_rows(
        self,
        ts_ms: int,
        elapsed_s: float,
        frame: int,
        raw: np.ndarray,
        delta: np.ndarray,
    ) -> None:
        # One record per frame; the full raw/delta grids are written to the wide
        # capacitance.csv (one CSV row per frame) at save time.
        self._rec_rows.append(
            {
                "timestamp_ms": ts_ms,
                "elapsed_s": elapsed_s,
                "frame": frame,
                "raw": raw.copy(),
                "delta": delta.copy(),
            }
        )

    def _restart_stream(self) -> bool:
        self._stream_restarts += 1
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        time.sleep(0.8)
        if not adb_device_ready(self.adb):
            return False
        try:
            self._open_stream()
            self._stream_failures = 0
            return True
        except RuntimeError:
            return False

    def _read_frame(self) -> np.ndarray:
        if self._stream is not None:
            return self._stream.read()
        return read_matrix(self.adb, self.source)

    def _handle_stream_stall(self, reason: str) -> None:
        self._stream_failures += 1
        hint = "replug USB if touch is dead"
        with self._lock:
            self._read_error = reason
            self._status_text = (
                f"Stream stalled — restarting ({self._stream_restarts + 1})…  "
                f"{hint}"
            )
        if self._restart_stream():
            with self._lock:
                self._status_text = (
                    f"Stream restarted ({self._stream_restarts}). Waiting for frames…"
                )
                self._read_error = ""
            return
        backoff = min(5.0, 0.5 * self._stream_failures)
        with self._lock:
            self._status_text = (
                f"ADB not responding — replug USB, wait for device, then retry…  "
                f"({reason})"
            )
        time.sleep(backoff)

    def _read_loop(self) -> None:
        last_read = time.time()
        try:
            if self.read_mode == "stream":
                self._open_stream()
            while self.running:
                loop_start = time.time()
                try:
                    raw = self._read_frame()
                    now = time.time()
                    hz = 1.0 / (now - last_read) if now > last_read else 0.0
                    last_read = now
                    self._stream_failures = 0

                    if self.baseline is None:
                        self.baseline = raw.copy()
                        self._save_baseline()
                        with self._lock:
                            cap_note = (
                                f", host cap {self.stream_rate_hz:g} Hz"
                                if self.read_mode == "stream" and self.stream_rate_hz > 0
                                else ""
                            )
                            self._status_text = (
                                f"Baseline captured (first frame). "
                                f"Live via {self.read_mode}{cap_note}."
                            )
                            self._read_error = ""
                    else:
                        delta = raw - self.baseline
                        skipped = self._stream.skipped_frames if self._stream else 0
                        frame = self._frame_count + 1
                        rec_active = self.recording
                        rec_start = self._rec_start
                        with self._lock:
                            self._latest_raw = raw
                            self._latest_delta = delta
                            self._frame_count = frame
                            self._last_hz = hz
                            self._status_text = (
                                f"Frame {frame}  |  {hz:.2f} Hz  |  "
                                f"Δ {np.min(delta):.1f}…{np.max(delta):.1f}"
                            )
                            if skipped:
                                self._status_text += f"  |  skipped {skipped} bad frames"
                            if self._stream_restarts:
                                self._status_text += (
                                    f"  |  restarts {self._stream_restarts}"
                                )
                            self._read_error = ""

                        if rec_active:
                            self._append_recording_rows(
                                int(now * 1000),
                                now - rec_start,
                                frame,
                                raw,
                                delta,
                            )
                except TimeoutError as exc:
                    if self.read_mode == "stream":
                        self._handle_stream_stall(str(exc))
                    else:
                        with self._lock:
                            self._read_error = str(exc)
                            self._status_text = f"Read error: {exc}"
                        time.sleep(0.5)
                except RuntimeError as exc:
                    if self.read_mode == "stream":
                        self._handle_stream_stall(str(exc))
                    else:
                        with self._lock:
                            self._read_error = str(exc)
                            self._status_text = f"Read error: {exc}"
                        time.sleep(1.0)
                except Exception as exc:
                    with self._lock:
                        self._read_error = str(exc)
                        self._status_text = f"Read error: {exc}"
                    time.sleep(1.0)

                if self.interval_ms > 0:
                    time.sleep(self.interval_ms / 1000.0)
                elif self.read_mode == "stream" and self.stream_rate_hz > 0:
                    elapsed = time.time() - loop_start
                    time.sleep(max(0.0, (1.0 / self.stream_rate_hz) - elapsed))
        finally:
            if self._stream is not None:
                self._stream.close()
                self._stream = None

    def _animate(self, _frame: int) -> tuple:
        with self._lock:
            delta = None if self._latest_delta is None else self._latest_delta.copy()
            status = self._status_text
            is_rec = self.recording
            rec_el = (time.time() - self._rec_start) if is_rec else 0.0

        if delta is not None:
            self._im.set_data(delta)
            vmin, vmax = symmetric_limits(delta, self.color_max)
            self._im.set_clim(vmin, vmax)

        self._status_lbl.config(text=status)
        self._rec_badge.config(text=f"● REC  {rec_el:.1f}s" if is_rec else "")
        return (self._im,)

    def _toggle_record(self) -> None:
        if self.recording:
            self._stop_record()
        else:
            self._start_record()

    def _start_record(self) -> None:
        if not self.running:
            messagebox.showwarning("Not connected", "Wait for the adb read loop to start.")
            return
        if self.baseline is None:
            messagebox.showwarning("Not ready", "Wait for the first-frame baseline.")
            return
        if self.recording:
            return
        self.recording = True
        self._rec_rows = []
        self._rec_start = time.time()
        self._set_recording_buttons(recording=True)

    def _set_recording_buttons(self, *, recording: bool) -> None:
        if recording:
            self._rec_btn.config(text="■  Stop Recording   [R]", bg=C["red"])
        else:
            self._rec_btn.config(text="▶  Start Recording   [R]", bg=C["green"])

    def _stop_record(self) -> None:
        if not self.recording:
            return
        self.recording = False
        rows = list(self._rec_rows)
        duration = time.time() - self._rec_start

        self._set_recording_buttons(recording=False)
        self._rec_badge.config(text="")
        if self._timer:
            self._timer.cancel()
            self._timer = None
        self._countdown_lbl.config(text="")
        self._timed_btn.config(state="normal")

        self.root.after(0, self._save_and_show, rows, duration)

    def _start_timed(self) -> None:
        if not self.running:
            messagebox.showwarning("Not connected", "Wait for the adb read loop to start.")
            return
        if self.baseline is None:
            messagebox.showwarning("Not ready", "Wait for the first-frame baseline.")
            return
        if self.recording:
            return
        try:
            secs = float(self._dur_var.get())
            if secs <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid duration", "Enter a positive number of seconds.")
            return

        self._start_record()
        self._timed_btn.config(state="disabled")
        self._tick_countdown(secs)
        self._timer = threading.Timer(secs, lambda: self.root.after(0, self._stop_record))
        self._timer.start()

    def _tick_countdown(self, remaining: float) -> None:
        if not self.recording:
            return
        self._countdown_lbl.config(text=f"Remaining: {remaining:.0f} s")
        if remaining > 0:
            self.root.after(500, self._tick_countdown, remaining - 0.5)

    def _save_baseline(self) -> None:
        if self.baseline is None:
            return
        path = self.data_dir / "baseline.csv"
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["row", "col", "value"])
            for row in range(ROWS):
                for col in range(COLS):
                    writer.writerow([row, col, f"{self.baseline[row, col]:.6g}"])

    def _write_capacitance_csv(self, path: Path, rows: list[dict]) -> None:
        with path.open("w", newline="") as handle:
            writer = CapWideWriter(
                handle, raw_fmt="{:.6g}", delta_fmt="{:.6g}", elapsed_fmt="{:.3f}"
            )
            for row in rows:
                writer.write_frame(
                    row["timestamp_ms"],
                    row["elapsed_s"],
                    row["frame"],
                    row["raw"],
                    row["delta"],
                )

    def _save_and_show(self, rows: list[dict], duration: float) -> None:
        if not rows:
            self._result_lbl.config(text="Recording produced no data.", fg=C["red"])
            return

        session_id = datetime.datetime.fromtimestamp(self._rec_start).strftime(
            "%Y%m%d_%H%M%S"
        )
        session_dir = self.data_dir / f"session_{session_id}"
        session_dir.mkdir(parents=True, exist_ok=True)
        filename = session_dir / self.config.csv_filename
        self._write_capacitance_csv(filename, rows)

        if self.baseline is not None:
            np.savetxt(session_dir / "baseline_matrix.csv", self.baseline, delimiter=",", fmt="%.6g")

        frames = sorted({row["frame"] for row in rows})
        lines = [
            f"Duration : {duration:.1f} s     Frames : {len(frames)}",
            f"Rows     : {len(rows) * ROWS * COLS}  ({ROWS}×{COLS} cells × frames)",
            f"Source   : {self.source} ({self.read_mode})",
            f"Rate     : {len(frames) / max(duration, 1e-6):.2f} Hz avg",
            f"Saved →  session_{session_id}/{self.config.csv_filename}",
        ]
        self._result_lbl.config(text="\n".join(lines), fg=C["green"])
        print(f"Saved recording to {filename}", flush=True)

    def _on_close(self) -> None:
        if self.recording:
            self._stop_record()
        self.running = False
        if self._timer:
            self._timer.cancel()
        self._anim.event_source.stop()
        self.root.destroy()


def run_snapshot(
    config: MonitorConfig,
    adb: Adb,
    source: str,
    snapshot: Path,
    color_max: float | None,
) -> int:
    try:
        baseline = read_matrix(adb, source)
        current = read_matrix(adb, source)
        values = current - baseline
    except (RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    vmin, vmax = symmetric_limits(values, color_max)
    figure, axis = plt.subplots(figsize=(8, 11))
    axis.imshow(
        values,
        origin="upper",
        aspect="equal",
        cmap=build_rwgb_colormap(),
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    axis.set(title=config.snapshot_title)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(snapshot, dpi=180)
    plt.close(figure)
    print(f"Saved snapshot to {snapshot}")
    return 0


def run_main(config: MonitorConfig) -> int:
    parser = argparse.ArgumentParser(description=config.doc or None)
    parser.add_argument("--adb", help="Path to adb executable")
    parser.add_argument("--serial", help="ADB device serial")
    parser.add_argument(
        "--source",
        choices=config.live_sources,
        default=config.default_source,
        help="Controller matrix to stream (raw=raw_data, baseline=basline_data, delta=diff_data)",
    )
    parser.add_argument(
        "--read-mode",
        choices=("stream", "poll"),
        default="stream",
        help="stream: one persistent adb shell (fast); poll: adb cat per frame",
    )
    parser.add_argument("--interval-ms", type=int, default=DEFAULT_INTERVAL_MS)
    parser.add_argument(
        "--stream-rate-hz",
        type=float,
        default=DEFAULT_STREAM_RATE_HZ,
        help="Optional host-side max rate in stream mode; 0 = as fast as adb allows",
    )
    parser.add_argument(
        "--touch-safe-sleep-ms",
        type=int,
        default=DEFAULT_TOUCH_SAFE_SLEEP_MS,
        help="Minimum sleep between device-side reads (default 30 ms)",
    )
    parser.add_argument(
        "--color-max",
        type=float,
        default=config.default_color_max,
        help="Fixed symmetric color limit ±value; 0 = auto from data (default)",
    )
    parser.add_argument("--data-dir", type=Path, default=config.log_dir)
    parser.add_argument("--snapshot", type=Path, help="Save one PNG and exit")
    args = parser.parse_args()

    if args.interval_ms < 0:
        parser.error("--interval-ms cannot be negative")
    if args.stream_rate_hz < 0:
        parser.error("--stream-rate-hz cannot be negative")
    if args.touch_safe_sleep_ms < 0:
        parser.error("--touch-safe-sleep-ms cannot be negative")

    color_max = None if args.color_max == 0 else args.color_max

    try:
        adb = Adb(find_adb(args.adb), args.serial)
    except FileNotFoundError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if args.snapshot:
        return run_snapshot(config, adb, args.source, args.snapshot, color_max)

    root = tk.Tk()
    OnePlusLiveMapMonitor(
        root,
        config=config,
        adb=adb,
        source=args.source,
        read_mode=args.read_mode,
        interval_ms=args.interval_ms,
        stream_rate_hz=args.stream_rate_hz,
        touch_safe_sleep_ms=args.touch_safe_sleep_ms,
        color_max=color_max,
        data_dir=args.data_dir,
    )
    root.mainloop()
    return 0


def main() -> int:
    return run_main(CAPACITANCE_CONFIG)


if __name__ == "__main__":
    raise SystemExit(main())
