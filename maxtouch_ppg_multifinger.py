#!/usr/bin/env python3
"""Log maXTouch T37 capacitance (single-page region) and PPG green channel.

Capacitance is read over USB HID from the maXTouch T37 diagnostic object. By default
the script streams PAGE0_WIDTH columns × (PAGE0_ROW_FRACTION × matrix Y rows)
from the origin (default 4 columns × one-third of row length).

PPG uses the same MAX30101 binary protocol as oneplus_ppg_monitor.py (0xAA sync).
Pass --ppg-pick-port to choose the Arduino serial port from a dropdown before
recording; the session timer starts only after that selection.
Live viewing matches oneplus_ppg_monitor.py / pixel_tablet_ppg_monitor.py: a
capacitance heatmap, a live mean-Δ line plot per auto-detected 6×6 finger patch
(detection runs once ~10 s after start), plus streaming PPG green trace. After
finger detection, a strategy window opens per finger that refreshes every ~1 s
with bandpassed CPG waveforms (All-N mean, Top-8 variance, Best px, Top-4
PPG-corr mean) and FFT from the past ~10 s, while time-series plots show the
past ~20 s. Optional --pixel-stream opens extra windows with every pixel's live
Δ (heatmap + stacked time series) for each detected finger.

Recording runs until you close the viewer window (or Ctrl+C with --no-viewer).
Activity labels are saved to capacitance_labels.csv in the session folder.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import select
import signal
import struct
import subprocess
import sys
import threading
import termios
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from activity_labels import (
    draw_label_markers,
    labels_path_for_session,
    load_activity_labels,
    save_activity_labels,
    sync_metadata_labels,
)
from maxtouch_hid import DEBUG_DIAGNOSTIC_T37, MaxTouchHid
from finger_live_view import (
    CAP_TRACE_WINDOW_S,
    FINGER_PATCH_SIZE,
    FINGER_PIXEL_STREAM_DELAY_AFTER_DETECT_S,
    STRATEGY_STREAM_DISPLAY_WINDOW_S,
    STRATEGY_STREAM_MIN_SPAN_S,
    STRATEGY_STREAM_PROCESS_WINDOW_S,
    STRATEGY_STREAM_UPDATE_S,
    FingerPatch,
    FingerPixelStreamWindows,
    FingerStrategyStreamWindows,
    detect_finger_patches,
)
from oneplus_monitor import build_rwgb_colormap, symmetric_limits

# Edit these defaults before running.
RUN_NAME = "maxtouch_moving_ppg"
OUT_DIR = Path("data_maxtouch/cap_log")
REGION_ORIGIN_X = 0
REGION_ORIGIN_Y = 0
PAGE0_WIDTH = 6  # mutual-sense X columns
PAGE0_ROW_FRACTION = 1.0 / 3.0  # fraction of full Y row length (matrix_y)
RUN_VIEWER = True
VIEW_INVERT_DELTA = True
VIEW_COLOR_MAX = 0.0  # 0 = auto-scale symmetric limits from latest frame
VIEW_REFRESH_MS = 50  # ~20 Hz UI poll; drain all new cap frames each tick
PPG_ENABLED = True
PPG_PORT = ""  # empty → auto-detect /dev/cu.usbmodem* or /dev/cu.usbserial*
PPG_BAUD = 115_200
PPG_SYNC = 0xAA
PPG_CSV_NAME = "ppg_green.csv"
PPG_AUTO_KILL_PORT_OWNER = True
PPG_PICK_PORT = False  # set True or pass --ppg-pick-port to show serial-port dropdown
SERIAL_PORT_SKIP = frozenset(
    {
        "Bluetooth-Incoming-Port",
        "debug-console",
    }
)

CAP_CSV_NAME = "capacitance.csv"
CAP_CSV_FIELDS = ("timestamp_ms", "elapsed_s", "frame", "row", "col", "raw", "delta")
PPG_CSV_FIELDS = ("timestamp_ms", "t_s", "green")
FINGER_DETECT_DELAY_S = 10.0
ENABLE_PIXEL_STREAM = False


@dataclass(frozen=True)
class Region:
    start_x: int
    start_y: int
    width: int
    height: int


@dataclass(frozen=True)
class Settings:
    name: str
    out_dir: Path
    region_origin_x: int
    region_origin_y: int
    page0_width: int
    page0_row_fraction: float
    fixed_region: str
    run_viewer: bool
    view_invert_delta: bool
    view_color_max: float | None
    view_refresh_ms: int
    ppg_enabled: bool
    ppg_port: str
    ppg_baud: int
    ppg_auto_kill_port_owner: bool
    ppg_pick_port: bool
    finger_detect_delay_s: float
    finger_patch_size: int
    enable_pixel_stream: bool


def region_pages(region: Region, matrix_y: int, page_size: int) -> list[int]:
    pages: set[int] = set()
    for row in range(region.height):
        for col in range(region.width):
            node = (region.start_x + col) * matrix_y + region.start_y + row
            pages.add((node * 2) // page_size)
    return sorted(pages)


def max_same_page_region_from_origin(
    start_x: int,
    start_y: int,
    matrix_x: int,
    matrix_y: int,
    page_size: int,
) -> Region:
    start_node = start_x * matrix_y + start_y
    start_page = (start_node * 2) // page_size
    best = Region(start_x, start_y, 1, 1)
    best_area = 1
    for width in range(1, matrix_x - start_x + 1):
        for height in range(1, matrix_y - start_y + 1):
            candidate = Region(start_x, start_y, width, height)
            if region_pages(candidate, matrix_y, page_size) == [start_page]:
                area = width * height
                if area > best_area:
                    best = candidate
                    best_area = area
    return best


def default_page0_region(
    matrix_x: int,
    matrix_y: int,
    origin_x: int,
    origin_y: int,
    width: int,
    row_fraction: float,
) -> Region:
    """Fixed width from origin; height = fraction of full mutual-sense Y length."""
    if width <= 0:
        raise ValueError("page0 width must be positive")
    if not 0 < row_fraction <= 1:
        raise ValueError("page0_row_fraction must be in (0, 1]")
    width = min(width, matrix_x - origin_x)
    available_rows = matrix_y - origin_y
    height = max(1, int(available_rows * row_fraction))
    height = min(height, available_rows)
    if width <= 0 or height <= 0:
        raise ValueError("page0 region origin is outside the matrix")
    return Region(origin_x, origin_y, width, height)


def cap_grid_for_display(grid: np.ndarray) -> np.ndarray:
    """Orient capacitance grid for viewing (90° rotate, then flip vertical)."""
    return np.flipud(np.rot90(grid, k=1))



def estimated_region_hz(page_commands: int) -> float:
    if page_commands <= 0:
        return 0.0
    return 110.0 / page_commands


def parse_fixed_region(text: str) -> Region:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 4:
        raise ValueError("region must be start_x,start_y,width,height")
    start_x, start_y, width, height = [int(part) for part in parts]
    if start_x < 0 or start_y < 0 or width <= 0 or height <= 0:
        raise ValueError("region requires nonnegative origin and positive size")
    return Region(start_x, start_y, width, height)


def resolve_region(
    mxt: MaxTouchHid,
    args: Settings,
) -> tuple[Region, int, list[int], float]:
    assert mxt.id_info is not None
    matrix_x = mxt.id_info.matrix_x
    matrix_y = mxt.id_info.matrix_y
    t37_size = mxt.object_size(DEBUG_DIAGNOSTIC_T37, 0)
    if t37_size is None:
        raise RuntimeError("T37 object not found")
    page_size = t37_size - 2

    if args.fixed_region:
        region = parse_fixed_region(args.fixed_region)
    else:
        region = default_page0_region(
            matrix_x,
            matrix_y,
            args.region_origin_x,
            args.region_origin_y,
            args.page0_width,
            args.page0_row_fraction,
        )

    if (
        region.start_x + region.width > matrix_x
        or region.start_y + region.height > matrix_y
    ):
        raise ValueError(
            f"region {region} exceeds matrix {matrix_x}x{matrix_y}"
        )

    pages = region_pages(region, matrix_y, page_size)
    page_commands = max(pages) + 1 if pages else 0
    return region, page_size, pages, estimated_region_hz(page_commands)


def make_run_dir(args: Settings) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.out_dir / f"session_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_metadata(
    run_dir: Path,
    args: Settings,
    mxt: MaxTouchHid,
    region: Region,
    pages: list[int],
    page_size: int,
    estimated_hz: float,
    *,
    ppg_port_override: str | None = None,
) -> None:
    assert mxt.id_info is not None
    metadata = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": args.name,
        "duration_seconds": None,
        "activity_labels": [],
        "maxtouch": {
            "matrix_shape": f"{mxt.id_info.matrix_x}x{mxt.id_info.matrix_y}",
            "region": {
                "start_x": region.start_x,
                "start_y": region.start_y,
                "width": region.width,
                "height": region.height,
            },
            "t37_page_size_bytes": page_size,
            "t37_pages_spanned": pages,
            "page_commands_per_frame": max(pages) + 1 if pages else 0,
            "estimated_region_hz": estimated_hz,
            "notes": [
                "Default region: PAGE0_WIDTH columns × PAGE0_ROW_FRACTION of matrix Y rows.",
                "T37 pages are 0-indexed in firmware; wider regions may span multiple pages.",
            ],
            "page0_width": args.page0_width,
            "page0_row_fraction": args.page0_row_fraction,
            "matrix_y_rows": mxt.id_info.matrix_y,
        },
        "arduino": {
            "enabled": args.ppg_enabled,
            "port": ppg_port_override or args.ppg_port or "auto",
            "baud": args.ppg_baud,
            "csv": PPG_CSV_NAME,
            "protocol": "MAX30101 binary (0xAA + red/ir/green uint32)",
            "auto_kill_port_owner": args.ppg_auto_kill_port_owner,
        },
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def load_session_t0_s(run_dir: Path) -> float:
    meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    started = datetime.fromisoformat(meta["started_utc"])
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    return started.timestamp()


def finalize_session_metadata(run_dir: Path, labels: list[tuple[float, str]]) -> None:
    meta_path = run_dir / "metadata.json"
    if not meta_path.is_file():
        return
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return

    duration: float | None = None
    cap_summary = run_dir / "capacitance_summary.json"
    if cap_summary.is_file():
        try:
            summary = json.loads(cap_summary.read_text(encoding="utf-8"))
            duration = float(summary.get("actual_duration_s", 0.0))
        except (json.JSONDecodeError, TypeError, ValueError):
            duration = None

    metadata["duration_seconds"] = duration
    metadata["activity_labels"] = [
        {"elapsed_s": elapsed, "label": label} for elapsed, label in labels
    ]
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def record_capacitance(
    mxt: MaxTouchHid,
    run_dir: Path,
    region: Region,
    args: Settings,
    pages: list[int],
    estimated_hz: float,
    stop_event: threading.Event,
) -> int:
    csv_path = run_dir / CAP_CSV_NAME
    summary_path = run_dir / "capacitance_summary.json"
    start = time.monotonic()
    frame = 0
    successes = 0
    completed_at = start
    errors: dict[str, str] = {}

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CAP_CSV_FIELDS)
        while not stop_event.is_set():
            cycle_start = time.monotonic()
            elapsed = cycle_start - start
            try:
                crop, _read_s = mxt.read_mutual_deltas_region(
                    region.start_x,
                    region.start_y,
                    region.width,
                    region.height,
                )
            except (RuntimeError, TimeoutError, OSError) as error:
                errors["read"] = str(error)
                time.sleep(0.01)
                continue

            frame += 1
            ts_ms = int(time.time() * 1000)
            for row in range(region.height):
                for col in range(region.width):
                    value = float(crop[row][col])
                    writer.writerow(
                        [
                            ts_ms,
                            f"{elapsed:.6f}",
                            frame,
                            row,
                            col,
                            f"{value:.6g}",
                            f"{value:.2f}",
                        ]
                    )
            handle.flush()
            successes += 1
            completed_at = time.monotonic()

    actual_duration_s = max(0.0, completed_at - start)
    achieved_frame_rate_hz = (successes / actual_duration_s) if actual_duration_s else 0.0
    summary_path.write_text(
        json.dumps(
            {
                "frames": frame,
                "matrix_records": successes,
                "errors": errors,
                "csv": CAP_CSV_NAME,
                "actual_duration_s": actual_duration_s,
                "achieved_frame_rate_hz": achieved_frame_rate_hz,
                "region": {
                    "start_x": region.start_x,
                    "start_y": region.start_y,
                    "width": region.width,
                    "height": region.height,
                },
                "t37_pages_spanned": pages,
                "page_commands_per_frame": max(pages) + 1 if pages else 0,
                "estimated_region_hz": estimated_hz,
                "matrix_shape": f"{region.width}x{region.height}",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if successes == 0:
        print("No capacitance frames were recorded. See capacitance_summary.json.")
        return 1
    print(
        f"Capacitance achieved frame rate: {achieved_frame_rate_hz:.3f} Hz "
        f"({successes} frames over {actual_duration_s:.1f} s)"
    )
    return 0


@dataclass
class LiveTrace:
    label: str
    x: list[float]
    y: list[float]


class CapHeatmapTailer:
    """Tail capacitance.csv and rebuild the latest delta grid per frame."""

    def __init__(
        self,
        csv_path: Path,
        n_rows: int,
        n_cols: int,
        invert_delta: bool,
    ) -> None:
        self.csv_path = csv_path
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.invert_delta = invert_delta
        self.offset = 0
        self.ready = False
        self._grid = np.zeros((n_rows, n_cols), dtype=np.float32)
        self._current_frame = -1
        self._cells_seen = 0
        self.latest_grid: np.ndarray | None = None
        self.frame_count = -1
        self.elapsed_s = 0.0
        self.timestamp_ms = 0

    def _accept_row(self, row: list[str]) -> None:
        if len(row) != 7:
            return
        try:
            timestamp_ms = int(row[0])
            elapsed = float(row[1])
            frame = int(row[2])
            row_index = int(row[3])
            col_index = int(row[4])
            delta = float(row[6])
        except ValueError:
            return
        if not (0 <= row_index < self.n_rows and 0 <= col_index < self.n_cols):
            return

        if frame != self._current_frame:
            self._current_frame = frame
            self._cells_seen = 0
            self._grid.fill(0.0)
            self.timestamp_ms = timestamp_ms

        value = -delta if self.invert_delta else delta
        self._grid[row_index, col_index] = value
        self._cells_seen += 1
        self.elapsed_s = elapsed
        if self._cells_seen >= self.n_rows * self.n_cols:
            self.latest_grid = self._grid.copy()
            self.frame_count = frame

    def poll_completed_frames(self) -> list[tuple[np.ndarray, int, float, int]]:
        """Return every newly completed frame since the last poll."""
        if not self.csv_path.exists():
            return []
        completed: list[tuple[np.ndarray, int, float, int]] = []
        with self.csv_path.open("r", encoding="utf-8", newline="") as handle:
            if self.offset:
                handle.seek(self.offset)
            reader = csv.reader(handle)
            if not self.ready:
                next(reader, None)
                self.ready = True
            for row in reader:
                frame_before = self.frame_count
                self._accept_row(row)
                if self.latest_grid is not None and self.frame_count != frame_before:
                    completed.append(
                        (
                            self.latest_grid.copy(),
                            self.frame_count,
                            self.elapsed_s,
                            self.timestamp_ms,
                        )
                    )
            self.offset = handle.tell()
        return completed

    def poll(self) -> bool:
        return bool(self.poll_completed_frames())


class PpgLogTailer:
    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path
        self.offset = 0
        self.ready = False
        self.traces: dict[str, LiveTrace] = {}

    def poll(self) -> bool:
        if not self.csv_path.exists():
            return False
        with self.csv_path.open("r", encoding="utf-8", newline="") as handle:
            if self.offset:
                handle.seek(self.offset)
            if not self.ready:
                reader = csv.DictReader(handle)
                self.ready = True
            else:
                reader = csv.DictReader(handle, fieldnames=list(PPG_CSV_FIELDS))
            trace = self.traces.setdefault("green", LiveTrace("green", [], []))
            for record in reader:
                timestamp_ms = record.get("timestamp_ms", "")
                green = record.get("green", "")
                if timestamp_ms == "" or green == "":
                    continue
                trace.x.append(float(timestamp_ms))
                trace.y.append(float(green))
            self.offset = handle.tell()
        return True


def list_ppg_serial_ports() -> list[str]:
    return sorted(
        path
        for path in glob.glob("/dev/cu.*")
        if Path(path).name.removeprefix("cu.") not in SERIAL_PORT_SKIP
    )


def preferred_ppg_port(ports: list[str], requested: str = "") -> str:
    if requested and requested in ports:
        return requested
    for port in ports:
        if "11301" in port:
            return port
    for port in ports:
        if any(key in port for key in ("usbmodem", "usbserial", "SLAB_USBtoUART")):
            return port
    return ports[0] if ports else ""


def find_ppg_port(requested: str) -> str:
    if requested:
        return requested
    ports = list_ppg_serial_ports()
    if not ports:
        raise FileNotFoundError("No PPG serial port found under /dev/cu.*")
    return preferred_ppg_port(ports)


def pick_ppg_serial_port(default: str = "") -> str:
    """Show a dropdown of /dev/cu.* ports; return the selected path."""
    import tkinter as tk
    from tkinter import messagebox, ttk

    result: list[str | None] = [None]

    root = tk.Tk()
    root.title("PPG serial port")
    root.resizable(False, False)

    selected_port = tk.StringVar(master=root)
    port_combo = ttk.Combobox(
        root,
        textvariable=selected_port,
        values=[],
        state="readonly",
        width=52,
    )

    def refresh_ports(*, keep_selection: bool = True) -> None:
        ports = list_ppg_serial_ports()
        if not ports:
            port_combo["values"] = []
            selected_port.set("")
            status_var.set("No serial ports found — plug in the Arduino and click Refresh.")
            return
        port_combo["values"] = ports
        current = selected_port.get().strip()
        if keep_selection and current in ports:
            selected_port.set(current)
        else:
            selected_port.set(preferred_ppg_port(ports, default))
        status_var.set(f"{len(ports)} port(s) available.")

    def on_start() -> None:
        port = selected_port.get().strip()
        if not port:
            messagebox.showerror(
                "Serial port",
                "Select a PPG serial port, or plug in the Arduino and click Refresh.",
                parent=root,
            )
            return
        if port not in list_ppg_serial_ports():
            messagebox.showerror(
                "Serial port",
                f"{port} is no longer available. Click Refresh and try again.",
                parent=root,
            )
            return
        result[0] = port
        root.destroy()

    def on_cancel() -> None:
        root.destroy()

    tk.Label(
        root,
        text="Choose the Arduino / MAX30101 serial port for PPG green:",
    ).pack(padx=12, pady=(12, 4), anchor="w")
    port_combo.pack(padx=12, pady=(0, 4))
    status_var = tk.StringVar(master=root)
    tk.Label(root, textvariable=status_var, fg="#555555").pack(
        padx=12, pady=(0, 8), anchor="w"
    )

    buttons = tk.Frame(root)
    buttons.pack(padx=12, pady=(0, 12))
    tk.Button(buttons, text="Refresh", command=refresh_ports, width=10).pack(
        side="left", padx=4
    )
    tk.Button(buttons, text="Start", command=on_start, width=10).pack(side="left", padx=4)
    tk.Button(buttons, text="Cancel", command=on_cancel, width=10).pack(
        side="left", padx=4
    )

    root.protocol("WM_DELETE_WINDOW", on_cancel)
    refresh_ports(keep_selection=False)
    root.mainloop()

    if result[0] is None:
        raise SystemExit("cancelled")
    return result[0]


def configure_serial(fd: int, baud: int) -> None:
    baud_attr = getattr(termios, f"B{baud}", None)
    if baud_attr is None:
        raise ValueError(f"Unsupported baud rate for termios: {baud}")
    attrs = termios.tcgetattr(fd)
    attrs[0] = 0
    attrs[1] = 0
    attrs[2] = baud_attr | termios.CS8 | termios.CREAD | termios.CLOCAL
    attrs[3] = 0
    attrs[4] = baud_attr
    attrs[5] = baud_attr
    attrs[6][termios.VMIN] = 0
    attrs[6][termios.VTIME] = 1
    termios.tcsetattr(fd, termios.TCSANOW, attrs)
    termios.tcflush(fd, termios.TCIOFLUSH)


def port_owner_pids(port: str) -> list[int]:
    result = subprocess.run(
        ["lsof", "-t", port],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    pids: list[int] = []
    for line in result.stdout.splitlines():
        try:
            pids.append(int(line.strip()))
        except ValueError:
            continue
    return pids


def kill_port_owners(port: str) -> str:
    pids = port_owner_pids(port)
    for pid in pids:
        if pid == os.getpid():
            continue
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    if pids:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            remaining = [pid for pid in port_owner_pids(port) if pid != os.getpid()]
            if not remaining:
                break
            time.sleep(0.1)
    return ", ".join(str(pid) for pid in pids)


def open_serial_port(port: str, auto_kill_port_owner: bool) -> int:
    try:
        return os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    except OSError as exc:
        if exc.errno != 16 or not auto_kill_port_owner:
            raise
        killed = kill_port_owners(port)
        try:
            return os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        except OSError as retry_exc:
            raise OSError(
                retry_exc.errno,
                f"{retry_exc.strerror}; auto-killed port owner PID(s): {killed or 'none'}",
                retry_exc.filename,
            ) from retry_exc


def record_ppg_green(
    run_dir: Path,
    args: Settings,
    stop_event: threading.Event,
    *,
    ppg_port: str | None = None,
) -> None:
    csv_path = run_dir / PPG_CSV_NAME
    summary_path = run_dir / "ppg_summary.json"
    started = time.monotonic()
    rec_t0: float | None = None
    rows = 0
    error = ""
    port = ""

    try:
        port = find_ppg_port(ppg_port or args.ppg_port)
        fd = open_serial_port(port, args.ppg_auto_kill_port_owner)
        configure_serial(fd, args.ppg_baud)
        termios.tcflush(fd, termios.TCIFLUSH)
        with os.fdopen(fd, "rb", buffering=0) as serial_handle:
            buffer = bytearray()
            with csv_path.open("w", newline="", encoding="utf-8") as csv_handle:
                writer = csv.writer(csv_handle)
                writer.writerow(PPG_CSV_FIELDS)
                csv_handle.flush()
                while not stop_event.is_set():
                    readable, _, _ = select.select([serial_handle], [], [], 0.05)
                    if not readable:
                        continue
                    chunk = os.read(serial_handle.fileno(), 4096)
                    if not chunk:
                        continue
                    buffer.extend(chunk)
                    while buffer:
                        sync_idx = buffer.find(bytes([PPG_SYNC]))
                        if sync_idx < 0:
                            buffer.clear()
                            break
                        if sync_idx > 0:
                            del buffer[:sync_idx]
                        if len(buffer) < 13:
                            break
                        packet = bytes(buffer[1:13])
                        del buffer[:13]
                        _, _, green = struct.unpack("<III", packet)
                        now = time.time()
                        if rec_t0 is None:
                            rec_t0 = now
                        elapsed = now - rec_t0
                        writer.writerow([int(now * 1000), f"{elapsed:.6f}", int(green)])
                        csv_handle.flush()
                        rows += 1
    except (OSError, ValueError, FileNotFoundError) as exc:
        error = str(exc)
        if "Resource busy" in error:
            error += " (close Serial Monitor or any other app using this port)"
        if not csv_path.exists():
            with csv_path.open("w", newline="", encoding="utf-8") as csv_handle:
                csv.writer(csv_handle).writerow(PPG_CSV_FIELDS)
    finally:
        duration = max(0.0, time.monotonic() - started)
        summary_path.write_text(
            json.dumps(
                {
                    "enabled": args.ppg_enabled,
                    "port": port or args.ppg_port or "auto",
                    "baud": args.ppg_baud,
                    "csv": PPG_CSV_NAME,
                    "protocol": "MAX30101 green channel (0xAA sync)",
                    "auto_kill_port_owner": args.ppg_auto_kill_port_owner,
                    "samples": rows,
                    "actual_duration_s": duration,
                    "achieved_sample_rate_hz": rows / duration if duration else 0.0,
                    "error": error,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if error:
            print(f"PPG unavailable: {error}")
        elif rows:
            print(f"PPG green achieved sample rate: {rows / duration if duration else 0.0:.3f} Hz")


class ActivityLabelPanel:
    """Track activity labels during live recording and persist on save."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.labels: list[tuple[float, str]] = []
        self._label_artists: list = []

    def add(self, elapsed_s: float, text: str) -> None:
        text = text.strip()
        if not text:
            return
        self.labels.append((elapsed_s, text))
        self.labels.sort(key=lambda item: item[0])

    def save(self) -> None:
        path = labels_path_for_session(self.run_dir)
        save_activity_labels(path, self.labels)
        sync_metadata_labels(self.run_dir, self.labels)
        if self.labels:
            print(f"Saved {len(self.labels)} activity label(s) → {path.name}")

    def redraw_markers(self, axes) -> None:
        for artist in self._label_artists:
            try:
                artist.remove()
            except ValueError:
                pass
        self._label_artists = draw_label_markers(axes, self.labels)


class LiveCsvViewer:
    def __init__(
        self,
        run_dir: Path,
        n_rows: int,
        n_cols: int,
        refresh_ms: int,
        cap_tailer: CapHeatmapTailer,
        ppg_tailer: PpgLogTailer | None,
        stop_event: threading.Event,
        label_panel: ActivityLabelPanel,
        *,
        color_max: float | None,
        finger_detect_delay_s: float = FINGER_DETECT_DELAY_S,
        finger_patch_size: int = FINGER_PATCH_SIZE,
        enable_pixel_stream: bool = ENABLE_PIXEL_STREAM,
        estimated_cap_hz: float = 0.0,
        include_ppg_corr: bool = True,
    ) -> None:
        self.run_dir = run_dir
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.cap_tailer = cap_tailer
        self.ppg_tailer = ppg_tailer
        self.refresh_ms = refresh_ms
        self.stop_event = stop_event
        self.label_panel = label_panel
        self.color_max = color_max
        self.finger_detect_delay_s = finger_detect_delay_s
        self.finger_patch_size = finger_patch_size
        self.enable_pixel_stream = enable_pixel_stream
        self.estimated_cap_hz = estimated_cap_hz
        self.include_ppg_corr = include_ppg_corr
        self.session_t0_s = load_session_t0_s(run_dir)
        self.finger_patches: list[FingerPatch] | None = None
        self.finger_detect_attempted = False
        self.finger_detected_at_s: float | None = None
        self.pixel_stream_windows: FingerPixelStreamWindows | None = None
        self.pixel_stream_opened = False
        self.strategy_stream_windows: FingerStrategyStreamWindows | None = None
        self.strategy_stream_opened = False
        self.cap_trace_times: list[float] = []
        self.cap_trace_values: dict[str, list[float]] = {}
        self._last_traced_frame = -1

    def aligned_s(self, timestamp_ms: float) -> float:
        return timestamp_ms / 1000.0 - self.session_t0_s

    def aligned_elapsed(self) -> float:
        if self.cap_tailer.timestamp_ms > 0:
            return self.aligned_s(self.cap_tailer.timestamp_ms)
        return self.cap_tailer.elapsed_s

    def measured_cap_hz(self) -> float | None:
        times = self.cap_trace_times
        if len(times) < 2:
            return None
        window_s = 5.0
        start = times[-1] - window_s
        start_idx = int(np.searchsorted(times, start))
        window = times[start_idx:]
        if len(window) < 2:
            return None
        dt = window[-1] - window[0]
        if dt <= 0:
            return None
        return (len(window) - 1) / dt

    def show(self) -> None:
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from matplotlib.patches import Rectangle
        from matplotlib.widgets import Button, TextBox

        cmap = build_rwgb_colormap()
        grid = cap_grid_for_display(
            np.zeros((self.n_rows, self.n_cols), dtype=np.float32)
        )
        display_rows, display_cols = grid.shape
        vmin, vmax = symmetric_limits(grid, self.color_max)
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

        if self.ppg_tailer is not None:
            fig = plt.figure(figsize=(12, 11.2))
            spec = GridSpec(
                3, 1, height_ratios=[3.5, 1.0, 1.2], hspace=0.35,
                left=0.08, right=0.96, top=0.94, bottom=0.16,
            )
            ax_cap = fig.add_subplot(spec[0])
            ax_cap_trace = fig.add_subplot(spec[1])
            ax_ppg = fig.add_subplot(spec[2], sharex=ax_cap_trace)
        else:
            fig = plt.figure(figsize=(12, 8.2))
            spec = GridSpec(
                2, 1, height_ratios=[3.5, 1.0], hspace=0.35,
                left=0.08, right=0.96, top=0.94, bottom=0.16,
            )
            ax_cap = fig.add_subplot(spec[0])
            ax_cap_trace = fig.add_subplot(spec[1])
            ax_ppg = None

        label_axes = [ax_cap_trace]
        if ax_ppg is not None:
            label_axes.append(ax_ppg)
        recent_label_text = fig.text(
            0.08, 0.11, "Recent: —", ha="left", va="center", fontsize=9, color="#555555"
        )

        def refresh_recent_label_text() -> None:
            if not self.label_panel.labels:
                recent_label_text.set_text("Recent: —")
                return
            elapsed, text = self.label_panel.labels[-1]
            recent_label_text.set_text(f"Recent: t={elapsed:.1f}s  {text}")

        def add_label(_event=None) -> None:
            elapsed = self.aligned_elapsed()
            self.label_panel.add(elapsed, label_box.text)
            label_box.set_val("")
            refresh_recent_label_text()
            self.label_panel.redraw_markers(label_axes)
            fig.canvas.draw_idle()

        label_box_ax = fig.add_axes([0.08, 0.03, 0.62, 0.045])
        label_box_ax.set_zorder(200)
        label_box_ax.set_navigate(False)
        label_box = TextBox(label_box_ax, "Label activity: ", initial="")
        label_box.label.set_fontsize(9)
        label_box.on_submit(add_label)

        label_btn_ax = fig.add_axes([0.72, 0.03, 0.12, 0.045])
        label_btn_ax.set_zorder(200)
        label_btn_ax.set_navigate(False)
        label_btn = Button(label_btn_ax, "Add Label")
        label_btn.label.set_fontsize(9)
        label_btn.on_clicked(add_label)

        im = ax_cap.imshow(
            grid,
            norm=norm,
            cmap=cmap,
            aspect="auto",
            origin="upper",
            interpolation="nearest",
        )
        ax_cap.set_xlabel(f"Mutual-sense Y ({self.n_rows} rows)")
        ax_cap.set_ylabel(f"Mutual-sense X ({self.n_cols} columns)")
        ax_cap.set_xticks(np.arange(-0.5, display_cols, 1), minor=True)
        ax_cap.set_yticks(np.arange(-0.5, display_rows, 1), minor=True)
        ax_cap.grid(which="minor", color="lightgrey", linestyle="-", linewidth=0.3)
        ax_cap.tick_params(which="minor", size=0)
        cap_title = ax_cap.set_title(
            f"Live capacitance Δ — {self.run_dir.name}  |  "
            f"{self.n_cols}×{self.n_rows} region"
        )
        cbar = fig.colorbar(im, ax=ax_cap, fraction=0.046, pad=0.04)
        cbar.set_label("Δ (T37 diagnostic units)")

        patch_overlay: list[object] = []
        cap_trace_title = ax_cap_trace.set_title(
            f"Live capacitance mean ({self.finger_patch_size}×{self.finger_patch_size} patch)"
        )
        ax_cap_trace.set_ylabel("Mean Δ")
        ax_cap_trace.grid(alpha=0.3)
        cap_trace_artists: dict[str, object] = {}
        cap_trace_status = ax_cap_trace.text(
            0.02,
            0.95,
            f"Finger detection at t={self.finger_detect_delay_s:.0f}s…",
            transform=ax_cap_trace.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="#555555",
        )

        ppg_artists: dict[str, object] = {}
        if ax_ppg is not None:
            ax_ppg.set_title("Live PPG green")
            ax_ppg.set_xlabel("Elapsed seconds (session start)")
            ax_ppg.set_ylabel("Green ADC")
            plt.setp(ax_cap_trace.get_xticklabels(), visible=False)
        else:
            ax_cap_trace.set_xlabel("Elapsed seconds (session start)")

        def sync_time_axes(latest_s: float) -> None:
            start = max(0.0, latest_s - CAP_TRACE_WINDOW_S)
            end = max(latest_s, start + 1.0)
            ax_cap_trace.set_xlim(start, end)

        def update_cap_trace_title() -> None:
            rate = self.measured_cap_hz()
            rate_text = f"{rate:.1f} Hz" if rate is not None else f"~{self.estimated_cap_hz:.1f} Hz est."
            cap_trace_title.set_text(
                f"Live capacitance mean ({self.finger_patch_size}×{self.finger_patch_size} patch)  |  "
                f"{rate_text}"
            )

        def update_ppg() -> None:
            if self.ppg_tailer is None or ax_ppg is None:
                return
            if not self.ppg_tailer.poll():
                return
            latest_s = 0.0
            for name, trace in self.ppg_tailer.traces.items():
                if not trace.x:
                    continue
                xs = [self.aligned_s(timestamp_ms) for timestamp_ms in trace.x]
                latest_s = max(latest_s, xs[-1])
                if name not in ppg_artists:
                    (artist,) = ax_ppg.plot(xs, trace.y, label=name, linewidth=1)
                    ppg_artists[name] = artist
                    ax_ppg.legend(loc="upper right", fontsize="small")
                else:
                    ppg_artists[name].set_data(xs, trace.y)
            if self.ppg_tailer.traces:
                xs = [
                    self.aligned_s(timestamp_ms)
                    for trace in self.ppg_tailer.traces.values()
                    for timestamp_ms in trace.x
                ]
                ys = [value for trace in self.ppg_tailer.traces.values() for value in trace.y]
                if xs and ys:
                    y_pad = max(1.0, (max(ys) - min(ys)) * 0.10)
                    ax_ppg.set_ylim(min(ys) - y_pad, max(ys) + y_pad)
                if latest_s > 0:
                    cap_latest = self.cap_trace_times[-1] if self.cap_trace_times else 0.0
                    sync_time_axes(max(latest_s, cap_latest))
                ax_ppg.figure.canvas.draw_idle()

        def redraw_patch_overlay() -> None:
            for artist in patch_overlay:
                try:
                    artist.remove()
                except ValueError:
                    pass
            patch_overlay.clear()
            if not self.finger_patches:
                return
            for patch in self.finger_patches:
                width = patch.x_max - patch.x_min
                height = patch.y_max - patch.y_min
                rect = Rectangle(
                    (patch.x_min - 0.5, patch.y_min - 0.5),
                    width,
                    height,
                    linewidth=2,
                    edgecolor="lime",
                    facecolor="none",
                    linestyle="--",
                    alpha=0.85,
                    zorder=11,
                )
                ax_cap.add_patch(rect)
                patch_overlay.append(rect)
                cy = (patch.y_min + patch.y_max) // 2
                cx = (patch.x_min + patch.x_max) // 2
                label = ax_cap.text(
                    cx,
                    cy,
                    patch.name,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="yellow",
                    fontweight="bold",
                    zorder=12,
                    clip_on=True,
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="black",
                        alpha=0.55,
                        edgecolor="none",
                    ),
                )
                patch_overlay.append(label)

        def maybe_detect_fingers(display_grid: np.ndarray) -> None:
            if self.finger_detect_attempted:
                return
            if self.aligned_elapsed() < self.finger_detect_delay_s:
                return
            self.finger_detect_attempted = True
            self.finger_patches = detect_finger_patches(
                display_grid, self.finger_patch_size
            )
            if self.finger_patches:
                self.finger_detected_at_s = self.aligned_elapsed()
                names = ", ".join(patch.name for patch in self.finger_patches)
                if self.enable_pixel_stream:
                    cap_trace_status.set_text(
                        f"Detected {len(self.finger_patches)} finger patch(es): {names}  |  "
                        f"per-pixel windows at t="
                        f"{self.finger_detected_at_s + FINGER_PIXEL_STREAM_DELAY_AFTER_DETECT_S:.0f}s"
                    )
                else:
                    cap_trace_status.set_text(
                        f"Detected {len(self.finger_patches)} finger patch(es): {names}  |  "
                        f"CPG strategy windows (last {STRATEGY_STREAM_DISPLAY_WINDOW_S:.0f}s)"
                    )
                print(
                    f"t={self.aligned_elapsed():.1f}s: auto-detected "
                    f"{len(self.finger_patches)} finger region(s) ({names})"
                )
                for patch in self.finger_patches:
                    self.cap_trace_values.setdefault(patch.name, [])
                redraw_patch_overlay()
            else:
                cap_trace_status.set_text(
                    f"No finger regions detected at t={self.aligned_elapsed():.1f}s"
                )
                print(
                    f"t={self.aligned_elapsed():.1f}s: no finger regions detected"
                )

        def update_cap_trace(display_grid: np.ndarray) -> None:
            maybe_detect_fingers(display_grid)
            if not self.finger_patches:
                return
            if self.cap_tailer.frame_count == self._last_traced_frame:
                return
            self._last_traced_frame = self.cap_tailer.frame_count

            elapsed = self.aligned_elapsed()
            self.cap_trace_times.append(elapsed)
            for patch in self.finger_patches:
                values = self.cap_trace_values.setdefault(patch.name, [])
                values.append(patch.mean(display_grid))

            start = max(0.0, elapsed - CAP_TRACE_WINDOW_S)
            start_idx = int(np.searchsorted(self.cap_trace_times, start))
            times = self.cap_trace_times[start_idx:]
            if not times:
                return

            for index, patch in enumerate(self.finger_patches):
                color = f"C{index % 10}"
                ys = self.cap_trace_values[patch.name][start_idx:]
                if patch.name not in cap_trace_artists:
                    (line,) = ax_cap_trace.plot([], [], color=color, linewidth=1, label=patch.name)
                    cap_trace_artists[patch.name] = line
                else:
                    cap_trace_artists[patch.name].set_data(times, ys)
            ax_cap_trace.legend(loc="upper right", fontsize="small")
            ppg_latest = 0.0
            if self.ppg_tailer is not None:
                green = self.ppg_tailer.traces.get("green")
                if green and green.x:
                    ppg_latest = self.aligned_s(green.x[-1])
            sync_time_axes(max(elapsed, ppg_latest))
            ax_cap_trace.relim()
            ax_cap_trace.autoscale_view(scalex=False, scaley=True)
            update_cap_trace_title()

        def maybe_open_pixel_stream_windows(display_grid: np.ndarray) -> None:
            if not self.enable_pixel_stream:
                return
            if self.pixel_stream_opened or not self.finger_patches:
                return
            if self.finger_detected_at_s is None:
                return
            open_at = self.finger_detected_at_s + FINGER_PIXEL_STREAM_DELAY_AFTER_DETECT_S
            if self.aligned_elapsed() < open_at:
                return
            self.pixel_stream_opened = True
            self.pixel_stream_windows = FingerPixelStreamWindows(
                self.run_dir,
                color_max=self.color_max,
                ppg_tailer=self.ppg_tailer,
                session_t0_s=self.session_t0_s,
            )
            self.pixel_stream_windows.open(self.finger_patches)
            names = ", ".join(patch.name for patch in self.finger_patches)
            print(
                f"t={self.aligned_elapsed():.1f}s: opened per-pixel stream window(s) "
                f"for {len(self.finger_patches)} finger(s) ({names})"
            )
            cap_trace_status.set_text(
                f"Streaming every pixel + CPG strategies for: {names}"
            )

        def maybe_open_strategy_stream_windows(display_grid: np.ndarray) -> None:
            if self.strategy_stream_opened or not self.finger_patches:
                return
            if self.finger_detected_at_s is None:
                return
            open_at = self.finger_detected_at_s + FINGER_PIXEL_STREAM_DELAY_AFTER_DETECT_S
            if self.aligned_elapsed() < open_at:
                return
            self.strategy_stream_opened = True
            self.strategy_stream_windows = FingerStrategyStreamWindows(
                self.run_dir,
                ppg_tailer=self.ppg_tailer,
                session_t0_s=self.session_t0_s,
                include_ppg_corr=self.include_ppg_corr,
            )
            self.strategy_stream_windows.open(self.finger_patches)
            names = ", ".join(patch.name for patch in self.finger_patches)
            print(
                f"t={self.aligned_elapsed():.1f}s: opened CPG strategy window(s) "
                f"for {len(self.finger_patches)} finger(s) ({names})"
            )

        def update_strategy_stream_windows(
            display_grid: np.ndarray,
            *,
            elapsed_s: float,
            append_only: bool = False,
        ) -> None:
            maybe_open_strategy_stream_windows(display_grid)
            if self.strategy_stream_windows is None:
                return
            self.strategy_stream_windows.append_frame(display_grid, elapsed_s)
            if not append_only:
                self.strategy_stream_windows.refresh(elapsed_s)
                self.strategy_stream_windows.prune_closed()

        def update_pixel_stream_windows(
            display_grid: np.ndarray,
            *,
            elapsed_s: float,
            frame_count: int,
        ) -> None:
            if not self.enable_pixel_stream:
                return
            maybe_open_pixel_stream_windows(display_grid)
            if self.pixel_stream_windows is None:
                return
            self.pixel_stream_windows.update(
                display_grid,
                elapsed_s,
                frame_count,
            )
            self.pixel_stream_windows.prune_closed()

        def update_pixel_stream_ppg() -> None:
            if not self.enable_pixel_stream or self.pixel_stream_windows is None:
                return
            self.pixel_stream_windows.update_ppg()

        def update() -> None:
            new_frames = self.cap_tailer.poll_completed_frames()
            latest_aligned_s = self.aligned_elapsed()
            latest_display: np.ndarray | None = None
            if new_frames:
                for grid, frame_count, _elapsed_s, timestamp_ms in new_frames:
                    display_grid = cap_grid_for_display(grid)
                    aligned_s = self.aligned_s(timestamp_ms)
                    latest_aligned_s = aligned_s
                    latest_display = display_grid
                    update_pixel_stream_windows(
                        display_grid,
                        elapsed_s=aligned_s,
                        frame_count=frame_count,
                    )
                    update_strategy_stream_windows(
                        display_grid,
                        elapsed_s=aligned_s,
                        append_only=True,
                    )

                latest_grid, frame_count, _elapsed_s, timestamp_ms = new_frames[-1]
                latest = cap_grid_for_display(latest_grid)
                latest_display = latest
                aligned_s = self.aligned_s(timestamp_ms)
                latest_aligned_s = aligned_s
                im.set_data(latest)
                vmin, vmax = symmetric_limits(latest, self.color_max)
                im.set_norm(mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax))
                cap_title.set_text(
                    f"Live capacitance Δ — {self.run_dir.name}  |  "
                    f"{self.n_cols}×{self.n_rows}  |  "
                    f"frame {frame_count}  |  "
                    f"t={aligned_s:.2f}s  |  "
                    f"Δ [{np.min(latest):.0f}, {np.max(latest):.0f}]"
                )
                update_cap_trace(latest)
                fig.canvas.draw_idle()
            update_ppg()
            update_pixel_stream_ppg()
            if self.strategy_stream_windows is not None:
                if latest_display is not None:
                    maybe_open_strategy_stream_windows(latest_display)
                self.strategy_stream_windows.refresh(latest_aligned_s)
                self.strategy_stream_windows.prune_closed()

        fig.show()
        try:
            while plt.fignum_exists(fig.number):
                if self.stop_event.is_set():
                    plt.close(fig)
                    break
                update()
                plt.pause(self.refresh_ms / 1000.0)
        finally:
            if not self.stop_event.is_set():
                self.stop_event.set()
            self.label_panel.save()


def build_settings(argv: list[str] | None = None) -> Settings:
    import argparse

    parser = argparse.ArgumentParser(
        description="Record maXTouch page-0 capacitance region and PPG green channel"
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--name", default=RUN_NAME)
    parser.add_argument("--region-origin-x", type=int, default=REGION_ORIGIN_X)
    parser.add_argument("--region-origin-y", type=int, default=REGION_ORIGIN_Y)
    parser.add_argument(
        "--page0-width",
        type=int,
        default=PAGE0_WIDTH,
        help="mutual-sense X columns from origin (default: 4)",
    )
    parser.add_argument(
        "--page0-row-fraction",
        type=float,
        default=PAGE0_ROW_FRACTION,
        help="fraction of full Y row length for region height (default: 1/3)",
    )
    parser.add_argument(
        "--region",
        default="",
        help="override auto page-0 region as start_x,start_y,width,height",
    )
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--view-color-max", type=float, default=VIEW_COLOR_MAX)
    parser.add_argument("--view-refresh-ms", type=int, default=VIEW_REFRESH_MS)
    parser.add_argument("--ppg-port", default=PPG_PORT)
    parser.add_argument("--ppg-baud", type=int, default=PPG_BAUD)
    parser.add_argument("--no-ppg", action="store_true")
    parser.add_argument(
        "--ppg-pick-port",
        action="store_true",
        default=PPG_PICK_PORT,
        help="show a serial-port dropdown before recording (session timer starts after selection)",
    )
    parser.add_argument("--ppg-no-auto-kill", action="store_true")
    parser.add_argument(
        "--finger-detect-delay",
        type=float,
        default=FINGER_DETECT_DELAY_S,
        help="seconds after recording start to auto-detect finger 6×6 patches once (default: 10)",
    )
    parser.add_argument(
        "--finger-patch-size",
        type=int,
        default=FINGER_PATCH_SIZE,
        help="fixed patch size around each detected finger (default: 6)",
    )
    parser.add_argument(
        "--pixel-stream",
        action="store_true",
        default=ENABLE_PIXEL_STREAM,
        help="after finger detection, open per-pixel Δ windows (36 stacked traces per finger; off by default)",
    )
    ns = parser.parse_args(argv)

    if ns.page0_width <= 0:
        parser.error("--page0-width must be positive")
    if not 0 < ns.page0_row_fraction <= 1:
        parser.error("--page0-row-fraction must be in (0, 1]")
    if ns.finger_detect_delay < 0:
        parser.error("--finger-detect-delay must be nonnegative")
    if ns.finger_patch_size <= 0:
        parser.error("--finger-patch-size must be positive")

    return Settings(
        name=ns.name,
        out_dir=ns.out_dir,
        region_origin_x=ns.region_origin_x,
        region_origin_y=ns.region_origin_y,
        page0_width=ns.page0_width,
        page0_row_fraction=ns.page0_row_fraction,
        fixed_region=ns.region,
        run_viewer=not ns.no_viewer,
        view_invert_delta=VIEW_INVERT_DELTA,
        view_color_max=None if ns.view_color_max == 0 else ns.view_color_max,
        view_refresh_ms=ns.view_refresh_ms,
        ppg_enabled=not ns.no_ppg,
        ppg_port=ns.ppg_port,
        ppg_baud=ns.ppg_baud,
        ppg_auto_kill_port_owner=not ns.ppg_no_auto_kill,
        ppg_pick_port=ns.ppg_pick_port,
        finger_detect_delay_s=ns.finger_detect_delay,
        finger_patch_size=ns.finger_patch_size,
        enable_pixel_stream=ns.pixel_stream,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_settings(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    mxt = MaxTouchHid()
    try:
        mxt.load_info()
        region, page_size, pages, estimated_hz = resolve_region(mxt, args)
        print(
            f"Region: ({region.start_x},{region.start_y}) "
            f"{region.width}×{region.height} columns×rows  |  "
            f"T37 pages {pages}  |  ~{estimated_hz:.1f} Hz"
        )

        resolved_ppg_port = args.ppg_port
        if args.ppg_enabled and args.ppg_pick_port:
            resolved_ppg_port = pick_ppg_serial_port(args.ppg_port)

        run_dir = make_run_dir(args)
        write_metadata(
            run_dir,
            args,
            mxt,
            region,
            pages,
            page_size,
            estimated_hz,
            ppg_port_override=resolved_ppg_port or None,
        )
        print(f"Recording to {run_dir}")

        stop_event = threading.Event()
        ppg_thread: threading.Thread | None = None
        if args.ppg_enabled:
            print(f"PPG serial port: {find_ppg_port(resolved_ppg_port)}")
            ppg_thread = threading.Thread(
                target=record_ppg_green,
                args=(run_dir, args, stop_event),
                kwargs={"ppg_port": resolved_ppg_port},
            )
            ppg_thread.start()

        label_panel = ActivityLabelPanel(run_dir)
        existing_labels = load_activity_labels(labels_path_for_session(run_dir))
        if existing_labels:
            label_panel.labels = list(existing_labels)

        if args.run_viewer:
            result: dict[str, int] = {}

            def worker() -> None:
                result["status"] = record_capacitance(
                    mxt, run_dir, region, args, pages, estimated_hz, stop_event
                )

            thread = threading.Thread(target=worker, daemon=True)
            thread.start()
            cap_tailer = CapHeatmapTailer(
                run_dir / CAP_CSV_NAME,
                region.height,
                region.width,
                args.view_invert_delta,
            )
            ppg_tailer = (
                PpgLogTailer(run_dir / PPG_CSV_NAME) if args.ppg_enabled else None
            )
            viewer = LiveCsvViewer(
                run_dir=run_dir,
                n_rows=region.height,
                n_cols=region.width,
                refresh_ms=args.view_refresh_ms,
                cap_tailer=cap_tailer,
                ppg_tailer=ppg_tailer,
                stop_event=stop_event,
                label_panel=label_panel,
                color_max=args.view_color_max,
                finger_detect_delay_s=args.finger_detect_delay_s,
                finger_patch_size=args.finger_patch_size,
                enable_pixel_stream=args.enable_pixel_stream,
                estimated_cap_hz=estimated_hz,
            )
            print("Recording… close the viewer window to stop.")
            viewer.show()
            thread.join()
            if ppg_thread is not None:
                ppg_thread.join()
            finalize_session_metadata(run_dir, label_panel.labels)
            print(f"Wrote {run_dir}")
            return result.get("status", 1)

        print("Recording… press Ctrl+C to stop.")
        try:
            status = record_capacitance(
                mxt, run_dir, region, args, pages, estimated_hz, stop_event
            )
        except KeyboardInterrupt:
            stop_event.set()
            status = 0
        if ppg_thread is not None:
            ppg_thread.join()
        label_panel.save()
        finalize_session_metadata(run_dir, label_panel.labels)
        print(f"Wrote {run_dir}")
        return status
    finally:
        mxt.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
