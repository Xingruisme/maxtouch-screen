#!/usr/bin/env python3
"""Read maXTouch eval kit on macOS via HID Control interface (report ID 1, cmd 0x51)."""

from __future__ import annotations

import argparse
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

if sys.platform.startswith("linux"):
    # The default hid module on Linux uses libusb, which reports usage_page 0 for every interface.
    import hidraw as hid
else:
    import hid

VENDOR_ID = 0x03EB
PRODUCT_ID = 0x214E
REPORT_ID = 1
IIC_CMD = 0x51
DELTAS_MODE = 0x10
PAGE_UP = 0x01
DEBUG_DIAGNOSTIC_T37 = 37
GEN_COMMANDPROCESSOR_T6 = 6
T6_DIAGNOSTIC_OFFSET = 0x05
TOUCH_MULTITOUCHSCREEN_T100 = 100
T100_TCHAUX_OFFSET = 0x03
T100_TCHAUX_PEAK_BIT = 0x10
SPT_MESSAGECOUNT_T44 = 44
GEN_MESSAGEPROCESSOR_T5 = 5
READ_TIMEOUT_S = 0.05
WRITE_TIMEOUT_S = 0.05


@dataclass
class IdInfo:
    family: int
    variant: int
    version: int
    build: int
    matrix_x: int
    matrix_y: int
    num_objects: int


@dataclass
class MxtObject:
    type_id: int
    address: int
    size: int
    instances: int
    num_report_ids: int


@dataclass
class T100PeakSample:
    timestamp: float
    report_id: int
    status: int
    x: int
    y: int
    peak: int


class MaxTouchHid:
    def __init__(self) -> None:
        self._h = self._open_control()
        self.id_info: Optional[IdInfo] = None
        self.objects: List[MxtObject] = []
        self._report_id_to_type: Dict[int, int] = {}

    def _open_control(self) -> hid.device:
        for d in hid.enumerate(VENDOR_ID, PRODUCT_ID):
            if d["interface_number"] == 0 and d["usage_page"] == 0xFF00:
                h = hid.device()
                h.open_path(d["path"])
                h.set_nonblocking(1)
                return h
        raise RuntimeError("Atmel maXTouch Control interface not found (is the kit plugged in?)")

    def close(self) -> None:
        self._h.close()

    def _read_response(self, timeout_s: float) -> Optional[List[int]]:
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            resp = self._h.read(64)
            if resp:
                return resp
        return None

    def read_register(
        self,
        address: int,
        count: int,
        *,
        attempts: int = 4,
    ) -> bytes:
        out = bytearray()
        remaining = count
        addr = address
        while remaining > 0:
            chunk = min(remaining, 58)
            pkt = bytearray(64)
            pkt[0] = REPORT_ID
            pkt[1] = IIC_CMD
            pkt[2] = 2
            pkt[3] = chunk
            struct.pack_into("<H", pkt, 4, addr)
            resp = None
            for attempt in range(attempts):
                if attempt:
                    while self._h.read(64):
                        pass
                    time.sleep(0.002 * attempt)
                self._h.write(bytes(pkt))
                resp = self._read_response(READ_TIMEOUT_S)
                if resp and len(resp) >= 3 and resp[1] == 0 and resp[2] > 0:
                    break
            else:
                raise RuntimeError(
                    f"read failed after {attempts} attempts "
                    f"addr=0x{addr:04X} "
                    f"resp={list(resp[:8]) if resp else None}"
                )
            n = resp[2]
            out.extend(resp[3 : 3 + n])
            remaining -= n
            addr += n
            if n == 0:
                break
        return bytes(out)

    def write_register(self, address: int, data: bytes) -> None:
        if len(data) > 58:
            raise ValueError("write too large for HID packet")
        pkt = bytearray(64)
        pkt[0] = REPORT_ID
        pkt[1] = IIC_CMD
        pkt[2] = 2 + len(data)
        pkt[3] = 0
        struct.pack_into("<H", pkt, 4, address)
        pkt[6 : 6 + len(data)] = data
        self._h.write(bytes(pkt))
        resp = self._read_response(WRITE_TIMEOUT_S)
        if not resp or resp[1] != 0x04:
            raise RuntimeError(f"write failed addr=0x{address:04X} resp={resp[:4] if resp else None}")

    def load_info(self) -> None:
        header = self.read_register(0, 7)
        id_info = IdInfo(
            family=header[0],
            variant=header[1],
            version=header[2],
            build=header[3],
            matrix_x=header[4],
            matrix_y=header[5],
            num_objects=header[6],
        )
        self.id_info = id_info
        table = self.read_register(7, id_info.num_objects * 6)
        self.objects = []
        for i in range(id_info.num_objects):
            chunk = table[i * 6 : (i + 1) * 6]
            self.objects.append(
                MxtObject(
                    type_id=chunk[0],
                    address=chunk[1] | (chunk[2] << 8),
                    size=chunk[3] + 1,
                    instances=chunk[4] + 1,
                    num_report_ids=chunk[5],
                )
            )
        self._build_report_id_map()

    def _build_report_id_map(self) -> None:
        self._report_id_to_type = {}
        report_id = 0
        for obj in self.objects:
            for _instance in range(obj.instances):
                for _ in range(obj.num_report_ids):
                    report_id += 1
                    self._report_id_to_type[report_id] = obj.type_id

    def object_address(self, type_id: int, instance: int = 0) -> Optional[int]:
        count = 0
        for o in self.objects:
            if o.type_id == type_id:
                if count == instance:
                    return o.address
                count += 1
        return None

    def object_size(self, type_id: int, instance: int = 0) -> Optional[int]:
        count = 0
        for o in self.objects:
            if o.type_id == type_id:
                if count == instance:
                    return o.size
                count += 1
        return None

    def read_mutual_deltas_frame(self) -> Tuple[List[int], float]:
        assert self.id_info is not None
        t6 = self.object_address(GEN_COMMANDPROCESSOR_T6, 0)
        t37 = self.object_address(DEBUG_DIAGNOSTIC_T37, 0)
        t37_size = self.object_size(DEBUG_DIAGNOSTIC_T37, 0)
        if t6 is None or t37 is None or t37_size is None:
            raise RuntimeError("T6 or T37 object not found in object table")

        diag_cmd = t6 + T6_DIAGNOSTIC_OFFSET
        page_size = t37_size - 2
        nodes = self.id_info.matrix_x * self.id_info.matrix_y
        pages_per_pass = (nodes * 2 + page_size - 1) // page_size

        values: List[int] = []
        t0 = time.perf_counter()

        for page in range(pages_per_pass):
            if page == 0:
                self.write_register(diag_cmd, bytes([DELTAS_MODE]))
            else:
                self.write_register(diag_cmd, bytes([PAGE_UP]))

            for _ in range(500):
                status = self.read_register(diag_cmd, 1)[0]
                if status == 0:
                    break
                time.sleep(0.0005)
            else:
                raise TimeoutError("T6 diagnostic command timed out")

            page_raw = self.read_register(t37, t37_size)
            mode = page_raw[0]
            page_num = page_raw[1]
            if mode != DELTAS_MODE:
                raise RuntimeError(f"unexpected T37 mode 0x{mode:02X}")
            if page_num != page:
                raise RuntimeError(f"unexpected T37 page {page_num} (expected {page})")

            data = page_raw[2:]
            for i in range(0, len(data) - 1, 2):
                if len(values) >= nodes:
                    break
                values.append(struct.unpack_from("<h", data, i)[0])

        elapsed = time.perf_counter() - t0
        return values[:nodes], elapsed

    def read_mutual_deltas_region(
        self,
        start_x: int,
        start_y: int,
        width: int,
        height: int,
    ) -> Tuple[List[List[int]], float]:
        """Read only T37 byte ranges needed for a fixed rectangular region."""
        assert self.id_info is not None
        t6 = self.object_address(GEN_COMMANDPROCESSOR_T6, 0)
        t37 = self.object_address(DEBUG_DIAGNOSTIC_T37, 0)
        t37_size = self.object_size(DEBUG_DIAGNOSTIC_T37, 0)
        if t6 is None or t37 is None or t37_size is None:
            raise RuntimeError("T6 or T37 object not found in object table")

        matrix_y = self.id_info.matrix_y
        page_size = t37_size - 2
        wanted: Dict[int, List[Tuple[int, int, int]]] = {}
        for row in range(height):
            for col in range(width):
                # T37 diagnostic data is X-major: each X line contains all Y nodes.
                node = (start_x + col) * matrix_y + start_y + row
                byte_offset = node * 2
                page = byte_offset // page_size
                offset = byte_offset % page_size
                wanted.setdefault(page, []).append((offset, row, col))

        result = [[0 for _ in range(width)] for _ in range(height)]
        diag_cmd = t6 + T6_DIAGNOSTIC_OFFSET
        t0 = time.perf_counter()

        for page in range(max(wanted) + 1):
            command = DELTAS_MODE if page == 0 else PAGE_UP
            self.write_register(diag_cmd, bytes([command]))
            for _ in range(500):
                if self.read_register(diag_cmd, 1)[0] == 0:
                    break
                time.sleep(0.0005)
            else:
                raise TimeoutError("T6 diagnostic command timed out")

            if page not in wanted:
                continue

            header = self.read_register(t37, 2)
            if header[0] != DELTAS_MODE or header[1] != page:
                raise RuntimeError(
                    f"unexpected T37 header mode=0x{header[0]:02X} page={header[1]}"
                )

            entries = wanted[page]
            first = min(offset for offset, _, _ in entries)
            last = max(offset for offset, _, _ in entries) + 2
            raw = self.read_register(t37 + 2 + first, last - first)
            for offset, row, col in entries:
                result[row][col] = struct.unpack_from("<h", raw, offset - first)[0]

        return result, time.perf_counter() - t0

    def enable_t100_peak_deltas(self) -> int:
        """Enable peak-delta field in T100 touch reports. Returns previous TCHAUX byte."""
        t100 = self.object_address(TOUCH_MULTITOUCHSCREEN_T100, 0)
        if t100 is None:
            raise RuntimeError("T100 object not found")
        prev = self.read_register(t100 + T100_TCHAUX_OFFSET, 1)[0]
        if not (prev & T100_TCHAUX_PEAK_BIT):
            self.write_register(t100 + T100_TCHAUX_OFFSET, bytes([prev | T100_TCHAUX_PEAK_BIT]))
        return prev

    def restore_t100_tchaux(self, value: int) -> None:
        t100 = self.object_address(TOUCH_MULTITOUCHSCREEN_T100, 0)
        if t100 is not None:
            self.write_register(t100 + T100_TCHAUX_OFFSET, bytes([value]))

    def _parse_t100_peak(self, msg: bytes, timestamp: float) -> Optional[T100PeakSample]:
        if len(msg) < 7:
            return None
        if self._report_id_to_type.get(msg[0]) != TOUCH_MULTITOUCHSCREEN_T100:
            return None
        return T100PeakSample(
            timestamp=timestamp,
            report_id=msg[0],
            status=msg[1],
            x=msg[2] | (msg[3] << 8),
            y=msg[4] | (msg[5] << 8),
            peak=msg[6],
        )

    def stream_t100_peaks(
        self,
        seconds: float,
        *,
        min_peak: int = 1,
        exclude_xy: Optional[Tuple[int, int]] = (1, 43),
    ) -> List[T100PeakSample]:
        """Poll T44/T5 for T100 touch reports with peak-delta byte (CPG-friendly ~80–100 Hz)."""
        t44 = self.object_address(SPT_MESSAGECOUNT_T44, 0)
        t5 = self.object_address(GEN_MESSAGEPROCESSOR_T5, 0)
        t5_size = self.object_size(GEN_MESSAGEPROCESSOR_T5, 0)
        if t44 is None or t5 is None or t5_size is None:
            raise RuntimeError("T44 or T5 object not found")

        samples: List[T100PeakSample] = []
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            if not self.read_register(t44, 1)[0]:
                continue
            msg = self.read_register(t5, t5_size)
            ts = time.perf_counter() - t0
            sample = self._parse_t100_peak(msg, ts)
            if sample is None:
                continue
            if sample.peak < min_peak:
                continue
            if exclude_xy is not None and (sample.x, sample.y) == exclude_xy:
                continue
            samples.append(sample)
        return samples

    def wait_for_t100_peak(
        self,
        timeout_s: float,
        *,
        min_peak: int = 1,
        exclude_xy: Optional[Tuple[int, int]] = (1, 43),
    ) -> Optional[T100PeakSample]:
        """Wait for and immediately return the first qualifying T100 touch report."""
        t44 = self.object_address(SPT_MESSAGECOUNT_T44, 0)
        t5 = self.object_address(GEN_MESSAGEPROCESSOR_T5, 0)
        t5_size = self.object_size(GEN_MESSAGEPROCESSOR_T5, 0)
        if t44 is None or t5 is None or t5_size is None:
            raise RuntimeError("T44 or T5 object not found")

        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout_s:
            if not self.read_register(t44, 1)[0]:
                continue
            msg = self.read_register(t5, t5_size)
            sample = self._parse_t100_peak(msg, time.perf_counter() - t0)
            if sample is None or sample.peak < min_peak:
                continue
            if exclude_xy is not None and (sample.x, sample.y) == exclude_xy:
                continue
            return sample
        return None


def open_digitizer() -> Optional[hid.device]:
    for d in hid.enumerate(VENDOR_ID, PRODUCT_ID):
        if d["usage_page"] == 0x0D and d["usage"] == 0x04:
            h = hid.device()
            h.open_path(d["path"])
            h.set_nonblocking(1)
            return h
    return None


def sample_digitizer(seconds: float) -> Tuple[int, float, List[str]]:
    h = open_digitizer()
    if h is None:
        return 0, 0.0, []
    lines: List[str] = []
    count = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        data = h.read(64)
        if data:
            count += 1
            lines.append(bytes(data).hex())
        time.sleep(0.001)
    elapsed = time.perf_counter() - t0
    h.close()
    return count, elapsed, lines


def matrix_to_rows(values: List[int], x: int, y: int) -> List[List[int]]:
    if len(values) < x * y:
        values = values + [0] * (x * y - len(values))
    return [values[row * x : (row + 1) * x] for row in range(y)]


def crop_around_peak(
    values: List[int],
    matrix_x: int,
    matrix_y: int,
    width: int,
    height: int,
) -> Tuple[List[List[int]], int, int, int, int]:
    """Return a fixed-size matrix crop centered on the largest absolute delta."""
    if width <= 0 or height <= 0:
        raise ValueError("crop dimensions must be positive")
    if width > matrix_x or height > matrix_y:
        raise ValueError(
            f"crop {width}x{height} exceeds matrix size {matrix_x}x{matrix_y}"
        )

    rows = matrix_to_rows(values, matrix_x, matrix_y)
    peak_index = max(range(matrix_x * matrix_y), key=lambda i: abs(values[i]))
    peak_x = peak_index % matrix_x
    peak_y = peak_index // matrix_x

    start_x = min(max(peak_x - width // 2, 0), matrix_x - width)
    start_y = min(max(peak_y - height // 2, 0), matrix_y - height)
    crop = [
        row[start_x : start_x + width]
        for row in rows[start_y : start_y + height]
    ]
    return crop, start_x, start_y, peak_x, peak_y


def crop_region(
    values: List[int],
    matrix_x: int,
    matrix_y: int,
    start_x: int,
    start_y: int,
    width: int,
    height: int,
) -> List[List[int]]:
    rows = matrix_to_rows(values, matrix_x, matrix_y)
    return [
        row[start_x : start_x + width]
        for row in rows[start_y : start_y + height]
    ]


def heat_color(value: int, scale: int) -> str:
    """Map signed deltas to blue (negative), black (zero), or red (positive)."""
    level = min(abs(value) / max(scale, 1), 1.0)
    low = int(20 * (1.0 - level))
    high = int(255 * level)
    if value >= 0:
        return f"#{high:02x}{low:02x}{low:02x}"
    return f"#{low:02x}{low:02x}{high:02x}"


class LiveHeatmap:
    def __init__(self, width: int, height: int, start_x: int, start_y: int) -> None:
        import tkinter as tk

        self._tk = tk
        self._open = True
        self._cell = max(42, min(70, 620 // max(width, height)))
        self._width = width
        self._height = height
        self._root = tk.Tk()
        self._root.title("maXTouch Live Finger Capacitance")
        self._root.protocol("WM_DELETE_WINDOW", self.close)
        self._status = tk.StringVar()
        tk.Label(self._root, textvariable=self._status, font=("TkDefaultFont", 13)).pack(
            padx=8, pady=8
        )
        self._canvas = tk.Canvas(
            self._root,
            width=width * self._cell,
            height=height * self._cell,
            background="black",
            highlightthickness=0,
        )
        self._canvas.pack(padx=10, pady=(0, 10))
        self._rectangles: List[int] = []
        self._labels: List[int] = []
        for row in range(height):
            for col in range(width):
                x0 = col * self._cell
                y0 = row * self._cell
                self._rectangles.append(
                    self._canvas.create_rectangle(
                        x0, y0, x0 + self._cell, y0 + self._cell,
                        fill="black", outline="#404040",
                    )
                )
                self._labels.append(
                    self._canvas.create_text(
                        x0 + self._cell / 2,
                        y0 + self._cell / 2,
                        text="0",
                        fill="white",
                        font=("TkFixedFont", 10),
                    )
                )
        self._origin = (start_x, start_y)
        self._root.update()

    @property
    def is_open(self) -> bool:
        return self._open

    def close(self) -> None:
        self._open = False

    def update(self, values: List[List[int]], frame: int, elapsed: float) -> None:
        flat = [value for row in values for value in row]
        scale = max((abs(value) for value in flat), default=1)
        for index, value in enumerate(flat):
            color = heat_color(value, scale)
            self._canvas.itemconfigure(self._rectangles[index], fill=color)
            self._canvas.itemconfigure(
                self._labels[index],
                text=str(value),
                fill="white" if abs(value) > scale * 0.25 else "#d0d0d0",
            )
        hz = 1.0 / elapsed if elapsed > 0 else 0.0
        self._status.set(
            f"Frame {frame}   {hz:.1f} Hz   origin={self._origin}   "
            f"range={min(flat)}..{max(flat)}"
        )
        try:
            self._root.update_idletasks()
            self._root.update()
        except self._tk.TclError:
            self._open = False

    def destroy(self) -> None:
        try:
            self._root.destroy()
        except self._tk.TclError:
            pass


def peak_rate_hz(samples: List[T100PeakSample]) -> Dict[int, float]:
    by_rid: Dict[int, List[T100PeakSample]] = {}
    for sample in samples:
        by_rid.setdefault(sample.report_id, []).append(sample)

    rates: Dict[int, float] = {}
    for report_id, group in by_rid.items():
        if len(group) < 2:
            rates[report_id] = 0.0
            continue
        duration = group[-1].timestamp - group[0].timestamp
        rates[report_id] = (len(group) - 1) / duration if duration > 0 else 0.0
    return rates


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire maXTouch data on macOS via HID")
    parser.add_argument(
        "--mode",
        choices=("peaks", "matrix", "finger-map", "both"),
        default="peaks",
        help=(
            "peaks: T100 peak delta; matrix: full T37 map; "
            "finger-map: continuously visualize a cropped T37 map"
        ),
    )
    parser.add_argument("--touch-seconds", type=float, default=5.0, help="Digitizer HID sample window")
    parser.add_argument("--stream-seconds", type=float, default=10.0, help="T100 peak stream duration")
    parser.add_argument("--frames", type=int, default=3, help="Mutual-cap delta frames to capture")
    parser.add_argument("--crop-width", type=int, default=4, help="Finger-map width in sensor nodes")
    parser.add_argument("--crop-height", type=int, default=4, help="Finger-map height in sensor nodes")
    parser.add_argument(
        "--touch-threshold",
        type=int,
        default=1,
        help="Minimum T100 peak delta that triggers finger-map capture",
    )
    parser.add_argument(
        "--wait-seconds",
        type=float,
        default=30.0,
        help="Maximum time to wait for a finger in finger-map mode",
    )
    parser.add_argument(
        "--capture-seconds",
        type=float,
        default=30.0,
        help="Duration of fixed-region capacitance capture",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data_maxtouch"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mxt = MaxTouchHid()
    try:
        mxt.load_info()
        assert mxt.id_info is not None
        idn = mxt.id_info
        print("=== Chip ===")
        print(
            f"family=0x{idn.family:02X} variant=0x{idn.variant:02X} "
            f"version={idn.version}.{idn.build:02X} matrix={idn.matrix_x}x{idn.matrix_y} "
            f"nodes={idn.matrix_x * idn.matrix_y}"
        )
        t37 = mxt.object_address(DEBUG_DIAGNOSTIC_T37, 0)
        t37_size = mxt.object_size(DEBUG_DIAGNOSTIC_T37, 0)
        print(f"T37 @ 0x{t37:04X} size={t37_size}" if t37 else "T37 not found")

        if args.mode in ("peaks", "both"):
            prev_tchaux = mxt.enable_t100_peak_deltas()
            print(f"\n=== T100 peak stream ({args.stream_seconds:.0f}s) — rest finger on panel ===")
            peaks = mxt.stream_t100_peaks(args.stream_seconds)
            rates = peak_rate_hz(peaks)
            out = args.out_dir / "t100_peaks.csv"
            with out.open("w") as f:
                f.write("t_s,report_id,status,x,y,peak\n")
                for s in peaks:
                    f.write(f"{s.timestamp:.6f},{s.report_id},{s.status},{s.x},{s.y},{s.peak}\n")
            if peaks:
                vals = [s.peak for s in peaks]
                rate_str = ", ".join(f"slot 0x{rid:02X} ~{hz:.0f} Hz" for rid, hz in sorted(rates.items()) if hz > 0)
                print(
                    f"saved {out.name}: {len(peaks)} samples  "
                    f"peak range {min(vals)}..{max(vals)}"
                )
                if rate_str:
                    print(f"  rates: {rate_str}")
            else:
                print("no peak samples (is a finger on the panel?)")
            mxt.restore_t100_tchaux(prev_tchaux)

        if args.mode == "finger-map":
            if args.crop_width > idn.matrix_x or args.crop_height > idn.matrix_y:
                parser.error(
                    f"crop {args.crop_width}x{args.crop_height} exceeds "
                    f"matrix {idn.matrix_x}x{idn.matrix_y}"
                )

            start_x = 0
            start_y = 0
            print(
                f"\n=== Fixed top-left finger map "
                f"({args.crop_width}x{args.crop_height}) ==="
            )
            print("place and hold finger in the top-left region")
            crop, elapsed = mxt.read_mutual_deltas_region(
                start_x,
                start_y,
                args.crop_width,
                args.crop_height,
            )
            try:
                print(
                    f"locked region origin=({start_x},{start_y}), "
                    "close window or press Ctrl-C to stop"
                )
                heatmap = LiveHeatmap(
                    args.crop_width,
                    args.crop_height,
                    start_x,
                    start_y,
                )
                frame = 0
                capture_started = time.perf_counter()
                last_fps_print = capture_started
                captured_rows: List[str] = []
                try:
                    while (
                        heatmap.is_open
                        and time.perf_counter() - capture_started < args.capture_seconds
                    ):
                        frame += 1
                        if frame > 1:
                            crop, elapsed = mxt.read_mutual_deltas_region(
                                start_x,
                                start_y,
                                args.crop_width,
                                args.crop_height,
                            )
                        timestamp = time.perf_counter() - capture_started
                        captured_rows.append(
                            f"{timestamp:.6f},{frame},"
                            + ",".join(str(value) for row in crop for value in row)
                        )
                        heatmap.update(crop, frame, elapsed)
                        now = time.perf_counter()
                        if now - last_fps_print >= 1.0:
                            instant_fps = 1.0 / elapsed if elapsed > 0 else 0.0
                            average_fps = frame / (now - capture_started)
                            print(
                                f"\rFPS: {instant_fps:.1f} current, "
                                f"{average_fps:.1f} average  frame={frame}",
                                end="",
                                flush=True,
                            )
                            last_fps_print = now
                except KeyboardInterrupt:
                    print("\nstopped by user")
                finally:
                    heatmap.destroy()
                out = args.out_dir / "finger_capacitance_30s.csv"
                with out.open("w") as f:
                    columns = [
                        f"x{x}_y{y}"
                        for y in range(args.crop_height)
                        for x in range(args.crop_width)
                    ]
                    f.write("t_s,frame," + ",".join(columns) + "\n")
                    f.write("\n".join(captured_rows) + "\n")
                duration = time.perf_counter() - capture_started
                average_fps = frame / duration if duration > 0 else 0.0
                print(
                    f"\ncaptured {frame} frame(s) in {duration:.2f}s "
                    f"({average_fps:.1f} FPS); saved {out}"
                )
            except KeyboardInterrupt:
                print("\nstopped by user")

        if args.mode in ("matrix", "both"):
            print(f"\n=== Digitizer HID ({args.touch_seconds:.0f}s) — touch the panel now ===")
            touch_n, touch_dt, touch_lines = sample_digitizer(args.touch_seconds)
            touch_hz = touch_n / touch_dt if touch_dt > 0 else 0.0
            print(f"digitizer reports: {touch_n} in {touch_dt:.2f}s  (~{touch_hz:.1f} Hz)")
            (args.out_dir / "digitizer_sample.txt").write_text("\n".join(touch_lines) + "\n")

            print(f"\n=== Mutual-cap delta map ({args.frames} frame(s)) — keep finger on panel ===")
            frame_times: List[float] = []
            for i in range(args.frames):
                values, elapsed = mxt.read_mutual_deltas_frame()
                frame_times.append(elapsed)
                hz = 1.0 / elapsed if elapsed > 0 else 0.0
                print(f"frame {i + 1}: read {len(values)} nodes in {elapsed:.3f}s (~{hz:.1f} Hz frame rate)")
                rows = matrix_to_rows(values, idn.matrix_x, idn.matrix_y)
                out = args.out_dir / f"mutual_deltas_frame{i + 1}.csv"
                with out.open("w") as f:
                    for row in rows:
                        f.write(",".join(str(v) for v in row) + "\n")
                nonzero = sum(1 for v in values if v != 0)
                print(f"  saved {out.name}  nonzero nodes={nonzero}  min={min(values)} max={max(values)}")

            avg_frame = sum(frame_times) / len(frame_times)
            print(f"\nAverage full-matrix frame time: {avg_frame:.3f}s (~{1.0 / avg_frame:.1f} Hz)")
    finally:
        mxt.close()


if __name__ == "__main__":
    main()
