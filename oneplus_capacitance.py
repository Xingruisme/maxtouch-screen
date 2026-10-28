#!/usr/bin/env python3
"""Read OnePlus Pad (OPD2203) Novatek NT36532 touch-controller matrices via adb.

Root is required. The pad exposes mutual-capacitance maps under
``/proc/touchpanel/debug_info/``:

  * ``baseline`` — ``basline_data`` (kernel typo) and ``raw_data`` sections
  * ``delta`` — controller ``diff_data`` (current − factory baseline)

Values are controller units, not calibrated farads.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.colors import ListedColormap

REPO_ROOT = Path(__file__).resolve().parent
DATA_ONEPLUS = REPO_ROOT / "data_oneplus"
CAP_LOG_DIR = DATA_ONEPLUS / "cap_log"

ROWS = 38
COLS = 54
# Shapes we know how to recover from a wide capacitance.csv header when the
# caller did not pass rows/cols (Pad 38×54; 8 Pro 17×38; Pixel 6 16×34; Pixel 4 15×32).
KNOWN_MATRIX_SHAPES = ((38, 54), (17, 38), (16, 34), (15, 32), (50, 32))
CAP_DELTA_COLOR_MIN = -800.0
CAP_DELTA_COLOR_MAX = 800.0

MATRIX_SOURCES = {
    "raw": ("/proc/touchpanel/debug_info/baseline", "raw_data"),
    "baseline": ("/proc/touchpanel/debug_info/baseline", "basline_data"),
    "delta": ("/proc/touchpanel/debug_info/delta", "diff_data"),
}
MATRIX_KINDS = tuple(MATRIX_SOURCES)
DEFAULT_MATRIX_SOURCE = "delta"  # ~8 Hz; "raw" (raw_data) is ~0.8 Hz via baseline proc

ROW_RE = re.compile(r"^\[\s*\d+\]\s*(.*)")
NUMBER_RE = re.compile(r"-?\d+")


class Adb:
    def __init__(self, executable: str, serial: str | None = None) -> None:
        self.base = [executable]
        if serial:
            self.base += ["-s", serial]

    def run(self, *args: str, check: bool = True, timeout: float | None = 20) -> str:
        result = subprocess.run(
            [*self.base, *args],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        if check and result.returncode:
            raise RuntimeError(result.stdout.strip() or f"adb exited {result.returncode}")
        return result.stdout

    def shell(self, command: str, root: bool = False, **kwargs: object) -> str:
        if root:
            command = f"su -c {shell_quote(command)}"
        return self.run("shell", command, **kwargs)


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


def device_epoch_ms(adb: Adb) -> int:
    """Current wall-clock time on the device, in ms since the Unix epoch.

    This is the same clock the capacitance stream stamps each frame with
    (``date +%s%3N`` in ``MatrixStream``) and the same clock the camera app
    stamps each frame with (``System.currentTimeMillis()``), so subtracting
    one shared anchor from both device timestamps puts capacitance and
    camera samples on one timeline that is immune to adb/USB latency and
    host scheduling jitter -- unlike timestamping on arrival at the host.
    """
    output = adb.shell("date +%s%3N")
    match = NUMBER_RE.search(output)
    if match is None:
        raise RuntimeError(f"could not read device clock from {output!r}")
    return int(match.group())


def find_adb(requested: str | None) -> str:
    candidates = [
        requested,
        shutil.which("adb"),
        str(REPO_ROOT / "intermediate-temp/platform-tools/adb"),
        str(REPO_ROOT / "adb"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    raise FileNotFoundError("adb not found; pass --adb /path/to/adb")


ONEPLUS_PAD_MODELS = frozenset({"OPD2203"})


def list_adb_devices(executable: str) -> list[dict[str, str]]:
    """Return connected adb devices with parsed ``adb devices -l`` fields."""
    result = subprocess.run(
        [executable, "devices", "-l"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=10,
        check=False,
    )
    devices: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("List of devices"):
            continue
        parts = stripped.split()
        if len(parts) < 2 or parts[1] != "device":
            continue
        info = {"serial": parts[0]}
        for token in parts[2:]:
            if ":" in token:
                key, value = token.split(":", 1)
                info[key] = value
        devices.append(info)
    return devices


def is_oneplus_pad_device(info: dict[str, str]) -> bool:
    model = info.get("model", "")
    if model in ONEPLUS_PAD_MODELS:
        return True
    haystack = " ".join(
        info.get(key, "") for key in ("model", "product", "device")
    ).upper()
    return "OPD2203" in haystack or "OP59BCL1" in haystack


def resolve_adb_serial(executable: str, requested: str | None = None) -> str | None:
    """Pick an adb serial when none was given explicitly.

    With multiple devices connected, prefer the OnePlus Pad (OPD2203).
    """
    if requested:
        return requested

    devices = list_adb_devices(executable)
    if not devices:
        return None
    if len(devices) == 1:
        return devices[0]["serial"]

    oneplus = [device for device in devices if is_oneplus_pad_device(device)]
    if len(oneplus) == 1:
        chosen = oneplus[0]
        print(
            "Multiple adb devices; auto-selected OnePlus Pad "
            f"{chosen['serial']} (model={chosen.get('model', '?')})"
        )
        return chosen["serial"]
    if len(oneplus) > 1:
        chosen = oneplus[0]
        print(
            "Multiple OnePlus devices; using "
            f"{chosen['serial']} (model={chosen.get('model', '?')})"
        )
        return chosen["serial"]

    serials = [device["serial"] for device in devices]
    raise RuntimeError(
        "Multiple adb devices connected and none look like a OnePlus Pad "
        f"(OPD2203). Set SERIAL in the script or pass --serial. "
        f"Connected: {', '.join(serials)}"
    )


def parse_section_matrix(text: str, section: str) -> list[list[int]]:
    """Extract one labeled section from a touchpanel debug proc dump."""
    in_section = False
    rows: list[list[int]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.endswith(":"):
            if in_section and rows:
                break
            in_section = stripped[:-1] == section
            if in_section:
                rows = []
            continue
        if not in_section:
            continue
        match = ROW_RE.match(stripped)
        if not match:
            continue
        values = [int(value) for value in NUMBER_RE.findall(match.group(1))]
        if values:
            rows.append(values)
    if not rows:
        raise ValueError(f"No matrix rows found for section {section!r}")
    return rows


def parse_section_from_lines(lines: list[str], section: str) -> list[list[int]]:
    return parse_section_matrix("\n".join(lines), section)


def read_controller(adb: Adb, kind: str) -> tuple[list[list[int]], list[int]]:
    path, section = MATRIX_SOURCES[kind]
    text = adb.shell(f"cat {path}", root=True)
    rows = parse_section_matrix(text, section)
    _validate_shape(rows, kind)
    return rows, []


def _validate_shape(rows: list[list[int]], kind: str) -> None:
    if len(rows) != ROWS:
        raise ValueError(f"{kind} returned {len(rows)} rows; expected {ROWS}")
    widths = {len(row) for row in rows}
    if widths != {COLS}:
        raise ValueError(f"{kind} returned column widths {sorted(widths)}; expected {COLS}")


# ---------------------------------------------------------------------------
# capacitance.csv formats
# ---------------------------------------------------------------------------
#
# Two on-disk layouts are supported. Both store the identical information; the
# wide layout is ~16x smaller because it drops the redundant ``row``/``col``
# enumeration and the per-pixel repetition of ``timestamp_ms``/``elapsed_s``/
# ``frame``.
#
#   * long (legacy): one row per pixel per frame
#       timestamp_ms, elapsed_s, frame, row, col, raw, delta
#
#   * wide (current): one row per frame
#       timestamp_ms, elapsed_s, frame, raw0..rawN-1, delta0..deltaN-1
#     where N = ROWS*COLS and the flattened pixel index k maps to
#     (row, col) = (k // COLS, k % COLS)  (C/row-major order).
#
# ``load_capacitance_session`` auto-detects the layout, so code that reads
# sessions works with either. New recordings are written wide.
#
# Thermal-log sessions under ``data_oneplus/cap_thermal_log`` are rewritten
# after recording so consecutive bit-identical frames are dropped. Each kept
# row is the first occurrence of that matrix; it stays current until the next
# unique row (or ``original_last_elapsed_s`` in capacitance_summary.json).
# Use ``held_time_mask`` for time-window queries on those files.

CAP_CSV_NAME = "capacitance.csv"
CAP_CSV_FIELDS_LONG = ("timestamp_ms", "elapsed_s", "frame", "row", "col", "raw", "delta")
CAP_WIDE_META_FIELDS = ("timestamp_ms", "elapsed_s", "frame")


def cap_wide_fields(rows: int = ROWS, cols: int = COLS) -> tuple[str, ...]:
    """Header for the wide capacitance layout (meta + flattened raw + delta)."""
    n = rows * cols
    return (
        *CAP_WIDE_META_FIELDS,
        *(f"raw{k}" for k in range(n)),
        *(f"delta{k}" for k in range(n)),
    )


def cap_csv_is_wide(path: Path) -> bool:
    """True if ``path`` uses the wide layout (one row per frame)."""
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        header = handle.readline()
    fields = [f.strip() for f in header.strip().split(",")]
    # The long layout is the only one that carries explicit row/col columns.
    return not ("row" in fields and "col" in fields)


class CapWideWriter:
    """Stream frames into a wide capacitance.csv (one row per frame).

    ``raw_fmt``/``delta_fmt``/``elapsed_fmt`` control number formatting so each
    caller can preserve the precision it recorded with.
    """

    def __init__(
        self,
        handle,
        *,
        rows: int = ROWS,
        cols: int = COLS,
        raw_fmt: str = "{:.6g}",
        delta_fmt: str = "{:.6g}",
        elapsed_fmt: str = "{:.6f}",
    ) -> None:
        self.rows = rows
        self.cols = cols
        self._raw_fmt = raw_fmt
        self._delta_fmt = delta_fmt
        self._elapsed_fmt = elapsed_fmt
        self._writer = csv.writer(handle)
        self._writer.writerow(cap_wide_fields(rows, cols))

    def write_frame(
        self,
        timestamp_ms: int,
        elapsed_s: float,
        frame: int,
        raw: np.ndarray,
        delta: np.ndarray,
    ) -> None:
        raw_flat = np.asarray(raw, dtype=float).reshape(-1)
        delta_flat = np.asarray(delta, dtype=float).reshape(-1)
        row = [
            int(timestamp_ms),
            self._elapsed_fmt.format(float(elapsed_s)),
            int(frame),
        ]
        row.extend(self._raw_fmt.format(v) for v in raw_flat)
        row.extend(self._delta_fmt.format(v) for v in delta_flat)
        self._writer.writerow(row)


@dataclass
class CapSession:
    """A loaded capacitance recording.

    Attributes:
        timestamp_ms: (T,) wall-clock ms per frame.
        elapsed_s:    (T,) seconds since recording start.
        frame_ids:    (T,) the ``frame`` column values (as recorded).
        raw:          (T, ROWS, COLS) raw controller matrix, float32.
        delta:        (T, ROWS, COLS) delta (raw - baseline), float32.
        hold_end_s:   last original sample time when unique-only files drop
                      trailing duplicates; defaults to ``elapsed_s[-1]``.
    """

    timestamp_ms: np.ndarray
    elapsed_s: np.ndarray
    frame_ids: np.ndarray
    raw: np.ndarray
    delta: np.ndarray
    hold_end_s: float | None = None

    @property
    def n_frames(self) -> int:
        return int(self.raw.shape[0])

    @property
    def last_hold_s(self) -> float:
        if self.hold_end_s is not None:
            return float(self.hold_end_s)
        if self.elapsed_s.size:
            return float(self.elapsed_s[-1])
        return 0.0


def _read_long_arrays(
    path: Path, rows: int, cols: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read a legacy long-format capacitance.csv into dense per-frame arrays."""
    n = rows * cols
    ts_by_frame: dict[int, int] = {}
    el_by_frame: dict[int, float] = {}
    raw_by_frame: dict[int, np.ndarray] = {}
    delta_by_frame: dict[int, np.ndarray] = {}
    filled: dict[int, int] = {}

    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"{path} is empty")
        idx = {name: i for i, name in enumerate(header)}
        required = {"timestamp_ms", "elapsed_s", "frame", "row", "col", "raw", "delta"}
        if not required.issubset(idx):
            raise ValueError(f"{path} missing required columns: {sorted(required)}")
        ti, ei, fi = idx["timestamp_ms"], idx["elapsed_s"], idx["frame"]
        ri, ci, rawi, di = idx["row"], idx["col"], idx["raw"], idx["delta"]
        for rec in reader:
            if not rec:
                continue
            frame = int(rec[fi])
            r = int(rec[ri])
            c = int(rec[ci])
            if not (0 <= r < rows and 0 <= c < cols):
                continue
            if frame not in raw_by_frame:
                raw_by_frame[frame] = np.zeros((rows, cols), dtype=np.float32)
                delta_by_frame[frame] = np.zeros((rows, cols), dtype=np.float32)
                filled[frame] = 0
            ts_by_frame[frame] = int(rec[ti])
            el_by_frame[frame] = float(rec[ei])
            raw_by_frame[frame][r, c] = float(rec[rawi])
            delta_by_frame[frame][r, c] = float(rec[di])
            filled[frame] += 1

    frame_ids = sorted(raw_by_frame)
    # Drop any incomplete (partially recorded) frames.
    frame_ids = [f for f in frame_ids if filled[f] == n]
    if not frame_ids:
        raise ValueError(f"{path} contains no complete frames")
    raw = np.stack([raw_by_frame[f] for f in frame_ids], axis=0)
    delta = np.stack([delta_by_frame[f] for f in frame_ids], axis=0)
    ts = np.asarray([ts_by_frame[f] for f in frame_ids], dtype=np.int64)
    el = np.asarray([el_by_frame[f] for f in frame_ids], dtype=np.float64)
    fids = np.asarray(frame_ids, dtype=np.int64)
    return ts, el, fids, raw, delta


def _read_wide_arrays(
    path: Path, rows: int, cols: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read a wide-format capacitance.csv into dense per-frame arrays."""
    n = rows * cols
    expected_cols = 3 + 2 * n
    ts_list: list[int] = []
    el_list: list[float] = []
    fid_list: list[int] = []
    raw_rows: list[np.ndarray] = []
    delta_rows: list[np.ndarray] = []

    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"{path} is empty")
        if len(header) != expected_cols:
            raise ValueError(
                f"{path} has {len(header)} columns; expected {expected_cols} "
                f"(3 meta + 2*{rows}x{cols})"
            )
        for rec in reader:
            if not rec:
                continue
            if len(rec) != expected_cols:
                continue  # skip a truncated trailing row
            ts_list.append(int(rec[0]))
            el_list.append(float(rec[1]))
            fid_list.append(int(rec[2]))
            values = np.array(rec[3:], dtype=np.float32)
            raw_rows.append(values[:n])
            delta_rows.append(values[n:])

    if not raw_rows:
        raise ValueError(f"{path} contains no frames")
    raw = np.stack(raw_rows, axis=0).reshape(-1, rows, cols)
    delta = np.stack(delta_rows, axis=0).reshape(-1, rows, cols)
    ts = np.asarray(ts_list, dtype=np.int64)
    el = np.asarray(el_list, dtype=np.float64)
    fids = np.asarray(fid_list, dtype=np.int64)
    return ts, el, fids, raw, delta


_SESSION_CACHE: dict[tuple, CapSession] = {}
_SESSION_CACHE_LIMIT = 4


def infer_cap_csv_shape(
    path: Path, *, rows: int = ROWS, cols: int = COLS
) -> tuple[int, int]:
    """Recover (rows, cols) from metadata.json or a wide capacitance.csv header."""
    path = Path(path)
    meta_path = path.parent / "metadata.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            meta = {}
        for key in ("pixel_tablet", "pixel4", "pixel6", "op8pro", "oneplus", "device"):
            block = meta.get(key)
            if not isinstance(block, dict):
                continue
            shape = block.get("matrix_shape")
            if isinstance(shape, str) and "x" in shape.lower():
                row_text, col_text = shape.lower().split("x", 1)
                try:
                    return int(row_text), int(col_text)
                except ValueError:
                    pass
    if path.is_file() and cap_csv_is_wide(path):
        with path.open("r", newline="", encoding="utf-8") as handle:
            header = [field.strip() for field in handle.readline().strip().split(",")]
        if len(header) >= 5 and (len(header) - 3) % 2 == 0:
            n_pixels = (len(header) - 3) // 2
            if rows * cols == n_pixels:
                return rows, cols
            for known_rows, known_cols in KNOWN_MATRIX_SHAPES:
                if known_rows * known_cols == n_pixels:
                    return known_rows, known_cols
    return rows, cols


def load_capacitance_session(
    path: Path, *, rows: int | None = None, cols: int | None = None
) -> CapSession:
    """Load a capacitance.csv (wide or legacy long layout) into a CapSession.

    The layout is auto-detected from the header, so callers do not need to know
    which format a given session was recorded in. Parsed sessions are cached by
    (path, mtime, size) so per-region loops re-read the file only when it changes.
    If ``rows``/``cols`` are omitted, the grid is inferred from metadata.json or
    the wide header (Pad 38×54, OnePlus 8 Pro 17×38, Pixel 6 16×34, Pixel 4 15×32,
    Pixel Tablet 50×32).
    """
    path = Path(path)
    if rows is None or cols is None:
        inferred_rows, inferred_cols = infer_cap_csv_shape(path)
        rows = inferred_rows if rows is None else rows
        cols = inferred_cols if cols is None else cols
    stat = path.stat()
    key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size, rows, cols)
    cached = _SESSION_CACHE.get(key)
    if cached is not None:
        return cached
    if cap_csv_is_wide(path):
        ts, el, fids, raw, delta = _read_wide_arrays(path, rows, cols)
    else:
        ts, el, fids, raw, delta = _read_long_arrays(path, rows, cols)
    session = CapSession(
        timestamp_ms=ts,
        elapsed_s=el,
        frame_ids=fids,
        raw=raw,
        delta=delta,
        hold_end_s=_hold_end_from_summary(path, el),
    )
    if len(_SESSION_CACHE) >= _SESSION_CACHE_LIMIT:
        _SESSION_CACHE.pop(next(iter(_SESSION_CACHE)))
    _SESSION_CACHE[key] = session
    return session


def _hold_end_from_summary(csv_path: Path, _elapsed_s: np.ndarray) -> float | None:
    summary_path = Path(csv_path).parent / "capacitance_summary.json"
    if not summary_path.is_file():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = summary.get("original_last_elapsed_s")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def capacitance_hold_end_s(session_dir: Path, elapsed_s: np.ndarray) -> float:
    """Last time the final unique frame was still current.

    Unique-only thermal CSVs drop trailing duplicates, so the last row's
    ``elapsed_s`` is when that matrix first appeared. The original last
    sample time is stored on ``capacitance_summary.json``.
    """
    hold_end = _hold_end_from_summary(Path(session_dir) / CAP_CSV_NAME, elapsed_s)
    if hold_end is not None:
        return hold_end
    if len(elapsed_s):
        return float(elapsed_s[-1])
    return 0.0


def held_time_mask(
    elapsed_s: np.ndarray,
    start_s: float,
    end_s: float,
    *,
    hold_end_s: float | None = None,
) -> np.ndarray:
    """True for unique frames whose hold interval overlaps ``[start_s, end_s]``.

    Frame ``i`` is treated as current from ``elapsed_s[i]`` until
    ``elapsed_s[i + 1]``. The last frame stays current until ``hold_end_s``
    (or its own timestamp when that is omitted).
    """
    elapsed = np.asarray(elapsed_s, dtype=float)
    if elapsed.size == 0:
        return np.zeros(0, dtype=bool)
    ends = np.empty(len(elapsed), dtype=float)
    if len(elapsed) > 1:
        ends[:-1] = elapsed[1:]
    last_end = float(elapsed[-1]) if hold_end_s is None else float(hold_end_s)
    ends[-1] = max(last_end, float(elapsed[-1]))
    return (elapsed <= end_s) & (ends >= start_s)


def build_rwgb_colormap() -> ListedColormap:
    """White–green–red diverging map used by ``measure_oneplus.py`` for Δ display."""
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


def cap_delta_normlims(values: np.ndarray) -> tuple[float, float]:
    vmin = max(float(np.min(values)), CAP_DELTA_COLOR_MIN)
    vmax = min(float(np.max(values)), CAP_DELTA_COLOR_MAX)
    if vmin >= 0:
        vmin = CAP_DELTA_COLOR_MIN
    if vmax <= 0:
        vmax = CAP_DELTA_COLOR_MAX
    return vmin, vmax
