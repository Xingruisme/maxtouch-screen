#!/usr/bin/env python3
"""Pixel 6 (oriole) STM FingerTipS FTM5 capacitance stream.

Mutual-sense maps are polled through Magisk ``su``:

    echo 13 > /sys/devices/platform/10d40000.spi/spi_master/spi11/spi11.0/stm_fts_cmd
    cat  .../stm_fts_cmd

Returns ``{ err rows cols ... }`` — a **16 × 34** signed grid. Unique rate is
~3.1 Hz (the ``stm_fts_cmd`` dump cost, not USB RTT). Display off typically
stops the scan; keep the screen on.
"""

from __future__ import annotations

import json
import re
import select
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1] / "screen-bloodflow-core-main"  # shared core modules (sibling folder)
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))  # after the device folder, so local copies win

from oneplus_capacitance import (  # noqa: E402
    Adb,
    CapWideWriter,
    list_adb_devices,
)
from oneplus_monitor import adb_device_ready  # noqa: E402

DEVICE_DIR = Path(__file__).resolve().parent
DATA_ROOT = DEVICE_DIR / "data"

ROWS = 16
COLS = 34
FTS_CMD = (
    "/sys/devices/platform/10d40000.spi/spi_master/spi11/spi11.0/stm_fts_cmd"
)
MATRIX_SOURCES = {
    "delta": FTS_CMD,
    "raw": FTS_CMD,
}
DEFAULT_MATRIX_SOURCE = "delta"
CAP_CSV_NAME = "capacitance.csv"
SU_CANDIDATES = ("/debug_ramdisk/su", "su")

PIXEL6_MODELS = frozenset({"Pixel 6", "oriole"})
PIXEL6_TOKENS = ("PIXEL 6", "PIXEL6", "ORIOLE")


def is_pixel6_device(info: dict[str, str]) -> bool:
    model = info.get("model", "")
    if model in PIXEL6_MODELS:
        return True
    haystack = " ".join(
        info.get(key, "") for key in ("model", "product", "device")
    ).upper()
    return any(token in haystack for token in PIXEL6_TOKENS)


def resolve_adb_serial(executable: str, requested: str | None = None) -> str | None:
    """Pick an adb serial when none was given; prefer the Pixel 6."""
    if requested:
        return requested

    devices = list_adb_devices(executable)
    if not devices:
        return None
    if len(devices) == 1:
        return devices[0]["serial"]

    matches = [device for device in devices if is_pixel6_device(device)]
    if len(matches) >= 1:
        chosen = matches[0]
        print(
            "Multiple adb devices; auto-selected Pixel 6 "
            f"{chosen['serial']} (model={chosen.get('model', '?')})"
        )
        return chosen["serial"]

    serials = [device["serial"] for device in devices]
    raise RuntimeError(
        "Multiple adb devices connected and none look like a Pixel 6 "
        f"(oriole). Set SERIAL in the script or pass --serial. "
        f"Connected: {', '.join(serials)}"
    )


def parse_fts_frame(text: str) -> np.ndarray:
    """Parse an ``stm_fts_cmd`` mutual-frame dump into a (16, 34) array."""
    cleaned = text.strip()
    if "Not in resume" in cleaned or "error" in cleaned.lower() and "{" not in cleaned:
        raise RuntimeError(
            f"touch IC not ready (is the screen on?): {cleaned[:120]}"
        )
    if cleaned.startswith("{"):
        cleaned = cleaned[1:]
    if cleaned.endswith("}"):
        cleaned = cleaned[:-1]
    parts = re.sub(r"\s+", " ", cleaned).strip().split()
    if len(parts) < 3:
        raise ValueError(f"short stm_fts_cmd response: {text[:120]}")
    err = int(parts[0], 16) if parts[0].startswith("0") else int(parts[0])
    if err != 0:
        raise ValueError(f"stm_fts_cmd error {err:#x}")
    rows, cols = int(parts[1]), int(parts[2])
    vals = [int(x) for x in parts[3:]]
    if len(vals) != rows * cols:
        raise ValueError(f"expected {rows * cols} cells, got {len(vals)}")
    if rows != ROWS or cols != COLS:
        raise ValueError(f"expected {ROWS}x{COLS}, got {rows}x{cols}")
    return np.asarray(vals, dtype=float).reshape(ROWS, COLS)


def read_matrix(adb: Adb, kind: str) -> np.ndarray:
    if kind not in MATRIX_SOURCES:
        raise ValueError(f"unknown source {kind!r}")
    last_error: Exception | None = None
    command = f"echo 13 > {FTS_CMD} && cat {FTS_CMD}"
    for su in SU_CANDIDATES:
        try:
            text = adb.shell(f"{su} -c '{command}'")
            return parse_fts_frame(text)
        except (RuntimeError, ValueError, subprocess.SubprocessError) as exc:
            last_error = exc
    raise RuntimeError(f"could not read {kind}: {last_error}")


def _su_for_stream(adb: Adb) -> str:
    for su in SU_CANDIDATES:
        try:
            output = adb.shell(f"test -x {su} && echo OK || echo MISSING", check=False)
        except (RuntimeError, subprocess.SubprocessError):
            continue
        if "OK" in output:
            return su
    return SU_CANDIDATES[0]


class MatrixStream:
    """Read 16×34 frames from one persistent Magisk root shell."""

    def __init__(
        self,
        adb: Adb,
        source: str,
        *,
        touch_safe_sleep_ms: int = 0,
    ) -> None:
        if source not in MATRIX_SOURCES:
            raise ValueError(f"unknown source {source!r}")
        self.adb = adb
        self.source = source
        self.touch_safe_sleep_ms = touch_safe_sleep_ms
        sleep_s = max(touch_safe_sleep_ms, 0) / 1000.0
        sleep_clause = f" sleep {sleep_s:.3f};" if sleep_s > 0 else ""
        remote_loop = (
            "while true; do "
            "echo FRAME_START; "
            f"echo 13 > {FTS_CMD} && cat {FTS_CMD}; "
            "echo FRAME_END $(date +%s%3N);"
            f"{sleep_clause} "
            "done"
        )
        su = _su_for_stream(adb)
        command = f"{su} -c '{remote_loop}'"
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
                frame_text = "".join(self._frame_lines)
                self._frame_lines = []
                try:
                    return parse_fts_frame(frame_text)
                except (RuntimeError, ValueError):
                    self.skipped_frames += 1
                    continue
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


def write_baseline_matrix(run_dir: Path, baseline: np.ndarray) -> None:
    np.savetxt(run_dir / "baseline_matrix.csv", baseline, delimiter=",", fmt="%.6g")


def record_capacitance(
    adb: Adb,
    run_dir: Path,
    args,
    stop_event,
    session_t0: float,
    device_epoch_anchor_ms: int | None = None,
) -> int:
    """Wide capacitance.csv logger, same schema as the Pad / 8 Pro monitors."""
    csv_path = run_dir / CAP_CSV_NAME
    period = 1.0 / args.rate if args.rate > 0 else 0.0
    start = session_t0
    frame = 0
    successes = 0
    errors: dict[str, str] = {}
    completed_at = start
    baseline_buf: list[np.ndarray] = []
    baseline: np.ndarray | None = None
    stream: MatrixStream | None = None
    stream_restarts = 0

    def open_stream() -> MatrixStream:
        nonlocal stream
        if stream is not None:
            stream.close()
        stream = MatrixStream(
            adb,
            args.source,
            touch_safe_sleep_ms=args.touch_safe_sleep_ms,
        )
        return stream

    if args.read_mode == "stream":
        open_stream()

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = CapWideWriter(
            handle,
            rows=ROWS,
            cols=COLS,
            raw_fmt="{:.6g}",
            delta_fmt="{:.2f}",
            elapsed_fmt="{:.6f}",
        )
        while not stop_event.is_set():
            cycle_start = time.monotonic()
            try:
                if stream is not None:
                    matrix = stream.read(timeout=8.0)
                else:
                    matrix = read_matrix(adb, args.source)
            except (RuntimeError, ValueError, TimeoutError) as error:
                message = str(error)
                errors["read"] = message
                if stream is not None and adb_device_ready(adb):
                    stream_restarts += 1
                    time.sleep(0.8)
                    try:
                        open_stream()
                        continue
                    except RuntimeError:
                        pass
                time.sleep(max(0.0, period - (time.monotonic() - cycle_start)))
                continue

            if matrix.shape != (ROWS, COLS):
                errors["shape"] = f"expected {ROWS}x{COLS}, got {matrix.shape}"
                time.sleep(max(0.0, period - (time.monotonic() - cycle_start)))
                continue

            if baseline is None:
                baseline_buf.append(matrix.copy())
                if len(baseline_buf) < args.baseline_frames:
                    time.sleep(max(0.0, period - (time.monotonic() - cycle_start)))
                    continue
                baseline = np.mean(np.stack(baseline_buf, axis=0), axis=0)
                write_baseline_matrix(run_dir, baseline)
                frame = 0

            delta = matrix - baseline
            frame += 1
            sample_time = time.monotonic()
            device_ts = stream.last_device_ts_ms if stream is not None else None
            if device_ts is not None and device_epoch_anchor_ms is not None:
                elapsed = (device_ts - device_epoch_anchor_ms) / 1000.0
            else:
                elapsed = sample_time - start
            ts_ms = device_ts if device_ts is not None else int(time.time() * 1000)
            writer.write_frame(ts_ms, elapsed, frame, matrix, delta)
            handle.flush()
            successes += 1
            completed_at = time.monotonic()
            if period > 0:
                time.sleep(max(0.0, period - (time.monotonic() - cycle_start)))

    if stream is not None:
        stream.close()

    actual_duration_s = max(0.0, completed_at - start)
    achieved_frame_rate_hz = (frame / actual_duration_s) if actual_duration_s else 0.0
    (run_dir / "capacitance_summary.json").write_text(
        json.dumps(
            {
                "frames": frame,
                "matrix_records": successes,
                "source": args.source,
                "read_mode": args.read_mode,
                "errors": errors,
                "csv": CAP_CSV_NAME,
                "requested_sample_rate_hz": args.rate,
                "actual_duration_s": actual_duration_s,
                "achieved_frame_rate_hz": achieved_frame_rate_hz,
                "baseline_frames": args.baseline_frames,
                "matrix_shape": f"{ROWS}x{COLS}",
                "stream_restarts": stream_restarts,
                "elapsed_clock": (
                    "device" if device_epoch_anchor_ms is not None else "host_arrival"
                ),
                "device_epoch_anchor_ms": device_epoch_anchor_ms,
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
        f"({frame} frames over {actual_duration_s:.1f} s)"
    )
    return 0
