#!/usr/bin/env python3
"""Live finger-patch detection, per-pixel streaming, and CPG strategy windows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from maxtouch_caplog_strategies import (
    BP_HIGH,
    BP_LOW,
    HR_HIGH,
    HR_LOW,
    RESAMPLE_HZ,
    ac_snr,
    build_strategies,
    butter_bandpass,
    estimate_hr,
    pearson_r,
    resample_to_uniform,
    zscore,
    zscore_psd_curve,
)
from oneplus_monitor import build_rwgb_colormap, symmetric_limits
from region_stats import (
    auto_detect_touch_regions,
    calculate_region_stats,
    coalesce_bipolar_touch_regions,
    default_finger_name,
    default_palm_name,
    dipole_mean,
    order_fingers_anatomically,
)

FINGER_PATCH_SIZE = 8
CAP_TRACE_WINDOW_S = 60.0
FINGER_PIXEL_STREAM_DELAY_AFTER_DETECT_S = 0.0
STRATEGY_STREAM_PROCESS_WINDOW_S = 10.0
STRATEGY_STREAM_DISPLAY_WINDOW_S = 20.0
STRATEGY_STREAM_UPDATE_S = 1.0
STRATEGY_STREAM_MIN_SPAN_S = 3.0
STRATEGY_SELECTION_REFRESH_S = 5.0   # how often to re-pick top-4 pixels vs PPG

CPG_EXTRACTION_PATCH_MEAN = "patch_mean"
CPG_EXTRACTION_TOP4_PPG_CORR = "top4_ppg_corr"


class PpgTraceLike(Protocol):
    x: list[float]
    y: list[float]


class PpgTailerLike(Protocol):
    traces: dict[str, PpgTraceLike]


@dataclass(frozen=True)
class FingerPatch:
    """A crop in display coordinates around an auto-detected contact region.

    Fingers use a fixed ``patch_size`` × ``patch_size`` square centered on the
    blob. Palms set ``kind="palm"`` and use the blob's full bounding box; when
    ``mask`` is provided (crop-relative bool array) the CPG mean is taken over
    the contact pixels only, ignoring non-contact pixels inside the box.
    """

    name: str
    y_min: int
    y_max: int
    x_min: int
    x_max: int
    patch_size: int
    kind: str = "finger"
    mask: np.ndarray | None = field(default=None, compare=False)
    bipolar_contact: bool = False

    @property
    def is_palm(self) -> bool:
        return self.kind == "palm"

    def _masked_values(self, crop: np.ndarray) -> np.ndarray:
        if self.mask is not None and self.mask.shape == crop.shape and self.mask.any():
            return crop[self.mask]
        return crop

    def mean(self, display_grid: np.ndarray) -> float:
        crop = display_grid[self.y_min : self.y_max, self.x_min : self.x_max]
        if crop.size == 0:
            return float("nan")
        if self.bipolar_contact:
            return dipole_mean(self._masked_values(crop))
        if self.mask is not None and self.mask.shape == crop.shape and self.mask.any():
            return float(np.mean(crop[self.mask]))
        if self.is_palm:
            return float(np.mean(crop))
        if crop.shape != (self.patch_size, self.patch_size):
            padded = np.zeros((self.patch_size, self.patch_size), dtype=crop.dtype)
            cy = (self.y_min + self.y_max) // 2
            cx = (self.x_min + self.x_max) // 2
            offset_before = self.patch_size // 2
            y_offset = max(0, offset_before - cy)
            x_offset = max(0, offset_before - cx)
            padded[
                y_offset : y_offset + crop.shape[0],
                x_offset : x_offset + crop.shape[1],
            ] = crop
            crop = padded
        return float(np.mean(crop))

    def crop(self, display_grid: np.ndarray) -> np.ndarray:
        return display_grid[self.y_min : self.y_max, self.x_min : self.x_max]

    def pixel_coords(self) -> list[tuple[int, int, str]]:
        """Column-major patch order: r1c1, r2c1, …, r1c2, r2c2, … (1-based labels)."""
        coords: list[tuple[int, int, str]] = []
        patch_h = self.y_max - self.y_min
        patch_w = self.x_max - self.x_min
        use_mask = (
            self.bipolar_contact
            and self.mask is not None
            and self.mask.shape == (patch_h, patch_w)
        )
        for col in range(patch_w):
            for row in range(patch_h):
                if use_mask and not self.mask[row, col]:
                    continue
                y = self.y_min + row
                x = self.x_min + col
                coords.append((y, x, f"r{row + 1}c{col + 1}"))
        return coords


_COLUMN_TRACE_RGB = (
    (1.0, 0.0, 0.0),    # red
    (1.0, 0.53, 0.0),   # orange
    (1.0, 0.85, 0.0),   # yellow
    (0.0, 0.67, 0.0),   # green
    (0.0, 0.4, 1.0),    # blue
    (0.53, 0.0, 0.8),   # purple
)


def pixel_trace_colors(n_rows: int, n_cols: int) -> list[tuple[float, float, float]]:
    """One hue per column; within each column, r1 light → rN dark."""
    import colorsys

    colors: list[tuple[float, float, float]] = []
    for col in range(n_cols):
        base = _COLUMN_TRACE_RGB[col % len(_COLUMN_TRACE_RGB)]
        hue, _lightness, saturation = colorsys.rgb_to_hls(*base)
        saturation = max(saturation, 0.55)
        for row in range(n_rows):
            row_frac = row / max(n_rows - 1, 1)
            lightness = 0.88 - row_frac * 0.62
            colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors


def pixel_stream_lane_step(signals: list[np.ndarray], *, overlap: float = 0.65) -> float:
    ptps = [
        float(np.percentile(np.abs(signal), 99) - np.percentile(np.abs(signal), 1))
        for signal in signals
        if signal.size > 0
    ]
    ptps = [value if value > 1e-9 else 1.0 for value in ptps]
    if not ptps:
        return 1.0
    return float(np.median(ptps)) * overlap


class FingerPixelStreamWindows:
    """One matplotlib window per finger: heatmap, stacked per-pixel traces, optional PPG."""

    def __init__(
        self,
        run_dir: Path,
        *,
        color_max: float | None,
        trace_window_s: float = CAP_TRACE_WINDOW_S,
        ppg_tailer: PpgTailerLike | None = None,
        session_t0_s: float = 0.0,
        ppg_x_is_elapsed: bool = False,
    ) -> None:
        self.run_dir = run_dir
        self.color_max = color_max
        self.trace_window_s = trace_window_s
        self.ppg_tailer = ppg_tailer
        self.session_t0_s = session_t0_s
        self.ppg_x_is_elapsed = ppg_x_is_elapsed
        self._windows: list[dict[str, object]] = []

    def ppg_elapsed_s(self, raw_x: float) -> float:
        if self.ppg_x_is_elapsed:
            return raw_x
        return raw_x / 1000.0 - self.session_t0_s

    def _sync_time_axes(self, window: dict[str, object], latest_s: float) -> None:
        start = max(0.0, latest_s - self.trace_window_s)
        end = max(latest_s, start + 1.0)
        window["ax_trace"].set_xlim(start, end)  # type: ignore[attr-defined]

    def _ppg_latest_aligned(self) -> float:
        if self.ppg_tailer is None:
            return 0.0
        green = self.ppg_tailer.traces.get("green")
        if green is None or not green.x:
            return 0.0
        return self.ppg_elapsed_s(green.x[-1])

    def open(self, patches: list[FingerPatch]) -> None:
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        cmap = build_rwgb_colormap()
        for patch in patches:
            coords = patch.pixel_coords()
            n_pixels = len(coords)
            patch_h = patch.y_max - patch.y_min
            patch_w = patch.x_max - patch.x_min
            trace_h = min(24.0, 2.0 + 0.20 * max(n_pixels, 1))
            ppg_ratio = 1.2 if self.ppg_tailer is not None else 0.0
            fig_h = 4.5 + trace_h * 0.55 + (1.8 if ppg_ratio else 0.0)
            fig = plt.figure(figsize=(11, fig_h))
            height_ratios = [max(2.5, patch_h * 0.55), trace_h]
            if ppg_ratio:
                height_ratios.append(ppg_ratio)
            spec = GridSpec(
                len(height_ratios),
                1,
                height_ratios=height_ratios,
                hspace=0.32,
                left=0.08,
                right=0.96,
                top=0.92,
                bottom=0.10,
            )
            ax_heat = fig.add_subplot(spec[0])
            ax_trace = fig.add_subplot(spec[1])
            ax_ppg = None
            if self.ppg_tailer is not None:
                ax_ppg = fig.add_subplot(spec[2], sharex=ax_trace)

            blank = np.zeros((patch_h, patch_w), dtype=np.float32)
            vmin, vmax = symmetric_limits(blank, self.color_max)
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
            im = ax_heat.imshow(
                blank,
                norm=norm,
                cmap=cmap,
                aspect="equal",
                origin="upper",
                interpolation="nearest",
            )
            ax_heat.set_xticks(np.arange(patch_w))
            ax_heat.set_yticks(np.arange(patch_h))
            ax_heat.set_xticklabels(
                [str(patch.x_min + col) for col in range(patch_w)], fontsize=8
            )
            ax_heat.set_yticklabels(
                [str(patch.y_min + row) for row in range(patch_h)], fontsize=8
            )
            ax_heat.set_xlabel("display x")
            ax_heat.set_ylabel("display y")
            heat_texts: list[list[object]] = []
            for row in range(patch_h):
                row_texts: list[object] = []
                for col in range(patch_w):
                    text = ax_heat.text(
                        col,
                        row,
                        "—",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="black",
                    )
                    row_texts.append(text)
                heat_texts.append(row_texts)
            fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04, label="Δ")

            colors = pixel_trace_colors(patch_h, patch_w)
            trace_lines: dict[str, object] = {}
            for index, (_y, _x, label) in enumerate(coords):
                (line,) = ax_trace.plot([], [], linewidth=0.8, alpha=0.85, color=colors[index])
                trace_lines[label] = line
            lane_labels: list[object] = []
            for index, (_y, _x, label) in enumerate(coords):
                text = ax_trace.text(
                    0.0,
                    0.0,
                    label,
                    fontsize=5,
                    color=colors[index],
                    ha="left",
                    va="center",
                    clip_on=False,
                )
                lane_labels.append(text)
            ax_trace.set_ylabel("Δ (stacked lanes)")
            ax_trace.set_yticks([])
            ax_trace.grid(True, alpha=0.25, axis="x")
            ppg_line = None
            if ax_ppg is not None:
                ax_ppg.set_title("Live PPG green")
                ax_ppg.set_ylabel("Green ADC")
                ax_ppg.set_xlabel("Elapsed seconds (session start)")
                ax_ppg.grid(True, alpha=0.25, axis="x")
                plt.setp(ax_trace.get_xticklabels(), visible=False)
            else:
                ax_trace.set_xlabel("Elapsed seconds (session start)")

            title = fig.suptitle(
                f"{patch.name} — live per-pixel Δ  |  {self.run_dir.name}",
                fontsize=11,
            )
            self._windows.append(
                {
                    "patch": patch,
                    "fig": fig,
                    "ax_heat": ax_heat,
                    "ax_trace": ax_trace,
                    "ax_ppg": ax_ppg,
                    "ppg_line": ppg_line,
                    "im": im,
                    "title": title,
                    "heat_texts": heat_texts,
                    "coords": coords,
                    "trace_lines": trace_lines,
                    "lane_labels": lane_labels,
                    "colors": colors,
                    "times": [],
                    "values": {label: [] for _y, _x, label in coords},
                    "last_frame": -1,
                    "lane_step": 1.0,
                    "stream_hz": None,
                }
            )
            fig.show()

    def update(self, display_grid: np.ndarray, elapsed_s: float, frame_count: int) -> None:
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt

        for window in self._windows:
            fig = window["fig"]
            if not plt.fignum_exists(fig.number):
                continue
            if frame_count == window["last_frame"]:
                continue
            window["last_frame"] = frame_count

            patch: FingerPatch = window["patch"]  # type: ignore[assignment]
            crop = patch.crop(display_grid)
            im = window["im"]
            vmin, vmax = symmetric_limits(crop, self.color_max)
            im.set_data(crop)
            im.set_norm(mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax))

            heat_texts: list[list[object]] = window["heat_texts"]  # type: ignore[assignment]
            for row in range(crop.shape[0]):
                for col in range(crop.shape[1]):
                    heat_texts[row][col].set_text(f"{crop[row, col]:.0f}")

            coords: list[tuple[int, int, str]] = window["coords"]  # type: ignore[assignment]
            times: list[float] = window["times"]  # type: ignore[assignment]
            values: dict[str, list[float]] = window["values"]  # type: ignore[assignment]
            times.append(elapsed_s)
            for y, x, label in coords:
                values[label].append(float(display_grid[y, x]))

            start = max(0.0, elapsed_s - self.trace_window_s)
            start_idx = int(np.searchsorted(times, start))
            trace_times = times[start_idx:]
            if not trace_times:
                continue

            stacked_signals = [
                np.asarray(values[label][start_idx:], dtype=np.float64) for _y, _x, label in coords
            ]
            window["lane_step"] = pixel_stream_lane_step(stacked_signals)
            step = window["lane_step"]

            trace_lines: dict[str, object] = window["trace_lines"]  # type: ignore[assignment]
            lane_labels: list[object] = window["lane_labels"]  # type: ignore[assignment]
            for index, (_y, _x, label) in enumerate(coords):
                signal = stacked_signals[index]
                centered = signal - float(np.mean(signal)) if signal.size else signal
                y_off = -index * step
                trace_lines[label].set_data(trace_times, centered + y_off)
                lane_labels[index].set_position((trace_times[0], y_off))
                lane_labels[index].set_text(f" {label}")

            if len(times) >= 2:
                window_s = 5.0
                start = times[-1] - window_s
                start_idx = int(np.searchsorted(times, start))
                hz_window = times[start_idx:]
                if len(hz_window) >= 2 and hz_window[-1] > hz_window[0]:
                    window["stream_hz"] = (len(hz_window) - 1) / (
                        hz_window[-1] - hz_window[0]
                    )

            ax_trace = window["ax_trace"]
            ax_trace.relim()
            ax_trace.autoscale_view()
            self._sync_time_axes(
                window,
                max(trace_times[-1], self._ppg_latest_aligned()),
            )

            stream_hz = window.get("stream_hz")
            hz_text = f"{stream_hz:.1f} Hz" if stream_hz is not None else "— Hz"
            title = window["title"]
            title.set_text(
                f"{patch.name} — live per-pixel Δ  |  {self.run_dir.name}  |  "
                f"frame {frame_count}  |  t={elapsed_s:.2f}s  |  "
                f"{len(coords)} pixels  |  {hz_text}"
            )
            fig.canvas.draw_idle()

    def update_ppg(self) -> None:
        if self.ppg_tailer is None:
            return
        import matplotlib.pyplot as plt

        green = self.ppg_tailer.traces.get("green")
        if green is None or not green.x:
            return

        xs = [self.ppg_elapsed_s(raw_x) for raw_x in green.x]
        for window in self._windows:
            fig = window["fig"]
            ax_ppg = window.get("ax_ppg")
            if ax_ppg is None or not plt.fignum_exists(fig.number):
                continue

            latest_s = max(xs[-1], window["times"][-1] if window["times"] else 0.0)  # type: ignore[index]
            start = max(0.0, latest_s - self.trace_window_s)
            start_idx = int(np.searchsorted(xs, start))
            plot_xs = xs[start_idx:]
            plot_ys = green.y[start_idx:]
            if not plot_xs:
                continue

            ppg_line = window.get("ppg_line")
            if ppg_line is None:
                (line,) = ax_ppg.plot(plot_xs, plot_ys, color="C0", linewidth=1, label="green")
                ax_ppg.legend(loc="upper right", fontsize="small")
                window["ppg_line"] = line
            else:
                ppg_line.set_data(plot_xs, plot_ys)

            y_pad = max(1.0, (max(plot_ys) - min(plot_ys)) * 0.10)
            ax_ppg.set_ylim(min(plot_ys) - y_pad, max(plot_ys) + y_pad)
            self._sync_time_axes(window, latest_s)
            fig.canvas.draw_idle()

    def prune_closed(self) -> None:
        import matplotlib.pyplot as plt

        self._windows = [
            window
            for window in self._windows
            if plt.fignum_exists(window["fig"].number)
        ]

    def any_open(self) -> bool:
        import matplotlib.pyplot as plt

        return any(plt.fignum_exists(window["fig"].number) for window in self._windows)


def select_live_strategy_names(
    names: list[str],
    n_pixels: int,
    *,
    include_ppg_corr: bool = True,
) -> list[str]:
    """Subset of build_strategies keys shown in the live strategy window."""
    selected: list[str] = []
    for name in names:
        if name == f"All-{n_pixels} mean":
            selected.append(name)
        elif name == "Top-8 variance":
            selected.append(name)
        elif name.startswith("Best px ("):
            selected.append(name)
        elif include_ppg_corr and name == "Top-4 PPG-corr mean":
            selected.append(name)
    return selected


@dataclass
class LiveStrategyResult:
    t_proc: np.ndarray
    filtered_proc: dict[str, np.ndarray]
    metrics: dict[str, dict]
    strat_names: list[str]
    ppg_proc: np.ndarray | None
    t_disp: np.ndarray
    filtered_disp: dict[str, np.ndarray]
    ppg_disp: np.ndarray | None


def _resample_cap_window(
    times: list[float],
    values: dict[str, list[float]],
    pixel_cols: list[str],
    *,
    t_end: float,
    window_s: float,
    ppg_times: list[float] | None = None,
    ppg_values: list[float] | None = None,
    min_span_s: float = STRATEGY_STREAM_MIN_SPAN_S,
) -> tuple[np.ndarray, "pd.DataFrame", np.ndarray | None] | None:
    """Resample trailing capacitance (and optional PPG) to a uniform grid."""
    import pandas as pd
    from scipy import interpolate

    if len(times) < 4:
        return None
    t_start = max(times[0], t_end - window_s)
    start_idx = int(np.searchsorted(times, t_start))
    t_use = np.asarray(times[start_idx:], dtype=np.float64)
    if t_use.size < 4 or (t_use[-1] - t_use[0]) < min_span_s:
        return None

    cap_matrix = np.column_stack(
        [np.asarray(values[col][start_idx:], dtype=np.float64) for col in pixel_cols]
    )
    t_cap_u = np.arange(t_use[0], t_use[-1], 1.0 / RESAMPLE_HZ)
    if t_cap_u.size < max(4, int(min_span_s * RESAMPLE_HZ * 0.5)):
        return None

    cap_uniform: dict[str, np.ndarray] = {}
    for index, col in enumerate(pixel_cols):
        col_vals = cap_matrix[:, index]
        fn = interpolate.interp1d(
            t_use,
            col_vals,
            kind="linear",
            bounds_error=False,
            fill_value=(col_vals[0], col_vals[-1]),
        )
        cap_uniform[col] = fn(t_cap_u)
    cap_u_df = pd.DataFrame(cap_uniform)

    ppg_interp: np.ndarray | None = None
    if ppg_times and ppg_values and len(ppg_times) >= 4:
        ppg_t = np.asarray(ppg_times, dtype=np.float64)
        ppg_v = np.asarray(ppg_values, dtype=np.float64)
        ppg_mask = (ppg_t >= t_use[0]) & (ppg_t <= t_use[-1])
        if int(ppg_mask.sum()) >= 4:
            t_ppg_u, ppg_u = resample_to_uniform(
                ppg_t[ppg_mask], ppg_v[ppg_mask], RESAMPLE_HZ
            )
            ppg_filt_full = butter_bandpass(ppg_u, BP_LOW, BP_HIGH, RESAMPLE_HZ)
            t_min = max(t_cap_u[0], t_ppg_u[0])
            t_max = min(t_cap_u[-1], t_ppg_u[-1])
            cap_mask = (t_cap_u >= t_min) & (t_cap_u <= t_max)
            ppg_mask_u = (t_ppg_u >= t_min) & (t_ppg_u <= t_max)
            t_common = t_cap_u[cap_mask]
            cap_u_df = cap_u_df.loc[cap_mask].reset_index(drop=True)
            ppg_interp = np.interp(
                t_common, t_ppg_u[ppg_mask_u], ppg_filt_full[ppg_mask_u]
            )
            return t_common, cap_u_df, ppg_interp

    return t_cap_u, cap_u_df, ppg_interp


def _live_strategy_selection(
    cap_u_df: "pd.DataFrame",
    pixel_cols: list[str],
    n_y: int,
    n_x: int,
    ppg_interp: np.ndarray | None,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    """Pixel indices chosen on the processing window."""
    X = cap_u_df[pixel_cols].values.astype(float)
    n_pixels = len(pixel_cols)
    ac_var = np.var(X - np.mean(X, axis=0, keepdims=True), axis=0)
    selection: dict[str, object] = {
        "n_pixels": n_pixels,
        "top8_idx": np.argsort(ac_var)[-8:] if n_pixels >= 8 else None,
        "best_px_idx": int(np.argmax(ac_var)),
        "top4_idx": None,
    }
    weights = np.zeros(n_pixels)
    if ppg_interp is not None:
        for i in range(n_pixels):
            col_data = X[:, i]
            if col_data.std() < 1e-10:
                continue
            px_bp = butter_bandpass(col_data - col_data.mean(), BP_LOW, BP_HIGH, RESAMPLE_HZ)
            weights[i] = abs(pearson_r(ppg_interp, px_bp))
        if weights.max() > 0:
            top_k = min(4, n_pixels)
            selection["top4_idx"] = np.argsort(weights)[-top_k:]
    return selection, ac_var, weights


def refresh_top4_ppg_corr_selection(
    times: list[float],
    pixel_values: dict[str, list[float]],
    pixel_cols: list[str],
    *,
    ppg_times: list[float] | None = None,
    ppg_values: list[float] | None = None,
    window_s: float = STRATEGY_STREAM_PROCESS_WINDOW_S,
) -> dict[str, object] | None:
    """Re-pick the top-4 PPG-correlated pixels from a trailing cap+PPG window."""
    if len(times) < 4:
        return None
    proc = _resample_cap_window(
        times,
        pixel_values,
        pixel_cols,
        t_end=times[-1],
        window_s=window_s,
        ppg_times=ppg_times,
        ppg_values=ppg_values,
    )
    if proc is None:
        return None
    _t_common, cap_u_df, ppg_interp = proc
    patch_h = max(int(name[1 : name.index("c")]) for name in pixel_cols)
    patch_w = max(int(name[name.index("c") + 1 :]) for name in pixel_cols)
    selection, _ac_var, _weights = _live_strategy_selection(
        cap_u_df, pixel_cols, patch_h, patch_w, ppg_interp
    )
    return selection


def top4_ppg_corr_sample(
    patch: FingerPatch,
    display_grid: np.ndarray,
    selection: dict[str, object] | None,
    pixel_cols: list[str],
) -> float:
    """Mean of the currently selected top-4 PPG-correlated pixels in this patch."""
    top4_idx = None if selection is None else selection.get("top4_idx")
    if top4_idx is None:
        return patch.mean(display_grid)
    coord_by_label = {label: (y, x) for y, x, label in patch.pixel_coords()}
    values: list[float] = []
    for idx in top4_idx:
        label = pixel_cols[int(idx)]
        pos = coord_by_label.get(label)
        if pos is None:
            continue
        values.append(float(display_grid[pos[0], pos[1]]))
    if not values:
        return patch.mean(display_grid)
    if patch.bipolar_contact:
        return dipole_mean(np.asarray(values, dtype=np.float64))
    return float(np.mean(values))


def rebuild_top4_history(
    pixel_raw: dict[str, list[float]],
    pixel_cols: list[str],
    selection: dict[str, object] | None,
    *,
    bipolar_contact: bool = False,
) -> list[float]:
    """Recompute the full CPG history for one finger under the current selection.

    Called when the top-4 pixel set is re-picked: re-averaging the stored
    per-pixel history keeps the whole trace consistent with the current pixels,
    so the band-pass sees no step discontinuity (which otherwise rings as a bump
    every re-selection interval).
    """
    top4_idx = None if selection is None else selection.get("top4_idx")
    labels = pixel_cols if top4_idx is None else [pixel_cols[int(i)] for i in top4_idx]
    columns = [np.asarray(pixel_raw[label], dtype=np.float64) for label in labels if label in pixel_raw]
    if not columns:
        return []
    length = min(col.size for col in columns)
    if length == 0:
        return []
    stacked = np.vstack([col[:length] for col in columns])
    if not bipolar_contact:
        return stacked.mean(axis=0).tolist()
    return [dipole_mean(stacked[:, t]) for t in range(length)]


def _signals_from_selection(
    cap_u_df: "pd.DataFrame",
    pixel_cols: list[str],
    selection: dict[str, object],
    strat_names: list[str],
) -> dict[str, np.ndarray]:
    X = cap_u_df[pixel_cols].values.astype(float)
    n_pixels = int(selection["n_pixels"])
    strategies: dict[str, np.ndarray] = {}
    all_mean = f"All-{n_pixels} mean"
    if all_mean in strat_names:
        strategies[all_mean] = X.mean(axis=1)
    if "Top-8 variance" in strat_names and selection["top8_idx"] is not None:
        strategies["Top-8 variance"] = X[:, selection["top8_idx"]].mean(axis=1)  # type: ignore[index]
    for name in strat_names:
        if name.startswith("Best px ("):
            strategies[name] = X[:, int(selection["best_px_idx"])]
    if "Top-4 PPG-corr mean" in strat_names:
        top4_idx = selection["top4_idx"]
        if top4_idx is not None:
            strategies["Top-4 PPG-corr mean"] = X[:, top4_idx].mean(axis=1)  # type: ignore[index]
        else:
            strategies["Top-4 PPG-corr mean"] = X.mean(axis=1)
    return strategies


def _filter_strategies(strategies: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    filtered: dict[str, np.ndarray] = {}
    for name, sig in strategies.items():
        filtered[name] = butter_bandpass(sig - sig.mean(), BP_LOW, BP_HIGH, RESAMPLE_HZ)
    return filtered


def process_live_strategies(
    times: list[float],
    values: dict[str, list[float]],
    pixel_cols: list[str],
    *,
    process_window_s: float,
    display_window_s: float,
    ppg_times: list[float] | None = None,
    ppg_values: list[float] | None = None,
    include_ppg_corr: bool = True,
) -> LiveStrategyResult | None:
    """Process trailing cap data: strategy pick + metrics/PSD on process window, traces on display window."""
    if len(times) < 4:
        return None
    t_end = times[-1]
    proc_ppg_t, proc_ppg_v = [], []
    if ppg_times and ppg_values:
        proc_start = max(times[0], t_end - process_window_s)
        for t, v in zip(ppg_times, ppg_values):
            if proc_start <= t <= t_end:
                proc_ppg_t.append(t)
                proc_ppg_v.append(v)

    proc = _resample_cap_window(
        times,
        values,
        pixel_cols,
        t_end=t_end,
        window_s=process_window_s,
        ppg_times=proc_ppg_t or None,
        ppg_values=proc_ppg_v or None,
    )
    if proc is None:
        return None
    t_proc, cap_proc, ppg_proc = proc

    n_y = max(int(name[1 : name.index("c")]) for name in pixel_cols)
    n_x = max(int(name[name.index("c") + 1 :]) for name in pixel_cols)
    selection, _ac_var, _weights = _live_strategy_selection(
        cap_proc, pixel_cols, n_y, n_x, ppg_proc
    )
    all_strategies, _, _ = build_strategies(cap_proc, pixel_cols, n_y, n_x, ppg_proc)
    strat_names = select_live_strategy_names(
        list(all_strategies.keys()),
        len(pixel_cols),
        include_ppg_corr=include_ppg_corr,
    )
    proc_raw = _signals_from_selection(cap_proc, pixel_cols, selection, strat_names)
    if not strat_names:
        return None

    filtered_proc = _filter_strategies(proc_raw)
    metrics: dict[str, dict] = {}
    for name in strat_names:
        sig_filt = filtered_proc[name]
        entry = dict(
            snr_db=ac_snr(sig_filt, RESAMPLE_HZ, HR_LOW, HR_HIGH),
            hr_bpm=estimate_hr(sig_filt, RESAMPLE_HZ, HR_LOW, HR_HIGH),
        )
        if ppg_proc is not None:
            entry["pearson_r"] = abs(pearson_r(sig_filt, ppg_proc))
        metrics[name] = entry

    disp_start = max(times[0], t_end - display_window_s)
    disp_ppg_t, disp_ppg_v = [], []
    if ppg_times and ppg_values:
        for t, v in zip(ppg_times, ppg_values):
            if disp_start <= t <= t_end:
                disp_ppg_t.append(t)
                disp_ppg_v.append(v)

    disp = _resample_cap_window(
        times,
        values,
        pixel_cols,
        t_end=t_end,
        window_s=display_window_s,
        ppg_times=disp_ppg_t or None,
        ppg_values=disp_ppg_v or None,
        min_span_s=0.5,
    )
    if disp is None:
        t_disp, filtered_disp, ppg_disp = t_proc, filtered_proc, ppg_proc
    else:
        t_disp, cap_disp, ppg_disp = disp
        disp_raw = _signals_from_selection(cap_disp, pixel_cols, selection, strat_names)
        filtered_disp = _filter_strategies(disp_raw)

    return LiveStrategyResult(
        t_proc=t_proc,
        filtered_proc=filtered_proc,
        metrics=metrics,
        strat_names=strat_names,
        ppg_proc=ppg_proc,
        t_disp=t_disp,
        filtered_disp=filtered_disp,
        ppg_disp=ppg_disp,
    )


class FingerStrategyStreamWindows:
    """One matplotlib window per finger: live CPG strategy waveforms + FFT."""

    def __init__(
        self,
        run_dir: Path,
        *,
        process_window_s: float = STRATEGY_STREAM_PROCESS_WINDOW_S,
        display_window_s: float = STRATEGY_STREAM_DISPLAY_WINDOW_S,
        update_interval_s: float = STRATEGY_STREAM_UPDATE_S,
        ppg_tailer: PpgTailerLike | None = None,
        session_t0_s: float = 0.0,
        ppg_x_is_elapsed: bool = False,
        include_ppg_corr: bool = True,
    ) -> None:
        self.run_dir = run_dir
        self.process_window_s = process_window_s
        self.display_window_s = display_window_s
        self.update_interval_s = update_interval_s
        self.ppg_tailer = ppg_tailer
        self.session_t0_s = session_t0_s
        self.ppg_x_is_elapsed = ppg_x_is_elapsed
        self.include_ppg_corr = include_ppg_corr
        self._windows: list[dict[str, object]] = []

    def ppg_elapsed_s(self, raw_x: float) -> float:
        if self.ppg_x_is_elapsed:
            return raw_x
        return raw_x / 1000.0 - self.session_t0_s

    def open(self, patches: list[FingerPatch]) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        for patch in patches:
            coords = patch.pixel_coords()
            pixel_cols = [label for _y, _x, label in coords]
            n_strats = 4 if self.include_ppg_corr else 3
            has_ppg = self.ppg_tailer is not None
            fig_h = 3.2 + 1.35 * n_strats + (0.85 if has_ppg else 0.0)
            fig = plt.figure(figsize=(11, fig_h))
            height_ratios = [1.35] + [0.95] * n_strats
            if has_ppg:
                height_ratios.append(0.85)
            spec = GridSpec(
                len(height_ratios),
                1,
                height_ratios=height_ratios,
                hspace=0.42,
                left=0.10,
                right=0.96,
                top=0.92,
                bottom=0.08,
            )
            ax_psd = fig.add_subplot(spec[0])
            ax_psd.set_xlabel("Frequency (bpm)")
            ax_psd.set_ylabel("Norm. PSD")
            ax_psd.set_title(
                "Strategy PSD (last {:.0f} s)".format(self.process_window_s)
            )
            ax_psd.grid(True, alpha=0.25)

            ts_axes: list[object] = []
            ax_ts_ref = None
            for index in range(n_strats):
                if ax_ts_ref is None:
                    ax = fig.add_subplot(spec[index + 1])
                    ax_ts_ref = ax
                else:
                    ax = fig.add_subplot(spec[index + 1], sharex=ax_ts_ref)
                ax.set_ylabel("z-score")
                ax.grid(True, alpha=0.25, axis="x")
                if has_ppg or index < n_strats - 1:
                    plt.setp(ax.get_xticklabels(), visible=False)
                else:
                    ax.set_xlabel(
                        f"Time in window (s, last {self.display_window_s:.0f} s)"
                    )
                ts_axes.append(ax)

            ax_ppg = None
            if has_ppg:
                ax_ppg = fig.add_subplot(spec[n_strats + 1], sharex=ax_ts_ref)
                ax_ppg.set_ylabel("z-score")
                ax_ppg.set_xlabel(f"Time in window (s, last {self.display_window_s:.0f} s)")
                ax_ppg.grid(True, alpha=0.25, axis="x")

            title = fig.suptitle(
                f"{patch.name} — live CPG strategies  |  {self.run_dir.name}",
                fontsize=11,
            )
            self._windows.append(
                {
                    "patch": patch,
                    "pixel_cols": pixel_cols,
                    "fig": fig,
                    "title": title,
                    "ax_psd": ax_psd,
                    "ts_axes": ts_axes,
                    "ax_ppg": ax_ppg,
                    "times": [],
                    "values": {label: [] for _y, _x, label in coords},
                    "last_processed_s": -1e9,
                    "strat_names": [],
                    "status_text": fig.text(
                        0.10,
                        0.02,
                        "Collecting data…",
                        ha="left",
                        va="bottom",
                        fontsize=8,
                        color="#555555",
                    ),
                }
            )
            fig.show()

    def _ppg_window(self, t_start: float, t_end: float) -> tuple[list[float], list[float]]:
        if self.ppg_tailer is None:
            return [], []
        green = self.ppg_tailer.traces.get("green")
        if green is None or not green.x:
            return [], []
        times: list[float] = []
        values: list[float] = []
        for raw_x, value in zip(green.x, green.y):
            t = self.ppg_elapsed_s(raw_x)
            if t_start <= t <= t_end:
                times.append(t)
                values.append(float(value))
        return times, values

    def append_frame(self, display_grid: np.ndarray, elapsed_s: float) -> None:
        for window in self._windows:
            patch: FingerPatch = window["patch"]  # type: ignore[assignment]
            times: list[float] = window["times"]  # type: ignore[assignment]
            values: dict[str, list[float]] = window["values"]  # type: ignore[assignment]
            times.append(elapsed_s)
            for y, x, label in patch.pixel_coords():
                values[label].append(float(display_grid[y, x]))

    def refresh(self, elapsed_s: float) -> None:
        import matplotlib.pyplot as plt

        for window in self._windows:
            fig = window["fig"]
            if not plt.fignum_exists(fig.number):
                continue

            patch: FingerPatch = window["patch"]  # type: ignore[assignment]
            pixel_cols: list[str] = window["pixel_cols"]  # type: ignore[assignment]
            times: list[float] = window["times"]  # type: ignore[assignment]
            values: dict[str, list[float]] = window["values"]  # type: ignore[assignment]

            last_processed_s: float = window["last_processed_s"]  # type: ignore[assignment]
            if elapsed_s - last_processed_s < self.update_interval_s:
                continue

            t_end = elapsed_s
            disp_start = max(0.0, t_end - self.display_window_s)
            ppg_times, ppg_values = self._ppg_window(disp_start, t_end)
            result = process_live_strategies(
                times,
                values,
                pixel_cols,
                process_window_s=self.process_window_s,
                display_window_s=self.display_window_s,
                ppg_times=ppg_times or None,
                ppg_values=ppg_values or None,
                include_ppg_corr=self.include_ppg_corr,
            )
            if result is None:
                status = window["status_text"]
                status.set_text(  # type: ignore[attr-defined]
                    f"Collecting ≥{STRATEGY_STREAM_MIN_SPAN_S:.0f}s in last "
                    f"{self.process_window_s:.0f}s process window…  t={elapsed_s:.1f}s"
                )
                fig.canvas.draw_idle()
                continue

            window["last_processed_s"] = elapsed_s
            window["strat_names"] = result.strat_names
            metrics = result.metrics
            strat_names = result.strat_names
            t_plot = result.t_disp - result.t_disp[0]
            ppg_norm = zscore(result.ppg_disp) if result.ppg_disp is not None else None

            ax_psd = window["ax_psd"]
            ax_psd.cla()
            ax_psd.set_xlabel("Frequency (bpm)")
            ax_psd.set_ylabel("Norm. PSD")
            ax_psd.grid(True, alpha=0.25)

            colors = plt.cm.tab10(np.linspace(0, 1, max(len(strat_names), 1)))
            cap_psds = {
                name: zscore_psd_curve(result.filtered_proc[name], RESAMPLE_HZ)
                for name in strat_names
            }
            cap_peak_scale = max((p.max() for _, p in cap_psds.values()), default=0.0) + 1e-30
            short = [
                name.replace(" mean", "").replace(" variance", "") for name in strat_names
            ]
            if result.ppg_proc is not None:
                f_ppg, p_ppg = zscore_psd_curve(result.ppg_proc, RESAMPLE_HZ)
                ax_psd.plot(
                    f_ppg * 60,
                    p_ppg / (p_ppg.max() + 1e-30),
                    "k-",
                    lw=2,
                    label="PPG",
                    zorder=10,
                )
            for index, name in enumerate(strat_names):
                f_s, p_s = cap_psds[name]
                ax_psd.plot(
                    f_s * 60,
                    p_s / cap_peak_scale,
                    color=colors[index],
                    alpha=0.85,
                    lw=1.2,
                    label=short[index],
                )
            ax_psd.set_ylim(0.0, 1.08)
            ax_psd.legend(fontsize=7, ncol=2, loc="upper right")
            ax_psd.set_title(
                f"Strategy PSD (last {self.process_window_s:.0f} s process window, "
                f"refresh {self.update_interval_s:.0f} s)"
            )

            ts_axes: list[object] = window["ts_axes"]  # type: ignore[assignment]
            ax_ppg = window.get("ax_ppg")
            for ax in ts_axes:
                ax.cla()
                ax.grid(True, alpha=0.25, axis="x")
            for index, name in enumerate(strat_names):
                if index >= len(ts_axes):
                    break
                ax = ts_axes[index]
                if ax_ppg is not None or index < len(strat_names) - 1:
                    plt.setp(ax.get_xticklabels(), visible=False)
                cap_norm = zscore(result.filtered_disp[name])
                cap_label = (
                    f"Cap  SNR={metrics[name]['snr_db']:+.1f}dB  "
                    f"HR={metrics[name]['hr_bpm']:.1f}bpm"
                )
                if "pearson_r" in metrics[name]:
                    cap_label = (
                        f"Cap  |r|={metrics[name]['pearson_r']:.3f}  "
                        f"SNR={metrics[name]['snr_db']:+.1f}dB  "
                        f"HR={metrics[name]['hr_bpm']:.1f}bpm"
                    )
                ax.plot(t_plot, cap_norm, color=colors[index], lw=0.9, label=cap_label)
                ax.set_title(name, fontsize=9)
                ax.legend(fontsize=7, loc="upper right")
                ax.set_ylabel("z-score")
                if ax_ppg is None and index == len(strat_names) - 1:
                    ax.set_xlabel(
                        f"Time in window (s, last {self.display_window_s:.0f} s)"
                    )

            if ax_ppg is not None:
                ax_ppg.cla()
                ax_ppg.grid(True, alpha=0.25, axis="x")
                ax_ppg.set_ylabel("z-score")
                ax_ppg.set_xlabel(
                    f"Time in window (s, last {self.display_window_s:.0f} s)"
                )
                ax_ppg.set_title("PPG green", fontsize=9)
                if ppg_norm is not None:
                    ax_ppg.plot(t_plot, ppg_norm, color="green", lw=0.9, label="PPG")
                    ax_ppg.legend(fontsize=7, loc="upper right")

            if t_plot.size:
                x_end = max(float(t_plot[-1]), 0.01)
                for ax in ts_axes[: len(strat_names)]:
                    ax.set_xlim(0.0, x_end)
                if ax_ppg is not None:
                    ax_ppg.set_xlim(0.0, x_end)

            for index in range(len(strat_names), len(ts_axes)):
                ts_axes[index].axis("off")

            hr_bits = [
                f"{short[i]} {metrics[strat_names[i]]['hr_bpm']:.0f}bpm"
                for i in range(len(strat_names))
            ]
            status = window["status_text"]
            status.set_text(  # type: ignore[attr-defined]
                f"Updated t={elapsed_s:.1f}s  |  " + "  |  ".join(hr_bits)
            )
            title = window["title"]
            title.set_text(  # type: ignore[attr-defined]
                f"{patch.name} — live CPG strategies  |  {self.run_dir.name}  |  "
                f"t={elapsed_s:.1f}s"
            )
            fig.canvas.draw_idle()

    def prune_closed(self) -> None:
        import matplotlib.pyplot as plt

        self._windows = [
            window
            for window in self._windows
            if plt.fignum_exists(window["fig"].number)
        ]

    def any_open(self) -> bool:
        import matplotlib.pyplot as plt

        return any(plt.fignum_exists(window["fig"].number) for window in self._windows)


def patch_bounds_from_centroid(
    cy: int,
    cx: int,
    crop_size: int,
    img_h: int,
    img_w: int,
    *,
    pixel_count: int = 0,
    region_height: int = 0,
    region_width: int = 0,
) -> tuple[int, int, int, int]:
    if pixel_count <= 2 and (region_height <= 2 or region_width <= 2):
        offset_before = crop_size // 2 - 1 if crop_size > 2 else 0
        offset_after = crop_size - offset_before
    else:
        offset_before = crop_size // 2
        offset_after = crop_size // 2
    y_min = max(0, cy - offset_before)
    y_max = min(img_h, cy + offset_after)
    x_min = max(0, cx - offset_before)
    x_max = min(img_w, cx + offset_after)
    return y_min, y_max, x_min, x_max


def _palm_patch_from_mask(mask: np.ndarray, patch_size: int) -> FingerPatch:
    """Full-bounding-box patch carrying the crop-relative contact mask."""
    y_coords, x_coords = np.where(mask)
    y_min = int(y_coords.min())
    y_max = int(y_coords.max()) + 1
    x_min = int(x_coords.min())
    x_max = int(x_coords.max()) + 1
    crop_mask = mask[y_min:y_max, x_min:x_max].copy()
    return FingerPatch(
        name="",
        y_min=y_min,
        y_max=y_max,
        x_min=x_min,
        x_max=x_max,
        patch_size=patch_size,
        kind="palm",
        mask=crop_mask,
    )


def detect_finger_patches(
    display_grid: np.ndarray,
    patch_size: int = FINGER_PATCH_SIZE,
    *,
    bipolar_contact: bool = False,
) -> list[FingerPatch]:
    """Auto-detect contact regions and return patch crops sorted left-to-right.

    Small compact blobs become fixed ``patch_size`` square finger patches;
    large irregular blobs become palm patches (full bounding box + contact
    mask, ``kind="palm"``). Fingers are named thumb…little, palms ``palm``.

    ``bipolar_contact`` (Pixel 4 STM FTS): detect on |Δ|, merge a +/− dipole
    into one finger region, and extract CPG with a constructive dipole mean.
    ``display_grid`` must be signed in that mode.
    """
    detect_grid = (
        np.abs(display_grid).astype(np.float64) if bipolar_contact else display_grid
    )
    finger_masks, palm_masks = auto_detect_touch_regions(detect_grid)
    if bipolar_contact:
        finger_masks, palm_masks = coalesce_bipolar_touch_regions(
            finger_masks, palm_masks, display_grid
        )
    if not finger_masks and not palm_masks:
        return []

    img_h, img_w = display_grid.shape
    # Order fingers by hand anatomy (thumb→little) so names land on the right
    # digit regardless of hand rotation on the panel.
    ordered_fingers = order_fingers_anatomically(finger_masks, detect_grid)
    n_fingers = len(ordered_fingers)
    result: list[FingerPatch] = []
    for index, mask in enumerate(ordered_fingers):
        if bipolar_contact:
            y_coords, x_coords = np.where(mask)
            y_min = int(y_coords.min())
            y_max = int(y_coords.max()) + 1
            x_min = int(x_coords.min())
            x_max = int(x_coords.max()) + 1
            result.append(
                FingerPatch(
                    name=default_finger_name(index, n_fingers),
                    y_min=y_min,
                    y_max=y_max,
                    x_min=x_min,
                    x_max=x_max,
                    patch_size=patch_size,
                    mask=mask[y_min:y_max, x_min:x_max].copy(),
                    bipolar_contact=True,
                )
            )
            continue
        stats = calculate_region_stats(mask, display_grid)
        cy, cx = stats["centroid_y"], stats["centroid_x"]
        y_min, y_max, x_min, x_max = patch_bounds_from_centroid(
            cy,
            cx,
            patch_size,
            img_h,
            img_w,
            pixel_count=int(stats["pixel_count"]),
            region_height=int(stats["height_pixels"]),
            region_width=int(stats["width_pixels"]),
        )
        result.append(
            FingerPatch(
                name=default_finger_name(index, n_fingers),
                y_min=y_min,
                y_max=y_max,
                x_min=x_min,
                x_max=x_max,
                patch_size=patch_size,
            )
        )

    palm_patches = [_palm_patch_from_mask(mask, patch_size) for mask in palm_masks]
    palm_patches.sort(key=lambda p: p.x_min)
    for index, patch in enumerate(palm_patches):
        result.append(
            FingerPatch(
                name=default_palm_name(index),
                y_min=patch.y_min,
                y_max=patch.y_max,
                x_min=patch.x_min,
                x_max=patch.x_max,
                patch_size=patch.patch_size,
                kind="palm",
                mask=patch.mask,
                bipolar_contact=bipolar_contact,
            )
        )
    return result

