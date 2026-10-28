#!/usr/bin/env python3
"""maXTouch T37 capacitive pulse (CPG) extraction from saved region NPZ files.

Loads a finger crop defined in ``{session}_regions.npz`` (from
``measure_maxtouch.py``), pulls ``delta`` values from long-format
``capacitance.csv``, checks frame staleness, then compares pixel-combination
strategies against ``ppg_green.csv`` when present.

Usage:
    python3 maxtouch_caplog_strategies.py
    python3 maxtouch_caplog_strategies.py --session data_maxtouch/cap_log/session_3finger_6row_1
    python3 maxtouch_caplog_strategies.py --batch
    python3 maxtouch_caplog_view.py data_maxtouch/cap_log/session_3finger_6row_2
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz  # np.trapz removed in NumPy 2.0
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button
from scipy import interpolate, signal

from activity_labels import draw_label_markers, load_activity_labels, resolve_labels_path
from maxtouch_capacitance import CAP_LOG_DIR

TRIM_END_S = 2.0
RESAMPLE_HZ = 25.0
BP_LOW, BP_HIGH = 0.5, 10.0
HR_LOW, HR_HIGH = 0.5, 3.0
LAG_WIN_S = 1.0
CAP_FILENAME = "capacitance.csv"
REGIONS_SUFFIX = "_regions.npz"
PPG_FILENAME = "ppg_green.csv"
HEADER_ROW_RATIOS = [1.1, 1.1, 0.65, 0.95, 0.95]
TS_ROW_RATIO = 0.9

DEFAULT_DATA_DIR = CAP_LOG_DIR
MAX_VISIBLE_SESSIONS = 16
MAX_VISIBLE_REGIONS = 8

# Interactive scrollable main panel layout (figure coordinates)
SCROLL_MAIN = dict(left=0.18, right=0.958, bottom=0.04, top=0.90)
SCROLL_STEP = 0.065
SCROLL_HEATMAP_H = 0.24
SCROLL_PEARSON_H = 0.14
SCROLL_PSD_H = 0.16
SCROLL_PX_PSD_H = 0.16
SCROLL_TS_H = 0.13
SCROLL_GAP = 0.034

SESSIONS = {
    "3finger_6row_1": "data_maxtouch/cap_log/session_3finger_6row_1",
    "3finger_6row_2": "data_maxtouch/cap_log/session_3finger_6row_2",
    "3finger_6row_3": "data_maxtouch/cap_log/session_3finger_6row_3",
}

VARIANCE_CMAP = LinearSegmentedColormap.from_list(
    "variance_white_red",
    plt.cm.RdBu_r(np.linspace(0.5, 1.0, 256)),
)


def natural_sort_key(text: str) -> list:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def find_viewable_sessions(data_dir: Path) -> list[Path]:
    folders: list[Path] = []
    if not data_dir.exists():
        return folders
    for item in sorted(data_dir.iterdir(), key=lambda p: natural_sort_key(p.name)):
        if not item.is_dir() or not item.name.startswith("session_"):
            continue
        if (item / CAP_FILENAME).is_file() and regions_npz_path(item).is_file():
            folders.append(item)
    return folders


def register_sidebar_axis(ax: plt.Axes, *, facecolor: str = "#f0f0f0") -> plt.Axes:
    ax.set_zorder(200)
    ax.set_navigate(False)
    ax.set_facecolor(facecolor)
    ax.set_clip_on(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


def event_in_axes(fig: plt.Figure, ax: plt.Axes, event) -> bool:
    if event.inaxes == ax:
        return True
    if event.x is None or event.y is None:
        return False
    fx, fy = fig.transFigure.inverted().transform((event.x, event.y))
    pos = ax.get_position()
    return pos.x0 <= fx <= pos.x1 and pos.y0 <= fy <= pos.y1


def event_ydata(ax: plt.Axes, event) -> float | None:
    if event.x is None or event.y is None:
        return None
    if event.inaxes == ax and event.ydata is not None:
        return float(event.ydata)
    try:
        return float(ax.transData.inverted().transform((event.x, event.y))[1])
    except Exception:
        return None


def hit_list_row(y: float, *, max_visible: int, scroll_offset: int, total: int) -> int | None:
    visible = min(max_visible, total - scroll_offset)
    for row in range(visible):
        y_pos = max_visible - row - 0.5
        if y_pos - 0.5 <= y <= y_pos + 0.5:
            return scroll_offset + row
    return None


def hit_region_row(y: float, num_regions: int, *, max_visible: int) -> int | None:
    for idx in range(num_regions):
        y_pos = max_visible - idx - 0.5
        if y_pos - 0.5 <= y <= y_pos + 0.5:
            return idx
    return None


def butter_bandpass(data: np.ndarray, lo: float, hi: float, fs: float, order: int = 2) -> np.ndarray:
    hi = min(hi, fs * 0.45)
    if hi <= lo:
        hi = lo + 0.1
    sos = signal.butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, data)


def resample_to_uniform(t: np.ndarray, x: np.ndarray, fs_out: float) -> tuple[np.ndarray, np.ndarray]:
    t_uniform = np.arange(t[0], t[-1], 1.0 / fs_out)
    fn = interpolate.interp1d(t, x, kind="linear", bounds_error=False, fill_value=(x[0], x[-1]))
    return t_uniform, fn(t_uniform)


def zscore(sig: np.ndarray) -> np.ndarray:
    return (sig - sig.mean()) / (sig.std() + 1e-12)


def pearson_r(sig1: np.ndarray, sig2: np.ndarray) -> float:
    s1 = sig1 - sig1.mean()
    s2 = sig2 - sig2.mean()
    denom = np.sqrt((s1 ** 2).sum() * (s2 ** 2).sum()) + 1e-12
    return float(np.dot(s1, s2) / denom)


def best_xcorr(sig1: np.ndarray, sig2: np.ndarray, fs: float, lag_win_s: float) -> tuple[float, float]:
    s1 = (sig1 - sig1.mean()) / (sig1.std() + 1e-12)
    s2 = (sig2 - sig2.mean()) / (sig2.std() + 1e-12)
    xc = np.correlate(s1, s2, mode="full") / len(s1)
    lags = np.arange(-(len(s1) - 1), len(s1)) / fs
    win = np.abs(lags) <= lag_win_s
    best_lag = lags[win][np.argmax(np.abs(xc[win]))]
    best_r = float(np.max(np.abs(xc[win])))
    return best_r, float(best_lag)


def ac_snr(sig: np.ndarray, fs: float, hr_lo: float, hr_hi: float) -> float:
    f, pxx = signal.welch(sig, fs=fs, nperseg=min(256, len(sig) // 4))
    hr_mask = (f >= hr_lo) & (f <= hr_hi)
    p_hr = _trapz(pxx[hr_mask], f[hr_mask])
    p_out = _trapz(pxx[~hr_mask], f[~hr_mask]) + 1e-30
    return float(10 * np.log10(p_hr / p_out))


def estimate_hr(sig: np.ndarray, fs: float, hr_lo: float, hr_hi: float) -> float:
    f, pxx = signal.welch(sig, fs=fs, nperseg=min(512, len(sig) // 4))
    hr_mask = (f >= hr_lo) & (f <= hr_hi)
    dom_f = f[hr_mask][np.argmax(pxx[hr_mask])]
    return float(dom_f * 60.0)


def dominant_psd_curve(sig: np.ndarray, fs: float, hr_lo: float = 0.3, hr_hi: float = 4.0):
    f, pxx = signal.welch(sig, fs=fs, nperseg=min(512, len(sig) // 4))
    mask = (f >= hr_lo) & (f <= hr_hi)
    return f[mask], pxx[mask]


def zscore_psd_curve(sig: np.ndarray, fs: float, hr_lo: float = 0.3, hr_hi: float = 4.0):
    """Z-score the signal, then compute PSD in the HR band."""
    return dominant_psd_curve(zscore(sig), fs, hr_lo, hr_hi)


def regions_npz_path(session_dir: Path) -> Path:
    return session_dir / f"{session_dir.name}{REGIONS_SUFFIX}"


def region_count(session_dir: Path) -> int:
    npz = np.load(regions_npz_path(session_dir), allow_pickle=True)
    if "num_regions" in npz.files:
        return int(npz["num_regions"])
    return sum(1 for key in npz.files if key.startswith("region_") and key.endswith("_crop_bounds"))


def sanitize_region_slug(name: str) -> str:
    slug = re.sub(r"[^\w\-]+", "_", name.strip(), flags=re.UNICODE)
    return slug[:64] or "region"


def region_name_from_npz(npz, region_idx: int = 0) -> str:
    if "region_names" in npz.files:
        names = [str(name) for name in npz["region_names"]]
        if 0 <= region_idx < len(names):
            return names[region_idx]
    key = f"region_{region_idx}_name"
    if key in npz.files:
        return str(npz[key])
    num = int(npz["num_regions"]) if "num_regions" in npz.files else 1
    return f"region_{region_idx}" if num > 1 else "region"


def region_label(session_dir: Path, region_idx: int) -> str:
    npz = np.load(regions_npz_path(session_dir), allow_pickle=True)
    return region_name_from_npz(npz, region_idx)


def region_output_suffix(session_dir: Path, region_idx: int, num_regions: int) -> str:
    if num_regions <= 1:
        return ""
    return f"_{sanitize_region_slug(region_label(session_dir, region_idx))}"


def strategy_comparison_plot_path(session_data: dict) -> Path:
    session_dir = session_data["session_dir"]
    region_idx = session_data.get("region_idx", 0)
    num_regions = session_data.get("num_regions", 1)
    out_suffix = region_output_suffix(session_dir, region_idx, num_regions)
    return session_dir / f"cpg_strategy_comparison{out_suffix}.png"


def analyze_staleness(matrices: list[np.ndarray], elapsed_s: np.ndarray) -> dict:
    """Report bit-identical consecutive frames (Pixel-style stale cache)."""
    if len(matrices) < 2:
        return {"frames": len(matrices), "stale_steps_pct": 0.0}

    identical = [int(np.array_equal(a, b)) for a, b in zip(matrices[:-1], matrices[1:])]
    max_diffs = [float(np.max(np.abs(b - a))) for a, b in zip(matrices[:-1], matrices[1:])]

    longest = 1
    streak = 1
    for i, is_ident in enumerate(identical):
        if is_ident:
            streak += 1
            longest = max(longest, streak)
        else:
            streak = 1

    duration = float(elapsed_s[-1] - elapsed_s[0]) if len(elapsed_s) > 1 else 0.0
    return {
        "frames": len(matrices),
        "duration_s": duration,
        "avg_hz": len(matrices) / max(duration, 1e-6),
        "stale_steps": int(sum(identical)),
        "stale_steps_pct": 100.0 * sum(identical) / len(identical),
        "changing_steps": int(len(identical) - sum(identical)),
        "unique_matrices": len({m.tobytes() for m in matrices}),
        "longest_identical_streak": int(longest),
        "median_max_abs_diff": float(np.median(max_diffs)),
        "max_max_abs_diff": float(np.max(max_diffs)),
    }


def print_staleness_report(report: dict) -> None:
    print("\n  Staleness check (bit-identical consecutive frames):")
    print(f"    Frames       : {report['frames']}  ({report.get('duration_s', 0):.1f} s @ "
          f"{report.get('avg_hz', 0):.1f} Hz)")
    print(f"    Stale steps  : {report['stale_steps']} / {report['frames'] - 1} "
          f"({report['stale_steps_pct']:.1f}%)")
    print(f"    Unique frames: {report['unique_matrices']} / {report['frames']}")
    print(f"    Longest stale run: {report['longest_identical_streak']} frames")
    print(f"    median max|Δ| between frames: {report['median_max_abs_diff']:.1f}")
    if report["stale_steps_pct"] > 50:
        print("    ⚠ Still mostly stale — proc cache may not be refreshing.")
    elif report["stale_steps_pct"] < 25:
        print("    ✓ Mostly live — consecutive frames usually differ.")


def load_cap_region(
    session_dir: Path,
    *,
    region_idx: int = 0,
    restrict_to_npz_frames: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str], dict, int, int, dict, dict | None]:
    """Load region delta series from CSV using crop bounds in the regions NPZ.

    ``region_idx`` selects which saved finger region to load (0, 1, … from
    ``measure_maxtouch``). By default uses ``frame_indices`` from the NPZ.
    Pass ``restrict_to_npz_frames=False`` to analyse the full capacitance CSV.
    """
    session_dir = session_dir.resolve()
    npz_path = regions_npz_path(session_dir)
    if not npz_path.is_file():
        raise FileNotFoundError(f"Regions file not found: {npz_path}")

    npz = np.load(npz_path, allow_pickle=True)
    bounds_key = f"region_{region_idx}_crop_bounds"
    if bounds_key not in npz.files:
        raise ValueError(f"{npz_path} has no {bounds_key} (num_regions={region_count(session_dir)})")
    y_min, y_max, x_min, x_max = map(int, npz[bounds_key])
    n_y = y_max - y_min
    n_x = x_max - x_min

    source_file = str(npz["source_file"]) if "source_file" in npz.files else ""
    if source_file and Path(source_file).resolve() != session_dir:
        print(f"  WARNING: NPZ was saved from {source_file}")
        print(f"           Using spatial crop from NPZ + pixels from {session_dir}/{CAP_FILENAME}")

    frame_filter: set[int] | None = None
    if restrict_to_npz_frames and "frame_indices" in npz.files:
        frame_filter = set(npz["frame_indices"].astype(int).tolist())

    csv_path = session_dir / CAP_FILENAME

    def read_frames(apply_filter: set[int] | None) -> tuple[list[int], list[np.ndarray], np.ndarray]:
        per_frame: dict[int, dict] = defaultdict(
            lambda: {"elapsed_s": 0.0, "grid": np.zeros((n_y, n_x), dtype=np.float32)}
        )
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"elapsed_s", "frame", "row", "col", "delta"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError(f"{csv_path} missing columns {required}")
            for row in reader:
                frame = int(row["frame"])
                if apply_filter is not None and frame not in apply_filter:
                    continue
                row_idx = int(row["row"])
                col_idx = int(row["col"])
                if not (y_min <= row_idx < y_max and x_min <= col_idx < x_max):
                    continue
                per_frame[frame]["elapsed_s"] = float(row["elapsed_s"])
                per_frame[frame]["grid"][row_idx - y_min, col_idx - x_min] = float(row["delta"])
        if not per_frame:
            raise ValueError(f"No region pixels loaded from {csv_path}")
        frame_ids = sorted(per_frame)
        matrices = [per_frame[fid]["grid"] for fid in frame_ids]
        elapsed = np.asarray([per_frame[fid]["elapsed_s"] for fid in frame_ids], dtype=np.float64)
        return frame_ids, matrices, elapsed

    full_ids, full_mats, full_elapsed = read_frames(None)
    full_staleness = analyze_staleness(full_mats, full_elapsed)

    if frame_filter is not None:
        frame_ids, matrices, elapsed = read_frames(frame_filter)
        npz_staleness = analyze_staleness(matrices, elapsed)
    else:
        frame_ids, matrices, elapsed = full_ids, full_mats, full_elapsed
        npz_staleness = None

    pixel_cols = [f"r{ry}c{rx}" for ry in range(n_y) for rx in range(n_x)]
    rows = [dict(zip(pixel_cols, matrix.reshape(-1))) for matrix in matrices]
    pivot = pd.DataFrame(rows, index=frame_ids)

    window_info = dict(
        region_idx=region_idx,
        y_min=y_min,
        y_max=y_max,
        x_min=x_min,
        x_max=x_max,
        csv_row_lo=y_min,
        csv_row_hi=y_max - 1,
        csv_col_lo=x_min,
        csv_col_hi=x_max - 1,
        start_frame=int(npz["start_frame"]) if "start_frame" in npz.files else frame_ids[0],
    )
    return elapsed, pivot, pixel_cols, window_info, n_y, n_x, full_staleness, npz_staleness


def _parse_pixel_name(name: str) -> tuple[int, int]:
    after_r = name[1:]
    c_idx = after_r.index("c")
    return int(after_r[:c_idx]), int(after_r[c_idx + 1:])


def build_strategies(
    cap_u_df: pd.DataFrame,
    pixel_cols: list[str],
    n_y: int,
    n_x: int,
    ppg_filt: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    X = cap_u_df[pixel_cols].values.astype(float)
    n_pixels = len(pixel_cols)
    strategies: dict[str, np.ndarray] = {}

    strategies[f"All-{n_pixels} mean"] = X.mean(axis=1)

    ac_var = np.var(X - X.mean(axis=0, keepdims=True), axis=0)
    for k in (8, 16):
        if n_pixels >= k:
            idx = np.argsort(ac_var)[-k:]
            strategies[f"Top-{k} variance"] = X[:, idx].mean(axis=1)

    best_px_idx = int(np.argmax(ac_var))
    strategies[f"Best px ({pixel_cols[best_px_idx]})"] = X[:, best_px_idx]

    dc_means = X.mean(axis=0)
    noise_px_idx = int(np.argmin(np.abs(dc_means)))
    strategies[f"Noise px ({pixel_cols[noise_px_idx]})"] = X[:, noise_px_idx]

    weights = np.zeros(n_pixels)
    if ppg_filt is not None:
        for i in range(n_pixels):
            col_data = X[:, i]
            if col_data.std() < 1e-10:
                continue
            px_bp = butter_bandpass(col_data - col_data.mean(), BP_LOW, BP_HIGH, RESAMPLE_HZ)
            weights[i] = abs(pearson_r(ppg_filt, px_bp))
        if weights.max() > 0:
            top_k = min(4, n_pixels)
            top_idx = np.argsort(weights)[-top_k:]
            strategies["Top-4 PPG-corr mean"] = X[:, top_idx].mean(axis=1)
        else:
            strategies["Top-4 PPG-corr mean"] = X.mean(axis=1)

    return strategies, ac_var, weights


def load_ppg(session_dir: Path) -> pd.DataFrame | None:
    path = session_dir / PPG_FILENAME
    if not path.is_file():
        return None
    ppg = pd.read_csv(path)
    if "t_s" not in ppg.columns and "elapsed_s" in ppg.columns:
        ppg = ppg.rename(columns={"elapsed_s": "t_s"})
    if "t_s" not in ppg.columns or "green" not in ppg.columns:
        raise ValueError(f"{path} needs t_s (or elapsed_s) and green columns")
    return ppg


def process_session(
    session_dir: Path,
    *,
    region_idx: int = 0,
    num_regions: int = 1,
    use_npz_frames: bool = True,
) -> dict | None:
    session_dir = session_dir.resolve()
    session_name = session_dir.name
    name = region_label(session_dir, region_idx)
    print(f"\n{'=' * 60}")
    print(f"  {session_name}  |  {name}  |  {session_dir}")
    print(f"{'=' * 60}")

    t_frames, pivot, pixel_cols, window_info, n_y, n_x, full_staleness, npz_staleness = load_cap_region(
        session_dir, region_idx=region_idx, restrict_to_npz_frames=use_npz_frames
    )
    if use_npz_frames and npz_staleness is not None:
        print("  NPZ saved frame window:")
        print_staleness_report(npz_staleness)
        staleness = npz_staleness
    else:
        print("  Full session (crop region):")
        print_staleness_report(full_staleness)
        staleness = full_staleness

    ppg_df = load_ppg(session_dir)
    has_ppg = ppg_df is not None

    t_start = float(t_frames[0])
    t_end = float(t_frames[-1]) - TRIM_END_S
    if has_ppg:
        t_start = max(t_start, float(ppg_df["t_s"].min()))
        t_end = min(t_end, float(ppg_df["t_s"].max()) - TRIM_END_S)

    keep = (t_frames >= t_start) & (t_frames <= t_end)
    t_use = t_frames[keep]
    pivot_use = pivot.iloc[keep].reset_index(drop=True)

    if len(t_use) < 10:
        print("  Too few frames after trim, skipping.")
        return None

    print(f"  Window: {t_use[0]:.1f} – {t_use[-1]:.1f} s  ({len(t_use)} frames)")
    print(f"  Crop: rows [{window_info['y_min']},{window_info['y_max']})  "
          f"cols [{window_info['x_min']},{window_info['x_max']})  ({n_y}×{n_x})")

    ppg_interp: np.ndarray | None = None
    ppg_hr: float | None = None
    if has_ppg:
        ppg_trim = ppg_df[(ppg_df["t_s"] >= t_start) & (ppg_df["t_s"] <= t_end)].copy()
        t_ppg_u, ppg_u = resample_to_uniform(
            ppg_trim["t_s"].values, ppg_trim["green"].values.astype(float), RESAMPLE_HZ
        )
        ppg_filt_full = butter_bandpass(ppg_u, BP_LOW, BP_HIGH, RESAMPLE_HZ)
        ppg_hr = estimate_hr(ppg_filt_full, RESAMPLE_HZ, HR_LOW, HR_HIGH)
        print(f"  PPG reference HR: {ppg_hr:.1f} bpm")

    t_cap_u = np.arange(t_use[0], t_use[-1], 1.0 / RESAMPLE_HZ)
    cap_uniform = {}
    for col in pixel_cols:
        vals = pivot_use[col].values.astype(float)
        fn = interpolate.interp1d(t_use, vals, kind="linear", bounds_error=False, fill_value=(vals[0], vals[-1]))
        cap_uniform[col] = fn(t_cap_u)
    cap_u_df = pd.DataFrame(cap_uniform)

    if has_ppg:
        t_ppg_u, ppg_u = resample_to_uniform(
            ppg_trim["t_s"].values, ppg_trim["green"].values.astype(float), RESAMPLE_HZ
        )
        ppg_filt_full = butter_bandpass(ppg_u, BP_LOW, BP_HIGH, RESAMPLE_HZ)
        t_min = max(t_cap_u[0], t_ppg_u[0])
        t_max = min(t_cap_u[-1], t_ppg_u[-1])
        cap_mask = (t_cap_u >= t_min) & (t_cap_u <= t_max)
        ppg_mask = (t_ppg_u >= t_min) & (t_ppg_u <= t_max)
        t_common = t_cap_u[cap_mask]
        cap_u_df = cap_u_df[cap_mask].reset_index(drop=True)
        ppg_interp = np.interp(t_common, t_ppg_u[ppg_mask], ppg_filt_full[ppg_mask])
    else:
        t_common = t_cap_u
        cap_u_df = cap_u_df.reset_index(drop=True)

    if len(t_common) < 20:
        print("  Too few resampled samples, skipping.")
        return None

    strategies, ac_var, weights = build_strategies(cap_u_df, pixel_cols, n_y, n_x, ppg_interp)
    results: dict[str, dict] = {}
    filtered_signals: dict[str, np.ndarray] = {}

    print()
    for name, sig in strategies.items():
        sig_filt = butter_bandpass(sig - sig.mean(), BP_LOW, BP_HIGH, RESAMPLE_HZ)
        snr = ac_snr(sig_filt, RESAMPLE_HZ, HR_LOW, HR_HIGH)
        hr_est = estimate_hr(sig_filt, RESAMPLE_HZ, HR_LOW, HR_HIGH)
        entry = dict(snr_db=snr, hr_bpm=hr_est)
        if ppg_interp is not None:
            r = abs(pearson_r(sig_filt, ppg_interp))
            entry.update(pearson_r=r, hr_err=abs(hr_est - ppg_hr))
            print(
                f"  {name:<32}  |r|={r:.3f}  "
                f"SNR={snr:+.1f}dB  HR={hr_est:.1f}bpm  ΔHR={entry['hr_err']:.1f}bpm"
            )
        else:
            print(f"  {name:<32}  SNR={snr:+.1f}dB  HR={hr_est:.1f}bpm")
        results[name] = entry
        filtered_signals[name] = sig_filt

    cap_dc = cap_u_df[pixel_cols].mean().values
    pixel_filtered: dict[str, np.ndarray] = {}
    for col in pixel_cols:
        vals = cap_u_df[col].values.astype(float)
        pixel_filtered[col] = butter_bandpass(vals - vals.mean(), BP_LOW, BP_HIGH, RESAMPLE_HZ)

    noise_px_idx = int(np.argmin(np.abs(cap_dc)))
    noise_px_col = pixel_cols[noise_px_idx]
    if weights.sum() > 0:
        top3_px_cols = [pixel_cols[i] for i in np.argsort(weights)[-3:][::-1]]
    else:
        top3_px_cols = [pixel_cols[i] for i in np.argsort(ac_var)[-3:][::-1]]
    out_suffix = region_output_suffix(session_dir, region_idx, num_regions)
    summary_path = session_dir / f"cpg_analysis{out_suffix}.json"
    with summary_path.open("w") as handle:
        json.dump(
            {
                "region_idx": region_idx,
                "region_name": name,
                "staleness": staleness,
                "results": results,
                "window": window_info,
            },
            handle,
            indent=2,
        )
    print(f"\n  Saved summary → {summary_path}")

    return dict(
        session_name=session_name,
        session_dir=session_dir,
        region_idx=region_idx,
        region_name=name,
        num_regions=num_regions,
        t_common=t_common,
        ppg_interp=ppg_interp,
        strategies=strategies,
        filtered_signals=filtered_signals,
        results=results,
        ac_var=ac_var,
        weights=weights,
        ppg_hr=ppg_hr,
        cap_dc=cap_dc,
        pixel_cols=pixel_cols,
        pixel_filtered=pixel_filtered,
        noise_px_col=noise_px_col,
        top3_px_cols=top3_px_cols,
        n_y=n_y,
        n_x=n_x,
        window_info=window_info,
        staleness=staleness,
        has_ppg=has_ppg,
    )


def _to_grid(values: np.ndarray, pixel_cols: list[str], n_y: int, n_x: int) -> np.ndarray:
    arr = np.zeros((n_y, n_x))
    for i, col in enumerate(pixel_cols):
        ry, rx = _parse_pixel_name(col)
        if ry < n_y and rx < n_x:
            arr[ry, rx] = values[i]
    return arr


def _scroll_content_height(n_ts: int) -> float:
    fixed = (
        SCROLL_HEATMAP_H
        + SCROLL_GAP
        + SCROLL_PEARSON_H
        + SCROLL_GAP
        + SCROLL_PSD_H
        + SCROLL_GAP
        + SCROLL_PX_PSD_H
    )
    ts_block = (SCROLL_GAP + n_ts * (SCROLL_TS_H + SCROLL_GAP)) if n_ts else 0.0
    return fixed + ts_block


def _scroll_band_y(offset_from_top: float, band_h: float, scroll_offset: float) -> float:
    content_top = SCROLL_MAIN["top"] + scroll_offset
    return content_top - offset_from_top - band_h


def _scroll_col_x(col_start: int, col_end: int) -> tuple[float, float]:
    width = SCROLL_MAIN["right"] - SCROLL_MAIN["left"]
    x0 = SCROLL_MAIN["left"] + (col_start / 6.0) * width
    return x0, ((col_end - col_start) / 6.0) * width


def _plot_session_header(
    fig: plt.Figure,
    session_data: dict,
    *,
    gs: gridspec.GridSpec | None = None,
    scroll_offset: float = 0.0,
    clip_patch: Rectangle | None = None,
) -> tuple[list[str], list, np.ndarray, float]:
    """Draw heatmaps + Pearson + PSD panels. Returns (strat_names, colors, psd_ylim, content_top_offset)."""
    strategies = session_data["strategies"]
    filtered_signals = session_data["filtered_signals"]
    results = session_data["results"]
    ac_var = session_data["ac_var"]
    cap_dc = session_data["cap_dc"]
    pixel_cols = session_data["pixel_cols"]
    n_y = session_data["n_y"]
    n_x = session_data["n_x"]
    ppg_interp = session_data["ppg_interp"]
    has_ppg = session_data["has_ppg"]

    strat_names = list(strategies.keys())
    n_strats = len(strat_names)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_strats, 1)))
    short = [n.replace(" mean", "").replace(" variance", "") for n in strat_names]

    hm_var = _to_grid(ac_var, pixel_cols, n_y, n_x)
    hm_dc = _to_grid(cap_dc, pixel_cols, n_y, n_x)
    hm_wt = _to_grid(session_data["weights"], pixel_cols, n_y, n_x)
    psd_ylim = (0.0, 1.08)

    scrollable = gs is None

    def add_ax(rect_or_subplot):
        if scrollable:
            ax = fig.add_axes(rect_or_subplot)
            if clip_patch is not None:
                ax.set_clip_path(clip_patch)
            return ax
        return fig.add_subplot(rect_or_subplot)

    def draw_heatmap(ax, data, plot_title, cmap, vmin=None, vmax=None, fmt=".0f"):
        if vmin is None:
            vmin = float(data.min())
        if vmax is None:
            vmax = float(data.max())
        im = ax.imshow(data, cmap=cmap, aspect="equal", vmin=vmin, vmax=vmax, origin="upper")
        if not scrollable:
            mid = (vmin + vmax) / 2
            for ry in range(n_y):
                for rx in range(n_x):
                    val = data[ry, rx]
                    ax.text(
                        rx,
                        ry,
                        f"{val:{fmt}}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if val > mid else "black",
                    )
        ax.set_title(plot_title, fontsize=11 if not scrollable else 10, pad=6)
        ax.set_xlabel("col (X)" if not scrollable else "col", fontsize=9)
        ax.set_ylabel("row (Y)" if not scrollable else "row", fontsize=9)
        ax.tick_params(labelsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04 if scrollable else 0.046)
        return im

    if scrollable:
        y0 = _scroll_band_y(0.0, SCROLL_HEATMAP_H, scroll_offset)
        heatmap_specs = (
            ((0, 2), hm_var, "Pixel AC variance", VARIANCE_CMAP, 0, hm_var.max() + 1e-10, ".0f"),
            (
                (2, 4),
                hm_dc,
                "Pixel DC delta mean",
                "RdBu_r",
                -max(float(np.abs(hm_dc).max()), 1e-6),
                max(float(np.abs(hm_dc).max()), 1e-6),
                ".0f",
            ),
        )
        for cols, data, plot_title, cmap, vmin, vmax, fmt in heatmap_specs:
            x, w = _scroll_col_x(*cols)
            draw_heatmap(add_ax([x, y0, w, SCROLL_HEATMAP_H]), data, plot_title, cmap, vmin, vmax, fmt)
        x, w = _scroll_col_x(4, 6)
        if has_ppg:
            wt_lim = max(float(np.abs(hm_wt).max()), 1e-6)
            draw_heatmap(
                add_ax([x, y0, w, SCROLL_HEATMAP_H]),
                hm_wt,
                "|Pearson r| with PPG",
                "RdBu_r",
                -wt_lim,
                wt_lim,
                ".2f",
            )
        else:
            ax = add_ax([x, y0, w, SCROLL_HEATMAP_H])
            ax.axis("off")
            ax.text(0.5, 0.5, "No PPG file\n(ppg_green.csv)", ha="center", va="center", fontsize=10)

        off1 = SCROLL_HEATMAP_H + SCROLL_GAP
        y1 = _scroll_band_y(off1, SCROLL_PEARSON_H, scroll_offset)
        x, w = _scroll_col_x(0, 6)
        ax_xcorr = add_ax([x, y1, w, SCROLL_PEARSON_H])
    else:
        draw_heatmap(
            add_ax(gs[0:2, 0:2]),
            hm_var,
            "Pixel AC variance",
            VARIANCE_CMAP,
            0,
            hm_var.max() + 1e-10,
        )
        dc_lim = max(float(np.abs(hm_dc).max()), 1e-6)
        draw_heatmap(add_ax(gs[0:2, 2:4]), hm_dc, "Pixel DC delta mean", "RdBu_r", -dc_lim, dc_lim)
        if has_ppg:
            wt_lim = max(float(np.abs(hm_wt).max()), 1e-6)
            draw_heatmap(
                add_ax(gs[0:2, 4:6]),
                hm_wt,
                "|Pearson r| with PPG",
                "RdBu_r",
                -wt_lim,
                wt_lim,
                ".2f",
            )
        else:
            ax = add_ax(gs[0:2, 4:6])
            ax.axis("off")
            ax.text(0.5, 0.5, "No PPG file\n(ppg_green.csv)", ha="center", va="center", fontsize=12)
        ax_xcorr = add_ax(gs[2, :])

    if has_ppg and all("pearson_r" in results[n] for n in strat_names):
        pearson_vals = [results[n]["pearson_r"] for n in strat_names]
        ax_xcorr.barh(short, pearson_vals, color=colors, height=0.62 if scrollable else 0.8)
        ax_xcorr.set_xlabel("|Pearson r| with PPG", fontsize=9 if scrollable else None)
        ax_xcorr.set_title("Pearson correlation with PPG", fontsize=10 if scrollable else None, pad=8)
        ax_xcorr.set_xlim(0, 1)
        if scrollable:
            ax_xcorr.tick_params(labelsize=8, pad=3)
    else:
        ax_xcorr.axis("off")
        ax_xcorr.text(
            0.5,
            0.5,
            "No PPG — Pearson correlation not available",
            ha="center",
            va="center",
            fontsize=12 if not scrollable else 10,
            transform=ax_xcorr.transAxes,
        )

    cap_psds = {name: zscore_psd_curve(filtered_signals[name], RESAMPLE_HZ) for name in strat_names}
    cap_peak_scale = max(p.max() for _, p in cap_psds.values()) + 1e-30
    strongest_cap = max(cap_psds, key=lambda name: cap_psds[name][1].max())

    if scrollable:
        off2 = SCROLL_HEATMAP_H + SCROLL_GAP + SCROLL_PEARSON_H + SCROLL_GAP
        y2 = _scroll_band_y(off2, SCROLL_PSD_H, scroll_offset)
        x, w = _scroll_col_x(0, 6)
        ax_psd = add_ax([x, y2, w, SCROLL_PSD_H])
    else:
        ax_psd = add_ax(gs[3, :])

    if has_ppg and ppg_interp is not None:
        f_ppg, p_ppg = zscore_psd_curve(ppg_interp, RESAMPLE_HZ)
        p_ppg_norm = p_ppg / (p_ppg.max() + 1e-30)
        ax_psd.plot(f_ppg * 60, p_ppg_norm, "k-", lw=2, label="PPG", zorder=10)
    for i, name in enumerate(strat_names):
        f_s, p_s = cap_psds[name]
        p_s_norm = p_s / cap_peak_scale
        ax_psd.plot(f_s * 60, p_s_norm, color=colors[i], alpha=0.75, lw=1, label=short[i])
    ax_psd.set_xlabel("Frequency (bpm)", fontsize=9 if scrollable else None)
    ax_psd.set_ylabel("Norm. PSD (PPG & strongest cap peak = 1)", fontsize=9 if scrollable else None)
    ax_psd.set_ylim(psd_ylim)
    ax_psd.axhline(1.0, color="0.65", lw=0.8, ls=":", zorder=1)
    ax_psd.legend(fontsize=8, ncol=3 if not scrollable else 2, loc="upper right")
    ax_psd.set_title(
        f"PSD comparison (cap ÷ {strongest_cap} peak; PPG ÷ own peak)",
        fontsize=10 if scrollable else None,
        pad=10 if scrollable else None,
    )
    ax_psd.autoscale(False)
    if scrollable:
        ax_psd.tick_params(labelsize=8)

    pixel_filtered = session_data.get("pixel_filtered", {})
    noise_px_col = session_data.get("noise_px_col")
    top3_px_cols = session_data.get("top3_px_cols", [])
    weights = session_data["weights"]

    if scrollable:
        off3 = SCROLL_HEATMAP_H + SCROLL_GAP + SCROLL_PEARSON_H + SCROLL_GAP + SCROLL_PSD_H + SCROLL_GAP
        y3 = _scroll_band_y(off3, SCROLL_PX_PSD_H, scroll_offset)
        x, w = _scroll_col_x(0, 6)
        ax_px_psd = add_ax([x, y3, w, SCROLL_PX_PSD_H])
    else:
        ax_px_psd = add_ax(gs[4, :])

    if has_ppg and ppg_interp is not None and pixel_filtered and noise_px_col:
        ax_ppg_psd = ax_px_psd.twinx()
        f_ppg, p_ppg = dominant_psd_curve(ppg_interp, RESAMPLE_HZ)
        ax_ppg_psd.plot(f_ppg * 60, p_ppg, "k-", lw=2, label="PPG", zorder=10)

        f_noise, p_noise = dominant_psd_curve(pixel_filtered[noise_px_col], RESAMPLE_HZ)
        ax_px_psd.plot(
            f_noise * 60,
            p_noise,
            color="0.45",
            lw=1.5,
            ls="--",
            label=f"Noise {noise_px_col}",
        )

        px_colors = ["C0", "C1", "C2"]
        for j, col in enumerate(top3_px_cols):
            f_px, p_px = dominant_psd_curve(pixel_filtered[col], RESAMPLE_HZ)
            px_idx = pixel_cols.index(col)
            ax_px_psd.plot(
                f_px * 60,
                p_px,
                color=px_colors[j % len(px_colors)],
                lw=1.5,
                label=f"{col} |r|={weights[px_idx]:.2f}",
            )
        ax_px_psd.set_xlabel("Frequency (bpm)", fontsize=9 if scrollable else None)
        ax_px_psd.set_ylabel("Cap PSD (noise + top-3)", fontsize=9 if scrollable else None)
        ax_ppg_psd.set_ylabel("PPG PSD", color="k")
        ax_ppg_psd.tick_params(axis="y", labelcolor="k")
        cap_lines, cap_labels = ax_px_psd.get_legend_handles_labels()
        ppg_lines, ppg_labels = ax_ppg_psd.get_legend_handles_labels()
        ax_px_psd.legend(
            cap_lines + ppg_lines,
            cap_labels + ppg_labels,
            fontsize=8,
            ncol=3,
            loc="upper right",
        )
        ax_px_psd.set_title("PSD — PPG vs noise vs top-3 PPG-correlated pixels", fontsize=10 if scrollable else None)
        if scrollable:
            ax_px_psd.tick_params(labelsize=8)
    else:
        ax_px_psd.axis("off")
        ax_px_psd.text(
            0.5,
            0.5,
            "No PPG — pixel PSD comparison not available",
            ha="center",
            va="center",
            fontsize=12 if not scrollable else 10,
            transform=ax_px_psd.transAxes,
        )

    header_offset = (
        SCROLL_HEATMAP_H
        + SCROLL_GAP
        + SCROLL_PEARSON_H
        + SCROLL_GAP
        + SCROLL_PSD_H
        + SCROLL_GAP
        + SCROLL_PX_PSD_H
        + SCROLL_GAP
    )
    return strat_names, colors, psd_ylim, header_offset


def plot_session_scrollable(
    fig: plt.Figure,
    session_data: dict,
    *,
    scroll_offset: float = 0.0,
    include_timeseries: bool = True,
    clip_patch: Rectangle | None = None,
) -> float:
    """Draw strategy plots into a fixed viewport; scroll vertically via ``scroll_offset``."""
    t_common = session_data["t_common"]
    filtered_signals = session_data["filtered_signals"]
    results = session_data["results"]
    ppg_interp = session_data["ppg_interp"]
    ppg_hr = session_data["ppg_hr"]
    has_ppg = session_data["has_ppg"]
    staleness = session_data["staleness"]
    session_name = session_data["session_name"]
    region_idx = session_data.get("region_idx", 0)
    num_regions = session_data.get("num_regions", 1)
    region_name = session_data.get("region_name", region_label(session_data["session_dir"], region_idx))

    ts_names = list(session_data["strategies"].keys()) if include_timeseries else []
    n_ts = len(ts_names)
    t_plot = t_common - t_common[0]
    activity_labels = load_activity_labels(resolve_labels_path(session_data["session_dir"]))

    region_tag = f"  ·  {region_name}" if num_regions > 1 else ""
    title = f"{session_name}{region_tag} — CPG strategies  [stale {staleness['stale_steps_pct']:.0f}%]"
    if has_ppg and ppg_hr is not None:
        title += f"  PPG HR {ppg_hr:.1f} bpm"
    fig.suptitle(title, fontsize=13, y=0.97)

    strat_names, colors, _, header_offset = _plot_session_header(
        fig,
        session_data,
        scroll_offset=scroll_offset,
        clip_patch=clip_patch,
    )

    for i, name in enumerate(ts_names):
        band_off = header_offset + i * (SCROLL_TS_H + SCROLL_GAP)
        y_ts = _scroll_band_y(band_off, SCROLL_TS_H, scroll_offset)
        x, w = _scroll_col_x(0, 6)
        ax = fig.add_axes([x, y_ts, w, SCROLL_TS_H])
        if clip_patch is not None:
            ax.set_clip_path(clip_patch)
        cap_norm = zscore(filtered_signals[name])
        label = f"Cap  SNR={results[name]['snr_db']:.1f}dB  HR={results[name]['hr_bpm']:.1f}bpm"
        if has_ppg and "pearson_r" in results[name]:
            label = f"Cap  |r|={results[name]['pearson_r']:.3f}  " + label
        if has_ppg and ppg_interp is not None:
            ax.plot(t_plot, zscore(ppg_interp), color="green", alpha=0.55, lw=0.8, label="PPG")
        ax.plot(t_plot, cap_norm, color=colors[i], lw=0.85, label=label)
        ax.set_title(name, fontsize=9, pad=8)
        ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
        ax.tick_params(labelsize=8)
        ax.margins(x=0.01, y=0.15)
        if activity_labels:
            draw_label_markers(ax, activity_labels, x_offset=-float(t_common[0]))
        if i == n_ts - 1:
            ax.set_xlabel("Time (s)", fontsize=9)

    return _scroll_content_height(n_ts)


def plot_session(
    session_data: dict,
    out_path: Path | None = None,
    *,
    include_timeseries: bool = True,
    show: bool = False,
    fig: plt.Figure | None = None,
) -> plt.Figure | None:
    t_common = session_data["t_common"]
    filtered_signals = session_data["filtered_signals"]
    results = session_data["results"]
    ppg_interp = session_data["ppg_interp"]
    ppg_hr = session_data["ppg_hr"]
    has_ppg = session_data["has_ppg"]
    staleness = session_data["staleness"]
    session_name = session_data["session_name"]
    region_idx = session_data.get("region_idx", 0)
    num_regions = session_data.get("num_regions", 1)
    region_name = session_data.get(
        "region_name",
        region_label(session_data["session_dir"], region_idx),
    )

    strat_names = list(session_data["strategies"].keys())
    n_strats = len(strat_names)
    t_plot = t_common - t_common[0]
    activity_labels = load_activity_labels(resolve_labels_path(session_data["session_dir"]))
    n_header = len(HEADER_ROW_RATIOS)
    if include_timeseries:
        height_ratios = HEADER_ROW_RATIOS + [TS_ROW_RATIO] * n_strats
        n_rows = n_header + n_strats
        fig_h = 11 + 2.4 * n_strats
    else:
        height_ratios = HEADER_ROW_RATIOS
        n_rows = n_header
        fig_h = 12

    region_tag = f"  ·  {region_name}" if num_regions > 1 else ""
    title = f"{session_name}{region_tag} — CPG strategies  [stale {staleness['stale_steps_pct']:.0f}%]"
    if has_ppg and ppg_hr is not None:
        title += f"  PPG HR {ppg_hr:.1f} bpm"

    owns_fig = fig is None
    if owns_fig:
        fig = plt.figure(figsize=(22, fig_h))
    fig.set_size_inches(22, fig_h)
    fig.suptitle(title, fontsize=16, y=0.998)
    gs = gridspec.GridSpec(
        n_rows,
        6,
        figure=fig,
        height_ratios=height_ratios,
        left=0.06,
        right=0.98,
        top=0.96,
        bottom=0.04,
        hspace=0.55,
        wspace=0.6,
    )

    _, colors, psd_ylim, _ = _plot_session_header(fig, session_data, gs=gs)

    if include_timeseries:
        ppg_norm = zscore(ppg_interp) if has_ppg and ppg_interp is not None else None
        for i, name in enumerate(strat_names):
            ax = fig.add_subplot(gs[n_header + i, :])
            cap_norm = zscore(filtered_signals[name])
            if ppg_norm is not None:
                ax.plot(t_plot, ppg_norm, color="green", alpha=0.55, lw=0.8, label="PPG")
            label = f"Cap  SNR={results[name]['snr_db']:.1f}dB  HR={results[name]['hr_bpm']:.1f}bpm"
            if has_ppg and "pearson_r" in results[name]:
                label = f"Cap  |r|={results[name]['pearson_r']:.3f}  " + label
            ax.plot(t_plot, cap_norm, color=colors[i], lw=0.8, label=label)
            ax.set_title(name, fontsize=10)
            ax.legend(fontsize=8, loc="upper right")
            if activity_labels:
                draw_label_markers(ax, activity_labels, x_offset=-float(t_common[0]))
            if i == n_strats - 1:
                ax.set_xlabel("Time (s)")

    for ax in fig.axes:
        if ax.get_ylabel() and "Norm. PSD" in ax.get_ylabel():
            ax.set_ylim(psd_ylim)
            ax.autoscale(False)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"  Saved plot → {out_path}")
    if show:
        fig.canvas.draw_idle()
        return fig
    if owns_fig:
        plt.close(fig)
    return None


def interactive_main(
    data_dir: Path,
    *,
    start_folder: Path,
    start_region: int = 0,
    use_npz_frames: bool = True,
    include_timeseries: bool = True,
) -> int:
    print(
        """
maXTouch CPG strategies (interactive)
  Up/Down     previous / next session
  [ / ]       previous / next saved region
  h           help
  Scroll wheel over the main panel (or drag the bar on the right) to see all time series
  Page Up / Page Down also scroll the main panel
  Click a session or region in the left sidebar
  Save plot   export cpg_strategy_comparison.png to the session folder
"""
    )

    available_folders = find_viewable_sessions(data_dir)
    if not available_folders:
        print(f"No viewable sessions in {data_dir}", file=sys.stderr)
        return 1

    if start_folder.resolve() not in {p.resolve() for p in available_folders}:
        available_folders.append(start_folder)
        available_folders.sort(key=lambda p: natural_sort_key(p.name))

    current_file_idx = [available_folders.index(start_folder)]
    current_region_idx = [start_region]
    file_scroll_offset = [0]
    file_rects: list[Rectangle] = []
    file_text_items = []
    region_rects: list[Rectangle] = []
    region_text_items = []
    chrome_axes: set[plt.Axes] = set()
    current_data: list[dict | None] = [None]
    content_scroll = [0.0]
    content_height = [0.0]
    scroll_track_rect = [None]
    scroll_thumb_rect = [None]
    scroll_dragging = [False]
    scroll_drag_offset = [0.0]

    viewport_h = SCROLL_MAIN["top"] - SCROLL_MAIN["bottom"]

    def max_content_scroll() -> float:
        return max(0.0, content_height[0] - viewport_h)

    def main_panel_contains(event) -> bool:
        if event.x is None or event.y is None:
            return False
        fx, fy = fig.transFigure.inverted().transform((event.x, event.y))
        return (
            SCROLL_MAIN["left"] <= fx <= SCROLL_MAIN["right"]
            and SCROLL_MAIN["bottom"] <= fy <= SCROLL_MAIN["top"]
        )

    def clear_main_content() -> None:
        for ax in fig.axes[:]:
            if ax not in chrome_axes:
                ax.remove()

    def update_main_scrollbar() -> None:
        if scroll_track_rect[0] is not None:
            scroll_track_rect[0].remove()
            scroll_track_rect[0] = None
        if scroll_thumb_rect[0] is not None:
            scroll_thumb_rect[0].remove()
            scroll_thumb_rect[0] = None

        max_scroll = max_content_scroll()
        if max_scroll <= 1e-6:
            main_scroll_ax.set_visible(False)
            return
        main_scroll_ax.set_visible(True)
        track_h = 1.0
        thumb_h = max(viewport_h / max(content_height[0], viewport_h) * track_h, 0.08)
        thumb_y = (1.0 - thumb_h) * (1.0 - content_scroll[0] / max_scroll) if max_scroll else 0.0
        scroll_track_rect[0] = Rectangle(
            (0.08, 0.0),
            0.84,
            track_h,
            facecolor="#e0e0e0",
            edgecolor="none",
            transform=main_scroll_ax.transAxes,
            zorder=1,
        )
        scroll_thumb_rect[0] = Rectangle(
            (0.08, thumb_y),
            0.84,
            thumb_h,
            facecolor="#888888",
            edgecolor="none",
            transform=main_scroll_ax.transAxes,
            zorder=2,
            picker=5,
        )
        main_scroll_ax.add_patch(scroll_track_rect[0])
        main_scroll_ax.add_patch(scroll_thumb_rect[0])

    def redraw_strategies(data: dict, *, reset_scroll: bool = False) -> None:
        if reset_scroll:
            content_scroll[0] = 0.0
        current_data[0] = data
        clear_main_content()
        content_height[0] = plot_session_scrollable(
            fig,
            data,
            scroll_offset=content_scroll[0],
            include_timeseries=include_timeseries,
            clip_patch=viewport_clip,
        )
        content_scroll[0] = min(content_scroll[0], max_content_scroll())
        if content_scroll[0] > 0:
            clear_main_content()
            content_height[0] = plot_session_scrollable(
                fig,
                data,
                scroll_offset=content_scroll[0],
                include_timeseries=include_timeseries,
                clip_patch=viewport_clip,
            )
        for ax in chrome_axes:
            ax.set_zorder(200)
        update_main_scrollbar()
        fig.canvas.draw_idle()

    def set_content_scroll(offset: float) -> None:
        offset = max(0.0, min(offset, max_content_scroll()))
        if abs(offset - content_scroll[0]) < 1e-6:
            return
        content_scroll[0] = offset
        if current_data[0] is not None:
            redraw_strategies(current_data[0])

    def scroll_from_thumb_y(thumb_y: float, thumb_h: float) -> float:
        max_scroll = max_content_scroll()
        if max_scroll <= 0:
            return 0.0
        track_h = 1.0 - thumb_h
        if track_h <= 1e-6:
            return 0.0
        frac = 1.0 - (thumb_y / track_h)
        return max(0.0, min(max_scroll, frac * max_scroll))

    def update_file_list_display() -> None:
        for rect in file_rects:
            rect.remove()
        for txt in file_text_items:
            txt.remove()
        file_rects.clear()
        file_text_items.clear()

        start_idx = file_scroll_offset[0]
        end_idx = min(start_idx + MAX_VISIBLE_SESSIONS, len(available_folders))
        for idx, i in enumerate(range(start_idx, end_idx)):
            folder = available_folders[i]
            y_pos = MAX_VISIBLE_SESSIONS - idx - 0.5
            selected = i == current_file_idx[0]
            rect = Rectangle(
                (0, y_pos - 0.45),
                0.98,
                0.9,
                facecolor="lightblue" if selected else "white",
                edgecolor="gray",
                linewidth=0.5,
                zorder=2,
                picker=5,
                gid=str(i),
            )
            session_list_ax.add_patch(rect)
            file_rects.append(rect)
            label = folder.name
            if len(label) > 24:
                label = label[:21] + "..."
            txt = session_list_ax.text(
                0.03,
                y_pos,
                label,
                fontsize=7,
                va="center",
                zorder=3,
                fontweight="bold" if selected else "normal",
            )
            file_text_items.append(txt)
        fig.canvas.draw_idle()

    def update_region_list_display() -> None:
        for rect in region_rects:
            rect.remove()
        for txt in region_text_items:
            txt.remove()
        region_rects.clear()
        region_text_items.clear()

        session_dir = available_folders[current_file_idx[0]]
        num_regions = region_count(session_dir)
        for idx in range(num_regions):
            y_pos = MAX_VISIBLE_REGIONS - idx - 0.5
            selected = idx == current_region_idx[0]
            name = region_label(session_dir, idx)
            rect = Rectangle(
                (0, y_pos - 0.45),
                0.98,
                0.9,
                facecolor="#c8e6c9" if selected else "white",
                edgecolor="gray",
                linewidth=0.5,
                zorder=2,
                picker=5,
                gid=str(idx),
            )
            region_list_ax.add_patch(rect)
            region_rects.append(rect)
            txt = region_list_ax.text(
                0.08,
                y_pos,
                name,
                fontsize=8,
                va="center",
                zorder=3,
                fontweight="bold" if selected else "normal",
            )
            region_text_items.append(txt)
        if num_regions == 0:
            region_list_ax.text(0.05, 0.5, "No regions", fontsize=8, va="center", transform=region_list_ax.transAxes)
        region_list_ax.set_ylim(0, max(MAX_VISIBLE_REGIONS, num_regions))
        fig.canvas.draw_idle()

    def load_selection(file_idx: int, *, region_idx: int | None = None) -> None:
        if file_idx < 0 or file_idx >= len(available_folders):
            return
        session_dir = available_folders[file_idx]
        if region_idx is None:
            region_idx = current_region_idx[0]
        num_regions = region_count(session_dir)
        if num_regions == 0:
            return
        region_idx = max(0, min(region_idx, num_regions - 1))

        data = process_session(
            session_dir,
            region_idx=region_idx,
            num_regions=num_regions,
            use_npz_frames=use_npz_frames,
        )
        if data is None:
            return

        current_file_idx[0] = file_idx
        current_region_idx[0] = region_idx

        if file_idx < file_scroll_offset[0]:
            file_scroll_offset[0] = file_idx
        elif file_idx >= file_scroll_offset[0] + MAX_VISIBLE_SESSIONS:
            file_scroll_offset[0] = max(0, file_idx - MAX_VISIBLE_SESSIONS + 1)

        redraw_strategies(data, reset_scroll=True)
        fig.canvas.manager.set_window_title(f"{session_dir.name} · {data['region_name']} — CPG strategies")

        update_file_list_display()
        update_region_list_display()
        print(f"Loaded {session_dir.name} · {data['region_name']}")

    def load_region_by_index(region_idx: int) -> None:
        load_selection(current_file_idx[0], region_idx=region_idx)

    def change_region(delta: int) -> None:
        session_dir = available_folders[current_file_idx[0]]
        num_regions = region_count(session_dir)
        if num_regions <= 1:
            return
        new_region = (current_region_idx[0] + delta) % num_regions
        load_selection(current_file_idx[0], region_idx=new_region)

    num_regions = region_count(start_folder)
    if start_region < 0 or start_region >= num_regions:
        print(f"Region {start_region} out of range (num_regions={num_regions})", file=sys.stderr)
        return 1

    fig = plt.figure(figsize=(20, 11))
    fig.canvas.manager.set_window_title("maXTouch CPG strategies")

    viewport_clip = Rectangle(
        (SCROLL_MAIN["left"], SCROLL_MAIN["bottom"]),
        SCROLL_MAIN["right"] - SCROLL_MAIN["left"],
        viewport_h,
        transform=fig.transFigure,
        visible=False,
    )

    session_list_ax = register_sidebar_axis(fig.add_axes([0.01, 0.54, 0.15, 0.40]), facecolor="#eef4fb")
    chrome_axes.add(session_list_ax)
    session_list_ax.set_title("Sessions", fontsize=9, pad=5)
    session_list_ax.set_xlim(0, 1)
    session_list_ax.set_ylim(0, MAX_VISIBLE_SESSIONS)
    session_list_ax.axis("off")

    region_list_ax = register_sidebar_axis(fig.add_axes([0.01, 0.12, 0.15, 0.38]), facecolor="#eef8ee")
    chrome_axes.add(region_list_ax)
    region_list_ax.set_title("Regions", fontsize=9, pad=5)
    region_list_ax.set_xlim(0, 1)
    region_list_ax.set_ylim(0, MAX_VISIBLE_REGIONS)
    region_list_ax.axis("off")

    save_btn_ax = fig.add_axes([0.01, 0.04, 0.15, 0.06])
    chrome_axes.add(save_btn_ax)

    def on_save_plot(_event) -> None:
        data = current_data[0]
        if data is None:
            return
        out_path = strategy_comparison_plot_path(data)
        plot_session(data, out_path, include_timeseries=include_timeseries)
        save_btn.label.set_text("Saved")
        fig.canvas.draw_idle()

    save_btn = Button(save_btn_ax, "Save plot")
    save_btn.on_clicked(on_save_plot)

    main_scroll_ax = register_sidebar_axis(
        fig.add_axes([0.962, SCROLL_MAIN["bottom"], 0.014, viewport_h]),
        facecolor="#f5f5f5",
    )
    chrome_axes.add(main_scroll_ax)
    main_scroll_ax.set_xlim(0, 1)
    main_scroll_ax.set_ylim(0, 1)
    main_scroll_ax.axis("off")
    main_scroll_ax.set_visible(False)

    def on_sidebar_click(event) -> None:
        if event.button != 1 or event.x is None or event.y is None:
            return
        if event_in_axes(fig, session_list_ax, event):
            y = event_ydata(session_list_ax, event)
            if y is None:
                return
            file_idx = hit_list_row(
                y,
                max_visible=MAX_VISIBLE_SESSIONS,
                scroll_offset=file_scroll_offset[0],
                total=len(available_folders),
            )
            if file_idx is not None and file_idx != current_file_idx[0]:
                load_selection(file_idx, region_idx=0)
            return
        if event_in_axes(fig, region_list_ax, event):
            y = event_ydata(region_list_ax, event)
            if y is None:
                return
            session_dir = available_folders[current_file_idx[0]]
            num_regions = region_count(session_dir)
            region_idx = hit_region_row(y, num_regions, max_visible=MAX_VISIBLE_REGIONS)
            if region_idx is not None and region_idx != current_region_idx[0]:
                load_region_by_index(region_idx)

    def on_sidebar_scroll(event) -> None:
        if event_in_axes(fig, session_list_ax, event):
            step = 3
            if event.button == "up":
                file_scroll_offset[0] = max(0, file_scroll_offset[0] - step)
            elif event.button == "down":
                max_offset = max(0, len(available_folders) - MAX_VISIBLE_SESSIONS)
                file_scroll_offset[0] = min(max_offset, file_scroll_offset[0] + step)
            else:
                return
            update_file_list_display()
            return
        if main_panel_contains(event) or event_in_axes(fig, main_scroll_ax, event):
            wheel = getattr(event, "step", None) or 1
            step = SCROLL_STEP * wheel
            if event.button == "up":
                set_content_scroll(content_scroll[0] - step)
            elif event.button == "down":
                set_content_scroll(content_scroll[0] + step)

    def on_press(event) -> None:
        if event.button != 1 or event.x is None or event.y is None:
            return
        if scroll_thumb_rect[0] is not None and event_in_axes(fig, main_scroll_ax, event):
            y = event_ydata(main_scroll_ax, event)
            if y is None:
                return
            thumb_y = scroll_thumb_rect[0].get_y()
            thumb_h = scroll_thumb_rect[0].get_height()
            if thumb_y <= y <= thumb_y + thumb_h:
                scroll_dragging[0] = True
                scroll_drag_offset[0] = y - thumb_y
                return
            max_scroll = max_content_scroll()
            if max_scroll <= 0:
                return
            track_h = 1.0
            thumb_h = max(viewport_h / max(content_height[0], viewport_h) * track_h, 0.08)
            thumb_y = max(0.0, min(track_h - thumb_h, y - thumb_h / 2))
            set_content_scroll(scroll_from_thumb_y(thumb_y, thumb_h))

    def on_motion(event) -> None:
        if not scroll_dragging[0] or event.y is None:
            return
        y = event_ydata(main_scroll_ax, event)
        if y is None:
            return
        track_h = 1.0
        thumb_h = scroll_thumb_rect[0].get_height() if scroll_thumb_rect[0] is not None else 0.08
        thumb_y = max(0.0, min(track_h - thumb_h, y - scroll_drag_offset[0]))
        set_content_scroll(scroll_from_thumb_y(thumb_y, thumb_h))

    def on_release(event) -> None:
        scroll_dragging[0] = False

    def on_key(event) -> None:
        if event.key == "up":
            load_selection(current_file_idx[0] - 1, region_idx=0)
        elif event.key == "down":
            load_selection(current_file_idx[0] + 1, region_idx=0)
        elif event.key == "[":
            change_region(-1)
        elif event.key == "]":
            change_region(1)
        elif event.key in ("pageup", "pagedown"):
            step = viewport_h * 0.85
            if event.key == "pageup":
                set_content_scroll(content_scroll[0] - step)
            else:
                set_content_scroll(content_scroll[0] + step)

    fig.canvas.mpl_connect("button_press_event", on_sidebar_click)
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("scroll_event", on_sidebar_scroll)
    fig.canvas.mpl_connect("key_press_event", on_key)

    update_file_list_display()
    update_region_list_display()
    load_selection(current_file_idx[0], region_idx=start_region)
    plt.show()
    return 0


def batch_main(args: argparse.Namespace) -> int:
    session_dirs = [Path(p) for p in SESSIONS.values()] if not args.session else args.session
    exit_code = 0
    suffix = "_summary" if args.summary_only else ""

    for session_dir in session_dirs:
        session_dir = session_dir.resolve()
        if not (session_dir / CAP_FILENAME).is_file():
            print(f"Missing {session_dir / CAP_FILENAME}", file=sys.stderr)
            exit_code = 1
            continue
        if not regions_npz_path(session_dir).is_file():
            print(f"Missing {regions_npz_path(session_dir)}", file=sys.stderr)
            exit_code = 1
            continue

        num_regions = region_count(session_dir)
        if args.region is not None:
            if args.region < 0 or args.region >= num_regions:
                print(
                    f"Region {args.region} out of range for {session_dir.name} "
                    f"(num_regions={num_regions})",
                    file=sys.stderr,
                )
                exit_code = 1
                continue
            region_indices = [args.region]
        else:
            region_indices = list(range(num_regions))

        if num_regions > 1:
            print(f"\n  {session_dir.name}: {num_regions} saved region(s) — analysing separately")

        for region_idx in region_indices:
            data = process_session(
                session_dir,
                region_idx=region_idx,
                num_regions=num_regions,
                use_npz_frames=not args.full_session,
            )
            if data is None:
                exit_code = 1
                continue

            out_suffix = region_output_suffix(session_dir, region_idx, num_regions)
            plot_session(
                data,
                session_dir / f"cpg_strategy_comparison{out_suffix}{suffix}.png",
                include_timeseries=not args.summary_only,
            )

    if len(session_dirs) > 1:
        print(f"\nProcessed {len(session_dirs)} session(s).")
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract CPG from maXTouch cap_log region NPZ + capacitance CSV.")
    parser.add_argument(
        "--session",
        type=Path,
        action="append",
        help="Session folder (repeatable). Interactive: start here. With --batch: export PNGs.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory to scan for session folders (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--region",
        type=int,
        default=0,
        help="Saved region index (default: 0). Batch: analyse only this region. Interactive: start here.",
    )
    parser.add_argument(
        "--full-session",
        action="store_true",
        help="Analyse the full capacitance CSV instead of frame_indices from the regions NPZ.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Omit per-strategy time-series rows in the output PNG.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch-export PNGs for default sessions (or --session folders).",
    )
    args = parser.parse_args()

    if args.batch:
        return batch_main(args)

    available = find_viewable_sessions(args.data_dir)
    if not available:
        print(f"No viewable sessions in {args.data_dir}", file=sys.stderr)
        return 1

    start_folder = available[0]
    if args.session:
        start_folder = args.session[0].resolve()
        if not regions_npz_path(start_folder).is_file():
            print(f"Missing {regions_npz_path(start_folder)}", file=sys.stderr)
            return 1
        known = {p.resolve() for p in available}
        if start_folder not in known:
            available.append(start_folder)
            available.sort(key=lambda p: natural_sort_key(p.name))
    elif len(available) > 1:
        print(f"Found {len(available)} sessions. Starting with: {start_folder.name}")

    num_regions = region_count(start_folder)
    if args.region < 0 or args.region >= num_regions:
        parser.error(f"Region {args.region} out of range for {start_folder.name} (num_regions={num_regions})")

    return interactive_main(
        args.data_dir,
        start_folder=start_folder,
        start_region=args.region,
        use_npz_frames=not args.full_session,
        include_timeseries=not args.summary_only,
    )


if __name__ == "__main__":
    raise SystemExit(main())
