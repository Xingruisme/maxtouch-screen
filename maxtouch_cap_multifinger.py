#!/usr/bin/env python3
"""Log maXTouch T37 capacitance (single-page region) without PPG.

Capacitance is read over USB HID from the maXTouch T37 diagnostic object. By default
the script streams PAGE0_WIDTH columns × (PAGE0_ROW_FRACTION × matrix Y rows)
from the origin (default 6 columns × one-third of row length).

Live viewing shows a capacitance heatmap, a live mean-Δ line plot per auto-detected
6×6 finger patch (detection runs once ~10 s after start), and CPG strategy windows
(All-N mean, Top-8 variance, Best px — no PPG-correlation channel). Optional
--pixel-stream opens extra windows with every pixel's live Δ for each detected finger.

Recording runs until you close the viewer window (or Ctrl+C with --no-viewer).
Activity labels are saved to capacitance_labels.csv in the session folder.
"""

from __future__ import annotations

import json
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from activity_labels import (
    labels_path_for_session,
    load_activity_labels,
)
from maxtouch_hid import MaxTouchHid
from maxtouch_ppg_multifinger import (
    CAP_CSV_NAME,
    ENABLE_PIXEL_STREAM,
    FINGER_DETECT_DELAY_S,
    FINGER_PATCH_SIZE,
    OUT_DIR,
    PAGE0_ROW_FRACTION,
    PAGE0_WIDTH,
    REGION_ORIGIN_X,
    REGION_ORIGIN_Y,
    RUN_VIEWER,
    VIEW_COLOR_MAX,
    VIEW_INVERT_DELTA,
    VIEW_REFRESH_MS,
    ActivityLabelPanel,
    CapHeatmapTailer,
    LiveCsvViewer,
    Region,
    Settings,
    finalize_session_metadata,
    make_run_dir,
    record_capacitance,
    resolve_region,
)

# Edit these defaults before running.
RUN_NAME = "maxtouch_cap"


@dataclass(frozen=True)
class CapSettings:
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
    finger_detect_delay_s: float
    finger_patch_size: int
    enable_pixel_stream: bool


def as_recording_settings(args: CapSettings) -> Settings:
    """Adapt cap-only settings for shared recording/viewer helpers."""
    return Settings(
        name=args.name,
        out_dir=args.out_dir,
        region_origin_x=args.region_origin_x,
        region_origin_y=args.region_origin_y,
        page0_width=args.page0_width,
        page0_row_fraction=args.page0_row_fraction,
        fixed_region=args.fixed_region,
        run_viewer=args.run_viewer,
        view_invert_delta=args.view_invert_delta,
        view_color_max=args.view_color_max,
        view_refresh_ms=args.view_refresh_ms,
        ppg_enabled=False,
        ppg_port="",
        ppg_baud=115_200,
        ppg_auto_kill_port_owner=True,
        ppg_pick_port=False,
        finger_detect_delay_s=args.finger_detect_delay_s,
        finger_patch_size=args.finger_patch_size,
        enable_pixel_stream=args.enable_pixel_stream,
    )


def write_metadata(
    run_dir: Path,
    args: CapSettings,
    mxt: MaxTouchHid,
    region: Region,
    pages: list[int],
    page_size: int,
    estimated_hz: float,
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
                "Capacitance-only session (no PPG).",
                "Default region: PAGE0_WIDTH columns × PAGE0_ROW_FRACTION of matrix Y rows.",
                "T37 pages are 0-indexed in firmware; wider regions may span multiple pages.",
            ],
            "page0_width": args.page0_width,
            "page0_row_fraction": args.page0_row_fraction,
            "matrix_y_rows": mxt.id_info.matrix_y,
        },
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def build_settings(argv: list[str] | None = None) -> CapSettings:
    import argparse

    parser = argparse.ArgumentParser(
        description="Record maXTouch page-0 capacitance region (no PPG)"
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--name", default=RUN_NAME)
    parser.add_argument("--region-origin-x", type=int, default=REGION_ORIGIN_X)
    parser.add_argument("--region-origin-y", type=int, default=REGION_ORIGIN_Y)
    parser.add_argument(
        "--page0-width",
        type=int,
        default=PAGE0_WIDTH,
        help="mutual-sense X columns from origin (default: 6)",
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

    return CapSettings(
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
        finger_detect_delay_s=ns.finger_detect_delay,
        finger_patch_size=ns.finger_patch_size,
        enable_pixel_stream=ns.pixel_stream,
    )


def main(argv: list[str] | None = None) -> int:
    cap_args = build_settings(argv)
    rec_args = as_recording_settings(cap_args)
    cap_args.out_dir.mkdir(parents=True, exist_ok=True)

    mxt = MaxTouchHid()
    try:
        mxt.load_info()
        region, page_size, pages, estimated_hz = resolve_region(mxt, rec_args)
        print(
            f"Region: ({region.start_x},{region.start_y}) "
            f"{region.width}×{region.height} columns×rows  |  "
            f"T37 pages {pages}  |  ~{estimated_hz:.1f} Hz"
        )

        run_dir = make_run_dir(rec_args)
        write_metadata(run_dir, cap_args, mxt, region, pages, page_size, estimated_hz)
        print(f"Recording to {run_dir}")

        stop_event = threading.Event()
        label_panel = ActivityLabelPanel(run_dir)
        existing_labels = load_activity_labels(labels_path_for_session(run_dir))
        if existing_labels:
            label_panel.labels = list(existing_labels)

        if cap_args.run_viewer:
            result: dict[str, int] = {}

            def worker() -> None:
                result["status"] = record_capacitance(
                    mxt, run_dir, region, rec_args, pages, estimated_hz, stop_event
                )

            thread = threading.Thread(target=worker, daemon=True)
            thread.start()
            cap_tailer = CapHeatmapTailer(
                run_dir / CAP_CSV_NAME,
                region.height,
                region.width,
                cap_args.view_invert_delta,
            )
            viewer = LiveCsvViewer(
                run_dir=run_dir,
                n_rows=region.height,
                n_cols=region.width,
                refresh_ms=cap_args.view_refresh_ms,
                cap_tailer=cap_tailer,
                ppg_tailer=None,
                stop_event=stop_event,
                label_panel=label_panel,
                color_max=cap_args.view_color_max,
                finger_detect_delay_s=cap_args.finger_detect_delay_s,
                finger_patch_size=cap_args.finger_patch_size,
                enable_pixel_stream=cap_args.enable_pixel_stream,
                estimated_cap_hz=estimated_hz,
                include_ppg_corr=False,
            )
            print("Recording… close the viewer window to stop.")
            viewer.show()
            thread.join()
            finalize_session_metadata(run_dir, label_panel.labels)
            print(f"Wrote {run_dir}")
            return result.get("status", 1)

        print("Recording… press Ctrl+C to stop.")
        try:
            status = record_capacitance(
                mxt, run_dir, region, rec_args, pages, estimated_hz, stop_event
            )
        except KeyboardInterrupt:
            stop_event.set()
            status = 0
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
        raise SystemExit(1) from error
