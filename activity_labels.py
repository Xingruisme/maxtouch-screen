"""Activity label sidecar for capacitance session recordings.

Sidecar file: ``capacitance_labels.csv`` next to ``capacitance.csv`` with columns
``elapsed_s,label`` (same convention as thermistor / PPG log viewers).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

LABELS_FILENAME = "capacitance_labels.csv"
TOUCH_LABELS_FILENAME = "touch_labels.csv"
CAP_FILENAME = "capacitance.csv"
TOUCH_CSV_FILENAME = "touch_events.csv"


def labels_path_for_session(session_dir: Path) -> Path:
    return session_dir / LABELS_FILENAME


def touch_labels_path_for_session(session_dir: Path) -> Path:
    return session_dir / TOUCH_LABELS_FILENAME


def resolve_touch_labels_path(session_dir: Path) -> Path:
    primary = touch_labels_path_for_session(session_dir)
    if primary.is_file():
        return primary
    legacy = labels_path_for_csv(session_dir / TOUCH_CSV_FILENAME)
    return legacy if legacy.is_file() else primary


def labels_path_for_csv(csv_path: Path) -> Path:
    stem = csv_path.stem
    return csv_path.with_name(f"{stem}_labels.csv")


def resolve_labels_path(session_dir: Path) -> Path:
    """Prefer ``capacitance_labels.csv``; fall back to legacy sidecar naming."""
    primary = labels_path_for_session(session_dir)
    if primary.is_file():
        return primary
    legacy = labels_path_for_csv(session_dir / CAP_FILENAME)
    return legacy if legacy.is_file() else primary


def load_activity_labels(path: Path) -> list[tuple[float, str]]:
    if not path.is_file():
        return []
    out: list[tuple[float, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = (row.get("label") or "").strip()
            if not label:
                continue
            try:
                elapsed = float(row["elapsed_s"])
            except (KeyError, TypeError, ValueError):
                continue
            out.append((elapsed, label))
    out.sort(key=lambda item: item[0])
    return out


def save_activity_labels(path: Path, labels: list[tuple[float, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["elapsed_s", "label"])
        for elapsed, label in labels:
            writer.writerow([f"{elapsed:.6f}", label])


def sync_metadata_labels(session_dir: Path, labels: list[tuple[float, str]]) -> None:
    meta_path = session_dir / "metadata.json"
    if not meta_path.is_file():
        return
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    metadata["activity_labels"] = [
        {"elapsed_s": elapsed, "label": label} for elapsed, label in labels
    ]
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def draw_label_markers(
    axes,
    labels: list[tuple[float, str]],
    *,
    x_offset: float = 0.0,
    color: str = "#E65100",
    fontsize: int = 7,
) -> list:
    """Draw vertical markers and rotated text on one or more matplotlib axes."""
    axis_list = list(axes) if isinstance(axes, (list, tuple)) else [axes]
    artists: list = []
    for elapsed, text in labels:
        x = elapsed + x_offset
        for axis in axis_list:
            artists.append(
                axis.axvline(x, color=color, lw=0.9, ls=":", alpha=0.75, zorder=4)
            )
        if axis_list:
            artists.append(
                axis_list[0].text(
                    x,
                    0.98,
                    text,
                    transform=axis_list[0].get_xaxis_transform(),
                    rotation=90,
                    va="top",
                    ha="right",
                    fontsize=fontsize,
                    color=color,
                    clip_on=True,
                    zorder=5,
                )
            )
    return artists
