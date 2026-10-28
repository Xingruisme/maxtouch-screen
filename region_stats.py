#!/usr/bin/env python3
"""
Common utilities for calculating and displaying region statistics.
Used by both measure.py and load_regions.py
"""

import numpy as np
from scipy import ndimage as ndi

FINGER_DEFAULT_NAMES = ("thumb", "index", "middle", "ring", "little")

# Palm classification: fingertip blobs on the OnePlus 38x54 matrix are compact
# (~8-12 px, <=5 px span); palm/thenar contact is a large irregular blob
# (often 100+ px spanning 10-20 rows/cols). A blob is treated as a palm when it
# exceeds either bound below.
PALM_MIN_PIXELS = 30
PALM_MIN_SPAN = 9
PALM_DEFAULT_NAME = "palm"


def is_palm_mask(
    mask: "np.ndarray",
    *,
    min_pixels: int = PALM_MIN_PIXELS,
    min_span: int = PALM_MIN_SPAN,
) -> bool:
    """True when a region mask is palm-sized rather than fingertip-sized."""
    pixel_count = int(np.sum(mask))
    if pixel_count == 0:
        return False
    if pixel_count >= min_pixels:
        return True
    y_coords, x_coords = np.where(mask)
    span = max(
        int(y_coords.max() - y_coords.min() + 1),
        int(x_coords.max() - x_coords.min() + 1),
    )
    return span >= min_span


def default_palm_name(index: int) -> str:
    return PALM_DEFAULT_NAME if index == 0 else f"{PALM_DEFAULT_NAME}_{index + 1}"


def default_finger_name(index: int, count: int) -> str:
    """Default left-to-right finger label for ``index`` among ``count`` regions.

    Five fingers use thumb through little; fewer than five skip thumb and start
    at index (e.g. 3 fingers → index, middle, ring).
    """
    if count <= 0 or index < 0:
        return f"region_{index + 1}"
    if count >= len(FINGER_DEFAULT_NAMES):
        if index < len(FINGER_DEFAULT_NAMES):
            return FINGER_DEFAULT_NAMES[index]
        return f"region_{index + 1}"
    name_idx = 1 + index
    if name_idx < len(FINGER_DEFAULT_NAMES):
        return FINGER_DEFAULT_NAMES[name_idx]
    return f"region_{index + 1}"


def order_fingers_anatomically(finger_masks, data):
    """Order finger masks thumb→little using hand anatomy, not raw x position.

    Fingers of one hand fan out along a single line (roughly perpendicular to
    the forearm), whatever the hand's rotation on the panel. We therefore:

      1. take each finger's centroid and find the line they spread along
         (principal axis via SVD), and sort the fingers along it,
      2. pick which end is the thumb: the thumb–index web is the widest space
         between neighbouring digits, so the end whose gap to its neighbour is
         larger is the thumb end,
      3. return the masks ordered from the thumb end, so the caller's default
         naming (thumb, index, middle, ring, little) lands on the right digit.

    With fewer than three fingers the thumb end is ambiguous, so the masks are
    returned in principal-axis order unchanged.
    """
    masks = list(finger_masks)
    n = len(masks)
    if n <= 1:
        return masks

    centroids = []
    for mask in masks:
        stats = calculate_region_stats(mask, data)
        centroids.append((float(stats["centroid_x"]), float(stats["centroid_y"])))
    pts = np.asarray(centroids, dtype=float)
    centered = pts - pts.mean(axis=0)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    order = list(np.argsort(centered @ axis))

    if n >= 3:
        ordered_pts = pts[order]
        gap_start = float(np.linalg.norm(ordered_pts[1] - ordered_pts[0]))
        gap_end = float(np.linalg.norm(ordered_pts[-1] - ordered_pts[-2]))
        if gap_end > gap_start:  # thumb is at the far (largest-gap) end
            order = order[::-1]
    return [masks[i] for i in order]


def compute_region_area(region_mask, pixel_spacing_mm=None):
    """
    Compute the area of a region in pixels and optionally in mm².

    Parameters:
    -----------
    region_mask : np.ndarray
        Binary mask of the region
    pixel_spacing_mm : float or tuple, optional
        Physical spacing between pixels in mm. If tuple, (y_spacing, x_spacing).
        If None, returns pixel count only.

    Returns:
    --------
    area_info : dict
        Dictionary with area information including dimensions
    """
    area_pixels = np.sum(region_mask)

    # Get bounding box dimensions
    y_coords, x_coords = np.where(region_mask)
    y_min, y_max = y_coords.min(), y_coords.max()
    x_min, x_max = x_coords.min(), x_coords.max()

    # Dimensions in pixels (inclusive)
    width_pixels = x_max - x_min + 1
    height_pixels = y_max - y_min + 1

    # Compute perimeter using edge detection
    # Find boundary pixels (pixels that have at least one non-region neighbor)
    eroded = ndi.binary_erosion(region_mask)
    boundary = region_mask & ~eroded
    perimeter_pixels = np.sum(boundary)

    # Compute perimeter roughness (ratio of actual perimeter to ideal circular perimeter)
    # For a circle: perimeter = 2 * pi * sqrt(area / pi) = 2 * sqrt(pi * area)
    if area_pixels > 0:
        ideal_perimeter = 2 * np.sqrt(np.pi * area_pixels)
        perimeter_roughness = perimeter_pixels / ideal_perimeter
    else:
        ideal_perimeter = 0
        perimeter_roughness = 0

    area_info = {
        'area_pixels': int(area_pixels),
        'width_pixels': int(width_pixels),
        'height_pixels': int(height_pixels),
        'perimeter_pixels': int(perimeter_pixels),
        'perimeter_roughness': float(perimeter_roughness),
        'bbox': (y_min, y_max, x_min, x_max)
    }

    if pixel_spacing_mm is not None:
        if isinstance(pixel_spacing_mm, (int, float)):
            # Uniform spacing
            area_mm2 = area_pixels * (pixel_spacing_mm ** 2)
            width_mm = width_pixels * pixel_spacing_mm
            height_mm = height_pixels * pixel_spacing_mm
            perimeter_mm = perimeter_pixels * pixel_spacing_mm
        else:
            # Different spacing in x and y - use average for perimeter
            y_spacing, x_spacing = pixel_spacing_mm
            area_mm2 = area_pixels * y_spacing * x_spacing
            width_mm = width_pixels * x_spacing
            height_mm = height_pixels * y_spacing
            perimeter_mm = perimeter_pixels * np.mean([y_spacing, x_spacing])

        area_info['area_mm2'] = area_mm2
        area_info['width_mm'] = width_mm
        area_info['height_mm'] = height_mm
        area_info['perimeter_mm'] = perimeter_mm
        area_info['pixel_spacing'] = pixel_spacing_mm

    return area_info


def calculate_region_stats(region_mask, data, pixel_spacing_mm=None):
    """
    Calculate statistics for a single region.

    Parameters:
    -----------
    region_mask : np.ndarray
        Binary mask of the region
    data : np.ndarray
        2D data array
    pixel_spacing_mm : float or tuple, optional
        Physical spacing between pixels in mm

    Returns:
    --------
    stats : dict
        Dictionary containing all region statistics
    """
    region_values = data[region_mask]
    y_coords, x_coords = np.where(region_mask)

    # Special handling for 2-pixel regions: bias towards pixel with largest absolute value
    if len(y_coords) == 2:
        val0 = data[y_coords[0], x_coords[0]]
        val1 = data[y_coords[1], x_coords[1]]
        if abs(val0) > abs(val1):
            centroid_y = y_coords[0]
            centroid_x = x_coords[0]
        else:
            centroid_y = y_coords[1]
            centroid_x = x_coords[1]
    else:
        # Normal centroid calculation with rounding
        centroid_y = int(np.mean(y_coords) + 0.5)
        centroid_x = int(np.mean(x_coords) + 0.5)

    centroid_value = data[centroid_y, centroid_x]

    # Compute area info
    area_info = compute_region_area(region_mask, pixel_spacing_mm)

    stats = {
        'pixel_count': np.sum(region_mask),
        'centroid_y': centroid_y,
        'centroid_x': centroid_x,
        'centroid_value': centroid_value,
        'sum': np.sum(region_values),
        'mean': np.mean(region_values),
        'std': np.std(region_values),
        'min': np.min(region_values),
        'max': np.max(region_values),
        'area_pixels': area_info['area_pixels'],
        'width_pixels': area_info['width_pixels'],
        'height_pixels': area_info['height_pixels'],
        'perimeter_pixels': area_info['perimeter_pixels'],
        'perimeter_roughness': area_info['perimeter_roughness'],
        'bbox': area_info['bbox']
    }

    # Add mm measurements if available
    if 'area_mm2' in area_info:
        stats['area_mm2'] = area_info['area_mm2']
        stats['width_mm'] = area_info['width_mm']
        stats['height_mm'] = area_info['height_mm']
        stats['perimeter_mm'] = area_info['perimeter_mm']
        stats['pixel_spacing'] = area_info['pixel_spacing']

    return stats


def calculate_summary_stats(individual_regions, combined_mask, data, pixel_spacing_mm=None):
    """
    Calculate summary statistics for all regions combined and per-region averages.

    Parameters:
    -----------
    individual_regions : list of np.ndarray
        List of binary masks for individual regions
    combined_mask : np.ndarray
        Combined binary mask of all regions
    data : np.ndarray
        2D data array
    pixel_spacing_mm : float or tuple, optional
        Physical spacing between pixels in mm

    Returns:
    --------
    summary : dict
        Dictionary containing combined and average statistics
    """
    num_regions = len(individual_regions)
    all_selected_values = data[combined_mask]

    # Combined statistics
    y_coords_all, x_coords_all = np.where(combined_mask)
    centroid_y_all = int(np.mean(y_coords_all) + 0.5)
    centroid_x_all = int(np.mean(x_coords_all) + 0.5)
    centroid_value_all = data[centroid_y_all, centroid_x_all]

    area_info_all = compute_region_area(combined_mask, pixel_spacing_mm)

    # Per-region averages
    avg_perimeter_pixels = np.mean([compute_region_area(r, pixel_spacing_mm)['perimeter_pixels']
                                     for r in individual_regions])
    avg_roughness = np.mean([compute_region_area(r, pixel_spacing_mm)['perimeter_roughness']
                             for r in individual_regions])
    avg_centroid_value = np.mean([data[int(np.mean(np.where(r)[0]) + 0.5),
                                       int(np.mean(np.where(r)[1]) + 0.5)]
                                  for r in individual_regions])
    avg_sum = np.mean([np.sum(data[r]) for r in individual_regions])

    summary = {
        'num_regions': num_regions,
        'combined': {
            'pixel_count': int(np.sum(combined_mask)),
            'centroid_y': centroid_y_all,
            'centroid_x': centroid_x_all,
            'centroid_value': centroid_value_all,
            'sum': np.sum(all_selected_values),
            'mean': np.mean(all_selected_values),
            'std': np.std(all_selected_values),
            'min': np.min(all_selected_values),
            'max': np.max(all_selected_values),
            'area_pixels': area_info_all['area_pixels'],
            'width_pixels': area_info_all['width_pixels'],
            'height_pixels': area_info_all['height_pixels'],
            'perimeter_pixels': area_info_all['perimeter_pixels'],
            'perimeter_roughness': area_info_all['perimeter_roughness']
        },
        'averages': {
            'perimeter_pixels': avg_perimeter_pixels,
            'perimeter_roughness': avg_roughness,
            'centroid_value': avg_centroid_value,
            'sum': avg_sum
        }
    }

    # Add mm measurements if available
    if 'area_mm2' in area_info_all:
        summary['combined']['area_mm2'] = area_info_all['area_mm2']
        summary['combined']['width_mm'] = area_info_all['width_mm']
        summary['combined']['height_mm'] = area_info_all['height_mm']
        summary['combined']['perimeter_mm'] = area_info_all['perimeter_mm']

    return summary


def print_region_stats(stats, region_idx, total_regions, region_name=None):
    """
    Print statistics for a single region.

    Parameters:
    -----------
    stats : dict
        Statistics dictionary from calculate_region_stats()
    region_idx : int
        Region index (1-based for display)
    total_regions : int
        Total number of regions
    region_name : str, optional
        Human-readable region label (shown instead of a bare index)
    """
    if region_name:
        print(f"\n--- {region_name} (region {region_idx} of {total_regions}) ---")
    else:
        print(f"\n--- Region {region_idx} of {total_regions} ---")
    print(f"Pixel count: {stats['pixel_count']}")

    # Display dimensions
    print(f"Dimensions: {stats['width_pixels']} x {stats['height_pixels']} pixels", end='')
    if 'width_mm' in stats:
        print(f" ({stats['width_mm']:.1f} x {stats['height_mm']:.1f} mm)")
    else:
        print()

    # Display area
    print(f"Area: {stats['area_pixels']} pixels", end='')
    if 'area_mm2' in stats:
        print(f" ({stats['area_mm2']:.2f} mm²)")
    else:
        print()

    # Display perimeter
    print(f"Perimeter: {stats['perimeter_pixels']} pixels", end='')
    if 'perimeter_mm' in stats:
        print(f" ({stats['perimeter_mm']:.2f} mm)")
    else:
        print()
    print(f"Perimeter roughness: {stats['perimeter_roughness']:.3f}")

    print(f"Centroid: ({stats['centroid_x']}, {stats['centroid_y']})")
    print(f"Centroid value: {stats['centroid_value']:.2f}")
    print(f"Sum: {stats['sum']:.2f}")
    print(f"Average value: {stats['mean']:.2f}")
    print(f"Std dev: {stats['std']:.2f}")
    print(f"Min value: {stats['min']:.2f}")
    print(f"Max value: {stats['max']:.2f}")


def print_summary_stats(summary):
    """
    Print summary statistics for all regions.

    Parameters:
    -----------
    summary : dict
        Summary statistics dictionary from calculate_summary_stats()
    """
    combined = summary['combined']
    averages = summary['averages']

    print(f"\n--- All Selected Regions Combined ---")
    print(f"Number of regions: {summary['num_regions']}")
    print(f"Pixel count: {combined['pixel_count']}")

    # Display combined dimensions
    print(f"Dimensions: {combined['width_pixels']} x {combined['height_pixels']} pixels", end='')
    if 'width_mm' in combined:
        print(f" ({combined['width_mm']:.1f} x {combined['height_mm']:.1f} mm)")
    else:
        print()

    # Display combined area
    print(f"Area: {combined['area_pixels']} pixels", end='')
    if 'area_mm2' in combined:
        print(f" ({combined['area_mm2']:.2f} mm²)")
    else:
        print()

    # Note: Perimeter, roughness, and centroid are not meaningful for disconnected regions
    print(f"Sum: {combined['sum']:.2f}")
    print(f"Average value: {combined['mean']:.2f}")
    print(f"Std dev: {combined['std']:.2f}")
    print(f"Min value: {combined['min']:.2f}")
    print(f"Max value: {combined['max']:.2f}")

    # Print per-region averages
    print(f"\n--- Per-Region Averages ---")
    print(f"Average perimeter: {averages['perimeter_pixels']:.2f} pixels")
    print(f"Average perimeter roughness: {averages['perimeter_roughness']:.3f}")
    print(f"Average centroid value: {averages['centroid_value']:.2f}")
    print(f"Stdev centroid value: {np.std([combined['centroid_value'] for _ in range(summary['num_regions'])]):.2f}")
    print(f"Average sum: {averages['sum']:.2f}")


def clamp_extreme_values(
    data: np.ndarray,
    lower_limit: float = -2000.0,
    upper_limit: float = 2000.0,
) -> np.ndarray:
    """Zero out extreme outliers before blob detection."""
    clamped = data.copy()
    clamped[(clamped < lower_limit) | (clamped > upper_limit)] = 0
    return clamped


def zscore_threshold(
    data: np.ndarray,
    z_threshold: float = 2.0,
    mode: str = "both",
) -> float:
    """Return a z-score threshold on the requested polarity."""
    mean = float(np.mean(data))
    std = float(np.std(data))
    if mode == "positive":
        return mean + z_threshold * std
    if mode == "negative":
        return mean - z_threshold * std

    max_pos = float(np.max(data - mean))
    max_neg = float(np.abs(np.min(data - mean)))
    if max_pos > max_neg:
        return mean + z_threshold * std
    return mean - z_threshold * std


def _bbox_chebyshev_gap(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Gap between two mask bounding boxes (0 if they touch or overlap)."""
    ya, xa = np.where(mask_a)
    yb, xb = np.where(mask_b)
    if ya.size == 0 or yb.size == 0:
        return float("inf")
    y_gap = max(0, int(ya.min()) - int(yb.max()) - 1, int(yb.min()) - int(ya.max()) - 1)
    x_gap = max(0, int(xa.min()) - int(xb.max()) - 1, int(xb.min()) - int(xa.max()) - 1)
    return float(max(y_gap, x_gap))


def mask_polarity(mask: np.ndarray, signed: np.ndarray) -> tuple[float, int]:
    """Return (purity, sign) for a contact mask on a signed Δ grid.

    Purity is the fraction of non-zero pixels that share the dominant sign.
    A split dipole lobe is ~0.9; a whole fingertip that already contains both
    poles is ~0.5–0.7.
    """
    vals = np.asarray(signed, dtype=np.float64)[mask]
    pos = int(np.sum(vals > 0))
    neg = int(np.sum(vals < 0))
    total = pos + neg
    if total == 0:
        return 0.0, 0
    if pos >= neg:
        return pos / total, 1
    return neg / total, -1


def merge_bipolar_contact_masks(
    masks: list[np.ndarray],
    signed: np.ndarray,
    *,
    min_purity: float = 0.7,
    max_bbox_gap: float = 1.0,
    max_centroid_dist: float = 6.0,
) -> list[np.ndarray]:
    """Join opposite-polarity lobes that belong to one fingertip.

    Pixel 4 STM FTS contact is often a +/− dipole with a near-zero gap.
    Detection on |Δ| can split that into two blobs (``region_6`` / ``region_7``
    next to an already-correct finger). Only *unipolar* neighbours are paired,
    so a fingertip that already contains both poles is left alone and two
    adjacent fingers are not glued together.
    """
    if len(masks) < 2:
        return list(masks)

    infos: list[dict] = []
    for mask in masks:
        purity, sign = mask_polarity(mask, signed)
        ys, xs = np.where(mask)
        infos.append(
            {
                "mask": mask,
                "purity": purity,
                "sign": sign,
                "cy": float(ys.mean()) if ys.size else 0.0,
                "cx": float(xs.mean()) if xs.size else 0.0,
            }
        )

    used = [False] * len(infos)
    pairs: list[tuple[float, int, int]] = []
    for i in range(len(infos)):
        for j in range(i + 1, len(infos)):
            a, b = infos[i], infos[j]
            if a["sign"] == 0 or b["sign"] == 0 or a["sign"] == b["sign"]:
                continue
            if a["purity"] < min_purity or b["purity"] < min_purity:
                continue
            gap = _bbox_chebyshev_gap(a["mask"], b["mask"])
            dist = float(
                np.hypot(a["cy"] - b["cy"], a["cx"] - b["cx"])
            )
            if gap > max_bbox_gap or dist > max_centroid_dist:
                continue
            pairs.append((dist, i, j))
    pairs.sort()

    merged: list[np.ndarray] = []
    for _dist, i, j in pairs:
        if used[i] or used[j]:
            continue
        used[i] = used[j] = True
        merged.append(infos[i]["mask"] | infos[j]["mask"])
    for i, info in enumerate(infos):
        if not used[i]:
            merged.append(info["mask"])
    return merged


def coalesce_bipolar_touch_regions(
    finger_masks: list[np.ndarray],
    palm_masks: list[np.ndarray],
    signed: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Merge split +/− lobes, then reclassify any palm-sized union."""
    fingers = merge_bipolar_contact_masks(finger_masks, signed)
    still_fingers: list[np.ndarray] = []
    palms = list(palm_masks)
    for mask in fingers:
        if is_palm_mask(mask):
            palms.append(mask)
        else:
            still_fingers.append(mask)
    return still_fingers, palms


def constructive_pixel_mean(
    cap_pixels: np.ndarray,
    indices: np.ndarray | None = None,
    *,
    signed_correlations: np.ndarray | None = None,
) -> np.ndarray:
    """Average selected pixels so a +/− dipole adds instead of cancelling.

    Live Pixel 4 CPG uses ``mean(+) − mean(−)``. The viewer previously took a
    plain mean of the top-4 |PPG|-correlated pixels, which often mixed a red
    lobe and a green lobe and wiped out the pulse.

    When ``signed_correlations`` is given, each pixel is flipped to agree with
    the PPG. Otherwise pixels are flipped by the sign of their DC mean.
    """
    pixels = np.asarray(cap_pixels, dtype=np.float64)
    if pixels.ndim != 2:
        raise ValueError("cap_pixels must be (time, pixel)")
    if indices is None:
        idx = np.arange(pixels.shape[1])
    else:
        idx = np.asarray(indices, dtype=int)
        idx = idx[(idx >= 0) & (idx < pixels.shape[1])]
    if idx.size == 0:
        return np.full(pixels.shape[0], np.nan)
    selected = pixels[:, idx]
    if signed_correlations is not None:
        signs = np.sign(np.asarray(signed_correlations, dtype=np.float64)[idx])
    else:
        signs = np.sign(np.nanmean(selected, axis=0))
    signs = np.where(signs == 0, 1.0, signs)
    return (selected * signs).mean(axis=1)


def dipole_mean(values: np.ndarray) -> float:
    """Constructive mean of a +/− contact dipole.

    ``mean(positive) − mean(negative)`` so the two lobes add instead of
    cancelling. Falls back to a plain mean when only one polarity is present.
    """
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    pos = arr[arr > 0]
    neg = arr[arr < 0]
    if pos.size and neg.size:
        return float(pos.mean() - neg.mean())
    return float(arr.mean())


def auto_detect_finger_regions(
    data: np.ndarray,
    *,
    z_threshold: float = 2.0,
    min_size: int = 3,
    lower_limit: float = -2000.0,
    upper_limit: float = 2000.0,
) -> list[np.ndarray]:
    """Detect finger blobs as connected components above a z-score threshold.

    Picks positive or negative polarity automatically based on whichever side
    of the mean has the larger deviation (typical capacitance delta sign varies
    by device/session).
    """
    data_clamped = clamp_extreme_values(data, lower_limit, upper_limit)
    threshold = zscore_threshold(data_clamped, z_threshold, mode="both")
    mean = float(np.mean(data_clamped))
    mask = data_clamped > threshold if threshold > mean else data_clamped < threshold
    if not mask.any():
        return []

    structure = np.ones((3, 3), dtype=bool)
    labeled, num_features = ndi.label(mask, structure=structure)
    regions: list[np.ndarray] = []
    for i in range(1, num_features + 1):
        region_mask = labeled == i
        if int(region_mask.sum()) >= min_size:
            regions.append(region_mask)
    return regions


def auto_detect_touch_regions(
    data: np.ndarray,
    *,
    z_threshold: float = 2.0,
    min_size: int = 3,
    lower_limit: float = -2000.0,
    upper_limit: float = 2000.0,
    palm_min_pixels: int = PALM_MIN_PIXELS,
    palm_min_span: int = PALM_MIN_SPAN,
    palm_bridge_z: float = 0.5,
    palm_absorb_gap: float = 6.0,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Detect a hand's contact and split it into fingertips and palm regions.

    Rather than classifying every blob in isolation, this reconstructs the
    whole-hand footprint so it can tell a fingertip apart from the raised
    metacarpal pads that sit under the fingers (which are anatomically part of
    the palm, not separate touches):

      1. detect strong contact blobs (:func:`auto_detect_finger_regions`) and
         mark palm-sized ones (large/irregular) as palm seeds,
      2. grow a "hand body" from the palm seeds at a *relaxed* threshold
         (``palm_bridge_z``): the raised pads are continuous raised tissue, so
         lowering the threshold bridges each pad to the palm mass, whereas a
         real fingertip stays in its own valley,
      3. absorb a small blob into the palm when it shares that relaxed hand
         body with a palm seed AND lies within ``palm_absorb_gap`` pixels of it
         (the distance guard stops a stray low-signal bridge from swallowing a
         genuinely separate fingertip),
      4. group the palm seeds + absorbed pads into one mask per hand body.

    Returns ``(finger_masks, palm_masks)``.
    """
    blobs = auto_detect_finger_regions(
        data,
        z_threshold=z_threshold,
        min_size=min_size,
        lower_limit=lower_limit,
        upper_limit=upper_limit,
    )
    if not blobs:
        return [], []

    palm_like = [
        is_palm_mask(mask, min_pixels=palm_min_pixels, min_span=palm_min_span)
        for mask in blobs
    ]
    palm_blobs = [mask for mask, is_palm in zip(blobs, palm_like) if is_palm]
    finger_blobs = [mask for mask, is_palm in zip(blobs, palm_like) if not is_palm]
    if not palm_blobs:
        return finger_blobs, []

    structure = np.ones((3, 3), dtype=bool)
    palm_union = np.zeros_like(palm_blobs[0], dtype=bool)
    for mask in palm_blobs:
        palm_union |= mask

    # Relaxed hand-body footprint: palm seeds plus everything continuous with
    # them at the lower threshold. Connected components of this footprint that
    # contain a palm seed are the palm body of each hand.
    data_clamped = clamp_extreme_values(data, lower_limit, upper_limit)
    mean = float(np.mean(data_clamped))
    bridge_thr = zscore_threshold(data_clamped, palm_bridge_z, mode="both")
    relaxed = data_clamped > bridge_thr if bridge_thr > mean else data_clamped < bridge_thr
    body_labels, _ = ndi.label(relaxed | palm_union, structure=structure)
    palm_body_ids = set(int(v) for v in np.unique(body_labels[palm_union]) if v > 0)

    # Euclidean distance from every pixel to the nearest strong palm pixel.
    gap_to_palm = ndi.distance_transform_edt(~palm_union)

    finger_masks: list[np.ndarray] = []
    pad_blobs: list[np.ndarray] = []
    for mask in finger_blobs:
        blob_body_ids = set(int(v) for v in np.unique(body_labels[mask]) if v > 0)
        shares_body = bool(blob_body_ids & palm_body_ids)
        near_palm = float(gap_to_palm[mask].min()) <= palm_absorb_gap
        if shares_body and near_palm:
            pad_blobs.append(mask)
        else:
            finger_masks.append(mask)

    # Group palm seeds + absorbed pads into one mask per relaxed hand body.
    palm_masks_by_id: dict[int, np.ndarray] = {
        body_id: np.zeros_like(palm_union, dtype=bool) for body_id in palm_body_ids
    }
    for mask in palm_blobs + pad_blobs:
        ids = [int(v) for v in np.unique(body_labels[mask]) if int(v) in palm_masks_by_id]
        if ids:
            palm_masks_by_id[ids[0]] |= mask

    palm_masks = [mask for mask in palm_masks_by_id.values() if mask.any()]
    return finger_masks, palm_masks
