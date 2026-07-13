"""Synthetic imaging-sonar sector generator for closed-loop simulations."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, Sequence, Tuple

import numpy as np

try:
    from scipy import ndimage  # type: ignore
except Exception:  # pragma: no cover - exercised when scipy is unavailable.
    ndimage = None  # type: ignore

Vector3 = Tuple[float, float, float]

TARGET_TYPES = {"submarine", "torpedo", "ship", "reef", "unknown"}
DEFAULT_SONAR_PARAMS: Dict[str, float] = {
    "sector_width_deg": 60.0,
    "max_range_m": 10.0,
    "range_resolution_m": 0.1,
    "angular_resolution_deg": 1.0,
    "noise_level": 0.055,
}


def generate_sonar_image(
    uuv_position: tuple[float, float, float],
    target_position: tuple[float, float, float],
    target_type: str,
    target_heading_deg: float,
    sonar_params: dict | None = None,
) -> dict:
    """Generate a deterministic RGB sector image for a nearby target.

    The returned image uses a polar sonar layout: rows are range bins from near
    to far, columns are angle bins from sector-left to sector-right.
    """

    params = _merged_params(sonar_params)
    sector_width = float(params["sector_width_deg"])
    max_range = float(params["max_range_m"])
    range_resolution = float(params["range_resolution_m"])
    angular_resolution = float(params["angular_resolution_deg"])
    height = max(1, int(round(max_range / range_resolution)))
    width = max(1, int(round(sector_width / angular_resolution)))
    target_kind = target_type if target_type in TARGET_TYPES else "unknown"

    absolute_bearing = _bearing_from_to(uuv_position, target_position)
    uuv_heading = float(params.get("uuv_heading_deg", absolute_bearing)) % 360.0
    target_bearing = _angle_delta(uuv_heading, absolute_bearing)
    sector_center = float(params.get("sector_center_deg", target_bearing))
    sector_delta = _angle_delta(sector_center, target_bearing)
    target_range = _horizontal_range(uuv_position, target_position)
    target_depth = float(target_position[2]) if len(target_position) > 2 else -50.0
    noise_level = float(params["noise_level"])

    ranges = (np.arange(height, dtype=np.float32) + 0.5) * range_resolution
    base = np.exp(-ranges / (0.4 * max_range))[:, None] * 0.16
    intensity = np.repeat(base, width, axis=1)
    intensity += _near_field_reverberation(ranges, width, max_range)

    rng = np.random.default_rng(_stable_seed(uuv_position, target_position, target_kind, target_heading_deg))
    echo_strength = 0.0
    has_wake = False
    in_sector = abs(sector_delta) <= sector_width / 2.0 and target_range <= max_range
    if in_sector:
        center_row = (target_range / range_resolution) - 0.5
        center_col = (sector_delta + sector_width / 2.0) / angular_resolution - 0.5
        echo_strength = _target_strength(target_kind, rng)
        echo, has_wake = _target_echo(
            kind=target_kind,
            height=height,
            width=width,
            center_row=center_row,
            center_col=center_col,
            strength=echo_strength,
            rng=rng,
        )
        intensity += echo

    noise = rng.normal(0.0, noise_level, size=(height, width))
    intensity = np.clip(intensity + noise, 0.0, 1.0)
    image = _sonar_colormap(intensity)
    return {
        "image": image,
        "intensity": intensity.astype(np.float32),
        "sector_center_deg": round(float(sector_center), 3),
        "sector_width_deg": sector_width,
        "max_range_m": max_range,
        "echo_strength": round(float(echo_strength), 3),
        "noise_level": round(noise_level, 3),
        "target_bearing_deg": round(float(target_bearing), 3),
        "target_range_m": round(float(target_range), 3),
        "target_depth_m": round(target_depth, 3),
        "target_type_hint": target_kind,
        "has_wake_hint": has_wake,
    }


def _merged_params(values: dict | None) -> Dict[str, float]:
    params = dict(DEFAULT_SONAR_PARAMS)
    if values:
        params.update({key: value for key, value in values.items() if value is not None})
    return params


def _target_echo(
    *,
    kind: str,
    height: int,
    width: int,
    center_row: float,
    center_col: float,
    strength: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, bool]:
    rows, cols = np.indices((height, width), dtype=np.float32)
    has_wake = False
    if kind == "submarine":
        mask = _ellipse(rows, cols, center_row, center_col, range_sigma=2.0, angle_sigma=9.0)
        echo = _gaussian_filter(mask.astype(np.float32), sigma=(0.8, 1.6)) * strength
    elif kind == "torpedo":
        mask = _ellipse(rows, cols, center_row, center_col, range_sigma=0.9, angle_sigma=8.5)
        echo = _gaussian_filter(mask.astype(np.float32), sigma=(0.35, 1.0)) * strength
    elif kind == "ship":
        hull = _ellipse(rows, cols, center_row, center_col, range_sigma=3.2, angle_sigma=3.8)
        echo = _gaussian_filter(hull.astype(np.float32), sigma=(0.8, 0.8)) * strength
        echo += _wake(rows, cols, center_row, center_col, strength)
        has_wake = True
    elif kind == "reef":
        blobs = np.zeros((height, width), dtype=bool)
        for _ in range(5):
            row = center_row + rng.normal(0.0, 2.0)
            col = center_col + rng.normal(0.0, 2.4)
            blobs |= _ellipse(rows, cols, row, col, range_sigma=rng.uniform(1.1, 2.7), angle_sigma=rng.uniform(1.0, 2.8))
        blobs &= rng.random((height, width)) > 0.18
        blobs = _binary_dilation(blobs, iterations=1)
        echo = _gaussian_filter(blobs.astype(np.float32), sigma=(0.7, 0.7)) * strength
    else:
        echo = np.zeros((height, width), dtype=np.float32)
        for _ in range(9):
            row = center_row + rng.normal(0.0, 3.0)
            col = center_col + rng.normal(0.0, 4.0)
            spot = _ellipse(rows, cols, row, col, range_sigma=0.8, angle_sigma=0.8)
            echo += _gaussian_filter(spot.astype(np.float32), sigma=(0.7, 0.7)) * strength * rng.uniform(0.35, 0.8)
    return np.clip(echo, 0.0, 1.0), has_wake


def _ellipse(
    rows: np.ndarray,
    cols: np.ndarray,
    center_row: float,
    center_col: float,
    *,
    range_sigma: float,
    angle_sigma: float,
) -> np.ndarray:
    return ((rows - center_row) / range_sigma) ** 2 + ((cols - center_col) / angle_sigma) ** 2 <= 1.0


def _wake(
    rows: np.ndarray,
    cols: np.ndarray,
    center_row: float,
    center_col: float,
    strength: float,
) -> np.ndarray:
    tail = np.zeros_like(rows, dtype=np.float32)
    downstream = rows > center_row + 2.0
    stripe = np.abs(cols - center_col) < 2.0
    decay = np.exp(-(rows - center_row - 2.0) / 8.0)
    tail[downstream & stripe] = decay[downstream & stripe] * strength * 0.38
    return _gaussian_filter(tail, sigma=(1.1, 0.9))


def _near_field_reverberation(ranges: np.ndarray, width: int, max_range: float) -> np.ndarray:
    near = ranges < min(5.0, max_range)
    stripes = (np.sin(ranges[:, None] * 7.0) + 1.0) * 0.018
    angular_texture = (np.cos(np.linspace(-math.pi, math.pi, width, dtype=np.float32))[None, :] + 1.0) * 0.006
    return np.where(near[:, None], stripes + angular_texture, 0.0)


def _target_strength(kind: str, rng: np.random.Generator) -> float:
    ranges = {
        "submarine": (0.80, 0.95),
        "torpedo": (0.80, 0.95),
        "ship": (0.70, 0.90),
        "reef": (0.30, 0.50),
        "unknown": (0.20, 0.40),
    }
    low, high = ranges.get(kind, ranges["unknown"])
    return float(rng.uniform(low, high))


def _sonar_colormap(intensity: np.ndarray) -> np.ndarray:
    stops = np.array(
        [
            [6, 18, 45],
            [0, 115, 150],
            [245, 198, 75],
            [255, 255, 245],
        ],
        dtype=np.float32,
    )
    values = np.clip(intensity, 0.0, 1.0) * (len(stops) - 1)
    left = np.floor(values).astype(int)
    right = np.clip(left + 1, 0, len(stops) - 1)
    fraction = (values - left)[..., None]
    rgb = stops[left] * (1.0 - fraction) + stops[right] * fraction
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _bearing_from_to(origin: Sequence[float], target: Sequence[float]) -> float:
    dx = float(target[0]) - float(origin[0])
    dy = float(target[1]) - float(origin[1])
    return (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0


def _horizontal_range(origin: Sequence[float], target: Sequence[float]) -> float:
    return math.hypot(float(target[0]) - float(origin[0]), float(target[1]) - float(origin[1]))


def _angle_delta(start: float, end: float) -> float:
    return (float(end) % 360.0 - float(start) % 360.0 + 180.0) % 360.0 - 180.0


def _stable_seed(
    uuv_position: Sequence[float],
    target_position: Sequence[float],
    target_type: str,
    target_heading_deg: float,
) -> int:
    json_like = (
        tuple(round(float(value), 2) for value in uuv_position),
        tuple(round(float(value), 2) for value in target_position),
        target_type,
        round(float(target_heading_deg), 2),
    )
    digest = hashlib.sha256(repr(json_like).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _gaussian_filter(values: np.ndarray, sigma: tuple[float, float]) -> np.ndarray:
    if ndimage is not None:
        return ndimage.gaussian_filter(values, sigma=sigma)
    result = values.astype(np.float32)
    for axis, axis_sigma in enumerate(sigma):
        result = _convolve_axis(result, _gaussian_kernel(float(axis_sigma)), axis)
    return result


def _gaussian_kernel(sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.array([1.0], dtype=np.float32)
    radius = max(1, int(math.ceil(sigma * 3.0)))
    x_values = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x_values**2) / (2.0 * sigma * sigma))
    return (kernel / np.sum(kernel)).astype(np.float32)


def _convolve_axis(values: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    radius = len(kernel) // 2
    pad_width = [(0, 0)] * values.ndim
    pad_width[axis] = (radius, radius)
    padded = np.pad(values, pad_width, mode="edge")
    result = np.zeros_like(values, dtype=np.float32)
    for offset, weight in enumerate(kernel):
        start = offset
        stop = start + values.shape[axis]
        slices = [slice(None)] * values.ndim
        slices[axis] = slice(start, stop)
        result += padded[tuple(slices)] * float(weight)
    return result


def _binary_dilation(values: np.ndarray, iterations: int = 1) -> np.ndarray:
    if ndimage is not None:
        return ndimage.binary_dilation(values, iterations=iterations)
    result = values.astype(bool)
    for _ in range(max(0, iterations)):
        padded = np.pad(result, ((1, 1), (1, 1)), mode="constant", constant_values=False)
        expanded = np.zeros_like(result, dtype=bool)
        for row_offset in range(3):
            for col_offset in range(3):
                expanded |= padded[row_offset : row_offset + result.shape[0], col_offset : col_offset + result.shape[1]]
        result = expanded
    return result
