"""Rule-based target recognition over synthetic imaging-sonar output."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

try:
    from scipy import ndimage  # type: ignore
except Exception:  # pragma: no cover - exercised when scipy is unavailable.
    ndimage = None  # type: ignore

REAL_TARGET_TYPES = {"submarine", "torpedo", "ship"}
HIGH_VALUE_TARGET_TYPES = {"submarine", "torpedo"}


def recognize_target(
    sonar_output: dict,
    sonar_params: dict | None = None,
) -> dict:
    """Recognize target type from a synthetic sonar image without ML models."""

    intensity = _intensity_from_output(sonar_output)
    target_depth = float(sonar_output.get("target_depth_m", -50.0))
    target_range = float(sonar_output.get("target_range_m", 999.0))
    echo_strength = float(sonar_output.get("echo_strength", 0.0))
    features = _extract_features(intensity, echo_strength, bool(sonar_output.get("has_wake_hint")))

    if echo_strength <= 0.0 or features["size_pixels"] <= 0:
        target_type = "unknown"
        confidence = 0.2
        reasoning = "未形成稳定高亮回波，目标尚未进入成像声呐有效扇区。"
    else:
        target_type, confidence, reasoning = _match_rules(features)

    if target_range > float((sonar_params or {}).get("clear_range_m", 8.0)):
        confidence *= 0.6
        reasoning += " 目标处于扇区远端，置信度按距离衰减。"

    confidence = round(max(0.0, min(1.0, confidence)), 3)
    return {
        "target_type": target_type,
        "confidence": confidence,
        "is_real_target": target_type in REAL_TARGET_TYPES,
        "is_high_value_target": target_type in HIGH_VALUE_TARGET_TYPES,
        "is_blue_target": False,
        "target_depth_m": round(target_depth, 3),
        "features": features,
        "reasoning": reasoning,
    }


def _intensity_from_output(sonar_output: dict) -> np.ndarray:
    if isinstance(sonar_output.get("intensity"), np.ndarray):
        return np.clip(sonar_output["intensity"].astype(np.float32), 0.0, 1.0)
    image = np.asarray(sonar_output.get("image"), dtype=np.float32)
    if image.ndim != 3 or image.shape[2] < 3:
        return np.zeros((1, 1), dtype=np.float32)
    luminance = image[..., 0] * 0.2126 + image[..., 1] * 0.7152 + image[..., 2] * 0.0722
    return np.clip(luminance / 255.0, 0.0, 1.0)


def _extract_features(intensity: np.ndarray, echo_strength: float, has_wake_hint: bool = False) -> Dict[str, Any]:
    if intensity.size == 0:
        return _empty_features(echo_strength)
    threshold = max(float(np.max(intensity)) * 0.52, float(np.mean(intensity) + np.std(intensity) * 1.4))
    mask = intensity > threshold
    labels, count = _label(mask)
    if count <= 0:
        return _empty_features(echo_strength)

    component_sizes = _label_sums(mask, labels, np.arange(1, count + 1))
    component_id = int(np.argmax(component_sizes)) + 1
    component = labels == component_id
    size_pixels = int(np.sum(component))
    mean_echo = float(np.mean(intensity[component])) if size_pixels else 0.0
    aspect_ratio = _aspect_ratio(component)
    has_wake = _has_wake(intensity, component) or has_wake_hint
    irregularity = _irregularity(component)
    shape = _shape_label(aspect_ratio, size_pixels, mean_echo, has_wake, irregularity)
    return {
        "shape": shape,
        "echo_strength": round(max(echo_strength, mean_echo), 3),
        "size_pixels": size_pixels,
        "aspect_ratio": round(aspect_ratio, 3),
        "has_wake": has_wake,
        "irregularity": round(irregularity, 3),
    }


def _match_rules(features: Dict[str, Any]) -> tuple[str, float, str]:
    aspect = float(features["aspect_ratio"])
    echo = float(features["echo_strength"])
    size = int(features["size_pixels"])
    has_wake = bool(features["has_wake"])
    irregularity = float(features.get("irregularity", 0.0))

    if aspect >= 3.0 and echo >= 0.8 and size <= 45:
        return "torpedo", 0.88, "细长强回波，尺寸小，符合鱼雷特征。"
    if aspect >= 2.5 and echo >= 0.7 and not has_wake and size > 45:
        return "submarine", 0.85, "长条状强回波，无尾流，符合潜艇特征。"
    if aspect < 2.2 and echo >= 0.65 and has_wake:
        return "ship", 0.80, "块状强回波伴尾流，符合水面舰特征。"
    if echo < 0.3:
        return "unknown", 0.50, "弥散弱回波，特征不明确，需继续抵近获取更清晰图像。"
    if aspect >= 2.5 and echo < 0.5:
        return "unknown", 0.50, "弥散弱回波，特征不明确，需继续抵近获取更清晰图像。"
    if irregularity >= 0.32 and echo < 0.58:
        return "reef", 0.90, "不规则弱回波，符合礁石特征。"
    if echo < 0.5 and size < 70:
        return "unknown", 0.50, "弥散弱回波，特征不明确，需继续抵近获取更清晰图像。"
    return "unknown", 0.50, "特征不明确，需继续抵近获取更清晰图像。"


def _aspect_ratio(component: np.ndarray) -> float:
    rows, cols = np.nonzero(component)
    if len(rows) < 2:
        return 1.0
    points = np.column_stack([rows.astype(np.float32), cols.astype(np.float32)])
    centered = points - np.mean(points, axis=0)
    covariance = np.cov(centered, rowvar=False)
    values = np.linalg.eigvalsh(covariance)
    major = max(float(values[-1]), 1e-6)
    minor = max(float(values[0]), 1e-6)
    return float(np.sqrt(major / minor))


def _has_wake(intensity: np.ndarray, component: np.ndarray) -> bool:
    rows, cols = np.nonzero(component)
    if len(rows) == 0:
        return False
    row_end = int(np.max(rows))
    center_col = int(round(float(np.mean(cols))))
    row_slice = slice(min(intensity.shape[0], row_end + 1), min(intensity.shape[0], row_end + 14))
    col_slice = slice(max(0, center_col - 4), min(intensity.shape[1], center_col + 5))
    if row_slice.start >= row_slice.stop:
        return False
    wake_region = intensity[row_slice, col_slice]
    background = np.median(intensity)
    return float(np.mean(wake_region)) > float(background + 0.08)


def _irregularity(component: np.ndarray) -> float:
    if not np.any(component):
        return 0.0
    filled = _binary_fill_holes(component)
    eroded = _binary_erosion(filled)
    perimeter = float(np.sum(filled ^ eroded))
    area = float(np.sum(filled))
    if area <= 0:
        return 0.0
    return perimeter / area


def _shape_label(
    aspect_ratio: float,
    size_pixels: int,
    echo_strength: float,
    has_wake: bool,
    irregularity: float,
) -> str:
    if aspect_ratio >= 4.0 and size_pixels <= 45:
        return "small_elongated"
    if aspect_ratio >= 2.5:
        return "elongated"
    if has_wake:
        return "blobby"
    if irregularity >= 0.32 and echo_strength < 0.58:
        return "irregular"
    if size_pixels < 70 and echo_strength < 0.5:
        return "scattered"
    return "blobby"


def _empty_features(echo_strength: float) -> Dict[str, Any]:
    return {
        "shape": "scattered",
        "echo_strength": round(max(0.0, echo_strength), 3),
        "size_pixels": 0,
        "aspect_ratio": 1.0,
        "has_wake": False,
        "irregularity": 0.0,
    }


def _label(mask: np.ndarray) -> tuple[np.ndarray, int]:
    if ndimage is not None:
        return ndimage.label(mask)
    labels = np.zeros(mask.shape, dtype=np.int32)
    current_label = 0
    height, width = mask.shape
    for row in range(height):
        for col in range(width):
            if not mask[row, col] or labels[row, col] != 0:
                continue
            current_label += 1
            stack = [(row, col)]
            labels[row, col] = current_label
            while stack:
                item_row, item_col = stack.pop()
                for next_row in range(max(0, item_row - 1), min(height, item_row + 2)):
                    for next_col in range(max(0, item_col - 1), min(width, item_col + 2)):
                        if mask[next_row, next_col] and labels[next_row, next_col] == 0:
                            labels[next_row, next_col] = current_label
                            stack.append((next_row, next_col))
    return labels, current_label


def _label_sums(values: np.ndarray, labels: np.ndarray, indexes: np.ndarray) -> np.ndarray:
    if ndimage is not None:
        return ndimage.sum(values, labels, index=indexes)
    return np.array([np.sum(values[labels == int(index)]) for index in indexes], dtype=np.float32)


def _binary_fill_holes(values: np.ndarray) -> np.ndarray:
    if ndimage is not None:
        return ndimage.binary_fill_holes(values)
    # The synthetic blobs are small and mostly filled already; returning the
    # component keeps the fallback deterministic without adding heavy geometry.
    return values.astype(bool)


def _binary_erosion(values: np.ndarray) -> np.ndarray:
    if ndimage is not None:
        return ndimage.binary_erosion(values)
    source = values.astype(bool)
    padded = np.pad(source, ((1, 1), (1, 1)), mode="constant", constant_values=False)
    eroded = np.ones_like(source, dtype=bool)
    for row_offset in range(3):
        for col_offset in range(3):
            eroded &= padded[row_offset : row_offset + source.shape[0], col_offset : col_offset + source.shape[1]]
    return eroded
