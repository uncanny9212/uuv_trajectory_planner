"""Natural-language task parsing for the web MVP."""

from __future__ import annotations

import copy
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from uuv_trajectory_planner.main import sample_payload


NumberPair = Tuple[float, float]


def payload_from_message(message: str) -> Dict[str, Any]:
    """Convert a short Chinese or English mission sentence into planner JSON.

    The parser is intentionally conservative: it starts from known working
    examples, then applies explicit numbers found in the user's message.
    """

    text = message.strip()
    scenario = "area_coverage" if _looks_like_coverage(text) else "general"
    payload = copy.deepcopy(sample_payload(scenario))
    payload["timestamp"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    if scenario == "area_coverage":
        _apply_coverage_details(payload, text)
    else:
        _apply_general_details(payload, text)
    return payload


def _looks_like_coverage(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in ["覆盖", "扫测", "扫描", "巡航", "cover", "coverage", "scan", "survey"])


def _apply_coverage_details(payload: Dict[str, Any], text: str) -> None:
    size = _find_area_size(text)
    if size:
        width, height = size
        polygon = [[0, 0], [width, 0], [width, height], [0, height]]
        payload["mission"]["coverage_area"] = polygon
        payload["environment"]["boundaries"] = polygon

    sweep_width = _find_named_number(text, ["扫宽", "扫测宽度", "航迹间距", "间距", "sweep", "swath", "width"])
    if sweep_width:
        payload["mission"]["constraints"]["sweep_width"] = sweep_width

    coverage_rate = _find_percentage(text)
    if coverage_rate:
        payload["mission"]["constraints"]["coverage_required"] = coverage_rate

    speed = _find_named_number(text, ["速度", "航速", "speed"])
    if speed:
        payload["mission"]["constraints"]["max_speed"] = speed
        payload["uuv_state"]["speed"] = min(speed, payload["uuv_state"].get("speed", speed))


def _apply_general_details(payload: Dict[str, Any], text: str) -> None:
    payload.setdefault("environment", {})["obstacles"] = []
    payload.setdefault("environment", {})["baits"] = []

    pairs = _coordinate_pairs(text)
    if pairs:
        start = pairs[0]
        payload["uuv_state"]["position"] = [start[0], start[1], -50.0]
    if len(pairs) >= 2:
        target = pairs[1]
        payload["mission"]["target_position"] = [target[0], target[1], -50.0]

    safety = _find_named_number(text, ["安全距离", "避障距离", "safe", "clearance"])
    if safety:
        payload["mission"]["constraints"]["min_obstacle_distance"] = safety

    speed = _find_named_number(text, ["速度", "航速", "speed"])
    if speed:
        payload["mission"]["constraints"]["max_speed"] = speed
        payload["uuv_state"]["speed"] = min(speed, payload["uuv_state"].get("speed", speed))

    obstacle_specs = _obstacle_specs(text)
    if obstacle_specs:
        payload["environment"]["obstacles"] = obstacle_specs
    bait_specs = _bait_specs(text)
    if bait_specs:
        payload["environment"]["baits"] = bait_specs


def _coordinate_pairs(text: str) -> List[NumberPair]:
    pairs: List[NumberPair] = []
    patterns = [
        r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)(?:\s*,\s*-?\d+(?:\.\d+)?)?\s*\]",
        r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)(?:\s*,\s*-?\d+(?:\.\d+)?)?\s*\)",
        r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            pair = (float(match.group(1)), float(match.group(2)))
            if pair not in pairs:
                pairs.append(pair)
        if pairs:
            break
    return pairs


def _find_area_size(text: str) -> Optional[NumberPair]:
    patterns = [
        r"(\d+(?:\.\d+)?)\s*(?:m|米)?\s*[xX×*]\s*(\d+(?:\.\d+)?)\s*(?:m|米)?",
        r"(\d+(?:\.\d+)?)\s*(?:米|m)?\s*(?:乘| by )\s*(\d+(?:\.\d+)?)\s*(?:米|m)?",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return (float(match.group(1)), float(match.group(2)))
    return None


def _find_named_number(text: str, names: Sequence[str]) -> Optional[float]:
    for name in names:
        pattern = rf"{re.escape(name)}\s*(?:为|=|:|：)?\s*(\d+(?:\.\d+)?)"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def _find_percentage(text: str) -> Optional[float]:
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if match:
        return max(0.0, min(1.0, float(match.group(1)) / 100.0))
    coverage = _find_named_number(text, ["覆盖率", "coverage"])
    if coverage:
        return coverage / 100.0 if coverage > 1.0 else coverage
    return None


def _obstacle_specs(text: str) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    pattern = (
        r"(?:障碍物|obstacle)\s*([A-Za-z0-9_-]+)?"
        r".*?\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)"
        r"(?:\s*,\s*(-?\d+(?:\.\d+)?))?\s*\)?"
        r".*?(?:半径|radius|r)\s*(?:=|:|：|为)?\s*(\d+(?:\.\d+)?)"
    )
    for index, match in enumerate(re.finditer(pattern, text, flags=re.IGNORECASE)):
        z = float(match.group(4)) if match.group(4) is not None else -50.0
        specs.append(
            {
                "id": match.group(1) or f"O{index + 1:03d}",
                "type": "static",
                "position": [float(match.group(2)), float(match.group(3)), z],
                "radius": float(match.group(5)),
            }
        )
    return specs


def _bait_specs(text: str) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    pattern = (
        r"(?:饵物|诱饵|bait)\s*([A-Za-z0-9_-]+)?"
        r".*?\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)"
        r"(?:\s*,\s*(-?\d+(?:\.\d+)?))?\s*\)?"
        r"(?:.*?(?:半径|逼近半径|radius|r)\s*(?:=|:|：|为)?\s*(\d+(?:\.\d+)?))?"
    )
    for index, match in enumerate(re.finditer(pattern, text, flags=re.IGNORECASE)):
        z = float(match.group(4)) if match.group(4) is not None else -50.0
        radius = float(match.group(5)) if match.group(5) is not None else 40.0
        specs.append(
            {
                "id": match.group(1) or f"B{index + 1:03d}",
                "position": [float(match.group(2)), float(match.group(3)), z],
                "radius": radius,
            }
        )
    return specs
