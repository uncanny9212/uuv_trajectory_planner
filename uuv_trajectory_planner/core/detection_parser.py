"""Parse UUV detection semantics into situation-awareness payloads."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from uuv_trajectory_planner.models.situation_awareness import SituationAwareness


Vector3 = Tuple[float, float, float]

_NUMBER = r"[-+]?\d+(?:\.\d+)?"
_COORDINATE_RE = re.compile(
    rf"[\(\[（]\s*({_NUMBER})\s*[,，]\s*({_NUMBER})"
    rf"(?:\s*[,，]\s*({_NUMBER}))?\s*[\)\]）]"
)


@dataclass(frozen=True)
class _DetectedTarget:
    """Internal target candidate extracted from detection text."""

    position: Vector3
    context: str
    radius: Optional[float] = None


class DetectionParser:
    """UUV detection semantic parser.

    Converts natural-language sonar/radar target descriptions into the same
    dictionary shape accepted by ``SituationAwareness.from_dict()``.
    """

    def __init__(self, default_detection_range: float = 500.0, default_depth: float = -50.0):
        """Create a parser with defaults used when sensor text is incomplete.

        Args:
            default_detection_range: Range in meters used when a bearing is
                present but no range is reported.
            default_depth: Depth in meters, expressed as a negative z value.
        """

        self.default_detection_range = float(default_detection_range)
        self.default_depth = -abs(float(default_depth))

    def parse(
        self,
        detection_text: str,
        uuv_position: Sequence[float] = (0.0, 0.0, -50.0),
    ) -> Dict[str, Any]:
        """Parse UUV detection text into a SituationAwareness-compatible payload.

        Args:
            detection_text: Natural-language UUV target detection description.
            uuv_position: Current UUV position ``(x, y, z)`` used for
                bearing-to-coordinate conversion.

        Returns:
            A dictionary accepted by ``SituationAwareness.from_dict()``.

        Raises:
            ValueError: If no usable target or coverage area can be parsed.
        """

        text = detection_text.strip()
        if not text:
            raise ValueError("detection_text must not be empty")

        current_position = self._extract_uuv_position(text) or self._normalize_position(uuv_position)
        scenario = self._detect_scenario(text)
        targets = self._extract_targets(text, current_position)
        classified = self._classify_targets(targets)
        coverage_area = self._extract_coverage_area(text)
        boundaries = coverage_area or [[0, 0], [1200, 0], [1200, 1000], [0, 1000]]

        mission: Dict[str, Any] = {
            "type": "trajectory_planning",
            "scenario": scenario,
            "constraints": {
                "max_speed": 3.0,
                "min_obstacle_distance": 50.0,
            },
        }
        orbit_turns = self._extract_orbit_turns(text)
        if orbit_turns:
            mission["constraints"]["orbit_turns"] = orbit_turns
        orbit_radius = self._extract_orbit_radius(text)
        if orbit_radius is not None:
            mission["constraints"]["orbit_radius"] = orbit_radius

        if scenario == "area_coverage":
            mission["coverage_area"] = boundaries
        else:
            if not classified["baits"]:
                raise ValueError("unable to parse an approach target from detection semantics")
            mission["target_position"] = classified["baits"][0]["position"]

        first_bearing, _ = self._extract_bearing_distance(text)
        payload = {
            "timestamp": self._extract_timestamp(text),
            "uuv_state": {
                "position": list(current_position),
                "heading": float(first_bearing or 0.0),
                "speed": 2.0,
                "battery": 1.0,
            },
            "mission": mission,
            "environment": {
                "obstacles": classified["obstacles"],
                "baits": classified["baits"],
                "boundaries": boundaries,
                "water_current": [0.0, 0.0, 0.0],
            },
        }

        SituationAwareness.from_dict(payload)
        return payload

    def _extract_bearing_distance(self, text: str) -> Tuple[Optional[float], Optional[float]]:
        """Extract bearing and distance from a detection sentence.

        Returns:
            ``(bearing_deg, distance_m)`` when a bearing is found. Distance is
            ``None`` when it is not explicitly present. Returns ``(None, None)``
            when no bearing can be parsed.
        """

        bearing = self._extract_bearing(text)
        if bearing is None:
            return (None, None)
        return (bearing, self._extract_distance(text))

    def _estimate_coordinates(
        self,
        bearing_deg: float,
        distance: float,
        uuv_position: Sequence[float],
    ) -> Vector3:
        """Estimate target coordinates from bearing/range and UUV position.

        Args:
            bearing_deg: Bearing in degrees where 0 is north and values
                increase clockwise.
            distance: Range in meters.
            uuv_position: Current UUV ``(x, y, z)`` position.

        Returns:
            Estimated absolute ``(x, y, z)`` coordinate.
        """

        current = self._normalize_position(uuv_position)
        math_angle_deg = 90.0 - bearing_deg
        math_angle_rad = math.radians(math_angle_deg)
        x = current[0] + distance * math.cos(math_angle_rad)
        y = current[1] + distance * math.sin(math_angle_rad)
        return (round(x, 3), round(y, 3), current[2])

    def _extract_depth(self, text: str) -> float:
        """Extract depth as a negative z value, or return the parser default."""

        patterns = [
            rf"(?:水深|深度|海面下|depth)\s*(?:为|=|:|：)?\s*(?:约|大约|大概|approximately|approx\.?|around)?\s*({_NUMBER})\s*(?:米|m|meters?)?",
            rf"(?:at\s+)?({_NUMBER})\s*(?:米|m|meters?)\s*(?:below\s+surface|underwater)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return -abs(float(match.group(1)))
        return self.default_depth

    def _classify_targets(self, targets: Sequence[_DetectedTarget]) -> Dict[str, List[Dict[str, Any]]]:
        """Classify detected targets into planner baits and obstacles."""

        baits: List[Dict[str, Any]] = []
        obstacles: List[Dict[str, Any]] = []
        for target in targets:
            if self._is_obstacle_context(target.context):
                obstacle: Dict[str, Any] = {
                    "id": f"O{len(obstacles) + 1:03d}",
                    "type": "moving" if self._is_moving_context(target.context) else "static",
                    "position": list(target.position),
                    "radius": float(target.radius if target.radius is not None else 50.0),
                }
                if obstacle["type"] == "moving":
                    obstacle["velocity"] = [0.0, 0.0, 0.0]
                obstacles.append(obstacle)
            else:
                baits.append(
                    {
                        "id": f"B{len(baits) + 1:03d}",
                        "position": list(target.position),
                        "radius": float(target.radius if target.radius is not None else 40.0),
                    }
                )
        return {"baits": baits, "obstacles": obstacles}

    def _extract_targets(self, text: str, uuv_position: Vector3) -> List[_DetectedTarget]:
        coordinate_targets = self._extract_coordinate_targets(text)
        bearing_targets = self._extract_bearing_targets(text, uuv_position, has_coordinates=bool(coordinate_targets))
        targets = coordinate_targets + bearing_targets
        if targets:
            return targets
        if self._detect_scenario(text) == "area_coverage":
            return []
        raise ValueError("unable to parse target position or bearing from detection semantics")

    def _extract_coordinate_targets(self, text: str) -> List[_DetectedTarget]:
        targets: List[_DetectedTarget] = []
        global_depth = self._extract_depth(text)

        for match in _COORDINATE_RE.finditer(text):
            context = self._coordinate_context(text, match.start(), match.end())
            prefix_context = self._prefix_context(text, match.start())
            if self._is_uuv_context(prefix_context) or self._is_coverage_context(prefix_context):
                continue
            z = float(match.group(3)) if match.group(3) is not None else self._extract_depth(context)
            if match.group(3) is None and z == self.default_depth:
                z = global_depth
            targets.append(
                _DetectedTarget(
                    position=(float(match.group(1)), float(match.group(2)), z),
                    context=context,
                    radius=self._extract_radius(context),
                )
            )
        return targets

    def _extract_bearing_targets(
        self,
        text: str,
        uuv_position: Vector3,
        has_coordinates: bool,
    ) -> List[_DetectedTarget]:
        targets: List[_DetectedTarget] = []
        segments = self._target_segments(text)

        if not has_coordinates:
            compact_targets = self._extract_compact_bearing_targets(text, uuv_position)
            if len(compact_targets) > 1:
                return compact_targets

            bearing_segments = [
                segment
                for segment in segments
                if self._looks_like_target_segment(segment) and self._extract_bearing_distance(segment)[0] is not None
            ]
            if not bearing_segments:
                bearing_segments = [text]

            for segment in bearing_segments:
                bearing, distance = self._extract_bearing_distance(segment)
                if bearing is None:
                    continue
                range_m = distance if distance is not None else self.default_detection_range
                position = self._estimate_coordinates(bearing, range_m, uuv_position)
                depth = self._extract_depth(segment)
                targets.append(
                    _DetectedTarget(
                        position=(position[0], position[1], depth),
                        context=segment,
                        radius=self._extract_radius(segment),
                    )
                )
            return targets

        for segment in segments:
            if _COORDINATE_RE.search(segment) or not self._looks_like_target_segment(segment):
                continue
            bearing, distance = self._extract_bearing_distance(segment)
            if bearing is None:
                continue
            range_m = distance if distance is not None else self.default_detection_range
            position = self._estimate_coordinates(bearing, range_m, uuv_position)
            depth = self._extract_depth(segment)
            targets.append(
                _DetectedTarget(
                    position=(position[0], position[1], depth),
                    context=segment,
                    radius=self._extract_radius(segment),
                )
            )
        return targets

    def _extract_compact_bearing_targets(self, text: str, uuv_position: Vector3) -> List[_DetectedTarget]:
        targets: List[_DetectedTarget] = []
        occupied: List[Tuple[int, int]] = []

        chinese_pattern = re.compile(
            rf"([东西南北]\s*偏\s*[东西南北]\s*{_NUMBER}\s*(?:度|°)?\s*(?:方向|方位)?)"
            rf"\s*(?:距离|航程|range|distance)?\s*(?:约|大约|大概)?\s*({_NUMBER})\s*(?:米|m|meters?)?"
            rf"(?:\s*(?:距离|范围|附近|左右))?",
            flags=re.IGNORECASE,
        )
        numeric_pattern = re.compile(
            rf"(?<![\d.])({_NUMBER})\s*(?:度|°)\s*(?:方向|方位)?"
            rf"\s*(?:距离|航程|range|distance)?\s*(?:约|大约|大概)?\s*({_NUMBER})\s*(?:米|m|meters?)?"
            rf"(?:\s*(?:距离|范围|附近|左右))?",
            flags=re.IGNORECASE,
        )

        for match in chinese_pattern.finditer(text):
            target = self._target_from_bearing_match(match, uuv_position)
            if target is not None:
                targets.append(target)
                occupied.append((match.start(), match.end()))

        for match in numeric_pattern.finditer(text):
            if any(start <= match.start() < end for start, end in occupied):
                continue
            target = self._target_from_bearing_match(match, uuv_position)
            if target is not None:
                targets.append(target)

        targets.sort(key=lambda target: text.find(target.context))
        if len(targets) <= 1:
            return []
        return targets

    def _target_from_bearing_match(
        self,
        match: re.Match[str],
        uuv_position: Vector3,
    ) -> Optional[_DetectedTarget]:
        context = match.group(0)
        bearing = self._extract_bearing(context)
        if bearing is None:
            return None
        distance = float(match.group(2))
        position = self._estimate_coordinates(bearing, distance, uuv_position)
        return _DetectedTarget(
            position=(position[0], position[1], self._extract_depth(context)),
            context=context,
            radius=self._extract_radius(context),
        )

    def _extract_bearing(self, text: str) -> Optional[float]:
        chinese = self._extract_chinese_bearing(text)
        if chinese is not None:
            return chinese

        numeric_patterns = [
            rf"(?<![\d.])({_NUMBER})\s*(?:度|°|degrees?|deg)\s*(?:方向|方位|bearing|azimuth|heading)?",
            rf"(?:方位角|方位|方向|bearing|azimuth|heading)\s*(?:为|=|:|：)?\s*({_NUMBER})\s*(?:度|°|degrees?|deg)?",
        ]
        for pattern in numeric_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return float(match.group(1)) % 360.0

        if self._looks_like_target_segment(text):
            match = re.search(rf"(?<![\d.])({_NUMBER})\s*(?:度|°)\s*(?:方向|方位|处)?", text)
            if match:
                return float(match.group(1)) % 360.0

        lowered = text.lower()
        cardinals = {
            "northeast": 45.0,
            "north east": 45.0,
            "southeast": 135.0,
            "south east": 135.0,
            "southwest": 225.0,
            "south west": 225.0,
            "northwest": 315.0,
            "north west": 315.0,
            "north": 0.0,
            "east": 90.0,
            "south": 180.0,
            "west": 270.0,
        }
        for word, value in cardinals.items():
            if re.search(rf"\b{re.escape(word)}\b", lowered):
                return value
        return None

    def _extract_chinese_bearing(self, text: str) -> Optional[float]:
        direction_map = {
            ("北", "东"): lambda value: value,
            ("北", "西"): lambda value: 360.0 - value,
            ("南", "东"): lambda value: 180.0 - value,
            ("南", "西"): lambda value: 180.0 + value,
            ("东", "北"): lambda value: 90.0 - value,
            ("东", "南"): lambda value: 90.0 + value,
            ("西", "北"): lambda value: 270.0 + value,
            ("西", "南"): lambda value: 270.0 - value,
        }
        match = re.search(rf"([东西南北])\s*偏\s*([东西南北])\s*({_NUMBER})\s*(?:度|°)?", text)
        if match:
            first, second, amount = match.group(1), match.group(2), float(match.group(3))
            converter = direction_map.get((first, second))
            if converter is not None:
                return converter(amount) % 360.0

        exact = {
            "正北": 0.0,
            "北方": 0.0,
            "正东": 90.0,
            "东方": 90.0,
            "正南": 180.0,
            "南方": 180.0,
            "正西": 270.0,
            "西方": 270.0,
            "东北": 45.0,
            "东南": 135.0,
            "西南": 225.0,
            "西北": 315.0,
        }
        for word, value in exact.items():
            if word in text:
                return value
        return None

    def _extract_distance(self, text: str) -> Optional[float]:
        patterns = [
            rf"(?:距离|航程|range|distance)\s*(?:为|=|:|：)?\s*(?:约|大约|大概|approximately|approx\.?|around)?\s*({_NUMBER})\s*(?:米|m|meters?)?",
            rf"({_NUMBER})\s*(?:米|m|meters?)\s*(?:处|away|range)",
            rf"({_NUMBER})\s*(?:米|m|meters?)\s*(?:距离|范围|附近|左右)",
            rf"(?:方向|方位)\s*(?:约|大约|大概)?\s*({_NUMBER})\s*(?:米|m|meters?)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return float(match.group(1))
        return None

    def _extract_radius(self, text: str) -> Optional[float]:
        patterns = [
            rf"(?:半径|radius|r)\s*(?:为|=|:|：)?\s*(?:约|大约|approximately|approx\.?|around)?\s*({_NUMBER})\s*(?:米|m|meters?)?",
            rf"({_NUMBER})\s*(?:米|m|meters?)\s*(?:半径|radius)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return float(match.group(1))
        return None

    def _extract_orbit_turns(self, text: str) -> int:
        patterns = [
            rf"(?:围着|围绕|环绕|绕行|绕)\S{{0,12}}?([一二两三四五六七八九十\d]+)\s*(?:圈|周|circle|circles|turns?)",
            rf"([一二两三四五六七八九十\d]+)\s*(?:圈|周)\S{{0,8}}?(?:环绕|绕行|绕)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return self._parse_count(match.group(1))
        if self._is_close_recon_context(text):
            return 2
        return 0

    def _extract_orbit_radius(self, text: str) -> Optional[float]:
        patterns = [
            rf"(?:环绕半径|绕行半径|绕圈半径|orbit\s+radius|circle\s+radius)\s*(?:为|=|:|：)?\s*(?:约|大约)?\s*({_NUMBER})\s*(?:米|m)?",
            rf"(?:以|按)\s*(?:半径)?\s*({_NUMBER})\s*(?:米|m)\s*(?:半径)?\s*(?:围绕|环绕|绕行|绕)",
            rf"(?:周边|周围|附近)\s*({_NUMBER})\s*(?:米|m)\s*(?:左右|范围|区域)?\S{{0,10}}?(?:围绕|环绕|绕行|绕|侦察)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return float(match.group(1))
        if self._is_close_recon_context(text):
            return 10.0
        return None

    def _is_close_recon_context(self, text: str) -> bool:
        lowered = text.lower()
        keywords = [
            "抵近侦察",
            "抵近侦查",
            "抵近观察",
            "抵近探测",
            "近距侦察",
            "近距侦查",
            "close reconnaissance",
            "close recon",
            "close inspection",
        ]
        if any(keyword in lowered for keyword in keywords):
            return True
        return bool(re.search(r"抵近\S{0,6}(?:侦察|侦查|观察|探测)", text))

    def _parse_count(self, value: str) -> int:
        if re.fullmatch(r"\d+", value):
            return max(0, int(value))
        digits = {
            "一": 1,
            "二": 2,
            "两": 2,
            "三": 3,
            "四": 4,
            "五": 5,
            "六": 6,
            "七": 7,
            "八": 8,
            "九": 9,
        }
        if value == "十":
            return 10
        if value.startswith("十"):
            return 10 + digits.get(value[1:], 0)
        if value.endswith("十"):
            return digits.get(value[:-1], 0) * 10
        if "十" in value:
            tens, ones = value.split("十", 1)
            return digits.get(tens, 1) * 10 + digits.get(ones, 0)
        return digits.get(value, 0)

    def _extract_timestamp(self, text: str) -> str:
        iso_match = re.search(
            r"(\d{4}-\d{2}-\d{2})[T\s](\d{2}:\d{2}(?::\d{2})?)(?:\s*(Z|[+-]\d{2}:?\d{2}))?",
            text,
        )
        if iso_match:
            time_part = iso_match.group(2)
            if len(time_part) == 5:
                time_part = f"{time_part}:00"
            offset = iso_match.group(3) or "Z"
            if offset != "Z" and len(offset) == 5 and ":" not in offset:
                offset = f"{offset[:3]}:{offset[3:]}"
            return f"{iso_match.group(1)}T{time_part}{offset}"
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    def _extract_uuv_position(self, text: str) -> Optional[Vector3]:
        pattern = re.compile(
            r"(?:当前位置|本艇位置|UUV位置|起始点|起点|从|current\s+position|start|starting\s+point|ownship|uuv\s+position)"
            rf".{{0,20}}?{_COORDINATE_RE.pattern}",
            flags=re.IGNORECASE,
        )
        match = pattern.search(text)
        if not match:
            return None
        groups = match.groups()
        z = float(groups[2]) if groups[2] is not None else self._extract_depth(match.group(0))
        return (float(groups[0]), float(groups[1]), z)

    def _extract_coverage_area(self, text: str) -> List[List[float]]:
        size_patterns = [
            rf"({_NUMBER})\s*(?:m|米)?\s*[xX×*]\s*({_NUMBER})\s*(?:m|米)?",
            rf"({_NUMBER})\s*(?:米|m)?\s*(?:乘|by)\s*({_NUMBER})\s*(?:米|m)?",
        ]
        for pattern in size_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                width, height = float(match.group(1)), float(match.group(2))
                return [[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]]

        if self._detect_scenario(text) != "area_coverage":
            return []

        points: List[List[float]] = []
        for match in _COORDINATE_RE.finditer(text):
            context = self._local_context(text, match.start(), match.end())
            if self._is_uuv_context(context):
                continue
            points.append([float(match.group(1)), float(match.group(2))])
        return points if len(points) >= 3 else []

    def _detect_scenario(self, text: str) -> str:
        lowered = text.lower()
        if any(keyword in lowered for keyword in ["覆盖", "扫描", "扫测", "搜索区域", "cover", "coverage", "scan", "survey"]):
            return "area_coverage"
        return "general"

    def _target_segments(self, text: str) -> List[str]:
        segments: List[str] = []
        for line in text.splitlines():
            cleaned = line.strip(" \t-*")
            if not cleaned:
                continue
            parts = re.split(r"(?=\b\d+[.、)]\s*)", cleaned)
            for part in parts:
                segment = part.strip(" \t-*")
                if not segment:
                    continue
                segments.extend(self._split_compact_target_segments(segment))
        return segments

    def _split_compact_target_segments(self, text: str) -> List[str]:
        pieces = [piece.strip(" \t-*") for piece in re.split(r"[、；;。]+", text) if piece.strip(" \t-*")]
        if len(pieces) <= 1:
            return [text]
        if all(self._looks_like_target_segment(piece) and self._extract_bearing(piece) is not None for piece in pieces):
            return pieces
        return [text]

    def _local_context(self, text: str, start: int, end: int) -> str:
        left = 0
        for delimiter in ["\n", "；", ";", "。", "，以及", "以及", " and "]:
            index = text.rfind(delimiter, 0, start)
            if index >= 0:
                left = max(left, index + len(delimiter))

        right = len(text)
        for delimiter in ["\n", "；", ";", "。", "，以及", "以及", " and "]:
            index = text.find(delimiter, end)
            if index >= 0:
                right = min(right, index)
        return text[left:right].strip(" \t-*")

    def _prefix_context(self, text: str, start: int) -> str:
        left = 0
        for delimiter in ["\n", "；", ";", "。", "，", ","]:
            index = text.rfind(delimiter, 0, start)
            if index >= 0:
                left = max(left, index + len(delimiter))
        return text[left:start].strip(" \t-*")

    def _coordinate_context(self, text: str, start: int, end: int) -> str:
        prefix = self._prefix_context(text, start)
        suffix = self._coordinate_suffix(text, end)
        context = f"{prefix}{suffix}".strip(" \t-*，,")
        return context or self._local_context(text, start, end)

    def _coordinate_suffix(self, text: str, start: int) -> str:
        suffix_parts: List[str] = []
        cursor = start
        while cursor < len(text):
            delimiter = re.match(r"\s*(，|,|、|；|;|。|\n|以及|and)\s*", text[cursor:], flags=re.IGNORECASE)
            if not delimiter:
                break
            marker = delimiter.group(1)
            if marker in {"；", ";", "。", "\n", "以及", "and"}:
                break

            segment_start = cursor + delimiter.end()
            next_delimiter = re.search(r"(，|,|、|；|;|。|\n|以及|and)", text[segment_start:], flags=re.IGNORECASE)
            segment_end = segment_start + next_delimiter.start() if next_delimiter else len(text)
            segment = text[segment_start:segment_end].strip(" \t-*")
            if not segment or self._starts_new_coordinate_target(segment):
                break
            if not self._is_coordinate_attribute(segment):
                break

            suffix_parts.append(segment)
            cursor = segment_end
        return "，".join(suffix_parts)

    def _starts_new_coordinate_target(self, text: str) -> bool:
        if _COORDINATE_RE.search(text):
            return True
        patterns = [
            r"^(?:\d+[.、)]\s*)?(?:目标|接触|礁石|浅滩|沉船|障碍物)[A-Za-z0-9_-]*\s*(?:在|位于|位置|方向|方位|:|：)",
            r"^(?:target|contact|reef|shoal|wreck|obstacle|hazard)[A-Za-z0-9_-]*\s*(?:at|position|bearing|range|:)",
        ]
        return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)

    def _is_coordinate_attribute(self, text: str) -> bool:
        lowered = text.lower()
        keywords = [
            "半径",
            "radius",
            "置信度",
            "confidence",
            "类型",
            "type",
            "疑似",
            "可能",
            "静止",
            "固定",
            "移动",
            "运动",
            "moving",
            "static",
            "强度",
            "备注",
            "水深",
            "深度",
            "depth",
            "速度",
            "velocity",
            "礁石",
            "浅滩",
            "沉船",
            "障碍",
            "reef",
            "shoal",
            "wreck",
            "obstacle",
            "hazard",
        ]
        return any(keyword in lowered for keyword in keywords)

    def _normalize_position(self, position: Sequence[float]) -> Vector3:
        values = [float(value) for value in position]
        if len(values) < 2:
            raise ValueError("uuv_position must contain at least x and y")
        z = values[2] if len(values) >= 3 else self.default_depth
        return (values[0], values[1], z)

    def _looks_like_target_segment(self, text: str) -> bool:
        lowered = text.lower()
        keywords = [
            "目标",
            "接触",
            "疑似",
            "探测",
            "target",
            "contact",
            "detected",
            "possible",
            "obstacle",
            "reef",
        ]
        return any(keyword in lowered for keyword in keywords) or bool(re.match(r"\d+[.、)]", text))

    def _is_obstacle_context(self, text: str) -> bool:
        lowered = text.lower()
        obstacle_keywords = [
            "礁石",
            "浅滩",
            "沉船",
            "固定障碍",
            "障碍物",
            "reef",
            "shoal",
            "wreck",
            "rock",
            "obstacle",
            "hazard",
        ]
        return any(keyword in lowered for keyword in obstacle_keywords)

    def _is_moving_context(self, text: str) -> bool:
        lowered = text.lower()
        return any(keyword in lowered for keyword in ["移动", "运动", "moving", "dynamic"])

    def _is_uuv_context(self, text: str) -> bool:
        lowered = text.lower()
        keywords = [
            "当前位置",
            "本艇位置",
            "uuv位置",
            "起始点",
            "起点",
            "出发",
            "current position",
            "start",
            "starting point",
            "ownship",
            "uuv position",
        ]
        return any(keyword in lowered for keyword in keywords)

    def _is_coverage_context(self, text: str) -> bool:
        lowered = text.lower()
        keywords = ["覆盖区域", "搜索区域", "扫描区域", "coverage area", "search area", "survey area"]
        return any(keyword in lowered for keyword in keywords)


def parse_detection_text(
    detection_text: str,
    uuv_position: Sequence[float] = (0.0, 0.0, -50.0),
    default_detection_range: float = 500.0,
    default_depth: float = -50.0,
) -> Dict[str, Any]:
    """Convenience wrapper for one-shot UUV detection semantic parsing."""

    parser = DetectionParser(
        default_detection_range=default_detection_range,
        default_depth=default_depth,
    )
    return parser.parse(detection_text, uuv_position=uuv_position)


__all__ = ["DetectionParser", "parse_detection_text"]
