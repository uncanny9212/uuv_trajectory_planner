"""Rolling next-step planner for passive-bearing UUV missions."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

Vector3 = Tuple[float, float, float]
TrendName = Literal["increasing", "decreasing", "stable", "unknown"]
DecisionName = Literal["adjust_heading", "advance", "approach_complete", "orbit", "give_up", "wait"]


def _normalize_angle(angle: float) -> float:
    return float(angle) % 360.0


def _signed_angle_delta(start: float, end: float) -> float:
    """Return the shortest signed delta from start to end in degrees."""

    return (_normalize_angle(end) - _normalize_angle(start) + 180.0) % 360.0 - 180.0


def _vector3(values: Optional[Sequence[Any]], default: Vector3 = (0.0, 0.0, -50.0)) -> Vector3:
    if values is None:
        return default
    if len(values) < 3:
        raise ValueError("uuv_state.position must contain x, y and z")
    return (float(values[0]), float(values[1]), float(values[2]))


def _parse_timestamp_seconds(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return None

    iso_text = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        return datetime.fromisoformat(iso_text).timestamp()
    except ValueError:
        pass

    parts = text.split(":")
    if len(parts) == 3:
        try:
            hours, minutes, seconds = parts
            return int(hours) * 3600.0 + int(minutes) * 60.0 + float(seconds)
        except ValueError:
            return None
    return None


@dataclass
class BearingObservation:
    """One passive-bearing observation."""

    angle: float
    timestamp: Optional[Any] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BearingObservation":
        if "angle" not in data:
            raise ValueError("bearing observation requires angle")
        return cls(angle=_normalize_angle(float(data["angle"])), timestamp=data.get("timestamp"))


@dataclass
class RollingConstraints:
    """Configurable thresholds for rolling next-step decisions."""

    approach_range: float = 100.0
    orbit_turns: int = 2
    orbit_radius: float = 10.0
    max_iterations: int = 50
    default_step: float = 300.0
    observation_interval_seconds: float = 30.0
    stable_angle_threshold: float = 5.0
    max_heading_correction: float = 10.0
    battery_give_up_threshold: float = 0.2

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "RollingConstraints":
        values = data or {}
        return cls(
            approach_range=max(0.0, float(values.get("approach_range", cls.approach_range))),
            orbit_turns=max(0, int(float(values.get("orbit_turns", cls.orbit_turns)))),
            orbit_radius=max(0.0, float(values.get("orbit_radius", cls.orbit_radius))),
            max_iterations=max(0, int(float(values.get("max_iterations", cls.max_iterations)))),
            default_step=max(0.0, float(values.get("default_step", cls.default_step))),
            observation_interval_seconds=max(
                1.0,
                float(values.get("observation_interval_seconds", cls.observation_interval_seconds)),
            ),
            stable_angle_threshold=max(
                0.0,
                float(values.get("stable_angle_threshold", cls.stable_angle_threshold)),
            ),
            max_heading_correction=max(
                0.0,
                float(values.get("max_heading_correction", cls.max_heading_correction)),
            ),
            battery_give_up_threshold=max(
                0.0,
                min(1.0, float(values.get("battery_give_up_threshold", cls.battery_give_up_threshold))),
            ),
        )


@dataclass
class UUVRollingState:
    """Current UUV navigation state used by the rolling planner."""

    position: Vector3 = (0.0, 0.0, -50.0)
    heading: float = 0.0
    speed: float = 1.5
    battery: float = 1.0

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "UUVRollingState":
        values = data or {}
        return cls(
            position=_vector3(values.get("position")),
            heading=_normalize_angle(float(values.get("heading", 0.0))),
            speed=max(0.0, float(values.get("speed", 1.5))),
            battery=max(0.0, min(1.0, float(values.get("battery", 1.0)))),
        )


@dataclass
class BearingTrend:
    """Trend summary derived from a bearing history."""

    trend: TrendName
    rate: float
    confidence: float
    delta: float
    sample_count: int


@dataclass
class RollingDecision:
    """Next-step decision emitted by the rolling planner."""

    decision: DecisionName
    next_heading: float
    advance_distance: float
    expected_duration: float
    reasoning: str
    confidence: float
    mode: str
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.next_heading = _normalize_angle(self.next_heading)
        self.advance_distance = max(0.0, self.advance_distance)
        self.expected_duration = max(0.0, self.expected_duration)
        self.confidence = max(0.0, min(1.0, self.confidence))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "next_heading": round(self.next_heading, 3),
            "advance_distance": round(self.advance_distance, 3),
            "expected_duration": round(self.expected_duration, 3),
            "reasoning": self.reasoning,
            "confidence": round(self.confidence, 3),
            "mode": self.mode,
            "warnings": self.warnings,
        }


class RollingPlanner:
    """Make one incremental UUV decision from passive bearing history."""

    def plan(self, payload: Dict[str, Any]) -> RollingDecision:
        """Return a next-step decision for the current rolling-planning cycle."""

        observations = self._bearing_history(payload.get("bearing_history", []))
        state = UUVRollingState.from_dict(payload.get("uuv_state"))
        constraints = RollingConstraints.from_dict(payload.get("constraints"))
        estimated_range_missing = payload.get("estimated_range") is None
        estimated_range = self._estimated_range(payload, constraints)
        iteration = self._iteration(payload)
        mission_context = str(payload.get("mission_context", "抵近侦察"))

        trend = self.analyze_bearing_trend(observations, constraints)
        estimated_distance = self.estimate_distance(estimated_range, trend)
        warnings: List[str] = []
        if estimated_range_missing:
            warnings.append("未提供距离估计，使用默认步长外推")

        if state.battery < constraints.battery_give_up_threshold:
            warnings.append("电量低于安全阈值")
            return self._decision(
                decision="give_up",
                state=state,
                observations=observations,
                trend=trend,
                estimated_distance=estimated_distance,
                constraints=constraints,
                iteration=iteration,
                mission_context=mission_context,
                next_heading=(state.heading + 180.0) % 360.0,
                advance_distance=0.0,
                confidence=0.9,
                mode="abort",
                warnings=warnings,
            )

        if constraints.max_iterations and iteration >= constraints.max_iterations:
            warnings.append("已达到最大滚动迭代次数")
            return self._decision(
                decision="give_up",
                state=state,
                observations=observations,
                trend=trend,
                estimated_distance=estimated_distance,
                constraints=constraints,
                iteration=iteration,
                mission_context=mission_context,
                next_heading=(state.heading + 180.0) % 360.0,
                advance_distance=0.0,
                confidence=0.78,
                mode="abort",
                warnings=warnings,
            )

        if estimated_distance <= constraints.approach_range:
            decision: DecisionName = "orbit" if constraints.orbit_turns > 0 else "approach_complete"
            mode = "orbit" if decision == "orbit" else "approach_complete"
            next_heading = self._orbit_heading(state, observations)
            duration = self._orbit_duration(state, constraints) if decision == "orbit" else 0.0
            warnings.append("估算距离已进入抵近阈值")
            return self._decision(
                decision=decision,
                state=state,
                observations=observations,
                trend=trend,
                estimated_distance=estimated_distance,
                constraints=constraints,
                iteration=iteration,
                mission_context=mission_context,
                next_heading=next_heading,
                advance_distance=0.0,
                expected_duration=duration,
                confidence=max(0.82, trend.confidence),
                mode=mode,
                warnings=warnings,
            )

        if trend.trend == "unknown":
            warnings.append("方位观测不足，建议等待更多信息")
            return self._decision(
                decision="wait",
                state=state,
                observations=observations,
                trend=trend,
                estimated_distance=estimated_distance,
                constraints=constraints,
                iteration=iteration,
                mission_context=mission_context,
                next_heading=state.heading,
                advance_distance=0.0,
                confidence=0.35,
                mode="observe",
                warnings=warnings,
            )

        if trend.trend == "stable" and iteration > 10:
            warnings.append("方位长期稳定但未进入抵近阈值，需要重新评估目标")
            return self._decision(
                decision="give_up",
                state=state,
                observations=observations,
                trend=trend,
                estimated_distance=estimated_distance,
                constraints=constraints,
                iteration=iteration,
                mission_context=mission_context,
                next_heading=(state.heading + 180.0) % 360.0,
                advance_distance=0.0,
                confidence=0.62,
                mode="abort",
                warnings=warnings,
            )

        if trend.trend == "stable" and iteration > 3:
            warnings.append("方位稳定但迭代次数偏多，建议暂停获取补充观测")
            return self._decision(
                decision="wait",
                state=state,
                observations=observations,
                trend=trend,
                estimated_distance=estimated_distance,
                constraints=constraints,
                iteration=iteration,
                mission_context=mission_context,
                next_heading=state.heading,
                advance_distance=0.0,
                confidence=0.58,
                mode="observe",
                warnings=warnings,
            )

        next_heading = self.decide_heading(state, trend, constraints)
        advance_distance = self._advance_distance(estimated_distance, constraints)
        if trend.trend == "stable":
            decision = "advance"
            confidence = min(0.9, max(0.68, trend.confidence))
        else:
            decision = "adjust_heading"
            confidence = min(0.92, max(0.55, trend.confidence))

        return self._decision(
            decision=decision,
            state=state,
            observations=observations,
            trend=trend,
            estimated_distance=estimated_distance,
            constraints=constraints,
            iteration=iteration,
            mission_context=mission_context,
            next_heading=next_heading,
            advance_distance=advance_distance,
            confidence=confidence,
            mode="approach",
            warnings=warnings,
        )

    def analyze_bearing_trend(
        self,
        bearing_history: Sequence[BearingObservation],
        constraints: Optional[RollingConstraints] = None,
    ) -> BearingTrend:
        """Analyze whether bearing angles are increasing, decreasing or stable."""

        settings = constraints or RollingConstraints()
        if len(bearing_history) < 2:
            return BearingTrend("unknown", 0.0, 0.0, 0.0, len(bearing_history))

        unwrapped = [bearing_history[0].angle]
        step_deltas: List[float] = []
        for previous, current in zip(bearing_history[:-1], bearing_history[1:]):
            delta = _signed_angle_delta(previous.angle, current.angle)
            step_deltas.append(delta)
            unwrapped.append(unwrapped[-1] + delta)

        net_delta = unwrapped[-1] - unwrapped[0]
        elapsed = self._elapsed_seconds(bearing_history, settings)
        rate = net_delta / elapsed if elapsed > 0.0 else 0.0
        sample_factor = min(0.14, max(0, len(bearing_history) - 2) * 0.04)

        if abs(net_delta) < settings.stable_angle_threshold:
            confidence = min(0.94, 0.76 + sample_factor)
            return BearingTrend("stable", 0.0, confidence, net_delta, len(bearing_history))

        direction = 1 if net_delta > 0.0 else -1
        useful_steps = [delta for delta in step_deltas if abs(delta) >= 0.5]
        if useful_steps:
            same_direction = sum(1 for delta in useful_steps if delta * direction > 0.0)
            consistency = same_direction / len(useful_steps)
        else:
            consistency = 0.0
        confidence = min(0.95, 0.55 + sample_factor + consistency * 0.18)
        trend: TrendName = "increasing" if net_delta > 0.0 else "decreasing"
        return BearingTrend(trend, rate, confidence, net_delta, len(bearing_history))

    def estimate_distance(self, estimated_range: float, trend: BearingTrend) -> float:
        """Estimate current range from the latest external range guess and bearing trend."""

        if trend.trend == "stable":
            return estimated_range * 0.7
        return estimated_range

    def decide_heading(
        self,
        state: UUVRollingState,
        trend: BearingTrend,
        constraints: Optional[RollingConstraints] = None,
    ) -> float:
        """Choose the next heading with a bounded tangent-rule correction."""

        settings = constraints or RollingConstraints()
        if trend.trend == "stable":
            return state.heading
        if trend.trend == "unknown":
            return state.heading

        correction = min(settings.max_heading_correction, max(2.0, abs(trend.delta) * 0.4))
        if trend.trend == "increasing":
            return (state.heading + correction) % 360.0
        return (state.heading - correction) % 360.0

    def _decision(
        self,
        *,
        decision: DecisionName,
        state: UUVRollingState,
        observations: Sequence[BearingObservation],
        trend: BearingTrend,
        estimated_distance: float,
        constraints: RollingConstraints,
        iteration: int,
        mission_context: str,
        next_heading: float,
        advance_distance: float,
        confidence: float,
        mode: str,
        warnings: List[str],
        expected_duration: Optional[float] = None,
    ) -> RollingDecision:
        duration = (
            expected_duration
            if expected_duration is not None
            else advance_distance / max(state.speed, 0.1)
        )
        reasoning = self._reasoning(
            decision=decision,
            state=state,
            observations=observations,
            trend=trend,
            estimated_distance=estimated_distance,
            constraints=constraints,
            iteration=iteration,
            mission_context=mission_context,
            next_heading=next_heading,
            advance_distance=advance_distance,
            confidence=confidence,
            warnings=warnings,
        )
        return RollingDecision(
            decision=decision,
            next_heading=next_heading,
            advance_distance=advance_distance,
            expected_duration=duration,
            reasoning=reasoning,
            confidence=confidence,
            mode=mode,
            warnings=warnings,
        )

    def _reasoning(
        self,
        *,
        decision: DecisionName,
        state: UUVRollingState,
        observations: Sequence[BearingObservation],
        trend: BearingTrend,
        estimated_distance: float,
        constraints: RollingConstraints,
        iteration: int,
        mission_context: str,
        next_heading: float,
        advance_distance: float,
        confidence: float,
        warnings: Sequence[str],
    ) -> str:
        latest_bearing = observations[-1].angle if observations else None
        history_text = self._history_text(observations)
        trend_text = {
            "increasing": "持续增大",
            "decreasing": "持续减小",
            "stable": "基本稳定",
            "unknown": "未知",
        }[trend.trend]
        side_text = {
            "increasing": "目标相对位置偏向右前方，应适当右转修正航向",
            "decreasing": "目标相对位置偏向左前方，应适当左转修正航向",
            "stable": "目标大体位于当前航向前方，可保持航向继续抵近",
            "unknown": "观测样本不足，暂不宜执行新的机动",
        }[trend.trend]
        latest_text = f"{latest_bearing:.1f}°" if latest_bearing is not None else "无"
        warning_text = "；".join(warnings) if warnings else "无"
        remaining_iterations = max(0, constraints.max_iterations - iteration)

        return (
            f"1.【态势理解】任务背景为{mission_context}；当前方位角为{latest_text}，"
            f"UUV航向为{state.heading:.1f}°，方位角趋势为{trend_text}（{history_text}），"
            f"估算距离约{estimated_distance:.1f}m。"
            f"2.【问题分析】方位净变化{trend.delta:.1f}°，变化率{trend.rate:.4f}°/s；"
            f"{side_text}；抵近阈值为{constraints.approach_range:.1f}m，"
            f"剩余滚动迭代预算{remaining_iterations}次。"
            f"3.【决策结论】决策类型为{decision}，下一航向{next_heading:.1f}°，"
            f"前进距离{advance_distance:.1f}m。"
            f"4.【置信度评估】置信度{confidence:.2f}；趋势分析置信度{trend.confidence:.2f}，"
            f"告警信息：{warning_text}。"
        )

    def _bearing_history(self, values: Sequence[Dict[str, Any]]) -> List[BearingObservation]:
        return [BearingObservation.from_dict(item) for item in values]

    def _elapsed_seconds(
        self,
        bearing_history: Sequence[BearingObservation],
        constraints: RollingConstraints,
    ) -> float:
        first = _parse_timestamp_seconds(bearing_history[0].timestamp)
        last = _parse_timestamp_seconds(bearing_history[-1].timestamp)
        if first is not None and last is not None and last > first:
            return last - first
        return constraints.observation_interval_seconds * max(1, len(bearing_history) - 1)

    def _iteration(self, payload: Dict[str, Any]) -> int:
        value = payload.get("iterations", payload.get("iteration", 0))
        return max(0, int(float(value)))

    def _estimated_range(self, payload: Dict[str, Any], constraints: RollingConstraints) -> float:
        value = payload.get("estimated_range")
        if value is None:
            return constraints.approach_range + constraints.default_step
        return max(0.0, float(value))

    def _advance_distance(self, estimated_distance: float, constraints: RollingConstraints) -> float:
        if estimated_distance <= 0.0:
            return constraints.default_step
        remaining_to_threshold = max(0.0, estimated_distance - constraints.approach_range)
        return min(constraints.default_step, remaining_to_threshold)

    def _orbit_heading(
        self,
        state: UUVRollingState,
        observations: Sequence[BearingObservation],
    ) -> float:
        if not observations:
            return state.heading
        return (observations[-1].angle + 90.0) % 360.0

    def _orbit_duration(self, state: UUVRollingState, constraints: RollingConstraints) -> float:
        circumference = 2.0 * math.pi * constraints.orbit_radius * max(1, constraints.orbit_turns)
        return circumference / max(state.speed, 0.1)

    def _history_text(self, observations: Sequence[BearingObservation]) -> str:
        if not observations:
            return "无方位历史"
        if len(observations) == 1:
            return f"{observations[0].angle:.1f}°"
        return f"{observations[0].angle:.1f}°→{observations[-1].angle:.1f}°"


def plan_rolling(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience wrapper returning a serializable rolling decision."""

    return RollingPlanner().plan(payload).to_dict()
