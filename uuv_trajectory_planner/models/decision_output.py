"""Structured decision output models."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

Vector3 = Tuple[float, float, float]


@dataclass
class TrajectoryPoint:
    """One executable trajectory waypoint."""

    point_id: int
    coordinates: Vector3
    heading: float
    speed: float
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "point_id": self.point_id,
            "coordinates": [round(value, 3) for value in self.coordinates],
            "heading": round(self.heading, 3),
            "speed": round(self.speed, 3),
            "timestamp": round(self.timestamp, 3),
        }


@dataclass
class AlternativePlan:
    """Compact description of a possible alternative strategy."""

    strategy: str
    distance: float
    energy_cost: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "distance": round(self.distance, 3),
            "energy_cost": self.energy_cost,
        }


@dataclass
class DecisionOutput:
    """Top-level decision command emitted by the ReAct engine."""

    decision_type: str
    scenario: str
    trajectory: List[TrajectoryPoint]
    total_distance: float
    estimated_time: float
    constraints_satisfied: List[str]
    confidence: float
    reasoning_chain: str
    alternatives: List[AlternativePlan] = field(default_factory=list)
    decision_id: Optional[str] = None
    status: str = "success"
    timestamp: Optional[str] = None
    feedback: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.decision_id is None:
            compact_time = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            self.decision_id = f"D{compact_time}"
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        self.confidence = max(0.0, min(1.0, self.confidence))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "decision_type": self.decision_type,
            "scenario": self.scenario,
            "status": self.status,
            "trajectory": [point.to_dict() for point in self.trajectory],
            "total_distance": round(self.total_distance, 3),
            "estimated_time": round(self.estimated_time, 3),
            "constraints_satisfied": self.constraints_satisfied,
            "confidence": round(self.confidence, 3),
            "reasoning_chain": self.reasoning_chain,
            "alternatives": [alternative.to_dict() for alternative in self.alternatives],
            "feedback": self.feedback,
            "timestamp": self.timestamp,
        }

    def to_json(self, pretty: bool = False) -> str:
        indent = 2 if pretty else None
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
