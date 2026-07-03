"""Structured situation-awareness input models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from uuv_trajectory_planner.models.constraints import PlanningConstraints

Vector2 = Tuple[float, float]
Vector3 = Tuple[float, float, float]


def _vector(values: Sequence[Any], size: int, default_z: float = 0.0) -> Tuple[float, ...]:
    if len(values) < size:
        if size == 3 and len(values) == 2:
            return (float(values[0]), float(values[1]), float(default_z))
        raise ValueError("Vector has fewer coordinates than required")
    return tuple(float(values[index]) for index in range(size))


def _vector2_list(values: Optional[Iterable[Sequence[Any]]]) -> List[Vector2]:
    return [(_vector(item, 2)[0], _vector(item, 2)[1]) for item in values or []]


@dataclass
class UUVState:
    """Current UUV navigation state."""

    position: Vector3 = (0.0, 0.0, -50.0)
    heading: float = 0.0
    speed: float = 1.5
    battery: float = 1.0

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "UUVState":
        values = data or {}
        return cls(
            position=_vector(values.get("position", [0.0, 0.0, -50.0]), 3),  # type: ignore[arg-type]
            heading=float(values.get("heading", 0.0)),
            speed=float(values.get("speed", 1.5)),
            battery=float(values.get("battery", 1.0)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "position": list(self.position),
            "heading": self.heading,
            "speed": self.speed,
            "battery": self.battery,
        }


@dataclass
class Obstacle:
    """Static or moving circular obstacle."""

    id: str
    type: str
    position: Vector3
    radius: float
    velocity: Vector3 = (0.0, 0.0, 0.0)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Obstacle":
        return cls(
            id=str(data.get("id", "obstacle")),
            type=str(data.get("type", "static")),
            position=_vector(data.get("position", [0.0, 0.0, -50.0]), 3),  # type: ignore[arg-type]
            radius=float(data.get("radius", 0.0)),
            velocity=_vector(data.get("velocity", [0.0, 0.0, 0.0]), 3),  # type: ignore[arg-type]
        )

    def predicted(self, seconds: float) -> "Obstacle":
        """Return a conservative predicted obstacle copy."""

        return Obstacle(
            id=f"{self.id}@{int(seconds)}s",
            type=self.type,
            position=(
                self.position[0] + self.velocity[0] * seconds,
                self.position[1] + self.velocity[1] * seconds,
                self.position[2] + self.velocity[2] * seconds,
            ),
            radius=self.radius,
            velocity=self.velocity,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "position": list(self.position),
            "radius": self.radius,
            "velocity": list(self.velocity),
        }


@dataclass
class BaitTarget:
    """Attractive target that the UUV should approach during planning."""

    id: str
    position: Vector3
    radius: float = 40.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaitTarget":
        return cls(
            id=str(data.get("id", "bait")),
            position=_vector(data.get("position", [0.0, 0.0, -50.0]), 3),  # type: ignore[arg-type]
            radius=float(data.get("radius", data.get("approach_radius", 40.0))),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "position": list(self.position),
            "radius": self.radius,
        }


@dataclass
class Mission:
    """Mission request parsed from input JSON."""

    type: str = "trajectory_planning"
    scenario: str = "general"
    target_position: Optional[Vector3] = None
    coverage_area: List[Vector2] = field(default_factory=list)
    constraints: PlanningConstraints = field(default_factory=PlanningConstraints)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "Mission":
        values = data or {}
        target = values.get("target_position")
        area = (
            values.get("coverage_area")
            or values.get("area_boundary")
            or values.get("search_area")
            or values.get("boundary")
            or []
        )
        return cls(
            type=str(values.get("type", "trajectory_planning")),
            scenario=str(values.get("scenario", "general")),
            target_position=_vector(target, 3) if target is not None else None,  # type: ignore[arg-type]
            coverage_area=_vector2_list(area),
            constraints=PlanningConstraints.from_dict(values.get("constraints")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "scenario": self.scenario,
            "target_position": list(self.target_position) if self.target_position else None,
            "coverage_area": [list(point) for point in self.coverage_area],
            "constraints": self.constraints.to_dict(),
        }


@dataclass
class Environment:
    """Mission environment parsed from input JSON."""

    obstacles: List[Obstacle] = field(default_factory=list)
    baits: List[BaitTarget] = field(default_factory=list)
    boundaries: List[Vector2] = field(default_factory=list)
    water_current: Vector3 = (0.0, 0.0, 0.0)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "Environment":
        values = data or {}
        return cls(
            obstacles=[Obstacle.from_dict(item) for item in values.get("obstacles", [])],
            baits=[
                BaitTarget.from_dict(item)
                for item in (
                    values.get("baits")
                    or values.get("bait_targets")
                    or values.get("attractors")
                    or []
                )
            ],
            boundaries=_vector2_list(values.get("boundaries", [])),
            water_current=_vector(values.get("water_current", [0.0, 0.0, 0.0]), 3),  # type: ignore[arg-type]
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "obstacles": [obstacle.to_dict() for obstacle in self.obstacles],
            "baits": [bait.to_dict() for bait in self.baits],
            "boundaries": [list(point) for point in self.boundaries],
            "water_current": list(self.water_current),
        }


@dataclass
class SituationAwareness:
    """Top-level structured UUV situation."""

    timestamp: str
    uuv_state: UUVState
    mission: Mission
    environment: Environment

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SituationAwareness":
        timestamp = str(
            data.get("timestamp") or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        )
        situation = cls(
            timestamp=timestamp,
            uuv_state=UUVState.from_dict(data.get("uuv_state")),
            mission=Mission.from_dict(data.get("mission")),
            environment=Environment.from_dict(data.get("environment")),
        )
        situation.validate()
        return situation

    def validate(self) -> None:
        """Validate the minimal mission fields required by each scenario."""

        scenario = self.mission.scenario
        if scenario == "general" and self.mission.target_position is None:
            raise ValueError("general scenario requires mission.target_position")
        if scenario == "area_coverage":
            area = self.mission.coverage_area or self.environment.boundaries
            if len(area) < 3:
                raise ValueError("area_coverage scenario requires a polygon boundary")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "uuv_state": self.uuv_state.to_dict(),
            "mission": self.mission.to_dict(),
            "environment": self.environment.to_dict(),
        }
