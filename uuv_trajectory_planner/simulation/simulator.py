"""Ground-truth simulator for passive-bearing UUV approach tests."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

Vector3 = Tuple[float, float, float]


def vector3(values: Sequence[float]) -> Vector3:
    """Normalize a sequence into an ``(x, y, z)`` tuple."""

    if len(values) < 2:
        raise ValueError("position must contain at least x and y")
    z = float(values[2]) if len(values) > 2 else -50.0
    return (float(values[0]), float(values[1]), z)


def bearing_from_to(origin: Sequence[float], target: Sequence[float]) -> float:
    """Return bearing from origin to target, where 0 deg is north and 90 deg is east."""

    dx = float(target[0]) - float(origin[0])
    dy = float(target[1]) - float(origin[1])
    return (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0


def move_by_heading(position: Sequence[float], heading: float, distance: float) -> Vector3:
    """Move from position along a navigation heading."""

    radians = math.radians(heading)
    return (
        float(position[0]) + math.sin(radians) * distance,
        float(position[1]) + math.cos(radians) * distance,
        float(position[2]) if len(position) > 2 else -50.0,
    )


@dataclass
class UUVSimulator:
    """Maintain target truth and UUV state during a rolling-planning test."""

    target_position: Vector3
    uuv_position: Vector3 = (0.0, 0.0, -50.0)
    heading: float = 0.0
    speed: float = 2.0
    battery: float = 0.9
    total_distance: float = 0.0

    @classmethod
    def create(
        cls,
        target_position: Sequence[float],
        uuv_position: Sequence[float] = (0.0, 0.0, -50.0),
        heading: float = 0.0,
        speed: float = 2.0,
        battery: float = 0.9,
    ) -> "UUVSimulator":
        return cls(
            target_position=vector3(target_position),
            uuv_position=vector3(uuv_position),
            heading=float(heading) % 360.0,
            speed=max(0.1, float(speed)),
            battery=max(0.0, min(1.0, float(battery))),
        )

    def bearing_to_target(self) -> float:
        """Calculate the current passive bearing from UUV to target."""

        return bearing_from_to(self.uuv_position, self.target_position)

    def distance_to_target(self) -> float:
        """Return horizontal distance from UUV to target."""

        return math.hypot(
            self.target_position[0] - self.uuv_position[0],
            self.target_position[1] - self.uuv_position[1],
        )

    def move(self, heading: float, distance: float) -> None:
        """Move the UUV and update accumulated path length."""

        travel = max(0.0, float(distance))
        self.heading = float(heading) % 360.0
        self.uuv_position = move_by_heading(self.uuv_position, self.heading, travel)
        self.total_distance += travel

    def uuv_state(self) -> Dict[str, Any]:
        """Return the state shape expected by ``RollingPlanner``."""

        return {
            "position": [round(value, 3) for value in self.uuv_position],
            "heading": round(self.heading, 3),
            "speed": round(self.speed, 3),
            "battery": round(self.battery, 3),
        }

    def history_item(self, iteration: int) -> Dict[str, Any]:
        """Return a serializable UUV history entry."""

        return {
            "iteration": iteration,
            "position": [round(value, 3) for value in self.uuv_position],
            "heading": round(self.heading, 3),
            "distance_to_target": round(self.distance_to_target(), 3),
            "bearing_to_target": round(self.bearing_to_target(), 3),
            "total_distance": round(self.total_distance, 3),
        }
