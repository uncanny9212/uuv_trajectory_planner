"""Planning constraint models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class PlanningConstraints:
    """Mission-level trajectory planning constraints."""

    max_speed: float = 3.0
    min_obstacle_distance: float = 50.0
    max_energy_cost: Optional[float] = None
    min_turning_radius: float = 30.0
    max_turn_angle_deg: float = 30.0
    sweep_width: float = 50.0
    coverage_required: float = 0.95
    optimization: str = "distance"
    orbit_turns: int = 0
    orbit_radius: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "PlanningConstraints":
        """Create constraints from user JSON."""

        values = data or {}
        return cls(
            max_speed=float(values.get("max_speed", cls.max_speed)),
            min_obstacle_distance=float(
                values.get("min_obstacle_distance", cls.min_obstacle_distance)
            ),
            max_energy_cost=(
                float(values["max_energy_cost"]) if values.get("max_energy_cost") is not None else None
            ),
            min_turning_radius=float(values.get("min_turning_radius", cls.min_turning_radius)),
            max_turn_angle_deg=float(values.get("max_turn_angle_deg", cls.max_turn_angle_deg)),
            sweep_width=float(values.get("sweep_width", values.get("swath_width", cls.sweep_width))),
            coverage_required=float(values.get("coverage_required", cls.coverage_required)),
            optimization=str(values.get("optimization", cls.optimization)),
            orbit_turns=max(0, int(float(values.get("orbit_turns", values.get("circle_turns", cls.orbit_turns))))),
            orbit_radius=(
                float(values["orbit_radius"])
                if values.get("orbit_radius") is not None
                else (
                    float(values["circle_radius"])
                    if values.get("circle_radius") is not None
                    else cls.orbit_radius
                )
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize constraints."""

        return {
            "max_speed": self.max_speed,
            "min_obstacle_distance": self.min_obstacle_distance,
            "max_energy_cost": self.max_energy_cost,
            "min_turning_radius": self.min_turning_radius,
            "max_turn_angle_deg": self.max_turn_angle_deg,
            "sweep_width": self.sweep_width,
            "coverage_required": self.coverage_required,
            "optimization": self.optimization,
            "orbit_turns": self.orbit_turns,
            "orbit_radius": self.orbit_radius,
        }
