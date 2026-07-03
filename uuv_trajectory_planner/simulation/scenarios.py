"""Predefined rolling-planner simulation scenarios."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

Vector3 = Tuple[float, float, float]

DEFAULT_SIMULATION_CONSTRAINTS: Dict[str, Any] = {
    "approach_range": 50.0,
    "orbit_turns": 2,
    "orbit_radius": 10.0,
    "max_iterations": 100,
    "default_step": 100.0,
    "stable_angle_threshold": 1.0,
}


@dataclass(frozen=True)
class SimulationScenario:
    """One ground-truth scenario for passive-bearing rolling tests."""

    name: str
    target_position: Vector3
    start_position: Vector3 = (0.0, 0.0, -50.0)
    initial_heading: float = 0.0
    expected_behavior: str = ""
    constraints: Dict[str, Any] = field(default_factory=dict)
    speed: float = 2.0
    battery: float = 0.9

    def merged_constraints(self) -> Dict[str, Any]:
        values = dict(DEFAULT_SIMULATION_CONSTRAINTS)
        values.update(self.constraints)
        return values

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "target_position": list(self.target_position),
            "start_position": list(self.start_position),
            "initial_heading": self.initial_heading,
            "expected_behavior": self.expected_behavior,
            "constraints": self.merged_constraints(),
            "speed": self.speed,
            "battery": self.battery,
        }


def default_scenarios() -> List[SimulationScenario]:
    """Return the required batch of representative rolling-planning scenes."""

    return [
        SimulationScenario(
            name="正前方",
            target_position=(800.0, 0.0, -50.0),
            initial_heading=90.0,
            expected_behavior="方位角稳定，直航抵近",
            constraints={"default_step": 250.0, "stable_angle_threshold": 5.0},
        ),
        SimulationScenario(
            name="右前方",
            target_position=(800.0, 300.0, -50.0),
            initial_heading=55.0,
            expected_behavior="方位角增大，右转修正",
            constraints={
                "default_step": 30.0,
                "max_heading_correction": 1.0,
                "max_iterations": 120,
                "stable_angle_threshold": 0.5,
            },
        ),
        SimulationScenario(
            name="左前方",
            target_position=(600.0, -400.0, -50.0),
            initial_heading=125.0,
            expected_behavior="方位角减小，左转修正",
            constraints={
                "default_step": 80.0,
                "max_heading_correction": 0.5,
                "max_iterations": 120,
                "stable_angle_threshold": 0.5,
            },
        ),
        SimulationScenario(
            name="远距离",
            target_position=(2000.0, 500.0, -50.0),
            initial_heading=70.0,
            expected_behavior="多次迭代抵近",
            constraints={"default_step": 150.0, "max_heading_correction": 1.0, "max_iterations": 120},
        ),
        SimulationScenario(
            name="近距离",
            target_position=(200.0, 50.0, -50.0),
            initial_heading=60.0,
            expected_behavior="快速进入抵近模式",
            constraints={"default_step": 80.0},
        ),
    ]


def scenario_by_name(name: str, scenarios: Sequence[SimulationScenario] | None = None) -> SimulationScenario:
    """Return a predefined scenario by name."""

    available = list(scenarios or default_scenarios())
    for scenario in available:
        if scenario.name == name:
            return scenario
    names = "、".join(scenario.name for scenario in available)
    raise ValueError(f"未知仿真场景：{name}。可选场景：{names}")
