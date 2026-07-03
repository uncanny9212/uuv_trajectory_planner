"""General trajectory planning acceptance tests."""

from __future__ import annotations

import math
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List

from uuv_trajectory_planner.core.react_engine import ReActEngine
from uuv_trajectory_planner.models.decision_output import DecisionOutput
from uuv_trajectory_planner.models.situation_awareness import SituationAwareness
from uuv_trajectory_planner.reporting import write_constraint_report
from uuv_trajectory_planner.visualization import save_trajectory_plot


OUTPUT_DIR = Path("test_outputs")


def general_payload(obstacles: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "timestamp": "2026-06-26T10:00:00Z",
        "uuv_state": {"position": [0.0, 0.0, -50.0], "heading": 45.0, "speed": 2.0, "battery": 0.85},
        "mission": {
            "type": "trajectory_planning",
            "scenario": "general",
            "target_position": [1000.0, 800.0, -50.0],
            "constraints": {
                "max_speed": 3.0,
                "min_obstacle_distance": 50.0,
                "max_energy_cost": 1000.0,
            },
        },
        "environment": {
            "obstacles": obstacles,
            "boundaries": [[-100, -100], [1200, -100], [1200, 1000], [-100, 1000]],
            "water_current": [0.5, 0.2, 0.0],
        },
    }


def run_case(name: str, payload: Dict[str, Any]) -> DecisionOutput:
    engine = ReActEngine()
    started = time.perf_counter()
    decision = engine.run(payload)
    elapsed = time.perf_counter() - started
    situation = SituationAwareness.from_dict(payload)
    write_constraint_report(decision, str(OUTPUT_DIR / f"{name}_report.json"), elapsed)
    save_trajectory_plot(decision, situation, str(OUTPUT_DIR / f"{name}.png"))
    return decision


class TestGeneralPlanning(unittest.TestCase):
    """Validate the requested point-to-point planning scenarios."""

    def test_simple_path_without_obstacles(self) -> None:
        decision = run_case("general_01_simple", general_payload([]))

        self.assertEqual(decision.status, "success")
        self.assertIn("obstacle_avoidance", decision.constraints_satisfied)
        self.assertIn("smoothness", decision.constraints_satisfied)
        self.assertGreaterEqual(decision.confidence, 0.8)
        self.assertLessEqual(decision.feedback["validation"]["max_turn_angle"], 30.0)

    def test_single_obstacle_avoidance(self) -> None:
        payload = general_payload(
            [{"id": "O001", "type": "static", "position": [420.0, 330.0, -50.0], "radius": 90.0}]
        )
        decision = run_case("general_02_single_obstacle", payload)

        self.assertEqual(decision.status, "success")
        self.assertIn("obstacle_avoidance", decision.constraints_satisfied)
        self.assertGreaterEqual(decision.feedback["validation"]["min_obstacle_clearance"], 50.0)
        self.assertGreater(decision.total_distance, 0.0)

    def test_multiple_obstacle_avoidance(self) -> None:
        payload = general_payload(
            [
                {"id": "O001", "type": "static", "position": [260.0, 220.0, -50.0], "radius": 70.0},
                {"id": "O002", "type": "static", "position": [520.0, 410.0, -50.0], "radius": 80.0},
                {"id": "O003", "type": "static", "position": [760.0, 610.0, -50.0], "radius": 65.0},
            ]
        )
        decision = run_case("general_03_multiple_obstacles", payload)

        self.assertEqual(decision.status, "success")
        self.assertIn("obstacle_avoidance", decision.constraints_satisfied)
        self.assertGreaterEqual(decision.feedback["validation"]["min_obstacle_clearance"], 50.0)
        self.assertGreaterEqual(decision.confidence, 0.55)

    def test_dynamic_obstacle_prediction_avoidance(self) -> None:
        payload = general_payload(
            [
                {
                    "id": "M001",
                    "type": "moving",
                    "position": [380.0, 280.0, -50.0],
                    "velocity": [1.0, 0.6, 0.0],
                    "radius": 60.0,
                }
            ]
        )
        decision = run_case("general_04_dynamic_obstacle", payload)

        self.assertEqual(decision.status, "success")
        self.assertIn("obstacle_avoidance", decision.constraints_satisfied)
        self.assertNotEqual(decision.feedback["algorithm"], "direct_line")
        self.assertGreaterEqual(decision.feedback["validation"]["min_obstacle_clearance"], 50.0)

    def test_bait_guided_path_approaches_bait_while_avoiding_obstacles(self) -> None:
        payload = general_payload(
            [{"id": "O001", "type": "static", "position": [520.0, 410.0, -50.0], "radius": 70.0}]
        )
        payload["environment"]["baits"] = [
            {"id": "B001", "position": [250.0, 520.0, -50.0], "radius": 45.0}
        ]
        decision = run_case("general_05_bait_guided", payload)

        self.assertEqual(decision.status, "success")
        self.assertIn("obstacle_avoidance", decision.constraints_satisfied)
        self.assertIn("bait_approach", decision.constraints_satisfied)
        self.assertEqual(decision.feedback["validation"]["baits_reached"], 1.0)
        self.assertLessEqual(decision.feedback["validation"]["bait_min_distances"]["B001"], 45.0)

    def test_bait_inside_obstacle_uses_safe_approach_point(self) -> None:
        payload = general_payload(
            [{"id": "O001", "type": "static", "position": [520.0, 350.0, -50.0], "radius": 80.0}]
        )
        payload["environment"]["baits"] = [
            {"id": "B001", "position": [520.0, 350.0, -50.0], "radius": 45.0}
        ]
        decision = run_case("general_06_bait_inside_obstacle", payload)

        self.assertEqual(decision.status, "success")
        self.assertIn("obstacle_avoidance", decision.constraints_satisfied)
        self.assertGreaterEqual(decision.feedback["validation"]["min_obstacle_clearance"], 50.0)
        self.assertGreater(decision.feedback["validation"]["bait_min_distances"]["B001"], 45.0)

    def test_orbit_instruction_adds_circular_target_path(self) -> None:
        payload = general_payload([])
        payload["mission"]["target_position"] = [150.0, 259.808, -50.0]
        payload["mission"]["constraints"]["orbit_turns"] = 2
        payload["mission"]["constraints"]["orbit_radius"] = 10.0
        payload["environment"]["baits"] = [
            {"id": "B001", "position": [150.0, 259.808, -50.0], "radius": 40.0}
        ]

        decision = run_case("general_07_orbit_target", payload)

        self.assertEqual(decision.status, "success")
        self.assertIn("bait_approach", decision.constraints_satisfied)
        self.assertIn("target_orbit", decision.constraints_satisfied)
        self.assertEqual(decision.feedback["validation"]["orbit_turns_planned"], 2.0)
        self.assertGreater(decision.feedback["validation"]["orbit_waypoint_count"], 40.0)
        self.assertGreater(len(decision.trajectory), 40)
        orbit_points = [
            point
            for point in decision.trajectory
            if 8.0
            <= math.hypot(point.coordinates[0] - 150.0, point.coordinates[1] - 259.808)
            <= 12.0
        ]
        self.assertGreater(len(orbit_points), 30)

    def test_close_reconnaissance_orbits_each_bait_target(self) -> None:
        payload = general_payload([])
        payload["mission"]["target_position"] = [200.0, 346.41, -50.0]
        payload["mission"]["constraints"]["orbit_turns"] = 2
        payload["mission"]["constraints"]["orbit_radius"] = 10.0
        payload["environment"]["baits"] = [
            {"id": "B001", "position": [200.0, 346.41, -50.0], "radius": 40.0},
            {"id": "B002", "position": [424.264, 424.264, -50.0], "radius": 40.0},
        ]

        decision = run_case("general_08_multi_close_recon", payload)
        last = decision.trajectory[-1].coordinates

        self.assertEqual(decision.status, "success")
        self.assertIn("bait_approach", decision.constraints_satisfied)
        self.assertIn("target_orbit", decision.constraints_satisfied)
        self.assertEqual(decision.feedback["validation"]["baits_reached"], 2.0)
        self.assertEqual(decision.feedback["validation"]["orbit_targets_planned"], 2.0)
        self.assertLess(math.hypot(last[0] - 424.264, last[1] - 424.264), 15.0)
        self.assertGreater(math.hypot(last[0] - 200.0, last[1] - 346.41), 100.0)


if __name__ == "__main__":
    unittest.main()
