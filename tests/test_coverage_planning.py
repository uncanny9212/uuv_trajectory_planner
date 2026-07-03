"""Area coverage planning acceptance tests."""

from __future__ import annotations

import time
import unittest
from pathlib import Path
from typing import Any, Dict

from uuv_trajectory_planner.core.react_engine import ReActEngine
from uuv_trajectory_planner.models.decision_output import DecisionOutput
from uuv_trajectory_planner.models.situation_awareness import SituationAwareness
from uuv_trajectory_planner.reporting import write_constraint_report
from uuv_trajectory_planner.visualization import save_trajectory_plot


OUTPUT_DIR = Path("test_outputs")


def coverage_payload() -> Dict[str, Any]:
    return {
        "timestamp": "2026-06-26T10:00:00Z",
        "uuv_state": {"position": [0.0, 0.0, -50.0], "heading": 0.0, "speed": 2.0, "battery": 0.9},
        "mission": {
            "type": "trajectory_planning",
            "scenario": "area_coverage",
            "coverage_area": [[0, 0], [500, 0], [500, 500], [0, 500]],
            "constraints": {
                "max_speed": 3.0,
                "sweep_width": 50.0,
                "coverage_required": 0.95,
            },
        },
        "environment": {
            "obstacles": [],
            "boundaries": [[0, 0], [500, 0], [500, 500], [0, 500]],
            "water_current": [0.0, 0.0, 0.0],
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


class TestCoveragePlanning(unittest.TestCase):
    """Validate the requested rectangular coverage scenario."""

    def test_rectangular_area_coverage_scan(self) -> None:
        decision = run_case("coverage_05_rectangle", coverage_payload())

        self.assertEqual(decision.status, "success")
        self.assertIn("coverage_rate", decision.constraints_satisfied)
        self.assertGreaterEqual(decision.feedback["validation"]["coverage_rate"], 0.95)
        self.assertGreaterEqual(decision.feedback["validation"]["lane_count"], 10)
        self.assertGreater(decision.total_distance, 4500.0)


if __name__ == "__main__":
    unittest.main()
