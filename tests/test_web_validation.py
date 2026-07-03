"""Web-input validation tests."""

from __future__ import annotations

import unittest

from uuv_trajectory_planner.main import sample_payload
from uuv_trajectory_planner.web_server import json_safe, validate_bait_clearance


class TestWebValidation(unittest.TestCase):
    """Validate UI/API guardrails for bait coordinates."""

    def test_bait_inside_obstacle_safety_zone_is_rejected(self) -> None:
        payload = sample_payload("general")
        payload["mission"]["constraints"]["min_obstacle_distance"] = 50.0
        payload["environment"]["obstacles"] = [
            {"id": "O001", "type": "static", "position": [300.0, 200.0, -50.0], "radius": 80.0}
        ]
        payload["environment"]["baits"] = [
            {"id": "B001", "position": [350.0, 230.0, -50.0], "radius": 40.0}
        ]

        with self.assertRaisesRegex(ValueError, "坐标不可用"):
            validate_bait_clearance(payload)

    def test_bait_outside_obstacle_safety_zone_is_allowed(self) -> None:
        payload = sample_payload("general")
        payload["mission"]["constraints"]["min_obstacle_distance"] = 50.0
        payload["environment"]["obstacles"] = [
            {"id": "O001", "type": "static", "position": [300.0, 200.0, -50.0], "radius": 80.0}
        ]
        payload["environment"]["baits"] = [
            {"id": "B001", "position": [520.0, 350.0, -50.0], "radius": 40.0}
        ]

        validate_bait_clearance(payload)

    def test_json_safe_replaces_non_finite_numbers(self) -> None:
        payload = {"clearance": float("inf"), "nested": [1.0, float("-inf"), float("nan")]}

        self.assertEqual(json_safe(payload), {"clearance": None, "nested": [1.0, None, None]})


if __name__ == "__main__":
    unittest.main()
