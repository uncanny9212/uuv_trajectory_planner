"""Web-input validation tests."""

from __future__ import annotations

import unittest

from uuv_trajectory_planner.main import sample_payload
from uuv_trajectory_planner.web_server import json_safe, rolling_preview_from_text, validate_bait_clearance


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

    def test_bearing_without_range_uses_rolling_planner(self) -> None:
        result = rolling_preview_from_text("有1个目标在北偏东20度。抵近侦察")

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["mode"], "rolling")
        self.assertEqual(result["rolling"]["detected_bearing"], 20.0)
        decisions = [item["decision"]["decision"] for item in result["rolling"]["iterations"]]
        self.assertEqual(decisions[0], "wait")
        self.assertIn("adjust_heading", decisions)
        self.assertEqual(decisions[-1], "orbit")
        self.assertNotIn("target_position", result["payload"])

    def test_bearing_with_explicit_range_uses_standard_detection_path(self) -> None:
        result = rolling_preview_from_text("目标在北偏东20度，距离500米。抵近侦察")

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
