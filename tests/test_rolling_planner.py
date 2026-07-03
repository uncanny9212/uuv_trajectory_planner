"""Rolling planner decision tests."""

from __future__ import annotations

import unittest

from uuv_trajectory_planner.core.rolling_planner import RollingPlanner, plan_rolling


def rolling_payload() -> dict:
    return {
        "bearing_history": [
            {"angle": 30, "timestamp": "10:00:00"},
            {"angle": 32, "timestamp": "10:00:30"},
            {"angle": 35, "timestamp": "10:01:00"},
            {"angle": 38, "timestamp": "10:01:30"},
            {"angle": 42, "timestamp": "10:02:00"},
        ],
        "uuv_state": {
            "position": [0, 0, -50],
            "heading": 30,
            "speed": 2.0,
            "battery": 0.8,
        },
        "estimated_range": 800,
        "constraints": {
            "approach_range": 100,
            "orbit_turns": 2,
            "orbit_radius": 10,
            "max_iterations": 50,
            "default_step": 300,
        },
        "mission_context": "抵近侦察",
    }


class TestRollingPlanner(unittest.TestCase):
    """Validate passive-bearing rolling decisions."""

    def test_increasing_bearing_adjusts_heading_right(self) -> None:
        decision = RollingPlanner().plan(rolling_payload())

        self.assertEqual(decision.decision, "adjust_heading")
        self.assertEqual(decision.mode, "approach")
        self.assertAlmostEqual(decision.next_heading, 34.8)
        self.assertEqual(decision.advance_distance, 300.0)
        self.assertEqual(decision.expected_duration, 150.0)
        self.assertGreaterEqual(decision.confidence, 0.8)
        self.assertIn("持续增大", decision.reasoning)

    def test_decreasing_bearing_adjusts_heading_left(self) -> None:
        payload = rolling_payload()
        payload["bearing_history"] = [
            {"angle": 60, "timestamp": "10:00:00"},
            {"angle": 57, "timestamp": "10:00:30"},
            {"angle": 54, "timestamp": "10:01:00"},
        ]
        payload["uuv_state"]["heading"] = 60

        decision = RollingPlanner().plan(payload)

        self.assertEqual(decision.decision, "adjust_heading")
        self.assertLess(decision.next_heading, 60.0)
        self.assertIn("持续减小", decision.reasoning)

    def test_bearing_wraparound_keeps_increasing_trend(self) -> None:
        payload = rolling_payload()
        payload["bearing_history"] = [
            {"angle": 350, "timestamp": "10:00:00"},
            {"angle": 355, "timestamp": "10:00:30"},
            {"angle": 2, "timestamp": "10:01:00"},
            {"angle": 5, "timestamp": "10:01:30"},
        ]
        payload["uuv_state"]["heading"] = 350

        decision = RollingPlanner().plan(payload)

        self.assertEqual(decision.decision, "adjust_heading")
        self.assertGreater(decision.next_heading, 350.0)
        self.assertIn("350.0°→5.0°", decision.reasoning)

    def test_close_range_switches_to_orbit(self) -> None:
        payload = rolling_payload()
        payload["estimated_range"] = 80

        decision = RollingPlanner().plan(payload)

        self.assertEqual(decision.decision, "orbit")
        self.assertEqual(decision.mode, "orbit")
        self.assertEqual(decision.advance_distance, 0.0)
        self.assertGreater(decision.expected_duration, 0.0)
        self.assertIn("估算距离已进入抵近阈值", decision.warnings)

    def test_low_battery_gives_up(self) -> None:
        payload = rolling_payload()
        payload["uuv_state"]["battery"] = 0.1

        decision = RollingPlanner().plan(payload)

        self.assertEqual(decision.decision, "give_up")
        self.assertEqual(decision.mode, "abort")
        self.assertEqual(decision.next_heading, 210.0)
        self.assertIn("电量低于安全阈值", decision.warnings)

    def test_insufficient_bearing_history_waits(self) -> None:
        payload = rolling_payload()
        payload["bearing_history"] = [{"angle": 30, "timestamp": "10:00:00"}]

        decision = RollingPlanner().plan(payload)

        self.assertEqual(decision.decision, "wait")
        self.assertEqual(decision.mode, "observe")
        self.assertEqual(decision.advance_distance, 0.0)
        self.assertLess(decision.confidence, 0.5)

    def test_missing_estimated_range_uses_conservative_default(self) -> None:
        payload = rolling_payload()
        del payload["estimated_range"]

        decision = RollingPlanner().plan(payload)

        self.assertEqual(decision.decision, "adjust_heading")
        self.assertEqual(decision.mode, "approach")
        self.assertEqual(decision.advance_distance, 300.0)
        self.assertIn("未提供距离估计，使用默认步长外推", decision.warnings)

    def test_plan_rolling_returns_serializable_dict(self) -> None:
        output = plan_rolling(rolling_payload())

        self.assertEqual(output["decision"], "adjust_heading")
        self.assertEqual(output["mode"], "approach")
        self.assertIn("reasoning", output)


if __name__ == "__main__":
    unittest.main()
