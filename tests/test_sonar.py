"""Synthetic sonar simulation and recognition tests."""

from __future__ import annotations

import unittest

from uuv_trajectory_planner.core.sonar_recognizer import recognize_target
from uuv_trajectory_planner.core.sonar_simulator import generate_sonar_image
from uuv_trajectory_planner.simulation import SimulationRunner


class TestSonarModules(unittest.TestCase):
    """Validate deterministic sonar image generation and rule recognition."""

    def test_generate_sonar_image_shape_and_metadata(self) -> None:
        output = generate_sonar_image((0.0, 0.0, -50.0), (0.0, 6.0, -40.0), "submarine", 30.0)

        self.assertEqual(output["image"].shape, (100, 60, 3))
        self.assertEqual(output["image"].dtype.name, "uint8")
        self.assertEqual(output["target_range_m"], 6.0)
        self.assertEqual(output["target_depth_m"], -40.0)
        self.assertGreater(output["echo_strength"], 0.0)

    def test_rule_recognizer_classifies_five_target_examples(self) -> None:
        expected = {
            "submarine": "submarine",
            "torpedo": "torpedo",
            "ship": "ship",
            "reef": "reef",
            "unknown": "unknown",
        }
        for target_type, expected_type in expected.items():
            with self.subTest(target_type=target_type):
                output = generate_sonar_image((0.0, 0.0, -50.0), (0.0, 6.0, -40.0), target_type, 45.0)
                recognition = recognize_target(output)

                self.assertEqual(recognition["target_type"], expected_type)
                self.assertIn("reasoning", recognition)


class TestSonarRunnerIntegration(unittest.TestCase):
    """Validate sonar-triggered target exclusion and post-mission overrides."""

    def test_blue_high_value_target_requests_authorization_inside_envelope(self) -> None:
        result = SimulationRunner().run_interactive(
            {
                "target_positions": [(800, 300, -40)],
                "target_profiles": [{"target_type": "submarine", "target_heading_deg": 20, "is_blue_target": True}],
                "bearing_text": "目标在北偏东70度。逐目标抵近侦察。",
                "approach_range": 50,
                "sonar_trigger_range": 15,
                "orbit_turns": 1,
                "bearing_noise_deg": 0,
            }
        )

        self.assertEqual(result["target_runs"][0]["sonar_recognition"]["target_type"], "submarine")
        self.assertTrue(result["target_runs"][0]["sonar_recognition"]["is_blue_target"])
        sonar_event = result["target_runs"][0]["sonar_events"][-1]
        self.assertEqual(len(sonar_event["image_rgb"]), 100)
        self.assertEqual(len(sonar_event["image_rgb"][0]), 60)
        self.assertEqual(len(sonar_event["image_rgb"][0][0]), 3)
        self.assertEqual(result["post_mission_decision"]["action"], "simulated_strike_request")
        self.assertEqual(result["post_mission_decision"]["decision_basis"], "sonar_value_depth_iff")

    def test_blue_high_value_target_tracks_when_depth_outside_envelope(self) -> None:
        result = SimulationRunner().run_interactive(
            {
                "target_positions": [(800, 300, -80)],
                "target_profiles": [{"target_type": "torpedo", "target_heading_deg": 20, "is_blue_target": True}],
                "bearing_text": "目标在北偏东70度。逐目标抵近侦察。",
                "approach_range": 50,
                "sonar_trigger_range": 15,
                "orbit_turns": 1,
                "bearing_noise_deg": 0,
            }
        )

        self.assertEqual(result["post_mission_decision"]["action"], "track_and_report")
        self.assertFalse(result["post_mission_decision"]["requires_authorization"])

    def test_red_neutral_submarine_never_requests_strike(self) -> None:
        result = SimulationRunner().run_interactive(
            {
                "target_positions": [(800, 300, -80)],
                "target_profiles": [
                    {"target_type": "submarine", "target_heading_deg": 20, "is_blue_target": False}
                ],
                "bearing_text": "目标在北偏东70度。逐目标抵近侦察。",
                "approach_range": 50,
                "sonar_trigger_range": 15,
                "orbit_turns": 1,
                "bearing_noise_deg": 0,
            }
        )

        decision = result["post_mission_decision"]
        self.assertEqual(decision["action"], "track_and_report")
        self.assertEqual(decision["decision_basis"], "iff_constraint")
        self.assertFalse(decision["requires_authorization"])
        self.assertIn("红方/中立", decision["reasoning"])

    def test_reef_target_is_excluded_by_sonar(self) -> None:
        result = SimulationRunner().run_interactive(
            {
                "target_positions": [(800, 300, -40)],
                "target_profiles": [{"target_type": "reef", "target_heading_deg": 20, "is_blue_target": False}],
                "bearing_text": "目标在北偏东70度。逐目标抵近侦察。",
                "approach_range": 50,
                "sonar_trigger_range": 15,
                "orbit_turns": 1,
                "bearing_noise_deg": 0,
            }
        )

        self.assertEqual(result["target_runs"][0]["status"], "excluded")
        self.assertTrue(result["target_runs"][0]["excluded_as_false_target"])
        self.assertEqual(result["excluded_target_count"], 1)
        self.assertEqual(result["post_mission_decision"]["decision_basis"], "sonar_exclusion")


if __name__ == "__main__":
    unittest.main()
