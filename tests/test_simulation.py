"""Simulation acceptance tests for rolling planner validation."""

from __future__ import annotations

import unittest

from uuv_trajectory_planner.simulation import (
    SimulationRunner,
    UUVSimulator,
    default_scenarios,
    parse_target_positions,
    scenario_by_name,
)
from uuv_trajectory_planner.simulation.runner import bearing_delta_series
from uuv_trajectory_planner.simulation.simulator import bearing_from_to
from uuv_trajectory_planner.simulation.visualization import simulation_plot_data_url


class TestSimulation(unittest.TestCase):
    """Validate the ground-truth simulation loop."""

    def test_bearing_uses_navigation_coordinates(self) -> None:
        origin = (0.0, 0.0, -50.0)

        self.assertAlmostEqual(bearing_from_to(origin, (0.0, 100.0, -50.0)), 0.0)
        self.assertAlmostEqual(bearing_from_to(origin, (100.0, 0.0, -50.0)), 90.0)
        self.assertAlmostEqual(bearing_from_to(origin, (0.0, -100.0, -50.0)), 180.0)
        self.assertAlmostEqual(bearing_from_to(origin, (-100.0, 0.0, -50.0)), 270.0)

    def test_uuv_move_uses_same_heading_coordinates(self) -> None:
        simulator = UUVSimulator.create(target_position=(100.0, 100.0, -50.0))

        simulator.move(0.0, 100.0)
        self.assertAlmostEqual(simulator.uuv_position[0], 0.0)
        self.assertAlmostEqual(simulator.uuv_position[1], 100.0)

        simulator.move(90.0, 50.0)
        self.assertAlmostEqual(simulator.uuv_position[0], 50.0)
        self.assertAlmostEqual(simulator.uuv_position[1], 100.0)

    def test_single_scenario_drives_rolling_planner_to_success(self) -> None:
        result = SimulationRunner().run(scenario_by_name("右前方"))

        self.assertEqual(result["status"], "success")
        self.assertLessEqual(result["final_distance"], result["scenario"]["constraints"]["approach_range"])
        self.assertIn("adjust_heading", [decision["decision"] for decision in result["decisions"]])
        self.assertGreater(sum(1 for delta in bearing_delta_series(result) if delta > 0.0), 3)

    def test_batch_scenarios_generate_report(self) -> None:
        report = SimulationRunner().run_batch(default_scenarios())

        self.assertEqual(report["scenario_count"], 5)
        self.assertEqual(report["success_count"], 5)
        self.assertEqual(report["success_rate"], 1.0)
        self.assertGreater(report["average_iterations"], 3.0)

    def test_target_position_text_supports_multiple_truth_targets(self) -> None:
        positions = parse_target_positions("(800, 300, -50)\n[600, 400]")

        self.assertEqual(positions, [(800.0, 300.0, -50.0), (600.0, 400.0, -50.0)])

    def test_target_positions_must_use_fixed_world_grid(self) -> None:
        with self.assertRaisesRegex(ValueError, "0～2000"):
            parse_target_positions("(-1, 300, -50)")
        with self.assertRaisesRegex(ValueError, "分辨率"):
            parse_target_positions("(800.5, 300, -50)")

    def test_interactive_simulation_approaches_and_orbits_selected_target(self) -> None:
        result = SimulationRunner().run_interactive(
            {
                "target_positions_text": "(800, 300, -50)\n(0, 1000, -50)",
                "bearing_text": "当前方位北偏东70度，抵近侦察",
                "default_step": 180,
                "approach_range": 50,
                "bearing_noise_deg": 1.0,
                "orbit_turns": 5,
                "max_iterations": 100,
            }
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["active_target_index"], 0)
        self.assertEqual(result["coordinate_system"]["x_range"], [0.0, 2000.0])
        self.assertEqual(result["coordinate_system"]["y_range"], [0.0, 2000.0])
        self.assertEqual(result["coordinate_system"]["resolution"], 1.0)
        self.assertEqual(result["coordinate_system"]["start_position"], [0.0, 0.0, -50.0])
        self.assertEqual(result["uuv_history"][0]["position"], [0.0, 0.0, -50.0])
        self.assertLessEqual(result["final_distance"], result["constraints"]["approach_range"])
        self.assertEqual(result["orbit_turns_completed"], 7)
        self.assertEqual(result["target_runs"][0]["target_depth"], 50.0)
        self.assertTrue(result["target_runs"][0]["is_deep_target"])
        self.assertGreaterEqual(len(result["orbit_history"]), 7 * 36)
        self.assertTrue(any(decision["executed_distance"] > 0 for decision in result["decisions"]))

    def test_interactive_simulation_visits_multiple_targets_in_bearing_order(self) -> None:
        result = SimulationRunner().run_interactive(
            {
                "target_positions": [(800, 300, -50), (300, 800, -50)],
                "bearing_text": "目标1在北偏东70度，目标2在北偏东21度。分别抵近侦察",
                "default_step": 180,
                "approach_range": 50,
                "bearing_noise_deg": 1.0,
                "orbit_turns": 1,
                "orbit_radius": 10,
                "max_iterations": 100,
            }
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["target_route"], [0, 1])
        self.assertEqual(result["completed_target_count"], 2)
        self.assertEqual(result["orbit_turns_completed"], 6)
        self.assertEqual([run["status"] for run in result["target_runs"]], ["success", "success"])
        self.assertEqual([run["orbit_turns_completed"] for run in result["target_runs"]], [3, 3])
        self.assertEqual(
            [(segment["kind"], segment["target_index"]) for segment in result["trajectory_segments"]],
            [("approach", 0), ("orbit", 0), ("approach", 1), ("orbit", 1)],
        )

    def test_interactive_simulation_flags_false_bearing_feedback(self) -> None:
        result = SimulationRunner().run_interactive(
            {
                "target_positions": [(800, 300, -50)],
                "bearing_text": "当前方位正北，抵近侦察",
                "default_step": 180,
                "approach_range": 50,
                "bearing_noise_deg": 0.0,
                "orbit_turns": 1,
                "orbit_radius": 10,
                "max_iterations": 100,
            }
        )

        self.assertEqual(result["status"], "success")
        self.assertTrue(result["false_information_detected"])
        self.assertEqual(result["bearing_assessments"][0]["status"], "suspect_false")
        self.assertFalse(result["decisions"][0]["target_discovered"])
        self.assertNotIn("discovered_position", result["decisions"][0])
        self.assertIn("疑似虚假信息", result["decisions"][1]["feedback"])
        self.assertIn("初始方位疑似虚假信息", result["decisions"][1]["warnings"])

    def test_visualization_returns_png_data_url(self) -> None:
        result = SimulationRunner().run(scenario_by_name("近距离"))
        data_url = simulation_plot_data_url(result)

        self.assertTrue(data_url.startswith("data:image/png;base64,"))


if __name__ == "__main__":
    unittest.main()
