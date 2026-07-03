"""Detection semantic parser tests."""

from __future__ import annotations

import time
import unittest

from uuv_trajectory_planner.core.detection_parser import DetectionParser, parse_detection_text
from uuv_trajectory_planner.models.situation_awareness import SituationAwareness


class TestDetectionParser(unittest.TestCase):
    """Validate UUV detection text parsing into planner input."""

    def test_full_3d_position(self) -> None:
        parser = DetectionParser()
        result = parser.parse(
            """
            探测到可能的目标：
            - 位置：(200, 300, -80)
            - 类型：疑似渔船
            - 置信度：0.75
            - 探测时间：2026-06-30 09:30:00
            """
        )

        self.assertEqual(result["timestamp"], "2026-06-30T09:30:00Z")
        self.assertEqual(result["mission"]["target_position"], [200.0, 300.0, -80.0])
        self.assertEqual(len(result["environment"]["baits"]), 1)
        self.assertEqual(result["environment"]["baits"][0]["position"], [200.0, 300.0, -80.0])
        SituationAwareness.from_dict(result)

    def test_2d_position_uses_default_depth(self) -> None:
        parser = DetectionParser()
        result = parser.parse(
            """
            探测到可能的目标：
            - 位置：(200, 300)
            - 类型：疑似渔船
            """
        )

        self.assertEqual(result["mission"]["target_position"], [200.0, 300.0, -50.0])

    def test_bearing_and_range_are_converted_to_coordinates(self) -> None:
        parser = DetectionParser(default_detection_range=500.0)
        result = parser.parse(
            """
            声呐接触：
            - 方向：北偏东30度
            - 距离：约800米
            """
        )

        target = result["mission"]["target_position"]
        self.assertAlmostEqual(target[0], 400.0, delta=1.0)
        self.assertAlmostEqual(target[1], 692.8, delta=1.0)
        self.assertEqual(target[2], -50.0)
        self.assertEqual(result["uuv_state"]["heading"], 30.0)

    def test_bearing_uses_semantic_depth(self) -> None:
        parser = DetectionParser()
        result = parser.parse(
            """
            声呐接触：
            - 方向：北偏东30度
            - 距离：约800米
            - 水深：80米
            """
        )

        self.assertEqual(result["mission"]["target_position"][2], -80.0)

    def test_multi_target_mix_classifies_baits_and_obstacles(self) -> None:
        parser = DetectionParser()
        result = parser.parse(
            """
            检测到3个接触：
            1. 目标A：位置(150, 200)，静止，置信度0.8
            2. 目标B：北偏东45度方向，距离约600米，移动中
            3. 目标C：位置(400, 500)，半径约50米，疑似礁石
            """
        )

        self.assertEqual(len(result["environment"]["baits"]), 2)
        self.assertEqual(len(result["environment"]["obstacles"]), 1)
        self.assertEqual(result["environment"]["obstacles"][0]["radius"], 50.0)
        self.assertAlmostEqual(result["environment"]["baits"][1]["position"][0], 424.264, places=3)
        self.assertAlmostEqual(result["environment"]["baits"][1]["position"][1], 424.264, places=3)

    def test_same_line_target_and_reef_have_independent_contexts(self) -> None:
        parser = DetectionParser()
        result = parser.parse("目标在(200, 300)，礁石在(400, 500)，半径50米")

        self.assertEqual(len(result["environment"]["baits"]), 1)
        self.assertEqual(result["environment"]["baits"][0]["position"], [200.0, 300.0, -50.0])
        self.assertEqual(len(result["environment"]["obstacles"]), 1)
        self.assertEqual(result["environment"]["obstacles"][0]["position"], [400.0, 500.0, -50.0])
        self.assertEqual(result["environment"]["obstacles"][0]["radius"], 50.0)

    def test_same_line_multiple_targets_are_all_baits(self) -> None:
        parser = DetectionParser()
        result = parser.parse("目标A在(100, 100)，目标B在(200, 200)，目标C在(300, 300)")

        self.assertEqual(len(result["environment"]["baits"]), 3)
        self.assertEqual(result["environment"]["obstacles"], [])

    def test_same_line_target_and_multiple_reefs_are_split(self) -> None:
        parser = DetectionParser()
        result = parser.parse("目标在(150, 150)，礁石A在(400, 500)，半径50米，礁石B在(600, 700)，半径30米")

        self.assertEqual(len(result["environment"]["baits"]), 1)
        self.assertEqual(result["environment"]["baits"][0]["position"], [150.0, 150.0, -50.0])
        self.assertEqual(len(result["environment"]["obstacles"]), 2)
        self.assertEqual(result["environment"]["obstacles"][0]["position"], [400.0, 500.0, -50.0])
        self.assertEqual(result["environment"]["obstacles"][0]["radius"], 50.0)
        self.assertEqual(result["environment"]["obstacles"][1]["position"], [600.0, 700.0, -50.0])
        self.assertEqual(result["environment"]["obstacles"][1]["radius"], 30.0)

    def test_english_position_input_is_supported(self) -> None:
        result = parse_detection_text(
            """
            Detected possible target at position (200, 300)
            Bearing: 030 degrees
            Range: approximately 800 meters
            """
        )

        self.assertEqual(result["mission"]["target_position"], [200.0, 300.0, -50.0])
        self.assertEqual(len(result["environment"]["baits"]), 1)

    def test_default_range_is_used_when_only_bearing_is_present(self) -> None:
        parser = DetectionParser(default_detection_range=500.0)
        result = parser.parse("声呐接触：方向：正东")

        target = result["mission"]["target_position"]
        self.assertAlmostEqual(target[0], 500.0, places=3)
        self.assertAlmostEqual(target[1], 0.0, places=3)

    def test_multiple_suspected_targets_with_compact_bearings(self) -> None:
        parser = DetectionParser(default_detection_range=500.0)
        result = parser.parse("北偏东39度有1个疑似目标、68度有1个疑似目标")

        baits = result["environment"]["baits"]
        self.assertEqual(len(baits), 2)
        self.assertAlmostEqual(baits[0]["position"][0], 314.66, places=2)
        self.assertAlmostEqual(baits[0]["position"][1], 388.573, places=2)
        self.assertAlmostEqual(baits[1]["position"][0], 463.592, places=2)
        self.assertAlmostEqual(baits[1]["position"][1], 187.303, places=2)

    def test_orbit_instruction_is_parsed_as_mission_constraint(self) -> None:
        parser = DetectionParser(default_detection_range=500.0)
        result = parser.parse("有一个疑似目标在北偏东30度方向，距离300米，首先接近他，然后围着他绕两圈")

        self.assertEqual(result["mission"]["constraints"]["orbit_turns"], 2)
        self.assertEqual(len(result["environment"]["baits"]), 1)
        self.assertAlmostEqual(result["mission"]["target_position"][0], 150.0, places=3)
        self.assertAlmostEqual(result["mission"]["target_position"][1], 259.808, places=3)

    def test_close_reconnaissance_sets_default_orbit_action(self) -> None:
        parser = DetectionParser(default_detection_range=500.0)
        result = parser.parse("北偏东45度方向400m距离附近有一个位置目标，需要uuv去抵近侦察")

        constraints = result["mission"]["constraints"]
        self.assertEqual(constraints["orbit_turns"], 2)
        self.assertEqual(constraints["orbit_radius"], 10.0)
        self.assertEqual(len(result["environment"]["baits"]), 1)
        self.assertAlmostEqual(result["mission"]["target_position"][0], 282.843, places=3)
        self.assertAlmostEqual(result["mission"]["target_position"][1], 282.843, places=3)

    def test_multiple_close_reconnaissance_targets_are_split(self) -> None:
        parser = DetectionParser(default_detection_range=500.0)
        result = parser.parse("北偏东30度方向400m、45度方向600m距离分别有2个目标，需要uuv分别去抵近侦察")

        constraints = result["mission"]["constraints"]
        baits = result["environment"]["baits"]
        self.assertEqual(constraints["orbit_turns"], 2)
        self.assertEqual(constraints["orbit_radius"], 10.0)
        self.assertEqual(len(baits), 2)
        self.assertAlmostEqual(baits[0]["position"][0], 200.0, places=3)
        self.assertAlmostEqual(baits[0]["position"][1], 346.41, places=3)
        self.assertAlmostEqual(baits[1]["position"][0], 424.264, places=3)
        self.assertAlmostEqual(baits[1]["position"][1], 424.264, places=3)

    def test_current_position_is_not_treated_as_target(self) -> None:
        parser = DetectionParser()
        result = parser.parse("当前位置：(10, 20, -60)，探测到目标位置：(210, 120)")

        self.assertEqual(result["uuv_state"]["position"], [10.0, 20.0, -60.0])
        self.assertEqual(result["mission"]["target_position"], [210.0, 120.0, -50.0])
        self.assertEqual(len(result["environment"]["baits"]), 1)

    def test_coverage_scenario_returns_valid_coverage_payload(self) -> None:
        parser = DetectionParser()
        result = parser.parse("扫描 500m x 300m 搜索区域")

        self.assertEqual(result["mission"]["scenario"], "area_coverage")
        self.assertEqual(result["mission"]["coverage_area"], [[0.0, 0.0], [500.0, 0.0], [500.0, 300.0], [0.0, 300.0]])
        SituationAwareness.from_dict(result)

    def test_parse_time_stays_under_100ms(self) -> None:
        parser = DetectionParser()
        started = time.perf_counter()
        parser.parse("声呐接触：方向：北偏东30度，距离：约800米")
        elapsed = time.perf_counter() - started

        self.assertLess(elapsed, 0.1)


if __name__ == "__main__":
    unittest.main()
