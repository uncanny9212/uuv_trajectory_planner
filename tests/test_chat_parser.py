"""Chat parser behavior tests."""

from __future__ import annotations

import unittest

from uuv_trajectory_planner.core.chat_parser import payload_from_message


class TestChatParser(unittest.TestCase):
    """Validate semantic generation of manual scene objects."""

    def test_general_message_starts_with_empty_scene_objects(self) -> None:
        payload = payload_from_message("从 (0,0) 出发到 (1000,800)，安全距离 50，航速 3。")

        self.assertEqual(payload["environment"]["obstacles"], [])
        self.assertEqual(payload["environment"]["baits"], [])

    def test_general_message_generates_objects_from_semantics(self) -> None:
        payload = payload_from_message(
            "从 (0,0) 出发到 (1000,800)，安全距离 50，航速 3。"
            "障碍物 O001 位于 (300,200,-50)，半径 80。"
            "饵物 B001 位于 (520,350,-50)，逼近半径 45。"
        )

        self.assertEqual(
            payload["environment"]["obstacles"],
            [{"id": "O001", "type": "static", "position": [300.0, 200.0, -50.0], "radius": 80.0}],
        )
        self.assertEqual(
            payload["environment"]["baits"],
            [{"id": "B001", "position": [520.0, 350.0, -50.0], "radius": 45.0}],
        )


if __name__ == "__main__":
    unittest.main()
