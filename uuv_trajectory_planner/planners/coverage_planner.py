"""Area coverage planner."""

from __future__ import annotations

from typing import List, Sequence

from uuv_trajectory_planner.models.decision_output import AlternativePlan, DecisionOutput
from uuv_trajectory_planner.models.situation_awareness import SituationAwareness, Vector2, Vector3
from uuv_trajectory_planner.planners import utils


class CoveragePlanner:
    """Generate a boustrophedon coverage trajectory for simple polygons."""

    def __init__(self, samples_per_axis: int = 60) -> None:
        self.samples_per_axis = samples_per_axis

    def plan(self, situation: SituationAwareness, reasoning: str = "") -> DecisionOutput:
        """Generate a structured area coverage trajectory decision."""

        constraints = situation.mission.constraints
        polygon = situation.mission.coverage_area or situation.environment.boundaries
        if len(polygon) < 3:
            raise ValueError("area coverage requires polygon boundary")

        sweep_width = constraints.sweep_width
        speed = max(0.1, min(constraints.max_speed, situation.uuv_state.speed or constraints.max_speed))
        lane_path = self._boustrophedon_path(polygon, sweep_width, situation.uuv_state.position[2])
        start_connected = [situation.uuv_state.position] + lane_path
        path = utils.densify_path(start_connected, max(35.0, sweep_width))
        coverage_rate = utils.estimate_coverage_rate(
            polygon,
            lane_path,
            sweep_width,
            samples_per_axis=self.samples_per_axis,
        )
        total_distance = utils.path_distance(path)
        estimated_time = total_distance / speed
        satisfied: List[str] = ["scan_spacing", "boundary_handling"]
        if coverage_rate >= constraints.coverage_required:
            satisfied.append("coverage_rate")
        if constraints.max_speed > 0:
            satisfied.append("speed_limit")

        min_x, min_y, max_x, max_y = utils.polygon_bounds(polygon)
        reason = (
            f"态势分析：覆盖区域边界框为{max_x - min_x:.1f}m x {max_y - min_y:.1f}m，"
            f"扫测宽度{sweep_width:.1f}m，目标覆盖率{constraints.coverage_required:.0%}。"
            "约束识别：扫描线保持半扫宽边界余量，相邻航迹间距等于扫测宽度。"
            "算法选择：采用往返式Boustrophedon扫描，适合矩形和简单多边形区域。"
            f"验证检查：栅格估算覆盖率{coverage_rate:.1%}，总航程{total_distance:.1f}m。"
        )
        if reasoning:
            reason = f"{reasoning} {reason}"

        lane_distance = utils.path_distance(lane_path)
        alternatives = [
            AlternativePlan("boustrophedon", total_distance, "balanced"),
            AlternativePlan("spiral_scan", lane_distance * 1.08, "lower_turn_count_for_round_areas"),
        ]
        confidence = 0.72 + min(0.2, coverage_rate * 0.2)
        if coverage_rate < constraints.coverage_required:
            confidence -= 0.25

        return DecisionOutput(
            decision_type="trajectory_planning",
            scenario="area_coverage",
            trajectory=utils.build_trajectory_points(path, speed),
            total_distance=total_distance,
            estimated_time=estimated_time,
            constraints_satisfied=satisfied,
            confidence=confidence,
            reasoning_chain=reason,
            alternatives=alternatives,
            feedback={
                "validation": {
                    "coverage_rate": coverage_rate,
                    "polygon_area": utils.polygon_area(polygon),
                    "sweep_width": sweep_width,
                    "lane_count": len(lane_path) // 2,
                },
                "algorithm": "boustrophedon",
            },
        )

    def _boustrophedon_path(
        self,
        polygon: Sequence[Vector2],
        sweep_width: float,
        z: float,
    ) -> List[Vector3]:
        min_x, min_y, max_x, max_y = utils.polygon_bounds(polygon)
        y = min_y + sweep_width / 2.0
        lanes: List[Vector3] = []
        reverse = False
        while y <= max_y - sweep_width / 2.0 + 1e-9:
            xs = utils.scanline_intersections(polygon, y)
            if len(xs) >= 2:
                pairs = [(xs[index], xs[index + 1]) for index in range(0, len(xs) - 1, 2)]
                if reverse:
                    pairs = list(reversed(pairs))
                for left, right in pairs:
                    start_x = left + sweep_width / 2.0
                    end_x = right - sweep_width / 2.0
                    if end_x < start_x:
                        start_x, end_x = left, right
                    if reverse:
                        lanes.extend([(end_x, y, z), (start_x, y, z)])
                    else:
                        lanes.extend([(start_x, y, z), (end_x, y, z)])
                    reverse = not reverse
            y += sweep_width
        if not lanes:
            center_y = (min_y + max_y) / 2.0
            lanes = [(min_x, center_y, z), (max_x, center_y, z)]
        return lanes
