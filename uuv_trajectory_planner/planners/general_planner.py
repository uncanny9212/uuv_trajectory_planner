"""General point-to-point trajectory planner."""

from __future__ import annotations

import heapq
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from uuv_trajectory_planner.models.decision_output import AlternativePlan, DecisionOutput
from uuv_trajectory_planner.models.situation_awareness import BaitTarget, Obstacle, SituationAwareness, Vector3
from uuv_trajectory_planner.planners import utils

GridNode = Tuple[int, int]


class GeneralTrajectoryPlanner:
    """Plan a safe trajectory from the current UUV state to a target position."""

    def __init__(
        self,
        grid_resolution: float = 50.0,
        max_search_nodes: int = 25000,
        smoothing_iterations: int = 4,
        dynamic_prediction_seconds: Optional[Sequence[float]] = None,
    ) -> None:
        self.grid_resolution = grid_resolution
        self.max_search_nodes = max_search_nodes
        self.smoothing_iterations = smoothing_iterations
        self.dynamic_prediction_seconds = list(dynamic_prediction_seconds or [60.0, 120.0, 180.0])

    def plan(self, situation: SituationAwareness, reasoning: str = "") -> DecisionOutput:
        """Generate a structured point-to-point trajectory decision."""

        constraints = situation.mission.constraints
        start = situation.uuv_state.position
        if situation.mission.target_position is None:
            raise ValueError("general planner requires target_position")
        goal = situation.mission.target_position
        speed = max(0.1, min(constraints.max_speed, situation.uuv_state.speed or constraints.max_speed))
        planning_obstacles = self._planning_obstacles(situation.environment.obstacles)
        clearance_buffer = max(10.0, self.grid_resolution * 0.25)
        required_clearance = constraints.min_obstacle_distance + clearance_buffer
        inflation = required_clearance + max(5.0, self.grid_resolution * 0.25)
        bait_waypoints = self._bait_waypoints(
            start,
            situation,
            planning_obstacles,
            required_clearance,
        )
        route_goals, orbit_waypoints = self._route_goals_with_orbits(
            start,
            goal,
            bait_waypoints,
            situation,
            planning_obstacles,
            required_clearance,
        )

        path, algorithms = self._plan_route(
            start,
            route_goals,
            situation,
            planning_obstacles,
            inflation,
            required_clearance,
        )
        if orbit_waypoints:
            algorithms.append("orbit_pattern")
        algorithm = "+".join(sorted(set(algorithms)))
        satisfied, report = utils.validate_general_path(path, situation.environment.obstacles, constraints)
        bait_report = self._bait_report(path, situation.environment.baits)
        bait_report["bait_waypoints"] = {
            bait.id: list(waypoint) for bait, waypoint in zip(situation.environment.baits, bait_waypoints)
        }
        report.update(bait_report)
        if situation.environment.baits and bait_report["baits_reached"] == len(situation.environment.baits):
            satisfied.append("bait_approach")
        report["orbit_turns_requested"] = float(constraints.orbit_turns)
        report["orbit_turns_planned"] = float(constraints.orbit_turns if orbit_waypoints else 0)
        report["orbit_waypoint_count"] = float(len(orbit_waypoints))
        report["orbit_targets_planned"] = float(self._orbit_target_count(orbit_waypoints, situation))
        if constraints.orbit_turns > 0 and orbit_waypoints:
            satisfied.append("target_orbit")
        total_distance = report["total_distance"]
        estimated_time = total_distance / speed
        trajectory = utils.build_trajectory_points(path, speed)
        confidence = self._confidence(satisfied, report, len(situation.environment.obstacles), algorithm)
        bait_text = (
            f"，需要逼近{len(situation.environment.baits)}个饵物"
            if situation.environment.baits
            else ""
        )
        orbit_text = (
            f"，并围绕目标环绕{constraints.orbit_turns}圈"
            if constraints.orbit_turns > 0 and orbit_waypoints
            else ("，但环绕动作因安全边界未生成" if constraints.orbit_turns > 0 else "")
        )

        reason = (
            f"态势分析：起点{tuple(round(v, 1) for v in start)}，目标"
            f"{tuple(round(v, 1) for v in goal)}，检测到{len(situation.environment.obstacles)}个障碍物{bait_text}{orbit_text}。"
            f"约束识别：安全距离{constraints.min_obstacle_distance:.1f}m，最大航速"
            f"{constraints.max_speed:.1f}m/s，最大转角{constraints.max_turn_angle_deg:.1f}度。"
            f"算法选择：采用{algorithm}分段生成路径，将饵物作为吸引航点；若饵物位于障碍物安全区内，"
            "选择最近的安全逼近点；若有环绕要求，则在目标周围追加圆周航点。"
            f"验证检查：最小障碍物净距{report['min_obstacle_clearance']:.1f}m，"
            f"最大相邻航向变化{report['max_turn_angle']:.1f}度，"
            f"已逼近饵物{bait_report['baits_reached']}/{bait_report['bait_count']}。"
        )
        if reasoning:
            reason = f"{reasoning} {reason}"

        alternatives = [
            AlternativePlan("shortest_path", utils.distance_3d(start, goal), "higher_if_obstacles_present"),
            AlternativePlan("bait_guided", total_distance, "balanced"),
        ]
        return DecisionOutput(
            decision_type="trajectory_planning",
            scenario="general",
            trajectory=trajectory,
            total_distance=total_distance,
            estimated_time=estimated_time,
            constraints_satisfied=satisfied,
            confidence=confidence,
            reasoning_chain=reason,
            alternatives=alternatives,
            feedback={"validation": report, "algorithm": algorithm},
        )

    def _bait_waypoints(
        self,
        start: Vector3,
        situation: SituationAwareness,
        obstacles: Sequence[Obstacle],
        required_clearance: float,
    ) -> List[Vector3]:
        waypoints: List[Vector3] = []
        current = start
        for bait in situation.environment.baits:
            waypoint = self._safe_bait_waypoint(
                current,
                bait,
                situation,
                obstacles,
                required_clearance,
            )
            waypoints.append(waypoint)
            current = waypoint
        return waypoints

    def _safe_bait_waypoint(
        self,
        current: Vector3,
        bait: BaitTarget,
        situation: SituationAwareness,
        obstacles: Sequence[Obstacle],
        required_clearance: float,
    ) -> Vector3:
        if self._point_is_safe(bait.position, situation, obstacles, required_clearance):
            return bait.position

        candidate_radii = [
            max(5.0, bait.radius * 0.5),
            max(10.0, bait.radius),
            bait.radius + required_clearance * 0.5,
            bait.radius + required_clearance,
            bait.radius + required_clearance * 1.5,
            bait.radius + required_clearance * 2.0,
        ]
        best: Optional[Vector3] = None
        best_score = float("inf")
        for radius in candidate_radii:
            for step in range(48):
                angle = 2.0 * math.pi * step / 48.0
                candidate = (
                    bait.position[0] + math.cos(angle) * radius,
                    bait.position[1] + math.sin(angle) * radius,
                    bait.position[2],
                )
                if not self._point_is_safe(candidate, situation, obstacles, required_clearance):
                    continue
                score = utils.distance_2d(candidate, bait.position) * 1000.0 + utils.distance_2d(current, candidate)
                if score < best_score:
                    best = candidate
                    best_score = score
        return best if best is not None else bait.position

    def _route_goals_with_orbits(
        self,
        start: Vector3,
        goal: Vector3,
        bait_waypoints: Sequence[Vector3],
        situation: SituationAwareness,
        obstacles: Sequence[Obstacle],
        required_clearance: float,
    ) -> Tuple[List[Vector3], List[Vector3]]:
        constraints = situation.mission.constraints
        route_goals: List[Vector3] = []
        orbit_waypoints: List[Vector3] = []
        current = start

        for bait, waypoint in zip(situation.environment.baits, bait_waypoints):
            route_goals.append(waypoint)
            current = waypoint
            if constraints.orbit_turns > 0:
                center = bait.position if self._point_is_safe(bait.position, situation, obstacles, required_clearance) else waypoint
                target_orbit = self._orbit_waypoints(
                    center,
                    current,
                    situation,
                    obstacles,
                    required_clearance,
                )
                if target_orbit:
                    route_goals.extend(target_orbit)
                    orbit_waypoints.extend(target_orbit)
                    current = target_orbit[-1]

        if not self._route_contains_goal(route_goals, goal):
            route_goals.append(goal)
            current = goal
            if constraints.orbit_turns > 0 and not situation.environment.baits:
                target_orbit = self._orbit_waypoints(
                    goal,
                    current,
                    situation,
                    obstacles,
                    required_clearance,
                )
                route_goals.extend(target_orbit)
                orbit_waypoints.extend(target_orbit)

        return utils.dedupe_path(route_goals), orbit_waypoints

    def _route_contains_goal(self, route_goals: Sequence[Vector3], goal: Vector3) -> bool:
        tolerance = max(1.0, self.grid_resolution * 0.05)
        return any(utils.distance_3d(point, goal) <= tolerance for point in route_goals)

    def _orbit_target_count(self, orbit_waypoints: Sequence[Vector3], situation: SituationAwareness) -> int:
        if not orbit_waypoints:
            return 0
        constraints = situation.mission.constraints
        points_per_target = max(1, constraints.orbit_turns * 24)
        return min(len(situation.environment.baits) or 1, int(math.ceil(len(orbit_waypoints) / points_per_target)))

    def _orbit_waypoints(
        self,
        center: Vector3,
        approach_from: Vector3,
        situation: SituationAwareness,
        obstacles: Sequence[Obstacle],
        required_clearance: float,
    ) -> List[Vector3]:
        constraints = situation.mission.constraints
        if constraints.orbit_turns <= 0:
            return []

        bait_radius = next(
            (
                bait.radius
                for bait in situation.environment.baits
                if utils.distance_2d(bait.position, center) <= max(1.0, bait.radius)
            ),
            40.0,
        )
        base_radius = constraints.orbit_radius or max(bait_radius, constraints.min_turning_radius)
        candidate_radii = [
            base_radius,
            base_radius + self.grid_resolution * 0.5,
            base_radius + self.grid_resolution,
            base_radius + required_clearance,
        ]
        start_angle = math.atan2(approach_from[1] - center[1], approach_from[0] - center[0])
        points_per_turn = max(24, int(math.ceil((2.0 * math.pi * base_radius) / max(10.0, self.grid_resolution * 0.35))))
        total_steps = points_per_turn * constraints.orbit_turns

        for radius in candidate_radii:
            waypoints = [
                (
                    center[0] + math.cos(start_angle - 2.0 * math.pi * step / points_per_turn) * radius,
                    center[1] + math.sin(start_angle - 2.0 * math.pi * step / points_per_turn) * radius,
                    center[2],
                )
                for step in range(total_steps + 1)
            ]
            if self._orbit_is_safe(center, waypoints, situation, obstacles, required_clearance):
                return waypoints
        return []

    def _orbit_is_safe(
        self,
        center: Vector3,
        waypoints: Sequence[Vector3],
        situation: SituationAwareness,
        obstacles: Sequence[Obstacle],
        required_clearance: float,
    ) -> bool:
        if not waypoints:
            return False
        if not utils.is_segment_safe(center, waypoints[0], obstacles, required_clearance):
            return False
        for point in waypoints:
            if not self._point_is_safe(point, situation, obstacles, required_clearance):
                return False
        for start, end in zip(waypoints[:-1], waypoints[1:]):
            if not utils.is_segment_safe(start, end, obstacles, required_clearance):
                return False
        return True

    def _point_is_safe(
        self,
        point: Vector3,
        situation: SituationAwareness,
        obstacles: Sequence[Obstacle],
        required_clearance: float,
    ) -> bool:
        if situation.environment.boundaries and not utils.point_in_polygon(point, situation.environment.boundaries):
            return False
        return all(
            utils.distance_2d(point, obstacle.position) - obstacle.radius >= required_clearance
            for obstacle in obstacles
        )

    def _plan_route(
        self,
        start: Vector3,
        route_goals: Sequence[Vector3],
        situation: SituationAwareness,
        planning_obstacles: Sequence[Obstacle],
        inflation: float,
        min_obstacle_distance: float,
    ) -> Tuple[List[Vector3], List[str]]:
        path: List[Vector3] = [start]
        algorithms: List[str] = []
        segment_start = start
        for segment_goal in route_goals:
            raw_segment, algorithm = self._plan_segment(
                segment_start,
                segment_goal,
                situation,
                planning_obstacles,
                inflation,
                min_obstacle_distance,
            )
            segment = self._post_process_path(raw_segment, planning_obstacles, min_obstacle_distance)
            path.extend(segment[1:])
            algorithms.append(algorithm)
            segment_start = segment_goal
        return utils.dedupe_path(path), algorithms

    def _plan_segment(
        self,
        start: Vector3,
        goal: Vector3,
        situation: SituationAwareness,
        planning_obstacles: Sequence[Obstacle],
        inflation: float,
        min_obstacle_distance: float,
    ) -> Tuple[List[Vector3], str]:
        if utils.is_segment_safe(start, goal, planning_obstacles, inflation):
            return [start, goal], "direct_line"
        raw_path = self._a_star(start, goal, situation, planning_obstacles, inflation)
        if raw_path:
            return raw_path, "a_star_grid"
        return self._detour_fallback(start, goal, situation.environment.obstacles, min_obstacle_distance), "detour_fallback"

    def _bait_report(self, path: Sequence[Vector3], baits: Sequence[BaitTarget]) -> Dict[str, float]:
        distances: Dict[str, float] = {}
        reached = 0
        for bait in baits:
            distance = utils.distance_point_to_polyline(bait.position, path)
            distances[bait.id] = distance
            if distance <= bait.radius:
                reached += 1
        return {
            "bait_count": float(len(baits)),
            "baits_reached": float(reached),
            "bait_min_distances": distances,  # type: ignore[dict-item]
        }

    def _planning_obstacles(self, obstacles: Iterable[Obstacle]) -> List[Obstacle]:
        """Expand moving obstacles into future conservative samples."""

        expanded: List[Obstacle] = []
        for obstacle in obstacles:
            expanded.append(obstacle)
            if obstacle.type == "moving" or any(abs(value) > 1e-9 for value in obstacle.velocity):
                expanded.extend(obstacle.predicted(seconds) for seconds in self.dynamic_prediction_seconds)
        return expanded

    def _bounds(
        self,
        start: Vector3,
        goal: Vector3,
        situation: SituationAwareness,
        obstacles: Sequence[Obstacle],
    ) -> Tuple[float, float, float, float]:
        if situation.environment.boundaries:
            min_x, min_y, max_x, max_y = utils.polygon_bounds(situation.environment.boundaries)
        else:
            xs = [start[0], goal[0]] + [obstacle.position[0] for obstacle in obstacles]
            ys = [start[1], goal[1]] + [obstacle.position[1] for obstacle in obstacles]
            margin = max(200.0, self.grid_resolution * 4.0)
            min_x, max_x = min(xs) - margin, max(xs) + margin
            min_y, max_y = min(ys) - margin, max(ys) + margin
        return min_x, min_y, max_x, max_y

    def _a_star(
        self,
        start: Vector3,
        goal: Vector3,
        situation: SituationAwareness,
        obstacles: Sequence[Obstacle],
        inflation: float,
    ) -> List[Vector3]:
        min_x, min_y, max_x, max_y = self._bounds(start, goal, situation, obstacles)
        resolution = max(20.0, self.grid_resolution)
        boundary = situation.environment.boundaries

        def to_node(point: Sequence[float]) -> GridNode:
            return (int(round((point[0] - min_x) / resolution)), int(round((point[1] - min_y) / resolution)))

        def to_point(node: GridNode) -> Vector3:
            return (min_x + node[0] * resolution, min_y + node[1] * resolution, start[2])

        max_i = int(math.ceil((max_x - min_x) / resolution))
        max_j = int(math.ceil((max_y - min_y) / resolution))
        start_node = to_node(start)
        goal_node = to_node(goal)

        def node_is_safe(node: GridNode) -> bool:
            if node[0] < 0 or node[1] < 0 or node[0] > max_i or node[1] > max_j:
                return False
            point = to_point(node)
            if boundary and not utils.point_in_polygon(point, boundary):
                if utils.distance_2d(point, start) > resolution and utils.distance_2d(point, goal) > resolution:
                    return False
            for obstacle in obstacles:
                if utils.distance_2d(point, obstacle.position) < obstacle.radius + inflation:
                    return False
            return True

        neighbors = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
        frontier: List[Tuple[float, GridNode]] = [(0.0, start_node)]
        came_from: Dict[GridNode, Optional[GridNode]] = {start_node: None}
        cost_so_far: Dict[GridNode, float] = {start_node: 0.0}
        visited = 0

        while frontier and visited < self.max_search_nodes:
            _, current = heapq.heappop(frontier)
            visited += 1
            if current == goal_node:
                break
            for di, dj in neighbors:
                candidate = (current[0] + di, current[1] + dj)
                if not node_is_safe(candidate):
                    continue
                current_point = to_point(current)
                candidate_point = to_point(candidate)
                if not utils.is_segment_safe(current_point, candidate_point, obstacles, inflation):
                    continue
                step_cost = utils.distance_2d(current_point, candidate_point)
                new_cost = cost_so_far[current] + step_cost
                if candidate not in cost_so_far or new_cost < cost_so_far[candidate]:
                    cost_so_far[candidate] = new_cost
                    priority = new_cost + utils.distance_2d(candidate_point, goal)
                    heapq.heappush(frontier, (priority, candidate))
                    came_from[candidate] = current

        if goal_node not in came_from:
            return []

        node_path: List[GridNode] = []
        cursor: Optional[GridNode] = goal_node
        while cursor is not None:
            node_path.append(cursor)
            cursor = came_from[cursor]
        node_path.reverse()
        path = [start]
        path.extend(to_point(node) for node in node_path[1:-1])
        path.append(goal)
        return utils.dedupe_path(path)

    def _detour_fallback(
        self,
        start: Vector3,
        goal: Vector3,
        obstacles: Sequence[Obstacle],
        min_distance: float,
    ) -> List[Vector3]:
        """Conservative fallback that bends around obstacles intersecting the direct route."""

        route: List[Vector3] = [start]
        dx = goal[0] - start[0]
        dy = goal[1] - start[1]
        length = math.hypot(dx, dy) or 1.0
        perpendicular = (-dy / length, dx / length)
        blocking = [
            obstacle
            for obstacle in obstacles
            if utils.segment_point_distance_2d(start, goal, obstacle.position) < obstacle.radius + min_distance
        ]
        blocking.sort(key=lambda obstacle: utils.distance_2d(start, obstacle.position))
        side = 1.0
        for obstacle in blocking:
            offset = obstacle.radius + min_distance + self.grid_resolution
            route.append(
                (
                    obstacle.position[0] + perpendicular[0] * offset * side,
                    obstacle.position[1] + perpendicular[1] * offset * side,
                    start[2],
                )
            )
            side *= -1.0
        route.append(goal)
        return utils.dedupe_path(route)

    def _post_process_path(
        self,
        raw_path: Sequence[Vector3],
        obstacles: Sequence[Obstacle],
        min_obstacle_distance: float,
    ) -> List[Vector3]:
        if len(raw_path) <= 2:
            return utils.densify_path(raw_path, max(40.0, self.grid_resolution))
        shortcut = utils.shortcut_path(raw_path, obstacles, min_obstacle_distance + self.grid_resolution * 0.15)
        best = utils.densify_path(shortcut, max(35.0, self.grid_resolution * 0.75))
        for iterations in range(self.smoothing_iterations, self.smoothing_iterations + 4):
            candidate = utils.densify_path(utils.chaikin_smooth(shortcut, iterations), max(30.0, self.grid_resolution * 0.5))
            if utils.min_obstacle_clearance(candidate, obstacles) >= min_obstacle_distance:
                best = candidate
                if utils.max_turn_angle(candidate) <= 30.0:
                    break
        return utils.dedupe_path(best)

    def _confidence(
        self,
        satisfied: Sequence[str],
        report: Dict[str, float],
        obstacle_count: int,
        algorithm: str,
    ) -> float:
        score = 0.65
        score += 0.12 if "obstacle_avoidance" in satisfied else -0.25
        score += 0.08 if "smoothness" in satisfied else -0.08
        score += 0.08 if "bait_approach" in satisfied else 0.0
        score += min(0.1, max(0.0, report.get("min_obstacle_clearance", 0.0)) / 1000.0)
        score -= min(0.12, obstacle_count * 0.02)
        if algorithm == "detour_fallback":
            score -= 0.08
        return max(0.1, min(0.98, score))
