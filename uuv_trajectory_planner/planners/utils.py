"""Geometry and trajectory helper functions."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

from uuv_trajectory_planner.models.constraints import PlanningConstraints
from uuv_trajectory_planner.models.decision_output import TrajectoryPoint
from uuv_trajectory_planner.models.situation_awareness import Obstacle, Vector2, Vector3


def distance_2d(a: Sequence[float], b: Sequence[float]) -> float:
    """Return planar distance."""

    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def distance_3d(a: Sequence[float], b: Sequence[float]) -> float:
    """Return 3D distance."""

    return math.sqrt(
        (float(a[0]) - float(b[0])) ** 2
        + (float(a[1]) - float(b[1])) ** 2
        + (float(a[2]) - float(b[2])) ** 2
    )


def heading_between(a: Sequence[float], b: Sequence[float]) -> float:
    """Return heading in degrees from point a to point b."""

    angle = math.degrees(math.atan2(float(b[1]) - float(a[1]), float(b[0]) - float(a[0])))
    return (angle + 360.0) % 360.0


def heading_delta(a: float, b: float) -> float:
    """Smallest absolute difference between two headings."""

    diff = (b - a + 180.0) % 360.0 - 180.0
    return abs(diff)


def turn_angle(previous: Sequence[float], current: Sequence[float], following: Sequence[float]) -> float:
    """Return local turn angle in degrees."""

    return heading_delta(heading_between(previous, current), heading_between(current, following))


def path_distance(path: Sequence[Vector3]) -> float:
    """Return polyline length."""

    return sum(distance_3d(path[index], path[index + 1]) for index in range(len(path) - 1))


def segment_point_distance_2d(a: Sequence[float], b: Sequence[float], p: Sequence[float]) -> float:
    """Return distance from point p to segment ab in the horizontal plane."""

    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    px, py = float(p[0]), float(p[1])
    dx = bx - ax
    dy = by - ay
    if dx == 0.0 and dy == 0.0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    nearest_x = ax + t * dx
    nearest_y = ay + t * dy
    return math.hypot(px - nearest_x, py - nearest_y)


def is_segment_safe(
    a: Sequence[float],
    b: Sequence[float],
    obstacles: Iterable[Obstacle],
    min_obstacle_distance: float,
) -> bool:
    """Check whether a segment clears all circular obstacles."""

    for obstacle in obstacles:
        center_distance = segment_point_distance_2d(a, b, obstacle.position)
        if center_distance < obstacle.radius + min_obstacle_distance:
            return False
    return True


def min_obstacle_clearance(
    path: Sequence[Vector3],
    obstacles: Iterable[Obstacle],
) -> float:
    """Return the minimum center-to-edge clearance of a path."""

    if len(path) < 2:
        return float("inf")
    clearance = float("inf")
    for index in range(len(path) - 1):
        for obstacle in obstacles:
            center_distance = segment_point_distance_2d(path[index], path[index + 1], obstacle.position)
            clearance = min(clearance, center_distance - obstacle.radius)
    return clearance


def dedupe_path(path: Sequence[Vector3], tolerance: float = 1e-6) -> List[Vector3]:
    """Remove consecutive duplicate waypoints."""

    result: List[Vector3] = []
    for point in path:
        if not result or distance_3d(result[-1], point) > tolerance:
            result.append(point)
    return result


def densify_path(path: Sequence[Vector3], max_step: float) -> List[Vector3]:
    """Insert intermediate points so each segment length stays below max_step."""

    if len(path) < 2:
        return list(path)
    result: List[Vector3] = [path[0]]
    for start, end in zip(path[:-1], path[1:]):
        segment_length = distance_3d(start, end)
        count = max(1, int(math.ceil(segment_length / max_step)))
        for step in range(1, count + 1):
            ratio = step / count
            result.append(
                (
                    start[0] + (end[0] - start[0]) * ratio,
                    start[1] + (end[1] - start[1]) * ratio,
                    start[2] + (end[2] - start[2]) * ratio,
                )
            )
    return dedupe_path(result)


def chaikin_smooth(path: Sequence[Vector3], iterations: int = 3) -> List[Vector3]:
    """Smooth a path using Chaikin corner cutting."""

    if len(path) < 3:
        return list(path)
    smoothed = list(path)
    for _ in range(iterations):
        next_path: List[Vector3] = [smoothed[0]]
        for start, end in zip(smoothed[:-1], smoothed[1:]):
            q = (
                0.75 * start[0] + 0.25 * end[0],
                0.75 * start[1] + 0.25 * end[1],
                0.75 * start[2] + 0.25 * end[2],
            )
            r = (
                0.25 * start[0] + 0.75 * end[0],
                0.25 * start[1] + 0.75 * end[1],
                0.25 * start[2] + 0.75 * end[2],
            )
            next_path.extend([q, r])
        next_path.append(smoothed[-1])
        smoothed = dedupe_path(next_path)
    return smoothed


def shortcut_path(
    path: Sequence[Vector3],
    obstacles: Iterable[Obstacle],
    min_obstacle_distance: float,
) -> List[Vector3]:
    """Remove unnecessary intermediate waypoints while preserving obstacle clearance."""

    if len(path) < 3:
        return list(path)
    obstacle_list = list(obstacles)
    result: List[Vector3] = [path[0]]
    index = 0
    while index < len(path) - 1:
        next_index = len(path) - 1
        while next_index > index + 1:
            if is_segment_safe(path[index], path[next_index], obstacle_list, min_obstacle_distance):
                break
            next_index -= 1
        result.append(path[next_index])
        index = next_index
    return dedupe_path(result)


def max_turn_angle(path: Sequence[Vector3]) -> float:
    """Return maximum local heading change across a path."""

    if len(path) < 3:
        return 0.0
    return max(turn_angle(path[index - 1], path[index], path[index + 1]) for index in range(1, len(path) - 1))


def build_trajectory_points(path: Sequence[Vector3], speed: float) -> List[TrajectoryPoint]:
    """Convert a raw path to executable trajectory points."""

    if not path:
        return []
    trajectory: List[TrajectoryPoint] = []
    elapsed = 0.0
    for index, point in enumerate(path):
        if index < len(path) - 1:
            heading = heading_between(point, path[index + 1])
        elif index > 0:
            heading = heading_between(path[index - 1], point)
        else:
            heading = 0.0
        if index > 0:
            elapsed += distance_3d(path[index - 1], point) / max(speed, 0.1)
        trajectory.append(
            TrajectoryPoint(
                point_id=index + 1,
                coordinates=point,
                heading=heading,
                speed=speed,
                timestamp=elapsed,
            )
        )
    return trajectory


def validate_general_path(
    path: Sequence[Vector3],
    obstacles: Iterable[Obstacle],
    constraints: PlanningConstraints,
) -> Tuple[List[str], Dict[str, float]]:
    """Validate obstacle and smoothness constraints for a general path."""

    obstacle_list = list(obstacles)
    clearance = min_obstacle_clearance(path, obstacle_list)
    turn = max_turn_angle(path)
    satisfied: List[str] = []
    if clearance >= constraints.min_obstacle_distance - 1e-6:
        satisfied.append("obstacle_avoidance")
    if turn <= constraints.max_turn_angle_deg + 1e-6:
        satisfied.append("smoothness")
    if constraints.max_speed > 0:
        satisfied.append("speed_limit")
    return satisfied, {
        "min_obstacle_clearance": clearance,
        "max_turn_angle": turn,
        "total_distance": path_distance(path),
    }


def polygon_area(polygon: Sequence[Vector2]) -> float:
    """Return absolute polygon area."""

    if len(polygon) < 3:
        return 0.0
    total = 0.0
    for index, point in enumerate(polygon):
        next_point = polygon[(index + 1) % len(polygon)]
        total += point[0] * next_point[1] - next_point[0] * point[1]
    return abs(total) / 2.0


def polygon_bounds(polygon: Sequence[Vector2]) -> Tuple[float, float, float, float]:
    """Return min_x, min_y, max_x, max_y."""

    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def point_in_polygon(point: Sequence[float], polygon: Sequence[Vector2]) -> bool:
    """Ray-casting point-in-polygon test."""

    x, y = float(point[0]), float(point[1])
    inside = False
    j = len(polygon) - 1
    for i, current in enumerate(polygon):
        previous = polygon[j]
        if ((current[1] > y) != (previous[1] > y)) and (
            x < (previous[0] - current[0]) * (y - current[1]) / ((previous[1] - current[1]) or 1e-12) + current[0]
        ):
            inside = not inside
        j = i
    return inside


def scanline_intersections(polygon: Sequence[Vector2], y: float) -> List[float]:
    """Return sorted x intersections between polygon edges and a horizontal scanline."""

    intersections: List[float] = []
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        y1, y2 = start[1], end[1]
        if y1 == y2:
            continue
        if min(y1, y2) <= y < max(y1, y2):
            ratio = (y - y1) / (y2 - y1)
            intersections.append(start[0] + ratio * (end[0] - start[0]))
    return sorted(intersections)


def distance_point_to_polyline(point: Sequence[float], path: Sequence[Vector3]) -> float:
    """Return distance from a point to a trajectory polyline."""

    if len(path) < 2:
        return float("inf")
    return min(segment_point_distance_2d(path[index], path[index + 1], point) for index in range(len(path) - 1))


def estimate_coverage_rate(
    polygon: Sequence[Vector2],
    path: Sequence[Vector3],
    sweep_width: float,
    samples_per_axis: int = 60,
) -> float:
    """Estimate area coverage using deterministic grid sampling."""

    min_x, min_y, max_x, max_y = polygon_bounds(polygon)
    covered = 0
    inside = 0
    if max_x == min_x or max_y == min_y:
        return 0.0
    for x_index in range(samples_per_axis):
        x = min_x + (x_index + 0.5) * (max_x - min_x) / samples_per_axis
        for y_index in range(samples_per_axis):
            y = min_y + (y_index + 0.5) * (max_y - min_y) / samples_per_axis
            if point_in_polygon((x, y), polygon):
                inside += 1
                if distance_point_to_polyline((x, y), path) <= sweep_width / 2.0 + 1e-6:
                    covered += 1
    return covered / inside if inside else 0.0
