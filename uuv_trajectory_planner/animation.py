"""Animated GIF rendering for trajectory decisions."""

from __future__ import annotations

import base64
import io
import math
from typing import Callable, List, Sequence, Tuple

from PIL import Image, ImageDraw

from uuv_trajectory_planner.models.decision_output import DecisionOutput
from uuv_trajectory_planner.models.situation_awareness import SituationAwareness

Point2 = Tuple[float, float]


def decision_gif_data_url(
    situation: SituationAwareness,
    decision: DecisionOutput,
    width: int = 720,
    height: int = 480,
    frame_count: int = 56,
) -> str:
    """Render an animated GIF and return it as a data URL."""

    frames = render_decision_gif(situation, decision, width, height, frame_count)
    buffer = io.BytesIO()
    frames[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=70,
        loop=0,
        optimize=True,
        disposal=2,
    )
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/gif;base64,{encoded}"


def render_decision_gif(
    situation: SituationAwareness,
    decision: DecisionOutput,
    width: int,
    height: int,
    frame_count: int,
) -> List[Image.Image]:
    """Create GIF frames for a trajectory decision."""

    path = [(point.coordinates[0], point.coordinates[1]) for point in decision.trajectory]
    boundary = situation.mission.coverage_area or situation.environment.boundaries
    points = _scene_points(situation, path, boundary)
    to_canvas = _projector(points, width, height)
    frames: List[Image.Image] = []
    steps = max(2, min(frame_count, max(12, len(path))))

    for frame_index in range(steps):
        progress = frame_index / (steps - 1)
        visible_path = _partial_path(path, progress)
        image = Image.new("RGB", (width, height), "#ffffff")
        draw = ImageDraw.Draw(image)
        _draw_grid(draw, width, height)
        _draw_boundary(draw, boundary, to_canvas)
        _draw_obstacles(draw, situation, to_canvas)
        _draw_baits(draw, situation, to_canvas)
        _draw_path(draw, visible_path, to_canvas)
        _draw_start_end(draw, path, visible_path, to_canvas)
        _draw_metrics(draw, decision, progress)
        frames.append(image)
    return frames


def _scene_points(situation: SituationAwareness, path: Sequence[Point2], boundary: Sequence[Point2]) -> List[Point2]:
    points: List[Point2] = list(path)
    points.extend(boundary)
    min_distance = situation.mission.constraints.min_obstacle_distance
    for obstacle in situation.environment.obstacles:
        radius = obstacle.radius + min_distance
        points.append((obstacle.position[0] - radius, obstacle.position[1] - radius))
        points.append((obstacle.position[0] + radius, obstacle.position[1] + radius))
    for bait in situation.environment.baits:
        points.append((bait.position[0] - bait.radius, bait.position[1] - bait.radius))
        points.append((bait.position[0] + bait.radius, bait.position[1] + bait.radius))
    if not points:
        points.append((0.0, 0.0))
    return points


def _projector(points: Sequence[Point2], width: int, height: int) -> Callable[[Sequence[float]], Tuple[int, int]]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    padding = max(30.0, (max_x - min_x + max_y - min_y) * 0.04)
    min_x -= padding
    max_x += padding
    min_y -= padding
    max_y += padding
    margin = 42
    scale = min((width - margin * 2) / max(1.0, max_x - min_x), (height - margin * 2) / max(1.0, max_y - min_y))

    def to_canvas(point: Sequence[float]) -> Tuple[int, int]:
        x = margin + (float(point[0]) - min_x) * scale
        y = height - margin - (float(point[1]) - min_y) * scale
        return int(round(x)), int(round(y))

    return to_canvas


def _partial_path(path: Sequence[Point2], progress: float) -> List[Point2]:
    if len(path) <= 1:
        return list(path)
    target = progress * (len(path) - 1)
    whole = int(target)
    partial = list(path[: whole + 1])
    if whole < len(path) - 1:
        ratio = target - whole
        start = path[whole]
        end = path[whole + 1]
        partial.append((start[0] + (end[0] - start[0]) * ratio, start[1] + (end[1] - start[1]) * ratio))
    return partial


def _draw_grid(draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
    for x in range(42, width - 41, 60):
        draw.line([(x, 42), (x, height - 42)], fill="#edf1f2", width=1)
    for y in range(42, height - 41, 60):
        draw.line([(42, y), (width - 42, y)], fill="#edf1f2", width=1)


def _draw_boundary(
    draw: ImageDraw.ImageDraw,
    boundary: Sequence[Point2],
    to_canvas: Callable[[Sequence[float]], Tuple[int, int]],
) -> None:
    if len(boundary) < 3:
        return
    projected = [to_canvas(point) for point in boundary]
    draw.line(projected + [projected[0]], fill="#5a686e", width=2)


def _draw_obstacles(
    draw: ImageDraw.ImageDraw,
    situation: SituationAwareness,
    to_canvas: Callable[[Sequence[float]], Tuple[int, int]],
) -> None:
    min_distance = situation.mission.constraints.min_obstacle_distance
    for obstacle in situation.environment.obstacles:
        x, y = to_canvas(obstacle.position)
        edge = to_canvas((obstacle.position[0] + obstacle.radius, obstacle.position[1]))
        safe_edge = to_canvas((obstacle.position[0] + obstacle.radius + min_distance, obstacle.position[1]))
        radius = abs(edge[0] - x)
        safe_radius = abs(safe_edge[0] - x)
        draw.ellipse((x - safe_radius, y - safe_radius, x + safe_radius, y + safe_radius), fill="#f8ded9")
        draw.ellipse((x - safe_radius, y - safe_radius, x + safe_radius, y + safe_radius), outline="#e6a79c", width=2)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="#e99184", outline="#c04d31", width=2)


def _draw_baits(
    draw: ImageDraw.ImageDraw,
    situation: SituationAwareness,
    to_canvas: Callable[[Sequence[float]], Tuple[int, int]],
) -> None:
    for bait in situation.environment.baits:
        x, y = to_canvas(bait.position)
        edge = to_canvas((bait.position[0] + bait.radius, bait.position[1]))
        radius = max(8, abs(edge[0] - x))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="#fff1b8", outline="#c69214", width=2)
        points = []
        for index in range(10):
            angle = -math.pi / 2 + index * math.pi / 5
            current_radius = 8 if index % 2 == 0 else 4
            points.append((x + math.cos(angle) * current_radius, y + math.sin(angle) * current_radius))
        draw.polygon(points, fill="#d99a00")


def _draw_path(
    draw: ImageDraw.ImageDraw,
    path: Sequence[Point2],
    to_canvas: Callable[[Sequence[float]], Tuple[int, int]],
) -> None:
    if len(path) < 2:
        return
    projected = [to_canvas(point) for point in path]
    draw.line(projected, fill="#087c89", width=4, joint="curve")
    x, y = projected[-1]
    draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill="#087c89", outline="#ffffff", width=2)


def _draw_start_end(
    draw: ImageDraw.ImageDraw,
    path: Sequence[Point2],
    visible_path: Sequence[Point2],
    to_canvas: Callable[[Sequence[float]], Tuple[int, int]],
) -> None:
    if not path:
        return
    sx, sy = to_canvas(path[0])
    draw.ellipse((sx - 7, sy - 7, sx + 7, sy + 7), fill="#157f4f")
    if len(visible_path) == len(path):
        ex, ey = to_canvas(path[-1])
        draw.ellipse((ex - 7, ey - 7, ex + 7, ey + 7), fill="#7a56a3")


def _draw_metrics(draw: ImageDraw.ImageDraw, decision: DecisionOutput, progress: float) -> None:
    text = (
        f"{decision.scenario} | {decision.total_distance:.1f} m | "
        f"confidence {decision.confidence:.2f} | {progress:.0%}"
    )
    draw.rounded_rectangle((14, 14, 430, 38), radius=6, fill="#ffffff", outline="#d8e0e3")
    draw.text((24, 20), text, fill="#172026")
