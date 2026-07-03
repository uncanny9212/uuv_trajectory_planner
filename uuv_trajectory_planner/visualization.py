"""Trajectory visualization helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from uuv_trajectory_planner.models.decision_output import DecisionOutput
from uuv_trajectory_planner.models.situation_awareness import SituationAwareness, Vector2


def save_trajectory_plot(decision: DecisionOutput, situation: SituationAwareness, output_path: str) -> None:
    """Save a PNG visualization for a decision trajectory."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = output.parent / ".matplotlib"
    cache_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon

    xs = [point.coordinates[0] for point in decision.trajectory]
    ys = [point.coordinates[1] for point in decision.trajectory]
    fig, ax = plt.subplots(figsize=(8, 6), dpi=130)
    ax.plot(xs, ys, color="#1f77b4", linewidth=2, marker="o", markersize=2.5, label="trajectory")

    boundary = situation.mission.coverage_area or situation.environment.boundaries
    if boundary:
        _draw_polygon(ax, boundary)

    for obstacle in situation.environment.obstacles:
        safety_radius = obstacle.radius + situation.mission.constraints.min_obstacle_distance
        ax.add_patch(Circle((obstacle.position[0], obstacle.position[1]), safety_radius, color="#d62728", alpha=0.12))
        ax.add_patch(Circle((obstacle.position[0], obstacle.position[1]), obstacle.radius, color="#d62728", alpha=0.35))
        ax.text(obstacle.position[0], obstacle.position[1], obstacle.id, ha="center", va="center", fontsize=8)

    ax.scatter([xs[0]], [ys[0]], color="#2ca02c", s=60, label="start", zorder=5)
    ax.scatter([xs[-1]], [ys[-1]], color="#9467bd", s=60, label="end", zorder=5)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_title(f"{decision.scenario} | distance={decision.total_distance:.1f}m | confidence={decision.confidence:.2f}")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def _draw_polygon(ax: object, boundary: Sequence[Vector2]) -> None:
    from matplotlib.patches import Polygon

    polygon = Polygon(boundary, closed=True, fill=False, edgecolor="#444444", linewidth=1.5, linestyle="--")
    ax.add_patch(polygon)
