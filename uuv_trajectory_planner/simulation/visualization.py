"""Visualization helpers for UUV rolling simulations."""

from __future__ import annotations

import base64
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Dict

os.environ.setdefault("MPLCONFIGDIR", "/tmp/uuv_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def save_simulation_plot(result: Dict[str, Any], output_path: str) -> str:
    """Save trajectory and bearing-history plots for one simulation result."""

    fig = _build_figure(result)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return str(target)


def simulation_plot_data_url(result: Dict[str, Any]) -> str:
    """Return a PNG data URL for web display."""

    fig = _build_figure(result)
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _build_figure(result: Dict[str, Any]) -> Any:
    scenario = result.get("scenario", {})
    target = result.get("active_target_position") or scenario.get("target_position", [0.0, 0.0, -50.0])
    target_positions = result.get("target_positions", [target])
    constraints = result.get("constraints") or scenario.get("constraints", {})
    approach_range = float(constraints.get("approach_range", 50.0))
    uuv_history = result.get("uuv_history", [])
    bearing_history = result.get("bearing_history", [])
    orbit_history = result.get("orbit_history", [])

    fig, (path_ax, bearing_ax) = plt.subplots(1, 2, figsize=(10.8, 4.4))
    fig.suptitle("UUV rolling simulation", fontsize=13)

    xs = [float(item["position"][0]) for item in uuv_history]
    ys = [float(item["position"][1]) for item in uuv_history]
    if xs and ys:
        path_ax.plot(xs, ys, marker="o", linewidth=2.0, color="#087c89", label="UUV path")
        path_ax.scatter(xs[0], ys[0], color="#157f4f", s=58, label="start", zorder=4)
        path_ax.scatter(xs[-1], ys[-1], color="#715f12", s=58, label="final", zorder=4)
    for index, item in enumerate(target_positions):
        label = "all truth targets" if index == 0 else None
        path_ax.scatter(float(item[0]), float(item[1]), color="#b2bdc2", s=45, marker="o", label=label, zorder=3)
    path_ax.scatter(float(target[0]), float(target[1]), color="#c04d31", s=90, marker="x", label="active target", zorder=5)
    if orbit_history:
        orbit_xs = [float(item["position"][0]) for item in orbit_history]
        orbit_ys = [float(item["position"][1]) for item in orbit_history]
        path_ax.plot(orbit_xs, orbit_ys, linewidth=1.6, color="#6a5acd", alpha=0.85, label="5-turn orbit")
    circle = plt.Circle((float(target[0]), float(target[1])), approach_range, fill=False, color="#c04d31", linestyle="--")
    path_ax.add_patch(circle)
    path_ax.set_aspect("equal", adjustable="datalim")
    path_ax.set_xlabel("x / m")
    path_ax.set_ylabel("y / m")
    path_ax.grid(True, alpha=0.25)
    path_ax.legend(loc="best", fontsize=8)

    bearing_values = [float(item["angle"]) for item in bearing_history]
    bearing_ax.plot(range(len(bearing_values)), bearing_values, marker="o", color="#4f6f91", linewidth=2.0)
    bearing_ax.set_xlabel("observation")
    bearing_ax.set_ylabel("bearing / deg")
    bearing_ax.set_ylim(0, 360)
    bearing_ax.grid(True, alpha=0.25)

    status = result.get("status", "--")
    final_distance = float(result.get("final_distance", 0.0))
    efficiency = float(result.get("path_efficiency", 0.0))
    fig.text(
        0.5,
        0.01,
        f"status={status}  final distance={final_distance:.1f}m  efficiency={efficiency:.2f}",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    return fig
