"""Command-line interface for the UUV trajectory planner MVP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from uuv_trajectory_planner.core.react_engine import ReActEngine


def sample_payload(scenario: str) -> Dict[str, Any]:
    """Return built-in demo payloads."""

    if scenario == "area_coverage":
        return {
            "timestamp": "2026-06-26T10:00:00Z",
            "uuv_state": {"position": [0.0, 0.0, -50.0], "heading": 0.0, "speed": 2.0, "battery": 0.9},
            "mission": {
                "type": "trajectory_planning",
                "scenario": "area_coverage",
                "coverage_area": [[0, 0], [500, 0], [500, 500], [0, 500]],
                "constraints": {
                    "max_speed": 3.0,
                    "sweep_width": 50.0,
                    "coverage_required": 0.95,
                },
            },
            "environment": {"obstacles": [], "boundaries": [[0, 0], [500, 0], [500, 500], [0, 500]]},
        }
    return {
        "timestamp": "2026-06-26T10:00:00Z",
        "uuv_state": {"position": [0.0, 0.0, -50.0], "heading": 45.0, "speed": 2.0, "battery": 0.85},
        "mission": {
            "type": "trajectory_planning",
            "scenario": "general",
            "target_position": [1000.0, 800.0, -50.0],
            "constraints": {
                "max_speed": 3.0,
                "min_obstacle_distance": 50.0,
                "max_energy_cost": 1000.0,
            },
        },
        "environment": {
            "obstacles": [
                {"id": "O001", "type": "static", "position": [300.0, 200.0, -50.0], "radius": 80.0},
                {
                    "id": "O002",
                    "type": "moving",
                    "position": [600.0, 500.0, -50.0],
                    "velocity": [1.0, 0.5, 0.0],
                    "radius": 60.0,
                },
                {"id": "O003", "type": "static", "position": [780.0, 650.0, -50.0], "radius": 45.0},
            ],
            "boundaries": [[0, 0], [1200, 0], [1200, 1000], [0, 1000]],
            "water_current": [0.5, 0.2, 0.0],
        },
    }


def load_payload(input_path: Optional[str], scenario: str) -> Dict[str, Any]:
    """Load JSON payload or return built-in sample."""

    if not input_path:
        return sample_payload(scenario)
    path = Path(input_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("input JSON must be an object")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UUV trajectory planner MVP")
    parser.add_argument("--input", help="Path to situation-awareness JSON")
    parser.add_argument(
        "--scenario",
        choices=["general", "area_coverage"],
        default="general",
        help="Built-in sample scenario when --input is omitted",
    )
    parser.add_argument("--config", help="Optional YAML config path")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = load_payload(args.input, args.scenario)
    engine = ReActEngine(config_path=args.config)
    decision = engine.run(payload)
    print(decision.to_json(pretty=args.pretty))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
