"""Test and demo artifact reporting helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from uuv_trajectory_planner.models.decision_output import DecisionOutput


def write_constraint_report(decision: DecisionOutput, output_path: str, elapsed_seconds: float) -> None:
    """Write a compact JSON validation report."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "decision_id": decision.decision_id,
        "scenario": decision.scenario,
        "status": decision.status,
        "confidence": decision.confidence,
        "constraints_satisfied": decision.constraints_satisfied,
        "total_distance": decision.total_distance,
        "estimated_time": decision.estimated_time,
        "decision_time_seconds": elapsed_seconds,
        "feedback": decision.feedback,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
