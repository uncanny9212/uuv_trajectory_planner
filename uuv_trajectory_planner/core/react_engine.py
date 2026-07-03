"""ReAct-style UUV trajectory planning engine."""

from __future__ import annotations

from typing import Any, Dict, Optional

from uuv_trajectory_planner.config import load_config
from uuv_trajectory_planner.core.llm_client import LLMClient
from uuv_trajectory_planner.core.memory_manager import MemoryManager
from uuv_trajectory_planner.models.decision_output import DecisionOutput
from uuv_trajectory_planner.models.situation_awareness import SituationAwareness
from uuv_trajectory_planner.planners.coverage_planner import CoveragePlanner
from uuv_trajectory_planner.planners.general_planner import GeneralTrajectoryPlanner


class ReActEngine:
    """Observe, reason, act, and feed back over UUV trajectory missions."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        config = load_config(config_path)
        planner_config = config.get("planner", {})
        llm_config = config.get("llm", {})
        memory_config = config.get("memory", {})
        self.memory = MemoryManager(window_size=int(memory_config.get("window_size", 5)))
        self.llm_client = LLMClient(
            model=llm_config.get("model"),
            timeout_seconds=int(llm_config.get("timeout_seconds", 20)),
        )
        self.general_planner = GeneralTrajectoryPlanner(
            grid_resolution=float(planner_config.get("grid_resolution", 50.0)),
            max_search_nodes=int(planner_config.get("max_search_nodes", 25000)),
            smoothing_iterations=int(planner_config.get("smoothing_iterations", 4)),
            dynamic_prediction_seconds=planner_config.get("dynamic_obstacle_prediction_seconds", [60, 120, 180]),
        )
        self.coverage_planner = CoveragePlanner()

    def observe(self, payload: Dict[str, Any]) -> SituationAwareness:
        """Parse and validate structured situation JSON."""

        return SituationAwareness.from_dict(payload)

    def reason(self, situation: SituationAwareness) -> str:
        """Generate explainable planning rationale."""

        return self.llm_client.reason(situation, self.memory.recent())

    def act(self, situation: SituationAwareness, reasoning: str) -> DecisionOutput:
        """Dispatch to the planner selected by mission scenario."""

        if situation.mission.scenario == "area_coverage":
            return self.coverage_planner.plan(situation, reasoning)
        return self.general_planner.plan(situation, reasoning)

    def feedback(self, decision: DecisionOutput) -> Dict[str, Any]:
        """Simulate execution feedback and store it in sliding memory."""

        result = {
            "decision_id": decision.decision_id,
            "status": decision.status,
            "trajectory_points": len(decision.trajectory),
            "confidence": decision.confidence,
            "constraints_satisfied": decision.constraints_satisfied,
        }
        decision.feedback["react_feedback"] = result
        self.memory.add(result)
        return result

    def run(self, payload: Dict[str, Any]) -> DecisionOutput:
        """Execute one full Observe -> Reason -> Act -> Feedback loop."""

        situation = self.observe(payload)
        reasoning = self.reason(situation)
        decision = self.act(situation, reasoning)
        self.feedback(decision)
        return decision
