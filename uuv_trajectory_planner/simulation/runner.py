"""Simulation runner that drives ``RollingPlanner`` with ground-truth bearings."""

from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Optional, Sequence

from uuv_trajectory_planner.core.rolling_planner import RollingPlanner
from uuv_trajectory_planner.simulation.scenarios import (
    DEFAULT_SIMULATION_CONSTRAINTS,
    SimulationScenario,
    default_scenarios,
)
from uuv_trajectory_planner.simulation.simulator import UUVSimulator, Vector3, bearing_from_to, vector3

_NUMBER_RE = r"[-+]?\d+(?:\.\d+)?"
_POSITION_RE = re.compile(
    rf"[\(\[（]\s*({_NUMBER_RE})\s*[,，]\s*({_NUMBER_RE})"
    rf"(?:\s*[,，]\s*({_NUMBER_RE}))?\s*[\)\]）]"
)
WORLD_MIN = 0.0
WORLD_MAX = 2000.0
WORLD_RESOLUTION = 1.0
DEFAULT_INTERACTIVE_START: Vector3 = (0.0, 0.0, -50.0)
DEFAULT_FALSE_BEARING_THRESHOLD_DEG = 25.0
DEEP_TARGET_THRESHOLD_M = 10.0
DEEP_TARGET_EXTRA_ORBIT_TURNS = 2


class SimulationRunner:
    """Run single or batch passive-bearing approach simulations."""

    def __init__(self, planner: Optional[RollingPlanner] = None) -> None:
        self.planner = planner or RollingPlanner()

    def run(self, scenario: SimulationScenario) -> Dict[str, Any]:
        """Run one scenario and return a serializable result report."""

        constraints = scenario.merged_constraints()
        simulator = UUVSimulator.create(
            target_position=scenario.target_position,
            uuv_position=scenario.start_position,
            heading=scenario.initial_heading,
            speed=scenario.speed,
            battery=scenario.battery,
        )
        initial_distance = simulator.distance_to_target()
        approach_range = float(constraints["approach_range"])
        max_iterations = int(constraints["max_iterations"])

        bearing_history: List[Dict[str, Any]] = [
            self._bearing_observation(simulator, 0),
        ]
        uuv_history: List[Dict[str, Any]] = [simulator.history_item(0)]
        decisions: List[Dict[str, Any]] = []
        status = "failed"
        failure_reason = "达到最大迭代次数"

        if initial_distance <= approach_range:
            status = "success"
            failure_reason = ""

        for iteration in range(max_iterations):
            if status == "success":
                break

            decision = self.planner.plan(
                {
                    "bearing_history": list(bearing_history),
                    "uuv_state": simulator.uuv_state(),
                    "estimated_range": self._range_hint(simulator.distance_to_target()),
                    "constraints": constraints,
                    "iteration": iteration,
                    "mission_context": "仿真抵近侦察",
                }
            ).to_dict()
            decisions.append(
                {
                    "iteration": iteration + 1,
                    "distance_before": round(simulator.distance_to_target(), 3),
                    **decision,
                }
            )

            if decision["decision"] == "give_up":
                failure_reason = "滚动规划Agent决定放弃"
                break

            if decision["decision"] in ("orbit", "approach_complete"):
                if simulator.distance_to_target() <= approach_range:
                    status = "success"
                    failure_reason = ""
                else:
                    failure_reason = "尚未进入抵近距离但Agent提前切换模式"
                break

            if decision["decision"] in ("advance", "adjust_heading"):
                simulator.move(float(decision["next_heading"]), float(decision["advance_distance"]))

            uuv_history.append(simulator.history_item(iteration + 1))
            bearing_history.append(self._bearing_observation(simulator, iteration + 1))

            if simulator.distance_to_target() <= approach_range:
                status = "success"
                failure_reason = ""
                break

        final_distance = simulator.distance_to_target()
        total_distance = simulator.total_distance
        path_efficiency = self._path_efficiency(initial_distance, final_distance, total_distance)
        result = {
            "scenario": scenario.to_dict(),
            "status": status,
            "iterations": len(decisions),
            "total_distance": round(total_distance, 3),
            "final_distance": round(final_distance, 3),
            "path_efficiency": round(path_efficiency, 3),
            "decisions": decisions,
            "uuv_history": uuv_history,
            "bearing_history": bearing_history,
            "failure_reason": failure_reason,
            "summary": self._summary(
                scenario=scenario,
                status=status,
                iterations=len(decisions),
                total_distance=total_distance,
                final_distance=final_distance,
                path_efficiency=path_efficiency,
                failure_reason=failure_reason,
            ),
        }
        return result

    def run_batch(self, scenarios: Optional[Sequence[SimulationScenario]] = None) -> Dict[str, Any]:
        """Run all provided scenarios and return aggregate statistics."""

        selected = list(scenarios or default_scenarios())
        results = [self.run(scenario) for scenario in selected]
        successes = [result for result in results if result["status"] == "success"]
        average_iterations = (
            sum(result["iterations"] for result in results) / len(results)
            if results
            else 0.0
        )
        average_efficiency = (
            sum(result["path_efficiency"] for result in results) / len(results)
            if results
            else 0.0
        )
        return {
            "scenario_count": len(results),
            "success_count": len(successes),
            "success_rate": round(len(successes) / len(results), 3) if results else 0.0,
            "average_iterations": round(average_iterations, 3),
            "average_path_efficiency": round(average_efficiency, 3),
            "results": results,
            "summary": (
                f"批量仿真完成：{len(successes)}/{len(results)} 个场景成功，"
                f"平均迭代 {average_iterations:.1f} 轮，平均路径效率 {average_efficiency:.2f}。"
            ),
        }

    def run_interactive(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run a closed-loop simulation over one or more truth targets."""

        target_positions = parse_target_positions(
            payload.get("target_positions", payload.get("target_positions_text", []))
        )
        start_position = self._interactive_start_position(payload)
        constraints = self._interactive_constraints(payload)
        initial_bearings = self._initial_bearings(payload)
        target_route, bearing_assessments = self._target_route(
            target_positions,
            start_position,
            initial_bearings,
            float(constraints["false_bearing_threshold"]),
        )
        bias_deg = float(payload.get("bearing_bias_deg", payload.get("bearing_bias", 0.0)))
        noise_deg = max(0.0, float(payload.get("bearing_noise_deg", payload.get("bearing_noise", 0.0))))

        uuv_history: List[Dict[str, Any]] = []
        bearing_history: List[Dict[str, Any]] = []
        decisions: List[Dict[str, Any]] = []
        orbit_history: List[Dict[str, Any]] = []
        trajectory_segments: List[Dict[str, Any]] = []
        target_runs: List[Dict[str, Any]] = []
        current_position = start_position
        global_iteration = 0
        approach_distance = 0.0
        orbit_distance = 0.0
        completed_target_count = 0
        final_distance = 0.0
        status = "success"
        failure_reason = ""

        for route_position, target_index in enumerate(target_route, start=1):
            target = target_positions[target_index]
            bearing_assessment = self._assessment_for_target(bearing_assessments, target_index)
            if route_position == 1 and initial_bearings:
                initial_observation = initial_bearings[0]
                observation_source = "user"
            else:
                initial_observation = (bearing_from_to(current_position, target) + bias_deg) % 360.0
                observation_source = "simulated"

            leg = self._run_target_leg(
                target_position=target,
                start_position=current_position,
                target_index=target_index,
                target_sequence=route_position,
                global_start_iteration=global_iteration,
                initial_observation=initial_observation,
                observation_source=observation_source,
                constraints=constraints,
                speed=float(payload.get("speed", 2.0)),
                battery=float(payload.get("battery", 0.9)),
                bias_deg=bias_deg,
                noise_deg=noise_deg,
                bearing_assessment=bearing_assessment,
            )

            uuv_history.extend(leg["uuv_history"])
            bearing_history.extend(leg["bearing_history"])
            decisions.extend(leg["decisions"])
            orbit_history.extend(leg["orbit_history"])
            trajectory_segments.extend(leg["trajectory_segments"])
            target_runs.append(leg["target_run"])
            approach_distance += float(leg["approach_distance"])
            orbit_distance += float(leg["orbit_distance"])
            final_distance = float(leg["final_distance"])
            global_iteration = int(leg["end_iteration"])
            current_position = vector3(leg["end_position"])

            if leg["status"] == "success":
                completed_target_count += 1
                continue
            status = "failed"
            failure_reason = str(leg["failure_reason"])
            break

        total_distance = approach_distance + orbit_distance
        path_efficiency = 1.0 if status == "success" else completed_target_count / max(1, len(target_route))
        active_target_index = target_route[0] if target_route else 0
        active_target = target_positions[active_target_index]
        first_bearing = initial_bearings[0] if initial_bearings else bearing_from_to(start_position, active_target)
        result = {
            "mode": "interactive",
            "status": status,
            "target_positions": [list(position) for position in target_positions],
            "target_route": target_route,
            "target_runs": target_runs,
            "completed_target_count": completed_target_count,
            "bearing_assessments": bearing_assessments,
            "false_information_detected": any(
                item["status"] != "trusted" for item in bearing_assessments
            ),
            "active_target_index": active_target_index,
            "active_target_position": list(active_target),
            "coordinate_system": {
                "x_range": [WORLD_MIN, WORLD_MAX],
                "y_range": [WORLD_MIN, WORLD_MAX],
                "resolution": WORLD_RESOLUTION,
                "start_position": list(start_position),
                "bearing_reference": "默认基于当前起点位置，0°为正北，90°为正东",
            },
            "bearing_measurement": {
                "initial_bearing": round(float(first_bearing), 3),
                "initial_bearings": [round(float(value), 3) for value in initial_bearings],
                "bias_deg": round(bias_deg, 3),
                "noise_deg": round(noise_deg, 3),
            },
            "constraints": constraints,
            "iterations": len(decisions),
            "discovered_iteration": target_runs[-1]["discovered_iteration"] if target_runs else None,
            "approach_distance": round(approach_distance, 3),
            "orbit_distance": round(orbit_distance, 3),
            "total_distance": round(total_distance, 3),
            "final_distance": round(final_distance, 3),
            "path_efficiency": round(path_efficiency, 3),
            "decisions": decisions,
            "uuv_history": uuv_history,
            "bearing_history": bearing_history,
            "orbit_history": orbit_history,
            "trajectory_segments": trajectory_segments,
            "orbit_turns_completed": sum(
                int(run.get("orbit_turns_completed", 0)) for run in target_runs
            ),
            "failure_reason": failure_reason,
            "summary": self._interactive_summary(
                status=status,
                target_count=len(target_positions),
                completed_target_count=completed_target_count,
                active_target_index=active_target_index,
                initial_bearing=float(first_bearing),
                iterations=len(decisions),
                approach_distance=approach_distance,
                orbit_distance=orbit_distance,
                final_distance=final_distance,
                total_orbit_turns=sum(
                    int(run.get("orbit_turns_completed", 0)) for run in target_runs
                ),
                failure_reason=failure_reason,
            ),
        }
        return result

    def _run_target_leg(
        self,
        *,
        target_position: Vector3,
        start_position: Vector3,
        target_index: int,
        target_sequence: int,
        global_start_iteration: int,
        initial_observation: float,
        observation_source: str,
        constraints: Dict[str, Any],
        speed: float,
        battery: float,
        bias_deg: float,
        noise_deg: float,
        bearing_assessment: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        approach_range = float(constraints["approach_range"])
        max_iterations = int(constraints["max_iterations"])
        orbit_turns = self._orbit_turns_for_target(target_position, constraints)
        leg_constraints = dict(constraints)
        leg_constraints["orbit_turns"] = orbit_turns
        is_deep_target = self._target_depth(target_position) > DEEP_TARGET_THRESHOLD_M
        simulator = UUVSimulator.create(
            target_position=target_position,
            uuv_position=start_position,
            heading=float(initial_observation),
            speed=speed,
            battery=battery,
        )
        initial_distance = simulator.distance_to_target()
        bearing_history: List[Dict[str, Any]] = [
            self._bearing_observation(
                simulator,
                global_start_iteration,
                observed_angle=float(initial_observation),
                source=observation_source,
                target_index=target_index,
                target_sequence=target_sequence,
            )
        ]
        uuv_history: List[Dict[str, Any]] = [
            self._history_item(simulator, global_start_iteration, target_index, target_sequence)
        ]
        decisions: List[Dict[str, Any]] = []
        status = "failed"
        failure_reason = "达到最大迭代次数"
        discovered_iteration: Optional[int] = None
        false_information_reported = False

        if self._within_approach_range(initial_distance, approach_range):
            status = "success"
            failure_reason = ""
            discovered_iteration = global_start_iteration

        for local_iteration in range(max_iterations):
            if status == "success":
                break

            distance_before = simulator.distance_to_target()
            if self._within_approach_range(distance_before, approach_range):
                status = "success"
                failure_reason = ""
                discovered_iteration = global_start_iteration + local_iteration
                break

            latest_observation = bearing_history[-1]
            decision = self.planner.plan(
                {
                    "bearing_history": list(bearing_history),
                    "uuv_state": simulator.uuv_state(),
                    "estimated_range": None,
                    "constraints": leg_constraints,
                    "iteration": local_iteration,
                    "mission_context": f"多目标闭环抵近侦察：第{target_sequence}个目标",
                }
            ).to_dict()

            executed_action = decision["decision"]
            executed_heading = float(decision["next_heading"])
            executed_distance = float(decision["advance_distance"])
            feedback_note = "按滚动规划输出执行。"
            decision_iteration = global_start_iteration + local_iteration + 1
            information_feedback = self._false_information_feedback_from_observations(
                bearing_history,
                float(constraints["false_bearing_threshold"]),
            )
            if information_feedback and not false_information_reported:
                decision = dict(decision)
                decision["warnings"] = list(decision.get("warnings", [])) + ["初始方位疑似虚假信息"]
                executed_action = "bearing_recheck_advance"
                executed_heading = float(latest_observation["angle"])
                executed_distance = self._safe_advance_distance(distance_before, approach_range, leg_constraints)
                feedback_note = information_feedback
                false_information_reported = True

            if decision["decision"] == "give_up":
                if self._can_keep_approaching_from_stable_bearing(decision, bearing_history, distance_before):
                    executed_action = "stable_bearing_advance"
                    executed_heading = float(latest_observation["angle"])
                    executed_distance = self._safe_advance_distance(distance_before, approach_range, leg_constraints)
                    feedback_note = "方位长期稳定但仍未发现目标，仿真按最新方位继续抵近。"
                else:
                    failure_reason = f"第{target_sequence}个目标：滚动规划Agent决定放弃"
                    decisions.append(
                        self._decision_record(
                            iteration=decision_iteration,
                            target_index=target_index,
                            target_sequence=target_sequence,
                            distance_before=distance_before,
                            distance_after=distance_before,
                            observation=latest_observation,
                            decision=decision,
                            executed_action=executed_action,
                            executed_heading=executed_heading,
                            executed_distance=0.0,
                            feedback_note=failure_reason,
                            target_discovered=False,
                        )
                    )
                    break

            if decision["decision"] in ("orbit", "approach_complete"):
                target_discovered = self._within_approach_range(distance_before, approach_range)
                if target_discovered:
                    status = "success"
                    failure_reason = ""
                    discovered_iteration = decision_iteration
                else:
                    failure_reason = f"第{target_sequence}个目标：尚未进入发现半径但Agent提前切换模式"
                decisions.append(
                    self._decision_record(
                        iteration=decision_iteration,
                        target_index=target_index,
                        target_sequence=target_sequence,
                        distance_before=distance_before,
                        distance_after=distance_before,
                        observation=latest_observation,
                        decision=decision,
                        executed_action=decision["decision"],
                        executed_heading=executed_heading,
                        executed_distance=0.0,
                        feedback_note=failure_reason or "进入目标绕航阶段。",
                        target_discovered=target_discovered,
                        discovered_position=target_position if target_discovered else None,
                    )
                )
                break

            if decision["decision"] == "wait" and len(bearing_history) > 1 and distance_before > approach_range:
                executed_action = "stable_bearing_advance"
                executed_heading = float(latest_observation["angle"])
                executed_distance = self._safe_advance_distance(distance_before, approach_range, leg_constraints)
                feedback_note = "已有连续方位观测，按最新方位继续前进并再次探测。"

            if executed_action in ("advance", "adjust_heading"):
                executed_action = "bearing_only_advance"
                executed_heading = float(latest_observation["angle"])
                executed_distance = self._safe_advance_distance(distance_before, approach_range, leg_constraints)
                feedback_note = "距离未知，按最新探测方位前进固定航段后复测。"

            if executed_action in (
                "advance",
                "adjust_heading",
                "stable_bearing_advance",
                "bearing_recheck_advance",
                "bearing_only_advance",
            ):
                simulator.move(executed_heading, executed_distance)

            distance_after = simulator.distance_to_target()
            target_discovered = self._within_approach_range(distance_after, approach_range)
            decisions.append(
                self._decision_record(
                    iteration=decision_iteration,
                    target_index=target_index,
                    target_sequence=target_sequence,
                    distance_before=distance_before,
                    distance_after=distance_after,
                    observation=latest_observation,
                    decision=decision,
                    executed_action=executed_action,
                    executed_heading=executed_heading,
                    executed_distance=executed_distance,
                    feedback_note=feedback_note,
                    target_discovered=target_discovered,
                    discovered_position=target_position if target_discovered else None,
                )
            )

            uuv_history.append(self._history_item(simulator, decision_iteration, target_index, target_sequence))

            if target_discovered:
                status = "success"
                failure_reason = ""
                discovered_iteration = decision_iteration
                break

            bearing_history.append(
                self._bearing_observation(
                    simulator,
                    decision_iteration,
                    observed_angle=self._measured_bearing(simulator, decision_iteration, bias_deg, noise_deg),
                    source="simulated",
                    target_index=target_index,
                    target_sequence=target_sequence,
                )
            )

        orbit_history: List[Dict[str, Any]] = []
        orbit_distance = 0.0
        if status == "success" and orbit_turns > 0:
            orbit_history, orbit_distance = self._orbit_history(
                simulator,
                leg_constraints,
                target_index=target_index,
                target_sequence=target_sequence,
                step_offset=global_start_iteration + len(decisions),
            )

        end_position = vector3(orbit_history[-1]["position"]) if orbit_history else simulator.uuv_position
        final_distance = math.hypot(target_position[0] - end_position[0], target_position[1] - end_position[1])
        approach_points = [item["position"] for item in uuv_history]
        orbit_points = [item["position"] for item in orbit_history]
        trajectory_segments = [
            {
                "kind": "approach",
                "target_index": target_index,
                "target_sequence": target_sequence,
                "distance": round(simulator.total_distance, 3),
                "points": approach_points,
            }
        ]
        if orbit_points:
            trajectory_segments.append(
                {
                    "kind": "orbit",
                    "target_index": target_index,
                    "target_sequence": target_sequence,
                    "distance": round(orbit_distance, 3),
                    "points": orbit_points,
                }
            )
        return {
            "status": status,
            "failure_reason": failure_reason,
            "final_distance": round(final_distance, 3),
            "approach_distance": round(simulator.total_distance, 3),
            "orbit_distance": round(orbit_distance, 3),
            "end_position": [round(value, 3) for value in end_position],
            "end_iteration": global_start_iteration + len(decisions),
            "uuv_history": uuv_history,
            "bearing_history": bearing_history,
            "decisions": decisions,
            "orbit_history": orbit_history,
            "trajectory_segments": trajectory_segments,
            "target_run": {
                "target_index": target_index,
                "target_sequence": target_sequence,
                "target_position": list(target_position),
                "target_depth": round(self._target_depth(target_position), 3),
                "is_deep_target": is_deep_target,
                "status": status,
                "iterations": len(decisions),
                "discovered_iteration": discovered_iteration,
                "approach_distance": round(simulator.total_distance, 3),
                "orbit_distance": round(orbit_distance, 3),
                "final_distance": round(final_distance, 3),
                "orbit_turns_completed": orbit_turns if status == "success" else 0,
                "failure_reason": failure_reason,
            },
        }

    def _bearing_observation(
        self,
        simulator: UUVSimulator,
        iteration: int,
        observed_angle: Optional[float] = None,
        source: str = "truth",
        target_index: Optional[int] = None,
        target_sequence: Optional[int] = None,
    ) -> Dict[str, Any]:
        true_angle = simulator.bearing_to_target()
        angle = true_angle if observed_angle is None else float(observed_angle) % 360.0
        observation = {
            "angle": round(angle, 3),
            "timestamp": self._timestamp(iteration),
            "true_angle": round(true_angle, 3),
            "bearing_error": round(_signed_angle_delta(true_angle, angle), 3),
            "source": source,
        }
        if target_index is not None:
            observation["target_index"] = target_index
        if target_sequence is not None:
            observation["target_sequence"] = target_sequence
        return observation

    def _history_item(
        self,
        simulator: UUVSimulator,
        iteration: int,
        target_index: int,
        target_sequence: int,
    ) -> Dict[str, Any]:
        item = simulator.history_item(iteration)
        item["target_index"] = target_index
        item["target_sequence"] = target_sequence
        return item

    def _timestamp(self, iteration: int) -> str:
        seconds = iteration * 30
        minutes, second = divmod(seconds, 60)
        hour, minute = divmod(minutes, 60)
        return f"{hour:02d}:{minute:02d}:{second:02d}"

    def _path_efficiency(self, initial_distance: float, final_distance: float, total_distance: float) -> float:
        if total_distance <= 1e-9:
            return 1.0 if initial_distance <= final_distance else 0.0
        ideal_progress = max(0.0, initial_distance - final_distance)
        return max(0.0, min(1.0, ideal_progress / total_distance))

    def _range_hint(self, true_distance: float) -> float:
        # RollingPlanner discounts stable bearings by 0.7; compensate so a
        # straight, stable approach can still reach the simulator threshold.
        return max(0.0, true_distance / 0.7)

    def _interactive_constraints(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        constraints = dict(DEFAULT_SIMULATION_CONSTRAINTS)
        constraints.update(
            {
                "approach_range": float(payload.get("approach_range", constraints["approach_range"])),
                "orbit_turns": int(float(payload.get("orbit_turns", 5))),
                "orbit_radius": float(payload.get("orbit_radius", constraints["orbit_radius"])),
                "max_iterations": int(float(payload.get("max_iterations", constraints["max_iterations"]))),
                "default_step": float(payload.get("default_step", constraints["default_step"])),
                "stable_angle_threshold": float(
                    payload.get("stable_angle_threshold", constraints["stable_angle_threshold"])
                ),
                "max_heading_correction": float(payload.get("max_heading_correction", 2.0)),
                "false_bearing_threshold": float(
                    payload.get("false_bearing_threshold", DEFAULT_FALSE_BEARING_THRESHOLD_DEG)
                ),
                "deep_target_threshold": DEEP_TARGET_THRESHOLD_M,
                "deep_target_extra_orbit_turns": DEEP_TARGET_EXTRA_ORBIT_TURNS,
            }
        )
        constraints["approach_range"] = max(0.0, constraints["approach_range"])
        constraints["orbit_turns"] = max(0, constraints["orbit_turns"])
        constraints["orbit_radius"] = max(0.0, constraints["orbit_radius"])
        constraints["max_iterations"] = max(1, constraints["max_iterations"])
        constraints["default_step"] = max(1.0, constraints["default_step"])
        constraints["false_bearing_threshold"] = max(0.0, constraints["false_bearing_threshold"])
        return constraints

    def _interactive_start_position(self, payload: Dict[str, Any]) -> Vector3:
        if payload.get("allow_start_override"):
            start = vector3(payload.get("start_position", payload.get("uuv_position", DEFAULT_INTERACTIVE_START)))
            return _validate_world_position(start, "起点")
        return DEFAULT_INTERACTIVE_START

    def _initial_bearing(self, payload: Dict[str, Any]) -> Optional[float]:
        value = payload.get("initial_bearing")
        if value is not None and str(value).strip() != "":
            return float(value) % 360.0
        text = str(payload.get("bearing_text", "")).strip()
        if not text:
            return None
        from uuv_trajectory_planner.core.detection_parser import DetectionParser

        bearing = DetectionParser()._extract_bearing(text)  # pylint: disable=protected-access
        return None if bearing is None else float(bearing) % 360.0

    def _initial_bearings(self, payload: Dict[str, Any]) -> List[float]:
        explicit = payload.get("initial_bearings")
        bearings: List[float] = []
        if explicit is not None and str(explicit).strip() != "":
            if isinstance(explicit, str):
                bearings.extend(float(value) % 360.0 for value in re.findall(_NUMBER_RE, explicit))
            elif isinstance(explicit, Sequence):
                bearings.extend(float(value) % 360.0 for value in explicit)
            else:
                bearings.append(float(explicit) % 360.0)

        text = str(payload.get("bearing_text", "")).strip()
        if text:
            from uuv_trajectory_planner.core.detection_parser import DetectionParser

            parser = DetectionParser()
            patterns = [
                rf"[东西南北]\s*偏\s*[东西南北]\s*{_NUMBER_RE}\s*(?:度|°)?",
                rf"(?:方位角|方位|方向|bearing|azimuth|heading)\s*(?:为|=|:|：)?\s*{_NUMBER_RE}\s*(?:度|°|degrees?|deg)?",
                rf"(?<![\d.]){_NUMBER_RE}\s*(?:度|°|degrees?|deg)\s*(?:方向|方位|bearing|azimuth|heading)?",
                r"正北|北方|正东|东方|正南|南方|正西|西方|东北|东南|西南|西北",
            ]
            matches: List[tuple[int, int, str]] = []
            for pattern in patterns:
                for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                    if any(start <= match.start() < end or match.start() <= start < match.end() for start, end, _ in matches):
                        continue
                    matches.append((match.start(), match.end(), match.group(0)))
            for _, _, candidate in sorted(matches, key=lambda item: item[0]):
                bearing = parser._extract_bearing(candidate)  # pylint: disable=protected-access
                if bearing is not None:
                    bearings.append(float(bearing) % 360.0)

        if not bearings:
            single = self._initial_bearing(payload)
            if single is not None:
                bearings.append(single)

        deduped: List[float] = []
        for bearing in bearings:
            if not any(abs(_signed_angle_delta(existing, bearing)) < 1e-6 for existing in deduped):
                deduped.append(bearing)
        return deduped

    def _target_route(
        self,
        targets: Sequence[Vector3],
        start_position: Vector3,
        initial_bearings: Sequence[float],
        false_bearing_threshold: float,
    ) -> tuple[List[int], List[Dict[str, Any]]]:
        remaining = set(range(len(targets)))
        route: List[int] = []
        assessments: List[Dict[str, Any]] = []

        for bearing_index, bearing in enumerate(initial_bearings, start=1):
            if not remaining:
                assessments.append(
                    {
                        "bearing_index": bearing_index,
                        "bearing": round(float(bearing) % 360.0, 3),
                        "matched_target_index": None,
                        "matched_true_bearing": None,
                        "delta_deg": None,
                        "threshold_deg": round(false_bearing_threshold, 3),
                        "status": "unmatched",
                        "message": "该方位没有可匹配的真实饵物，判定为疑似虚假信息。",
                    }
                )
                break
            selected = min(
                remaining,
                key=lambda index: abs(_signed_angle_delta(bearing, bearing_from_to(start_position, targets[index]))),
            )
            matched_true_bearing = bearing_from_to(start_position, targets[selected])
            delta = abs(_signed_angle_delta(bearing, matched_true_bearing))
            status = "suspect_false" if delta > false_bearing_threshold else "trusted"
            message = (
                f"输入方位与第{selected + 1}个真实饵物方位偏差{delta:.1f}°，超过阈值"
                f"{false_bearing_threshold:.1f}°，判定为疑似虚假信息。"
                if status == "suspect_false"
                else f"输入方位与第{selected + 1}个真实饵物方位偏差{delta:.1f}°，在可信阈值内。"
            )
            assessments.append(
                {
                    "bearing_index": bearing_index,
                    "bearing": round(float(bearing) % 360.0, 3),
                    "matched_target_index": selected,
                    "matched_true_bearing": round(matched_true_bearing, 3),
                    "delta_deg": round(delta, 3),
                    "threshold_deg": round(false_bearing_threshold, 3),
                    "status": status,
                    "message": message,
                }
            )
            route.append(selected)
            remaining.remove(selected)

        if initial_bearings:
            return (route, assessments)

        current = targets[route[-1]] if route else start_position
        while remaining:
            selected = min(
                remaining,
                key=lambda index: math.hypot(targets[index][0] - current[0], targets[index][1] - current[1]),
            )
            route.append(selected)
            remaining.remove(selected)
            current = targets[selected]
        return (route, assessments)

    def _assessment_for_target(
        self,
        assessments: Sequence[Dict[str, Any]],
        target_index: int,
    ) -> Optional[Dict[str, Any]]:
        for assessment in assessments:
            if assessment.get("matched_target_index") == target_index:
                return assessment
        return None

    def _false_information_feedback(self, assessment: Optional[Dict[str, Any]]) -> str:
        if not assessment or assessment.get("status") != "suspect_false":
            return ""
        return (
            f"驾驶员反馈：收到的{assessment['bearing']:.1f}°方位与后续探测趋势不一致，"
            "判定疑似虚假信息，等待复测后修正航向。"
        )

    def _false_information_feedback_from_observations(
        self,
        bearing_history: Sequence[Dict[str, Any]],
        threshold_deg: float,
    ) -> str:
        if len(bearing_history) < 2:
            return ""
        first = bearing_history[0]
        latest = bearing_history[-1]
        if first.get("source") != "user":
            return ""
        delta = abs(_signed_angle_delta(float(first["angle"]), float(latest["angle"])))
        if delta <= threshold_deg:
            return ""
        return (
            f"初始方位{float(first['angle']):.1f}°与复测方位{float(latest['angle']):.1f}°"
            f"差异{delta:.1f}°，疑似虚假信息；当前仅采信复测方位，距离和目标坐标仍未知。"
        )

    def _select_active_target(
        self,
        targets: Sequence[Vector3],
        start_position: Vector3,
        initial_bearing: Optional[float],
    ) -> int:
        if initial_bearing is None:
            distances = [
                math.hypot(target[0] - start_position[0], target[1] - start_position[1])
                for target in targets
            ]
            return min(range(len(targets)), key=lambda index: distances[index])
        return min(
            range(len(targets)),
            key=lambda index: abs(_signed_angle_delta(initial_bearing, bearing_from_to(start_position, targets[index]))),
        )

    def _measured_bearing(
        self,
        simulator: UUVSimulator,
        iteration: int,
        bias_deg: float,
        noise_deg: float,
    ) -> float:
        deterministic_noise = noise_deg * math.sin(iteration * 1.61803398875)
        return (simulator.bearing_to_target() + bias_deg + deterministic_noise) % 360.0

    def _safe_advance_distance(
        self,
        distance_before: float,
        approach_range: float,
        constraints: Dict[str, Any],
    ) -> float:
        remaining = max(0.0, distance_before - approach_range)
        if remaining <= 1e-6:
            return 0.0
        return min(float(constraints["default_step"]), remaining)

    def _target_depth(self, target_position: Sequence[float]) -> float:
        return abs(float(target_position[2])) if len(target_position) > 2 else 50.0

    def _orbit_turns_for_target(
        self,
        target_position: Sequence[float],
        constraints: Dict[str, Any],
    ) -> int:
        base_turns = max(0, int(constraints["orbit_turns"]))
        if self._target_depth(target_position) > DEEP_TARGET_THRESHOLD_M:
            return base_turns + DEEP_TARGET_EXTRA_ORBIT_TURNS
        return base_turns

    def _within_approach_range(self, distance: float, approach_range: float) -> bool:
        return distance <= approach_range + 1e-6

    def _can_keep_approaching_from_stable_bearing(
        self,
        decision: Dict[str, Any],
        bearing_history: Sequence[Dict[str, Any]],
        distance_before: float,
    ) -> bool:
        warnings = "；".join(str(item) for item in decision.get("warnings", []))
        return len(bearing_history) > 1 and distance_before > 0.0 and "方位长期稳定" in warnings

    def _decision_record(
        self,
        *,
        iteration: int,
        target_index: Optional[int] = None,
        target_sequence: Optional[int] = None,
        distance_before: float,
        distance_after: float,
        observation: Dict[str, Any],
        decision: Dict[str, Any],
        executed_action: str,
        executed_heading: float,
        executed_distance: float,
        feedback_note: str,
        target_discovered: bool = False,
        discovered_position: Optional[Vector3] = None,
    ) -> Dict[str, Any]:
        record = {
            "iteration": iteration,
            "distance_before": round(distance_before, 3),
            "distance_after": round(distance_after, 3),
            "observation": observation,
            **decision,
            "executed_action": executed_action,
            "executed_heading": round(executed_heading, 3),
            "executed_distance": round(executed_distance, 3),
            "feedback": feedback_note,
            "target_discovered": target_discovered,
        }
        if target_discovered and discovered_position is not None:
            record["discovered_position"] = [round(value, 3) for value in discovered_position]
        if target_index is not None:
            record["target_index"] = target_index
        if target_sequence is not None:
            record["target_sequence"] = target_sequence
        return record

    def _orbit_history(
        self,
        simulator: UUVSimulator,
        constraints: Dict[str, Any],
        target_index: Optional[int] = None,
        target_sequence: Optional[int] = None,
        step_offset: int = 0,
    ) -> tuple[List[Dict[str, Any]], float]:
        turns = max(0, int(constraints["orbit_turns"]))
        radius = max(0.0, float(constraints["orbit_radius"]))
        if turns == 0 or radius <= 0.0:
            return ([], 0.0)

        points_per_turn = 36
        total_segments = turns * points_per_turn
        current = simulator.uuv_position
        target = simulator.target_position
        start_angle = math.atan2(current[1] - target[1], current[0] - target[0])
        points: List[Dict[str, Any]] = [
            {
                "step": step_offset,
                "local_step": 0,
                "turn": 0.0,
                "position": [round(current[0], 3), round(current[1], 3), round(current[2], 3)],
            }
        ]
        if target_index is not None:
            points[0]["target_index"] = target_index
        if target_sequence is not None:
            points[0]["target_sequence"] = target_sequence
        for index in range(total_segments + 1):
            angle = start_angle + (2.0 * math.pi * index / points_per_turn)
            position = [
                round(target[0] + math.cos(angle) * radius, 3),
                round(target[1] + math.sin(angle) * radius, 3),
                round(target[2], 3),
            ]
            item: Dict[str, Any] = {
                "step": step_offset + index + 1,
                "local_step": index + 1,
                "turn": round(index / points_per_turn, 3),
                "position": position,
            }
            if target_index is not None:
                item["target_index"] = target_index
            if target_sequence is not None:
                item["target_sequence"] = target_sequence
            points.append(item)
        entry_position = points[1]["position"] if len(points) > 1 else points[0]["position"]
        entry_distance = math.hypot(current[0] - entry_position[0], current[1] - entry_position[1])
        orbit_distance = entry_distance + (2.0 * math.pi * radius * turns)
        return (points, orbit_distance)

    def _summary(
        self,
        *,
        scenario: SimulationScenario,
        status: str,
        iterations: int,
        total_distance: float,
        final_distance: float,
        path_efficiency: float,
        failure_reason: str,
    ) -> str:
        status_text = "成功抵近目标" if status == "success" else f"失败：{failure_reason}"
        return (
            f"场景“{scenario.name}”{status_text}；共迭代{iterations}轮，"
            f"实际航程{total_distance:.1f}m，最终距离{final_distance:.1f}m，"
            f"路径效率{path_efficiency:.2f}。预期行为：{scenario.expected_behavior}。"
        )

    def _interactive_summary(
        self,
        *,
        status: str,
        target_count: int,
        completed_target_count: int,
        active_target_index: int,
        initial_bearing: float,
        iterations: int,
        approach_distance: float,
        orbit_distance: float,
        final_distance: float,
        total_orbit_turns: int,
        failure_reason: str,
    ) -> str:
        if status == "success":
            target_text = (
                f"已连续完成{completed_target_count}/{target_count}个目标"
                if target_count > 1
                else f"按初始对话方位{initial_bearing:.1f}°选择第{active_target_index + 1}个目标"
            )
            return (
                f"闭环仿真完成：共读取{target_count}个真实饵物坐标，{target_text}；滚动迭代{iterations}轮后"
                f"完成抵近侦察，抵近航程{approach_distance:.1f}m，累计绕航{total_orbit_turns}圈"
                f"（绕航约{orbit_distance:.1f}m），最终距目标{final_distance:.1f}m。"
            )
        return (
            f"闭环仿真未完成：共读取{target_count}个真实饵物坐标，已完成{completed_target_count}/{target_count}个目标；"
            f"首个任务按初始对话方位{initial_bearing:.1f}°选择第{active_target_index + 1}个目标；已滚动迭代{iterations}轮，"
            f"最终距目标{final_distance:.1f}m。原因：{failure_reason}。"
        )


def parse_target_positions(values: Any) -> List[Vector3]:
    """Parse multiple truth target positions from UI text or JSON-like data."""

    if isinstance(values, str):
        text = values.strip()
        if not text:
            raise ValueError("请输入至少一个真实目标坐标")
        parsed = _json_positions(text)
        if parsed:
            return parsed

        positions = [
            vector3([match.group(1), match.group(2), match.group(3) if match.group(3) is not None else -50.0])
            for match in _POSITION_RE.finditer(text)
        ]
        if positions:
            return _validate_world_positions(positions)

        line_positions: List[Vector3] = []
        for line in text.splitlines():
            numbers = re.findall(_NUMBER_RE, line)
            if len(numbers) >= 2:
                line_positions.append(vector3(numbers[:3]))
        if line_positions:
            return _validate_world_positions(line_positions)
        raise ValueError("无法识别真实目标坐标，请使用 (x,y,z) 或每行 x,y,z")

    if isinstance(values, dict):
        position = values.get("position", values.get("target_position"))
        if position is None:
            raise ValueError("目标坐标对象需要包含 position")
        return _validate_world_positions([vector3(position)])

    if isinstance(values, Sequence):
        items = list(values)
        if not items:
            raise ValueError("请输入至少一个真实目标坐标")
        if all(isinstance(item, (int, float)) for item in items[:3]):
            return _validate_world_positions([vector3(items)])
        positions = []
        for item in items:
            if isinstance(item, dict):
                item = item.get("position", item.get("target_position"))
            positions.append(vector3(item))
        return _validate_world_positions(positions)

    raise ValueError("请输入至少一个真实目标坐标")


def _json_positions(text: str) -> List[Vector3]:
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        return []
    return parse_target_positions(decoded)


def _validate_world_positions(positions: Sequence[Vector3]) -> List[Vector3]:
    return [_validate_world_position(position, f"目标{index}") for index, position in enumerate(positions, start=1)]


def _validate_world_position(position: Vector3, label: str) -> Vector3:
    x, y, z = position
    if not (WORLD_MIN <= x <= WORLD_MAX and WORLD_MIN <= y <= WORLD_MAX):
        raise ValueError(f"{label}坐标超出范围：X/Y 必须位于 {WORLD_MIN:.0f}～{WORLD_MAX:.0f}")
    if not _is_grid_value(x) or not _is_grid_value(y):
        raise ValueError(f"{label}坐标不符合分辨率：X/Y 分辨率为 {WORLD_RESOLUTION:.0f}m")
    return (float(round(x)), float(round(y)), z)


def _is_grid_value(value: float) -> bool:
    return abs((float(value) - WORLD_MIN) / WORLD_RESOLUTION - round((float(value) - WORLD_MIN) / WORLD_RESOLUTION)) < 1e-9


def _signed_angle_delta(start: float, end: float) -> float:
    return (float(end) % 360.0 - float(start) % 360.0 + 180.0) % 360.0 - 180.0


def result_distance_series(result: Dict[str, Any]) -> List[float]:
    """Return distance-to-target series for quick analysis."""

    return [float(item["distance_to_target"]) for item in result.get("uuv_history", [])]


def bearing_delta_series(result: Dict[str, Any]) -> List[float]:
    """Return signed deltas between consecutive bearing observations."""

    values = [float(item["angle"]) for item in result.get("bearing_history", [])]
    deltas = []
    for previous, current in zip(values[:-1], values[1:]):
        deltas.append((current - previous + 180.0) % 360.0 - 180.0)
    return deltas
