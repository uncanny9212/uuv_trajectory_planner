"""Small local web server for the UUV planner MVP."""

from __future__ import annotations

import argparse
import json
import math
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from uuv_trajectory_planner.animation import decision_gif_data_url
from uuv_trajectory_planner.core.chat_parser import payload_from_message
from uuv_trajectory_planner.core.detection_parser import DetectionParser
from uuv_trajectory_planner.core.llm_client import LLMClient
from uuv_trajectory_planner.core.react_engine import ReActEngine
from uuv_trajectory_planner.core.rolling_planner import RollingPlanner
from uuv_trajectory_planner.models.situation_awareness import SituationAwareness
from uuv_trajectory_planner.main import sample_payload
from uuv_trajectory_planner.simulation import SimulationRunner, default_scenarios, scenario_by_name
from uuv_trajectory_planner.simulation.visualization import simulation_plot_data_url


STATIC_DIR = Path(__file__).resolve().parent / "web"


def rolling_preview_from_text(text: str, body: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Build a rolling-planning preview when text has bearing but no range."""

    values = body or {}
    parser = DetectionParser(
        default_detection_range=float(values.get("default_detection_range", 800.0)),
        default_depth=float(values.get("default_depth", -50.0)),
    )
    bearing, distance = parser._extract_bearing_distance(text)  # pylint: disable=protected-access
    if bearing is None or distance is not None:
        return None

    mission_context = "抵近侦察" if parser._is_close_recon_context(text) else "方位滚动规划"  # pylint: disable=protected-access
    constraints = {
        "approach_range": float(values.get("approach_range", 100.0)),
        "orbit_turns": int(
            values.get("orbit_turns")
            or parser._extract_orbit_turns(text)  # pylint: disable=protected-access
            or (2 if mission_context == "抵近侦察" else 0)
        ),
        "orbit_radius": float(
            values.get("orbit_radius")
            or parser._extract_orbit_radius(text)  # pylint: disable=protected-access
            or 10.0
        ),
        "max_iterations": int(values.get("max_iterations", 8)),
        "default_step": float(values.get("default_step", 300.0)),
    }
    range_proxy = max(
        constraints["approach_range"] + constraints["default_step"],
        float(values.get("rolling_initial_range", values.get("default_detection_range", 800.0))),
    )
    uuv_position = values.get("uuv_position", [0.0, 0.0, -50.0])
    if not isinstance(uuv_position, list) or len(uuv_position) < 2:
        raise ValueError("UUV位置必须至少包含 x 和 y")
    if len(uuv_position) == 2:
        uuv_position = [uuv_position[0], uuv_position[1], -50.0]

    planner = RollingPlanner()
    bearing_history = [{"angle": float(bearing), "timestamp": "00:00:00"}]
    uuv_state = {
        "position": [float(uuv_position[0]), float(uuv_position[1]), float(uuv_position[2])],
        "heading": float(values.get("uuv_heading", bearing)),
        "speed": float(values.get("uuv_speed", 2.0)),
        "battery": float(values.get("battery", 0.8)),
    }
    iterations = []

    for iteration in range(constraints["max_iterations"]):
        rolling_payload = {
            "bearing_history": list(bearing_history),
            "uuv_state": dict(uuv_state),
            "estimated_range": range_proxy,
            "constraints": constraints,
            "iteration": iteration,
            "mission_context": mission_context,
        }
        decision = planner.plan(rolling_payload).to_dict()
        feedback = _rolling_feedback(
            decision=decision,
            bearing_history=bearing_history,
            uuv_state=uuv_state,
            range_proxy=range_proxy,
            approach_range=constraints["approach_range"],
            iteration=iteration,
        )
        iterations.append(
            {
                "iteration": iteration + 1,
                "observation": bearing_history[-1],
                "range_proxy": round(range_proxy, 3),
                "decision": decision,
                "feedback": feedback,
            }
        )

        if decision["decision"] in ("orbit", "approach_complete", "give_up"):
            break

        if feedback.get("updated_position") is not None:
            uuv_state["position"] = feedback["updated_position"]
        uuv_state["heading"] = decision["next_heading"]
        range_proxy = float(feedback.get("updated_range_proxy", range_proxy))
        if feedback.get("next_bearing") is not None:
            bearing_history.append(
                {
                    "angle": feedback["next_bearing"],
                    "timestamp": _rolling_timestamp_seconds(len(bearing_history) * 30),
                }
            )

    summary = _rolling_summary(text, bearing, range_proxy, constraints, iterations)
    return {
        "mode": "rolling",
        "input": text,
        "payload": {
            "bearing_history": bearing_history,
            "uuv_state": uuv_state,
            "estimated_range": range_proxy,
            "constraints": constraints,
            "mission_context": mission_context,
        },
        "rolling": {
            "detected_bearing": round(float(bearing), 3),
            "explicit_distance": False,
            "summary": summary,
            "iterations": iterations,
        },
        "animation_url": None,
    }


def _rolling_feedback(
    *,
    decision: Dict[str, Any],
    bearing_history: Sequence[Dict[str, Any]],
    uuv_state: Dict[str, Any],
    range_proxy: float,
    approach_range: float,
    iteration: int,
) -> Dict[str, Any]:
    decision_type = decision["decision"]
    if decision_type == "wait":
        next_bearing = _simulated_next_bearing(float(bearing_history[-1]["angle"]), iteration)
        return {
            "action": "补充方位观测",
            "executed_distance": 0.0,
            "updated_position": None,
            "updated_range_proxy": round(range_proxy, 3),
            "next_bearing": round(next_bearing, 3),
            "note": "无距离输入，先等待下一帧方位反馈。",
        }
    if decision_type not in ("advance", "adjust_heading"):
        return {
            "action": "模式切换",
            "executed_distance": 0.0,
            "updated_position": None,
            "updated_range_proxy": round(range_proxy, 3),
            "next_bearing": None,
            "note": "滚动规划已结束本轮抵近阶段。",
        }

    distance = float(decision["advance_distance"])
    next_position = _move_by_heading(uuv_state["position"], float(decision["next_heading"]), distance)
    updated_range = max(approach_range, range_proxy - distance)
    next_bearing = _simulated_next_bearing(float(bearing_history[-1]["angle"]), iteration)
    return {
        "action": "执行下一步机动",
        "executed_distance": round(distance, 3),
        "updated_position": [round(value, 3) for value in next_position],
        "updated_range_proxy": round(updated_range, 3),
        "next_bearing": round(next_bearing, 3),
        "note": "用执行反馈更新位置和距离代理，再进入下一轮决策。",
    }


def _move_by_heading(position: Sequence[float], heading: float, distance: float) -> list[float]:
    radians = math.radians(heading)
    return [
        float(position[0]) + math.sin(radians) * distance,
        float(position[1]) + math.cos(radians) * distance,
        float(position[2]),
    ]


def _simulated_next_bearing(current_bearing: float, iteration: int) -> float:
    return (current_bearing + max(0.4, 1.8 - iteration * 0.35)) % 360.0


def _rolling_timestamp_seconds(seconds: int) -> str:
    minutes, second = divmod(seconds, 60)
    hour, minute = divmod(minutes, 60)
    return f"{hour:02d}:{minute:02d}:{second:02d}"


def _rolling_summary(
    text: str,
    bearing: float,
    final_range_proxy: float,
    constraints: Dict[str, Any],
    iterations: Sequence[Dict[str, Any]],
) -> str:
    last_decision = iterations[-1]["decision"]["decision"] if iterations else "wait"
    return (
        f"识别到无距离方位输入：“{text}”。初始方位为{bearing:.1f}°，未把默认距离转换成固定目标坐标；"
        f"系统改用滚动规划，根据方位反馈逐轮决策。共执行{len(iterations)}轮，"
        f"最终距离代理约{final_range_proxy:.1f}m，抵近阈值{constraints['approach_range']:.1f}m，"
        f"末轮决策为{last_decision}。"
    )


def json_safe(value: Any) -> Any:
    """Return a JSON-standard-safe copy of API response data."""

    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def validate_bait_clearance(payload: Dict[str, Any]) -> None:
    """Reject bait coordinates that are inside any obstacle safety zone."""

    mission = payload.get("mission", {})
    constraints = mission.get("constraints", {})
    safety_distance = float(constraints.get("min_obstacle_distance", 50.0))
    environment = payload.get("environment", {})
    obstacles = environment.get("obstacles", [])
    baits = environment.get("baits", [])
    for bait in baits:
        bait_position = bait.get("position", [0.0, 0.0, -50.0])
        bait_id = str(bait.get("id", "bait"))
        for obstacle in obstacles:
            obstacle_position = obstacle.get("position", [0.0, 0.0, -50.0])
            obstacle_id = str(obstacle.get("id", "obstacle"))
            limit = float(obstacle.get("radius", 0.0)) + safety_distance
            distance = math.hypot(
                float(bait_position[0]) - float(obstacle_position[0]),
                float(bait_position[1]) - float(obstacle_position[1]),
            )
            if distance < limit:
                raise ValueError(
                    f"饵物 {bait_id} 坐标不可用：位于障碍物 {obstacle_id} 的安全距离内，"
                    f"至少需要 {limit:.1f}m，当前距离 {distance:.1f}m。"
                )


class PlannerWebHandler(BaseHTTPRequestHandler):
    """Serve the web UI and planning API."""

    engine = ReActEngine()
    simulation_runner = SimulationRunner()
    llm_client = LLMClient(timeout_seconds=6)

    def do_HEAD(self) -> None:
        """Support lightweight health checks for the static entry."""

        requested = self.path.split("?", 1)[0]
        if requested in ("", "/"):
            requested = "/index.html"
        target = (STATIC_DIR / requested.lstrip("/")).resolve()
        if not str(target).startswith(str(STATIC_DIR.resolve())) or not target.exists() or not target.is_file():
            self.send_response(HTTPStatus.NOT_FOUND)
            self.end_headers()
            return
        content_type, _ = mimetypes.guess_type(str(target))
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(target.stat().st_size))
        self.end_headers()

    def do_GET(self) -> None:
        """Serve static UI files."""

        if self.path == "/api/sample/general":
            self._send_json(sample_payload("general"))
            return
        if self.path == "/api/sample/area_coverage":
            self._send_json(sample_payload("area_coverage"))
            return
        if self.path == "/api/simulation/scenarios":
            self._send_json({"scenarios": [scenario.to_dict() for scenario in default_scenarios()]})
            return

        requested = self.path.split("?", 1)[0]
        if requested in ("", "/"):
            requested = "/index.html"
        target = (STATIC_DIR / requested.lstrip("/")).resolve()
        if not str(target).startswith(str(STATIC_DIR.resolve())) or not target.exists() or not target.is_file():
            self._send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        content_type, _ = mimetypes.guess_type(str(target))
        body = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        """Run trajectory planning for submitted JSON."""

        if self.path == "/api/plan":
            self._handle_plan()
            return
        if self.path == "/api/chat-plan":
            self._handle_chat_plan()
            return
        if self.path == "/api/agent-chat":
            self._handle_agent_chat()
            return
        if self.path == "/api/detection-parse":
            self._handle_detection_parse()
            return
        if self.path == "/api/detection-plan":
            self._handle_detection_plan()
            return
        if self.path == "/api/simulation/run":
            self._handle_simulation_run()
            return
        if self.path == "/api/simulation/interactive":
            self._handle_simulation_interactive()
            return
        if self.path == "/api/simulation/batch":
            self._handle_simulation_batch()
            return
        self._send_error(HTTPStatus.NOT_FOUND, f"Not found: {self.path}")

    def _handle_plan(self) -> None:
        try:
            payload = self._read_json_body()
            if not isinstance(payload, dict):
                raise ValueError("Request body must be a JSON object")
            decision = self.engine.run(payload)
            self._send_json(decision.to_dict())
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))

    def _handle_chat_plan(self) -> None:
        try:
            body = self._read_json_body()
            if not isinstance(body, dict):
                raise ValueError("Request body must be a JSON object")
            if isinstance(body.get("messages"), list):
                self._send_json({"mode": "agent_chat", **self._agent_chat_response(body)})
                return
            message = str(body.get("message", "")).strip()
            if not message:
                raise ValueError("请输入任务描述")
            rolling_result = rolling_preview_from_text(message, body)
            if rolling_result is not None:
                self._send_json(rolling_result)
                return
            payload = payload_from_message(message)
            self._apply_manual_overrides(payload, body)
            validate_bait_clearance(payload)
            situation = SituationAwareness.from_dict(payload)
            decision = self.engine.run(payload)
            gif_url = decision_gif_data_url(situation, decision)
            self._send_json({"input": message, "payload": payload, "decision": decision.to_dict(), "animation_url": gif_url})
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))

    def _handle_agent_chat(self) -> None:
        try:
            body = self._read_json_body()
            if not isinstance(body, dict):
                raise ValueError("Request body must be a JSON object")
            self._send_json({"mode": "agent_chat", **self._agent_chat_response(body)})
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))

    def _agent_chat_response(self, body: Dict[str, Any]) -> Dict[str, Any]:
        messages = body.get("messages", [])
        if not isinstance(messages, list) or not messages:
            raise ValueError("请输入对话内容")
        cleaned_messages = []
        for item in messages[-20:]:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).strip()
            content = str(item.get("content", "")).strip()
            if role in ("user", "assistant") and content:
                cleaned_messages.append({"role": role, "content": content})
        if not cleaned_messages:
            raise ValueError("请输入有效对话内容")
        context = body.get("context", {}) if isinstance(body.get("context", {}), dict) else {}
        return self.llm_client.chat(cleaned_messages, context)

    def _handle_detection_parse(self) -> None:
        try:
            body = self._read_json_body()
            if not isinstance(body, dict):
                raise ValueError("Request body must be a JSON object")
            detection_text = str(body.get("detection_text", "")).strip()
            rolling_result = rolling_preview_from_text(detection_text, body)
            if rolling_result is not None:
                self._send_json(rolling_result)
                return
            payload = self._payload_from_detection_body(body)
            self._send_json({"payload": payload})
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))

    def _handle_detection_plan(self) -> None:
        try:
            body = self._read_json_body()
            if not isinstance(body, dict):
                raise ValueError("Request body must be a JSON object")
            detection_text = str(body.get("detection_text", "")).strip()
            rolling_result = rolling_preview_from_text(detection_text, body)
            if rolling_result is not None:
                self._send_json(rolling_result)
                return
            payload = self._payload_from_detection_body(body)
            validate_bait_clearance(payload)
            situation = SituationAwareness.from_dict(payload)
            decision = self.engine.run(payload)
            gif_url = decision_gif_data_url(situation, decision)
            self._send_json({"payload": payload, "decision": decision.to_dict(), "animation_url": gif_url})
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))

    def _handle_simulation_run(self) -> None:
        try:
            body = self._read_json_body()
            if not isinstance(body, dict):
                raise ValueError("Request body must be a JSON object")
            scenario = scenario_by_name(str(body.get("scenario", "正前方")))
            result = self.simulation_runner.run(scenario)
            self._send_json(
                {
                    "mode": "simulation",
                    "result": result,
                    "plot_url": simulation_plot_data_url(result),
                }
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))

    def _handle_simulation_interactive(self) -> None:
        try:
            body = self._read_json_body()
            if not isinstance(body, dict):
                raise ValueError("Request body must be a JSON object")
            result = self.simulation_runner.run_interactive(body)
            self._send_json(
                {
                    "mode": "interactive_simulation",
                    "result": result,
                }
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))

    def _handle_simulation_batch(self) -> None:
        try:
            report = self.simulation_runner.run_batch()
            first_result = report["results"][0] if report["results"] else None
            self._send_json(
                {
                    "mode": "simulation_batch",
                    "report": report,
                    "plot_url": simulation_plot_data_url(first_result) if first_result else None,
                }
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))

    def _payload_from_detection_body(self, body: Any) -> Dict[str, Any]:
        if not isinstance(body, dict):
            raise ValueError("Request body must be a JSON object")
        detection_text = str(body.get("detection_text", "")).strip()
        if not detection_text:
            raise ValueError("请输入探测语义")

        parser = DetectionParser(
            default_detection_range=float(body.get("default_detection_range", 500.0)),
            default_depth=float(body.get("default_depth", -50.0)),
        )
        uuv_position = body.get("uuv_position", [0.0, 0.0, -50.0])
        if not isinstance(uuv_position, list) or len(uuv_position) < 2:
            raise ValueError("UUV位置必须至少包含 x 和 y")
        payload = parser.parse(detection_text, uuv_position=uuv_position)
        if body.get("min_obstacle_distance") is not None:
            payload["mission"].setdefault("constraints", {})["min_obstacle_distance"] = float(
                body["min_obstacle_distance"]
            )
        return payload

    def _apply_manual_overrides(self, payload: Dict[str, Any], body: Dict[str, Any]) -> None:
        """Apply UI-provided obstacle, bait, and safety-distance controls."""

        if body.get("min_obstacle_distance") is not None:
            payload["mission"].setdefault("constraints", {})["min_obstacle_distance"] = float(
                body["min_obstacle_distance"]
            )

        obstacles = body.get("obstacles")
        if isinstance(obstacles, list):
            payload.setdefault("environment", {})["obstacles"] = [
                {
                    "id": item["id"],
                    "type": item.get("type", "static"),
                    "position": item["position"],
                    "radius": item["radius"],
                    "velocity": item.get("velocity", [0.0, 0.0, 0.0]),
                }
                for item in self._clean_spatial_items(obstacles, "O", 50.0)
            ]

        baits = body.get("baits")
        if isinstance(baits, list):
            payload.setdefault("environment", {})["baits"] = [
                {
                    "id": item["id"],
                    "position": item["position"],
                    "radius": item["radius"],
                }
                for item in self._clean_spatial_items(baits, "B", 40.0)
            ]

    def _clean_spatial_items(self, items: Sequence[Any], prefix: str, default_radius: float) -> list[Dict[str, Any]]:
        cleaned = []
        for index, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                continue
            position = item.get("position")
            if not isinstance(position, list) or len(position) < 2:
                position = [item.get("x", 0.0), item.get("y", 0.0), item.get("z", -50.0)]
            if len(position) == 2:
                position = [position[0], position[1], -50.0]
            cleaned.append(
                {
                    "id": str(item.get("id") or f"{prefix}{index:03d}"),
                    "position": [float(position[0]), float(position[1]), float(position[2])],
                    "radius": float(item.get("radius", default_radius)),
                    "type": str(item.get("type") or "static"),
                    "velocity": item.get("velocity", [0.0, 0.0, 0.0]),
                }
            )
        return cleaned

    def _read_json_body(self) -> Any:
        if self.path not in (
            "/api/plan",
            "/api/chat-plan",
            "/api/agent-chat",
            "/api/detection-parse",
            "/api/detection-plan",
            "/api/simulation/run",
            "/api/simulation/interactive",
            "/api/simulation/batch",
        ):
            self._send_error(HTTPStatus.NOT_FOUND, "Not found")
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length)
        return json.loads(raw_body.decode("utf-8"))

    def log_message(self, format: str, *args: Any) -> None:
        """Reduce request log noise."""

        return

    def _send_json(self, payload: Dict[str, Any]) -> None:
        body = json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        body = json.dumps({"status": "error", "message": message}, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the local planner web server."""

    server = ThreadingHTTPServer((host, port), PlannerWebHandler)
    print(f"UUV planner web UI: http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UUV planner web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run_server(host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
