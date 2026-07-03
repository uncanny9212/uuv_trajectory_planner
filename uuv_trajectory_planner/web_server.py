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
from uuv_trajectory_planner.core.react_engine import ReActEngine
from uuv_trajectory_planner.models.situation_awareness import SituationAwareness
from uuv_trajectory_planner.main import sample_payload


STATIC_DIR = Path(__file__).resolve().parent / "web"


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
        if self.path == "/api/detection-parse":
            self._handle_detection_parse()
            return
        if self.path == "/api/detection-plan":
            self._handle_detection_plan()
            return
        self._send_error(HTTPStatus.NOT_FOUND, "Not found")

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
            message = str(body.get("message", "")).strip()
            if not message:
                raise ValueError("请输入任务描述")
            payload = payload_from_message(message)
            self._apply_manual_overrides(payload, body)
            validate_bait_clearance(payload)
            situation = SituationAwareness.from_dict(payload)
            decision = self.engine.run(payload)
            gif_url = decision_gif_data_url(situation, decision)
            self._send_json({"input": message, "payload": payload, "decision": decision.to_dict(), "animation_url": gif_url})
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))

    def _handle_detection_parse(self) -> None:
        try:
            body = self._read_json_body()
            payload = self._payload_from_detection_body(body)
            self._send_json({"payload": payload})
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))

    def _handle_detection_plan(self) -> None:
        try:
            body = self._read_json_body()
            payload = self._payload_from_detection_body(body)
            validate_bait_clearance(payload)
            situation = SituationAwareness.from_dict(payload)
            decision = self.engine.run(payload)
            gif_url = decision_gif_data_url(situation, decision)
            self._send_json({"payload": payload, "decision": decision.to_dict(), "animation_url": gif_url})
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
        if self.path not in ("/api/plan", "/api/chat-plan", "/api/detection-parse", "/api/detection-plan"):
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
