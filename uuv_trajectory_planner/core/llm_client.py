"""LLM client with deterministic local fallback."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from uuv_trajectory_planner.models.situation_awareness import SituationAwareness


class LLMClient:
    """Wrap cloud LLM calls while preserving an offline MVP path."""

    def __init__(self, model: Optional[str] = None, timeout_seconds: int = 20) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5.5")
        self.timeout_seconds = timeout_seconds
        self.api_key = os.getenv("OPENAI_API_KEY")

    def reason(self, situation: SituationAwareness, history: Optional[List[Dict[str, Any]]] = None) -> str:
        """Return a concise explainable planning rationale."""

        if self.api_key:
            cloud_reasoning = self._try_cloud_reasoning(situation, history or [])
            if cloud_reasoning:
                return cloud_reasoning
        return self._local_reasoning(situation, history or [])

    def _try_cloud_reasoning(
        self,
        situation: SituationAwareness,
        history: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Try OpenAI SDK if installed and configured."""

        try:
            from openai import OpenAI  # type: ignore
        except Exception:
            return None

        system_prompt = (
            "你是UUV端侧智能体，负责水下自主轨迹规划。"
            "请输出简洁、可审计的决策说明，包含态势分析、约束识别、算法选择和验证检查。"
            "不要输出完整隐藏思维过程，只输出可解释决策摘要。"
        )
        user_payload = {
            "situation": situation.to_dict(),
            "recent_decisions": history[-5:],
        }
        try:
            client = OpenAI(api_key=self.api_key, timeout=self.timeout_seconds)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                ],
                temperature=0.2,
            )
            content = response.choices[0].message.content
            return content.strip() if content else None
        except Exception:
            return None

    def _local_reasoning(self, situation: SituationAwareness, history: List[Dict[str, Any]]) -> str:
        """Generate deterministic reasoning text for offline operation."""

        obstacle_count = len(situation.environment.obstacles)
        scenario = situation.mission.scenario
        history_text = f"参考最近{len(history)}轮反馈。" if history else "无历史反馈。"
        if scenario == "area_coverage":
            return (
                f"本地推理：{history_text} 当前任务为区域覆盖规划，检测到{obstacle_count}个障碍物；"
                "优先保证覆盖率和边界余量，选择往返式扫描并在执行后估算覆盖率。"
            )
        return (
            f"本地推理：{history_text} 当前任务为通用轨迹规划，检测到{obstacle_count}个障碍物；"
            "优先满足安全距离和航向平滑约束，先尝试直达路径，不可行时切换到A*栅格规划。"
        )
