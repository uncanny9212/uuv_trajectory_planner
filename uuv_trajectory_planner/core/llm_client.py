"""LLM client with deterministic local fallback."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

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

    def chat(
        self,
        messages: List[Dict[str, str]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Answer an operator chat turn, using the cloud LLM when configured."""

        values = context or {}
        api_key = str(values.get("api_key", "")).strip()
        api_base_url = str(values.get("api_base_url", "")).strip()
        local_reply = self._local_chat(messages, values)
        if self._looks_like_url(api_key):
            return {
                "reply": local_reply,
                "source": "local",
                "model": "local-rule",
                "llm_requested": True,
                "fallback_reason": "API_KEY 字段看起来是网页链接，请粘贴实际密钥字符串，而不是页面地址。",
            }
        if api_key:
            cloud_reply, fallback_reason = self._try_cloud_chat(
                messages,
                values,
                api_key=api_key,
                api_base_url=api_base_url,
            )
            if cloud_reply:
                return {"reply": cloud_reply, "source": "llm", "model": self.model, "llm_requested": True}
            return {
                "reply": local_reply,
                "source": "local",
                "model": "local-rule",
                "llm_requested": True,
                "fallback_reason": fallback_reason
                or "LLM 调用失败，已切换本地模式。请检查 API_KEY、API地址和网络连通性。",
            }
        return {
            "reply": local_reply,
            "source": "local",
            "model": "local-rule",
            "llm_requested": False,
            "fallback_reason": "未输入 API_KEY，使用本地模式。",
        }

    def _try_cloud_chat(
        self,
        messages: List[Dict[str, str]],
        context: Dict[str, Any],
        *,
        api_key: str,
        api_base_url: str = "",
    ) -> Tuple[Optional[str], Optional[str]]:
        system_prompt, payload = self._chat_prompt_payload(messages, context)
        try:
            from openai import OpenAI  # type: ignore
        except Exception:
            return self._try_cloud_chat_http(
                system_prompt,
                payload,
                api_key=api_key,
                api_base_url=api_base_url,
            )

        try:
            client_kwargs: Dict[str, Any] = {"api_key": api_key, "timeout": self.timeout_seconds}
            if api_base_url:
                client_kwargs["base_url"] = self._sdk_base_url(api_base_url)
            client = OpenAI(**client_kwargs)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                temperature=0.2,
            )
            content = response.choices[0].message.content
            return (content.strip(), None) if content else (None, "LLM 返回为空，已切换本地模式。")
        except Exception as exc:  # pylint: disable=broad-exception-caught
            return None, self._cloud_chat_error_message(exc)

    def _try_cloud_chat_http(
        self,
        system_prompt: str,
        payload: Dict[str, Any],
        *,
        api_key: str,
        api_base_url: str = "",
    ) -> Tuple[Optional[str], Optional[str]]:
        import urllib.error
        import urllib.request

        request_payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                "temperature": 0.2,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            self._chat_completions_url(api_base_url),
            data=request_payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310
                data = json.loads(response.read().decode("utf-8"))
            content = data.get("choices", [{}])[0].get("message", {}).get("content")
            return (content.strip(), None) if content else (None, "LLM 返回为空，已切换本地模式。")
        except urllib.error.HTTPError as exc:
            return None, self._cloud_http_error_message(exc)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            return None, self._cloud_chat_error_message(exc)

    @staticmethod
    def _chat_prompt_payload(
        messages: List[Dict[str, str]],
        context: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        system_prompt = (
            "你是UUV闭环仿真Agent，负责和用户对话确认任务意图、目标坐标、发现距离、航段策略、"
            "驾驶员反馈语气，以及任务后基于深度的自主决策规则。"
            "回答要简洁、可执行；不要声称知道真实目标距离，发现半径外只能使用方位信息。"
            "当用户给出目标坐标或参数时，请复述已识别内容；当信息不足时，请提出一个最关键的问题。"
            "红方或中立目标不得进入模拟打击与授权流程，只能跟踪、复核、协同或返航。"
            "攻击相关内容仅作为仿真授权流程表达，不提供真实武器控制细节。"
        )
        payload = {
            "context": {key: value for key, value in context.items() if key != "api_key"},
            "conversation": messages[-12:],
        }
        return system_prompt, payload

    @staticmethod
    def _chat_completions_url(api_base_url: str) -> str:
        base_url = (api_base_url or "https://api.openai.com/v1").strip().rstrip("/")
        if base_url.endswith("/chat/completions"):
            return base_url
        return f"{base_url}/chat/completions"

    @staticmethod
    def _sdk_base_url(api_base_url: str) -> str:
        base_url = api_base_url.strip().rstrip("/")
        if base_url.endswith("/chat/completions"):
            return base_url[: -len("/chat/completions")]
        return base_url

    @staticmethod
    def _looks_like_url(value: str) -> bool:
        normalized = value.strip().lower()
        return normalized.startswith("http://") or normalized.startswith("https://")

    @staticmethod
    def _cloud_chat_error_message(exc: Exception) -> str:
        message = str(exc).lower()
        if "401" in message or "unauthorized" in message or "incorrect api key" in message:
            return "API_KEY 未通过认证，请确认粘贴的是实际密钥字符串。"
        if "404" in message or "not found" in message:
            return "API地址或模型不可用，请检查 API地址和模型配置。"
        if "timeout" in message or "timed out" in message:
            return "LLM 请求超时，已切换本地模式。请检查网络或稍后重试。"
        return "LLM 调用失败，已切换本地模式。请检查 API_KEY、API地址和网络连通性。"

    @staticmethod
    def _cloud_http_error_message(exc: Exception) -> str:
        status = getattr(exc, "code", None)
        if status == 401:
            return "API_KEY 未通过认证，请确认粘贴的是实际密钥字符串。"
        if status == 404:
            return "API地址或模型不可用，请检查 API地址和模型配置。"
        if status == 429:
            return "LLM 请求被限流，已切换本地模式。请稍后重试。"
        return "LLM 调用失败，已切换本地模式。请检查 API_KEY、API地址和网络连通性。"

    def _local_chat(self, messages: List[Dict[str, str]], context: Dict[str, Any]) -> str:
        last = next((item.get("content", "") for item in reversed(messages) if item.get("role") == "user"), "")
        target_count = int(context.get("target_count", 0) or 0)
        discovery_range = context.get("discovery_range", 50)
        if any(word in last for word in ["深度", "决策", "返航", "二次", "协同", "打击", "授权"]):
            return (
                "当前任务后决策先检查IFF：红方或中立目标不执行打击，只能跟踪、复核、协同或返航。"
                "其他目标再结合已确认深度决策：10m及以内返航，10到30m二次查看，"
                "30到60m召集其他UUV，超过60m才可能进入模拟打击待机并请求授权。"
            )
        if any(word in last for word in ["坐标", "目标", "饵物"]):
            return (
                f"收到。我会从对话中识别目标坐标和深度；当前已在参数区看到{target_count}个目标，"
                f"发现距离为{discovery_range}m。"
            )
        if any(word in last for word in ["语气", "反馈", "驾驶员"]):
            return "收到。驾驶员反馈可按简练、正式、口语化或详细风格表达，并会和轨迹进度同步。"
        return (
            "收到。我会把你的对话作为任务语义输入，用于识别方位、目标坐标、发现距离和反馈偏好；"
            "开始仿真后，UUV会按方位滚动抵近，并在发现目标后基于深度自主决策。"
        )

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
