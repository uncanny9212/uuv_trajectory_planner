"""LLM chat mode behavior tests."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from uuv_trajectory_planner.core.llm_client import LLMClient


class FakeHttpResponse:
    """Small context manager used to emulate urllib responses."""

    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeHttpResponse":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload, ensure_ascii=False).encode("utf-8")


class TestLLMClient(unittest.TestCase):
    """Validate cloud/local chat switching feedback."""

    def test_url_in_api_key_falls_back_with_clear_reason(self) -> None:
        client = LLMClient(model="test-model")

        response = client.chat(
            [{"role": "user", "content": "你好"}],
            {"api_key": "https://platform.openai.com/api-keys"},
        )

        self.assertEqual(response["source"], "local")
        self.assertTrue(response["llm_requested"])
        self.assertIn("网页链接", response["fallback_reason"])

    def test_missing_api_key_reports_local_mode(self) -> None:
        client = LLMClient(model="test-model")

        response = client.chat([{"role": "user", "content": "你好"}], {})

        self.assertEqual(response["source"], "local")
        self.assertFalse(response["llm_requested"])
        self.assertIn("未输入", response["fallback_reason"])

    def test_cloud_failure_reports_fallback_reason(self) -> None:
        client = LLMClient(model="test-model")

        def fail_chat(messages, context, *, api_key, api_base_url=""):
            return None, "测试云端失败"

        client._try_cloud_chat = fail_chat  # type: ignore[method-assign]
        response = client.chat([{"role": "user", "content": "你好"}], {"api_key": "sk-test"})

        self.assertEqual(response["source"], "local")
        self.assertTrue(response["llm_requested"])
        self.assertEqual(response["fallback_reason"], "测试云端失败")

    def test_cloud_success_reports_llm_mode(self) -> None:
        client = LLMClient(model="test-model")

        def pass_chat(messages, context, *, api_key, api_base_url=""):
            return "云端回复", None

        client._try_cloud_chat = pass_chat  # type: ignore[method-assign]
        response = client.chat([{"role": "user", "content": "你好"}], {"api_key": "sk-test"})

        self.assertEqual(response["source"], "llm")
        self.assertTrue(response["llm_requested"])
        self.assertEqual(response["reply"], "云端回复")

    def test_chat_url_normalization(self) -> None:
        self.assertEqual(
            LLMClient._chat_completions_url("https://api.openai.com/v1"),
            "https://api.openai.com/v1/chat/completions",
        )
        self.assertEqual(
            LLMClient._chat_completions_url("https://api.openai.com/v1/chat/completions"),
            "https://api.openai.com/v1/chat/completions",
        )

    def test_http_chat_path_parses_cloud_reply(self) -> None:
        client = LLMClient(model="test-model")
        response_payload = {"choices": [{"message": {"content": "云端HTTP回复"}}]}

        with patch("urllib.request.urlopen", return_value=FakeHttpResponse(response_payload)) as mocked:
            reply, reason = client._try_cloud_chat_http(
                "系统提示",
                {"conversation": [{"role": "user", "content": "你好"}]},
                api_key="sk-test",
                api_base_url="https://api.openai.com/v1",
            )

        request = mocked.call_args.args[0]
        self.assertEqual(reply, "云端HTTP回复")
        self.assertIsNone(reason)
        self.assertEqual(request.full_url, "https://api.openai.com/v1/chat/completions")


if __name__ == "__main__":
    unittest.main()
