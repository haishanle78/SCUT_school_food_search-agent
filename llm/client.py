"""OpenAI 兼容的 LLM HTTP 客户端：原生支持 function calling（tools）。

零第三方依赖，urllib 直连。兼容 DeepSeek / 智谱 / OpenAI 等一切
OpenAI chat/completions 协议的服务。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib import error, request

import config


@dataclass
class ToolCall:
    """LLM 发起的一次工具调用请求。"""

    id: str
    name: str
    arguments: str  # JSON 字符串，执行时再解析

    def parse_arguments(self) -> Dict[str, Any]:
        try:
            return json.loads(self.arguments or "{}")
        except json.JSONDecodeError:
            return {}


@dataclass
class AssistantMessage:
    """LLM 的一个回合：要么是最终回答，要么带工具调用请求。"""

    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)

    @property
    def wants_tool(self) -> bool:
        return bool(self.tool_calls)


class LLMClient:
    def __init__(self) -> None:
        self.enabled = config.LLM_ENABLED

    # ---- 对外主入口 ----
    def chat(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> AssistantMessage:
        """发起一轮对话。messages 遵循 OpenAI 协议（含 role=tool 的回填消息）。"""
        if not self.enabled:
            raise RuntimeError("LLM 未配置：请复制 .env.example 为 .env 并填入 API Key")

        payload: Dict[str, Any] = {
            "model": config.LLM_MODEL,
            "messages": messages,
            "temperature": config.LLM_TEMPERATURE,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            config.LLM_BASE_URL,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config.LLM_API_KEY}",
            },
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"LLM 调用失败: HTTP {exc.code} {detail[:300]}") from exc
        except Exception as exc:
            raise RuntimeError(f"LLM 网络异常: {exc}") from exc

        return self._parse_response(result)

    # ---- 协议解析 ----
    # 模型偶尔把流式协议片段当正文吐出来：
    #   {"index":0,"finish_reason":"tool_calls","delta":{"role":"assistant",...}}
    # 这是它自己想要发起工具调用却写成了文本，用户不该看到这种原始 JSON。
    _PROTOCOL_LEAK_RE = re.compile(
        r'\{\s*"index"\s*:\s*\d+\s*,\s*"finish_reason"'
        r'|"tool_calls"\s*:\s*\[\s*\{\s*"id"\s*:'
    )

    @classmethod
    def _strip_protocol_leak(cls, content: str) -> str:
        if not content:
            return content
        match = cls._PROTOCOL_LEAK_RE.search(content)
        if not match:
            return content
        cut = content.rfind("{", 0, match.start() + 1)
        cleaned = content[: cut if cut != -1 else match.start()]
        return cleaned.rstrip(" \t\r\n{[,:\"")

    def _parse_response(self, result: Dict[str, Any]) -> AssistantMessage:
        try:
            msg = result["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"LLM 返回格式异常: {json.dumps(result, ensure_ascii=False)[:300]}") from exc

        assistant = AssistantMessage(content=self._strip_protocol_leak((msg.get("content") or "").strip()))
        for tc in msg.get("tool_calls") or []:
            try:
                fn = tc["function"]
                assistant.tool_calls.append(ToolCall(id=tc.get("id", ""), name=fn["name"], arguments=fn.get("arguments", "{}")))
            except (KeyError, TypeError):
                continue
        return assistant
