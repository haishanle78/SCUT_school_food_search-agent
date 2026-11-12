"""OpenAI 兼容的 LLM HTTP 客户端：原生支持 function calling（tools）。

零第三方依赖，urllib 直连。兼容 DeepSeek / 智谱 / OpenAI 等一切
OpenAI chat/completions 协议的服务。

并发治理：所有请求经由全局 KeyPool（llm/keypool.py）——
  - 排队：空闲 Key 不足时等待，超过 LLM_QUEUE_WAIT 抛 LLMBusy（上层友好提示，不降级离线）；
  - 分流：调度器挑最闲的 Key 放行，单 Key 并发不超过 LLM_CONCURRENCY_PER_KEY；
  - 熔断换钥匙：限流类错误（429/1214/并发超限）熔断当前 Key 并自动换下一把重试，
    用户无感；非限流错误（如上下文被拒 400）按原逻辑向上抛。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib import error, request

import config
from llm.keypool import KeyPool, LLMBusy

__all__ = ["ToolCall", "AssistantMessage", "LLMClient", "LLMBusy", "get_pool", "reset_pool_for_tests"]


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


# 全局唯一 Key 池：跨会话/跨用户共享（进程内单例）。
_POOL: Optional[KeyPool] = None


def get_pool() -> KeyPool:
    global _POOL
    if _POOL is None:
        _POOL = KeyPool(
            config.LLM_KEYS,
            config.LLM_CONCURRENCY_PER_KEY,
            config.LLM_QUEUE_WAIT,
            config.LLM_BREAKER_COOLDOWN,
        )
    return _POOL


def reset_pool_for_tests() -> None:
    """测试专用：丢弃全局池，下次 get_pool() 会按当前 config 重建。"""
    global _POOL
    _POOL = None


class LLMClient:
    def __init__(self, pool: Optional[KeyPool] = None) -> None:
        self.pool = pool if pool is not None else get_pool()
        self.enabled = config.LLM_ENABLED

    # ---- 对外主入口 ----
    def chat(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> AssistantMessage:
        """发起一轮对话。messages 遵循 OpenAI 协议（含 role=tool 的回填消息）。

        经 Key 池调度：排队 → 取 Key → 调用 → 限流类错误自动换钥匙重试。
        """
        if not self.enabled or not self.pool.enabled:
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
        failovers = self.pool.size

        for attempt in range(failovers):
            slot = self.pool.acquire()  # 排队超时抛 LLMBusy，向上透传
            try:
                result = self._do_chat(body, slot.key)
                self.pool.mark_success(slot)
                return self._parse_response(result)
            except RuntimeError as exc:
                tripped = self.pool.mark_failure(slot, exc)
                if tripped and attempt < failovers - 1:
                    continue  # 该 Key 被熔断，换下一把，用户无感
                raise
            finally:
                self.pool.release(slot)

        raise RuntimeError("LLM 调用失败：所有 Key 均不可用")  # 理论上到不了这里

    def _do_chat(self, body: bytes, api_key: str) -> Dict[str, Any]:
        """真正发 HTTP 的地方（测试里会被替换）。"""
        req = request.Request(
            config.LLM_BASE_URL,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"LLM 调用失败: HTTP {exc.code} {detail[:300]}") from exc
        except Exception as exc:
            raise RuntimeError(f"LLM 网络异常: {exc}") from exc

    # ---- 协议解析 ----
    def _parse_response(self, result: Dict[str, Any]) -> AssistantMessage:
        try:
            msg = result["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"LLM 返回格式异常: {json.dumps(result, ensure_ascii=False)[:300]}") from exc

        assistant = AssistantMessage(content=(msg.get("content") or "").strip())
        for tc in msg.get("tool_calls") or []:
            try:
                fn = tc["function"]
                assistant.tool_calls.append(ToolCall(id=tc.get("id", ""), name=fn["name"], arguments=fn.get("arguments", "{}")))
            except (KeyError, TypeError):
                continue
        return assistant
