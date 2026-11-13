"""记忆系统：三层设计。

1. 工作记忆  = 本轮发给 LLM 的 messages（由 ConversationMemory 裁剪维护）
2. 短期记忆 = 会话历史（最近 N 轮，超出的旧轮次压缩丢弃）
3. 长期记忆 = data/user_profile.json（口味画像、忌口、去过的店，跨会话持久）
"""

from __future__ import annotations

import copy
import json
import os
import re
from typing import Any, Dict, List

import config


def _split_terms(text: str) -> List[str]:
    """把「不吃香菜、不要葱」这类口语串拆成独立条目，便于逐条去重累积。"""
    parts = re.split(r"[、,，;；/\s]+", (text or "").strip())
    return [p.strip() for p in parts if p.strip()]


class ConversationMemory:
    """会话记忆：维护 OpenAI 协议的 messages 列表，只保留最近 HISTORY_TURNS 条消息。"""

    def __init__(self, system_prompt: str) -> None:
        self.messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_assistant(self, content: str, tool_calls: List[Dict[str, Any]] | None = None) -> None:
        msg: Dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self.messages.append(msg)

    def add_tool_result(self, tool_call_id: str, result: str) -> None:
        self.messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": result})

    def trim(self) -> None:
        """裁剪掉过早的轮次（保留 system + 最近 HISTORY_TURNS 条）。

        关键约束（OpenAI 协议）：role=tool 的消息必须紧跟在带 tool_calls 的
        assistant 消息之后，成组出现。裁剪不能把一组切开，否则 API 直接报
        HTTP 400「Messages with role 'tool' must be a response to a preceding
        message with 'tool_calls'」。
        """
        keep = 1 + config.HISTORY_TURNS
        if len(self.messages) <= keep:
            return

        head = self.messages[0]  # system
        rest = self.messages[1:]

        # 按"原子组"切分：assistant(tool_calls)+其后的 tool 结果是一个整体
        groups: List[List[Dict[str, Any]]] = []
        i, n = 0, len(rest)
        while i < n:
            msg = rest[i]
            if msg.get("role") == "tool":
                # 悬空的 tool 消息（前面没有 assistant tool_calls）——直接丢弃
                i += 1
                continue
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                group = [msg]
                i += 1
                while i < n and rest[i].get("role") == "tool":
                    group.append(rest[i])
                    i += 1
                groups.append(group)
            else:
                groups.append([msg])
                i += 1

        # 从最近的组往前收集，凑满 keep 条为止；组是原子的，绝不切开
        selected: List[List[Dict[str, Any]]] = []
        count = 0
        for group in reversed(groups):
            if count + len(group) > keep and selected:
                break
            selected.insert(0, group)
            count += len(group)
            if count >= keep:
                break

        self.messages = [head] + [m for g in selected for m in g]

    def as_payload(self) -> List[Dict[str, Any]]:
        return self._sanitize(self.messages)

    @staticmethod
    def _sanitize(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """发送前自检：修复不完整的工具调用组，保证发给 LLM 的序列永远合法。

        规则（OpenAI 协议）：assistant 带 tool_calls 后，必须紧跟与每个
        tool_call_id 一一对应的 tool 消息。本方法：
          1. 丢弃悬空的 tool 消息（找不到对应的调用请求）；
          2. 若某组 tool_calls 的结果不齐全，整组删除（assistant + 相关结果）。
        """
        cleaned: List[Dict[str, Any]] = []
        pending: List[str] = []   # 等待响应的 tool_call_id
        group_ids: set = set()    # 当前组全部 tool_call_id（核销与否都要记得，整组删除时用）

        def _drop_incomplete_group() -> None:
            nonlocal pending, group_ids
            cleaned[:] = [
                m for m in cleaned
                if not (
                    (m.get("role") == "assistant" and any(tc.get("id") in group_ids for tc in (m.get("tool_calls") or [])))
                    or (m.get("role") == "tool" and m.get("tool_call_id") in group_ids)
                )
            ]
            pending = []
            group_ids = set()

        for msg in messages:
            role = msg.get("role")
            if role == "tool":
                tid = msg.get("tool_call_id")
                if tid in pending:
                    pending.remove(tid)
                    cleaned.append(msg)
                # else: 悬空 tool 消息，直接丢弃
                continue
            if role == "assistant" and msg.get("tool_calls"):
                if pending:
                    _drop_incomplete_group()
                pending = [tc.get("id") for tc in msg["tool_calls"]]
                group_ids = set(pending)
                cleaned.append(msg)
                continue
            # user / 普通 assistant 消息：若上一组 tool_calls 还有结果没回填，整组剔除
            if pending:
                _drop_incomplete_group()
            cleaned.append(msg)

        if pending:
            _drop_incomplete_group()
        return cleaned


class UserProfile:
    """长期记忆：用户画像，跨会话持久化到 user_profile.json。"""

    DEFAULTS: Dict[str, Any] = {
        "口味偏好": [],
        "忌口": [],
        "常去店铺": [],
        "不想再去": [],
        "预算习惯": "未记录",
        "备注": [],
    }

    def __init__(self, path: str = config.PROFILE_PATH) -> None:
        self.path = path
        # 必须深拷贝：DEFAULTS 是类属性，dict() 只是浅拷贝，
        # 所有实例会共享同一批 list。A 用户记了「不吃香菜」，
        # 这个值就写进了类的默认列表，之后新建的每个用户都会莫名其妙带着它——
        # 也就是「不同用户画像串数据」。
        self.data: Dict[str, Any] = copy.deepcopy(self.DEFAULTS)
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as handle:
                    loaded = json.load(handle)
                for key, value in loaded.items():
                    self.data[key] = value
            except (json.JSONDecodeError, OSError):
                pass

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(self.data, handle, ensure_ascii=False, indent=2)

    def update(self, key: str, value: Any) -> None:
        """写入画像。列表型字段只追加、不覆盖，字符串型字段才整体替换。

        历史 bug：模型有时把忌口当成一句话传（value="不吃香菜、不要葱"），
        有时传列表。旧逻辑遇到 list 会「整体替换」——后一次记的忌口把前一次冲掉，
        用户会发现自己说过三次忌口，系统只记得最后一次。
        """
        if key not in self.data:
            self.data[key] = []

        # 统一把传入值拍平成字符串列表（兼容 "a、b" / "a,b" / ["a","b"]）
        incoming: List[str] = []
        if isinstance(value, (list, tuple, set)):
            for item in value:
                incoming.extend(_split_terms(str(item)))
        else:
            incoming.extend(_split_terms(str(value)))
        incoming = [t for t in incoming if t]
        if not incoming:
            return

        if isinstance(self.data[key], list):
            for term in incoming:
                if term not in self.data[key]:
                    self.data[key].append(term)
        else:
            # 预算是单值语义，后一次覆盖前一次是合理的
            self.data[key] = incoming[0] if len(incoming) == 1 else "、".join(incoming)
        self.save()

    def as_prompt_text(self) -> str:
        """把画像渲染进 system prompt，让 LLM 记住用户是谁。"""
        parts = [f"{key}: {'、'.join(v) if isinstance(v, list) else v}" for key, v in self.data.items() if v]
        return "\n".join(parts) if parts else "（暂无用户画像记录）"
