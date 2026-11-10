"""协议健壮性压力测试：验证历史裁剪/发送自检永不产生非法消息序列。

协议铁律（OpenAI 兼容）：assistant(tool_calls) 后必须紧跟与其每个
tool_call_id 一一对应的 tool 结果消息。任何违反都会被 DeepSeek 等
服务端以 HTTP 400 拒绝。

跑法：python tests/test_protocol.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from agent.core import FoodAdvisorAgent, _trace
from agent.memory import ConversationMemory, UserProfile
from llm.client import AssistantMessage, ToolCall
from tools import build_registry


def validate(payload, ctx=""):
    """挂起清单核销式校验：唯一正确的协议检查方法。"""
    pending = []
    for i, msg in enumerate(payload):
        role = msg.get("role")
        if role == "assistant" and msg.get("tool_calls"):
            assert not pending, f"{ctx} 位置{i}: 上一组 tool_calls 未核销完"
            pending = [tc.get("id") for tc in msg["tool_calls"]]
        elif role == "tool":
            assert msg.get("tool_call_id") in pending, f"{ctx} 位置{i}: 悬空 tool 消息"
            pending.remove(msg["tool_call_id"])
        else:
            assert not pending, f"{ctx} 位置{i}: tool_calls 缺少结果回填"
    assert not pending, f"{ctx}: 结尾仍有未核销的 tool_calls {pending}"


def part1_memory_stress():
    """Part 1：直接压 ConversationMemory——50 轮长会话，强制触发裁剪。"""
    m = ConversationMemory("system")
    for turn in range(50):
        m.add_user(f"问题{turn}")
        m.add_assistant("", tool_calls=[
            {"id": f"t{turn}-{k}", "type": "function", "function": {"name": "food_search", "arguments": "{}"}}
            for k in range(3)  # 每轮 3 个并行工具调用，制造连续 tool 消息
        ])
        for k in range(3):
            m.add_tool_result(f"t{turn}-{k}", "结果")
        m.add_assistant(f"回答{turn}")
        m.trim()
        validate(m.as_payload(), f"[memory 第{turn}轮]")
    print("Part 1 记忆压力测试（50轮×3工具）: 通过")


class FakeLLM:
    """假 LLM：按脚本回放，不走网络；记录所有收到的 payload 供校验。"""

    def __init__(self, script):
        self.script = script
        self.i = 0
        self.enabled = True
        self.seen = 0

    def chat(self, messages, tools=None):
        validate(messages, "[FakeLLM 收到的 payload]")
        self.seen += 1
        msg = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        if isinstance(msg, str) and msg == "END":
            return AssistantMessage(content="最终回答")
        return msg


def make_agent(script):
    agent = FoodAdvisorAgent.__new__(FoodAdvisorAgent)  # 跳过真实 LLM 初始化
    agent.llm = FakeLLM(script)
    agent.profile = UserProfile()
    agent.poi_store = {}
    agent.functions, agent.schemas = build_registry(agent.profile, agent.poi_store)
    agent.memory = ConversationMemory("")
    agent._trace_fn = _trace  # 正常初始化流程会设置，__new__ 路径需手动补齐
    agent.last_cards = None
    agent.last_summary = ""
    agent.last_dropped = []
    agent.origin_key = config.DEFAULT_ORIGIN
    agent.origin = config.origin_by_key(agent.origin_key)
    agent._needed_modes = ["walking"]
    return agent


def part2_full_loop():
    """Part 2：完整 run() 循环——多工具/未知工具/坏参数/MAX_STEPS 用尽混合轰炸。"""
    script = [
        AssistantMessage(tool_calls=[
            ToolCall(id="c1", name="food_search", arguments='{"query": "辣"}'),
            ToolCall(id="c2", name="plan_route", arguments='{"place_name": "测试店"}'),
        ]),
        AssistantMessage(tool_calls=[
            ToolCall(id="c3", name="no_such_tool", arguments="{}"),          # 未知工具
            ToolCall(id="c4", name="estimate_budget", arguments='{"错误参数": 1}'),  # 坏参数
        ]),
        AssistantMessage(tool_calls=[
            ToolCall(id="c5", name="remember_preference", arguments='{"key": "口味偏好", "value": "爱辣"}'),
        ]),
        "END",
    ]
    agent = make_agent(script)
    for turn in range(30):  # 30 轮，历史必然超过 HISTORY_TURNS=12，裁剪反复触发
        answer = agent.run(f"第{turn}轮：3人聚餐人均40元")
        assert isinstance(answer, str) and answer, f"第{turn}轮没有返回文本"
        validate(agent.memory.messages, f"[run 第{turn}轮结束后的原始记忆]")
    print(f"Part 2 完整ReAct循环压力测试（30轮，假LLM共响应{agent.llm.seen}次）: 通过")


def part3_max_steps_poison():
    """Part 3：MAX_STEPS 用尽（循环不带最终回答就退出）后继续对话——
    这是历史上最容易产生悬空消息的场景。"""
    script = [
        AssistantMessage(tool_calls=[ToolCall(id="m1", name="food_search", arguments="{}")]),
    ] * 20  # 永远调用工具，逼 run() 走到 MAX_STEPS=8 退出
    agent = make_agent(script)
    agent.config_max_steps = 8
    for turn in range(10):
        answer = agent.run(f"第{turn}轮问题")
        assert "最大推理步数" in answer or answer, "MAX_STEPS 场景应有兜底返回"
        validate(agent.memory.messages, f"[MAX_STEPS 第{turn}轮后的记忆]")
    print("Part 3 MAX_STEPS 用尽后继续对话: 通过")


if __name__ == "__main__":
    part1_memory_stress()
    part2_full_loop()
    part3_max_steps_poison()
    print("=== 全部压力测试通过：发给 LLM 的消息序列永远合法 ===")
