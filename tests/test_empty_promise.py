"""空话纠正测试：模型只回复「请稍等 / 我马上去查」却不调工具时，
必须被强制要求执行，多次不听话则用规则检索兜底——用户绝不能空手而归。

跑法：python tests/test_empty_promise.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from agent.core import FoodAdvisorAgent, _trace
from agent.memory import ConversationMemory, UserProfile
from llm.client import AssistantMessage, ToolCall
from tools import build_registry

PASSED = 0
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASSED
    if ok:
        PASSED += 1
        print(f"  ok  {name}")
    else:
        FAILED.append(name)
        print(f"  FAIL {name}  {detail}")


class FakeLLM:
    """按脚本回放的假 LLM；记录每次收到的消息，便于断言"是否发生了纠正"。"""

    def __init__(self, script):
        self.script = script
        self.i = 0
        self.enabled = True
        self.payloads: list[list] = []

    def chat(self, messages, tools=None):
        self.payloads.append([dict(m) for m in messages])
        msg = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        if isinstance(msg, str):
            return AssistantMessage(content=msg)
        return msg


def make_agent(script) -> FoodAdvisorAgent:
    agent = FoodAdvisorAgent.__new__(FoodAdvisorAgent)
    agent.llm = FakeLLM(script)
    agent.profile = UserProfile()
    agent.poi_store = {}
    agent.functions, agent.schemas = build_registry(agent.profile, agent.poi_store)
    agent.memory = ConversationMemory("")
    agent._trace_fn = _trace
    agent.last_cards = None
    agent.last_summary = ""
    agent.last_dropped = []
    agent.origin_key = config.DEFAULT_ORIGIN
    agent.origin = config.origin_by_key(agent.origin_key)
    agent._needed_modes = ["walking"]
    return agent


QUESTION = "想吃辣的，预算30以内，不想走超过15分钟"


def test_promise_detection() -> None:
    print("[1] 空话识别")
    agent = make_agent(["x"])

    check("『请稍等』被识别", agent._is_empty_promise("请稍等，我马上为你查询。", 0))
    check("『我这就去查』被识别", agent._is_empty_promise("好的，我这就去查一下附近有什么。", 0))
    check("正常回答不被误判", not agent._is_empty_promise(
        "推荐 cloud 意面，人均 40 元，步行 25 分钟。", 0))
    check("调过工具后不判定为空话", not agent._is_empty_promise("请稍等", 1))
    check("长回复不判定（多半是在解释）", not agent._is_empty_promise("请稍等。" * 60, 0))


def test_nudge_then_execute() -> None:
    print("[2] 首次只说空话 → 纠正后真的执行并给出结果")
    # 联网层 mock，避免测试消耗真实地图额度
    import json as _json
    from tools import route_planner as rp

    original_search = rp.search_poi
    rp.search_poi = lambda keyword, poi_store=None, radius=None: _json.dumps(
        {
            "状态": "ok",
            "结果": [{
                "uid": "mock1", "名称": "测试川菜馆", "地址": "测试路1号",
                "坐标": "23.0,113.4", "直线距离_米": 700, "查询品类": keyword,
            }],
        },
        ensure_ascii=False,
    )
    try:
        script = [
            "好的，根据你的需求，我帮你查一下附近有哪些辣的餐厅。请稍等，我马上为你查询。",
            AssistantMessage(tool_calls=[ToolCall(id="s1", name="food_search",
                                                  arguments='{"query": "辣", "taste": "辣", "budget_max": 30}')]),
            "推荐 cloud 意面 等几家，人均 40 元以内的可选。",
        ]
        agent = make_agent(script)
        answer = agent.run(QUESTION)

        check("最终有回复", bool(answer))
        check("没有把空话当答案返回", "请稍等" not in answer, answer[:60])
        check("发生了第二次模型调用（被纠正）", agent.llm.i >= 2, str(agent.llm.i))
        nudged = any(
            any(isinstance(m.get("content"), str) and "系统提醒" in m["content"] for m in payload)
            for payload in agent.llm.payloads[1:2]
        )
        check("纠正消息已注入", nudged)
    finally:
        rp.search_poi = original_search


def test_repeat_promise_fallback() -> None:
    print("[3] 连续两次只说空话 → 规则检索兜底，用户仍能拿到知识库结果")
    script = [
        "请稍等，我马上去查。",
        "马上就好，正在为你查询附近餐厅。",
        "请稍等片刻。",
    ]
    agent = make_agent(script)
    answer = agent.run(QUESTION)

    check("有回复且不是空话", bool(answer) and "请稍等" not in answer, answer[:80])
    check("走了兜底（标注知识库检索结果）", "知识库检索结果" in answer, answer[:80])
    check("兜底内容来自真实知识库", ("｜" in answer) or ("没有" in answer), answer[:120])


def test_max_steps_message() -> None:
    print("[4] 工具无限调用时的退出提示正常")
    script = [AssistantMessage(tool_calls=[ToolCall(id=f"t{i}", name="food_search", arguments="{}")])
              for i in range(20)]
    agent = make_agent(script)
    answer = agent.run(QUESTION)
    check("返回了可读提示", isinstance(answer, str) and bool(answer), answer[:60])


def test_text_walk_guard() -> None:
    print("[5] 纯文字回复提到的超限门店 → 自动补更正说明")
    agent = make_agent(["x"])
    agent.poi_store = {
        "u1": {"名称": "近店", "路线": {"起点": agent.origin_key, "步行": {"分钟": 8}}},
        "u2": {"名称": "远店", "路线": {"起点": agent.origin_key, "步行": {"分钟": 23}}},
    }
    reply = "推荐近店（步行8分钟）和远店（步行23分钟），都在15分钟内。"
    guarded = agent._guard_text_walk_limit(reply, "不想走超过15分钟")
    check("超限门店被点名更正", "远店" in guarded.split("\n")[0] and "23" in guarded.split("\n")[0], guarded[:90])
    check("达标门店不被误伤", "近店（步行" not in guarded.split("\n")[0], guarded[:80])

    no_minutes = agent._guard_text_walk_limit(reply, "想吃点好的")
    check("用户没提时间限制时不干预", no_minutes == reply)


def test_clarify_when_conditions_given() -> None:
    print("[6] 用户已给条件却反问 → 强制检索")
    script = [
        "你想吃川菜、湘菜还是辣味小吃呢？",
        "好的，我查到了几家适合的店。",
    ]
    agent = make_agent(script)
    answer = agent.run(QUESTION)
    check("反问被拦截，最终有实质回答", "川菜、湘菜还是" not in answer, answer[:80])
    check("触发了第二轮调用（被纠正）", agent.llm.i >= 2, str(agent.llm.i))
    nudged = any(
        any(isinstance(m.get("content"), str) and "不要再反问" in m["content"] for m in payload)
        for payload in agent.llm.payloads[1:2]
    )
    check("clarify 纠正消息已注入", nudged)


def test_no_nudge_for_open_question() -> None:
    print("[7] 用户没给条件时的正常追问 → 不干预")
    agent = make_agent(["x"])
    reason = agent._nudge_reason("想吃点什么口味的呢？辣的还是不辣？", 0, "帮我推荐点吃的")
    check("无约束的追问不触发纠正", reason == "", reason)


def test_card_missing_nudge() -> None:
    print("[8] 调过检索、正文列了店却没出卡片 → 强制补 format_recommendation")
    script = [
        "为你找到几家：测试川菜馆就不错，辣度和预算都符合。",
        "好的，已经生成推荐卡片了。",
    ]
    agent = make_agent(script)
    agent.poi_store["mock1"] = {"名称": "测试川菜馆", "地址": "测试路1号"}
    agent._used_food_search = True
    answer = agent._react_loop(QUESTION)
    check("卡片纠正消息已注入", any(
        any(isinstance(m.get("content"), str)
            and "format_recommendation" in m["content"]
            and "结构化卡片" in m["content"] for m in payload)
        for payload in agent.llm.payloads[1:]
    ))
    check("纠正后模型被再次调用", agent.llm.i >= 2, str(agent.llm.i))
    check("没有把纯文字列店当成最终答案直接放行", answer == "好的，已经生成推荐卡片了。", answer[:60])


def test_card_nudge_no_false_positive() -> None:
    print("[9] 不该触发卡片纠正的情况 → 不误伤")
    # a. 如实回答无匹配（没点名任何检索到的门店）
    agent = make_agent(["3公里内没有找到符合的门店，建议放宽预算后再试。"])
    agent._used_food_search = True
    agent.poi_store["mock1"] = {"名称": "测试川菜馆"}
    agent._react_loop(QUESTION)
    check("无匹配的如实回答不触发纠正", len(agent.llm.payloads) == 1)

    # b. 已经调用过 format_recommendation（全部被剔除后如实说明）
    agent2 = make_agent(["这几家都超出了步行范围，暂时没有完全符合的推荐。"])
    agent2._used_food_search = True
    agent2._format_attempted = True
    agent2.poi_store["mock1"] = {"名称": "测试川菜馆"}
    agent2._react_loop(QUESTION)
    check("已尝试出卡片后不再催", len(agent2.llm.payloads) == 1)

    # c. 没用过 food_search（门店咨询等场景）→ 不催
    agent3 = make_agent(["测试川菜馆就在东门附近，人均30左右，值得一试。"])
    agent3.poi_store["mock1"] = {"名称": "测试川菜馆"}
    agent3._react_loop("你知道测试川菜馆吗")
    check("非检索场景不催卡片", len(agent3.llm.payloads) == 1)


if __name__ == "__main__":
    config.KB_DEMO = False
    test_promise_detection()
    test_nudge_then_execute()
    test_repeat_promise_fallback()
    test_max_steps_message()
    test_text_walk_guard()
    test_clarify_when_conditions_given()
    test_no_nudge_for_open_question()
    test_card_missing_nudge()
    test_card_nudge_no_false_positive()
    print(f"\n通过 {PASSED} 项" + ("" if not FAILED else f"，失败 {len(FAILED)} 项：{FAILED}"))
    raise SystemExit(0 if not FAILED else 1)
