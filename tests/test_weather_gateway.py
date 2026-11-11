"""天气网关回归测试：决策翻译、缓存、降级、密钥卫生。

全程 mock 网络层（替换 weather_gateway._do_http），不消耗任何真实百度额度。
天气是增强项：所有失败路径必须优雅降级，不影响推荐主流程。

跑法：python tests/test_weather_gateway.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

# 模拟"已配置 AK"的环境（天气复用地图 AK）
config.MAP_ENABLED = True
config.BAIDU_MAP_AK = config.BAIDU_MAP_AK or "test-ak"
config.WEATHER_ENABLED = True

from agent.core import FoodAdvisorAgent  # noqa: E402
from agent.memory import ConversationMemory, UserProfile  # noqa: E402
from tools import build_registry, weather_gateway as wgw  # noqa: E402

PASS = 0


def ok(label: str) -> None:
    global PASS
    PASS += 1
    print(f"[通过] {label}")


class FakeTransport:
    """可编排的假天气接口：记录调用、按脚本返回 (状态, 原始响应)。"""

    def __init__(self, handler):
        self.handler = handler
        self.calls: list[dict] = []
        self.original = None

    def __enter__(self):
        self.original = wgw._do_http
        def wrapper(params):
            self.calls.append(dict(params))
            return self.handler(params)
        wgw._do_http = wrapper
        wgw.reset_for_tests()
        return self

    def __exit__(self, *exc):
        wgw._do_http = self.original
        wgw.reset_for_tests()


def baidu_ok(now_text="小雨", temp=26, today_day="中雨", name="番禺"):
    """百度天气 API 的标准成功响应结构。"""
    return {
        "status": 0,
        "message": "success",
        "result": {
            "location": {"country": "中国", "province": "广东省", "city": "广州市", "name": name, "id": "440113"},
            "now": {"text": now_text, "temp": temp, "wind_class": "3级", "wind_dir": "东北风", "humidity": 90},
            "forecast": [
                {"date": "2026-11-11", "text_day": today_day, "text_night": "阴", "high": 28, "low": 21},
            ],
        },
    }


def make_agent() -> FoodAdvisorAgent:
    agent = FoodAdvisorAgent.__new__(FoodAdvisorAgent)
    agent.profile = UserProfile()
    agent.poi_store = {}
    agent.functions, agent.schemas = build_registry(agent.profile, agent.poi_store)
    agent.memory = ConversationMemory("")
    agent._trace_fn = lambda text: None
    agent.last_cards = None
    agent.last_summary = ""
    agent.last_dropped = []
    agent.origin_key = config.DEFAULT_ORIGIN
    agent.origin = config.origin_by_key(agent.origin_key)
    agent._needed_modes = ["walking"]
    return agent


# ---------- 1. 正常返回 + 雨天决策翻译 ----------

def test_rain_decision():
    with FakeTransport(lambda params: (wgw.OK, baidu_ok(now_text="小雨", temp=26, today_day="中雨"))):
        data = wgw.fetch_weather()
    assert data["状态"] == wgw.OK
    assert data["位置"] == "番禺"
    assert data["现在"]["现象"] == "小雨" and data["现在"]["气温"] == 26
    sugg = data["建议"]
    assert sugg["场景"] == "rain", f"雨雪天气应翻译为 rain 场景: {sugg}"
    assert sugg["外卖优先"] is True
    assert sugg["步行宽容度"] == 10
    assert "雨" in sugg["提示文案"]
    ok("1 雨天翻译：小雨 → 外卖优先 + 步行宽容度 10（后端确定性规则）")


# ---------- 2. 高温 / 正常天气分支 ----------

def test_hot_and_normal():
    with FakeTransport(lambda params: (wgw.OK, baidu_ok(now_text="晴", temp=35, today_day="晴"))):
        hot = wgw.fetch_weather()["建议"]
    assert hot["场景"] == "hot" and hot["外卖优先"] is False and hot["步行宽容度"] == 15
    ok("2a 高温 35°C → hot 场景，步行宽容度放宽到 15")

    with FakeTransport(lambda params: (wgw.OK, baidu_ok(now_text="多云", temp=22, today_day="多云"))):
        normal = wgw.fetch_weather()["建议"]
    assert normal["场景"] == "normal" and normal["步行宽容度"] is None
    ok("2b 22°C 多云 → normal 场景，不注入特殊策略")


# ---------- 3. 缓存：第二次调用不发生网络请求 ----------

def test_cache():
    with FakeTransport(lambda params: (wgw.OK, baidu_ok())) as t:
        first = wgw.fetch_weather()
        assert first["缓存命中"] is False
        second = wgw.fetch_weather()
    assert len(t.calls) == 1, f"30 分钟缓存内两次调用应只请求 1 次，实际 {len(t.calls)} 次"
    assert second["缓存命中"] is True
    assert second["现在"]["现象"] == first["现在"]["现象"]
    ok("3 全站共享缓存：同 district 30 分钟内只打一次接口")


# ---------- 4. 失败降级：超时/接口故障/无 AK ----------

def test_failure_degradation():
    with FakeTransport(lambda params: (wgw.TIMEOUT, {})) as t:
        data = wgw.fetch_weather()
    assert data["状态"] == wgw.TIMEOUT
    assert "建议" not in data, "失败结果不得带任何天气建议，防止上层误用"
    ok("4a 接口超时：返回 timeout 状态，不抛异常、不给出天气建议")

    with FakeTransport(lambda params: (wgw.AUTH_ERROR, {})):
        data2 = wgw.fetch_weather()
    assert data2["状态"] == wgw.AUTH_ERROR
    ok("4b AK 权限异常：如实返回 auth_error")

    old = config.WEATHER_ENABLED
    config.WEATHER_ENABLED = False
    try:
        with FakeTransport(lambda params: (wgw.OK, baidu_ok())) as t2:
            data3 = wgw.fetch_weather()
        assert data3["状态"] == wgw.AUTH_ERROR and len(t2.calls) == 0, "关闭天气时不得发出任何请求"
        ok("4c WEATHER_ENABLED=False：不请求、直接降级（无 AK 环境静默可用）")
    finally:
        config.WEATHER_ENABLED = old


# ---------- 5. 字段缺失：不猜，按 SERVER_ERROR 处理 ----------

def test_missing_fields():
    with FakeTransport(lambda params: (wgw.OK, {"status": 0, "result": {"location": {"name": "番禺"}}})):
        data = wgw.fetch_weather()
    assert data["状态"] == wgw.SERVER_ERROR, "缺 now 字段应报 SERVER_ERROR 而不是猜默认值"
    ok("5 返回结构异常：不编造天气字段，按 SERVER_ERROR 降级")


# ---------- 6. core.py 注入：雨天段落出现，关闭时消失 ----------

def test_core_injection():
    with FakeTransport(lambda params: (wgw.OK, baidu_ok(now_text="小雨", temp=24, today_day="中雨"))):
        agent = make_agent()
        block = agent._weather_context()
        # 注意：_system_prompt 内部会再取一次天气，必须留在 mock 范围内
        # （FakeTransport 退出时会 reset_for_tests 清空缓存）
        prompt = agent._system_prompt()
    assert "【当前天气" in prompt and "雨天执行规则" in prompt
    ok("6a 雨天时 system prompt 注入天气段落与执行规则")

    old = config.WEATHER_ENABLED
    config.WEATHER_ENABLED = False
    try:
        agent2 = make_agent()
        block2 = agent2._weather_context()
        prompt2 = agent2._system_prompt()
        # 不是"什么都不说"：实测会编造天气，所以必须给出"查不到"的约束
        assert "无法获取实时天气" in block2 or "天气数据不可用" in block2, block2
        assert "【当前天气（实时" not in prompt2, "不得注入具体天气数据"
        assert "【天气数据不可用】" in prompt2, "应注入防编造约束"
        ok("6b 天气关闭时注入防编造约束，而非放任模型瞎猜天气")
    finally:
        config.WEATHER_ENABLED = old


# ---------- 7. 工具注册：get_weather 可被 LLM 调用 ----------

def test_tool_registered():
    agent = make_agent()
    assert "get_weather" in agent.functions, "get_weather 必须注册进工具表"
    names = [s["function"]["name"] for s in agent.schemas]
    assert "get_weather" in names, "get_weather 必须有 function schema"
    desc = next(s for s in agent.schemas if s["function"]["name"] == "get_weather")["function"]["description"]
    assert "禁止编造天气" in desc
    with FakeTransport(lambda params: (wgw.OK, baidu_ok())) as t:
        result = json.loads(agent.functions["get_weather"]())
    assert result["状态"] == wgw.OK and len(t.calls) == 1
    ok("7 get_weather 已注册 + schema 描述含路由纪律 + 工具可执行")


# ---------- 8. 密钥卫生：日志与统计不含 AK ----------

def test_secret_hygiene():
    masked = wgw._redacted({"district_id": "440113", "ak": "secret-ak"})
    assert masked["ak"] == "***" and masked["district_id"] == "440113"
    stats = wgw.weather_stats()
    assert not any("key" in k.lower() or k.lower() == "ak" for k in stats)
    ok("8 日志 AK 打码、统计接口不暴露密钥")


if __name__ == "__main__":
    test_rain_decision()
    test_hot_and_normal()
    test_cache()
    test_failure_degradation()
    test_missing_fields()
    test_core_injection()
    test_tool_registered()
    test_secret_hygiene()
    print(f"=== 全部 {PASS} 项天气网关测试通过 ===")
