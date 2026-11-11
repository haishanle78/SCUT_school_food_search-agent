"""地图治理与门店真实性的回归测试（对应修改任务书的 10 条验收标准）。

全程 mock 网络层（替换 map_gateway._do_http），不消耗任何真实百度额度。

跑法：python tests/test_map_gateway.py
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

config.MAP_ENABLED = True
config.BAIDU_MAP_AK = config.BAIDU_MAP_AK or "test-ak"
# 天气模块走真实网络，单测一律关闭（本文件只测地图治理）
config.WEATHER_ENABLED = False

from agent.core import FoodAdvisorAgent  # noqa: E402
from agent.memory import ConversationMemory, UserProfile  # noqa: E402
from tools import build_registry, map_gateway as gw  # noqa: E402
import tools.route_planner as rp  # noqa: E402

PASS = 0


def ok(label: str) -> None:
    global PASS
    PASS += 1
    print(f"[通过] {label}")


class FakeTransport:
    """可编排的假百度：记录调用、按脚本返回 MapResult。"""

    def __init__(self, handler):
        self.handler = handler
        self.calls: list[tuple[str, dict]] = []
        self.original = None

    def __enter__(self):
        self.original = gw._do_http
        def wrapper(path, params):
            self.calls.append((path, dict(params)))
            return self.handler(path, params)
        gw._do_http = wrapper
        gw.reset_for_tests()
        return self

    def __exit__(self, *exc):
        gw._do_http = self.original
        gw.reset_for_tests()


def poi(uid, name, lat=23.0180, lng=113.4120, tag="美食;中餐厅", address="番禺区测试路1号", area="番禺区"):
    return {
        "uid": uid,
        "name": name,
        "address": address,
        "province": "广东省",
        "city": "广州市",
        "area": area,
        "location": {"lat": lat, "lng": lng},
        "detail_info": {"tag": tag},
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


# ---------- 验收1/2：指定店名不擅自替换 ----------

def test_specified_store_not_substituted():
    def handler(path, params):
        query = params.get("query", "")
        if "停车场" in query or "B记" in query:
            return gw.MapResult(gw.OK, payload={"status": 0, "results": [
                poi("p1", "B记鱼粥-门前停车场", tag="停车场;停车位", address="荣华里-5号楼"),
            ]})
        # 只有一家"相似但不同"的粥店
        return gw.MapResult(gw.OK, payload={"status": 0, "results": [
            poi("s1", "恒源客家佬粥粉面", lat=23.0190, lng=113.4130),
        ]})

    with FakeTransport(handler):
        data = json.loads(rp.find_restaurant("B记鱼粥", poi_store={}))
    assert data["状态"] == "auxiliary_only", f"停车场不应被当作门店: {data}"
    assert "停车场" in data["提示"] and "不能作为门店" in data["提示"]
    assert data["uid" if "uid" in data else "店名"] == "B记鱼粥"
    ok("验收1/2 只返回『B记鱼粥-门前停车场』时：不生成门店，如实说明并要地标")

    # 名称相近但不同：只能给候选，不能当答案
    def handler2(path, params):
        return gw.MapResult(gw.OK, payload={"status": 0, "results": [
            poi("s1", "恒源客家佬粥粉面"),
        ]})

    with FakeTransport(handler2):
        data2 = json.loads(rp.find_restaurant("B记鱼粥", poi_store={}))
    assert data2["状态"] in ("uncertain", "no_result"), f"相似店不得直接认定为同一家: {data2}"
    assert data2["状态"] != "match", "相似店不能被判定为确认匹配"
    assert "不确定" in data2.get("提示", "") or "不能确认" in data2.get("提示", "")
    assert "B记鱼粥" in json.dumps(data2, ensure_ascii=False)
    ok("验收1 指定店名查不到时：不替换成恒源客家佬粥粉面，候选人需用户确认")


# ---------- 验收3：限流/配额要如实说明，不能说门店不存在 ----------

def test_rate_limit_and_quota_messages():
    with FakeTransport(lambda p, q: gw.MapResult(gw.RATE_LIMITED)) as t:
        data = json.loads(rp.search_poi("粤菜", poi_store={}))
    assert data["状态"] == gw.RATE_LIMITED
    assert "不代表附近没有店" in data["提示"] or "不代表" in data["提示"]
    assert "没有匹配" not in data["提示"]
    ok(f"验收3a 限流：明确说暂时查不了（重试 {len(t.calls)} 次），不说门店不存在")

    def quota_handler(path, params):
        return gw.MapResult(gw.QUOTA_EXHAUSTED, payload={"status": 302}, message="天配额超限")

    with FakeTransport(quota_handler) as t2:
        data2 = json.loads(rp.search_poi("粤菜", poi_store={}))
        snap = gw.snapshot()
    assert data2["状态"] == gw.QUOTA_EXHAUSTED and "额度" in data2["提示"]
    assert len(t2.calls) == 1, f"配额耗尽不应重试，实际 {len(t2.calls)} 次"
    assert snap["breaker_seconds_left"] > 0, "配额异常后应熔断，避免继续白烧请求"
    ok("验收3b 配额耗尽：一次即停 + 熔断，不再重复请求")


# ---------- 验收4/6：缓存复用 + 只查需要的交通方式 ----------

def test_cache_and_modes():
    def handler(path, params):
        if path.startswith("/direction/v2/walking"):
            return gw.MapResult(gw.OK, payload={"status": 0, "result": {"routes": [{"duration": 720, "distance": 900}]}})
        if path.startswith("/direction/v2/riding"):
            return gw.MapResult(gw.OK, payload={"status": 0, "result": {"routes": [{"duration": 300, "distance": 950}]}})
        return gw.MapResult(gw.OK, payload={"status": 0, "result": {"routes": [{"duration": 240, "distance": 1200}]}})

    with FakeTransport(handler) as t:
        store = {"u1": {"uid": "u1", "名称": "测试粥店", "地址": "番禺", "行政区": "番禺区",
                        "坐标": "23.018,113.412", "直线距离_米": 800, "是餐饮": True, "在范围内": True}}
        first = json.loads(rp.verify_route("u1", poi_store=store))
        assert first["路线耗时"]["步行"]["分钟"] == 12
        assert len(t.calls) == 1, f"默认只应查步行，实际 {len(t.calls)} 次"
        ok("验收6 默认只查步行，不自动额外查骑行/驾车")

        second = json.loads(rp.verify_route("u1", poi_store=store))
        assert len(t.calls) == 1, f"同一起终点应命中缓存，实际又请求了 {len(t.calls)} 次"
        assert "缓存命中" in second["路线耗时"]["来源"] or "复用" in second["路线耗时"]["来源"]
        assert second["路线耗时"]["查询时间"] == first["路线耗时"]["查询时间"], "复用结果不得把旧时间刷成现在"
        ok("验收4 重复查询同一起终点：缓存命中，不再请求百度")

        # 已核实过的方式直接复用；换方式才新增一次请求
        rp.verify_route("u1", poi_store=store, mode="riding")
        assert len(t.calls) == 2, f"切到骑行应只新增 1 次，实际 {len(t.calls)} 次"
        ok("验收4b 按方式细化缓存：换交通方式才新增请求")


# ---------- 验收5：并发同一请求只打一次百度 ----------

def test_single_flight():
    def slow_handler(path, params):
        time.sleep(0.25)
        return gw.MapResult(gw.OK, payload={"status": 0, "results": [poi("x1", "并发测试粥店")]})

    with FakeTransport(slow_handler) as t:
        results = []
        barrier = threading.Barrier(4)

        def worker():
            barrier.wait()
            results.append(json.loads(rp.search_poi("粥", poi_store={})))

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

    place_calls = [c for c in t.calls if c[0] == "/place/v2/search"]
    assert len(place_calls) == 1, f"4 个并发相同请求应合并为 1 次，实际 {len(place_calls)} 次"
    assert all(r.get("结果") for r in results)
    ok("验收5 并发相同请求：single-flight 合并，只调用一次百度")


# ---------- 验收7：卡片硬校验（范围/路线/餐饮） ----------

def test_card_validation():
    agent = make_agent()
    agent.poi_store.update({
        "ok1": {"uid": "ok1", "名称": "合规粥店", "地址": "番禺", "行政区": "番禺区", "坐标": "23.018,113.412",
                "直线距离_米": 900, "是餐饮": True, "在范围内": True,
                "路线": {"步行": {"分钟": 12, "距离_米": 900}, "查询时间": "t", "来源": "测试", "起点": agent.origin_key}},
        "far1": {"uid": "far1", "名称": "超范围店", "地址": "荔湾", "行政区": "荔湾区", "坐标": "23.13,113.24",
                 "直线距离_米": 25000, "是餐饮": True, "在范围内": False,
                 "路线": {"步行": {"分钟": 300}, "起点": agent.origin_key}},
        "park1": {"uid": "park1", "名称": "某店-门前停车场", "地址": "番禺", "行政区": "番禺区", "坐标": "23.018,113.412",
                  "直线距离_米": 500, "是餐饮": False, "在范围内": True,
                  "路线": {"步行": {"分钟": 5}, "起点": agent.origin_key}},
        "noroute": {"uid": "noroute", "名称": "未核实路线店", "地址": "番禺", "行政区": "番禺区", "坐标": "23.018,113.412",
                    "直线距离_米": 700, "是餐饮": True, "在范围内": True},
    })
    cards = agent._inject_card_facts([
        {"uid": "ok1", "cuisine": "粥", "reason": "近"},
        {"uid": "far1", "cuisine": "粤菜", "reason": "跨区"},
        {"uid": "park1", "cuisine": "粥", "reason": "停车场"},
        {"uid": "noroute", "cuisine": "粥", "reason": "没路线"},
        {"uid": "ghost", "cuisine": "粥", "reason": "编造"},
    ])
    assert [c["uid"] for c in cards] == ["ok1"], f"只有合规卡片能通过: {[c['uid'] for c in cards]}"
    reasons = " ".join(d["原因"] for d in agent.last_dropped)
    assert "超出" in reasons and "附属设施" in reasons and "路线未核实" in reasons and "未核实" in reasons
    ok("验收7 超范围/附属设施/路线未核实/uid 编造：全部被后端拦截")

    card = cards[0]
    assert card["identity_verified"] and card["route_verified"]
    assert card["price_per_person"] is None and card["price_source"] == "待确认", "无来源人均必须标待确认"
    ok("验收7b 未知人均/营业时间标『待确认』，不编造")


# ---------- 验收8：官方路线链接（起终点带坐标+名称） ----------

def test_route_link():
    link = rp.build_route_link("测试粥店", "23.018,113.412")
    assert "origin=latlng:23.017973,113.411971|name:" in link
    assert "destination=latlng:23.018,113.412|name:" in link
    assert "coord_type=bd09ll" in link and "mode=walking" in link
    from urllib.parse import unquote
    assert "测试粥店" in unquote(link)
    ok("验收8 路线链接按官方 URI 规范：坐标+名称+坐标系，手机可直接导航")


# ---------- 验收9：密钥不进入前端/日志 ----------

def test_secret_hygiene():
    html = Path("web/static/index.html").read_text(encoding="utf-8")
    for bad in ("BAIDU_MAP_AK", "OPENAI_API_KEY", "DEEPSEEK_API_KEY", "glm-4-flash\n"):
        assert bad not in html, f"前端不应出现 {bad}"
    snapshot = gw.snapshot()
    assert not any("ak" == k.lower() or "key" in k.lower() for k in snapshot), "状态接口不得暴露密钥"
    assert "***" in gw._redacted("/place/v2/search", {"ak": "secret", "query": "粥"})
    ok("验收9 前端/状态接口/日志均不含密钥（AK 打码）")


# ---------- 验收10 相关：全局限速跨会话 ----------

def test_global_rate_limit():
    old_qps, old_wait = config.MAP_QPS, config.MAP_MAX_WAIT
    config.MAP_QPS, config.MAP_MAX_WAIT = 1.0, 0.05
    try:
        def handler(path, params):
            return gw.MapResult(gw.OK, payload={"status": 0, "results": []})

        with FakeTransport(handler):
            statuses = []
            for i in range(5):
                res = gw.request("/place/v2/search", {"query": f"q{i}"}, kind="search", ttl=0)
                statuses.append(res.status)
            snap = gw.snapshot()
        assert statuses.count(gw.RATE_LIMITED) >= 1, f"突发请求应被全局限速拦住: {statuses}"
        assert snap["rate_limited"] >= 1
        ok("验收5b 全局限速：突发请求被令牌桶拦截（跨会话共享）")
    finally:
        config.MAP_QPS, config.MAP_MAX_WAIT = old_qps, old_wait


if __name__ == "__main__":
    test_specified_store_not_substituted()
    test_rate_limit_and_quota_messages()
    test_cache_and_modes()
    test_single_flight()
    test_card_validation()
    test_route_link()
    test_secret_hygiene()
    test_global_rate_limit()
    print(f"=== 全部 {PASS} 项地图治理测试通过 ===")
