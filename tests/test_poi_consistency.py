"""POI 一致性回归测试：门店身份、地理范围、路线事实的铁律验证。

覆盖场景（对应真实线上事故的防御）：
  1. 跨区远店（如荔湾区的"广味居"）混不进 3 公里推荐；
  2. 步行 15 分钟上限过滤：超限店降级为替代方案，全部超限时如实报告无匹配；
  3. 同名不同分店：uid 区分，路线不张冠李戴；
  4. 地图无结果 / 接口失败：返回"暂无法确认"，不编造耗时，不退回示例数据；
  5. 卡片事实由后端注入：模型编写的地址/耗时被覆盖，uid 无效的卡片被剔除；
  6. 官方路线链接格式（起终点坐标|名称，coord_type=bd09ll）。

跑法：python tests/test_poi_consistency.py （全程 mock，不消耗 API 配额）
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

# 测试全程 mock 网络，但代码里用 MAP_ENABLED 做前置开关。
# 这里强制打开并给一个假 AK，保证没有 .env / 没有真实 AK 的队友也能直接跑测试。
config.MAP_ENABLED = True
config.BAIDU_MAP_AK = config.BAIDU_MAP_AK or "test-ak"

import tools.route_planner as rp
from agent.core import FoodAdvisorAgent
from agent.memory import ConversationMemory, UserProfile
from tools import build_registry

PASS = 0


def ok(label):
    global PASS
    PASS += 1
    print(f"[通过] {label}")


# ---------- 公共 fixture ----------

ORIGIN_LAT, ORIGIN_LNG = 23.017973, 113.411971


def make_poi(uid, name, address, area, lat, lng):
    """构造标准化 POI（模拟百度返回）。"""
    return {
        "uid": uid,
        "名称": name,
        "地址": address,
        "行政区": area,
        "坐标": f"{lat},{lng}",
        "直线距离_米": rp._haversine_m(ORIGIN_LAT, ORIGIN_LNG, lat, lng),
    }


def make_agent():
    """构造可测的 agent（跳过 LLM 初始化，直测后端逻辑）。"""
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


# ---------- 场景1：跨区远店混不进 3 公里 ----------

def test_far_store_rejected():
    near = {"uid": "near-1", "name": "番禺粤菜馆", "location": {"lat": 23.018, "lng": 113.412}}
    far = {"uid": "far-1", "name": "广味居", "location": {"lat": 23.13, "lng": 113.24}}  # 荔湾区，约 25km
    original = rp._poi_query
    rp._poi_query = lambda keyword: [far, near]
    try:
        store = {}
        result = json.loads(rp.search_poi("粤菜", poi_store=store))
    finally:
        rp._poi_query = original

    names = [x["名称"] for x in result.get("结果") or []]
    assert "广味居" not in names, f"跨区远店混入推荐: {names}"
    assert "番禺粤菜馆" in names
    rejected = [x["名称"] for x in (result.get("已剔除") or result.get("已剔除超范围") or [])]
    assert "广味居" in rejected, "远店应出现在已剔除列表"
    ok("场景1 跨区远店（荔湾区）被 3 公里硬限制剔除")


# ---------- 场景2：步行 15 分钟上限过滤 ----------

def test_walk_limit_filter():
    agent = make_agent()
    # 两家店：一家步行 10 分钟（达标），一家 40 分钟（超限）
    agent.poi_store["uid-ok"] = dict(make_poi("uid-ok", "近的店", "近的地址", "番禺", 23.018, 113.412),
                                      路线={"步行": {"分钟": 10, "距离_米": 700}, "骑行": {"分钟": 4, "距离_米": 720},
                                           "查询时间": "2026-11-10 13:00", "来源": "测试"})
    agent.poi_store["uid-far"] = dict(make_poi("uid-far", "远的店", "远的地址", "番禺", 23.04, 113.44),
                                      路线={"步行": {"分钟": 40, "距离_米": 2800}, "骑行": {"分钟": 14, "距离_米": 2900},
                                           "查询时间": "2026-11-10 13:00", "来源": "测试"})
    agent.last_cards = agent._inject_card_facts([
        {"uid": "uid-ok", "cuisine": "粤菜", "reason": "近"},
        {"uid": "uid-far", "cuisine": "粤菜", "reason": "远"},
    ])
    reply = agent._filter_cards_by_walk_limit("模型说都符合", "想吃粤菜，步行15分钟以内")
    assert "远的店" in reply and "超过" in reply, "超限店未在文案中标注"
    assert agent.last_cards[0]["name"] == "近的店", "达标店应排在前面"
    assert agent.last_cards[1]["exceeds_walk_limit"] is True
    ok("场景2a 混合情况：达标在前，超限标注为替代")

    # 全部超限：如实报告无匹配，不强凑
    agent2 = make_agent()
    agent2.poi_store["uid-far"] = dict(make_poi("uid-far", "远店A", "地址A", "番禺", 23.04, 113.44),
                                       路线={"步行": {"分钟": 40, "距离_米": 2800}, "查询时间": "t", "来源": "测试"})
    agent2.last_cards = agent2._inject_card_facts([{"uid": "uid-far", "cuisine": "粤菜", "reason": "r"}])
    reply2 = agent2._filter_cards_by_walk_limit("模型硬说符合", "步行15分钟")
    assert "没有完全匹配" in reply2, "全部超限时应如实说明无匹配"
    assert "模型硬说符合" not in reply2, "模型误导文案应被替换"
    ok("场景2b 全部超限：如实报告无匹配，替换误导文案")


# ---------- 场景3：同名不同分店，uid 区分 ----------

def test_same_name_different_uid():
    store = {
        "uid-A": dict(make_poi("uid-A", "隆江猪脚饭(南村店)", "南村地址", "番禺", 23.015, 113.410)),
        "uid-B": dict(make_poi("uid-B", "隆江猪脚饭(市桥店)", "市桥地址", "番禺", 23.03, 113.38)),
    }
    original_dir = rp._baidu_direction
    calls = []
    def fake_direction(mode, dest_coord):
        calls.append(dest_coord)
        return {"分钟": 12 if "23.015" in dest_coord else 25, "距离_米": 800 if "23.015" in dest_coord else 1800}
    rp._baidu_direction = fake_direction
    try:
        ra = json.loads(rp.verify_route("uid-A", poi_store=store, mode="all"))
        rb = json.loads(rp.verify_route("uid-B", poi_store=store, mode="all"))
    finally:
        rp._baidu_direction = original_dir

    assert ra["目的地"].endswith("南村店)") and ra["路线耗时"]["步行"]["分钟"] == 12
    assert rb["目的地"].endswith("市桥店)") and rb["路线耗时"]["步行"]["分钟"] == 25
    # 每家店调用 3 次（步行/骑行/打车）；两家店的坐标集合必须各自独立
    coords_a = {c for c in calls[:3]}
    coords_b = {c for c in calls[3:6]}
    assert coords_a != coords_b and len(coords_a) == 1 and len(coords_b) == 1, "同名店必须按各自坐标分别规划"
    ok("场景3 同名不同分店：uid 区分，路线不张冠李戴")


# ---------- 场景4：地图无结果 / 接口失败 ----------

def test_no_result_and_failure():
    # 4a 无结果：如实提示，不推荐
    original = rp._poi_query
    rp._poi_query = lambda keyword: []
    try:
        r = json.loads(rp.search_poi("披萨", poi_store={}))
    finally:
        rp._poi_query = original
    assert "确实没有" in r["提示"] or r.get("状态") == "no_result"
    ok("场景4a 地图无结果：如实提示不推荐")

    # 4b 未知 uid 且详情失败：暂无法确认，禁止编造
    original_detail = rp._poi_detail_by_uid
    rp._poi_detail_by_uid = lambda uid: None
    try:
        v = json.loads(rp.verify_route("ghost-uid", poi_store={}))
    finally:
        rp._poi_detail_by_uid = original_detail
    assert "暂无法确认" in v["提示"] and "禁止" in v["要求"]
    ok("场景4b 未知门店：暂无法确认 + 禁止凭店名猜测")

    # 4c 路线接口失败：暂无法确认，不写示例耗时
    store = {"uid-x": make_poi("uid-x", "接口失败的店", "地址", "番禺", 23.018, 113.412)}
    original_dir = rp._baidu_direction
    rp._baidu_direction = lambda mode, coord: None
    try:
        v2 = json.loads(rp.verify_route("uid-x", poi_store=store))
    finally:
        rp._baidu_direction = original_dir
    assert "暂无法确认" in v2["提示"]
    assert "路线" not in store["uid-x"], "失败时不得写入路线事实"
    ok("场景4c 路线接口失败：不写入任何冒充的路线数据")


# ---------- 场景5：卡片事实由后端注入 ----------

def test_backend_fact_injection():
    agent = make_agent()
    agent.poi_store["uid-1"] = dict(
        make_poi("uid-1", "真·完整店名(南村店)", "真实完整地址xx路xx号", "广东省 广州市 番禺区", 23.018, 113.412),
        路线={"步行": {"分钟": 10, "距离_米": 700}, "骑行": {"分钟": 3, "距离_米": 720},
             "查询时间": "2026-11-10 13:00", "来源": "百度地图 direction API（实时）"},
    )
    # 模型试图编造地址和耗时（会被后端覆盖）
    recs = [{
        "uid": "uid-1",
        "name": "模型瞎编的店名",
        "cuisine": "粤菜",
        "price_per_person": 18,
        "reason": "近",
        "transport": {"walk_min": 999},  # 模型编造
    }]
    cards = agent._inject_card_facts(recs)
    c = cards[0]
    assert c["name"] == "真·完整店名(南村店)", "店名必须来自后端 POI 事实"
    assert c["address"] == "真实完整地址xx路xx号"
    assert c["transport"]["walk_min"] == 10, "耗时必须来自后端路线事实（覆盖模型编造值 999）"
    assert c["transport"]["来源"].startswith("百度地图")
    assert c["origin"] and c["route_link"], "卡片须带出发点与官方路线链接"
    ok("场景5a 后端事实注入：模型编造的名称/地址/耗时被覆盖")

    # uid 无效的卡片被整体剔除
    cards2 = agent._inject_card_facts([{"uid": "ghost", "cuisine": "x", "reason": "r"}])
    assert cards2 == [], "uid 无法验证的推荐必须剔除"
    ok("场景5b uid 无法验证的卡片整体剔除（防张冠李戴）")


# ---------- 场景6：官方路线链接格式 ----------

def test_route_link_format():
    link = rp.build_route_link("测试店名", "23.018,113.412")
    assert link.startswith("https://api.map.baidu.com/direction?")
    assert "origin=latlng:23.017973,113.411971|name:" in link, "起点须按官方格式携带坐标与名称"
    assert "destination=latlng:23.018,113.412|name:" in link, "终点须按官方格式携带坐标与名称"
    assert "coord_type=bd09ll" in link, "必须声明坐标系"
    assert "mode=walking" in link
    from urllib.parse import unquote
    assert config.origin_by_key(config.DEFAULT_ORIGIN)["name"] in unquote(link), "起点须带名称"
    assert "测试店名" in unquote(link), "终点须带名称"
    ok("场景6 官方路线链接：起终点坐标+名称+坐标系齐全")


if __name__ == "__main__":
    test_far_store_rejected()
    test_walk_limit_filter()
    test_same_name_different_uid()
    test_no_result_and_failure()
    test_backend_fact_injection()
    test_route_link_format()
    print(f"=== 全部 {PASS} 项 POI 一致性测试通过 ===")
