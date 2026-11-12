"""外卖店卡片链路测试：外卖不需要步行路线，有坐标就能出卡片。

覆盖三条规则：
1. 外卖门店（类别=外卖）有坐标但没路线 → 允许出卡片，交通字段是配送参考，
   链接是地图标注而不是导航；
2. 外卖门店没有坐标 → 仍然剔除（不能拿一个不知道在哪的店推荐给用户）；
3. 堂食门店没有路线 → 照旧剔除，不能被外卖豁免规则顺带放过。

跑法：python tests/test_delivery_cards.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from agent.core import FoodAdvisorAgent, _trace
from agent.memory import ConversationMemory, UserProfile
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


def make_agent() -> FoodAdvisorAgent:
    agent = FoodAdvisorAgent.__new__(FoodAdvisorAgent)
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


def test_delivery_card_without_route() -> None:
    print("[1] 外卖店有坐标、无路线 → 允许出卡片")
    agent = make_agent()
    agent.poi_store["dl1"] = {
        "uid": "dl1", "名称": "刘文祥麻辣烫(市桥店)", "地址": "市桥街某路1号",
        "行政区": "广东省 广州市 番禺区", "坐标": "22.956873,113.370643",
        "直线距离_米": 8004, "是餐饮": True, "在范围内": True,
        "来源": "本地知识库", "类别": "外卖", "人均": 25, "学生评分": 2.5,
        "平均配送分钟": 40, "路线": {},
    }
    cards = agent._inject_card_facts([{"uid": "dl1", "cuisine": "麻辣烫", "reason": "评论区说别点", "warning": "避雷"}])
    check("卡片未被剔除", len(cards) == 1, str(agent.last_dropped))
    card = cards[0] if cards else {}
    check("标记为外卖专属", card.get("delivery_only") is True)
    check("不冒充路线已核实", card.get("route_verified") is False)
    # 8 公里外的店不可能还是人工填的 40 分钟：时长必须跟距离自洽
    check("配送时长按距离估算", card.get("transport", {}).get("delivery_min") == 65, str(card.get("transport")))
    check("远距离店标注超配送半径", "超出常规配送半径" in str(card.get("transport", {}).get("距离说明")), str(card.get("transport")))
    check("链接是地图标注而非导航", "api.map.baidu.com/marker" in str(card.get("route_link")))
    check("名称与地址来自后端", card.get("name") == "刘文祥麻辣烫(市桥店)" and card.get("address") == "市桥街某路1号")
    check("评分来自知识库", card.get("rating") == 2.5)

    # 近店对照组：1.9 公里 → 30 分钟，且不该提示超出配送半径
    agent.poi_store["dl_near"] = {
        "uid": "dl_near", "名称": "烹湘", "地址": "兴业大道1166-1号铺",
        "行政区": "广东省 广州市 番禺区", "坐标": "23.00976,113.39542",
        "直线距离_米": 1924, "是餐饮": True, "在范围内": True,
        "来源": "本地知识库", "类别": "外卖", "人均": 28, "学生评分": 4.3,
        "平均配送分钟": 35, "路线": {},
    }
    near = agent._inject_card_facts([{"uid": "dl_near", "cuisine": "湘菜"}])
    near_t = (near[0].get("transport") if near else {}) or {}
    check("近店配送时长按距离算", near_t.get("delivery_min") == 30, str(near_t))
    check("近店不报超出配送半径", "超出" not in str(near_t.get("距离说明")), str(near_t))


def test_delivery_without_coord_dropped() -> None:
    print("[2] 外卖店没有坐标 → 仍然剔除")
    agent = make_agent()
    agent.poi_store["dl2"] = {
        "uid": "dl2", "名称": "遇见赣厨", "坐标": "", "是餐饮": True,
        "在范围内": True, "类别": "外卖", "路线": {},
    }
    cards = agent._inject_card_facts([{"uid": "dl2", "reason": "好吃"}])
    check("无坐标被剔除", not cards)
    check("剔除原因写明缺坐标", any("坐标" in d.get("原因", "") for d in agent.last_dropped), str(agent.last_dropped))


def test_dine_in_still_needs_route() -> None:
    print("[3] 堂食店没有路线 → 照旧剔除（不能被外卖规则放过）")
    agent = make_agent()
    agent.poi_store["r1"] = {
        "uid": "r1", "名称": "潮埕", "坐标": "23.01,113.39", "是餐饮": True,
        "在范围内": True, "类别": "周边堂食", "路线": {},
    }
    cards = agent._inject_card_facts([{"uid": "r1", "reason": "潮汕牛肉"}])
    check("堂食无路线被剔除", not cards)
    check("剔除原因是路线未核实", any("路线" in d.get("原因", "") for d in agent.last_dropped), str(agent.last_dropped))

    agent.poi_store["r1"]["路线"] = {"起点": agent.origin_key, "步行": {"分钟": 18, "距离_米": 1400}, "来源": "百度地图 direction API"}
    cards2 = agent._inject_card_facts([{"uid": "r1", "reason": "潮汕牛肉"}])
    check("补上路线后正常出卡片", len(cards2) == 1 and cards2[0]["route_verified"] is True)
    check("堂食链接是导航链接", "api.map.baidu.com/direction" in str(cards2[0]["route_link"]))


def test_local_hit_registers_poi() -> None:
    print("[4] 本地知识库命中 → 自动登记进 poi_store（否则卡片必被剔除）")
    import importlib

    fs = importlib.import_module("tools.food_search")  # tools.food_search 同时是函数名，必须拿模块本身

    store: dict = {}
    fs.food_search("外卖", poi_store=store)
    check("外卖本地命中已登记", bool(store), f"poi_store 条数 {len(store)}")
    kinds = {v.get("类别") for v in store.values()}
    check("登记条目类别为外卖", kinds == {"外卖"}, str(kinds))
    with_coord = [v for v in store.values() if v.get("坐标")]
    check("有坐标的条目带 uid", all(v.get("uid") for v in with_coord))


def test_far_branch_not_in_range() -> None:
    print("[5] 商圈店（6–11km）登记后不算「附近」，避免被当步行推荐")
    from tools.route_planner import register_kb_poi

    store: dict = {}
    register_kb_poi(store, {"名称": "寿司郎（番禺天河城）", "地图uid": "u_far", "坐标": "23.0,113.4",
                            "直线距离_米": 8000}, kind="周边堂食")
    check("超出 3 公里的堂食店标记为范围外", store["u_far"]["在范围内"] is False)

    store2: dict = {}
    register_kb_poi(store2, {"名称": "美食面家", "地图uid": "u_near", "坐标": "23.0,113.39",
                             "直线距离_米": 1491}, kind="周边堂食")
    check("3 公里内的堂食店标记为范围内", store2["u_near"]["在范围内"] is True)

    store3: dict = {}
    register_kb_poi(store3, {"店名": "刘文祥麻辣烫(市桥店)", "地图uid": "u_dl", "坐标": "22.95,113.37",
                             "直线距离_米": 8004}, kind="外卖")
    check("外卖店不受步行半径限制", store3["u_dl"]["在范围内"] is True)


def test_duplicate_uid_deduped() -> None:
    print("[6] 同一家店被模型重复推荐 → 只保留一张卡片")
    agent = make_agent()
    agent.poi_store["dl3"] = {
        "uid": "dl3", "名称": "烹湘·鲜炒湖南土菜(兴业大道店)", "地址": "兴业大道",
        "坐标": "23.00976,113.39542", "在范围内": True, "是餐饮": True,
        "类别": "外卖", "平均配送分钟": 35, "路线": {},
    }
    cards = agent._inject_card_facts([{"uid": "dl3"}, {"uid": "dl3"}, {"uid": "dl3"}])
    check("重复 uid 去重", len(cards) == 1, f"生成 {len(cards)} 张")


def test_delivery_text_not_walk_talk() -> None:
    print("[7] 外卖卡片不出现步行描述（模型拿不到路线数据，混进来也要被清掉）")
    agent = make_agent()
    agent.poi_store["dl4"] = {
        "uid": "dl4", "名称": "湘村民肴", "地址": "大龙街市莲路傍江东村段5号一楼",
        "坐标": "22.952078,113.415599", "在范围内": True, "是餐饮": True,
        "类别": "外卖", "平均配送分钟": 40, "直线距离_米": 7337, "路线": {},
    }
    cards = agent._inject_card_facts([{
        "uid": "dl4",
        "reason": "辣椒炒肉是招牌，适合喜欢辣味的朋友。",
        "warning": "距离校区较远，步行约139分钟，骑行约57分钟。",
        "worth_it": "步行半小时能到，值得跑一趟。",
    }])
    card = cards[0] if cards else {}
    check("卡片未被剔除", len(cards) == 1, str(agent.last_dropped))
    check("理由保留（未误删正常内容）", "辣椒炒肉" in str(card.get("reason")), str(card.get("reason")))
    check("避雷里的步行时长被清掉", "步行" not in str(card.get("warning")), str(card.get("warning")))
    check("值不值里的步行说法被清掉", "步行" not in str(card.get("worth_it")), str(card.get("worth_it")))

    # 外卖店不该被算路线：入口直接拦掉，模型拿不到步行分钟数
    from llm.client import ToolCall  # noqa: WPS433 —— 测试内导入，避免影响模块加载

    out = agent._execute_tool(ToolCall(id="t1", name="verify_route", arguments='{"uid": "dl4", "mode": "walking"}'))
    check("外卖店 verify_route 被拦下", "外卖店不需要路线" in out or "skipped" in out, out[:120])
    check("拦截不泄露步行时长", "步行" not in out or "不需要" in out, out[:120])


if __name__ == "__main__":
    config.KB_DEMO = False
    test_delivery_card_without_route()
    test_delivery_without_coord_dropped()
    test_dine_in_still_needs_route()
    test_local_hit_registers_poi()
    test_far_branch_not_in_range()
    test_duplicate_uid_deduped()
    test_delivery_text_not_walk_talk()
    print(f"\n通过 {PASSED} 项" + ("" if not FAILED else f"，失败 {len(FAILED)} 项：{FAILED}"))
    raise SystemExit(0 if not FAILED else 1)
