"""推荐流程回归测试：覆盖本次修复的 5 个线上问题 + 4 类既有缺陷。

全部使用模拟数据，不调用真实 LLM / 百度地图，不消耗任何额度。
跑法：python tests/test_recommendation_flow.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

# 全程离线：不碰真实网络
config.WEATHER_ENABLED = False
config.MAP_ENABLED = True
config.BAIDU_MAP_AKS = ["test-ak"]
config.BAIDU_MAP_AK = "test-ak"
config.KB_DEMO = False
config.MAP_USAGE_PATH = os.path.join(tempfile.mkdtemp(prefix="junshi_flow_"), "map_usage.json")

import importlib

from agent.core import FoodAdvisorAgent, _trace
from agent.memory import ConversationMemory, UserProfile
from tools import build_registry

fs = importlib.import_module("tools.food_search")

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


def make_agent(mode: str = "", profile: UserProfile | None = None) -> FoodAdvisorAgent:
    agent = FoodAdvisorAgent.__new__(FoodAdvisorAgent)
    agent.profile = profile or UserProfile(path=os.path.join(tempfile.mkdtemp(), "p.json"))
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
    agent._dining_mode = mode
    return agent


def fake_online(items):
    """替换联网兜底，返回固定候选（模拟百度地图检索结果）。"""

    def _fake(keyword, poi_store):
        for it in items:
            if poi_store is not None:
                poi_store[it["uid"]] = {
                    "uid": it["uid"],
                    "名称": it["名称"],
                    "地址": it.get("地址"),
                    "行政区": "广东省 广州市 番禺区",
                    "坐标": it.get("坐标"),
                    "直线距离_米": it.get("直线距离_米"),
                    "是餐饮": True,
                    "在范围内": True,
                    "类别": "联网检索",
                    "来源": "联网（百度地图）",
                    "路线": {},
                }
        return items, None

    return _fake


# ---------------- 1. 日料外卖：三家店不再被步行校验拦截 ----------------

def test_japanese_delivery_cards() -> None:
    print("[1] 日料外卖：联网搜到的三家店都要能出卡片（原来全被『缺步行』剔除）")
    items = [
        {"uid": "u" + "1" * 23, "名称": "禾小屋寿司(南村店)", "地址": "南村镇", "坐标": "23.010,113.410", "直线距离_米": 1200},
        {"uid": "u" + "2" * 23, "名称": "大雄寿司(南村店)", "地址": "南村镇", "坐标": "23.011,113.411", "直线距离_米": 1500},
        {"uid": "u" + "3" * 23, "名称": "鱼鲜风深海鱼刺身", "坐标": "23.012,113.412", "直线距离_米": 1800},
    ]
    original_online = fs._online_fallback
    original_all = fs._all_places
    fs._online_fallback = fake_online(items)
    fs._all_places = lambda: []  # 强制走联网分支：本地库里有日料会掩盖联网链路
    try:
        agent = make_agent(mode="delivery")
        raw = fs.food_search(query="日料", dining_mode="delivery", poi_store=agent.poi_store)
        data = json.loads(raw)
        check("检索走到联网", "联网检索" in data, str(data.keys()))
        check("返回 3 家候选", len(data.get("结果") or []) == 3, str(len(data.get("结果") or [])))
        check("用餐方式标为 delivery", data.get("用餐方式") == "delivery")
        check("明确提示配送能力待确认", "待确认" in str(data.get("配送提示")), str(data.get("配送提示")))

        recs = [{"uid": it["uid"], "cuisine": "日本料理", "reason": "新鲜刺身"} for it in items]
        cards = agent._inject_card_facts(recs)
        check("三家全部通过校验", len(cards) == 3, f"通过 {len(cards)}，剔除 {agent.last_dropped}")
        if cards:
            t = cards[0].get("transport") or {}
            check("外卖不出现步行时长", "walk_min" not in t, str(t))
            check("配送时长标为预估", t.get("delivery_eta_source") == "按直线距离预估", str(t))
            check("配送能力标待确认", t.get("配送能力") == "待确认", str(t))
            check("不冒充路线已核实", cards[0].get("route_verified") is False)
            check("用餐方式写进卡片", cards[0].get("dining_mode") == "delivery")
    finally:
        fs._online_fallback = original_online
        fs._all_places = original_all


# ---------------- 2. 美式汉堡外卖：参数不能把「外卖」当口味 ----------------

def test_burger_delivery_params() -> None:
    print("[2] 美式汉堡外卖：场景与菜系分开，taste 不再收到『外卖』")
    items = [
        {"uid": "b" + "1" * 23, "名称": "塔斯汀中国汉堡(南村店)", "坐标": "23.010,113.410", "直线距离_米": 900},
    ]
    original_online = fs._online_fallback
    original_all = fs._all_places
    fs._online_fallback = fake_online(items)
    fs._all_places = lambda: []
    try:
        agent = make_agent(mode="delivery")
        data = json.loads(fs.food_search(query="美式汉堡", dining_mode="delivery", poi_store=agent.poi_store))
        check("按品类检索到汉堡店", len(data.get("结果") or []) == 1, str(data))
        check("检索说明里不再出现『外卖』品类", "外卖」匹配" not in str(data.get("联网检索")), str(data.get("联网检索")[:60]))

        # 错误调用（模型把外卖塞进 taste）也要被纠正成场景，而不是当成口味去筛
        agent2 = make_agent()
        data2 = json.loads(fs.food_search(query="美式汉堡", taste="外卖", poi_store=agent2.poi_store))
        check("taste=外卖 被识别为场景而非口味", data2.get("用餐方式") == "delivery", str(data2.get("用餐方式")))
    finally:
        fs._online_fallback = original_online
        fs._all_places = original_all


# ---------------- 3. 菜系不能被静默放宽 ----------------

def test_cuisine_not_silently_relaxed() -> None:
    print("[3] 用户要的品类没有时：不得自动换成品类凑数")
    original = fs._online_fallback
    fs._online_fallback = lambda kw, store: ([], None)  # 联网也查不到
    try:
        data = json.loads(fs.food_search(query="美式汉堡", dining_mode="delivery"))
        check("没有硬凑出结果", not (data.get("结果") or []), str(data.get("结果")))
        advice = str(data.get("建议") or "")
        check("明确要求如实说没找到", "没找到" in advice, advice[:80])
        check("明确禁止用别的品类代替", "不要用其他品类代替" in advice, advice[:80])
    finally:
        fs._online_fallback = original


# ---------------- 4. 无坐标私域外卖 ----------------

def test_private_delivery_without_coord() -> None:
    print("[4] 无坐标私域商家：可推荐、位置待确认、不给导航、不编造")
    agent = make_agent(mode="delivery")
    agent.poi_store["p1"] = {
        "uid": "p1", "名称": "阿姨私房菜", "地址": "", "坐标": "",
        "是餐饮": True, "在范围内": True, "类别": "外卖", "路线": {},
    }
    cards = agent._inject_card_facts([{"uid": "p1", "cuisine": "家常菜", "reason": "阿姨手艺"}])
    check("不被剔除", len(cards) == 1, str(agent.last_dropped))
    card = cards[0] if cards else {}
    check("标位置待确认", card.get("coordinate_pending") is True)
    check("没有导航链接", card.get("route_link") is None)
    check("地址保持为空而不是编造", not card.get("address"), str(card.get("address")))
    check("配送能力不吹牛", (card.get("transport") or {}).get("配送能力") in ("待确认", "知识库标注可外卖"))


# ---------------- 5. 长历史：结构化压缩不能切断 JSON ----------------

def test_long_history_compaction() -> None:
    print("[5] 长历史裁剪：压缩后仍是合法 JSON，uid 完整保留")
    agent = make_agent(mode="delivery")
    agent.memory.messages.append({"role": "system", "content": "sys"})

    big_items = []
    for i in range(8):
        big_items.append({
            "uid": "420cd99c95e9111bc8213d%02d" % i,
            "名称": "候选门店%d（南村店）" % i,
            "类别": "联网检索",
            "菜系": "美式汉堡",
            "地址": "广州市番禺区南村镇兴业大道东%d号" % i,
            "坐标": "23.01%d,113.41%d" % (i, i),
            "直线距离_米": 800 + i * 200,
            "来源": "联网（百度地图）",
            "一句话点评": "很长的点评内容用来撑大体积" * 6,
            "招牌菜": ["汉堡", "薯条", "可乐"],
        })
    big = json.dumps({"用餐方式": "delivery", "结果": big_items}, ensure_ascii=False)
    agent.memory.messages.append({"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "food_search", "arguments": "{}"}}
    ]})
    agent.memory.messages.append({"role": "tool", "tool_call_id": "c1", "content": big})
    check("原始结果确实超长（会触发裁剪）", len(big) > config.TOOL_RESULT_MAX_CHARS, str(len(big)))

    compacted = agent._compact_tool_results(agent.memory.messages)
    tool_msgs = [m for m in compacted if m.get("role") == "tool"]
    check("工具消息还在", len(tool_msgs) == 1)
    try:
        parsed = json.loads(tool_msgs[0]["content"])
        ok_json = True
    except json.JSONDecodeError as exc:
        parsed, ok_json = {}, False
        print(f"        JSON 解析失败：{exc}")
    check("压缩后仍是合法 JSON（截断半截 JSON 是原 bug）", ok_json)
    items = parsed.get("结果") or []
    check("保留候选而不是清空", len(items) >= 1, str(len(items)))
    check("每个候选的 uid 完整", all(len(str(it.get("uid") or "")) >= 20 for it in items), str([it.get("uid") for it in items]))
    check("候选名称没丢", all(it.get("名称") for it in items))
    check("标明这是本地压缩而非上游错误", "本地上下文预算压缩" in str(parsed.get("_压缩说明")), str(parsed.get("_压缩说明"))[:60])


# ---------------- 6. 模型空回复 ----------------

def test_empty_model_reply() -> None:
    print("[6] 模型空回复：不能返回空白气泡")
    agent = make_agent(mode="delivery")
    agent.poi_store["e1"] = {
        "uid": "e1", "名称": "禾小屋寿司", "地址": "南村", "坐标": "23.01,113.41",
        "直线距离_米": 1200, "是餐饮": True, "在范围内": True,
        "类别": "联网检索", "来源": "联网（百度地图）", "路线": {},
    }
    reply = agent._ensure_reply("", "有没有日料外卖")
    check("空回复被兜底成有效文字", bool(reply and reply.strip()), repr(reply))
    check("兜底顺带补出了卡片", bool(agent.last_cards), str(len(agent.last_cards or [])))

    # 什么都没有时也必须给明确说明，而不是空
    agent2 = make_agent(mode="delivery")
    agent2.last_dropped = [{"uid": "x1", "名称": "某店", "原因": "路线未核实（缺 步行）"}]
    reply2 = agent2._ensure_reply("   ", "推荐点吃的")
    check("无候选时也给出可读内容", bool(reply2 and reply2.strip()), repr(reply2))
    check(
        "内容是推荐或明确说明，不是空白气泡",
        any(k in reply2 for k in ("兜底", "没有匹配", "没有检索到", "推荐", "待确认")),
        reply2[:100],
    )


# ---------------- 7. 超时重试去重 ----------------

def test_request_dedupe() -> None:
    print("[7] 超时重试：同一 request_id 只执行一次")
    from web import server

    key = "sid-test:req-abc"
    status, ev, cached = server.begin_dedupe(key)
    check("首次请求由我执行", status == "owner", status)
    # 同一请求并发到达：必须等第一次，不能双开
    status2, ev2, _ = server.begin_dedupe(key)
    check("并发同一请求进入等待", status2 == "wait", status2)
    payload = {"reply": "第一次的真实结果", "cards": []}
    server.finish_dedupe(key, payload)
    check("等待者被唤醒", ev2 is not None and ev2.wait(timeout=2), "事件未 set")
    status3, _, cached3 = server.begin_dedupe(key)
    check("重试命中缓存", status3 == "hit" and (cached3 or {}).get("reply") == "第一次的真实结果", str(cached3))

    key2 = "sid-test:req-fail"
    server.begin_dedupe(key2)
    server.finish_dedupe(key2, None)
    status4, _, _ = server.begin_dedupe(key2)
    check("失败的请求不缓存（允许真正重跑）", status4 == "owner", status4)


# ---------------- 8. 两个用户画像互不串 ----------------

def test_profile_isolation() -> None:
    print("[8] 两个独立用户：画像不串、忌口累积不覆盖")
    a = UserProfile(path=os.path.join(tempfile.mkdtemp(), "a.json"))
    b = UserProfile(path=os.path.join(tempfile.mkdtemp(), "b.json"))
    a.update("忌口", "不吃香菜")
    check("A 的忌口只属于 A", b.data["忌口"] == [], str(b.data["忌口"]))
    check("类默认值未被污染", UserProfile.DEFAULTS["忌口"] == [], str(UserProfile.DEFAULTS["忌口"]))

    a.update("忌口", "不要葱")
    a.update("忌口", "不吃香菜")  # 重复
    check("忌口累积且去重", a.data["忌口"] == ["不吃香菜", "不要葱"], str(a.data["忌口"]))

    a.update("忌口", "少辣、不要姜")
    check("口语串被拆成独立条目", "少辣" in a.data["忌口"] and "不要姜" in a.data["忌口"], str(a.data["忌口"]))

    # 持久化后重新加载仍然是各自的内容
    a2 = UserProfile(path=a.path)
    b2 = UserProfile(path=b.path)
    check("A 重新加载后仍是自己的忌口", a2.data["忌口"] == a.data["忌口"], str(a2.data["忌口"]))
    check("B 重新加载后仍为空", b2.data["忌口"] == [], str(b2.data["忌口"]))


# ---------------- 9. 外卖不出现「步行 0 分钟」 ----------------

def test_no_walk_zero_for_delivery() -> None:
    print("[9] 外卖结果不得出现『步行 0 分钟』")
    data = json.loads(fs.food_search(query="外卖"))
    items = data.get("结果") or []
    check("外卖检索有结果", bool(items), str(len(items)))
    walk_values = [(it.get("交通") or {}).get("步行") for it in items]
    check("外卖结果不带步行字段", all(v is None for v in walk_values), str(walk_values))


# ---------------- 10. 坏 Key 不能拖垮整个池 ----------------

def test_bad_key_failover() -> None:
    print("[10] 池里有一把失效 Key：必须自动换下一把，而不是整轮降级离线")
    import llm.client as lc

    saved = (config.LLM_PROVIDER, config.LLM_MODEL, config.LLM_BASE_URL, config.LLM_KEYS, config.LLM_ENABLED)
    config.LLM_PROVIDER = "openai"
    config.LLM_MODEL = "glm-4-flash"
    config.LLM_BASE_URL = "https://example.invalid/v1"
    # 第一把是失效 Key（401），第二把可用
    config.LLM_KEYS = ["bad" * 11, "good" * 11]
    config.LLM_ENABLED = True
    try:
        client = lc.LLMClient()
        tried: list[str] = []

        def fake_do_chat(body, key):
            tried.append(key[:4])
            if key.startswith("bad"):
                raise RuntimeError(
                    'LLM 调用失败: HTTP 401 {"error":{"code":"401","message":"令牌已过期或验证不正确"}}'
                )
            return {"choices": [{"message": {"role": "assistant", "content": "好"}, "finish_reason": "stop"}]}

        client._do_chat = fake_do_chat
        msg = client.chat([{"role": "user", "content": "hi"}])
        check("坏 Key 未导致整轮失败", (msg.content or "") == "好", str(msg.content))
        check("确实先撞了坏 Key 再换好的", tried[:1] != ["good"] and "good" in tried, str(tried))

        # 坏 Key 已被熔断，后续请求不该再撞它
        tried.clear()
        client.chat([{"role": "user", "content": "hi"}])
        check("坏 Key 被熔断后不再复用", tried == ["good"], str(tried))
    finally:
        (config.LLM_PROVIDER, config.LLM_MODEL, config.LLM_BASE_URL, config.LLM_KEYS, config.LLM_ENABLED) = saved


# ---------------- 11. 离线兜底必须认菜系 ----------------

def test_offline_fallback_respects_cuisine() -> None:
    print("[11] 离线兜底：问日料不能推酸菜鱼（菜系条件不能被丢掉）")
    agent = make_agent(mode="delivery")
    reply = agent._offline_fallback("有日料的外卖推荐吗")
    check("有内容返回", bool(reply and reply.strip()), repr(reply))
    # 关键：不能出现完全无关的品类
    check("没有推酸菜鱼", "酸菜鱼" not in reply, reply[:160])
    check("没有推面馆/牛肉面这类无关项", ("面馆" not in reply), reply[:160])
    # 认出日料后，结果里应出现日料相关词
    check(
        "结果与日料相关",
        any(k in reply for k in ("寿司", "日料", "天妇罗", "刺身", "日本料理")) or "没有" in reply,
        reply[:200],
    )


if __name__ == "__main__":
    test_japanese_delivery_cards()
    test_burger_delivery_params()
    test_cuisine_not_silently_relaxed()
    test_private_delivery_without_coord()
    test_long_history_compaction()
    test_empty_model_reply()
    test_request_dedupe()
    test_profile_isolation()
    test_no_walk_zero_for_delivery()
    test_bad_key_failover()
    test_offline_fallback_respects_cuisine()
    print(f"\n通过 {PASSED} 项" + ("" if not FAILED else f"，失败 {len(FAILED)} 项：{FAILED}"))
    raise SystemExit(0 if not FAILED else 1)
