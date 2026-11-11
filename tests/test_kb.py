"""知识库检索测试：检索召回、过滤、口碑统计、空库诚实性、演示隔离。

跑法：
    python tests/test_kb.py

全部离线（不联网、不调大模型、不消耗地图额度）。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from tools import kb_search

PASSED = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global PASSED
    if ok:
        PASSED += 1
        print(f"  [通过] {name}")
    else:
        print(f"  [失败] {name} {detail}")
        raise AssertionError(name)


def _names(items):
    return [i["名称"] for i in items]


def test_demo_retrieval() -> None:
    config.KB_DEMO = True
    kb_search.reset_cache()

    status = kb_search.kb_status()
    check("演示语料可加载", status["门店数"] >= 10 and status["评价数"] >= 30, str(status))
    check("演示模式被标记", status["demo"] is True)

    hits = kb_search.search_stores("糖水")
    check("关键词命中糖水窗口", hits and hits[0]["名称"] == "二饭三楼糖水窗口", str(_names(hits)[:3]))
    check("命中结果带评分与样本数", hits[0]["评分"] is not None and hits[0]["评分样本"] >= 3, str(hits[0]))
    check("命中结果带评价证据", bool(hits[0]["摘录"] and hits[0]["摘录"]["片段"]), str(hits[0].get("摘录")))

    spicy = kb_search.search_stores("辣的")
    check("口味词'辣'能召回辣味店", any("麻辣烫" in n or "湘聚楼" in n for n in _names(spicy)), str(_names(spicy)))

    breakfast = kb_search.search_stores("早餐吃什么")
    check("场景词'早餐'命中早餐档", "一饭一楼早餐档" in _names(breakfast), str(_names(breakfast)))

    night = kb_search.search_stores("宵夜")
    check("同义词'宵夜'扩展到夜宵店", any("砂锅粥" in n for n in _names(night)), str(_names(night)))

    empty_query = kb_search.search_stores("")
    check("空查询返回全部（受 top_k 限制）", len(empty_query) > 5, str(len(empty_query)))


def test_filters() -> None:
    config.KB_DEMO = True
    kb_search.reset_cache()

    cheap = kb_search.search_stores("", max_price=10)
    check("价格过滤：全部 ≤10 元", all((s["人均"] or 0) <= 10 for s in cheap), str([s["人均"] for s in cheap]))

    near = kb_search.search_stores("", max_walk=5)
    check("步行过滤：全部 ≤5 分钟", all((s["步行分钟"] or 0) <= 5 for s in near), str([s["步行分钟"] for s in near]))

    canteen = kb_search.search_stores("", scene="食堂窗口")
    check("场景过滤：只返回食堂窗口", all(s["类别"] == "食堂窗口" for s in canteen), str(_names(canteen)))

    group = kb_search.search_stores("", tag="适合聚餐")
    check("标签过滤：命中聚餐店", group and all(
        "适合聚餐" in (s["口味标签"] or []) or "适合聚餐" in " ".join(s["好评要点"]) for s in group
    ), str(_names(group)))

    by_price = kb_search.search_stores("", sort="price")
    prices = [s["人均"] for s in by_price if s["人均"] is not None]
    check("按价格排序单调不减", prices == sorted(prices), str(prices[:6]))


def test_aggregation() -> None:
    config.KB_DEMO = True
    kb_search.reset_cache()

    detail = kb_search.store_detail("二饭三楼糖水窗口")
    check("详情可查", detail is not None)
    ratings = [r["评分"] for r in detail["评价"] if r["评分"] is not None]
    expected = round(sum(ratings) / len(ratings), 2)
    check("均值由后端算且与原始评价一致", abs(detail["评分"] - expected) < 0.01, f"{detail['评分']} vs {expected}")
    check("样本数与评价条数一致", detail["样本"] == len(detail["评价"]), f"{detail['样本']} vs {len(detail['评价'])}")
    check("分布之和等于评分样本数", sum(detail["分布"].values()) == len(ratings), str(detail["分布"]))
    check("每条评价都带来源", all(r["来源"] for r in detail["评价"]))
    check("评价按日期倒序", [r["日期"] for r in detail["评价"]] == sorted([r["日期"] for r in detail["评价"]], reverse=True))

    no_review = kb_search.store_detail("轻食沙拉铺")
    check("无评价的店也能查到", no_review is not None)
    check("无评价时评分留空而非编造", no_review["评分"] is None and no_review["评价"] == [])

    check("未知店返回 None", kb_search.store_detail("不存在的店") is None)


def test_determinism() -> None:
    config.KB_DEMO = True
    kb_search.reset_cache()
    first = [s["名称"] for s in kb_search.search_stores("粥")]
    second = [s["名称"] for s in kb_search.search_stores("粥")]
    check("同查询结果稳定（可复现）", first == second, f"{first} vs {second}")


def test_real_data() -> None:
    """正式模式：读真实门店与评价，不得拿演示数据充数，评分必须可溯源。"""
    config.KB_DEMO = False
    kb_search.reset_cache()

    status = kb_search.kb_status()
    check("正式模式未被演示数据污染", status["demo"] is False)
    check("正式模式读到真实门店", status["门店数"] > 0, str(status))
    check("正式模式读到真实评价", status["评价数"] > 0, str(status))

    items = kb_search.search_stores("潮汕牛肉")
    check("真实门店可被检索到", any(i["名称"] == "潮埕" for i in items), str(_names(items)))
    for item in items:
        if item["评分样本"] == 0:
            continue
        check(f"{item['名称']} 有评价时须标明评分来源", item["评分来源"] in ("学生评价", "攻略整理"), str(item))

    # 来源标签要跟着评价的 source 走：全是攻略整理时不能自称“学生评价”
    guide_names = [i["名称"] for i in kb_search.search_stores("")]
    if guide_names:
        labels = {i["名称"]: i["评分来源"] for i in kb_search.search_stores("") if i["评分样本"]}
        check("攻略来源的评价不冒称学生评价",
              all(v in ("攻略整理", "学生评价") for v in labels.values()), str(labels))
        check("全是攻略评价时标注为攻略整理",
              all(v == "攻略整理" for v in labels.values()), str(labels))

    all_items = kb_search.search_stores("")
    quoted = [i for i in all_items if i["摘录"]]
    check("有评价的门店能给出评价摘录", len(quoted) == len(all_items), f"{len(quoted)}/{len(all_items)}")
    check("摘录必须带来源", all((i["摘录"] or {}).get("来源") for i in quoted))

    orphan = [i["名称"] for i in all_items if i["评分样本"] == 0]
    check("没有门店因店名对不上而缺评价（孤儿评价）", not orphan, str(orphan))
    check("未知门店详情为 None", kb_search.store_detail("不存在的店") is None)


def test_scene_alias() -> None:
    """场景取值归一：food_search 写作「饭堂窗口」，前端筛选传「食堂窗口」，必须能命中。"""
    window = {"名称": "测试窗口", "类别": "饭堂窗口", "人均": 15}
    check("饭堂窗口对外统一显示为食堂窗口", kb_search._scene(window) == "食堂窗口")
    check("前端传食堂窗口能筛到饭堂窗口", kb_search._passes(window, "食堂窗口", None, None, ""))
    check("传饭堂窗口也兼容", kb_search._passes(window, "饭堂窗口", None, None, ""))
    check("其它场景不会误命中", not kb_search._passes(window, "周边堂食", None, None, ""))


def main() -> None:
    print("知识库检索测试")
    test_demo_retrieval()
    test_filters()
    test_aggregation()
    test_determinism()
    test_scene_alias()
    test_real_data()
    print(f"\n全部通过：{PASSED} 项")


if __name__ == "__main__":
    main()
