"""food_search 检索顺序测试：先本地知识库，无匹配自动联网兜底。

全部 mock 联网层（route_planner.search_poi），不消耗任何地图额度。
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

# 干净环境（无 .env）也要能跑：联网开关强制打开，AK 用假值；
# 本测试对 route_planner.search_poi 全程 mock，不会真的发请求。
config.MAP_ENABLED = True
config.BAIDU_MAP_AK = config.BAIDU_MAP_AK or "test-ak"

import importlib

fs = importlib.import_module("tools.food_search")  # 注意：tools 包属性里 food_search 是函数，须用 importlib 拿模块
from tools import route_planner as rp

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


def _with_mock(payload, calls):
    """把 route_planner.search_poi 替换成返回固定 JSON 的假实现。"""

    def fake(keyword, poi_store=None, radius=None):
        calls.append(keyword)
        if isinstance(payload, Exception):
            raise payload
        if callable(payload):
            return payload(keyword)
        return json.dumps(payload, ensure_ascii=False)

    return fake


class _empty_local:
    """临时把本地知识库清空，稳定测「本地无匹配 → 联网」分支。

    不能挑一个「当前知识库里没有的品类」来测：知识库是活的，
    今天没有川菜、明天加了川菜，测试就会莫名其妙变成假失败。
    """

    def __enter__(self):
        self._orig = fs._all_places
        fs._all_places = lambda: []
        return self

    def __exit__(self, *exc):
        fs._all_places = self._orig
        return False


def run(query: str, **kwargs) -> dict:
    return json.loads(fs.food_search(query, **kwargs))


def test_kb_hit_no_online() -> None:
    print("[1] 本地命中 → 不联网（不耗额度）")
    calls: list[str] = []
    original = rp.search_poi
    rp.search_poi = _with_mock({"状态": "ok", "结果": []}, calls)
    try:
        data = run("面")
        check("本地有关键词「面」的门店", len(data.get("结果") or []) > 0)
        check("结果来源均为知识库", all(i["来源"] == "知识库" for i in data["结果"]))
        check("未发起联网检索", calls == [], str(calls))
    finally:
        rp.search_poi = original


def test_kb_miss_online_fallback() -> None:
    print("[2] 本地无匹配 → 自动联网兜底")
    calls: list[str] = []
    original = rp.search_poi
    online = {
        "状态": "ok",
        "结果": [
            {
                "uid": "u1",
                "名称": "测试川菜馆",
                "地址": "测试路1号",
                "坐标": "23.0,113.4",
                "直线距离_米": 800,
                "查询品类": "川菜",
            },
            {
                "uid": "u2",
                "名称": "测试湘菜馆",
                "地址": "测试路2号",
                "坐标": "23.0,113.4",
                "直线距离_米": 1500,
                "查询品类": "川菜",
            },
        ],
    }
    rp.search_poi = _with_mock(online, calls)
    try:
        with _empty_local():
            data = run("川菜")
        # 本地库此刻为空，必然走联网；若没联网说明兜底链断了
        check("知识库为空时应联网", len(calls) >= 1, str(calls))
        check("联网结果带 uid", all(i.get("uid") for i in data.get("结果", [])))
        check("来源标注为联网", all(i["来源"] == "联网（百度地图）" for i in data.get("结果", [])))
        check("返回包含联网说明", "联网" in str(data.get("联网检索", "")))
    finally:
        rp.search_poi = original


def test_kb_miss_quota_note() -> None:
    print("[3] 联网失败（配额/限流）→ 如实说明，不影响本地")
    calls: list[str] = []
    original = rp.search_poi
    rp.search_poi = _with_mock({"状态": "quota_exhausted", "提示": "地图接口今日配额已用完"}, calls)
    try:
        with _empty_local():
            data = run("川菜")
        check("结果为空", data.get("结果") == [])
        check("附带联网状态说明", "配额" in str(data.get("联网检索", "")), str(data))
    finally:
        rp.search_poi = original


def test_scene_word_no_online() -> None:
    print("[4] 场景词（外卖/聚餐）无匹配 → 不发起联网")
    calls: list[str] = []
    original = rp.search_poi
    rp.search_poi = _with_mock({"状态": "ok", "结果": []}, calls)
    try:
        data = run("外卖")
        check("未发起联网检索", calls == [], str(calls))
        check("返回提示或结果", ("提示" in data) or ("建议" in data) or data.get("结果"), str(data)[:120])
    finally:
        rp.search_poi = original


def test_exception_safe() -> None:
    print("[5] 联网抛异常 → 不拖垮检索")
    calls: list[str] = []
    original = rp.search_poi
    rp.search_poi = _with_mock(RuntimeError("network down"), calls)
    try:
        with _empty_local():
            data = run("川菜")
        check("返回了合法 JSON 且结果为空", data.get("结果") == [])
        check("异常被捕获并说明", "异常" in str(data.get("联网检索", "")), str(data)[:120])
    finally:
        rp.search_poi = original


def test_delivery_scene_isolation() -> None:
    print("[6] 场景互斥：堂食检索不混入外卖店")
    # 用本地库里确实有货的品类（「面」3 家），且放宽步行时间——
    # 否则测的是「数据里没有 15 分钟内的店」而不是「场景互斥」
    dine_in = run("面", max_minutes=40)
    cats = {i.get("类别") for i in dine_in.get("结果", [])}
    check("堂食检索结果里没有外卖条目", "外卖" not in cats, str(cats))
    check("堂食检索仍有结果", bool(dine_in.get("结果")), str(dine_in)[:100])

    takeaway = run("外卖")
    cats2 = {i.get("类别") for i in takeaway.get("结果", [])}
    check("外卖检索能命中外卖店", "外卖" in cats2, str(cats2))

    spicy_takeaway = run("外卖", taste="辣")
    names = [i["名称"] for i in spicy_takeaway.get("结果", [])]
    check("外卖+口味组合可用", bool(names), str(names)[:100])
    # 经过同义词扩展后「辣」会连带湘菜/川菜，所以这三个字都算辣味店
    check(
        "外卖+口味结果都是辣味店",
        all(any(k in json.dumps(i, ensure_ascii=False) for k in ("辣", "湘", "川"))
            for i in spicy_takeaway.get("结果", [])),
        str(names),
    )

    # 口语口味词要归一到知识库标签：模型常把「推荐几个外卖」传成 taste="爱辣"
    slang = run("外卖", taste="爱辣")
    check("口语口味「爱辣」不被过滤成空", bool(slang.get("结果")), str(slang)[:120])
    check(
        "「爱辣」与「辣」召回一致",
        {i["名称"] for i in slang.get("结果", [])} == {i["名称"] for i in spicy_takeaway.get("结果", [])},
        f"爱辣={[i['名称'] for i in slang.get('结果', [])]} 辣={names}",
    )

    # 口味词不是地图能检索的品类，拿它联网只会白烧额度
    pure_taste = run("外卖", taste="清淡")
    check("场景词+纯口味词不触发联网", "联网检索" not in pure_taste, str(pure_taste)[:120])


def main() -> None:
    config.KB_DEMO = False
    test_kb_hit_no_online()
    test_kb_miss_online_fallback()
    test_kb_miss_quota_note()
    test_scene_word_no_online()
    test_exception_safe()
    test_delivery_scene_isolation()
    test_store_name_query()
    print(f"\n通过 {PASSED} 项" + ("" if not FAILED else f"，失败 {len(FAILED)} 项：{FAILED}"))
    raise SystemExit(0 if not FAILED else 1)


def test_store_name_query() -> None:
    print("[7] 问某家店（含疑问尾巴）→ 按店名命中，且不因场景互斥漏掉外卖店")
    calls: list[str] = []
    original = rp.search_poi
    rp.search_poi = _with_mock({"状态": "ok", "结果": []}, calls)
    try:
        data = run("达海美食怎么样")
        names = [i["名称"] for i in data.get("结果") or []]
        check("整句问句能命中该店", names[:1] == ["达海美食"], str(names))
        check("带回按店名命中说明", "按店名命中" in str(data.get("匹配方式", "")), str(data)[:80])

        data2 = run("小意司日料多少钱")
        names2 = [i["名称"] for i in data2.get("结果") or []]
        check("外卖店按店名也能命中（不被场景互斥排掉）", names2[:1] == ["小意司日料"], str(names2))
        check("店名查询不联网", calls == [], str(calls))
        check("外卖店带下单方式", bool((data2.get("结果") or [{}])[0].get("下单方式")))
    finally:
        rp.search_poi = original


if __name__ == "__main__":
    main()
