"""美食检索官：先查本地知识库，无匹配时自动联网（百度地图）检索。

数据来源（合规）：
  - 本地知识库：大众点评/小红书内容的人工整理 + 学生问卷实测；
  - 联网兜底：百度地图地点检索（出发点 3 公里内真实门店，带 uid/坐标）。
示例数据已隔离到 data/examples/（仅作字段模板参考），正式推荐
只使用 data/ 下的真实数据与地图实时返回的门店，绝不返回示例数据冒充真实门店。

检索顺序：
  1. 本地知识库有匹配 → 只返回本地结果（不消耗地图额度）；
  2. 本地无匹配且有关键词 → 自动调 search_poi 联网检索，返回真实门店
     （写入会话 poi_store，后续 verify_route / 推荐卡片用同一 uid 引用）；
  3. 联网失败（限流/配额/网络）→ 如实返回状态说明，不编造门店。
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import config
from tools.kb_index import SYNONYMS

# 场景词不参与菜系/品类匹配：『外卖』说的是怎么吃，不是吃什么。
# 历史包袱：这些词曾被塞进 query，现在统一转到 dining_mode，但仍要兜住老调用。
_SCENE_WORDS = {"外卖", "外卖推荐", "聚餐", "聚会"}
_DELIVERY_WORDS = {"外卖", "外卖推荐", "delivery", "deliver", "takeout", "takeaway", "配送"}
_DINE_IN_WORDS = {"堂食", "到店", "dine_in", "dinein", "in_store", "到店吃"}


def normalize_dining_mode(value: str) -> str:
    """把各种口语/英文写法收敛成 delivery / dine_in / 空（未指定）。

    未指定 ≠ 堂食：用户没提用餐方式时，两种店都可以进候选，
    卡片再按实际门店类别决定要不要核实路线。
    """
    text = (value or "").strip().lower()
    if not text:
        return ""
    if text in _DELIVERY_WORDS or "外卖" in text:
        return "delivery"
    if text in _DINE_IN_WORDS or "堂食" in text or "到店" in text:
        return "dine_in"
    return ""


def _load(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data.get("restaurants") or data.get("窗口") or data.get("店铺") or []
    except (json.JSONDecodeError, OSError):
        return []


def _find_place(place_name: str) -> Dict[str, Any] | None:
    """按名称在真实知识库（周边餐厅+饭堂窗口+外卖）中查找门店。"""
    for path, kind in (
        (config.RESTAURANTS_PATH, "周边堂食"),
        (config.CANTEEN_PATH, "饭堂窗口"),
        (config.DELIVERY_PATH, "外卖"),
    ):
        for item in _load(path):
            name = item.get("名称") or item.get("店名") or item.get("窗口名")
            if place_name in str(name or ""):
                hit = dict(item)
                hit.setdefault("类别", kind)
                return hit
    return None


def _all_places() -> List[Dict[str, Any]]:
    """合并三类知识库，统一字段结构。"""
    places: List[Dict[str, Any]] = []

    for item in _load(config.RESTAURANTS_PATH):
        entry = dict(item)
        entry["类别"] = "周边堂食"
        places.append(entry)

    for item in _load(config.CANTEEN_PATH):
        entry = dict(item)
        entry.setdefault("人均", entry.get("价格", 15))
        entry.setdefault("菜系", "饭堂")
        entry["交通"] = entry.get("交通", {"步行": entry.get("步行分钟", 6)})
        entry["类别"] = "饭堂窗口"
        places.append(entry)

    for item in _load(config.DELIVERY_PATH):
        entry = dict(item)
        # 外卖没有「步行 X 分钟」这回事。曾经这里默认填 {"步行": 0}，
        # 前端就渲染成「步行 0 分钟」——一个凭空捏造的事实。
        # 现在干脆不给交通字段：没有数据就没有，卡片只显示配送参考。
        entry.pop("交通", None)
        entry["类别"] = "外卖"
        places.append(entry)

    return places


def _taste_terms(taste: str) -> List[str]:
    """把模型给的口语口味词归一成知识库里可能出现的词。

    小模型经常把「推荐几个外卖」传成 taste="爱辣"（用户根本没提口味），
    或者把「想吃点清淡的」传成 taste="清淡点"。这里用 SYNONYMS 做一次扩展：
    「爱辣」→ 辣/麻辣/香辣/重口/川菜/湘菜，任一词命中即算匹配。
    认不出的词保留原样（下面还有「全灭则放宽」兜底，不会死锁）。
    """
    text = (taste or "").strip().lower()
    if not text:
        return []
    terms: List[str] = []
    for key, alts in SYNONYMS.items():
        if key and key in text:
            terms.append(key)
            terms.extend(alts)
    return terms or [text]


def _is_taste_word(term: str) -> bool:
    """判断是不是纯口味/需求词（这类词拿去地图检索没有意义）。"""
    return any(key and key in (term or "") for key in SYNONYMS)


def _match(
    item: Dict[str, Any],
    query: str,
    taste_terms: List[str],
    budget_max: float | None,
    max_minutes: int | None,
    want_delivery: bool,
    dining_mode: str,
) -> bool:
    kind = str(item.get("类别") or "")
    is_delivery_store = kind == "外卖"

    # 场景互斥：用户没说要外卖时，外卖店不进结果。
    # 否则「辣的、步行15分钟」会把外卖店排到前面——外卖店没有步行字段，
    # 天然绕过时间筛选，堂食推荐会被挤掉。
    if is_delivery_store and not want_delivery:
        return False

    # 饭堂窗口是明确不能配送的堂食档口，点外卖时不该混进来。
    # 周边堂食 / 联网检索则保留：地图上的店不代表不能送，只是后端没有配送依据，
    # 卡片会标「配送能力待确认」——既不武断排除，也不假装已确认。
    if want_delivery and kind == "饭堂窗口":
        return False

    if budget_max is not None:
        try:
            if float(item.get("人均", 0)) > float(budget_max):
                return False
        except (TypeError, ValueError):
            pass

    # 步行上限只对堂食有意义：外卖不走路，拿步行分钟筛外卖是错的。
    if max_minutes is not None and not want_delivery:
        try:
            walk = int(item.get("交通", {}).get("步行", 999))
            if walk > int(max_minutes):
                return False
        except (TypeError, ValueError):
            pass

    # 只有「私域外卖库」里明确标注不可外卖的店才排除；
    # 其他店配送能力未知，交给卡片标注，不在这里一刀切。
    if want_delivery and is_delivery_store and item.get("可外卖") is False:
        return False

    if query in {"聚餐", "聚会"}:
        try:
            if int(item.get("适合聚餐人数", 0) or 0) < 4:
                return False
        except (TypeError, ValueError):
            return False

    # 关键词匹配：店名、菜系、招牌菜、标签、一句话点评
    # （"外卖/聚餐/聚会"是场景词而非内容词，上面已按场景字段过滤，这里跳过）
    if query and query not in _SCENE_WORDS:
        haystack = " ".join(
            str(item.get(field, ""))
            for field in ("名称", "店名", "窗口名", "菜系", "一句话点评", "口味标签")
        ) + " " + " ".join(str(x) for x in item.get("招牌菜", []) or [])
        # 子串双向匹配：用户说「日料」能命中菜系里的「日本料理」，
        # 说「日本料理」也能命中标签里的「日料」。只认单向会漏掉大量同义表达。
        q = query.lower()
        if q not in haystack.lower() and haystack.lower() not in q:
            return False

    if taste_terms:
        tag = f"{item.get('菜系', '')}{' '.join(str(t) for t in item.get('口味标签', []) or [])}{item.get('一句话点评', '')}"
        if not any(term in tag for term in taste_terms):
            return False
    return True


# ---------------- 联网兜底（百度地图） ----------------


def _online_fallback(keyword: str, poi_store: Optional[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """联网检索兜底。返回 (在线条目列表, 状态说明)；说明非 None 表示联网不可用/无结果。"""
    from tools import route_planner as rp  # 惰性导入，避免包初始化顺序问题

    try:
        raw = rp.search_poi(keyword, poi_store=poi_store if poi_store is not None else {})
        data = json.loads(raw)
    except Exception as exc:  # noqa: BLE001 —— 联网失败不能拖垮本地检索
        return [], f"联网检索异常：{exc}"
    if data.get("状态") != "ok":
        note = data.get("提示") or data.get("建议") or "联网检索暂不可用"
        return [], note
    return data.get("结果") or [], None


def food_search(
    query: str = "",
    taste: str = "",
    dining_mode: str = "",
    budget_max: float | None = None,
    max_minutes: int | None = None,
    poi_store: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    """先查本地知识库；无匹配时自动联网（百度地图 3 公里内真实门店）。

    三个维度各管一件事，绝不混用：
      query       —— 吃什么品类（汉堡/日料/粥/粤菜），只管菜系
      taste       —— 什么口味（辣/清淡/甜），只管口味
      dining_mode —— 怎么吃（delivery 外卖 / dine_in 到店 / 空 未指定），只管场景

    本地结果按学生评分降序、步行时间升序；联网结果按直线距离升序，带 uid。
    """
    raw_query = (query or "").strip()
    raw_taste = (taste or "").strip()
    mode = normalize_dining_mode(dining_mode)
    # 向后兼容/纠错：模型常把场景词塞进 query 或 taste（截图里就出现过
    # query="日本料理", taste="外卖"）。场景词必须转到 dining_mode，
    # 否则 taste="外卖" 会被拿去当口味标签匹配，一条都命中不了。
    for slot in ("query", "taste"):
        raw = raw_query if slot == "query" else raw_taste
        if raw in _SCENE_WORDS:
            if "外卖" in raw and not mode:
                mode = "delivery"
            raw = "" if "外卖" in raw else raw
            if slot == "query":
                raw_query = raw
            else:
                raw_taste = raw
    query, taste = raw_query, raw_taste
    want_delivery = mode == "delivery"

    if budget_max is not None:
        budget_max = float(budget_max)
    # 外卖不走路：用户若顺手给了步行上限，对外卖场景直接忽略，避免拿它筛掉可送的店
    if max_minutes is not None and not want_delivery:
        max_minutes = int(max_minutes)
    elif want_delivery:
        max_minutes = None

    places = _all_places()
    terms = _taste_terms(taste)
    hits = [p for p in places if _match(p, query, terms, budget_max, max_minutes, want_delivery, mode)]
    taste_relaxed = ""
    if not hits and terms:
        # 口味词把结果全灭了：可能是模型自己加的口味（用户原话里并没有），
        # 或者知识库还没有这个口味标签。此时放宽【口味】给结果，但绝不放宽菜系——
        # 用户明确要汉堡，就不能拿酸菜鱼凑数。
        hits = [p for p in places if _match(p, query, [], budget_max, max_minutes, want_delivery, mode)]
        if hits:
            taste_relaxed = (
                f"没有严格符合「{taste}」口味的门店，以下结果已放宽【口味】约束"
                + (f"，但「{query}」这个菜系条件没有放宽" if query else "")
                + "。请在回答里说明这一点并询问用户是否接受，不要假装完全符合。"
            )
    if want_delivery:
        # 外卖的优先级和堂食相反：先看能不能送到（距离），再看评分。
        # 按评分排会把 7 公里外的店排第一，学生点外卖第一眼关心的是「多久能到」。
        hits.sort(key=lambda p: (
            int(p.get("直线距离_米") or 99999),
            -float(p.get("学生评分", 0) or 0),
        ))
    else:
        hits.sort(key=lambda p: (-float(p.get("学生评分", 0) or 0), int(p.get("交通", {}).get("步行", 0) or 0)))

    # ---- ① 本地有匹配：只返回本地结果，不消耗地图额度 ----
    if hits:
        # 本地店也要登记进 poi_store：它们不经过 search_poi，不登记的话
        # format_recommendation 会判成「uid 未核实」把整批卡片剔除。
        if poi_store is not None:
            from tools.route_planner import register_kb_poi

            for p in hits[:8]:
                register_kb_poi(poi_store, p, kind=str(p.get("类别") or ""))
        return json.dumps(
            {
                "用餐方式": mode or "未指定",
                **({"口味说明": taste_relaxed, "需询问用户": True} if taste_relaxed else {}),
                "结果": [
                    {
                        "名称": p.get("名称") or p.get("店名") or p.get("窗口名"),
                        "类别": p.get("类别"),
                        "菜系": p.get("菜系"),
                        "人均": p.get("人均"),
                        "招牌菜": p.get("招牌菜"),
                        # 外卖不提供步行交通：填 0 会被前端渲染成「步行 0 分钟」，是假事实
                        "交通": None if want_delivery else p.get("交通"),
                        "地址": p.get("地址"),
                        "营业时间": p.get("营业时间"),
                        "高峰时段": p.get("高峰时段"),
                        "适合聚餐人数": p.get("适合聚餐人数"),
                        "学生评分": p.get("学生评分"),
                        "避雷": p.get("避雷"),
                        "一句话点评": p.get("一句话点评"),
                        "可外卖": p.get("可外卖"),
                        "平均配送分钟": p.get("平均配送分钟"),
                        "配送说明": "配送能力/费用/时长未核实，需标待确认" if want_delivery else None,
                        "uid": p.get("地图uid"),
                        "坐标": p.get("坐标"),
                        "来源": "知识库",
                    }
                    for p in hits[:5]
                ]
            },
            ensure_ascii=False,
        )

    # ---- ② 本地无匹配：自动联网兜底（仅限可搜索的具体关键词） ----
    # 口味/需求词（辣、清淡、便宜…）不是地图能检索的品类，拿它联网只会白烧额度，
    # 还会返回「出发点 3 公里内没有『爱辣』的餐饮门店」这种没用的结论。
    if query and query not in _SCENE_WORDS:
        online_kw = query
    elif taste and not _is_taste_word(taste):
        online_kw = taste
    else:
        online_kw = ""
    online_items: List[Dict[str, Any]] = []
    online_note: Optional[str] = None
    if online_kw and config.MAP_ENABLED:
        online_items, online_note = _online_fallback(online_kw, poi_store)

    if online_items:
        out: Dict[str, Any] = {
            "用餐方式": mode or "未指定",
            "联网检索": f"知识库无「{online_kw}」匹配，已自动联网检索百度地图（出发点 3 公里内真实门店）",
            "结果": [
                {
                    "名称": it.get("名称"),
                    # 类别保持事实：「联网检索」不等于「可配送」，两者不能混为一谈
                    "类别": "联网检索",
                    "菜系": it.get("查询品类") or online_kw,
                    "人均": None,
                    "地址": it.get("地址"),
                    "直线距离_米": it.get("直线距离_米"),
                    "坐标": it.get("坐标"),
                    "uid": it.get("uid"),
                    "可外卖": None,
                    "来源": "联网（百度地图）",
                }
                for it in online_items[:8]
            ],
        }
        if want_delivery:
            # 关键事实边界：地图搜到这家店 ≠ 确认它送外卖。
            # 措辞尽量中性，避免模型把「不要编造」这类内部指令原样念给用户。
            out["配送提示"] = (
                "这些门店来自地图检索，只能说明店铺存在。"
                "配送能力、配送费、配送时长均未核实，统一标「待确认」。"
            )
        elif budget_max is not None:
            out["联网检索"] += "（联网结果未按预算筛选，请按需核实后再推荐）"
        return json.dumps(out, ensure_ascii=False)

    # ---- ③ 两边都没有：如实说明 ----
    out = {"结果": [], "用餐方式": mode or "未指定"}
    if online_note is not None:
        out["联网检索"] = online_note
        out["建议"] = "本地知识库无匹配，联网检索暂不可用；可放宽口味/预算/时间约束稍后再试"
    elif online_kw:
        out["提示"] = f"本地知识库与联网检索均无「{online_kw}」匹配"
        out["建议"] = (
            f"附近没有「{online_kw}」这个品类。应当如实说明没找到，"
            "并询问用户是否换菜系或放宽条件；不要用其他品类代替。"
        )
    elif not places:
        out["提示"] = "本地知识库暂无真实数据（示例数据已隔离，不用于正式推荐）"
        out["下一步"] = "必须改用 search_poi 联网检索周边真实门店，检索结果会带 uid/地址/坐标/距离，再用 verify_route 核实路线"
    else:
        out["提示"] = "知识库有真实数据但无匹配结果"
        out["建议"] = "可放宽口味/预算/时间约束后重试"
    return json.dumps(out, ensure_ascii=False)
