"""路线参谋：门店事实与路线事实的唯一权威来源。

数据一致性铁律（本模块的立身之本）：
  1. 门店身份以百度 POI uid 贯穿全程：检索 → 路线 → 推荐卡片 → 地图跳转，
     引用的永远是同一家店。禁止仅凭店名定位（同名不同分店会张冠李戴）。
  2. 「附近推荐」严格限制在出发点 3 公里内：百度 API radius_limit=true，
     后端再用坐标直线距离复核（双保险），超出即剔除。
     但「指定门店查询」不受 3 公里限制——先在 3 公里内找，找不到再全市找，
     并明确标注"超出校区范围"，而不是谎称"这家店不存在"。
  3. 路线只查用户真正需要的交通方式（默认步行），避免"一家店 × 三种方式"
     把百度额度烧穿；缓存与限速由 tools/map_gateway 统一治理。
  4. 路线事实（距离、耗时、交通方式、查询时间、来源）只由后端计算并
     保存在 poi_store；失败就是失败，返回"暂无法确认"，绝不退回静态数据冒充。
  5. 停车场、公交站等附属 POI 不得当餐馆推荐。

数据来源合规声明：使用百度地图开放平台官方 API，非爬虫。
"""

from __future__ import annotations

import contextvars
import difflib
import json
import math
import re
import threading
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import config
from tools import map_gateway as gw
from tools.food_search import _load

RADIUS_M = 3000            # 「附近推荐」检索半径（米）
RADIUS_TOLERANCE_M = 3200  # 后端复核容差（直线距离略大于路网属正常）
ORIGIN_NAME = config.origin_by_key(config.DEFAULT_ORIGIN)["name"]  # 兼容旧引用

MODE_LABEL = {"walking": "步行", "riding": "骑行", "driving": "打车"}
LABEL_MODE = {v: k for k, v in MODE_LABEL.items()}

# 出发点通过 contextvar 传递：工具函数签名保持简单，同时保证多线程会话互不串味。
_ORIGIN_CTX: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar("origin", default=None)
_TLS = threading.local()


def set_origin(preset: Optional[Dict[str, Any]]) -> None:
    """设置当前会话的出发点（由 agent 在每轮开始时调用）。"""
    _ORIGIN_CTX.set(preset)


def current_origin() -> Dict[str, Any]:
    return _ORIGIN_CTX.get() or config.origin_by_key(config.DEFAULT_ORIGIN)


def _origin_pair() -> Tuple[float, float]:
    lat, lng = str(current_origin()["coord"]).split(",")
    return float(lat), float(lng)


# ---------------- 基础工具 ----------------

def _haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> int:
    """球面直线距离（米），用于后端复核检索结果确实在范围内。"""
    rad = math.pi / 180
    dlat = (lat2 - lat1) * rad
    dlng = (lng2 - lng1) * rad
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1 * rad) * math.cos(lat2 * rad) * math.sin(dlng / 2) ** 2
    return round(2 * 6371000 * math.asin(math.sqrt(a)))


# ---------------- 餐饮 / 附属设施判别 ----------------

_DINING_HINTS = (
    "餐厅", "餐馆", "菜馆", "饭店", "酒楼", "食堂", "美食", "餐饮", "小吃", "快餐",
    "面馆", "粉店", "米粉", "粥", "烧腊", "火锅", "烧烤", "咖啡", "茶饮", "奶茶",
    "甜品", "烘焙", "披萨", "汉堡", "炸鸡", "麻辣烫", "饺子", "包子", "饭店",
)
_NON_DINING_HINTS = (
    "停车场", "停车库", "停车位", "公交站", "公交车站", "地铁站", "充电站", "充电桩",
    "加油站", "公共厕所", "卫生间", "售票", "出入口", "门禁", "保安亭", "岗亭",
    "银行", "ATM", "atm", "快递", "驿站", "物业", "公寓", "宿舍", "教学楼",
    "办公楼", "体育馆", "球场", "田径场", "医院", "药房", "超市", "便利店",
    "市场", "书店", "打印", "洗衣",
)


def _hit(text: str, words: Tuple[str, ...]) -> bool:
    return any(w in text for w in words)


def is_dining(name: str, tag: str = "", *, category_search: bool = False) -> Optional[bool]:
    """判断是否为餐饮门店。返回 True / False / None（未知）。

    - 明确命中"停车场/公交站/银行…"→ False，这类附属 POI 绝不能当餐馆推荐；
    - 明确命中"餐厅/粥/烧腊…"→ True；
    - 品类检索（用户本来就在搜吃的）→ 默认 True（餐饮品类下检索到的地点）；
    - 指定店名查询 → 返回 None（未知），不硬下结论，由匹配流程再判断。
    """
    text = f"{name} {tag}"
    if _hit(text, _NON_DINING_HINTS):
        return False
    if _hit(text, _DINING_HINTS):
        return True
    return True if category_search else None


# ---------------- POI 检索（uid 的起点） ----------------

def _normalize_poi(r: Dict[str, Any], *, category_search: bool = False) -> Optional[Dict[str, Any]]:
    """把百度 POI 原始结果转成标准门店事实；缺 uid 或坐标的丢弃。"""
    uid = r.get("uid")
    loc = r.get("location") or {}
    if not uid or not loc.get("lat") or not loc.get("lng"):
        return None
    olat, olng = _origin_pair()
    detail = r.get("detail_info") or {}
    tag = str(detail.get("tag") or r.get("tag") or "")
    name = r.get("name") or ""
    distance = _haversine_m(olat, olng, float(loc["lat"]), float(loc["lng"]))
    return {
        "uid": str(uid),
        "名称": name,
        "地址": r.get("address") or "",
        "行政区": " ".join(x for x in (r.get("province"), r.get("city"), r.get("area")) if x),
        "坐标": f"{loc['lat']},{loc['lng']}",
        "直线距离_米": distance,
        "类型标签": tag,
        "是餐饮": is_dining(name, tag, category_search=category_search),
        "在范围内": distance <= RADIUS_TOLERANCE_M,
        "范围说明": (
            f"出发点 {RADIUS_M / 1000:.0f} 公里内" if distance <= RADIUS_TOLERANCE_M
            else f"超出校区范围（直线 {distance} 米）"
        ),
        "来源": "百度地图 place/v2/search",
    }


def _poi_query(keyword: str, radius: Optional[int] = RADIUS_M) -> List[Dict[str, Any]]:
    """百度地点检索（带全局限速/缓存）。radius=None 表示不限半径（用于指定门店查询）。

    这是唯一直接碰地图检索的地方；测试会替换本函数。
    """
    olat, olng = _origin_pair()
    params: Dict[str, Any] = {"query": keyword, "scope": 2}
    if radius is not None:
        params.update({"location": f"{olat},{olng}", "radius": radius, "radius_limit": "true"})
    else:
        params["region"] = "广州"
    key = f"search|{keyword}|{radius}|{olat},{olng}"
    result = gw.request("/place/v2/search", params, kind="search", cache_key=key,
                        ttl=config.MAP_CACHE_TTL_SEARCH)
    _TLS.search_status = result.status
    _TLS.search_message = result.message
    if not result.ok:
        return []
    return (result.payload.get("results") or [])


def _search_status() -> str:
    return getattr(_TLS, "search_status", gw.OK)


def _query(keyword: str, radius: Optional[int] = RADIUS_M) -> List[Dict[str, Any]]:
    """调用 _poi_query 的兼容包装（测试夹具可能只接受一个参数）。"""
    try:
        return _poi_query(keyword, radius)
    except TypeError:
        return _poi_query(keyword)  # type: ignore[call-arg]


def _error_payload(status: str, keyword: str, message: str = "") -> Dict[str, Any]:
    """把地图异常翻译成人话——重点是：故障 ≠ 没有这家店。"""
    if status == gw.RATE_LIMITED:
        return {
            "状态": status,
            "提示": f"地图服务当前限流，暂时无法查询「{keyword}」。这是接口繁忙，不代表附近没有店。",
            "要求": "如实告诉用户『暂时查不了，稍后再试』；禁止改搜别的品类、禁止编造推荐。",
        }
    if status == gw.QUOTA_EXHAUSTED:
        return {
            "状态": status,
            "提示": f"今日地图检索额度已用完，暂时无法查询「{keyword}」。这同样不代表没有这家店。",
            "要求": "如实说明查询额度受限；可以用本地知识库已核实门店作答，禁止编造。",
        }
    if status == gw.AUTH_ERROR:
        return {
            "状态": status,
            "提示": "地图服务配置异常（AK 或服务权限），联网查询暂不可用。",
            "要求": "如实告知系统暂时无法联网查询；不要把它说成『没有门店』。",
        }
    return {
        "状态": status,
        "提示": f"地图查询失败：{message or gw.STATUS_LABEL.get(status, status)}",
        "要求": "如实告知暂时查不到真实门店；禁止凭记忆或想象编写门店信息。",
    }


def search_poi(
    keyword: str,
    poi_store: Optional[Dict[str, Dict[str, Any]]] = None,
    radius: int = RADIUS_M,
) -> str:
    """联网检索周边真实门店（默认 3 公里硬限制）。仅当知识库查不到时由 LLM 调用。

    返回的每家店带 uid——后续 verify_route、推荐卡片、地图跳转都用这个 uid
    引用同一家店。抽象口味词（清淡/辣/聚餐）自动展开为具体品类。
    """
    store = poi_store if poi_store is not None else {}
    if not config.MAP_ENABLED:
        return json.dumps(
            {"提示": "未配置百度地图 AK，无法联网检索真实门店", "建议": "在 .env 中填写 BAIDU_MAP_AK"},
            ensure_ascii=False,
        )

    expansions = {
        "清淡": ["粥", "粉面", "粤菜"],
        "辣": ["川菜", "湘菜", "麻辣烫"],
        "聚餐": ["粤菜", "火锅", "湘菜"],
        "外卖": ["快餐", "粉面", "烧腊"],
        "粤菜": ["粤菜", "烧腊", "肠粉"],
    }
    queries = expansions.get(keyword.strip(), [keyword])
    merged: List[Dict[str, Any]] = []
    seen: set[str] = set()
    rejected: List[Dict[str, Any]] = []
    statuses: List[str] = []

    for query in queries:
        results = _query(query, radius)
        status = _search_status()
        statuses.append(status)
        if status != gw.OK:
            continue
        for r in results:
            poi = _normalize_poi(r, category_search=True)
            if poi is None or poi["uid"] in seen:
                continue
            if poi.get("是餐饮") is False:
                rejected.append({"名称": poi["名称"], "原因": "附属设施（停车场/车站等），非餐饮门店"})
                continue
            if poi["直线距离_米"] > RADIUS_TOLERANCE_M + 200:
                rejected.append({"名称": poi["名称"], "直线距离_米": poi["直线距离_米"], "原因": "超出检索半径"})
                continue
            seen.add(poi["uid"])
            poi["查询品类"] = query
            merged.append(poi)
        if len(merged) >= 8:
            break

    if not merged:
        # 关键：区分「接口故障」与「确实没有」
        ok_queries = [s for s in statuses if s == gw.OK]
        if not ok_queries:
            payload = _error_payload(statuses[0] if statuses else gw.SERVER_ERROR, keyword)
            payload["已尝试品类"] = queries
            return json.dumps(payload, ensure_ascii=False)
        payload = {
            "状态": gw.NO_RESULT,
            "提示": f"查询成功，但出发点 {radius} 米内确实没有「{keyword}」的餐饮门店",
            "已尝试品类": queries,
            "建议": "换具体品类词（如粥、粉面、粤菜、烧腊）再试；确实没有就如实告知用户，不要推荐范围外的店",
        }
        if rejected:
            payload["已剔除"] = rejected
        return json.dumps(payload, ensure_ascii=False)

    # 写入会话级门店事实库（uid → 事实）
    for poi in merged:
        store.setdefault(poi["uid"], poi)

    out = {
        "状态": gw.OK,
        "提示": f"以下为出发点 {radius} 米内的真实门店（按直线距离升序），后续请用 uid 引用",
        "出发点": current_origin()["name"],
        "结果": sorted(merged, key=lambda p: p["直线距离_米"])[:8],
    }
    if rejected:
        out["已剔除"] = rejected
    return json.dumps(out, ensure_ascii=False)


# ---------------- 指定门店查询（不许擅自替换） ----------------

def _norm_name(text: str) -> str:
    return re.sub(r"[\s\-—·、,，.。()（）\[\]【】]", "", str(text)).lower()


def _name_similarity(a: str, b: str) -> float:
    """名称相似度：序列相似 + 包含关系加成，只用于找候选，不能单独认定同一家店。"""
    na, nb = _norm_name(a), _norm_name(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    ratio = difflib.SequenceMatcher(None, na, nb).ratio()
    if na in nb or nb in na:
        ratio = max(ratio, min(len(na), len(nb)) / max(len(na), len(nb)) * 0.95)
    return round(ratio, 3)


def _kb_lookup(name: str) -> Optional[Dict[str, Any]]:
    """本地真实知识库里的门店（正式名称 + 已核实别名）。"""
    for path in (config.RESTAURANTS_PATH, config.CANTEEN_PATH):
        for item in _load(path):
            official = item.get("正式名称") or item.get("名称") or item.get("窗口名") or ""
            aliases = item.get("别名") or []
            if isinstance(aliases, str):
                aliases = [aliases]
            for candidate in [official, *aliases]:
                if candidate and _name_similarity(name, candidate) >= 0.9:
                    return item
    return None


def find_restaurant(
    name: str,
    poi_store: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    """按用户给的店名查这家店本身（不是找替代品）。

    流程：本地知识库 → 3 公里内地图检索 → 全市检索（明确标注超出范围）。
    名称相似度只用于挑候选；没有明确匹配就如实说"不确定"，并请用户补充
    校门/地标/地址，绝不拿别的店顶替。
    """
    store = poi_store if poi_store is not None else {}
    store_name = str(name or "").strip()
    if not store_name:
        return json.dumps({"提示": "请告诉我店名，我帮你查具体位置和路线"}, ensure_ascii=False)

    # 1) 本地真实知识库（有别名与核实日期，最可信）
    #    但坐标缺失时不能就此收工：本地记录只能证明「这家店存在」，
    #    证明不了它在哪。此时继续联网检索，把坐标补上。
    kb = _kb_lookup(store_name)
    if kb and str(kb.get("坐标") or "").strip():
        coord = kb.get("坐标")
        return json.dumps(
            {
                "状态": "kb_hit",
                "来源": "本地真实知识库",
                "名称": kb.get("正式名称") or kb.get("名称"),
                "别名": kb.get("别名") or [],
                "地址": kb.get("地址") or "",
                "坐标": coord,
                "提示": "在本地知识库中确认了这家店；需要路线请用 plan_route 或 verify_route",
                "要求": "回答要基于这条记录，不要引入知识库里没有的店",
            },
            ensure_ascii=False,
        )

    if not config.MAP_ENABLED:
        return json.dumps(
            {"提示": "未配置百度地图，暂时无法确认这家店", "要求": "如实告知，不要换成别的店"},
            ensure_ascii=False,
        )

    # 2) 3 公里内检索 → 3) 全市检索
    attempts = [(RADIUS_M, f"{RADIUS_M / 1000:.0f} 公里内"), (None, "广州市范围")]
    candidates: List[Dict[str, Any]] = []
    scope_text = attempts[0][1]
    statuses: List[str] = []
    for radius, label in attempts:
        raw = _query(store_name, radius)
        status = _search_status()
        statuses.append(status)
        if status != gw.OK:
            continue
        normalized = [p for p in (_normalize_poi(r, category_search=False) for r in raw) if p]
        if normalized:
            candidates = normalized
            scope_text = label
            break

    if not candidates:
        ok_statuses = [s for s in statuses if s == gw.OK]
        if not ok_statuses:
            payload = _error_payload(statuses[0] if statuses else gw.SERVER_ERROR, store_name)
            payload["店名"] = store_name
            return json.dumps(payload, ensure_ascii=False)
        return json.dumps(
            {
                "状态": gw.NO_RESULT,
                "店名": store_name,
                "提示": f"地图上没有检索到「{store_name}」这家店（3 公里内与广州市范围都没有）。",
                "要求": (
                    "如实告诉用户没查到，并请对方补充校门/地标/地址再确认；"
                    "绝对不要用别的店当答案，也不要说成『附近没有好吃的』。"
                ),
            },
            ensure_ascii=False,
        )

    scored = []
    for poi in candidates:
        score = _name_similarity(store_name, poi["名称"])
        scored.append((score, poi))
    scored.sort(key=lambda x: (-x[0], x[1]["直线距离_米"]))
    best_score, best = scored[0]

    # 附属设施（如"XX-门前停车场"）只能当线索：先在全市范围再找一次餐馆本体
    if best.get("是餐饮") is False:
        if scope_text != "广州市范围":
            raw_far = _query(store_name, radius=None)
            if _search_status() == gw.OK:
                far_pois = [p for p in (_normalize_poi(r, category_search=False) for r in raw_far) if p]
                far_scored = sorted(
                    ((_name_similarity(store_name, p["名称"]), p) for p in far_pois
                     if p.get("是餐饮") is not False),
                    key=lambda x: (-x[0], x[1]["直线距离_米"]),
                )
                if far_scored and far_scored[0][0] >= 0.6:
                    score2, poi2 = far_scored[0]
                    poi2["允许超范围"] = True
                    store.setdefault(poi2["uid"], poi2)
                    return json.dumps(
                        {
                            "状态": "match",
                            "店名": store_name,
                            "uid": poi2["uid"],
                            "名称": poi2["名称"],
                            "匹配度": score2,
                            "地址": poi2["地址"],
                            "行政区": poi2["行政区"],
                            "坐标": poi2["坐标"],
                            "直线距离_米": poi2["直线距离_米"],
                            "范围": poi2["范围说明"],
                            "检索范围": "广州市范围（校区内只找到停车场）",
                            "提示": (
                                f"校区附近只检索到「{best['名称']}」（停车场），"
                                f"这家餐馆本体在更远处：{poi2['范围说明']}。"
                            ),
                            "要求": "明确告知用户距离很远（给出真实直线距离），不要把它包装成『附近推荐』。",
                        },
                        ensure_ascii=False,
                    )
        others = [p for s, p in scored if p.get("是餐饮") is not False and s >= 0.5]
        return json.dumps(
            {
                "状态": "auxiliary_only",
                "店名": store_name,
                "线索": {"名称": best["名称"], "地址": best["地址"], "uid": best["uid"], "匹配度": best_score},
                "其他候选": [{"名称": p["名称"], "地址": p["地址"], "uid": p["uid"]} for p in others[:3]],
                "提示": (
                    f"只找到「{best['名称']}」，这是停车场等附属地点，不是餐馆本体，不能作为门店推荐。"
                ),
                "要求": "把这条线索告诉用户，并请对方确认具体位置/校门；不要生成餐馆推荐卡片。",
            },
            ensure_ascii=False,
        )

    if best_score < 0.6:
        return json.dumps(
            {
                "状态": "uncertain",
                "店名": store_name,
                "候选": [
                    {"名称": p["名称"], "地址": p["地址"], "匹配度": s, "uid": p["uid"]}
                    for s, p in scored[:3]
                ],
                "提示": (
                    f"没有找到与「{store_name}」明确对应的门店，只有名称相近的候选，不能确认是同一家。"
                ),
                "要求": (
                    "如实说明不确定，并请用户补充校门/地标/地址；"
                    "相近候选只能作为『可能』提出，不能直接当作答案，也不能替用户改推荐其他店。"
                ),
            },
            ensure_ascii=False,
        )

    poi = best
    store.setdefault(poi["uid"], poi)
    return json.dumps(
        {
            "状态": "match",
            "店名": store_name,
            "uid": poi["uid"],
            "名称": poi["名称"],
            "匹配度": best_score,
            "地址": poi["地址"],
            "行政区": poi["行政区"],
            "坐标": poi["坐标"],
            "直线距离_米": poi["直线距离_米"],
            "范围": poi["范围说明"],
            "检索范围": scope_text,
            "是餐饮": poi.get("是餐饮"),
            "提示": "已确认这家店的身份；需要路线请调用 verify_route(uid)。若超出校区范围要明确告知用户。",
            "要求": "只描述这家店；不要替换成别的店，也不要顺手推荐其他店（除非用户明确要求替代方案）。",
        },
        ensure_ascii=False,
    )


# ---------------- 路线核实（事实只由这里产生） ----------------

def _baidu_direction(mode: str, dest_coord: str) -> Optional[Dict[str, Any]]:
    """路线规划 API（测试会替换本函数）。返回 {分钟, 距离_米} 或 None。"""
    olat, olng = _origin_pair()
    origin = f"{olat},{olng}"
    key = f"route|{mode}|{origin}|{dest_coord}"
    result = gw.request(
        f"/direction/v2/{mode}",
        {"origin": origin, "destination": dest_coord},
        kind="route",
        cache_key=key,
        ttl=config.MAP_CACHE_TTL_ROUTE,
    )
    _TLS.route_status = result.status
    _TLS.route_cached = result.cached
    if not result.ok:
        return None
    routes = (result.payload.get("result") or {}).get("routes") or []
    if not routes:
        _TLS.route_status = gw.NO_RESULT
        return None
    route = routes[0]
    return {"分钟": round(route.get("duration", 0) / 60), "距离_米": round(route.get("distance", 0))}


def _route_status() -> str:
    return getattr(_TLS, "route_status", gw.OK)


def _poi_detail_by_uid(uid: str) -> Optional[Dict[str, Any]]:
    """按 uid 精确查询门店详情（用于 LLM 传了未检索过的 uid 时兜底）。"""
    result = gw.request(
        "/place/v2/detail",
        {"uid": uid, "scope": 2},
        kind="search",
        cache_key=f"detail|{uid}",
        ttl=config.MAP_CACHE_TTL_SEARCH,
    )
    _TLS.search_status = result.status
    if not result.ok:
        return None
    return _normalize_poi(result.payload.get("result") or {}, category_search=False)


def verify_route(
    uid: str,
    poi_store: Optional[Dict[str, Dict[str, Any]]] = None,
    mode: str = "walking",
) -> str:
    """按 uid 核实该门店的真实路线，并保存路线事实。

    mode: walking（默认，只查步行）/ riding / driving / all。
    原则：用户不需要的交通方式不查——百度路线额度有限，一家店查三种方式
    会把额度消耗三倍。
    """
    store = poi_store if poi_store is not None else {}
    uid = str(uid).strip()
    poi = store.get(uid)
    if poi is None:
        poi = _poi_detail_by_uid(uid)  # 精确按 uid 定位，绝不按店名模糊匹配
        if poi is None:
            status = _search_status()
            if status not in (gw.OK, gw.NO_RESULT):
                return json.dumps(_error_payload(status, uid), ensure_ascii=False)
            return json.dumps(
                {
                    "uid": uid,
                    "提示": "暂无法确认该门店（uid 未在本次会话检索结果中，且按 uid 查详情失败）",
                    "要求": "请改用 search_poi 检索周边门店后再传 uid；禁止凭店名猜测路线耗时",
                },
                ensure_ascii=False,
            )
        store[uid] = poi

    if poi.get("是餐饮") is False:
        return json.dumps(
            {
                "uid": uid,
                "名称": poi.get("名称"),
                "提示": "这是停车场/车站等附属地点，不是餐饮门店，不能作为推荐",
                "要求": "不要为它生成推荐卡片",
            },
            ensure_ascii=False,
        )

    coord = poi.get("坐标")
    if not coord or not config.MAP_ENABLED:
        return json.dumps(
            {"uid": uid, "名称": poi.get("名称"), "提示": "暂无法确认路线（无坐标或未配置百度地图）", "要求": "不要编造耗时"},
            ensure_ascii=False,
        )

    modes = list(MODE_LABEL) if mode == "all" else [mode if mode in MODE_LABEL else "walking"]
    origin = current_origin()
    existing = poi.get("路线") or {}
    route_facts: Dict[str, Any] = {
        k: v for k, v in existing.items() if k in ("步行", "骑行", "打车", "查询时间", "来源")
    } if existing.get("起点") == origin["key"] else {}

    failures: List[str] = []
    cached_any = False
    fetched_new = False
    for m in modes:
        label = MODE_LABEL[m]
        if label in route_facts:  # 已经核实过这种方式，直接复用
            continue
        fact = _baidu_direction(m, str(coord))
        if fact:
            route_facts[label] = fact
            fetched_new = True
            cached_any = cached_any or bool(getattr(_TLS, "route_cached", False))
        else:
            failures.append(f"{label}:{_route_status()}")

    if not any(route_facts.get(MODE_LABEL[m]) for m in modes):
        status = _route_status()
        if status in (gw.OK, gw.NO_RESULT):
            return json.dumps(
                {
                    "uid": uid,
                    "名称": poi.get("名称"),
                    "提示": "暂无法确认路线（路线接口没有返回可用结果），禁止编造耗时",
                    "失败原因": failures,
                },
                ensure_ascii=False,
            )
        payload = _error_payload(status, poi.get("名称") or uid)
        payload.update({"uid": uid, "名称": poi.get("名称"), "失败原因": failures})
        payload["提示"] = "暂无法确认路线：" + payload["提示"]
        return json.dumps(payload, ensure_ascii=False)

    if fetched_new:
        # 有新查询才更新时间戳，避免把旧结果说成刚刚查的
        route_facts["查询时间"] = gw.now_text()
        route_facts["来源"] = (
            "百度地图 direction API（缓存命中）" if cached_any else "百度地图 direction API（实时）"
        )
    else:
        route_facts.setdefault("查询时间", gw.now_text())
        route_facts["来源"] = "百度地图 direction API（复用本次会话已核实结果）"
    route_facts["起点"] = origin["key"]
    poi["路线"] = route_facts  # 事实入库：后续卡片/跳转/过滤全部引用它

    return json.dumps(
        {
            "uid": uid,
            "目的地": poi["名称"],
            "地址": poi.get("地址"),
            "行政区": poi.get("行政区"),
            "出发点": origin["name"],
            "路线耗时": route_facts,
            "路线已核实": True,
            "失败方式": failures or None,
        },
        ensure_ascii=False,
    )


# ---------------- 知识库路线（真实数据专用） ----------------

def plan_route(place_name: str, mode: str = "walking") -> str:
    """知识库内真实门店的路线（须有坐标字段）。不做店名联网模糊定位。"""
    for path in (config.RESTAURANTS_PATH, config.CANTEEN_PATH):
        for item in _load(path):
            name = item.get("正式名称") or item.get("名称") or item.get("窗口名")
            if place_name in str(name or ""):
                coord = item.get("坐标")
                if not coord:
                    return json.dumps(
                        {"提示": f"知识库中「{place_name}」缺少坐标，暂无法规划路线"},
                        ensure_ascii=False,
                    )
                modes = list(MODE_LABEL) if mode == "all" else [mode if mode in MODE_LABEL else "walking"]
                route_facts: Dict[str, Any] = {}
                failures = []
                for m in modes:
                    fact = _baidu_direction(m, str(coord))
                    if fact:
                        route_facts[MODE_LABEL[m]] = fact
                    else:
                        failures.append(f"{MODE_LABEL[m]}:{_route_status()}")
                if not route_facts:
                    payload = _error_payload(_route_status(), str(name))
                    payload["失败原因"] = failures
                    return json.dumps(payload, ensure_ascii=False)
                route_facts["查询时间"] = gw.now_text()
                route_facts["来源"] = "百度地图 direction API"
                route_facts["起点"] = current_origin()["key"]
                walk = int(route_facts.get("步行", {}).get("分钟", 0) or 0)
                return json.dumps(
                    {
                        "目的地": name,
                        "出发点": current_origin()["name"],
                        "交通耗时_分钟": route_facts,
                        "值不值得去": "值得跑一趟" if 0 < walk <= 20 else "路程偏远，建议聚餐/约会场景再去",
                    },
                    ensure_ascii=False,
                )
    return json.dumps(
        {
            "提示": f"「{place_name}」不在本地真实知识库中（示例数据已隔离）",
            "下一步": "指定门店用 find_restaurant 确认身份，附近推荐用 search_poi 拿 uid，再用 verify_route(uid) 核实路线；禁止凭店名猜测",
        },
        ensure_ascii=False,
    )


# ---------------- 官方路线跳转链接 ----------------

def build_route_link(
    dest_name: str,
    dest_coord: str,
    mode: str = "walking",
    origin: Optional[Dict[str, Any]] = None,
) -> str:
    """百度官方 direction URI。

    按官方规范，起终点格式为 `latlng:纬度,经度|name:名称`，坐标是导航依据，
    名称用于展示；坐标系显式声明 bd09ll。
    文档：https://lbsyun.baidu.com/docs/webapi?title=mapadjustment%2Furi%2Fweb
    """
    preset = origin or current_origin()
    olat, olng = str(preset["coord"]).split(",")
    dlat, dlng = str(dest_coord).split(",")
    origin_part = f"latlng:{olat},{olng}|name:{quote(str(preset['name']))}"
    dest_part = f"latlng:{dlat},{dlng}|name:{quote(str(dest_name))}"
    return (
        "https://api.map.baidu.com/direction?"
        f"origin={origin_part}&destination={dest_part}"
        f"&mode={mode}&region=%E5%B9%BF%E5%B7%9E&coord_type=bd09ll&output=html&src=webapp.junshi"
    )
