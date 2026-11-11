"""知识库检索层：门店检索 + 评价证据 + 口碑统计。

分工（与 core.py 的「事实注入」铁律一致）：
- 本模块负责**检索与统计**：BM25 找门店、找评价原文，评分均值/样本数/分布全部
  由代码算出来，带来源与日期；
- LLM 只负责**解释**：推荐理由、值不值这类文字；
- 谁都不能绕过本模块编数字——这是"知识库"区别于"模型印象"的地方。

演示模式（KB_DEMO=1）：
读 data/examples/demo_*.json 的演示语料，接口返回 demo=True，前端会挂横幅。
正式模式只读 data/ 下的真实数据；真实库为空时如实返回空，绝不拿示例冒充。
"""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any, Dict, List, Optional

import config
from tools.food_search import _all_places
from tools.kb_index import BM25Index, normalize, normalize_scores

_CACHE: Dict[str, Any] = {"signature": None}

_DEMO_STORES = os.path.join(config.DATA_DIR, "examples", "demo_stores.json")
_DEMO_REVIEWS = os.path.join(config.DATA_DIR, "examples", "demo_reviews.jsonl")


# ---------- 语料读取 ----------
def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue  # 单行脏数据不能让整个知识库不可用
            if isinstance(item, dict):
                item.setdefault("id", f"line{line_no}")
                rows.append(item)
    return rows


def is_demo() -> bool:
    return bool(config.KB_DEMO)


def load_stores() -> List[Dict[str, Any]]:
    if is_demo():
        if not os.path.exists(_DEMO_STORES):
            return []
        with open(_DEMO_STORES, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return list(payload.get("stores") or [])
    return [s for s in _all_places() if _name(s)]


def load_reviews() -> List[Dict[str, Any]]:
    return _read_jsonl(_DEMO_REVIEWS if is_demo() else config.KB_REVIEWS_PATH)


# ---------- 索引 ----------
def _signature() -> str:
    paths = [_DEMO_STORES, _DEMO_REVIEWS] if is_demo() else [
        config.RESTAURANTS_PATH, config.CANTEEN_PATH, config.DELIVERY_PATH, config.KB_REVIEWS_PATH,
    ]
    parts = []
    for path in paths:
        try:
            parts.append(f"{path}:{os.path.getmtime(path):.0f}:{os.path.getsize(path)}")
        except OSError:
            parts.append(f"{path}:missing")
    return ("demo|" if is_demo() else "real|") + "|".join(parts)


def reset_cache() -> None:
    _CACHE["signature"] = None


def _name(store: Dict[str, Any]) -> str:
    return str(store.get("名称") or store.get("店名") or store.get("窗口名") or "").strip()


def _store_text(store: Dict[str, Any]) -> str:
    """店名重复两次，让店名命中优先于描述词。"""
    name = _name(store)
    parts: List[str] = [name, name]
    for field in ("别名", "菜系", "一句话点评", "类别", "地址"):
        if store.get(field):
            parts.append(str(store[field]))
    for field in ("招牌菜", "口味标签"):
        parts.extend(str(v) for v in (store.get(field) or []))
    return " ".join(parts)


def _review_text(review: Dict[str, Any]) -> str:
    parts = [str(review.get("text") or "")]
    parts.extend(str(v) for v in (review.get("tags") or []))
    if review.get("meal"):
        parts.append(str(review["meal"]))
    return " ".join(parts)


def _rating_of(review: Dict[str, Any]) -> Optional[float]:
    try:
        value = review.get("rating")
        return round(float(value), 2) if value is not None else None
    except (TypeError, ValueError):
        return None


def _ensure_index() -> None:
    signature = _signature()
    if _CACHE.get("signature") == signature:
        return

    stores = [s for s in load_stores() if _name(s)]
    reviews = [r for r in load_reviews() if str(r.get("store") or "").strip()]

    by_store: Dict[str, List[Dict[str, Any]]] = {}
    for review in reviews:
        by_store.setdefault(normalize(review.get("store")), []).append(review)

    store_docs = [
        {"id": normalize(_name(s)), "store": _name(s), "text": _store_text(s), "meta": s} for s in stores
    ]
    review_docs = [
        {"id": str(r.get("id")), "store": str(r.get("store")), "text": _review_text(r), "meta": r}
        for r in reviews
    ]

    _CACHE.update(
        signature=signature,
        stores=stores,
        reviews=reviews,
        store_index=BM25Index(store_docs),
        review_index=BM25Index(review_docs),
        by_store=by_store,
        by_name={normalize(_name(s)): s for s in stores},
    )


# ---------- 口碑聚合（确定性，模型无法改写） ----------
def _reviews_for(name: str) -> List[Dict[str, Any]]:
    _ensure_index()
    key = normalize(name)
    if key in _CACHE["by_store"]:
        return _CACHE["by_store"][key]
    for store_key, rows in _CACHE["by_store"].items():  # 容错「南村店」这类后缀差异
        if store_key and (store_key in key or key in store_key):
            return rows
    return []


def _aggregate(name: str) -> Optional[Dict[str, Any]]:
    reviews = _reviews_for(name)
    if not reviews:
        return None
    values = [v for v in (_rating_of(r) for r in reviews) if v is not None]
    average = round(sum(values) / len(values), 2) if values else None
    distribution = Counter(str(int(round(v))) for v in values)

    good, bad = Counter(), Counter()
    for review in reviews:
        score = _rating_of(review)
        target = good if (score is None or score >= 3.5) else bad
        for tag in review.get("tags") or []:
            target[str(tag)] += 1

    return {
        "评分": average,
        "样本": len(reviews),
        "评分样本": len(values),
        "分布": {star: distribution.get(star, 0) for star in ("5", "4", "3", "2", "1")},
        "好评要点": [t for t, _ in good.most_common(3)],
        "差评要点": [t for t, _ in bad.most_common(3)],
    }


def _source_label(name: str) -> str:
    """按评价来源判断这个分数是问卷（学生填的）还是攻略整理（单人观点）。"""
    sources = [str(r.get("source") or "") for r in _reviews_for(name)]
    if not sources:
        return "学生评价"
    if any("问卷" in s for s in sources):
        return "学生评价"
    if all(("攻略" in s) or ("整理" in s) for s in sources if s):
        return "攻略整理"
    return "学生评价"


def _quote(name: str, query: str = "") -> Optional[Dict[str, Any]]:
    """挑一条最能代表口碑的评价：有查询词就按相关度，否则取信息量最大的一条。"""
    _ensure_index()
    reviews = _reviews_for(name)
    if not reviews:
        return None
    picked = None
    if query.strip():
        # 优先用相关度高的正面评价做门面；负面评价留给详情页与「避雷」字段
        for doc, _score in _CACHE["review_index"].search(query, top_k=12):
            if normalize(doc.get("store")) != normalize(name):
                continue
            review = doc.get("meta") or {}
            score = _rating_of(review)
            if score is not None and score < 4:
                continue
            picked = review
            break
    if picked is None:
        positives = [r for r in reviews if (_rating_of(r) or 0) >= 4]
        pool = positives or reviews
        picked = max(pool, key=lambda r: (len(str(r.get("text") or "")), str(r.get("id"))))
    return {
        "片段": str(picked.get("text") or "")[:70],
        "评分": _rating_of(picked),
        "日期": picked.get("date") or "",
        "来源": picked.get("source") or "学生评价",
    }


# ---------- 门店检索 ----------
def _walk_minutes(store: Dict[str, Any]) -> Optional[int]:
    try:
        return int((store.get("交通") or {}).get("步行"))
    except (TypeError, ValueError):
        return None


def _store_tags(store: Dict[str, Any]) -> List[str]:
    return [str(t) for t in (store.get("口味标签") or [])]


_SCENE_ALIASES = {"饭堂窗口": "食堂窗口", "饭堂": "食堂窗口", "食堂": "食堂窗口", "周边": "周边堂食"}


def _scene(store: Dict[str, Any]) -> str:
    """统一场景取值：food_search 里饭堂写作「饭堂窗口」，对外一律用「食堂窗口」。"""
    raw = str(store.get("类别") or "").strip()
    return _SCENE_ALIASES.get(raw, raw)


def _passes(store: Dict[str, Any], scene: str, max_price, max_walk, tag: str) -> bool:
    if scene and _scene(store) != _SCENE_ALIASES.get(scene, scene):
        return False
    if max_price:
        try:
            if float(store.get("人均") or 0) > float(max_price):
                return False
        except (TypeError, ValueError):
            pass
    if max_walk:
        walk = _walk_minutes(store)
        if walk is not None and walk > int(max_walk):
            return False
    if tag:
        haystack = " ".join(_store_tags(store)) + " " + " ".join(
            str(t) for r in _reviews_for(_name(store)) for t in (r.get("tags") or [])
        )
        if tag not in haystack:
            return False
    return True


def search_stores(
    query: str = "",
    scene: str = "",
    max_price: Optional[float] = None,
    max_walk: Optional[int] = None,
    tag: str = "",
    sort: str = "relevance",
    top_k: int = 20,
) -> List[Dict[str, Any]]:
    _ensure_index()
    stores = _CACHE["stores"]
    if not stores:
        return []

    pairs: List[tuple] = []
    if query.strip():
        for doc, score in _CACHE["store_index"].search(query, top_k=max(top_k * 4, 30)):
            store = doc.get("meta") or {}
            if _passes(store, scene, max_price, max_walk, tag):
                pairs.append((store, score))
    else:
        pairs = [(s, 0.0) for s in stores if _passes(s, scene, max_price, max_walk, tag)]

    relevance = normalize_scores([(s, sc) for s, sc in pairs])

    items: List[Dict[str, Any]] = []
    for (store, _score), rel in zip(pairs, relevance):
        stats = _aggregate(_name(store)) or {}
        rating = stats.get("评分")
        if rating is None:
            rating = store.get("学生评分")
        samples = stats.get("样本", 0)
        items.append(
            {
                "名称": _name(store),
                "类别": _scene(store),
                "菜系": store.get("菜系"),
                "商圈": store.get("商圈"),
                "人均": store.get("人均"),
                "招牌菜": store.get("招牌菜"),
                "口味标签": _store_tags(store),
                "步行分钟": _walk_minutes(store),
                "地址": store.get("地址"),
                "营业时间": store.get("营业时间"),
                "适合聚餐人数": store.get("适合聚餐人数"),
                "可外卖": store.get("可外卖"),
                "评分": rating,
                "评分样本": samples,
                "评分来源": _source_label(_name(store)) if stats.get("评分") is not None else ("攻略整理" if rating is not None else None),
                "样本不足": bool(samples and samples < config.KB_MIN_SAMPLES),
                "好评要点": stats.get("好评要点", []),
                "差评要点": stats.get("差评要点", []),
                "摘录": _quote(_name(store), query),
                "相关度": round(rel, 3),
            }
        )

    if sort == "rating":
        items.sort(key=lambda i: (-(i["评分"] or 0), -(i["评分样本"] or 0)))
    elif sort == "distance":
        items.sort(key=lambda i: (i["步行分钟"] if i["步行分钟"] is not None else 999))
    elif sort == "price":
        items.sort(key=lambda i: (i["人均"] if i["人均"] is not None else 9999))
    else:
        items.sort(key=lambda i: (-i["相关度"], -(i["评分"] or 0)))
    return items[:top_k]


def store_detail(name: str) -> Optional[Dict[str, Any]]:
    """单店详情：全部评价 + 评分分布，供弹窗渲染。"""
    _ensure_index()
    store = _CACHE["by_name"].get(normalize(name))
    if store is None:
        for key, candidate in _CACHE["by_name"].items():
            if key and (key in normalize(name) or normalize(name) in key):
                store = candidate
                break
    if store is None:
        return None

    stats = _aggregate(_name(store)) or {}
    detail_rating = stats.get("评分")
    if detail_rating is None:
        detail_rating = store.get("学生评分")
    reviews = sorted(
        _reviews_for(_name(store)),
        key=lambda r: (str(r.get("date") or ""), str(r.get("id"))),
        reverse=True,
    )
    return {
        "名称": _name(store),
        "类别": _scene(store),
        "菜系": store.get("菜系"),
        "商圈": store.get("商圈"),
        "人均": store.get("人均"),
        "地址": store.get("地址"),
        "营业时间": store.get("营业时间"),
        "招牌菜": store.get("招牌菜"),
        "步行分钟": _walk_minutes(store),
        "评分": detail_rating,
        "评分来源": _source_label(_name(store)) if stats.get("评分") is not None else ("攻略整理" if detail_rating is not None else None),
        "样本": stats.get("样本", 0),
        "分布": stats.get("分布", {}),
        "好评要点": stats.get("好评要点", []),
        "差评要点": stats.get("差评要点", []),
        "评价": [
            {
                "片段": str(r.get("text") or ""),
                "评分": _rating_of(r),
                "日期": r.get("date") or "",
                "来源": r.get("source") or "学生评价",
                "餐段": r.get("meal") or "",
            }
            for r in reviews
        ],
    }


def kb_status() -> Dict[str, Any]:
    """知识库概览：给前端顶部显示规模，也给 /api/status 复用。"""
    _ensure_index()
    stores = _CACHE["stores"]
    reviews = _CACHE["reviews"]
    return {
        "demo": is_demo(),
        "门店数": len(stores),
        "评价数": len(reviews),
        "有评价门店数": len({normalize(r.get("store")) for r in reviews}),
        "最小样本数": config.KB_MIN_SAMPLES,
    }
