"""美食检索官：在本地知识库中按条件筛选。

数据来源（合规）：大众点评/小红书内容的人工整理 + 学生问卷实测。
示例数据已隔离到 data/examples/（仅作字段模板参考），正式推荐
只使用 data/ 下的真实数据——没有真实数据时明确说明并引导联网，
绝不返回示例数据冒充真实门店。
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import config


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
    """按名称在真实知识库（周边餐厅+饭堂）中查找门店。"""
    for path in (config.RESTAURANTS_PATH, config.CANTEEN_PATH):
        for item in _load(path):
            name = item.get("名称") or item.get("店名") or item.get("窗口名")
            if place_name in str(name or ""):
                return item
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
        entry.setdefault("交通", {"步行": 0})
        entry["类别"] = "外卖"
        places.append(entry)

    return places


def _match(item: Dict[str, Any], query: str, taste: str, budget_max: float | None, max_minutes: int | None) -> bool:
    if budget_max is not None:
        try:
            if float(item.get("人均", 0)) > float(budget_max):
                return False
        except (TypeError, ValueError):
            pass
    if max_minutes is not None:
        try:
            walk = int(item.get("交通", {}).get("步行", 999))
            if item.get("类别") != "外卖" and walk > int(max_minutes):
                return False
        except (TypeError, ValueError):
            pass

    if query in {"外卖", "外卖推荐"}:
        if not item.get("可外卖") and item.get("类别") != "外卖":
            return False
    if query in {"聚餐", "聚会"}:
        try:
            if int(item.get("适合聚餐人数", 0) or 0) < 4:
                return False
        except (TypeError, ValueError):
            return False

    # 关键词/口味匹配：店名、菜系、招牌菜、标签、一句话点评
    # （"外卖/聚餐/聚会"是场景词而非内容词，上面已按场景字段过滤，这里跳过）
    if query and query not in {"外卖", "外卖推荐", "聚餐", "聚会"}:
        haystack = " ".join(
            str(item.get(field, ""))
            for field in ("名称", "店名", "窗口名", "菜系", "一句话点评", "口味标签")
        ) + " " + " ".join(str(x) for x in item.get("招牌菜", []) or [])
        if query.lower() not in haystack.lower():
            return False

    if taste:
        tag = f"{item.get('菜系', '')}{' '.join(str(t) for t in item.get('口味标签', []) or [])}{item.get('一句话点评', '')}"
        if taste not in tag:
            return False
    return True


def food_search(query: str, taste: str = "", budget_max: float | None = None, max_minutes: int | None = None) -> str:
    """检索知识库，返回 Top5（按学生评分降序、步行时间升序）。"""
    if budget_max is not None:
        budget_max = float(budget_max)
    if max_minutes is not None:
        max_minutes = int(max_minutes)

    hits = [p for p in _all_places() if _match(p, query, taste, budget_max, max_minutes)]
    hits.sort(key=lambda p: (-float(p.get("学生评分", 0) or 0), int(p.get("交通", {}).get("步行", 0) or 0)))

    if not hits:
        if not _all_places():
            return json.dumps(
                {
                    "提示": "本地知识库暂无真实数据（示例数据已隔离，不用于正式推荐）",
                    "下一步": "必须改用 search_poi 联网检索周边真实门店，检索结果会带 uid/地址/坐标/距离，再用 verify_route 核实路线",
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {"提示": "知识库有真实数据但无匹配结果", "建议": "可用 search_poi 联网搜索，或放宽口味/预算/时间约束"},
            ensure_ascii=False,
        )

    return json.dumps(
        {
            "结果": [
                {
                    "名称": p.get("名称") or p.get("店名") or p.get("窗口名"),
                    "类别": p.get("类别"),
                    "菜系": p.get("菜系"),
                    "人均": p.get("人均"),
                    "招牌菜": p.get("招牌菜"),
                    "交通": p.get("交通"),
                    "营业时间": p.get("营业时间"),
                    "高峰时段": p.get("高峰时段"),
                    "适合聚餐人数": p.get("适合聚餐人数"),
                    "学生评分": p.get("学生评分"),
                    "避雷": p.get("避雷"),
                    "一句话点评": p.get("一句话点评"),
                }
                for p in hits[:5]
            ]
        },
        ensure_ascii=False,
    )
