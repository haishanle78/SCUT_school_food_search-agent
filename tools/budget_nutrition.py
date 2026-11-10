"""预算营养师：预算核算 + 荤素搭配建议。"""

from __future__ import annotations

import json
from typing import Any, Dict

from tools.food_search import _find_place


def estimate_budget(place_name: str, party_size: int = 1) -> str:
    place = _find_place(place_name)
    if place is None:
        return json.dumps(
            {
                "提示": f"「{place_name}」不在本地真实知识库中",
                "要求": "人均由你按品类常识预估，并在最终回答中注明'预估'；不要声称来自点评数据",
            },
            ensure_ascii=False,
        )

    per_person = float(place.get("人均", 0) or 0)
    size = int(party_size) if party_size else 1
    walk = int(place.get("交通", {}).get("步行", 0) or 0)

    traffic_fee = "0 元（步行可达）" if walk <= 15 else "往返打车约 30-40 元（偏僻路段建议拼车）"
    return json.dumps(
        {
            "店铺": place_name,
            "人均": per_person,
            "人数": size,
            "餐费合计_元": round(per_person * size),
            "交通提示": traffic_fee,
        },
        ensure_ascii=False,
    )


def health_hint(place_name: str) -> str:
    place = _find_place(place_name)
    if place is None:
        return json.dumps(
            {
                "提示": f"「{place_name}」不在本地真实知识库中，无招牌菜数据",
                "要求": "搭配建议由你按菜系常识给出，注明是常识建议",
            },
            ensure_ascii=False,
        )

    dishes = place.get("招牌菜") or []
    if isinstance(dishes, str):
        dishes = [dishes]
    dishes = [str(d) for d in dishes]

    meat_keywords = ("肉", "鱼", "虾", "鸡", "牛", "羊", "排骨", "烧", "烤", "煲")
    veg_keywords = ("菜", "蔬", "豆腐", "菌", "瓜", "笋")
    meat_count = sum(1 for d in dishes if any(k in d for k in meat_keywords))
    veg_count = sum(1 for d in dishes if any(k in d for k in veg_keywords))

    if meat_count >= 2 and veg_count == 0:
        advice = f"招牌菜偏肉（{meat_count} 道硬菜），建议加 1-2 个绿叶菜或豆腐，荤素均衡还解腻。"
    elif veg_count >= 2 and meat_count == 0:
        advice = "招牌偏素，经常健身的同学记得补蛋白，加个荤菜。"
    else:
        advice = "荤素搭配尚可，正常点即可；记得别全点招牌菜，看人数加减。"

    return json.dumps({"店铺": place_name, "招牌菜": dishes, "搭配建议": advice}, ensure_ascii=False)
