"""按店名补门店坐标：调用百度 place/v2/search 检索，把坐标/地址写回 data/restaurants.json。

只用于数据维护，不参与线上问答链路。

用法：
    python tools/enrich_coords.py --dry-run     # 只查询、打印结果，不写盘
    python tools/enrich_coords.py               # 查询并写盘（会先备份原文件）
    python tools/enrich_coords.py --only 潮埕    # 只补指定门店

设计要点：
- 命中判定从严：候选名称必须包含门店核心名（括号、空格全部归一后再比）；
- 名称带分店后缀（如「太二（番禺天河城）」）的门店：用「核心名 + 后缀」做检索词，
  距离上限放宽到 12km（番禺各商圈直线 6–11km，属正常；珠江新城/天河区 15km+ 仍会被拒）；
- 无后缀门店（周边 2–3km 那批）：沿用 6km 上限。
- 走 map_gateway，自动享受限速、缓存、日配额与错误分类。
- 写盘前备份为 data/restaurants.json.bak。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from tools import map_gateway as gw
from tools import route_planner as rp

CAMPUS_ACCEPT_M = 6000     # 无分店后缀门店：离校区直线距离上限
BRANCH_ACCEPT_M = 12000    # 带分店后缀门店：离校区直线距离上限（番禺/琶洲各商圈 6–11km）


def _load() -> dict:
    with open(config.RESTAURANTS_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _save(payload: dict) -> None:
    shutil.copyfile(config.RESTAURANTS_PATH, config.RESTAURANTS_PATH + ".bak")
    with open(config.RESTAURANTS_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _clean(s: str) -> str:
    """去掉空格与各种括号，用于名称比对。"""
    return re.sub(r"[（）()\s]", "", str(s or ""))


def _split_name(name: str):
    """「太二（番禺天河城）」→ 核心名「太二」+ 后缀「番禺天河城」。"""
    m = re.match(r"^(.+?)[（(](.+)[)）]$", str(name).strip())
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return str(name).strip(), ""


def _candidates(name: str) -> list:
    """检索词尝试顺序：核心名+分店后缀 → 全名 → 纯中文名。"""
    core, suffix = _split_name(name)
    tries = []
    if suffix:
        tries.append(f"{core} {suffix}")
    tries.append(name)
    stripped = "".join(ch for ch in name if "\u4e00" <= ch <= "\u9fff")
    if stripped and stripped not in tries:
        tries.append(stripped)
    for keyword in tries:
        raw = rp._query(keyword, None)
        if rp._search_status() != gw.OK:
            return [], rp._search_status()
        items = [p for p in (rp._normalize_poi(r, category_search=False) for r in raw) if p]
        if items:
            return items, gw.OK
    return [], gw.NO_RESULT


FACILITY_PAT = re.compile(r"停车场|接送点|充电|加油|有限公司|分公司|集团公司|仓库")


def _pick(name: str, items: list):
    """从严挑一个候选：名称归一后必须包含核心名；同分店后缀者优先；再取最近。

    过滤规则（都是真踩过的坑）：
    - 停车场/接送点/公司注册名等附属设施一律不收（「B记鱼粥-门前停车场」）；
    - 明确非餐饮的 POI 不收（is_dining 为 False）；
    - 带分店后缀的门店：候选名里没有该后缀时，只接受 8km 内的（防「山姆的牛
      琶洲保利店」错配到 10km 外的哈街总店这种跨商圈错配）。
    """
    core, suffix = _split_name(name)
    core_c, suffix_c = _clean(core), _clean(suffix)
    accept_m = BRANCH_ACCEPT_M if suffix else CAMPUS_ACCEPT_M

    matched = []
    for item in items:
        cand = _clean(item.get("名称"))
        if not cand or not core_c or core_c not in cand:
            continue
        if FACILITY_PAT.search(cand):
            continue
        if item.get("是餐饮") is False:
            continue
        matched.append(item)
    if not matched:
        return None, "店名对不上", accept_m

    # 同分店后缀的优先（如候选名里带「番禺天河城」）
    if suffix_c:
        preferred = [i for i in matched if suffix_c in _clean(i.get("名称"))]
        pool = preferred or matched
    else:
        pool = matched
    pool.sort(key=lambda i: i.get("直线距离_米") or 10 ** 9)
    best = pool[0]
    dist = best.get("直线距离_米") or 10 ** 9
    # 后缀门店但候选名不含后缀 → 只信 8km 内的
    if suffix_c and suffix_c not in _clean(best.get("名称")) and dist > 8000:
        return None, f"最近的同名店（不含分店后缀）直线 {dist} 米，疑似跨商圈错配", accept_m
    if dist > accept_m:
        return None, f"最近的同名家直线 {dist} 米，超出 {accept_m} 米", accept_m
    exact = _clean(best.get("名称")) == _clean(name)
    return best, ("完全同名" if exact else ("名称包含" if not suffix_c or suffix_c in _clean(best.get("名称")) else "核心名命中")), accept_m


def main() -> int:
    parser = argparse.ArgumentParser(description="补门店坐标")
    parser.add_argument("--dry-run", action="store_true", help="只打印不写盘")
    parser.add_argument("--only", default="", help="只处理名称包含该关键词的门店")
    args = parser.parse_args()

    if not config.MAP_ENABLED:
        print("未配置百度地图 AK：请在 .env 里填写 BAIDU_MAP_AK 后重试。")
        return 2

    payload = _load()
    stores = payload.get("restaurants") or []
    if not stores:
        print("门店库为空，先填 data/restaurants.json")
        return 1

    filled, skipped, failed = [], [], []
    for store in stores:
        name = str(store.get("名称") or "").strip()
        if not name or (args.only and args.only not in name):
            continue
        if store.get("坐标") and not args.only:
            print(f"  跳过 {name}：已有坐标 {store['坐标']}")
            continue

        items, status = _candidates(name)
        if status != gw.OK:
            failed.append((name, status))
            print(f"  [失败] {name}：地图返回 {status}")
            continue
        best, why, _accept = _pick(name, items)
        if best is None:
            skipped.append((name, why))
            print(f"  [留空] {name}：{why}")
            continue

        store["坐标"] = best["坐标"]
        store["地址"] = best.get("地址") or store.get("地址") or ""
        store["行政区"] = best.get("行政区") or ""
        store["地图uid"] = best.get("uid") or ""
        store["直线距离_米"] = best.get("直线距离_米")
        store["坐标核实日期"] = gw.now_text()[:10]
        filled.append((name, best["名称"], best["坐标"], best.get("直线距离_米")))
        print(f"  [写入] {name} → {best['名称']} | {best['坐标']} | 直线 {best.get('直线距离_米')} 米（{why}）")

    snapshot = gw.snapshot()
    print(f"\n合计：写入 {len(filled)} 家，留空 {len(skipped)} 家，失败 {len(failed)} 家")
    print(f"地图额度：今日检索 {snapshot.get('daily_search')}/{snapshot.get('daily_search_limit')}，缓存命中 {snapshot.get('cache_hits')} 次")

    if args.dry_run:
        print("（dry-run，未写盘）")
    elif filled:
        _save(payload)
        print(f"已写入 {config.RESTAURANTS_PATH}（备份 .bak）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

