"""天气统一网关：复用百度地图 AK 的天气 API，为 agent 提供实时天气感知。

为什么独立成网关而不并入 map_gateway：
  天气接口的免费额度与地点检索/路线规划相互独立；若走 map_gateway.request()，
  调用会被计入"地点检索"日配额（默认 80 次/日），而天气每 30 分钟刷新一次、
  全站共享一份缓存也会日消耗 ~48 次，等于吃掉大半检索额度。所以单独建轻量网关：
  只保留缓存 + 错误分类 + AK 脱敏，不需要令牌桶/熔断/single-flight
  （天气 QPS 宽松、单 key 缓存天然合并了全站请求）。

决策翻译在后端完成（与"事实字段后端注入"同一铁律）：
  "小雨" → 外卖优先、步行宽容度 10 分钟，由本模块的确定性规则给出，
  模型只负责在回答中引用，不允许自行从天气现象推断策略。

失败哲学：天气是增强项不是依赖项。任何异常（无 AK / 超时 / 接口故障）
都以非 ok 状态返回，调用方（core.py）据此静默跳过天气注入，主流程不受影响。
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Dict, Tuple
from urllib import error as urlerror
from urllib import request as urlrequest
from urllib.parse import urlencode

import config

# 错误分类直接复用地图网关的常量与状态码映射（百度系接口状态码语义一致）
from tools.map_gateway import (  # noqa: F401  (re-export 供测试断言)
    AUTH_ERROR,
    NETWORK_ERROR,
    OK,
    QUOTA_EXHAUSTED,
    RATE_LIMITED,
    SERVER_ERROR,
    TIMEOUT,
    classify,
    now_text,
)

logger = logging.getLogger("weather_gateway")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

API_PATH = "/weather/v1/"

# ---------------- 内部状态 ----------------
_lock = threading.RLock()
_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_stats: Dict[str, int] = {"calls": 0, "cache_hits": 0, "errors": 0}

_RAIN_WORDS = ("雨", "雷", "雪", "冰雹", "冻")


def _redacted(params: Dict[str, Any]) -> Dict[str, Any]:
    """日志脱敏：AK 永远打码。"""
    return {k: ("***" if k == "ak" else v) for k, v in params.items()}


# ---------------- 缓存 ----------------
def _cache_get(key: str) -> Dict[str, Any] | None:
    with _lock:
        item = _cache.get(key)
        if not item:
            return None
        expire, data = item
        if expire < time.time():
            _cache.pop(key, None)
            return None
    return {**data, "缓存命中": True}


def _cache_put(key: str, data: Dict[str, Any]) -> None:
    ttl = max(60, config.WEATHER_CACHE_TTL)
    with _lock:
        _cache[key] = (time.time() + ttl, {k: v for k, v in data.items() if k != "缓存命中"})


# ---------------- 决策翻译（确定性规则，后端职责） ----------------
def decide(
    now_text_val: str,
    temp: float,
    today_day_text: str,
    upcoming_rain: bool = False,
) -> Dict[str, Any]:
    """天气现象 → 推荐策略。规则写死在这里，模型不允许自行解释天气。

    upcoming_rain 来自未来 3 小时逐时预报（降水概率/降水量/现象词），
    这样"现在晴但一小时后下雨"也能提前给出外卖优先策略。
    """
    combined = f"{now_text_val}{today_day_text}"
    raining_now = any(w in combined for w in _RAIN_WORDS)
    if raining_now or upcoming_rain:
        if raining_now:
            hint = "正在下雨，建议优先外卖，或选步行 10 分钟内的近店；去堂食记得提醒用户带伞"
        else:
            hint = "现在没下雨，但未来 3 小时可能有雨，建议优先外卖或步行 10 分钟内的近店，出门提醒带伞"
        return {
            "场景": "rain",
            "外卖优先": True,
            "步行宽容度": 10,
            "提示文案": hint,
        }
    if temp >= 33:
        return {
            "场景": "hot",
            "外卖优先": False,
            "步行宽容度": 15,
            "提示文案": "天气炎热（≥33°C），优先步行 15 分钟内的近店，远店建议打车，主动提醒防暑",
        }
    if temp <= 5:
        return {
            "场景": "cold",
            "外卖优先": False,
            "步行宽容度": 15,
            "提示文案": "气温较低（≤5°C），优先近店或热食，远店建议打车",
        }
    return {"场景": "normal", "外卖优先": False, "步行宽容度": None, "提示文案": "天气适宜出行，按用户需求正常推荐"}


def _num(value: Any, default: float = 0.0) -> float:
    """百度字段可能是数字或字符串，缺失时给默认值，绝不让解析异常冒泡。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _hour_brief(hour: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "时间": hour.get("data_time") or "",
        "现象": str(hour.get("text") or ""),
        "气温": hour.get("temp_fc"),
        "降水概率": hour.get("pop"),
        "降水量": hour.get("prec_1h"),
    }


def _translate(payload: Dict[str, Any]) -> Dict[str, Any]:
    """百度原始响应 → 统一结构。缺关键字段时按 SERVER_ERROR 处理，不猜。

    百度天气 v1（data_type=all）真实结构：
      result.now            {text, temp, feels_like, rh, wind_class, ...}
      result.forecasts[]    7 天预报 {date, text_day, text_night, high, low, ...}
      result.forecast_hours[] 24 小时逐时 {data_time, text, temp_fc, pop, prec_1h}
      result.alerts[]       气象预警
    """
    result = payload.get("result") or {}
    now = result.get("now") or {}
    location = result.get("location") or {}
    # forecasts 为官方字段；forecast 作兼容兜底（部分版本/镜像返回单数）
    forecasts = result.get("forecasts") or result.get("forecast") or []
    hours = result.get("forecast_hours") or []
    alerts = result.get("alerts") or []
    if not now or "temp" not in now or "text" not in now:
        return {"状态": SERVER_ERROR, "说明": "天气接口返回缺少 now 字段"}

    try:
        temp = float(now.get("temp"))
    except (TypeError, ValueError):
        return {"状态": SERVER_ERROR, "说明": "气温字段无法解析"}

    today = forecasts[0] if forecasts else {}
    today_day_text = str(today.get("text_day") or "")

    next3 = hours[:3]
    upcoming_rain = any(
        any(w in str(h.get("text") or "") for w in _RAIN_WORDS)
        or _num(h.get("pop")) >= 60
        or _num(h.get("prec_1h")) > 0
        for h in next3
    )
    alert_titles = [str(a.get("title") or a.get("type") or "") for a in alerts][:2]
    alert_titles = [t for t in alert_titles if t]

    return {
        "状态": OK,
        "说明": "查询成功",
        "位置": location.get("name") or "校区",
        "查询时间": now_text(),
        "现在": {
            "现象": str(now.get("text") or ""),
            "气温": temp,
            "体感": now.get("feels_like"),
            "湿度": now.get("rh"),
            "风向": now.get("wind_dir") or "",
            "风力": now.get("wind_class") or "",
        },
        "今日": {
            "白天": today_day_text,
            "夜间": str(today.get("text_night") or ""),
            "最高温": today.get("high"),
            "最低温": today.get("low"),
        },
        "未来三小时": [_hour_brief(h) for h in next3],
        "预警": alert_titles,
        "建议": decide(str(now.get("text") or ""), temp, today_day_text, upcoming_rain=upcoming_rain),
    }


# ---------------- HTTP ----------------
def _do_http(params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """真正发请求的地方（测试里会被替换）。返回 (统一状态, 百度原始响应)。"""
    query = dict(params)
    # 天气接口复用地图 AK 池：优先用池里第一把，第一把被熔断/失效时自动换下一把。
    # 天气额度与检索/路线相互独立，不占那两项的日配额。
    pool = list(config.BAIDU_MAP_AKS) or ([config.BAIDU_MAP_AK] if config.BAIDU_MAP_AK else [])
    ak = pool[0] if pool else ""
    query.update({"output": "json", "ak": ak})
    url = f"https://api.map.baidu.com{API_PATH}?{urlencode(query)}"
    try:
        with urlrequest.urlopen(url, timeout=config.WEATHER_TIMEOUT) as resp:
            raw = resp.read().decode("utf-8", "ignore")
    except urlerror.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", "ignore")
        except Exception:
            pass
        status = NETWORK_ERROR
        try:
            status = classify(int(json.loads(body).get("status", -1)))
        except (ValueError, TypeError, json.JSONDecodeError):
            pass
        logger.warning("weather %s -> %s (HTTP %s)", _redacted(params), status, exc.code)
        return status, {}
    except (urlerror.URLError, TimeoutError) as exc:
        reason = getattr(exc, "reason", exc)
        status = TIMEOUT if "timed out" in str(reason).lower() else NETWORK_ERROR
        logger.warning("weather %s -> %s (%s)", _redacted(params), status, reason)
        return status, {}
    except OSError as exc:
        logger.warning("weather %s -> %s (%s)", _redacted(params), NETWORK_ERROR, exc)
        return NETWORK_ERROR, {}

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("weather %s -> server_error（返回非 JSON）", _redacted(params))
        return SERVER_ERROR, {}

    status = SERVER_ERROR
    try:
        status = classify(int(data.get("status", -1)))
    except (TypeError, ValueError):
        pass
    if status != OK:
        logger.warning("weather %s -> %s msg=%s", _redacted(params), status, data.get("message") or "")
    return status, data


def fetch_weather(district_id: str | None = None) -> Dict[str, Any]:
    """统一入口。返回统一结构 dict；非 ok 状态时调用方应静默跳过天气。

    全站共享一份缓存（30 分钟 TTL）：第一次请求打百度，之后所有会话/所有轮次
    直接复用，既省额度又不给每轮提问增加延迟。
    """
    if not config.WEATHER_ENABLED:
        return {"状态": AUTH_ERROR, "说明": "未配置百度地图 AK，天气服务不可用"}

    key = f"weather|{district_id or config.WEATHER_DISTRICT_ID}"
    cached = _cache_get(key)
    if cached is not None:
        with _lock:
            _stats["cache_hits"] += 1
        return cached

    # data_type=all：一次取回实况(now) + 预报(forecast) + 生活指数(indexes)
    params = {"district_id": district_id or config.WEATHER_DISTRICT_ID, "data_type": "all"}
    with _lock:
        _stats["calls"] += 1
    status, payload = _do_http(params)
    if status != OK:
        with _lock:
            _stats["errors"] += 1
        return {"状态": status, "说明": "天气接口暂时不可用，本轮不注入天气信息"}

    data = _translate(payload)
    if data.get("状态") != OK:
        with _lock:
            _stats["errors"] += 1
        return data
    data["缓存命中"] = False
    _cache_put(key, data)
    return data


def get_weather() -> str:
    """工具函数：用户主动问天气时由 LLM 调用。返回 JSON 文本。"""
    return json.dumps(fetch_weather(), ensure_ascii=False)


def weather_stats() -> Dict[str, Any]:
    """调用统计（不含任何密钥），供 /api/status 后续接入。"""
    with _lock:
        return {
            "calls": _stats["calls"],
            "cache_hits": _stats["cache_hits"],
            "errors": _stats["errors"],
            "cache_entries": len(_cache),
            "ttl_seconds": config.WEATHER_CACHE_TTL,
            "enabled": config.WEATHER_ENABLED,
        }


def reset_for_tests() -> None:
    """测试专用：清空缓存与统计。"""
    with _lock:
        _cache.clear()
        for key in _stats:
            _stats[key] = 0
