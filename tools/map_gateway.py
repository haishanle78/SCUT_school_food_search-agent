"""百度地图统一网关：所有地图请求的唯一出口。

为什么要有这层：
  百度个人开发者权益有限（地点检索 100 次/日、3 次/秒；路线 5000 次/日、3 次/秒），
  而 agent 很容易在"每家店 × 三种交通方式 × 每次重问"上把额度烧穿，还会因为并发
  触发 401（并发超限）而被误判成"附近没有店"。所以集中在这一层做治理：

  1. 全局限速：令牌桶，跨会话/跨用户共享（进程内）；多实例部署需外部共享存储，见 README；
  2. 缓存 + 并发合并：同一 key 的并发请求只真正打一次百度；
  3. 错误分类：区分"查到了""查了但没有""限流""配额耗尽""权限异常""超时"；
  4. 有限重试：只对限流/超时/网络抖动重试，配额与权限异常立刻熔断，不再白烧请求；
  5. 调用统计：次数、缓存命中、限流次数，通过 /api/status 暴露；
  6. 日志脱敏：AK 永远不进日志。

对外只暴露 request() 与 snapshot()；业务层无需关心这些细节。
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import date, datetime
from typing import Any, Dict, Optional, Tuple
from urllib import error as urlerror
from urllib import request as urlrequest
from urllib.parse import urlencode

import config

logger = logging.getLogger("map_gateway")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

# ---------------- 状态常量 ----------------
OK = "ok"
NO_RESULT = "no_result"
RATE_LIMITED = "rate_limited"
QUOTA_EXHAUSTED = "quota_exhausted"
AUTH_ERROR = "auth_error"
TIMEOUT = "timeout"
NETWORK_ERROR = "network_error"
BAD_PARAM = "bad_param"
SERVER_ERROR = "server_error"

STATUS_LABEL = {
    OK: "查询成功",
    NO_RESULT: "查询成功但没有匹配结果",
    RATE_LIMITED: "地图接口限流（请求过快或并发超限）",
    QUOTA_EXHAUSTED: "地图接口今日配额已用完",
    AUTH_ERROR: "地图 AK 或服务权限异常",
    TIMEOUT: "地图接口超时",
    NETWORK_ERROR: "地图接口网络异常",
    BAD_PARAM: "地图请求参数非法",
    SERVER_ERROR: "地图服务内部错误",
}

# 百度业务状态码分类（https://lbsyun.baidu.com/faq/details?id=2298）
_AUTH_CODES = {3, 4, 5, 101, 200, 201, 202, 203, 204, 210, 211, 240, 250, 251, 260, 261, 300}
_QUOTA_CODES = {301, 302}
_RATE_CODES = {401}
_RETRYABLE = {RATE_LIMITED, TIMEOUT, NETWORK_ERROR, SERVER_ERROR}


class MapResult:
    """统一返回：业务层只看 status / payload，不再猜"是不是没找到"。"""

    __slots__ = ("status", "payload", "cached", "attempts", "message")

    def __init__(
        self,
        status: str,
        payload: Optional[Dict[str, Any]] = None,
        cached: bool = False,
        attempts: int = 1,
        message: str = "",
    ) -> None:
        self.status = status
        self.payload = payload or {}
        self.cached = cached
        self.attempts = attempts
        self.message = message or STATUS_LABEL.get(status, status)

    @property
    def ok(self) -> bool:
        return self.status == OK

    def as_dict(self) -> Dict[str, Any]:
        return {
            "状态": self.status,
            "说明": self.message,
            "缓存命中": self.cached,
            "尝试次数": self.attempts,
        }

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"MapResult(status={self.status}, cached={self.cached}, attempts={self.attempts})"


# ---------------- 内部状态 ----------------
_lock = threading.RLock()
_cache: Dict[str, Tuple[float, MapResult]] = {}
_inflight: Dict[str, threading.Event] = {}
_tokens: float = float(config.MAP_QPS)
_last_refill: float = time.monotonic()
_usage_day: str = date.today().isoformat()
_usage: Dict[str, int] = {"search": 0, "route": 0}
_breaker: Dict[str, Any] = {"until": 0.0, "status": None}
_stats: Dict[str, int] = {
    "calls": 0,
    "cache_hits": 0,
    "rate_limited": 0,
    "quota": 0,
    "auth": 0,
    "timeout": 0,
    "network": 0,
    "no_result": 0,
    "errors": 0,
}
_CACHE_MAX = 512


def classify(status_code: int) -> str:
    """百度业务码 → 统一状态。0 成功；302 配额；401 并发/限流；其余权限类。"""
    if status_code == 0:
        return OK
    if status_code in _QUOTA_CODES:
        return QUOTA_EXHAUSTED
    if status_code in _RATE_CODES:
        return RATE_LIMITED
    if status_code in _AUTH_CODES:
        return AUTH_ERROR
    if status_code == 2:
        return BAD_PARAM
    return SERVER_ERROR


# ---------------- 计数持久化（当日额度跨进程/重启可见） ----------------
def _load_usage() -> None:
    global _usage, _usage_day
    path = getattr(config, "MAP_USAGE_PATH", "")
    if not path or not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if data.get("day") == _usage_day:
            _usage = {k: int(data.get(k, 0)) for k in ("search", "route")}
    except (OSError, ValueError):
        pass


def _save_usage_locked() -> None:
    path = getattr(config, "MAP_USAGE_PATH", "")
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump({"day": _usage_day, "search": _usage["search"], "route": _usage["route"]}, handle)
        os.replace(tmp, path)
    except OSError:
        pass  # 只读文件系统（云端容器）不应影响主流程


def _roll_day_locked() -> None:
    global _usage_day, _usage
    today = date.today().isoformat()
    if today != _usage_day:
        _usage_day = today
        _usage = {"search": 0, "route": 0}
        _save_usage_locked()


def _reserve(kind: str) -> Optional[str]:
    """取一个调用名额：先查日配额，再按全局 QPS 排队。返回 None 表示可以发请求。"""
    global _tokens, _last_refill
    deadline = time.monotonic() + max(0.2, config.MAP_MAX_WAIT)
    limit = config.MAP_DAILY_SEARCH if kind == "search" else config.MAP_DAILY_ROUTE
    while True:
        with _lock:
            _roll_day_locked()
            if _usage[kind] >= limit:
                _stats["quota"] += 1
                return QUOTA_EXHAUSTED
            now = time.monotonic()
            rate = max(config.MAP_QPS, 0.1)
            _tokens = min(rate, _tokens + (now - _last_refill) * rate)
            _last_refill = now
            if _tokens >= 1.0:
                _tokens -= 1.0
                _usage[kind] += 1
                _save_usage_locked()
                return None
            wait = (1.0 - _tokens) / rate
        if time.monotonic() + wait > deadline:
            _stats["rate_limited"] += 1
            return RATE_LIMITED
        time.sleep(min(max(wait, 0.02), 0.25))


# ---------------- 缓存 ----------------
def _cache_get(key: str) -> Optional[MapResult]:
    with _lock:
        item = _cache.get(key)
        if not item:
            return None
        expire, result = item
        if expire < time.time():
            _cache.pop(key, None)
            return None
    clone = MapResult(result.status, result.payload, cached=True, attempts=result.attempts, message=result.message)
    return clone


def _cache_put(key: str, result: MapResult, ttl: int) -> None:
    if ttl <= 0:
        return
    with _lock:
        if len(_cache) >= _CACHE_MAX:
            for old in sorted(_cache, key=lambda k: _cache[k][0])[:_CACHE_MAX // 4]:
                _cache.pop(old, None)
        _cache[key] = (time.time() + ttl, result)


# ---------------- HTTP ----------------
def _redacted(path: str, params: Dict[str, Any]) -> str:
    safe = {k: ("***" if k == "ak" else v) for k, v in params.items()}
    return f"{path} {safe}"


def _do_http(path: str, params: Dict[str, Any]) -> MapResult:
    """真正发请求的地方（测试里会被替换）。"""
    query = dict(params)
    query.update({"output": "json", "ak": config.BAIDU_MAP_AK})
    url = f"https://api.map.baidu.com{path}?{urlencode(query)}"
    started = time.monotonic()
    try:
        with urlrequest.urlopen(url, timeout=config.MAP_TIMEOUT) as resp:
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
        logger.warning("map %s -> %s (HTTP %s, %.0fms)", _redacted(path, params), status, exc.code,
                       (time.monotonic() - started) * 1000)
        return MapResult(status, message=f"HTTP {exc.code}")
    except (urlerror.URLError, TimeoutError) as exc:
        reason = getattr(exc, "reason", exc)
        status = TIMEOUT if "timed out" in str(reason).lower() else NETWORK_ERROR
        logger.warning("map %s -> %s (%s)", _redacted(path, params), status, reason)
        return MapResult(status, message=str(reason))
    except OSError as exc:
        logger.warning("map %s -> %s (%s)", _redacted(path, params), NETWORK_ERROR, exc)
        return MapResult(NETWORK_ERROR, message=str(exc))

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("map %s -> server_error（返回非 JSON，%.0fms）", _redacted(path, params),
                       (time.monotonic() - started) * 1000)
        return MapResult(SERVER_ERROR, message="返回内容不是 JSON")

    code = data.get("status", -1)
    try:
        status = classify(int(code))
    except (TypeError, ValueError):
        status = SERVER_ERROR
    elapsed = (time.monotonic() - started) * 1000
    if status == OK:
        logger.info("map %s -> ok %.0fms", _redacted(path, params), elapsed)
    else:
        logger.warning("map %s -> %s status=%s msg=%s", _redacted(path, params), status, code,
                       data.get("message") or "")
    return MapResult(status, payload=data, message=str(data.get("message") or ""))


def _trip_breaker(status: str) -> None:
    with _lock:
        _breaker["until"] = time.time() + max(30, config.MAP_AUTH_COOLDOWN)
        _breaker["status"] = status


def _request_with_retry(path: str, params: Dict[str, Any], kind: str) -> MapResult:
    attempts = 0
    while True:
        attempts += 1
        blocked = _reserve(kind)
        if blocked:
            return MapResult(blocked, attempts=attempts)
        result = _do_http(path, params)
        result.attempts = attempts
        _stats["calls"] += 1
        _stats[kind + "_calls"] = _stats.get(kind + "_calls", 0) + 1
        if result.status == OK:
            return result
        if result.status == NO_RESULT:
            _stats["no_result"] += 1
            return result
        _stats["errors"] += 1
        if result.status == RATE_LIMITED:
            _stats["rate_limited"] += 1
        elif result.status == QUOTA_EXHAUSTED:
            _stats["quota"] += 1
        elif result.status == AUTH_ERROR:
            _stats["auth"] += 1
        elif result.status == TIMEOUT:
            _stats["timeout"] += 1
        elif result.status == NETWORK_ERROR:
            _stats["network"] += 1

        if result.status in (QUOTA_EXHAUSTED, AUTH_ERROR):
            _trip_breaker(result.status)  # 再重试只会白烧，直接熔断
            return result
        if result.status in _RETRYABLE and attempts <= max(0, config.MAP_RETRY):
            time.sleep(config.MAP_BACKOFF * attempts)
            continue
        return result


def request(
    path: str,
    params: Dict[str, Any],
    *,
    kind: str = "search",
    cache_key: Optional[str] = None,
    ttl: Optional[int] = None,
) -> MapResult:
    """统一入口：缓存 → 并发合并 → 限速 → 重试 → 分类。

    kind: "search"（地点检索）或 "route"（路线规划），用于分别计算日配额。
    """
    if not config.MAP_ENABLED:
        return MapResult(AUTH_ERROR, message="未配置百度地图 AK")

    with _lock:
        if _breaker["until"] > time.time():
            return MapResult(
                _breaker["status"] or QUOTA_EXHAUSTED,
                message="地图服务暂不可用（配额/权限异常后熔断中，稍后自动恢复）",
            )

    key = cache_key or f"{path}|{json.dumps(params, sort_keys=True, ensure_ascii=False)}"
    ttl = config.MAP_CACHE_TTL_SEARCH if ttl is None else ttl
    cached = _cache_get(key)
    if cached:
        _stats["cache_hits"] += 1
        return cached

    with _lock:
        event = _inflight.get(key)
        leader = event is None
        if leader:
            event = threading.Event()
            _inflight[key] = event

    if not leader:  # 同一请求正在飞，等它的结果，不再打一次百度
        event.wait(timeout=config.MAP_TIMEOUT + 2)
        cached = _cache_get(key)
        if cached:
            _stats["cache_hits"] += 1
            return cached
        return MapResult(NETWORK_ERROR, message="并发请求未能取到结果，请稍后重试")

    try:
        result = _request_with_retry(path, params, kind)
        if result.ok:
            _cache_put(key, result, ttl)
        return result
    finally:
        with _lock:
            _inflight.pop(key, None)
        event.set()


def snapshot() -> Dict[str, Any]:
    """给 /api/status 用的调用统计（不含任何密钥）。"""
    with _lock:
        stats = dict(_stats)
        usage = dict(_usage)
        breaker_until = _breaker["until"]
        entries = len(_cache)
        day = _usage_day
    return {
        **stats,
        "qps": config.MAP_QPS,
        "cache_entries": entries,
        "day": day,
        "daily_search": usage["search"],
        "daily_search_limit": config.MAP_DAILY_SEARCH,
        "daily_route": usage["route"],
        "daily_route_limit": config.MAP_DAILY_ROUTE,
        "breaker_seconds_left": max(0, round(breaker_until - time.time())),
        "cache_hit_rate": round(stats["cache_hits"] / max(1, stats["cache_hits"] + stats["calls"]), 3),
    }


def reset_for_tests() -> None:
    """测试专用：清空缓存、统计、熔断与当日计数。"""
    global _tokens, _last_refill, _usage, _usage_day
    with _lock:
        _cache.clear()
        _inflight.clear()
        for key in _stats:
            _stats[key] = 0
        _tokens = float(config.MAP_QPS)
        _last_refill = time.monotonic()
        _usage = {"search": 0, "route": 0}
        _usage_day = date.today().isoformat()
        _breaker["until"] = 0.0
        _breaker["status"] = None


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


_load_usage()
