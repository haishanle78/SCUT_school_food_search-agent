"""百度地图统一网关：所有地图请求的唯一出口。

为什么要有这层：
  百度个人开发者权益有限（每把 AK：地点检索 100 次/日、3 次/秒；路线 5000 次/日、
  3 次/秒），而 agent 很容易在"每家店 × 三种交通方式 × 每次重问"上把额度烧穿，
  还会因为并发触发 401（并发超限）而被误判成"附近没有店"。所以集中在这一层治理：

  1. AK 池：多把 AK 轮流放行（BAIDU_MAP_AKS，逗号分隔），池子总容量线性扩展；
  2. 每 AK 独立记账：独立令牌桶（MAP_QPS）、独立日配额（MAP_DAILY_*）、独立熔断；
  3. 调度：优先选「未熔断、未超额、令牌最充足」的 AK，轮询起点保证负载摊匀；
  4. 缓存 + 并发合并：同一 key 的并发请求只真正打一次百度（缓存不区分 AK，命中不耗额度）；
  5. 错误分类：区分"查到了""查了但没有""限流""配额耗尽""权限异常""超时"；
  6. 有限重试 + 故障转移：限流/超时按退避重试；某把 AK 配额/权限异常 → 只熔断
     那把 AK 并立即换下一把重试，用户无感；全部 AK 不可用才对外报错；
  7. 调用统计：次数、缓存命中、限流次数、每 AK 用量，通过 /api/status 暴露；
  8. 日志脱敏：AK 永远打码，不写入日志。

对外只暴露 request() 与 snapshot()；业务层无需关心这些细节。

多实例部署提示：AK 池状态是进程内的（日计数另落盘跨进程可见）；单容器单进程
成立，若平台扩多实例需外部共享存储（Redis），见 README「已知限制」。
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple
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

# 每 AK 独立状态：ak -> {bucket, last_refill, day, used, breaker_until, breaker_status}
_ak_state: Dict[str, Dict[str, Any]] = {}
_ak_rr = 0  # 轮询起点，保证负载摊匀
_usage_day: str = date.today().isoformat()


def _ak_pool() -> List[str]:
    aks = list(getattr(config, "BAIDU_MAP_AKS", []) or [])
    if not aks and getattr(config, "BAIDU_MAP_AK", ""):
        aks = [config.BAIDU_MAP_AK]
    return [a for a in aks if a]


def _per_ak_limit(kind: str) -> int:
    return config.MAP_DAILY_SEARCH if kind == "search" else config.MAP_DAILY_ROUTE


def _state_for(ak: str) -> Dict[str, Any]:
    st = _ak_state.get(ak)
    if st is None:
        st = {
            "bucket": float(config.MAP_QPS),
            "last_refill": time.monotonic(),
            "day": date.today().isoformat(),
            "used": {"search": 0, "route": 0},
            "breaker_until": 0.0,
            "breaker_status": "",
        }
        _ak_state[ak] = st
    return st


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
    global _usage_day
    path = getattr(config, "MAP_USAGE_PATH", "")
    if not path or not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if data.get("day") != _usage_day:
            return
        if isinstance(data.get("aks"), dict):
            for ak, used in data["aks"].items():
                st = _state_for(ak)
                st["used"] = {k: int(used.get(k, 0)) for k in ("search", "route")}
            return
        pool = _ak_pool()
        if pool:  # 旧格式：聚合计数归到第一把 AK
            st = _state_for(pool[0])
            st["used"] = {k: int(data.get(k, 0) or 0) for k in ("search", "route")}
    except (OSError, ValueError, TypeError, AttributeError):
        pass


def _save_usage_locked() -> None:
    path = getattr(config, "MAP_USAGE_PATH", "")
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp"
        payload = {
            "day": _usage_day,
            "aks": {ak: dict(st["used"]) for ak, st in _ak_state.items()},
        }
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        os.replace(tmp, path)
    except OSError:
        pass  # 只读文件系统（云端容器）不应影响主流程


def _roll_day_locked(st: Dict[str, Any]) -> bool:
    """单 AK 跨天滚动。返回 True 表示发生了滚动（调用方负责落盘）。"""
    global _usage_day
    today = date.today().isoformat()
    if today != st["day"]:
        st["day"] = today
        st["used"] = {"search": 0, "route": 0}
        _usage_day = today
        return True
    return False


def _sync_usage_locked() -> None:
    """以磁盘计数为准做一次同步（取最大值），修正多进程互相覆盖。

    服务进程与数据维护脚本可能同时调用地图；各自启动时快照一次计数，
    之后各自写盘会互相覆盖（后写的旧值把新值抹掉，额度被意外"重置"）。
    每次取名额前同步一次，可容忍毫秒级竞态。
    兼容旧格式：磁盘只有聚合 search/route 时，归到池中第一把 AK，避免额度被低估。
    """
    path = getattr(config, "MAP_USAGE_PATH", "")
    if not path or not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return
    if data.get("day") != _usage_day:
        return
    pool = _ak_pool()
    pool_set = set(pool)
    aks_payload = data.get("aks")
    if isinstance(aks_payload, dict):
        for ak, used in aks_payload.items():
            # 只同步当前池内的 AK：换过 AK 集合或测试用假 AK 时，不把历史计数带进来
            if ak not in pool_set or not isinstance(used, dict):
                continue
            st = _state_for(ak)
            for key in ("search", "route"):
                st["used"][key] = max(st["used"][key], int(used.get(key, 0) or 0))
        return
    if len(pool) == 1:  # 旧格式（聚合计数）只在单 AK 时可归属，多 AK 无法归属则跳过
        st = _state_for(pool[0])
        for key in ("search", "route"):
            st["used"][key] = max(st["used"][key], int(data.get(key, 0) or 0))


def _reserve(kind: str) -> Tuple[Optional[str], Optional[str]]:
    """取一个调用名额。

    返回 (None, ak) 表示可以发请求；否则返回 (blocked_status, None)。
    blocked_status: RATE_LIMITED（等不到令牌）/ QUOTA_EXHAUSTED（全部超额）/
                    AUTH_ERROR（全部熔断且熔断原因是权限类）。
    """
    global _ak_rr
    aks = _ak_pool()
    if not aks:
        return AUTH_ERROR, None
    deadline = time.monotonic() + max(0.2, config.MAP_MAX_WAIT)
    while True:
        with _lock:
            now_mono = time.monotonic()
            now_wall = time.time()
            _sync_usage_locked()  # 先对齐磁盘计数，再做配额判断
            rolled = False
            usable: List[Tuple[str, Dict[str, Any]]] = []
            for ak in aks:
                st = _state_for(ak)
                rolled = _roll_day_locked(st) or rolled
                if st["breaker_until"] <= now_wall and st["used"][kind] < _per_ak_limit(kind):
                    usable.append((ak, st))
            if rolled:
                _save_usage_locked()
            if not usable:
                broken = [ak for ak in aks if _ak_state[ak]["breaker_until"] > now_wall]
                if len(broken) == len(aks):
                    status = QUOTA_EXHAUSTED
                    for ak in broken:
                        if _ak_state[ak]["breaker_status"] == AUTH_ERROR:
                            status = AUTH_ERROR
                            break
                    return status, None
                _stats["quota"] += 1
                return QUOTA_EXHAUSTED, None
            rate = max(config.MAP_QPS, 0.1)
            n = len(usable)
            for i in range(n):
                ak, st = usable[(_ak_rr + i) % n]
                st["bucket"] = min(rate, st["bucket"] + (now_mono - st["last_refill"]) * rate)
                st["last_refill"] = now_mono
                if st["bucket"] >= 1.0:
                    st["bucket"] -= 1.0
                    st["used"][kind] += 1
                    _ak_rr = (_ak_rr + i + 1) % n
                    _save_usage_locked()
                    return None, ak
            best = max(st["bucket"] for _, st in usable)
            wait = (1.0 - min(best, 1.0)) / rate
        if time.monotonic() + wait > deadline:
            _stats["rate_limited"] += 1
            return RATE_LIMITED, None
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


def _do_http(path: str, params: Dict[str, Any], ak: Optional[str] = None) -> MapResult:
    """真正发请求的地方（测试里会被替换）。ak 为空时回落到第一把 AK。"""
    query = dict(params)
    query.update({"output": "json", "ak": ak or (config.BAIDU_MAP_AKS or [config.BAIDU_MAP_AK])[0]})
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


def _trip_breaker(ak: str, status: str) -> None:
    """只熔断出问题的那把 AK；其他 AK 继续服务。"""
    with _lock:
        st = _state_for(ak)
        st["breaker_until"] = time.time() + max(30, config.MAP_AUTH_COOLDOWN)
        st["breaker_status"] = status


def _request_with_retry(path: str, params: Dict[str, Any], kind: str) -> MapResult:
    attempts = 0
    n_aks = max(1, len(_ak_pool()))
    max_attempts = n_aks + max(0, config.MAP_RETRY)
    last: Optional[MapResult] = None
    while attempts < max_attempts:
        attempts += 1
        blocked, ak = _reserve(kind)
        if blocked:
            return MapResult(blocked, attempts=attempts)
        result = _do_http(path, params, ak)
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
        last = result

        if result.status in (QUOTA_EXHAUSTED, AUTH_ERROR) and ak:
            # 这把 AK 熔断，其余的继续试；全部不可用时下一轮 _reserve 会给出结论
            _trip_breaker(ak, result.status)
            continue
        if result.status in _RETRYABLE and attempts <= max(0, config.MAP_RETRY):
            time.sleep(config.MAP_BACKOFF * attempts)
            continue
        return result
    return last or MapResult(SERVER_ERROR, message="地图请求未能执行")


def request(
    path: str,
    params: Dict[str, Any],
    *,
    kind: str = "search",
    cache_key: Optional[str] = None,
    ttl: Optional[int] = None,
) -> MapResult:
    """统一入口：缓存 → 并发合并 → AK 池调度 → 限速 → 重试/换 AK → 分类。

    kind: "search"（地点检索）或 "route"（路线规划），用于分别计算日配额。
    """
    if not config.MAP_ENABLED:
        return MapResult(AUTH_ERROR, message="未配置百度地图 AK")

    with _lock:
        now_wall = time.time()
        aks = _ak_pool()
        broken = [ak for ak in aks if _ak_state.get(ak, {}).get("breaker_until", 0) > now_wall]
        if aks and len(broken) == len(aks):
            status = QUOTA_EXHAUSTED
            for ak in broken:
                if _ak_state[ak]["breaker_status"] == AUTH_ERROR:
                    status = AUTH_ERROR
                    break
            left = max(round(_ak_state[ak]["breaker_until"] - now_wall) for ak in broken)
            return MapResult(
                status,
                message=f"地图服务暂不可用（全部 AK 熔断中，约 {left} 秒后自动恢复）",
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


def _mask_ak(ak: str) -> str:
    if len(ak) <= 8:
        return ak[:2] + "***"
    return ak[:6] + "***" + ak[-4:]


def snapshot() -> Dict[str, Any]:
    """给 /api/status 用的调用统计（不含任何密钥）。"""
    with _lock:
        stats = dict(_stats)
        entries = len(_cache)
        now_wall = time.time()
        aks = _ak_pool()
        ak_usage = []
        breaker_left = 0
        total_search = total_route = 0
        for ak in aks:
            st = _ak_state.get(ak)
            if st is None:
                ak_usage.append({"ak": _mask_ak(ak), "search": 0, "route": 0, "breaker_seconds_left": 0})
                continue
            left = max(0, round(st["breaker_until"] - now_wall))
            breaker_left = max(breaker_left, left)
            total_search += st["used"]["search"]
            total_route += st["used"]["route"]
            ak_usage.append({
                "ak": _mask_ak(ak),
                "search": st["used"]["search"],
                "route": st["used"]["route"],
                "breaker_seconds_left": left,
            })
    return {
        **stats,
        "qps": config.MAP_QPS,
        "cache_entries": entries,
        "day": _usage_day,
        "daily_search": total_search,
        "daily_search_limit": config.MAP_DAILY_SEARCH,
        "daily_route": total_route,
        "daily_route_limit": config.MAP_DAILY_ROUTE,
        "ak_count": len(aks),
        "ak_daily_search_limit_total": config.MAP_DAILY_SEARCH * len(aks),
        "ak_daily_route_limit_total": config.MAP_DAILY_ROUTE * len(aks),
        "ak_usage": ak_usage,
        "breaker_seconds_left": breaker_left,
        "cache_hit_rate": round(stats["cache_hits"] / max(1, stats["cache_hits"] + stats["calls"]), 3),
    }


def reset_for_tests() -> None:
    """测试专用：清空缓存、统计、熔断与当日计数。"""
    global _ak_state, _ak_rr, _usage_day
    with _lock:
        _cache.clear()
        _inflight.clear()
        for key in _stats:
            _stats[key] = 0
        _ak_state = {}
        _ak_rr = 0
        _usage_day = date.today().isoformat()


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


_load_usage()
