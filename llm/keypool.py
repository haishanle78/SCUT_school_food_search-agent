"""LLM Key 池：多把 API Key 轮流放行 + 每把独立并发上限 + 排队 + 熔断。

为什么要有这层（对照 map_gateway 的思路，面向 LLM 出口）：
  glm-4-flash 免费档单 Key 并发上限为 2；上下文超过 8K token 后速率还会被
  大幅限制。没有这层时，N 个访客同时提问就是 N 路并发直接打到智谱——
  超限的请求报 429/1214，而 agent 会把 LLM 异常当作"降级离线"信号，
  导致大批用户同时收到离线兜底回答（当前知识库为空 = 集体翻车）。

治理手段：
  1. 每把 Key 一个 BoundedSemaphore（默认 2 路并发），Key 之间互不影响；
  2. 调度器挑「未熔断且最闲」的 Key，FIFO 排队，排队超过 LLM_QUEUE_WAIT
     秒抛 LLMBusy —— 上层收到后返回友好提示，绝不降级到离线模式；
  3. 限流类错误（HTTP 429 / 1214 / 并发超限文案）触发该 Key 熔断
     LLM_BREAKER_COOLDOWN 秒，本次请求自动换下一把 Key，用户无感；
  4. 每 Key 独立记账（当日调用数、在途数），snapshot() 暴露给 /api/status，
     永不包含密钥明文。

多实例部署提示：池状态是进程内的，单容器单进程成立；若平台扩多实例，
需换外部共享存储（Redis），见 README「已知限制」。
"""

from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional


class LLMBusy(RuntimeError):
    """排队超时：当前没有可用的 Key 名额。

    与普通 RuntimeError 的区别：上层必须把它当「稍后再试」处理，
    绝不能触发离线降级（离线模式查空知识库只会更糟）。
    retry_after：建议等待秒数（全部 Key 冷却时 = 最早解冻剩余时间）。
    """

    def __init__(self, message: str, retry_after: int = 0) -> None:
        super().__init__(message)
        self.retry_after = max(0, int(retry_after))


class _Slot:
    """一把钥匙的运行时状态。"""

    __slots__ = ("key", "sem", "in_flight", "used", "day", "breaker_until", "breaker_status")

    def __init__(self, key: str, concurrency: int) -> None:
        self.key = key
        self.sem = threading.BoundedSemaphore(concurrency)
        self.in_flight = 0
        self.used = 0
        self.day = time.strftime("%Y-%m-%d")
        self.breaker_until = 0.0
        self.breaker_status = ""


class KeyPool:
    def __init__(
        self,
        keys: List[str],
        concurrency_per_key: int,
        queue_wait: float,
        breaker_cooldown: int,
    ) -> None:
        self._slots = [_Slot(k, max(1, concurrency_per_key)) for k in keys if k]
        self._lock = threading.Lock()
        self._queue_wait = max(0.0, queue_wait)
        self._cooldown = max(30, breaker_cooldown)
        self._concurrency = max(1, concurrency_per_key)

    @property
    def size(self) -> int:
        return len(self._slots)

    @property
    def enabled(self) -> bool:
        return bool(self._slots)

    # ---------- 取号 ----------
    def acquire(self, timeout: Optional[float] = None) -> _Slot:
        """取一个空闲 Key；排队超过 timeout（默认 queue_wait）抛 LLMBusy。"""
        deadline = time.monotonic() + (self._queue_wait if timeout is None else max(0.0, timeout))
        while True:
            with self._lock:
                now = time.time()
                candidates = [s for s in self._slots if s.breaker_until <= now]
                # 最闲优先；同等忙碌时优先今日调用少的，把负载摊匀
                candidates.sort(key=lambda s: (s.in_flight, s.used))
                for s in candidates:
                    self._roll_day_locked(s)
                    if s.sem.acquire(blocking=False):
                        s.in_flight += 1
                        s.used += 1
                        return s
            if time.monotonic() >= deadline:
                with self._lock:
                    now_wall = time.time()
                    lefts = [s.breaker_until - now_wall for s in self._slots]
                cooling = [x for x in lefts if x > 0]
                # 全部 Key 都在冷却 → 告诉用户真实等待时长；否则只是排队没抢到名额
                all_cooling = bool(cooling) and len(cooling) == len(self._slots)
                retry_after = int(min(cooling)) + 1 if all_cooling else 10
                if all_cooling:
                    reason = "所有模型通道都在冷却（刚被上游限流）"
                else:
                    reason = "模型通道正被其他请求占用"
                raise LLMBusy(
                    f"{reason}，排队 {self._queue_wait:.0f} 秒仍未排到",
                    retry_after=retry_after,
                )
            time.sleep(0.05)

    def release(self, slot: _Slot) -> None:
        with self._lock:
            slot.in_flight = max(0, slot.in_flight - 1)
        slot.sem.release()

    # ---------- 结果回填 ----------
    def mark_success(self, slot: _Slot) -> None:
        with self._lock:
            slot.breaker_status = ""

    def mark_failure(self, slot: _Slot, exc: Exception) -> bool:
        """按错误类型处理：限流/并发类 → 熔断该 Key 并返回 True（可换钥匙重试）。"""
        text = str(exc)
        retryable = _looks_rate_limited(text)
        if retryable:
            with self._lock:
                slot.breaker_until = time.time() + self._cooldown
                slot.breaker_status = text[:120]
        return retryable

    # ---------- 统计 ----------
    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            now = time.time()
            keys = [
                {
                    "key": _mask(s.key),
                    "in_flight": s.in_flight,
                    "used_today": s.used,
                    "breaker_seconds_left": max(0, round(s.breaker_until - now)),
                    "breaker_status": s.breaker_status,
                }
                for s in self._slots
            ]
        return {
            "keys": len(self._slots),
            "concurrency_per_key": self._concurrency,
            "queue_wait": self._queue_wait,
            "slots": keys,
        }

    def reset_for_tests(self) -> None:
        with self._lock:
            for s in self._slots:
                s.in_flight = 0
                s.used = 0
                s.day = time.strftime("%Y-%m-%d")
                s.breaker_until = 0.0
                s.breaker_status = ""

    def _roll_day_locked(self, slot: _Slot) -> None:
        today = time.strftime("%Y-%m-%d")
        if today != slot.day:
            slot.day = today
            slot.used = 0


def _looks_rate_limited(text: str) -> bool:
    """判断是否为「换一把 Key 有意义」的错误。"""
    lowered = text.lower()
    signals = ("429", "1214", "1302", "并发", "限流", "rate limit", "too many requests", "quota")
    return any(sig in lowered for sig in signals)


def _mask(key: str) -> str:
    if len(key) <= 8:
        return key[:2] + "***"
    return key[:6] + "***" + key[-4:]
