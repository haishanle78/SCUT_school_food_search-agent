"""并发治理测试：LLM Key 池（排队/分流/熔断/故障转移）+ 地图 AK 池 + 上下文预算。

全程 mock 网络层（替换 LLMClient._do_chat / map_gateway._do_http），不消耗真实额度。

跑法：python tests/test_concurrency.py
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

config.MAP_ENABLED = True

from agent.core import FoodAdvisorAgent  # noqa: E402
from agent.memory import ConversationMemory, UserProfile  # noqa: E402
from llm.client import LLMClient, reset_pool_for_tests  # noqa: E402
from llm.keypool import KeyPool, LLMBusy  # noqa: E402
from tools import build_registry, map_gateway as gw  # noqa: E402

PASS = 0


def ok(label: str) -> None:
    global PASS
    PASS += 1
    print(f"[通过] {label}")


def fake_llm_payload(content="ok"):
    return {"choices": [{"message": {"content": content}}]}


class FakeChat:
    """按 Key 编排的假 LLM：记录每把钥匙被调用的次数与峰值并发。"""

    def __init__(self, behavior, delay=0.0, on_call=None):
        # behavior: {key: "ok" | "rate" | Exception}
        self.behavior = behavior
        self.delay = delay
        self.on_call = on_call
        self.calls = {}
        self.concurrent = {}
        self.max_concurrent = {}
        self.lock = threading.Lock()

    def __call__(self, body, api_key):
        with self.lock:
            self.calls[api_key] = self.calls.get(api_key, 0) + 1
            self.concurrent[api_key] = self.concurrent.get(api_key, 0) + 1
            self.max_concurrent[api_key] = max(
                self.max_concurrent.get(api_key, 0), self.concurrent[api_key]
            )
        if self.on_call:
            self.on_call(api_key)
        try:
            if self.delay:
                time.sleep(self.delay)
            action = self.behavior.get(api_key, "ok")
            if action == "rate":
                raise RuntimeError("LLM 调用失败: HTTP 429 {\"error\":{\"code\":\"1214\"}} 并发超限")
            if isinstance(action, Exception):
                raise action
            return fake_llm_payload()
        finally:
            with self.lock:
                self.concurrent[api_key] -= 1


def make_agent():
    agent = FoodAdvisorAgent.__new__(FoodAdvisorAgent)
    agent.profile = UserProfile()
    agent.poi_store = {}
    agent.functions, agent.schemas = build_registry(agent.profile, agent.poi_store)
    agent.memory = ConversationMemory("")
    agent._trace_fn = lambda text: None
    agent.last_cards = None
    agent.last_summary = ""
    agent.last_dropped = []
    agent.origin_key = config.DEFAULT_ORIGIN
    agent.origin = config.origin_by_key(agent.origin_key)
    agent._needed_modes = ["walking"]
    return agent


# ---------- LLM 池：熔断 + 故障转移 ----------

def test_llm_failover():
    pool = KeyPool(["k1", "k2"], concurrency_per_key=2, queue_wait=5, breaker_cooldown=600)
    client = LLMClient(pool=pool)
    client.enabled = True
    fake = FakeChat({"k1": "rate", "k2": "ok"})
    client._do_chat = fake

    reply = client.chat([{"role": "user", "content": "测试"}])
    assert reply.content == "ok"
    assert fake.calls.get("k1", 0) == 1, f"k1 只应被尝试 1 次: {fake.calls}"
    assert fake.calls.get("k2", 0) == 1
    snap = pool.snapshot()
    k1 = next(s for s in snap["slots"] if s["key"].startswith("k1***") or "k1" in s["key"])
    assert k1["breaker_seconds_left"] > 0, "限流 Key 应被熔断"
    ok("LLM 故障转移：k1 限流即熔断，请求自动换 k2，用户拿到正常回答")


# ---------- LLM 池：排队超时 ----------

def test_llm_busy():
    pool = KeyPool(["k1"], concurrency_per_key=1, queue_wait=0.2, breaker_cooldown=600)
    client = LLMClient(pool=pool)
    client.enabled = True
    slot_held = threading.Event()
    fake = FakeChat({"k1": "ok"}, delay=0.5, on_call=lambda key: slot_held.set())
    client._do_chat = fake

    result_box = []

    def worker():
        try:
            client.chat([{"role": "user", "content": "占坑"}])
            result_box.append("done")
        except Exception as exc:  # noqa: BLE001
            result_box.append(exc)

    t = threading.Thread(target=worker)
    t.start()
    assert slot_held.wait(timeout=2), "占坑请求应先拿到唯一名额"
    try:
        client.chat([{"role": "user", "content": "第二个"}])
        result_box.append("unexpected-success")
    except LLMBusy:
        result_box.append("busy")
    t.join()

    assert "done" in result_box, f"占坑请求应正常完成: {result_box}"
    assert "busy" in result_box, f"排队超时应抛 LLMBusy: {result_box}"
    assert len(result_box) == 2, f"两个请求各有一个明确结果: {result_box}"
    ok("LLM 排队超时：唯一通道被占用时，第二个请求排队 0.2 秒后抛 LLMBusy（不降级）")


# ---------- LLM 池：并发不超上限（防雪崩核心） ----------

def test_llm_no_avalanche():
    pool = KeyPool(["k1"], concurrency_per_key=2, queue_wait=10, breaker_cooldown=600)
    client = LLMClient(pool=pool)
    client.enabled = True
    fake = FakeChat({"k1": "ok"}, delay=0.05)
    client._do_chat = fake

    errors = []
    barrier = threading.Barrier(20)

    def worker(i):
        barrier.wait()
        try:
            client.chat([{"role": "user", "content": f"问题{i}"}])
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert not errors, f"20 路并发应有 0 个失败（排队消化），实际 {len(errors)} 个: {errors[:2]}"
    assert fake.max_concurrent.get("k1", 0) <= 2, (
        f"单 Key 并发绝不能超过 2，实际峰值 {fake.max_concurrent.get('k1')}"
    )
    assert fake.calls.get("k1", 0) == 20
    ok("LLM 防雪崩：20 路并发全部排队消化成功，单 Key 峰值并发 ≤ 2，0 个报错")


# ---------- Agent：LLMBusy 不降级离线 ----------

def test_agent_busy_message():
    agent = make_agent()
    busy_client = SimpleNamespace(
        enabled=True,
        chat=lambda messages, tools=None: (_ for _ in ()).throw(LLMBusy("所有模型通道都在忙")),
    )
    agent.llm = busy_client
    reply = agent.run("附近有什么好吃的")
    assert "再" in reply and "通道" in reply, f"应返回排队提示: {reply}"
    assert "离线" not in reply and "知识库" not in reply, "绝不能降级到离线模式"
    ok("Agent 收到 LLMBusy：返回『稍后再试』提示，绝不降级到离线模式")


# ---------- 上下文预算：防 8K 限速档 ----------

def test_context_budget():
    agent = make_agent()
    agent.llm = SimpleNamespace(enabled=False)
    long_tool = "x" * 5000
    messages = [
        {"role": "system", "content": "系统提示"},
        {"role": "user", "content": "问"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": long_tool},
    ]
    old_limit, old_tool = config.LLM_CONTEXT_CHAR_LIMIT, config.TOOL_RESULT_MAX_CHARS
    before = agent._payload_chars(messages, agent.schemas)
    config.LLM_CONTEXT_CHAR_LIMIT, config.TOOL_RESULT_MAX_CHARS = before - 2000, 600
    try:
        assert agent._payload_over_budget(messages), "超预算应被检测到"
        shrunk = agent._shrink(messages)
        assert len(shrunk[-1]["content"]) <= 600 + 40, "工具结果应被截断"
        assert shrunk[0]["role"] == "system", "system 提示必须保留"
        after = agent._payload_chars(shrunk, agent.schemas)
        assert after <= config.LLM_CONTEXT_CHAR_LIMIT, f"截断后应回到预算内: {after}"
        assert len(shrunk) == len(messages), "仅靠截断就能达标时，不得压缩历史"
    finally:
        config.LLM_CONTEXT_CHAR_LIMIT, config.TOOL_RESULT_MAX_CHARS = old_limit, old_tool
    ok("上下文预算：超限主动截断工具结果并保留 system，绝不带着超标上下文发请求")


# ---------- 地图 AK 池：故障转移 ----------

class FakeMap:
    """按 AK 编排的假百度。"""

    def __init__(self, behavior):
        self.behavior = behavior  # {ak: MapResult 或 callable(ak)}
        self.calls = []
        self.original = None

    def __enter__(self):
        self.original = gw._do_http

        def wrapper(path, params, ak=None):
            self.calls.append((path, ak))
            action = self.behavior.get(ak, gw.MapResult(gw.OK, payload={"status": 0, "results": []}))
            if callable(action):
                return action(ak)
            return action

        gw._do_http = wrapper
        gw.reset_for_tests()
        return self

    def __exit__(self, *exc):
        gw._do_http = self.original
        gw.reset_for_tests()


def test_map_failover():
    old_aks, old_ak = config.BAIDU_MAP_AKS, config.BAIDU_MAP_AK
    config.BAIDU_MAP_AKS = ["ak1", "ak2"]
    try:
        def ok_result(ak):
            return gw.MapResult(gw.OK, payload={"status": 0, "results": [{"uid": "u1"}]})

        with FakeMap({
            "ak1": gw.MapResult(gw.QUOTA_EXHAUSTED, payload={"status": 302}, message="天配额超限"),
            "ak2": ok_result,
        }) as fake:
            result = gw.request("/place/v2/search", {"query": "粤菜"}, kind="search", ttl=0)
            snap = gw.snapshot()
        assert result.ok, f"ak1 配额耗尽应自动换 ak2 成功: {result.status} {result.message}"
        assert ("/place/v2/search", "ak1") in [c for c in fake.calls if c[1]]
        assert snap["ak_usage"][0]["breaker_seconds_left"] > 0, "ak1 应被熔断"
        assert snap["ak_usage"][1]["breaker_seconds_left"] == 0, "ak2 不应被连坐"
        ok("地图故障转移：ak1 配额耗尽即熔断，自动换 ak2 成功，互不连坐")
    finally:
        config.BAIDU_MAP_AKS, config.BAIDU_MAP_AK = old_aks, old_ak


def test_map_per_ak_quota():
    old_aks = config.BAIDU_MAP_AKS
    old_limit = config.MAP_DAILY_SEARCH
    config.BAIDU_MAP_AKS = ["ak1", "ak2"]
    config.MAP_DAILY_SEARCH = 1  # 每把 AK 只允许 1 次检索
    try:
        with FakeMap({}):
            r1 = gw.request("/place/v2/search", {"query": "q1"}, kind="search", ttl=0)
            r2 = gw.request("/place/v2/search", {"query": "q2"}, kind="search", ttl=0)
            r3 = gw.request("/place/v2/search", {"query": "q3"}, kind="search", ttl=0)
        assert r1.ok and r2.ok, "两把 AK 各自的第一次调用都应成功"
        assert r3.status == gw.QUOTA_EXHAUSTED, f"两把 AK 用满后应报配额耗尽: {r3.status}"
        ok("地图每 AK 独立日配额：各用 1 次成功，池子总量用尽后如实报配额耗尽")
    finally:
        config.BAIDU_MAP_AKS = old_aks
        config.MAP_DAILY_SEARCH = old_limit


def test_map_ak_pool_capacity():
    old_aks = config.BAIDU_MAP_AKS
    config.BAIDU_MAP_AKS = ["ak1", "ak2", "ak3", "ak4"]
    try:
        with FakeMap({}):
            for i in range(8):
                r = gw.request("/place/v2/search", {"query": f"batch{i}"}, kind="search", ttl=0)
                assert r.ok, f"第 {i+1} 次调用应成功: {r.status}"
            snap = gw.snapshot()
        assert snap["ak_count"] == 4
        assert snap["ak_daily_search_limit_total"] == config.MAP_DAILY_SEARCH * 4
        used = sorted(a["search"] for a in snap["ak_usage"])
        assert used == [2, 2, 2, 2], f"8 次调用应被轮询摊匀到 4 把 AK: {used}"
        ok("AK 池容量与负载均衡：8 次调用轮询摊匀到 4 把 AK（各 2 次）")
    finally:
        config.BAIDU_MAP_AKS = old_aks


if __name__ == "__main__":
    reset_pool_for_tests()
    test_llm_failover()
    test_llm_busy()
    test_llm_no_avalanche()
    test_agent_busy_message()
    test_context_budget()
    test_map_failover()
    test_map_per_ak_quota()
    test_map_ak_pool_capacity()
    reset_pool_for_tests()
    print(f"=== 全部 {PASS} 项并发治理测试通过 ===")
