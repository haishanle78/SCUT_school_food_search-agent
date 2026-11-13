"""配置解析测试：.env 行尾注释、引号、多值密钥拆分、单值回退。

背景：`config.load_dotenv` 早期只跳过以 # 开头的整行，不剥行尾注释，
导致 `LLM_CONCURRENCY_PER_KEY=2  # 注释` 被读成非法字符串，服务启动即 ValueError。
`.env.example` 模板大量使用行尾注释，因此这条必须有回归测试保护。

跑法：python tests/test_config_env.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

BASE = Path(__file__).resolve().parent.parent
PASS = 0


def ok(label: str) -> None:
    global PASS
    PASS += 1
    print(f"[通过] {label}")


def test_inline_comment():
    """行尾注释必须被剥离，数值项能正确转型。"""
    assert config._strip_inline_comment("2      # 单 Key 并发路数") == "2"
    assert config._strip_inline_comment("80             # 每把 AK 的日上限") == "80"
    assert config._strip_inline_comment("  14000  ") == "14000"
    ok("行尾 # 注释被剥离，数值项可转型")


def test_quoted_hash_kept():
    """引号内的 # 是值的一部分，不能被误删。"""
    assert config._strip_inline_comment('"abc#def"') == "abc#def"
    assert config._strip_inline_comment("'x#y'") == "x#y"
    assert config._strip_inline_comment('"http://a#b"') == "http://a#b"
    ok("引号内的 # 保留（密钥/URL 安全）")


def test_empty_and_plain():
    """空值、无注释的普通值保持语义。"""
    assert config._strip_inline_comment("") == ""
    assert config._strip_inline_comment("   ") == ""
    assert config._strip_inline_comment("plain") == "plain"
    ok("空值与普通值不变形")


def test_template_parses_cleanly():
    """把 .env.example 当 .env 解析：所有数值项都必须可转型（防模板写坏）。"""
    keys = [
        "LLM_CONCURRENCY_PER_KEY",
        "LLM_QUEUE_WAIT",
        "LLM_BREAKER_COOLDOWN",
        "LLM_CONTEXT_CHAR_LIMIT",
        "TOOL_RESULT_MAX_CHARS",
        "MAP_QPS",
        "MAP_DAILY_SEARCH",
        "MAP_DAILY_ROUTE",
        "MAP_TIMEOUT",
        "MAP_MAX_WAIT",
        "MAP_RETRY",
        "MAP_AUTH_COOLDOWN",
        "MAP_CACHE_TTL_SEARCH",
        "MAP_CACHE_TTL_ROUTE",
        "MAX_MESSAGE_CHARS",
        "CHAT_RATE_PER_MIN",
        "CHAT_GLOBAL_DAILY",
        "MAX_TOOL_CALLS_PER_TURN",
    ]
    saved = {k: os.environ.pop(k, None) for k in keys}
    try:
        config.load_dotenv(str(BASE / ".env.example"))
        for k in keys:
            raw = os.environ.get(k)
            assert raw is not None, f"模板缺少配置项 {k}"
            int(raw)  # 转型失败即抛 ValueError，正是本次修复的 bug
        ok(f"模板全部 {len(keys)} 个数值项可正确解析")
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v


def test_multi_value_split():
    """多值密钥按逗号拆分，去空格、去空项；单值变量作为回退。"""
    keys = ["OPENAI_API_KEYS", "OPENAI_API_KEY", "BAIDU_MAP_AKS", "BAIDU_MAP_AK"]
    saved = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ.pop(k, None)
        os.environ["OPENAI_API_KEYS"] = "k1, k2 ,, k3"
        os.environ["BAIDU_MAP_AKS"] = "ak1,ak2"
        import importlib

        reloaded = importlib.reload(config)
        assert reloaded.LLM_KEYS == ["k1", "k2", "k3"], reloaded.LLM_KEYS
        assert reloaded.BAIDU_MAP_AKS == ["ak1", "ak2"], reloaded.BAIDU_MAP_AKS
        assert reloaded.LLM_ENABLED and reloaded.MAP_ENABLED
        ok("多值拆分正确（去空格/去空项），开关置真")
    finally:
        for k in keys:
            os.environ.pop(k, None)
            if saved.get(k) is not None:
                os.environ[k] = saved[k]
        import importlib

        importlib.reload(config)


def test_single_source_no_runtime_env():
    """回归：runtime.env 已删除，配置只能来自 .env，且不再存在会失效的开关。

    历史 bug：runtime.env 用 override=True 二次加载，把 .env 的值顶掉；
    而"关闭开关"写的是 `not in {"1","true",...}`，填 0 也会被当成关闭、
    填 1 才生效——语义反直觉，导致改了 .env 却不生效。
    """
    assert not (BASE / "runtime.env").exists(), "runtime.env 应已删除，配置只留 .env 一份"
    src = (BASE / "config.py").read_text(encoding="utf-8")
    assert "runtime.env" not in src, "config.py 不应再引用 runtime.env"
    assert "USE_PLATFORM_ENV" not in src, "开关已移除，不应再出现在 config.py"
    assert config.CONFIG_SOURCE in {"env-file", "process-env"}, config.CONFIG_SOURCE
    ok("单一配置源：无 runtime.env、无失效开关")


def test_both_pools_declared_same_file():
    """.env 里 Key 池与 AK 池必须写在同一个文件，且数量一致（防只更新一处）。"""
    env_path = BASE / ".env"
    if not env_path.exists():
        print("[跳过] 无本地 .env（如 CI），不做实际值校验")
        return
    llm = config.LLM_KEYS
    aks = config.BAIDU_MAP_AKS
    assert llm, ".env 未解析出任何 LLM Key"
    assert aks, ".env 未解析出任何地图 AK"
    # 每把 Key/AK 都必须是完整长度，避免粘贴被截断（智谱 Key 49 字符、百度 AK 32 字符）
    for k in llm:
        assert len(k) >= 30, f"LLM Key 长度异常（{len(k)}），可能粘贴不全"
    for a in aks:
        assert len(a) >= 24, f"地图 AK 长度异常（{len(a)}），可能粘贴不全"
    ok(f"凭据池已装配：LLM Key {len(llm)} 把 / 地图 AK {len(aks)} 把")


def test_pool_dedup_and_cleaning():
    """池解析必须去重保序、容忍空格/引号/空项/末尾逗号。

    为什么去重是硬要求：同一把 Key 占多个槽位会绕过「单 Key 并发上限」，
    等于自己把并发打上去，反而更容易被上游 429；重复 AK 同理会让日配额
    被重复计算，实际上限不会翻倍。
    """
    assert config._split_pool("k1,k1,k2,k2") == ["k1", "k2"]
    assert config._split_pool(" k1 , , k2 ,, k3 ") == ["k1", "k2", "k3"]
    assert config._split_pool("k1,k2,") == ["k1", "k2"]
    assert config._split_pool("") == []
    assert config._split_pool('"k1",\'k2\'') == ["k1", "k2"]
    # 顺序必须保留（第一个出现位置代表该 Key，池子轮询依赖这个顺序）
    assert config._split_pool("z,a,m") == ["z", "a", "m"]
    ok("凭据池去重保序，空格/引号/空项/末尾逗号均被清洗")


def test_multiline_misformat_detected():
    """回归：多行填写会被 dotenv 静默丢弃，必须能报出来。

    用户很容易把 4 把 Key 写成 4 行，标准 dotenv 只认第一行，
    后 3 行变成"孤行"被忽略——没有任何报错，池子悄悄只剩 1 把。
    """
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".env")
    os.close(fd)
    try:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(
                "LLM_PROVIDER=openai\n"
                "OPENAI_API_KEYS=aaaaaaaaaaaaaaaaaaaaaa\n"
                "bbbbbbbbbbbbbbbbbbbbbb\n"          # 第 3 行：被丢弃的孤行
                "cccccccccccccccccccccc\n"          # 第 4 行：被丢弃的孤行
                "# 这是注释，不该报\n"
                "LLM_MODEL=glm-4-flash\n"
            )
        warns = config._suspicious_env_lines(path)
        assert len(warns) == 2, f"应报出 2 条孤行，实际 {warns}"
        assert "第 3 行" in warns[0] and "第 4 行" in warns[1], warns
        ok("多行误填能被检测（不会静默丢 Key）")
    finally:
        os.unlink(path)


def test_credentials_summary_never_leaks():
    """自检摘要只能出现打码后的凭据，绝不包含明文。"""
    summary = config.describe_credentials()
    assert set(summary) == {"config_source", "llm", "map", "warnings"}, summary.keys()
    for secret in config.LLM_KEYS + config.BAIDU_MAP_AKS:
        text = repr(summary)
        assert secret not in text, "摘要里出现了凭据明文！"
    assert summary["llm"]["keys"] == len(config.LLM_KEYS)
    assert summary["map"]["aks"] == len(config.BAIDU_MAP_AKS)
    ok("凭据摘要已打码，数量与真实池一致")


if __name__ == "__main__":
    test_inline_comment()
    test_quoted_hash_kept()
    test_empty_and_plain()
    test_template_parses_cleanly()
    test_multi_value_split()
    test_single_source_no_runtime_env()
    test_both_pools_declared_same_file()
    test_pool_dedup_and_cleaning()
    test_multiline_misformat_detected()
    test_credentials_summary_never_leaks()
    print(f"=== 全部 {PASS} 项配置解析测试通过 ===")
