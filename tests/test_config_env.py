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


if __name__ == "__main__":
    test_inline_comment()
    test_quoted_hash_kept()
    test_empty_and_plain()
    test_template_parses_cleanly()
    test_multi_value_split()
    print(f"=== 全部 {PASS} 项配置解析测试通过 ===")
