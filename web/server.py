"""网页版服务器：标准库 http.server 实现，零第三方依赖，可发布为在线服务。

部署适配（WorkBuddy 发布规范）：
  - 监听 PORT 环境变量（本地默认 8787），绑定 0.0.0.0；
  - 每个浏览器会话（session_id）拥有独立的 agent 实例：对话记忆、
    用户画像互不干扰，画像持久化到 data/sessions/<sid>.json；
  - API 密钥只存在于服务端 .env / 环境变量，前端永远接触不到。

跑法：python web/server.py   然后浏览器打开 http://localhost:8787
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 项目根加入搜索路径

import config
from agent.core import FoodAdvisorAgent, _trace
from agent.memory import ConversationMemory, UserProfile
from llm.client import get_pool
from tools import map_gateway
from tools import kb_search as kb

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
INDEX_HTML = (BASE_DIR / "static" / "index.html").read_text(encoding="utf-8")
SESSIONS_DIR = PROJECT_DIR / "data" / "sessions"
MAX_SESSIONS = 200  # 内存中最多保留的活跃会话数

# 代码指纹：只对源码文件做哈希，不含任何密钥/画像数据。
# 用途：/api/status 暴露该指纹，可随时比对「线上运行的代码」与「本地 SDK 源码」是否一致。
BUILD_FILES = (
    "config.py",
    "agent/core.py",
    "agent/memory.py",
    "llm/client.py",
    "tools/__init__.py",
    "tools/food_search.py",
    "tools/route_planner.py",
    "tools/budget_nutrition.py",
    "tools/map_gateway.py",
    "tools/kb_index.py",
    "tools/kb_search.py",
    "web/server.py",
    "web/static/index.html",
)


def build_fingerprint() -> dict:
    digest = hashlib.sha256()
    count = 0
    for rel in BUILD_FILES:
        path = PROJECT_DIR / rel
        if not path.exists():
            continue
        # 换行归一化：Windows 工作区是 CRLF，git 检出/CI 是 LF，
        # 若按原始字节哈希，同一份代码会算出不同指纹（队友会误报"线上不一致"）。
        raw = path.read_bytes().replace(b"\r\n", b"\n")
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(raw)).encode("ascii"))
        digest.update(b"\0")
        digest.update(raw)
        count += 1
    return {"hash": digest.hexdigest()[:12], "files": count}


BUILD = build_fingerprint()
START_TS = time.time()


class Session:
    """一个访客 = 一个独立的 agent（独立记忆 + 独立画像 + 独立限流窗口）。"""

    def __init__(self, sid: str) -> None:
        self.lock = threading.Lock()
        self.last_used = time.time()
        self.recent_calls: list[float] = []  # 最近提问时间戳，用于每分钟限流
        safe = "".join(c for c in sid if c.isalnum() or c in "-_")[:64] or "anonymous"
        self.profile_path = str(SESSIONS_DIR / f"{safe}.json")
        self.profile = UserProfile(path=self.profile_path)
        self.agent = FoodAdvisorAgent(profile=self.profile, trace_fn=_trace)

    def allow_chat(self) -> tuple[bool, int]:
        """每会话每分钟提问上限；返回 (是否放行, 建议等待秒数)。"""
        now = time.time()
        self.recent_calls = [t for t in self.recent_calls if now - t < 60]
        if len(self.recent_calls) >= config.CHAT_RATE_PER_MIN:
            wait = int(60 - (now - self.recent_calls[0])) + 1
            return False, max(1, wait)
        self.recent_calls.append(now)
        return True, 0


_sessions: dict[str, Session] = {}
_sessions_guard = threading.Lock()

# 全站每日提问上限：防止被刷 → 地图/模型额度被烧穿
_global_guard = threading.Lock()
_global_day = time.strftime("%Y-%m-%d")
_global_count = 0


def allow_global_chat() -> tuple[bool, int]:
    global _global_day, _global_count
    today = time.strftime("%Y-%m-%d")
    with _global_guard:
        if today != _global_day:
            _global_day, _global_count = today, 0
        if _global_count >= config.CHAT_GLOBAL_DAILY:
            return False, 0
        _global_count += 1
        return True, config.CHAT_GLOBAL_DAILY - _global_count


def get_session(sid: str) -> Session:
    with _sessions_guard:
        sess = _sessions.get(sid)
        if sess is None:
            # 超限时淘汰最久未用的会话
            if len(_sessions) >= MAX_SESSIONS:
                oldest = min(_sessions, key=lambda k: _sessions[k].last_used)
                _sessions.pop(oldest, None)
            sess = Session(sid)
            _sessions[sid] = sess
        sess.last_used = time.time()
        return sess


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    timeout = 20  # 单连接读超时，避免半开连接长期占用线程

    def _send(self, code: int, payload, ctype: str = "application/json; charset=utf-8") -> None:
        data = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            # 手机切后台/网络抖动导致的断连是常态，不能让它冒泡成线程异常
            pass

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length).decode("utf-8") if length else "{}"
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def do_GET(self) -> None:  # noqa: N802（http.server 约定）
        try:
            self._handle_get()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:  # 兜底：单请求出错不影响服务存活
            print(f"  [错误] GET {self.path} -> {type(exc).__name__}: {exc}")
            try:
                self._send(500, {"error": "服务器内部错误，请稍后重试"})
            except Exception:
                pass

    def _handle_get(self) -> None:
        path = urlparse(self.path).path  # 剥离 ?sharecode= 等查询参数再路由
        if path in ("/", "/index.html"):
            self._send(200, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/status":
            origin = config.origin_by_key(config.DEFAULT_ORIGIN)
            self._send(200, {
                "llm": config.LLM_ENABLED,
                "llm_model": config.LLM_MODEL if config.LLM_ENABLED else "",
                "llm_pool": get_pool().snapshot() if config.LLM_ENABLED else None,
                "map": config.MAP_ENABLED,
                "origin": f"{origin['name']}（{origin['coord']}）",
                "build": BUILD["hash"],
                "build_files": BUILD["files"],
                "config_source": config.CONFIG_SOURCE,
                "limits": {
                    "message_chars": config.MAX_MESSAGE_CHARS,
                    "chat_per_min": config.CHAT_RATE_PER_MIN,
                    "chat_daily": config.CHAT_GLOBAL_DAILY,
                    "tool_calls_per_turn": config.MAX_TOOL_CALLS_PER_TURN,
                },
                "map_stats": map_gateway.snapshot(),
            })
        elif path == "/api/health":
            self._send(200, {
                "ok": True,
                "pid": os.getpid(),
                "uptime_sec": round(time.time() - START_TS),
                "build": BUILD["hash"],
                "sessions": len(_sessions),
            })
        elif path == "/api/origins":
            self._send(200, {
                "default": config.DEFAULT_ORIGIN,
                "origins": [
                    {k: p[k] for k in ("key", "name", "coord", "source", "coord_type")}
                    for p in config.ORIGIN_PRESETS
                ],
            })
        elif path == "/api/kb/status":
            # 知识库规模（门店数/评价数/是否演示数据），前端顶部展示
            self._send(200, kb.kb_status())
        elif path == "/api/kb/search":
            params = parse_qs(urlparse(self.path).query)

            def value(key: str, default: str = "") -> str:
                return (params.get(key) or [default])[0]

            try:
                max_price = float(value("max_price")) if value("max_price") else None
            except ValueError:
                max_price = None
            try:
                max_walk = int(value("max_walk")) if value("max_walk") else None
            except ValueError:
                max_walk = None

            items = kb.search_stores(
                query=value("q"),
                scene=value("scene"),
                max_price=max_price,
                max_walk=max_walk,
                tag=value("tag"),
                sort=value("sort", "relevance"),
            )
            self._send(200, {
                "demo": kb.is_demo(),
                "total": len(items),
                "items": items,
                "status": kb.kb_status(),
            })
        elif path == "/api/kb/store":
            params = parse_qs(urlparse(self.path).query)
            name = (params.get("name") or [""])[0]
            detail = kb.store_detail(name)
            if detail is None:
                self._send(404, {"error": "知识库中没有这家店", "name": name})
            else:
                self._send(200, {"demo": kb.is_demo(), "store": detail})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            self._handle_post()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:  # 兜底：单请求出错不影响服务存活
            print(f"  [错误] POST {self.path} -> {type(exc).__name__}: {exc}")
            try:
                self._send(500, {"error": "服务器内部错误，请稍后重试"})
            except Exception:
                pass

    def _handle_post(self) -> None:
        path = urlparse(self.path).path  # 剥离 ?sharecode= 等查询参数再路由
        body = self._read_body()
        sid = str(body.get("session_id") or "anonymous")

        if path == "/api/profile":
            sess = get_session(sid)
            self._send(200, {"profile": sess.profile.data})
            return

        if path == "/api/chat":
            message = str(body.get("message", "")).strip()
            if not message:
                self._send(400, {"error": "消息不能为空"})
                return
            if len(message) > config.MAX_MESSAGE_CHARS:
                self._send(413, {
                    "error": f"消息太长（{len(message)} 字），请控制在 {config.MAX_MESSAGE_CHARS} 字以内",
                })
                return

            ok_global, _left = allow_global_chat()
            if not ok_global:
                self._send(429, {
                    "error": "今天全站提问次数已达上限，明天再来吧（额度用于保证比赛演示期间稳定）",
                })
                return

            sess = get_session(sid)
            allowed, wait = sess.allow_chat()
            if not allowed:
                self._send(429, {
                    "error": f"问得有点快，{wait} 秒后再试（每分钟最多 {config.CHAT_RATE_PER_MIN} 次）",
                    "retry_after": wait,
                })
                return

            # 出发点切换（页面选择校门/宿舍区；坐标均来自地图实测）
            origin_key = str(body.get("origin") or "").strip()
            if origin_key:
                sess.agent.set_origin(origin_key)

            trace: list[str] = []

            def collector(text: str) -> None:
                trace.append(text)
                print(f"  [推理] {text}")  # 终端同步可见，方便调试

            with sess.lock:
                sess.agent._trace_fn = collector
                try:
                    reply = sess.agent.run(message)
                except Exception as exc:  # 兜底：不让服务器崩
                    reply = f"（服务器内部错误：{exc}）"
                finally:
                    sess.agent._trace_fn = _trace
                cards = sess.agent.last_cards
                summary = sess.agent.last_summary
                dropped = sess.agent.last_dropped
                origin_name = sess.agent.origin["name"]
            self._send(200, {
                "reply": reply,
                "trace": trace,
                "cards": cards,
                "summary": summary,
                "dropped": dropped,
                "origin": origin_name,
                "profile": sess.profile.data,
                "map_stats": map_gateway.snapshot(),
            })
            return

        if path == "/api/reset":
            sess = get_session(sid)
            with sess.lock:
                sess.agent.memory = ConversationMemory("")
            self._send(200, {"ok": True})
            return

        self._send(404, {"error": "not found"})

    def log_message(self, fmt: str, *args) -> None:  # 安静模式
        pass


class Server(ThreadingHTTPServer):
    daemon_threads = True          # 请求线程随主进程退出，不阻塞重启
    allow_reuse_address = True
    request_queue_size = 64


def main() -> None:
    port = int(os.getenv("PORT", "8787"))
    mode = "LLM 模式" if config.LLM_ENABLED else "离线模式"
    print("=" * 56)
    print("  国际校区干饭军师 · 网页版已启动")
    print(f"  模式：{mode} | 百度地图：{'已连接' if config.MAP_ENABLED else '未配置'}")
    print(f"  端口：{port} | build：{BUILD['hash']} | 配置源：{config.CONFIG_SOURCE}")
    print(f"  限制：{config.MAX_MESSAGE_CHARS} 字 / 每分钟 {config.CHAT_RATE_PER_MIN} 次 / 日 {config.CHAT_GLOBAL_DAILY} 次")
    print("  关闭服务：Ctrl+C")
    print("=" * 56)
    while True:  # 单次异常退出后自动重启监听，避免线上整站不可用
        try:
            server = Server(("0.0.0.0", port), Handler)
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n服务器已关闭。")
            break
        except Exception as exc:
            print(f"[警告] 服务异常退出（{type(exc).__name__}: {exc}），2 秒后重启监听")
            time.sleep(2)


if __name__ == "__main__":
    main()
