"""全局配置：从 .env 与默认值加载。

唯一配置源：进程环境变量 > .env > 代码默认值。没有第二层覆盖文件。
"""

from __future__ import annotations

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _strip_inline_comment(value: str) -> str:
    """去掉行尾 # 注释，但保留引号内的 #（模板里大量使用行尾注释）。"""
    value = value.strip()
    if not value:
        return value
    if value[0] in {'"', "'"}:
        quote = value[0]
        end = value.find(quote, 1)
        if end > 0:
            return value[1:end].strip()
        return value[1:].strip()
    return value.split("#", 1)[0].strip()


def load_dotenv(path: str, override: bool = False) -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = _strip_inline_comment(value)
            if key and value and (override or not os.getenv(key)):
                os.environ[key] = value


# .env 是唯一配置源：本地与云端都只读这一个文件，不再有第二层覆盖。
# 这样「改了 .env 却不生效」这类问题从根上消失——不需要记任何开关。
# 优先级：进程环境变量（平台注入/命令行） > .env 文件 > 代码默认值。
# 云端没有 .env 时，只要平台环境变量按 .env 里的同名键填好，照样能跑。
load_dotenv(os.path.join(BASE_DIR, ".env"))
CONFIG_SOURCE = "env-file"
if not os.path.exists(os.path.join(BASE_DIR, ".env")):
    CONFIG_SOURCE = "process-env"


# ---------- LLM ----------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "").strip().lower()
LLM_API_KEY = os.getenv("DEEPSEEK_API_KEY" if LLM_PROVIDER == "deepseek" else "OPENAI_API_KEY", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "").strip() or ("deepseek-chat" if LLM_PROVIDER == "deepseek" else "gpt-4o-mini")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").strip() or (
    "https://api.deepseek.com/v1/chat/completions"
    if LLM_PROVIDER == "deepseek"
    else "https://api.openai.com/v1/chat/completions"
)
LLM_TEMPERATURE = 0.4  # 推荐类任务要稳，别太发散

# 多 Key 分流：逗号分隔。兼容旧的单 Key 变量（OPENAI_API_KEY / DEEPSEEK_API_KEY）。
# 去重保序是硬要求：同一把 Key 若占了多个槽位，会绕过"单 Key 并发上限"的约束，
# 等于自己把并发打高，反而更容易被上游 429。所以先按出现顺序去重再建池。
def _split_pool(raw: str) -> list[str]:
    """把逗号分隔的凭据串解析成去重列表；顺序保留（首个位置代表该 Key）。"""
    seen: set[str] = set()
    out: list[str] = []
    for item in raw.split(","):
        token = item.strip().strip('"').strip("'").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


_keys_raw = os.getenv("OPENAI_API_KEYS", "") or os.getenv("DEEPSEEK_API_KEYS", "")
LLM_KEYS = _split_pool(_keys_raw)
if not LLM_KEYS and LLM_API_KEY and "在此填入" not in LLM_API_KEY:
    LLM_KEYS = [LLM_API_KEY]

# 占位 key 检测：.env 里还是模板文本时视为未配置
LLM_ENABLED = bool(LLM_PROVIDER in {"deepseek", "openai"} and LLM_KEYS)

# ---------- LLM 并发治理（Key 池） ----------
# glm-4-flash 免费档：单 Key 并发上限 2；上下文超 8K token 后速率被大幅限制。
LLM_CONCURRENCY_PER_KEY = int(os.getenv("LLM_CONCURRENCY_PER_KEY", "2"))   # 单 Key 并发路数
LLM_QUEUE_WAIT = float(os.getenv("LLM_QUEUE_WAIT", "10"))                  # 排队等待上限（秒），超时返回"稍后再试"
LLM_BREAKER_COOLDOWN = int(os.getenv("LLM_BREAKER_COOLDOWN", "120"))       # Key 限流后的熔断时长（秒）
# 上下文字符预算（messages + tools 的 JSON 总字符数）。中文约 1 字符 ≈ 1 token，
# 14000 字符 ≈ 8K token 出头，超限会主动裁剪，避免触发智谱的 8K 限速档。
LLM_CONTEXT_CHAR_LIMIT = int(os.getenv("LLM_CONTEXT_CHAR_LIMIT", "14000"))
TOOL_RESULT_MAX_CHARS = int(os.getenv("TOOL_RESULT_MAX_CHARS", "600"))     # 裁剪时单条工具结果保留长度

# ---------- 地图 ----------
# 多 AK 分流：逗号分隔（每个 AK 需为服务端类型应用，不勾 SN 校验）。
# 兼容旧的单 AK 变量 BAIDU_MAP_AK。同样去重保序：重复 AK 会让日配额被重复计算，
# 实际上限并不会翻倍，只会在"以为还有额度"时被打回配额耗尽。
_aks_raw = os.getenv("BAIDU_MAP_AKS", "")
BAIDU_MAP_AKS = _split_pool(_aks_raw)
BAIDU_MAP_AK = os.getenv("BAIDU_MAP_AK", "").strip().strip('"').strip("'").strip()
if not BAIDU_MAP_AKS and BAIDU_MAP_AK:
    BAIDU_MAP_AKS = [BAIDU_MAP_AK]
if BAIDU_MAP_AKS and not BAIDU_MAP_AK:
    BAIDU_MAP_AK = BAIDU_MAP_AKS[0]
MAP_ENABLED = bool(BAIDU_MAP_AKS)

# 出发点：华工广州国际校区（番禺区南村镇兴业大道东777号）
# 坐标系：百度 bd09ll。以下校门坐标均由百度地图地点检索实测获得（2026-11-10），
# 不是估算值；如需新增校门，用拾取器取点后注明来源：
# https://api.map.baidu.com/lbsapi/getpoint/index.html
CAMPUS_LOCATION = os.getenv("CAMPUS_LOCATION", "23.017973,113.411971")

ORIGIN_PRESETS = [
    {"key": "dorm", "name": "宿舍区（校区中心）", "coord": "23.017973,113.411971",
     "source": "百度地点检索实测 2026-11-10", "coord_type": "bd09ll"},
    {"key": "south_gate", "name": "南门", "coord": "23.014617,113.414821",
     "source": "百度地点检索实测 2026-11-10", "coord_type": "bd09ll"},
    {"key": "north_gate", "name": "北门", "coord": "23.022913,113.410872",
     "source": "百度地点检索实测 2026-11-10", "coord_type": "bd09ll"},
    {"key": "southwest_gate", "name": "西南门", "coord": "23.014569,113.407333",
     "source": "百度地点检索实测 2026-11-10", "coord_type": "bd09ll"},
    {"key": "northwest_gate", "name": "西北门", "coord": "23.020240,113.408009",
     "source": "百度地点检索实测 2026-11-10", "coord_type": "bd09ll"},
]
DEFAULT_ORIGIN = os.getenv("ORIGIN_KEY", "dorm")


def origin_by_key(key: str | None) -> dict:
    """按 key 取出发点；未知 key 回落到默认，绝不返回空。"""
    for preset in ORIGIN_PRESETS:
        if preset["key"] == (key or ""):
            return dict(preset)
    for preset in ORIGIN_PRESETS:
        if preset["key"] == DEFAULT_ORIGIN:
            return dict(preset)
    return dict(ORIGIN_PRESETS[0])

# ---------- Agent ----------
MAX_STEPS = 8          # ReAct 循环最大步数（防死循环）
HISTORY_TURNS = 12     # 会话记忆保留的最近轮数

# ---------- 数据文件 ----------
DATA_DIR = os.path.join(BASE_DIR, "data")
RESTAURANTS_PATH = os.path.join(DATA_DIR, "restaurants.json")
CANTEEN_PATH = os.path.join(DATA_DIR, "canteen.json")
DELIVERY_PATH = os.path.join(DATA_DIR, "delivery.json")
PROFILE_PATH = os.path.join(DATA_DIR, "user_profile.json")

# ---------- 知识库（评分与评价检索） ----------
# 真实评价语料：一行一条 JSON（字段见 data/kb/README.md）
KB_DIR = os.path.join(DATA_DIR, "kb")
KB_REVIEWS_PATH = os.path.join(KB_DIR, "reviews.jsonl")
# 演示模式：只读 KB_DEMO_DIR 下的 demo_stores.json / demo_reviews.jsonl，
# 绝不与真实库混用。仓库**不附带**演示语料（避免假商家混进正式推荐）；
# 需要演示或测试时，请自备该目录，测试用例会临时生成固定数据。
KB_DEMO = os.getenv("KB_DEMO", "").strip().lower() in {"1", "true", "yes", "on"}
KB_DEMO_DIR = os.getenv("KB_DEMO_DIR", os.path.join(DATA_DIR, "examples"))
KB_MIN_SAMPLES = int(os.getenv("KB_MIN_SAMPLES", "3"))  # 少于该样本数不出均值，只报样本数

# ---------- 地图调用治理（控制额度，避免触发并发/配额限制） ----------
# 百度个人开发者默认权益（每把 AK）：地点检索 100 次/日、3 次/秒；路线规划 5000 次/日、3 次/秒。
# 下面两个日配额是【每把 AK】的额度，池子总容量 = 配额 × AK 数量。留余量：默认 2 QPS、
# 检索 80 次/日、路线 4000 次/日。以控制台实际配置为准。
MAP_QPS = float(os.getenv("MAP_QPS", "2"))                    # 服务端全局限速（跨会话共享）
MAP_DAILY_SEARCH = int(os.getenv("MAP_DAILY_SEARCH", "80"))   # 地点检索日上限
MAP_DAILY_ROUTE = int(os.getenv("MAP_DAILY_ROUTE", "4000"))   # 路线规划日上限
MAP_TIMEOUT = float(os.getenv("MAP_TIMEOUT", "8"))            # 单次请求超时（秒）
MAP_MAX_WAIT = float(os.getenv("MAP_MAX_WAIT", "4"))          # 等令牌的最长时间，超时直接降级
MAP_RETRY = int(os.getenv("MAP_RETRY", "1"))                  # 限流/超时的有限重试次数
MAP_BACKOFF = float(os.getenv("MAP_BACKOFF", "0.6"))          # 重试退避基数（秒）
MAP_AUTH_COOLDOWN = int(os.getenv("MAP_AUTH_COOLDOWN", "600"))  # 权限/配额异常后的熔断时长（秒）
MAP_CACHE_TTL_SEARCH = int(os.getenv("MAP_CACHE_TTL_SEARCH", "600"))  # 检索缓存（秒）
MAP_CACHE_TTL_ROUTE = int(os.getenv("MAP_CACHE_TTL_ROUTE", "600"))    # 路线缓存（秒）
MAP_USAGE_PATH = os.path.join(DATA_DIR, "map_usage.json")             # 当日调用计数（持久化）

# ---------- 天气感知（复用百度地图 AK；天气接口免费额度独立，不占检索/路线配额） ----------
WEATHER_DISTRICT_ID = os.getenv("WEATHER_DISTRICT_ID", "440113")   # 广州市番禺区（校区所在行政区划代码）
WEATHER_TIMEOUT = float(os.getenv("WEATHER_TIMEOUT", "6"))         # 单次天气请求超时（秒）
WEATHER_CACHE_TTL = int(os.getenv("WEATHER_CACHE_TTL", "1800"))    # 缓存 30 分钟：天气变化慢，避免频繁打接口
WEATHER_ENABLED = bool(BAIDU_MAP_AK)                               # 有地图 AK 才启用；无 AK 时静默降级

# ---------- 接口安全与预算 ----------
MAX_MESSAGE_CHARS = int(os.getenv("MAX_MESSAGE_CHARS", "400"))       # 单条消息长度上限
CHAT_RATE_PER_MIN = int(os.getenv("CHAT_RATE_PER_MIN", "8"))         # 每会话每分钟提问上限
CHAT_GLOBAL_DAILY = int(os.getenv("CHAT_GLOBAL_DAILY", "300"))       # 全站每日提问上限
MAX_TOOL_CALLS_PER_TURN = int(os.getenv("MAX_TOOL_CALLS_PER_TURN", "10"))  # 单轮工具调用预算


# ---------- 启动自检：把「填了什么、几把、有没有填错」讲清楚 ----------
def _mask_secret(token: str) -> str:
    """凭据打码：只留头尾，足够核对是否填对同一把，又不会泄露。"""
    if len(token) <= 8:
        return token[:2] + "***"
    return token[:6] + "***" + token[-4:]


def _suspicious_env_lines(path: str) -> list[str]:
    """找出疑似「多行填写」造成的孤行。

    标准 dotenv 不支持把一个变量拆成多行：写成
        OPENAI_API_KEYS=key1
        key2
    时 key2 那行既没有 `=` 也不以 `#` 开头，会被静默丢弃。
    这里把它揪出来，避免"以为填了 4 把、实际只生效 1 把"。
    """
    if not os.path.exists(path):
        return []
    found: list[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line or line.startswith("#") or "=" in line:
                continue
            if len(line) >= 16:  # 像一段凭据，而不是手误的零散字符
                found.append(f"第 {lineno} 行: {line[:6]}...{line[-4:]}（{len(line)} 字符）")
    return found


def describe_credentials() -> dict:
    """返回可安全打印的凭据装配摘要（绝不包含明文）。"""
    env_path = os.path.join(BASE_DIR, ".env")
    return {
        "config_source": CONFIG_SOURCE,
        "llm": {
            "provider": LLM_PROVIDER or "(未设置)",
            "model": LLM_MODEL,
            "base_url": LLM_BASE_URL,
            "enabled": LLM_ENABLED,
            "keys": len(LLM_KEYS),
            "concurrency_total": len(LLM_KEYS) * LLM_CONCURRENCY_PER_KEY,
            "masked": [_mask_secret(k) for k in LLM_KEYS],
        },
        "map": {
            "enabled": MAP_ENABLED,
            "aks": len(BAIDU_MAP_AKS),
            "masked": [_mask_secret(a) for a in BAIDU_MAP_AKS],
            "daily_search_total": MAP_DAILY_SEARCH * max(1, len(BAIDU_MAP_AKS)),
            "daily_route_total": MAP_DAILY_ROUTE * max(1, len(BAIDU_MAP_AKS)),
        },
        "warnings": _suspicious_env_lines(env_path),
    }
