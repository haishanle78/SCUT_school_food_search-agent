"""全局配置：从 .env 与默认值加载。"""

from __future__ import annotations

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


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
            value = value.strip().strip('"').strip("'")
            if key and value and (override or not os.getenv(key)):
                os.environ[key] = value


load_dotenv(os.path.join(BASE_DIR, ".env"))
# 云端可能保留上一次部署的环境变量，且 .env 不一定随源码上传。
# runtime.env 是随项目发布的服务端专用配置，覆盖旧值，保证线上模型/坐标一致。
# 但平台侧若改用「环境变量/密钥管理」注入新凭据，设置 USE_PLATFORM_ENV=1 即可让
# 平台注入值优先，避免打包配置把云端新密钥覆盖回旧值。
CONFIG_SOURCE = "platform-env"
if os.getenv("USE_PLATFORM_ENV", "").strip().lower() not in {"1", "true", "yes", "on"}:
    load_dotenv(os.path.join(BASE_DIR, "runtime.env"), override=True)
    CONFIG_SOURCE = "runtime.env"


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

# 占位 key 检测：.env 里还是模板文本时视为未配置
LLM_ENABLED = bool(LLM_PROVIDER in {"deepseek", "openai"} and LLM_API_KEY and "在此填入" not in LLM_API_KEY)

# ---------- 地图 ----------
BAIDU_MAP_AK = os.getenv("BAIDU_MAP_AK", "").strip()
MAP_ENABLED = bool(BAIDU_MAP_AK)

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

# ---------- 地图调用治理（控制额度，避免触发并发/配额限制） ----------
# 百度个人开发者默认权益：地点检索 100 次/日、3 次/秒；路线规划 5000 次/日、3 次/秒。
# 这里留余量：默认 2 QPS、检索 80 次/日、路线 4000 次/日。以控制台实际配置为准。
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

# ---------- 接口安全与预算 ----------
MAX_MESSAGE_CHARS = int(os.getenv("MAX_MESSAGE_CHARS", "400"))       # 单条消息长度上限
CHAT_RATE_PER_MIN = int(os.getenv("CHAT_RATE_PER_MIN", "8"))         # 每会话每分钟提问上限
CHAT_GLOBAL_DAILY = int(os.getenv("CHAT_GLOBAL_DAILY", "300"))       # 全站每日提问上限
MAX_TOOL_CALLS_PER_TURN = int(os.getenv("MAX_TOOL_CALLS_PER_TURN", "10"))  # 单轮工具调用预算
