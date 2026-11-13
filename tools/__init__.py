"""工具注册表：把工具函数 + OpenAI function calling schema 打包给 Agent。

⚠️ 工具的 description 就是 LLM 的"路由依据"——写得越准确，选工具越准。

数据一致性分工（重要）：
  - 模型负责：判断意图（指定门店/泛推荐/路线/比较）、选择门店（uid）、
    解释推荐理由/避雷/值不值；
  - 后端负责：门店身份（uid → 名称/地址/行政区/坐标）、是否餐饮门店、
    范围复核（附近推荐 3 公里）、路线事实、路线跳转链接、未知字段标注。
  卡片中的事实字段一律由后端从 poi_store 注入，不接受模型编写。
"""

from __future__ import annotations

import inspect
import json
from typing import Any, Callable, Dict, List, Tuple

from agent.memory import UserProfile
from tools.food_search import food_search
from tools.route_planner import find_restaurant, plan_route, search_poi, verify_route
from tools.budget_nutrition import estimate_budget, health_hint
from tools.weather_gateway import get_weather

Registry = Dict[str, Callable[..., str]]


def _adapt(fn: Callable[..., str]) -> Callable[..., str]:
    """容错适配层：模型多传的参数直接忽略，而不是让整个工具报 TypeError。

    小模型（如免费档 GLM）偶尔会自作主张加参数（radius、origin、location…）。
    严格报错会让一次提问彻底失败，丢弃未知参数并显式告知模型才是正确做法。
    """
    params = set(inspect.signature(fn).parameters)

    def call(**kwargs):
        used = {k: v for k, v in kwargs.items() if k in params}
        ignored = sorted(set(kwargs) - params)
        try:
            result = fn(**used)
        except TypeError as exc:
            return json.dumps(
                {
                    "error": f"参数不匹配: {exc}",
                    "可用参数": sorted(params),
                    "已收到参数": sorted(kwargs),
                },
                ensure_ascii=False,
            )
        if ignored and isinstance(result, str) and result.lstrip().startswith("{"):
            try:
                data = json.loads(result)
                data["_已忽略参数"] = ignored
                return json.dumps(data, ensure_ascii=False)
            except json.JSONDecodeError:
                pass
        return result

    return call


def remember_preference(key: str, value: str, profile: UserProfile) -> str:
    """（闭包注入 profile，见 build_registry）LLM 主动维护用户画像。"""
    try:
        profile.update(key, value)
        return f"已记入画像: {key} += {value}"
    except Exception as exc:
        return f"画像更新失败: {exc}"


def format_recommendation(recommendations, summary=""):
    """LLM 给出最终回答前调用：把 Top3 推荐结构化，前端渲染成推荐卡片。

    工具本身只返回确认文案；core.py 会截获参数，并按 uid 从 poi_store
    注入后端事实字段，同时做硬校验（身份/餐饮/范围/路线）。
    """
    return (
        f"已提交 {len(recommendations)} 个候选。后端会逐条校验门店身份、是否餐饮门店、"
        "是否在范围内、路线是否已核实；未通过校验的条目会被剔除，未知字段显示为待确认。"
        "请在最终回答中围绕通过的店铺给整体建议（怎么选、什么时间去、值不值得跑一趟）。"
    )


def build_registry(
    profile: UserProfile,
    poi_store: Dict[str, Dict[str, Any]] | None = None,
) -> Tuple[Registry, List[Dict[str, Any]]]:
    store = poi_store if poi_store is not None else {}
    functions: Registry = {
        "food_search": _adapt(
            lambda query="", taste="", dining_mode="", budget_max=None, max_minutes=None: food_search(
                query,
                taste=taste,
                dining_mode=dining_mode,
                budget_max=budget_max,
                max_minutes=max_minutes,
                poi_store=store,
            )
        ),
        "plan_route": _adapt(plan_route),
        "find_restaurant": _adapt(lambda name: find_restaurant(name, poi_store=store)),
        "search_poi": _adapt(lambda keyword: search_poi(keyword, poi_store=store)),
        "verify_route": _adapt(lambda uid, mode="walking": verify_route(uid, poi_store=store, mode=mode)),
        "estimate_budget": _adapt(estimate_budget),
        "health_hint": _adapt(health_hint),
        "get_weather": _adapt(get_weather),
        "remember_preference": _adapt(lambda key, value: remember_preference(key, value, profile)),
        "format_recommendation": _adapt(format_recommendation),
    }

    schemas: List[Dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "food_search",
                "description": (
                    "美食检索（先本地、后联网）：优先查本地知识库（问卷/点评人工整理，"
                    "含学生评分/招牌菜/避雷）；本地无匹配时【自动】联网检索百度地图 "
                    "3 公里内真实门店（带 uid/地址/坐标，自动剔除停车场等附属设施）。"
                    "用户问『附近吃什么/推荐餐厅/想吃X』时用这个工具即可，"
                    "它已经内置联网兜底——【不需要】再单独调 search_poi，"
                    "除非返回明确提示联网失败或需要换品类词重查。禁止编造门店。"
                    "\n三个参数各管一件事，不要混用："
                    "query 只放菜系/品类（汉堡、日料、粥、粤菜），"
                    "taste 只放口味（辣、清淡、甜），"
                    "dining_mode 只放用餐方式（外卖=delivery，到店=dine_in）。"
                    "把『外卖』塞进 taste、或把『辣』塞进 query 都会导致检索方向错误。"
                    "用户问某一家具体店（『达海美食怎么样』『小意司日料多少钱』）时也用这个工具，"
                    "query 直接传店名，会按店名精确命中并返回该店真实评分/招牌菜/一句话点评/下单方式；"
                    "回答只能用返回字段，没写的一律说『资料里没提到』，禁止编造菜品和环境。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "菜系/品类关键词：汉堡、日料、寿司、粥、粉面、粤菜、烧腊、川菜。"
                                "只放『吃什么品类』，不要放口味词（辣/清淡）和场景词（外卖/聚餐）。"
                                "不知道品类就留空，不要填『外卖』。"
                            ),
                        },
                        "taste": {
                            "type": "string",
                            "description": (
                                "口味偏好，只能填：辣、清淡、甜、重口、鲜。"
                                "『外卖』不是口味，请放 dining_mode。"
                            ),
                        },
                        "dining_mode": {
                            "type": "string",
                            "enum": ["delivery", "dine_in"],
                            "description": (
                                "本次用餐方式。用户说『外卖/送到宿舍/不想出门/点个XX』→ delivery；"
                                "说『去店里/堂食/走过去』或没提→ dine_in。"
                                "delivery 时不核实步行路线，配送能力/费用/时长无依据时标待确认。"
                            ),
                        },
                        "budget_max": {"type": "number", "description": "人均预算上限（元）"},
                        "max_minutes": {
                            "type": "integer",
                            "description": "可接受的步行时间上限（分钟）；外卖场景下该参数无意义，不要传",
                        },
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "find_restaurant",
                "description": (
                    "【指定门店查询】用户说出具体店名（如『你知道B记鱼粥吗』『XX店在哪』）时用这个，"
                    "先确认这家店到底是什么、在哪。返回匹配度、地址、坐标、uid、是否超出校区范围。"
                    "找不到明确匹配时它会要求你如实说明并追问地标——"
                    "绝不能把别的店当成答案，也不能因为没找到就改推荐别的店。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "用户原话里的店名，尽量保留完整（含分店名）"},
                    },
                    "required": ["name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_poi",
                "description": (
                    "【附近推荐】联网检索真实门店：默认出发点 3 公里内（硬限制，跨区远店会被剔除），"
                    "自动排除停车场等附属设施。返回每家店的 uid、名称、地址、行政区、坐标、直线距离。"
                    "搜索用具体品类词（粥/粉面/粤菜/烧腊/川菜），抽象口味词会被展开。"
                    "接口限流/配额用尽时会明确返回错误状态——那不是『附近没有店』，要如实说明。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keyword": {"type": "string", "description": "搜索品类关键词，如'粤菜'、'粥'、'奶茶'"},
                    },
                    "required": ["keyword"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "verify_route",
                "description": (
                    "路线核实：按 uid 查询该门店从当前出发点的真实路线。"
                    "mode 默认 walking（只查步行，省额度）；用户明确要骑行/打车时才传 riding/driving。"
                    "这是路线数据的唯一权威来源，推荐前必须对候选店调用（用户只关心步行就只查步行）。"
                    "失败会返回具体原因（限流/配额/权限/超时），如实转述，禁止编造耗时。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "uid": {"type": "string", "description": "search_poi 或 find_restaurant 返回的门店 uid"},
                        "mode": {
                            "type": "string",
                            "enum": ["walking", "riding", "driving", "all"],
                            "description": "交通方式：默认 walking；用户要骑行/打车才选；all 仅在同一家店需要对比多种方式时使用",
                        },
                    },
                    "required": ["uid"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "plan_route",
                "description": (
                    "知识库内真实门店的路线查询（须有坐标）。知识库无真实数据时会引导改用 "
                    "find_restaurant + verify_route；不做店名联网模糊定位。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "place_name": {"type": "string", "description": "店铺名称（须与知识库中名称一致）"},
                        "mode": {
                            "type": "string",
                            "enum": ["walking", "riding", "driving", "all"],
                            "description": "交通方式，默认 walking",
                        },
                    },
                    "required": ["place_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "estimate_budget",
                "description": "预算营养师：按知识库店铺人均与聚餐人数核算总预算。知识库外门店返回提示，此时人均由你预估并注明'预估'。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "place_name": {"type": "string", "description": "店铺名称"},
                        "party_size": {"type": "integer", "description": "用餐人数，默认 1"},
                    },
                    "required": ["place_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "health_hint",
                "description": "预算营养师：根据知识库店铺招牌菜给出荤素搭配建议。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "place_name": {"type": "string", "description": "店铺名称"},
                    },
                    "required": ["place_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": (
                    "查询当前实时天气（校区所在番禺区，含气温/现象/今日预报/出行建议）。"
                    "用户主动问天气（『下雨了吗』『现在天气怎么样』『明天适合出门吃饭吗』）时调用；"
                    "天气会自动注入系统提示，普通推荐任务无需调用本工具。"
                    "服务不可用或异常时如实说明『暂时查不了天气』，不影响推荐本身，禁止编造天气。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "remember_preference",
                "description": "把用户的口味偏好、忌口、常去/不想再去的店、预算习惯记入长期画像，之后的对话都要遵守。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string", "enum": ["口味偏好", "忌口", "常去店铺", "不想再去", "预算习惯", "备注"]},
                        "value": {"type": "string", "description": "要记录的内容，一条即可，如'不吃香菜'"},
                    },
                    "required": ["key", "value"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "format_recommendation",
                "description": (
                    "推荐卡片生成器：确定最终推荐后、输出最终回答之前调用。"
                    "每家店必须传 uid（来自 search_poi / find_restaurant 或 food_search 知识库结果）；"
                    "堂食门店要先用 verify_route 核实路线，外卖店（类别=外卖）有坐标即可，不需要 verify_route。"
                    "名称、地址、坐标、耗时、距离、路线/地图链接由后端注入，你只填解释性字段和预估人均。"
                    "后端会做硬校验：非餐饮门店、未核实路线的堂食店、超出范围的条目会被剔除，"
                    "人均/评分/营业时间没有来源时前端显示『待确认』——不要编造。"
                    "没有合适的店就别调用本工具，直接如实告诉用户没有匹配。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "recommendations": {
                            "type": "array",
                            "description": "推荐店铺列表，按推荐度排序，最多 3 家。每家必须带 uid。",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "uid": {"type": "string", "description": "门店唯一编号（必填）"},
                                    "cuisine": {"type": "string", "description": "菜系/品类（可按品类推断）"},
                                    "price_per_person": {"type": "number", "description": "人均预估（元）；没有来源就留空，不要编"},
                                    "reason": {"type": "string", "description": "推荐理由，一句话"},
                                    "warning": {"type": "string", "description": "避雷提示，一句话"},
                                    "delivery": {"type": "boolean", "description": "是否支持外卖配送（不确定就省略）"},
                                    "worth_it": {"type": "string", "description": "值不值得跑一趟的一句话判断"},
                                },
                                "required": ["uid", "cuisine", "reason"],
                            },
                        },
                        "summary": {"type": "string", "description": "一句话总结这组推荐的适用场景"},
                    },
                    "required": ["recommendations"],
                },
            },
        },
    ]
    return functions, schemas
