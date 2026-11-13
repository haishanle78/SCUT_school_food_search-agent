"""ReAct 核心：LLM ↔ 工具 的完整调用循环。

这是智能体的"大脑"。每一轮：
  LLM 读取对话历史 → 决定「直接回答」还是「调用工具」→
  执行工具 → 结果回填 → 再思考……直到给出最终推荐。

事实与解释的分工（数据一致性铁律）：
  - 模型负责：选择门店（uid）、推荐理由、避雷提示、值不值判断；
  - 后端负责：门店身份（名称/地址/行政区/坐标）、路线事实（耗时/距离/
    来源/查询时间）、3 公里范围复核、步行上限过滤、官方路线跳转链接。
  卡片事实字段一律由 _inject_card_facts 从 poi_store 注入，覆盖模型编写值。

LLM 不可用时降级为规则编排（离线演示模式），保证任何场合都能跑。
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

import config
from agent.memory import ConversationMemory, UserProfile
from llm.client import AssistantMessage, LLMClient, ToolCall
from llm.keypool import LLMBusy
from tools import build_registry
from tools import map_gateway, weather_gateway
from tools.route_planner import (
    build_place_link,
    build_route_link,
    current_origin,
    enrich_poi_from_kb,
    set_origin,
)

MODE_KEYWORDS = {
    "riding": ("骑行", "单车", "自行车", "骑车"),
    "driving": ("打车", "开车", "驾车", "滴滴", "网约车"),
}

# 天气拿不到时注入的约束。实测发现：完全不提天气，模型会顺口编
# "现在天气晴朗，适合出门吃饭"——没有数据支撑的判断，违反"不编造"铁律。
# 所以宁可在 prompt 里明确告诉它"查不到"，让它如实回答。
WEATHER_UNAVAILABLE_HINT = (
    "\n【天气数据不可用】实时天气本次获取失败（未配置或接口异常）。\n"
    "若用户询问天气，如实说明暂时查不到实时天气；"
    "绝对不要凭空断言“今天晴朗”“温度适宜”“适合出门”这类没有数据支撑的判断。\n"
    "与天气无关的推荐照常进行，不要因此降低回答质量。\n\n"
)


def _trace(text: str) -> None:
    """推理过程可视化：答辩时评委能看到每一步在干什么。"""
    print(f"  [推理] {text}")


_WALK_TALK_RE = re.compile(r"步行|骑行|走路|走过去|打车|坐车|地铁|自驾|单车|自行车")


def _strip_walk_talk(text: Any) -> str:
    """外卖卡片文案清洗：删掉「步行 38 分钟」「配送约 35 分钟」这类与后端事实冲突的描述。

    模型即使被禁止，也会把 verify_route 拿到的数据顺手写进推荐理由，
    结果卡片上出现「推荐外卖 + 步行 139 分钟」这种自相矛盾的话。
    外卖卡片的时间只来自后端按距离估算的 transport.delivery_min，
    模型自己猜的分钟数（往往和后端算的差 5-10 分钟）一律删掉，只留口味/价格/分量这些解释。
    """
    raw = str(text or "").strip()
    if not raw:
        return ""
    kept = []
    for seg in re.split(r"(?<=[。；;！!？?，,])", raw):
        seg = seg.strip()
        if not seg:
            continue
        if _WALK_TALK_RE.search(seg):
            continue
        if re.search(r"\d+\s*分钟", seg):
            continue
        kept.append(seg)
    out = "".join(kept).strip(" ，,。;；")
    return out or "详见卡片上的配送参考"


def _estimate_delivery_min(distance_m: Any, fallback: Any = None) -> Any:
    """按直线距离估算外卖参考配送时长。

    知识库里的「平均配送分钟」是人工按品类猜的，7 公里外的店也写 40 分钟就不自洽了。
    卡片上的距离和时长是并排展示的，两者必须互相说得通，否则一眼假。
    """
    try:
        dist = float(distance_m)
    except (TypeError, ValueError):
        return fallback
    if dist <= 0:
        return fallback
    if dist <= 1000:
        return 25
    if dist <= 2000:
        return 30
    if dist <= 3500:
        return 38
    if dist <= 5000:
        return 45
    if dist <= 8000:
        return 55
    return 65
def _as_price(value: Any) -> float | None:
    """人均只接受正数。模型可能回传 '待确认' / '约30' 这类文本，
    一律视为"无来源"（显示待确认），绝不把占位文本当成人均事实。
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num if num > 0 else None


# 工具参数的典型字段：出现这些的 JSON 块一定是模型把调用参数当正文输出了
_ECHO_FIELDS = ('"uid"', '"cuisine"', '"recommendations"', '"reason"', '"price_per_person"')


def _strip_tool_echo(text: str) -> str:
    """清掉模型误写进正文的工具参数与内部日志。

    实测线上出现过：回答里整段重复
        {"uid": "b0c56ab7...", "cuisine": "肠粉", "reason": "..."}
    后面还跟着「剔除未通过校验的推荐：石磨肠粉（路线未核实…）」——
    这是模型的工具调用参数 + 后端内部日志，用户完全不该看到。

    做法：按括号配平扫描，只删除「含工具字段特征」的 {…} / […] 块，
    正常的中文括注、百分比、普通方括号不受影响。
    """
    raw = str(text or "")
    if not raw:
        return raw

    # 1) 先去掉 ```json ... ``` 代码块
    raw = re.sub(r"```(?:json)?\s*[\s\S]*?```", "", raw)

    # 2) 括号配平扫描，删除含工具字段的 JSON 块
    out: List[str] = []
    i, n = 0, len(raw)
    while i < n:
        ch = raw[i]
        if ch not in "{[":
            out.append(ch)
            i += 1
            continue
        depth, j, in_str, esc = 0, i, False, False
        while j < n:
            c = raw[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c in "{[":
                    depth += 1
                elif c in "}]":
                    depth -= 1
                    if depth == 0:
                        j += 1
                        break
            j += 1
        block = raw[i:j]
        if len(block) > 16 and any(k in block for k in _ECHO_FIELDS):
            i = j  # 整块丢弃
            continue
        out.append(block)
        i = j

    cleaned = "".join(out)

    # 3) 去掉后端内部日志（剔除原因不该给用户看）
    cleaned = re.sub(r"剔除未通过校验的推荐[:：][\s\S]*$", "", cleaned)
    # 4) 收拾多余空行与残留在行首的逗号
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"^[\s,，;；]*", "", cleaned)
    cleaned = cleaned.strip()

    # 清理过头（整段都是 JSON）就返回空，交给 _ensure_reply 兜底重写一句
    return cleaned


class FoodAdvisorAgent:
    def __init__(
        self,
        profile: UserProfile | None = None,
        trace_fn=None,
        origin_key: str | None = None,
    ) -> None:
        self.llm = LLMClient()
        self.profile = profile or UserProfile()
        # 出发点：页面可切换（宿舍区/各校门），坐标均来自地图实测
        self.origin_key = origin_key or config.DEFAULT_ORIGIN
        self.origin = config.origin_by_key(self.origin_key)
        set_origin(self.origin)
        # 会话级门店事实库：uid → {名称/地址/行政区/坐标/直线距离/路线}
        # search_poi 写入门店事实，verify_route 写入路线事实，
        # _inject_card_facts 读取它生成卡片事实字段。
        self.poi_store: Dict[str, Dict[str, Any]] = {}
        # 工具注册表：注入 profile（画像维护）与 poi_store（门店事实贯穿）
        self.functions, self.schemas = build_registry(self.profile, self.poi_store)
        self.memory = ConversationMemory("")
        # 推理过程输出通道：默认打印到终端，网页版注入收集器在前端展示
        self._trace_fn = trace_fn or _trace
        # 本轮的结构化推荐卡片（LLM 调用 format_recommendation 工具时截获，
        # 并经 _inject_card_facts 注入后端事实 + 硬校验）
        self.last_cards: List[Dict[str, Any]] | None = None
        self.last_summary: str = ""
        self.last_dropped: List[Dict[str, str]] = []
        # 本轮用户需要的交通方式（默认只查步行，省地图额度）
        self._needed_modes: List[str] = ["walking"]

    # 单轮标记（run() 每轮重置；放类级默认值，测试里 __new__ 构造的实例也能用）
    _format_attempted: bool = False   # 本轮是否已调用过 format_recommendation
    _used_food_search: bool = False   # 本轮是否成功调用过 food_search
    # 本次用餐方式：delivery（点外卖）/ dine_in（到店）/ ""（未指定）
    # 与门店自身的「类别」分开：类别说的是这家店是什么，用餐方式说的是这顿怎么吃。
    # 混在一起的后果就是「外卖推荐被步行路线校验拦下」——地图搜到的店类别是
    # 「联网检索」，永远不等于「外卖」，于是统统按堂食要求步行路线，全军覆没。
    _dining_mode: str = ""
    # 用户是否明确要「附近」：决定了超出 3 公里的店是直接剔除，还是保留并标注
    _nearby_strict: bool = False
    # 本轮诊断信息（只用于日志定位，不含任何密钥；对外不下发）
    _turn_log: Dict[str, Any] = {}

    def set_origin(self, origin_key: str) -> None:
        """切换出发点：路线事实按起点分别记录，切换后旧路线自动作废。"""
        self.origin_key = origin_key
        self.origin = config.origin_by_key(origin_key)
        set_origin(self.origin)

    @staticmethod
    def _detect_modes(text: str) -> List[str]:
        modes = ["walking"]
        for mode, words in MODE_KEYWORDS.items():
            if any(w in text for w in words):
                modes.append(mode)
        return modes

    # ---------- 对外主入口 ----------
    def run(self, user_input: str) -> str:
        self.last_cards = None
        self.last_summary = ""
        self.last_dropped = []
        self._format_attempted = False
        self._used_food_search = False
        self._dining_mode = ""
        self._turn_log = {}
        # 「附近/周边/走过去」这类明确的空间约束 → 超过 3 公里的店不推荐；
        # 只说「想吃日料」没说附近 → 附近没有时可以给远处同类店，但要标注距离
        self._nearby_strict = any(
            w in user_input for w in ("附近", "周边", "旁边", "走路", "步行", "走过去", "楼下")
        )
        self._needed_modes = self._detect_modes(user_input)
        set_origin(self.origin)
        # 每轮刷新 system prompt（画像可能上一轮刚被工具更新过）
        self.memory.messages[0] = {"role": "system", "content": self._system_prompt()}
        self.memory.add_user(user_input)
        self.memory.trim()

        if self.llm.enabled:
            try:
                reply = self._react_loop(user_input)
                # 先清掉模型误吐的工具参数/内部日志，再进兜底：
                # 清理后成空也能被 _ensure_reply 救回来
                reply = _strip_tool_echo(reply)
                reply = self._ensure_reply(reply, user_input)
                reply = self._filter_cards_by_walk_limit(reply, user_input)
                reply = self._guard_text_walk_limit(reply, user_input)
                out = self._guard_walking_constraint(reply, user_input)
                self._log_turn(out)
                return out
            except LLMBusy as busy:
                # 排队超时 ≠ 服务坏了：给用户明确预期，绝不能降级到离线模式
                wait = getattr(busy, "retry_after", 10) or 10
                self._trace_fn(f"模型通道排队超时（建议 {wait} 秒后重试），返回稍后再试提示")
                self._turn_log["end_reason"] = "llm_busy"
                if wait >= 60:
                    text = f"模型接口刚被上游限流，正在冷却，约 {max(1, round(wait / 60))} 分钟后自动恢复，不用重开页面。"
                else:
                    text = f"这一步模型响应慢了一点，通道正忙，请等 {wait} 秒左右再发一次，你的问题我还记着呢。"
                self._log_turn(text)
                return text + "（这不是你的操作问题，也不是服务挂了。）"
            except RuntimeError as exc:
                self._trace_fn(f"LLM 异常，切换离线模式: {exc}")
                self._turn_log["end_reason"] = "llm_error"
        out = self._offline_fallback(user_input)
        self._log_turn(out, offline=True)
        return out

    def _log_turn(self, reply: str, offline: bool = False) -> None:
        """一轮结束的诊断日志：够定位问题，又不泄露任何凭据。

        记录：用餐方式、工具调用次数、卡片数、剔除原因、结束原因、回复长度。
        用户反馈「空回答」「卡片全没了」时，看这一行就知道卡在哪一环。
        """
        try:
            text = (reply or "").strip()
            if not text:
                reason = "empty_reply"
            elif offline:
                reason = self._turn_log.get("end_reason") or "offline_fallback"
            elif self.last_cards:
                reason = "cards"
            else:
                reason = self._turn_log.get("end_reason") or "text_only"
            drops = [str(d.get("原因") or "")[:24] for d in (self.last_dropped or [])][:3]
            print(
                "  [诊断] 用餐方式=%s 工具=%d 卡片=%d 剔除=%d %s%s"
                % (
                    self._dining_mode or "未指定",
                    self._turn_log.get("tool_calls", 0),
                    len(self.last_cards or []),
                    len(self.last_dropped or []),
                    f"原因={reason}",
                    f" 剔除样例={'/'.join(drops)}" if drops else "",
                )
            )
        except Exception:  # 日志绝不能反过来影响主流程
            pass

    # ---------- 最终结果检查：绝不返回空白气泡 ----------
    def _ensure_reply(self, reply: str, user_input: str) -> str:
        """模型可能返回空字符串：工具跑了、卡也剔了，就是不说话，前端就是个空气泡。

        这里做次数受限的补救，顺序是「能不麻烦用户就不麻烦」：
          1. 本轮已检索到候选 → 直接用后端事实生成卡片，附一句说明；
          2. 有卡片但没文字 → 补一句整体建议；
          3. 什么都没有 → 走规则检索，给出知识库真实结果；
          4. 连规则检索都没结果 → 明确说明原因（工具错误/剔除原因），不糊弄。
        """
        text = (reply or "").strip()
        if text:
            # 有文字但只有句号/表情这类无效内容时也走补救
            if len(text) >= 2:
                return reply
        else:
            self._trace_fn("模型返回空回答，进入兜底流程")

        # ① 已有卡片、只是没写文字：补一句，把控制权交回卡片
        if self.last_cards:
            names = "、".join(str(c.get("name")) for c in self.last_cards[:3])
            self._trace_fn("空回答但有卡片，补写整体建议")
            return f"按你的条件挑出这几家（{names}），门店与路线数据都来自本轮真实检索，具体见下方卡片。"

        # ② 本轮检索过候选但没出卡片：用后端事实直接补卡片
        if self.poi_store:
            recs: List[Dict[str, Any]] = []
            for uid, poi in self.poi_store.items():
                if poi.get("是餐饮") is False:
                    continue
                # 用户明确要「附近」时才跳过超范围店；只是想吃某品类时，
                # 远处的同类店也要给出来（卡片会标距离），否则永远只有一句「没有」
                if self._nearby_strict and poi.get("在范围内") is False:
                    continue
                recs.append({
                    "uid": uid,
                    "cuisine": str(poi.get("菜系") or ""),
                    "reason": "",  # 模型没给理由就不编，卡片只展示事实
                })
                if len(recs) >= 3:
                    break
            if recs:
                cards = self._inject_card_facts(recs)
                if cards:
                    self.last_cards = cards
                    self.last_summary = "（模型未给出文字说明，以下为已核实门店）"
                    self._trace_fn(f"空回答兜底：用已检索候选补出 {len(cards)} 张卡片")
                    return "这是按你刚才的条件查到的门店，信息都来自本轮真实检索，具体见下方卡片。"

        # ③ 规则检索兜底
        fallback = self._offline_fallback(user_input, label="【兜底 · 知识库真实数据】")
        if (fallback or "").strip():
            self._trace_fn("空回答兜底：改用规则检索")
            return fallback

        # ④ 实在没有：如实说明，不留空白
        reasons: List[str] = []
        for d in self.last_dropped[:3]:
            reasons.append(f"{d.get('名称') or d.get('uid')}（{d.get('原因')}）")
        detail = ("；".join(reasons)) if reasons else "本轮没有检索到可用门店"
        self._trace_fn(f"空回答且无可兜底内容，如实说明：{detail}")
        return (
            "这次没能给出推荐：模型没有返回有效内容，"
            f"兜底检索也没有可用结果（{detail}）。\n"
            "可以换个说法再问一次，比如直接说想吃什么品类、预算或能接受的步行时间。"
        )

    def _auto_verify_route(self, uid: str, modes: List[str]) -> Dict[str, Any]:
        """卡片校验阶段自动补查路线（模型漏调 verify_route 时兜住）。

        只对调用方给的 uid 查，次数由 auto_route_budget 控制。
        任何异常都吞掉并返回空路线——补查失败不能反过来把卡片也搞没。
        """
        from tools.route_planner import verify_route

        for mode in modes or ["walking"]:
            try:
                verify_route(uid, poi_store=self.poi_store, mode=mode)
            except Exception as exc:  # noqa: BLE001 —— 补查是增强，失败就降级标注
                self._trace_fn(f"自动补查路线异常（{mode}）: {exc}")
                break
        poi = self.poi_store.get(uid) or {}
        route = dict(poi.get("路线") or {})
        if route.get("起点") and route["起点"] != self.origin_key:
            return {}  # 出发点变了，旧路线不作数
        return route

    # ---------- 卡片事实注入 + 硬校验：模型只做选择和解释 ----------
    def _inject_card_facts(self, recs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """按 uid 从 poi_store 注入事实字段，并做发布前硬校验。

        硬校验（不通过就整条剔除，这些是"能不能推荐"的底线）：
          1. 身份已核实：uid 必须来自本次 search_poi / find_restaurant / 详情接口；
          2. 是餐饮门店（停车场、车站等附属设施一律拒绝）；
          3. 在推荐范围内（附近推荐必须 3 公里内）。

        软校验（不通过仍出卡片，但如实标注）：
          4. 路线是否已核实 —— 后端会先自动补查；补查不到就标「路线待核实」。
             路线属于「值不值得跑一趟」的增强信息，不是推荐资格。
             早期把它当硬门槛，模型漏调 verify_route 时整批卡片消失，
             用户只看到「搜到 8 家、一张卡都没有」。

        字段策略：名称/地址/行政区/坐标/耗时/距离/路线链接由后端注入；
        人均/评分/营业时间/外卖能力没有后端来源时一律置空（前端显示"待确认"），
        模型填的人均只能标为"预估"。
        """
        cards: List[Dict[str, Any]] = []
        seen_uids: set[str] = set()
        # 自动补查路线的预算：只给排在前面的候选查（卡片本来就只展示前几家），
        # 避免一次检索把地图路线额度烧穿。
        auto_route_budget = 3
        for rec in recs:
            uid = str(rec.get("uid") or "").strip()
            if uid and uid in seen_uids:
                continue  # 同一家店被重复推荐（模型常见的复制粘贴），只保留第一次
            seen_uids.add(uid)
            route_pending = False
            poi = self.poi_store.get(uid)
            if poi is None:
                self.last_dropped.append({"uid": uid or "?", "原因": "uid 未核实（不在本次检索结果中）"})
                continue
            if poi.get("是餐饮") is False:
                self.last_dropped.append({"uid": uid, "名称": poi.get("名称"), "原因": "附属设施，非餐饮门店"})
                continue

            # 超出 3 公里：用户明确要「附近」才剔除；只是想吃某品类时保留，
            # 卡片上标清距离——「附近没有日料，最近的在番禺天河城约 7.5 公里」
            # 比「没有匹配」有用得多。
            out_of_range = poi.get("在范围内") is False
            if out_of_range and self._nearby_strict:
                self.last_dropped.append({"uid": uid, "名称": poi.get("名称"), "原因": "超出校区推荐范围"})
                continue

            # 【两层概念必须分开】
            #   门店类别 = 这家店是什么（外卖专营 / 周边堂食 / 饭堂窗口 / 联网检索）
            #   用餐方式 = 这一顿怎么吃（delivery 点外卖 / dine_in 到店）
            # 以前只认「类别 == 外卖」，而地图搜到的店类别是「联网检索」，
            # 于是「日料外卖」的三家店全被按堂食要求步行路线，一个都出不来。
            store_is_delivery = str(poi.get("类别") or "") == "外卖"
            mode_is_delivery = self._dining_mode == "delivery"
            is_delivery = mode_is_delivery or store_is_delivery
            has_coord = bool(str(poi.get("坐标") or "").strip())

            # 地图结果缺人均/评分时，用知识库同名记录补（高置信才补，标来源）
            if poi.get("来源") != "本地知识库":
                poi = enrich_poi_from_kb(poi)
            route = dict(poi.get("路线") or {})
            if route.get("起点") and route["起点"] != self.origin_key:
                route = {}  # 出发点变了，旧路线作废（不能让用户在旧起点数据上做决定）

            if is_delivery:
                # 外卖不需要步行路线——用户根本不会走过去，查了就是白烧额度。
                # 没有坐标也不再剔除：私域商家常常没有坐标，位置待确认比编一个坐标诚实。
                # 但「能推荐」不等于「确认能送」：配送能力/费用/时长一律标待确认。
                verified = True
            else:
                needed = [m for m in self._needed_modes if m in ("walking", "riding", "driving")]
                label = {"walking": "步行", "riding": "骑行", "driving": "打车"}
                verified = all(route.get(label[m]) for m in needed) if needed else bool(route)
                if not verified and auto_route_budget > 0 and str(poi.get("坐标") or "").strip():
                    # 模型经常漏调 verify_route 就直接调 format_recommendation，
                    # 以前这会让所有候选因「缺步行」被整批剔除——用户看到的是
                    # 「搜到 8 家、一张卡都没有」。路线是增强信息，不该当准入条件。
                    # 这里后端自己补查一次，只给排在前面的候选（省地图额度）。
                    auto_route_budget -= 1
                    route = self._auto_verify_route(uid, needed)
                    verified = all(route.get(label[m]) for m in needed) if needed else bool(route)
                    if verified:
                        self._trace_fn(f"自动补查路线成功: {poi.get('名称')}")
                if not verified:
                    # 补查也没拿到（配额/限流/接口异常/无路线）→ 不再剔除，
                    # 照常出卡片，但如实标「路线待核实」，让用户自己判断要不要去。
                    route_pending = True
                    missing = "、".join(label[m] for m in needed if not route.get(label[m]))
                    self._trace_fn(
                        f"路线待核实（{missing or '所需方式'}）但仍出卡片: {poi.get('名称')}"
                    )

            transport: Dict[str, Any] = {}
            if is_delivery:
                dist = poi.get("直线距离_米")
                # 配送时长只可能是估算：后端从来没拿到过商家的真实出餐+骑手时长
                eta = _estimate_delivery_min(dist, poi.get("平均配送分钟"))
                if eta is not None:
                    transport["delivery_min"] = eta
                    transport["delivery_eta_source"] = "按直线距离预估"
                # 地图搜到店 ≠ 确认能送：这是本次最容易说谎的地方，必须显式标注
                transport["配送能力"] = "待确认" if not store_is_delivery else "知识库标注可外卖"
                transport["配送费"] = "待确认"
                if not has_coord:
                    transport["距离说明"] = "位置待确认（商家未提供坐标）"
                elif isinstance(dist, (int, float)) and dist > 0:
                    # 卡片上的距离和配送时长必须自洽：7 公里外的店不可能 40 分钟送到
                    transport["距离说明"] = (
                        f"距校区约 {dist / 1000:.1f} 公里（直线）"
                        + ("，超出常规配送半径，可能需要自取或配送费较高" if dist > 4000 else "")
                    )
                transport["来源"] = poi.get("来源") or "本地知识库"
                transport["查询时间"] = ""
            else:
                for lbl, key in (("步行", "walk_min"), ("骑行", "ride_min"), ("打车", "taxi_min")):
                    fact = route.get(lbl)
                    if fact:
                        transport[key] = fact.get("分钟")
                transport["来源"] = route.get("来源", "")
                transport["查询时间"] = route.get("查询时间", "")
                if route.get("步行"):
                    transport["步行距离_米"] = route["步行"].get("距离_米")
                if route_pending:
                    # 如实说明：不是「很近」也不是「很远」，是后端没拿到
                    transport["路线状态"] = "待核实（接口未返回可用结果）"
                if not has_coord:
                    # 知识库里很多店只有商圈没有坐标：给出「大概在哪、多远」比
                    # 干巴巴一句「位置待确认」有用，但必须标明是估算。
                    area = str(poi.get("商圈") or "")
                    area_dist = poi.get("商圈距离_米")
                    if area and isinstance(area_dist, (int, float)) and area_dist > 0:
                        transport["位置说明"] = (
                            f"在{area}，距校区约 {area_dist / 1000:.1f} 公里（商圈估算）"
                        )
                    elif area:
                        transport["位置说明"] = f"在{area}，具体位置待确认"
                    else:
                        transport["位置说明"] = "无地图点位，位置待确认"

            # 人均：知识库有数据才算事实，否则模型值只能标"预估"（非数字一律视为无来源）
            kb_price = _as_price(poi.get("人均"))
            model_price = _as_price(rec.get("price_per_person"))
            if kb_price is not None:
                price, price_source = kb_price, "知识库"
            elif model_price is not None:
                price, price_source = model_price, "模型预估"
            else:
                price, price_source = None, "待确认"

            card = {
                "uid": uid,
                "identity_verified": True,          # 身份已核实
                "route_verified": bool(verified) and not is_delivery,  # 外卖没有路线，不冒充已核实
                "route_pending": bool(route_pending),  # 路线没拿到，但卡片照出，标待核实
                "delivery_only": is_delivery,
                # 本次用餐方式与门店类别分开暴露，前端据此决定显示「配送参考」还是「路线」
                "dining_mode": "delivery" if is_delivery else "dine_in",
                "store_kind": poi.get("类别") or "",
                # 超范围不等于不能推荐：泛品类推荐时给远店并标距离，是有用信息
                "out_of_range": bool(out_of_range),
                "area": poi.get("商圈") or "",
                "area_distance_m": poi.get("商圈距离_米"),
                "synthetic_uid": bool(poi.get("合成uid")),  # 无地图点位，只有知识库资料
                "name": poi.get("名称"),             # 后端事实：完整店名（含分店）
                "address": poi.get("地址"),           # 后端事实：完整地址
                "district": poi.get("行政区"),         # 后端事实：行政区
                "straight_distance_m": poi.get("直线距离_米"),
                "transport": transport,              # 后端事实：真实路线 / 配送参考
                # 没有坐标就不给任何链接：既不能导航到一个不存在的位置，也不能伪造地图标注。
                # 私域外卖商家常常只有店名，位置待确认是诚实状态，不是缺陷。
                "coordinate_pending": not has_coord,
                "route_link": (
                    build_place_link(poi.get("名称", ""), str(poi.get("坐标", "")), str(poi.get("地址") or ""))
                    if is_delivery
                    else build_route_link(
                        poi.get("名称", ""), str(poi.get("坐标", "")),
                        mode=self._needed_modes[0] if self._needed_modes else "walking",
                        origin=self.origin,
                    )
                ) if has_coord else None,
                "origin": self.origin["name"],       # 出发点，便于核对
            # 解释性字段（模型职责）
            "cuisine": rec.get("cuisine", ""),
            "reason": _strip_walk_talk(rec.get("reason", "")) if is_delivery else rec.get("reason", ""),
            "warning": _strip_walk_talk(rec.get("warning", "")) if is_delivery else rec.get("warning", ""),
            "worth_it": _strip_walk_talk(rec.get("worth_it", "")) if is_delivery else rec.get("worth_it", ""),
                # 事实/未知分离
                "price_per_person": price,
                "price_source": price_source,
                "price_note": poi.get("人均性质") or (poi.get("测评来源") or ""),
                "rating": poi.get("学生评分"),        # 只有知识库才有评分
                "open_hours": poi.get("营业时间") or "待确认",
                "delivery": rec.get("delivery") if rec.get("delivery") is not None else poi.get("可外卖"),
            }
            cards.append(card)

        if self.last_dropped:
            self._trace_fn(
                "剔除未通过校验的推荐: "
                + "；".join(f"{d.get('名称') or d['uid']}（{d['原因']}）" for d in self.last_dropped)
            )
        return cards

    def _filter_cards_by_walk_limit(self, reply: str, user_input: str) -> str:
        """步行上限过滤：超限卡片降级为「替代方案」，全部超限时如实报告无匹配。

        注意：路线拿不到的卡片既不算达标、也不算超限，是「无法判断」。
        早期把 walk_min 为 None 的当成超限，等于二次惩罚——
        卡片刚因为路线缺失逃过一劫，又被这里标成「替代方案」。
        """
        _q, _t, _b, minutes = self._parse_constraints(user_input)
        if not minutes or not self.last_cards:
            return reply

        within = [c for c in self.last_cards if self._walk_min(c) is not None and self._walk_min(c) <= minutes]
        over = [c for c in self.last_cards if self._walk_min(c) is not None and self._walk_min(c) > minutes]
        unknown = [c for c in self.last_cards if self._walk_min(c) is None]

        if over:
            for c in over:
                c["exceeds_walk_limit"] = True

        unknown_note = ""
        if unknown:
            names = "、".join(str(c.get("name")) for c in unknown[:3])
            unknown_note = (
                f"\n\n另外 {names} 的步行时长暂时没查到（接口没返回），"
                f"无法确认是否在 {minutes} 分钟内，去之前建议点卡片里的路线链接看一眼。"
            )

        if not within and over:
            # 没有一家满足：如实说明，不强凑 Top3
            self.last_cards = over + unknown  # 保留卡片作为"替代方案"展示
            self.last_summary = f"没有步行 {minutes} 分钟内的完全匹配，以下为距离最近的替代（建议骑行/打车）"
            lines = [
                f"实话实说：按你“步行不超过 {minutes} 分钟”的条件，{self.origin['name']} 附近没有完全匹配的店。",
                "以下几家的真实路线都超过上限，仅作为骑行/打车的替代参考：",
            ]
            for i, c in enumerate(over[:3], 1):
                t = c.get("transport") or {}
                price = c.get("price_per_person")
                price_text = f"人均约 {price} 元（预估）" if price is not None else "人均待确认"
                parts = []
                for key, label in (("walk_min", "步行"), ("ride_min", "骑行"), ("taxi_min", "打车")):
                    if t.get(key) is not None:
                        parts.append(f"{label} {t[key]} 分钟")
                lines.append(
                    f"{i}. {c.get('name')}｜{c.get('district', '')}｜{price_text}｜" + " / ".join(parts or ["路线待确认"])
                )
            lines.append(f"如果坚持步行 {minutes} 分钟内，建议点外卖，或放宽到骑行时间，我再帮你筛。")
            return "\n".join(lines) + unknown_note

        if within and over:
            names = "、".join(str(c.get("name")) for c in over)
            self.last_cards = within + over + unknown  # 达标的在前
            return (
                f"先说明：{names} 超过步行 {minutes} 分钟上限，放在后面作为替代；前面的店满足条件。"
                f"\n\n{reply}{unknown_note}"
            )

        if within and unknown:
            self.last_cards = within + unknown
            return f"{reply}{unknown_note}"

        return reply

    def _guard_walking_constraint(self, reply: str, user_input: str) -> str:
        """最后防线：不允许把超时店描述成符合步行上限（防止文案与事实矛盾）。"""
        _q, _t, _b, minutes = self._parse_constraints(user_input)
        if not minutes or not self.last_cards:
            return reply
        bad = [
            c for c in self.last_cards
            if self._walk_min(c) is not None and self._walk_min(c) > minutes
        ]
        if bad and all(c.get("exceeds_walk_limit") for c in bad):
            return reply  # 已由 _filter_cards_by_walk_limit 处理并声明
        if bad:
            names = "、".join(str(c.get("name")) for c in bad)
            return f"（注意：{names} 步行超过 {minutes} 分钟，属于替代方案而非完全匹配。）\n\n{reply}"
        return reply

    def _guard_text_walk_limit(self, reply: str, user_input: str) -> str:
        """纯文字回复（没出卡片）也要守住步行上限。

        模型有时不调 format_recommendation，直接在正文里列店并声称「都在 X 分钟内」，
        绕过了卡片校验。这里用后端已核实的路线事实复核：正文提到的超限门店，
        统一补一句更正说明，避免把替代方案说成完全匹配。
        """
        _q, _t, _b, minutes = self._parse_constraints(user_input)
        if not minutes or not reply or self.last_cards:
            return reply

        over = []
        for poi in self.poi_store.values():
            name = str(poi.get("名称") or "")
            if not name or name not in reply:
                continue
            route = poi.get("路线") or {}
            if route.get("起点") and route["起点"] != self.origin_key:
                continue  # 出发点变了，旧路线不作数
            try:
                walk = int((route.get("步行") or {}).get("分钟"))
            except (TypeError, ValueError):
                continue
            if walk > minutes:
                over.append(f"{name}（步行 {walk} 分钟）")

        if not over:
            return reply
        return (
            f"⚠️ 更正：{'、'.join(over)} 超出你说的 {minutes} 分钟步行上限，"
            "属于替代方案而非完全匹配（下文如称『都在范围内』，以此处为准）。\n\n" + reply
        )

    @staticmethod
    def _walk_min(card: Dict[str, Any]) -> int | None:
        try:
            v = (card.get("transport") or {}).get("walk_min")
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    # ---------- LLM 模式：完整 ReAct 循环 ----------
    def _chat(self):
        """调用 LLM；带两道防线：

        1. 上下文预算（主动）：messages + tools 的 JSON 字符数超过
           LLM_CONTEXT_CHAR_LIMIT 时主动裁剪——先截断过长的工具结果，
           仍超限则压缩到最近 8 条。 glm-4-flash 免费档上下文超 8K token
           后速率被限制为极低档，被动等 400 再压缩为时已晚（请求已经在
           限速档里排队了），所以必须在发请求前主动收缩。
        2. 400 重试（被动）：上下文仍被拒时压缩历史重试一次（历史逻辑保留）。
        """
        messages = self.memory.as_payload()
        if self._payload_over_budget(messages):
            self._trace_fn("上下文超过预算，主动裁剪（防 8K 限速档）")
            messages = self._shrink(messages)
        try:
            return self.llm.chat(messages, tools=self.schemas)
        except RuntimeError as exc:
            if "400" not in str(exc):
                raise
            self._trace_fn("LLM 拒绝了当前上下文（400），压缩历史后重试一次")
            compact = [messages[0]] + ConversationMemory._sanitize(messages[-8:])
            return self.llm.chat(compact, tools=self.schemas)

    @staticmethod
    def _payload_chars(messages, tools=None) -> int:
        try:
            total = len(json.dumps(messages, ensure_ascii=False))
            if tools:
                total += len(json.dumps(tools, ensure_ascii=False))
            return total
        except (TypeError, ValueError):
            return 0

    def _payload_over_budget(self, messages) -> bool:
        return self._payload_chars(messages, self.schemas) > config.LLM_CONTEXT_CHAR_LIMIT

    def _shrink(self, messages):
        """两级裁剪：先【结构化压缩】工具结果 → 仍超限再压缩历史到最近 8 条。

        为什么不能直接截断字符串：工具结果是 JSON，硬切前 N 个字符会把
        `[{"uid": "420cd99c..."` 切成半截，uid 丢失、括号不闭合，
        LLM 拿到一段没法解析的乱码，既选不出店也说不清话——日志里那次
        「汉堡搜到了候选 → 触发裁剪 → 空回答」就是这么来的。
        """
        budget = config.LLM_CONTEXT_CHAR_LIMIT
        shrunk = self._compact_tool_results(messages)
        if self._payload_chars(shrunk, self.schemas) <= budget:
            return shrunk
        compact = [shrunk[0]] + ConversationMemory._sanitize(shrunk[-8:])
        return compact

    # 压缩时无论如何都要保住的字段：没有它们，模型就没法引用门店
    _MUST_KEEP_FIELDS = (
        "uid", "名称", "类别", "菜系", "来源", "坐标", "直线距离_米",
        "地址", "人均", "可外卖", "状态", "提示", "建议", "用餐方式",
    )

    @staticmethod
    def _compress_tool_json(content: str, keep_items: int, keep_fields: tuple | None = None) -> str:
        """把一段 JSON 工具结果结构化压缩，保证输出仍是合法 JSON。

        做法：保留所有控制类字段（状态/提示/建议/用餐方式…），
        列表型候选只留前 keep_items 条；keep_fields 非空时，每条候选只留这些关键字段。
        这样名称和 uid 永远完整，模型仍能引用真实门店。
        """
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError, ValueError):
            return ""
        if not isinstance(data, dict):
            return ""

        out: Dict[str, Any] = {}
        trimmed = 0
        for key, value in data.items():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                kept = []
                for item in value[:keep_items]:
                    if keep_fields:
                        kept.append({f: item.get(f) for f in keep_fields if f in item})
                    else:
                        kept.append(item)
                out[key] = kept
                if len(value) > keep_items:
                    trimmed += len(value) - keep_items
            else:
                out[key] = value
        if trimmed:
            # 明确告诉模型：这是本地压缩，不是上游故障，uid 依然有效
            out["_压缩说明"] = (
                f"结果已本地压缩：候选共减去 {trimmed} 条，保留 {keep_items} 条；"
                "uid 仍然有效，可直接用于 verify_route / format_recommendation。"
                "这是本地上下文预算压缩，不是接口限流或配额错误。"
            )
        return json.dumps(out, ensure_ascii=False)

    def _compact_tool_results(self, messages):
        """按「新旧分级」压缩工具结果：最近一组保内容，更早的只留骨架。"""
        # 找出最后一组工具调用（assistant tool_calls + 其后的 tool 结果）的起点
        last_group_start = 0
        for i, m in enumerate(messages):
            if m.get("role") == "assistant" and m.get("tool_calls"):
                last_group_start = i

        limit = config.TOOL_RESULT_MAX_CHARS
        out: List[Dict[str, Any]] = []
        for i, m in enumerate(messages):
            if m.get("role") != "tool":
                out.append(m)
                continue
            content = m.get("content") or ""
            if len(content) <= limit:
                out.append(m)
                continue

            is_recent = i >= last_group_start
            # 最近一组：保留完整字段，只减少候选数量（模型还要用这些写推荐理由）
            compressed = self._compress_tool_json(content, keep_items=4 if is_recent else 1,
                                                 keep_fields=None if is_recent else self._MUST_KEEP_FIELDS)
            if compressed and len(compressed) < len(content):
                out.append(dict(m, content=compressed))
                continue
            # 兜底：结构化压缩失败（非 JSON 结果）时，保头部并说明是本地裁剪
            out.append(dict(m, content=content[:limit] + "…（本地压缩：结果过长已省略尾部，非上游错误）"))
        return out

    def _react_loop(self, user_input: str = "") -> str:
        tool_calls_used = 0
        nudges = 0
        card_nudges = 0
        for _step in range(config.MAX_STEPS):
            assistant = self._chat()

            if not assistant.wants_tool:
                # 空话纠正：只承诺「稍等/马上查」或反问用户已知条件，都不算完成任务。
                reason = self._nudge_reason(assistant.content, tool_calls_used, user_input)
                if reason and nudges < 2:
                    nudges += 1
                    self._trace_fn(f"模型未执行（{reason}），要求立即检索")
                    self.memory.add_assistant(assistant.content)
                    self.memory.add_user(self._NUDGE_TEXT[reason])
                    self.memory.trim()
                    continue

                # 卡片缺失纠正：调过检索、正文里点名了真实门店，却没出结构化卡片。
                # 前端只有 format_recommendation 截获的卡片才算推荐结果，
                # 纯文字列店用户抓不住重点（名称/耗时等事实也无后端校验）。
                if (
                    not self.last_cards
                    and not self._format_attempted
                    and self._used_food_search
                    and card_nudges < 1
                    and self._mentions_poi_store(assistant.content)
                ):
                    card_nudges += 1
                    self._trace_fn("正文推荐了门店但未生成卡片，要求补 format_recommendation")
                    self.memory.add_assistant(assistant.content)
                    self.memory.add_user(self._CARD_NUDGE_TEXT)
                    self.memory.trim()
                    continue

                self.memory.add_assistant(assistant.content)
                self.memory.trim()
                if reason and user_input:
                    # 兜底：纠正两次仍不干活，就用规则检索给出真实知识库结果，
                    # 宁可少一点"智能"，也不能让用户空手而归。
                    self._trace_fn("模型仍未调用工具，改用规则检索兜底")
                    self._turn_log.update(tool_calls=tool_calls_used, end_reason="nudge_exhausted")
                    return self._offline_fallback(user_input, label="【知识库检索结果】")
                self._turn_log.update(tool_calls=tool_calls_used, end_reason="model_final")
                return assistant.content

            # 1) 把 LLM 的工具调用请求原样回填到对话（协议要求）
            self.memory.add_assistant(assistant.content or "", tool_calls=self._raw_tool_calls(assistant))

            # 2) 逐个执行工具，结果回填（LLM 下一轮就能"观察"到结果）
            for tc in assistant.tool_calls:
                tool_calls_used += 1
                if tool_calls_used > config.MAX_TOOL_CALLS_PER_TURN:
                    # 工具调用预算（防止一次提问把地图额度烧穿）
                    result = json.dumps(
                        {
                            "error": f"本轮工具调用已达上限（{config.MAX_TOOL_CALLS_PER_TURN} 次）",
                            "要求": "基于已有信息作答；证据不足就如实说明暂时无法确认，不要继续调用工具",
                        },
                        ensure_ascii=False,
                    )
                    self._trace_fn("工具调用预算用尽，停止继续调用")
                else:
                    result = self._execute_tool(tc)
                self.memory.add_tool_result(tc.id, result)

            self.memory.trim()
            self._turn_log["tool_calls"] = tool_calls_used

        self._turn_log.update(tool_calls=tool_calls_used, end_reason="max_steps")
        return "（达到最大推理步数，请把需求说得更具体些，比如口味、预算、能接受的步行时间。）"

    # 「我这就去查」「请稍等」这类空承诺（真去查会调工具，不会只说不做）
    _PROMISE_RE = re.compile(
        r"(请稍等|稍等|稍候|马上(就|去|为)?|立刻|立即|这就(去|为)|正在(查|检索|搜)|"
        r"查询中|我(帮你|帮您|去|来)查|让我(查|看)|等我)"
    )

    # 「没查就推荐」判定：用户要的是吃喝推荐 + 回复已经成段
    _REC_RE = re.compile(r"推荐|哪家|有什么|附近|求|想吃|去哪")
    _FOOD_RE = re.compile(r"吃|外卖|饭|餐|食堂|窗口|菜|夜宵|下午茶|粉|面|粥")

    def _is_empty_promise(self, content: str, used_tools: int) -> bool:
        """判断这条回复是不是「只说要查、实际没查」的空话。"""
        if used_tools or self.last_cards:
            return False
        text = (content or "").strip()
        if not text or len(text) > 200:
            return False
        return bool(self._PROMISE_RE.search(text))

    def _nudge_reason(self, content: str, used_tools: int, user_input: str) -> str:
        """判断是否需要逼模型干活，返回原因（空串表示不需要）。

        三类小模型常见"不干活"：
          promise  —— 只回「请稍等/马上查」，实际没调工具；
          clarify  —— 用户已经给了口味/预算/步行时间，它却反问「你想吃什么口味」；
          no_tools —— 一轮工具都没调，却直接写出一整段推荐（店名全是记忆里的品牌，
                      不是知识库或地图返回的事实）。这类回答最长最像样，也最危险。
        """
        if used_tools or self.last_cards:
            return ""
        text = (content or "").strip()
        if not text:
            return ""
        if len(text) <= 220 and self._PROMISE_RE.search(text):
            return "promise"
        _q, taste, budget, minutes = self._parse_constraints(user_input)
        if len(text) <= 220 and (taste or budget is not None or minutes is not None) and ("？" in text or "?" in text):
            return "clarify"
        # 成段的推荐却没有一次检索：正文里的店名不可能有出处
        if len(text) >= 40 and self._REC_RE.search(user_input) and self._FOOD_RE.search(user_input):
            return "no_tools"
        return ""

    _NUDGE_TEXT = {
        "promise": (
            "（系统提醒：你刚才只说了要去查，但没有调用任何工具，用户看不到任何结果。"
            "现在立刻调用合适的工具完成检索，然后直接给出结论；"
            "禁止再回复「请稍等 / 马上查」这类空话，也不要把检索过程当成答案。）"
        ),
        "clarify": (
            "（系统提醒：用户已经给出了口味/预算/步行时间等条件，不要再反问这些。"
            "现在立刻调用 food_search 检索并给出结论；确实缺关键信息时，"
            "最多只问一个问题，同时必须先基于现有条件给出可执行的推荐。）"
        ),
        "no_tools": (
            "（系统提醒：这一轮你还没有调用任何工具，就写出了一整段推荐——里面的店名全部来自你的记忆，"
            "而本项目只允许推荐知识库或地图返回的门店，那些品牌在本校区附近有没有、开不开、送不送外卖都无法确认。"
            "现在立刻调用 food_search 检索（外卖场景 query 用「外卖」），拿到真实门店后再推荐；"
            "检索不到就如实说没有匹配。）"
        ),
    }

    _CARD_NUDGE_TEXT = (
        "（系统提醒：你已经在正文里列出门店，但推荐结果必须通过调用 format_recommendation "
        "生成结构化卡片，纯文字列表不算完成推荐。请立即调用 format_recommendation，"
        "传入这些门店的 uid 和解释性字段（菜系、预估人均、推荐理由、避雷点、值不值得跑一趟）；"
        "名称/地址/耗时/地图链接由后端自动注入，未核实过路线的门店先 verify_route。"
        "出完卡片后，再用一两句话给整体建议，不要重复罗列卡片字段。）"
    )

    def _mentions_poi_store(self, text: str) -> bool:
        """正文里是否点名了本轮检索核实的真实门店（联网结果也会进 poi_store）。

        只提知识库/联网检索里出现过的店名才触发卡片纠正；
        回答「没有匹配」或纯闲聊时不会误伤。
        """
        if not text:
            return False
        return any(
            (name := str(poi.get("名称") or "")) and name in text
            for poi in self.poi_store.values()
        )

    def _execute_tool(self, tc: ToolCall) -> str:
        self._trace_fn(f"调用工具 {tc.name}({tc.arguments})")
        func = self.functions.get(tc.name)
        if func is None:
            return json.dumps({"error": f"未知工具: {tc.name}，可用: {list(self.functions)}"}, ensure_ascii=False)        # food_search 传了 dining_mode 就记下来：卡片校验要用它判断「这顿怎么吃」，
        # 而不是去猜门店类别（地图结果的类别永远是「联网检索」，猜不出外卖场景）。
        if tc.name == "food_search":
            _args = tc.parse_arguments()
            from tools.food_search import normalize_dining_mode

            mode = normalize_dining_mode(str(_args.get("dining_mode") or ""))
            if not mode:
                # 模型没传 dining_mode 时，从 query/taste 里的场景词兜底识别（兼容老行为）
                for raw in (str(_args.get("query") or ""), str(_args.get("taste") or "")):
                    mode = normalize_dining_mode(raw)
                    if mode:
                        break
            if mode:
                self._dining_mode = mode
        # 外卖店没有「走过去」这回事：模型一旦拿到步行/骑行分钟数，就会写进推荐理由，
        # 出现「推荐外卖，步行 139 分钟」这种自相矛盾的结论。直接在入口拦掉，同时省地图额度。
        if tc.name == "verify_route":
            _args = tc.parse_arguments()
            _poi = self.poi_store.get(str(_args.get("uid") or "")) or {}
            _store_is_delivery = str(_poi.get("类别") or "") == "外卖"
            if _store_is_delivery or self._dining_mode == "delivery":
                return json.dumps(
                    {
                        "状态": "skipped",
                        "名称": _poi.get("名称"),
                        "说明": (
                            "这次是点外卖，不需要路线核实，用户不会走过去"
                            if self._dining_mode == "delivery"
                            else "外卖店不需要路线核实，用户不会走过去"
                        ),
                        "可用事实": {
                            "地址": _poi.get("地址"),
                            "直线距离_米": _poi.get("直线距离_米"),
                        },
                        "要求": "推荐外卖时只说口味、价格、配送；距离用「距校区约 X 公里」，"
                                "不要提步行/骑行几分钟，也不要再对这家店调 verify_route。"
                                "配送能力/费用/时长没有依据时写「待确认」。",
                    },
                    ensure_ascii=False,
                )
        try:
            output = func(**tc.parse_arguments())
        except TypeError as exc:
            return json.dumps({"error": f"参数不匹配: {exc}，请检查参数名"}, ensure_ascii=False)
        except Exception as exc:
            # 把错误返回给 LLM，它有机会反思并换一种方式重试
            return json.dumps({"error": f"工具执行失败: {exc}，请调整参数或改用其他工具"}, ensure_ascii=False)
        if tc.name == "food_search":
            self._used_food_search = True
        if tc.name == "format_recommendation":
            # 截获结构化推荐 → 注入后端事实 + 硬校验 → 生成卡片
            args = tc.parse_arguments()
            recs = args.get("recommendations") or []
            if isinstance(recs, list) and recs:
                self._format_attempted = True
                self.last_cards = self._inject_card_facts(recs)
                self.last_summary = str(args.get("summary") or "")
                if not self.last_cards:
                    return json.dumps(
                        {
                            "error": "所有推荐条目都未通过校验（门店身份/餐饮类型/范围/路线核实）",
                            "剔除明细": self.last_dropped,
                            "要求": "必须先 search_poi 或 find_restaurant 拿 uid，再 verify_route 核实路线，然后才能推荐；"
                                    "如果确实没有合规的店，就如实告诉用户没有匹配，不要硬凑。",
                        },
                        ensure_ascii=False,
                    )
                output = (
                    f"已生成 {len(self.last_cards)} 张推荐卡片（后端已校验门店身份、范围与路线）。"
                    + (f"被剔除：{self.last_dropped}" if self.last_dropped else "")
                    + " 请在最终回答中围绕通过的店铺给出整体建议，不要重复罗列卡片字段。"
                )
            elif isinstance(recs, list):
                self._format_attempted = True  # 传了空列表也算尝试过，不再反复催
                self.last_cards = None
        self._trace_fn(f"工具返回: {str(output)[:120]}…")
        return output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)

    @staticmethod
    def _raw_tool_calls(assistant: AssistantMessage) -> List[Dict[str, Any]]:
        return [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.arguments},
            }
            for tc in assistant.tool_calls
        ]

    # ---------- 离线模式：规则编排兜底 ----------
    def _offline_fallback(self, user_input: str, label: str = "【离线 · 知识库真实数据 Top3】") -> str:
        self._trace_fn("离线模式：规则解析约束 → 直接检索知识库")
        query, taste, budget, minutes = self._parse_constraints(user_input)
        mode = self._dining_mode or self._detect_dining_mode(user_input)
        result = self.functions["food_search"](
            query=query, taste=taste, dining_mode=mode, budget_max=budget, max_minutes=minutes
        )
        data = json.loads(result)
        if not data.get("结果"):
            from tools.food_search import _all_places
            total = len(_all_places())
            if total:
                return (
                    f"（按你给的条件，本地知识库（共 {total} 家）里没有匹配的门店。"
                    "可以放宽预算或步行时间，或换个口味再问我一次。）"
                )
            return (
                "（离线模式：本地知识库暂无真实数据。示例数据已隔离、不用于正式推荐；"
                "联网检索与 LLM 推理需要配置 API Key 后使用。）"
            )
        lines = [label]
        for i, item in enumerate(data["结果"][:3], 1):
            name = item.get("名称") or "?"
            if mode == "delivery":
                # 外卖没有步行：只报距离和预估时长，两个都必须带「预估/待确认」字样
                dist = item.get("直线距离_米")
                dist_text = f"距校区约 {dist / 1000:.1f} 公里" if isinstance(dist, (int, float)) and dist > 0 else "位置待确认"
                eta = _estimate_delivery_min(dist, item.get("平均配送分钟"))
                eta_text = f"配送约 {eta} 分钟（预估）" if eta is not None else "配送时长待确认"
                lines.append(
                    f"{i}. {name}｜{item.get('菜系', '')}｜人均{item.get('人均', '待确认')}元｜"
                    f"{dist_text}｜{eta_text}｜配送能力待确认｜{item.get('一句话点评', '')}"
                )
            else:
                walk = (item.get("交通") or {}).get("步行")
                # 联网检索结果没有路线事实：不能写「步行?分钟」，那是残缺信息冒充事实
                walk_text = f"步行{walk}分钟" if walk is not None else "路线待核实"
                price = item.get("人均")
                price_text = f"人均{price}元" if price is not None else "人均待确认"
                rating = item.get("学生评分")
                rating_text = f"｜评分{rating}" if rating is not None else ""
                lines.append(
                    f"{i}. {name}｜{item.get('菜系', '')}｜{price_text}｜"
                    f"{walk_text}{rating_text}｜{item.get('一句话点评', '')}"
                )
        return "\n".join(lines)

    @staticmethod
    def _detect_dining_mode(text: str) -> str:
        """从用户原话识别用餐方式（离线兜底与卡片校验共用同一套判定）。"""
        from tools.food_search import normalize_dining_mode

        raw = text or ""
        if any(w in raw for w in ("外卖", "送到宿舍", "不想出门", "点个", "点份", "配送")):
            return "delivery"
        if any(w in raw for w in ("堂食", "到店", "去店里", "走过去", "店里吃")):
            return "dine_in"
        return normalize_dining_mode("")

    @staticmethod
    def _parse_constraints(text: str):
        """规则解析：没有 LLM 时也能从问句中抽取约束。

        这里最容易出错的是「把什么当成 query」：
          query 必须是菜系/品类（日料、汉堡、粥），绝不能是口味（辣）或场景（外卖）。
        早期版本把「外卖」和口味词都塞进 query，导致离线兜底时品类条件丢失，
        问「有日料的外卖推荐吗」会返回一堆酸菜鱼、面馆。
        """
        import re

        from tools.food_search import detect_cuisine

        taste = ""
        for kw in ("辣", "清淡", "甜", "重口"):
            if kw in text:
                taste = "辣" if kw == "重口" else kw
                break

        budget = None
        m = re.search(r"(?:预算|人均)\s*(\d+)", text)
        if m:
            budget = float(m.group(1))

        minutes = None
        m = re.search(r"(\d+)\s*分钟", text)
        if m:
            minutes = int(m.group(1))

        # 场景词不再伪装成 query：用餐方式由 _detect_dining_mode 单独处理
        if "聚餐" in text or "聚会" in text:
            query = "聚餐"
        else:
            # 认不出菜系就留空，让检索按其他条件走；绝不拿口味词顶替品类
            query = detect_cuisine(text)
        return query, taste, budget, minutes

    def _weather_context(self) -> str:
        """实时天气 → system prompt 段落（天气感知的核心注入点）。

        天气是增强项不是依赖项：WEATHER_ENABLED 为 False、接口异常、
        任何状态非 ok，都不影响推荐主流程。
        但此时不能什么都不说——模型会顺口编「今天晴朗、适合出门」这类没有
        依据的话（实测过）。所以改为注入一段"数据不可用"的约束，让它如实说查不到。
        缓存 30 分钟，每轮刷新 system prompt 不会带来额外请求。
        """
        if not getattr(config, "WEATHER_ENABLED", False):
            return WEATHER_UNAVAILABLE_HINT
        try:
            data = weather_gateway.fetch_weather()
        except Exception as exc:  # 天气模块任何意外都不能影响推荐主流程
            self._trace_fn(f"天气模块异常（已忽略）: {exc}")
            return WEATHER_UNAVAILABLE_HINT
        if data.get("状态") != weather_gateway.OK:
            self._trace_fn(f"天气不可用（{data.get('状态')}），本轮不注入天气")
            return WEATHER_UNAVAILABLE_HINT

        now = data.get("现在") or {}
        sugg = data.get("建议") or {}
        lines = [
            f"\n【当前天气（实时，{data.get('查询时间', '')} 查询，全站缓存 30 分钟）】"
            f"{data.get('位置', '')} {now.get('现象', '')} {now.get('气温', '')}°C",
            f"天气策略（后端确定性规则，直接遵循，不要自行从天气推断）：{sugg.get('提示文案', '')}",
        ]
        if sugg.get("外卖优先"):
            lines.append(
                "雨天执行规则：支持外卖的门店排前面；堂食店只把步行 ≤ 10 分钟的作为首选，"
                "更远的降为替代方案并注明雨天因素；回答里提一句下雨的推荐理由。"
            )
        elif sugg.get("场景") == "hot":
            lines.append("高温执行规则：优先步行 ≤ 15 分钟的近店，主动提醒防暑，远店建议打车。")
        else:
            lines.append("天气适宜出行，按用户需求正常推荐即可。")
        return "\n".join(lines) + "\n\n"

    # ---------- System Prompt：人设 + 意图分流 + 工具策略 ----------
    def _system_prompt(self) -> str:
        return (
            "你是「国际校区干饭军师」，华南理工大学国际校区学生的美食决策助手。\n"
            "校区痛点：饭堂选择少、位置偏、交通不便，学生不知道吃什么、去哪吃、值不值得跑一趟。\n\n"
            "【第一步：先判断意图，不要所有问题都走『搜品类→推三家→算路线』】\n"
            "A. 指定门店（用户说出具体店名：『你知道B记鱼粥吗』『XX店在哪』）\n"
            "   → 先调 find_restaurant(店名) 确认这家店的身份；\n"
            "   → 找到：只讲这家店（地址、是否超出校区范围、值不值得去），要路线再 verify_route(uid, mode)；\n"
            "   → 没找到明确匹配：如实说没查到，并追问校门/地标/地址。**绝不换成别的店，也不改推荐其他店**；\n"
            "   → 只查到『XX-门前停车场』这类附属地点：说明这是停车场不是餐馆本体，请用户确认具体位置；\n"
            "   → 返回 kb_no_coord（资料里有、但无核实坐标）：只讲资料（人均/口碑/类别），"
            "明确标注来源与『位置待核实』，**不要给步行时间/距离/导航，也不要出推荐卡片**，"
            "更不能因为地图只搜到停车场就说这家店不存在。\n"
            "B. 附近推荐（『附近有什么好吃的粥』『推荐个粤菜』）\n"
            "   → food_search 先查本地库 → 无匹配自动联网（3 公里硬限制，无需再调 search_poi）→ 需要时 verify_route 核实\n"
            "   → format_recommendation 出卡片 → 给整体建议。\n"
            "   → **food_search 的三个参数各管一件事，绝对不能混用**：\n"
            "     query       = 吃什么品类（汉堡 / 日料 / 粥 / 粤菜），只说菜系；\n"
            "     taste       = 什么口味（辣 / 清淡 / 甜），只说口味；\n"
            "     dining_mode = 怎么吃（delivery 点外卖 / dine_in 到店）。\n"
            "     把「外卖」写进 taste 或 query 都是错的——它是场景，必须放 dining_mode。\n"
            "     例：『有没有日料外卖』→ query=日料, dining_mode=delivery；\n"
            "     『想吃点辣的』→ taste=辣, dining_mode=dine_in。\n"
            "   → **菜系条件不能擅自放宽**：用户要汉堡，附近没有汉堡就如实说没有，\n"
            "     绝不能换成酸菜鱼之类无关品类凑数。工具返回「需询问用户」时，\n"
            "     必须在回答里说明放宽了什么、没放宽什么，并问用户是否接受。\n"
            "C. 路线查询（『怎么去X』）→ find_restaurant 确认身份 → verify_route(uid, mode)。\n"
            "D. 门店比较（『这两家哪家近』）→ 各自 find_restaurant + verify_route，逐项对比，不要合成泛推荐。\n\n"
            "【地图额度纪律（很重要）】\n"
            "1. 只查用户需要的交通方式：默认 walking；用户说骑行/打车才查对应方式；"
            "不要对一家店把步行/骑行/打车全查一遍（每多查一种就多消耗一份额度）。\n"
            "2. 先用 search_poi 筛出候选，再对少量候选 verify_route；用户只是问『知不知道这家店』时不用算路线。\n"
            "3. 同一家店不要重复 verify_route；工具已返回路线事实就直接复用。\n\n"
            "【错误 ≠ 没有门店】\n"
            "- 工具返回『限流 / 配额用尽 / 权限异常 / 超时』是接口故障，不是附近没有店："
            "如实告诉用户『暂时查不了，稍后再试』，禁止改搜别的品类，禁止编造门店和耗时。\n"
            "- 只有工具明确说『查询成功但没有匹配』时，才能说附近没有合适的店。\n\n"
            "【诚实铁律】\n"
            "- **任何具体门店名都必须来自本轮工具返回**：未调用 food_search / find_restaurant / search_poi "
            "检索之前，禁止在回答里列出任何店名（包括你记忆里的常见品牌）；检索结果里没有的店，一个字都不要提。\n"
            "- 门店事实（名称/地址/行政区/耗时/距离/是否在范围内）一律以工具返回为准，禁止凭记忆或想象编写。\n"
            "- 3 公里内没有合适的店就如实说没有匹配，禁止推荐范围外的远店，禁止强凑 Top3。\n"
            "- 用户设了步行/预算上限时，超限的店只能作为「替代方案」并明确标注，不能说成符合条件。\n"
            "- **同名 vs 同类，规则完全不同**：\n"
            "  · 用户点名某家店（「紫藏在哪」「有没有一家叫达海煲仔饭的」）→ 只能用这家，"
            "**绝不能用同品类的其他店顶替**；查不到就如实说查不到，可以补一句「同类型的还有 A/B」，"
            "但必须讲清那是另一家店，由用户决定要不要看。\n"
            "  · 用户只说想吃什么品类（「想吃日料」「有没有汉堡」）→ **可以推荐同类店，名字不同没问题**，"
            "这是正常推荐不是顶替。但必须说明：这是同品类推荐、在哪（商圈/距离）、要不要打车。\n"
            "- 卡片上标了「超出附近范围」的店，回答里要明确说「不在步行范围，建议打车/骑车过去」，"
            "不能含糊说成「附近有一家」。\n"
            "- 人均没有依据就写「待确认」，你的估算必须写「预估」；评分、营业时间、能否外卖无来源就写待确认。\n"
            "- 用户提到口味/忌口/去过的店时，调用 remember_preference 记入画像。\n"
            "- **不要回复「请稍等 / 我马上去查 / 正在检索」**：工具调用在用户界面是不可见的，"
            "只说要查却什么都不做，用户会一直等不到结果。要么当场调工具查完再答，要么直接回答。\n"
            "- **工具返回里的「要求 / 建议 / 提示 / 配送提示」是给你的内部约束，不是给用户看的话**："
            "照它去做，但绝对不要原样念出来。工具说「不要编造」，你只需做到不编造，"
            "不要在回答里写「我不会编造」；工具说「标待确认」，你只需在正文或卡片里写「待确认」。\n"
            "- 语气亲切接地气，像个懂吃的学长。\n\n"
            "【推荐卡片】\n"
            "确定推荐后调用 format_recommendation，每家只传 uid + 解释性字段（菜系、预估人均、理由、避雷、值不值）；"
            "名称/地址/耗时/距离/地图链接由后端注入。后端会校验身份、餐饮类型、范围和路线，"
            "不通过的会被剔除——没有合规的店就不要硬凑。\n"
            "**点外卖时（dining_mode=delivery）任何店都不需要 verify_route**："
            "外卖不是走过去的，卡片会显示配送参考与地图位置；到店吃（dine_in）才必须先 verify_route。\n"
            "**回答正文里绝对不要出现 JSON、uid 字段名或工具参数**："
            "那些是给程序看的，用户只看到一堆花括号会很困惑。要用平常话描述结果。\n"
            "**但「地图搜到这家店」不等于「确认它能送到校区」**：配送能力、配送费、配送时长"
            "没有依据时一律写「待确认」，配送时长只能说「预估 X 分钟」。\n"
            "**外卖店的正文、reason、warning 里只写口味、价格和配送，禁止写「步行几分钟/距离几公里」**——"
            "外卖不是走过去的，后端也从未为外卖店算过步行路线，写了就是编造；"
            "要提醒远近就说「距校区约 X 公里，配送可能偏慢」。"
            "私域商家可能没有坐标，此时位置写「待确认」，不要编造地址或坐标。\n\n"
            f"【当前出发点】{self.origin['name']}（坐标来自地图实测，检索与路线均以此为起点）\n"
            "默认就用这个出发点，**不要反问用户在哪个位置**；用户主动说『从南门/北门走』时，"
            "提示他可以在页面顶部切换出发点。系统没有手机实时定位，不要声称知道用户的当前位置。\n"
            "用户给了口味/预算/时间就走流程检索，缺的项自己按常识取舍，不要为了一个次要条件反复追问。\n"
            f"{self._weather_context()}"
            f"【用户画像（长期记忆）】\n{self.profile.as_prompt_text()}"
        )
