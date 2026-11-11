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
from typing import Any, Dict, List

import config
from agent.memory import ConversationMemory, UserProfile
from llm.client import AssistantMessage, LLMClient, ToolCall
from llm.keypool import LLMBusy
from tools import build_registry
from tools.route_planner import build_route_link, current_origin, set_origin
from tools import map_gateway

MODE_KEYWORDS = {
    "riding": ("骑行", "单车", "自行车", "骑车"),
    "driving": ("打车", "开车", "驾车", "滴滴", "网约车"),
}


def _trace(text: str) -> None:
    """推理过程可视化：答辩时评委能看到每一步在干什么。"""
    print(f"  [推理] {text}")


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
        self._needed_modes = self._detect_modes(user_input)
        set_origin(self.origin)
        # 每轮刷新 system prompt（画像可能上一轮刚被工具更新过）
        self.memory.messages[0] = {"role": "system", "content": self._system_prompt()}
        self.memory.add_user(user_input)
        self.memory.trim()

        if self.llm.enabled:
            try:
                reply = self._react_loop()
                reply = self._filter_cards_by_walk_limit(reply, user_input)
                return self._guard_walking_constraint(reply, user_input)
            except LLMBusy:
                # 排队超时 ≠ 服务坏了：给用户明确预期，绝不能降级到离线模式
                self._trace_fn("模型通道排队超时，返回稍后再试提示")
                return (
                    "现在提问的同学有点多，模型通道都占满了。请等 10 秒左右再发一次，"
                    "你的问题我还记着呢。"
                )
            except RuntimeError as exc:
                self._trace_fn(f"LLM 异常，切换离线模式: {exc}")
        return self._offline_fallback(user_input)

    # ---------- 卡片事实注入 + 硬校验：模型只做选择和解释 ----------
    def _inject_card_facts(self, recs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """按 uid 从 poi_store 注入事实字段，并做发布前硬校验。

        校验项（任一不通过则整条剔除，绕不过去）：
          1. 身份已核实：uid 必须来自本次 search_poi / find_restaurant / 详情接口；
          2. 是餐饮门店（停车场、车站等附属设施一律拒绝）；
          3. 在推荐范围内（附近推荐必须 3 公里内）；
          4. 用户需要的交通方式路线已核实成功（且起点与当前出发点一致）。

        字段策略：名称/地址/行政区/坐标/耗时/距离/路线链接由后端注入；
        人均/评分/营业时间/外卖能力没有后端来源时一律置空（前端显示"待确认"），
        模型填的人均只能标为"预估"。
        """
        cards: List[Dict[str, Any]] = []
        for rec in recs:
            uid = str(rec.get("uid") or "").strip()
            poi = self.poi_store.get(uid)
            if poi is None:
                self.last_dropped.append({"uid": uid or "?", "原因": "uid 未核实（不在本次检索结果中）"})
                continue
            if poi.get("是餐饮") is False:
                self.last_dropped.append({"uid": uid, "名称": poi.get("名称"), "原因": "附属设施，非餐饮门店"})
                continue
            if poi.get("在范围内") is False:
                self.last_dropped.append({"uid": uid, "名称": poi.get("名称"), "原因": "超出校区推荐范围"})
                continue

            route = dict(poi.get("路线") or {})
            if route.get("起点") and route["起点"] != self.origin_key:
                route = {}  # 出发点变了，旧路线作废（不能让用户在旧起点数据上做决定）
            needed = [m for m in self._needed_modes if m in ("walking", "riding", "driving")]
            label = {"walking": "步行", "riding": "骑行", "driving": "打车"}
            verified = all(route.get(label[m]) for m in needed) if needed else bool(route)
            if not verified:
                missing = "、".join(label[m] for m in needed if not route.get(label[m]))
                self.last_dropped.append(
                    {"uid": uid, "名称": poi.get("名称"), "原因": f"路线未核实（缺 {missing or '所需方式'}）"}
                )
                continue

            transport: Dict[str, Any] = {}
            for lbl, key in (("步行", "walk_min"), ("骑行", "ride_min"), ("打车", "taxi_min")):
                fact = route.get(lbl)
                if fact:
                    transport[key] = fact.get("分钟")
            transport["来源"] = route.get("来源", "")
            transport["查询时间"] = route.get("查询时间", "")
            if route.get("步行"):
                transport["步行距离_米"] = route["步行"].get("距离_米")

            # 人均：知识库有数据才算事实，否则模型值只能标"预估"
            kb_price = poi.get("人均")
            model_price = rec.get("price_per_person")
            if kb_price is not None:
                price, price_source = kb_price, "知识库"
            elif model_price is not None:
                price, price_source = model_price, "模型预估"
            else:
                price, price_source = None, "待确认"

            card = {
                "uid": uid,
                "identity_verified": True,          # 身份已核实
                "route_verified": True,             # 路线已核实（本次所需方式）
                "name": poi.get("名称"),             # 后端事实：完整店名（含分店）
                "address": poi.get("地址"),           # 后端事实：完整地址
                "district": poi.get("行政区"),         # 后端事实：行政区
                "straight_distance_m": poi.get("直线距离_米"),
                "transport": transport,              # 后端事实：真实路线
                "route_link": build_route_link(
                    poi.get("名称", ""), str(poi.get("坐标", "")),
                    mode=self._needed_modes[0] if self._needed_modes else "walking",
                    origin=self.origin,
                ) if poi.get("坐标") else None,
                "origin": self.origin["name"],       # 出发点，便于核对
                # 解释性字段（模型职责）
                "cuisine": rec.get("cuisine", ""),
                "reason": rec.get("reason", ""),
                "warning": rec.get("warning", ""),
                "worth_it": rec.get("worth_it", ""),
                # 事实/未知分离
                "price_per_person": price,
                "price_source": price_source,
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
        """步行上限过滤：超限卡片降级为「替代方案」，全部超限时如实报告无匹配。"""
        _q, _t, _b, minutes = self._parse_constraints(user_input)
        if not minutes or not self.last_cards:
            return reply

        within = [c for c in self.last_cards if self._walk_min(c) is not None and self._walk_min(c) <= minutes]
        over = [c for c in self.last_cards if self._walk_min(c) is None or self._walk_min(c) > minutes]

        if over:
            for c in over:
                c["exceeds_walk_limit"] = True

        if not within and over:
            # 没有一家满足：如实说明，不强凑 Top3
            self.last_cards = over  # 保留卡片作为"替代方案"展示
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
            return "\n".join(lines)

        if within and over:
            names = "、".join(str(c.get("name")) for c in over)
            self.last_cards = within + over  # 达标的在前
            return f"先说明：{names} 超过步行 {minutes} 分钟上限，放在后面作为替代；前面的店满足条件。\n\n{reply}"

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
        """两级裁剪：先截断工具结果 → 仍超限再压缩历史到最近 8 条。"""
        budget = config.LLM_CONTEXT_CHAR_LIMIT
        shrunk = self._truncate_tool_results(messages)
        if self._payload_chars(shrunk, self.schemas) <= budget:
            return shrunk
        compact = [shrunk[0]] + ConversationMemory._sanitize(shrunk[-8:])
        return compact

    @staticmethod
    def _truncate_tool_results(messages):
        limit = config.TOOL_RESULT_MAX_CHARS
        out = []
        for m in messages:
            if m.get("role") == "tool" and len(m.get("content") or "") > limit:
                m = dict(m, content=m["content"][:limit] + "…（工具结果过长已截断；结论性字段已在前文）")
            out.append(m)
        return out

    def _react_loop(self) -> str:
        tool_calls_used = 0
        for _step in range(config.MAX_STEPS):
            assistant = self._chat()

            if not assistant.wants_tool:
                self.memory.add_assistant(assistant.content)
                self.memory.trim()
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

        return "（达到最大推理步数，请把需求说得更具体些，比如口味、预算、能接受的步行时间。）"

    def _execute_tool(self, tc: ToolCall) -> str:
        self._trace_fn(f"调用工具 {tc.name}({tc.arguments})")
        func = self.functions.get(tc.name)
        if func is None:
            return json.dumps({"error": f"未知工具: {tc.name}，可用: {list(self.functions)}"}, ensure_ascii=False)
        try:
            output = func(**tc.parse_arguments())
        except TypeError as exc:
            return json.dumps({"error": f"参数不匹配: {exc}，请检查参数名"}, ensure_ascii=False)
        except Exception as exc:
            # 把错误返回给 LLM，它有机会反思并换一种方式重试
            return json.dumps({"error": f"工具执行失败: {exc}，请调整参数或改用其他工具"}, ensure_ascii=False)
        if tc.name == "format_recommendation":
            # 截获结构化推荐 → 注入后端事实 + 硬校验 → 生成卡片
            args = tc.parse_arguments()
            recs = args.get("recommendations") or []
            if isinstance(recs, list) and recs:
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
    def _offline_fallback(self, user_input: str) -> str:
        self._trace_fn("离线模式：规则解析约束 → 直接检索知识库")
        query, taste, budget, minutes = self._parse_constraints(user_input)
        result = self.functions["food_search"](query=query, taste=taste, budget_max=budget, max_minutes=minutes)
        data = json.loads(result)
        if not data.get("结果"):
            return (
                "（离线模式：本地知识库暂无真实数据。示例数据已隔离、不用于正式推荐；"
                "联网检索与 LLM 推理需要配置 API Key 后使用。）"
            )
        lines = ["【离线 · 知识库真实数据 Top3】"]
        for i, item in enumerate(data["结果"][:3], 1):
            name = item.get("名称") or "?"
            walk = (item.get("交通") or {}).get("步行", "?")
            lines.append(
                f"{i}. {name}｜{item.get('菜系', '')}｜人均{item.get('人均', '?')}元｜"
                f"步行{walk}分钟｜评分{item.get('学生评分', '?')}｜{item.get('一句话点评', '')}"
            )
        return "\n".join(lines)

    @staticmethod
    def _parse_constraints(text: str):
        """规则解析：没有 LLM 时也能从问句中抽取约束。"""
        import re

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

        if "外卖" in text:
            query = "外卖"
        elif "聚餐" in text or "聚会" in text:
            query = "聚餐"
        elif taste:
            query = taste
        else:
            query = ""
        return query, taste, budget, minutes

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
            "   → 只查到『XX-门前停车场』这类附属地点：说明这是停车场不是餐馆本体，请用户确认具体位置。\n"
            "B. 附近推荐（『附近有什么好吃的粥』『推荐个粤菜』）\n"
            "   → food_search 查本地库 → 空则 search_poi 联网（3 公里硬限制）→ verify_route 核实\n"
            "   → format_recommendation 出卡片 → 给整体建议。\n"
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
            "- 门店事实（名称/地址/行政区/耗时/距离/是否在范围内）一律以工具返回为准，禁止凭记忆或想象编写。\n"
            "- 3 公里内没有合适的店就如实说没有匹配，禁止推荐范围外的远店，禁止强凑 Top3。\n"
            "- 用户设了步行/预算上限时，超限的店只能作为「替代方案」并明确标注，不能说成符合条件。\n"
            "- 人均没有依据就写「待确认」，你的估算必须写「预估」；评分、营业时间、能否外卖无来源就写待确认。\n"
            "- 用户提到口味/忌口/去过的店时，调用 remember_preference 记入画像。\n"
            "- 语气亲切接地气，像个懂吃的学长。\n\n"
            "【推荐卡片】\n"
            "确定推荐后调用 format_recommendation，每家只传 uid + 解释性字段（菜系、预估人均、理由、避雷、值不值）；"
            "名称/地址/耗时/距离/地图链接由后端注入。后端会校验身份、餐饮类型、范围和路线，"
            "不通过的会被剔除——没有合规的店就不要硬凑。\n\n"
            f"【当前出发点】{self.origin['name']}（坐标来自地图实测，检索与路线均以此为起点）\n"
            "如果用户说『从南门/北门走』，提示他可以在页面顶部切换出发点；"
            "系统没有手机实时定位，不要声称知道用户的当前位置。\n\n"
            f"【用户画像（长期记忆）】\n{self.profile.as_prompt_text()}"
        )
