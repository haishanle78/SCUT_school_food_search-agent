# 国际校区干饭军师 · 智能体工程

> 华工广州国际校区美食指引 Agent：解决「今天吃什么 / 去哪吃 / 值不值得跑一趟」。
> 纯 Python 标准库实现，零第三方依赖，Python 3.10+ 即可运行。

线上演示：`https://campus-food-advisor.app.workbuddy.host/?sharecode=NDHJM5B3NI`
（部署在 WorkBuddy 云端容器，与本机无关）

---

## 1. 目录结构

```
SDK/
├── main.py                  # CLI 入口：交互对话 / --demo 演示
├── config.py                # 配置：.env → runtime.env（可关）→ 环境变量
├── verify_deploy.py         # 核对线上代码与本地源码是否一致（build 指纹）
├── .env.example             # 配置模板（复制为 .env 后填 Key）
├── agent/
│   ├── core.py              # ReAct 主循环 + 意图分流 + 卡片硬校验 + 工具预算
│   └── memory.py            # 三层记忆 + 消息协议自检
├── llm/
│   └── client.py            # OpenAI 兼容 HTTP 客户端（智谱 / DeepSeek / OpenAI）
├── tools/
│   ├── map_gateway.py       # ★ 地图统一网关：限速/缓存/去重/错误分类/重试/统计
│   ├── route_planner.py     # 门店检索、指定门店查询、路线核实、官方路线链接
│   ├── food_search.py       # 本地知识库检索（真实库为空时明确提示转联网）
│   ├── budget_nutrition.py  # 预算核算 + 荤素搭配
│   └── __init__.py          # 9 个工具注册 + 参数容错适配层
├── web/
│   ├── server.py            # 单端口服务：限流、消息长度、每日上限、地图统计
│   └── static/index.html    # 聊天页：出发点选择 + 推荐卡片 + 推理过程
├── data/
│   ├── restaurants.json     # 真实餐厅库（空容器，等问卷数据）
│   ├── canteen.json / delivery.json
│   ├── examples/            # 示例数据，仅作字段模板，正式推荐不读取
│   └── map_usage.json       # 地图调用计数（运行时生成，勿入库）
├── tests/
│   ├── test_protocol.py     # 消息协议压力测试
│   ├── test_poi_consistency.py  # POI 一致性 10 项
│   └── test_map_gateway.py  # 地图治理+门店真实性 13 项（对应验收标准）
└── data_collect/问卷模板.md
```

---

## 2. 快速开始

```bash
python main.py --demo      # 离线演示，无需任何 Key
python web/server.py       # 网页版 → http://localhost:8787
python main.py             # 配好 .env 后的交互模式（/profile 看画像，/quit 退出）
```

要开大模型：复制 `.env.example` 为 `.env` 填 Key（见第 3 节）。

---

## 3. 配置项

```ini
# 模型：二选一
LLM_PROVIDER=openai
OPENAI_API_KEY=你的智谱Key
LLM_MODEL=glm-4-flash
LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4/chat/completions
# 或 DeepSeek（付费）
# LLM_PROVIDER=deepseek
# DEEPSEEK_API_KEY=...

# 地图
BAIDU_MAP_AK=你的AK
CAMPUS_LOCATION=23.017973,113.411971
ORIGIN_KEY=dorm                 # 默认出发点：dorm / south_gate / north_gate / southwest_gate / northwest_gate
# 需要向百度申请哪些接口/配额，见 docs/地图接入说明.md（可直接转给百度同学）

# 地图治理（默认值已按"个人开发者额度留余量"设好）
MAP_QPS=2                       # 全局 QPS（官方 3，留 1 余量）
MAP_DAILY_SEARCH=80             # 地点检索日上限（官方 100）
MAP_DAILY_ROUTE=4000            # 路线日上限（官方 5000）
MAP_TIMEOUT=8                   # 单次请求超时（秒）
MAP_MAX_WAIT=4                  # 排队等令牌上限，超时直接降级提示
MAP_RETRY=1                     # 限流/超时的重试次数（配额/权限异常不重试）
MAP_AUTH_COOLDOWN=600           # 配额/权限异常后的熔断时长（秒）
MAP_CACHE_TTL_SEARCH=600        # 检索缓存
MAP_CACHE_TTL_ROUTE=600         # 路线缓存

# 接口安全
MAX_MESSAGE_CHARS=400
CHAT_RATE_PER_MIN=8             # 每会话每分钟提问上限
CHAT_GLOBAL_DAILY=300           # 全站每日提问上限
MAX_TOOL_CALLS_PER_TURN=10      # 单轮工具调用预算

# 云端用平台环境变量注入密钥时，设为 1 让平台值优先（否则 runtime.env 会覆盖）
# USE_PLATFORM_ENV=1
```

- 智谱 Key：`open.bigmodel.cn` 注册；百度 AK：`lbsyun.baidu.com` 创建**服务端**类型应用（不勾 SN）。
- `.env` / `runtime.env` 只放服务端，永不进前端、日志、分享包。

---

## 4. 架构要点（答辩对照）

**四要素**：LLM（tool-calling 循环，最多 8 步）｜记忆（会话历史 + 画像 + 知识库）｜
工具（9 个）｜评测护栏（3 套测试 + 确定性兜底）。

### 4.1 地图额度治理（`tools/map_gateway.py`）

所有百度请求走同一入口，统一实现：

| 能力 | 做法 |
|---|---|
| 全局限速 | 令牌桶，默认 2 QPS，跨会话共享；等不到令牌就降级提示，不硬打 |
| 日配额 | 检索 80/日、路线 4000/日，计数落盘 `data/map_usage.json`，跨进程共享 |
| 缓存 | 键 = 接口+参数（含起点/终点/方式），默认 10 分钟；命中不耗额度 |
| 并发合并 | 同一请求同时到达只真正调用一次（single-flight） |
| 错误分类 | `ok / no_result / rate_limited / quota_exhausted / auth_error / timeout / network_error …` |
| 有限重试 | 只对限流/超时/网络重试 1 次；配额与权限异常立刻熔断 10 分钟 |
| 统计 | 调用数、缓存命中率、限流次数 → `/api/status` 的 `map_stats` |
| 日志脱敏 | AK 永远打码，不写入日志 |

**已知限制**：QPS 限速是进程内状态；单容器单进程够用，若平台扩成多实例，
QPS 需要外部共享存储（Redis）才能严格全局，日配额已通过文件计数跨进程近似共享。

### 4.2 只查需要的交通方式

`verify_route(uid, mode)` 默认 `walking`，用户说骑行/打车才查对应方式；
同一家店已核实过的结果直接复用；换方式才新增一次请求 —— 一家店查三种方式的三倍额度消耗被消掉了。

### 4.3 意图分流（解决"问 B记鱼粥却推荐别家"）

| 用户意图 | 工具路径 |
|---|---|
| 指定门店（"你知道B记鱼粥吗"） | `find_restaurant` 确认身份 →（需要时）`verify_route` |
| 附近推荐（"附近有什么好吃的粥"） | `food_search` → `search_poi` → `verify_route` → `format_recommendation` |
| 路线（"怎么去X"） | `find_restaurant` → `verify_route(uid, mode)` |
| 门店比较 | 各自 `find_restaurant` + `verify_route`，逐项对比 |

`find_restaurant` 的规则：保留原店名 → 本地知识库（含别名）→ 3 公里检索 → 全市检索
（明确标注"超出校区范围"）→ 名称相似度只用于挑候选（≥0.6 才算匹配）。
**没有明确匹配就如实说不确定并追问校门/地标，绝不拿别的店顶替。**

### 4.4 附属设施过滤

名称/标签命中"停车场、公交站、充电桩、银行、公厕、超市…"一律不算餐馆；
只查到"B记鱼粥-门前停车场"时，会再全市找餐馆本体，找不到就只把停车场当线索并要用户确认位置。
（实测：2026-11-10 真实调用就是这个结果，卡片数为 0。）

### 4.5 卡片硬校验（模型不能靠提示词绕过）

`format_recommendation` 的条目必须同时满足：身份已核实（uid 在本次结果里）、
是餐饮门店、在推荐范围内、用户需要的交通方式路线已核实且起点一致。任一不过 → 整条剔除，
并在 `dropped` 里给出原因。事实字段（名称/地址/行政区/坐标/耗时/距离/路线链接）由后端注入；
人均/评分/营业时间/外卖无来源时显示"待确认"，模型估的标"预估"。

### 4.6 出发点与导航

- 5 个出发点（宿舍区 + 南门/北门/西南门/西北门），坐标均由百度地点检索**实测**（2026-11-10），
  页面顶部可切换；系统没有手机定位，不会声称知道"你的当前位置"；
- 路线链接按百度官方 URI 规范：`origin=latlng:纬度,经度|name:名称&destination=...&coord_type=bd09ll`，
  手机点开即导航页，起终点与卡片是同一门店、同一坐标。

### 4.7 接口安全

消息长度 ≤400 字、每会话 8 次/分钟、全站 300 次/日、单轮工具调用 ≤10 次；
超限返回 429/413 并在页面友好提示。密钥只在服务端，状态接口只暴露统计不暴露凭据。

---

## 5. 真实数据怎么入库

字段模板见 `data/examples/restaurants.json`，建议字段：
`正式名称 / 别名 / 分店 / 地址 / 坐标 / 坐标系 / 餐饮类型 / 人均及来源 / 学生评分 / 招牌菜 / 避雷 / 营业时间 / 高峰时段 / 可外卖 / 核实日期 / 数据来源`。

- **坐标必填**（bd09ll），否则无法核实路线；
- 只收**校区 3 公里内**的店；不同地图提供商的 POI ID 不能混用；
- 数据来源写清楚（问卷 / 点评整理 / 小红书整理），这是"原创数据"的证据。

---

## 6. 测试

```bash
python tests/test_protocol.py         # 3 部分：协议裁剪/自愈、ReAct 压力、步数耗尽
python tests/test_poi_consistency.py  # 10 项：范围/同名分店/失败不编造/事实注入/链接格式
python tests/test_map_gateway.py      # 13 项：限流提示、配额熔断、缓存、并发合并、只查步行、
                                      #        附属POI拒绝、指定店不替换、卡片硬校验、密钥卫生
```

全部用 mock 网络，不消耗真实额度；发布前必须全绿。

---

## 7. 部署与一致性校验

- 发布：owner 在 WorkBuddy 用「发布应用」把整个目录上传到云端容器跑 `python web/server.py`（平台注入 `PORT`）。
  **发布是快照**：本地改完必须重新发布才上线。
- 校验：`python verify_deploy.py`，比对本地与线上 `/api/status` 的 `build` 指纹（11 个源码文件）。

---

## 8. 团队协作

- `.env` / `runtime.env` / `data/sessions/` / `data/map_usage.json` / `__pycache__` 永不入库（见 `.gitignore`）；
- 私有仓库 + 分支：`main` 可发布；
- 发布权只在 owner（绑账号）：队友改 → 推送 → owner 拉取自测 → 发布 → `verify_deploy.py` 复核。

---

## 9. 进度与待办

- [x] ReAct 循环 + 9 工具 + 三层记忆 + 协议自愈
- [x] 地图统一网关（限速/缓存/去重/错误分类/熔断/统计/脱敏）
- [x] 意图分流 + 指定门店不替换 + 附属设施过滤
- [x] 卡片硬校验、未知字段待确认、官方导航链接
- [x] 5 个实测出发点、页面可切换
- [x] 接口限流与工具预算、build 指纹校验
- [x] **真实门店库 57 家**（周边 2–3km 10 家 + 5 个商圈 47 家，两篇攻略人工整理）+ 58 条评价
- [ ] 问卷实测评价补到每店 ≥3 条（当前每店 1 条，属单人观点，页面标「样本不足 3」）
- [ ] 坐标补录（AK 到手后跑 `tools/enrich_coords.py`，一次性约 60–120 次检索）
- [ ] 天气感知（雨天推外卖/近店）
- [ ] 定时推送（工作日 11:00 / 17:00）
- [ ] 可选：高德做第二来源核对、RapidFuzz 提升别名匹配、Tavily 搜公开网页线索

---

## 10. 知识库检索页（RAG 骨架）

页面顶部新增「AI 推荐 / 浏览检索」两个视图。检索页不调用大模型、不调用地图接口，
只读本地索引，因此**不消耗任何额度，断网也能演示**；直接访问 `/#kb` 可进入。

### 10.1 实现分层

| 文件 | 职责 |
|---|---|
| `tools/kb_index.py` | 检索内核：中文单字+二字 bigram 分词、同义词扩展、BM25 打分（纯标准库）|
| `tools/kb_search.py` | 门店检索、评价证据、口碑统计、详情聚合、演示/真实语料切换 |
| `web/server.py` | 三个只读接口 `/api/kb/status` `/api/kb/search` `/api/kb/store` |
| `web/static/index.html` | 视图切换 + 检索页 + 店铺详情弹窗 |
| `data/kb/reviews.jsonl` | 真实评价语料 58 条（两家攻略整理，字段见 `data/kb/README.md`）|
| `tests/test_kb.py` | 49 项：召回、过滤、统计一致性、来源标注、孤儿评价、演示隔离 |

### 10.2 两条铁律

1. **评分与评价只由后端统计**：均值、样本数、分布、摘录全部由 `kb_search` 算出，
   带来源与日期；模型与前端都不允许自己编数字。样本少于 `KB_MIN_SAMPLES`（默认 3）
   时标记「样本不足」，只报条数。
2. **演示数据与真实数据物理隔离**：`KB_DEMO=1` 才读 `data/examples/demo_*.json`，
   接口返回 `demo=true`，页面顶部挂横幅；正式模式只读 `data/`，空库如实返回 0 条。

### 10.3 跑法

```bash
# 界面联调（演示语料）
KB_DEMO=1 python web/server.py     # Windows: set KB_DEMO=1 && python web/server.py
# 打开 http://localhost:8787/#kb

# 只看检索层
python tests/test_kb.py
```

