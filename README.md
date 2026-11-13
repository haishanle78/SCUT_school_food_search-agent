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
├── config.py                # 配置：环境变量 → .env → 代码默认值（单一配置源）
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
│   ├── weather_gateway.py   # ★ 天气统一网关：30 分钟缓存/错误分类/决策翻译（雨→外卖优先）
│   └── __init__.py          # 10 个工具注册 + 参数容错适配层
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
│   ├── test_map_gateway.py  # 地图治理+门店真实性 13 项（对应验收标准）
│   └── test_weather_gateway.py  # 天气网关 12 项（决策翻译/缓存/降级/注入/密钥卫生）
└── data_collect/问卷模板.md
```

---

## 2. 快速开始

```bash
python main.py --demo      # 离线演示，无需任何 Key
python web/server.py       # 网页版 → http://localhost:8787
python main.py             # 配好 .env 后的交互模式（/profile 看画像，/quit 退出）
```

要开大模型需要密钥，四步（详见第 3 节）：

```bash
cp .env.example .env      # 1. 复制模板（.env 不进 git，只存在你本机）
# 2. 编辑 .env：填 OPENAI_API_KEYS / BAIDU_MAP_AKS（值见下节）
python web/server.py      # 3. 启动 → http://localhost:8787
curl localhost:8787/api/status   # 4. 验证：llm=true、map=true、llm_pool.keys=你的Key数
```

---

## 3. 配置项

```ini
# ===== 模型 =====
LLM_PROVIDER=openai
LLM_MODEL=glm-4-flash
LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4/chat/completions

# 单 Key 写法（兼容，池子等价于 1 把钥匙 / 2 路并发）
OPENAI_API_KEY=你的智谱Key
# 多 Key 写法（推荐，容量线性翻倍）：逗号分隔，不能有空格和引号
# OPENAI_API_KEYS=key1,key2,key3,key4
# 两者同时存在时以 OPENAI_API_KEYS 为准

# 或 DeepSeek（付费）
# LLM_PROVIDER=deepseek
# DEEPSEEK_API_KEY=...
# DEEPSEEK_API_KEYS=key1,key2

# ===== LLM Key 池与并发治理（默认值即为 glm-4-flash 免费档的安全值）=====
LLM_CONCURRENCY_PER_KEY=2       # 每把 Key 同时放行的请求数（免费档官方限制 2，别调大）
LLM_QUEUE_WAIT=10               # 抢不到名额时的排队上限（秒），超时回"稍后再试"
LLM_BREAKER_COOLDOWN=600        # 某把 Key 被 429/1214 限流后熔断多久再试（秒）
LLM_CONTEXT_CHAR_LIMIT=14000    # 单次请求字符预算，超限主动裁剪（防触发 8K token 限速档）
TOOL_RESULT_MAX_CHARS=600       # 裁剪时单条工具结果保留的字数

# ===== 地图 =====
# 单 AK 写法（兼容，池子等价于 1 把钥匙）
BAIDU_MAP_AK=你的AK
# 多 AK 写法（推荐）：逗号分隔，每把 AK 独立日配额/独立限速/独立熔断
# BAIDU_MAP_AKS=ak1,ak2,ak3,ak4
# 两者同时存在时以 BAIDU_MAP_AKS 为准

CAMPUS_LOCATION=23.017973,113.411971
ORIGIN_KEY=dorm                 # 默认出发点：dorm / south_gate / north_gate / southwest_gate / northwest_gate
# 需要向百度申请哪些接口/配额，见 docs/地图接入说明.md（可直接转给百度同学）

# 地图治理（默认值已按"个人开发者额度留余量"设好；日配额为每把 AK 的额度）
MAP_QPS=2                       # 每 AK 全局限速（官方 3，留 1 余量）
MAP_DAILY_SEARCH=80             # 每把 AK 的地点检索日上限（官方 100）
MAP_DAILY_ROUTE=4000            # 每把 AK 的路线日上限（官方 5000）
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

# 只读 .env，没有第二层覆盖文件；云端无 .env 时把同名键填进平台环境变量即可
```

**怎么填（照做即可）**

1. `cp .env.example .env`——`.env` 只在你本机存在，git 不跟踪，不进任何 PR。
2. LLM：把 `OPENAI_API_KEYS=` 这行的 `#` 去掉，填真实 Key；只有 1 把就填 `OPENAI_API_KEYS=那一把`（与单 Key 写法等价）。
3. 地图：把 `BAIDU_MAP_AKS=` 这行的 `#` 去掉，填真实 AK，同样逗号分隔。
4. 其余参数**保持默认**。`LLM_CONCURRENCY_PER_KEY=2` 是免费档硬限制，调大会换来 429。
5. 验证：启动后访问 `/api/status`，确认 `llm`、`map` 均为 `true`，`llm_pool.keys` 等于你填的 Key 数量，`map_stats.ak_usage` 里 AK 数量对得上。

**密钥从哪来**

- 智谱 Key：`open.bigmodel.cn` 注册 → 控制台创建 API Key（`glm-4-flash` 免费）。
- 百度 AK：`lbsyun.baidu.com` 创建**服务端**类型应用，**不勾 SN 校验**，白名单可留空。

**安全纪律**

- `.env` 只放服务端，永不进前端、日志、分享包、截图、PPT。
- 不要 `git add -f .env`（`.gitignore` 已拦截，强制添加会泄露）。
- 云端部署：把 `.env` 里的同名键填进平台「环境变量/密钥管理」，或把 `.env` 一并上传；两者都没有时才走代码默认值（无 Key = 离线模式，无 AK = 不查地图）。

**排错**

| 现象 | 原因 | 处理 |
|---|---|---|
| `/api/status` 里 `llm=false` | `.env` 未创建 / 变量名拼错 / 值仍是占位符「在此填入」 | 检查文件名是 `.env` 不是 `.env.txt`；占位符会被识别为未配置 |
| 一直返回"当前提问的人较多" | 所有 Key 都在熔断中或排队超时 | 看 `/api/status` 的 `llm_pool`，多数是 Key 无效或并发被占满 |
| 地图卡片全被剔除 | AK 配额用尽或类型不对（非服务端应用） | 看 `map_stats.ak_usage` 的熔断标记，换 AK 或改应用类型 |
| 改了 `.env` 但不生效 | 进程已有同名环境变量时 `.env` 不会覆盖（`load_dotenv(override=False)`），或改完没重启 | 重启服务；Windows 下确认文件名是 `.env` 而非 `.env.txt` |
| 行尾注释导致启动报 `ValueError` | 旧版解析器不剥行尾 `#` | 已修复并有回归测试 `tests/test_config_env.py`；模板可放心带注释 |

---

## 4. 平台能力与自研边界（答辩口径）

评委会问「哪些是 WorkBuddy 的，哪些是你们写的」。这张表就是答案，别把自研工具说成平台原生 Skill。

| 能力 | 归属 | 说明 |
|---|---|---|
| 托管运行、公网链接、访问控制 | **WorkBuddy 平台** | 发布后由平台容器运行 `python web/server.py`，注入 `PORT`；链接与分享码由平台发放 |
| 资料库（知识库表格/文档/检索） | **WorkBuddy 平台** | 用于人协作编辑与材料展示；应用运行时仍读仓库内 `data/*.json`，两者同源、导出同步 |
| 定时任务 / 自动化（工作流） | **WorkBuddy 平台** | 饭点推送等；需在平台侧单独配置，不是代码里的定时器 |
| 专家 / 连接器 | **WorkBuddy 平台** | 会话级能力，**不会进入已部署的应用**；连接器可用于批量取数（如补 POI 坐标） |
| 大模型、地图、天气接口 | **第三方服务** | 智谱 GLM（免费档）、百度地图；密钥自管，只在服务端 |
| ReAct 循环、10 个工具、三层记忆、卡片硬校验、地图网关、BM25 知识库检索、检索页/聊天前端 | **本仓自研代码** | 全部 Python/JS 实现，纯标准库；不是平台原生 Skill，也不冒充 |
| 10 个测试文件、build 指纹校验 | **本仓自研** | `python tests/*.py` 可在无 `.env` 的干净环境跑（外部接口全部 mock） |

**一句话**：平台负责「托管、协作、调度」，代码负责「agent 逻辑与产品形态」；两边数据同源，评分点的三个词都能指到具体产物。

---

## 5. 架构要点（答辩对照）

**四要素**：LLM（tool-calling 循环，最多 8 步）｜记忆（会话历史 + 画像 + 知识库）｜
工具（10 个）｜评测护栏（4 套测试 + 确定性兜底）。

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

### 4.8 出口资源池化：LLM Key 池（`llm/keypool.py`）与地图 AK 池

LLM 与地图是两类"会烧额度、会被限流"的出口资源，统一按「池化」治理
（对照 `map_gateway` 的思路，二者机制同构）：

| 能力 | LLM Key 池 | 地图 AK 池 |
|---|---|---|
| 池化 | `OPENAI_API_KEYS=k1,k2,k3,k4`（兼容旧单 Key 变量） | `BAIDU_MAP_AKS=ak1,ak2,ak3,ak4` |
| 并发上限 | 每 Key 信号量 2 路（glm-4-flash 免费档限制） | 每 AK 令牌桶 2 QPS |
| 调度 | 最闲优先 + 轮询摊匀，FIFO 排队 | 同左 |
| 排队超时 | `LLM_QUEUE_WAIT=10` 秒 → LLMBusy → 前端"稍后再试" | `MAP_MAX_WAIT=4` 秒 → 降级提示 |
| 熔断 | 429/1214/并发超限 → 该 Key 熔断 10 分钟，换钥匙重试，用户无感 | 配额/权限异常 → 该 AK 熔断 10 分钟，换 AK 重试 |
| 上下文纪律 | 每次请求前按字符预算（`LLM_CONTEXT_CHAR_LIMIT=14000`）主动裁剪：先截断工具结果 → 再压缩历史，坚决不进智谱 8K token 限速档 | — |
| 统计 | `/api/status` 的 `llm_pool`（Key 数、在途、今日用量，密钥打码） | `map_stats` 的 `ak_usage`（每 AK 用量与熔断状态） |

容量账（4 Key / 4 AK 配置下）：LLM 并发 8 路 ≈ 每分钟 ~48 人次提问吞吐；
地图检索 320 次/日、路线 16000 次/日、QPS 8。

**已知限制**：两个池的状态均为进程内实现（LLM 池纯内存，地图日计数落盘）；
单容器单进程成立。若平台扩多实例，LLM 池需换外部共享存储（如 Redis），
日配额文件计数为跨进程近似共享。

---

## 6. 真实数据怎么入库

字段模板见 `data/examples/restaurants.json`，建议字段：
`正式名称 / 别名 / 分店 / 地址 / 坐标 / 坐标系 / 餐饮类型 / 人均及来源 / 学生评分 / 招牌菜 / 避雷 / 营业时间 / 高峰时段 / 可外卖 / 核实日期 / 数据来源`。

- **坐标必填**（bd09ll），否则无法核实路线；
- 只收**校区 3 公里内**的店；不同地图提供商的 POI ID 不能混用；
- 数据来源写清楚（问卷 / 点评整理 / 小红书整理），这是"原创数据"的证据。

---

## 7. 测试

```bash
python tests/test_protocol.py         # 3 部分：协议裁剪/自愈、ReAct 压力、步数耗尽
python tests/test_poi_consistency.py  # 10 项：范围/同名分店/失败不编造/事实注入/链接格式
python tests/test_map_gateway.py      # 13 项：限流提示、配额熔断、缓存、并发合并、只查步行、
                                      #        附属POI拒绝、指定店不替换、卡片硬校验、密钥卫生
python tests/test_concurrency.py      # 8 项：LLM Key 池排队/分流/熔断/故障转移、20 路并发
                                      #        防雪崩、上下文预算、地图 AK 池故障转移/独立配额/负载均衡
python tests/test_weather_gateway.py  # 12 项：雨/高温决策翻译、全站缓存、超时降级、
                                      #        system prompt 注入/摘除、工具注册、密钥卫生
```

全部用 mock 网络，不消耗真实额度；发布前必须全绿。

---

## 8. 部署与一致性校验

- 发布：owner 在 WorkBuddy 用「发布应用」把整个目录上传到云端容器跑 `python web/server.py`（平台注入 `PORT`）。
  **发布是快照**：本地改完必须重新发布才上线。
- 校验：`python verify_deploy.py`，比对本地与线上 `/api/status` 的 `build` 指纹（14 个源码文件）。

---

## 9. 团队协作

- `.env` / `data/sessions/` / `data/map_usage.json` / `__pycache__` 永不入库（见 `.gitignore`）；
- 私有仓库 + 分支：`main` 可发布；
- 发布权只在 owner（绑账号）：队友改 → 推送 → owner 拉取自测 → 发布 → `verify_deploy.py` 复核；
- **匿名要求**：参赛网页、截图、录屏、PPT 一律不出现姓名/学号/指导教师；Git 提交历史不必为匿名改写。

---

## 10. 进度与待办

- [x] ReAct 循环 + 9 工具 + 三层记忆 + 协议自愈
- [x] 地图统一网关（限速/缓存/去重/错误分类/熔断/统计/脱敏）
- [x] 意图分流 + 指定门店不替换 + 附属设施过滤
- [x] 卡片硬校验、未知字段待确认、官方导航链接
- [x] 5 个实测出发点、页面可切换
- [x] 接口限流与工具预算、build 指纹校验
- [x] **真实门店库 57 家**（周边 2–3km 10 家 + 5 个商圈 47 家，两篇攻略人工整理）+ 58 条评价
- [x] 出口资源池化：LLM Key 池（排队/分流/熔断）+ 地图 AK 池 + 上下文预算（防 8K 限速）
- [x] 天气感知（`tools/weather_gateway.py`：百度天气 API + 30 分钟全站缓存，
      雨→外卖优先/步行≤10 分钟、高温→近店优先，决策由后端确定性规则翻译，失败静默降级）
- [ ] 问卷实测评价补到每店 ≥3 条（当前每店 1 条，属单人观点，页面标「样本不足 3」）
- [ ] 坐标补录（AK 到手后跑 `tools/enrich_coords.py`，一次性约 60–120 次检索）
- [ ] 定时推送（工作日 11:00 / 17:00）
- [ ] 可选：高德做第二来源核对、RapidFuzz 提升别名匹配、Tavily 搜公开网页线索

---

## 11. 知识库检索页（RAG 骨架）

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
| `tests/test_kb.py` | 58 项：召回、过滤、统计一致性、来源标注、孤儿评价、缺语料降级 |

### 10.2 两条铁律

1. **评分与评价只由后端统计**：均值、样本数、分布、摘录全部由 `kb_search` 算出，
   带来源与日期；模型与前端都不允许自己编数字。样本少于 `KB_MIN_SAMPLES`（默认 3）
   时标记「样本不足」，只报条数。
2. **演示数据与真实数据物理隔离**：`KB_DEMO=1` 才读 `KB_DEMO_DIR`（默认
   `data/examples`）下的演示语料，接口返回 `demo=true`，页面顶部挂横幅；
   正式模式只读 `data/`，空库如实返回 0 条。
   **仓库不附带任何演示语料**（避免假商家混进正式推荐）；需要演示时自备该目录，
   测试由 `tests/test_kb.py` 在临时目录生成固定小样本，用完即弃。

### 10.3 跑法

```bash
# 界面联调（需自备演示语料目录：demo_stores.json + demo_reviews.jsonl）
KB_DEMO=1 KB_DEMO_DIR=path/to/fixtures python web/server.py
# Windows: set KB_DEMO=1 && set KB_DEMO_DIR=path\to\fixtures && python web/server.py
# 打开 http://localhost:8787/#kb

# 只看检索层
python tests/test_kb.py
```

