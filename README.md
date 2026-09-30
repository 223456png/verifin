# VeriFin — 金融证据校验 Agent

> 让金融 Agent 不只「检索相关文本」，而是验证证据在**实体、时间、指标、口径**上是否真正匹配；不匹配则自动诊断、重规划（Replan）并重试。
>
> A financial evidence-verification agent built on LangGraph: retrieval is just the input — the real output is a **verifiable, traceable Claim-Evidence binding**, with automatic replanning when evidence fails four-factor verification. Ships with an **MCP server** so any agent client can reuse its retrieval / verification / calculation capabilities.

[![CI](https://github.com/223456png/verifin/actions/workflows/ci.yml/badge.svg)](https://github.com/223456png/verifin/actions/workflows/ci.yml)
[![tests](https://img.shields.io/badge/tests-225%20passed-brightgreen)](tests/)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)
[![MCP](https://img.shields.io/badge/MCP-server-8A2BE2)](src/verifin/mcp/server.py)

---

## TL;DR

| | |
|---|---|
| **一句话** | LangGraph 多智能体金融 RAG，核心不是「检索到」，而是「证据经得起四要素校验」 |
| **技术栈** | LangGraph · ChromaDB + BM25 混合检索 · FastMCP · FastAPI · DeepSeek/OpenAI 兼容 LLM（可降级） |
| **最硬的三个数字** | 真实 FinQA 100 样本 **EM 8.0% → 29.0%**（LLM 程序生成，净 +8 题零回归）· 文档召回 **56% → 88%** · 多跳合成基准 EM **37.5% → 100%** |
| **最值钱的工程点** | 有**消融数据否定自己初始架构**的实证（LLM planner 反而更差），以及一个三层防御的 **PoT 安全沙箱** |
| **225 项测试** | 全部离线可复现，默认路径零外部 API 依赖 |

---

## 为什么做这个项目

金融 RAG 的检索环节普遍把「语义相关」当「证据可用」，导致四类高频错误：

| 问题 | 具体表现 |
|---|---|
| **证据混淆** | 检索到 "Apple 2023 revenue" 与 "Apple 2024 revenue"，语义都相关，但用户问的是 2024 |
| **指标错配** | 拿 "gross margin" 当 "operating margin" 用 |
| **口径不一** | 同一指标在不同文档定义不同（adjusted vs GAAP），数值冲突无仲裁 |
| **证据链断裂** | 多跳问题需要 A 文档的 revenue + B 文档的 cost，只找到其中一个 |

VeriFin 的答案：**以证据校验为中心的 Agent 循环** —— 检索 → 四要素校验（Entity / Period / Metric / Definition）→ 冲突仲裁 → 不通过则诊断原因、重构查询、重试；数值问题由 PoT 程序执行器完成受限算术求值。全程轨迹可观测、可回放、可消融。

---

## 核心特性

- **MCP Server：把能力暴露为标准协议** —— FastMCP stdio 服务，四个**只读**工具（`retrieve` / `expand_document` / `verify_claim` / `calculate`），任何 MCP 客户端（Claude Desktop / Cursor / 自建 Agent）无需理解内部实现即可复用。不暴露任何写路径与 shell，协议级冒烟测试六步全通 → 见 [MCP Server](#mcp-server)
- **LangGraph 状态机编排**：planner → retriever → verifier → calculator → replanner → reporter 六节点循环，checkpoint 持久化多轮会话；领域逻辑（校验/计算/仲裁）全部为无状态 Tool，与框架解耦、可独立单测
- **混合检索 + 多查询合并**：BM25 + Dense 双路召回（可插拔 embedding：离线 hash / 语义 bge-small），多子任务检索结果按 chunk_id 去重合并（保留原始问题的强词法信号）；small-to-big 父文档补全，解决表格数值块 BM25 信号弱的问题
- **四要素校验器**：实体词边界匹配、期间年份交集、指标词典同义词归一、口径 token 重叠率——全部规则化、零 LLM 依赖、确定性可测
- **冲突仲裁与重规划**：≤2% 数值差异融合均值，>5% 强制 Replan；来源信任度加权（年报申报 > 新闻稿 > 研报）；Replan 按失败原因（缺年份 / 缺指标 / 缺实体）重构查询
- **PoT 安全计算器 + 程序模板执行器**：`calc_expression` 三层防御的受限求值沙箱（AST 节点白名单 + `__builtins__` 置空 + SIGALRM 超时 + 幂次决定性 guard，**禁任意代码 / 禁 IO**，并有反向测试锁定）；FinQA 推导型问题（占 93%）经模板检测 → 候选数值枚举 → 单位归一 → 锚分剪枝 → 确定性求值；五类模板——年份键控（growth_pct / difference）、实体键控（cross_entity_diff 跨实体差值 / argmax_relay 三实体比较接力，实体锚定扫描取证）与短语键控（**ratio**："what percentage of X are Y" 比率题，行标签 × 问题短语双向词重叠锚定分子/分母）
- **FinQA DSL 多步执行器 + 可插拔 LLM 程序生成（可降级）**：模板外的多步算术（table_sum / table_average / `#N` 步骤引用链）由 `LLMProgramGenerator` 经 `complete(prompt)->str` 注入任意 LLM 生成 DSL 程序，编号候选（`vN` 单值 / `tN` 值组）供其消歧；输出先过结构校验（算子白名单 / 引用越界 / 幻觉候选拒绝）再逐步 PoT 求值——**LLM 输出永不直接执行**；LLM 不可用 / 异常 / 输出非法时自动降级回确定性模板路径（默认行为与无 LLM 基线完全一致）
- **最小 Tool-Calling Harness（模型驱动循环，~150 行零框架）**：与固定状态机互补的 `ToolCallingHarness`——LLM 按 OpenAI function calling 协议自主决定调什么工具、何时停（`OpenAICompatLLM.chat` 新增 messages+tools 支持，失败返回 None 触发 llm_error 停机）；工具层复用 `ToolRegistry`，未知工具/执行异常以 tool 消息回喂模型自主纠正，三停机条件（finished / max_turns / llm_error）防死循环。**架构定位是诚实分层**：graph 的确定性路由是 EM 基线（LLM planner 自由分解 24% < 规则混合 29% 的消融数据支撑），harness 是开放任务的能力层，不混跑评测口径
- **会话偏好记忆**：多轮对话中抽取实体/期间/指标/来源偏好并作用于后续检索与仲裁（ConvFinQA 多轮继承）
- **可插拔 LLM 规划层（可降级）**：`LLMPlanner` 经 `complete(prompt)->str` 注入任意 LLM provider（OpenAI / Anthropic / 本地模型），把金融问题分解为检索子任务 + 四要素 + 计算规格；LLM 不可用 / 异常 / 输出非法时自动降级到确定性规则 planner（零外部依赖兜底，测试与 demo 开箱即用）
- **多跳 QA 合成管线**：从带来源标识的原子 chunk 抽取种子事实 → 跨文档配对（增长链 / 跨实体差值 / 三实体接力）→ 按推理深度合并 → **四重校验**（语义 / 推理 / 来源跨度 / 反伪多跳），确定性合成 2-hop/3-hop 评测基准——反伪多跳用数值级比较（年份 "2023" 不误含 "20"），实测拒掉与单文档现成数值重合的差值题
- **RAGAs 风格生成质量指标（规则近似）**：Faithfulness（答案数值声明被「过校验证据 ∪ PoT 计算产物」支持的比例）与 Answer Relevancy（问题内容词覆盖率）——确定性、可复现、可进 CI，与 LLM-as-judge 的差异如实声明
- **全链路评测体系**：合成消融套件（5 配置 × 20 样本）+ 真实 FinQA/ConvFinQA 端到端评测 + LLM-as-judge，报告含答案类型分解与诚实边界声明
- **FastAPI 服务层 + Web Demo**：单线程执行器解决 chromadb SQLite 线程亲和问题；Web Demo 可视化 Agent 轨迹、证据卡片与计算过程

---

## 架构

```text
                         ┌──────────────────────────────┐
                         │         User Query           │
                         └──────────────┬───────────────┘
                                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         LangGraph Agent Loop                        │
│                                                                     │
│   ┌──────────┐    ┌───────────┐    ┌──────────┐    ┌───────────┐   │
│   │ Planner  │───▶│ Retriever │───▶│ Verifier │───▶│Calculator │   │
│   │ 任务分解  │    │ BM25+Dense│    │ 四要素校验│    │ PoT 求值  │   │
│   │ 子查询生成│    │ 多查询合并 │    │ 冲突仲裁  │    │ 程序模板  │   │
│   └──────────┘    └───────────┘    └────┬─────┘    └─────┬─────┘   │
│        ▲                                  │                │         │
│        │           校验失败/数值冲突        │                │         │
│        │                                  ▼                ▼         │
│   ┌────┴─────────┐              ┌──────────────┐  ┌──────────────┐  │
│   │  Replanner   │◀─────────────│ 4 类失败诊断  │  │ 候选锚定剪枝  │  │
│   │ 查询重构重试  │              │ (年份/指标/   │  │ (行标签词重叠 │  │
│   └──────────────┘              │  实体/口径)   │  │  双向覆盖率) │  │
│                                 └──────────────┘  └──────────────┘  │
│                                                                     │
│   ┌──────────┐                                                      │
│   │ Reporter │──▶ 带引用的可溯源答案 + 轨迹回放                      │
│   └──────────┘                                                      │
└─────────────────────────────────────────────────────────────────────┘
          ▼                    ▼                     ▼
   ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐
   │ FastAPI     │    │  Web Demo    │    │   MCP Server     │
   │ /ask /health│    │ 轨迹可视化    │    │ 4 只读工具 / stdio│
   └─────────────┘    └──────────────┘    └──────────────────┘
```

**关键设计决策**

| 决策 | 理由 |
|---|---|
| 领域逻辑与 LangGraph 解耦 | Verifier/Calculator/仲裁器均为纯函数 Tool，框架可替换，单测无图依赖 |
| 无 LLM 的确定性规则引擎 | 校验/计算/仲裁全程可解释、可回放、结果确定——评测数字可信的前提 |
| 单线程执行器承载 Agent 执行 | chromadb SQLite 句柄线程亲和，ASGI 工作线程直接调用会抛异常 |
| anchor score 锚定体系 | 表格行 metric 命中（0.0）< 行标签词重叠（0.5）< 散文证据（1.0）< 距离兜底（50+），跨层级候选可比较 |
| MCP 只读暴露 | 对外复用「检索 / 校验 / 计算」三件套，不暴露写路径与 shell——能力开放与攻击面收敛的平衡 |

---

## MCP Server

VeriFin 把核心能力封装为 **MCP（Model Context Protocol）服务**，任何支持 MCP 的客户端都能直接调用——不需要读它的源码，也不需要把金融文档喂给通用模型去「自由发挥」。

启动：

```bash
# 需先构建索引（默认读取项目根 ./bge_env）
VERIFIN_INDEX_DIR=./bge_env python -m verifin.mcp.server

# 或使用安装后的入口点
VERIFIN_INDEX_DIR=./bge_env verifin-mcp
```

### 工具列表

| 工具 | 入参 | 返回 | 用途 |
|---|---|---|---|
| `retrieve` | `query`, `top_k`=10, `use_reranker`=True | `[{chunk_id, doc_id, score, content, metadata}]` | BM25 + Dense + RRF 混合检索，可选 Cross-Encoder 精排 |
| `expand_document` | `doc_id` | `[{chunk_id, content, metadata}]` | 父文档补全——表格被分块切断时的标准补救 |
| `verify_claim` | `claim`, `evidence_text` | 逐要素匹配明细 + `PASS`/`REJECT` | 四要素规则校验（实体 / 期间 / 指标 / 定义） |
| `calculate` | `expression` | `{expression, value, unit, error, is_valid, execution_time_ms}` | PoT 安全计算：AST 白名单受限求值，支持算术与 math 白名单函数。求值失败不抛异常，改为 `error` 非空、`value` 为 `null` |

### 客户端配置

**Claude Desktop**（`claude_desktop_config.json`）：

```json
{
  "mcpServers": {
    "verifin": {
      "command": "python",
      "args": ["-m", "verifin.mcp.server"],
      "env": { "VERIFIN_INDEX_DIR": "/absolute/path/to/bge_env" }
    }
  }
}
```

**Cursor / 通用 MCP 客户端**（`.cursor/mcp.json` 或等价配置）：

```json
{
  "mcpServers": {
    "verifin": {
      "command": "python",
      "args": ["-m", "verifin.mcp.server"],
      "env": { "VERIFIN_INDEX_DIR": "${workspaceFolder}/bge_env" }
    }
  }
}
```

### 安全边界

- **只读设计**：四个工具全部无副作用，不暴露写路径、文件系统遍历或 shell
- **计算沙箱**：`calculate` 为三层防御的受限求值——AST 节点白名单拒绝 `Import`/`Attribute`/未注册调用，`__builtins__` 置空仅暴露 math 白名单，SIGALRM 超时（仅 Linux 主线程生效，API 工作线程 / Windows 无此兜底）+ 幂次/指数 guard（嵌套幂与非 int 指数一律快速拒绝）保证快速确定性拒绝
- **反向测试锁定**：`tests/test_calculator.py` 显式断言 `open('/etc/passwd')`、`__import__('os')`、`().__class__` 均被拒绝
- **惰性加载**：导入模块不触发磁盘 IO，索引在首次工具调用时加载

### 协议级冒烟测试

```bash
python scripts/test_mcp_server.py
# [1] initialize OK  [2] tools/list OK  [3] calculate  [4] verify_claim  [5] retrieve  [6] expand_document
```

---

## 评测结果

### 合成评测套件（20 样本 × 5 配置消融，多轮）

| 配置 | 准确率 | Faithfulness | Answer Relevancy | 消融贡献 |
|---|---|---|---|---|
| **完整系统** | **100.0%** | **100.0%** | **98.2%** | — |
| − Verifier | 80.0% | — | — | **+20 pp** |
| − Replanner | 85.0% | — | — | **+15 pp** |
| − 偏好记忆（多轮子集 40% vs 100%） | 85.0% | — | — | **+15 pp** |

> Faithfulness / Answer Relevancy 为 RAGAs 同名指标的**确定性规则近似**
> （非 LLM-as-judge）：前者按「通过四要素校验的证据 ∪ PoT 计算产物」
> 计算答案数值声明的支持率，与验证闭环绑定；详见报告「生成质量指标」节。

### 多跳合成基准（16 条：2-hop 14 / 3-hop 2，四重校验产出）

| 子集 | 样本数 | Phase 9.6 | Phase 9.7 | 说明 |
|---|---|---|---|---|
| growth_chain（单实体跨年增长） | 6 | 100% | **100%** | PoT 计算链端到端全对 |
| cross_company_diff（跨实体差值） | 8 | 0% | **100%** | 实体锚定取证 + `subtract(A,B)` 差值链 |
| argmax_relay（三实体接力） | 2 | 0% | **100%** | `greater` 比较链 + 胜者第二指标 `lookup` |
| **整体** | 16 | 37.5% | **100%** | Faithfulness 100%、Tool F1 100% |

> Phase 9.6 的 37.5% 是「检索到证据 ≠ 算得出答案」的直接实证（gold 文档全部
> 召回但跨实体算术无模板可执行）；Phase 9.7 新增 `cross_entity_diff` /
> `argmax_relay` 实体键控模板与实体锚定数值扫描（实体提及 + 指标同义词 +
> 年份三信号最近邻）后三类子集全部命中。诚实边界：该基准由本项目合成管线
> 产出，问题形态与模板同源，真实 FinQA 的跨实体/多步算术仍属 Phase 10。
> 复现：`make multihop-synth && make multihop`。

### 真实 FinQA test（100 样本抽样）

| 指标 | Phase 8 | Phase 9 | Phase 10 | Phase 11 | Phase 12 | 提升 |
|---|---|---|---|---|---|---|
| 答案 EM | 1.0% | 6.0% | 7.0% | 8.0% | **21.0%** | **21×** |
| 模板可检测子集 EM | — | 24.0% | 18.4% | 21.1% | **36.8%** | — |
| 程序可检测规模 | — | 25 题 | 38 题 | 38 题 | **57 题** | +19（ratio 变体 + average） |
| 文档级召回 | 48% | 56% | 56% | **88.0%** | 88.0% | **+40 pp** |

**Phase 11 检索召回修复（+32 pp doc recall）**：归因诊断（`scripts/diagnose_recall.py` 六组对照 + `scripts/trace_gold_loss.py` 44 失败样本逐个跑图追踪）发现检索层本身 88%，端到端 56% 的损耗 59% 来自"四要素校验失败 chunk 被永久拉黑后**从证据池驱逐**"（verifier 对每个子任务按各自 claim 重新校验，chunk 在子任务 A 失败不代表在子任务 B 下无价值）。修复四件套：①排除语义修正（只防新检索重复命中、不驱逐历史证据）②表格 chunk 豁免（`is_table` 标记，四要素对表格天然过严）③replanner 原地打转终止（重复子任务视为无新信号）④端到端切 bge 索引（`make benchmark-finqa`）。EM 净 +2/-1（43/72 新对，73 诚实披露为更大证据池下 ratio 锚定变化）。

**Phase 12 程序覆盖扩展 + 12.2 跨文档污染守卫（EM 8.0% → 21.0%）**：①句式扩展——ratio_of_to / pct_of_to / as_pct_of / hyphen_ratio / ROI-return 关键词 + average 年份区间模板，100 样本程序可检测子集 38 → 57 题，全测 1147 六族句式未检出 **122 → 9（-93%**，设计目标 ≥60%）；②v12 引入的三例回归逐样本复现（`scripts/repro_regressions.py`）定位到**跨公司同构表污染**——CME 题召回 GPN 股权计划表，分母短语命中其列头/行标签锚到异公司值 766801（正确分母为同表 total 行 1217121），`_ratio_candidates` 新增跨文档污染守卫：分母候选与分子最优锚定不同文档 → 剔除异文档候选，清空后 total 行回退限定分子同文档；③reranker 失败缓存（HF 不可达时每进程只尝试一次加载，防 hub 内部 5×8s 重试 × 每次检索的 40s 级阻塞，此前 100 样本评测被拖至 34 分钟）+ `RERANKER_MODEL` 环境变量支持本地模型目录离线加载。**同降级 reranker 口径**（与 Phase 12.1 基线一致）3/3 回归样本修复，错误分布 calculation_error 38 → 33、verifier_reject 29 → 26，EM 13.0% → 21.0%。200 项测试全绿。

**Phase 12.3 真实 LLM 程序生成接入（EM 21.0% → 29.0%）**：可插拔 LLM provider（OpenAI 兼容直连 DeepSeek / 本地桥接双通道，`complete` 永不抛异常、失败空串触发既有确定性降级）接入 LLM 程序生成路径，`--llm-mode` 单变量消融开关 + 逐调用统计（成功率/延迟）落盘。**100 样本消融（同降级 reranker 口径）**：规则模板 21.0% → **LLM 程序生成 29.0%（+8pp，净 +8 题 / 零回归**——LLM 只接管确定性模板未检出的 46 题，全部执行成功，均值 762ms）；LLM planner 消融（24.0%，+5/-2）显示其分解虽 100% 成功但与四要素校验的下游兼容性差，**最终架构取规则 planner + LLM 程序生成的混合**——数据驱动的架构决策而非默认全 LLM。错误分布 verifier_reject 29 → 19。

**Phase 12.4 MCP Server + LLM-as-judge**：FastMCP stdio 服务四工具只读暴露，协议级冒烟六步全通；LLM-as-judge（DeepSeek，40 样本子集，0 调用失败）分层发现——**答错 27 题 faithfulness 0%**（噪声证据放行后算错值的直接实证），答对 11 题 36%，judge 与 EM 一致性 82%。

**答案类型分解**（诚实口径）：提取型 4% / 推导型 93%（需 program 算术）/ 布尔型 3%。确定性模板覆盖 growth_pct / difference / cross_entity_diff / argmax_relay / **ratio**（"what percentage of X are Y"，FinQA test 占 ~17%）五类。Phase 10 ratio 模板在 100 样本上检出执行 13 题（此前 0）。Phase 10.1 锚定改进（年份 token / "after \<year\>"→thereafter 行 / 正文脚注扫描 / 分母 total 行兜底 / 列一致性 / 负值剪枝）后，**理想检索口径**（gold 文档全文直接喂给执行器，剥离检索层）ratio 锚定精度 **2/13 → 7/13**。复现：`scripts/verify_ratio_ideal.py`。

### 真实 ConvFinQA dev 多轮（100 轮）

| 指标 | Phase 8 | Phase 9 |
|---|---|---|
| 文档级召回 | 52% | **65%** |

### Agent 行为指标

| 指标 | Phase 9 | Phase 10 | 目标 |
|---|---|---|---|
| 工具调用精准率 Tool F1 | 95.6% | **94.8%** | ≥ 85% ✓ |
| 重规划有效率 | 45.7% | **50.0%** | ≥ 60% |
| 平均轨迹步数 | 7.65 | **7.57** | ≤ 4 |

> Phase 10 Tool F1 微降为**口径变化**而非行为退化：pseudo-gold calc 触发词
> 补齐比率类（portion/fraction/percent of）后，更多样本被要求 calc 工具族
> （比率题此前静默落入提取路径，现经 ratio 模板真正进入 calculator）。

### 检索升级（单查询直接检索口径，100 样本）

| 配置 | doc recall@20 |
|---|---|
| hash + hybrid | 81.0% |
| **bge + hybrid** | **86.0%（+5.0 pp）** |
| bge + hybrid + rerank | 85.0%（-1.0 pp，通用精排对金融表格无益） |

> 语义 embedding（bge-small-en-v1.5）替换无语义哈希向量带来明确提升；通用
> cross-encoder rerank 在此口径无益，属诚实边界。复现：`scripts/compare_retrieval.py`。

> **完整评测结论、逐 Phase 归因与已知局限**见 [`docs/EVALUATION.md`](docs/EVALUATION.md)。

---

## 快速开始

### 安装

```bash
git clone https://github.com/223456png/verifin.git
cd verifin
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env                                 # 可选：按需配置索引与 LLM 密钥
```

### 方式一：零数据演示（内置合成语料）

```bash
# 启动 API + Web Demo（http://127.0.0.1:8000）
make demo
# 等价于：python -m verifin.api.app
```

打开浏览器访问 `http://127.0.0.1:8000`，直接提问，可视化查看 Agent 轨迹、证据卡片与计算过程。

### 方式二：Docker 一键启动

```bash
docker compose up --build
# 服务暴露在 http://127.0.0.1:8000
```

### 方式二 b：Hugging Face Space 自托管

`deploy/hf-space/` 提供等价的 Docker Space 部署资产（合成语料零依赖模式，锁定仓库提交构建）——可直接创建 Docker Space 后推送该目录，或参考其 Dockerfile 自行部署到任意容器平台。前端更新用 `make space-sync` 同步。

### 方式三：真实 FinQA 评测

```bash
# 1. 准备数据（FinQA 数据集放入 data/finqa/）
#    https://github.com/czyssrs/FinQA

# 2. 构建索引（默认 hash，离线零依赖）
python scripts/build_index.py --dataset finqa --data-dir ./data/finqa

# 2b. 构建语义索引（bge-small，需 `[bge]` 依赖 + 模型权重，doc recall +5 pp）
# pip install -e ".[bge]"                 # 安装 sentence-transformers
# HF_ENDPOINT=https://hf-mirror.com \     # 国内网络可走镜像首次下载
#   python scripts/build_index.py --dataset finqa --data-dir ./data/finqa \
#     --embedding bge --persist-dir ./bge_env

# 3. 运行评测
python scripts/run_benchmark.py --dataset finqa --max-samples 100 --output ./results

# 4. 消融实验
python scripts/run_benchmark.py --dataset finqa --ablation --max-samples 50

# 5. 多轮评测（ConvFinQA）
python scripts/run_benchmark.py --dataset convfinqa --multi-turn
```

### 方式四：作为 MCP 服务接入 Agent 客户端

```bash
VERIFIN_INDEX_DIR=./bge_env python -m verifin.mcp.server
```

配置方式见上方 [MCP Server](#mcp-server) 章节。

### API 调用

```bash
# 健康检查
curl http://127.0.0.1:8000/health

# 单轮问答
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What was the growth rate of net revenue from 2014 to 2015?"}'

# 多轮（同一 conversation_id 复用会话状态与偏好记忆）
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What about 2016?", "conversation_id": "demo-session"}'
```

### 运行测试

```bash
python -m pytest tests/ -q          # 225 项全绿
python -m pytest tests/ --cov       # 带覆盖率
```

---

## 项目结构

```text
src/verifin/
├── ingestion/        # 文档加载、解析、分块（表格序列化保结构）
├── indexing/         # BM25 + ChromaDB 向量索引 + SQLite 元数据
├── retrieval/        # 混合检索、多查询合并、small-to-big 补全
├── core/             # LangGraph 图与节点（planner/retriever/verifier/
│                     #   calculator/replanner/reporter）+ 会话状态
├── tools/            # 无状态领域工具：四要素校验器、冲突仲裁、
│                     #   PoT 计算器、程序模板执行器、表格解析
├── harness.py        # 最小 tool-calling harness（模型驱动工具循环，
│                     #   与 graph 确定性路由互补，共享 ToolRegistry）
├── mcp/              # MCP Server（FastMCP，四只读工具，惰性加载索引）
├── benchmark/        # 数据集适配、指标、消融框架、报告生成、轨迹分析
├── api/              # FastAPI 服务层（单线程执行器，线程安全）
└── ui/               # Web Demo 静态页（轨迹可视化）
```

**文档**

| 路径 | 内容 |
|---|---|
| [`docs/EVALUATION.md`](docs/EVALUATION.md) | 评测结论汇总、关键工程改动归因与诚实边界 |

---

## Roadmap

- [x] **Phase 10**：ratio 比率模板（短语锚定，FinQA test 占 ~17% 的题型，EM 6.0% → 7.0%）+ FinQA DSL 多步执行器（`#N` 引用 / table_sum / table_average）+ 可插拔 LLM 程序生成器（结构校验 + 失败降级，mock provider 单测锁定语义）；Phase 10.1 ratio 锚定改进（年份 token / thereafter 行 / 正文脚注扫描 / 分母 total 行兜底 / 列一致性 / 负值剪枝，理想检索口径 2/13 → 7/13）
- [x] **Phase 11**：检索召回修复——损耗归因（检索层 88% vs 端到端 56%，44 失败样本逐个跑图追踪：59% 是 exclude 拉黑驱逐 gold）+ 排除语义修正（不驱逐历史证据）+ 表格 chunk 豁免 + replanner 打转终止 + bge 索引端到端切换。doc recall **56% → 88.0%（+32 pp）**、EM **7.0% → 8.0%**
- [x] **Phase 12**：程序句式覆盖扩展（ratio 5 变体 + average 年区间模板 + ROI/growth 关键词，可检测子集 38 → 57 题、全测六族未检出 122 → 9）+ 12.2 跨文档污染守卫（CME/GPN 同构表召回污染根因，分母限分子同文档 + total 回退同文档优先）+ reranker 失败缓存与 `RERANKER_MODEL` 离线路径。同降级 reranker 口径 EM **13.0% → 21.0%**、可检测子集 EM **21.1% → 36.8%**
- [x] **Phase 12.3**：真实 LLM 程序生成接入——可插拔 provider（OpenAI 兼容直连 / 本地桥接双通道）+ `--llm-mode` 单变量消融。EM **21.0% → 29.0%**（净 +8 题零回归）；LLM planner 消融揭示下游兼容性问题，架构定为**规则 planner + LLM 程序生成**
- [x] **Phase 12.4**：**MCP Server**（FastMCP stdio 四只读工具，协议冒烟六步全通）+ LLM-as-judge 评测体系（faithfulness/relevancy 分层发现）
- [x] 真实语义 embedding（bge-small-en-v1.5）替换 hash-Dense（doc recall +5 pp）
- [x] 可插拔 LLM 规划层（任意 provider，失败降级规则 planner，零外部依赖兜底）
- [x] 多跳 QA 合成管线（种子抽取 + 四重校验，16 条基准 + 能力边界实证）
- [x] 实体键控程序模板（cross_entity_diff / argmax_relay），多跳基准 EM 37.5% → 100%
- [x] **Phase 13**：最小 tool-calling harness——`ToolCallingHarness` 循环 + `OpenAICompatLLM.chat`（messages+tools function calling 协议），17 项测试锁定循环语义（工具回喂 / 三停机 / 单例隔离），不动 graph 与 benchmark 一行
- [x] **Phase 13.1**：LLM 调用接 Structured Outputs——`response_format` 透传（OpenAI `json_schema` strict 形态，planner/programmer 各自 schema：`financial_plan` / `dsl_program`），端点不支持时自动摘除回退（400/422 即摘、不吃退避预算），`--json-schema` 消融开关；模型层保证输出结构 + 手工解析降级双层防线并存，8 项测试锁定透传/回退/接线语义。**消融实测**（FinQA test 100 样本，vs v125）：EM 31.0% vs 30.0%（噪声量级，零重试本来就稳），代价 +36% 延迟——json-schema 买到的是结构确定性而非准确率，详见 `docs/EVALUATION.md`
- [ ] 金融域专用 rerank（通用 ms-marco 对数值表格无益，需财务语料微调）
- [ ] TAT-QA 数据集评测（跨表推理）
- [ ] 重规划策略优化（多跳基准首答全对暂无触发；真实 FinQA 上有效率 50% → 目标 60%+）
- [ ] LLM 辅助实体链接（Phase 12.5b 实测为诚实的负向结果：内容实体候选过于稀疏，根因是候选来源应从 doc_id 前缀扩充）

---

## 贡献与开发

```bash
make install          # 安装依赖
make test             # 运行测试
ruff check src tests  # 静态检查（CI 同款）
```

提交前请确保 `pytest` 与 `ruff check` 均通过；CI 会在 push 与 PR 时自动运行（Python 3.10 / 3.11 / 3.12 矩阵）。

---

## License

[MIT](LICENSE)
