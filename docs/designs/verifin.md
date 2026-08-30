---
design_type: initiative
created_at: 2026-08-29
---

# VeriFin — Financial Evidence Verification Agent

> 原项目代号 PEV（Financial Evidence Verification Agent），正式更名为 **VeriFin**。
> 一句话定位：让金融 Agent 不只「检索相关文本」，而是验证证据在**实体、时间、指标、口径**上是否真正匹配，不匹配则自动触发重规划（Replan）。

## Problem（要解决的问题）

金融 RAG 系统的检索环节普遍「语义相关」即「证据可用」，导致四类错误：

| 问题 | 具体表现 |
|---|---|
| 证据混淆 | 检索到 "Apple 2023 revenue" 与 "Apple 2024 revenue"，语义都相关，但用户问的是 2024 operating margin |
| 指标错配 | 拿 "gross margin" 当 "operating margin" 用 |
| 口径不一 | 不同财报对同一指标定义不同（adjusted vs GAAP） |
| 跨文档证据链断裂 | 多跳问题需要 A 文档的 revenue + B 文档的 cost，但只找到其中一个 |

## Vision（愿景）

构建一个以**证据校验**为中心的 Agent：检索只是输入，真正产出是**通过四要素校验、可溯源（带引用）的 Claim-Evidence 绑定**。校验失败时自动诊断原因、重构查询（Replan）并重试，而非直接输出错误答案。数值型问题（如「增长多少 %」）由 PoT 计算器执行**受限表达式求值**，让 Agent 不仅答「有没有」，还答「算没算对」。

VeriFin 的核心校验逻辑（Verifier/Replanner）设计为无状态、可独立单元测试的 Tool 集合；LangGraph 作为一等编排层负责状态持久化、回滚点设置与循环控制，两者通过标准 Tool 接口解耦，框架更换不影响领域逻辑。

## HOTL Contracts（战略级）

```yaml
intent: 构建一个分 7 个 Phase 交付的金融证据验证 Agent（ReAct 驱动），每个 Phase 都有可运行、可测试、可消融的产出
constraints:
  - 任何 Phase 不得破坏前序 Phase 的公开 API 与测试
  - 默认路径不依赖网络与外部服务（模型权重除外，且需可降级）
  - calc_expression 只允许受限沙箱求值（禁任意代码执行、禁 IO）
  - 每 Phase 附带测试/指标证明其增量价值
success_criteria:
  - 7 个 Phase 全部交付，FinQA/ConvFinQA/TAT-QA 至少一个数据集上有消融对比
  - Phase 7 产出 3 项 Agent 行为指标（重规划有效率 / 工具调用精准率 / 平均轨迹步数）与消融报告
  - Phase 1 为全部后续阶段的基座：ingestion + 双索引 + 元数据
risk_level: medium
```

## Non-goals

- 不做自有大模型训练 / 微调（仅用开源权重 + API 推理）
- 不做生产级多租户服务、鉴权与计费
- 不做 PDF OCR 与扫描件还原（数字格式 PDF 优先；OCR 列为远期扩展）
- 不追求 SOTA 学术指标，重在「验证机制」本身可消融、可解释

## Stakeholders

- **开发者本人**：面向 Agent 开发岗位的作品集项目
- **评审者（面试官）**：关注架构清晰度、可观测性、测试完备度、消融实验严谨性
- **数据来源**：Kaggle ICAIF-24 Finance RAG Challenge（FinQA / ConvFinQA / TAT-QA）

## Architecture（7 Phase 全景，ReAct 驱动）

```text
┌─────────────────────────────────────────────────────────────────────┐
│ Phase 1  Ingestion & Indexing (基座)                              │
│  PDF/文本解析 → 标题层级分块(结构感知) → BM25 + ChromaDB 双索引      │
│  + SQLite 元数据存储                                               │
├─────────────────────────────────────────────────────────────────────┤
│ Phase 2  Hybrid Retrieval + Rerank (多路召回)                     │
│  BM25 + Dense 融合 → BGE Rerank → 统一 SearchResult 输出          │
│  (语义分块作为实验性开关在此评估)                                   │
├─────────────────────────────────────────────────────────────────────┤
│ Phase 3  ReAct Agent Core (LangGraph 状态机)  ← 核心升级点        │
│  State → Tool Registry(检索/计算/校验) → Planner 生成初始          │
│  子任务队列 → 条件路由(ACCEPT/REJECT/REPLAN) → 执行中动态调整      │
├─────────────────────────────────────────────────────────────────────┤
│ Phase 4  Verifier Tools (四要素校验 + PoT 计算器)  ← PoT 回归     │
│  实体/时间/指标/口径 逐项校验 + 证据 Schema 抽取                    │
│  + 表达式安全执行(受限沙箱 + ast 语法校验)                          │
├─────────────────────────────────────────────────────────────────────┤
│ Phase 5  Replanner with Memory (带记忆的重规划)                    │
│  失败原因诊断 → 上下文注入(历史失败记录) → 查询改写 → 重试(≤3轮)   │
├─────────────────────────────────────────────────────────────────────┤
│ Phase 6  Synthesizer with Citation (证据链绑定)                   │
│  仅绑定 Verifier=ACCEPT 的证据 → 按引用溯源生成最终答案             │
├─────────────────────────────────────────────────────────────────────┤
│ Phase 7  Benchmark + 消融 + Agent 行为分析  ← 新增 Agent 指标     │
│  FinQA/ConvFinQA 准确率 + 重规划有效率 + 工具调用精准率 + 步数分布  │
└─────────────────────────────────────────────────────────────────────┘
```

**循环状态与路由**（贯穿 Phase 3–6 的统一 State）：

```python
state = {"messages": [...], "retrieved_docs": [...], "verify_flags": {},
         "retry_count": 0, "tool_call_history": [], "hooks": []}
```

条件路由：Verifier → ACCEPT 进 Synthesizer；REJECT 且 `retry_count < 3` → Replanner → Retriever（重试）；超出轮数 → 降级输出（附失败原因，不硬编答案）。

**Plan-Execute-Replan 闭环（动态，而非静态 DAG）**：Planner 节点首轮仅生成**初始**子任务队列；LangGraph 循环允许在执行中动态追加/调整后续步骤——例如 Verifier 发现口径不一致，Replanner 可在当前子任务中动态插入「口径对比」步骤后再继续。验收信号：同一多跳问题在不同上下文下，Agent 的执行路径长度可以不同（动态性证明）。

**可观测性：Verification Hooks**。在每个关键决策节点注入轻量校验钩子，写结构化事件进入 `state["hooks"]`，作为 Phase 7 轨迹分析的数据源：

- **Retrieve 后**：召回重叠率、年份/实体冲突快检
- **Calculate 后**：数值合理性检查（如 margin > 100% 报警）
- **Synthesize 前**：claim-证据绑定完整性（每个 claim 必须有 ACCEPT 证据）

**Tool Registry**（Phase 3 建立，Phase 4 增补）：

| Tool | 职责 | Phase |
|---|---|---|
| `retrieve(query)` | 多路召回 + rerank，返回 SearchResult | 2–3 |
| `verify_claim(claim, chunk)` | 四要素逐项校验，写 verify_flags | 4 |
| `calc_expression(expr)` | PoT：受限沙箱求值（RestrictedPython 首选 / ast 降级），返回 (value, unit) | 4 |
| `extract_evidence(chunk)` | 结构化证据抽取（Pydantic schema） | 4 |

Agent 根据问题类型自主决定是否调用 `calc_expression`（如「增长多少 %」类问题），调用轨迹记入 `tool_call_history`。

## Phase breakdown

| Phase | 名称 | 核心产出 | 验收信号 |
|---|---|---|---|
| 1 | Ingestion & Indexing | 解析/分块/加载 + BM25 + ChromaDB + SQLite 元数据 | 6+ 测试全绿；build_index.py 端到端跑通 |
| 2 | Hybrid Retrieval + Rerank | 双路融合(RRF) + BGE Rerank，统一 SearchResult | 召回率优于单路 BM25 |
| 3 | ReAct Agent Core | LangGraph StateGraph、Tool Registry、Planner 初始子任务队列、条件路由 | 多跳分解正确；同一问题在不同上下文下执行路径长度可不同（动态性证明） |
| 4 | Verifier Tools | 证据 Schema 抽取、四要素校验、PoT 计算器 | 指标/年份混用被拦截；(3910-2470)/2470 = 58.3% 算对 |
| 5 | Replanner with Memory | 失败诊断 + 历史上下文注入 + 查询改写 + ≤3 轮重试 | 首轮失败案例重规划成功 |
| 6 | Synthesizer with Citation | 仅绑定 ACCEPT 证据 + 引用溯源答案 | 答案中每句可点回证据 chunk |
| 7 | Benchmark + 消融 + Agent 行为分析 | 三数据集评估、组件消融、轨迹分析、错误分类 | 消融结论 + Agent 指标报告成文 |

**Phase 7 的 Agent 行为指标**（区别于传统 RAG 的 QA 准确率；目标值为初始目标，实测后校准，方法论在报告与代码注释中显式声明）：

| 指标 | 计算方法 | 初始目标 |
|---|---|---|
| 重规划有效率 (Replan Hit Rate) | 首轮校验失败 → 重规划后通过的比例 | ≥ 60% |
| 工具调用精准率 (Tool F1) | Agent 选择的工具与 pseudo-gold 工具标签的匹配度 | ≥ 85% |
| 平均轨迹步数 (Avg. Trajectory Steps) | 提问 → 最终答案经历的 Loop 轮数 | ≤ 4 轮 |

> Tool F1 的黄金标签：FinQA 等数据集不随附工具轨迹，Phase 7 以规则构造 pseudo-gold（如问题含 growth/percentage/change → 必须调 `calc_expression`），并在消融报告中声明该替代方法的局限。

## Target 目录结构（随 Phase 演进落地，每节点标注落地 Phase）

```text
verifin/
├── pyproject.toml
├── src/verifin/
│   ├── schemas.py              # P1  核心数据结构（DocumentChunk 等）
│   ├── embedding.py            # P1  可插拔 embedding 工厂（哈希默认 / BGE 可选）
│   ├── ingestion/              # P1
│   │   ├── parser.py           #      PDF/文本解析
│   │   ├── chunker.py          #      标题层级分块（512 token 上限）
│   │   └── loader.py           #      数据集加载（容错多格式）
│   ├── indexing/               # P1
│   │   ├── bm25_index.py       #      稀疏检索索引 + 持久化
│   │   ├── vector_index.py     #      ChromaDB 稠密向量索引
│   │   └── metadata_store.py   #      SQLite 元数据（检索后过滤/展示）
│   ├── core/                   # P3  LangGraph 编排（薄适配层）
│   │   ├── state.py            #      State 定义（messages/retrieved_docs/verify_flags/hooks...）
│   │   ├── graph.py            #      条件路由 ACCEPT/REJECT/REPLAN 与编译
│   │   ├── nodes.py            #      节点函数（planner/retriever/verifier/replanner/synthesizer）
│   │   └── runner.py           #      AgentRunner（invoke/astream 封装）
│   ├── tools/                  # P2-P5 领域 Tool（无状态、可单测，不 import LangGraph）
│   │   ├── retriever.py        # P2   多路召回 + rerank
│   │   ├── evidence.py         # P4   证据 schema + 结构化抽取
│   │   ├── verifier.py         # P4   四要素校验
│   │   ├── calculator.py       # P4   PoT 沙箱（RestrictedPython/ast）
│   │   └── replanner.py        # P5   带记忆重规划
│   ├── synthesizer/            # P6
│   │   └── synthesize.py       #      claim-证据绑定 + 引用溯源
│   └── hooks/                  # P3-P6 共享可观测钩子
│       └── event_logger.py     #      结构化事件写入 state["hooks"]
├── tests/
├── scripts/build_index.py      # P1  索引构建入口
├── data/                       # P1  样例数据（真实数据集落地位置）
└── benchmark/                  # P7
    ├── metrics.py              #      含 Tool F1 pseudo-gold 生成
    └── ablation.py             #      组件消融
```

## Risks

1. **依赖体积**：sentence-transformers 引入 torch（数 GB），CI/沙箱安装困难 → 作为可选依赖（`[bge]` extra），默认用本地确定性 embedding 跑通全流程。
2. **数据获取**：Kaggle 数据集需凭证，沙箱无法下载 → 内置合成样例数据 + 容错 loader，真实数据落地即可无缝接入。
3. **评测成本**：Phase 7 全量评估需要 GPU 或耐心 → 设计为可子集跑的评估脚本。
4. **API 漂移**：chromadb 0.5→0.6+ 接口变化 → 索引层封装在独立模块，升版本只需改一处。
5. **calc_expression 安全（高关注）**：求值层首选 RestrictedPython 沙箱（禁用 `__builtins__`、仅暴露 math 白名单函数、超时强制终止），`ast` 仅做语法树校验；无法引入时降级为纯 `ast` 白名单求值（仅字面量/四则/幂/括号/比较）。诚实边界：二者都不是形式化安全屏障，因此 calc Tool 本身不暴露任何 IO 能力（无文件/网络/进程面），并叠加 Calculate 钩子做数值合理性二次检查（如 margin>100% 报警）——三层防御。
6. **Tool F1 标签缺失**：黄金工具轨迹不可得 → pseudo-gold 规则生成 + 方法论显式声明，防止数字被过度解读。
7. **LangGraph API 迭代快**：0.2→0.3→1.0 的 StateGraph/Send 接口有变动 → 编排层独立成薄适配模块，领域 Tool 不 import LangGraph。