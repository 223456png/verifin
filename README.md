# VeriFin — 金融证据校验 Agent

> 让金融 Agent 不只「检索相关文本」，而是验证证据在**实体、时间、指标、口径**上是否真正匹配；不匹配则自动诊断、重规划（Replan）并重试。
>
> A financial evidence-verification agent built on LangGraph: retrieval is just the input — the real output is a **verifiable, traceable Claim-Evidence binding**, with automatic replanning when evidence fails four-factor verification.

[![tests](https://img.shields.io/badge/tests-162%20passed-brightgreen)]()
[![python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![license](https://img.shields.io/badge/license-MIT-lightgrey)]()

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

- **LangGraph 状态机编排**：planner → retriever → verifier → calculator → replanner → reporter 六节点循环，checkpoint 持久化多轮会话；领域逻辑（校验/计算/仲裁）全部为无状态 Tool，与框架解耦、可独立单测
- **混合检索 + 多查询合并**：BM25 + Dense 双路召回（可插拔 embedding：离线 hash / 语义 bge-small），多子任务检索结果按 chunk_id 去重合并（保留原始问题的强词法信号）；small-to-big 父文档补全，解决表格数值块 BM25 信号弱的问题
- **四要素校验器**：实体词边界匹配、期间年份交集、指标词典同义词归一、口径 token 重叠率——全部规则化、零 LLM 依赖、确定性可测
- **冲突仲裁与重规划**：≤2% 数值差异融合均值，>5% 强制 Replan；来源信任度加权（年报申报 > 新闻稿 > 研报）；Replan 按失败原因（缺年份 / 缺指标 / 缺实体）重构查询
- **PoT 安全计算器 + 程序模板执行器**：`calc_expression` AST 白名单沙箱求值（禁任意代码 / 禁 IO）；FinQA 推导型问题（占 93%）经模板检测 → 候选数值枚举 → 单位归一 → 锚分剪枝 → 确定性求值；四类模板——年份键控（growth_pct / difference）与实体键控（cross_entity_diff 跨实体差值 / argmax_relay 三实体比较接力），后者经实体锚定扫描（实体提及 + 指标同义词 + 年份三信号最近邻）取证，跨文档证据链不施加同 chunk 偏好
- **会话偏好记忆**：多轮对话中抽取实体/期间/指标/来源偏好并作用于后续检索与仲裁（ConvFinQA 多轮继承）
- **可插拔 LLM 规划层（可降级）**：`LLMPlanner` 经 `complete(prompt)->str` 注入任意 LLM provider（OpenAI / Anthropic / 本地模型），把金融问题分解为检索子任务 + 四要素 + 计算规格；LLM 不可用 / 异常 / 输出非法时自动降级到确定性规则 planner（零外部依赖兜底，测试与 demo 开箱即用）
- **多跳 QA 合成管线**：从带来源标识的原子 chunk 抽取种子事实 → 跨文档配对（增长链 / 跨实体差值 / 三实体接力）→ 按推理深度合并 → **四重校验**（语义 / 推理 / 来源跨度 / 反伪多跳），确定性合成 2-hop/3-hop 评测基准——反伪多跳用数值级比较（年份 "2023" 不误含 "20"），实测拒掉与单文档现成数值重合的差值题
- **RAGAs 风格生成质量指标（规则近似）**：Faithfulness（答案数值声明被「过校验证据 ∪ PoT 计算产物」支持的比例）与 Answer Relevancy（问题内容词覆盖率）——确定性、可复现、可进 CI，与 LLM-as-judge 的差异如实声明
- **全链路评测体系**：合成消融套件（5 配置 × 20 样本）+ 真实 FinQA/ConvFinQA 端到端评测，报告含答案类型分解与诚实边界声明
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
                                        ▼
                         ┌──────────────────────────────┐
                         │  FastAPI (/ask /health)      │
                         │  + Web Demo（轨迹可视化）     │
                         └──────────────────────────────┘
```

**关键设计决策**

| 决策 | 理由 |
|---|---|
| 领域逻辑与 LangGraph 解耦 | Verifier/Calculator/仲裁器均为纯函数 Tool，框架可替换，单测无图依赖 |
| 无 LLM 的确定性规则引擎 | 校验/计算/仲裁全程可解释、可回放、结果确定——评测数字可信的前提 |
| 单线程执行器承载 Agent 执行 | chromadb SQLite 句柄线程亲和，ASGI 工作线程直接调用会抛异常 |
| anchor score 锚定体系 | 表格行 metric 命中（0.0）< 行标签词重叠（0.5）< 散文证据（1.0）< 距离兜底（50+），跨层级候选可比较 |

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

| 指标 | Phase 8 | Phase 9 | 提升 |
|---|---|---|---|
| 答案 EM | 1.0% | **6.0%** | **6×** |
| 模板可检测子集 EM | — | **24.0%** | — |
| 文档级召回 recall@20 | 48% | **56%** | +8 pp |

**答案类型分解**（诚实口径）：提取型 4% / 推导型 93%（需 program 算术）/ 布尔型 3%。规则模板覆盖 growth_pct / difference 两类（检测子集 25%），table_sum / exp_avg / 多步 add 链属 LLM program 生成范畴（见 Roadmap）。

### 真实 ConvFinQA dev 多轮（100 轮）

| 指标 | Phase 8 | Phase 9 |
|---|---|---|
| 文档级召回 | 52% | **65%** |

### Agent 行为指标

| 指标 | 实测 | 目标 |
|---|---|---|
| 工具调用精准率 Tool F1 | **95.6%** | ≥ 85% ✓ |
| 重规划有效率 | 45.7% | ≥ 60% |
| 平均轨迹步数 | 7.65 | ≤ 4 |

### 检索升级（单查询直接检索口径，100 样本）

| 配置 | doc recall@20 |
|---|---|
| hash + hybrid | 81.0% |
| **bge + hybrid** | **86.0%（+5.0 pp）** |
| bge + hybrid + rerank | 85.0%（-1.0 pp，通用精排对金融表格无益） |

> 语义 embedding（bge-small-en-v1.5）替换无语义哈希向量带来明确提升；通用
> cross-encoder rerank 在此口径无益，属诚实边界。复现：`scripts/compare_retrieval.py`。

完整报告见 `results/finqa_v8/report.md`、`results/convfinqa_check/report.md`、`results/synth_check4/report.md`、`results/multihop_check/report.md`（含错误分析与诚实声明）。

---

## 快速开始

### 安装

```bash
git clone https://github.com/<your-username>/verifin.git
cd verifin
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### 方式一：零数据演示（内置合成语料）

```bash
# 启动 API + Web Demo（http://127.0.0.1:8000）
python -m verifin.api.app
```

打开浏览器访问 `http://127.0.0.1:8000`，直接提问，可视化查看 Agent 轨迹、证据卡片与计算过程。

### 方式二：真实 FinQA 评测

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
python -m pytest tests/ -q        # 162 项全绿
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
├── benchmark/        # 数据集适配、指标、消融框架、报告生成、轨迹分析
├── api/              # FastAPI 服务层（单线程执行器，线程安全）
└── ui/               # Web Demo 静态页（轨迹可视化）
```

**设计文档**（`docs/designs/`）：9 个 Phase 的迭代设计记录，每个 Phase 含问题定义、方案、验收标准——完整呈现从 ingestion 到 FinQA 集成的演进路径。

---

## Roadmap

- [ ] **Phase 10**：LLM program 生成（table_sum / exp_avg / 多步 add 链），覆盖 FinQA 剩余 ~75% 推导型问题
- [x] 真实语义 embedding（bge-small-en-v1.5）替换 hash-Dense（doc recall +5 pp）
- [x] 可插拔 LLM 规划层（任意 provider，失败降级规则 planner，零外部依赖兜底）
- [x] 多跳 QA 合成管线（种子抽取 + 四重校验，16 条基准 + 能力边界实证）
- [x] 实体键控程序模板（cross_entity_diff / argmax_relay），多跳基准 EM 37.5% → 100%
- [ ] 金融域专用 rerank（通用 ms-marco 对数值表格无益，需财务语料微调）
- [ ] TAT-QA 数据集评测（跨表推理）
- [ ] 重规划策略优化（多跳基准首答全对暂无触发；真实 FinQA 上有效率 46% → 目标 60%+）

---

## License

MIT
