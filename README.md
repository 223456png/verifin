# VeriFin

金融问答的 RAG 有个老问题：检索到一段“看起来相关”的文字，就直接拿去当证据。可实际上年份可能对不上、指标可能不是同一个、口径可能一个是 GAAP、一个是 adjusted。模型不操心这些，照样算，最后给你一个看着挺像回事、其实站不住的答案。

VeriFin 想把“证据到底能不能用”这一步补上。每条证据都要过实体、时间、指标、口径四项检查，过不了就诊断原因、改写查询再试；涉及数字的地方不让模型心算，交给受限的程序执行器。整个循环用 LangGraph 编排，核心能力也包成了 MCP 服务，别的客户端可以直接调。

一句话：检索只是入口，我要的是一个能溯源、经得起核对的 Claim–Evidence 绑定。

## 为什么做这个

金融 RAG 最容易在四个地方翻车：

| 问题 | 长什么样 |
|---|---|
| 证据混淆 | 检索到 Apple 2023 和 2024 的营收，都“相关”，但用户问的是 2024 |
| 指标错配 | 把 gross margin 当成 operating margin 用 |
| 口径不一 | 同一指标在不同文档定义不同（adjusted vs GAAP），数值冲突没人裁 |
| 证据链断裂 | 多跳问题要 A 文档的 revenue 加 B 文档的 cost，只找到一个 |

这些错检索层发现不了，因为 chunk 确实“相关”。所以我把重心放在校验，而不是把召回率再往上卷。

## 它怎么跑

```text
用户 Query
    │
    ▼
┌──────────┐   子任务（查询变体 + 四要素槽位 + 计算规格）
│ Planner  │───────────────────────────────┐
└──────────┘                               ▼
                                    ┌───────────┐  BM25 + Dense 双路
       ┌───────────────────────────▶│ Retriever │  RRF 融合、去重合并
       │                            └───────────┘  small-to-big 补全
       │                                    │
       │  重构查询重试                        ▼
       │                            ┌───────────┐
┌──────┴───────┐   失败诊断          │ Verifier  │ 四要素核验 + 冲突仲裁
│  Replanner   │◀───────────────────│           │
└──────┬───────┘   REJECT           └───────────┘
       │  预算耗尽 → END 降级              │
       │                    ┌─────────────┼─────────────┐
       │                    ▼             ▼             ▼
       │              ┌───────────┐ ┌──────────┐ ┌───────────┐
       │              │ Calculator│ │Synthesizer│ │ 回环检索  │
       │              │ PoT 沙箱  │ │ 带引用答案 │ └───────────┘
       │              └─────┬─────┘ └────┬─────┘
       └────────────────────┴────────────┴──▶ END
```

六个节点：planner → retriever → verifier → calculator → replanner → synthesizer。verifier 之后按校验结果分流，失败就走 replanner 回环，重试预算用完就降级出答案。

## 几个关键设计

**校验用规则，不用 LLM。** 四要素全部是确定性规则：实体做词边界匹配（防止 Apple 命中 Applebee）、期间取年份交集、指标靠同义词词典归一、口径看定义描述的 token 重叠率。好处是同一个输入永远同一个结果，能进评测口径、能单测。代价是会漏检，所以我让规则只做保守拒绝，漏的交给 replanner 和冲突仲裁兜底。

**冲突按可信度裁，不投票。** 数值差在 2% 以内当口径噪声，融合均值；超过 5% 就按来源可信层级裁，年报（经审计）> 新闻稿 > 研报。三个研报说 110 亿，也压不过一个年报说 120 亿。只有最可信那一层自己都对不上时，才判 REJECT 去重规划。

**检索双路，融合只看排名。** BM25 抓精确数值和指标名，向量抓同义改写，两路都用。融合用 RRF 而不是分数加权，因为 BM25 的分数没有上界、cosine 有界，直接加权没意义。表格被切碎后单独一块词法信号弱，就用 small-to-big：检索用小块，返回时补全父文档。

**数字不让模型心算。** FinQA 里九成以上是推导题，必须算准，所以数值都走程序。程序在受限求值器里跑：AST 白名单只放行算术和 math 函数，`__builtins__` 置空，长度和指数有硬上限，`9**9**9` 这种直接拒。反向测试里 `open()`、`__import__`、`().__class__` 都是锁死的。

模板覆盖不到的多步算术，交给 LLM 生成程序。但 LLM 的输出永远不当命令执行，先过算子白名单、引用越界、幻觉候选三道检查，不通过就退回确定性模板，行为跟不用 LLM 时一致。

**领域逻辑和框架解耦。** 校验、计算、仲裁都写成无状态 Tool，统一注册。LangGraph 图、MCP 服务、tool-calling harness 三个入口共用同一套，框架换了逻辑不用动，单测也不用起图。

**对外只开只读能力。** MCP 服务暴露四个工具：检索、父文档补全、校验、计算。不开放写路径和 shell，既是能力复用，也是收敛攻击面。

**LLM 可插拔，也能降级。** planner 和程序生成都走 `complete(prompt) -> str` 接口，接 OpenAI 兼容模型都行。LLM 不可用或输出非法时，自动退回规则 planner / 确定性模板，所以 `make demo` 不配 key 也能跑。

## 快速开始

```bash
git clone https://github.com/223456png/verifin.git
cd verifin
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

零数据演示（内置合成语料，不需要 API key）：

```bash
make demo        # 启动 API + Web Demo，访问 http://127.0.0.1:8000
```

Docker 启动：

```bash
docker compose up --build
```

真实 FinQA 评测：

```bash
# 1. 数据：从 https://github.com/czyssrs/FinQA 下载 test 集，放成 data/finqa/test.json
#    （约 1,147 题；数据集许可不允许入库，需要自己准备）
#    注意：目录为空时 run_benchmark 会降级成内置合成语料并打警告，
#    这时的报告数字是合成口径，不代表真实 FinQA 结果。
python scripts/build_index.py --dataset finqa --data-dir ./data/finqa
python scripts/run_benchmark.py --dataset finqa --max-samples 100 --output ./results
python scripts/run_benchmark.py --dataset finqa --ablation --max-samples 50
```

想换成语义 embedding（bge-small，doc recall 会高一些）：

```bash
pip install -e ".[bge]"
python scripts/build_index.py --dataset finqa --data-dir ./data/finqa \
  --embedding bge --persist-dir ./bge_env
```

复现文章里 EM 29.0% 那一版（Phase 12.3，LLM 程序生成）还需要 LLM key 和对应开关：

```bash
python scripts/run_benchmark.py --dataset finqa --max-samples 100 --output ./results_llm_repro \
  --llm-api-key $DEEPSEEK_API_KEY --llm-mode programmer
```

规则模板基线（21.0%）和调用统计会落在同一个目录。没有 key 时跑出来的只是规则基线口径。

作为 MCP 服务用：

```bash
VERIFIN_INDEX_DIR=./bge_env python -m verifin.mcp.server
```

Claude Desktop / Cursor 的配置：

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

接口调用：

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What was the growth rate of net revenue from 2014 to 2015?"}'
```

跑测试：

```bash
python -m pytest tests/ -q     # 244 项，全离线
```

## 评测结果

所有数字都能在仓库里复现。下面这些表是逐阶段实测的，最后有一节说清楚哪些数字的边界在哪。

### 合成消融（20 样本 × 5 配置）

| 配置 | 准确率 | Faithfulness | Answer Relevancy | 贡献 |
|---|---|---|---|---|
| 完整系统 | 100.0% | 100.0% | 98.2% | — |
| 去掉 Verifier | 80.0% | — | — | +20 pp |
| 去掉 Replanner | 100.0% | — | — | +0 |
| 去掉偏好记忆 | 85.0% | — | — | +15 pp |
| 去掉 Dense 检索 | 100.0% | — | — | +0 |

去掉 Verifier 掉 20 个点，是这条链路上单点贡献最大的一层。

Replanner 这里是 +0，需要解释一下：合成集里的冲突基本都被上面的确定性仲裁消化掉了，20 个样本里重规划触发 0 次，所以它没有可贡献的空间。这不是 Replanner 没用——真实 FinQA 上重规划有效率是 50.0%（见下面的行为指标）。Dense 的 +0 同理，说明这两层在这个合成集上还没到瓶颈。

合成语料里我特意埋了 3 个冲突样本：新闻稿 110 亿美元、年报 120 亿美元，差 8.3%。仲裁后都收敛到年报的值，没触发重规划，这也是完整系统 100% 的来源之一。

Faithfulness 和 Answer Relevancy 是 RAGAs 同名指标的规则近似，不是用 RAGAs 库、也不是 LLM 打分：前者算“答案里的数值声明有多少能被通过校验的证据或计算结果支持”，所以跟校验闭环绑在一起。

### 多跳合成基准（16 条）

| 子集 | 条数 | Phase 9.6 | Phase 9.7 |
|---|---|---|---|
| 单实体跨年增长 | 6 | 100% | 100% |
| 跨实体差值 | 8 | 0% | 100% |
| 三实体接力 | 2 | 0% | 100% |
| 整体 | 16 | 37.5% | 100% |

Phase 9.6 的 37.5% 很说明问题：gold 文档全召回了，但跨实体算术没有模板可执行，所以算不出来。9.7 补上实体键控模板后三类子集全中。

要说清楚的是，这个基准是项目自己合成的，出题形态和模板同源，有自证成分。真实 FinQA 的跨实体、多步算术仍然偏弱，这也是为什么我没有把它当主要卖点。

### 真实 FinQA test（100 样本）

| 指标 | Phase 8 | Phase 9 | Phase 10 | Phase 11 | Phase 12 |
|---|---|---|---|---|---|
| 答案 EM | 1.0% | 6.0% | 7.0% | 8.0% | 21.0% |
| 模板可检测子集 EM | — | 24.0% | 18.4% | 21.1% | 36.8% |
| 可程序化题数 | — | 25 | 38 | 38 | 57 |
| 文档级召回 | 48% | 56% | 56% | 88.0% | 88.0% |

最终版（v125）EM 是 30%；纯 LLM 程序生成那一版实测 29%，两个数都有出处，对外统一说 30%。

中间有个转折值得说：端到端召回一度只有 56%，但检索层其实是 88%。逐样本追下来发现，损耗的 59% 来自“校验失败的 chunk 被永久拉黑、从证据池驱逐”——同一块证据在子任务 A 上失败，不代表在子任务 B 上没用。修掉这个驱逐逻辑之后，召回追平到 88%。

### 检索升级（单查询直接检索，100 样本）

| 配置 | doc recall@20 |
|---|---|
| hash + 混合检索 | 81.0% |
| bge + 混合检索 | 86.0%（+5 pp） |
| bge + 混合检索 + rerank | 85.0%（−1 pp） |

通用 cross-encoder rerank 在这个金融表格场景里掉了 1 个点，这个负结果我如实写在这里，金融域专用的 rerank 还在 roadmap 上。

### 生成质量与 Agent 行为

- 答案类型：提取型 4%、推导型 93%、布尔型 3%，所以数值计算是绕不开的。
- 工具调用精准率 Tool F1：94.8%（目标 ≥85%）。
- 重规划有效率：50.0%（目标 60%，没达标，如实写）。
- 平均轨迹步数：7.57（目标 ≤4，也偏多）。

更多逐阶段归因和已知问题在 [`docs/EVALUATION.md`](docs/EVALUATION.md)。

## 项目结构

```text
src/verifin/
├── ingestion/     # 文档加载、解析、分块（表格保结构）
├── indexing/      # BM25 + ChromaDB 向量索引 + SQLite 元数据
├── retrieval/     # 混合检索、多查询合并、small-to-big
├── core/          # LangGraph 图与六个节点 + 会话状态
├── tools/         # 四要素校验、冲突仲裁、PoT 计算器、程序模板
├── harness.py     # 最小 tool-calling 循环（与图互补，共用 ToolRegistry）
├── mcp/           # MCP Server（四个只读工具）
├── benchmark/     # 数据集适配、指标、消融、报告
├── api/           # FastAPI 服务层
└── ui/            # Web Demo
```

## Roadmap

已经做完的、还没做的都列在这里，省得看代码猜。

- [x] ratio 比率模板 + FinQA DSL 多步执行器 + 可插拔 LLM 程序生成
- [x] 检索召回修复（56% → 88%）
- [x] 程序句式扩展、跨文档污染守卫
- [x] 真实 LLM 程序生成接入（EM 21% → 29%）
- [x] MCP Server + LLM-as-judge 评测
- [x] 最小 tool-calling harness
- [ ] 金融域专用 rerank（通用模型对数值表格无益）
- [ ] TAT-QA 跨表推理评测
- [ ] 重规划策略优化（有效率 50% → 60%）
- [ ] LLM 辅助实体链接（试过一版，效果不好，原因写在 EVALUATION.md 里）

## 开发

```bash
make install
make test
ruff check src tests
```

提交前 `pytest` 和 `ruff check` 都要过，CI 会在 push 和 PR 时跑（Python 3.10 / 3.11 / 3.12）。

## License

MIT，见 [LICENSE](LICENSE)。
