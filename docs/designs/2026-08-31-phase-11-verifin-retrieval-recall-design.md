---
design_type: phase
created_at: 2026-08-31
---

# Phase 11 — 检索召回修复：exclude 误杀治理与 bge 索引切换 设计文档

日期：2026-08-31
范围：端到端 doc recall 56% → 75%+（FinQA test 100 样本）；exclude 拉黑策略软化、端到端基准切换 bge 索引、replanner 原地打转治理

## 1. Intent Contract

- **intent**: 消除端到端 doc recall 与检索层能力之间 32pp 的状态管理损耗，
  把 FinQA test 100 样本的 doc recall 从 56% 提升到 75%+（保守口径 70%+），
  EM 不回归且预期上升。
- **constraints**:
  - 检索层（hybrid_retriever.py / fusion.py / 索引构建）**不改**——诊断证明
    检索层本身 88%（bge 子任务合并口径），损耗不在检索；
  - 默认无 LLM 路径行为可复现（评测数字零外部依赖）；
  - 全量回归零破坏（181 项测试全绿）；
  - exclude 的原始意图（避免重复检索失败方向）保留——只豁免表格 chunk，
    不整体废弃排除机制。
- **success_criteria**:
  - FinQA test 100 样本复跑：doc recall ≥ 70%（目标 75%+）、EM ≥ 7.0%、
    正确题目集合不减（lost=[]）；
  - LOSS-BY-EXCLUDE 样本（26 个）复跑后 ≥ 80% 转为 gold 命中；
  - 181 项回归 + Phase 11 新增单测全绿。
- **risk_level**: medium（改动 replanner 排除策略影响重规划循环行为；
  有 trace_gold_loss.py 逐样本追踪兜底）

## 2. 验证契约的数据基础（Phase 11 前置诊断）

| 口径（FinQA test 100 样本，top_k=25） | doc recall |
|---|---|
| 端到端终态（v10_1 报告值，hash 索引） | 56% |
| 检索层模拟：hash 单查询 | 83% |
| 检索层模拟：bge 单查询 | 86% |
| 检索层模拟：hash 子任务合并 | 85% |
| 检索层模拟：bge 子任务合并（±rerank） | 88% |

44 个端到端失败样本逐个跑图追踪（scripts/trace_gold_loss.py）：

| 损耗形态 | 样本数 | 占比 | 机制 |
|---|---|---|---|
| LOSS-BY-EXCLUDE | 26 | 59% | 四要素校验失败 → chunk 永久拉黑 → 重规划重检索到 gold chunk 又被过滤 → 终态丢失（单样本累积拉黑 25–58 chunk） |
| LOSS-BY-RETRIEVAL | 12 | 27% | 短查询无实体/无年份（"what is the debt-to-asset ratio?"）、replanner 子任务跑偏 |
| 复跑已命中 | 6 | 14% | rerank 模型加载状态差异（v10_1 跑时模型缺失降级） |

根因：**四要素校验对表格 chunk 天然过严**（表格行标签没有实体/期间散文文本），
校验失败被误判为"chunk 无法回答"而永久排除——表格密集语料的系统性误杀。
外部 AI 建议的"表格感知"方向正确，但落点应在**校验豁免**而非 query 改写。

## 3. Verification Contract

```
verify_steps:
  - 单测: pytest tests/ -q → 181+ 项全绿（含 Phase 11 新增）
  - 追踪回归: python scripts/trace_gold_loss.py --idx <26 个 exclude 样本>
    → LOSS-BY-EXCLUDE 转为 GOLD-RETRIEVED ≥ 80%
  - 检索层不变: python scripts/diagnose_recall.py → 六组口径与诊断基线一致
    （±1pp，证明检索层未被波及）
  - 端到端: run_benchmark.py --dataset finqa --max-samples 100
    --index-dir ./bge_env --output ./results/finqa_v11
    → doc recall ≥ 70%、EM ≥ 7.0%、正确题集合不减
```

## 4. Governance Contract

```
approval_gates:
  - G1: exclude 软化策略选定（表格 chunk 豁免 vs 全量降权）——本设计已定表格豁免
  - G2: 端到端复跑前后正确题目集合对比（lost 必须为空）
rollback:
  - exclude 豁免为独立小改（replanner 累积处一处过滤），revert 即回 Phase 10 行为
  - bge 索引切换仅为运行参数（--index-dir），不改代码
ownership: 本会话 agent 负责，用户验收
```

## 5. Scope

| In | Out |
|---|---|
| exclude 拉黑的表格 chunk 豁免（replanner_node 累积处过滤） | 检索层任何改动（hybrid/fusion/rerank/索引构建） |
| 端到端基准切换 bge 索引（Makefile benchmark 目标 + 文档） | 表格感知 query 改写（诊断证伪优先级） |
| replanner 重复子任务检测终止（原地打转治理） | 相邻块滑窗召回（small-to-big 已覆盖，doc 级召回无增益） |
| 诊断脚本固化（diagnose_recall.py / trace_gold_loss.py 入库） | RRF k 调参（rerank 无损已间接证伪） |
| FinQA 复跑 + 文档更新 | LLM 路径（保持默认关闭） |

## 6. Decisions

| # | 决策 | 选择 | 拒绝的替代方案 |
|---|---|---|---|
| D1 | exclude 软化方式 | **表格 chunk 豁免**：replanner 累积 exclude_chunk_ids 时，过滤掉"含 markdown 表格行"的 chunk（经 metadata store 查 content 判定）——表格 chunk 四要素失败是结构性的（无散文四要素文本），语义上不该排除 | ①全量降权不排除：改变 exclude 语义面太大，重规划失去探索新方向能力；②doc 级保护（同 doc 永不全排除）：粒度太粗，真无关 doc 无法排除 |
| D2 | 端到端索引 | **bge_env**（已构建，manifest 记录 embedding=bge）+ Makefile 新增 benchmark-finqa 目标固化 | 在根目录重建 bge 索引（覆盖 hash 保底索引，破坏离线零依赖测试路径） |
| D3 | replanner 打转治理 | 新子任务与既有 sub_tasks 文本相同 → 视为"无新信号"终止重规划（走 synthesizer 给部分答案） | 允许重复但限流（治标不治本，浪费轮次且失败记忆污染） |
| D4 | 诊断脚本 | 固化入库 scripts/（可复现归因，面试可展示） | 一次性脚本丢弃（Phase 11 结论无法回溯） |
| D5 | LOSS-BY-RETRIEVAL 的 12 样本 | 不做专项处理——bge 切换的语义召回预计自然挽回部分，剩余属查询理解瓶颈（无实体短查询），Phase 12 范畴 | 指标词词典扩展/查询改写（诊断显示收益上限 12pp 且改写有丢词法信号风险，优先级低于 exclude 修复） |

## 7. Surface

- **src/verifin/core/nodes.py**（replanner_node）：exclude 累积处新增表格 chunk
  豁免过滤；新子任务重复检测终止。touches 约 30 行。
- **Makefile**：新增 benchmark-finqa 目标（--index-dir ./bge_env）。
- **scripts/diagnose_recall.py、scripts/trace_gold_loss.py**：已存在的前置诊断
  脚本纳入版本管理（本次设计过程中产出）。
- **tests/**：新增 exclude 豁免语义单测（表格 chunk 不排除 / 散文 chunk 照旧 /
  重复子任务终止）约 3 项。
- **results/finqa_v11/**、README、resume_numbers.md：复跑结果与文档更新。

## 8. Risks & Open Questions

- **R1（中）**：exclude 豁免后 replanner 可能重复检索同一批表格 chunk 浪费轮次
  （轨迹步数上升）——D3 的重复终止可对冲；复跑时观测 avg_steps 如实报告。
- **R2（低）**：bge 索引查询需 sentence-transformers 模型权重（沙箱环境重置
  会丢缓存，需 HF_ENDPOINT 镜像重拉）——Makefile 目标注释写明环境要求；
  hash 索引路径保留为离线保底。
- **R3（低）**：26 个 exclude 样本中部分可能同时存在真检索问题（豁免后仍不
  命中）——success_criteria 已放宽到 ≥80% 转化而非全量。
- **Q1（开放）**：四要素校验本身是否要给表格 chunk 加"结构性豁免"标记
  （verdict 不再是 FAIL 而是表格默认部分通过）？影响 faithfulness 口径，
  本 phase 不动，留 Phase 12。

## 9. 实施附录（2026-09-01 复盘）

D1 的表格豁免落地后追踪回归仅 10/25 转化——逐样本诊断发现被拉黑的 gold
chunk 一半是**散文 chunk**（四要素因 metric/period 错配失败，非表格结构
问题），表格豁免覆盖不了。更深层的机制问题在 retriever_node：排除过滤
作用于 merged（含历史轮已入池的 previous）——即排除会从累积状态**驱逐**
已检索到的 gold chunk，而 verifier 对每个子任务按各自 claim 重新校验全部
docs，chunk 在子任务 A 失败不代表在子任务 B 下无价值。

**D1 演进**：排除语义修正（只过滤本轮新检索结果、不驱逐历史证据）成为
主修复——追加后 25 样本追踪 **23/25 转化（92%）**。D1 表格豁免保留
（避免表格 chunk 进入排除列表浪费轮次），D3 打转终止保留。

最终验收（results/finqa_v11）：doc recall **56% → 88.0%**（超 75% 目标）、
EM **7.0% → 8.0%**（净 +2/-1：43/72 新对，73 由对转错——更大证据池下
ratio 锚定选了不同操作数，如实披露）、184 项测试全绿、检索层零改动
（git diff 正交性确认）。
