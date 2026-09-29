# VeriFin 评测总结与已知局限

> 本文汇总 VeriFin 各阶段的评测结论、关键工程改动的归因过程，以及**已知局限**。
> 所有数字均可回溯到 `results/` 下对应的报告文件或 `scripts/` 下的复现脚本。
> 最后更新：Phase 12.4（MCP Server + LLM-as-judge）。

---

## 一、系统定位

LangGraph 多智能体金融证据校验 RAG 系统：BM25 + Dense 混合检索（多查询合并 + small-to-big 父文档补全）、四要素规则校验（表格 chunk 豁免与排除语义修正）、冲突仲裁与重规划（带失败记忆与原地打转终止）、会话偏好记忆、PoT 安全计算器 + 程序模板执行器（五类模板）+ FinQA DSL 多步执行器与可插拔 LLM 程序生成（失败降级）；检索升级（语义 embedding 替换哈希向量）、可插拔 LLM 规划层（失败降级规则 planner）、多跳 QA 合成管线（四重校验）、MCP Server 能力暴露，`200` 项测试全绿，完成真实 FinQA / ConvFinQA 评测并提供 FastAPI 服务层与 Web Demo。

---

## 二、核心指标汇总

### 1. 端到端准确率（合成评测套件，20 样本 × 5 配置消融，多轮）

完整系统 **100.0%**，Faithfulness **100.0%** / Answer Relevancy **98.2%**（RAGAs 风格规则近似）。

### 2. 消融贡献（合成套件，多轮口径）

| 移除组件 | 准确率 | 贡献 |
|---|---|---|
| Verifier | 80.0% | **+20 pp** |
| Replanner | 85.0% | **+15 pp** |
| 偏好记忆 | 85.0%（多轮子集 40% vs 100%） | **+15 pp** |

### 3. 真实 FinQA test（1,147 题，评测 100 样本）

| 指标 | Phase 8 | Phase 9 | Phase 10 | Phase 11 | Phase 12 | Phase 12.3 |
|---|---|---|---|---|---|---|
| 答案 EM | 1.0% | 6.0% | 7.0% | 8.0% | 21.0% | **29.0%** |
| 模板可检测子集 EM | — | 24.0% | 18.4% | 21.1% | 36.8% | — |
| 程序可检测规模 | — | 25 题 | 38 题 | 38 题 | 57 题 | — |
| 文档级召回 | 48% | 56% | 56% | **88.0%** | 88.0% | 88.0% |

**文档召回 48% → 88.0%（+32 pp，Phase 11）**：损耗归因（44 失败样本逐个跑图追踪）发现 59% 是「四要素校验失败 chunk 被永久拉黑后从证据池驱逐」的状态管理损耗，而非检索能力不足（检索层本身 88%，子任务合并口径）。修复 = 排除只防新检索重复命中、不驱逐历史证据 + 表格 chunk 豁免 + replanner 原地打转终止 + 端到端切 bge 索引。

### 4. 真实 ConvFinQA dev 多轮（100 轮）

文档级召回 **52% → 65%（+13 pp）**，多轮继承/偏好机制端到端跑通。

### 5. 多跳合成基准（16 条：2-hop 14 / 3-hop 2）

整体 EM **37.5% → 100%**（Phase 9.6 → 9.7）——跨实体差值 8 条、三实体接力 2 条此前 0%（「检索到证据 ≠ 算得出答案」的实证），新增实体键控程序模板后三类子集全对；Faithfulness 100%、Tool F1 100%。

### 6. 检索升级（单查询直接检索口径，FinQA test 100 样本）

语义 embedding `bge-small` 替换无语义 hash 向量，doc recall@20 **81.0% → 86.0%（+5.0 pp）**；cross-encoder rerank 在此口径下 **-1.0 pp**，如实报告。

### 7. 工程与行为指标

| 指标 | Phase 9 | Phase 10 | Phase 11 |
|---|---|---|---|
| Tool F1 | 95.6% | 94.8% | 95.0% |
| 平均轨迹步数 | 7.65 | 7.57 | 6.67 |
| 重规划有效率 | 45.7% | 50.0% | 3% |

> Phase 11 重规划有效率降至 3% 是**打转终止策略**的直接结果：失败样本提前退出而非继续无效重规划，指标口径随之变化。

### 8. 测试资产

`200` 项全绿，覆盖程序执行器五模板 / DSL 多步执行器 / LLM 程序生成降级语义 / 表格解析 / 检索合并 / LLM 规划层 / 多跳合成四重校验 / 实体锚定与短语锚定扫描回归 / Phase 10.1 年份 token 与 thereafter 锚定 / Phase 11 表格豁免与打转终止语义 / Phase 12 跨文档污染守卫 / **PoT 沙箱反向拒绝用例**。

---

## 三、关键工程改动归因

以下每一条都是「发现问题 → 定位根因 → 修复 → 验证」的完整闭环，供技术评审参考。

### Phase 9：检索与锚定基础

1. **多子任务检索合并（multi-query merge）**：计算流基期/比较期检索此前每轮**覆盖**上一轮结果（最终只剩最后一个子任务的文档，原始问题的强词法信号被丢弃）→ 按 chunk_id 去重合并（上限 80）+ top_k 20→25。这是 LangChain MultiQueryRetriever 的同思路实现。
2. **表格解析修复**：无分隔行（`|---|`）的 markdown 表格此前 `rows[2:]` 会把**首条数据行当分隔行丢掉**（如 "2014 net revenue" 行），修复后水平/垂直表格的年份候选不再缺行。
3. **候选锚定体系（anchor score）**：表格行 metric 命中 + 年份在行标签（0.0）< 行标签 × 问题**双向**词重叠（0.5）< 散文证据 metric 命中（1.0）< 新闻源惩罚（+0.5）< 距离兜底（50+）；父文档补全候选仅 +0.25 来源惩罚（不抹平语义层级）；跨 chunk 组合 +1.0 惩罚（双期数值几乎总在同表）。
4. **数值抽取净化**："Form 10-K" 的 10 此前被抽为数值 10.0，与真实值 12000 构成**假 major conflict** 导致重规划螺旋耗尽——数字后紧跟 `-字母` 的 SEC 报表编号（10-K/10-Q/S-1）已剔除。
5. **FinQA 方向惯例**："percentage DECREASE from 2009 to 2010" 的金标 program 为 `subtract(2009值, 2010值), divide(#0, 2009值)`（降幅取正），与标准 (target-base)/base 相反——检测到 decrease 措辞时翻转求值方向。

### Phase 9.5：检索升级

1. **语义 embedding 替换哈希向量**：Phase 1 的 HashEmbedding 是 SHA256 词袋哈希投影（无语义，仅"词重叠越多越相似"），在真实财报上的 Dense 路为负贡献。接入 `bge-small-en-v1.5`（sentence-transformers，384 维），通过 `--embedding bge` 构建 + manifest 记录 + 读取时自动按 manifest 装配，单查询直接检索 doc recall@20 **81.0% → 86.0%（+5.0 pp）**。
2. **可插拔 embedding 工厂**：`get_embedding_function("hash"|"bge")` 统一入口，hash 为离线零依赖保底（测试/CI/demo 开箱即用），bge 为显式语义升级；索引 manifest 持久化嵌入选择，跨进程读取无需手工指定。
3. **rerank 接回默认路径但如实降级**：CrossEncoder 精排此前"写了但未接入主流程"，现从 manifest/参数贯通；但实测通用 `ms-marco-MiniLM` 对金融表格 doc 召回 **-1.0 pp**，保留为可选并在报告如实标注。

### Phase 9.5：LLM 规划层

1. **可插拔 provider 注入**：`LLMPlanner` 通过 `complete(prompt) -> str` callable 接入任意 LLM，不绑定 SDK——调用方自行封装网络层，一行注入即可切换。
2. **失败降级（graceful fallback）**：LLM 未注入 / 调用异常 / 输出非法（缺 sub_tasks、非 JSON、结构错）时 `plan()` 返回 `None`，planner 节点自动回退到确定性规则 planner，行为与无 LLM 基线完全一致（测试逐项断言相等）。
3. **合并策略**：规则输出为骨（偏好注入 / 计算检测 / 四要素抽取齐全），LLM 仅增强——sub_tasks 采用 LLM 分解、claims 规则优先 + LLM 补齐、calculation_spec 规则优先 + LLM 仅补规则未检测到的计算；避免 LLM 幻觉污染确定性链路。

### Phase 9.7：实体键控计算流

1. **两类新程序模板**：`cross_entity_diff`（"How much higher was A revenue than B revenue in 2024" → `subtract(A,B)`）与 `argmax_relay`（"Among A, B and C, which had the highest revenue in 2024, and what was its gross margin?" → `greater` 比较链 + 胜者第二指标 `lookup`）。检测顺序先具体后一般，防止模板互抢。
2. **实体锚定数值扫描**：`collect_entity_values` 取「实体提及 + 指标同义词 + 年份」三信号最近邻的数字；chunk 未提及目标实体/指标/年份时返回空，防跨指标串值。
3. **多跳语义的锚定差异**：growth_pct 双期数值几乎总在同表（同 chunk 偏好 + 跨 chunk 惩罚），跨实体双数值天然位于不同 chunk——**不施加**同 chunk 偏好，这是单实体与多跳检索结构差异的直接体现。
4. **pseudo-gold 触发词扩展**："higher/lower than" / "difference between" 同样要求 calc 工具族，跨实体子集 Tool F1 从误罚到 100%。

### Phase 10：比率模板 + LLM 程序生成

1. **ratio 短语键控模板**："what percentage/percent/portion/fraction of X are Y"（FinQA test 占 ~17%）此前无模板可执行（静默落入提取路径必然错）——`collect_phrase_values` 以「行标签 × 问题短语实词**双向**词重叠」锚定分子/分母表格数值，`divide(部分, 整体)` 双刻度输出。
2. **FinQA DSL 多步执行器**：`parse_dsl`（顶层逗号切分、括号深度感知、算子白名单 + 元数校验）+ `execute_dsl`（`#N` 步骤引用、`vN`/`tN` 候选绑定、`const_K` 常量；聚合算子展开为算术表达式后过 PoT 沙箱；任何非法返回 error 不抛出）。
3. **可插拔 LLM 程序生成器**：`LLMProgramGenerator` 经 `complete(prompt)->str` 注入任意 provider；候选编号契约使幻觉引用可机检（引用未编号候选直接拒绝）；输出先过 `validate_dsl` 结构校验再逐步求值——**LLM 输出永不直接执行**。五类降级路径均以 mock provider 单测锁定。
4. **规则优先、LLM 兜底的合并策略**：确定性模板检出优先执行；模板失败或未检出且 programmer 可用时才走 LLM 路径。
5. **pseudo-gold 口径修正**：calc 触发词补齐 portion/fraction/percent of——Tool F1 95.6% → 94.8% 是**口径变化**而非退化。
6. **Phase 10.1 ratio 锚定改进（六项）**：① 问题年份 token 直接匹配表格年份行；② "after \<year\>" 映射 thereafter 行；③ 正文/脚注级数值扫描（脚注标记 "(1)" 剔除防误抽）；④ 分母 total 行兜底——**仅当分母短语扫描完全无候选时注入**；⑤ 列一致性排序；⑥ 负值剪枝。**理想检索口径** ratio 锚定精度 **2/13 → 7/13**。

### Phase 11：检索召回修复

1. **损耗归因先行（不盲改检索）**：外部建议全部指向检索层调优（query 改写/滑窗/RRF 调参），但归因诊断推翻了这个前提——检索层本身 doc recall@25 达 **88%**（`scripts/diagnose_recall.py` 六组对照），端到端终态却只有 56%。对 44 个失败样本**逐个跑图追踪**（`scripts/trace_gold_loss.py`）：59% 是 LOSS-BY-EXCLUDE、27% 是真检索失败、14% 复跑已命中。
2. **排除语义修正（核心修复）**：`exclude_chunk_ids` 原实现把失败 chunk 从 `retrieved_docs` 整体移除；但 verifier 对每个子任务按**各自 claim** 重新校验——chunk 在子任务 A 失败不代表在细化后的子任务 B 下无价值。修正为：排除**只过滤本轮新检索结果**，**不驱逐历史证据**。
3. **表格 chunk 豁免**：四要素校验对表格天然过严（行标签无散文实体/期间文本），`verify_claim_batch` 富化 `is_table` 标记，`get_failed_chunk_ids` 跳过表格条目。
4. **replanner 原地打转终止**：新子任务与既有 sub_tasks 文本相同视为无新信号 → 终止路径。
5. **端到端切 bge 索引**：Makefile `benchmark-finqa` 目标固化（hash 索引保留为离线保底路径）。
6. **验证闭环**：26 个 LOSS-BY-EXCLUDE 样本复跑 **23/25 转为 gold 命中（92%）**；检索层零改动（git diff 正交性）。

### Phase 12：程序覆盖扩展与跨文档污染守卫

1. **跨公司同构表污染**（真 bug）：CME 题召回 GPN 股权计划表，分母短语 "outstanding options" 命中异公司行标签 → 分母锚到 766801（正确为同表 total 行 1217121）→ `_ratio_candidates` 跨文档污染守卫（分母候选限分子最优锚定同文档，清空后 total 回退同文档优先）。
2. **reranker 失效伪影**：沙箱 HF SSL 阻断 → 每次检索 5×8s 重试卡 40s、评测拖至 34 分钟且检索质量降级 → 失败缓存（每进程只试一次）+ `RERANKER_MODEL` 本地路径离线加载。

### Phase 12.3：真实 LLM 程序生成接入

1. LLM 程序生成接管确定性模板未检出的 46 题，**净 +8 题 / 零回归**，46/46 调用全成功（DeepSeek 直连，均值 762ms）。
2. **LLM planner 消融（24.0%，+5/-2）**：规划 100/100 成功但端到端反低于 LLM program-only——**分解质量 ≠ 与四要素校验的下游兼容性**。据此把架构从「全 LLM」改为**规则 planner + LLM 程序生成**，这是数据驱动的架构决策，而非默认全 LLM。

### Phase 12.4：MCP Server + LLM-as-judge

1. **MCP Server**：FastMCP stdio 服务四工具只读暴露（retrieve / expand_document / verify_claim / calculate），协议级冒烟六步全通；任何 MCP 客户端可复用 VeriFin 检索 + 校验 + 计算能力。
2. **LLM-as-judge**（DeepSeek，40 样本子集，0 调用失败）：faithfulness + relevancy 逐样本判定，证据锚定四要素校验通过池。
3. **分层发现**：答错 27 题 faithfulness **0%**（噪声证据放行后算错值——verifier 精度弱点的直接实证）；答对 11 题 36%（judge 证据切片 6/32 偏窄）；judge 与 EM 一致性 82%。

### Phase 13 / 13.1：Tool-Calling Harness 与 Structured Outputs（能力层，不进评测口径）

1. **架构定位（诚实分层）**：`ToolCallingHarness` 是与固定状态机互补的**开放任务能力层**——LLM 按 OpenAI function calling 协议自主决定调什么工具、何时停（三停机条件：finished / max_turns / llm_error）。**它与 graph 的确定性路由不在同一评测口径**：EM 基线全部来自 graph 流水线（LLM planner 自由分解 24% < 规则混合 29% 的消融数据支撑确定性路由的选择），harness 不参与 EM 数字，因此本文件的指标表不含 Phase 13 行——这是口径隔离，不是漏测（harness 行为由 17 项单测锁定：调度、回喂、三停机）。
2. **Phase 13.1 Structured Outputs**：`response_format` 透传（OpenAI `json_schema` strict 形态，planner / programmer 各自 schema：`financial_plan` / `dsl_program`），端点不支持时自动摘除回退（400/422 即摘、不吃退避预算），`--json-schema` 消融开关。8 项测试锁定透传 / 回退 / 接线语义；**真实评测数字待跑**（需 LLM key，见第五节回溯索引说明）。

---

## 四、已知局限与诚实边界

以下几点是项目**主动披露**的能力边界，不应被解读为已完成的能力。

- **默认路径为无 LLM 的确定性规则引擎**。FinQA test 答案类型分解：**提取型 4% / 推导型（需 program 算术）93% / 布尔型 3%**——确定性模板覆盖 growth_pct / difference / cross_entity_diff / argmax_relay / ratio 五类（真实 FinQA 计算子集 38%），table_sum / exp_avg 及多步 add 链由 LLM 程序生成路径覆盖。
- **ratio 模板的真实精度仍然有限**：100 样本检出执行 13 题对 1 题（端到端）。Phase 10.1 六项锚定改进后，**理想检索口径**（剥离检索层）2/13 → 7/13。端到端仍对 1 题说明主要瓶颈已从锚定层转移到检索层；剩余 6 题理想检索失败属短语锚定精度上限（行标签与问题短语零词重叠 / 量纲与刻度错配）。
- **真实大语料上整体 EM 仍低**：Phase 11 后 doc recall 88%，**program 覆盖成为第一瓶颈**；检索残余 12% 失败中一半是无实体短查询（"what is the debt-to-asset ratio?"）的查询理解问题。
- **Phase 11 净账**：EM 7.0% → 8.0% 是净 +2/-1（43/72 新对、73 由对转错——更大证据池下 ratio 锚定选了不同操作数）；doc recall +32pp 的收益尚未被 program 覆盖兑现。
- **Phase 12 环境差声明**：21.0% 与本地 13.0%（Phase 12.1 降级口径）的差含环境差（沙箱重建 bge 索引与本地索引的 chunk 边界可能不一致），归因证据以 3/3 回归样本修复 + 单测锁定 + 同环境错误分布改善为准。
- **Phase 12.3 口径声明**：29.0% 与 21.0% 同口径可比（同索引同 reranker 降级）；LLM planner 变体曾以 z-ai 桥接跑过（95% 调用被限流拒绝，数据不可信已弃用），上表数字全部为 DeepSeek 直连完整跑出。
- **LLM-as-judge 为 40 样本子集口径**：全量 100 样本受沙箱限流中断；忠实性按"结论可由证据数值算术推导"判定（PoT 架构下的正确定义）。
- **通用 cross-encoder rerank 对金融表格 doc 召回无益**（-1 pp）——通用段落精排与金融数值表格的语义分布不匹配，金融域专用 rerank 待做。
- **多跳基准由本项目合成管线产出**，问题形态与模板同源，不能等同于真实场景的跨实体推理能力。
- **Phase 12.5b（LLM 辅助实体链接）为诚实的负向结果**：v126 实测 EM 29%（-1 vs v125），链接器仅激活 1/100——内容实体候选过于稀疏（公司名多在 doc_id 而非正文）。根因定性为信息论边界：问题本身不含消歧信息时实体链接无法提供增益。代码保留（缓存 + 校验 + 降级完备），未来接 doc_id 前缀候选可复用。

---

## 五、数字回溯索引

| 数字 | 来源 |
|---|---|
| 合成 100% / 消融贡献 | `results/synth_check3/report.md` |
| FinQA EM 6.0%（Phase 9）/ det 24% / recall 56% | `results/finqa_v8/report.md` |
| FinQA EM 7.0%（Phase 10）/ PoT 子集 38 题 | `results/finqa_v10/report.md` |
| ratio 理想检索 2/13→7/13（Phase 10.1） | `scripts/verify_ratio_ideal.py` + `results/finqa_v10_1/report.md` |
| FinQA EM 8.0% / doc recall 88.0%（Phase 11） | `results/finqa_v11/report.md` + `scripts/diagnose_recall.py` + `scripts/trace_gold_loss.py` |
| FinQA EM 21.0%（Phase 12.2）/ 六族未检出 122→9 | `results_v122_norank/report.md` + `scripts/repro_regressions.py` |
| FinQA EM 29.0%（Phase 12.3） | `scripts/summarize_llm_ablation.py` 输出 |
| ConvFinQA recall 65% | `results/convfinqa_check/report.md` |
| 检索升级 recall 81%→86% | `scripts/compare_retrieval.py`（FinQA test 100 样本实测） |
| 多跳基准 EM 37.5%→100% | `results/multihop_check/report.md` |

> `results/` 下的原始报告为运行期产物，未纳入版本控制（`.gitignore` 排除）；可用
> `make benchmark` / `make multihop` 等目标复现。
