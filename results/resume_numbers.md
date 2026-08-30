# VeriFin — 简历数字（Phase 10 收官，2026-08-30）

> 本文件是从评测报告（results/synth_check4、results/finqa_v8、results/finqa_v10、
> results/convfinqa_check、results/multihop_check）提炼的面试/简历可引用数字与
> 诚实边界说明。所有数字均可回溯到对应报告文件。

## 一句话定位

LangGraph 多智能体金融证据校验 RAG 系统：BM25+Dense 混合检索（多查询合并 +
small-to-big 父文档补全）、四要素规则校验、冲突仲裁与重规划、会话偏好记忆、
PoT 安全计算器 + 程序模板执行器（五类模板）+ FinQA DSL 多步执行器与可插拔
LLM 程序生成（失败降级）；检索升级（语义 embedding 替换哈希向量）、可插拔
LLM 规划层（失败降级规则 planner）、多跳 QA 合成管线（四重校验）与
175 项测试全绿，完成真实 FinQA / ConvFinQA 评测并提供 FastAPI 服务层与 Web Demo。

## 可引用数字

1. **端到端准确率（合成评测套件，20 样本 × 5 配置消融，多轮）**：完整系统 **100.0%**，
   Faithfulness **100.0%** / Answer Relevancy **98.2%**（RAGAs 风格规则近似）
2. **消融贡献（合成套件，多轮口径）**：Verifier **+20 pp**（100% vs 80%）、
   Replanner **+15 pp**（100% vs 85%）、偏好记忆 **+15 pp**（多轮子集 100% vs 40%）
3. **真实 FinQA test（1,147 题，评测 100 样本）**：
   - 答案 EM **1.0% → 6.0% → 7.0%（7 倍，Phase 10）**；PoT 计算子集规模
     25 → **38 题**（ratio 模板 +13），模板可检测子集 EM 24% → 18.4%
     （子集扩容稀释，绝对正确数 6 → 7）
   - 文档级召回 recall@20：**48% → 56%**（多子任务检索合并 + top_k 25）
4. **真实 ConvFinQA dev 多轮（100 轮）**：文档级召回 **52% → 65%（+13 pp）**，
   多轮继承/偏好机制端到端跑通
5. **工程指标（真实数据，Phase 10）**：Tool F1 94.8%、平均轨迹步数 7.57、
   重规划有效率 50%（pseudo-gold 触发词补齐比率类后的新口径）
6. **检索升级（单查询直接检索口径，FinQA test 100 样本）**：语义 embedding
   bge-small 替换无语义 hash 向量，doc recall@20 **81.0% → 86.0%（+5.0 pp）**；
   cross-encoder rerank 在此口径下 **-1.0 pp**（通用精排对金融表格 doc 召回无益，
   如实报告）
7. **多跳合成基准（16 条：2-hop 14 / 3-hop 2）**：整体 EM **37.5% → 100%**
   （Phase 9.6 → 9.7）——跨实体差值 8 条、三实体接力 2 条此前 0%（"检索到
   证据 ≠ 算得出答案"的实证），新增实体键控程序模板后三类子集全对；
   Faithfulness 100%、Tool F1 100%。诚实边界：基准由本项目合成管线产出，
   问题形态与模板同源
8. **测试资产**：175 项全绿（含程序执行器五模板 / DSL 多步执行器 / LLM 程序
   生成降级语义 / 表格解析 / 检索合并 / LLM 规划层 / 多跳合成四重校验 /
   实体锚定与短语锚定扫描回归）

## Phase 9 关键工程改动（面试可展开）

1. **多子任务检索合并（multi-query merge）**：计算流基期/比较期检索此前每轮
   **覆盖**上一轮结果（最终只剩最后一个子任务的文档，原始问题的强词法信号被
   丢弃）→ 按 chunk_id 去重合并（上限 80）+ top_k 20→25。这是 LangChain
   MultiQueryRetriever 的同思路实现。
2. **表格解析修复**：无分隔行（`|---|`）的 markdown 表格此前 `rows[2:]` 会把
   **首条数据行当分隔行丢掉**（如 "2014 net revenue" 行），修复后水平/垂直
   表格的年份候选不再缺行。
3. **候选锚定体系（anchor score）**：表格行 metric 命中+年份在行标签（0.0）<
   行标签×问题**双向**词重叠（0.5）< 散文证据 metric 命中（1.0）< 新闻源
   惩罚（+0.5）< 距离兜底（50+）；父文档补全候选仅 +0.25 来源惩罚（不抹平
   语义层级）；跨 chunk 组合 +1.0 惩罚（双期数值几乎总在同表）。
4. **数值抽取净化**："Form 10-K" 的 10 此前被抽为数值 10.0，与真实值 12000
   构成**假 major conflict** 导致重规划螺旋耗尽——数字后紧跟 `-字母` 的
   SEC 报表编号（10-K/10-Q/S-1）已剔除。
5. **FinQA 方向惯例**："percentage DECREASE from 2009 to 2010" 的金标 program
   为 `subtract(2009值, 2010值), divide(#0, 2009值)`（降幅取正），与标准
   (target-base)/base 相反——检测到 decrease 措辞时翻转求值方向。

## Phase 9.5 检索升级（面试可展开）

1. **语义 embedding 替换哈希向量**：Phase 1 的 HashEmbedding 是 SHA256 词袋
   哈希投影（无语义，仅"词重叠越多越相似"），在真实财报上的 Dense 路为负贡献。
   接入 `bge-small-en-v1.5`（sentence-transformers，384 维），通过
   `--embedding bge` 构建 + manifest 记录 + 读取时自动按 manifest 装配，
   单查询直接检索 doc recall@20 **81.0% → 86.0%（+5.0 pp）**。
2. **可插拔 embedding 工厂**：`get_embedding_function("hash"|"bge")` 统一入口，
   hash 为离线零依赖保底（测试/CI/demo 开箱即用），bge 为显式语义升级；
   索引 manifest 持久化嵌入选择，跨进程读取无需手工指定。
3. **rerank 接回默认路径但如实降级**：CrossEncoder 精排此前"写了但未接入主流程"，
   现从 manifest/参数贯通；但实测通用 `ms-marco-MiniLM` 对金融表格 doc 召回
   **-1.0 pp**（通用段落精排对数值表格敏感性低），保留为可选并在报告如实标注。

## Phase 9.5 LLM 规划层（面试可展开）

1. **可插拔 provider 注入**：`LLMPlanner` 通过 `complete(prompt) -> str` callable
   接入任意 LLM（OpenAI / Anthropic / 本地模型），不绑定 SDK——调用方自行封装
   网络层，一行注入即可切换。
2. **失败降级（graceful fallback）**：LLM 未注入 / 调用异常 / 输出非法（缺
   sub_tasks、非 JSON、结构错）时 `plan()` 返回 `None`，planner 节点自动回退到
   确定性规则 planner，行为与无 LLM 基线完全一致（测试逐项断言相等）。
3. **合并策略**：规则输出为骨（偏好注入 / 计算检测 / 四要素抽取齐全），LLM 仅
   增强——sub_tasks 采用 LLM 分解、claims 规则优先 + LLM 补齐、calculation_spec
   规则优先 + LLM 仅补规则未检测到的计算；避免 LLM 幻觉污染确定性链路。
4. **边界诚实**：默认路径仍为无 LLM 规则引擎（评测数字零外部依赖可复现）；
   LLM 规划层为可选增强，未接入真实 provider 跑 FinQA 全量（无 API key 不
   虚报数字），以 11 项单测锁定降级与合并语义。

## Phase 9.7 实体键控计算流（面试可展开）

1. **两类新程序模板**：`cross_entity_diff`（"How much higher was A revenue
   than B revenue in 2024" → `subtract(A,B)`）与 `argmax_relay`（"Among A, B
   and C, which had the highest revenue in 2024, and what was its gross
   margin?" → `greater` 比较链 + 胜者第二指标 `lookup`）。检测顺序先具体后
   一般（argmax_relay → growth_pct → cross_entity_diff → difference），
   双年份归 growth、单年份比较级归 cross-entity，防止模板互抢。
2. **实体锚定数值扫描**：`collect_entity_values` 取「实体提及 + 指标同义词 +
   年份」三信号最近邻的数字；chunk 未提及目标实体/指标/年份时返回空，
   防跨指标串值（gross margin 问题不误收同实体 revenue 数值）。
3. **多跳语义的锚定差异**：growth_pct 双期数值几乎总在同表（同 chunk 偏好 +
   跨 chunk 惩罚），跨实体双数值天然位于不同 chunk（跨文档证据链）——
   **不施加**同 chunk 偏好，这是单实体与多跳检索结构差异的直接体现。
4. **pseudo-gold 触发词扩展**："higher/lower than" / "difference between"
   （含比较级与 than 分离形态）同样要求 calc 工具族，跨实体子集 Tool F1
   从误罚到 100%。

## Phase 10 比率模板 + LLM 程序生成（面试可展开）

1. **ratio 短语键控模板**："what percentage/percent/portion/fraction of X are
   Y"（FinQA test 占 ~17%）此前无模板可执行（静默落入提取路径必然错）——
   `collect_phrase_values` 以「行标签 × 问题短语实词**双向**词重叠」锚定
   分子/分母表格数值（全覆盖命中 = 0 分锚定；单 token 命中 +0.25 惩罚防
   "total"/"current" 泛化词假阳性），`divide(部分, 整体)` 双刻度输出
   （裸小数 + ×100，FinQA exe_ans 两种形态并存）。真实 FinQA 100 样本
   检出执行 13 题（此前 0）、新增 1 题正确、零回归。
2. **FinQA DSL 多步执行器**：`parse_dsl`（顶层逗号切分、括号深度感知、
   算子白名单 + 元数校验）+ `execute_dsl`（`#N` 步骤引用、`vN`/`tN` 候选
   绑定、`const_K` 常量；聚合算子展开为算术表达式后过 PoT 沙箱；任何非法
   返回 error 不抛出——LLM 输出不可信，失败即降级）。
3. **可插拔 LLM 程序生成器**：`LLMProgramGenerator` 经 `complete(prompt)->str`
   注入任意 provider；候选编号契约（单值 `vN` + 行标签上下文、值组 `tN`）
   使幻觉引用可机检（引用未编号候选直接拒绝）；输出先过 `validate_dsl`
   结构校验再逐步求值——LLM 输出**永不直接执行**。五类降级路径
   （未注入/异常/非 JSON/校验失败/空候选）均以 mock provider 单测锁定。
4. **规则优先、LLM 兜底的合并策略**：确定性模板检出优先执行；模板失败
   或未检出且 programmer 可用时才走 LLM 路径——与 Phase 9.5 LLMPlanner
   同构，避免 LLM 幻觉污染确定性链路。
5. **pseudo-gold 口径修正**：calc 触发词补齐 portion/fraction/percent of
   （比率题同样要求 calc 工具族）——Tool F1 95.6% → 94.8% 是**口径变化**
   而非退化（此前比率题被静默豁免 calc 要求）。

## 诚实边界（面试必答，已写入报告中）

- Agent 默认路径为**无 LLM 的确定性规则引擎**。FinQA test 答案类型分解：
  **提取型 4% / 推导型（需 program 算术）93% / 布尔型 3%**——确定性模板
  覆盖 growth_pct / difference / cross_entity_diff / argmax_relay / ratio
  五类（真实 FinQA 计算子集 38%），table_sum/exp_avg 及多步 add 链由
  Phase 10 LLM 程序生成路径覆盖（基础设施就绪）。
- **ratio 模板的真实精度**：100 样本检出执行 13 题对 1 题——失败主因是
  操作数锚定误差（短语命中的表格行与金标操作数不一致），属短语锚定的
  精度瓶颈，如实报告（EM 净增 +1 样本、零回归）。
- **LLM 路径未接真实 provider 全量评测**（无 API key 不虚报数字）：DSL
  执行器 + 校验 + 降级以 mock provider 单测锁定语义；默认路径（无 LLM）
  行为与 Phase 9 完全一致，评测数字零外部依赖可复现。
- hash-Dense embedding 无语义，已在 Phase 9.5 用 bge-small 替换（单查询 doc
  recall 81%→86%，+5 pp）；但通用 `ms-marco` cross-encoder rerank 对金融表格
  doc 召回无益（-1 pp）——通用段落精排与金融数值表格的语义分布不匹配，
  金融域专用 rerank 待做。
- 真实大语料上整体 EM 仍低：检索（doc recall 56%）与 program 覆盖是两大瓶颈，
  均已在报告中按口径分解如实披露。

## 数字回溯索引

| 数字 | 报告 |
|------|------|
| 合成 100% / 消融贡献 | [results/synth_check3/report.md](synth_check3/report.md) |
| FinQA EM 6.0%（Phase 9）/ det 24% / recall 56% | [results/finqa_v8/report.md](finqa_v8/report.md) |
| FinQA EM 7.0%（Phase 10）/ PoT 子集 38 题 | [results/finqa_v10/report.md](finqa_v10/report.md) |
| ConvFinQA recall 65% | [results/convfinqa_check/report.md](convfinqa_check/report.md) |
| 检索升级 recall 81%→86% | `scripts/compare_retrieval.py`（FinQA test 100 样本实测） |
| 多跳基准 EM 37.5%→100% | [results/multihop_check/report.md](multihop_check/report.md) |
