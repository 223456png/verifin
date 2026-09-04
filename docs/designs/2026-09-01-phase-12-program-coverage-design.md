# Phase 12 — VeriFin 程序覆盖（Program Coverage）设计

- 日期：2026-09-01
- 范围：phase（VeriFin 多阶段计划第 12 期）
- 前置：Phase 11 检索召回（doc recall 56% → 88.0%）

## Intent Contract

**问题**：FinQA 100 样本端到端 EM 仅 8%，92 个失败样本中 64 个 gold program 含
`divide`，其中 **39 个是 `detect_program` 未检出任何计算模板**（模式覆盖缺口，
非检索缺口——`gt_in_retrieved=True` 占 33/39）。检索已把 gold 文档带回来，
但问题句式不在 5 个确定性模板（argmax_relay / growth_pct / ratio /
cross_entity_diff / difference）的覆盖面内，走散文提取路径答非所问。

**意图**：扩展确定性程序模板的**句式覆盖**，把未检出的比率/均值/增长问句
路由进现有计算执行路径（复用 `_execute_ratio` 与年份锚定候选收集），
在不引入新 LLM 依赖的前提下提升端到端 EM。

**成功标准**：
1. 全测试集（1147 题）`detect_program` 对目标句式的未检出数下降 ≥60%
   （ratio_of_to 48→0、pct_of_to 22→≤4、average_years 17→≤4、
   hyphen_ratio 4→0、as_pct_of 11→≤2、roi_return 20→≤2）；
2. FinQA 100 样本端到端 EM 相对 v11 基线（8%）提升 ≥6 个百分点；
3. 既有 184 项测试全绿（无回归——新句式不得抢占既有模板的问题）。

## Verification Contract

- 单元测试：`detect_program` 对每个新句式 ≥2 正例 + ≥3 反例
  （"due to expire" 类假连接词不得触发 pct_of_to；"average" 不得劫持
  双年份 growth 问句）；`_execute_average` 数值正确性 + 单位一致性剪枝。
- 端到端：`make benchmark-finqa`（bge 索引，100 样本）对比 v11 results。
- 诊断脚本复核：失败样本中 kind=NONE 且含 divide 的数量从 39 显著下降。

## Governance Contract

- 风险等级：medium（检测顺序变化可能劫持既有模板流量）
- 门禁：单元测试 + 全量回归绿 → 自动；benchmark EM 回退 >2pp → 人工
- 回滚：`detect_program` 新句式全部在既有检测顺序之后追加，出问题可逐条禁用

## Scope

**做**：
1. ratio 句式扩展（5 种新表面形式，全部复用 `_execute_ratio`）；
2. average 新模板（kind="average"，年份区间均值，复用年份锚定候选路径）；
3. growth/difference 关键词扩展（roi / return on investment / rate of
   return / growth 单年份 → base=year-1）；
4. `_ratio_candidates` 短语内年份锚定（"ratio of X in 2012 to Y in 2013"）。

**不做**：
- 锚定精度问题（已检出但锚错值的 24 个样本——growth_pct 14 + ratio 10，
  属于 Phase 13 的锚定语义问题，与句式覆盖正交）；
- verifier_reject 根因（29 个样本，四要素对表格/比率的口径问题）；
- LLM 程序生成路径的召回（llm_programmer 已有，本期只扩确定性模板）；
- 复杂多步程序（"combined amount of X and Y then ratio"——常数除法、
  多跳 add+divide 链，确定性模板性价比低，留给 LLM 路径）。

## Decisions

- **D1 检测顺序**：新 ratio 句式插在既有 `_RATIO_RE`（what percentage of
  DEN is NUM）**之后**、cross_entity_diff 之前。既有句式优先，防止
  "what percentage of X are Y" 被 pct_of_to 的贪婪 "to" 匹配劫持。
- **D2 pct_of_to 假连接词排除**：` to ` 前是 due / compared / according /
  prior / relative / next 时视为短语内连接词而非分母引导词，不触发。
- **D3 ratio 句式双年份放行**：ratio_of_to / pct_of_to 不设年份 ≤1 门槛
  （"ratio of the purchase in december 2012 to the purchase in january
  2013" 的两个年份分别在分子/分母短语内）；growth_pct 检测在前且需
  growth 关键词，无关键词双年份 ratio 不会误入 growth。
- **D4 average 候选复用**：ProgramSpec 增加 `periods: List[str]`；年份
  锚定候选收集循环从 `(base_period, target_period)` 扩为
  `periods or (base, target)`——growth/difference 流量 periods 为空，行为不变。
- **D5 growth 单年份归 difference 模板**：`_DIFFERENCE_KEYWORDS` 追加
  "growth"、"grew"（substring 匹配，"growth rate in X for 2003" →
  base=2002, target=2003，与 FinQA `subtract(cur, prev), divide(#0, prev)`
  惯例一致）；growth_pct（双年份）检测在前，不冲突。
- **D6 ROI 归 growth_pct**：`_GROWTH_KEYWORDS` 追加 "roi"、
  "return on investment"、"rate of return"。gold program
  `subtract(v_end, 100), divide(#0, 100)` 中 const_100 是表格里的指数
  基期值，年份锚定扫描天然覆盖，无需特判。
- **D7 hyphen_ratio 词法映射**："debt-to-asset ratio" → numerator="debt"、
  denominator="asset"（连字符两侧各取一词，复数归一在 `_match_metric`
  的同义词层处理不了时靠行标签词重叠兜底）。

## Surface

- `verifin/tools/program_executor.py`：新正则 ×5 + `_execute_average` +
  `ProgramSpec.periods` + `detect_program` 分派 + `execute()` average 分支；
- `verifin/core/nodes.py`：`detect_calculation_need` 序列化 periods；
  `_calculator_impl` 重建 periods；年份锚定扫描循环扩展；
  `_ratio_candidates` 短语内年份提取；
- `tests/test_program_executor.py` / `tests/test_calculation.py`：新增用例。

## Risks & Open Questions

- pct_of_to 在 "due to / related to" 类问句的误触发（D2 排除清单有限，
  靠 sanity 剪枝 |fraction|≤20 + 非负约束兜底）；
- average 的 candidates 桶按 period 键控，表格列年份与行标签年份不一致
  的表可能锚空（沿用 growth 流的既有风险，不新增处理）；
- EM 提升幅度依赖锚定质量：句式检出只是把问题送进计算流，锚错值的
  样本仍会失败（Phase 13 的范畴）。
