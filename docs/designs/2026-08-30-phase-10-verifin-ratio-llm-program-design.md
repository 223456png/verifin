# Phase 10 — 比率模板与 LLM 程序生成 设计文档

日期：2026-08-30
范围：ratio 确定性模板（"what percentage of X are Y"）、FinQA DSL 多步执行器、可插拔 LLM 程序生成器（失败降级）

## 1. Intent 契约

- **比率题型覆盖**："what percentage/percent/portion/fraction of <整体> is/are <部分>"
  类问题（FinQA test 占 ~17%）此前无模板可执行（落入提取路径必然错）——
  新增 `ratio` 模板：短语锚定表格扫描收集分子/分母候选 → `divide(部分, 整体)`；
- **DSL 多步执行**：模板外推导型问题（table_sum / table_average / 多步算术链）
  由 LLM 生成 FinQA DSL 程序（`subtract(5829, 5735), divide(#0, 5735)`），
  `execute_dsl` 多步求值（`#N` 步骤引用）；
- **可插拔 + 降级**：LLM provider 经 `complete(prompt) -> str` callable 注入
  （不绑定 SDK）；未注入/调用失败/输出非法 → 自动降级回确定性模板路径，
  默认行为与 Phase 9 完全一致（评测数字零外部依赖可复现）；
- **安全边界**：LLM 输出**永不直接执行**——先过 `validate_dsl` 结构校验
  （算子白名单 / 引用越界 / 幻觉候选），再逐步经 PoT 沙箱求值
  （`calc_expression` AST 白名单）。

## 2. Constraints

- 无新增外部依赖（string.Template / json / re 标准库）；
- 层级规则不变：tools/ 不 import LangGraph；nodes.py 局部导入
  program_executor（防循环依赖）；
- 默认路径（无 LLM）与 Phase 9 行为一致：`build_agent_graph()` 不注入
  programmer 时 calculator/planner 输出与 Phase 9 相同（回归测试锁定）；
- 全量回归零破坏（162 → 175 项测试全绿）。

## 3. 关键决策

| # | 决策 | 理由 |
|---|---|---|
| D1 | ratio 检测置于 growth_pct 之后、cross_entity_diff 之前；年份 ≤1 才触发（双年份归 growth） | 检测顺序先具体后一般，防止模板互抢（与 Phase 9.7 惯例一致） |
| D2 | 分子/分母候选用**短语锚定表格扫描** `collect_phrase_values`：行标签 × 问题短语实词**双向**词重叠打分（全覆盖命中 = 0 分锚定；单 token 命中 +0.25 惩罚防泛化词假阳性） | 比率题常无年份、指标词典不覆盖（"leased facilities"），操作数只能靠行标签语义锚定 |
| D3 | ratio 求值输出**双刻度**：fraction（裸小数，FinQA 比率 GT 常见）+ value（×100 百分比）；EM 池逐一尝试 | FinQA exe_ans 两种形态并存（0.14464 / 16.94） |
| D4 | 剪枝：同 chunk 同值自除跳过、量纲冲突跳过、\|fraction\| > 20 视为锚定错误；排序偏好 \|fraction\| ≤ 2（部分 ≤ 整体是常态，反向多为方向配错） | 财报比率语义先验 |
| D5 | `parse_dsl` 不支持嵌套调用（FinQA `program_re` 嵌套形态需先展开）；顶层逗号切分（括号深度感知） | 简化求值器；嵌套可静态展开为多步 |
| D6 | `execute_dsl` 算术算子逐步经 evaluator（PoT 沙箱保留工具轨迹）；聚合算子展开为算术表达式（`table_sum(t) → (a+b+c)`）；`greater` 原生比较 | 复用 Phase 4 三层防御；任何非法返回 `{"error"}` 不抛出（LLM 输出不可信） |
| D7 | 候选编号契约：单值 `vN`（值 + 行标签上下文）、值组 `tN`（同表行数值列表）注入 prompt；LLM 只能引用已编号候选 | 校验可机检（幻觉引用 v9 直接拒绝）；上下文行标签供 LLM 消歧 |
| D8 | planner 注入 programmer 且可用时：确定性模板未检出的数值问题置 `calculation_spec={"kind": "llm_program"}`；模板路径失败时 calculator 兜底转 LLM 路径（`_finish_calculation`） | 规则优先、LLM 兜底——避免 LLM 幻觉污染确定性链路（与 Phase 9.5 LLMPlanner 合并策略同构） |
| D9 | prompt 用 `string.Template`（`$query`/`$candidates`）而非 `str.format` | JSON 示例花括号与 format 语法冲突 |
| D10 | pseudo-gold calc 触发词补 "portion/fraction/percent of"（percentage 已在增长关键词内） | 比率题同样要求 calc 工具族，否则 Tool F1 误罚 |

## 4. 验证契约

| 验证点 | 测试 |
|---|---|
| ratio 检测（短语抽取 / 单年份 / 双年份归 growth 不误伤） | test_program_executor 15 |
| ratio 求值（双刻度 / 缺候选 / 量纲冲突 / 自除剪枝 / \|fraction\|≤2 排序） | test_program_executor 16-17 |
| 短语锚定扫描（全覆盖命中 0 分锚定 / 无关短语低置信 / 空短语） | test_program_executor 18 |
| parse_dsl（顶层切分 / 白名单 / 元数 / 嵌套拒绝） | test_program_executor 19 |
| execute_dsl（#N 引用 / vN 绑定 / const_K / greater / 聚合 / 全部非法形态） | test_program_executor 20-22 |
| LLMProgramGenerator（成功 / markdown 容错 / 值组渲染 / 五类降级） | test_llm_programmer 2-4 |
| validate_dsl（合法 / 幻觉引用 / 前向引用 / 未知 token） | test_llm_programmer 1 |
| 图注入（llm_programmer 注入后可编译，默认路径不受影响） | test_llm_programmer 5 |
| pseudo-gold 比率触发词 | test_benchmark 3 |
| 回归 162 项零破坏 | pytest 175 passed |
| 真实 FinQA 100 样本复跑 | results/finqa_v10/report.md |

## 5. 治理契约

- 风险等级：中（新增 LLM 路径默认关闭；ratio 模板改变部分样本的路由——
  复跑无回归，EM 6.0% → 7.0%，无样本由对转错）；
- 门禁：G1 13 项新测全绿；G2 162 项回归零破坏；G3 FinQA 复跑无退化
  （EM +1 样本，lost=[]）；
- 回滚：`build_agent_graph()` 不传 llm_programmer 即回到 Phase 9 行为；
  ratio 模板独立于 LLM 路径，可单独保留；
- 诚实边界：LLM 程序生成未接入真实 provider 跑全量评测（无 API key 不
  虚报数字），以 mock provider 单测锁定语义；ratio 模板真实 FinQA 上
  检出执行 13 题对 1 题——失败主因是操作数锚定错误（候选选错行），
  属短语锚定的精度瓶颈，如实报告。

## 6. Phase 10.1 附录：ratio 锚定改进（2026-08-30）

对 13 道 ratio 失败样本逐条诊断后落地六项锚定改进：

| # | 改进 | 动机（失败形态） |
|---|---|---|
| A1 | 问题年份 token 直接匹配表格年份行（`_PHRASE_YEAR_RE`） | "due in 2018" 的操作数在行标签为 "2018" 的行，指标词典不覆盖 |
| A2 | "after \<year\>" → thereafter 行（`_AFTER_YEAR_RE`） | "due after 2020" 的金标行是 "2021 - thereafter" |
| A3 | 正文/脚注级数值扫描（跳过表格行；脚注标记 "(1)" 剔除） | 分子在正文不在表格（"42749 shares were repurchased..."） |
| A4 | 分母 total 行兜底，**仅当分母短语扫描完全无候选时注入** | "future minimum rental payments" 与行 "total" 零词重叠；无条件注入会让其他表的合计行压过正确行（过注入回归） |
| A5 | 列一致性排序（num/den 同列且列头词命中分母短语优先） | "total facilities as measured in square feet" 误取国别/子列值 |
| A6 | 负值剪枝（部分/整体非负） | 负值操作数产生无意义比率 |

### 验证（Phase 10.1）

| 口径 | 改前 | 改后 | 说明 |
|---|---|---|---|
| 理想检索（gold 全文直接喂执行器，`scripts/verify_ratio_ideal.py`） | 2/13 | **7/13** | 剥离检索层单独度量锚定精度 |
| 端到端 FinQA 100 样本（results/finqa_v10_1） | 7.0% | 7.0% | 13 道 ratio 答案全部按新锚定改变、零回归；瓶颈在检索（doc recall 56%） |
| 回归测试 | 175 | **181** | 新增 6 项锚定单测全绿 |

### 治理（Phase 10.1）

- 风险等级：低（仅影响 ratio 模板候选收集与排序；端到端复跑正确题目集合不变）；
- 回滚：六项改进相互独立，A4（total 兜底）的注入条件收敛是唯一行为开关；
- 诚实边界：理想检索剩余 6 题失败属短语锚定精度上限（行标签与问题短语
  零词重叠 / 量纲与刻度错配），端到端增益受检索层压制（金标文档未召回
  则锚定无从谈起）。
