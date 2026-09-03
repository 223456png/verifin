---
intent: 扩展确定性程序模板句式覆盖（ratio 5 变体 + average + growth/difference 关键词），把 39 个未检出的 divide 失败样本路由进计算路径，FinQA 100 样本 EM 8% → ≥14%
success_criteria: 全测试集句式未检出下降 ≥60%；benchmark EM 提升 ≥6pp；既有测试全绿
risk_level: medium
auto_approve: true
---

## Steps

- [ ] **Step 1: ratio 句式扩展（program_executor.py detect_program）**
action: 新增 _RATIO_OF_TO_RE（ratio of NUM to DEN，含 hyphen 变体）、_PCT_OF_TO_RE（percent of NUM to DEN，排除 due/compared/according/prior/relative/next to）、_AS_PCT_OF_RE（NUM as a percentage of DEN）、_REPRESENTED_PCT_RE（NUM represented what percentage of DEN），插在既有 _RATIO_RE 之后；双年份放行（D3）；词干提取 num/den 短语
verify: .venv/bin/python -m pytest tests/test_program_executor.py -x -q
loop: false

- [ ] **Step 2: average 模板（ProgramSpec.periods + _execute_average）**
action: ProgramSpec 增加 periods 字段；_AVERAGE_RE 检测（average X between/from Y1 and/to Y2 → periods=Y1..Y2 有序去重）；execute() 分派 _execute_average（每年桶 top-1、单位一致、sum/n，alternates 取次优组合）
verify: .venv/bin/python -m pytest tests/test_program_executor.py -x -q
loop: false

- [ ] **Step 3: growth/difference 关键词扩展**
action: _GROWTH_KEYWORDS += roi/return on investment/rate of return；_DIFFERENCE_KEYWORDS += growth/grew
verify: .venv/bin/python -m pytest tests/test_program_executor.py tests/test_calculation.py -x -q
loop: false

- [ ] **Step 4: 节点管道接线（nodes.py）**
action: detect_calculation_need 序列化 periods；_calculator_impl 重建 periods；average 走默认年份锚定路径（扫描循环扩展为 periods or (base, target)）；_ratio_candidates 按 num/den 短语内首个 20xx 年份分别锚定
verify: .venv/bin/python -m pytest tests/test_calculation.py tests/test_agent.py -x -q
loop: false

- [ ] **Step 5: 单元测试补齐**
action: 每个新句式 ≥2 正例 + ≥3 反例（due to expire 不触发、average 不劫持双年份 growth、新句式不抢既有 _RATIO_RE 流量）；_execute_average 数值 + 单位冲突剪枝用例
verify: .venv/bin/python -m pytest tests/ -q
loop: false

- [ ] **Step 6: 句式覆盖复核**
action: 全测试集 1147 题重跑句式频率统计，确认 ratio_of_to 48→0、pct_of_to 22→≤4、average_years 17→≤4、roi_return 20→≤2
verify: .venv/bin/python -c "覆盖统计脚本"
loop: false

- [ ] **Step 7: FinQA benchmark 端到端验证**
action: make benchmark-finqa（100 样本，bge 索引），对比 v11（EM 8%）；失败样本重新分解 kind 分布确认 NONE-divide 下降
verify: EM ≥ 14%
loop: until EM >= 14% 或连续 2 轮无提升
max_iterations: 3
gate: auto

- [ ] **Step 8: 文档更新**
action: README 徽章/阶段成果 + results/resume_numbers.md 追加 Phase 12 指标与诚实边界
verify: .venv/bin/python -m pytest tests/ -q
loop: false
