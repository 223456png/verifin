---
intent: 消除端到端 doc recall 与检索层能力之间 32pp 的状态管理损耗（exclude 拉黑误杀治理 + bge 索引切换 + replanner 打转治理），FinQA test 100 样本 doc recall 56% → 75%+，EM 不回归
success_criteria: FinQA 复跑 doc recall ≥ 70%（目标 75%+）、EM ≥ 7.0%、正确题集合不减；26 个 LOSS-BY-EXCLUDE 样本 ≥ 80% 转为 gold 命中；181+ 项测试全绿
risk_level: medium
auto_approve: true
branch: main
worktree: false
---

## Steps

- [ ] **Step 1: 表格 chunk 豁免——失败测试先行**
action: 读 src/verifin/tools/replanner.py 与 src/verifin/tools/verifier.py，确认 verify_flags.results 的字段结构与 exclude_chunk_ids 生成路径。在 tests/ 中新增失败测试：构造含表格 chunk（content 含 markdown 管道行）且四要素校验失败的 verify_flags，断言 Replanner/replanner_node 生成的 exclude_chunk_ids **不含**该表格 chunk；同时断言纯散文 chunk 照旧被排除。
loop: until 新测试失败原因正确（豁免逻辑缺失）
max_iterations: 3
verify: .venv/bin/python -m pytest tests/ -q -k "table and exclu" 期望 1 failed
gate: auto

- [ ] **Step 2: 实现 exclude 表格 chunk 豁免**
action: 在 verify 结果（verify_claim_batch 输出的 results 条目）增加 is_table 布尔标记（content 含两个及以上 "|" 管道行即视为表格 chunk，判定函数放 verifier.py）；replanner.py 生成 exclude_chunk_ids 时跳过 is_table=True 的失败条目。不改变散文 chunk 的排除行为。
loop: until Step 1 新测试通过且全量回归绿
max_iterations: 4
verify: .venv/bin/python -m pytest tests/ -q
gate: auto

- [ ] **Step 3: replanner 重复子任务终止——测试先行**
action: 在 tests/ 中新增失败测试：sub_tasks 已含 "retrieve revenue 2016 calculation method"，replanner_node 再次收到相同 new_sub_task 时断言 next_step 走向终止/合成（不再追加重复子任务、不再进入 retriever 循环）。
loop: until 新测试失败原因正确
max_iterations: 3
verify: .venv/bin/python -m pytest tests/ -q -k "duplicate and subtask" 期望 1 failed
gate: auto

- [ ] **Step 4: 实现重复子任务检测终止**
action: src/verifin/core/nodes.py 的 replanner_node：plan.new_sub_task 与既有 sub_tasks 文本相同 → 视为无新信号，复用 plan.continue=False 的终止路径（人性化诊断 + END），不追加子任务。
loop: until Step 3 新测试通过且全量回归绿
max_iterations: 4
verify: .venv/bin/python -m pytest tests/ -q
gate: auto

- [ ] **Step 5: Makefile 固化 bge 基准目标**
action: Makefile 新增 benchmark-finqa 目标：`.venv/bin/python scripts/run_benchmark.py --dataset finqa --max-samples 100 --index-dir ./bge_env --output ./results`，注释写明需先构建 bge 索引（scripts/compare_retrieval.py 头部的构建命令）且查询需 sentence-transformers 模型（HF_ENDPOINT=https://hf-mirror.com）。
loop: false
verify: grep -A2 "benchmark-finqa" Makefile
gate: auto

- [ ] **Step 6: 追踪回归——26 个 exclude 样本转化验证**
action: 用修复后代码重跑 scripts/trace_gold_loss.py --idx 17 18 19 26 30 32 33 34 35 38 42 59 61 65 67 69 79 80 81 89 91 92 93 94 97（25 个可稳定复现的 LOSS-BY-EXCLUDE 样本）--index-dir . ，统计 GOLD-RETRIEVED 转化率。低于 80% 时逐样本诊断（预期失败原因：表格判定未覆盖该 chunk 形态，如 HTML 表格序列化后无管道符——按实际形态补判定规则后重跑）。
loop: until 转化率 ≥ 80% 或连续 2 轮无法提升（如实记录残余）
max_iterations: 3
verify: .venv/bin/python scripts/trace_gold_loss.py --idx 17 18 19 26 30 32 33 34 35 38 42 59 61 65 67 69 79 80 81 89 91 92 93 94 97 --index-dir . 2>&1 | grep -c "GOLD-RETRIEVED" 期望 ≥ 20
gate: auto

- [ ] **Step 7: 检索层不变验证**
action: 重跑 scripts/diagnose_recall.py，六组口径与设计文档基线一致（A1≈83%、A2≈86%、B1≈85%、B2≈88%、B3≈86%、B4≈88%，±1pp 漂移可接受）——证明本 phase 改动未波及检索层。
loop: until 六组口径均在 ±1pp 内
max_iterations: 2
verify: .venv/bin/python scripts/diagnose_recall.py 2>&1 | tail -8
gate: auto

- [ ] **Step 8: FinQA 端到端复跑（bge 索引）**
action: HF_HUB_OFFLINE=1 .venv/bin/python scripts/run_benchmark.py --dataset finqa --max-samples 100 --index-dir ./bge_env --output ./results/finqa_v11（模型缓存缺失时改用 HF_ENDPOINT=https://hf-mirror.com）。产出 report 后核对三个门限：doc recall ≥ 70%、EM ≥ 7.0%、正确题目集合相对 v10_1 不减（对比 results.json 的 is_correct 索引集合，lost 必须为空）。
loop: until 三门限全过（EM 若因 ratio 外噪声轻微波动 ±1 题，如实报告并由用户裁决是否重跑）
max_iterations: 2
verify: .venv/bin/python -c "import json; rows=json.load(open('results/finqa_v11/full/results.json')); old=json.load(open('results/finqa_v10_1/full/results.json')); new={i for i,r in enumerate(rows) if r['is_correct']}; prev={i for i,r in enumerate(old) if r['is_correct']}; assert not (prev-new), f'lost={sorted(prev-new)}'; print('EM', len(new)/len(rows), 'lost ok')"
gate: human

- [ ] **Step 9: 文档更新 + commit push**
action: results/resume_numbers.md（Phase 11 小节：损耗归因表 + exclude 豁免 + bge 切换 + 复跑数字 + 诚实边界）、README（Phase 11 条目、报告链接、测试数）、设计文档如需补充实测数字一并更新。git add 相关文件（src/tests/Makefile/scripts/docs/results），commit message 中文（Phase 11：检索召回修复——exclude 表格豁免/bge 切换/replanner 打转治理，doc recall 56%→N%），用 git -c credential-helper 推送 main。
loop: until commit 与 push 成功
max_iterations: 3
verify: git log --oneline -1 && git status --short
gate: auto
