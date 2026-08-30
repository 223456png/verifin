# Phase 5 — 多跳推理增强与对话体验优化 工作流（执行计划）

日期：2026-08-29
方法：TDD（RED → GREEN），回归门槛 = Phase 1-4 全部 46 项

## 执行步骤

| # | 步骤 | 产出 | 验证 |
|---|---|---|---|
| 1 | 编写 5 个测试文件（RED） | test_reconciliation(5) test_context(4) test_unit_parser(3) test_table_extraction(3) test_messages(2) | pytest 报 ImportError |
| 2 | tools/unit_parser.py | to_base / convert_unit / normalize_value（基准 million） | test_unit_parser 3 绿 |
| 3 | tools/context.py + inheritance_rules.json + pyproject package-data | infer_metric(query, history) 规则驱动 | test_context 4 绿 |
| 4 | tools/evidence.py 表格适配器 | parse_markdown_tables / parse_simple_html_tables / 表格式证据增强 | test_table_extraction 3 绿 |
| 5 | tools/verifier.py | resolve_conflicts + 结果富化（value/unit/period/source_type）+ major→REJECT | test_reconciliation 4 绿 |
| 6 | ui/messages.py + 节点升级 | 中文错误模板；synthesizer 失败路径全中文；state.claims；planner 继承；replanner consolidated 定向 | test_messages 2 绿 + test_reconciliation 集成 1 绿 |
| 7 | registry 注册 convert_unit | register_builtin_tools 增 1 工具 | 手检 list_tools |
| 8 | 全量验收 | pytest 63 项 + ruff + compileall | 输出证据 |
| 9 | 端到端演示（双文档 10% 冲突 → consolidated Replan） | 轨迹 + 中文诊断 | 演示输出 |
| 10 | 验收汇报（+GitHub 发布选项） | 对照成功标准 | — |

## 门禁

- **G1（步骤 6 前）**：17 项领域新测全绿；
- **G2（完成前）**：Phase 1-4 回归 46 项零破坏（合计 63）；
- **G3**：端到端冲突仲裁演示 + 中文报错零 Traceback。

## 风险与对策

- 回归风险：resolve_conflicts 仅介入「passed ≥2 且含数值」路径；单证路径字节级等价（D1）
- 中文输出与既有英文成功路径并存：仅失败路径中文化，成功路径不变（Phase 4 断言不受影响）
- JSON 资源随包分发：setuptools package-data 显式声明 inheritance_rules.json