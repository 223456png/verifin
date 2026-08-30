# Phase 6 — 对话状态与用户偏好记忆 工作流（执行计划）

日期：2026-08-29
方法：TDD（RED → GREEN），回归门槛 = Phase 1-5 全部 66 项

## 执行步骤

| # | 步骤 | 产出 | 验证 |
|---|---|---|---|
| 1 | 编写 5 个测试文件（RED） | test_dialog_state(5) test_preference_extractor(5) test_preference_applier(3) test_table_parser_adv(4) test_dialog_integration(4) | pytest ImportError/断言失败 |
| 2 | core/dialog_state.py | DialogState + update/promote + to_dict/from_dict | test_dialog_state 5 绿 |
| 3 | config/preference_rules.json + core/preference_extractor.py | 显式规则匹配 | test_preference_extractor 5 绿 |
| 4 | core/preference_applier.py | inherit/override/检索词构建 | test_preference_applier 2 绿 |
| 5 | tools/table_parser.py | 旋转/水平多级/跨页拼接 | test_table_parser_adv 4 绿 |
| 6 | 节点与周边升级 | state.dialog_state；planner 偏好注入 + hook；verifier 偏好加权；registry 2 工具；ui 确认消息 | test_preference_applier 仲裁 1 绿 + 集成 4 绿 |
| 7 | 全量验收 | pytest 87 + ruff + compileall | 输出证据 |
| 8 | 端到端演示 | 3 轮继承 + 偏好确认轨迹 | 演示输出 |
| 9 | 验收汇报 | 对照成功标准 | — |

## 门禁

- G1（步骤 6 前）：18 项领域新测全绿；
- G2（完成前）：Phase 1-5 回归 66 项零破坏（合计 87）；
- G3：演示含 preference_applied 字段与中文偏好确认。

## 风险与对策

- 回归风险：无偏好时 planner 子任务文本与 Phase 5 完全一致（D5）；resolve_conflicts 新参数可选
- 表格新解析仅作 fallback：标准结构命中后不再尝试旋转/交叉解析，避免覆盖既有测试行为
- 规则 JSON 随包分发：package-data 增加 config/*.json