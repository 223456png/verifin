# Phase 4 — Verifier Tools 工作流（执行计划）

日期：2026-08-29
方法：TDD（RED → GREEN），回归门槛 = Phase 1-3 全部 25 项

## 执行步骤

| # | 步骤 | 产出 | 验证 |
|---|---|---|---|
| 1 | 编写 4 个测试文件（RED） | tests/test_evidence.py(4) test_verifier.py(7) test_calculator.py(5) test_integration_verifier.py(3) | pytest 报 ModuleNotFoundError |
| 2 | schemas.py 扩展 | Evidence / VerificationResult / ExpressionResult / VerifyFlags | 测试文件可 import |
| 3 | tools/evidence.py | EvidenceExtractor + METRIC_KEYWORDS + extract_entity_candidates | test_evidence 4 绿 |
| 4 | tools/verifier.py | Verifier + extract_claim + verify_claim/verify_claim_batch 包装 | test_verifier 7 绿 |
| 5 | tools/calculator.py | ExpressionCalculator + calc_expression 包装 | test_calculator 5 绿 |
| 6 | registry.register_builtin_tools 替换占位 | retrieve/verify_claim/verify_claim_batch/calc_expression/extract_evidence | 手检 registry.list_tools |
| 7 | core/nodes.py 升级 | verifier_node 真实校验；replanner 诊断重建子任务；synthesizer 仅绑定通过证据 | test_agent 11 项回归绿 |
| 8 | state.py verify_flags 类型强化（validator + docstring） | 写入值统一为 dict 形态 | 序列化测试回归绿 |
| 9 | test_agent.py 适配真实内置工具 | _register_stub_retrieve 调 register_builtin_tools 后仅 stub retrieve | 全量回归绿 |
| 10 | 全量验收 | pytest 44 项 + ruff F401/I001 + compileall | 输出证据 |
| 11 | 端到端演示（指标错配拦截） | 轨迹含 replanner，二次检索后 ACCEPT | 演示输出 |
| 12 | 验收汇报 | 对照 Section 十 验收标准 | — |

## 门禁（Gates）

- **G1（步骤 6 前）**：15 项领域单测全绿，无对外部服务依赖；
- **G2（步骤 9 前）**：test_agent 全绿（25 项回归无破坏）；
- **G3（完成前）**：44 项全绿 + 演示轨迹证据 + lint 干净。

## 风险与对策

- 回归破坏：verify_flags 新结构必须保留 `flags[task]["passed"]` 布尔语义（D6）；
- 计算超时不确定性：以长度/指数 guard 提供确定性拒绝，SIGALRM 仅作兜底（D7）；
- 指标词典歧义：最长同义词优先 + canonical key（D2），测试覆盖 revenue vs gross_margin 对抗样例。