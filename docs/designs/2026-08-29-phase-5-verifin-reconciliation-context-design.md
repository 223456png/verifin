# Phase 5 — 多跳推理增强与对话体验优化 设计文档

日期：2026-08-29
范围：多文档冲突仲裁、上下文指标继承、表格证据、单位归一化、人性化诊断

## 1. Intent 契约（为什么做）

Phase 4 已能拦截"指标错配"（revenue vs gross_margin），但当**多份文档对同一指标给出不同数值**时，
系统没有仲裁能力；同时 Planner 的指标抽取不理解**上下文继承**（"去年赚了多少钱"的隐含指标），
财报核心数据（表格）与单位口径（$12,400m vs $12.4b）也未被充分利用。

Phase 5 赋予 Agent 三件事：
1. **Cross-Doc Reconciliation**：多文档同指标异值 → 时间/来源加权仲裁，小差异均值融合，大差异强制重规划；
2. **Implicit Metric Inference**：基于历史对话的指标继承（赚了多少钱 → 净利润/增速），规则由 JSON 驱动；
3. **Evidence 解析增强 + 体验优化**：表格提取、单位归一化、中文业务化诊断（零 Traceback）。

## 2. Constraints 约束

- 无新增外部依赖：表格解析用正则/纯标准库（csv/collections），单位换算为纯函数；
- 领域模块（tools/）不 import LangGraph；编排层（core/）保持薄适配；
- Phase 1-4 全部 46 项测试回归通过；
- `verify_flags[sub_task]` 结构向后兼容（`passed` 布尔语义不变）；
- inheritance_rules.json 以包数据随发行版携带（setuptools package-data）。

## 3. 关键决策

| # | 决策 | 理由 |
|---|---|---|
| D1 | 仲裁逻辑 `resolve_conflicts(passed_results)` 为 **tools/verifier.py 的纯函数**，批量校验后自动调用 | 可独立单测；不影响四要素逐项语义 |
| D2 | 可信度：来源标记 source_type(filing=1.0 > news=0.7 > research=0.5 > unknown=0.6；年份权重 1.0/0.66/0.33（按新旧排序） | 规则可解释，测试可断言加权均值偏向"最新财报" |
| D3 | 相对差异 `2*|a-b|/(a+b)`：≤2% 均值融合；2%~5% 保留双值与 minor 预警；(>5%) major → 决策 REJECT → 强制 Replan（合并报表口径） | 与提示词阈值一致；中间区间不伪造数值 |
| D4 | 单位归一化以 **百万（million）为基准**：billion×1000、亿×100、万×0.01；百分号不可跨量纲换算（% vs 绝对值 → major 冲突） | 默认口径统一，避免单位差误判 |
| D5 | 上下文继承：`extract_claim` 显式指标优先；无显式指标时 `infer_metric(query, history)` 查 inheritance_rules.json（关键词 + 继承对） | 规则可被主提示词/配置驱动，Planner 不硬编码 |
| D6 | `AgentState.claims: Dict[str, dict]` 新字段：planner 写入每个子任务的最终 claim，verifier 优先读取（fallback extract_claim） | 隐含指标无法从子任务文本反推时依然可校验；新增字段向后兼容 |
| D7 | 表格解析：markdown 表 + 简单 HTML `<table>` 行扫描，支持两级表头（组/明细）与首列空单元格承接（合并单元格 carry-forward） | 纯标准库即可覆盖三类常见结构；复杂排印留待 Phase 7 |
| D8 | 诊断消息集中于 `ui/messages.py`：错误码 → 中文模板（含指标中文名映射）；synthesizer 失败路径全中文输出 | 开发者/终端用户双受众分离；测试断言无技术黑话与 Traceback |
| D9 | `convert_unit` 注册为 Agent 工具（unit_parser 纯函数），calculator 白名单不引入字符串常量 | PoT 沙箱保持纯数值白名单的最小攻击面 |

## 4. 验证契约（如何确认）

| 步骤 | 验证 |
|---|---|
| 2 份冲突证据自动仲裁 | test_reconciliation：1% 差异 → ACCEPT+均值；10% → REJECT+consolidated 重规划（集成轨迹） |
| 隐含指标识别 | test_context：历史 revenue→profit（net income）；历史 growth→growth_rate；显式指标优先；JSON 规则加载 |
| 表格提取 ≥3 结构 | test_table_extraction：单表头 / 双级表头 / 合并单元格 carry-forward |
| 单位归一化 | test_unit_parser：billion↔million、亿↔million；$12,400m ≡ $12.4b 仲裁一致 |
| 零 Traceback 友好报错 | test_messages：中文模板、无 expected/got/mismatch 技术词；PoT 拦截原因中文化 |
| 回归 | `pytest -q`：63 passed（46 回归 + 17 新增）；ruff F401/I001/F841/F811 clean；compileall OK |

## 5. 治理契约

- **风险等级**：中（新增仲裁可能改变既有 ACCEPT 行为——仅当 passed 结果 ≥2 条且数值可比较时才介入；单条证据路径完全不变）
- **批准门禁**：G1 领域单测全绿（reconciliation/context/unit/table/messages 共 17）；G2 Phase 1-4 回归 46 项零破坏；G3 端到端演示轨迹含 consolidated 重规划
- **回滚策略**：resolve_conflicts 默认关闭于单证路径；`claims` 字段缺省空 dict，旧 checkpoint 可反序列化