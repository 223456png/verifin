# Phase 6 — 对话状态与用户偏好记忆 设计文档

日期：2026-08-29
范围：多轮对话状态跟踪、偏好抽取/注入、复杂表格结构扩展

## 1. Intent 契约

- **多轮继承**：连续对话中记住实体/指标/期间偏好，后续问题缺省字段自动继承（用户免于重复表述）；
- **偏好记忆**：显式指令（“只看年报”）立即生效并覆盖旧偏好；隐式推断（连续同实体/单一指标）兜底；
- **偏好注入**：Planner 检索词加权、Verifier 冲突仲裁来源加权（+0.2）、Synthesizer 附带偏好确认；
- **表格扩展**：多级表头（垂直 + 水平交叉）、旋转表头（指标在顶/年份在列）、跨页拼接去重。

## 2. Constraints

- 无新增外部依赖（json/re/collections 标准库）；
- 领域层级规则不变：tools/ 与新增 core/dialog_state.py 均不 import LangGraph；
- 新对话重置偏好（DialogState 仅存于 AgentState，随 checkpoint/线程隔离，不跨会话持久化）；
- Phase 1-5 全部 66 项回归通过；`AgentState` 新增字段仅累加（dialog_state 默认空 dict）。

## 3. 关键决策

| # | 决策 | 理由 |
|---|---|---|
| D1 | `DialogState` 为 dataclass，经 asdict/from_dict 存入 `AgentState.dialog_state`（Dict[str,Any]）；planner 每轮读改写回 | 与 claims/verify_flags 的 dict 化模式一致，JSON 可序列化、checkpoint 可复用 |
| D2 | 优先级：显式指令 > 显式 claim > 隐式推断 > 系统默认；显式字段覆盖旧偏好并写 `preference_source` 来源 | 与规格「显式用户指令 > 隐式推断 > 系统默认」一致，可追溯 |
| D3 | 继承顺序：`extract_claim(文本) → 上下文指标推断(Phase5) → 偏好继承 → 显式指令覆盖` | 每层只填上一层的空缺，显式永远最后胜出 |
| D4 | 隐式规则：连续 3 轮同实体（含继承生效的实体）→ entity_preference；≥2 轮单一指标 → metric_preference；显式来源保留不降级 | 规则值由 preference_rules.json 驱动（implicit 段），可配置 |
| D5 | 偏好注入检索查询：`retrieve {entity} {metric} {period} [consolidated] [<source_token>] more specifically`；仅在有偏好时追加，无偏好输出与 Phase 5 字节一致 | 回归零破坏的关键 |
| D6 | 仲裁来源加权：偏好来源与证据 source_type 命中同一别名组 → 信任度 +0.2，conflict 输出 `preference_note`（如“已按用户偏好「年报」上调可信度”） | 规格“优先采信年报”的可解释实现 |
| D7 | 复杂表格：新模块 tools/table_parser.py（复用 evidence 的 TableSpec/parse_tables）；旋转表头=指标在列头+年份在第一列；(metric, period) 交叉定位；跨页=首列标签归一后按行拼接、剔除重复表头行与完全重复行 | 与 Phase 5 解析器分层不冲突 |
| D8 | 偏好确认消息集中在 ui/messages.py（source_label 中文映射 + preference_confirmation），仅成功路径前置一行 | 终端中文化延续 Phase 5，英文断言不受影响 |
| D9 | registry 注册 get_preferences / set_preference 工具（纯 dict 操作，供 Agent/Phase 7 LLM 调用） | 规格要求；薄包装，不引入新依赖 |

## 4. 验证契约

| 验证点 | 测试 |
|---|---|
| 空字段从偏好继承（entity/metric/period 三场景） | test_dialog_state 3 |
| 新指令覆盖旧偏好 + 来源追溯 | test_dialog_state / test_preference_extractor |
| 显式指令抽取（年报/合并口径/期间/新闻稿/研报） | test_preference_extractor 5 |
| 隐式推断（连续同实体、单一指标） | test_dialog_state 2 |
| 注入 Planner（检索词含偏好 + hook preference_applied） | test_preference_applier 2 |
| 注入 Verifier（仲裁加权 + preference_note） | test_preference_applier 1 |
| 多级表头（水平交叉）/旋转（2 种） | test_table_parser_adv 3 |
| 跨页拼接去重 | test_table_parser_adv 1 |
| 端到端（3 轮继承、偏好确认、显式覆盖、偏好+表格+仲裁联合） | test_dialog_integration 4 |
| 回归 66 项 + lint + compileall | pytest 87 passed |

## 5. 治理契约

- 风险等级：中（planner/verifier 输出新增偏好痕迹；无偏好时行为与 Phase 5 字节一致）
- 门禁：G1 21 项新测全绿；G2 66 项回归零破坏；G3 演示轨迹含 preference_applied 与中文偏好确认
- 回滚：dialog_state 缺省空 dict；resolve_conflicts/verify_claim_batch 新参数均可选