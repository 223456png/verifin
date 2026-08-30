---
design_type: phase
created_at: 2026-08-29
---

# Phase 3 — VeriFin ReAct Agent Core（LangGraph 状态机）设计

## Intent Contract

```yaml
intent: 基于 LangGraph 构建 ReAct Agent 核心编排层：State 定义、Tool Registry（检索/计算/校验占位）、Planner 初始子任务队列、条件路由（ACCEPT/REJECT/REPLAN），并以确定性测试证明同一问题在不同上下文下执行路径长度可不同
constraints:
  - 领域 Tool（retriever/calculator/verifier）不 import LangGraph；编排层（core/）是薄适配层
  - Phase 1-2 公开 API 与 14 项测试不得破坏（回归全绿）
  - AgentState 可 JSON 序列化/反序列化（为 Phase 6 Checkpoint 做准备）
  - 全流程不依赖 LLM 调用（Planner 规则拆解、Verifier 关键词占位），保证离线可运行与可测
  - 重试上限 max_retries=3；耗尽的降级路径必须路由到 END（不得死循环）
success_criteria:
  - tests/test_agent.py 9 项测试全部通过（注册表单例、工具注册执行、planner 子任务、verifier 双路由、replanner 重试上限、完整运行成功、含重规划运行、动态路径长度、状态序列化）
  - 同一问题两次运行（第二次携带 verify_flags 上下文）路径长度不同（确定性断言）
  - Phase 1-2 回归 14 项全绿（合计 23 项）
risk_level: low
```

## Verification Contract

```yaml
verify_steps:
  - run tests: cd /workspace/verifin && .venv/bin/python -m pytest tests/ -v
  - check: 单例同一性；工具执行结果与历史记录；sub_tasks 非空；verifier 双路由 next_step；replanner 达到上限路由 end；完整图运行含/不含 replanner 的路径；动态路径两次运行长度不同；state JSON 往返一致
  - run demo: 用 stub 检索工具跑一次含重规划的真实图执行，打印节点轨迹 hooks 与最终答案
  - confirm: 全部测试通过（9 + 14），演示轨迹中节点访问顺序符合 ReAct 循环设计
```

## Governance Contract

```yaml
approval_gates:
  - 本设计文档经人工批准后方可开始实现（HOTL 硬门）
  - 实现完成后运行 code-review 流程（final），BLOCK 项修复后方可视为完成
rollback: 新增代码集中于 src/verifin/core/、tools/registry.py、tests/test_agent.py，回滚 = 删除新增文件并还原 schemas.py/pyproject.toml 改动；无持久化状态损失
ownership: 用户为项目 owner 并批准设计；AI agent 负责实现、测试与验证
```

## Scope

| In | Out |
|---|---|
| ToolCall/ToolResult/AgentState 数据结构 | LLM Planner 拆解（Phase 7 增强） |
| ToolRegistry 单例（注册/获取/执行/历史/重置） | 工具 schema 校验执行（Phase 4 与证据 schema 一并做） |
| 5 节点：planner/retriever/verifier/replanner/synthesizer（验证与计算均为占位实现） | 四要素校验（Phase 4）、PoT 计算器（Phase 4） |
| StateGraph 编译 + 条件路由（verifier 三向 / replanner 双向） | Checkpoint 持久化与恢复（Phase 6；仅带 MemorySaver 编译入口） |
| AgentRunner（run/astream 封装） | LangGraph 平台化服务（LangGraph Server） |
| 9 项 Agent 测试（stub 工具注入，不耦合真实索引） | 与真实索引/真实 LLM 的集成测试 |
| hooks 节点轨迹记录 | 轨迹分析报告（Phase 7） |

## Decisions

| # | 决策 | 选择 | 被否决的替代方案 |
|---|---|---|---|
| 1 | 依赖定位 | langgraph + langchain-core 进 core dependencies（编排层一等公民，与总规划一致） | 放 optional extra（编排是主链路，optional 会让默认安装跑不起 Agent） |
| 2 | LangGraph 版本策略 | pyproject 写 `langgraph>=0.2.0`（会解析到最新 1.x），代码做双版本兼容：MemorySaver 导入回退、Annotated+add_messages 状态、条件路由 1.x 签名，执行期以实测版本锁定行为 | 锁死 0.2.x（丢失官方维护与修复） |
| 3 | 图接线 | verifier 三向条件路由（synthesizer/replanner/retriever[多子任务回环]）、replanner 双向条件路由（retriever/END）、其余静态边 | spec 原案 replanner 静态边（重试耗尽时死循环）；verifier 只双路由（sub_tasks 队列空转） |
| 4 | 节点返回语义 | next_step 存入 state 作为路由信号；节点返回局部更新 dict（LangGraph 合并） | 节点间直接调用函数（失去状态机与可观测性） |
| 5 | Planner | 规则拆解：抽取大写实体与四位数年份，sub_tasks = [原查询] +（命中指标词时追加 "retrieve {metric} for {year}"）；无 LLM | 直接调 LLM 拆解（引入 API 依赖与不确定性，违背离线可测约束） |
| 6 | Verifier（占位） | 关键词校验：查询中的大写实体词（len>1）至少一个出现在检索文档拼接文本中 → 写 verify_flags[sub_task]={"passed": True, "entity": ...}；若该 sub_task 已有 passed 标记则直接放行（支撑动态路径） | 四要素完整校验（Phase 4 范围，本 Phase 仅占位） |
| 7 | Replanner | 失败时追加 "retrieve {query} with broader terms" 子任务，retry_count+1 回 retriever；达到上限路由 END 并附降级说明 | 静态固定重试（无增量信息，违背 Plan-Execute-Replan 闭环） |
| 8 | 工具注册 | builtin 注册 retrieve + verify_claim/calc_expression 占位；测试经 registry.reset() 后注册 stub 覆盖 | Agent 测试直连真实索引（flag：网络/产物依赖，破坏离线可测） |
| 9 | 动态路径证明 | hooks 记录每次节点访问；同问题第二次运行携带 verify_flags passed 上下文 → retriever 检测后跳过检索 → 路径 7 节点 vs 4 节点的确定性断言 | 依赖随机 LLM 行为证明动态性（不可复现） |
| 10 | 状态序列化 | AgentState 用 pydantic BaseModel（model_dump_json/validate 往返）；dataclass 子类型 ToolCall/ToolResult 以 dict 形态存于 field | TypedDict 手写状态（无校验、无序列化协议） |

## Surface

**公开 API（src/verifin）**：`schemas.py` 新增 `ToolCall`（tool_name/args/call_id）与 `ToolResult`（call_id/success/output/error/duration_ms）dataclass；`core/state.py` 定义 `AgentState(BaseModel)`：messages（Annotated[List[dict], add_messages]）、retrieved_docs、verify_flags、tool_call_history、hooks、retry_count、max_retries、sub_tasks、current_sub_task_index、next_step（Literal 路由信号）；`tools/registry.py` 提供 `ToolRegistry`（线程安全单例：register/get/list_tools/execute/reset）与 `register_builtin_tools()`；`core/nodes.py` 提供 5 个节点函数（planner/retriever/verifier/replanner/synthesizer），全部只读写 state、通过 ToolRegistry 调工具；`core/graph.py` 提供 `build_agent_graph()`（StateGraph + MemorySaver 编译，条件路由如 Decisions #3）；`core/runner.py` 提供 `AgentRunner.run(query) -> dict` 与 `astream`。

**运行环（一次 ReAct 循环）**：planner（拆解子任务）→ retriever（执行当前子任务检索，写 retrieved_docs）→ verifier（关键词校验，ACCEPT 且队列未完 → 回 retriever；ACCEPT 且队列完 → synthesizer；REJECT → replanner）→ replanner（追加子任务回 retriever 或 END 降级）→ synthesizer（仅绑定 passed 证据，生成带 [doc_id] 引用答案，messages 追加 assistant 消息，END）。

**动过的文件**：新增 `src/verifin/core/{__init__,state,nodes,graph,runner}.py`、`src/verifin/tools/registry.py`、`tests/test_agent.py`；修改 `src/verifin/schemas.py`（+2 dataclass）、`src/verifin/tools/__init__.py`（导出 register_builtin_tools）、`pyproject.toml`（+langgraph/langchain-core）、`docs/designs/verifin.md`（目标目录树补 runner.py）；`docs/designs/` 新增本设计文档。

## Risks & Open Questions

1. **LangGraph 1.x 与 0.2 时代 spec 的 API 漂移（中）**：安装将解析到最新 1.x，MemorySaver/条件路由/add_messages 的细节差异执行期验证——缓解：兼容导入 + 立即跑 9 项测试锁定行为；若 1.x 行为有实质差异，以实测调整并回写本设计说明。
2. **占位逻辑的诚实边界（低）**：关键词校验与规则拆解只是 Phase 3 的契约占位，测试必须验证「机制」而非「效果」（断言路径与信号，不断言检索质量）；真实校验/规划在 Phase 4/7 替换时以同样测试面回归。
3. **单例与测试隔离（低）**：reset() 解决跨测试注册污染；多线程安全由锁保证，Phase 3 无并发场景。
4. **开放问题**：MemorySaver 内存 checkpoint 是否作为 Phase 3 验收内容（仅编译入口 + 可通过 thread_id 隔离多次运行，持久化留 Phase 6）；`sub_tasks` 的拆解质量指标（覆盖实体/年份）留 Phase 7 与 Tool F1 同口径评估。