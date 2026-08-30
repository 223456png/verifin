---
intent: 基于 LangGraph 构建 ReAct Agent 核心编排层：State 定义、Tool Registry（检索/计算/校验占位）、Planner 初始子任务队列、条件路由（ACCEPT/REJECT/REPLAN），并以确定性测试证明同一问题在不同上下文下执行路径长度可不同
success_criteria: tests/test_agent.py 9 项测试全部通过；同一问题两次运行路径长度不同（确定性断言）；Phase 1-2 回归 14 项全绿（合计 23 项）
risk_level: low
auto_approve: true
worktree: false
---

# Phase 3 — VeriFin ReAct Agent Core 工作流

> 项目根目录：/workspace/verifin。Python 命令一律 `.venv/bin/python`。设计文档：docs/designs/2026-08-29-phase-3-verifin-agent-core-design.md（已批准）。
> 环境事实：langgraph 将解析到最新 1.x，步骤 1 实测兼容性（MemorySaver 导入回退、add_messages、条件路由）。

## Steps

- [ ] **Step 1: 安装并验证 LangGraph 依赖**
action: 在 /workspace/verifin 执行 `uv pip install --python .venv/bin/python -e ".[dev]"`（pyproject 已新增 langgraph>=0.2.0 与 langchain-core>=0.3.0）；随后用 python 脚本验证：打印 langgraph 与 langchain_core 版本；`from langgraph.graph import StateGraph, END` 可用；`from langgraph.graph.message import add_messages` 与 `from langchain_core.messages import add_messages` 至少一个可用；MemorySaver 从 langgraph.checkpoint.memory 导入（失败则回退 InMemorySaver，并把实际可用名记录在 scripts 输出）
verify: .venv/bin/python -c "
import importlib, langgraph, langchain_core
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
ms = None
try:
    from langgraph.checkpoint.memory import MemorySaver; ms = MemorySaver
except Exception:
    from langgraph.checkpoint.memory import InMemorySaver; ms = InMemorySaver
print('langgraph-ok', langgraph.__version__, '| langchain-core', langchain_core.__version__, '| saver', ms.__name__)"
loop: until 退出码 0 且输出以 langgraph-ok 开头
max_iterations: 3

- [ ] **Step 2: 编写规格测试 tests/test_agent.py（TDD RED 基线）**
action: 编写 tests/test_agent.py，9 项测试：(1) test_tool_registry_singleton —— 两次取实例 is 同一对象；(2) test_tool_registration_and_execution —— reset 后注册 lambda 工具，execute 断言 output 正确、history 记录、未注册工具返回 success=False；(3) test_planner_node_creates_sub_tasks —— 输入含实体与年份的问题，断言 sub_tasks 非空且含原查询；(4) test_verifier_routes_correctly —— 构造 ACCEPT（文档含实体）断言 next_step=="synthesizer"；REJECT 断言 next_step=="replanner"；(5) test_replanner_retry_limit —— retry_count==max_retries 时断言 next_step=="end"；(6) test_agent_full_run_success —— autouse fixture reset registry 并注册 stub retrieve（返回含实体文档）与 stub verify/calc，AgentRunner.run() 后断言最终 next_step=="end"、messages 有 assistant 答案、路径不含 replanner；(7) test_agent_full_run_with_replan —— stub retrieve 首次返回不含实体的文档、之后返回含实体文档，断言 hooks 节点序列包含 replanner 且最终成功；(8) test_agent_dynamic_path_length —— 同一问题跑两次：第一次初始 state 无上下文（路径 7 次节点访问）、第二次传递 verify_flags 含 passed 标记（路径 4 次访问），断言两次长度不同且第二次更短；(9) test_agent_state_serializable —— AgentState 填充数据后 model_dump_json → model_validate_json 往返，断言关键字段一致。测试使用 stub 工具注入（不耦合真实索引）；hooks 路径断言基于节点访问顺序
verify:
  - type: artifact
    path: tests/test_agent.py
    assert:
      kind: exists

- [ ] **Step 3: 扩展 schemas.py 并实现 tools/registry.py**
action: schemas.py 新增 ToolCall(tool_name/args: dict/call_id: str) 与 ToolResult(call_id/success: bool/output: Any/error: Optional[str]=None/duration_ms: float=0.0) dataclass；tools/registry.py 实现 ToolRegistry 线程安全单例（__new__ + _lock）：register(name, func, description, schema)、get、list_tools（LLM function-calling 描述列表）、execute(call)（未注册工具 success=False；执行成功记录 history 并返回 ToolResult 带 duration_ms；异常返回 success=False + error）、reset()（清空 tools 与 history）；模块级 register_builtin_tools() 注册 retrieve（from tools.retriever import retrieve）与 verify_claim/calc_expression 占位 lambda；tools/__init__.py 导出 ToolRegistry 与 register_builtin_tools
verify: .venv/bin/python -c "
from verifin.tools.registry import ToolRegistry, register_builtin_tools
from verifin.schemas import ToolCall
r1, r2 = ToolRegistry(), ToolRegistry()
assert r1 is r2
r1.reset()
r1.register('add', lambda a, b: a + b, 'add two numbers', {'type': 'object'})
res = r1.execute(ToolCall(tool_name='add', args={'a': 1, 'b': 2}, call_id='c1'))
assert res.success and res.output == 3 and res.duration_ms >= 0, res
assert r1.execute(ToolCall(tool_name='nope', args={}, call_id='c2')).success is False
print('registry-ok', len(r1.list_tools()))"
loop: until 退出码 0 且输出以 registry-ok 开头
max_iterations: 5

- [ ] **Step 4: 实现 src/verifin/core/state.py 与 core/__init__.py**
action: AgentState(BaseModel)：messages: Annotated[List[dict], add_messages]（从 langgraph.graph.message 导入，回退 langchain_core.messages）；retrieved_docs: List[dict]、verify_flags: Dict[str, Any]、tool_call_history: List[ToolCall]、hooks: List[dict]、retry_count: int=0、max_retries: int=3、sub_tasks: List[str]、current_sub_task_index: int=0、next_step: Literal["planner","retriever","verifier","replanner","synthesizer","end"]="planner"；全部 Field(default_factory=...) 防共享；core/__init__.py 导出 build_agent_graph/AgentState/AgentRunner（延迟导入避免循环）
verify: .venv/bin/python -c "
from verifin.core.state import AgentState
s = AgentState(messages=[{'role': 'user', 'content': 'Q'}])
j = s.model_dump_json()
s2 = AgentState.model_validate_json(j)
assert s2.messages[0]['content'] == 'Q' and s2.max_retries == 3 and s2.next_step == 'planner'
print('state-ok', len(j))"
loop: until 退出码 0 且输出以 state-ok 开头
max_iterations: 5

- [ ] **Step 5: 实现 src/verifin/core/nodes.py（5 个节点函数）**
action: 实现（全部只读写 state、经 ToolRegistry 调工具、向 hooks 追加 {"node": name, "ts", "retry_count", "sub_task"}）：planner_node —— 规则拆解：取 messages 最后一条用户消息，抽取大写实体词（len>1）与 4 位年份，sub_tasks=[query]+（含指标词 revenue/margin/cost/income/earnings/profit 且有年份时追加 "retrieve {metric} for {year} more specifically"），current_sub_task_index=0，next_step="retriever"，tool_call_history 记录 planner 调用；retriever_node —— 取当前 sub_task：若 verify_flags 已有该 sub_task 的 passed 标记则跳过检索（hooks 记 skip、docs 保持），否则 ToolRegistry.execute(retrieve, {"query": sub_task})，成功后把输出宽容转换为 doc dict 列表（SearchResultSet → dataclasses.asdict；已是 dict 列表则原样；单 dict 包列表）写入 retrieved_docs（本次子任务的文档覆盖本轮 docs），记录 ToolCall 到 tool_call_history，next_step="verifier"；verifier_node —— 占位关键词校验：query 中大写实体词任一出现在本轮检索文档拼接文本中，或 verify_flags 已有 passed 标记 → verify_flags[sub_task]={"passed": True, "entity": 命中词}，index+1；若还有未执行子任务 → next_step="retriever" 否则 "synthesizer"；失败 → verify_flags[sub_task]={"passed": False, "reason": "entity not found"}，next_step="replanner"；replanner_node —— retry_count<max_retries 时追加 sub_task "retrieve {query} with broader terms"、retry_count+1、next_step="retriever"；否则 next_step="end" 并记录降级 hooks；synthesizer_node —— 绑定 verify_flags 中 passed 的 sub_task 与 retrieved_docs 中对应文档，生成带 [doc_id] 引用的答案文本，messages 追加 {"role": "assistant", "content": answer}，next_step="end"
verify: .venv/bin/python -c "
from verifin.core.state import AgentState
from verifin.core.nodes import planner_node
s = AgentState(messages=[{'role': 'user', 'content': 'What was NovaTech revenue in 2024?'}])
out = planner_node(s)
assert out['sub_tasks'] and 'NovaTech revenue in 2024?' in out['sub_tasks'][0], out
assert out['next_step'] == 'retriever'
print('nodes-ok', out['sub_tasks'])"
loop: until 退出码 0 且输出以 nodes-ok 开头
max_iterations: 5

- [ ] **Step 6: 实现 src/verifin/core/graph.py 与 runner.py**
action: graph.py：build_agent_graph() 用 StateGraph(AgentState) 添加 5 节点，set_entry_point("planner")；条件路由 route_after_verifier(state)：next_step=="synthesizer"→"synthesizer"、"replanner"→"replanner"、"retriever"→"retriever"，否则 END；route_after_replanner：next_step=="retriever"→"retriever" 否则 END；静态边 planner→retriever、retriever→verifier、synthesizer→END；compile(checkpointer=MemorySaver 或 InMemorySaver 实测回退)；runner.py：AgentRunner(graph, thread_id="default")：run(query, initial_state=None) 组装 AgentState → graph.invoke(state, config={"configurable": {"thread_id": ...}}) 返回最终状态 dict；astream 同名异步封装
verify: .venv/bin/python -c "
from verifin.core.graph import build_agent_graph
g = build_agent_graph()
print('graph-ok', type(g).__name__, sorted(n for n in g.get_graph().nodes if n != '__start__'))"
loop: until 退出码 0 且输出以 graph-ok 开头
max_iterations: 5

- [ ] **Step 7: 运行 tool registry 规格测试**
verify: .venv/bin/python -m pytest tests/test_agent.py -k "tool_registry or tool_registration" -x -q
loop: until 退出码 0 且输出 passed
max_iterations: 5

- [ ] **Step 8: 运行 planner / verifier / replanner 规格测试**
verify: .venv/bin/python -m pytest tests/test_agent.py -k "planner_node or verifier_routes or replanner_retry" -x -q
loop: until 退出码 0 且输出 passed
max_iterations: 5

- [ ] **Step 9: 运行完整运行 / 动态路径 / 序列化 规格测试**
verify: .venv/bin/python -m pytest tests/test_agent.py -k "agent_full_run or dynamic_path or state_serializable" -x -q
loop: until 退出码 0 且输出 passed
max_iterations: 5

- [ ] **Step 10: 全量验收（Phase 3 9 项 + Phase 1-2 回归 14 项）**
verify:
  - type: shell
    command: .venv/bin/python -m pytest tests/ -v
  - type: shell
    command: .venv/bin/python -m compileall -q src/verifin
loop: until 退出码 0 且 pytest 输出 23 passed
max_iterations: 3

- [ ] **Step 11: 端到端演示（含重规划的 ReAct 循环）**
action: 用 stub retrieve（首查返回不含实体、复查返回含实体）执行 AgentRunner，打印 hooks 节点轨迹、verify_flags、retry_count 与最终答案；再跑一次携带 verify_flags 上下文的同一问题，打印两条轨迹长度对比
verify: .venv/bin/python - <<'PYEOF'
from verifin.tools.registry import ToolRegistry
from verifin.schemas import ToolCall
from verifin.core.runner import AgentRunner
from verifin.core.graph import build_agent_graph

reg = ToolRegistry(); reg.reset()
state_box = {"n": 0}
def fake_retrieve(query: str, top_k: int = 10):
    state_box["n"] += 1
    if state_box["n"] == 1:
        return [{"chunk_id": "d1", "content": "Helios Energy capex plan for 2025.", "metadata": {"doc_id": "other"}}]
    return [{"chunk_id": "d2", "content": "NovaTech revenue in 2024 was $12,400 million.", "metadata": {"doc_id": "nova"}}]
reg.register("retrieve", fake_retrieve, "stub", {"type": "object"})
reg.register("verify_claim", lambda claim, chunk: {"passed": True}, "stub", {"type": "object"})
reg.register("calc_expression", lambda expr: {"value": None}, "stub", {"type": "object"})

runner = AgentRunner(build_agent_graph(), thread_id="demo1")
final = runner.run("What was NovaTech revenue in 2024?")
nodes = [h["node"] for h in final["hooks"]]
print("path:", " -> ".join(nodes), "| len:", len(nodes), "| retries:", final["retry_count"])
answer = [m for m in final["messages"] if m.get("role") == "assistant"]
print("answer:", answer[-1]["content"] if answer else "N/A")
print("demo-ok")
PYEOF
loop: until 退出码 0 且输出含 demo-ok
max_iterations: 5

- [ ] **Step 12: 代码审查与人工验收**
action: 对 Phase 3 新增/修改代码运行 HOTL code-review（final，对照本工作流与 Phase 3 设计文档契约）；修复 BLOCK 项并重跑 Step 10；随后向用户汇报：测试结果、动态路径数字、演示轨迹、与验收标准逐项对照结论
gate: human