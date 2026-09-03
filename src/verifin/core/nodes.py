"""ReAct 循环节点：planner / retriever / verifier / replanner / synthesizer。

约定：
- 节点接收 ``AgentState`` 或 dict（LangGraph 传参形态兼容），内部统一为 dict 读取；
- 节点返回**局部更新 dict**（只含变化的字段），由 LangGraph 合并入状态；
- 领域逻辑经 ToolRegistry 调用；工具未注册时回退直接调用领域实现（两者均不依赖 LangGraph）；
- 每次节点访问向 ``hooks`` 追加 ``{"node", "ts", "retry_count", "sub_task"}``。
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger

from verifin.core.dialog_state import DialogState
from verifin.core.preference_applier import apply_preferences, build_retrieval_query
from verifin.core.preference_extractor import extract_preferences
from verifin.schemas import ToolCall
from verifin.tools.context import infer_metric
from verifin.tools.failure_memory import FailureMemoryStore
from verifin.tools.registry import ToolRegistry
from verifin.tools.replanner import Replanner
from verifin.tools.verifier import extract_claim, verify_claim_batch
from verifin.ui.messages import friendly_message, preference_confirmation


def _as_dict(state: Any) -> dict:
    """把 AgentState（pydantic）或 dict 统一为 dict 视图。"""
    if isinstance(state, dict):
        return state
    return state.model_dump()


def _latest_user_query(state: dict) -> str:
    """取最新用户消息文本（兼容 dict 的 role/type 与 langchain Message 对象）。"""
    for message in reversed(state.get("messages") or []):
        if isinstance(message, dict):
            is_user = message.get("role") == "user" or message.get("type") == "human"
            content = message.get("content") if is_user else None
        else:
            # langchain Message：add_messages reducer 会把 dict 转成 Message 对象
            content = getattr(message, "content", None) if getattr(message, "type", "") == "human" else None
        if content:
            return content
    return ""


def _user_messages(state: dict) -> List[str]:
    """按序返回全部用户消息文本（含最新一条）。"""
    contents: List[str] = []
    for message in state.get("messages") or []:
        if isinstance(message, dict):
            is_user = message.get("role") == "user" or message.get("type") == "human"
            content = message.get("content") if is_user else None
        else:
            content = (
                getattr(message, "content", None)
                if getattr(message, "type", "") == "human"
                else None
            )
        if content:
            contents.append(str(content))
    return contents


def _hook(
    state: dict,
    node: str,
    sub_task: Optional[str],
    hook_extra: Optional[dict] = None,
    **updates: Any,
) -> dict:
    """追加可观测钩子并附带局部更新（hook_extra 如 preference_applied 并入钩子条目）。"""
    entry = {
        "node": node,
        "ts": datetime.now(timezone.utc).isoformat(),
        "retry_count": state.get("retry_count", 0),
        "sub_task": sub_task,
    }
    if hook_extra:
        entry.update(hook_extra)
    return {"hooks": list(state.get("hooks") or []) + [entry], **updates}


def _call_tool(name: str, **args: Any) -> Any:
    """经 ToolRegistry 执行工具；未注册/失败返回 None（节点层可做领域回退）。"""
    registry = ToolRegistry()
    func = registry.get(name)
    if func is None:
        return None
    call = ToolCall(tool_name=name, args=args, call_id=uuid.uuid4().hex)
    result = registry.execute(call)
    return result.output if result.success else None


# Phase 8：数值计算需求的关键词与双年份检测
_CALC_KEYWORDS = (
    "growth", "percentage", "increase", "decrease", "change",
    "grew", "grown", "rose", "yoy", "compare", "difference",
)
_YEAR_RE = re.compile(r"\b20\d{2}\b")

# 计算取值的来源可信度（与 verifier 的 _SOURCE_TRUST 对齐；filing > news > research）
_SOURCE_TRUST = {
    "filing": 1.0, "annual report": 1.0, "10-k": 1.0, "10k": 1.0, "sec": 1.0,
    "news": 0.7, "press release": 0.7,
    "research": 0.5, "analyst": 0.5,
    "unknown": 0.6,
}


def _detect_calculation(query: str) -> Optional[dict]:
    """检测数值计算需求（Phase 8 入口，Phase 9 委托 :func:`detect_program`）。

    - **growth_pct**：计算关键词 + 双年份（"from 2023 to 2024"；
      "compare to" 反序表达交换基期/比较期）；
    - **difference**（Phase 9）：净变动关键词 + 单年份 → 基期推断为上一年
      （FinQA 高频模式 ``subtract(cur, prev)``）；
    - 无关键词/无年份 → None（普通提取路径，防误伤）。
    """
    from verifin.tools.program_executor import detect_program

    spec = detect_program(query)
    if spec is None:
        return None
    return {
        "kind": spec.kind,
        "base_period": spec.base_period,
        "target_period": spec.target_period,
        "reversed": spec.reversed,
        # Phase 9.7 实体键控模板透传（跨实体差值 / argmax 接力）；
        # argmax 的主指标由检测正则给出（claim 词典归一可能取到第二指标）
        "metric": spec.metric,
        "entity_a": spec.entity_a,
        "entity_b": spec.entity_b,
        "entities": list(spec.entities),
        "second_metric": spec.second_metric,
        # Phase 10 ratio 模板透传（分子/分母短语）
        "numerator": spec.numerator,
        "denominator": spec.denominator,
        # Phase 12 average 模板透传（年份区间）
        "periods": list(spec.periods),
    }


def planner_node(state: Any) -> dict:
    """规则拆解 + 偏好注入：初始子任务队列（无 LLM，Phase 7 升级）。

    - 显式指标优先；无显式指标按历史上下文继承隐含指标（Phase 5）；
    - 会话偏好（Phase 6）：显式指令覆盖旧偏好 → claim 空字段从偏好继承 →
      隐式规则提升；检索子任务拼接 consolidated / 来源偏好 token；
    - 每个子任务的最终 claim 写入 ``state.claims``；偏好变化记录在 hook 的
      ``preference_applied`` 与 ``state.dialog_state``；
    - 数值计算（Phase 8）：增长/百分比类多跳问题（关键词 + 双年份）→
      拆为基期/比较期两个检索子任务 + 写 ``calculation_spec``，
      由 calculator 节点在证据校验后执行。
    """
    s = _as_dict(state)
    query = _latest_user_query(s)
    history = _user_messages(s)[:-1]

    claim = dict(extract_claim(query))
    if not claim.get("metric"):
        inferred = infer_metric(query, history)
        if inferred:
            claim["metric"] = inferred

    dialog = DialogState.from_dict(s.get("dialog_state") or {})
    explicit_prefs = extract_preferences(query)
    dialog, claim, applied = apply_preferences(dialog, claim, explicit_prefs)

    calc_spec = _detect_calculation(query)
    if calc_spec is not None:
        # 检测正则给出的主指标（argmax）优先，其余用 claim 词典归一结果
        calc_spec["metric"] = calc_spec.get("metric") or claim.get("metric")
        calc_spec["entity"] = claim.get("entity")

    claims: Dict[str, dict] = {query: claim}
    if calc_spec is not None and calc_spec.get("kind") == "cross_entity_diff":
        # Phase 9.7 跨实体差值流：两侧实体各一个携带四要素的检索子任务
        year = calc_spec.get("base_period") or claim.get("period")
        claim_a = {
            **claim,
            "entity": calc_spec.get("entity_a") or claim.get("entity"),
            "period": year,
        }
        claim_b = {**claim, "entity": calc_spec.get("entity_b"), "period": year}
        task_a = build_retrieval_query(claim_a, dialog)
        task_b = build_retrieval_query(claim_b, dialog)
        sub_tasks = [task_a, task_b]
        claims[task_a] = dict(claim_a)
        claims[task_b] = dict(claim_b)
    elif calc_spec is not None and calc_spec.get("kind") == "argmax_relay":
        # Phase 9.7 argmax 接力流：各实体的主指标子任务 + 第二指标子任务
        # （胜者规划期未知 → 第二指标按指标+期间检索，全部候选实体的
        # 第二指标 chunk 一并召回，由 calculator 比较后择胜者）
        year = calc_spec.get("base_period") or claim.get("period")
        sub_tasks = []
        for entity in calc_spec.get("entities") or []:
            entity_claim = {
                **claim, "entity": entity, "period": year,
                "metric": calc_spec.get("metric") or claim.get("metric"),
            }
            task = build_retrieval_query(entity_claim, dialog)
            sub_tasks.append(task)
            claims[task] = dict(entity_claim)
        second_claim = {
            **claim, "entity": None, "period": year,
            "metric": calc_spec.get("second_metric") or claim.get("metric"),
        }
        second_task = build_retrieval_query(second_claim, dialog)
        sub_tasks.append(second_task)
        claims[second_task] = dict(second_claim)
    elif calc_spec is not None and calc_spec.get("kind") != "ratio":
        # 计算流（年份键控）：基期/比较期各一个携带四要素的检索子任务；
        # 实体/指标缺失时（真实数据全小写问题）原问题先入队兜底检索
        # （Phase 10 ratio 走默认单检索路径——双操作数几乎总在同表，
        # 由 calculator 按分子/分母短语锚定取数）
        base_claim = {**claim, "period": calc_spec["base_period"]}
        target_claim = {**claim, "period": calc_spec["target_period"]}
        sub_tasks = [
            build_retrieval_query(base_claim, dialog),
            build_retrieval_query(target_claim, dialog),
        ]
        if not (claim.get("entity") and claim.get("metric")):
            sub_tasks.insert(0, query)
        if len(sub_tasks) == 3:
            task_claims = (
                (sub_tasks[0], claim), (sub_tasks[1], base_claim),
                (sub_tasks[2], target_claim),
            )
        else:
            task_claims = (
                (sub_tasks[0], base_claim), (sub_tasks[1], target_claim)
            )
        for task, task_claim in task_claims:
            claims[task] = dict(task_claim)
    else:
        sub_tasks = [query]
        if claim.get("metric") and claim.get("period"):
            refined = build_retrieval_query(claim, dialog)
            sub_tasks.append(refined)
            claims[refined] = dict(extract_claim(refined))

    return _hook(
        s, "planner", query,
        hook_extra={"preference_applied": applied, "calculation_requested": calc_spec is not None},
        sub_tasks=sub_tasks,
        claims=claims,
        dialog_state=dialog.to_dict(),
        current_sub_task_index=0,
        retry_count=0,
        failure_history=[],
        exclude_chunk_ids=[],
        calculation_requested=calc_spec is not None,
        calculation_spec=calc_spec or {},
        calculation_result={},
        next_step="retriever",
    )


def make_planner_node(llm_planner=None, llm_programmer=None):
    """构造 planner 节点工厂：注入 LLM 规划器 / LLM 程序生成器（可降级）。

    ``llm_planner`` 需提供 ``available`` 属性与 ``plan(query, history) -> Optional[dict]``
    方法（见 :class:`verifin.core.llm_planner.LLMPlanner`）；None 或不可用时节点行为
    与确定性规则 planner 完全一致。

    ``llm_programmer``（Phase 10）需提供 ``available`` 属性（见
    :class:`verifin.tools.llm_programmer.LLMProgramGenerator`）；注入且可用时，
    确定性模板未检出的数值问题置 ``calculation_spec={"kind": "llm_program"}``
    （calculator 走 LLM 程序生成路径，失败自动降级诊断）。

    合并策略（LLM 可用且输出合法时）：
    - ``sub_tasks``：采用 LLM 的分解结果；
    - ``claims``：规则 claim 优先（含 query 的可靠四要素），LLM 补充其余子任务；
    - ``calculation_spec``：规则 detect_program 优先，LLM 仅补充规则未检测到的计算。
    """
    def _node(state: Any) -> dict:
        use_llm_program = (
            llm_programmer is not None
            and getattr(llm_programmer, "available", False)
        )
        if llm_planner is None or not getattr(llm_planner, "available", False):
            if not use_llm_program:
                return planner_node(state)
            # Phase 10：无 LLM planner 但有 programmer → 模板未检出时走 LLM 程序
            result = planner_node(state)
            if not result.get("calculation_spec"):
                result["calculation_spec"] = {"kind": "llm_program"}
                result["calculation_requested"] = True
            return result

        # 先跑规则 planner（保证偏好注入/calc 检测/兜底齐全），再用 LLM 增强分解
        result = planner_node(state)
        s = _as_dict(state)
        query = _latest_user_query(s)
        history = _user_messages(s)[:-1]

        llm_plan = None
        try:
            llm_plan = llm_planner.plan(query, history)
        except Exception as exc:  # noqa: BLE001 - provider 异常降级
            logger.warning("LLM 规划异常，降级规则 planner: {}", exc)
            llm_plan = None

        if not llm_plan or not llm_plan.get("sub_tasks"):
            return result

        merged_claims = dict(result.get("claims") or {})
        for task in llm_plan["sub_tasks"]:
            if task in merged_claims:
                continue
            tc = llm_plan.get("claims", {}).get(task)
            merged_claims[task] = (
                dict(tc) if isinstance(tc, dict) else dict(extract_claim(task))
            )

        result["sub_tasks"] = llm_plan["sub_tasks"]
        result["claims"] = merged_claims
        if llm_plan.get("calculation_spec") and not result.get("calculation_spec"):
            result["calculation_spec"] = llm_plan["calculation_spec"]
            result["calculation_requested"] = True

        return result

    return _node


def _coerce_docs(output: Any) -> List[dict]:
    """把检索工具输出宽容转换为 doc dict 列表。"""
    if output is None:
        return []
    if isinstance(output, dict):
        return [output]
    if isinstance(output, list):
        return [item for item in output if isinstance(item, dict)]
    # SearchResultSet（dataclass）形态：取 results 列表
    if hasattr(output, "results"):
        from dataclasses import asdict

        return [asdict(item) for item in output.results]
    return []


def retriever_node(state: Any) -> dict:
    """执行当前子任务检索（verify_flags 已通过的子任务跳过，支撑动态路径）。"""
    s = _as_dict(state)
    sub_tasks: List[str] = s.get("sub_tasks") or []
    index: int = s.get("current_sub_task_index", 0)
    if index >= len(sub_tasks):
        return _hook(s, "retriever", None, next_step="synthesizer")
    task = sub_tasks[index]

    # 上下文注入：该子任务已通过 → 跳过检索，直接进入校验（路径缩短）
    if (s.get("verify_flags") or {}).get(task, {}).get("passed"):
        return _hook(
            s, "retriever", task, next_step="verifier", retrieved_docs=s.get("retrieved_docs") or []
        )

    call = ToolCall(
        tool_name="retrieve",
        args={"query": task, "top_k": 25},  # FinQA 实测 gold 常排在 21-25 名
        call_id=uuid.uuid4().hex,
    )
    result = ToolRegistry().execute(call)
    docs: List[dict] = []
    if result.success:
        docs = _coerce_docs(result.output)
    else:
        logger.warning("retrieve 工具执行失败: {}", result.error)
    # 带记忆（Phase 5 修正版 / Phase 11 语义修正）：排除**只过滤本轮新检索
    # 结果**（不在失败方向的 chunk 上重复检索），**不驱逐历史轮已入池的
    # 证据**——verifier 对每个子任务按各自 claim 重新校验全部 docs，
    # chunk 在子任务 A 失败不代表在细化后的子任务 B 下无价值；驱逐曾造
    # 成 gold 文档整批丢失（FinQA 26/44 doc recall 失败样本的主因）。
    excluded = set(s.get("exclude_chunk_ids") or [])
    if excluded:
        docs = [doc for doc in docs if doc.get("chunk_id") not in excluded]
    # 多子任务结果合并（Phase 9 修正）：计算流基期/比较期各自检索，若每轮
    # 覆盖上一轮结果，最终只剩最后一个子任务的文档——原始问题的强词法
    # 信号被丢弃（FinQA 实测 doc recall 48% → 82%）。按 chunk_id 去重
    # 合并（本轮新结果优先），上限 80（计算流 3 子任务 × top 25 不截断）；
    # multi-query merge 是 RAG 多查询检索的标准做法（LangChain
    # MultiQueryRetriever 同思路）。
    previous = [d for d in (s.get("retrieved_docs") or []) if isinstance(d, dict)]
    merged: List[dict] = []
    seen_chunk_ids = set()
    for doc in docs + previous:
        chunk_id = doc.get("chunk_id")
        if chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk_id)
        merged.append(doc)
        if len(merged) >= 80:
            break
    return _hook(
        s, "retriever", task,
        retrieved_docs=merged,
        tool_call_history=list(s.get("tool_call_history") or []) + [call],
        next_step="verifier",
    )


def _next_after_queue(state: dict) -> str:
    """子任务队列处理完毕后的路由：有计算需求 → calculator；否则 synthesizer。"""
    return "calculator" if state.get("calculation_requested") else "synthesizer"


def verifier_node(state: Any) -> dict:
    """四要素证据校验（真实 Verifier，经 ``verify_claim_batch`` 工具）。

    - 已有 passed 标记（历史上下文注入）→ 直接放行（动态路径缩短的关键）；
    - ACCEPT / PARTIAL 且队列未完 → 回 retriever 处理下一子任务；否则 → synthesizer；
    - REJECT → replanner（触发诊断性重规划）。
    """
    s = _as_dict(state)
    sub_tasks: List[str] = s.get("sub_tasks") or []
    index: int = s.get("current_sub_task_index", 0)
    task = sub_tasks[index] if 0 <= index < len(sub_tasks) else ""
    flags: Dict[str, Any] = dict(s.get("verify_flags") or {})

    existing = flags.get(task) or {}
    if existing.get("passed"):
        next_index = index + 1
        if next_index < len(sub_tasks):
            return _hook(
                s, "verifier", task,
                current_sub_task_index=next_index, next_step="retriever",
            )
        return _hook(s, "verifier", task, next_step=_next_after_queue(s))

    claim_map: dict = s.get("claims") or {}
    claim = claim_map.get(task) if isinstance(claim_map, dict) else None
    if not claim:
        claim = extract_claim(task) if task else {}
    docs = [doc for doc in (s.get("retrieved_docs") or []) if isinstance(doc, dict)]
    # 会话偏好传入仲裁（Phase 6：偏好来源可信度加权）
    preferences = s.get("dialog_state") or {}
    batch = _call_tool(
        "verify_claim_batch", claim=claim, chunks=docs, preferences=preferences
    )
    if not isinstance(batch, dict):
        # 工具未注册/失败 → 领域直接回退（保证节点可独立运行与单测）
        batch = verify_claim_batch(claim, docs, preferences)
    decision = batch.get("decision", "REJECT")
    flags[task] = {**batch, "sub_task_index": index}

    if decision in ("ACCEPT", "PARTIAL"):
        next_index = index + 1
        if next_index < len(sub_tasks):
            return _hook(s, "verifier", task, verify_flags=flags,
                         current_sub_task_index=next_index, next_step="retriever")
        return _hook(s, "verifier", task, verify_flags=flags,
                     next_step=_next_after_queue(s))
    logger.info(
        "子任务校验 REJECT: mismatches={} conflict={}",
        [result.get("mismatches") for result in batch.get("results", [])],
        (batch.get("conflict") or {}).get("level", "none"),
    )
    # Phase 8（真实数据适配）：计算问题时四要素校验（实体/指标在真实语料抽取困难）
    # 常被判 REJECT，重规划螺旋对计算流无增益——不走 replanner；
    # Phase 9 修正：基期/比较期检索子任务仍在队列时先跑完（多子任务检索
    # 结果合并可显著提升 doc recall——首个子任务单路检索常漏 gold），
    # 队列耗尽再进 calculator 做年份锚定数值兜底扫描。
    if s.get("calculation_requested"):
        next_index = index + 1
        if next_index < len(sub_tasks):
            return _hook(
                s, "verifier", task, verify_flags=flags,
                current_sub_task_index=next_index, next_step="retriever",
            )
        return _hook(s, "verifier", task, verify_flags=flags, next_step="calculator")
    return _hook(s, "verifier", task, verify_flags=flags, next_step="replanner")


def _scan_years_in_docs(docs: List[dict], base_year: str, target_year: str):
    """年份锚定数值扫描（Phase 8 兜底；Phase 9 由 calculator 节点的
    多候选收集替代，保留供单测直接调用）。
    """
    from verifin.tools.evidence import EvidenceExtractor

    base_hit: Optional[dict] = None
    target_hit: Optional[dict] = None
    for doc in docs:
        content = str(doc.get("content") or "")
        for year, holder in (
            (str(base_year), "base"), (str(target_year), "target"),
        ):
            if holder == "base" and base_hit is not None:
                continue
            if holder == "target" and target_hit is not None:
                continue
            value, unit, _, _ = EvidenceExtractor._anchored_value(
                content, year_filter=str(year), prefer="year"
            )
            if value is None:
                continue
            item = {
                "value": value,
                "unit": unit,
                "period": str(year),
                "chunk_id": doc.get("chunk_id"),
            }
            if holder == "base":
                base_hit = item
            else:
                target_hit = item
            if base_hit is not None and target_hit is not None:
                return base_hit, target_hit
    return base_hit, target_hit


def _entity_keyed_candidates(s: dict, program_spec) -> Dict[str, list]:
    """实体键控候选收集（Phase 9.7：跨实体差值 / argmax 接力计算流）。

    返回 ``{entity: [ValueCandidate]}``；argmax_relay 额外含
    ``"second::<entity>"`` 桶（胜者第二指标接力查找，键与
    :meth:`ProgramExecutor._execute_argmax_relay` 约定一致）。

    收集顺序（桶内按 anchor_score 升序取优）：
    1. 四要素校验通过的证据（claim 实体 + 指标双匹配；新闻来源 +0.5 惩罚）；
    2. 检索文档的实体锚定扫描（实体提及 + 指标同义词 + 年份三信号最近邻）；
    3. small-to-big 父文档补全（表格数值块 BM25 信号弱，按问题实词 ×
       已检索 chunk 重叠度排序补全，+0.25 来源惩罚）。
    """
    from verifin.tools.evidence import EvidenceExtractor
    from verifin.tools.program_executor import ValueCandidate

    if program_spec.kind == "cross_entity_diff":
        entities = [program_spec.entity_a, program_spec.entity_b]
        metric_pairs = [(program_spec.metric, "")]
    else:
        entities = list(program_spec.entities)
        metric_pairs = [
            (program_spec.metric, ""),
            (program_spec.second_metric, "second::"),
        ]
    year = program_spec.base_period or None
    query_text = _latest_user_query(s)

    candidates: Dict[str, List[ValueCandidate]] = {}

    def _add(prefix: str, entity: str, item: dict, chunk_id, anchor: float) -> None:
        key = f"{prefix}{entity}" if prefix else entity
        bucket = candidates.setdefault(key, [])
        if any(c.value == item["value"] and c.unit == item["unit"] for c in bucket):
            return
        bucket.append(ValueCandidate(
            value=item["value"],
            unit=item.get("unit"),
            period=item.get("period") or year,
            chunk_id=chunk_id,
            anchor_score=anchor,
        ))

    # 1. 校验通过证据（claim 实体 × 指标匹配）
    source_types: Dict[str, str] = {}
    docs = [d for d in (s.get("retrieved_docs") or []) if isinstance(d, dict)]
    for doc in docs:
        meta = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
        if meta.get("source_type"):
            source_types[str(doc.get("chunk_id"))] = str(meta["source_type"])
    entity_lookup = {str(e).lower(): e for e in entities if e}
    for flag in (s.get("verify_flags") or {}).values():
        if not isinstance(flag, dict):
            continue
        flag_claim = flag.get("claim") if isinstance(flag.get("claim"), dict) else {}
        claim_entity = str(flag_claim.get("entity") or "").lower()
        entity = entity_lookup.get(claim_entity)
        if entity is None:
            continue
        for metric, prefix in metric_pairs:
            if not metric or flag_claim.get("metric") != metric:
                continue
            for result in flag.get("results") or []:
                if not (
                    isinstance(result, dict)
                    and result.get("passed")
                    and result.get("value") is not None
                ):
                    continue
                if year and str(year) not in str(result.get("period") or ""):
                    continue
                source_penalty = (
                    0.5 if source_types.get(str(result.get("chunk_id"))) == "news" else 0.0
                )
                _add(
                    prefix, entity,
                    {"value": float(result["value"]), "unit": result.get("unit")},
                    result.get("chunk_id"), 1.0 + source_penalty,
                )

    # 2. 检索文档实体锚定扫描
    def _scan(content: str, chunk_id, penalty: float = 0.0) -> None:
        for entity in entities:
            if not entity:
                continue
            for metric, prefix in metric_pairs:
                if not metric:
                    continue
                for item in EvidenceExtractor.collect_entity_values(
                    content, entity, year, spec_metric=metric
                ):
                    _add(prefix, entity, item, chunk_id,
                         penalty + item.get("anchor_score", 1.0))

    for doc in docs:
        _scan(str(doc.get("content") or ""), doc.get("chunk_id"))

    # 3. small-to-big 父文档补全（问题实词 × 已检索 chunk 重叠度排序）
    doc_chunks: Dict[str, str] = {}
    for doc in docs:
        meta = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
        doc_id = str(meta.get("doc_id") or "")
        if doc_id:
            doc_chunks[doc_id] = doc_chunks.get(doc_id, "") + " " + str(doc.get("content") or "")
    query_tokens = EvidenceExtractor._content_tokens(query_text)
    doc_ids = sorted(
        doc_chunks,
        key=lambda d: -len(EvidenceExtractor._content_tokens(doc_chunks[d]) & query_tokens),
    )
    for doc_id in doc_ids[:10]:
        expanded = _call_tool("expand_document", doc_id=doc_id)
        if not isinstance(expanded, list):
            continue  # 工具未注册（如单元测试）→ 跳过
        for chunk in expanded:
            if isinstance(chunk, dict):
                _scan(
                    str(chunk.get("content") or ""),
                    chunk.get("chunk_id"),
                    penalty=0.25,
                )
    return candidates


def _ratio_candidates(s: dict, program_spec) -> Dict[str, list]:
    """比率流候选收集（Phase 10："what percentage of X are Y"）。

    返回 ``{"num": [ValueCandidate], "den": [ValueCandidate]}``——分子/分母
    各自按短语锚定扫描（行标签 × 短语实词重叠）。短语无实词
    （"due in 2018"）时回退年份锚定扫描（问题年份）。

    收集顺序（桶内按 anchor_score 升序取优）：
    1. 检索文档的短语锚定表格扫描（行标签词重叠锚定）；
    2. small-to-big 父文档补全（+0.25 来源惩罚）。
    """
    from verifin.tools.evidence import EvidenceExtractor
    from verifin.tools.program_executor import ValueCandidate

    candidates: Dict[str, List[ValueCandidate]] = {"num": [], "den": []}
    # total 行回退候选（Phase 10.1）：仅当分母短语扫描**完全无候选**时启用，
    # 无条件注入会让证据中其他表的合计行（anchor 0.75）压过正确行的
    # 弱短语命中（单 token 命中 anchor 1.0），系统性引入错表分母
    total_fallback: List[tuple] = []
    query_text = _latest_user_query(s)
    year = program_spec.base_period or None
    # Phase 12：短语内年份锚定——"ratio of the purchase in december 2012
    # to the purchase in january 2013" 的两个年份分别在分子/分母短语内
    # （spec.base_period 为空），按各自短语首个 20xx 年份分别锚定
    def _phrase_year(phrase: Optional[str]) -> Optional[str]:
        m = re.search(r"\b(20\d{2})\b", phrase or "")
        return m.group(1) if m else None

    phrase_anchors = (
        ("num", program_spec.numerator, _phrase_year(program_spec.numerator) or year),
        ("den", program_spec.denominator, _phrase_year(program_spec.denominator) or year),
    )

    def _add(bucket: str, item: dict, chunk_id, penalty: float) -> None:
        lst = candidates[bucket]
        if any(c.value == item["value"] and c.unit == item["unit"] for c in lst):
            return
        lst.append(
            ValueCandidate(
                value=float(item["value"]),
                unit=item.get("unit"),
                period=str(item.get("period") or year or ""),
                chunk_id=chunk_id,
                anchor_score=penalty + float(item.get("anchor_score", 0.0)),
                row_label=str(item.get("row_label") or ""),
                column=str(item.get("column") or ""),
            )
        )

    def _scan(content: str, chunk_id, penalty: float = 0.0) -> None:
        for bucket, phrase, phrase_year in phrase_anchors:
            if not phrase:
                continue
            items = EvidenceExtractor.collect_phrase_values(
                content, phrase, phrase_year
            )
            if not items:
                # 短语无实词（"due in 2018"）→ 年份锚定兜底
                if phrase_year:
                    items = EvidenceExtractor.collect_year_values(content, phrase_year)
            for item in items:
                _add(bucket, item, chunk_id, penalty)
        # 分母 total 行回退候选（Phase 10.1）：先缓存，仅当分母短语扫描
        # 完全无候选时兜底启用（"percentage of X" 的 X 几乎总是合计行，
        # 但 X 短语与行标签常零词重叠——"future minimum rental payments"
        # vs 行 "total"）
        for item in EvidenceExtractor.collect_total_values(content):
            total_fallback.append((item, chunk_id, penalty))

    for doc in [d for d in (s.get("retrieved_docs") or []) if isinstance(d, dict)]:
        _scan(str(doc.get("content") or ""), doc.get("chunk_id"))

    # 父文档补全（small-to-big）：比率题双操作数几乎总在同表，
    # 检索未直接命中数值 chunk 时按问题实词 × 文档重叠度补全
    doc_chunks: Dict[str, str] = {}
    for doc in [d for d in (s.get("retrieved_docs") or []) if isinstance(d, dict)]:
        meta = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
        doc_id = str(meta.get("doc_id") or "")
        if doc_id:
            doc_chunks[doc_id] = doc_chunks.get(doc_id, "") + " " + str(doc.get("content") or "")
    query_tokens = EvidenceExtractor._content_tokens(query_text)
    doc_ids = sorted(
        doc_chunks,
        key=lambda d: -len(EvidenceExtractor._content_tokens(doc_chunks[d]) & query_tokens),
    )
    for doc_id in doc_ids[:10]:
        expanded = _call_tool("expand_document", doc_id=doc_id)
        if not isinstance(expanded, list):
            continue
        for chunk in expanded:
            if isinstance(chunk, dict):
                _scan(
                    str(chunk.get("content") or ""),
                    chunk.get("chunk_id"),
                    penalty=0.25,
                )

    # 分母短语全空 → total 行回退兜底（直接 chunk 优先于父文档补全来源）
    if not candidates["den"]:
        for item, chunk_id, penalty in total_fallback:
            _add("den", item, chunk_id, penalty)
    return candidates


def _llm_program_candidates(s: dict, limit: int = 12) -> List[dict]:
    """LLM 程序生成的编号候选（Phase 10）：检索文档 + 父文档补全的表格数值。

    返回 ``[{"id": "v1", "value": 8.1, "context": "row: ... | col: ...",
    "chunk_id": ...}]``（单值 vN，按锚定分排序截断）；同一行标签的
    多个数值聚合成 ``tN`` 值组候选（聚合算子参数）。
    """
    from verifin.tools.evidence import EvidenceExtractor

    query_text = _latest_user_query(s)
    import re as _re

    years = list(dict.fromkeys(_re.findall(r"\b(?:19|20)\d{2}\b", query_text)))[:2]

    singles: List[dict] = []
    groups: List[dict] = []
    seen_values = set()
    seen_groups = set()

    def _scan(content: str, chunk_id, penalty: float = 0.0) -> None:
        for year in years:
            for item in EvidenceExtractor.collect_table_year_values(
                content, year, query=query_text
            ):
                key = (item.get("row_label"), item["value"])
                if key in seen_values:
                    continue
                seen_values.add(key)
                singles.append({
                    **item, "chunk_id": chunk_id,
                    "anchor_score": penalty + item.get("anchor_score", 0.0),
                })
        # 无年份 / 聚合：短语锚定取行组
        rows: Dict[str, List[float]] = {}
        for item in EvidenceExtractor.collect_phrase_values(
            content, query_text, max_n=12
        ):
            label = str(item.get("row_label") or "")
            if label and label not in seen_groups:
                rows.setdefault(label, []).append(item["value"])
        for label, values in rows.items():
            if len(values) < 2:
                continue  # 单值行已在 singles 覆盖
            seen_groups.add(label)
            groups.append({
                "label": label, "values": values, "chunk_id": chunk_id,
            })

    for doc in [d for d in (s.get("retrieved_docs") or []) if isinstance(d, dict)]:
        _scan(str(doc.get("content") or ""), doc.get("chunk_id"))

    doc_chunks: Dict[str, str] = {}
    for doc in [d for d in (s.get("retrieved_docs") or []) if isinstance(d, dict)]:
        meta = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
        doc_id = str(meta.get("doc_id") or "")
        if doc_id:
            doc_chunks[doc_id] = doc_chunks.get(doc_id, "") + " " + str(doc.get("content") or "")
    query_tokens = EvidenceExtractor._content_tokens(query_text)
    doc_ids = sorted(
        doc_chunks,
        key=lambda d: -len(EvidenceExtractor._content_tokens(doc_chunks[d]) & query_tokens),
    )
    for doc_id in doc_ids[:5]:
        expanded = _call_tool("expand_document", doc_id=doc_id)
        if not isinstance(expanded, list):
            continue
        for chunk in expanded:
            if isinstance(chunk, dict):
                _scan(
                    str(chunk.get("content") or ""),
                    chunk.get("chunk_id"),
                    penalty=0.25,
                )

    singles.sort(key=lambda item: item.get("anchor_score", 0.0))
    out: List[dict] = []
    for index, item in enumerate(singles[:limit], start=1):
        context = f"row: {item.get('row_label') or '?'}"
        if item.get("column"):
            context += f" | col: {item['column']}"
        out.append({
            "id": f"v{index}",
            "value": item["value"],
            "context": context,
            "chunk_id": item.get("chunk_id"),
        })
    for index, group in enumerate(groups[:4], start=1):
        out.append({
            "id": f"t{index}",
            "value": group["values"],
            "context": f"row group: {group['label']}",
            "chunk_id": group.get("chunk_id"),
        })
    return out


def calculator_node(state: Any) -> dict:
    """数值计算节点（Phase 8 接入主循环，Phase 9 升级为程序执行器）。

    流程（Phase 9 / 9.7 / 10）：
    1. **候选收集**：
       - 年份键控（growth_pct / difference）：四要素校验通过的证据值
         （高置信，``anchor_score=0``）+ 检索文档的年份锚定多候选扫描；
       - 实体键控（cross_entity_diff / argmax_relay）：校验证据 +
         实体锚定扫描（实体提及 + 指标同义词 + 年份三信号最近邻）；
       - 短语键控（ratio，Phase 10）：分子/分母短语锚定的表格扫描；
    2. **程序执行**：:class:`ProgramExecutor` 枚举组合（≤16），单位归一
       （million/% 不跨量纲）+ 合理性剪枝，经 ``calc_expression``（PoT
       安全求值，走 ToolRegistry 保留 calc 工具族调用轨迹）得出 primary
       + ≤2 个 alternates 的多刻度答案（difference / fraction /
       percentage，对齐 FinQA 裸数值 GT）；
    3. 降级：无可用组合 → calculation_result 记 error，synthesizer 输出诊断。
    """
    return _calculator_impl(state, llm_programmer=None)


def make_calculator_node(llm_programmer=None):
    """构造 calculator 节点工厂：注入 LLM 程序生成器（Phase 10，可降级）。

    ``llm_programmer`` 需提供 ``available`` 属性与
    ``generate(query, candidates) -> Optional[dict]`` 方法（见
    :class:`verifin.tools.llm_programmer.LLMProgramGenerator`）。注入且可用时：
    确定性模板路径先行，失败（无 spec / 执行 error）→ LLM 生成 FinQA DSL
    程序（编号候选值上下文）→ :func:`execute_dsl` 多步求值；LLM 不可用 /
    输出非法时行为与 Phase 9 完全一致。
    """
    def _node(state: Any) -> dict:
        return _calculator_impl(state, llm_programmer=llm_programmer)

    return _node


def _calculator_impl(state: Any, llm_programmer=None) -> dict:
    from verifin.tools.evidence import EvidenceExtractor
    from verifin.tools.program_executor import (
        ProgramExecutor,
        ProgramSpec,
        ValueCandidate,
    )

    s = _as_dict(state)
    spec = s.get("calculation_spec") or {}
    spec_metric = spec.get("metric")
    program_spec = ProgramSpec(
        kind=str(spec.get("kind") or "growth_pct"),
        base_period=str(spec.get("base_period") or ""),
        target_period=str(spec.get("target_period") or ""),
        metric=spec_metric,
        entity=spec.get("entity"),
        reversed=bool(spec.get("reversed")),
        entity_a=spec.get("entity_a"),
        entity_b=spec.get("entity_b"),
        entities=list(spec.get("entities") or []),
        second_metric=spec.get("second_metric"),
        numerator=spec.get("numerator"),
        denominator=spec.get("denominator"),
        periods=list(spec.get("periods") or []),
    )
    # Phase 12：均值流按 periods 区间逐年扫描；growth/difference 流 periods
    # 为空 → 行为不变（base/target 两年份）
    scan_years = tuple(
        dict.fromkeys(
            y for y in (
                *(program_spec.periods or []),
                program_spec.base_period,
                program_spec.target_period,
            ) if y
        )
    )

    def _registry_evaluator(expression: str) -> dict:
        out = _call_tool("calc_expression", expr=expression)
        if isinstance(out, dict):
            return out
        # 工具未注册 → 领域直接回退（节点可独立运行与单测）
        from verifin.tools.calculator import calc_expression

        return calc_expression(expression)

    # Phase 9.7：实体键控模板（跨实体差值 / argmax 接力）走独立候选路径
    if program_spec.kind in ("cross_entity_diff", "argmax_relay"):
        candidates = _entity_keyed_candidates(s, program_spec)
        program = ProgramExecutor(evaluator=_registry_evaluator).execute(
            program_spec, candidates
        )
        return _finish_calculation(s, program, llm_programmer, _registry_evaluator)

    # Phase 10：比率模板（"what percentage of X are Y"）走短语键控候选路径
    if program_spec.kind == "ratio":
        candidates = _ratio_candidates(s, program_spec)
        program = ProgramExecutor(evaluator=_registry_evaluator).execute(
            program_spec, candidates
        )
        return _finish_calculation(s, program, llm_programmer, _registry_evaluator)

    # Phase 10：LLM 程序生成路径（无确定性模板可检测时；llm_programmer
    # 未注入 / 不可用时直接走模板降级诊断，行为与 Phase 9 一致）
    if program_spec.kind == "llm_program":
        return _run_llm_program(s, llm_programmer, _registry_evaluator)

    # 1a. 校验通过证据 → 候选（指标匹配的 flag 优先）
    candidates: Dict[str, List[ValueCandidate]] = {}
    # 来源可信度（Phase 6 同款口径）：计算流证据候选对低可信来源
    # （新闻稿 vs 年报申报文件）加锚分惩罚——同一指标的数值冲突时
    # 申报口径优先（合成套件新闻稿 vs 年报冲突样本的预期行为）
    source_types: Dict[str, str] = {}
    for doc in [d for d in (s.get("retrieved_docs") or []) if isinstance(d, dict)]:
        meta = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
        if meta.get("source_type"):
            source_types[str(doc.get("chunk_id"))] = str(meta["source_type"])
    for flag in (s.get("verify_flags") or {}).values():
        if not isinstance(flag, dict):
            continue
        flag_claim = flag.get("claim") if isinstance(flag.get("claim"), dict) else {}
        metric_matched = bool(spec_metric and flag_claim.get("metric") == spec_metric)
        for result in flag.get("results") or []:
            if (
                isinstance(result, dict)
                and result.get("passed")
                and result.get("value") is not None
            ):
                period = str(result.get("period") or "")
                if not period:
                    continue
                source_penalty = (
                    0.5 if source_types.get(str(result.get("chunk_id"))) == "news" else 0.0
                )
                cand = ValueCandidate(
                    value=float(result["value"]),
                    unit=result.get("unit"),
                    period=period,
                    chunk_id=result.get("chunk_id"),
                    # anchor 刻度（Phase 9 校准）：0.0 表格行 metric 命中 +
                    # 年份在行标签（结构最强）< 0.5 表格行词重叠/父文档补全
                    # metric 命中 < 1.0 散文证据 metric 命中（无行标签结构
                    # 验证，仅四要素校验）< 5.0 散文无 metric 命中 < 50+ 距离兜底
                    anchor_score=(1.0 if metric_matched else 5.0) + source_penalty,
                )
                # 证据 period 可能是 "FY2023"/"2013-2023" 形态 → 子串归桶
                matched = [
                    year for year in scan_years
                    if year and year in period
                ]
                if matched:
                    for year in matched:
                        candidates.setdefault(year, []).append(cand)
                else:
                    candidates.setdefault(period, []).append(cand)

    # 1b. 检索文档候选扫描：表格感知（列年份对齐/行标签年份 + 指标匹配 +
    #     行标签 × 问题词重叠）优先，字符距离锚定兜底
    query_text = _latest_user_query(s)
    # Phase 12.1：chunk 级指标门控——与问题**零实词重叠**的 chunk 不产生
    # 候选。年份锚定兜底（collect_year_values）会从任意含年份的 chunk 抓数，
    # 跨公司无关文档（如包装销售散文对税务问题）的候选全为噪声；
    # 金标 chunk 的行标签/章节语境几乎总含问题指标词（实测 ADBE 表 4 词
    # 命中 vs IP 散文 0 命中）
    query_word_tokens = EvidenceExtractor._content_tokens(query_text)
    for doc in [d for d in (s.get("retrieved_docs") or []) if isinstance(d, dict)]:
        content = str(doc.get("content") or "")
        if not (EvidenceExtractor._content_tokens(content) & query_word_tokens):
            continue
        for year in scan_years:
            if not year:
                continue
            table_items = EvidenceExtractor.collect_table_year_values(
                content, year, spec_metric=spec_metric, query=query_text
            )
            distance_items = EvidenceExtractor.collect_year_values(content, year)
            scans = [(item, item.get("anchor_score", 50.0)) for item in table_items]
            scans += [(item, 50.0 + item.get("anchor_score", 0.0)) for item in distance_items]
            for item, anchor in scans:
                bucket = candidates.setdefault(year, [])
                if any(
                    cand.value == item["value"] and cand.unit == item["unit"]
                    for cand in bucket
                ):
                    continue
                bucket.append(
                    ValueCandidate(
                        value=item["value"],
                        unit=item["unit"],
                        period=year,
                        chunk_id=doc.get("chunk_id"),
                        anchor_score=anchor,
                    )
                )

    # 2. 父文档补全（small-to-big，Phase 9）：表格数值所在 chunk 的 BM25 信号弱
    #    （纯数字行），直接检索常漏——对已检索文档拉取全部 chunk 的**表格感知**
    #    候选（+0.5 来源惩罚：弱于直接检索的同质量候选，但 metric 命中的
    #    表格行仍强于散文抽取值）。
    #    补全顺序（Phase 9 修正）：按「问题实词 × 该文档已检索 chunk 内容」的
    #    重叠度排序——合并检索后的文档序号不反映相关度（金标文档可能排到
    #    40+ 名外），问题关键词命中的文档优先补全。
    from verifin.tools.evidence import EvidenceExtractor as _EE

    doc_chunks: Dict[str, str] = {}
    for doc in [d for d in (s.get("retrieved_docs") or []) if isinstance(d, dict)]:
        meta = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
        doc_id = str(meta.get("doc_id") or "")
        if doc_id:
            doc_chunks[doc_id] = doc_chunks.get(doc_id, "") + " " + str(doc.get("content") or "")
    query_content_tokens = _EE._content_tokens(query_text)
    doc_ids = sorted(
        doc_chunks,
        key=lambda d: -len(_EE._content_tokens(doc_chunks[d]) & query_content_tokens),
    )
    for doc_id in doc_ids[:10]:  # 补全文档数上限（防候选爆炸）
        expanded = _call_tool("expand_document", doc_id=doc_id)
        if not isinstance(expanded, list):
            continue  # 工具未注册（如单元测试）→ 跳过
        for chunk in expanded:
            if not isinstance(chunk, dict):
                continue
            content = str(chunk.get("content") or "")
            for year in scan_years:
                if not year:
                    continue
                for item in EvidenceExtractor.collect_table_year_values(
                    content, year, spec_metric=spec_metric, query=query_text
                ):
                    bucket = candidates.setdefault(year, [])
                    if any(
                        cand.value == item["value"] and cand.unit == item["unit"]
                        for cand in bucket
                    ):
                        continue
                    bucket.append(
                        ValueCandidate(
                            value=item["value"],
                            unit=item["unit"],
                            period=year,
                            chunk_id=chunk.get("chunk_id"),
                            # 父文档补全来源惩罚 +0.25（弱于直接检索的同质量
                            # 候选，但不得抹平语义层级：问题词重叠命中的补全行
                            # （0.5+0.25）仍应强于泛化词典命中的直接行（1.0））
                            anchor_score=0.25 + item.get("anchor_score", 0.0),
                        )
                    )

    # 3. 程序执行（PoT 求值经 ToolRegistry，保留 calc 工具族轨迹）
    program = ProgramExecutor(evaluator=_registry_evaluator).execute(
        program_spec, candidates
    )
    return _finish_calculation(s, program, llm_programmer, _registry_evaluator)


def _run_llm_program(s: dict, llm_programmer, evaluator) -> dict:
    """Phase 10 LLM 程序生成路径：编号候选 → LLM 生成 DSL → execute_dsl。

    任何环节失败（programmer 未注入 / 候选为空 / LLM 输出非法 / 求值
    出错）→ calculation_result 记 error，synthesizer 输出诊断（与模板
    降级口径一致）。
    """
    from verifin.tools.program_executor import execute_dsl

    query_text = _latest_user_query(s)

    def _fail(error: str) -> dict:
        return _hook(
            s, "calculator", None,
            calculation_result={"kind": "llm_program", "error": error},
            next_step="synthesizer",
        )

    if llm_programmer is None or not getattr(llm_programmer, "available", False):
        return _fail("llm programmer unavailable; no deterministic template matched")
    candidates = _llm_program_candidates(s)
    if not candidates:
        return _fail("no numeric candidates collected from retrieved documents")
    generated = llm_programmer.generate(query_text, candidates)
    if not generated:
        return _fail("llm program generation failed validation; falling back")
    out = execute_dsl(generated["program"], generated["bindings"], evaluator=evaluator)
    if out.get("error") or out.get("value") is None:
        return _fail(f"dsl execution failed: {out.get('error')}")
    # 证据 chunk：程序实际引用的候选（vN / tN）来源
    import re as _re

    referenced_ids = set(_re.findall(r"\b[vt]\d+\b", generated["program"]))
    referenced = [c for c in candidates if c["id"] in referenced_ids]
    if not referenced:
        referenced = candidates[:2]
    result = {
        "kind": "llm_program",
        "program": generated["program"],
        "expression": out.get("expression"),
        "value": out.get("value"),
        "unit": None,
        "steps": out.get("steps"),
        "evidence": [c.get("chunk_id") for c in referenced if c.get("chunk_id")],
    }
    return _hook(
        s, "calculator", None,
        hook_extra={"expression": result["expression"]},
        calculation_result=result,
        next_step="synthesizer",
    )


def _finish_calculation(s: dict, program, llm_programmer, evaluator) -> dict:
    """模板执行收尾：成功直接返回；失败且 LLM programmer 可用 → LLM 兜底。"""
    result = program.to_dict()
    if not result.get("error") or llm_programmer is None or not getattr(
        llm_programmer, "available", False
    ):
        return _hook(
            s, "calculator", None,
            hook_extra={"expression": program.expression} if program.expression else None,
            calculation_result=result,
            next_step="synthesizer",
        )
    # Phase 10：确定性模板失败 → LLM 程序生成兜底（可降级）
    return _run_llm_program(s, llm_programmer, evaluator)


def replanner_node(state: Any) -> dict:
    """带记忆的重规划节点（Phase 5 修正版）：委托 :class:`Replanner` 完成
    失败诊断 → 策略选择 → 模板化子任务生成。

    - 失败记忆持久化到 ``state.failure_history``（同一问题的重试链内累积，
      新提问由 planner 重置）；
    - 四要素校验失败的 chunk 写入 ``state.exclude_chunk_ids``（下轮检索过滤）；
    - 不继续（上限耗尽 / 重复失败无新信号）→ 人性化中文诊断 + END。
    """
    s = _as_dict(state)
    query = _latest_user_query(s)
    retry_count: int = s.get("retry_count", 0)
    max_retries: int = s.get("max_retries", 3)
    sub_tasks: List[str] = list(s.get("sub_tasks") or [])
    if not sub_tasks:
        return _hook(s, "replanner", query, next_step="end")

    index: int = s.get("current_sub_task_index", 0)
    failed_task = sub_tasks[index] if 0 <= index < len(sub_tasks) else query
    failed_flag = (s.get("verify_flags") or {}).get(failed_task) or {}
    if not isinstance(failed_flag, dict):
        failed_flag = {}

    store = FailureMemoryStore.from_history(s.get("failure_history") or [])
    plan = Replanner(failure_store=store).replan(
        sub_task=failed_task,
        verify_flags=failed_flag,
        retry_count=retry_count,
        max_retries=max_retries,
    )
    failure_history = store.to_history()

    if not plan["continue"]:
        logger.warning("重规划终止：{}", plan["reason"])
        # 失败路径也给出人性化中文诊断（不经过 synthesizer，直接作为终态消息）
        answer = "\n".join(_failure_message_lines(s))
        return _hook(
            s, "replanner", query,
            messages=[{"role": "assistant", "content": answer}],
            failure_history=failure_history,
            next_step="end",
        )

    new_task = plan["new_sub_task"]
    # Phase 11：重复子任务检测——新任务与既有 sub_tasks 文本相同说明重规划
    # 原地打转（无新检索方向，实测失败样本连续生成相同任务浪费轮次并污染
    # 失败记忆）→ 视为无新信号，走终止路径（人性化诊断 + END）
    if new_task and new_task in sub_tasks:
        logger.warning("重规划终止：新子任务与既有任务重复（原地打转）")
        answer = "\n".join(_failure_message_lines(s))
        return _hook(
            s, "replanner", query,
            messages=[{"role": "assistant", "content": answer}],
            failure_history=failure_history,
            hook_extra={"replan_strategy": "duplicate_task_terminated"},
            next_step="end",
        )
    new_index = len(sub_tasks)  # 新任务追加到队尾并切到它
    sub_tasks.append(new_task)
    # 失败 chunk 累积排除（去重保序）
    excluded = list(
        dict.fromkeys(
            list(s.get("exclude_chunk_ids") or []) + list(plan.get("exclude_chunk_ids") or [])
        )
    )
    return _hook(
        s, "replanner", query,
        hook_extra={"replan_strategy": plan["strategy"], "replan_reason": plan["reason"]},
        sub_tasks=sub_tasks,
        current_sub_task_index=new_index,
        retry_count=retry_count + 1,
        failure_history=failure_history,
        exclude_chunk_ids=excluded,
        next_step="retriever",
    )


_MISMATCH_RE = re.compile(
    r"^([a-z_]+):\s*expected '([^']*)'(?:,\s*got '([^']*)')?$", re.IGNORECASE
)


def _failure_message_lines(s: dict) -> List[str]:
    """把校验失败诊断为人性化中文消息（无技术黑话；去重，至多 3 条 + 冲突/降级提示）。"""
    flags: Dict[str, Any] = s.get("verify_flags") or {}
    lines: List[str] = []
    emitted: set = set()
    conflict_major = False
    for flag in flags.values():
        if not isinstance(flag, dict):
            continue
        conflict = flag.get("conflict")
        if isinstance(conflict, dict) and conflict.get("level") == "major":
            conflict_major = True
        for result in flag.get("results") or []:
            if not isinstance(result, dict) or result.get("passed"):
                continue
            for mismatch in result.get("mismatches") or []:
                match = _MISMATCH_RE.match(str(mismatch))
                if not match:
                    continue
                pillar, expected, got = match.group(1), match.group(2), match.group(3)
                key = (pillar, expected, got)
                if key in emitted:
                    continue
                emitted.add(key)
                lines.append(
                    friendly_message(
                        f"MISMATCH_{pillar.upper()}", expected=expected, got=got
                    )
                )
                if len(lines) >= 3:
                    break
            if len(lines) >= 3:
                break
        if len(lines) >= 3:
            break
    if conflict_major:
        lines.append(friendly_message("CONFLICT_MAJOR"))
    if s.get("retry_count", 0) >= s.get("max_retries", 3):
        lines.append(friendly_message("RETRY_EXHAUSTED"))
    if not lines:
        lines.append(friendly_message("NO_EVIDENCE"))
    return lines


def _calculation_line(s: dict) -> Optional[str]:
    """Phase 8：calculation_result 有值时生成计算答案行（数值 + 表达式 + 基期/比较期）。"""
    calc = s.get("calculation_result") or {}
    if not isinstance(calc, dict) or calc.get("value") is None:
        return None
    base = calc.get("base") or {}
    target = calc.get("target") or {}
    return (
        f"Calculated result ({calc.get('kind')}): {calc['value']}% "
        f"[expression: {calc.get('expression')}] "
        f"[base {base.get('value')} → target {target.get('value')}]"
    )


def synthesizer_node(state: Any) -> dict:
    """最终合成：仅绑定``results[].passed``通过的证据，生成带引用与校验裁决的答案。

    Phase 3 兼容：verify_flags 无 results 结构但有 passed 标记时，绑定全部检索文档。
    Phase 8：calculation_result 有值时首行输出计算答案（证据值绑定计算）。
    """
    s = _as_dict(state)
    flags: Dict[str, Any] = s.get("verify_flags") or {}
    docs: List[dict] = [doc for doc in (s.get("retrieved_docs") or []) if isinstance(doc, dict)]

    accepted_ids = {
        result.get("chunk_id")
        for flag in flags.values()
        if isinstance(flag, dict)
        for result in (flag.get("results") or [])
        if isinstance(result, dict) and result.get("passed")
    }
    verdicts = [
        flag.get("decision")
        for flag in flags.values()
        if isinstance(flag, dict) and flag.get("decision")
    ]
    verdict = verdicts[-1] if verdicts else "REJECT"
    calc_line = _calculation_line(s)

    if accepted_ids:
        verified_docs = [doc for doc in docs if doc.get("chunk_id") in accepted_ids]
    elif any(
        flag.get("passed") for flag in flags.values() if isinstance(flag, dict)
    ):
        verified_docs = docs
    else:
        verified_docs = []

    if verified_docs:
        lines = ["Based on verified evidence:", f"- verification verdict: {verdict}"]
        # Phase 6：偏好确认前置（如「已按您偏好的「年报」口径呈现结果。」）
        confirmation = preference_confirmation(s.get("dialog_state") or {})
        if confirmation:
            lines.insert(0, confirmation)
        if calc_line:
            lines.insert(0, calc_line)
        for doc in verified_docs:
            source = (
                (doc.get("metadata") or {}).get("doc_id")
                or doc.get("chunk_id", "unknown")
            )
            lines.append(f"- {doc.get('content', '')[:160]} [doc: {source}]")
        answer = "\n".join(lines)
    elif calc_line:
        # 计算完成但无可绑定文档（极端情况）：仍输出证据值计算答案
        answer = calc_line
    else:
        # 失败路径：人性化中文诊断（无技术黑话）
        answer = "\n".join(_failure_message_lines(s))
    return _hook(
        s, "synthesizer", None,
        messages=[{"role": "assistant", "content": answer}],
        next_step="end",
    )