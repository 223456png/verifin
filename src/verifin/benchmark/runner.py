"""批量评估运行器 + 消融变体图 + 内存检索环境。

设计要点（Phase 7 优化，对提示词的 4 处调整）：
1. **同步执行**：Agent 图执行（LangGraph invoke）本身是同步的，不做 asyncio 空壳
   包装；批量循环 + 断点续跑即可满足长任务可靠性；
2. **完整配置复用生产图**：``config == FULL_CONFIG`` 时直接使用
   :func:`verifin.core.graph.build_agent_graph`（消融基线 = 生产行为，
   这是消融结论成立的前提）；其余配置在本包内临时织入变体图，**不改核心图**；
3. **真实内存索引环境**：从 DocumentChunk 构建真实 BM25 + Chroma 向量 + SQLite
   元数据 + HybridRetriever（Reranker 无模型时按 Phase 2 契约自动降级），
   测试/演示全程跑真实检索流水线，不用 stub；
4. **多轮评估不用 checkpoint 线程**，改为显式携带 ``messages + dialog_state``
   到下一轮 initial_state——每轮 hooks/步数指标干净隔离，且更可调试。
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from langgraph.graph import END, StateGraph

from verifin.benchmark.metrics import (
    answer_relevancy,
    exact_match,
    faithfulness,
    ground_truth_in_text,
    tool_families,
    tool_f1,
)
from verifin.benchmark.trace import (
    classify_error,
    extract_answer_value,
    extract_trajectory,
)
from verifin.core.graph import build_agent_graph, route_after_replanner, route_after_verifier
from verifin.core.nodes import (
    _as_dict,
    _latest_user_query,
    calculator_node,
    planner_node,
    replanner_node,
    retriever_node,
    synthesizer_node,
    verifier_node,
)
from verifin.core.runner import AgentRunner
from verifin.core.state import AgentState
from verifin.embedding import get_embedding_function
from verifin.indexing.bm25_index import BM25Index
from verifin.indexing.metadata_store import MetadataStore
from verifin.indexing.vector_index import VectorIndex
from verifin.retrieval.hybrid_retriever import HybridRetriever, build_hybrid_retriever
from verifin.retrieval.reranker import Reranker
from verifin.schemas import DocumentChunk
from verifin.tools.registry import ToolRegistry, register_builtin_tools
from verifin.tools.verifier import extract_claim

FULL_CONFIG: Dict[str, bool] = {
    "verifier": True,
    "replanner": True,
    "preferences": True,
    "dense": True,
}


# ---------------------------------------------------------------------------
# 消融变体图（完整配置直接复用生产图）
# ---------------------------------------------------------------------------

def _planner_without_preferences(state: Any) -> dict:
    """去除偏好记忆的 planner 包装：每轮输入态清空 dialog_state（每轮独立）。"""
    if isinstance(state, dict):
        clone = dict(state)
        clone["dialog_state"] = {}
        return planner_node(clone)
    return planner_node(state.model_copy(update={"dialog_state": {}}))


def _synthesizer_without_verifier(state: Any) -> dict:
    """去除 Verifier 的合成包装：跳过四要素校验，把检索文档直接作为已通过证据。"""
    s = _as_dict(state)
    query = _latest_user_query(s)
    docs = [doc for doc in (s.get("retrieved_docs") or []) if isinstance(doc, dict)]
    s["verify_flags"] = {
        query: {
            "passed": True,
            "decision": "ACCEPT",
            "claim": extract_claim(query),
            "results": [
                {"chunk_id": doc.get("chunk_id"), "passed": True,
                 "mismatches": [], "missing": []}
                for doc in docs
            ],
        }
    }
    return synthesizer_node(s)


def _memory_saver():
    """按实测版本回退导入内存 checkpoint（与 core.graph 相同策略）。"""
    try:
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    except Exception:  # pragma: no cover - 版本回退
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()


def build_benchmark_graph(config: Dict[str, bool]):
    """按消融配置构建图；完整配置 == 生产图，其余为临时变体。"""
    config = dict(config)
    if config == FULL_CONFIG:
        return build_agent_graph()

    enable_verifier = bool(config.get("verifier", True))
    enable_replanner = bool(config.get("replanner", True)) and enable_verifier

    builder = StateGraph(AgentState)
    planner = (
        planner_node if config.get("preferences", True) else _planner_without_preferences
    )
    builder.add_node("planner", planner)
    builder.add_node("retriever", retriever_node)
    if enable_verifier:
        builder.add_node("verifier", verifier_node)
        builder.add_node("calculator", calculator_node)
    if enable_replanner:
        builder.add_node("replanner", replanner_node)
    builder.add_node(
        "synthesizer",
        synthesizer_node if enable_verifier else _synthesizer_without_verifier,
    )

    builder.set_entry_point("planner")
    builder.add_edge("planner", "retriever")
    if enable_verifier:
        builder.add_edge("retriever", "verifier")
        builder.add_edge("calculator", "synthesizer")
        builder.add_conditional_edges(
            "verifier",
            route_after_verifier,
            {
                "synthesizer": "synthesizer",
                "calculator": "calculator",
                "replanner": "replanner" if enable_replanner else "synthesizer",
                "retriever": "retriever",
                "end": END,
            },
        )
    else:
        builder.add_edge("retriever", "synthesizer")
    if enable_replanner:
        builder.add_conditional_edges(
            "replanner", route_after_replanner, {"retriever": "retriever", "end": END}
        )
    builder.add_edge("synthesizer", END)
    return builder.compile(checkpointer=_memory_saver())


# ---------------------------------------------------------------------------
# 内存检索环境（真实 BM25 + Dense + 融合，非 stub）
# ---------------------------------------------------------------------------

class RetrievalEnvironment:
    """从语料 chunk 构建的内存索引 + 检索工具（dense=False 时仅 BM25）。"""

    def __init__(
        self, chunks: List[DocumentChunk], workspace: Path, embedding: str = "hash"
    ) -> None:
        # 每次初始化使用唯一子目录：前一次运行的 SQLite/Chroma 句柄可能仍持有
        # 文件句柄，删除会触发 readonly 冲突——唯一目录天然幂等且可重复运行
        self.root = Path(workspace) / f".index_env-{uuid.uuid4().hex[:8]}"
        self.root.mkdir(parents=True, exist_ok=True)
        self.bm25 = BM25Index()
        self.bm25.build(chunks)
        self.metadata = MetadataStore(path=str(self.root / "metadata.db"))
        self.metadata.upsert_chunks(chunks)
        self.vector = VectorIndex(
            persist_dir=str(self.root / "chroma"),
            embedding=get_embedding_function(embedding),
        )
        self.vector.add_chunks(chunks)
        # Reranker 无 cross-encoder 模型时按 Phase 2 契约自动降级（跳过精排）
        self.hybrid = HybridRetriever(
            bm25_index=self.bm25,
            vector_index=self.vector,
            metadata_store=self.metadata,
            reranker=Reranker(),
        )

    @classmethod
    def from_persist_dir(cls, persist_dir: Path) -> "RetrievalEnvironment":
        """从 ``scripts/build_index.py`` 预构建的索引产物组装环境（真实数据集路径）。

        embedding 名从构建期 manifest.json 读取，reranker 启用（懒加载，模型
        缺失时按 Phase 2 契约自动降级透传，不影响回退路径）。
        """
        env = cls.__new__(cls)
        env.root = Path(persist_dir)
        env.hybrid = build_hybrid_retriever(persist_dir=persist_dir)
        env.bm25 = env.hybrid.bm25
        env.metadata = env.hybrid.metadata
        env.vector = env.hybrid.vector
        return env


def _register_retrieve_tool(env: RetrievalEnvironment, dense: bool) -> None:
    """把检索环境注册为 `retrieve` 工具（覆盖内置默认；dense=False 仅走 BM25）。

    同时注册 `expand_document`（small-to-big 文档补全）：检索 chunk 只定位到
    文档，证据抽取可拉取该文档全部 chunk——财报表格常被分块切断，表格数值
    所在 chunk 的 BM25 信号弱（纯数字行），父文档全量扫描是表格问答的
    标准补救（Phase 9）。
    """
    registry = ToolRegistry()

    def hybrid_retrieve(
        query: str, top_k: int = 10, use_reranker: bool = True,
        filter_metadata: Optional[dict] = None,
    ):
        return env.hybrid.search(
            query, top_k=top_k, use_reranker=use_reranker, filter_metadata=filter_metadata
        )

    def bm25_retrieve(
        query: str, top_k: int = 10, use_reranker: bool = True,
        filter_metadata: Optional[dict] = None,
    ) -> List[dict]:
        pool = max(top_k, 1) * 2
        hits = env.bm25.search(query, top_k=pool)
        docs: List[dict] = []
        for chunk_id, score in hits[:top_k]:
            chunk = env.metadata.get(chunk_id)
            if chunk is None:
                continue
            meta: dict = {"doc_id": chunk.doc_id, "doc_name": chunk.doc_name}
            meta.update(chunk.metadata or {})
            docs.append({
                "chunk_id": chunk_id,
                "content": chunk.content,
                "score": float(score),
                "metadata": meta,
            })
        return docs

    def expand_document(doc_id: str) -> List[dict]:
        """返回该文档的全部 chunk（父文档补全）。"""
        docs: List[dict] = []
        for chunk in env.metadata.filter(doc_id=str(doc_id)):
            meta: dict = {"doc_id": chunk.doc_id, "doc_name": chunk.doc_name}
            meta.update(chunk.metadata or {})
            docs.append({
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "metadata": meta,
            })
        return docs

    func = hybrid_retrieve if dense else bm25_retrieve
    mode = "hybrid" if dense else "bm25-only"
    registry.register(
        name="retrieve", func=func, description=f"stub-free {mode} retrieval env",
        schema={"type": "object"},
    )
    registry.register(
        name="expand_document", func=expand_document,
        description="parent-document expansion: all chunks of a doc",
        schema={"type": "object"},
    )


# ---------------------------------------------------------------------------
# 批量运行器（含断点续跑）
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """按配置批量评估数据集样本，产出轨迹结果 + 断点检查点。"""

    def __init__(
        self,
        config_name: str,
        dataset: Any,
        config: Dict[str, bool],
        output_dir: Path,
        multi_turn: bool = False,
        checkpoint_every: int = 10,
        index_dir: Optional[Path] = None,
    ) -> None:
        self.config_name = config_name
        self.dataset = dataset
        self.config = dict(config)
        self.output_dir = Path(output_dir) / config_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.multi_turn = multi_turn
        self.checkpoint_every = checkpoint_every
        self._carry: Dict[str, dict] = {}

        chunks = dataset.build_chunks()
        if chunks:
            # 合成/内置语料：真实内存索引（BM25 + Chroma + SQLite）
            self.env = RetrievalEnvironment(chunks, self.output_dir / ".index_env")
        elif index_dir is not None:
            # 真实数据集：加载 build_index.py 预构建的索引产物
            self.env = RetrievalEnvironment.from_persist_dir(index_dir)
        else:
            raise RuntimeError(
                "数据集无可索引语料且未指定 --index-dir；真实数据集请先运行 "
                "scripts/build_index.py --data-dir <dir> 构建索引，再经 --index-dir 指定"
            )
        self.graph = build_benchmark_graph(self.config)
        register_builtin_tools()
        _register_retrieve_tool(self.env, dense=bool(config.get("dense", True)))

    # ---- 批量执行 -------------------------------------------------------

    def run_all(
        self, max_samples: Optional[int] = None, resume: bool = False
    ) -> List[dict]:
        """批量运行；``resume=True`` 时从最新检查点续跑，跳过已完成样本。"""
        limit = min(max_samples or len(self.dataset), len(self.dataset))
        results, start = [], 0
        if resume:
            results, start = self._load_checkpoint()
            if start >= limit:
                return results
        for idx in range(start, limit):
            results.append(self.run_single(self.dataset[idx]))
            if (idx + 1) % self.checkpoint_every == 0:
                self._save_checkpoint(results, idx)
        self._save_results(results)
        return results

    # ---- 单样本 ----------------------------------------------------------

    def run_single(self, sample: dict) -> dict:
        """运行单个样本，返回含轨迹 / 数值答案 / 错误归因的完整结果 dict。

        多轮组（conversation_id）在 multi_turn 模式下手动携带上轮
        ``messages + dialog_state`` 到本轮 initial_state（每轮轨迹隔离）。
        """
        registry = ToolRegistry()
        before = len(registry.history())
        query = str(sample["query"])
        conversation_id = sample.get("conversation_id")

        if self.multi_turn and conversation_id:
            carry = self._carry.get(conversation_id) or {}
            previous_messages = list(carry.get("messages") or [])
            initial = AgentState(
                messages=previous_messages + [{"role": "user", "content": query}],
                dialog_state=dict(carry.get("dialog_state") or {}),
            )
            final = AgentRunner(self.graph).run(query, initial_state=initial)
        else:
            final = AgentRunner(self.graph).run(query)

        tools_used = [call.tool_name for call in registry.history()[before:]]
        for call in final.get("tool_call_history") or []:
            name = getattr(call, "tool_name", None)
            if name and name not in tools_used:
                tools_used.append(name)
        families_used = list(dict.fromkeys(tool_families(tools_used)))

        docs = [doc for doc in (final.get("retrieved_docs") or []) if isinstance(doc, dict)]
        retrieved_count = len(docs)
        # 文档级 gold 召回（FinQA gold 由 program 计算得出，GT 字面值常不出现
        # 在语料中，无法用「GT 文本命中」衡量；以 gold 文档是否被召回为准），
        # 再叠加 GT 字面值兜底（合成语料口径不变）
        gold_doc_ids = {
            str(item.get("doc_id"))
            for item in (sample.get("gold_evidence") or [])
            if isinstance(item, dict) and item.get("doc_id")
        }
        gt_in_retrieved = any(
            isinstance(doc.get("metadata"), dict)
            and str(doc["metadata"].get("doc_id")) in gold_doc_ids
            for doc in docs
        ) or any(
            ground_truth_in_text(str(sample["ground_truth"]), str(doc.get("content") or ""))
            for doc in docs
        )
        answer_candidates = extract_answer_value(final, query)
        answers = [m for m in final.get("messages") or [] if m.get("role") == "assistant"]
        answer = str(answers[-1]["content"]) if answers else ""
        ground_truth = str(sample["ground_truth"])

        # RAGAs 风格生成质量指标（规则近似）：
        # faithfulness 的证据池 = 通过四要素校验的 chunk 文本 + PoT 计算产物
        # （与验证闭环绑定，而非任意检索结果）
        calc = final.get("calculation_result") or {}
        calc_pool = []
        if isinstance(calc, dict):
            for key, unit_key in (
                ("value", "unit"), ("fraction", None), ("difference", "difference_unit"),
            ):
                if calc.get(key) is not None:
                    calc_pool.append((float(calc[key]), calc.get(unit_key)))
            for alternate in calc.get("alternates") or []:
                if not isinstance(alternate, dict):
                    continue
                if alternate.get("fraction") is not None:
                    calc_pool.append((float(alternate["fraction"]), None))
                if alternate.get("difference") is not None:
                    calc_pool.append(
                        (float(alternate["difference"]), alternate.get("difference_unit"))
                    )
        passed_ids = {
            str(result.get("chunk_id"))
            for flags in (final.get("verify_flags") or {}).values()
            if isinstance(flags, dict)
            for result in flags.get("results") or []
            if isinstance(result, dict) and result.get("passed") and result.get("chunk_id")
        }
        evidence_texts = [
            str(doc.get("content") or "")
            for doc in docs
            if str(doc.get("chunk_id")) in passed_ids
        ]

        result = {
            "query": query,
            "conversation_id": conversation_id,
            "answer": answer,
            "answer_value": [[value, unit] for value, unit in answer_candidates],
            "ground_truth": ground_truth,
            "gold_answer_type": sample.get("gold_answer_type") or "unknown",
            "is_correct": exact_match(answer_candidates, ground_truth),
            "tool_f1": tool_f1(families_used, _pseudo_gold_families(query)),
            "faithfulness": faithfulness(answer, evidence_texts, calc_pool),
            "answer_relevancy": answer_relevancy(query, answer),
            "trajectory": extract_trajectory(
                final, tools_used, families_used, retrieved_count,
                gt_in_retrieved, verifier_enabled=bool(self.config.get("verifier", True)),
            ),
            "verify_flags": final.get("verify_flags") or {},
            "dialog_state": final.get("dialog_state") or {},
        }
        result["error_type"] = classify_error(result)
        if self.multi_turn and conversation_id:
            self._carry[conversation_id] = {
                "messages": final.get("messages") or [],
                "dialog_state": final.get("dialog_state") or {},
            }
        return result

    # ---- 断点续跑 ---------------------------------------------------------

    def _save_checkpoint(self, results: List[dict], idx: int) -> None:
        """每 N 个样本保存中间结果（防长任务中断）。"""
        path = self.output_dir / f"checkpoint_{idx:06d}.json"
        path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    def _load_checkpoint(self):
        """从最新检查点恢复（返回已有结果与下一个索引）。"""
        files = sorted(self.output_dir.glob("checkpoint_*.json"))
        if not files:
            return [], 0
        latest = files[-1]
        results = json.loads(latest.read_text(encoding="utf-8"))
        next_idx = int(latest.stem.split("_")[1]) + 1
        return results, next_idx

    def _save_results(self, results: List[dict]) -> None:
        path = self.output_dir / "results.json"
        path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )


def _pseudo_gold_families(query: str) -> List[str]:
    """per-sample Tool F1 的 pseudo-gold 工具族（rules 同 metrics.generate_pseudo_gold）。"""
    from verifin.benchmark.metrics import generate_pseudo_gold

    return list(dict.fromkeys(tool_families(generate_pseudo_gold(query))))