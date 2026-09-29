"""FastAPI 应用工厂与路由实现。

线程模型（重要）：chromadb 的 SQLite 句柄**线程亲和**（跨线程使用抛
ProgrammingError），而 FastAPI 请求跑在 ASGI 工作线程——因此
``VeriFinService`` 用**单线程执行器**承载「索引装配 + Agent 图执行」
全程（bootstrap 与 ask 都提交到同一线程），主线程仅做 HTTP 编解码。
"""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from verifin.benchmark.dataset import synth_chunks
from verifin.benchmark.runner import (
    RetrievalEnvironment,
    _register_retrieve_tool,
)
from verifin.core.graph import build_agent_graph
from verifin.core.runner import AgentRunner
from verifin.schemas import DocumentChunk
from verifin.tools.registry import register_builtin_tools


def _web_dir() -> Path:
    """Web Demo 单页目录：源码 checkout（src 布局回溯到项目根）优先，cwd 兜底。"""
    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent.parent.parent / "web",  # <root>/src/verifin/api/app.py
        Path.cwd() / "web",
    ]
    return next((c for c in candidates if c.is_dir()), candidates[0])


class AskRequest(BaseModel):
    """``POST /ask`` 请求体。"""

    query: str = Field(..., min_length=1, max_length=2000,
                       description="用户问题（支持中英文金融问题）")
    conversation_id: Optional[str] = Field(
        None, description="多轮会话 id（同一 id 复用对话状态与偏好记忆）"
    )


class EvidenceItem(BaseModel):
    chunk_id: Optional[str] = None
    doc_id: Optional[str] = None
    doc_name: Optional[str] = None
    content: Optional[str] = None
    value: Optional[float] = None
    unit: Optional[str] = None
    period: Optional[str] = None


class AskResponse(BaseModel):
    """``POST /ask`` 响应体。"""

    conversation_id: str
    answer: str
    calculation: Dict[str, Any] = Field(default_factory=dict)
    evidence: List[EvidenceItem] = Field(default_factory=list)
    trajectory: Dict[str, Any] = Field(default_factory=dict)
    dialog_state: Dict[str, Any] = Field(default_factory=dict)


class VeriFinService:
    """Agent 图的会话态服务封装（单线程执行器，线程安全）。

    Args:
        index_dir: 预构建索引目录（``scripts/build_index.py`` 产物）；
            None 时用内置合成语料（零外部依赖演示模式）。
        dense: 混合检索是否启用 Dense 路（False 仅 BM25）。
        graph_factory: 图工厂（测试注入用；默认生产图）。
    """

    def __init__(
        self,
        index_dir: Optional[Path] = None,
        dense: bool = True,
        graph_factory: Optional[Callable] = None,
    ) -> None:
        self._index_dir = index_dir
        self._dense = dense
        self._graph_factory = graph_factory or build_agent_graph
        self._lock = threading.Lock()
        self._threads: Dict[str, str] = {}
        self._known_entities: List[str] = []
        # 单线程执行器：chromadb SQLite 线程亲和 → 装配与执行必须同线程
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="verifin-agent")
        self.graph, self.corpus = self._executor.submit(self._bootstrap).result()

    def _bootstrap(self):
        """在工作线程内装配检索环境 + 注册工具 + 编译图（线程亲和要求）。"""
        if self._index_dir is not None:
            env = RetrievalEnvironment.from_persist_dir(Path(self._index_dir))
            corpus = {"source": "persisted-index", "path": str(self._index_dir)}
            chunks = None
        else:
            chunks: List[DocumentChunk] = synth_chunks()
            env = RetrievalEnvironment(chunks, workspace=Path(".verifin_api_env"))
            corpus = {"source": "synthetic", "chunks": len(chunks)}
        register_builtin_tools()
        _register_retrieve_tool(env, dense=self._dense)
        self._known_entities = self._collect_entities(chunks)
        return self._graph_factory(), corpus

    @staticmethod
    def _collect_entities(chunks) -> List[str]:
        """从语料抽已知实体拼写表（查询归一化的大小写修正用）。

        合成语料实体集中在 content 首字母大写短语；persisted 索引路径
        暂不修正（英文问法本身可命中），后续可从 metadata 实体列扩充。
        """
        if not chunks:
            return []
        from collections import Counter

        from verifin.tools.evidence import extract_entity_candidates

        counter = Counter()
        for chunk in chunks:
            for name in extract_entity_candidates(chunk.content or ""):
                counter[name] += 1
        # 高频（≥2）且非单字母：语料里反复出现的实体名
        return [name for name, n in counter.most_common(50) if n >= 2 and len(name) >= 3]

    # ------------------------------------------------------------------

    def ask(self, query: str, conversation_id: Optional[str]) -> dict:
        """执行一轮问答（conversation_id 复用 thread_id → 多轮记忆）。

        入口先做查询归一化（中文指标词/年份/虚词 → 检索系统能懂的形态，
        已知实体大小写修正）——仅会话层生效，benchmark 口径不受影响。
        """
        from verifin.query_norm import normalize_query

        normalized = normalize_query(query, known_entities=self._known_entities)
        with self._lock:
            if conversation_id:
                thread_id = self._threads.setdefault(
                    conversation_id, f"conv-{uuid.uuid4().hex}"
                )
            else:
                conversation_id = uuid.uuid4().hex
                thread_id = f"conv-{uuid.uuid4().hex}"
                self._threads[conversation_id] = thread_id
        from concurrent.futures import TimeoutError as FutureTimeoutError
        try:
            final = self._executor.submit(
                self._run, normalized, thread_id
            ).result(timeout=120)
        except FutureTimeoutError:
            final = {
                "messages": [
                    {"role": "user", "content": query},
                    {
                        "role": "assistant",
                        "content": (
                            "本轮处理超时（超过 120 秒安全上限），已终止。"
                            "请尝试更明确的问题表述（公司名 + 年份 + 指标）后重试。"
                        ),
                    },
                ]
            }
        return self._to_response(conversation_id, final)

    def _run(self, query: str, thread_id: str) -> dict:
        return AgentRunner(self.graph).run(query, thread_id=thread_id)

    @property
    def session_count(self) -> int:
        with self._lock:
            return len(self._threads)

    # ------------------------------------------------------------------

    @staticmethod
    def _to_response(conversation_id: str, final: dict) -> dict:
        """最终状态 → API 响应（答案/计算/证据/轨迹）。"""
        answers = [
            m for m in (final.get("messages") or [])
            if isinstance(m, dict) and m.get("role") == "assistant"
        ]
        answer = str(answers[-1]["content"]) if answers else ""

        # 证据：verify_flags 中 passed 的 chunk（含抽取值；content/doc 信息
        # 从 retrieved_docs 按 chunk_id 补齐——verify 结果只带 chunk_id）
        chunk_meta: Dict[str, dict] = {}
        for doc in (final.get("retrieved_docs") or []):
            if not isinstance(doc, dict):
                continue
            meta = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
            chunk_meta[str(doc.get("chunk_id"))] = {
                "doc_id": meta.get("doc_id"),
                "doc_name": meta.get("doc_name"),
                "content": str(doc.get("content") or "")[:400],
            }
        evidence: List[dict] = []
        seen_chunks = set()
        verdict = ""
        for flag in (final.get("verify_flags") or {}).values():
            if not isinstance(flag, dict):
                continue
            if flag.get("decision"):
                verdict = str(flag.get("decision"))
            for result in flag.get("results") or []:
                if not isinstance(result, dict):
                    continue
                chunk_id = result.get("chunk_id")
                if chunk_id in seen_chunks:
                    continue
                if result.get("value") is None and not result.get("passed"):
                    # 无数值且未通过（缺失字段的中性记录）不进证据卡
                    continue
                seen_chunks.add(chunk_id)
                meta = chunk_meta.get(str(chunk_id), {})
                evidence.append({
                    "chunk_id": chunk_id,
                    "doc_id": meta.get("doc_id"),
                    "doc_name": meta.get("doc_name"),
                    "content": meta.get("content"),
                    "value": result.get("value"),
                    "unit": result.get("unit"),
                    "period": result.get("period"),
                    # 徽章跟随后端裁决：通过校验 = 采用，否则 = 排除展示
                    "status": "accepted" if result.get("passed") else "excluded",
                })

        hooks = final.get("hooks") or []
        trajectory = {
            "node_path": [h.get("node") for h in hooks if isinstance(h, dict)],
            "steps": len(hooks),
            "tool_calls": [
                {"tool": getattr(call, "tool_name", None),
                 "args": getattr(call, "args", None)}
                for call in (final.get("tool_call_history") or [])
            ],
        }
        dialog_state = dict(final.get("dialog_state") or {})
        dialog_state.setdefault("session_id", conversation_id)
        return {
            "conversation_id": conversation_id,
            "verdict": verdict or ("VERIFIED" if evidence else "REJECT"),
            "answer": answer,
            "calculation": final.get("calculation_result") or {},
            "evidence": evidence[:8],
            "trajectory": trajectory,
            "dialog_state": dialog_state,
        }


def _build_service(
    index_dir: Optional[Path] = None,
    dense: bool = True,
    graph_factory: Optional[Callable] = None,
) -> VeriFinService:
    """构建服务（在工作线程内装配真实检索环境 + 生产图）。"""
    return VeriFinService(index_dir=index_dir, dense=dense, graph_factory=graph_factory)


def create_app(
    index_dir: Optional[Path] = None,
    dense: bool = True,
    service: Optional[VeriFinService] = None,
) -> FastAPI:
    """应用工厂（测试可注入 service / index_dir）。"""
    app = FastAPI(
        title="VeriFin API",
        version="0.1.0",
        description="金融证据校验 Agent（LangGraph + 混合检索 + 程序执行器）",
    )
    svc = service or _build_service(index_dir=index_dir, dense=dense)
    app.state.service = svc

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "corpus": svc.corpus,
            "sessions": svc.session_count,
        }

    @app.post("/ask", response_model=AskResponse)
    def ask(payload: AskRequest) -> JSONResponse:
        if not payload.query.strip():
            raise HTTPException(status_code=422, detail="query 不能为空")
        result = svc.ask(payload.query, payload.conversation_id)
        return JSONResponse(result)

    @app.get("/")
    def index():
        page = _web_dir() / "index.html"
        if page.is_file():
            return FileResponse(page)
        return {
            "name": "VeriFin API",
            "docs": "/docs",
            "endpoints": ["/health", "/ask"],
        }

    return app


def main() -> None:  # pragma: no cover - 本地启动入口
    """``python -m verifin.api.app`` 启动开发服务器。"""
    import os

    import uvicorn

    index_dir = os.environ.get("VERIFIN_INDEX_DIR")
    app = create_app(
        index_dir=Path(index_dir) if index_dir else None,
        dense=os.environ.get("VERIFIN_DENSE", "1") not in ("0", "false"),
    )
    uvicorn.run(app, host=os.environ.get("VERIFIN_HOST", "127.0.0.1"),
                port=int(os.environ.get("VERIFIN_PORT", "8000")))


if __name__ == "__main__":  # pragma: no cover
    main()
