"""VeriFin 核心数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class ToolCall:
    """一次工具调用的记录（LLM function calling 与内部执行统一形态）。

    Attributes:
        tool_name: 工具名。
        args: 调用参数。
        call_id: 调用唯一标识（UUID），用于把结果回绑到调用。
    """

    tool_name: str
    args: dict
    call_id: str


@dataclass
class ToolResult:
    """一次工具调用的执行结果。

    Attributes:
        call_id: 回绑的 ToolCall.call_id。
        success: 是否成功执行。
        output: 工具实际返回数据（SearchResultSet / 校验结果 / float 等）。
        error: 失败原因（可选）。
        duration_ms: 执行耗时（毫秒）。
    """

    call_id: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0


@dataclass
class DocumentChunk:
    """一个不可再分的检索单元。

    Attributes:
        chunk_id: chunk 唯一标识（UUID hex）。
        doc_id: 所属文档唯一标识。
        doc_name: 文档文件名。
        content: 分块文本内容。
        page_num: 起始页码（文本文件为 1）。
        section_title: 所属章节标题（祖先标题链，以 " > " 连接）。
        section_level: 章节层级（1/2/3...，无标题时为 None）。
        metadata: 额外元数据（公司名、年份等；延续片含 is_continuation=True）。
    """

    chunk_id: str
    doc_id: str
    doc_name: str
    content: str
    page_num: Optional[int] = None
    section_title: Optional[str] = None
    section_level: Optional[int] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ParsedPage:
    """解析后的单页文本。"""

    page_num: int
    text: str


@dataclass
class ParsedDocument:
    """数据加载器产出的原始文档（解析前的内存表示）。"""

    doc_id: str
    doc_name: str
    raw_text: str
    format: str
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    """单条检索结果。

    Attributes:
        chunk_id: 命中 chunk 的 id。
        content: chunk 文本内容。
        score: 最终综合得分（min-max 归一化到 0-1）。
        bm25_score: BM25 原始分（未进稀疏路候选时为 None）。
        dense_score: 稠密相似度原始分（未进稠密路候选时为 None）。
        rerank_score: Reranker 原始分（未启用/降级时为 None）。
        metadata: chunk 元数据（doc_id/page_num/section_title 及扩展字段）。
    """

    chunk_id: str
    content: str
    score: float
    bm25_score: Optional[float] = None
    dense_score: Optional[float] = None
    rerank_score: Optional[float] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchResultSet:
    """一次检索的完整结果集。

    Attributes:
        query: 原始查询。
        results: 最终结果列表（按 score 降序）。
        total_count: 最终结果数。
        retrieval_time_ms: 整条流水线耗时（毫秒）。
        fusion_method: 融合方式（"rrf" | "weighted"）。
        reranker_used: 本次检索是否实际使用了 Reranker 精排。
    """

    query: str
    results: List[SearchResult]
    total_count: int
    retrieval_time_ms: float
    fusion_method: str
    reranker_used: bool


@dataclass
class Evidence:
    """结构化证据（从文档 chunk 中规则抽取，Phase 4）。"""

    chunk_id: str
    doc_id: str
    doc_name: str
    content: str
    page_num: Optional[int] = None
    section_title: Optional[str] = None

    # 四要素抽取结果（None 表示未抽取到）
    entity: Optional[str] = None
    period: Optional[str] = None
    metric: Optional[str] = None
    definition: Optional[str] = None

    # 数值抽取
    value: Optional[float] = None
    unit: Optional[str] = None  # "million" / "billion" / "%" / "M" / "B"

    # 置信度（已抽取要素数 / 4）
    extraction_confidence: float = 0.0


@dataclass
class VerificationResult:
    """单条证据的四要素校验结果。"""

    chunk_id: str
    passed: bool

    # 逐项校验结果（claim 未提供该要素时为 None）
    entity_match: Optional[bool] = None
    period_match: Optional[bool] = None
    metric_match: Optional[bool] = None
    definition_match: Optional[bool] = None

    # 不匹配详情（如 "metric: expected 'revenue', got 'cost'"）
    mismatches: List[str] = field(default_factory=list)
    # claim 要求但证据缺失的要素（如 "definition"）
    missing: List[str] = field(default_factory=list)

    # 综合置信度 = 命中要素数 / claim 提供要素数
    confidence: float = 0.0


@dataclass
class ExpressionResult:
    """PoT 计算器的求值结果。"""

    expression: str
    value: Optional[float] = None
    unit: Optional[str] = None
    error: Optional[str] = None
    is_valid: bool = False
    execution_time_ms: float = 0.0


@dataclass
class VerifyFlags:
    """写入 ``AgentState.verify_flags[sub_task]`` 的结构契约（Phase 4）。

    与 Phase 3 兼容：``passed`` 布尔语义不变（retriever 跳过 / synthesizer 绑定依赖它）。
    通常由 ``verify_claim_batch`` 直接产出 dict 形态，本类仅作文档与构造辅助。
    """

    claim: dict = field(default_factory=dict)
    results: List[dict] = field(default_factory=list)
    passed_count: int = 0
    total_count: int = 0
    decision: str = "REJECT"  # ACCEPT | PARTIAL | REJECT
    passed: bool = False
    confidence: float = 0.0