"""Phase 1 — ingestion + indexing 规格测试。

覆盖规格中的 6 个测试：
test_parser_pdf / test_chunker_by_heading / test_chunker_max_tokens /
test_bm25_index_build / test_bm25_search / test_vector_index_add_search

向量测试使用离线确定性 HashEmbedding，不依赖网络。
"""

from __future__ import annotations

import importlib
import importlib.util
import uuid
from pathlib import Path

import pytest

from verifin.embedding import HashEmbedding
from verifin.indexing.bm25_index import BM25Index
from verifin.indexing.vector_index import VectorIndex
from verifin.ingestion.chunker import MAX_TOKENS, chunk_documents, estimate_tokens
from verifin.ingestion.parser import parse_file
from verifin.schemas import DocumentChunk, ParsedPage


def _pymupdf():
    """按安装形态导入 pymupdf（新包名 pymupdf / 旧包名 fitz）。"""
    if importlib.util.find_spec("pymupdf"):
        return importlib.import_module("pymupdf")
    return importlib.import_module("fitz")


def _make_chunks(texts: list[str]) -> list[DocumentChunk]:
    """构造测试用 chunk 列表（直接构造，绕过 parser/chunker）。"""
    return [
        DocumentChunk(
            chunk_id=uuid.uuid4().hex,
            doc_id="d1",
            doc_name="report.md",
            content=text,
            page_num=1,
            section_title=None,
            section_level=None,
            metadata={},
        )
        for text in texts
    ]


@pytest.fixture()
def sample_pdf(tmp_path: Path) -> Path:
    """在临时目录生成一个含多行文本的示例 PDF。"""
    mupdf = _pymupdf()
    path = tmp_path / "sample.pdf"
    doc = mupdf.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "## Financial Results\n\nTotal revenue for fiscal 2024 was $12,400 million.",
    )
    doc.save(path)
    doc.close()
    return path


# 8.1 解析示例 PDF → 断言输出非空，包含文本
def test_parser_pdf(sample_pdf: Path) -> None:
    pages = parse_file(sample_pdf)
    assert pages, "PDF 解析输出不应为空"
    joined = "\n".join(page.text for page in pages)
    assert "Financial Results" in joined
    assert "revenue" in joined.lower()
    assert all(page.page_num >= 1 for page in pages)


# 8.2 输入带标题的文本 → 断言按标题切分正确
def test_chunker_by_heading() -> None:
    text = (
        "intro paragraph.\n\n"
        "# Annual Report\n\n"
        "## Financial Results\n\n"
        "### Revenue\nRevenue was 100.\n\n"
        "### Operating Margin\nMargin was 20%.\n\n"
        "## Balance Sheet\n\nAssets were 500.\n"
    )
    pages = [ParsedPage(page_num=1, text=text)]
    chunks = chunk_documents(pages, doc_id="d1", doc_name="report.md")
    titles = [(chunk.section_title, chunk.section_level) for chunk in chunks]
    assert len(chunks) == 4
    assert chunks[0].section_title is None  # 首个标题前的散段
    assert ("Annual Report > Financial Results > Revenue", 3) in titles
    assert ("Annual Report > Financial Results > Operating Margin", 3) in titles
    assert ("Annual Report > Balance Sheet", 2) in titles


# 8.3 输入长文本 → 断言每个 chunk ≤ 512 tokens
def test_chunker_max_tokens() -> None:
    long_text = "The company reported strong results this quarter. " * 2000
    pages = [ParsedPage(page_num=1, text=long_text)]
    chunks = chunk_documents(pages, doc_id="d1", doc_name="long.txt")
    assert len(chunks) > 1, "长文本应被二次切分为多个 chunk"
    for chunk in chunks:
        assert estimate_tokens(chunk.content) <= MAX_TOKENS


# 8.4 构建 BM25 索引 → 断言索引非空
def test_bm25_index_build() -> None:
    chunks = _make_chunks(
        ["operating margin was 30.7 percent", "gross margin was 44.1 percent"]
    )
    index = BM25Index()
    index.build(chunks)
    assert index.index is not None
    assert len(index.chunk_ids) == 2
    assert len(index.corpus) == len(index.chunk_ids)


# 8.5 搜索查询 → 断言返回结果数 = top_k
def test_bm25_search() -> None:
    texts = [
        f"Document {i} discusses operating margin and quarterly revenue for segment {i}."
        for i in range(8)
    ]
    chunks = _make_chunks(texts)
    index = BM25Index()
    index.build(chunks)
    results = index.search("operating margin", top_k=3)
    assert len(results) == 3, "返回结果数应等于 top_k"
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True), "分数应降序"
    assert all(chunk_id in index.chunk_ids for chunk_id, _ in results)


# 8.6 添加 chunk → 搜索 → 断言返回正确结果
def test_vector_index_add_search(tmp_path: Path) -> None:
    chunks = _make_chunks(
        [
            "Apple fiscal 2024 operating margin was 30.7 percent.",
            "Apple fiscal 2023 gross margin was 44.1 percent.",
            "Orange Inc fiscal 2024 revenue grew to 12.4 billion dollars.",
        ]
    )
    index = VectorIndex(persist_dir=str(tmp_path), embedding=HashEmbedding(dim=384))
    index.add_chunks(chunks)
    assert index.count() == 3
    results = index.search("Apple 2024 operating margin", top_k=1)
    assert len(results) == 1
    assert results[0][0] == chunks[0].chunk_id
