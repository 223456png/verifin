"""数据加载器：容错解析 ICAIF-24 风格数据集与裸文本文件。

支持格式：
- ``.jsonl`` / ``.json``：ICAIF-24 Finance RAG Challenge 常见字段
  （html_text / pre_text + post_text + table_ori / text / content）；
- ``.txt`` / ``.md``：裸文本直接读取。
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, List, Optional, Union

from loguru import logger

from verifin.schemas import ParsedDocument

PathLike = Union[str, Path]

_JSON_SUFFIXES = {".json", ".jsonl"}
_TEXT_SUFFIXES = {".txt", ".md", ".text"}
_PDF_SUFFIX = ".pdf"

_TAG_RE = re.compile(r"<[^>]+>")
_ENTITY_MAP = {
    "&amp;": "&",
    "&lt;": "<",
    "&gt;": ">",
    "&quot;": '"',
    "&#39;": "'",
    "&apos;": "'",
    "&nbsp;": " ",
}


def _html_to_text(html: str) -> str:
    """轻量 HTML → 文本（去标签 + 基础实体反转义）。"""
    text = _TAG_RE.sub(" ", html)
    for entity, char in _ENTITY_MAP.items():
        text = text.replace(entity, char)
    return text


def _field_to_lines(value: Any) -> List[str]:
    """字段值可为 str 或 str 列表，统一为文本行列表。"""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return [str(value)]


def _serialize_table(table: Any) -> str:
    """表格（list[list[str]]）序列化为 markdown 管道行（行首补 ``|``）。

    行首管道符保证数据行（首列为行标签，如 ``net revenue | 5829``）也能被
    :func:`verifin.tools.evidence.parse_markdown_tables` 解析为表格结构，
    供表格感知证据抽取按「行标签 + 年份列」定位数值（Phase 9）。
    """
    if not isinstance(table, list):
        return ""
    lines: List[str] = []
    for row in table:
        if isinstance(row, (list, tuple)):
            cells = ["" if cell is None else str(cell) for cell in row]
            lines.append("| " + " | ".join(cells))
        else:
            lines.append("| " + str(row))
    return "\n".join(lines)


def _extract_text(record: dict) -> Optional[str]:
    """从 ICAIF-24 风格记录中提取正文文本。"""
    if record.get("html_text"):
        return _html_to_text(str(record["html_text"]))
    parts: List[str] = []
    for field in ("pre_text", "post_text", "text", "content"):
        value = record.get(field)
        if value:
            raw = "\n".join(_field_to_lines(value))
            parts.append(_html_to_text(raw) if "<" in raw else raw)
    table = record.get("table_ori") or record.get("table")
    if table:
        parts.append("Table data:\n" + _serialize_table(table))
    return "\n\n".join(parts) if parts else None


def _read_records(path: Path) -> List[dict]:
    """读取 JSON/JSONL 文件为记录列表（整文件 JSON 或逐行 JSONL 均可）。"""
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return [data]
        return [item for item in data if isinstance(item, dict)]
    except json.JSONDecodeError:
        records: List[dict] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if isinstance(item, dict):
                    records.append(item)
            except json.JSONDecodeError as exc:
                logger.warning("跳过无法解析的行（{}）: {}", path.name, exc)
        return records


def load_dataset(data_dir: PathLike, limit: Optional[int] = None) -> List[ParsedDocument]:
    """加载数据集目录下的全部文档。

    Args:
        data_dir: 数据集目录（扫描 .json/.jsonl/.txt/.md，不递归）。
        limit: 最多加载的文档数（None 表示全部）。

    Returns:
        按文件顺序排列的 :class:`ParsedDocument` 列表（jsonl 内顺序保持）。
    """
    directory = Path(data_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"数据目录不存在: {directory}")
    files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and (
            path.suffix.lower() in _JSON_SUFFIXES | _TEXT_SUFFIXES
            or path.suffix.lower() == _PDF_SUFFIX
        )
    )
    documents: List[ParsedDocument] = []
    for path in files:
        if limit is not None and len(documents) >= limit:
            break
        suffix = path.suffix.lower()
        if suffix == _PDF_SUFFIX:
            # PDF 直接走 parser 的 pymupdf/pypdf 解析，按换页符合并分页文本
            from verifin.ingestion.parser import parse_file

            pages = parse_file(path)
            if not pages:
                logger.warning("PDF 解析为空，跳过: {}", path.name)
                continue
            documents.append(
                ParsedDocument(
                    doc_id=hashlib.sha256(path.name.encode("utf-8")).hexdigest()[:16],
                    doc_name=path.name,
                    raw_text="\f".join(page.text for page in pages),
                    format="pdf",
                    metadata={"source_file": path.name},
                )
            )
        elif suffix in _JSON_SUFFIXES:
            for record in _read_records(path):
                if limit is not None and len(documents) >= limit:
                    break
                text = _extract_text(record)
                if not text:
                    continue
                doc_id = str(record.get("id") or record.get("doc_id")
                              or hashlib.sha256(text.encode("utf-8")).hexdigest()[:16])
                documents.append(
                    ParsedDocument(
                        doc_id=doc_id,
                        doc_name=path.name,
                        raw_text=text,
                        format=suffix.lstrip("."),
                        metadata={"source_file": path.name},
                    )
                )
        else:
            documents.append(
                ParsedDocument(
                    doc_id=hashlib.sha256((path.name).encode("utf-8")).hexdigest()[:16],
                    doc_name=path.name,
                    raw_text=path.read_text(encoding="utf-8", errors="replace"),
                    format=suffix.lstrip("."),
                    metadata={"source_file": path.name},
                )
            )
    logger.info("加载完成：{} 个文档（目录 {})", len(documents), directory)
    return documents
