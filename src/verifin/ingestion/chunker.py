"""智能分块：按 Markdown 标题层级切分，保留祖先标题链，512 token 上限。"""

from __future__ import annotations

import math
import re
import uuid
from typing import List, Optional, Tuple

from verifin.schemas import DocumentChunk, ParsedPage

#: 单个 chunk 的估算 token 上限（对应规格「每个 chunk ≤ 512 tokens」）
MAX_TOKENS = 512

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_WORD_RE = re.compile(r"\w+")


def estimate_tokens(text: str) -> int:
    """轻量 token 估算：词数 × 1.3 向上取整（对英文财报文本近似 BPE 口径）。"""
    return math.ceil(len(_WORD_RE.findall(text)) * 1.3)


def _merge_paragraphs(lines: List[str]) -> List[str]:
    """把行列表合并为段落列表（空行作为段落边界，段内保留换行结构）。"""
    paragraphs: List[str] = []
    current: List[str] = []
    for line in lines:
        if line.strip():
            current.append(line)
        elif current:
            paragraphs.append("\n".join(current))
            current = []
    if current:
        paragraphs.append("\n".join(current))
    return paragraphs


def _split_by_words(text: str) -> List[str]:
    """按词累计切分单个超限段落，保证每个片段 ≤ MAX_TOKENS。"""
    pieces: List[str] = []
    current: List[str] = []
    for word in text.split():
        candidate = " ".join(current + [word])
        if current and estimate_tokens(candidate) > MAX_TOKENS:
            pieces.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        pieces.append(" ".join(current))
    return pieces


def _emit_section(
    chunks: List[DocumentChunk],
    titles: List[str],
    level: Optional[int],
    paragraph_lines: List[str],
    page_num: int,
    doc_id: str,
    doc_name: str,
    meta: dict,
) -> None:
    """把一个标题节（或标题前的散段）按 token 上限切分为若干 chunk 并追加。"""
    paragraphs = _merge_paragraphs(paragraph_lines)
    if not paragraphs:
        return

    section_title = " > ".join(titles) if titles else None
    emitted = False

    def make_chunk(parts: List[str], continuation: bool) -> None:
        nonlocal emitted
        chunk_meta = dict(meta)
        if continuation:
            chunk_meta["is_continuation"] = True
        chunks.append(
            DocumentChunk(
                chunk_id=uuid.uuid4().hex,
                doc_id=doc_id,
                doc_name=doc_name,
                content="\n\n".join(parts),
                page_num=page_num,
                section_title=section_title,
                section_level=level,
                metadata=chunk_meta,
            )
        )
        emitted = True

    current: List[str] = []
    for para in paragraphs:
        if current and estimate_tokens("\n\n".join(current + [para])) > MAX_TOKENS:
            # 加入下一段会超限：先落盘当前累积段落，再从该段重新开始
            make_chunk(current, continuation=emitted)
            current = []
        if not current and estimate_tokens(para) > MAX_TOKENS:
            # 单段本身超限 → 按词硬切
            for piece in _split_by_words(para):
                make_chunk([piece], continuation=emitted)
            continue
        current.append(para)
    if current:
        make_chunk(current, continuation=emitted)


def chunk_documents(
    pages: List[ParsedPage],
    doc_id: str,
    doc_name: str,
    metadata: Optional[dict] = None,
) -> List[DocumentChunk]:
    """将解析后的页面列表切分为 :class:`DocumentChunk` 列表。

    规则：
    - 任意级别 Markdown 标题（#~######）触发切分；
    - 维护标题栈保留祖先标题链（如 ``Annual Report > Financial Results > Revenue``）；
    - 每个 chunk 估算 token 数 ≤ :data:`MAX_TOKENS`，超限的节按段落二次切分，
      延续片在 metadata 中记录 ``is_continuation: True``；
    - 标题层级（section_level）取当前节末段标题的层级。
    """
    meta = dict(metadata or {})
    chunks: List[DocumentChunk] = []
    stack: List[Tuple[int, str]] = []  # (level, title) 标题栈
    section_lines: List[str] = []
    section_page: Optional[int] = None

    def flush_section() -> None:
        nonlocal section_lines, section_page
        if not section_lines:
            return
        _emit_section(
            chunks,
            [title for _, title in stack],
            stack[-1][0] if stack else None,
            section_lines,
            section_page or 1,
            doc_id,
            doc_name,
            meta,
        )
        section_lines = []
        section_page = None

    lines = (
        (page.page_num, line)
        for page in pages
        for line in page.text.splitlines()
    )
    for page_num, line in lines:
        match = _HEADING_RE.match(line.strip())
        if match:
            flush_section()
            level = len(match.group(1))
            title = match.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            continue
        if line.strip():
            if section_page is None:
                section_page = page_num
            section_lines.append(line)
        elif section_lines:
            section_lines.append("")  # 空行作段落分隔标记
    flush_section()
    return chunks