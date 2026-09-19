"""文档解析：PDF（pymupdf 主 / pypdf 降级回退）与纯文本/Markdown。

输出：按页面组织的 :class:`ParsedPage` 列表。
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from typing import List, Tuple, Union

from loguru import logger

from verifin.schemas import ParsedDocument, ParsedPage

PathLike = Union[str, Path]


def _load_pymupdf():
    """按安装形态导入 pymupdf（新包名 pymupdf / 旧包名 fitz）。"""
    if importlib.util.find_spec("pymupdf"):
        return importlib.import_module("pymupdf")
    if importlib.util.find_spec("fitz"):
        return importlib.import_module("fitz")
    raise ImportError("未安装 pymupdf/fitz")


def _parse_pdf_pymupdf(path: Path) -> List[ParsedPage]:
    """用 pymupdf 抽取每页文本 + 尽力而为的表格文本（TSV 行）。"""
    mupdf = _load_pymupdf()
    pages: List[ParsedPage] = []
    with mupdf.open(str(path)) as doc:
        for index, page in enumerate(doc, start=1):
            lines: List[str] = [page.get_text("text").strip()]
            try:  # 表格文本为尽力而为能力，失败不影响主流程
                for table in page.find_tables():
                    for row in table.extract():
                        cells = ["" if cell is None else str(cell) for cell in row]
                        lines.append(" | ".join(cells))
            except Exception as exc:  # pragma: no cover - 防御性
                logger.debug("表格抽取跳过（第 {} 页）: {}", index, exc)
            text = "\n".join(line for line in lines if line.strip())
            if text:
                pages.append(ParsedPage(page_num=index, text=text))
    return pages


def _parse_pdf_pypdf(path: Path) -> List[ParsedPage]:
    """pypdf 降级路径：逐页抽取纯文本。"""
    from pypdf import PdfReader

    pages: List[ParsedPage] = []
    reader = PdfReader(str(path))
    for index, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(ParsedPage(page_num=index, text=text))
    return pages


def _split_pages(text: str) -> List[ParsedPage]:
    """纯文本按换页符分页；无换页符时整体视为第 1 页。"""
    return [
        ParsedPage(page_num=index + 1, text=part.strip("\n"))
        for index, part in enumerate(text.split("\f"))
        if part.strip()
    ]


def parse_file(path: PathLike) -> List[ParsedPage]:
    """解析 PDF 或纯文本/Markdown 文件，返回分页文本列表。

    PDF 优先 pymupdf（文本 + 表格），ImportError 或解析异常时降级 pypdf。
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"文件不存在: {p}")
    if p.suffix.lower() == ".pdf":
        try:
            return _parse_pdf_pymupdf(p)
        except ImportError:
            logger.warning("pymupdf 不可用，降级 pypdf 解析 {}", p.name)
            return _parse_pdf_pypdf(p)
        except Exception as exc:
            logger.warning("pymupdf 解析失败（{}），降级 pypdf: {}", p.name, exc)
            return _parse_pdf_pypdf(p)
    text = p.read_text(encoding="utf-8", errors="replace")
    return _split_pages(text)


def parse_documents(
    documents: List[ParsedDocument],
) -> List[Tuple[ParsedDocument, List[ParsedPage]]]:
    """批量解析文档，返回 (文档, 分页文本) 配对列表，顺序与输入一致。"""
    return [(doc, _split_pages(doc.raw_text)) for doc in documents]
