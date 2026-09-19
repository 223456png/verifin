"""SQLite 元数据存储：检索后过滤与展示的数据源。

使用 stdlib sqlite3，零额外依赖；Phase 4 的证据存储将与本库同库扩展。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import List, Optional, Union

from verifin.schemas import DocumentChunk

PathLike = Union[str, Path]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    doc_name TEXT NOT NULL,
    page_num INTEGER,
    section_title TEXT,
    section_level INTEGER,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    content TEXT NOT NULL
);
"""

_COLUMNS = (
    "chunk_id, doc_id, doc_name, page_num, section_title, "
    "section_level, metadata_json, content"
)


class MetadataStore:
    """chunk 元数据存储：upsert / 单查 / 过滤 / 清空。"""

    def __init__(self, path: PathLike = "verifin_metadata.db") -> None:
        self.path = str(path)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def __enter__(self) -> "MetadataStore":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def upsert_chunks(self, chunks: List[DocumentChunk]) -> None:
        """批量写入（INSERT OR REPLACE，按 chunk_id 幂等）。"""
        rows = [
            (
                chunk.chunk_id,
                chunk.doc_id,
                chunk.doc_name,
                chunk.page_num,
                chunk.section_title,
                chunk.section_level,
                json.dumps(chunk.metadata, ensure_ascii=False),
                chunk.content,
            )
            for chunk in chunks
        ]
        self._conn.executemany(
            f"INSERT OR REPLACE INTO chunks ({_COLUMNS}) VALUES (?,?,?,?,?,?,?,?)",
            rows,
        )
        self._conn.commit()

    @staticmethod
    def _row_to_chunk(row: tuple) -> DocumentChunk:
        chunk_id, doc_id, doc_name, page_num, title, level, meta_json, content = row
        return DocumentChunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            doc_name=doc_name,
            content=content,
            page_num=page_num,
            section_title=title,
            section_level=level,
            metadata=json.loads(meta_json) if meta_json else {},
        )

    def get(self, chunk_id: str) -> Optional[DocumentChunk]:
        """按 chunk_id 单查。"""
        row = self._conn.execute(
            f"SELECT {_COLUMNS} FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        return self._row_to_chunk(row) if row else None

    def filter(
        self,
        doc_id: Optional[str] = None,
        page_num: Optional[int] = None,
        section_title: Optional[str] = None,
    ) -> List[DocumentChunk]:
        """按 doc_id / 页码 / 章节标题过滤（AND 语义，None 表示不过滤）。"""
        conditions: List[str] = []
        params: List[Union[str, int]] = []
        if doc_id is not None:
            conditions.append("doc_id = ?")
            params.append(doc_id)
        if page_num is not None:
            conditions.append("page_num = ?")
            params.append(page_num)
        if section_title is not None:
            conditions.append("section_title = ?")
            params.append(section_title)
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = self._conn.execute(
            f"SELECT {_COLUMNS} FROM chunks{where}", tuple(params)
        ).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def clear(self) -> None:
        """清空 chunks 表。"""
        self._conn.execute("DELETE FROM chunks")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __len__(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
