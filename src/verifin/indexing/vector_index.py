"""ChromaDB 持久化稠密向量索引（可插拔 embedding）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple, Union

import chromadb

from verifin.embedding import HashEmbedding
from verifin.schemas import DocumentChunk

PathLike = Union[str, Path]

_ADD_BATCH_SIZE = 1000  # Chroma 单批写入上限保护（实际限额 5461）


class VectorIndex:
    """基于 ChromaDB PersistentClient 的向量索引。

    Attributes:
        collection: Chroma collection，默认 ``verifin_chunks``，
            使用注入的 embedding function（默认离线 HashEmbedding）。
    """

    def __init__(
        self,
        persist_dir: PathLike = "chroma_db",
        collection_name: str = "verifin_chunks",
        embedding=None,
    ) -> None:
        self.persist_dir = str(persist_dir)
        self.embedding = embedding or HashEmbedding()
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding,
        )

    @staticmethod
    def _meta(chunk: DocumentChunk) -> dict:
        """转换为 Chroma 元数据（值仅允许 str/int/float/bool，过滤 None）。"""
        meta: dict = {"doc_id": chunk.doc_id, "doc_name": chunk.doc_name}
        if chunk.page_num is not None:
            meta["page_num"] = chunk.page_num
        if chunk.section_title:
            meta["section_title"] = chunk.section_title
        if chunk.section_level is not None:
            meta["section_level"] = chunk.section_level
        if chunk.metadata:
            meta["extra_json"] = json.dumps(chunk.metadata, ensure_ascii=False)
        return meta

    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """添加 chunk 到向量库：chunk_id 为 id、content 为文档、元数据随附。

        大语料分批写入：Chroma 单批上限 5461 条（Phase 8 真实 FinQA 全量构建暴露），
        按 1000 条/批切分，避免一次性 add 超限。
        """
        if not chunks:
            return
        for start in range(0, len(chunks), _ADD_BATCH_SIZE):
            batch = chunks[start:start + _ADD_BATCH_SIZE]
            self.collection.add(
                ids=[chunk.chunk_id for chunk in batch],
                documents=[chunk.content for chunk in batch],
                metadatas=[self._meta(chunk) for chunk in batch],
            )

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """返回 ``[(chunk_id, distance)]``，距离越小越相似。"""
        count = self.collection.count()
        if count == 0:
            return []
        result = self.collection.query(
            query_texts=[query], n_results=min(top_k, count)
        )
        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        return list(zip(ids, [float(d) for d in distances], strict=False))

    def count(self) -> int:
        return self.collection.count()
