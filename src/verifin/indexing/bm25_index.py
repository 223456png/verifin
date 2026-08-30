"""BM25 稀疏检索索引（rank_bm25.BM25Okapi）+ pickle 持久化。"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import List, Optional, Tuple, Union

from rank_bm25 import BM25Okapi

from verifin.schemas import DocumentChunk

PathLike = Union[str, Path]

_TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> List[str]:
    """小写 Unicode 字母数字词分词（与 chunker/embedding 的 token 口径一致）。"""
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """BM25Okapi 稀疏检索索引。

    在内存中构建，可 pickle 持久化 (corpus, chunk_ids, index) 三元组。
    Phase 2 的稀疏/稠密融合将以本类作为稀疏路入口。
    """

    def __init__(self) -> None:
        self.index: Optional[BM25Okapi] = None
        self.chunk_ids: List[str] = []
        self.corpus: List[str] = []

    def build(self, chunks: List[DocumentChunk]) -> None:
        """用 chunk 内容构建 BM25 索引。"""
        self.corpus = [chunk.content for chunk in chunks]
        self.chunk_ids = [chunk.chunk_id for chunk in chunks]
        if self.corpus:
            self.index = BM25Okapi([tokenize(text) for text in self.corpus])
        else:
            self.index = None

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """返回按 BM25 分数降序的 ``[(chunk_id, score)]``，最多 top_k 条。"""
        if self.index is None or not self.chunk_ids:
            return []
        scores = self.index.get_scores(tokenize(query))
        ordered = sorted(
            enumerate(scores), key=lambda item: item[1], reverse=True
        )[:top_k]
        return [(self.chunk_ids[i], float(score)) for i, score in ordered]

    def save(self, path: PathLike) -> None:
        """pickle 持久化 (corpus, chunk_ids, index) 三元组。"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as fh:
            pickle.dump((self.corpus, self.chunk_ids, self.index), fh)

    @classmethod
    def load(cls, path: PathLike) -> "BM25Index":
        """从 pickle 文件恢复索引实例。

        注意：仅用于加载本项目自产的可信产物，切勿加载不可信来源的 pickle。
        """
        with Path(path).open("rb") as fh:
            corpus, chunk_ids, index = pickle.load(fh)
        instance = cls()
        instance.corpus = corpus
        instance.chunk_ids = chunk_ids
        instance.index = index
        return instance

    def __len__(self) -> int:
        return len(self.chunk_ids)