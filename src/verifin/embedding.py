"""可插拔 embedding。

- ``HashEmbedding``：离线确定性哈希词袋 embedding（默认，测试/CI/降级用）。
- ``get_embedding_function``：工厂，支持 ``bge``（sentence-transformers，可选依赖）。
"""

from __future__ import annotations

import hashlib
import re
from typing import List, Union

import numpy as np
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from loguru import logger


class HashEmbedding(EmbeddingFunction):
    """基于 SHA256 哈希词袋的确定性 embedding。

    每个 token 哈希投影到 dim 个桶之一并累加计数，最后 L2 归一。
    无网络依赖、完全可复现；相似度语义为「重叠词越多，余弦相似度越高」。
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    @classmethod
    def name(cls) -> str:
        """chromadb 1.x 注册表身份（注册时会以类对象调用，须为 classmethod）。"""
        return "verifin-hash"

    def get_config(self) -> dict:
        """chromadb 1.x 要求返回可序列化配置（用于集合重建校验）。"""
        return {"dim": self.dim}

    @classmethod
    def build_from_config(cls, config: dict) -> "HashEmbedding":
        """chromadb 1.x 要求支持从配置重建实例。"""
        return cls(dim=int(config.get("dim", 256)))

    def _embed(self, text: str) -> List[float]:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in re.findall(r"\w+", text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            vector[int(digest, 16) % self.dim] += 1.0
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector.tolist()

    def __call__(self, input: Union[str, List[str]]) -> Embeddings:
        """chromadb EmbeddingFunction 接口：单文本返回单向量，列表返回向量列表。"""
        if isinstance(input, str):
            return self._embed(input)  # type: ignore[return-value]
        return [self._embed(text) for text in input]


def get_embedding_function(name: str = "hash", **kwargs) -> EmbeddingFunction:
    """embedding 工厂。

    Args:
        name: ``hash``（默认离线）| ``bge``（需安装 optional 依赖 ``[bge]``）。
        **kwargs: 传给具体实现的参数（如 ``dim``、``model_name``）。

    Raises:
        ImportError: 选择 ``bge`` 但 sentence-transformers 未安装。
        ValueError: 未知的 embedding 名称。
    """
    if name == "hash":
        return HashEmbedding(**kwargs)
    if name in {"bge", "sentence-transformers", "st"}:
        try:
            from chromadb.utils.embedding_functions import (
                SentenceTransformerEmbeddingFunction,
            )
        except ImportError as exc:  # pragma: no cover - 取决于可选依赖
            raise ImportError(
                "BGE embedding 需要可选依赖，请执行：pip install -e '.[bge]'"
            ) from exc
        model_name = kwargs.pop("model_name", "BAAI/bge-small-en-v1.5")
        return SentenceTransformerEmbeddingFunction(model_name=model_name, **kwargs)
    raise ValueError(f"未知 embedding 名称: {name!r}")


def _register_hash_embedding() -> None:
    """导入期向 chromadb 注册 HashEmbedding，保证跨进程读取已建集合可用。"""
    try:
        from chromadb.utils.embedding_functions import register_embedding_function

        register_embedding_function(HashEmbedding)
    except Exception as exc:  # pragma: no cover - 注册失败仅退化为 legacy 方式，不影响功能
        logger.debug("HashEmbedding 注册到 chromadb 注册表失败: {}", exc)


_register_hash_embedding()