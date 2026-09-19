"""Cross-Encoder Reranker 封装：懒加载 + 降级透传。

设计目标（Phase 2 设计 Decisions #4/#5）：
- 默认模型 cross-encoder/ms-marco-MiniLM-L-6-v2（~90MB，轻量精排）；
- sentence-transformers 为 optional 依赖（[rerank] extra）；
- 模型不可用时**降级透传**（保持融合顺序、零分），绝不抛出影响检索主流程。
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

from loguru import logger


class Reranker:
    """Cross-Encoder 精排器（懒加载，失败一次即缓存）。

    Attributes:
        model_name: HuggingFace cross-encoder 模型名或本地模型目录路径。
        available: 模型是否成功加载（降级时 False）。
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: Optional[str] = None) -> None:
        # None → 读环境变量 RERANKER_MODEL（支持本地目录离线加载），
        # 均未设置时用默认模型名。显式传参优先级最高（向后兼容）。
        self.model_name = (
            model_name
            if model_name is not None
            else os.environ.get("RERANKER_MODEL", self.DEFAULT_MODEL)
        )
        self._model = None
        self._available = False
        # Phase 12.2：加载失败缓存——失败一次后本进程不再重试。
        # 此前每次 rerank() 都重跑 _load_model()，HF 不可达时
        # （hub 内部重试 5×8s）每次检索卡 ~40s，100 样本评测被拖到
        # 34 分钟；网络恢复无法自动感知属可接受代价，进程内确定性
        # 降级优于可变长阻塞。
        self._load_attempted = False

    @property
    def available(self) -> bool:
        return self._available

    def _load_model(self) -> None:
        """懒加载模型；任何失败（依赖缺失/模型不可达）都降级，不抛出。

        每进程只尝试一次（成败皆缓存），防网络故障下的重复长阻塞。
        """
        if self._load_attempted:
            return
        self._load_attempted = True
        try:
            from sentence_transformers import CrossEncoder

            logger.info("加载 CrossEncoder 模型: {}", self.model_name)
            self._model = CrossEncoder(self.model_name)
            self._available = True
        except Exception as exc:
            logger.warning(
                "Reranker {} 不可用（{}），本进程后续检索跳过精排（失败已缓存，不再重试）",
                self.model_name, exc,
            )
            self._available = False

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, str]],
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """对候选 ``[(chunk_id, content)]`` 精排，返回按分数降序的 ``[(chunk_id, score)]``。

        模型不可用时按候选原序透传零分（降级契约）。
        """
        if not self._available:
            self._load_model()
        if not self._available:
            return [(chunk_id, 0.0) for chunk_id, _ in candidates[:top_k]]
        if not candidates:
            return []
        pairs = [(query, content) for _, content in candidates]
        try:
            raw_scores = self._model.predict(pairs, show_progress_bar=False)
        except Exception as exc:  # 预测期异常同样降级，保证不中断检索主流程
            logger.warning("Reranker 预测失败（{}），降级为透传", exc)
            self._available = False
            return [(chunk_id, 0.0) for chunk_id, _ in candidates[:top_k]]
        scored = [
            (candidates[i][0], float(raw_scores[i])) for i in range(len(candidates))
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]
