"""多路召回融合：Reciprocal Rank Fusion（RRF）与加权融合。"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple


class RRFusion:
    """倒数排名融合（Reciprocal Rank Fusion）。

    score(d) = Σ_{i=1..n} 1 / (rank_i(d) + k)，k 为标准常数 60。
    仅使用各列表的排名信息，天然对分数量纲不敏感。
    """

    def __init__(self, k: int = 60) -> None:
        self.k = k

    def fuse(
        self,
        ranking_lists: Sequence[Sequence[Tuple[str, float]]],
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """融合多个已按分数降序的排序列表。

        Args:
            ranking_lists: 多个 ``[(chunk_id, score)]`` 列表（各自降序）。
            top_k: 返回条数。

        Returns:
            按 RRF 分降序的 ``[(chunk_id, rrf_score)]``；同分按 chunk_id 升序稳定排序。
        """
        scores: Dict[str, float] = defaultdict(float)
        for ranking in ranking_lists:
            for rank, (chunk_id, _) in enumerate(ranking, start=1):
                scores[chunk_id] += 1.0 / (rank + self.k)
        return sorted(
            scores.items(), key=lambda item: (-item[1], item[0])
        )[:top_k]


def minmax_normalize(values: Sequence[float], degenerate: float) -> List[float]:
    """min-max 归一化到 0-1。

    Args:
        values: 数值序列。
        degenerate: 全同分（分母为 0）时的取值——加权融合用 0.5（避免单元素
            列表被归零），最终展示分用 0.0（全同分即无区分度）。

    Returns:
        归一化后的列表；空序列原样返回。
    """
    if not values:
        return list(values)
    low, high = min(values), max(values)
    if high - low < 1e-12:
        return [degenerate] * len(values)
    return [(value - low) / (high - low) for value in values]


def _minmax(values: Sequence[float]) -> List[float]:
    """加权融合内部的列表归一化：退化值取 0.5（见 :func:`minmax_normalize`）。"""
    return minmax_normalize(values, degenerate=0.5)


def weighted_fuse(
    ranking_lists: Sequence[Sequence[Tuple[str, float]]],
    weights: Sequence[float],
    top_k: int = 10,
) -> List[Tuple[str, float]]:
    """加权融合：各列表分数 min-max 归一后按权重求和。

    Args:
        ranking_lists: 多个 ``[(chunk_id, score)]`` 列表（各自降序）。
        weights: 与列表一一对应的权重（调用方需保证顺序一致）。
        top_k: 返回条数。

    Returns:
        按加权分降序的 ``[(chunk_id, weighted_score)]``；同分按 chunk_id 升序稳定排序。
    """
    if len(ranking_lists) != len(weights):
        raise ValueError("ranking_lists 与 weights 长度必须一致")
    scores: Dict[str, float] = defaultdict(float)
    for ranking, weight in zip(ranking_lists, weights, strict=False):
        if not ranking:
            continue
        ids = [item[0] for item in ranking]
        normalized = _minmax([item[1] for item in ranking])
        for chunk_id, value in zip(ids, normalized, strict=False):
            scores[chunk_id] += weight * value
    return sorted(
        scores.items(), key=lambda item: (-item[1], item[0])
    )[:top_k]
