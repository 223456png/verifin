"""pytest 全局夹具：单测不加载真实重排/嵌入模型。

真实 bge embedding 与 cross-encoder reranker 需要下载数百 MB 权重，且
transformers 每次加载会联网做 ETag 检查（走代理极慢）。单测统一让
Reranker 走**降级透传**路径（mock ``_load_model`` 失败），保持毫秒级；
真实召回/精排效果由 benchmark（真实数据集消融，scripts/run_benchmark.py）
验证。生产路径（FastAPI / scripts）不受此夹具影响。
"""

from __future__ import annotations

import pytest

from verifin.retrieval.reranker import Reranker


@pytest.fixture(autouse=True)
def _disable_real_reranker(monkeypatch) -> None:
    """所有测试内 Reranker 懒加载一律失败 → 降级透传，不联网加载模型。"""

    def _fail_load(_self: Reranker) -> None:
        _self._available = False

    monkeypatch.setattr(Reranker, "_load_model", _fail_load)
    yield