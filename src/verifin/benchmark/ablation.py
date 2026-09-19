"""消融实验：5 组配置矩阵 + AblationRunner 运行与摘要。

矩阵语义（Phase 7 §一 三个核心问题）：
- ``no_verifier``：答案与完整系统的差距 = **Verifier 的准确率贡献**；
- ``no_replanner``：Verifier REJECT 后不重规划直接结束，差距 = **Replanner 挽救率**；
- ``no_preferences``：每轮独立无记忆，多轮子集差距 = **偏好记忆的多轮增益**；
- ``no_dense``：仅 BM25 召回，差距 = **稠密检索路的贡献**。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from verifin.benchmark.metrics import summarize_results
from verifin.benchmark.runner import FULL_CONFIG, BenchmarkRunner

ABLATION_CONFIGS: Dict[str, Dict[str, bool]] = {
    "full": dict(FULL_CONFIG),
    "no_verifier": {"verifier": False, "replanner": False, "preferences": True, "dense": True},
    "no_replanner": {"verifier": True, "replanner": False, "preferences": True, "dense": True},
    "no_preferences": {"verifier": True, "replanner": True, "preferences": False, "dense": True},
    "no_dense": {"verifier": True, "replanner": True, "preferences": True, "dense": False},
}

CONFIG_LABELS = {
    "full": "完整系统",
    "no_verifier": "去掉 Verifier",
    "no_replanner": "去掉 Replanner",
    "no_preferences": "去掉偏好记忆",
    "no_dense": "仅 BM25（无 Dense）",
}


class AblationRunner:
    """顺序执行全部消融配置并产出摘要。"""

    def __init__(
        self,
        dataset: Any,
        output_dir: Path,
        multi_turn: bool = False,
        checkpoint_every: int = 10,
        index_dir: Optional[Path] = None,
    ) -> None:
        self.dataset = dataset
        self.output_dir = Path(output_dir)
        self.multi_turn = multi_turn
        self.checkpoint_every = checkpoint_every
        self.index_dir = index_dir

    def run_all(
        self, max_samples: Optional[int] = None, resume: bool = False
    ) -> Dict[str, List[dict]]:
        """运行所有消融配置，返回 ``{config_name: results}``。"""
        results: Dict[str, List[dict]] = {}
        for name, config in ABLATION_CONFIGS.items():
            print(f"运行消融配置: {name}（{CONFIG_LABELS.get(name, name)}）")
            runner = BenchmarkRunner(
                config_name=name,
                dataset=self.dataset,
                config=config,
                output_dir=self.output_dir,
                multi_turn=self.multi_turn,
                checkpoint_every=self.checkpoint_every,
                index_dir=self.index_dir,
            )
            results[name] = runner.run_all(max_samples=max_samples, resume=resume)
        return results

    def summarize(self, results: Dict[str, List[dict]]) -> Dict[str, dict]:
        """生成消融对比摘要（逐配置 4 指标 + 错误归因 + 重规划计数）。"""
        return {
            name: summarize_results(rows)
            for name, rows in results.items()
        }
