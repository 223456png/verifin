"""Phase 7 — Benchmark 测试（数据加载 / 指标 / 消融 / 报告 / 小规模端到端）。"""

from __future__ import annotations

import json

import pytest

from verifin.benchmark.ablation import ABLATION_CONFIGS
from verifin.benchmark.dataset import SyntheticDataset, load_dataset
from verifin.benchmark.metrics import (
    accuracy,
    exact_match,
    generate_pseudo_gold,
    replan_hit_rate,
    tool_f1,
    tool_families,
)
from verifin.benchmark.report import ReportGenerator
from verifin.benchmark.runner import FULL_CONFIG, BenchmarkRunner, build_benchmark_graph
from verifin.tools.registry import ToolRegistry


@pytest.fixture(autouse=True)
def _clean_registry():
    ToolRegistry().reset()
    yield
    ToolRegistry().reset()


def _result(is_correct, replan, decision):
    return {
        "query": "q",
        "is_correct": is_correct,
        "trajectory": {
            "replan_triggered": replan,
            "verifier_decision": decision,
            "steps": 5,
            "tool_families": ["retrieve", "verify"],
        },
    }


# 1. 数据集加载：FinQA 真实格式解析 / 降级合成语料 / 三种数据集工厂
def test_dataset_loader(tmp_path) -> None:
    # 目录不存在 → 降级内置合成语料（20 个样本）
    synthetic = load_dataset("finqa", tmp_path / "missing")
    assert len(synthetic) == 20
    assert synthetic.is_synthetic is True
    assert {"query", "ground_truth", "gold_evidence"} <= set(synthetic[0])

    # 真实 FinQA 格式（qa.question / qa.exe_ans）容错解析
    records = [
        {"id": "d1", "qa": {"question": "What was revenue in 2024?", "exe_ans": "$12,000 million"}},
        {"id": "d2", "qa": {"question": "What was margin in 2023?", "exe_ans": "15%"}},
    ]
    (tmp_path / "finqa_test.json").write_text(json.dumps(records), encoding="utf-8")
    real = load_dataset("finqa", tmp_path)
    assert real.is_synthetic is False
    assert len(real) == 2
    assert real[0]["query"] == "What was revenue in 2024?"
    assert real[0]["ground_truth"] == "$12,000 million"
    assert real[0]["gold_evidence"][0]["doc_id"] == "d1"

    # ConvFinQA / TAT-QA 容错降级 + 未知数据集报错
    assert len(load_dataset("convfinqa", tmp_path / "nope")) == 20
    assert len(load_dataset("tatqa", tmp_path / "nope")) == 20
    assert len(load_dataset("synthetic", tmp_path)) == 20
    with pytest.raises(ValueError):
        load_dataset("unknown", tmp_path)


# 2. 数值归一化精确匹配（单位折算 + 1% 容差）
def test_exact_match_numeric() -> None:
    assert exact_match((10800.0, "million"), "$10,800 million") is True
    assert exact_match((12.4, "billion"), "$12,400 million") is True  # 亿→百万折算
    assert exact_match((11000.0, "million"), "$10,800 million") is False  # 差 1.85% > 1%
    assert exact_match((15.0, "%"), "15%") is True
    assert exact_match((12000.0, "million"), "15%") is False  # 单位量纲不一致
    assert exact_match(None, "$10,800 million") is False
    assert accuracy([]) == 0.0  # 空结果守卫
    assert accuracy([{"is_correct": True}, {"is_correct": False}]) == 0.5


# 3. Tool F1：工具族归一 + precision/recall/F1 计算
def test_tool_f1_calculation() -> None:
    assert tool_families(["verify_claim_batch", "retrieve"]) == {"retrieve", "verify"}
    assert tool_f1(["retrieve"], ["retrieve", "verify"]) == pytest.approx(2 / 3)
    assert tool_f1(["retrieve", "verify"], ["retrieve", "verify"]) == 1.0
    assert tool_f1([], ["retrieve"]) == 0.0
    assert tool_f1(["retrieve"], []) == 1.0  # 无标要求 → 不罚
    # pseudo-gold：增长类多跳问题要求 calc 工具族
    assert "calc" in generate_pseudo_gold("What was the percentage growth in revenue?")
    assert "calc" not in generate_pseudo_gold("What was revenue in 2024?")
    # Phase 10：比率类（portion/fraction/percent of）要求 calc
    assert "calc" in generate_pseudo_gold(
        "what portion of total facilities are leased facilities?"
    )
    assert "calc" in generate_pseudo_gold(
        "what fraction of the portfolio was fixed rate?"
    )
    assert "calc" in generate_pseudo_gold(
        "what percent of total revenue came from leasing?"
    )


# 4. 重规划有效率：Replan 触发样本中裁决非 REJECT 的比例（PARTIAL 计入）
def test_replan_hit_rate() -> None:
    results = [
        _result(True, True, "ACCEPT"),
        _result(False, True, "REJECT"),
        _result(True, False, "ACCEPT"),  # 未触发重规划 → 不参与分母
    ]
    assert replan_hit_rate(results) == 0.5
    assert replan_hit_rate([_result(True, True, "PARTIAL")]) == 1.0
    assert replan_hit_rate([_result(True, False, "ACCEPT")]) == 0.0
    assert replan_hit_rate([]) == 0.0


# 5. 消融配置有效性：5 组配置 + 每组可编译出可调用图
def test_ablation_configs() -> None:
    assert set(ABLATION_CONFIGS) == {
        "full", "no_verifier", "no_replanner", "no_preferences", "no_dense",
    }
    for name, config in ABLATION_CONFIGS.items():
        assert all(isinstance(value, bool) for value in config.values()), name
        graph = build_benchmark_graph(config)
        assert hasattr(graph, "invoke"), f"{name} 图应可执行"
    # 完整配置 == 生产图复用路径
    assert build_benchmark_graph(dict(FULL_CONFIG)) is not None


# 6. 报告生成：包含全部必要章节并写盘 report.md
def test_report_generation(tmp_path) -> None:
    configs = {
        name: {
            "accuracy": 0.7, "avg_steps": 4.2, "replan_hit_rate": 0.5,
            "tool_f1": 0.9, "total": 10, "correct": 7,
            "replanned": 2, "replan_hits": 1,
            "error_counts": {"retrieval_failure": 2, "calculation_error": 1},
            "multi_turn_accuracy": 0.8,
        }
        for name in ABLATION_CONFIGS
    }
    summary = {
        "dataset": "finqa", "dataset_source": "synthetic",
        "dataset_total": 20, "total_samples": 10,
        "multi_turn": True, "run_date": "2026-08-29",
        "configs": configs,
    }
    text = ReportGenerator(tmp_path).generate(summary)
    for section in ("实验配置", "消融实验结果", "关键结论", "Agent 行为指标", "诚实声明"):
        assert section in text
    assert "| 完整系统 |" in text and "| 仅 BM25（无 Dense） |" in text
    assert "70.0%" in text  # 准确率格式化
    assert (tmp_path / "report.md").is_file()


# 7. 小规模端到端：真实内存索引 → Agent 全流水线 5 样本 + 消融 2 样本 + 断点续跑
def test_small_end_to_end(tmp_path) -> None:
    dataset = SyntheticDataset()
    assert dataset.build_chunks(), "合成语料应有可索引 chunk"

    runner = BenchmarkRunner(
        config_name="full", dataset=dataset, config=dict(FULL_CONFIG),
        output_dir=tmp_path, checkpoint_every=2,
    )
    results = runner.run_all(max_samples=5)
    assert len(results) == 5
    for row in results:
        assert {"query", "answer", "ground_truth", "is_correct", "trajectory"} <= set(row)
        assert isinstance(row["is_correct"], bool)
        assert row["trajectory"]["steps"] > 0
        assert "retrieve" in row["trajectory"]["tool_families"]
    # 断点 + 最终结果文件
    assert (tmp_path / "full" / "checkpoint_000001.json").is_file()
    assert (tmp_path / "full" / "checkpoint_000003.json").is_file()
    assert (tmp_path / "full" / "results.json").is_file()

    # 断点续跑：同配置同目录 resume → 从检查点恢复（不再重跑前 4 个）
    runner2 = BenchmarkRunner(
        config_name="full", dataset=dataset, config=dict(FULL_CONFIG),
        output_dir=tmp_path, checkpoint_every=2,
    )
    resumed = runner2.run_all(max_samples=5, resume=True)
    assert len(resumed) == 5

    # 消融配置可运行（no_verifier：无验证直接合成）
    ablation_runner = BenchmarkRunner(
        config_name="no_verifier", dataset=dataset,
        config=ABLATION_CONFIGS["no_verifier"], output_dir=tmp_path,
    )
    ablated = ablation_runner.run_all(max_samples=2)
    assert len(ablated) == 2
    assert all(isinstance(row["is_correct"], bool) for row in ablated)
