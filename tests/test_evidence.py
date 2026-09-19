"""Phase 4 — Evidence 规则抽取测试（tools/evidence.py）。"""

from __future__ import annotations

import pytest

from verifin.schemas import Evidence
from verifin.tools.evidence import EvidenceExtractor

EXTRACTOR = EvidenceExtractor()


def _chunk(
    content: str,
    doc_id: str = "nova-fy2024",
    doc_name: str = "novatech-fy2024.pdf",
    **extra,
) -> dict:
    chunk = {
        "chunk_id": "c1",
        "doc_id": doc_id,
        "doc_name": doc_name,
        "content": content,
        "metadata": {},
    }
    chunk.update(extra)
    return chunk


# 9.1.1 从内容/文件名中提取公司名
def test_extract_entity() -> None:
    evidence = EXTRACTOR.extract(_chunk("NovaTech reported strong results."))
    assert isinstance(evidence, Evidence)
    assert evidence.entity == "NovaTech"
    # 多词公司名（Title Case）
    evidence2 = EXTRACTOR.extract(_chunk("Helios Energy capex plan for 2025."))
    assert evidence2.entity == "Helios Energy"
    # 内容无首字母大写词 → 回退 doc_name 前缀（helion-fy2023.pdf → Helion）
    evidence3 = EXTRACTOR.extract(
        _chunk("operating results improved.", doc_name="helion-fy2023.pdf")
    )
    assert evidence3.entity == "Helion"
    # 完全抽取不到 → None
    evidence4 = EXTRACTOR.extract(
        _chunk("operating results improved.", doc_name="filing.pdf")
    )
    assert evidence4.entity is None


# 9.1.2 年份模式（2024 / FY2024 / Q2 2023）
def test_extract_period() -> None:
    assert EXTRACTOR.extract(_chunk("Revenue in FY2024 grew.")).period == "FY2024"
    assert EXTRACTOR.extract(_chunk("Q2 2023 was strong.")).period == "Q2 2023"
    assert EXTRACTOR.extract(_chunk("In 2024, growth was 5%.")).period == "2024"
    assert EXTRACTOR.extract(_chunk("No period information.")).period is None


# 9.1.3 指标关键词匹配（canonical key）
def test_extract_metric() -> None:
    assert EXTRACTOR.extract(_chunk("Operating margin was 15%.")).metric == "operating_margin"
    assert EXTRACTOR.extract(_chunk("Revenue grew 5%.")).metric == "revenue"
    assert EXTRACTOR.extract(_chunk("EBITDA improved.")).metric == "EBITDA"
    assert EXTRACTOR.extract(_chunk("The board approved a plan.")).metric is None


# 9.1.4 数值抽取（$10,800 million → 10800.0, million；42.5% → 42.5, %）
def test_extract_value() -> None:
    evidence = EXTRACTOR.extract(_chunk("Revenue reached $10,800 million."))
    assert evidence.value == pytest.approx(10800.0)
    assert evidence.unit == "million"

    evidence2 = EXTRACTOR.extract(_chunk("Margin was 42.5% in 2024."))
    assert evidence2.value == pytest.approx(42.5)
    assert evidence2.unit == "%"

    evidence3 = EXTRACTOR.extract(_chunk("No numbers here."))
    assert evidence3.value is None and evidence3.unit is None
