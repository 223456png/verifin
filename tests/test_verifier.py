"""Phase 4 — 四要素校验器测试（tools/verifier.py）。"""

from __future__ import annotations

from verifin.schemas import Evidence, VerificationResult
from verifin.tools.evidence import EvidenceExtractor
from verifin.tools.verifier import Verifier, extract_claim

VERIFIER = Verifier()
EXTRACTOR = EvidenceExtractor()


def _evidence(content: str, doc_id: str = "nova-fy2024") -> Evidence:
    return EXTRACTOR.extract(
        {"chunk_id": "c1", "doc_id": doc_id, "doc_name": f"{doc_id}.pdf",
         "content": content, "metadata": {}}
    )


# 9.2.1 四要素全部匹配 → PASS
def test_verifier_all_match() -> None:
    claim = {"entity": "NovaTech", "period": "2024", "metric": "revenue"}
    evidence = _evidence("NovaTech revenue in 2024 was $12,400 million.")
    result = VERIFIER.verify(claim, evidence)
    assert isinstance(result, VerificationResult)
    assert result.passed is True
    assert result.entity_match and result.period_match and result.metric_match
    assert result.mismatches == [] and result.missing == []
    assert result.confidence == 1.0


# 9.2.2 Entity 不匹配 → REJECT
def test_verifier_entity_mismatch() -> None:
    claim = {"entity": "NovaTech", "period": "2024", "metric": "revenue"}
    evidence = _evidence("Helios Energy revenue was $500 million in 2024.")
    result = VERIFIER.verify(claim, evidence)
    assert result.passed is False
    assert any("entity" in item for item in result.mismatches)
    assert result.entity_match is False


# 9.2.3 Period 不匹配 → REJECT
def test_verifier_period_mismatch() -> None:
    claim = {"entity": "NovaTech", "period": "2024", "metric": "revenue"}
    evidence = _evidence("NovaTech revenue in 2023 was $10,000 million.")
    result = VERIFIER.verify(claim, evidence)
    assert result.passed is False
    assert any("period" in item for item in result.mismatches)
    assert result.period_match is False


# 9.2.4 同义词匹配（operating margin ↔ EBIT margin）→ PASS
def test_verifier_metric_synonym() -> None:
    claim = {"entity": "NovaTech", "period": "2024", "metric": "operating_margin"}
    evidence = _evidence("NovaTech EBIT margin was 15% in 2024.")
    result = VERIFIER.verify(claim, evidence)
    assert result.passed is True
    assert result.metric_match is True


# 9.2.5 指标错配（gross margin vs operating margin）→ REJECT
def test_verifier_metric_mismatch() -> None:
    claim = {"entity": "NovaTech", "period": "2024", "metric": "operating_margin"}
    evidence = _evidence("NovaTech gross margin was 40% in 2024.")
    result = VERIFIER.verify(claim, evidence)
    assert result.passed is False
    assert result.metric_match is False
    assert any("metric" in item for item in result.mismatches)


# 9.2.6 Definition 关键词重叠 ≥50% → PASS
def test_verifier_definition_match() -> None:
    claim = {
        "entity": "NovaTech",
        "period": "2024",
        "metric": "operating_margin",
        "definition": "adjusted EBITDA / revenue",
    }
    evidence = _evidence(
        "NovaTech operating margin, calculated as adjusted EBITDA divided by revenue, was 15% in 2024."
    )
    result = VERIFIER.verify(claim, evidence)
    assert result.passed is True
    assert result.definition_match is True


# 口径不一（Definition 无重叠）→ REJECT（design D4 增强覆盖）
def test_verifier_definition_mismatch() -> None:
    claim = {
        "entity": "NovaTech",
        "period": "2024",
        "metric": "operating_margin",
        "definition": "adjusted EBITDA / revenue",
    }
    evidence = _evidence(
        "NovaTech margin, calculated as GAAP net income over sales, was 15% in 2024."
    )
    result = VERIFIER.verify(claim, evidence)
    assert result.passed is False
    assert result.definition_match is False


# claim 提供而证据缺失 → missing 列表记录（批量校验输入形态也验证）
def test_verify_batch_and_claim_extraction() -> None:
    claim = extract_claim("What was NovaTech revenue in 2024?")
    assert claim["entity"] == "NovaTech"
    assert claim["period"] == "2024"
    assert claim["metric"] == "revenue"
    assert claim["definition"] is None

    evidence_list = [
        _evidence("NovaTech revenue in 2024 was $12,400 million."),
        _evidence("Helios Energy revenue in 2024 was $500 million."),
    ]
    results = VERIFIER.verify_batch(claim, evidence_list)
    assert [r.passed for r in results] == [True, False]
    assert results[1].mismatches  # entity 不匹配


def test_verify_claim_empty_claim_not_vacuously_passed() -> None:
    """四要素全空（如 claim="qwerty 123"）绝不能真空通过。

    回归：此前 `verify_claim` 对无任何可校验要素的 claim 返回 passed=True
    （checked=0 → mismatches 为空），MCP 调用方会误以为该 claim 已被四要素校验。
    """
    from verifin.tools.verifier import verify_claim

    empty = extract_claim("qwerty 123")
    assert all(empty.get(k) is None for k in ("entity", "period", "metric", "definition"))

    result = verify_claim(empty, {"chunk_id": "evidence", "content": "some random text"})
    assert result["passed"] is False
    assert result["confidence"] == 0.0
    assert any("no checkable elements" in m for m in result["mismatches"])
