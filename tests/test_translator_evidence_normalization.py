import re

from src.agents.business_translator import (
    _ensure_evidence_section,
    _is_valid_evidence_source,
    _validate_report,
)


def test_ensure_evidence_section_rebuilds_single_canonical_block():
    report = """## Resumen\n\nTexto base.\n\n## Evidencia usada\n\nevidence:\n{claim: \"legacy\", source: \"inline_token\"}\n\nArtifacts:\n- data/old_a.csv\n\n## Evidencia usada\n\nevidence:\n{claim: \"legacy2\", source: \"inline_token_2\"}\n\nArtifacts:\n- data/old_b.csv\n"""
    evidence_paths = ["data/metrics.json", "data/scored_rows.csv", "data/metrics.json"]

    normalized = _ensure_evidence_section(report, evidence_paths)

    assert len(re.findall(r"(?im)^##\s+Evidencia usada\s*$", normalized)) == 1
    assert normalized.count("Artifacts:") == 1
    assert "- data/metrics.json" in normalized
    assert "- data/scored_rows.csv" in normalized
    assert normalized.count("- data/metrics.json") == 1
    assert "{claim:" in normalized
    assert "source:" in normalized


def test_ensure_evidence_section_repairs_text_before_rebuild():
    report = "## Resumen\n\nNecesita revisi\u00c3\u00b3n manual\n"
    normalized = _ensure_evidence_section(report, ["data/summary.md"])
    assert "Necesita revisi\u00f3n manual" in normalized


def test_missing_is_not_treated_as_valid_evidence_source():
    assert _is_valid_evidence_source("missing") is False


def test_validate_report_flags_substantive_missing_evidence_claims():
    report = """## Decisión Ejecutiva

**Estatus: NO_GO**

## Riesgos

Existe una fuga crítica en opportunity_id.

## Evidencia usada

evidence:
{claim: "Existe una fuga crítica en opportunity_id.", source: "missing"}
{claim: "No verificable con artifacts actuales", source: "missing"}

Artifacts:
- data/output_contract_report.json
"""

    validation = _validate_report(
        content=report,
        expected_decision="NO_GO",
        facts_context=[],
        metrics_payload={},
        plots=[],
        expected_language="es",
    )

    assert "unsupported_evidence_claims" in validation["critical_issues"]
    assert validation["unsupported_evidence_claims"] == [
        "Existe una fuga crítica en opportunity_id."
    ]
