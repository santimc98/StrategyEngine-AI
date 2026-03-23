import json
import os

from src.agents.business_translator import BusinessTranslatorAgent, _score_report_quality


class _EchoModel:
    def generate_content(self, prompt):
        class _Resp:
            def __init__(self, text):
                self.text = text
        return _Resp(prompt)


def test_translator_self_check_instructions_present(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "insights.json"), "w", encoding="utf-8") as f:
        json.dump({}, f)

    agent = BusinessTranslatorAgent(api_key="dummy_key")
    agent.model = _EchoModel()
    report = agent.generate_report(
        {"execution_output": "OK", "business_objective": "Objetivo de prueba"}
    )
    assert "Slot Coverage" in report
    assert "reporting_policy" in report


def test_translator_prompt_declares_source_of_truth_and_authoritative_outcome(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "insights.json"), "w", encoding="utf-8") as f:
        json.dump({}, f)
    with open(os.path.join("data", "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"run_outcome": "NO_GO"}, f)

    agent = BusinessTranslatorAgent(api_key="dummy_key")
    agent.model = _EchoModel()
    report = agent.generate_report(
        {"execution_output": "OK", "business_objective": "Objetivo de prueba"}
    )

    assert "=== SOURCE OF TRUTH AND PRECEDENCE ===" in report
    assert "The authoritative executive outcome for this report is: NO_GO" in report


def test_translator_quality_score_penalizes_decision_discrepancy_context():
    score = _score_report_quality(
        {
            "structure_issues": [],
            "decision_issue": [],
            "unverified_metrics": [],
            "unsupported_evidence_claims": [],
            "invalid_plots": [],
            "context_warnings": ["decision_discrepancy_authoritative_vs_derived"],
            "decision_discrepancy": {
                "authoritative_decision": "NO_GO",
                "derived_decision": "GO_WITH_LIMITATIONS",
                "run_outcome": "NO_GO",
            },
        }
    )

    assert score < 100


def test_translator_fallback_does_not_claim_missing_declared_artifacts_as_present(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)

    with open(os.path.join("data", "insights.json"), "w", encoding="utf-8") as f:
        json.dump({}, f)
    with open(os.path.join("data", "execution_contract.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "required_outputs": [
                    {"path": "artifacts/clean/dataset_cleaned.csv", "owner": "data_engineer", "required": True},
                    {"path": "artifacts/reports/quality_audit_report.json", "owner": "data_engineer", "required": True},
                ]
            },
            f,
        )
    with open(os.path.join("data", "output_contract_report.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "overall_status": "error",
                "present": [],
                "missing": [
                    "artifacts/clean/dataset_cleaned.csv",
                    "artifacts/reports/quality_audit_report.json",
                ],
            },
            f,
        )
    with open(os.path.join("data", "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"run_outcome": "NO_GO"}, f)
    with open(os.path.join("data", "produced_artifact_index.json"), "w", encoding="utf-8") as f:
        json.dump([], f)

    agent = BusinessTranslatorAgent(api_key="dummy_key")
    agent.model = _EchoModel()
    report = agent.generate_report(
        {"execution_output": "DE failed before producing artifacts", "business_objective": "Objetivo de prueba"}
    )

    assert "Confirmed artifact present: artifacts/clean/dataset_cleaned.csv" not in report
    assert 'source: "artifacts/clean/dataset_cleaned.csv"' not in report
    assert "- missing" in report
