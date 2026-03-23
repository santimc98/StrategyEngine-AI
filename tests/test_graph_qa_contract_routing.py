import json
from pathlib import Path

from src.graph import graph as graph_module


def test_check_data_success_routes_cleaning_runs_to_qa_when_contract_requires_it(tmp_path):
    cleaned = tmp_path / "dataset_cleaned.csv"
    cleaned.write_text("a\n1\n", encoding="utf-8")
    state = {
        "ml_data_path": str(cleaned),
        "cleaned_data_preview": "{\"ok\": true}",
        "execution_contract": {
            "scope": "cleaning_only",
            "active_workstreams": {
                "data_cleaning": True,
                "feature_engineering": True,
                "model_training": False,
            },
            "required_outputs": [
                {"path": "artifacts/clean/dataset_cleaned.csv", "owner": "data_engineer"},
                {"path": "artifacts/qa/data_validation_results.json", "owner": "qa_engineer"},
            ],
            "qa_gates": [{"name": "verify_exclusions", "severity": "HARD"}],
        },
        "qa_view": {
            "review_subject": "data_engineer",
            "artifacts_to_verify": ["artifacts/clean/dataset_cleaned.csv"],
            "qa_required_outputs": ["artifacts/qa/data_validation_results.json"],
            "qa_gates": [{"name": "verify_exclusions", "severity": "HARD"}],
        },
    }
    assert graph_module.check_data_success(state) == "success_cleaning_only_qa"


def test_run_qa_reviewer_persists_contract_driven_report_for_data_engineer(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    def fake_review(code, strategy, business_objective, qa_context):
        assert code.strip() == "print('clean')"
        assert qa_context.get("review_subject") == "data_engineer"
        return {
            "status": "APPROVED",
            "feedback": "QA Passed",
            "failed_gates": [],
            "required_fixes": [],
            "qa_gates_evaluated": ["verify_exclusions"],
            "evidence": [{"claim": "Artifact path declared", "source": "artifacts/clean/dataset_cleaned.csv"}],
        }

    monkeypatch.setattr(graph_module, "_abort_if_requested", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(graph_module, "_consume_budget", lambda *_args, **_kwargs: (True, {}, ""))
    monkeypatch.setattr(graph_module, "collect_static_qa_facts", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(graph_module, "run_static_qa_checks", lambda *_args, **_kwargs: {"status": "PASS", "facts": {}})
    monkeypatch.setattr(graph_module.qa_reviewer, "review_code", fake_review)

    state = {
        "cleaning_code": "print('clean')",
        "selected_strategy": {},
        "business_objective": "Audit cleaned outputs",
        "feedback_history": [],
        "execution_contract": {
            "required_outputs": [
                {"path": "artifacts/clean/dataset_cleaned.csv", "owner": "data_engineer"},
                {"path": "artifacts/qa/data_validation_results.json", "owner": "qa_engineer"},
            ],
        },
        "qa_view": {
            "review_subject": "data_engineer",
            "subject_required_outputs": ["artifacts/clean/dataset_cleaned.csv"],
            "artifacts_to_verify": ["artifacts/clean/dataset_cleaned.csv"],
            "qa_required_outputs": ["artifacts/qa/data_validation_results.json"],
            "subject_code_path_hint": "artifacts/data_engineer_last.py",
            "qa_gates": [{"name": "verify_exclusions", "severity": "HARD"}],
        },
    }

    result = graph_module.run_qa_reviewer(state)
    report_path = Path(result.get("qa_report_path") or "")
    assert result.get("qa_review_subject") == "data_engineer"
    assert report_path.as_posix() == "artifacts/qa/data_validation_results.json"
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["review_subject"] == "data_engineer"
    assert payload["status"] == "APPROVED"
