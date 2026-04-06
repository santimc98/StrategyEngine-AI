"""
Tests for result evaluator behavior with non-blocking metric feedback.

Policy:
- NEEDS_IMPROVEMENT for metric-only issues stays NEEDS_IMPROVEMENT in evaluator output.
- check_evaluation then treats non-blocking metric findings as advisory-only (no retry).
- NEEDS_IMPROVEMENT for compliance issues (audit rejected, missing outputs) keeps retry path.
"""
import json
import os
from pathlib import Path

from src.graph import graph as graph_mod


class _StubReviewerMetricIssue:
    """Stub that returns NEEDS_IMPROVEMENT for metric-only issue (no audit rejection)."""
    def evaluate_results(self, *_args, **_kwargs):
        return {"status": "NEEDS_IMPROVEMENT", "feedback": "Metrics below threshold", "retry_worth_it": True}


class _StubReviewerApproved:
    """Stub that returns APPROVED."""
    def evaluate_results(self, *_args, **_kwargs):
        return {"status": "APPROVED", "feedback": "All good"}


class _StubReviewerStaleDeliverabilityComplaint:
    def evaluate_results(self, *_args, **_kwargs):
        return {
            "status": "NEEDS_IMPROVEMENT",
            "feedback": (
                "The run is not deliverable because required scoring CSV and executive report "
                "are missing or stale, and upstream outputs appear stale."
            ),
            "retry_worth_it": True,
        }

    def review_code(self, *_args, **_kwargs):
        return {
            "status": "APPROVED",
            "feedback": "review ok",
            "failed_gates": [],
            "required_fixes": [],
            "hard_failures": [],
        }

class _StubQAApproved:
    def review_code(self, *_args, **_kwargs):
        return {
            "status": "APPROVED",
            "feedback": "QA Passed",
            "failed_gates": [],
            "required_fixes": [],
            "hard_failures": [],
        }

class _StubQARejectedNoGates:
    def review_code(self, *_args, **_kwargs):
        return {
            "status": "REJECTED",
            "feedback": "Output schema mismatch",
            "failed_gates": [],
            "required_fixes": [],
            "hard_failures": [],
        }


def test_result_evaluator_metric_issue_keeps_needs_improvement_and_stops_advisory(tmp_path, monkeypatch):
    """
    When NEEDS_IMPROVEMENT is for metric-only issues (iteration_type='metric'),
    evaluator should keep NEEDS_IMPROVEMENT and evaluation routing should stop as advisory-only.
    """
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"metric": 0.5}, f)

    state = {
        "execution_output": "OK",  # No traceback -> not compliance
        "selected_strategy": {},
        "business_objective": "",
        "execution_contract": {"spec_extraction": {"case_taxonomy": []}},
        "evaluation_spec": {},
        "iteration_count": 0,
        "feedback_history": [],
    }

    monkeypatch.setattr(graph_mod, "reviewer", _StubReviewerMetricIssue())
    result = graph_mod.run_result_evaluator(state)

    assert result["review_verdict"] == "NEEDS_IMPROVEMENT"
    assert not any("Metric iteration disabled" in item for item in result["feedback_history"])

    state.update(result)
    assert graph_mod.check_evaluation(state) == "approved"
    assert state.get("stop_reason") == "ADVISORY_ONLY"


def test_result_evaluator_compliance_issue_keeps_needs_improvement(tmp_path, monkeypatch):
    """
    When NEEDS_IMPROVEMENT is for compliance issues (traceback, audit rejection),
    it should stay NEEDS_IMPROVEMENT to allow retry for compliance fixes.
    """
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"metric": 0.5}, f)

    state = {
        "execution_output": "Traceback (most recent call last):\n  File...\nValueError: something wrong",
        "selected_strategy": {},
        "business_objective": "",
        "execution_contract": {"spec_extraction": {"case_taxonomy": []}},
        "evaluation_spec": {},
        "iteration_count": 0,
        "feedback_history": [],
    }

    monkeypatch.setattr(graph_mod, "reviewer", _StubReviewerMetricIssue())
    result = graph_mod.run_result_evaluator(state)

    # Compliance issues (traceback) should keep NEEDS_IMPROVEMENT for retry
    assert result["review_verdict"] == "NEEDS_IMPROVEMENT"
    # Should NOT have metric iteration disabled message (this is compliance)
    assert not any("Metric iteration disabled" in item for item in result["feedback_history"])


def test_result_evaluator_approved_stays_approved(tmp_path, monkeypatch):
    """When APPROVED, it should stay APPROVED (no changes)."""
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"metric": 0.9}, f)

    state = {
        "execution_output": "OK",
        "selected_strategy": {},
        "business_objective": "",
        "execution_contract": {"spec_extraction": {"case_taxonomy": []}},
        "evaluation_spec": {},
        "iteration_count": 0,
        "feedback_history": [],
    }

    monkeypatch.setattr(graph_mod, "reviewer", _StubReviewerApproved())
    result = graph_mod.run_result_evaluator(state)

    assert result["review_verdict"] == "APPROVED"


def test_result_evaluator_hard_gate_column_presence_blocks_success(tmp_path, monkeypatch):
    """
    HARD QA gate: explanation column missing in scored_rows.csv -> surfaced as advisory context.
    """
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"metric": 0.7}, f)
    with open(os.path.join("data", "scored_rows.csv"), "w", encoding="utf-8") as f:
        f.write("top_drivers\nfoo\n")

    state = {
        "execution_output": "OK",
        "selected_strategy": {},
        "business_objective": "",
        "execution_contract": {"spec_extraction": {"case_taxonomy": []}},
        "evaluation_spec": {
            "qa_gates": [
                {
                    "name": "explanation_column_presence",
                    "severity": "HARD",
                    "params": {"target_file": "data/scored_rows.csv", "column": "explanation"},
                }
            ]
        },
        "iteration_count": 0,
        "feedback_history": [],
    }

    monkeypatch.setattr(graph_mod, "reviewer", _StubReviewerApproved())
    monkeypatch.setattr(graph_mod, "qa_reviewer", _StubQAApproved())
    result = graph_mod.run_result_evaluator(state)

    assert result["review_verdict"] == "APPROVED"
    assert any("QA_GATE_FAIL" in item for item in result["feedback_history"])

    state.update(result)
    assert graph_mod.check_evaluation(state) == "approved"


def test_result_evaluator_qa_rejected_blocks_metric_downgrade(tmp_path, monkeypatch):
    """
    QA REJECTED (even without failed_gates) must block metric-only downgrade.
    """
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"metric": 0.5}, f)

    state = {
        "execution_output": "OK",
        "selected_strategy": {},
        "business_objective": "",
        "generated_code": "print('hello')",
        "execution_contract": {"spec_extraction": {"case_taxonomy": []}},
        "evaluation_spec": {},
        "iteration_count": 0,
        "feedback_history": [],
    }

    monkeypatch.setattr(graph_mod, "reviewer", _StubReviewerMetricIssue())
    monkeypatch.setattr(graph_mod, "qa_reviewer", _StubQARejectedNoGates())
    result = graph_mod.run_result_evaluator(state)

    assert result["review_verdict"] == "NEEDS_IMPROVEMENT"
    assert any(
        ("QA_CODE_AUDIT[REJECTED]" in item) or ("CODE_AUDIT_FINDINGS" in item)
        for item in result["feedback_history"]
    )
    assert "hard_failures" in result and "qa_rejected" in result["hard_failures"]

    state.update(result)
    assert graph_mod.check_evaluation(state) == "retry"


def test_result_evaluator_missing_contract_artifact_forces_retry(tmp_path, monkeypatch):
    """
    Missing contract-required artifacts must force NEEDS_IMPROVEMENT and retry context.
    """
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"metric": 0.9}, f)

    state = {
        "execution_output": "OK",
        "selected_strategy": {},
        "business_objective": "",
        "execution_contract": {
            "required_outputs": ["data/metrics.json", "static/plots/confidence_distribution.png"],
            "artifact_requirements": {
                "required_files": [{"path": "data/metrics.json"}],
            },
            "spec_extraction": {"case_taxonomy": []},
        },
        "evaluation_spec": {},
        "iteration_count": 0,
        "feedback_history": [],
    }

    monkeypatch.setattr(graph_mod, "reviewer", _StubReviewerApproved())
    result = graph_mod.run_result_evaluator(state)

    assert result["review_verdict"] == "NEEDS_IMPROVEMENT"
    gate_ctx = result.get("last_gate_context", {})
    assert "contract_required_artifacts_missing" in (gate_ctx.get("failed_gates") or [])
    assert any(
        "static/plots/confidence_distribution.png" in fix
        for fix in (gate_ctx.get("required_fixes") or [])
    )
    assert result.get("review_retry_worth_it") is True

    state.update(result)
    assert graph_mod.check_evaluation(state) == "retry"


def test_result_evaluator_ignores_stale_runtime_hard_failures_after_clean_retry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"metric": 0.9}, f)

    state = {
        "execution_output": "HEAVY_RUNNER: status=success reason=local_runner_mode",
        "execution_error": False,
        "sandbox_failed": False,
        "hard_failures": ["runtime_failure", "result_evaluator_failed_gate:runtime_failure"],
        "last_gate_context": {
            "status": "NEEDS_IMPROVEMENT",
            "failed_gates": ["runtime_failure"],
            "required_fixes": ["Fix runtime"],
            "hard_failures": ["runtime_failure"],
            "traceback": "Traceback (most recent call last): ...",
        },
        "selected_strategy": {},
        "business_objective": "",
        "generated_code": "print('hello')",
        "execution_contract": {"spec_extraction": {"case_taxonomy": []}},
        "evaluation_spec": {},
        "iteration_count": 0,
        "feedback_history": [],
    }

    monkeypatch.setattr(graph_mod, "reviewer", _StubReviewerApproved())
    monkeypatch.setattr(graph_mod, "qa_reviewer", _StubQAApproved())
    result = graph_mod.run_result_evaluator(state)

    assert result["review_verdict"] == "APPROVED"
    assert "hard_failures" not in result or result["hard_failures"] == []
    assert result["last_gate_context"].get("hard_failures") in ([], None)


def test_result_evaluator_downgrades_stale_non_ml_output_warning_after_clean_success(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("artifacts/ml").mkdir(parents=True, exist_ok=True)
    Path("artifacts/clean").mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(parents=True, exist_ok=True)

    Path("artifacts/ml/cv_metrics.json").write_text("{}", encoding="utf-8")
    Path("artifacts/ml/churn_risk_scores.csv").write_text("account_id,score\nA,0.9\n", encoding="utf-8")
    Path("artifacts/ml/model.pkl").write_text("stub", encoding="utf-8")
    Path("artifacts/clean/churn_snapshots_ml_ready.csv").write_text("x\n1\n", encoding="utf-8")
    Path("artifacts/clean/cleaning_manifest.json").write_text("{}", encoding="utf-8")
    Path("data/metrics.json").write_text('{"metric": 0.9}', encoding="utf-8")

    state = {
        "execution_output": (
            "HEAVY_RUNNER: status=success reason=local_runner_mode\n"
            "VALIDATION_WARNING: STALE_OUTPUTS: ['artifacts/clean/churn_snapshots_ml_ready.csv', "
            "'artifacts/clean/cleaning_manifest.json']"
        ),
        "execution_error": False,
        "sandbox_failed": False,
        "selected_strategy": {},
        "business_objective": "",
        "generated_code": "print('hello')",
        "execution_contract": {
            "required_outputs": [
                {"path": "artifacts/ml/cv_metrics.json", "owner": "ml_engineer"},
                {"path": "artifacts/ml/churn_risk_scores.csv", "owner": "ml_engineer"},
                {"path": "artifacts/ml/model.pkl", "owner": "ml_engineer"},
                {"path": "artifacts/clean/churn_snapshots_ml_ready.csv", "owner": "data_engineer"},
                {"path": "artifacts/clean/cleaning_manifest.json", "owner": "data_engineer"},
            ],
            "spec_extraction": {"case_taxonomy": []},
        },
        "output_contract_report": {
            "overall_status": "ok",
            "missing": [],
            "present": [
                "artifacts/ml/cv_metrics.json",
                "artifacts/ml/churn_risk_scores.csv",
                "artifacts/ml/model.pkl",
                "artifacts/clean/churn_snapshots_ml_ready.csv",
                "artifacts/clean/cleaning_manifest.json",
            ],
        },
        "evaluation_spec": {},
        "iteration_count": 0,
        "feedback_history": [],
    }

    monkeypatch.setattr(graph_mod, "reviewer", _StubReviewerStaleDeliverabilityComplaint())
    monkeypatch.setattr(graph_mod, "qa_reviewer", _StubQAApproved())
    result = graph_mod.run_result_evaluator(state)

    assert result["review_verdict"] == "APPROVE_WITH_WARNINGS"
    assert any("NONBLOCKING_STALE_OUTPUTS_IGNORED" in item for item in result["feedback_history"])
    assert result["iteration_handoff"].get("source") == "result_evaluator"
    assert result["iteration_handoff"].get("repair_policy") in (None, {})


def test_check_evaluation_advisory_needs_improvement_stops_without_retry():
    state = {
        "review_verdict": "NEEDS_IMPROVEMENT",
        "execution_output": "OK",
        "last_iteration_type": None,
        "last_gate_context": {
            "failed_gates": ["metric_gap"],
            "required_fixes": ["Improve KPI if possible."],
        },
    }
    assert graph_mod.check_evaluation(state) == "approved"
    assert state.get("stop_reason") == "ADVISORY_ONLY"


def test_check_evaluation_results_advisor_improve_retry_routes_to_retry(monkeypatch):
    monkeypatch.setenv("IMPROVEMENT_LOOP_ENABLED", "1")
    monkeypatch.setenv("MAX_IMPROVEMENT_ATTEMPTS", "2")
    monkeypatch.setenv("IMPROVEMENT_PATIENCE", "1")
    calls = {"count": 0}

    def _fake_bootstrap(working_state, _contract):
        calls["count"] += 1
        working_state["ml_improvement_round_active"] = True
        working_state["ml_improvement_round_count"] = 1
        working_state["ml_improvement_rounds_allowed"] = 2
        return True

    monkeypatch.setattr(graph_mod, "_bootstrap_metric_improvement_round", _fake_bootstrap)

    state = {
        "review_verdict": "APPROVE_WITH_WARNINGS",
        "reviewer_last_result": {"status": "APPROVED"},
        "qa_last_result": {"status": "APPROVED"},
        "execution_output": "OK",
        "last_iteration_type": None,
        "execution_error": False,
        "sandbox_failed": False,
        "results_last_result": {
            "iteration_recommendation": {"action": "RETRY", "mode": "improve"}
        },
        "primary_metric_snapshot": {
            "primary_metric_name": "roc_auc",
            "primary_metric_value": 0.71,
            "baseline_value": 0.71,
        },
        "execution_contract": {},
    }

    assert graph_mod.check_review_board_metric_improvement_route(state) == "bootstrap_improvement_round"
    updates = graph_mod.run_metric_improvement_bootstrap(state)
    merged_state = {**state, **updates}
    assert calls["count"] == 1
    assert merged_state.get("ml_improvement_round_active") is True
    assert graph_mod.check_metric_improvement_bootstrap_route(merged_state) == "retry"


def test_check_evaluation_results_advisor_stop_does_not_block_improvement_round(monkeypatch):
    monkeypatch.setenv("IMPROVEMENT_LOOP_ENABLED", "1")
    monkeypatch.setenv("MAX_IMPROVEMENT_ATTEMPTS", "3")
    monkeypatch.setenv("IMPROVEMENT_PATIENCE", "2")
    calls = {"count": 0}

    def _fake_bootstrap(working_state, _contract):
        calls["count"] += 1
        working_state["ml_improvement_round_active"] = True
        working_state["ml_improvement_round_count"] = 1
        working_state["ml_improvement_rounds_allowed"] = 3
        return True

    monkeypatch.setattr(graph_mod, "_bootstrap_metric_improvement_round", _fake_bootstrap)

    state = {
        "review_verdict": "APPROVED",
        "reviewer_last_result": {"status": "APPROVED"},
        "qa_last_result": {"status": "APPROVED"},
        "execution_output": "OK",
        "last_iteration_type": None,
        "execution_error": False,
        "sandbox_failed": False,
        "results_last_result": {
            "iteration_recommendation": {"action": "STOP"}
        },
        "primary_metric_snapshot": {
            "primary_metric_name": "roc_auc",
            "primary_metric_value": 0.73,
            "baseline_value": 0.73,
        },
        "execution_contract": {},
    }

    assert graph_mod.check_review_board_metric_improvement_route(state) == "bootstrap_improvement_round"
    updates = graph_mod.run_metric_improvement_bootstrap(state)
    merged_state = {**state, **updates}
    assert calls["count"] == 1
    assert merged_state.get("ml_improvement_round_active") is True
