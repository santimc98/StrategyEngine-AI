import pytest

from src.graph.graph import _build_iteration_handoff


def test_iteration_handoff_prioritizes_runtime_and_missing_outputs(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text("{}", encoding="utf-8")
    scored_path = tmp_path / "scored_rows.csv"

    state = {
        "iteration_count": 1,
        "execution_contract": {
            "required_outputs": [str(metrics_path), str(scored_path)],
        },
        "primary_metric_snapshot": {
            "primary_metric_name": "accuracy",
            "primary_metric_value": 0.71,
            "baseline_value": 0.68,
        },
        "execution_output": "Traceback (most recent call last)\nValueError: bad split",
        "last_runtime_error_tail": "ValueError: bad split",
    }
    gate_context = {
        "failed_gates": ["runtime_failure", "output_contract"],
        "required_fixes": ["Fix train/test split", f"Write {scored_path}"],
        "feedback": "Runtime crash during fit.",
    }
    oc_report = {"present": [str(metrics_path)], "missing": [str(scored_path)]}

    handoff = _build_iteration_handoff(
        state=state,
        status="NEEDS_IMPROVEMENT",
        gate_context=gate_context,
        oc_report=oc_report,
        review_result={"feedback": "Model crashed before scoring."},
        qa_result={"feedback": "Missing required scored_rows.csv."},
        evaluation_spec={"primary_metric": "accuracy"},
    )

    assert handoff["mode"] == "patch"
    assert handoff["next_iteration"] == 2
    assert handoff["contract_focus"]["missing_outputs"] == [str(scored_path)]
    assert any("runtime root cause" in item.lower() for item in handoff["patch_objectives"])
    assert any("missing contract outputs" in item.lower() for item in handoff["patch_objectives"])
    assert any(str(metrics_path) in item for item in handoff["must_preserve"])


def test_iteration_handoff_extracts_target_from_qa_gate_params():
    state = {
        "iteration_count": 0,
        "execution_contract": {"required_outputs": []},
        "primary_metric_snapshot": {
            "primary_metric_name": "roc_auc",
            "primary_metric_value": 0.79,
        },
    }

    handoff = _build_iteration_handoff(
        state=state,
        status="NEEDS_IMPROVEMENT",
        gate_context={},
        oc_report={},
        review_result={},
        qa_result={},
        evaluation_spec={
            "qa_gates": [
                {"name": "metric_gate", "params": {"target": 0.85}},
            ]
        },
    )

    metric_focus = handoff["metric_focus"]
    assert metric_focus["target_value"] == pytest.approx(0.85)
    assert metric_focus["target_source"].startswith("evaluation_spec.qa_gates")
    assert metric_focus["gap_to_target"] == pytest.approx(0.06, abs=1e-6)


def test_iteration_handoff_defers_metric_optimization_when_runtime_blockers_exist():
    state = {
        "iteration_count": 1,
        "execution_contract": {"required_outputs": ["data/metrics.json", "data/submission.csv"]},
        "primary_metric_snapshot": {
            "primary_metric_name": "roc_auc",
            "primary_metric_value": 0.91,
            "baseline_value": 0.905,
        },
        "ml_improvement_round_active": True,
        "ml_improvement_hypothesis_packet": {
            "action": "APPLY",
            "hypothesis": {"technique": "multi_seed_catboost_averaging"},
        },
        "ml_optimization_context": {
            "policy": {"phase": "explore"},
            "active_hypothesis": {
                "hypothesis": {"technique": "multi_seed_catboost_averaging"},
            },
        },
        "execution_output": "TIMEOUT: Script exceeded 7200s limit",
        "last_runtime_error_tail": "TIMEOUT: Script exceeded 7200s limit",
    }

    handoff = _build_iteration_handoff(
        state=state,
        status="NEEDS_IMPROVEMENT",
        gate_context={
            "failed_gates": ["runtime_failure"],
            "required_fixes": ["Reduce runtime cost."],
            "feedback": "Runtime timeout detected.",
        },
        oc_report={"present": [], "missing": ["data/metrics.json", "data/submission.csv"]},
        review_result={},
        qa_result={},
        evaluation_spec={"primary_metric": "roc_auc"},
    )

    assert handoff["mode"] == "patch"
    assert handoff["repair_policy"]["repair_first"] is True
    assert handoff["repair_policy"]["primary_focus"] == "runtime"
    assert handoff["editor_constraints"]["must_apply_hypothesis"] is False
    assert handoff["retry_context"]["error_type"] == "timeout"
    assert handoff["retry_context"]["cost_reduction_required"] is True
    assert handoff["deferred_optimization"]["active_technique"] == "multi_seed_catboost_averaging"
