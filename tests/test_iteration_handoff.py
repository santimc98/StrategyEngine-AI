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


def test_iteration_handoff_builds_repair_ground_truth_for_runtime_api_misuse():
    generated_code = (
        "import pathlib\n"
        "\n"
        "def run():\n"
        "    path = pathlib.Path('missing.txt')\n"
        "    return path.read_text(extra=True)\n"
        "\n"
        "run()\n"
    )
    runtime_output = (
        "Traceback (most recent call last)\n"
        "  File \"script.py\", line 7, in <module>\n"
        "    run()\n"
        "  File \"script.py\", line 5, in run\n"
        "    return path.read_text(extra=True)\n"
        "TypeError: Path.read_text() got an unexpected keyword argument 'extra'\n"
    )

    state = {
        "iteration_count": 1,
        "execution_contract": {"required_outputs": ["data/metrics.json", "data/submission.csv"]},
        "generated_code": generated_code,
        "execution_output": runtime_output,
        "last_runtime_error_tail": runtime_output,
    }

    handoff = _build_iteration_handoff(
        state=state,
        status="NEEDS_IMPROVEMENT",
        gate_context={
            "failed_gates": ["runtime_failure", "output_contract"],
            "required_fixes": ["Remove unsupported kwargs from the failing call."],
            "feedback": "Runtime crash before outputs were written.",
        },
        oc_report={"present": [], "missing": ["data/metrics.json", "data/submission.csv"]},
        review_result={"feedback": "Runtime failed before scoring."},
        qa_result={"feedback": "Missing required outputs."},
        evaluation_spec={"primary_metric": "accuracy"},
    )

    assert handoff["retry_context"]["error_type"] == "runtime_api_misuse"
    assert handoff["repair_policy"]["primary_focus"] == "runtime"
    repair_ground_truth = handoff["repair_ground_truth"]
    assert repair_ground_truth["root_cause_type"] == "runtime_api_misuse"
    assert any(
        fact.get("fact") == "unexpected_keyword_argument" and fact.get("value") == "extra"
        for fact in repair_ground_truth["verified_facts"]
    )
    assert any(
        site.get("expression") == "path.read_text" and "read_text" in str(site.get("resolved_symbol") or "")
        for site in repair_ground_truth["candidate_call_sites"]
    )
    assert any(
        env.get("fact") == "callable_signature"
        and "read_text" in str(env.get("resolved_symbol") or "")
        and "encoding" in str(env.get("value") or "")
        for env in repair_ground_truth["environment_facts"]
    )


def test_iteration_handoff_builds_patch_only_repair_scope_for_runtime_repair():
    generated_code = (
        "def check_writable(path):\n"
        "    os.remove(path)\n"
        "\n"
        "def main():\n"
        "    check_writable('tmp.txt')\n"
    )
    runtime_output = (
        "CRITICAL: Security Violations:\n"
        "Calling 'os.remove' is not allowed.\n"
    )

    state = {
        "iteration_count": 1,
        "execution_contract": {"required_outputs": ["data/metrics.json", "data/submission.csv"]},
        "generated_code": generated_code,
        "execution_output": runtime_output,
        "last_runtime_error_tail": runtime_output,
    }

    handoff = _build_iteration_handoff(
        state=state,
        status="NEEDS_IMPROVEMENT",
        gate_context={
            "failed_gates": ["runtime_failure", "contract_required_artifacts_missing"],
            "required_fixes": ["Remove os.remove from check_writable.", "Regenerate required outputs."],
            "feedback": "Security violation in output writability check.",
        },
        oc_report={"present": [], "missing": ["data/metrics.json", "data/submission.csv"]},
        review_result={
            "feedback": "Remove the prohibited os.remove call in check_writable.",
            "evidence": [{"claim": "Forbidden os.remove call.", "source": "script:2"}],
        },
        qa_result={},
        evaluation_spec={"primary_metric": "accuracy"},
    )

    repair_scope = handoff["repair_scope"]
    assert repair_scope["scope_policy"] == "patch_only"
    assert repair_scope["phase"] == "compliance_runtime"
    assert repair_scope["reviewer_guided"] is True
    assert any("script_line:2" == item for item in repair_scope["editable_targets"])
    assert any("training_strategy_and_model_family" == item for item in repair_scope["protected_regions"])
    assert any("Do not widen scope" in item for item in repair_scope["must_preserve_invariants"])
    assert handoff["editor_constraints"]["scope_policy"] == "patch_only"
    assert handoff["editor_constraints"]["freeze_unimplicated_regions"] is True


def test_iteration_handoff_prefers_real_traceback_over_heavy_runner_success_wrapper():
    runtime_tail = (
        "Traceback (most recent call last)\n"
        "  File \"script.py\", line 42, in <module>\n"
        "    main()\n"
        "  File \"script.py\", line 35, in main\n"
        "    assert len(submission_df) == EXPECTED_TEST_ROWS, 'submission mismatch'\n"
        "AssertionError: submission.csv row count mismatch: 316 vs 95\n"
    )
    state = {
        "iteration_count": 2,
        "execution_contract": {"required_outputs": ["data/metrics.json", "data/submission.csv"]},
        "execution_output": "HEAVY_RUNNER: status=success reason=local_runner_mode",
        "last_runtime_error_tail": runtime_tail,
    }

    handoff = _build_iteration_handoff(
        state=state,
        status="NEEDS_IMPROVEMENT",
        gate_context={
            "failed_gates": ["output_row_count_consistency"],
            "required_fixes": ["Filter submission rows to scoring subset only."],
        },
        oc_report={"present": [], "missing": ["data/metrics.json", "data/submission.csv"]},
        review_result={},
        qa_result={},
        evaluation_spec={"primary_metric": "roc_auc"},
    )

    assert handoff["retry_context"]["specific_error"].endswith("316 vs 95")
    assert handoff["repair_ground_truth"]["failure_signature"] == "AssertionError: submission.csv row count mismatch: 316 vs 95"
    assert not handoff["repair_ground_truth"]["failure_signature"].startswith("HEAVY_RUNNER")


def test_iteration_handoff_keeps_assertion_runtime_root_cause_even_with_missing_outputs():
    runtime_tail = (
        "Traceback (most recent call last)\n"
        "  File \"script.py\", line 42, in <module>\n"
        "    main()\n"
        "  File \"script.py\", line 35, in main\n"
        "    assert len(submission_df) == EXPECTED_TEST_ROWS, 'submission mismatch'\n"
        "AssertionError: submission.csv row count mismatch: 316 vs 95\n"
    )
    state = {
        "iteration_count": 3,
        "execution_contract": {"required_outputs": ["data/metrics.json", "data/scored_rows.csv", "data/submission.csv"]},
        "execution_output": runtime_tail,
        "last_runtime_error_tail": runtime_tail,
    }

    handoff = _build_iteration_handoff(
        state=state,
        status="NEEDS_IMPROVEMENT",
        gate_context={
            "failed_gates": ["output_row_count_consistency", "contract_required_artifacts_missing"],
            "required_fixes": ["Regenerate outputs after fixing row filtering."],
        },
        oc_report={"present": [], "missing": ["data/metrics.json", "data/scored_rows.csv", "data/submission.csv"]},
        review_result={},
        qa_result={},
        evaluation_spec={"primary_metric": "official_kaggle_metric"},
    )

    assert handoff["retry_context"]["error_type"] == "runtime_contract_assertion"
    assert handoff["repair_policy"]["primary_focus"] == "runtime"
    assert handoff["repair_ground_truth"]["root_cause_type"] == "runtime_contract_assertion"
    assert any(
        fact.get("fact") == "exception_type" and fact.get("value") == "AssertionError"
        for fact in handoff["repair_ground_truth"]["verified_facts"]
    )
