import pytest

from src.graph.graph import (
    _build_iteration_handoff,
    _build_review_guided_retry_context,
    _extract_verified_gate_feedback,
    prepare_runtime_fix,
)


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


def test_review_guided_retry_context_rejects_string_output_status_as_path_list():
    context = _build_review_guided_retry_context(
        failed_gates=["outlier_policy_applied"],
        hard_failures=["outlier_policy_applied"],
        missing_outputs="missing",
        present_outputs="present",
        required_fixes=["outlier_policy_applied: report schema mismatch"],
        evidence_focus=[],
    )

    assert context["missing_outputs"] == []
    assert context["working_components"] == []
    assert "Missing outputs: m" not in context["specific_error"]


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
    assert handoff["optimization_lane"]["active"] is True
    assert handoff["optimization_lane"]["resume_after_repair"] is True
    assert handoff["optimization_lane"]["repair_first"] is True
    assert handoff["optimization_lane"]["active_technique"] == "multi_seed_catboost_averaging"


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
    assert any(
        "does not accept 'extra'" in str(note or "")
        for note in repair_ground_truth.get("compatibility_notes", [])
    )


def test_prepare_runtime_fix_attaches_authoritative_repair_ground_truth_for_ml_runtime_retry(
    monkeypatch,
):
    monkeypatch.setattr(
        "src.graph.graph.failure_explainer.explain_ml_failure",
        lambda **kwargs: "",
    )
    generated_code = (
        "from sklearn.preprocessing import OneHotEncoder\n"
        "from sklearn.compose import ColumnTransformer\n"
        "\n"
        "def build_preprocessor(cat_cols):\n"
        "    return ColumnTransformer([\n"
        "        ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), cat_cols),\n"
        "    ])\n"
    )
    runtime_output = (
        "Traceback (most recent call last)\n"
        "  File \"script.py\", line 6, in <module>\n"
        "    build_preprocessor(['segment'])\n"
        "  File \"script.py\", line 5, in build_preprocessor\n"
        "    ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), cat_cols),\n"
        "TypeError: OneHotEncoder.__init__() got an unexpected keyword argument 'sparse'\n"
    )

    result = prepare_runtime_fix(
        {
            "execution_output": runtime_output,
            "generated_code": generated_code,
            "last_generated_code": generated_code,
            "execution_contract": {
                "required_outputs": [
                    "artifacts/ml/cv_metrics.json",
                    "artifacts/ml/model.pkl",
                ]
            },
            "output_contract_report": {
                "present": ["artifacts/ml/model.pkl"],
                "missing": ["artifacts/ml/cv_metrics.json"],
            },
            "iteration_handoff": {"mode": "build", "feedback": {}},
        }
    )

    handoff = result["iteration_handoff"]
    repair_ground_truth = handoff["repair_ground_truth"]
    assert repair_ground_truth["root_cause_type"] == "runtime_api_misuse"
    assert any(
        fact.get("fact") == "unexpected_keyword_argument" and fact.get("value") == "sparse"
        for fact in repair_ground_truth.get("verified_facts", [])
    )
    assert any(
        env.get("fact") == "callable_signature"
        and "sklearn" in str(env.get("resolved_symbol") or "")
        for env in repair_ground_truth.get("environment_facts", [])
    )
    assert any(
        fact.get("fact") == "unexpected_keyword_callable_mismatch"
        for fact in repair_ground_truth.get("verified_facts", [])
    )
    assert handoff["feedback"]["repair_ground_truth_summary"]["root_cause_type"] == "runtime_api_misuse"
    assert handoff["feedback"]["verified_environment_facts"]
    assert any(
        item.get("kind") == "verified_fact" and item.get("fact") == "unexpected_keyword_argument"
        for item in handoff["quality_focus"]["evidence"]
    )


def test_prepare_runtime_fix_reconciles_conflicting_ml_threshold_repair_guidance(monkeypatch):
    monkeypatch.setattr(
        "src.graph.graph.failure_explainer.explain_ml_failure",
        lambda **kwargs: (
            "WHERE: holdout validation guard\n"
            "WHY: holdout split has only 406 rows.\n"
            "FIX: Lower the minimum row threshold from 1000 to 400 to match the observed holdout size."
        ),
    )
    runtime_output = (
        "Traceback (most recent call last)\n"
        "  File \"script.py\", line 10, in <module>\n"
        "    raise ValueError('Temporal validation holdout has insufficient rows: 406 < 1000')\n"
        "ValueError: Temporal validation holdout has insufficient rows: 406 < 1000\n"
    )

    result = prepare_runtime_fix(
        {
            "execution_output": runtime_output,
            "generated_code": "print('baseline')\n",
            "last_generated_code": "print('baseline')\n",
            "execution_contract": {
                "required_outputs": ["artifacts/ml/cv_metrics.json"],
                "qa_gates": [
                    {
                        "name": "holdout_sample_credible",
                        "severity": "HARD",
                        "params": {"min_rows": 1000},
                    }
                ],
            },
            "output_contract_report": {
                "present": [],
                "missing": ["artifacts/ml/cv_metrics.json"],
            },
            "iteration_handoff": {"mode": "build", "feedback": {}},
        }
    )

    handoff = result["iteration_handoff"]
    repair_ground_truth = handoff["repair_ground_truth"]
    assert repair_ground_truth.get("governance_conflicts")
    assert any(
        fact.get("fact") == "governance_conflict_detected"
        for fact in repair_ground_truth.get("verified_facts", [])
    )
    assert all("400" not in str(item) for item in handoff["quality_focus"]["required_fixes"])
    assert any(
        "without weakening hard reviewer/qa/contract gates" in str(item).lower()
        for item in handoff["quality_focus"]["required_fixes"]
    )
    assert all(
        "400" not in str(item)
        for item in handoff["retry_context"].get("recommended_actions", [])
    )
    assert "Lower the minimum row threshold" not in str(result.get("ml_engineer_audit_override") or "")


def test_iteration_handoff_builds_callable_compatibility_facts_for_sklearn_api_misuse():
    generated_code = (
        "from sklearn.preprocessing import OneHotEncoder\n"
        "from sklearn.pipeline import Pipeline\n"
        "\n"
        "def build_preprocessor():\n"
        "    return Pipeline([('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))])\n"
        "\n"
        "build_preprocessor()\n"
    )
    runtime_output = (
        "Traceback (most recent call last)\n"
        "  File \"script.py\", line 7, in <module>\n"
        "    build_preprocessor()\n"
        "  File \"script.py\", line 5, in build_preprocessor\n"
        "    return Pipeline([('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))])\n"
        "TypeError: OneHotEncoder.__init__() got an unexpected keyword argument 'sparse'\n"
    )

    state = {
        "iteration_count": 1,
        "execution_contract": {"required_outputs": ["data/metrics.json"]},
        "generated_code": generated_code,
        "execution_output": runtime_output,
        "last_runtime_error_tail": runtime_output,
    }

    handoff = _build_iteration_handoff(
        state=state,
        status="NEEDS_IMPROVEMENT",
        gate_context={
            "failed_gates": ["runtime_failure"],
            "required_fixes": ["Patch the failing OneHotEncoder call using the verified signature."],
            "feedback": "Runtime crash in sklearn preprocessing.",
        },
        oc_report={"present": [], "missing": ["data/metrics.json"]},
        review_result={"feedback": "Runtime failed before metrics."},
        qa_result={},
        evaluation_spec={"primary_metric": "mae"},
    )

    repair_ground_truth = handoff["repair_ground_truth"]
    assert repair_ground_truth["root_cause_type"] == "runtime_api_misuse"
    assert any(
        fact.get("fact") == "unexpected_keyword_argument" and fact.get("value") == "sparse"
        for fact in repair_ground_truth["verified_facts"]
    )
    assert any(
        fact.get("fact") == "callable_accepted_parameters"
        and "OneHotEncoder" in str(fact.get("resolved_symbol") or "")
        and "sparse_output" in list(fact.get("value") or [])
        for fact in repair_ground_truth["verified_facts"]
    )
    assert any(
        fact.get("fact") == "unexpected_keyword_callable_mismatch"
        and fact.get("value", {}).get("unsupported_keyword") == "sparse"
        and "sparse_output" in list(fact.get("value", {}).get("accepted_parameters") or [])
        for fact in repair_ground_truth["verified_facts"]
    )
    assert any(
        "does not accept 'sparse'" in str(note or "") and "sparse_output" in str(note or "")
        for note in repair_ground_truth.get("compatibility_notes", [])
    )


def test_iteration_handoff_exposes_column_repair_context_for_parser_introduced_nulls(tmp_path):
    csv_path = tmp_path / "account_snapshots.csv"
    csv_path.write_text(
        "account_id,snapshot_month_end,churn_60d\n"
        "A001,2025-01-31,0\n"
        "A002,31/12/2024,1\n"
        "A003,12-31-2024,\n",
        encoding="utf-8",
    )
    runtime_output = (
        "Traceback (most recent call last)\n"
        "  File \"script.py\", line 88, in <module>\n"
        "    main()\n"
        "  File \"script.py\", line 72, in main\n"
        "    raise ValueError('identifier_and_split_key_completeness - Nulls introduced in primary identifiers after parsing: snapshot_month_end')\n"
        "ValueError: identifier_and_split_key_completeness - Nulls introduced in primary identifiers after parsing: snapshot_month_end\n"
    )
    state = {
        "iteration_count": 2,
        "execution_contract": {"required_outputs": ["artifacts/clean/accounts_snapshot_ml_ready.csv"]},
        "execution_output": runtime_output,
        "last_runtime_error_tail": runtime_output,
        "csv_path": str(csv_path),
        "csv_encoding": "utf-8",
        "csv_sep": ",",
        "csv_decimal": ".",
        "dataset_profile": {
            "column_profiles": {
                "snapshot_month_end": {
                    "null_pct": 0.0,
                    "null_count": 0,
                    "looks_datetime": True,
                    "observed_format_patterns": ["YYYY-MM-DD", "DD/MM/YYYY", "MM-DD-YYYY"],
                },
                "account_id": {
                    "null_pct": 0.0,
                    "null_count": 0,
                    "unique_count": 3,
                },
            }
        },
    }

    handoff = _build_iteration_handoff(
        state=state,
        status="NEEDS_IMPROVEMENT",
        gate_context={
            "failed_gates": ["identifier_and_split_key_completeness"],
            "required_fixes": ["Fix parsing/coercion for snapshot_month_end without blaming the raw CSV."],
            "feedback": "snapshot_month_end became null after parsing.",
        },
        oc_report={"present": [], "missing": ["artifacts/clean/accounts_snapshot_ml_ready.csv"]},
        review_result={},
        qa_result={},
        evaluation_spec={"primary_metric": "pr_auc"},
    )

    repair_ground_truth = handoff["repair_ground_truth"]
    assert any(
        fact.get("fact") == "column_repair_context"
        and fact.get("column") == "snapshot_month_end"
        and fact.get("value", {}).get("raw_non_null_ratio_exact") == pytest.approx(1.0)
        for fact in repair_ground_truth["verified_facts"]
    )
    assert any(
        "introduced by parsing or coercion" in directive.lower()
        for directive in repair_ground_truth.get("repair_directives", [])
    )
    assert any(
        "staged parsing" in directive.lower()
        for directive in repair_ground_truth.get("repair_directives", [])
    )
    assert any(
        delta.get("kind") == "parser_introduced_null_inflation"
        and delta.get("column") == "snapshot_month_end"
        for delta in repair_ground_truth.get("causal_deltas", [])
        if isinstance(delta, dict)
    )
    assert any(
        "snapshot_month_end" in str(goal or "").lower()
        and "null inflation" in str(goal or "").lower()
        for goal in repair_ground_truth.get("repair_goal", [])
    )
    repair_scope = handoff["repair_scope"]
    assert any(
        "snapshot_month_end: raw source is fully populated" in item
        for item in repair_scope.get("active_findings", [])
    )
    assert any(
        "parse_logic:snapshot_month_end" == item
        for item in repair_scope.get("editable_targets", [])
    )


def test_iteration_handoff_marks_rethink_required_for_repeated_de_failure_pattern(tmp_path):
    csv_path = tmp_path / "account_snapshots.csv"
    csv_path.write_text(
        "account_id,snapshot_month_end,churn_60d\n"
        "A001,2025-01-31,0\n"
        "A002,31/12/2024,1\n"
        "A003,12-31-2024,\n",
        encoding="utf-8",
    )
    runtime_output = (
        "Traceback (most recent call last)\n"
        "  File \"script.py\", line 88, in <module>\n"
        "    main()\n"
        "  File \"script.py\", line 72, in main\n"
        "    raise ValueError('identifier_and_split_key_completeness - Nulls introduced in primary identifiers after parsing: snapshot_month_end')\n"
        "ValueError: identifier_and_split_key_completeness - Nulls introduced in primary identifiers after parsing: snapshot_month_end\n"
    )
    state = {
        "iteration_count": 3,
        "execution_contract": {"required_outputs": ["artifacts/clean/accounts_snapshot_ml_ready.csv"]},
        "execution_output": runtime_output,
        "last_runtime_error_tail": runtime_output,
        "csv_path": str(csv_path),
        "csv_encoding": "utf-8",
        "csv_sep": ",",
        "csv_decimal": ".",
        "dataset_profile": {
            "column_profiles": {
                "snapshot_month_end": {
                    "null_pct": 0.0,
                    "null_count": 0,
                    "looks_datetime": True,
                    "observed_format_patterns": ["YYYY-MM-DD", "DD/MM/YYYY", "MM-DD-YYYY"],
                }
            }
        },
        "data_engineer_attempt_history": [
            {
                "attempt": 1,
                "source": "runtime_sandbox_execute",
                "status": "REJECTED",
                "failed_gates": ["identifier_and_split_key_completeness"],
                "required_fixes": ["Fix parsing/coercion for snapshot_month_end."],
                "runtime_error_tail": "ValueError: identifier_and_split_key_completeness - Nulls introduced in primary identifiers after parsing: snapshot_month_end",
                "feedback_summary": "snapshot_month_end became null after parsing.",
            },
            {
                "attempt": 2,
                "source": "runtime_heavy_runner_code",
                "status": "REJECTED",
                "failed_gates": ["identifier_and_split_key_completeness"],
                "required_fixes": ["Do not blame the raw CSV; rethink snapshot_month_end parsing."],
                "runtime_error_tail": "ValueError: identifier_and_split_key_completeness - Nulls introduced in primary identifiers after parsing: snapshot_month_end",
                "feedback_summary": "snapshot_month_end still becomes null after parsing.",
            },
        ],
    }

    handoff = _build_iteration_handoff(
        state=state,
        status="NEEDS_IMPROVEMENT",
        gate_context={
            "failed_gates": ["identifier_and_split_key_completeness"],
            "required_fixes": ["Fix parsing/coercion for snapshot_month_end without blaming the raw CSV."],
            "feedback": "snapshot_month_end became null after parsing again.",
        },
        oc_report={"present": [], "missing": ["artifacts/clean/accounts_snapshot_ml_ready.csv"]},
        review_result={},
        qa_result={},
        evaluation_spec={"primary_metric": "pr_auc"},
    )

    repair_ground_truth = handoff["repair_ground_truth"]
    assert repair_ground_truth.get("rethink_required") is True
    assert "same gate/failure pattern" in str(repair_ground_truth.get("rethink_reason") or "").lower()
    assert any(
        "rethink the implicated transformation strategy" in item.lower()
        for item in handoff["repair_scope"].get("must_preserve_invariants", [])
    )


def test_iteration_handoff_builds_artifact_schema_mismatch_from_verified_reviewer_evidence():
    state = {
        "iteration_count": 1,
        "execution_contract": {
            "required_outputs": ["artifacts/clean/accounts_snapshot_ml_ready.csv"],
        },
    }

    review_result = {
        "gate_results": [
            {
                "name": "outlier_policy_applied",
                "severity": "HARD",
                "passed": False,
                "issues": ["outlier_treatment_columns_missing_in_report"],
                "evidence": {
                    "report_present": True,
                    "report_file_exists": True,
                    "policy_target_columns": ["arr_current", "invoice_overdue_days"],
                    "report_columns_touched": [],
                    "missing_target_columns_in_report": ["arr_current", "invoice_overdue_days"],
                },
            }
        ]
    }

    handoff = _build_iteration_handoff(
        state=state,
        status="NEEDS_IMPROVEMENT",
        gate_context={},
        oc_report={"present": ["artifacts/clean/accounts_snapshot_ml_ready.csv"], "missing": []},
        review_result=review_result,
        qa_result={},
        evaluation_spec={"primary_metric": "pr_auc"},
    )

    repair_ground_truth = handoff["repair_ground_truth"]
    assert any(
        fact.get("fact") == "artifact_schema_mismatch"
        and fact.get("gate") == "outlier_policy_applied"
        for fact in repair_ground_truth.get("verified_facts", [])
    )
    assert any(
        delta.get("kind") == "artifact_schema_mismatch"
        and delta.get("gate") == "outlier_policy_applied"
        for delta in repair_ground_truth.get("causal_deltas", [])
        if isinstance(delta, dict)
    )
    assert any(
        "schema/field names" in str(goal or "").lower()
        for goal in repair_ground_truth.get("repair_goal", [])
    )
    repair_scope = handoff["repair_scope"]
    assert any(
        "outlier_policy_applied: the artifact exists" in item.lower()
        for item in repair_scope.get("active_findings", [])
    )
    assert "artifact_schema:outlier_policy_applied" in repair_scope.get("editable_targets", [])


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


def test_iteration_handoff_prefers_real_exception_over_heavy_runner_infra_marker():
    runtime_tail = (
        "HEAVY_RUNNER_ERROR_CONTEXT:\n"
        "Traceback (most recent call last):\n"
        "  File \"ml_script.py\", line 227, in main\n"
        "    if not df['case_id'].between(1, 20).all():\n"
        "AttributeError: 'DataFrame' object has no attribute 'between'\n"
        "HEAVY_RUNNER_INFRA_ERROR\n"
    )
    state = {
        "iteration_count": 2,
        "execution_contract": {"required_outputs": ["artifacts/ml/calibration_metrics.json"]},
        "execution_output": runtime_tail,
        "last_runtime_error_tail": runtime_tail,
    }

    handoff = _build_iteration_handoff(
        state=state,
        status="NEEDS_IMPROVEMENT",
        gate_context={
            "failed_gates": ["runtime_failure"],
            "required_fixes": ["Fix runtime failure."],
        },
        oc_report={"present": [], "missing": ["artifacts/ml/calibration_metrics.json"]},
        review_result={},
        qa_result={},
        evaluation_spec={"primary_metric": "mae"},
    )

    assert handoff["retry_context"]["specific_error"] == "AttributeError: 'DataFrame' object has no attribute 'between'"
    assert handoff["repair_ground_truth"]["failure_signature"] == "AttributeError: 'DataFrame' object has no attribute 'between'"
    assert "HEAVY_RUNNER_INFRA_ERROR" not in handoff["repair_ground_truth"]["failure_signature"]


def test_extract_verified_gate_feedback_filters_retry_packet_to_blocking_verified_failures():
    packet = {
        "failed_checks": [
            "identifier_columns_excluded_from_features",
            "arr_current_numeric_conversion_verified",
            "nps_forward_fill_temporal_integrity",
        ],
        "required_fixes": [
            "identifier_columns_excluded_from_features: csm_owner was not excluded from the feature set",
            "arr_current_numeric_conversion_verified: Column remains object type with currency strings",
            "nps_forward_fill_temporal_integrity: No evidence of partitioned forward-fill in cleaning_code",
        ],
        "hard_failures": [
            "identifier_columns_excluded_from_features",
            "arr_current_numeric_conversion_verified",
        ],
        "gate_results": [
            {
                "name": "training_cohort_filter_enforced",
                "severity": "HARD",
                "passed": None,
                "issues": [],
                "evidence": {"deterministic_support": "not_implemented"},
            },
            {
                "name": "arr_current_numeric_conversion_verified",
                "severity": "HARD",
                "passed": False,
                "issues": ["Column remains object type with currency strings"],
                "evidence": "column_stats_sample#arr_current",
            },
            {
                "name": "nps_forward_fill_temporal_integrity",
                "severity": "SOFT",
                "passed": False,
                "issues": ["No evidence of partitioned forward-fill in cleaning_code"],
                "evidence": "cleaning_code#nps_last_observed",
            },
        ],
    }

    verified = _extract_verified_gate_feedback(packet, hard_only=True)

    assert verified["failed_gates"] == ["arr_current_numeric_conversion_verified"]
    assert verified["hard_failures"] == ["arr_current_numeric_conversion_verified"]
    assert verified["required_fixes"] == [
        "arr_current_numeric_conversion_verified: Column remains object type with currency strings"
    ]


def test_iteration_handoff_uses_review_guided_retry_context_without_runtime_failure(tmp_path):
    metrics_path = tmp_path / "cv_metrics.json"
    metrics_path.write_text("{}", encoding="utf-8")
    state = {
        "iteration_count": 1,
        "execution_contract": {"required_outputs": [str(metrics_path)]},
        "execution_output": "HEAVY_RUNNER: status=success reason=local_runner_mode",
        "last_runtime_error_tail": "HEAVY_RUNNER: status=success reason=local_runner_mode",
        "primary_metric_snapshot": {
            "primary_metric_name": "top_decile_lift",
            "primary_metric_value": 9.92,
            "baseline_value": 9.80,
            "higher_is_better": True,
        },
        "reviewer_last_result": {
            "status": "APPROVED",
            "warnings": ["Temporal CV should stay grouped by snapshot month."],
        },
        "metrics_report": {
            "primary_metric_name": "top_decile_lift",
            "primary_metric_value": 9.92,
            "model_family": "LightGBM",
        },
    }

    handoff = _build_iteration_handoff(
        state=state,
        status="NEEDS_IMPROVEMENT",
        gate_context={
            "failed_gates": ["metric_above_random_baseline"],
            "required_fixes": ["Use the metric facts from cv_metrics.json and explain the reviewer/QA mismatch."],
            "feedback": "QA rejected on metric evidence.",
                "feedback_record": {},
            },
        oc_report={"present": [str(metrics_path)], "missing": []},
        review_result={"feedback": "Reviewer approved the baseline."},
        qa_result={"feedback": "QA rejected on metric evidence."},
        evaluation_spec={"primary_metric": "top_decile_lift"},
    )

    assert handoff["retry_context"]["error_type"] == "review_gate_failure"
    assert handoff["retry_context"]["repair_focus"] == "compliance"
    assert handoff["repair_ground_truth"]["root_cause_type"] == "review_gate_failure"
    assert handoff["feedback"]["runtime_error_tail"] == ""
    assert handoff["incumbent_brief"]["primary_metric"] == "top_decile_lift"
    assert handoff["incumbent_brief"]["incumbent_score"] == pytest.approx(9.92)


def test_iteration_handoff_reconciles_review_guided_fix_that_weakens_hard_ml_gate():
    state = {
        "iteration_count": 1,
        "execution_contract": {
            "required_outputs": [],
            "qa_gates": [
                {
                    "name": "holdout_sample_credible",
                    "severity": "HARD",
                    "params": {"min_rows": 1000},
                }
            ],
        },
        "primary_metric_snapshot": {
            "primary_metric_name": "pr_auc",
            "primary_metric_value": 0.8772,
        },
    }

    handoff = _build_iteration_handoff(
        state=state,
        status="NEEDS_IMPROVEMENT",
        gate_context={
            "failed_gates": ["holdout_sample_credible"],
            "required_fixes": ["Lower the minimum row threshold from 1000 to 400."],
            "feedback": "Reviewer requested a repair.",
            "feedback_record": {},
        },
        oc_report={"present": [], "missing": []},
        review_result={"feedback": "Holdout credibility gate failed."},
        qa_result={"feedback": "Holdout credibility gate failed."},
        evaluation_spec={"primary_metric": "pr_auc"},
    )

    assert all("400" not in str(item) for item in handoff["quality_focus"]["required_fixes"])
    assert handoff["repair_ground_truth"].get("governance_conflicts")
    assert any(
        "without weakening hard reviewer/qa/contract gates" in str(item).lower()
        for item in handoff["quality_focus"]["required_fixes"]
    )
