import json
from pathlib import Path

import pytest

from src.graph.graph import (
    _build_metric_round_contract_lock,
    _build_hybrid_bundle_signature,
    _metric_round_has_deterministic_blockers,
    _promote_best_attempt,
    _resolve_metric_round_hybrid_policy,
    check_evaluation,
    check_metric_improvement_bootstrap_route,
    _is_improvement,
    run_metric_improvement_bootstrap,
    run_metric_improvement_finalize,
    _should_run_metric_improvement_round,
    _snapshot_ml_outputs,
    _restore_ml_outputs,
    _evaluate_metric_round_hypothesis_application,
)
from src.graph import graph as graph_mod


def test_should_run_metric_improvement_round_defaults_to_true_after_baseline_approved() -> None:
    state = {
        "review_verdict": "APPROVED",
        "reviewer_last_result": {"status": "APPROVED"},
        "qa_last_result": {"status": "APPROVED"},
        "execution_error": False,
        "sandbox_failed": False,
        "ml_improvement_attempted": False,
    }
    contract = {}
    assert _should_run_metric_improvement_round(state, contract) is True


def test_should_run_metric_improvement_round_requires_real_reviewer_pair_approval() -> None:
    state = {
        "review_verdict": "APPROVED",
        "reviewer_last_result": {"status": "APPROVED"},
        "qa_last_result": {"status": "REJECTED"},
        "execution_error": False,
        "sandbox_failed": False,
        "ml_improvement_attempted": False,
    }
    assert _should_run_metric_improvement_round(state, {}) is False


def test_should_run_metric_improvement_round_accepts_ml_review_stack_fallback() -> None:
    state = {
        "review_verdict": "APPROVE_WITH_WARNINGS",
        "ml_review_stack": {
            "reviewer": {"status": "APPROVED"},
            "qa_reviewer": {"status": "APPROVE_WITH_WARNINGS"},
        },
        "execution_error": False,
        "sandbox_failed": False,
        "ml_improvement_attempted": False,
    }
    assert _should_run_metric_improvement_round(state, {}) is True


def test_should_run_metric_improvement_round_ignores_nonblocking_needs_improvement_when_no_structured_blockers() -> None:
    state = {
        "review_verdict": "NEEDS_IMPROVEMENT",
        "review_board_verdict": {
            "status": "APPROVE_WITH_WARNINGS",
            "final_review_verdict": "NEEDS_IMPROVEMENT",
            "failed_areas": ["metric_gap"],
            "required_actions": ["Try additional model families."],
        },
        "last_gate_context": {"failed_gates": [], "required_fixes": [], "hard_failures": []},
        "reviewer_last_result": {"status": "APPROVED"},
        "qa_last_result": {"status": "APPROVED"},
        "execution_error": False,
        "sandbox_failed": False,
        "ml_improvement_attempted": False,
    }

    assert _should_run_metric_improvement_round(state, {}) is True


def test_bootstrap_metric_improvement_round_supports_nested_metric_aliases(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    metrics_path = Path("data/metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "cv_results": {
                    "overall_oof_auc": 0.9160376006743025,
                    "std_fold_auc": 0.0009317617858213929,
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(graph_mod, "append_experiment_entry", lambda *args, **kwargs: None)
    monkeypatch.setattr(graph_mod, "log_run_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        graph_mod.results_advisor,
        "generate_critique_packet",
        lambda _ctx: {
            "metric_comparison": {
                "baseline_value": 0.9160376006743025,
                "candidate_value": 0.9160376006743025,
                "meets_min_delta": False,
            },
            "validation_signals": {"validation_mode": "cv"},
            "error_modes": [],
            "analysis_summary": "Baseline available for improvement loop bootstrap.",
        },
    )
    monkeypatch.setattr(
        graph_mod.strategist,
        "generate_iteration_hypothesis",
        lambda _ctx: {
            "action": "APPLY",
            "hypothesis": {
                "technique": "feature_interactions",
                "objective": "Test one bounded improvement.",
                "target_columns": ["ALL_NUMERIC"],
                "feature_scope": "model_features",
                "params": {},
                "expected_effect": {"target_error_modes": ["metric_stagnation"], "direction": "positive"},
            },
            "tracker_context": {"signature": "hyp_nested_auc_alias", "is_duplicate": False, "duplicate_of": None},
            "success_criteria": {"primary_metric_name": "ROC-AUC", "min_delta": 0.0005, "must_pass_active_gates": True},
        },
    )
    monkeypatch.setattr(graph_mod.results_advisor, "last_critique_meta", {"mode": "deterministic", "source": "deterministic", "provider": "none", "model": None})
    monkeypatch.setattr(graph_mod.strategist, "last_iteration_meta", {"mode": "deterministic", "source": "deterministic", "model": None})

    contract = {
        "validation_requirements": {"primary_metric": "ROC-AUC"},
        "iteration_policy": {"metric_improvement_rounds": 1, "metric_min_delta": 0.0005},
        "feature_engineering_plan": {"techniques": [{"technique": "feature_interactions"}], "derived_columns": [], "notes": ""},
        "artifact_requirements": {"required_files": [{"path": "data/metrics.json"}]},
        "required_outputs": ["data/metrics.json"],
        "column_roles": {},
    }
    state = {
        "review_verdict": "APPROVED",
        "reviewer_last_result": {"status": "APPROVED"},
        "qa_last_result": {"status": "APPROVED"},
        "execution_error": False,
        "sandbox_failed": False,
        "ml_improvement_attempted": False,
        "iteration_count": 0,
        "generated_code": "def train():\n    return None\n",
        "data_summary": "Summary ready",
        "steward_context_ready": True,
        "steward_context_quality": {"ready": True, "reasons": [], "warnings": []},
        "feedback_history": [],
    }

    activated = graph_mod._bootstrap_metric_improvement_round(state, contract)

    assert activated is True
    assert state.get("ml_improvement_round_active") is True
    assert state.get("ml_improvement_primary_metric_name") == "ROC-AUC"
    assert state.get("ml_improvement_round_baseline_metric") == pytest.approx(0.9160376006743025, abs=1e-12)
    handoff = state.get("iteration_handoff", {})
    assert handoff.get("optimization_focus", {}).get("primary_metric_name") == "ROC-AUC"


def test_metric_round_contract_lock_filters_baseline_only_reviewer_gates() -> None:
    contract = {
        "validation_requirements": {"primary_metric": "roc_auc"},
        "allowed_feature_sets": {"forbidden_features": ["target"], "model_features": ["f1", "f2"]},
        "qa_gates": [
            {"name": "leakage_prevention_feature_exclusion", "severity": "HARD"},
            {"name": "probability_output_validation", "severity": "HARD"},
        ],
        "reviewer_gates": [
            {"name": "baseline_simplicity_enforcement", "severity": "HARD", "params": {"forbidden_techniques": ["stacking"]}},
            {"name": "model_selection_priority", "severity": "SOFT", "params": {"primary": "CatBoostClassifier"}},
            {"name": "submission_schema_compliance", "severity": "HARD"},
        ],
    }

    lock = _build_metric_round_contract_lock(contract, ["data/metrics.json"], "roc_auc")

    assert lock["qa_gates"] == [
        "leakage_prevention_feature_exclusion",
        "probability_output_validation",
    ]
    assert lock["reviewer_gates"] == ["submission_schema_compliance"]


def test_bootstrap_metric_round_active_gates_context_excludes_baseline_only_gates(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    metrics_path = Path("data/metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps({"roc_auc": 0.916}), encoding="utf-8")

    captured = {}

    def _capture_critique(ctx):
        captured.update(ctx if isinstance(ctx, dict) else {})
        return {
            "metric_comparison": {
                "baseline_value": 0.916,
                "candidate_value": 0.916,
                "meets_min_delta": False,
            },
            "validation_signals": {"validation_mode": "cv"},
            "error_modes": [],
            "analysis_summary": "Baseline available for improvement loop bootstrap.",
        }

    monkeypatch.setattr(graph_mod, "append_experiment_entry", lambda *args, **kwargs: None)
    monkeypatch.setattr(graph_mod, "log_run_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(graph_mod.results_advisor, "generate_critique_packet", _capture_critique)
    monkeypatch.setattr(graph_mod.results_advisor, "last_critique_meta", {"mode": "deterministic", "source": "deterministic", "provider": "none", "model": None})
    monkeypatch.setattr(
        graph_mod.strategist,
        "generate_iteration_hypothesis",
        lambda _ctx: {
            "action": "APPLY",
            "hypothesis": {
                "technique": "feature_interactions",
                "objective": "Test one bounded improvement.",
                "target_columns": ["ALL_NUMERIC"],
                "feature_scope": "model_features",
                "params": {},
                "expected_effect": {"target_error_modes": ["metric_stagnation"], "direction": "positive"},
            },
            "tracker_context": {"signature": "hyp_filtered_active_gates", "is_duplicate": False, "duplicate_of": None},
            "success_criteria": {"primary_metric_name": "roc_auc", "min_delta": 0.0005, "must_pass_active_gates": True},
        },
    )
    monkeypatch.setattr(graph_mod.strategist, "last_iteration_meta", {"mode": "deterministic", "source": "deterministic", "model": None})

    contract = {
        "validation_requirements": {"primary_metric": "roc_auc"},
        "iteration_policy": {"metric_improvement_rounds": 1, "metric_min_delta": 0.0005},
        "feature_engineering_plan": {"techniques": [{"technique": "feature_interactions"}], "derived_columns": [], "notes": ""},
        "artifact_requirements": {"required_files": [{"path": "data/metrics.json"}]},
        "required_outputs": ["data/metrics.json"],
        "column_roles": {},
        "qa_gates": [{"name": "leakage_prevention_feature_exclusion", "severity": "HARD"}],
        "reviewer_gates": [
            {"name": "baseline_simplicity_enforcement", "severity": "HARD", "params": {"forbidden_techniques": ["stacking"]}},
            {"name": "submission_schema_compliance", "severity": "HARD"},
        ],
    }
    state = {
        "review_verdict": "APPROVED",
        "reviewer_last_result": {"status": "APPROVED"},
        "qa_last_result": {"status": "APPROVED"},
        "execution_error": False,
        "sandbox_failed": False,
        "ml_improvement_attempted": False,
        "iteration_count": 0,
        "generated_code": "def train():\n    return None\n",
        "data_summary": "Summary ready",
        "steward_context_ready": True,
        "steward_context_quality": {"ready": True, "reasons": [], "warnings": []},
        "feedback_history": [],
    }

    activated = graph_mod._bootstrap_metric_improvement_round(state, contract)

    assert activated is True
    assert captured.get("active_gates_context") == [
        "leakage_prevention_feature_exclusion",
        "submission_schema_compliance",
    ]


def test_is_improvement_respects_min_delta_threshold() -> None:
    assert _is_improvement(0.8000, 0.8003, True, 0.0005) is False
    assert _is_improvement(0.8000, 0.8010, True, 0.0005) is True


def test_snapshot_and_restore_ml_outputs(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    metrics_path = Path("artifacts/data/metrics.json")
    submission_path = Path("artifacts/data/submission.csv")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    submission_path.parent.mkdir(parents=True, exist_ok=True)

    baseline_metrics = {"roc_auc": 0.81234}
    baseline_submission = "id,pred\n1,0.2\n"
    metrics_path.write_text(json.dumps(baseline_metrics), encoding="utf-8")
    submission_path.write_text(baseline_submission, encoding="utf-8")

    output_paths = [str(metrics_path).replace("\\", "/"), str(submission_path).replace("\\", "/")]
    snapshot_dir = Path("work/ml_baseline_snapshot")
    _snapshot_ml_outputs(output_paths, snapshot_dir)

    metrics_path.write_text(json.dumps({"roc_auc": 0.70001}), encoding="utf-8")
    submission_path.write_text("id,pred\n1,0.9\n", encoding="utf-8")

    _restore_ml_outputs(snapshot_dir, output_paths)

    restored_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    restored_submission = submission_path.read_text(encoding="utf-8")
    assert restored_metrics == baseline_metrics
    assert restored_submission == baseline_submission


def test_check_evaluation_restores_baseline_when_improvement_is_below_delta(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    metrics_path = Path("data/metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    baseline = {"roc_auc": 0.8000}
    metrics_path.write_text(json.dumps(baseline), encoding="utf-8")
    snapshot_dir = Path("work/ml_baseline_snapshot")
    output_paths = ["data/metrics.json"]
    _snapshot_ml_outputs(output_paths, snapshot_dir)

    # Candidate round result (approved but below min_delta)
    metrics_path.write_text(json.dumps({"roc_auc": 0.8003}), encoding="utf-8")
    state = {
        "review_verdict": "APPROVED",
        "execution_contract": {},
        "ml_improvement_round_active": True,
        "ml_improvement_primary_metric_name": "roc_auc",
        "ml_improvement_baseline_metric": 0.8000,
        "ml_improvement_min_delta": 0.0005,
        "ml_improvement_higher_is_better": True,
        "ml_improvement_output_paths": output_paths,
        "ml_improvement_snapshot_dir": str(snapshot_dir),
        "ml_improvement_baseline_review_verdict": "APPROVED",
        "feedback_history": [],
    }

    route = check_evaluation(state)
    restored = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert route == "approved"
    assert state.get("ml_improvement_kept") == "baseline"
    assert restored == baseline


def test_metric_round_blockers_ignore_stale_review_board_payload() -> None:
    state = {
        "execution_output": "Training completed successfully.",
        "runtime_fix_terminal": False,
        "sandbox_failed": False,
        "output_contract_report": {"overall_status": "ok", "missing": []},
        "last_gate_context": {"failed_gates": [], "hard_failures": []},
        "review_board_verdict": {
            "deterministic_blockers": [
                "runtime_failed",
                "missing_required_artifact:data/metrics.json",
            ]
        },
    }

    assert _metric_round_has_deterministic_blockers(state, include_review_signals=False) is False


def test_promote_best_attempt_clears_runtime_blockers(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    best_dir = Path("artifacts/best_attempt")
    best_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "artifact_index": [],
        "output_contract_report": {"overall_status": "ok", "missing": []},
        "execution_output": "Recovered execution output",
        "plots_local": [],
    }
    (best_dir / "best_attempt.json").write_text(json.dumps(metadata), encoding="utf-8")

    updates = _promote_best_attempt(
        {
            "best_attempt_dir": str(best_dir),
            "runtime_fix_terminal": True,
            "sandbox_failed": True,
            "execution_error": True,
            "last_runtime_error_tail": "Traceback ...",
        }
    )

    assert updates["execution_output"] == "Recovered execution output"
    assert updates["execution_output_stale"] is True
    assert updates["runtime_fix_terminal"] is False
    assert updates["sandbox_failed"] is False
    assert updates["execution_error"] is False
    assert updates["last_runtime_error_tail"] is None


def test_hybrid_policy_dedup_uses_merged_target_signature() -> None:
    hypothesis_packet = {
        "action": "APPLY",
        "hypothesis": {
            "technique": "optuna_hpo",
            "objective": "Tune the incumbent model.",
            "target_columns": ["score_hint"],
            "feature_scope": "model_features",
            "params": {},
        },
        "tracker_context": {"signature": "hyp_seed"},
    }
    feature_engineering_plan = {
        "techniques": [
            {"technique": "optuna_hpo", "columns": ["score_hint"], "params": {}},
            {"technique": "kfold_target_encoding", "columns": ["contract_type"], "params": {}},
        ]
    }
    failed_bundle_signature = _build_hybrid_bundle_signature(
        ["optuna_hpo", "kfold_target_encoding"],
        ["score_hint", "contract_type", "score_hint"],
    )
    tracker_entries = [
        {
            "event": "candidate_evaluated",
            "signature": failed_bundle_signature,
            "improved_by_metric": False,
        }
    ]

    packet, policy_meta = _resolve_metric_round_hybrid_policy(
        round_id=3,
        rounds_allowed=4,
        no_improve_streak=0,
        patience=2,
        min_delta=0.0001,
        higher_is_better=True,
        hypothesis_packet=hypothesis_packet,
        feature_engineering_plan=feature_engineering_plan,
        tracker_entries=tracker_entries,
    )

    assert packet["hypothesis"]["technique"] == "optuna_hpo"
    assert packet["hypothesis"]["params"].get("bundle_techniques") is None
    assert policy_meta["bundle_size"] == 1


def test_finalize_round_syncs_review_board_verdict_with_kept_artifact(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    metrics_path = Path("data/metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    baseline = {"roc_auc": 0.8000}
    metrics_path.write_text(json.dumps(baseline), encoding="utf-8")
    snapshot_dir = Path("work/ml_baseline_snapshot")
    output_paths = ["data/metrics.json"]
    _snapshot_ml_outputs(output_paths, snapshot_dir)

    # Candidate is approved but below delta, so baseline should be restored.
    metrics_path.write_text(json.dumps({"roc_auc": 0.8002}), encoding="utf-8")
    board_payload = {
        "status": "APPROVED",
        "final_review_verdict": "APPROVED",
        "summary": "Candidate approved by review board.",
    }
    Path("data/review_board_verdict.json").write_text(json.dumps(board_payload), encoding="utf-8")
    state = {
        "review_verdict": "APPROVED",
        "execution_contract": {},
        "ml_improvement_round_active": True,
        "ml_improvement_primary_metric_name": "roc_auc",
        "ml_improvement_baseline_metric": 0.8000,
        "ml_improvement_min_delta": 0.0005,
        "ml_improvement_higher_is_better": True,
        "ml_improvement_output_paths": output_paths,
        "ml_improvement_snapshot_dir": str(snapshot_dir),
        "ml_improvement_baseline_review_verdict": "APPROVED",
        "review_board_verdict": board_payload,
        "feedback_history": [],
    }

    route = check_evaluation(state)

    assert route == "approved"
    synced_payload = state.get("review_board_verdict") if isinstance(state.get("review_board_verdict"), dict) else {}
    finalization = synced_payload.get("metric_round_finalization") if isinstance(synced_payload.get("metric_round_finalization"), dict) else {}
    assert finalization.get("kept") == "baseline"
    assert finalization.get("final_metric") == baseline["roc_auc"]
    assert "METRIC_IMPROVEMENT_FINAL:" in str(synced_payload.get("summary") or "")

    persisted = json.loads(Path("data/review_board_verdict.json").read_text(encoding="utf-8"))
    persisted_finalization = persisted.get("metric_round_finalization") if isinstance(persisted.get("metric_round_finalization"), dict) else {}
    assert persisted_finalization.get("kept") == "baseline"


def test_finalize_round_keeps_canonical_candidate_when_advisor_delta_signal_is_wrong(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    report_path = Path("artifacts/ml/evaluation_summary.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_payload = {
        "primary_metric": "mean_multi_horizon_log_loss",
        "primary_metric_value": 0.40341051810159645,
        "mean_multi_horizon_log_loss": 0.40341051810159645,
    }
    report_path.write_text(json.dumps(baseline_payload), encoding="utf-8")
    snapshot_dir = Path("work/ml_incumbent_snapshot_r1")
    output_paths = ["artifacts/ml/evaluation_summary.json"]
    _snapshot_ml_outputs(output_paths, snapshot_dir)

    candidate_payload = {
        "primary_metric": "mean_multi_horizon_log_loss",
        "primary_metric_value": 0.330041811925438,
        "mean_multi_horizon_log_loss": 0.330041811925438,
    }
    report_path.write_text(json.dumps(candidate_payload), encoding="utf-8")
    board_payload = {
        "status": "APPROVED",
        "final_review_verdict": "APPROVED",
        "summary": "Candidate approved by review board.",
    }
    Path("data").mkdir(parents=True, exist_ok=True)
    Path("data/review_board_verdict.json").write_text(json.dumps(board_payload), encoding="utf-8")

    monkeypatch.setattr(
        graph_mod.results_advisor,
        "generate_critique_packet",
        lambda _ctx: {
            "metric_comparison": {
                "baseline_value": 0.40341051810159645,
                "candidate_value": 0.330041811925438,
                "meets_min_delta": False,
            },
            "validation_signals": {"validation_mode": "cv"},
            "error_modes": [],
            "analysis_summary": "Candidate metrics appear empty even though the artifact is present.",
        },
    )
    monkeypatch.setattr(
        graph_mod.results_advisor,
        "last_critique_meta",
        {"mode": "deterministic", "source": "deterministic", "provider": "none", "model": None},
    )
    monkeypatch.setattr(graph_mod, "append_experiment_entry", lambda *args, **kwargs: None)
    monkeypatch.setattr(graph_mod, "append_hypothesis_memory", lambda *args, **kwargs: None)
    monkeypatch.setattr(graph_mod, "log_run_event", lambda *args, **kwargs: None)

    contract = {
        "validation_requirements": {"primary_metric": "mean_multi_horizon_log_loss"},
        "iteration_policy": {"metric_improvement_rounds": 3, "metric_improvement_patience": 2},
        "artifact_requirements": {
            "required_files": [{"path": "artifacts/ml/evaluation_summary.json"}],
        },
        "required_outputs": ["artifacts/ml/evaluation_summary.json"],
        "column_roles": {},
    }
    state = {
        "review_verdict": "APPROVED",
        "execution_contract": contract,
        "ml_improvement_round_active": True,
        "ml_improvement_round_count": 1,
        "ml_improvement_current_round_id": 1,
        "ml_improvement_primary_metric_name": "mean_multi_horizon_log_loss",
        "ml_improvement_round_baseline_metric": 0.40341051810159645,
        "ml_improvement_baseline_metric": 0.40341051810159645,
        "ml_improvement_min_delta": 0.0005,
        "ml_improvement_higher_is_better": False,
        "ml_improvement_output_paths": output_paths,
        "ml_improvement_snapshot_dir": str(snapshot_dir),
        "ml_improvement_baseline_review_verdict": "APPROVED",
        "ml_improvement_rounds_allowed": 3,
        "ml_improvement_patience": 2,
        "ml_improvement_no_improve_streak": 0,
        "review_board_verdict": board_payload,
        "feedback_history": [],
    }

    route = check_evaluation(state)

    assert route == "approved"
    assert state.get("ml_improvement_kept") == "improved"
    loop_state = state.get("metric_loop_state") if isinstance(state.get("metric_loop_state"), dict) else {}
    selection = loop_state.get("selection") if isinstance(loop_state.get("selection"), dict) else {}
    final_entry = loop_state.get("final") if isinstance(loop_state.get("final"), dict) else {}
    assert selection.get("selected_label") == "candidate"
    assert selection.get("improved_by_metric") is True
    assert selection.get("advisor_meets_min_delta") is False
    assert final_entry.get("label") == "candidate"
    assert final_entry.get("metric_value") == pytest.approx(0.330041811925438, abs=1e-12)
    synced_payload = state.get("review_board_verdict") if isinstance(state.get("review_board_verdict"), dict) else {}
    finalization = synced_payload.get("metric_round_finalization") if isinstance(synced_payload.get("metric_round_finalization"), dict) else {}
    assert finalization.get("kept") == "candidate"
    assert finalization.get("final_metric") == pytest.approx(0.330041811925438, abs=1e-12)


def test_check_evaluation_logs_metric_improvement_round_completion(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    events = []
    from src.graph import graph as graph_mod

    monkeypatch.setattr(
        graph_mod,
        "log_run_event",
        lambda run_id, event_type, payload, log_dir="logs": events.append((run_id, event_type, payload)),
    )
    monkeypatch.setattr(graph_mod, "append_experiment_entry", lambda *args, **kwargs: None)

    metrics_path = Path("data/metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps({"roc_auc": 0.8006}), encoding="utf-8")
    snapshot_dir = Path("work/ml_baseline_snapshot")
    output_paths = ["data/metrics.json"]
    _snapshot_ml_outputs(output_paths, snapshot_dir)

    state = {
        "run_id": "run_improvement_trace",
        "review_verdict": "APPROVED",
        "execution_contract": {},
        "ml_improvement_round_active": True,
        "ml_improvement_primary_metric_name": "roc_auc",
        "ml_improvement_baseline_metric": 0.8000,
        "ml_improvement_min_delta": 0.0005,
        "ml_improvement_higher_is_better": True,
        "ml_improvement_output_paths": output_paths,
        "ml_improvement_snapshot_dir": str(snapshot_dir),
        "ml_improvement_baseline_review_verdict": "APPROVED",
        "feedback_history": [],
    }

    route = check_evaluation(state)

    assert route == "approved"
    event_types = [evt[1] for evt in events]
    assert "metric_improvement_round_complete" in event_types


def test_finalize_round_requests_continue_when_budget_and_patience_allow(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    metrics_path = Path("data/metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps({"roc_auc": 0.8001}), encoding="utf-8")
    snapshot_dir = Path("work/ml_incumbent_snapshot_r1")
    output_paths = ["data/metrics.json"]
    _snapshot_ml_outputs(output_paths, snapshot_dir)

    state = {
        "review_verdict": "APPROVED",
        "execution_contract": {"iteration_policy": {"metric_improvement_rounds": 3, "metric_improvement_patience": 2}},
        "ml_improvement_round_active": True,
        "ml_improvement_round_count": 1,
        "ml_improvement_current_round_id": 1,
        "ml_improvement_primary_metric_name": "roc_auc",
        "ml_improvement_round_baseline_metric": 0.8000,
        "ml_improvement_baseline_metric": 0.8000,
        "ml_improvement_min_delta": 0.0005,
        "ml_improvement_higher_is_better": True,
        "ml_improvement_output_paths": output_paths,
        "ml_improvement_snapshot_dir": str(snapshot_dir),
        "ml_improvement_baseline_review_verdict": "APPROVED",
        "ml_improvement_rounds_allowed": 3,
        "ml_improvement_patience": 2,
        "ml_improvement_no_improve_streak": 0,
        "feedback_history": [],
    }
    updates = run_metric_improvement_finalize(state)
    assert updates.get("metric_improvement_nodes_managed") is False
    assert updates.get("ml_improvement_continue") is True
    assert updates.get("ml_improvement_attempted") is False


def test_finalize_round_forced_by_budget_exceeded_stops_loop(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    metrics_path = Path("data/metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps({"roc_auc": 0.8001}), encoding="utf-8")
    snapshot_dir = Path("work/ml_incumbent_snapshot_r1")
    output_paths = ["data/metrics.json"]
    _snapshot_ml_outputs(output_paths, snapshot_dir)

    state = {
        "review_verdict": "APPROVED",
        "execution_contract": {"iteration_policy": {"metric_improvement_rounds": 3, "metric_improvement_patience": 2}},
        "ml_improvement_round_active": True,
        "ml_improvement_force_finalize_reason": "budget_exceeded",
        "ml_improvement_round_count": 1,
        "ml_improvement_current_round_id": 1,
        "ml_improvement_primary_metric_name": "roc_auc",
        "ml_improvement_round_baseline_metric": 0.8000,
        "ml_improvement_baseline_metric": 0.8000,
        "ml_improvement_min_delta": 0.0005,
        "ml_improvement_higher_is_better": True,
        "ml_improvement_output_paths": output_paths,
        "ml_improvement_snapshot_dir": str(snapshot_dir),
        "ml_improvement_baseline_review_verdict": "APPROVED",
        "ml_improvement_rounds_allowed": 3,
        "ml_improvement_patience": 2,
        "ml_improvement_no_improve_streak": 0,
        "feedback_history": [],
    }
    updates = run_metric_improvement_finalize(state)

    assert updates.get("metric_improvement_nodes_managed") is True
    assert updates.get("ml_improvement_continue") is False
    assert updates.get("ml_improvement_attempted") is True
    assert updates.get("ml_improvement_loop_complete") is True
    assert updates.get("ml_improvement_force_finalize_reason") == ""


def test_bootstrap_node_activates_next_round_after_continue(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "metrics.json").write_text(json.dumps({"roc_auc": 0.8002}), encoding="utf-8")
    events = []
    from src.graph import graph as graph_mod

    monkeypatch.setattr(
        graph_mod,
        "log_run_event",
        lambda run_id, event_type, payload, log_dir="logs": events.append((run_id, event_type, payload)),
    )
    monkeypatch.setattr(graph_mod, "append_experiment_entry", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        graph_mod.results_advisor,
        "generate_critique_packet",
        lambda _ctx: {
            "packet_type": "advisor_critique_packet",
            "packet_version": "1.0",
            "run_id": "run_test",
            "iteration": 2,
            "timestamp_utc": "2026-02-19T10:00:00+00:00",
            "primary_metric_name": "roc_auc",
            "higher_is_better": True,
            "metric_comparison": {
                "baseline_value": 0.8002,
                "candidate_value": 0.8002,
                "delta_abs": 0.0,
                "delta_rel": 0.0,
                "min_delta_required": 0.0005,
                "meets_min_delta": False,
            },
            "validation_signals": {"validation_mode": "cv"},
            "error_modes": [],
            "risk_flags": [],
            "active_gates_context": [],
            "analysis_summary": "No gain over incumbent.",
            "strictly_no_code_advice": True,
        },
    )
    monkeypatch.setattr(
        graph_mod.strategist,
        "generate_iteration_hypothesis",
        lambda _ctx: {
            "packet_type": "iteration_hypothesis_packet",
            "packet_version": "1.0",
            "run_id": "run_test",
            "iteration": 3,
            "hypothesis_id": "h_abcdef12",
            "action": "APPLY",
            "hypothesis": {
                "technique": "missing_indicators",
                "objective": "Improve round.",
                "target_columns": ["ALL_NUMERIC"],
                "feature_scope": "model_features",
                "params": {},
                "expected_effect": {"target_error_modes": ["metric_stagnation"], "direction": "positive"},
            },
            "application_constraints": {
                "edit_mode": "incremental",
                "max_code_regions_to_change": 3,
                "forbid_replanning": True,
                "forbid_model_family_switch": True,
                "must_keep": ["data_split_logic", "cv_protocol", "output_paths_contract"],
            },
            "success_criteria": {
                "primary_metric_name": "roc_auc",
                "min_delta": 0.0005,
                "must_pass_active_gates": True,
            },
            "tracker_context": {"signature": "hyp_next_round", "is_duplicate": False, "duplicate_of": None},
            "explanation": "Single hypothesis selected.",
            "fallback_if_not_applicable": "NO_OP",
        },
    )
    monkeypatch.setattr(graph_mod.strategist, "last_iteration_meta", {"mode": "deterministic", "source": "deterministic", "model": None})
    monkeypatch.setattr(graph_mod.results_advisor, "last_critique_meta", {"mode": "deterministic", "source": "deterministic", "provider": "none", "model": None})

    state = {
        "run_id": "run_test",
        "review_verdict": "APPROVED",
        "reviewer_last_result": {"status": "APPROVED"},
        "qa_last_result": {"status": "APPROVED"},
        "execution_error": False,
        "sandbox_failed": False,
        "ml_improvement_continue": True,
        "ml_improvement_attempted": False,
        "ml_improvement_round_active": False,
        "metric_improvement_nodes_managed": False,
        "ml_improvement_round_count": 1,
        "ml_improvement_rounds_allowed": 3,
        "ml_improvement_no_improve_streak": 1,
        "ml_improvement_patience": 2,
        "ml_improvement_primary_metric_name": "roc_auc",
        "ml_improvement_min_delta": 0.0005,
        "generated_code": "def train():\n    pass\n",
        "execution_contract": {
            "iteration_policy": {"metric_improvement_rounds": 3, "metric_improvement_patience": 2},
            "feature_engineering_plan": {"techniques": [{"technique": "missing_indicators"}], "derived_columns": [], "notes": ""},
            "artifact_requirements": {"required_files": [{"path": "data/metrics.json"}]},
            "required_outputs": ["data/metrics.json"],
            "column_roles": {},
        },
        "feedback_history": [],
        "ml_review_stack": {
            "runtime": {"status": "OK", "runtime_fix_terminal": False},
            "result_evaluator": {"status": "APPROVED"},
            "reviewer": {"status": "APPROVED"},
            "qa_reviewer": {"status": "APPROVED"},
            "results_advisor": {"status": "APPROVED"},
            "metric_improvement_context": {"active": False, "review_mode": None},
        },
    }
    updates = run_metric_improvement_bootstrap(state)
    merged_state = {**state, **updates}
    route = check_metric_improvement_bootstrap_route(merged_state)
    assert route == "retry"
    assert merged_state.get("ml_improvement_round_active") is True
    assert int(merged_state.get("ml_improvement_round_count", 0)) == 2
    persisted_handoff = json.loads(Path("data/iteration_handoff.json").read_text(encoding="utf-8"))
    assert persisted_handoff.get("mode") == "optimize"
    assert persisted_handoff.get("source") == "actor_critic_metric_improvement"
    assert persisted_handoff.get("hypothesis_packet", {}).get("hypothesis", {}).get("technique") == "missing_indicators"
    persisted_stack = json.loads(Path("data/ml_review_stack.json").read_text(encoding="utf-8"))
    assert persisted_stack.get("iteration_handoff", {}).get("source") == "actor_critic_metric_improvement"
    assert persisted_stack.get("metric_improvement_context", {}).get("active") is True


def test_check_finalize_route_bootstraps_next_round_when_continue_flag_is_set() -> None:
    state = {
        "iteration_count": 3,
        "review_verdict": "APPROVED",
        "execution_contract": {},
        "metric_improvement_nodes_managed": False,
        "ml_improvement_continue": True,
        "ml_improvement_round_active": False,
        "ml_improvement_round_count": 1,
        "ml_improvement_rounds_allowed": 3,
        "primary_metric_snapshot": {
            "primary_metric_name": "roc_auc",
            "primary_metric_value": 0.8001,
            "baseline_value": 0.8000,
        },
    }

    route = graph_mod.check_finalize_metric_improvement_route(state)

    assert route == "bootstrap_improvement_round"


def test_metric_round_apply_guard_requires_material_edit() -> None:
    state = {
        "ml_improvement_round_active": True,
        "ml_improvement_hypothesis_packet": {
            "action": "APPLY",
            "hypothesis": {"technique": "missing_indicators", "target_columns": ["ALL_NUMERIC"]},
        },
    }
    previous_code = (
        "def build_features(df):\n"
        "    return df\n"
        "\n"
        "def train(df):\n"
        "    X = build_features(df)\n"
        "    return X\n"
    )

    no_change = _evaluate_metric_round_hypothesis_application(
        state,
        previous_code=previous_code,
        candidate_code=previous_code,
    )
    assert no_change.get("enforced") is True
    assert no_change.get("applied") is False

    changed_code = (
        "def build_features(df):\n"
        "    df = df.copy()\n"
        "    for col in df.select_dtypes(include=['number']).columns:\n"
        "        df[f\"{col}_is_missing\"] = df[col].isna().astype('Int64')\n"
        "    return df\n"
        "\n"
        "def train(df):\n"
        "    X = build_features(df)\n"
        "    return X\n"
    )
    applied = _evaluate_metric_round_hypothesis_application(
        state,
        previous_code=previous_code,
        candidate_code=changed_code,
    )
    assert applied.get("enforced") is True
    assert applied.get("applied") is True


def test_metric_round_guarded_mode_ignores_advisory_verdict(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    metrics_path = Path("data/metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps({"roc_auc": 0.8}), encoding="utf-8")
    snapshot_dir = Path("work/ml_incumbent_snapshot_r1")
    output_paths = ["data/metrics.json"]
    _snapshot_ml_outputs(output_paths, snapshot_dir)
    metrics_path.write_text(json.dumps({"roc_auc": 0.801}), encoding="utf-8")

    monkeypatch.setattr(
        graph_mod.results_advisor,
        "generate_critique_packet",
        lambda _ctx: {
            "metric_comparison": {
                "baseline_value": 0.8,
                "candidate_value": 0.801,
                "meets_min_delta": True,
            },
            "validation_signals": {"validation_mode": "cv"},
            "error_modes": [],
        },
    )
    monkeypatch.setattr(
        graph_mod.results_advisor,
        "last_critique_meta",
        {"mode": "deterministic", "source": "deterministic", "provider": "none", "model": None},
    )

    state = {
        "review_verdict": "NEEDS_IMPROVEMENT",
        "execution_contract": {
            "iteration_policy": {"metric_round_review_mode": "hybrid_guarded"},
        },
        "ml_improvement_round_active": True,
        "ml_improvement_review_mode": "hybrid_guarded",
        "ml_improvement_primary_metric_name": "roc_auc",
        "ml_improvement_round_baseline_metric": 0.8,
        "ml_improvement_baseline_metric": 0.8,
        "ml_improvement_min_delta": 0.0005,
        "ml_improvement_higher_is_better": True,
        "ml_improvement_output_paths": output_paths,
        "ml_improvement_snapshot_dir": str(snapshot_dir),
        "ml_improvement_baseline_review_verdict": "APPROVED",
        "last_gate_context": {
            "failed_gates": ["qa_non_active_gate"],
            "hard_failures": ["qa_non_active_gate"],
        },
        "feedback_history": [],
    }

    route = check_evaluation(state)

    assert route == "approved"
    assert state.get("ml_improvement_kept") == "improved"


def test_bootstrap_metric_round_prefers_canonical_metric_loop_state_over_stale_legacy_scalars(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    metrics_path = Path("data/metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps({"mean_multi_horizon_log_loss": 0.0862}), encoding="utf-8")

    monkeypatch.setattr(graph_mod, "append_experiment_entry", lambda *args, **kwargs: None)
    monkeypatch.setattr(graph_mod, "log_run_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        graph_mod.results_advisor,
        "generate_critique_packet",
        lambda _ctx: {
            "metric_comparison": {
                "baseline_value": 0.0862,
                "candidate_value": 0.0862,
                "meets_min_delta": False,
            },
            "validation_signals": {"validation_mode": "cv"},
            "error_modes": [],
            "analysis_summary": "Baseline ready.",
        },
    )
    monkeypatch.setattr(
        graph_mod.strategist,
        "generate_iteration_hypothesis",
        lambda _ctx: {
            "action": "APPLY",
            "hypothesis": {
                "technique": "missing_indicators",
                "objective": "Improve calibration.",
                "target_columns": ["label_12h", "label_24h"],
                "feature_scope": "model_features",
                "params": {},
            },
            "tracker_context": {"signature": "hyp_metric_loop_state", "is_duplicate": False, "duplicate_of": None},
            "success_criteria": {"primary_metric_name": "mean_multi_horizon_log_loss", "min_delta": 0.0005},
        },
    )
    monkeypatch.setattr(graph_mod.results_advisor, "last_critique_meta", {"mode": "deterministic", "source": "deterministic", "provider": "none", "model": None})
    monkeypatch.setattr(graph_mod.strategist, "last_iteration_meta", {"mode": "deterministic", "source": "deterministic", "model": None})

    contract = {
        "validation_requirements": {"primary_metric": "mean_multi_horizon_log_loss"},
        "iteration_policy": {"metric_improvement_rounds": 2, "metric_min_delta": 0.0005},
        "artifact_requirements": {"required_files": [{"path": "data/metrics.json"}]},
        "required_outputs": ["data/metrics.json"],
        "column_roles": {},
    }
    state = {
        "run_id": "run_metric_loop_state",
        "review_verdict": "APPROVED",
        "reviewer_last_result": {"status": "APPROVED"},
        "qa_last_result": {"status": "APPROVED"},
        "execution_error": False,
        "sandbox_failed": False,
        "ml_improvement_attempted": False,
        "iteration_count": 0,
        "generated_code": "def train():\n    return None\n",
        "feedback_history": [],
        # Deliberately stale/conflicting legacy values.
        "ml_improvement_best_metric": 0.1475,
        "ml_improvement_higher_is_better": True,
        "metric_loop_state": {
            "schema_version": "v1",
            "target": {
                "name": "mean_multi_horizon_log_loss",
                "canonical_name": "logloss",
                "higher_is_better": False,
                "min_delta": 0.0005,
                "source": "test",
            },
            "round": {
                "round_id": 0,
                "rounds_allowed": 2,
                "patience": 2,
                "no_improve_streak": 0,
                "status": "complete",
                "baseline": {
                    "label": "round_baseline",
                    "metric_name": "mean_multi_horizon_log_loss",
                    "metric_value": 0.0862,
                    "metrics_payload": {"mean_multi_horizon_log_loss": 0.0862},
                },
            },
            "incumbent": {
                "label": "incumbent",
                "metric_name": "mean_multi_horizon_log_loss",
                "metric_value": 0.0862,
                "metrics_payload": {"mean_multi_horizon_log_loss": 0.0862},
            },
            "best_observed": {
                "label": "incumbent",
                "metric_name": "mean_multi_horizon_log_loss",
                "metric_value": 0.0862,
                "source": "test",
            },
        },
    }

    activated = graph_mod._bootstrap_metric_improvement_round(state, contract)

    assert activated is True
    handoff = state.get("iteration_handoff", {})
    optimization_context = handoff.get("optimization_context") if isinstance(handoff.get("optimization_context"), dict) else {}
    metric_snapshot = optimization_context.get("metric_snapshot") if isinstance(optimization_context.get("metric_snapshot"), dict) else {}
    assert metric_snapshot.get("primary_metric_name") == "mean_multi_horizon_log_loss"
    assert metric_snapshot.get("higher_is_better") is False
    assert metric_snapshot.get("best_metric_so_far") == pytest.approx(0.0862, abs=1e-12)


def test_bootstrap_metric_round_ignores_uninitialized_legacy_defaults_and_uses_evaluation_report_alias(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    report_path = Path("reports/evaluation_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps({"metrics": {"mean_multi_horizon_log_loss": 0.21735549007130867}}),
        encoding="utf-8",
    )

    monkeypatch.setattr(graph_mod, "append_experiment_entry", lambda *args, **kwargs: None)
    monkeypatch.setattr(graph_mod, "log_run_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        graph_mod.results_advisor,
        "generate_critique_packet",
        lambda _ctx: {
            "metric_comparison": {
                "baseline_value": 0.21735549007130867,
                "candidate_value": 0.21735549007130867,
                "meets_min_delta": False,
            },
            "validation_signals": {"validation_mode": "cv"},
            "error_modes": [],
            "analysis_summary": "Baseline ready.",
        },
    )
    monkeypatch.setattr(
        graph_mod.strategist,
        "generate_iteration_hypothesis",
        lambda _ctx: {
            "action": "APPLY",
            "hypothesis": {
                "technique": "cross_horizon_features",
                "objective": "Improve multi-horizon log loss.",
                "target_columns": ["label_12h", "label_24h"],
                "feature_scope": "model_features",
                "params": {},
            },
            "tracker_context": {"signature": "hyp_bootstrap_alias", "is_duplicate": False, "duplicate_of": None},
            "success_criteria": {"primary_metric_name": "mean_multi_horizon_log_loss", "min_delta": 0.0005},
        },
    )
    monkeypatch.setattr(
        graph_mod.results_advisor,
        "last_critique_meta",
        {"mode": "deterministic", "source": "deterministic", "provider": "none", "model": None},
    )
    monkeypatch.setattr(
        graph_mod.strategist,
        "last_iteration_meta",
        {"mode": "deterministic", "source": "deterministic", "model": None},
    )

    contract = {
        "validation_requirements": {"primary_metric": "mean_multi_horizon_log_loss"},
        "iteration_policy": {"metric_improvement_rounds": 2, "metric_min_delta": 0.0005},
        "artifact_requirements": {"required_files": [{"path": "reports/evaluation_report.json"}]},
        "required_outputs": ["reports/evaluation_report.json"],
        "column_roles": {},
    }
    state = {
        "run_id": "run_metric_loop_alias",
        "review_verdict": "APPROVED",
        "reviewer_last_result": {"status": "APPROVED"},
        "qa_last_result": {"status": "APPROVED"},
        "execution_error": False,
        "sandbox_failed": False,
        "ml_improvement_attempted": False,
        "iteration_count": 0,
        "generated_code": "def train():\n    return None\n",
        "feedback_history": [],
        # Default-style legacy values that should be ignored before the first round.
        "ml_improvement_best_metric": 0.0,
        "ml_improvement_higher_is_better": True,
        "ml_improvement_round_count": 0,
        "ml_improvement_current_round_id": 0,
        "metric_loop_state": {},
        "primary_metric_snapshot": {},
    }

    activated = graph_mod._bootstrap_metric_improvement_round(state, contract)

    assert activated is True
    loop_state = state.get("metric_loop_state") if isinstance(state.get("metric_loop_state"), dict) else {}
    target = loop_state.get("target") if isinstance(loop_state.get("target"), dict) else {}
    handoff = state.get("iteration_handoff", {})
    optimization_context = handoff.get("optimization_context") if isinstance(handoff.get("optimization_context"), dict) else {}
    metric_snapshot = optimization_context.get("metric_snapshot") if isinstance(optimization_context.get("metric_snapshot"), dict) else {}
    assert target.get("higher_is_better") is False
    assert metric_snapshot.get("higher_is_better") is False
    assert metric_snapshot.get("baseline_metric") == pytest.approx(0.21735549007130867, abs=1e-12)
    assert metric_snapshot.get("best_metric_so_far") == pytest.approx(0.21735549007130867, abs=1e-12)


def test_finalize_metric_round_persists_canonical_metric_loop_state_without_mixing_baseline_and_candidate(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    metrics_path = Path("data/metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    baseline = {"mean_multi_horizon_log_loss": 0.0862}
    metrics_path.write_text(json.dumps(baseline), encoding="utf-8")
    snapshot_dir = Path("work/ml_incumbent_snapshot_r1")
    output_paths = ["data/metrics.json"]
    _snapshot_ml_outputs(output_paths, snapshot_dir)

    candidate = {"mean_multi_horizon_log_loss": 0.1264}
    metrics_path.write_text(json.dumps(candidate), encoding="utf-8")
    monkeypatch.setattr(
        graph_mod.results_advisor,
        "generate_critique_packet",
        lambda _ctx: {
            "metric_comparison": {
                "baseline_value": 0.0862,
                "candidate_value": 0.1264,
                "meets_min_delta": False,
            },
            "validation_signals": {"validation_mode": "cv"},
            "error_modes": [],
        },
    )
    monkeypatch.setattr(
        graph_mod.results_advisor,
        "last_critique_meta",
        {"mode": "deterministic", "source": "deterministic", "provider": "none", "model": None},
    )

    state = {
        "review_verdict": "APPROVED",
        "execution_contract": {
            "validation_requirements": {"primary_metric": "mean_multi_horizon_log_loss"},
            "artifact_requirements": {"required_files": [{"path": "data/metrics.json"}]},
            "required_outputs": ["data/metrics.json"],
            "iteration_policy": {"metric_improvement_rounds": 3, "metric_improvement_patience": 2, "metric_min_delta": 0.0005},
        },
        "ml_improvement_round_active": True,
        "ml_improvement_round_count": 1,
        "ml_improvement_current_round_id": 1,
        "ml_improvement_primary_metric_name": "mean_multi_horizon_log_loss",
        "ml_improvement_round_baseline_metric": 0.0862,
        "ml_improvement_round_baseline_metrics": baseline,
        "ml_improvement_baseline_metric": 0.0862,
        "ml_improvement_baseline_metrics": baseline,
        "ml_improvement_best_metric": 0.0862,
        "ml_improvement_min_delta": 0.0005,
        "ml_improvement_higher_is_better": False,
        "ml_improvement_output_paths": output_paths,
        "ml_improvement_snapshot_dir": str(snapshot_dir),
        "ml_improvement_baseline_review_verdict": "APPROVED",
        "ml_improvement_rounds_allowed": 3,
        "ml_improvement_patience": 2,
        "ml_improvement_no_improve_streak": 0,
        "feedback_history": [],
        "metric_loop_state": {
            "schema_version": "v1",
            "target": {
                "name": "mean_multi_horizon_log_loss",
                "canonical_name": "logloss",
                "higher_is_better": False,
                "min_delta": 0.0005,
                "source": "test",
            },
            "round": {
                "round_id": 1,
                "rounds_allowed": 3,
                "patience": 2,
                "no_improve_streak": 0,
                "status": "active",
                "baseline": {
                    "label": "round_baseline",
                    "metric_name": "mean_multi_horizon_log_loss",
                    "metric_value": 0.0862,
                    "metrics_payload": baseline,
                    "review_verdict": "APPROVED",
                },
            },
            "incumbent": {
                "label": "incumbent",
                "metric_name": "mean_multi_horizon_log_loss",
                "metric_value": 0.0862,
                "metrics_payload": baseline,
            },
            "candidate": {
                "label": "candidate",
                "metric_name": "mean_multi_horizon_log_loss",
                "metric_value": None,
                "metrics_payload": {},
                "status": "pending",
            },
            "best_observed": {
                "label": "incumbent",
                "metric_name": "mean_multi_horizon_log_loss",
                "metric_value": 0.0862,
                "source": "test",
            },
        },
    }

    route = check_evaluation(state)

    assert route == "approved"
    loop_state = state.get("metric_loop_state") if isinstance(state.get("metric_loop_state"), dict) else {}
    assert loop_state.get("round", {}).get("baseline", {}).get("metric_value") == pytest.approx(0.0862, abs=1e-12)
    assert loop_state.get("candidate", {}).get("metric_value") == pytest.approx(0.1264, abs=1e-12)
    assert loop_state.get("final", {}).get("label") == "baseline"
    assert loop_state.get("final", {}).get("metric_value") == pytest.approx(0.0862, abs=1e-12)
    assert loop_state.get("incumbent", {}).get("metric_value") == pytest.approx(0.0862, abs=1e-12)
    assert loop_state.get("best_observed", {}).get("metric_value") == pytest.approx(0.0862, abs=1e-12)
    persisted_loop_state = json.loads(Path("data/metric_loop_state.json").read_text(encoding="utf-8"))
    assert persisted_loop_state.get("final", {}).get("label") == "baseline"
