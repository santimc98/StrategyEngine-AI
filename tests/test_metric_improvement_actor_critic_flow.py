import json

from src.graph import graph as graph_mod


def test_bootstrap_metric_improvement_round_builds_actor_critic_handoff(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    events = []
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
            "iteration": 1,
            "timestamp_utc": "2026-02-19T10:00:00+00:00",
            "primary_metric_name": "roc_auc",
            "higher_is_better": True,
            "metric_comparison": {
                "baseline_value": 0.80,
                "candidate_value": 0.80,
                "delta_abs": 0.0,
                "delta_rel": 0.0,
                "min_delta_required": 0.0005,
                "meets_min_delta": False,
            },
            "validation_signals": {"validation_mode": "cv"},
            "error_modes": [{"id": "fold_instability", "severity": "medium", "confidence": 0.8, "evidence": "x", "affected_scope": "cross_validation", "metric_impact_direction": "negative"}],
            "risk_flags": ["potential_overfitting"],
            "active_gates_context": [],
            "analysis_summary": "No gain over baseline.",
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
            "iteration": 2,
            "hypothesis_id": "h_abcdef12",
            "action": "APPLY",
            "hypothesis": {
                "technique": "missing_indicators",
                "objective": "Improve fold stability.",
                "target_columns": ["ALL_NUMERIC"],
                "feature_scope": "model_features",
                "params": {"indicator_suffix": "_is_missing"},
                "expected_effect": {"target_error_modes": ["fold_instability"], "direction": "positive"},
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
            "tracker_context": {"signature": "hyp_1234567890ab", "is_duplicate": False, "duplicate_of": None},
            "explanation": "Single hypothesis selected.",
            "fallback_if_not_applicable": "NO_OP",
        },
    )
    monkeypatch.setattr(graph_mod.strategist, "last_iteration_meta", {"mode": "deterministic", "source": "deterministic", "model": None})
    monkeypatch.setattr(graph_mod.results_advisor, "last_critique_meta", {"mode": "deterministic", "source": "deterministic", "provider": "none", "model": None})

    metrics_path = tmp_path / "data" / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps({"roc_auc": 0.8}), encoding="utf-8")

    contract = {
        "iteration_policy": {"metric_improvement_rounds": 1, "metric_min_delta": 0.0005},
        "feature_engineering_plan": {"techniques": [{"technique": "missing_indicators"}], "derived_columns": [], "notes": ""},
        "artifact_requirements": {"required_files": [{"path": "data/metrics.json"}]},
        "required_outputs": ["data/metrics.json"],
        "column_roles": {},
    }
    state = {
        "run_id": "run_test",
        "review_verdict": "APPROVED",
        "reviewer_last_result": {"status": "APPROVED"},
        "qa_last_result": {"status": "APPROVED"},
        "execution_error": False,
        "sandbox_failed": False,
        "ml_improvement_attempted": False,
        "iteration_count": 1,
        "generated_code": "def train():\n    pass\n",
        "data_summary": "Steward summary: core columns validated.",
        "steward_context_ready": True,
        "steward_context_quality": {"ready": True, "reasons": [], "warnings": []},
        "ml_review_stack": {
            "results_advisor": {
                "summary_lines": ["Baseline metrics verified; proceed with incremental FE."],
                "recommendations": ["Apply one FE hypothesis and keep CV fixed."],
                "risks": [],
            }
        },
        "feedback_history": [],
    }
    activated = graph_mod._bootstrap_metric_improvement_round(state, contract)
    assert activated is True
    assert state.get("ml_improvement_round_active") is True
    handoff = state.get("iteration_handoff", {})
    assert handoff.get("mode") == "optimize"
    assert handoff.get("source") == "actor_critic_metric_improvement"
    quality_focus = handoff.get("quality_focus") if isinstance(handoff.get("quality_focus"), dict) else {}
    assert quality_focus.get("status") == "OPTIMIZATION_REQUIRED"
    assert quality_focus.get("failed_gates") == []
    optimization_focus = handoff.get("optimization_focus") if isinstance(handoff.get("optimization_focus"), dict) else {}
    assert optimization_focus.get("primary_metric_name") == "roc_auc"
    assert isinstance(handoff.get("critic_packet"), dict)
    assert isinstance(handoff.get("hypothesis_packet"), dict)
    constraints = handoff.get("editor_constraints") if isinstance(handoff.get("editor_constraints"), dict) else {}
    assert constraints.get("must_apply_hypothesis") is True
    assert constraints.get("forbid_noop") is True
    assert constraints.get("patch_intensity") == "aggressive"
    feedback_history = state.get("feedback_history", [])
    assert feedback_history
    assert "STEWARD_FEEDBACK" in feedback_history[-1]
    assert "RESULTS_ADVISOR_FEEDBACK" in feedback_history[-1]
    assert "ITERATION_HYPOTHESIS_PACKET" in feedback_history[-1]
    event_types = [evt[1] for evt in events]
    assert "metric_improvement_round_start" in event_types
    assert "metric_improvement_round_activated" in event_types
