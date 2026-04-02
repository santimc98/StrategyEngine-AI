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


def test_bootstrap_metric_improvement_round_exploit_phase_builds_compatible_bundle(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(graph_mod, "append_experiment_entry", lambda *args, **kwargs: None)
    monkeypatch.setattr(graph_mod, "log_run_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        graph_mod,
        "load_recent_experiment_entries",
        lambda run_id, k=20: [
            {
                "event": "candidate_evaluated",
                "approved": True,
                "delta": 0.0012,
                "technique": "Heart Rate Reserve Proxy",
            }
        ],
    )
    monkeypatch.setattr(
        graph_mod.results_advisor,
        "generate_critique_packet",
        lambda _ctx: {
            "packet_type": "advisor_critique_packet",
            "packet_version": "1.0",
            "run_id": "run_exploit",
            "iteration": 2,
            "timestamp_utc": "2026-02-19T10:00:00+00:00",
            "primary_metric_name": "roc_auc",
            "higher_is_better": True,
            "metric_comparison": {
                "baseline_value": 0.90,
                "candidate_value": 0.90,
                "delta_abs": 0.0,
                "delta_rel": 0.0,
                "min_delta_required": 0.0005,
                "meets_min_delta": False,
            },
            "validation_signals": {"validation_mode": "cv"},
            "error_modes": [],
            "risk_flags": [],
            "active_gates_context": [],
            "analysis_summary": "Stagnation detected.",
            "strictly_no_code_advice": True,
        },
    )
    monkeypatch.setattr(
        graph_mod.strategist,
        "generate_iteration_hypothesis",
        lambda _ctx: {
            "packet_type": "iteration_hypothesis_packet",
            "packet_version": "1.0",
            "run_id": "run_exploit",
            "iteration": 3,
            "hypothesis_id": "h_abcdef99",
            "action": "APPLY",
            "hypothesis": {
                "technique": "Type Casting",
                "objective": "Normalize dtypes",
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
            "tracker_context": {"signature": "hyp_type_casting", "is_duplicate": False, "duplicate_of": None},
            "explanation": "Single hypothesis selected.",
            "fallback_if_not_applicable": "NO_OP",
        },
    )
    monkeypatch.setattr(graph_mod.results_advisor, "last_critique_meta", {"mode": "deterministic", "source": "deterministic", "provider": "none", "model": None})
    monkeypatch.setattr(graph_mod.strategist, "last_iteration_meta", {"mode": "deterministic", "source": "deterministic", "model": None})

    metrics_path = tmp_path / "data" / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps({"roc_auc": 0.90}), encoding="utf-8")

    contract = {
        "iteration_policy": {"metric_improvement_rounds": 4, "metric_improvement_patience": 2, "metric_min_delta": 0.0005},
        "feature_engineering_plan": {
            "techniques": [
                {"technique": "Type Casting", "columns": ["ALL_NUMERIC"]},
                {"technique": "Heart Rate Reserve Proxy", "columns": ["max_hr", "age"]},
            ],
            "derived_columns": [],
            "notes": "",
        },
        "artifact_requirements": {"required_files": [{"path": "data/metrics.json"}]},
        "required_outputs": ["data/metrics.json"],
        "column_roles": {},
        "allowed_feature_sets": {"model_features": ["age", "max_hr"], "forbidden_features": []},
        "validation_requirements": {"primary_metric": "roc_auc", "split_column": "is_train"},
    }
    state = {
        "run_id": "run_exploit",
        "review_verdict": "APPROVED",
        "reviewer_last_result": {"status": "APPROVED"},
        "qa_last_result": {"status": "APPROVED"},
        "execution_error": False,
        "sandbox_failed": False,
        "ml_improvement_attempted": False,
        "ml_improvement_round_count": 1,
        "ml_improvement_no_improve_streak": 1,
        "ml_improvement_patience": 2,
        "iteration_count": 2,
        "generated_code": "def train():\n    pass\n",
        "feedback_history": [],
    }

    activated = graph_mod._bootstrap_metric_improvement_round(state, contract)
    assert activated is True
    handoff = state.get("iteration_handoff", {})
    optimization_context = handoff.get("optimization_context") if isinstance(handoff.get("optimization_context"), dict) else {}
    policy = optimization_context.get("policy") if isinstance(optimization_context.get("policy"), dict) else {}
    hypothesis = handoff.get("hypothesis_packet", {}).get("hypothesis", {})
    bundle = hypothesis.get("params", {}).get("bundle_techniques", [])

    assert policy.get("phase") == "exploit"
    assert bundle == []
    assert hypothesis.get("technique") == "Type Casting"
    assert policy.get("strategist_packet_preserved") is True


def test_bootstrap_metric_improvement_round_forces_apply_on_first_round_when_noop(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(graph_mod, "append_experiment_entry", lambda *args, **kwargs: None)
    monkeypatch.setattr(graph_mod, "log_run_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(graph_mod, "load_recent_experiment_entries", lambda run_id, k=20: [])
    monkeypatch.setattr(
        graph_mod.results_advisor,
        "generate_critique_packet",
        lambda _ctx: {
            "packet_type": "advisor_critique_packet",
            "packet_version": "1.0",
            "run_id": "run_force_apply",
            "iteration": 1,
            "timestamp_utc": "2026-02-19T10:00:00+00:00",
            "primary_metric_name": "roc_auc",
            "higher_is_better": True,
            "metric_comparison": {
                "baseline_value": 0.8,
                "candidate_value": 0.8,
                "delta_abs": 0.0,
                "delta_rel": 0.0,
                "min_delta_required": 0.0005,
                "meets_min_delta": False,
            },
            "validation_signals": {"validation_mode": "cv"},
            "error_modes": [],
            "risk_flags": [],
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
            "run_id": "run_force_apply",
            "iteration": 2,
            "hypothesis_id": "h_noop0001",
            "action": "NO_OP",
            "hypothesis": {
                "technique": "NO_OP",
                "objective": "No-op fallback",
                "target_columns": ["ALL_NUMERIC"],
                "feature_scope": "model_features",
                "params": {},
                "expected_effect": {"target_error_modes": ["metric_stagnation"], "direction": "neutral"},
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
            "tracker_context": {"signature": "hyp_noop", "is_duplicate": True, "duplicate_of": "hyp_noop"},
            "explanation": "No-op fallback",
            "fallback_if_not_applicable": "NO_OP",
        },
    )
    monkeypatch.setattr(graph_mod.results_advisor, "last_critique_meta", {"mode": "deterministic", "source": "deterministic", "provider": "none", "model": None})
    monkeypatch.setattr(graph_mod.strategist, "last_iteration_meta", {"mode": "deterministic", "source": "deterministic", "model": None})

    metrics_path = tmp_path / "data" / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps({"roc_auc": 0.8}), encoding="utf-8")

    contract = {
        "iteration_policy": {"metric_improvement_rounds": 2, "metric_improvement_patience": 2, "metric_min_delta": 0.0005},
        "feature_engineering_plan": {
            "techniques": [
                {"technique": "missing_indicators", "columns": ["ALL_NUMERIC"]},
                {"technique": "rare_category_grouping", "columns": ["ALL_CATEGORICAL"]},
            ],
            "derived_columns": [],
            "notes": "",
        },
        "artifact_requirements": {"required_files": [{"path": "data/metrics.json"}]},
        "required_outputs": ["data/metrics.json"],
        "column_roles": {},
    }
    state = {
        "run_id": "run_force_apply",
        "review_verdict": "APPROVED",
        "reviewer_last_result": {"status": "APPROVED"},
        "qa_last_result": {"status": "APPROVED"},
        "execution_error": False,
        "sandbox_failed": False,
        "ml_improvement_attempted": False,
        "ml_improvement_round_count": 0,
        "iteration_count": 1,
        "generated_code": "def train():\n    pass\n",
        "feedback_history": [],
    }

    activated = graph_mod._bootstrap_metric_improvement_round(state, contract)
    assert activated is False
    assert state.get("ml_improvement_loop_complete") is True
    assert state.get("stop_reason") == "IMPROVEMENT_ROUND_DUPLICATE_NOOP"


def test_bootstrap_metric_improvement_round_stops_on_duplicate_noop_after_first_round(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(graph_mod, "append_experiment_entry", lambda *args, **kwargs: None)
    monkeypatch.setattr(graph_mod, "log_run_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        graph_mod,
        "load_recent_experiment_entries",
        lambda run_id, k=20: [
            {
                "event": "hypothesis_proposed",
                "signature": "hyp_dup_same",
                "technique": "missing_indicators",
            }
        ],
    )
    monkeypatch.setattr(
        graph_mod.results_advisor,
        "generate_critique_packet",
        lambda _ctx: {
            "packet_type": "advisor_critique_packet",
            "packet_version": "1.0",
            "run_id": "run_dup_noop",
            "iteration": 2,
            "timestamp_utc": "2026-02-19T10:00:00+00:00",
            "primary_metric_name": "roc_auc",
            "higher_is_better": True,
            "metric_comparison": {
                "baseline_value": 0.8,
                "candidate_value": 0.8,
                "delta_abs": 0.0,
                "delta_rel": 0.0,
                "min_delta_required": 0.0005,
                "meets_min_delta": False,
            },
            "validation_signals": {"validation_mode": "cv"},
            "error_modes": [],
            "risk_flags": [],
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
            "run_id": "run_dup_noop",
            "iteration": 3,
            "hypothesis_id": "h_dup0002",
            "action": "NO_OP",
            "hypothesis": {
                "technique": "NO_OP",
                "objective": "No-op fallback",
                "target_columns": ["ALL_NUMERIC"],
                "feature_scope": "model_features",
                "params": {},
                "expected_effect": {"target_error_modes": ["metric_stagnation"], "direction": "neutral"},
            },
            "tracker_context": {"signature": "hyp_dup_same", "is_duplicate": True, "duplicate_of": "hyp_dup_same"},
            "fallback_if_not_applicable": "NO_OP",
        },
    )
    monkeypatch.setattr(graph_mod.results_advisor, "last_critique_meta", {"mode": "deterministic", "source": "deterministic", "provider": "none", "model": None})
    monkeypatch.setattr(graph_mod.strategist, "last_iteration_meta", {"mode": "deterministic", "source": "deterministic", "model": None})

    metrics_path = tmp_path / "data" / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps({"roc_auc": 0.8}), encoding="utf-8")

    contract = {
        "iteration_policy": {"metric_improvement_rounds": 3, "metric_improvement_patience": 2, "metric_min_delta": 0.0005},
        "feature_engineering_plan": {
            "techniques": [],
            "derived_columns": [],
            "notes": "",
        },
        "artifact_requirements": {"required_files": [{"path": "data/metrics.json"}]},
        "required_outputs": ["data/metrics.json"],
        "column_roles": {},
    }
    state = {
        "run_id": "run_dup_noop",
        "review_verdict": "APPROVED",
        "reviewer_last_result": {"status": "APPROVED"},
        "qa_last_result": {"status": "APPROVED"},
        "execution_error": False,
        "sandbox_failed": False,
        "ml_improvement_attempted": False,
        "ml_improvement_round_count": 1,
        "iteration_count": 2,
        "generated_code": "def train():\n    pass\n",
        "feedback_history": [],
    }

    activated = graph_mod._bootstrap_metric_improvement_round(state, contract)
    assert activated is False
    assert state.get("ml_improvement_loop_complete") is True
    assert state.get("ml_improvement_attempted") is True
    assert state.get("stop_reason") == "IMPROVEMENT_ROUND_DUPLICATE_NOOP"


def test_bootstrap_metric_improvement_round_recovers_diversity_after_negative_streak(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(graph_mod, "append_experiment_entry", lambda *args, **kwargs: None)
    monkeypatch.setattr(graph_mod, "log_run_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        graph_mod,
        "load_recent_experiment_entries",
        lambda run_id, k=20: [
            {
                "event": "candidate_evaluated",
                "delta": -0.0011,
                "technique": "missing_indicators",
                "approved": True,
                "signature": "hyp_miss_1",
            },
            {
                "event": "candidate_evaluated",
                "delta": -0.0008,
                "technique": "missing_indicators",
                "approved": True,
                "signature": "hyp_miss_2",
            },
        ],
    )
    monkeypatch.setattr(
        graph_mod.results_advisor,
        "generate_critique_packet",
        lambda _ctx: {
            "packet_type": "advisor_critique_packet",
            "packet_version": "1.0",
            "run_id": "run_diversity",
            "iteration": 3,
            "timestamp_utc": "2026-02-19T10:00:00+00:00",
            "primary_metric_name": "roc_auc",
            "higher_is_better": True,
            "metric_comparison": {
                "baseline_value": 0.9,
                "candidate_value": 0.9,
                "delta_abs": 0.0,
                "delta_rel": 0.0,
                "min_delta_required": 0.0005,
                "meets_min_delta": False,
            },
            "validation_signals": {"validation_mode": "cv"},
            "error_modes": [],
            "risk_flags": [],
            "active_gates_context": [],
            "analysis_summary": "Consecutive regression detected.",
            "strictly_no_code_advice": True,
        },
    )
    monkeypatch.setattr(
        graph_mod.strategist,
        "generate_iteration_hypothesis",
        lambda _ctx: {
            "packet_type": "iteration_hypothesis_packet",
            "packet_version": "1.0",
            "run_id": "run_diversity",
            "iteration": 4,
            "hypothesis_id": "h_div0004",
            "action": "APPLY",
            "hypothesis": {
                "technique": "missing_indicators",
                "objective": "Retry same technique",
                "target_columns": ["ALL_NUMERIC"],
                "feature_scope": "model_features",
                "params": {},
                "expected_effect": {"target_error_modes": ["metric_stagnation"], "direction": "positive"},
            },
            "tracker_context": {"signature": "hyp_miss_2", "is_duplicate": False, "duplicate_of": None},
            "fallback_if_not_applicable": "NO_OP",
        },
    )
    monkeypatch.setattr(graph_mod.results_advisor, "last_critique_meta", {"mode": "deterministic", "source": "deterministic", "provider": "none", "model": None})
    monkeypatch.setattr(graph_mod.strategist, "last_iteration_meta", {"mode": "deterministic", "source": "deterministic", "model": None})

    metrics_path = tmp_path / "data" / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps({"roc_auc": 0.9}), encoding="utf-8")

    contract = {
        "iteration_policy": {"metric_improvement_rounds": 5, "metric_improvement_patience": 3, "metric_min_delta": 0.0005},
        "feature_engineering_plan": {
            "techniques": [
                {"technique": "missing_indicators", "target_columns": ["ALL_NUMERIC"]},
                {"technique": "rare_category_grouping", "target_columns": ["ALL_CATEGORICAL"]},
            ],
            "derived_columns": [],
            "notes": "",
        },
        "artifact_requirements": {"required_files": [{"path": "data/metrics.json"}]},
        "required_outputs": ["data/metrics.json"],
        "column_roles": {},
    }
    state = {
        "run_id": "run_diversity",
        "review_verdict": "APPROVED",
        "reviewer_last_result": {"status": "APPROVED"},
        "qa_last_result": {"status": "APPROVED"},
        "execution_error": False,
        "sandbox_failed": False,
        "ml_improvement_attempted": False,
        "ml_improvement_round_count": 2,
        "ml_improvement_no_improve_streak": 2,
        "iteration_count": 3,
        "generated_code": "def train():\n    pass\n",
        "feedback_history": [],
    }

    activated = graph_mod._bootstrap_metric_improvement_round(state, contract)
    assert activated is True
    handoff = state.get("iteration_handoff") if isinstance(state.get("iteration_handoff"), dict) else {}
    hypothesis_packet = handoff.get("hypothesis_packet") if isinstance(handoff.get("hypothesis_packet"), dict) else {}
    technique = str(hypothesis_packet.get("hypothesis", {}).get("technique") or "").strip().lower()
    policy = handoff.get("optimization_context", {}).get("policy", {}) if isinstance(handoff.get("optimization_context"), dict) else {}
    assert "missing_indicators" in technique
    assert policy.get("diversity_recovery_applied") is False
    assert int(policy.get("negative_delta_streak", 0) or 0) >= 2
    assert policy.get("strategist_packet_preserved") is True
