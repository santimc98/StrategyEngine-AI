from src.agents.strategist import StrategistAgent
from src.utils.actor_critic_schemas import validate_iteration_hypothesis_packet


def test_generate_iteration_hypothesis_llm_mode_preserves_reasoned_selection(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("STRATEGIST_ITERATION_MODE", "llm")
    strategist = StrategistAgent()
    monkeypatch.setattr(
        strategist,
        "_generate_iteration_hypothesis_llm",
        lambda _ctx: {
            "packet_type": "iteration_hypothesis_packet",
            "packet_version": "1.0",
            "run_id": "run_test",
            "iteration": 2,
            "hypothesis_id": "h_reasoned1",
            "action": "APPLY",
            "hypothesis": {
                "technique": "target_encoding",
                "objective": "Exploit the incumbent signal on high-cardinality categoricals.",
                "target_columns": ["merchant_id"],
                "feature_scope": "model_features",
                "params": {"smoothing": 20},
                "expected_effect": {"target_error_modes": ["minority_class_recall_low"], "direction": "positive"},
            },
            "tracker_context": {"signature": "sig_target_encoding", "is_duplicate": False, "duplicate_of": None},
            "fallback_if_not_applicable": "NO_OP",
        },
    )
    monkeypatch.setattr(
        strategist,
        "_generate_iteration_hypothesis_deterministic",
        lambda _ctx: {
            "packet_type": "iteration_hypothesis_packet",
            "packet_version": "1.0",
            "run_id": "run_test",
            "iteration": 2,
            "hypothesis_id": "h_det0001",
            "action": "APPLY",
            "hypothesis": {
                "technique": "missing_indicators",
                "objective": "Deterministic fallback.",
                "target_columns": ["ALL_NUMERIC"],
                "feature_scope": "model_features",
                "params": {},
                "expected_effect": {"target_error_modes": ["metric_stagnation"], "direction": "positive"},
            },
            "tracker_context": {"signature": "sig_det", "is_duplicate": False, "duplicate_of": None},
            "fallback_if_not_applicable": "NO_OP",
        },
    )

    packet = strategist.generate_iteration_hypothesis(
        {
            "run_id": "run_test",
            "iteration": 2,
            "primary_metric_name": "roc_auc",
            "min_delta": 0.0005,
            "feature_engineering_plan": {"techniques": []},
            "critique_packet": {"error_modes": [{"id": "minority_class_recall_low"}]},
            "experiment_tracker": [],
        }
    )

    assert packet.get("hypothesis", {}).get("technique") == "target_encoding"
    assert strategist.last_iteration_meta.get("source") == "llm"


def test_generate_iteration_hypothesis_llm_mode_downgrades_duplicate_to_noop(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("STRATEGIST_ITERATION_MODE", "llm")
    strategist = StrategistAgent()
    monkeypatch.setattr(
        strategist,
        "_generate_iteration_hypothesis_llm",
        lambda _ctx: {
            "packet_type": "iteration_hypothesis_packet",
            "packet_version": "1.0",
            "run_id": "run_test",
            "iteration": 3,
            "hypothesis_id": "h_reasoned_dup",
            "action": "APPLY",
            "hypothesis": {
                "technique": "missing_indicators",
                "objective": "Retry prior signal.",
                "target_columns": ["ALL_NUMERIC"],
                "feature_scope": "model_features",
                "params": {},
                "expected_effect": {"target_error_modes": ["metric_stagnation"], "direction": "positive"},
            },
            "tracker_context": {"signature": "dup_sig", "is_duplicate": False, "duplicate_of": None},
            "fallback_if_not_applicable": "NO_OP",
        },
    )

    packet = strategist.generate_iteration_hypothesis(
        {
            "run_id": "run_test",
            "iteration": 3,
            "primary_metric_name": "roc_auc",
            "min_delta": 0.0005,
            "feature_engineering_plan": {"techniques": []},
            "critique_packet": {"error_modes": [{"id": "metric_stagnation"}]},
            "experiment_tracker": [{"signature": "dup_sig"}],
        }
    )

    assert packet.get("action") == "NO_OP"
    assert packet.get("tracker_context", {}).get("duplicate_of") == "dup_sig"
    assert strategist.last_iteration_meta.get("source") == "llm"


def test_generate_iteration_hypothesis_hybrid_mode_uses_deterministic_only_on_llm_failure(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("STRATEGIST_ITERATION_MODE", "hybrid")
    strategist = StrategistAgent()
    monkeypatch.setattr(strategist, "_generate_iteration_hypothesis_llm", lambda _ctx: {})
    monkeypatch.setattr(
        strategist,
        "_generate_iteration_hypothesis_deterministic",
        lambda _ctx: {
            "packet_type": "iteration_hypothesis_packet",
            "packet_version": "1.0",
            "run_id": "run_test",
            "iteration": 2,
            "hypothesis_id": "h_det0002",
            "action": "APPLY",
            "hypothesis": {
                "technique": "frequency_encoding",
                "objective": "Fallback deterministic hypothesis.",
                "target_columns": ["merchant_id"],
                "feature_scope": "model_features",
                "params": {},
                "expected_effect": {"target_error_modes": ["minority_class_recall_low"], "direction": "positive"},
            },
            "tracker_context": {"signature": "sig_freq", "is_duplicate": False, "duplicate_of": None},
            "fallback_if_not_applicable": "NO_OP",
        },
    )

    packet = strategist.generate_iteration_hypothesis(
        {
            "run_id": "run_test",
            "iteration": 2,
            "primary_metric_name": "roc_auc",
            "min_delta": 0.0005,
            "feature_engineering_plan": {"techniques": []},
            "critique_packet": {"error_modes": [{"id": "minority_class_recall_low"}]},
            "experiment_tracker": [],
        }
    )

    assert packet.get("hypothesis", {}).get("technique") == "frequency_encoding"
    assert strategist.last_iteration_meta.get("source") == "deterministic_fallback"


def test_generate_iteration_hypothesis_supports_all_numeric_macro(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("STRATEGIST_ITERATION_MODE", "deterministic")
    strategist = StrategistAgent()
    packet = strategist.generate_iteration_hypothesis(
        {
            "run_id": "run_test",
            "iteration": 2,
            "primary_metric_name": "roc_auc",
            "min_delta": 0.0005,
            "feature_engineering_plan": {
                "techniques": [
                    {
                        "technique": "missing_indicators",
                        "columns": ["ALL_NUMERIC"],
                        "rationale": "Improve stability.",
                    }
                ]
            },
            "critique_packet": {
                "error_modes": [{"id": "fold_instability"}],
                "analysis_summary": "Variance across folds.",
            },
            "experiment_tracker": [],
        }
    )
    valid, errors = validate_iteration_hypothesis_packet(packet)
    assert valid is True, errors
    assert packet.get("action") == "APPLY"
    assert "ALL_NUMERIC" in (packet.get("hypothesis", {}).get("target_columns") or [])


def test_generate_iteration_hypothesis_downgrades_duplicate_to_noop(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("STRATEGIST_ITERATION_MODE", "deterministic")
    strategist = StrategistAgent()
    first_packet = strategist.generate_iteration_hypothesis(
        {
            "run_id": "run_test",
            "iteration": 2,
            "primary_metric_name": "roc_auc",
            "min_delta": 0.0005,
            "feature_engineering_plan": {
                "techniques": [{"technique": "missing_indicators", "columns": ["ALL_NUMERIC"]}]
            },
            "critique_packet": {"error_modes": [{"id": "fold_instability"}]},
            "experiment_tracker": [],
        }
    )
    signature = (
        first_packet.get("tracker_context", {}).get("signature")
        if isinstance(first_packet.get("tracker_context"), dict)
        else None
    )
    second_packet = strategist.generate_iteration_hypothesis(
        {
            "run_id": "run_test",
            "iteration": 3,
            "primary_metric_name": "roc_auc",
            "min_delta": 0.0005,
            "feature_engineering_plan": {
                "techniques": [{"technique": "missing_indicators", "columns": ["ALL_NUMERIC"]}]
            },
            "critique_packet": {"error_modes": [{"id": "fold_instability"}]},
            "experiment_tracker": [{"signature": signature}],
        }
    )
    assert second_packet.get("action") == "NO_OP"
    assert second_packet.get("hypothesis", {}).get("technique") == "NO_OP"


def test_generate_iteration_hypothesis_fallback_uses_missingness_signal(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("STRATEGIST_ITERATION_MODE", "deterministic")
    strategist = StrategistAgent()
    packet = strategist.generate_iteration_hypothesis(
        {
            "run_id": "run_test",
            "iteration": 2,
            "primary_metric_name": "roc_auc",
            "min_delta": 0.0005,
            "feature_engineering_plan": {"techniques": []},
            "critique_packet": {"error_modes": [{"id": "fold_instability"}]},
            "dataset_profile": {"missingness": {"feature_a": 0.12, "feature_b": 0.0}},
            "experiment_tracker": [],
        }
    )
    assert packet.get("action") == "APPLY"
    assert packet.get("hypothesis", {}).get("technique") == "missing_indicators"


def test_generate_iteration_hypothesis_fallback_advances_when_first_signature_seen(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("STRATEGIST_ITERATION_MODE", "deterministic")
    strategist = StrategistAgent()
    context = {
        "run_id": "run_test",
        "iteration": 2,
        "primary_metric_name": "roc_auc",
        "min_delta": 0.0005,
        "feature_engineering_plan": {"techniques": []},
        "critique_packet": {"error_modes": [{"id": "fold_instability"}, {"id": "minority_class_recall_low"}]},
        "dataset_profile": {"high_cardinality_columns": ["merchant_id"]},
        "experiment_tracker": [],
    }
    first_packet = strategist.generate_iteration_hypothesis(context)
    signature = first_packet.get("tracker_context", {}).get("signature")
    second_packet = strategist.generate_iteration_hypothesis(
        {
            **context,
            "iteration": 3,
            "experiment_tracker": [{"signature": signature}],
        }
    )
    assert second_packet.get("action") == "APPLY"
    assert second_packet.get("hypothesis", {}).get("technique") != first_packet.get("hypothesis", {}).get("technique")


def test_generate_iteration_hypothesis_repairs_long_objective_without_noop(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("STRATEGIST_ITERATION_MODE", "deterministic")
    strategist = StrategistAgent()
    very_long_rationale = "Improve signal. " * 40
    packet = strategist.generate_iteration_hypothesis(
        {
            "run_id": "run_test",
            "iteration": 2,
            "primary_metric_name": "roc_auc",
            "min_delta": 0.0005,
            "feature_engineering_plan": {
                "techniques": [
                    {
                        "technique": "rare_category_grouping",
                        "columns": ["ALL_CATEGORICAL"],
                        "rationale": very_long_rationale,
                    }
                ]
            },
            "critique_packet": {"error_modes": [{"id": "minority_class_recall_low"}]},
            "experiment_tracker": [],
        }
    )
    valid, errors = validate_iteration_hypothesis_packet(packet)
    assert valid is True, errors
    assert packet.get("action") == "APPLY"
    objective = str(packet.get("hypothesis", {}).get("objective") or "")
    assert objective
    assert len(objective) <= 220


def test_generate_iteration_hypothesis_ranks_evidence_over_blueprint_order(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("STRATEGIST_ITERATION_MODE", "deterministic")
    strategist = StrategistAgent()
    packet = strategist.generate_iteration_hypothesis(
        {
            "run_id": "run_test",
            "iteration": 2,
            "primary_metric_name": "roc_auc",
            "min_delta": 0.0005,
            "feature_engineering_plan": {
                "techniques": [
                    {
                        "technique": "rare_category_grouping",
                        "columns": ["ALL_CATEGORICAL"],
                        "rationale": "Compress long-tail categories before the next round.",
                    }
                ]
            },
            "optimization_blueprint": {
                "improvement_actions": [
                    {
                        "technique": "stacking_ensemble",
                        "priority": 1,
                        "concrete_params": {"meta_model": "logistic_regression"},
                        "code_change_hint": "Try stacking as the next blueprint action.",
                    }
                ]
            },
            "critique_packet": {
                "error_modes": [{"id": "minority_class_recall_low", "severity": "high"}],
                "risk_flags": ["class_imbalance_sensitivity"],
            },
            "dataset_profile": {
                "basic_stats": {"n_rows": 300000},
                "high_cardinality_columns": ["merchant_id"],
                "column_types": {
                    "categorical": ["merchant_id", "segment"],
                    "numeric": ["amount"],
                },
                "cardinality": {"merchant_id": {"unique": 5000}},
            },
            "experiment_tracker": [],
        }
    )
    assert packet.get("action") == "APPLY"
    assert packet.get("hypothesis", {}).get("technique") == "rare_category_grouping"
    assert "Evidence-ranked" in str(packet.get("explanation") or "")


def test_generate_iteration_hypothesis_respects_optimization_policy(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("STRATEGIST_ITERATION_MODE", "deterministic")
    strategist = StrategistAgent()
    packet = strategist.generate_iteration_hypothesis(
        {
            "run_id": "run_test",
            "iteration": 2,
            "primary_metric_name": "roc_auc",
            "min_delta": 0.0005,
            "optimization_policy": {
                "allow_ensemble": False,
                "allow_feature_engineering": False,
                "allow_hpo": True,
            },
            "optimization_blueprint": {
                "improvement_actions": [
                    {
                        "technique": "stacking_ensemble",
                        "action_family": "ensemble_or_stacking",
                        "priority": 1,
                        "concrete_params": {"meta_model": "logistic_regression"},
                    },
                    {
                        "technique": "focused_lightgbm_hpo",
                        "action_family": "hyperparameter_search",
                        "priority": 2,
                        "concrete_params": {"n_trials": 8},
                    },
                    {
                        "technique": "kfold_target_encoding",
                        "action_family": "feature_engineering",
                        "priority": 3,
                        "concrete_params": {"smoothing": 10},
                    },
                ]
            },
            "critique_packet": {"error_modes": [{"id": "metric_stagnation"}]},
            "dataset_profile": {"basic_stats": {"n_rows": 10000}},
            "experiment_tracker": [],
        }
    )

    assert packet.get("action") == "APPLY"
    assert packet.get("hypothesis", {}).get("technique") == "focused_lightgbm_hpo"


def test_generate_iteration_hypothesis_downranks_recent_regressions(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("STRATEGIST_ITERATION_MODE", "deterministic")
    strategist = StrategistAgent()
    packet = strategist.generate_iteration_hypothesis(
        {
            "run_id": "run_test",
            "iteration": 4,
            "primary_metric_name": "roc_auc",
            "min_delta": 0.0005,
            "feature_engineering_plan": {
                "techniques": [
                    {
                        "technique": "target_encoding",
                        "columns": ["merchant_id"],
                        "rationale": "Encode high-cardinality categoricals with target statistics.",
                    },
                    {
                        "technique": "frequency_encoding",
                        "columns": ["merchant_id"],
                        "rationale": "Try a cheaper categorical compression alternative.",
                    },
                ]
            },
            "critique_packet": {
                "error_modes": [{"id": "minority_class_recall_low", "severity": "high"}],
                "risk_flags": ["class_imbalance_sensitivity"],
            },
            "dataset_profile": {
                "basic_stats": {"n_rows": 50000},
                "high_cardinality_columns": ["merchant_id"],
                "column_types": {
                    "categorical": ["merchant_id", "segment"],
                    "numeric": ["amount"],
                },
                "cardinality": {"merchant_id": {"unique": 2500}},
            },
            "experiment_tracker": [
                {
                    "technique": "target_encoding",
                    "delta": -0.0012,
                    "approved": False,
                }
            ],
        }
    )
    assert packet.get("action") == "APPLY"
    assert packet.get("hypothesis", {}).get("technique") == "frequency_encoding"
