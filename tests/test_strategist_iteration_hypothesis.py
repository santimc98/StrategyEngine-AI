from src.agents.strategist import StrategistAgent
from src.utils.actor_critic_schemas import validate_iteration_hypothesis_packet


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
