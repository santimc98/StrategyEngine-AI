from src.agents.results_advisor import ResultsAdvisorAgent
from src.utils.actor_critic_schemas import validate_advisor_critique_packet


def test_generate_critique_packet_includes_holdout_validation_signals() -> None:
    advisor = ResultsAdvisorAgent(api_key="")
    packet = advisor.generate_critique_packet(
        {
            "run_id": "run_test",
            "iteration": 1,
            "primary_metric_name": "roc_auc",
            "higher_is_better": True,
            "min_delta": 0.0005,
            "baseline_metrics": {
                "model_performance": {
                    "cv_roc_auc": 0.801,
                    "cv_std": 0.012,
                    "holdout_roc_auc": 0.796,
                }
            },
            "candidate_metrics": {
                "model_performance": {
                    "cv_roc_auc": 0.801,
                    "cv_std": 0.012,
                    "holdout_roc_auc": 0.796,
                }
            },
            "active_gates_context": ["required_artifacts_present", "target_variance_guard"],
            "dataset_profile": {"n_rows": 1200},
        }
    )
    valid, errors = validate_advisor_critique_packet(packet)
    assert valid is True, errors
    validation = packet.get("validation_signals", {})
    assert validation.get("validation_mode") in {"holdout", "cv_and_holdout"}
    assert packet.get("strictly_no_code_advice") is True


def test_generate_critique_packet_repairs_schema_drift_without_deterministic_fallback(monkeypatch) -> None:
    advisor = ResultsAdvisorAgent(api_key="")
    advisor.critique_mode = "hybrid"
    broken_packet = {
        "packet_type": "advisor_critique_packet",
        "packet_version": "1.0",
        "run_id": "run_test",
        "iteration": 2,
        "timestamp_utc": "2026-01-01T00:00:00Z",
        "primary_metric_name": "roc_auc",
        "higher_is_better": True,
        "metric_comparison": {
            "baseline_value": 0.8000,
            "candidate_value": 0.8010,
            "delta_abs": 0.0010,
            "delta_rel": 0.00125,
            "min_delta_required": 0.0005,
            "meets_min_delta": True,
        },
        "validation_signals": {
            "validation_mode": "cv_and_holdout",
            "cv": True,
            "holdout": {
                "metric_value": 0.799,
                "split_name": "holdout",
                "sample_count": 100,
                "class_distribution_shift": "low",
            },
            "generalization_gap": None,
        },
        "error_modes": [
            {
                "id": "overfit_risk",
                "severity": "critical",
                "confidence": 0.8,
                "evidence": "large gap",
                "affected_scope": "model",
                "metric_impact_direction": "down",
            }
        ],
        "risk_flags": ["overfit_risk"],
        "active_gates_context": ["required_artifacts_present"],
        "analysis_summary": "Schema drift packet",
        "strictly_no_code_advice": True,
    }
    monkeypatch.setattr(advisor, "_generate_critique_packet_llm", lambda context: broken_packet)

    packet = advisor.generate_critique_packet(
        {
            "run_id": "run_test",
            "iteration": 2,
            "primary_metric_name": "roc_auc",
            "higher_is_better": True,
            "min_delta": 0.0005,
            "baseline_metrics": {
                "model_performance": {
                    "cv_roc_auc": 0.8000,
                    "cv_std": 0.010,
                    "holdout_roc_auc": 0.799,
                }
            },
            "candidate_metrics": {
                "model_performance": {
                    "cv_roc_auc": 0.8010,
                    "cv_std": 0.011,
                    "holdout_roc_auc": 0.800,
                }
            },
            "active_gates_context": ["required_artifacts_present"],
            "dataset_profile": {"n_rows": 900},
        }
    )

    valid, errors = validate_advisor_critique_packet(packet)
    assert valid is True, errors
    assert packet.get("error_modes", [{}])[0].get("severity") == "high"
    assert packet.get("error_modes", [{}])[0].get("metric_impact_direction") == "negative"
    assert (advisor.last_critique_meta or {}).get("source") in {"llm_repair_normalized", "llm_repair_pass"}


def test_generate_critique_packet_treats_empty_candidate_metrics_as_missing() -> None:
    advisor = ResultsAdvisorAgent(api_key="")
    advisor.critique_mode = "deterministic"
    packet = advisor.generate_critique_packet(
        {
            "run_id": "run_test",
            "iteration": 3,
            "primary_metric_name": "mean_multi_horizon_log_loss",
            "higher_is_better": False,
            "min_delta": 0.0005,
            "baseline_metrics": {
                "primary_metric": "mean_multi_horizon_log_loss",
                "mean_multi_horizon_log_loss": 0.330041811925438,
            },
            "candidate_metrics": {},
            "active_gates_context": ["required_artifacts_present"],
            "dataset_profile": {"n_rows": 316},
        }
    )

    valid, errors = validate_advisor_critique_packet(packet)
    assert valid is True, errors
    comparison = packet.get("metric_comparison", {})
    assert comparison.get("baseline_value") == comparison.get("candidate_value")
    assert comparison.get("meets_min_delta") is False
