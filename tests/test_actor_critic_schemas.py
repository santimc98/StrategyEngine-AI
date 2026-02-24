from src.utils.actor_critic_schemas import (
    normalize_target_columns,
    validate_advisor_critique_packet,
    validate_experiment_hypothesis_packet_v2,
    validate_experiment_result_packet_v2,
    validate_iteration_hypothesis_packet,
)


def test_validate_advisor_critique_packet_supports_cv_and_holdout() -> None:
    packet = {
        "packet_type": "advisor_critique_packet",
        "packet_version": "1.0",
        "run_id": "run_x",
        "iteration": 1,
        "timestamp_utc": "2026-02-19T10:00:00+00:00",
        "primary_metric_name": "roc_auc",
        "higher_is_better": True,
        "metric_comparison": {
            "baseline_value": 0.80,
            "candidate_value": 0.81,
            "delta_abs": 0.01,
            "delta_rel": 0.0125,
            "min_delta_required": 0.0005,
            "meets_min_delta": True,
        },
        "validation_signals": {
            "validation_mode": "cv_and_holdout",
            "cv": {
                "cv_mean": 0.805,
                "cv_std": 0.01,
                "fold_count": 5,
                "variance_level": "medium",
            },
            "holdout": {
                "metric_value": 0.81,
                "split_name": "holdout",
                "sample_count": 1000,
                "class_distribution_shift": "low",
            },
            "generalization_gap": 0.005,
        },
        "error_modes": [],
        "risk_flags": [],
        "active_gates_context": ["required_artifacts_present"],
        "analysis_summary": "Candidate improves baseline and passes min delta.",
        "strictly_no_code_advice": True,
    }
    valid, errors = validate_advisor_critique_packet(packet)
    assert valid is True, errors


def test_validate_iteration_hypothesis_packet_accepts_all_numeric_macro() -> None:
    packet = {
        "packet_type": "iteration_hypothesis_packet",
        "packet_version": "1.0",
        "run_id": "run_x",
        "iteration": 2,
        "hypothesis_id": "h_abcdef12",
        "action": "APPLY",
        "hypothesis": {
            "technique": "missing_indicators",
            "objective": "Improve signal robustness with low-cost features.",
            "target_columns": ["ALL_NUMERIC"],
            "feature_scope": "model_features",
            "params": {"indicator_suffix": "_is_missing"},
            "expected_effect": {
                "target_error_modes": ["fold_instability"],
                "direction": "positive",
            },
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
        "tracker_context": {
            "signature": "hyp_1234567890ab",
            "is_duplicate": False,
            "duplicate_of": None,
        },
        "explanation": "Single hypothesis selected from critique and FE plan.",
        "fallback_if_not_applicable": "NO_OP",
    }
    valid, errors = validate_iteration_hypothesis_packet(packet)
    assert valid is True, errors
    assert normalize_target_columns(["ALL_NUMERIC", "a", "ALL_NUMERIC"]) == ["ALL_NUMERIC", "a"]


def test_validate_iteration_hypothesis_packet_rejects_duplicate_apply() -> None:
    packet = {
        "packet_type": "iteration_hypothesis_packet",
        "packet_version": "1.0",
        "run_id": "run_x",
        "iteration": 2,
        "hypothesis_id": "h_abcdef12",
        "action": "APPLY",
        "hypothesis": {
            "technique": "missing_indicators",
            "objective": "Improve signal robustness with low-cost features.",
            "target_columns": ["ALL_NUMERIC"],
            "feature_scope": "model_features",
            "params": {},
            "expected_effect": {
                "target_error_modes": ["fold_instability"],
                "direction": "positive",
            },
        },
        "application_constraints": {
            "edit_mode": "incremental",
            "max_code_regions_to_change": 3,
            "forbid_replanning": True,
            "forbid_model_family_switch": True,
            "must_keep": ["data_split_logic"],
        },
        "success_criteria": {
            "primary_metric_name": "roc_auc",
            "min_delta": 0.0005,
            "must_pass_active_gates": True,
        },
        "tracker_context": {
            "signature": "hyp_1234567890ab",
            "is_duplicate": True,
            "duplicate_of": "hyp_1234567890ab",
        },
        "explanation": "Duplicate should never be APPLY.",
        "fallback_if_not_applicable": "NO_OP",
    }
    valid, errors = validate_iteration_hypothesis_packet(packet)
    assert valid is False
    assert any("action=NO_OP" in msg or "NO_OP" in msg for msg in errors)


def test_validate_experiment_hypothesis_packet_v2_accepts_macro_targets() -> None:
    packet = {
        "packet_type": "experiment_hypothesis_packet",
        "packet_version": "2.0",
        "run_id": "run_x",
        "round": 2,
        "hypothesis_id": "h_abcdef12",
        "action": "APPLY",
        "technique": "missing_indicators",
        "objective": "Stabilize folds under missingness.",
        "target_columns": ["ALL_NUMERIC"],
        "params": {"suffix": "_is_missing"},
        "success_criteria": {
            "primary_metric_name": "roc_auc",
            "min_delta": 0.0005,
            "must_pass_active_gates": True,
        },
    }
    valid, errors = validate_experiment_hypothesis_packet_v2(packet)
    assert valid is True, errors


def test_validate_experiment_result_packet_v2_rejects_inconsistent_status() -> None:
    packet = {
        "packet_type": "experiment_result_packet",
        "packet_version": "2.0",
        "run_id": "run_x",
        "round": 2,
        "hypothesis_id": "h_abcdef12",
        "status": "REJECTED",
        "primary_metric_name": "roc_auc",
        "baseline_metric": 0.81,
        "candidate_metric": 0.812,
        "delta_abs": 0.002,
        "meets_min_delta": True,
        "gates_passed": True,
    }
    valid, errors = validate_experiment_result_packet_v2(packet)
    assert valid is False
    assert any("REJECTED status cannot have gates_passed=true" in err for err in errors)
