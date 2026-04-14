import pandas as pd
from pathlib import Path

from src.agents.cleaning_reviewer import _evaluate_gates_deterministic, normalize_gate_name


def test_id_integrity_excludes_target_columns_from_identifier_check():
    df = pd.DataFrame(
        {
            "id": ["0001", "0002", "0003", "0004"],
            "identity_hate": [0.0, 1.0, 0.0, 0.0],
            "__split": ["train", "train", "train", "train"],
        }
    )
    gates = [
        {
            "name": "id_integrity",
            "severity": "HARD",
            "params": {"detect_scientific_notation": True, "min_samples": 1},
        }
    ]

    result = _evaluate_gates_deterministic(
        gates=gates,
        required_columns=[],
        cleaned_header=list(df.columns),
        cleaned_csv_path="data/cleaned_data.csv",
        sample_str=df.astype(str),
        sample_infer=df,
        manifest={},
        raw_sample=None,
        column_roles={
            "id": ["id"],
            "targets": ["identity_hate"],
            "split_indicator": ["__split"],
        },
        allowed_feature_sets={},
    )

    assert result["status"] == "APPROVED"
    assert "id_integrity" not in result.get("failed_checks", [])
    gate_entry = next(
        gr
        for gr in result.get("gate_results", [])
        if normalize_gate_name(gr.get("name", "")) == "id_integrity"
    )
    evidence = gate_entry.get("evidence") or {}
    assert "id" in (evidence.get("candidate_columns") or [])
    assert "identity_hate" not in (evidence.get("candidate_columns") or [])


def test_row_count_sanity_skips_when_drop_matches_label_null_listwise_pattern():
    df = pd.DataFrame(
        {
            "id": ["a", "b"],
            "target_a": [0.0, 1.0],
            "target_b": [1.0, 0.0],
            "__split": ["train", "train"],
        }
    )
    gates = [
        {
            "name": "row_count_sanity",
            "severity": "SOFT",
            "params": {"max_drop_pct": 5.0},
        }
    ]
    manifest = {
        "row_counts": {
            "initial": 1000,
            "final": 500,
            "dropped": 500,
            "dropped_reason": "null_label_removal (listwise deletion on toxicity labels)",
        },
        "gate_results": {
            "null_label_removal": {
                "status": "PASSED",
                "rows_removed": 500,
            }
        },
        "null_stats": {
            "before": {"id": 0, "target_a": 500, "target_b": 500, "__split": 0},
            "after": {"id": 0, "target_a": 0, "target_b": 0, "__split": 0},
        },
    }

    result = _evaluate_gates_deterministic(
        gates=gates,
        required_columns=[],
        cleaned_header=list(df.columns),
        cleaned_csv_path="data/cleaned_data.csv",
        sample_str=df.astype(str),
        sample_infer=df,
        manifest=manifest,
        raw_sample=None,
        column_roles={
            "id": ["id"],
            "targets": ["target_a", "target_b"],
            "split_indicator": ["__split"],
        },
        allowed_feature_sets={},
    )

    assert result["status"] == "APPROVED"
    assert "row_count_sanity" not in result.get("failed_checks", [])
    gate_entry = next(
        gr
        for gr in result.get("gate_results", [])
        if normalize_gate_name(gr.get("name", "")) == "row_count_sanity"
    )
    evidence = gate_entry.get("evidence") or {}
    assert evidence.get("applies_if") is False
    assert evidence.get("skip_reason") == "drop_explained_by_label_null_listwise_removal"


def test_feature_coverage_sanity_flags_missing_features_against_inventory(monkeypatch):
    monkeypatch.setattr(
        "src.agents.cleaning_reviewer._load_column_inventory_names",
        lambda path="data/column_inventory.json": [
            "id",
            "target",
            "is_train",
            "age",
            "bp",
            "cholesterol",
            "max_hr",
        ],
    )

    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "target": [1, 0, 1, 0],
            "is_train": [1, 1, 0, 0],
        }
    )
    gates = [
        {
            "name": "feature_coverage_sanity",
            "severity": "SOFT",
            "params": {"min_feature_count": 3, "check_against": "data_atlas"},
        }
    ]

    result = _evaluate_gates_deterministic(
        gates=gates,
        required_columns=["id", "target", "is_train"],
        cleaned_header=list(df.columns),
        cleaned_csv_path="data/cleaned_data.csv",
        sample_str=df.astype(str),
        sample_infer=df,
        manifest={},
        raw_sample=None,
        column_roles={
            "identifiers": ["id"],
            "outcome": ["target"],
            "split_columns": ["is_train"],
        },
        allowed_feature_sets={"model_features": ["age", "bp", "cholesterol", "max_hr"]},
        dataset_profile={"dataset_semantics": {"primary_target": "target", "split_candidates": ["is_train"]}},
    )

    assert result["status"] == "APPROVE_WITH_WARNINGS"
    assert "feature_coverage_sanity" in result.get("failed_checks", [])
    gate_entry = next(
        gr
        for gr in result.get("gate_results", [])
        if normalize_gate_name(gr.get("name", "")) == "feature_coverage_sanity"
    )
    evidence = gate_entry.get("evidence") or {}
    assert evidence.get("cleaned_feature_count") == 0
    assert evidence.get("source_feature_count", 0) >= 3


def test_outlier_policy_applied_accepts_dict_shaped_report_columns():
    df = pd.DataFrame(
        {
            "employees": [10, 20, 30],
            "annual_revenue": [1000, 2000, 3000],
        }
    )
    gates = [
        {
            "name": "outlier_policy_applied",
            "severity": "HARD",
            "params": {"strict": True},
        }
    ]
    manifest = {
        "outlier_treatment": {
            "policy_applied": True,
        }
    }
    outlier_policy = {
        "enabled": True,
        "apply_stage": "data_engineer",
        "target_columns": ["employees", "annual_revenue"],
        "strict": True,
    }
    outlier_report = {
        "enabled": True,
        "columns": {
            "employees": {"clipped_count": 2, "action": "clipped_and_flagged"},
            "annual_revenue": {"clipped_count": 1, "action": "clipped_and_flagged"},
        },
    }

    result = _evaluate_gates_deterministic(
        gates=gates,
        required_columns=[],
        cleaned_header=list(df.columns),
        cleaned_csv_path="data/cleaned_data.csv",
        sample_str=df.astype(str),
        sample_infer=df,
        manifest=manifest,
        raw_sample=None,
        column_roles={},
        allowed_feature_sets={},
        outlier_policy=outlier_policy,
        outlier_report=outlier_report,
        outlier_report_path="data/outlier_treatment_report.json",
    )

    assert result["status"] == "APPROVED"
    gate_entry = next(
        gr
        for gr in result.get("gate_results", [])
        if normalize_gate_name(gr.get("name", "")) == "outlier_policy_applied"
    )
    evidence = gate_entry.get("evidence") or {}
    assert gate_entry.get("passed") is None
    assert evidence.get("semantic_review_required") is True
    assert set(evidence.get("policy_target_columns") or []) == {"employees", "annual_revenue"}


def test_outlier_policy_applied_accepts_actions_report():
    """Regression: run 8ef7d68e aborted because the extractor rejected the
    `actions: [{column, method, lower, upper, ...}]` schema the DE produced.
    This is a reasonable shape and must be recognized."""
    df = pd.DataFrame(
        {
            "employees": [10, 20, 30],
            "annual_revenue": [1000, 2000, 3000],
        }
    )
    gates = [
        {
            "name": "outlier_policy_applied",
            "severity": "HARD",
            "params": {"strict": True},
        }
    ]
    manifest = {"outlier_treatment": {"policy_applied": True}}
    outlier_policy = {
        "enabled": True,
        "apply_stage": "data_engineer",
        "target_columns": ["employees", "annual_revenue"],
        "strict": True,
    }
    outlier_report = {
        "enabled": True,
        "actions": [
            {
                "column": "employees",
                "method": "cap",
                "lower": 0.0,
                "upper": 10000.0,
                "before_min": 7.0,
                "before_max": 2839.0,
            },
            {
                "column": "annual_revenue",
                "method": "cap",
                "lower": 0.0,
                "upper": 1000000.0,
                "before_min": 10000.0,
                "before_max": 1876092.0,
            },
        ],
    }

    result = _evaluate_gates_deterministic(
        gates=gates,
        required_columns=[],
        cleaned_header=list(df.columns),
        cleaned_csv_path="data/cleaned_data.csv",
        sample_str=df.astype(str),
        sample_infer=df,
        manifest=manifest,
        raw_sample=None,
        column_roles={},
        allowed_feature_sets={},
        outlier_policy=outlier_policy,
        outlier_report=outlier_report,
        outlier_report_path="data/outlier_treatment_report.json",
    )

    assert result["status"] == "APPROVED"
    gate_entry = next(
        gr
        for gr in result.get("gate_results", [])
        if normalize_gate_name(gr.get("name", "")) == "outlier_policy_applied"
    )
    evidence = gate_entry.get("evidence") or {}
    assert gate_entry.get("passed") is None
    assert evidence.get("semantic_review_required") is True
    assert set(evidence.get("policy_target_columns") or []) == {"employees", "annual_revenue"}


def test_outlier_policy_applied_accepts_flat_top_level_column_report():
    df = pd.DataFrame(
        {
            "employees": [10, 20, 30],
            "annual_revenue": [1000, 2000, 3000],
        }
    )
    gates = [
        {
            "name": "outlier_policy_applied",
            "severity": "HARD",
            "params": {"strict": True},
        }
    ]
    manifest = {"outlier_treatment": {"policy_applied": True}}
    outlier_policy = {
        "enabled": True,
        "apply_stage": "data_engineer",
        "target_columns": ["employees", "annual_revenue"],
        "strict": True,
    }
    outlier_report = {
        "employees": {"threshold_99pct": 5000, "capped_count": 2},
        "annual_revenue": {"threshold_99pct": 9000, "capped_count": 1},
    }

    result = _evaluate_gates_deterministic(
        gates=gates,
        required_columns=[],
        cleaned_header=list(df.columns),
        cleaned_csv_path="data/cleaned_data.csv",
        sample_str=df.astype(str),
        sample_infer=df,
        manifest=manifest,
        raw_sample=None,
        column_roles={},
        allowed_feature_sets={},
        outlier_policy=outlier_policy,
        outlier_report=outlier_report,
        outlier_report_path="data/outlier_treatment_report.json",
    )

    gate_entry = next(
        gr
        for gr in result.get("gate_results", [])
        if normalize_gate_name(gr.get("name", "")) == "outlier_policy_applied"
    )
    evidence = gate_entry.get("evidence") or {}
    assert gate_entry.get("passed") is None
    assert evidence.get("semantic_review_required") is True
    assert set(evidence.get("policy_target_columns") or []) == {"employees", "annual_revenue"}


def test_outlier_policy_applied_accepts_columns_analyzed_and_decisions_report():
    df = pd.DataFrame(
        {
            "employees": [10, 20, 30],
            "annual_revenue": [1000, 2000, 3000],
        }
    )
    gates = [
        {
            "name": "outlier_policy_applied",
            "severity": "HARD",
            "params": {"strict": True},
        }
    ]
    manifest = {"outlier_treatment": {"policy_applied": True}}
    outlier_policy = {
        "enabled": True,
        "apply_stage": "data_engineer",
        "target_columns": ["employees", "annual_revenue"],
        "strict": True,
    }
    outlier_report = {
        "policy": {"method": "winsorize"},
        "columns_analyzed": [
            {"column": "employees", "pct99": 120},
            {"column": "annual_revenue", "pct99": 9500},
        ],
        "decisions": [
            {"column": "employees", "action": "capped_and_flagged"},
            {"column": "annual_revenue", "action": "capped_and_flagged"},
        ],
    }

    result = _evaluate_gates_deterministic(
        gates=gates,
        required_columns=[],
        cleaned_header=list(df.columns),
        cleaned_csv_path="data/cleaned_data.csv",
        sample_str=df.astype(str),
        sample_infer=df,
        manifest=manifest,
        raw_sample=None,
        column_roles={},
        allowed_feature_sets={},
        outlier_policy=outlier_policy,
        outlier_report=outlier_report,
        outlier_report_path="data/outlier_treatment_report.json",
    )

    assert result["status"] == "APPROVED"
    gate_entry = next(
        gr
        for gr in result.get("gate_results", [])
        if normalize_gate_name(gr.get("name", "")) == "outlier_policy_applied"
    )
    evidence = gate_entry.get("evidence") or {}
    assert gate_entry.get("passed") is None
    assert evidence.get("semantic_review_required") is True
    assert set(evidence.get("policy_target_columns") or []) == {"employees", "annual_revenue"}


def test_outlier_policy_applied_accepts_targets_dict_report():
    df = pd.DataFrame(
        {
            "employees": [10, 20, 30],
            "annual_revenue": [1000, 2000, 3000],
        }
    )
    gates = [
        {
            "name": "outlier_policy_applied",
            "severity": "HARD",
            "params": {"strict": True},
        }
    ]
    manifest = {"outlier_treatment": {"policy_applied": True}}
    outlier_policy = {
        "enabled": True,
        "apply_stage": "data_engineer",
        "target_columns": ["employees", "annual_revenue"],
        "strict": True,
    }
    outlier_report = {
        "enabled": True,
        "targets": {
            "employees": {"threshold_99pct": 5000, "capped_count": 2},
            "annual_revenue": {"threshold_99pct": 9000, "capped_count": 1},
        },
        "actions": ["clip_and_flag"],
    }

    result = _evaluate_gates_deterministic(
        gates=gates,
        required_columns=[],
        cleaned_header=list(df.columns),
        cleaned_csv_path="data/cleaned_data.csv",
        sample_str=df.astype(str),
        sample_infer=df,
        manifest=manifest,
        raw_sample=None,
        column_roles={},
        allowed_feature_sets={},
        outlier_policy=outlier_policy,
        outlier_report=outlier_report,
        outlier_report_path="data/outlier_treatment_report.json",
    )

    gate_entry = next(
        gr
        for gr in result.get("gate_results", [])
        if normalize_gate_name(gr.get("name", "")) == "outlier_policy_applied"
    )
    evidence = gate_entry.get("evidence") or {}
    assert gate_entry.get("passed") is None
    assert evidence.get("semantic_review_required") is True
    assert set(evidence.get("policy_target_columns") or []) == {"employees", "annual_revenue"}


def test_boolean_normalization_passes_for_nullable_integer_boolean_columns(tmp_path: Path):
    csv_path = tmp_path / "cleaned.csv"
    cleaned = pd.DataFrame(
        {
            "marketing_consent": [1, 0, None, 1],
            "demo_requested": [0, 1, 1, None],
            "mql_flag": [1, 0, 1, 0],
            "employees": [10, 20, 30, 40],
        }
    )
    cleaned.to_csv(csv_path, index=False)
    raw_sample = pd.DataFrame(
        {
            "marketing_consent": ["yes", "no", None, "si"],
            "demo_requested": ["0", "1", "yes", None],
            "mql_flag": ["true", "false", "1", "0"],
        }
    )
    gates = [
        {
            "name": "boolean_normalization",
            "severity": "HARD",
            "params": {"target_values": [0, 1, None]},
        }
    ]

    result = _evaluate_gates_deterministic(
        gates=gates,
        required_columns=[],
        cleaned_header=list(cleaned.columns),
        cleaned_csv_path=str(csv_path),
        sample_str=cleaned.astype("string"),
        sample_infer=cleaned,
        manifest={},
        raw_sample=raw_sample,
        column_roles={"pre_decision": ["marketing_consent", "demo_requested", "mql_flag"]},
        allowed_feature_sets={},
        column_dtype_targets={
            "marketing_consent": {"target_dtype": "int64", "nullable": True, "role": "pre_decision"},
            "demo_requested": {"target_dtype": "int64", "nullable": True, "role": "pre_decision"},
            "mql_flag": {"target_dtype": "int64", "nullable": True, "role": "pre_decision"},
            "employees": {"target_dtype": "float64", "nullable": True, "role": "pre_decision"},
        },
    )

    assert result["status"] == "APPROVED"
    gate_entry = next(
        gr
        for gr in result.get("gate_results", [])
        if normalize_gate_name(gr.get("name", "")) == "boolean_normalization"
    )
    evidence = gate_entry.get("evidence") or {}
    assert gate_entry.get("passed") is True
    assert set(evidence.get("columns_checked") or []) == {"marketing_consent", "demo_requested", "mql_flag"}


def test_boolean_normalization_passes_when_contract_requests_int8_but_csv_roundtrip_is_int64(tmp_path: Path):
    csv_path = tmp_path / "cleaned.csv"
    cleaned = pd.DataFrame(
        {
            "webinar_attended": [1, 0, 1, None],
            "demo_requested": [0, 1, 1, 0],
        }
    )
    cleaned.to_csv(csv_path, index=False)
    gates = [
        {
            "name": "boolean_columns_normalized",
            "severity": "HARD",
            "params": {
                "columns": ["webinar_attended", "demo_requested"],
                "expected_dtype": "int8",
                "allowed_values": [0, 1],
            },
        }
    ]

    result = _evaluate_gates_deterministic(
        gates=gates,
        required_columns=[],
        cleaned_header=list(cleaned.columns),
        cleaned_csv_path=str(csv_path),
        sample_str=cleaned.astype("string"),
        sample_infer=cleaned,
        manifest={},
        raw_sample=pd.DataFrame(
            {
                "webinar_attended": ["yes", "no", "1", None],
                "demo_requested": ["0", "1", "yes", "0"],
            }
        ),
        column_roles={"pre_decision": ["webinar_attended", "demo_requested"]},
        allowed_feature_sets={},
        column_dtype_targets={
            "webinar_attended": {"target_dtype": "int8", "nullable": True, "role": "pre_decision"},
            "demo_requested": {"target_dtype": "int8", "nullable": True, "role": "pre_decision"},
        },
    )

    gate_entry = next(
        gr
        for gr in result.get("gate_results", [])
        if normalize_gate_name(gr.get("name", "")) == "boolean_normalization"
    )
    evidence = gate_entry.get("evidence") or {}
    assert gate_entry.get("passed") is True
    assert evidence.get("expected_dtype") == "int8"
    assert evidence.get("csv_roundtrip_dtype_width_not_enforced") is True


def test_boolean_normalization_fails_when_cleaned_values_still_contain_text_tokens(tmp_path: Path):
    csv_path = tmp_path / "cleaned.csv"
    cleaned = pd.DataFrame(
        {
            "marketing_consent": ["yes", "0", None],
            "demo_requested": ["1", "no", "1"],
        }
    )
    cleaned.to_csv(csv_path, index=False)
    raw_sample = pd.DataFrame(
        {
            "marketing_consent": ["yes", "no", None],
            "demo_requested": ["1", "0", "yes"],
        }
    )
    gates = [
        {
            "name": "boolean_normalization",
            "severity": "HARD",
            "params": {"target_values": [0, 1, None]},
        }
    ]

    result = _evaluate_gates_deterministic(
        gates=gates,
        required_columns=[],
        cleaned_header=list(cleaned.columns),
        cleaned_csv_path=str(csv_path),
        sample_str=cleaned.astype("string"),
        sample_infer=cleaned,
        manifest={},
        raw_sample=raw_sample,
        column_roles={"pre_decision": ["marketing_consent", "demo_requested"]},
        allowed_feature_sets={},
        column_dtype_targets={
            "marketing_consent": {"target_dtype": "int64", "nullable": True, "role": "pre_decision"},
            "demo_requested": {"target_dtype": "int64", "nullable": True, "role": "pre_decision"},
        },
    )

    assert result["status"] == "REJECTED"
    assert "boolean_normalization" in result.get("failed_checks", [])
    gate_entry = next(
        gr
        for gr in result.get("gate_results", [])
        if normalize_gate_name(gr.get("name", "")) == "boolean_normalization"
    )
    evidence = gate_entry.get("evidence") or {}
    assert gate_entry.get("passed") is False
    assert "marketing_consent" in (evidence.get("invalid_values") or {})


def test_temporal_training_mask_allows_future_rows_outside_training_subset(tmp_path: Path):
    csv_path = tmp_path / "cleaned.csv"
    cleaned = pd.DataFrame(
        {
            "created_at": ["2024-12-15", "2024-12-31", "2025-01-05", "2025-02-01"],
            "won_90d": [1, 0, None, None],
            "lead_id": ["a", "b", "c", "d"],
        }
    )
    cleaned.to_csv(csv_path, index=False)
    gates = [
        {
            "name": "enforce_temporal_training_mask",
            "severity": "HARD",
            "params": {
                "column": "created_at",
                "training_cutoff": "2024-12-31",
                "rule": "Training rows must satisfy created_at <= '2024-12-31'",
            },
        },
        {
            "name": "target_not_null_in_training",
            "severity": "HARD",
            "params": {"column": "won_90d", "applies_to": "training_rows"},
        },
    ]

    result = _evaluate_gates_deterministic(
        gates=gates,
        required_columns=[],
        cleaned_header=list(cleaned.columns),
        cleaned_csv_path=str(csv_path),
        sample_str=cleaned.astype("string"),
        sample_infer=cleaned,
        manifest={},
        raw_sample=None,
        column_roles={"time_columns": ["created_at"], "outcome": ["won_90d"], "identifiers": ["lead_id"]},
        allowed_feature_sets={},
    )

    temporal_gate = next(
        gr
        for gr in result.get("gate_results", [])
        if normalize_gate_name(gr.get("name", "")) == "enforce_temporal_training_mask"
    )
    target_gate = next(
        gr
        for gr in result.get("gate_results", [])
        if normalize_gate_name(gr.get("name", "")) == "target_not_null_in_training"
    )
    temporal_evidence = temporal_gate.get("evidence") or {}
    target_evidence = target_gate.get("evidence") or {}
    assert result["status"] == "APPROVED"
    assert temporal_gate.get("passed") is True
    assert target_gate.get("passed") is True
    assert temporal_evidence.get("future_rows_present") == 2
    assert temporal_evidence.get("training_rows_count") == 2
    assert target_evidence.get("null_training_rows") == 0


def test_temporal_training_mask_fails_only_when_explicit_training_rows_cross_cutoff(tmp_path: Path):
    csv_path = tmp_path / "cleaned.csv"
    cleaned = pd.DataFrame(
        {
            "created_at": ["2024-12-20", "2025-01-05", "2025-01-10"],
            "__split": ["train", "train", "score"],
            "won_90d": [1, 0, None],
        }
    )
    cleaned.to_csv(csv_path, index=False)
    gates = [
        {
            "name": "enforce_temporal_training_mask",
            "severity": "HARD",
            "params": {
                "column": "created_at",
                "training_cutoff": "2024-12-31",
                "rule": "Training rows must satisfy created_at <= '2024-12-31'",
            },
        }
    ]

    result = _evaluate_gates_deterministic(
        gates=gates,
        required_columns=[],
        cleaned_header=list(cleaned.columns),
        cleaned_csv_path=str(csv_path),
        sample_str=cleaned.astype("string"),
        sample_infer=cleaned,
        manifest={},
        raw_sample=None,
        column_roles={"time_columns": ["created_at"], "split_indicator": ["__split"], "outcome": ["won_90d"]},
        allowed_feature_sets={},
    )

    gate_entry = next(
        gr
        for gr in result.get("gate_results", [])
        if normalize_gate_name(gr.get("name", "")) == "enforce_temporal_training_mask"
    )
    evidence = gate_entry.get("evidence") or {}
    assert result["status"] == "REJECTED"
    assert gate_entry.get("passed") is False
    assert evidence.get("training_mask_source") == "indicator:__split"
    assert evidence.get("training_rows_after_cutoff") == 1


def test_cleaning_reviewer_deterministically_passes_documented_first_attempt_style_cleaning(tmp_path: Path):
    csv_path = tmp_path / "cleaned.csv"
    cleaned = pd.DataFrame(
        {
            "account_id": ["ACC_001", "ACC_001", "ACC_002", "ACC_003"],
            "snapshot_month_end": ["2025-07-31", "2025-08-31", "2025-08-31", "2025-09-30"],
            "cohort_split": ["train", "temporal_holdout", "train", "score"],
            "churn_60d": [0.0, 1.0, 0.0, None],
            "arr_current": ["EUR 279,981", "$278,656", "280,147", "277,572"],
            "nps_last_observed": [72.0, 72.0, 55.0, 61.0],
        }
    )
    cleaned.to_csv(csv_path, index=False)

    manifest = {
        "cleaning_gates_status": {
            "arr_current_numeric_conversion_verified": "WARNING_dropped_from_features",
        },
        "conversions": [
            {
                "step": "arr_current_parsing",
                "description": "Parsed arr_current and dropped from model features when unsafe.",
                "dropped_from_features": True,
            }
        ],
        "imputation": {
            "nps_last_observed": {
                "strategy": "forward_fill_within_account_id",
                "missing_before": 2,
                "missing_after": 0,
            }
        },
    }
    gates = [
        {
            "name": "training_cohort_filter_enforced",
            "severity": "HARD",
            "params": {
                "required_condition": "churn_60d IS NOT NULL AND snapshot_month_end <= '2025-08-31'",
                "split_label": "train",
            },
        },
        {
            "name": "scoring_cohort_filter_enforced",
            "severity": "HARD",
            "params": {
                "required_condition": "churn_60d IS NULL",
                "split_label": "score",
            },
        },
        {
            "name": "identifier_columns_excluded_from_features",
            "severity": "HARD",
            "params": {
                "forbidden_as_features": ["account_id", "snapshot_month_end", "csm_owner"],
            },
        },
        {
            "name": "arr_current_numeric_conversion_verified",
            "severity": "HARD",
            "params": {"column": "arr_current"},
        },
        {
            "name": "nps_forward_fill_temporal_integrity",
            "severity": "SOFT",
            "params": {
                "column": "nps_last_observed",
                "group_key": "account_id",
                "sort_key": "snapshot_month_end",
            },
        },
    ]
    cleaning_code = (
        "df = df.sort_values(['account_id', 'snapshot_month_end']).reset_index(drop=True)\n"
        "df['nps_last_observed'] = df.groupby('account_id')['nps_last_observed'].ffill()\n"
    )

    result = _evaluate_gates_deterministic(
        gates=gates,
        required_columns=[],
        cleaned_header=list(cleaned.columns),
        cleaned_csv_path=str(csv_path),
        sample_str=cleaned.astype("string"),
        sample_infer=cleaned,
        manifest=manifest,
        raw_sample=None,
        column_roles={
            "identifier": ["account_id"],
            "split": ["cohort_split"],
            "outcome": ["churn_60d"],
        },
        allowed_feature_sets={},
        cleaning_code=cleaning_code,
    )

    assert result["status"] == "APPROVED"
    gate_map = {
        normalize_gate_name(entry.get("name", "")): entry
        for entry in result.get("gate_results", [])
        if isinstance(entry, dict)
    }
    assert gate_map["training_cohort_filter_enforced"]["passed"] is True
    assert gate_map["scoring_cohort_filter_enforced"]["passed"] is True
    assert gate_map["identifier_columns_excluded_from_features"]["passed"] is True
    assert gate_map["arr_current_numeric_conversion_verified"]["passed"] is True
    assert gate_map["nps_forward_fill_temporal_integrity"]["passed"] is True


def test_arr_current_numeric_conversion_fails_when_object_currency_strings_are_undocumented(tmp_path: Path):
    csv_path = tmp_path / "cleaned.csv"
    cleaned = pd.DataFrame(
        {
            "arr_current": ["EUR 279,981", "$278,656", "280,147"],
        }
    )
    cleaned.to_csv(csv_path, index=False)

    result = _evaluate_gates_deterministic(
        gates=[
            {
                "name": "arr_current_numeric_conversion_verified",
                "severity": "HARD",
                "params": {"column": "arr_current"},
            }
        ],
        required_columns=[],
        cleaned_header=list(cleaned.columns),
        cleaned_csv_path=str(csv_path),
        sample_str=cleaned.astype("string"),
        sample_infer=cleaned,
        manifest={},
        raw_sample=None,
        column_roles={},
        allowed_feature_sets={},
    )

    assert result["status"] == "REJECTED"
    gate_entry = next(
        gr
        for gr in result.get("gate_results", [])
        if normalize_gate_name(gr.get("name", "")) == "arr_current_numeric_conversion_verified"
    )
    assert gate_entry.get("passed") is False


def test_outlier_policy_applied_accepts_treatments_list_report():
    df = pd.DataFrame(
        {
            "employees": [10, 20, 30],
            "annual_revenue": [1000, 2000, 3000],
        }
    )
    gates = [
        {
            "name": "outlier_policy_applied",
            "severity": "HARD",
            "params": {"strict": True},
        }
    ]
    outlier_policy = {
        "enabled": True,
        "apply_stage": "data_engineer",
        "target_columns": ["employees", "annual_revenue"],
        "strict": True,
    }
    outlier_report = {
        "enabled": True,
        "treatments": [
            {"column": "employees", "method": "winsorize_1pct_99pct", "rows_capped": 2},
            {"column": "annual_revenue", "method": "winsorize_1pct_99pct", "rows_capped": 1},
        ],
    }

    result = _evaluate_gates_deterministic(
        gates=gates,
        required_columns=[],
        cleaned_header=list(df.columns),
        cleaned_csv_path="data/cleaned_data.csv",
        sample_str=df.astype(str),
        sample_infer=df,
        manifest={},
        raw_sample=None,
        column_roles={},
        allowed_feature_sets={},
        outlier_policy=outlier_policy,
        outlier_report=outlier_report,
        outlier_report_path="data/outlier_treatment_report.json",
    )

    gate_entry = next(
        gr
        for gr in result.get("gate_results", [])
        if normalize_gate_name(gr.get("name", "")) == "outlier_policy_applied"
    )
    evidence = gate_entry.get("evidence") or {}
    assert result["status"] == "APPROVED"
    assert gate_entry.get("passed") is None
    assert evidence.get("semantic_review_required") is True
    assert set(evidence.get("policy_target_columns") or []) == {"employees", "annual_revenue"}


def test_cleaning_reviewer_deterministically_evaluates_temporal_contract_gates(tmp_path: Path):
    csv_path = tmp_path / "account_snapshots_ml_ready.csv"
    cleaned = pd.DataFrame(
        {
            "account_id": ["A1", "A2", "A3", "A4"],
            "snapshot_month_end": ["2025-08-31", "2025-09-30", "2025-10-31", "2025-11-30"],
            "churn_60d": [0.0, 1.0, 0.0, None],
            "arr_current": [1000.0, 2000.0, 3000.0, 4000.0],
            "seat_utilization_30d": [0.7, 0.8, 0.9, 0.6],
        }
    )
    cleaned.to_csv(csv_path, index=False)
    gates = [
        {
            "name": "snapshot_month_end_parseable",
            "severity": "HARD",
            "params": {"column": "snapshot_month_end", "required_parse_ratio": 1.0},
        },
        {
            "name": "training_rows_churn_label_not_null",
            "severity": "HARD",
            "params": {
                "partition": "training",
                "filter": "snapshot_month_end <= '2025-09-30'",
                "column": "churn_60d",
                "condition": "IS NOT NULL",
            },
        },
        {
            "name": "exact_duplicates_removed_from_training",
            "severity": "HARD",
            "params": {
                "partition": "training",
                "filter": "snapshot_month_end <= '2025-09-30'",
                "condition": "is_exact_duplicate = FALSE",
            },
        },
        {
            "name": "leakage_columns_excluded_from_feature_matrix",
            "severity": "HARD",
            "params": {"forbidden_columns": ["final_account_status", "cancelled_at", "health_score"]},
        },
        {
            "name": "training_partition_temporal_ceiling",
            "severity": "HARD",
            "params": {
                "column": "snapshot_month_end",
                "max_allowed_date": "2025-09-30",
                "partition": "training",
            },
        },
        {
            "name": "arr_current_numeric_parseable",
            "severity": "SOFT",
            "params": {"column": "arr_current", "required_parse_ratio": 1.0},
        },
    ]

    result = _evaluate_gates_deterministic(
        gates=gates,
        required_columns=[],
        cleaned_header=list(cleaned.columns),
        cleaned_csv_path=str(csv_path),
        sample_str=cleaned.astype("string"),
        sample_infer=cleaned,
        manifest={},
        raw_sample=None,
        column_roles={"identifier": ["account_id"], "outcome": ["churn_60d"]},
        allowed_feature_sets={},
    )

    assert result["status"] == "APPROVED"
    gate_map = {
        normalize_gate_name(entry.get("name", "")): entry
        for entry in result.get("gate_results", [])
        if isinstance(entry, dict)
    }
    for gate_name in [
        "snapshot_month_end_parseable",
        "training_rows_churn_label_not_null",
        "exact_duplicates_removed_from_training",
        "leakage_columns_excluded_from_feature_matrix",
        "training_partition_temporal_ceiling",
        "arr_current_numeric_parseable",
    ]:
        assert gate_map[gate_name]["passed"] is True


def test_training_label_gate_fails_only_for_nulls_inside_filtered_partition(tmp_path: Path):
    csv_path = tmp_path / "account_snapshots_ml_ready.csv"
    cleaned = pd.DataFrame(
        {
            "snapshot_month_end": ["2025-08-31", "2025-09-30", "2025-11-30"],
            "churn_60d": [0.0, None, None],
        }
    )
    cleaned.to_csv(csv_path, index=False)

    result = _evaluate_gates_deterministic(
        gates=[
            {
                "name": "training_rows_churn_label_not_null",
                "severity": "HARD",
                "params": {
                    "partition": "training",
                    "filter": "snapshot_month_end <= '2025-09-30'",
                    "column": "churn_60d",
                },
            }
        ],
        required_columns=[],
        cleaned_header=list(cleaned.columns),
        cleaned_csv_path=str(csv_path),
        sample_str=cleaned.astype("string"),
        sample_infer=cleaned,
        manifest={},
        raw_sample=None,
        column_roles={},
        allowed_feature_sets={},
    )

    gate_entry = result["gate_results"][0]
    evidence = gate_entry.get("evidence") or {}
    assert result["status"] == "REJECTED"
    assert gate_entry.get("passed") is False
    assert evidence.get("null_rows_in_partition") == 1
