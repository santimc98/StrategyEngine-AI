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
    assert gate_entry.get("passed") is True
    assert set(evidence.get("report_columns_touched") or []) == {"employees", "annual_revenue"}


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
