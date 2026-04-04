import pandas as pd

from src.utils.data_atlas import (
    build_data_atlas,
    build_steward_focus_context,
    summarize_data_atlas,
    summarize_steward_focus_context,
    normalize_evidence_requests,
    build_default_evidence_requests,
    validate_steward_semantics,
)


def test_build_data_atlas_and_summary_basic():
    profile = {
        "columns": ["label", "__split", "pixel0", "pixel1"],
        "type_hints": {"label": "numeric", "__split": "categorical", "pixel0": "numeric", "pixel1": "numeric"},
        "missing_frac": {"label": 0.4, "__split": 0.0, "pixel0": 0.0, "pixel1": 0.0},
        "cardinality": {
            "label": {"unique": 10, "top_values": [{"value": "1", "count": 100}]},
            "__split": {"unique": 2, "top_values": [{"value": "train", "count": 42000}]},
            "pixel0": {"unique": 2, "top_values": [{"value": "0", "count": 68000}]},
            "pixel1": {"unique": 1, "top_values": [{"value": "0", "count": 70000}]},
        },
        "sampling": {"was_sampled": True, "sample_size": 7000, "strategy": "composite"},
        "duplicate_stats": {"row_dup_count": 12, "row_dup_frac": 0.0017},
        "temporal_analysis": {
            "is_time_series": True,
            "detected_datetime_columns": ["__split"],
            "details": [
                {
                    "column": "__split",
                    "granularity_hint": "monthly",
                    "parse_ratio": 1.0,
                    "unique_ratio": 0.02,
                    "duplicate_ratio": 0.98,
                    "max_rows_per_timestamp": 35000,
                    "time_span_days": 365.0,
                }
            ],
        },
    }
    atlas = build_data_atlas(profile, ["label", "__split", "pixel0", "pixel1"])
    assert atlas["coverage"]["total_columns"] == 4
    assert atlas["signals"]["constant_like_count"] >= 1
    assert atlas["duplicate_stats"]["row_dup_count"] == 12
    assert atlas["temporal_overview"]["is_time_series"] is True
    summary = summarize_data_atlas(atlas)
    assert "DATA_ATLAS_SUMMARY:" in summary
    assert "target_name_hints" in summary
    assert "temporal_overview" in summary
    assert "duplicate_rows" in summary


def test_normalize_and_default_evidence_requests():
    header = ["label", "__split", "pixel0"]
    reqs = normalize_evidence_requests(
        [
            {"kind": "missingness", "column": "label"},
            {"kind": "uniques", "column": "__split", "max_unique": 25},
            {"kind": "unknown", "column": "label"},
            {"kind": "uniques", "column": "bad_col"},
        ],
        header,
    )
    assert len(reqs) == 2
    defaults = build_default_evidence_requests("label", ["__split"], ["id"], header)
    assert any(item["column"] == "label" for item in defaults)
    assert any(item["column"] == "__split" for item in defaults)


def test_validate_steward_semantics_gate():
    ok = validate_steward_semantics(
        dataset_semantics={
            "primary_target": "label",
            "target_status": "confirmed",
            "split_candidates": ["__split"],
        },
        dataset_training_mask={"training_rows_rule": "rows where label is not missing", "scoring_rows_rule_primary": "all rows"},
        header_cols=["label", "__split", "pixel0"],
        target_missingness={"null_frac_exact": 0.4},
        column_sets={"explicit_columns": ["label", "__split"], "sets": []},
    )
    assert ok["ready"] is True

    bad = validate_steward_semantics(
        dataset_semantics={"primary_target": "unknown_col"},
        dataset_training_mask={"training_rows_rule": ""},
        header_cols=["label", "__split"],
        target_missingness={"null_frac_exact": 0.4},
        column_sets={},
    )
    assert bad["ready"] is False
    assert "missing_training_rows_rule" in bad["reasons"]


def test_validate_steward_semantics_blocks_invalid_target_status():
    result = validate_steward_semantics(
        dataset_semantics={
            "primary_target": "label",
            "target_status": "invalid",
            "target_status_reason": "label is 97% missing and behaves like a post-decision status",
            "recommended_primary_target": "renewal_score",
        },
        dataset_training_mask={
            "training_rows_rule": "rows where label is not missing",
            "scoring_rows_rule_primary": "all rows",
        },
        header_cols=["label", "renewal_score", "__split"],
        target_missingness={"null_frac_exact": 0.97},
        column_sets={"explicit_columns": ["label", "__split"], "sets": []},
    )
    assert result["ready"] is False
    assert "primary_target_invalid" in result["reasons"]
    assert result["target_status"] == "invalid"
    assert result["recommended_primary_target"] == "renewal_score"


def test_build_steward_focus_context_surfaces_label_maturity_and_identifier_grain():
    df = pd.DataFrame(
        {
            "account_id": ["A", "A", "A", "B", "B", "B"],
            "snapshot_month_end": [
                "2025-08-31",
                "2025-09-30",
                "2025-10-31",
                "2025-08-31",
                "2025-09-30",
                "2025-10-31",
            ],
            "churn_60d": [0, 1, None, 0, 0, None],
        }
    )

    context = build_steward_focus_context(
        df,
        primary_target="churn_60d",
        split_candidates=["snapshot_month_end"],
        id_candidates=["account_id"],
    )

    maturity = context["temporal_label_maturity"][0]
    assert maturity["split_column"] == "snapshot_month_end"
    assert maturity["latest_fully_labeled_bucket"] == "2025-09"
    assert maturity["earliest_bucket_with_unlabeled_rows"] == "2025-10"

    identifier = context["identifier_structure"][0]
    assert identifier["id_column"] == "account_id"
    assert identifier["duplicated_entity_rows"] == 6
    assert identifier["pair_diagnostics"][0]["duplicated_pair_rows"] == 0

    summary = summarize_steward_focus_context(context)
    assert "STEWARD_FOCUS_CONTEXT:" in summary
    assert "latest_fully_labeled=2025-09" in summary
    assert "earliest_unlabeled=2025-10" in summary
