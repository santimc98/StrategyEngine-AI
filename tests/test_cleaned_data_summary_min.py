import pandas as pd

from src.graph.graph import _build_cleaned_data_summary_min


def test_cleaned_data_summary_min_flags_missing_required_and_role_dtype_warnings():
    df = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "vendor_id": [1, 2, 1, 2],
            "pickup_datetime": [
                "2026-01-01 10:00:00",
                "2026-01-01 10:05:00",
                "2026-01-01 10:10:00",
                "2026-01-01 10:15:00",
            ],
            "trip_duration": [300.0, 450.0, None, 700.0],
            "__split": ["train", "train", "test", "test"],
        }
    )
    contract = {
        "canonical_columns": ["id", "vendor_id", "pickup_datetime", "trip_duration", "__split"],
        "column_roles": {
            "id": ["id"],
            "categorical_features": ["vendor_id"],
            "temporal_features": ["pickup_datetime"],
            "target": ["trip_duration"],
            "split_indicator": ["__split"],
        },
        "objective_analysis": {"problem_type": "regression"},
    }
    required_columns = ["id", "vendor_id", "pickup_datetime", "trip_duration", "__split", "store_and_fwd_flag"]

    summary = _build_cleaned_data_summary_min(
        df_clean=df,
        contract=contract,
        required_columns=required_columns,
        data_path="data/cleaned_trips.csv",
    )

    assert "store_and_fwd_flag" in summary.get("missing_required_columns", [])
    assert summary.get("split_column") == "__split"

    by_name = {entry["column_name"]: entry for entry in summary.get("column_summaries", [])}
    vendor = by_name["vendor_id"]
    assert vendor["mismatch_with_contract"]["role_dtype_warning"] == "categorical_role_with_numeric_dtype"
    assert vendor["in_train"]["rows"] == 2
    assert vendor["in_test"]["rows"] == 2


def test_cleaned_data_summary_min_includes_outlier_treatment_advisory():
    df = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 1000.0],
            "target": [10.0, 20.0, 30.0],
        }
    )
    contract = {
        "canonical_columns": ["feature_a", "target"],
        "column_roles": {"pre_decision": ["feature_a"], "outcome": ["target"]},
    }
    summary = _build_cleaned_data_summary_min(
        df_clean=df,
        contract=contract,
        required_columns=["feature_a", "target"],
        outlier_policy={
            "enabled": True,
            "apply_stage": "data_engineer",
            "target_columns": ["feature_a"],
            "report_path": "data/outlier_treatment_report.json",
            "strict": True,
        },
        outlier_report={
            "status": "applied",
            "columns_touched": ["feature_a"],
            "rows_affected": 1,
            "flags_created": ["feature_a_outlier_flag"],
        },
        cleaning_manifest={"outlier_treatment": {"policy_applied": True}},
    )

    outlier = summary.get("outlier_treatment") or {}
    assert outlier.get("enabled") is True
    assert outlier.get("report_present") is True
    assert outlier.get("status") == "applied"
    assert "feature_a" in (outlier.get("columns_touched") or [])


def test_cleaned_data_summary_min_includes_cleaned_ml_fact_packet():
    df = pd.DataFrame(
        {
            "account_id": ["a1", "a2", "a3", "a4"],
            "snapshot_month_end": pd.to_datetime(
                ["2025-08-31", "2025-09-30", "2025-10-31", "2025-11-30"]
            ),
            "churn_60d": [0, 1, 0, None],
            "cohort_split": ["train", "train", "holdout", "scoring"],
            "arr_current": [1000.0, 1500.0, 2000.0, 2500.0],
            "region": ["emea", "na", "emea", "latam"],
            "executive_sponsor_present": [True, False, True, None],
        }
    )
    contract = {
        "allowed_feature_sets": {
            "model_features": ["arr_current", "region", "executive_sponsor_present"]
        },
        "task_semantics": {
            "target_columns": ["churn_60d"],
            "prediction_unit": "account-snapshot",
            "temporal_ordering_column": "snapshot_month_end",
        },
        "evaluation_spec": {"primary_metric": "pr_auc"},
        "validation_requirements": {"method": "temporal_holdout_with_cv"},
        "column_roles": {
            "identifiers": ["account_id", "snapshot_month_end"],
            "target": ["churn_60d"],
            "split_indicator": ["cohort_split"],
            "numerical_features": ["arr_current"],
            "categorical_features": ["region"],
            "boolean_features": ["executive_sponsor_present"],
        },
    }

    summary = _build_cleaned_data_summary_min(
        df_clean=df,
        contract=contract,
        required_columns=["account_id", "snapshot_month_end", "churn_60d", "cohort_split"],
        cleaning_manifest={
            "conversions": [
                "parsed_numeric:arr_current",
                "parsed_datetime:snapshot_month_end",
                "parsed_target:churn_60d",
            ],
            "final_columns_ml_ready": list(df.columns),
        },
    )

    packet = summary.get("cleaned_ml_fact_packet") or {}
    assert packet.get("target_column") == "churn_60d"
    assert packet.get("row_count_total") == 4
    assert packet.get("rows_labeled_target") == 3
    assert packet.get("rows_unlabeled_target") == 1
    assert packet.get("expected_scoring_row_count") == 1
    assert packet.get("validation_relevant_facts", {}).get("validation_method") == "temporal_holdout_with_cv"

    split_counts = packet.get("split_value_counts") or []
    assert any(entry.get("split_value") == "holdout" for entry in split_counts)

    readiness = (packet.get("feature_readiness") or {}).get("buckets") or {}
    assert readiness.get("numeric_ready", {}).get("count") == 1
    assert readiness.get("categorical_ready", {}).get("count") == 1
    assert readiness.get("boolean_ready", {}).get("count") == 1

    normalization = packet.get("data_engineer_normalization") or []
    assert any(item.get("action") == "parsed_numeric" for item in normalization)
