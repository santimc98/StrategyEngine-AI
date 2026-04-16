import pandas as pd

from src.agents.steward import build_dataset_profile
from src.utils.data_profile_compact import convert_dataset_profile_to_data_profile


def test_build_dataset_profile_includes_numeric_and_text_summaries() -> None:
    df = pd.DataFrame(
        {
            "num": [1, 2, 3, 4, None],
            "cat": ["A", "B", "B", None, "C"],
        }
    )
    profile = build_dataset_profile(
        df=df,
        objective="test",
        dialect_info={"sep": ",", "decimal": ".", "diagnostics": {}},
        encoding="utf-8",
        file_size_bytes=123,
        was_sampled=False,
        sample_size=len(df),
        pii_findings=None,
    )

    assert "numeric_summary" in profile
    assert "text_summary" in profile
    assert "num" in profile["numeric_summary"]
    assert "cat" in profile["text_summary"]
    assert "duplicate_stats" in profile
    assert "row_dup_count" in profile["duplicate_stats"]
    assert "temporal_analysis" in profile


def test_build_dataset_profile_includes_temporal_analysis_for_datetime_like_columns() -> None:
    df = pd.DataFrame(
        {
            "account_id": ["A", "A", "B", "B"],
            "snapshot_month_end": ["2025-01-31", "2025/02/28", "31/03/2025", "04-30-2025"],
            "target": [0, 1, 0, None],
        }
    )
    profile = build_dataset_profile(
        df=df,
        objective="test",
        dialect_info={"sep": ",", "decimal": ".", "diagnostics": {}},
        encoding="utf-8",
        file_size_bytes=123,
        was_sampled=False,
        sample_size=len(df),
        pii_findings=None,
    )

    temporal = profile.get("temporal_analysis") or {}
    assert temporal.get("detected_datetime_columns")
    details = temporal.get("details") or []
    assert any(item.get("column") == "snapshot_month_end" for item in details if isinstance(item, dict))


def test_build_dataset_profile_adds_temporal_normalization_facts_for_mixed_period_dates() -> None:
    df = pd.DataFrame(
        {
            "account_id": ["A", "B", "C", "D", "E", "F"],
            "snapshot_month_end": [
                "2025-01-31",
                "01/31/2025",
                "2025/01/31",
                "2025-02-28",
                "02-28-2025",
                "2025-02-28T18:00:00",
            ],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )

    profile = build_dataset_profile(
        df=df,
        objective="test",
        dialect_info={"sep": ",", "decimal": ".", "diagnostics": {}},
        encoding="utf-8",
        file_size_bytes=123,
        was_sampled=False,
        sample_size=len(df),
        pii_findings=None,
    )

    facts = profile.get("temporal_normalization_facts") or []
    snapshot_fact = next(item for item in facts if item.get("column") == "snapshot_month_end")
    assert snapshot_fact["raw_unique_count"] == 6
    assert snapshot_fact["canonical_unique_counts"]["month_period"] == 2
    assert snapshot_fact["normalization_collapse_risk"] == "high"
    assert snapshot_fact["contract_gate_guidance"]["raw_unique_count_is_pre_normalization"] is True


def test_temporal_profile_parses_year_first_dates_without_day_month_flip() -> None:
    df = pd.DataFrame(
        {
            "account_id": ["A", "B", "C", "D", "E", "F", "G"],
            "account_created_at": [
                "04/10/2023",
                "30/04/2024",
                "2024-01-02",
                "2024-04-30",
                "2023/12/12",
                "2024-06-13T18:00:00",
                "2025-02-11",
            ],
            "target": [0, 1, 0, 1, 0, 1, 0],
        }
    )

    profile = build_dataset_profile(
        df=df,
        objective="test",
        dialect_info={"sep": ",", "decimal": ".", "diagnostics": {}},
        encoding="utf-8",
        file_size_bytes=123,
        was_sampled=False,
        sample_size=len(df),
        pii_findings=None,
    )

    facts = profile.get("temporal_normalization_facts") or []
    created_fact = next(item for item in facts if item.get("column") == "account_created_at")
    assert profile["type_hints"]["account_created_at"] == "datetime"
    assert created_fact["parse_policy"] == "format_family_aware_explicit_yearfirst"
    assert created_fact["ambiguous_date_resolution"]["preferences"]["slash"] == "dayfirst"
    assert 495 <= created_fact["time_span_days"] <= 497


def test_temporal_profile_marks_unresolved_ambiguous_spans_low_confidence() -> None:
    df = pd.DataFrame(
        {
            "event_date": ["01/02/2024", "02/03/2024", "2024-04-05", "2024-04-06"],
            "target": [0, 1, 0, 1],
        }
    )

    profile = build_dataset_profile(
        df=df,
        objective="test",
        dialect_info={"sep": ",", "decimal": ".", "diagnostics": {}},
        encoding="utf-8",
        file_size_bytes=123,
        was_sampled=False,
        sample_size=len(df),
        pii_findings=None,
    )

    facts = profile.get("temporal_normalization_facts") or []
    event_fact = next(item for item in facts if item.get("column") == "event_date")
    assert event_fact["time_span_confidence"] == "low"
    assert event_fact["ambiguous_date_resolution"]["unresolved_ambiguous_token_count"] == 2
    assert "time_span_days_policy" in event_fact["contract_gate_guidance"]


def test_temporal_analysis_does_not_treat_numeric_day_durations_as_dates() -> None:
    df = pd.DataFrame(
        {
            "account_id": ["A", "B", "C"],
            "login_days_14d": [1, 4, 9],
            "invoice_overdue_days": [0, 15, 30],
            "snapshot_month_end": ["2025-01-31", "2025-02-28", "2025-03-31"],
        }
    )

    profile = build_dataset_profile(
        df=df,
        objective="test",
        dialect_info={"sep": ",", "decimal": ".", "diagnostics": {}},
        encoding="utf-8",
        file_size_bytes=123,
        was_sampled=False,
        sample_size=len(df),
        pii_findings=None,
    )

    detected = set(profile.get("temporal_analysis", {}).get("detected_datetime_columns") or [])
    assert "snapshot_month_end" in detected
    assert "login_days_14d" not in detected
    assert "invoice_overdue_days" not in detected


def test_convert_dataset_profile_preserves_extended_fields() -> None:
    dataset_profile = {
        "rows": 5,
        "cols": 2,
        "columns": ["num", "cat"],
        "type_hints": {"num": "numeric", "cat": "categorical"},
        "missing_frac": {"num": 0.2, "cat": 0.2},
        "cardinality": {"num": {"unique": 4, "top_values": []}, "cat": {"unique": 3, "top_values": []}},
        "numeric_summary": {"num": {"count": 4, "mean": 2.5}},
        "text_summary": {"cat": {"count": 4, "avg_len": 1.0}},
        "duplicate_stats": {"row_dup_count": 0, "row_dup_frac": 0.0},
        "sampling": {"was_sampled": False, "sample_size": 5, "file_size_bytes": 123},
        "dialect": {"sep": ",", "decimal": ".", "encoding": "utf-8"},
        "pii_findings": {"detected": False, "findings": []},
        "temporal_normalization_facts": [
            {
                "column": "cat",
                "raw_unique_count": 3,
                "canonical_unique_counts": {"date": 2},
                "normalization_collapse_risk": "medium",
            }
        ],
        "cardinality_note": "test",
    }

    data_profile = convert_dataset_profile_to_data_profile(dataset_profile, contract={})

    assert data_profile.get("missingness") == dataset_profile["missing_frac"]
    assert data_profile.get("numeric_summary") == dataset_profile["numeric_summary"]
    assert data_profile.get("text_summary") == dataset_profile["text_summary"]
    assert data_profile.get("duplicate_stats") == dataset_profile["duplicate_stats"]
    assert data_profile.get("sampling") == dataset_profile["sampling"]
    assert data_profile.get("dialect") == dataset_profile["dialect"]
    assert data_profile.get("temporal_normalization_facts") == dataset_profile["temporal_normalization_facts"]
