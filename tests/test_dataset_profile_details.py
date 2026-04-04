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
        "cardinality_note": "test",
    }

    data_profile = convert_dataset_profile_to_data_profile(dataset_profile, contract={})

    assert data_profile.get("missingness") == dataset_profile["missing_frac"]
    assert data_profile.get("numeric_summary") == dataset_profile["numeric_summary"]
    assert data_profile.get("text_summary") == dataset_profile["text_summary"]
    assert data_profile.get("duplicate_stats") == dataset_profile["duplicate_stats"]
    assert data_profile.get("sampling") == dataset_profile["sampling"]
    assert data_profile.get("dialect") == dataset_profile["dialect"]
