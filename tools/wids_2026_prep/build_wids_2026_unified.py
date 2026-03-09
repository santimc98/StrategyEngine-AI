#!/usr/bin/env python3
"""
Build a single competition-aware CSV for WiDS 2026 from train/test CSVs.

Expected input files inside --input-dir:
- train.csv
- test.csv
- metadata.csv (optional)
- sample_submission.csv (optional)

Main output:
- wids_2026_unified.csv

The output is optimized for the current agent stack:
- explicit __split column
- flattened tabular columns
- train-only multi-horizon labels exposed as columns
- optional metadata merged onto both train and test
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


SPLIT_COLUMN = "__split"
DEFAULT_ID_CANDIDATES = ("event_id", "EventId", "id", "Id")
DEFAULT_HORIZONS = (12, 24, 48, 72)
OPTIONAL_METADATA_FILENAMES = ("metadata.csv", "metaData.csv", "data_dictionary.csv")
OPTIONAL_SAMPLE_FILENAMES = ("sample_submission.csv",)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def _resolve_optional_input(input_dir: Path, candidates: Iterable[str]) -> Path | None:
    for candidate in candidates:
        path = input_dir / candidate
        if path.exists():
            return path
    lower_map = {child.name.lower(): child for child in input_dir.iterdir() if child.is_file()}
    for candidate in candidates:
        matched = lower_map.get(candidate.lower())
        if matched is not None:
            return matched
    return None


def _coerce_event_indicator(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.fillna(0).gt(0)


def _infer_id_column(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    explicit: str | None,
) -> str:
    if explicit:
        if explicit not in train_df.columns or explicit not in test_df.columns:
            raise ValueError(f"Configured id column '{explicit}' must exist in both train and test.")
        return explicit

    shared = set(train_df.columns) & set(test_df.columns)
    for candidate in DEFAULT_ID_CANDIDATES:
        if candidate in shared:
            return candidate
    raise ValueError(
        "Could not infer an id column shared by train and test. "
        "Pass --id-column explicitly."
    )


def _merge_metadata(
    df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    id_column: str,
) -> tuple[pd.DataFrame, List[str]]:
    if metadata_df.empty:
        return df, []
    if id_column not in metadata_df.columns:
        raise ValueError(f"metadata.csv does not contain the id column '{id_column}'.")

    rename_map: Dict[str, str] = {}
    for column in metadata_df.columns:
        if column == id_column:
            continue
        if column in df.columns:
            rename_map[column] = f"meta__{column}"

    metadata_df = metadata_df.rename(columns=rename_map)
    merged = df.merge(metadata_df, how="left", on=id_column, validate="m:1")
    merged_columns = [col for col in metadata_df.columns if col != id_column]
    return merged, merged_columns


def _classify_metadata_table(metadata_df: pd.DataFrame, id_column: str) -> str:
    columns_lower = {str(col).strip().lower() for col in metadata_df.columns}
    if id_column in metadata_df.columns:
        return "row_level"
    if "column" in columns_lower and (
        "description" in columns_lower or "type" in columns_lower or "category" in columns_lower
    ):
        return "column_dictionary"
    return "unknown"


def _build_column_metadata_summary(
    metadata_df: pd.DataFrame,
    combined_df: pd.DataFrame,
) -> Dict[str, Any]:
    renamed = metadata_df.copy()
    renamed.columns = [str(col).strip().lower() for col in renamed.columns]

    summary: Dict[str, Any] = {
        "mode": "column_dictionary",
        "metadata_rows": int(len(renamed)),
        "metadata_columns": list(metadata_df.columns),
    }

    if "column" in renamed.columns:
        metadata_columns = [str(value).strip() for value in renamed["column"].tolist() if str(value).strip()]
        combined_columns = set(str(col) for col in combined_df.columns)
        present = [col for col in metadata_columns if col in combined_columns]
        missing = [col for col in metadata_columns if col not in combined_columns]
        summary["declared_dataset_columns"] = metadata_columns
        summary["declared_columns_present"] = present
        summary["declared_columns_missing"] = missing

    if "type" in renamed.columns and "column" in renamed.columns:
        type_map = {}
        for _, row in renamed[["column", "type"]].dropna(subset=["column"]).iterrows():
            type_map[str(row["column"]).strip()] = str(row["type"]).strip()
        summary["column_type_map"] = type_map

    return summary


def _ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column in columns:
        if column not in df.columns:
            df[column] = pd.NA
    return df


def _derive_horizon_labels(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    time_column: str,
    event_column: str,
    horizons: Iterable[int],
) -> tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    if time_column not in train_df.columns or event_column not in train_df.columns:
        return train_df, test_df, []

    train_time = pd.to_numeric(train_df[time_column], errors="coerce")
    train_event = _coerce_event_indicator(train_df[event_column])

    label_columns: List[str] = []
    for horizon in horizons:
        label_column = f"label_{int(horizon)}h"
        train_df[label_column] = (train_event & train_time.le(float(horizon))).astype("Int64")
        test_df[label_column] = pd.Series([pd.NA] * len(test_df), dtype="Int64")
        label_columns.append(label_column)
    return train_df, test_df, label_columns


def _align_train_test_columns(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_columns = list(train_df.columns)
    test_columns = list(test_df.columns)
    all_columns = list(dict.fromkeys(train_columns + [col for col in test_columns if col not in train_columns]))
    train_df = _ensure_columns(train_df.copy(), all_columns)
    test_df = _ensure_columns(test_df.copy(), all_columns)
    return train_df[all_columns], test_df[all_columns]


def _build_ordered_columns(
    combined_df: pd.DataFrame,
    id_column: str,
    label_columns: List[str],
    auxiliary_target_columns: List[str],
) -> List[str]:
    front = [id_column, SPLIT_COLUMN]
    special = set(front) | set(label_columns) | set(auxiliary_target_columns)
    feature_columns = [col for col in combined_df.columns if col not in special]
    ordered = front + feature_columns + label_columns + auxiliary_target_columns
    return [col for col in ordered if col in combined_df.columns]


def _validate_sample_submission(
    test_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    id_column: str,
) -> Dict[str, Any]:
    if id_column not in sample_df.columns:
        raise ValueError(f"sample_submission.csv does not contain the id column '{id_column}'.")
    test_ids = test_df[id_column].tolist()
    sample_ids = sample_df[id_column].tolist()
    same_ids = set(test_ids) == set(sample_ids)
    same_order = test_ids == sample_ids
    return {
        "sample_rows": int(len(sample_df)),
        "test_rows": int(len(test_df)),
        "same_ids": bool(same_ids),
        "same_order": bool(same_order),
        "submission_columns": list(sample_df.columns),
    }


def build_dataset(
    input_dir: Path,
    output_dir: Path,
    id_column: str | None,
    time_column: str,
    event_column: str,
) -> Dict[str, Any]:
    train_path = input_dir / "train.csv"
    test_path = input_dir / "test.csv"
    metadata_path = _resolve_optional_input(input_dir, OPTIONAL_METADATA_FILENAMES)
    sample_path = _resolve_optional_input(input_dir, OPTIONAL_SAMPLE_FILENAMES)

    train_df = _read_csv(train_path)
    test_df = _read_csv(test_path)

    resolved_id = _infer_id_column(train_df, test_df, id_column)

    metadata_columns: List[str] = []
    metadata_mode = "absent"
    metadata_summary: Dict[str, Any] | None = None
    if metadata_path is not None and metadata_path.exists():
        metadata_df = _read_csv(metadata_path)
        metadata_mode = _classify_metadata_table(metadata_df, resolved_id)
        if metadata_mode == "row_level":
            train_df, metadata_columns = _merge_metadata(train_df, metadata_df, resolved_id)
            test_df, _ = _merge_metadata(test_df, metadata_df, resolved_id)
        elif metadata_mode == "column_dictionary":
            metadata_summary = _build_column_metadata_summary(
                metadata_df=metadata_df,
                combined_df=pd.concat([train_df, test_df], ignore_index=True),
            )
        else:
            metadata_summary = {
                "mode": "unknown",
                "metadata_rows": int(len(metadata_df)),
                "metadata_columns": list(metadata_df.columns),
                "message": (
                    "Metadata file detected but not merged because it is neither row-level metadata "
                    "nor a recognized column dictionary."
                ),
            }

    train_df, test_df, label_columns = _derive_horizon_labels(
        train_df=train_df,
        test_df=test_df,
        time_column=time_column,
        event_column=event_column,
        horizons=DEFAULT_HORIZONS,
    )

    train_df[SPLIT_COLUMN] = "train"
    test_df[SPLIT_COLUMN] = "test"
    train_df, test_df = _align_train_test_columns(train_df, test_df)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        combined_df = pd.concat([train_df, test_df], ignore_index=True)

    auxiliary_target_columns = [
        column for column in (time_column, event_column) if column in combined_df.columns
    ]
    ordered_columns = _build_ordered_columns(
        combined_df=combined_df,
        id_column=resolved_id,
        label_columns=label_columns,
        auxiliary_target_columns=auxiliary_target_columns,
    )
    combined_df = combined_df[ordered_columns]

    sample_check: Dict[str, Any] | None = None
    if sample_path is not None and sample_path.exists():
        sample_df = _read_csv(sample_path)
        sample_check = _validate_sample_submission(test_df, sample_df, resolved_id)

    output_dir.mkdir(parents=True, exist_ok=True)
    unified_path = output_dir / "wids_2026_unified.csv"
    summary_path = output_dir / "prep_summary.json"
    metadata_summary_path = output_dir / "column_metadata_summary.json"
    combined_df.to_csv(unified_path, index=False)

    summary: Dict[str, Any] = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "id_column": resolved_id,
        "split_column": SPLIT_COLUMN,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "combined_rows": int(len(combined_df)),
        "column_count": int(len(combined_df.columns)),
        "label_columns": label_columns,
        "auxiliary_target_columns": auxiliary_target_columns,
        "metadata_path": str(metadata_path) if metadata_path is not None else None,
        "metadata_mode": metadata_mode,
        "metadata_merged": bool(metadata_columns),
        "metadata_columns_added": metadata_columns,
        "output_csv": str(unified_path),
    }
    if metadata_summary is not None:
        metadata_summary_path.write_text(json.dumps(metadata_summary, indent=2), encoding="utf-8")
        summary["metadata_summary_path"] = str(metadata_summary_path)
    if sample_check is not None:
        summary["sample_submission_check"] = sample_check

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Build a WiDS 2026 unified CSV from train/test inputs.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=script_dir / "input",
        help="Directory containing train.csv, test.csv, and optional metadata.csv/sample_submission.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "output",
        help="Directory where the unified CSV and summary JSON will be written",
    )
    parser.add_argument("--id-column", default=None, help="Explicit id column. Defaults to auto-detect.")
    parser.add_argument("--time-column", default="time_to_hit_hours", help="Time-to-event column in train.csv")
    parser.add_argument("--event-column", default="event", help="Event indicator column in train.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        id_column=args.id_column,
        time_column=args.time_column,
        event_column=args.event_column,
    )
    print("[OK] Unified dataset created")
    print(f"     output_csv: {summary['output_csv']}")
    print(f"     rows: train={summary['train_rows']} test={summary['test_rows']} total={summary['combined_rows']}")
    print(f"     id_column: {summary['id_column']}")
    print(f"     label_columns: {', '.join(summary['label_columns']) or 'none'}")
    if summary.get("sample_submission_check"):
        check = summary["sample_submission_check"]
        print(
            "     sample_submission_check: "
            f"same_ids={check['same_ids']} same_order={check['same_order']} columns={check['submission_columns']}"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
