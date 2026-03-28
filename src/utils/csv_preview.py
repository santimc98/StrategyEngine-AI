from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.utils.csv_dialect import load_output_dialect, read_csv_sample, sniff_csv_dialect


def _candidate_manifest_paths(csv_path: str) -> list[str]:
    path = Path(csv_path)
    candidates: list[str] = []
    seen: set[str] = set()
    for parent in [path.parent, *path.parents]:
        if not parent:
            continue
        for candidate in (
            parent / "cleaning_manifest.json",
            parent / "data" / "cleaning_manifest.json",
            parent / "artifacts" / "clean" / "cleaning_manifest.json",
        ):
            candidate_str = str(candidate)
            if candidate_str not in seen:
                candidates.append(candidate_str)
                seen.add(candidate_str)
    return candidates


def resolve_preview_dialect(csv_path: str) -> Dict[str, Any]:
    for manifest_path in _candidate_manifest_paths(csv_path):
        dialect = load_output_dialect(manifest_path)
        if isinstance(dialect, dict) and dialect.get("sep"):
            return dialect
    return sniff_csv_dialect(csv_path)


def _alternate_dialect(dialect: Dict[str, Any]) -> Dict[str, Any]:
    sep = str(dialect.get("sep") or ",")
    encoding = str(dialect.get("encoding") or "utf-8")
    if sep == ";":
        return {"sep": ",", "decimal": ".", "encoding": encoding}
    return {"sep": ";", "decimal": ",", "encoding": encoding}


def load_csv_preview(csv_path: str, max_rows: int = 50) -> Dict[str, Any]:
    if not csv_path:
        return {}

    base_dialect = resolve_preview_dialect(csv_path)
    candidate_dialects = [base_dialect]
    alt_dialect = _alternate_dialect(base_dialect)
    if alt_dialect != base_dialect:
        candidate_dialects.append(alt_dialect)

    best_payload: Dict[str, Any] = {}
    last_error: Optional[Exception] = None

    for dialect in candidate_dialects:
        sample = read_csv_sample(csv_path, dialect, max_rows)
        columns = sample.get("columns") or []
        if not columns:
            continue

        if not best_payload or len(columns) > int(best_payload.get("col_count") or 0):
            best_payload = {
                "columns": columns,
                "rows": sample.get("rows") or [],
                "row_count_total": sample.get("row_count_total"),
                "col_count": len(columns),
                "dialect_used": sample.get("dialect_used") or dialect,
            }

        if len(columns) <= 1:
            continue

        try:
            frame = pd.read_csv(
                csv_path,
                nrows=max_rows,
                sep=str(dialect.get("sep") or ","),
                decimal=str(dialect.get("decimal") or "."),
                encoding=str(dialect.get("encoding") or "utf-8"),
                engine="python",
            )
        except Exception as exc:
            last_error = exc
            frame = pd.DataFrame(sample.get("rows") or [], columns=columns)

        return {
            "df": frame,
            "row_count_total": sample.get("row_count_total"),
            "col_count": len(columns),
            "dialect_used": sample.get("dialect_used") or dialect,
        }

    if best_payload:
        return {
            "df": pd.DataFrame(best_payload.get("rows") or [], columns=best_payload.get("columns") or []),
            "row_count_total": best_payload.get("row_count_total"),
            "col_count": best_payload.get("col_count") or 0,
            "dialect_used": best_payload.get("dialect_used") or base_dialect,
            "warning": "single_column_or_ambiguous_dialect",
        }

    if last_error is not None:
        raise last_error
    return {}
