from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dateutil import parser as date_parser


_TARGET_HINT_TOKENS = {
    "target",
    "label",
    "y",
    "class",
    "outcome",
    "price",
    "saleprice",
    "score",
    "risk",
    "duration",
    "toxic",
}
_SPLIT_HINT_TOKENS = {"split", "fold", "train", "test", "partition", "is_train", "is_test"}
_ID_HINT_TOKENS = {"id", "uuid", "guid", "identifier", "key", "customer_id", "policy_id"}
_SUPPORTED_EVIDENCE_KINDS = {"missingness", "uniques", "column_profile"}
_VALID_TARGET_STATUSES = {"confirmed", "questioned", "invalid"}


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _tokenize_col(name: str) -> List[str]:
    if not isinstance(name, str):
        return []
    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")
    parts = [p for p in normalized.split("_") if p]
    return parts


def _has_hint(name: str, hint_tokens: set[str]) -> bool:
    tokens = _tokenize_col(name)
    if not tokens:
        return False
    if name.strip().lower() in hint_tokens:
        return True
    return any(tok in hint_tokens for tok in tokens)


def _normalize_column_inventory(column_inventory: Any, profile_columns: List[str]) -> List[str]:
    cols: List[str] = []
    if isinstance(column_inventory, list):
        cols = [str(c) for c in column_inventory if isinstance(c, str) and c.strip()]
    if cols:
        return cols
    return [str(c) for c in profile_columns if isinstance(c, str) and c.strip()]


def build_data_atlas(
    dataset_profile: Dict[str, Any],
    column_inventory: Any,
    column_sets: Optional[Dict[str, Any]] = None,
    *,
    max_top_values: int = 3,
) -> Dict[str, Any]:
    profile = dataset_profile if isinstance(dataset_profile, dict) else {}
    profile_columns = profile.get("columns") if isinstance(profile.get("columns"), list) else []
    columns = _normalize_column_inventory(column_inventory, profile_columns)

    type_hints = profile.get("type_hints") if isinstance(profile.get("type_hints"), dict) else {}
    missing_frac = profile.get("missing_frac") if isinstance(profile.get("missing_frac"), dict) else {}
    cardinality = profile.get("cardinality") if isinstance(profile.get("cardinality"), dict) else {}
    sampling = profile.get("sampling") if isinstance(profile.get("sampling"), dict) else {}
    compute_hints = profile.get("compute_hints") if isinstance(profile.get("compute_hints"), dict) else {}
    duplicate_stats = profile.get("duplicate_stats") if isinstance(profile.get("duplicate_stats"), dict) else {}
    temporal_analysis = profile.get("temporal_analysis") if isinstance(profile.get("temporal_analysis"), dict) else {}

    entries: List[Dict[str, Any]] = []
    constant_like_cols: List[str] = []
    high_missing_cols: List[Tuple[str, float]] = []
    split_name_hints: List[str] = []
    id_name_hints: List[str] = []
    target_name_hints: List[str] = []
    profiled_count = 0

    for col in columns:
        miss = _safe_float(missing_frac.get(col))
        uniq = None
        top_values: List[Dict[str, Any]] = []
        card_entry = cardinality.get(col)
        if isinstance(card_entry, dict):
            uniq = _safe_int(card_entry.get("unique"))
            raw_top = card_entry.get("top_values")
            if isinstance(raw_top, list):
                for item in raw_top[: max(1, int(max_top_values))]:
                    if not isinstance(item, dict):
                        continue
                    value = item.get("value")
                    count = _safe_int(item.get("count"))
                    top_values.append(
                        {
                            "value": str(value) if value is not None else "",
                            "count": count if count is not None else 0,
                        }
                    )
        type_hint = str(type_hints.get(col) or "unknown")
        constant_like = bool(uniq is not None and uniq <= 1)

        if col in type_hints or col in missing_frac or col in cardinality:
            profiled_count += 1
        if constant_like:
            constant_like_cols.append(col)
        if miss is not None:
            high_missing_cols.append((col, miss))
        if _has_hint(col, _SPLIT_HINT_TOKENS):
            split_name_hints.append(col)
        if _has_hint(col, _ID_HINT_TOKENS):
            id_name_hints.append(col)
        if _has_hint(col, _TARGET_HINT_TOKENS):
            target_name_hints.append(col)

        entries.append(
            {
                "name": col,
                "type_hint": type_hint,
                "missing_frac": miss,
                "unique_count": uniq,
                "constant_like": constant_like,
                "top_values": top_values,
            }
        )

    high_missing_cols.sort(key=lambda item: item[1], reverse=True)
    sets_meta = []
    if isinstance(column_sets, dict):
        raw_sets = column_sets.get("sets")
        if isinstance(raw_sets, list):
            for item in raw_sets[:12]:
                if not isinstance(item, dict):
                    continue
                selector = item.get("selector") if isinstance(item.get("selector"), dict) else {}
                sets_meta.append(
                    {
                        "name": str(item.get("name") or "SET"),
                        "selector_type": str(selector.get("type") or "unknown"),
                        "count": _safe_int(item.get("count")) or 0,
                    }
                )

    return {
        "coverage": {
            "total_columns": len(columns),
            "profiled_columns": profiled_count,
            "profile_coverage_ratio": round(float(profiled_count / len(columns)), 6) if columns else 0.0,
        },
        "sampling": {
            "was_sampled": bool(sampling.get("was_sampled")),
            "sample_size": _safe_int(sampling.get("sample_size")) or 0,
            "strategy": str(sampling.get("strategy") or "unknown"),
            "total_rows_in_file": _safe_int(sampling.get("total_rows_in_file")),
        },
        "compute_hints": compute_hints,
        "duplicate_stats": {
            "row_dup_count": _safe_int(duplicate_stats.get("row_dup_count")) or 0,
            "row_dup_frac": _safe_float(duplicate_stats.get("row_dup_frac")) or 0.0,
        },
        "temporal_overview": {
            "is_time_series": bool(temporal_analysis.get("is_time_series")),
            "detected_datetime_columns": [
                str(col)
                for col in (temporal_analysis.get("detected_datetime_columns") or [])
                if isinstance(col, str) and col.strip()
            ][:12],
            "details": [
                item
                for item in (temporal_analysis.get("details") or [])[:8]
                if isinstance(item, dict)
            ],
        },
        "signals": {
            "constant_like_count": len(constant_like_cols),
            "constant_like_sample": constant_like_cols[:30],
            "high_missing_top": [{"column": c, "missing_frac": m} for c, m in high_missing_cols[:30]],
            "target_name_hints": target_name_hints[:40],
            "split_name_hints": split_name_hints[:40],
            "id_name_hints": id_name_hints[:40],
        },
        "column_sets_overview": sets_meta,
        "columns": entries,
    }


def summarize_data_atlas(
    atlas: Dict[str, Any],
    *,
    max_columns: int = 60,
    max_lines: int = 80,
) -> str:
    if not isinstance(atlas, dict):
        return ""
    coverage = atlas.get("coverage") if isinstance(atlas.get("coverage"), dict) else {}
    sampling = atlas.get("sampling") if isinstance(atlas.get("sampling"), dict) else {}
    compute_hints = atlas.get("compute_hints") if isinstance(atlas.get("compute_hints"), dict) else {}
    duplicate_stats = atlas.get("duplicate_stats") if isinstance(atlas.get("duplicate_stats"), dict) else {}
    temporal_overview = atlas.get("temporal_overview") if isinstance(atlas.get("temporal_overview"), dict) else {}
    signals = atlas.get("signals") if isinstance(atlas.get("signals"), dict) else {}
    columns = atlas.get("columns") if isinstance(atlas.get("columns"), list) else []

    total_cols = int(coverage.get("total_columns") or 0)
    profiled_cols = int(coverage.get("profiled_columns") or 0)
    ratio = coverage.get("profile_coverage_ratio")

    lines: List[str] = ["DATA_ATLAS_SUMMARY:"]
    lines.append(f"- columns_total: {total_cols}")
    lines.append(f"- profiled_columns: {profiled_cols}")
    if ratio is not None:
        lines.append(f"- profile_coverage_ratio: {ratio}")
    lines.append(
        f"- sampling: strategy={sampling.get('strategy')}, was_sampled={sampling.get('was_sampled')}, "
        f"sample_size={sampling.get('sample_size')}, total_rows_in_file={sampling.get('total_rows_in_file')}"
    )
    if compute_hints:
        lines.append(
            f"- compute_hints: scale={compute_hints.get('scale_category')}, "
            f"estimated_memory_mb={compute_hints.get('estimated_memory_mb')}, "
            f"cv_feasible={compute_hints.get('cross_validation_feasible')}, "
            f"dl_feasible={compute_hints.get('deep_learning_feasible')}"
        )
    lines.append(
        f"- duplicate_rows: count={int(duplicate_stats.get('row_dup_count') or 0)}, "
        f"frac={duplicate_stats.get('row_dup_frac')}"
    )
    if temporal_overview:
        lines.append(
            f"- temporal_overview: is_time_series={bool(temporal_overview.get('is_time_series'))}, "
            f"detected_datetime_columns={list(temporal_overview.get('detected_datetime_columns') or [])[:8]}"
        )
        temporal_details = temporal_overview.get("details")
        if isinstance(temporal_details, list) and temporal_details:
            compact_details = []
            for item in temporal_details[:5]:
                if not isinstance(item, dict):
                    continue
                compact_details.append(
                    {
                        "column": str(item.get("column") or ""),
                        "granularity_hint": item.get("granularity_hint"),
                        "parse_ratio": item.get("parse_ratio"),
                        "unique_ratio": item.get("unique_ratio"),
                        "duplicate_ratio": item.get("duplicate_ratio"),
                        "max_rows_per_timestamp": item.get("max_rows_per_timestamp"),
                        "time_span_days": item.get("time_span_days"),
                    }
                )
            if compact_details:
                lines.append(f"- temporal_details: {compact_details}")
    lines.append(
        f"- constant_like_count: {int(signals.get('constant_like_count') or 0)} "
        f"(sample={list(signals.get('constant_like_sample') or [])[:8]})"
    )
    lines.append(f"- target_name_hints: {list(signals.get('target_name_hints') or [])[:12]}")
    lines.append(f"- split_name_hints: {list(signals.get('split_name_hints') or [])[:12]}")
    lines.append(f"- id_name_hints: {list(signals.get('id_name_hints') or [])[:12]}")

    high_missing = signals.get("high_missing_top")
    if isinstance(high_missing, list) and high_missing:
        lines.append(
            f"- high_missing_top: "
            f"{[{ 'column': str(x.get('column')), 'missing_frac': x.get('missing_frac')} for x in high_missing[:8] if isinstance(x, dict)]}"
        )

    priority: List[Dict[str, Any]] = []
    seen = set()
    hint_cols = (
        list(signals.get("target_name_hints") or [])
        + list(signals.get("split_name_hints") or [])
        + list(signals.get("id_name_hints") or [])
    )
    col_map = {str(item.get("name")): item for item in columns if isinstance(item, dict) and item.get("name")}
    for name in hint_cols:
        if name in col_map and name not in seen:
            seen.add(name)
            priority.append(col_map[name])
    for item in columns:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        if not name or name in seen:
            continue
        priority.append(item)
        seen.add(name)
        if len(priority) >= max(1, int(max_columns)):
            break

    lines.append("- column_snapshot:")
    for item in priority[: max(1, int(max_columns))]:
        name = str(item.get("name") or "")
        miss = item.get("missing_frac")
        uniq = item.get("unique_count")
        type_hint = str(item.get("type_hint") or "unknown")
        constant_like = bool(item.get("constant_like"))
        lines.append(
            f"  - {name}: type_hint={type_hint}, missing_frac={miss}, unique_count={uniq}, constant_like={constant_like}"
        )

    return "\n".join(lines[: max(8, int(max_lines))])


def normalize_evidence_requests(
    requests: Any,
    header_cols: List[str],
    *,
    max_items: int = 16,
) -> List[Dict[str, Any]]:
    allowed = set(str(c) for c in (header_cols or []) if c)
    if not allowed:
        return []
    normalized: List[Dict[str, Any]] = []
    seen = set()
    raw_list = requests if isinstance(requests, list) else []
    for item in raw_list:
        kind = None
        column = None
        max_unique = 20
        if isinstance(item, str):
            kind = "uniques"
            column = item
        elif isinstance(item, dict):
            kind = str(item.get("kind") or item.get("type") or "").strip().lower()
            column = item.get("column")
            maybe_unique = _safe_int(item.get("max_unique"))
            if maybe_unique is not None:
                max_unique = maybe_unique
        if kind not in _SUPPORTED_EVIDENCE_KINDS:
            continue
        column = str(column or "").strip()
        if not column or column not in allowed:
            continue
        if kind == "uniques":
            max_unique = max(5, min(int(max_unique), 100))
        else:
            max_unique = 0
        key = (kind, column, max_unique)
        if key in seen:
            continue
        seen.add(key)
        entry = {"kind": kind, "column": column}
        if kind == "uniques":
            entry["max_unique"] = max_unique
        normalized.append(entry)
        if len(normalized) >= max(1, int(max_items)):
            break
    return normalized


def build_default_evidence_requests(
    primary_target: Any,
    split_candidates: Any,
    id_candidates: Any,
    header_cols: List[str],
    *,
    max_items: int = 16,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    allowed = set(str(c) for c in (header_cols or []) if c)
    target = str(primary_target or "").strip()
    if target and target in allowed:
        candidates.append({"kind": "missingness", "column": target})
        candidates.append({"kind": "uniques", "column": target, "max_unique": 20})
    for col in (split_candidates if isinstance(split_candidates, list) else [])[:6]:
        name = str(col or "").strip()
        if name and name in allowed:
            candidates.append({"kind": "uniques", "column": name, "max_unique": 30})
            candidates.append({"kind": "missingness", "column": name})
    for col in (id_candidates if isinstance(id_candidates, list) else [])[:4]:
        name = str(col or "").strip()
        if name and name in allowed:
            candidates.append({"kind": "uniques", "column": name, "max_unique": 30})
    return normalize_evidence_requests(candidates, header_cols, max_items=max_items)


def validate_steward_semantics(
    dataset_semantics: Dict[str, Any],
    dataset_training_mask: Dict[str, Any],
    header_cols: List[str],
    target_missingness: Dict[str, Any],
    column_sets: Dict[str, Any],
) -> Dict[str, Any]:
    headers = set(str(c) for c in (header_cols or []) if c)
    semantics = dataset_semantics if isinstance(dataset_semantics, dict) else {}
    training_mask = dataset_training_mask if isinstance(dataset_training_mask, dict) else {}
    reasons: List[str] = []
    warnings: List[str] = []

    primary_target = str(semantics.get("primary_target") or "").strip()
    if not primary_target:
        reasons.append("missing_primary_target")
    elif primary_target not in headers:
        reasons.append(f"primary_target_not_in_header:{primary_target}")

    target_analysis = semantics.get("target_analysis") if isinstance(semantics.get("target_analysis"), dict) else {}
    target_status_raw = semantics.get("target_status")
    if not isinstance(target_status_raw, str) or not target_status_raw.strip():
        target_status_raw = target_analysis.get("target_status")
    target_status = str(target_status_raw or "").strip().lower()
    if target_status:
        if target_status not in _VALID_TARGET_STATUSES:
            reasons.append(f"invalid_target_status:{target_status}")
        elif target_status == "invalid":
            reasons.append("primary_target_invalid")
        elif target_status == "questioned":
            warnings.append("primary_target_questioned")

    recommended_primary_target_raw = semantics.get("recommended_primary_target")
    if not isinstance(recommended_primary_target_raw, str) or not recommended_primary_target_raw.strip():
        recommended_primary_target_raw = target_analysis.get("recommended_primary_target")
    recommended_primary_target = str(recommended_primary_target_raw or "").strip()
    if recommended_primary_target and recommended_primary_target not in headers:
        warnings.append(f"recommended_primary_target_not_in_header:{recommended_primary_target}")

    target_status_reason_raw = semantics.get("target_status_reason")
    if not isinstance(target_status_reason_raw, str) or not target_status_reason_raw.strip():
        target_status_reason_raw = target_analysis.get("target_status_reason")
    target_status_reason = str(target_status_reason_raw or "").strip()
    if target_status in {"questioned", "invalid"} and not target_status_reason:
        warnings.append("missing_target_status_reason")

    training_rows_rule = str(training_mask.get("training_rows_rule") or "").strip()
    scoring_rows_rule = str(training_mask.get("scoring_rows_rule_primary") or "").strip()
    if not training_rows_rule:
        reasons.append("missing_training_rows_rule")
    if not scoring_rows_rule:
        warnings.append("missing_scoring_rows_rule_primary")

    null_frac = _safe_float((target_missingness or {}).get("null_frac_exact"))
    if primary_target and null_frac is not None and 0.0 < null_frac < 1.0:
        tr_lower = training_rows_rule.lower()
        if primary_target.lower() not in tr_lower:
            warnings.append("training_rows_rule_not_explicit_about_primary_target")
        if "missing" not in tr_lower and "null" not in tr_lower:
            warnings.append("training_rows_rule_not_explicit_about_missingness")

    split_candidates = semantics.get("split_candidates")
    if isinstance(split_candidates, list):
        missing_split = [str(c) for c in split_candidates if str(c) not in headers]
        if missing_split:
            warnings.append(f"split_candidates_not_in_header:{missing_split[:8]}")

    if not isinstance(column_sets, dict):
        warnings.append("column_sets_not_dict")
    else:
        explicit = column_sets.get("explicit_columns")
        if isinstance(explicit, list):
            bad_explicit = [str(c) for c in explicit if str(c) not in headers]
            if bad_explicit:
                warnings.append(f"column_sets_explicit_not_in_header:{bad_explicit[:10]}")

    ready = len(reasons) == 0
    return {
        "ready": ready,
        "reasons": reasons,
        "warnings": warnings,
        "primary_target": primary_target,
        "target_status": target_status or "confirmed",
        "recommended_primary_target": recommended_primary_target,
        "target_status_reason": target_status_reason,
        "training_rows_rule": training_rows_rule,
    }


def _effective_non_missing_mask(series: pd.Series) -> pd.Series:
    try:
        text = series.astype(str).str.strip()
    except Exception:
        return series.notna()
    return series.notna() & ~text.str.lower().isin({"", "nan", "none", "null", "<na>", "nat"})


def _parse_datetime_multi(series: pd.Series) -> pd.Series:
    if not isinstance(series, pd.Series):
        return pd.Series(dtype="datetime64[ns]")
    mask = _effective_non_missing_mask(series)
    working = series.where(mask)
    parsed = pd.to_datetime(working, errors="coerce", utc=True)
    remaining = mask & parsed.isna()
    if bool(remaining.any()):
        parsed_dayfirst = pd.to_datetime(working[remaining], errors="coerce", utc=True, dayfirst=True)
        parsed.loc[remaining] = parsed_dayfirst
    remaining = mask & parsed.isna()
    if bool(remaining.any()):
        def _fallback_parse(value: Any) -> pd.Timestamp:
            text = str(value or "").strip()
            if not text or text.lower() in {"nan", "none", "null", "<na>", "nat"}:
                return pd.NaT
            for dayfirst in (False, True):
                try:
                    ts = pd.Timestamp(date_parser.parse(text, dayfirst=dayfirst))
                    if ts.tzinfo is None:
                        return ts.tz_localize("UTC")
                    return ts.tz_convert("UTC")
                except Exception:
                    continue
            return pd.NaT

        fallback = working[remaining].apply(_fallback_parse)
        parsed.loc[remaining] = fallback
    try:
        return parsed.dt.tz_convert(None)
    except Exception:
        try:
            return parsed.dt.tz_localize(None)
        except Exception:
            return parsed


def _bucketize_temporal_series(parsed: pd.Series, granularity_hint: str) -> pd.Series:
    if not isinstance(parsed, pd.Series):
        return pd.Series(dtype="object")
    granularity = str(granularity_hint or "").strip().lower()
    if granularity in {"monthly", "yearly"}:
        return parsed.dt.to_period("M").astype(str)
    if granularity == "weekly":
        return parsed.dt.to_period("W").astype(str)
    if granularity in {"hourly", "sub-minute"}:
        return parsed.dt.strftime("%Y-%m-%d %H:%M:%S")
    return parsed.dt.strftime("%Y-%m-%d")


def build_steward_focus_context(
    df: pd.DataFrame,
    *,
    primary_target: Any,
    split_candidates: Any,
    id_candidates: Any,
    max_bucket_count: int = 24,
) -> Dict[str, Any]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {
            "primary_target": str(primary_target or "").strip(),
            "split_candidates": [],
            "id_candidates": [],
            "temporal_label_maturity": [],
            "identifier_structure": [],
        }

    target = str(primary_target or "").strip()
    split_cols = [str(col).strip() for col in (split_candidates or []) if str(col).strip() in df.columns]
    id_cols = [str(col).strip() for col in (id_candidates or []) if str(col).strip() in df.columns]

    temporal_label_maturity: List[Dict[str, Any]] = []
    if target in df.columns:
        target_non_missing = _effective_non_missing_mask(df[target])
        for split_col in split_cols[:6]:
            parsed = _parse_datetime_multi(df[split_col])
            non_missing_split = _effective_non_missing_mask(df[split_col])
            parse_ratio = float((parsed.notna() & non_missing_split).sum() / max(int(non_missing_split.sum()), 1))
            if parse_ratio < 0.6:
                continue

            non_null = parsed.dropna().sort_values()
            granularity_hint = "unknown"
            if len(non_null) > 2:
                diffs = non_null.diff().dropna().dt.total_seconds()
                diffs = diffs[diffs > 0]
                if not diffs.empty:
                    median_seconds = float(diffs.median())
                    granularity_hint = (
                        "sub-minute" if median_seconds < 90 else
                        "hourly" if median_seconds < 5400 else
                        "daily" if median_seconds < 172800 else
                        "weekly" if median_seconds < 1209600 else
                        "monthly" if median_seconds < 38016000 else
                        "yearly"
                    )

            bucket = _bucketize_temporal_series(parsed, granularity_hint)
            working = pd.DataFrame(
                {
                    "bucket": bucket,
                    "target_present": target_non_missing,
                }
            )
            working = working[working["bucket"].notna()]
            if working.empty:
                continue
            grouped = (
                working.groupby("bucket", dropna=True)["target_present"]
                .agg(["count", "sum"])
                .reset_index()
                .rename(columns={"count": "rows", "sum": "labeled_rows"})
            )
            grouped["labeled_rows"] = grouped["labeled_rows"].astype(int)
            grouped["unlabeled_rows"] = grouped["rows"] - grouped["labeled_rows"]
            grouped["label_missing_frac"] = (
                grouped["unlabeled_rows"] / grouped["rows"].clip(lower=1)
            ).round(6)
            grouped = grouped.sort_values("bucket").reset_index(drop=True)

            earliest_unlabeled_bucket = None
            latest_fully_labeled_bucket = None
            for record in grouped.to_dict(orient="records"):
                if int(record["unlabeled_rows"]) > 0 and earliest_unlabeled_bucket is None:
                    earliest_unlabeled_bucket = str(record["bucket"])
                if int(record["unlabeled_rows"]) == 0:
                    latest_fully_labeled_bucket = str(record["bucket"])
            latest_pre_unlabeled = None
            if earliest_unlabeled_bucket is not None:
                for record in grouped.to_dict(orient="records"):
                    bucket_name = str(record["bucket"])
                    if bucket_name >= earliest_unlabeled_bucket:
                        break
                    if int(record["unlabeled_rows"]) == 0:
                        latest_pre_unlabeled = bucket_name

            temporal_label_maturity.append(
                {
                    "split_column": split_col,
                    "parse_ratio": round(parse_ratio, 4),
                    "granularity_hint": granularity_hint,
                    "raw_unique_count": int(df[split_col].nunique(dropna=True)),
                    "logical_unique_count": int(parsed.dropna().nunique()),
                    "latest_fully_labeled_bucket": latest_fully_labeled_bucket,
                    "latest_fully_labeled_before_unlabeled": latest_pre_unlabeled,
                    "earliest_bucket_with_unlabeled_rows": earliest_unlabeled_bucket,
                    "bucket_count": int(len(grouped)),
                    "bucket_label_maturity_sample": grouped.tail(max(1, int(max_bucket_count))).to_dict(orient="records"),
                }
            )

    identifier_structure: List[Dict[str, Any]] = []
    for id_col in id_cols[:4]:
        id_mask = _effective_non_missing_mask(df[id_col])
        id_series = df.loc[id_mask, id_col]
        entity_rows = int(len(id_series))
        unique_entities = int(id_series.nunique(dropna=True))
        duplicated_entity_rows = int(id_series.duplicated(keep=False).sum()) if entity_rows else 0
        entity_entry: Dict[str, Any] = {
            "id_column": id_col,
            "rows_with_id": entity_rows,
            "unique_entities": unique_entities,
            "duplicated_entity_rows": duplicated_entity_rows,
            "rows_per_entity_mean": round(entity_rows / max(unique_entities, 1), 4) if entity_rows else 0.0,
        }
        pair_diagnostics: List[Dict[str, Any]] = []
        for split_col in split_cols[:4]:
            subset = df[[id_col, split_col]].copy()
            subset = subset[_effective_non_missing_mask(subset[id_col]) & _effective_non_missing_mask(subset[split_col])]
            if subset.empty:
                continue
            dup_pair_rows = int(subset.duplicated(subset=[id_col, split_col], keep=False).sum())
            pair_diagnostics.append(
                {
                    "split_column": split_col,
                    "rows_with_pair": int(len(subset)),
                    "unique_pairs": int(subset.drop_duplicates(subset=[id_col, split_col]).shape[0]),
                    "duplicated_pair_rows": dup_pair_rows,
                    "duplicated_pair_frac": round(dup_pair_rows / max(len(subset), 1), 6),
                }
            )
        if pair_diagnostics:
            entity_entry["pair_diagnostics"] = pair_diagnostics
        identifier_structure.append(entity_entry)

    return {
        "primary_target": target,
        "split_candidates": split_cols,
        "id_candidates": id_cols,
        "temporal_label_maturity": temporal_label_maturity,
        "identifier_structure": identifier_structure,
    }


def summarize_steward_focus_context(
    focus_context: Dict[str, Any],
    *,
    max_lines: int = 40,
    max_buckets: int = 8,
) -> str:
    if not isinstance(focus_context, dict):
        return ""
    lines: List[str] = ["STEWARD_FOCUS_CONTEXT:"]
    primary_target = str(focus_context.get("primary_target") or "").strip()
    if primary_target:
        lines.append(f"- primary_target: {primary_target}")

    maturity = focus_context.get("temporal_label_maturity")
    if isinstance(maturity, list) and maturity:
        lines.append("- temporal_label_maturity:")
        for item in maturity[:4]:
            if not isinstance(item, dict):
                continue
            lines.append(
                "  - "
                f"{item.get('split_column')}: parse_ratio={item.get('parse_ratio')}, "
                f"granularity={item.get('granularity_hint')}, "
                f"raw_unique={item.get('raw_unique_count')}, logical_unique={item.get('logical_unique_count')}, "
                f"latest_fully_labeled={item.get('latest_fully_labeled_bucket')}, "
                f"latest_pre_unlabeled={item.get('latest_fully_labeled_before_unlabeled')}, "
                f"earliest_unlabeled={item.get('earliest_bucket_with_unlabeled_rows')}"
            )
            samples = item.get("bucket_label_maturity_sample")
            if isinstance(samples, list) and samples:
                compact = []
                for bucket in samples[-max(1, int(max_buckets)):]:
                    if not isinstance(bucket, dict):
                        continue
                    compact.append(
                        {
                            "bucket": bucket.get("bucket"),
                            "rows": bucket.get("rows"),
                            "unlabeled_rows": bucket.get("unlabeled_rows"),
                            "label_missing_frac": bucket.get("label_missing_frac"),
                        }
                    )
                if compact:
                    lines.append(f"    buckets_tail={compact}")

    identifier_structure = focus_context.get("identifier_structure")
    if isinstance(identifier_structure, list) and identifier_structure:
        lines.append("- identifier_structure:")
        for item in identifier_structure[:4]:
            if not isinstance(item, dict):
                continue
            lines.append(
                "  - "
                f"{item.get('id_column')}: rows_with_id={item.get('rows_with_id')}, "
                f"unique_entities={item.get('unique_entities')}, "
                f"duplicated_entity_rows={item.get('duplicated_entity_rows')}, "
                f"rows_per_entity_mean={item.get('rows_per_entity_mean')}"
            )
            pair_diagnostics = item.get("pair_diagnostics")
            if isinstance(pair_diagnostics, list) and pair_diagnostics:
                compact_pairs = []
                for pair in pair_diagnostics[:4]:
                    if not isinstance(pair, dict):
                        continue
                    compact_pairs.append(
                        {
                            "split_column": pair.get("split_column"),
                            "duplicated_pair_rows": pair.get("duplicated_pair_rows"),
                            "duplicated_pair_frac": pair.get("duplicated_pair_frac"),
                            "unique_pairs": pair.get("unique_pairs"),
                        }
                    )
                if compact_pairs:
                    lines.append(f"    pair_diagnostics={compact_pairs}")

    return "\n".join(lines[: max(8, int(max_lines))])


def resolve_steward_target_reconsideration_candidate(
    *,
    current_target: Any,
    steward_context_quality: Dict[str, Any] | None,
    dataset_semantics: Dict[str, Any] | None,
    header_cols: List[str] | None,
) -> Dict[str, Any]:
    quality = steward_context_quality if isinstance(steward_context_quality, dict) else {}
    semantics = dataset_semantics if isinstance(dataset_semantics, dict) else {}
    headers = {str(col).strip() for col in (header_cols or []) if str(col).strip()}
    current = str(current_target or "").strip()
    target_status = str(quality.get("target_status") or semantics.get("target_status") or "").strip().lower()
    reasons = quality.get("reasons") if isinstance(quality.get("reasons"), list) else []
    recommended = str(
        quality.get("recommended_primary_target")
        or semantics.get("recommended_primary_target")
        or ""
    ).strip()
    status_reason = str(
        quality.get("target_status_reason")
        or semantics.get("target_status_reason")
        or ""
    ).strip()
    is_invalid = target_status == "invalid" or "primary_target_invalid" in {str(item) for item in reasons}

    if not is_invalid:
        return {
            "should_retry": False,
            "candidate": "",
            "reason": "target_not_invalid",
            "target_status": target_status or "confirmed",
            "status_reason": status_reason,
        }
    if not recommended:
        return {
            "should_retry": False,
            "candidate": "",
            "reason": "no_recommended_primary_target",
            "target_status": target_status or "invalid",
            "status_reason": status_reason,
        }
    if recommended == current:
        return {
            "should_retry": False,
            "candidate": recommended,
            "reason": "recommended_matches_current_target",
            "target_status": target_status or "invalid",
            "status_reason": status_reason,
        }
    if recommended not in headers:
        return {
            "should_retry": False,
            "candidate": recommended,
            "reason": "recommended_target_not_in_header",
            "target_status": target_status or "invalid",
            "status_reason": status_reason,
        }
    return {
        "should_retry": True,
        "candidate": recommended,
        "reason": "recommended_primary_target_available",
        "target_status": target_status or "invalid",
        "status_reason": status_reason,
    }
