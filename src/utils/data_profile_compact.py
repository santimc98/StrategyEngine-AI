
from typing import Dict, Any, List
from datetime import datetime, timezone

# Tokens that indicate a column might be a split/fold indicator
SPLIT_CANDIDATE_TOKENS = {"split", "set", "fold", "train", "test", "partition", "is_train", "is_test"}
TARGET_TOKEN_HINTS = {"label", "target", "outcome", "response", "class", "y"}
ID_TOKEN_HINTS = {"id", "uuid", "guid", "key", "account", "customer", "user", "session", "device", "row", "index", "record"}


def convert_dataset_profile_to_data_profile(
    dataset_profile: Dict[str, Any],
    contract: Dict[str, Any],
    analysis_type: str | None = None,
) -> Dict[str, Any]:
    """
    Convert a dataset_profile.json (Steward output) to data_profile schema.

    This is the CANONICAL evidence conversion: dataset_profile is the source of truth,
    and we derive data_profile from it using the contract for context.

    Args:
        dataset_profile: The Steward-generated dataset profile (rows, cols, missing_frac, cardinality, etc.)
        contract: Execution contract with outcome_columns, column_roles, validation_requirements
        analysis_type: Optional analysis type (classification, regression)

    Returns:
        data_profile dict with the standard schema:
        - basic_stats, dtypes, missingness_top30, outcome_analysis, split_candidates,
        - constant_columns, high_cardinality_columns, leakage_flags, schema_version, generated_at
    """
    contract = contract or {}
    dataset_profile = dataset_profile or {}

    sample_rows = int(dataset_profile.get("rows", 0))
    n_cols = int(dataset_profile.get("cols", 0))
    columns = list(dataset_profile.get("columns", []))
    sampling = dataset_profile.get("sampling", {}) if isinstance(dataset_profile.get("sampling"), dict) else {}
    was_sampled = bool(sampling.get("was_sampled"))
    total_rows_in_file_raw = sampling.get("total_rows_in_file")
    total_rows_in_file = int(total_rows_in_file_raw) if isinstance(total_rows_in_file_raw, (int, float)) else 0
    # Prefer total_rows_in_file for scale-sensitive reasoning when profile was sampled.
    # This avoids underestimating memory and cardinality ratios on large datasets.
    n_rows = total_rows_in_file if (was_sampled and total_rows_in_file > 0) else sample_rows
    sampled_profile_uncertain = was_sampled and total_rows_in_file > 0 and sample_rows > 0 and sample_rows < total_rows_in_file

    # 1. basic_stats
    basic_stats = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "columns": columns,
    }

    # 2. dtypes from type_hints (map to dtype-like strings)
    type_hints = dataset_profile.get("type_hints", {})
    dtypes = {}
    for col in columns:
        hint = type_hints.get(col, "object")
        # Map type_hints to pandas-like dtypes for consistency
        if hint == "numeric":
            dtypes[col] = "float64"
        elif hint == "categorical":
            dtypes[col] = "object"
        elif hint == "datetime":
            dtypes[col] = "datetime64"
        elif hint == "boolean":
            dtypes[col] = "bool"
        else:
            dtypes[col] = "object"

    # 3. missingness_top30 from missing_frac
    missing_frac = dataset_profile.get("missing_frac", {})
    sorted_miss = sorted(missing_frac.items(), key=lambda x: x[1], reverse=True)
    missingness_top30 = {k: round(v, 4) for k, v in sorted_miss[:30]}

    # 4. outcome_analysis from contract + missing_frac + cardinality
    outcome_analysis = {}
    outcome_cols = _extract_outcome_columns(contract)
    cardinality = dataset_profile.get("cardinality", {})
    compute_hints = dataset_profile.get("compute_hints") if isinstance(dataset_profile.get("compute_hints"), dict) else {}

    for outcome_col in outcome_cols:
        if outcome_col not in columns:
            outcome_analysis[outcome_col] = {"present": False, "error": "column_not_found"}
            continue

        null_frac = missing_frac.get(outcome_col, 0.0)
        non_null_count = int(n_rows * (1 - null_frac)) if n_rows > 0 else 0

        analysis_entry = {
            "present": True,
            "non_null_count": non_null_count,
            "total_count": n_rows,
            "null_frac": round(null_frac, 4),
        }

        # Determine inferred_type from cardinality or analysis_type
        card_info = cardinality.get(outcome_col, {})
        n_unique = card_info.get("unique", 0)
        inferred = analysis_type or ""
        if not inferred:
            inferred = "classification" if n_unique <= 20 else "regression"
        analysis_entry["inferred_type"] = inferred
        analysis_entry["n_unique"] = n_unique

        # Add class_counts for classification
        if inferred == "classification":
            top_values = card_info.get("top_values", [])
            class_counts = {}
            for tv in top_values:
                val = str(tv.get("value", ""))
                # Skip nan values in class counts
                if val.lower() != "nan":
                    class_counts[val] = int(tv.get("count", 0))
            analysis_entry["n_classes"] = len(class_counts) if class_counts else n_unique
            analysis_entry["class_counts"] = class_counts
            if class_counts:
                min_class = min(class_counts.values())
                max_class = max(class_counts.values())
                total_classes = sum(class_counts.values())
                analysis_entry["class_imbalance_ratio"] = round(float(min_class / max(max_class, 1)), 6)
                analysis_entry["minority_class_share"] = round(float(min_class / max(total_classes, 1)), 6)

        outcome_analysis[outcome_col] = analysis_entry

    # 5. split_candidates: detect columns with split-related names and values
    split_candidates = []
    for col in columns:
        col_lower = col.lower().replace("_", " ").replace("-", " ")
        tokens = set(col_lower.split())
        if tokens & SPLIT_CANDIDATE_TOKENS:
            card_info = cardinality.get(col, {})
            top_values = card_info.get("top_values", [])
            unique_values_sample = [str(tv.get("value", "")) for tv in top_values[:20]]
            split_candidates.append({
                "column": col,
                "unique_values_sample": unique_values_sample,
            })

    # 6. constant_columns:
    # For sampled profiles we keep sample evidence separate and avoid hard "constant"
    # claims that can cause false deterministic filters downstream.
    constant_columns = []
    constant_columns_sample = []
    for col in columns:
        card_info = cardinality.get(col, {})
        n_unique = card_info.get("unique", 0)
        if n_unique <= 1:
            constant_columns_sample.append(col)
    if not sampled_profile_uncertain:
        constant_columns = list(constant_columns_sample)

    # 7. high_cardinality_columns: unique ratio > 0.95 and > 50 uniques.
    # For sampled profiles, treat this as advisory evidence only.
    high_cardinality_columns = []
    high_cardinality_columns_sample = []
    for col in columns:
        card_info = cardinality.get(col, {})
        n_unique = card_info.get("unique", 0)
        unique_ratio = n_unique / n_rows if n_rows > 0 else 0
        if unique_ratio > 0.95 and n_unique > 50:
            high_cardinality_columns_sample.append({
                "column": col,
                "n_unique": n_unique,
                "unique_ratio": round(unique_ratio, 4),
            })
    if not sampled_profile_uncertain:
        high_cardinality_columns = list(high_cardinality_columns_sample)

    # 8. leakage_flags: outcome column name appears in other columns
    leakage_flags = []
    outcome_names_lower = {c.lower() for c in outcome_cols}
    for col in columns:
        col_lower = col.lower()
        for outcome in outcome_names_lower:
            if outcome in col_lower and col not in outcome_cols:
                leakage_flags.append({
                    "column": col,
                    "reason": f"name_contains_outcome:{outcome}",
                    "severity": "SOFT",
                })

    # Build the data_profile
    data_profile = {
        "basic_stats": basic_stats,
        "dtypes": dtypes,
        "missingness": missing_frac,
        "missingness_top30": missingness_top30,
        "outcome_analysis": outcome_analysis,
        "split_candidates": split_candidates,
        "constant_columns": constant_columns,
        "constant_columns_sample": constant_columns_sample,
        "constant_columns_confidence": "low_sampled" if sampled_profile_uncertain else "high_full_or_complete",
        "high_cardinality_columns": high_cardinality_columns,
        "high_cardinality_columns_sample": high_cardinality_columns_sample,
        "high_cardinality_confidence": "low_sampled" if sampled_profile_uncertain else "high_full_or_complete",
        "leakage_flags": leakage_flags,
        "cardinality": dataset_profile.get("cardinality", {}),
        "numeric_summary": dataset_profile.get("numeric_summary", {}),
        "text_summary": dataset_profile.get("text_summary", {}),
        "duplicate_stats": dataset_profile.get("duplicate_stats", {}),
        "sampling": dataset_profile.get("sampling", {}),
        "compute_hints": compute_hints,
        "temporal_analysis": dataset_profile.get("temporal_analysis", {}),
        "temporal_normalization_facts": dataset_profile.get("temporal_normalization_facts", [])
        if isinstance(dataset_profile.get("temporal_normalization_facts"), list)
        else (
            dataset_profile.get("temporal_analysis", {}).get("normalization_facts", [])
            if isinstance(dataset_profile.get("temporal_analysis"), dict)
            else []
        ),
        "sampling_uncertainty": {
            "is_uncertain_for_column_level_deterministic_inference": sampled_profile_uncertain,
            "sample_rows": sample_rows,
            "estimated_total_rows": total_rows_in_file if total_rows_in_file > 0 else None,
            "row_coverage_ratio": round((sample_rows / total_rows_in_file), 6)
            if sampled_profile_uncertain and total_rows_in_file > 0
            else (1.0 if n_rows > 0 else None),
        },
        "dialect": dataset_profile.get("dialect", {}),
        "pii_findings": dataset_profile.get("pii_findings", {}),
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "converted_from_dataset_profile",
    }

    return data_profile


def _extract_outcome_columns(contract: Dict[str, Any]) -> List[str]:
    """
    Extract outcome columns from contract using V4.1 accessor.

    Priority (handled by get_outcome_columns):
      1. contract["outcome_columns"]
      2. column_roles["outcome"] (supports all V4.1 formats: role->list, column->role, list of dicts)
      3. objective_analysis target fields
      4. Empty list if nothing found

    This is NOT dataset-specific: it's schema-aware extraction.
    """
    try:
        from src.utils.contract_accessors import get_outcome_columns
        outcomes = get_outcome_columns(contract)
        if outcomes:
            return outcomes
    except ImportError:
        pass

    # Legacy fallback if contract accessors are not available
    outcome_cols = []

    # Try outcome_columns first
    raw_outcomes = contract.get("outcome_columns")
    if raw_outcomes:
        if isinstance(raw_outcomes, list):
            outcome_cols = [str(c) for c in raw_outcomes if c and str(c).lower() != "unknown"]
        elif isinstance(raw_outcomes, str) and raw_outcomes.lower() != "unknown":
            outcome_cols = [raw_outcomes]

    # Fallback: column_roles["outcome"]
    if not outcome_cols:
        roles = contract.get("column_roles", {})
        if isinstance(roles, dict):
            # V4.1 format A: role -> list[str]
            outcome_from_roles = roles.get("outcome", [])
            if isinstance(outcome_from_roles, list):
                outcome_cols = [str(c) for c in outcome_from_roles if c]
            elif isinstance(outcome_from_roles, str):
                outcome_cols = [outcome_from_roles]

            # V4.1 format C: column -> role (inverted)
            if not outcome_cols:
                for col, role in roles.items():
                    if isinstance(role, str) and role.lower() == "outcome":
                        outcome_cols.append(str(col))
                    elif isinstance(role, dict) and role.get("role", "").lower() == "outcome":
                        outcome_cols.append(str(col))

    return outcome_cols


def _is_dataset_profile_schema(profile: Dict[str, Any]) -> bool:
    """Detect if profile is in dataset_profile schema (has rows/missing_frac) vs data_profile schema."""
    # dataset_profile has: rows, cols, missing_frac, cardinality
    # data_profile has: basic_stats, missingness_top30, outcome_analysis
    has_ds_keys = "rows" in profile and "missing_frac" in profile
    has_dp_keys = "basic_stats" in profile and "outcome_analysis" in profile
    return has_ds_keys and not has_dp_keys


def _compact_temporal_normalization_facts(
    facts: Any,
    max_items: int = 12,
) -> List[Dict[str, Any]]:
    if not isinstance(facts, list):
        return []
    compacted: List[Dict[str, Any]] = []
    for item in facts:
        if not isinstance(item, dict):
            continue
        column = str(item.get("column") or "").strip()
        if not column:
            continue
        payload: Dict[str, Any] = {
            "column": column,
            "parse_ratio": item.get("parse_ratio"),
            "raw_unique_count": item.get("raw_unique_count"),
            "raw_format_families": item.get("raw_format_families") or [],
            "has_mixed_raw_formats": bool(item.get("has_mixed_raw_formats")),
            "has_time_or_timezone_component": bool(item.get("has_time_or_timezone_component")),
            "canonical_unique_counts": item.get("canonical_unique_counts") or {},
            "raw_to_canonical_unique_ratios": item.get("raw_to_canonical_unique_ratios") or {},
            "semantic_granularity_hints": item.get("semantic_granularity_hints") or [],
            "normalization_collapse_risk": item.get("normalization_collapse_risk") or "unknown",
            "gate_policy": "never copy raw_unique_count into cleaned-artifact expected_unique_range",
        }
        if item.get("time_span_days") is not None:
            payload["time_span_days"] = item.get("time_span_days")
        compacted.append(payload)
        if len(compacted) >= max_items:
            break
    return compacted


def compact_data_profile_for_llm(
    profile: Dict[str, Any],
    max_cols: int = 60,
    contract: Dict[str, Any] | None = None,
    analysis_type: str | None = None,
) -> Dict[str, Any]:
    """
    Compact the data profile for LLM consumption.
    Retains critical decision-making facts while reducing token usage.

    Accepts either:
    - data_profile schema (basic_stats, outcome_analysis, etc.)
    - dataset_profile schema (rows, missing_frac, cardinality) - auto-converts

    Args:
        profile: Data profile or dataset profile dict
        max_cols: Max columns before summarizing dtypes
        contract: Optional contract for conversion (needed if profile is dataset_profile schema)
        analysis_type: Optional analysis type for conversion

    Returns:
        Compacted profile suitable for LLM prompts
    """
    if not isinstance(profile, dict):
        return {}

    # Auto-detect and convert dataset_profile schema if needed
    if _is_dataset_profile_schema(profile):
        if contract is None:
            # Can't convert without contract, return minimal
            return {
                "basic_stats": {"n_rows": profile.get("rows", 0), "n_cols": profile.get("cols", 0)},
                "outcome_analysis": {},
                "split_candidates": [],
                "leakage_flags": [],
                "missingness_top30": dict(list(profile.get("missing_frac", {}).items())[:30]),
                "temporal_normalization_facts": profile.get("temporal_normalization_facts", [])
                if isinstance(profile.get("temporal_normalization_facts"), list)
                else (
                    profile.get("temporal_analysis", {}).get("normalization_facts", [])
                    if isinstance(profile.get("temporal_analysis"), dict)
                    else []
                ),
                "_warning": "dataset_profile detected but no contract provided for conversion",
            }
        profile = convert_dataset_profile_to_data_profile(profile, contract, analysis_type)

    compact = {}

    # 1. Basic Stats (Critical)
    compact["basic_stats"] = _compact_basic_stats_for_llm(
        profile.get("basic_stats", {}),
        max_cols=max_cols,
    )

    # 2. Outcome Analysis (Critical)
    compact["outcome_analysis"] = profile.get("outcome_analysis", {})

    # 3. Split Candidates (Critical for training rows policy)
    compact["split_candidates"] = profile.get("split_candidates", [])

    # 4. Leakage Flags (Critical for leakage policy)
    compact["leakage_flags"] = profile.get("leakage_flags", [])

    # 5. Missingness (Top 30 only)
    compact["missingness_top30"] = profile.get("missingness_top30", {})

    # 6. Constant columns (useful for feature exclusion)
    compact["constant_columns"] = profile.get("constant_columns", [])

    # 7. High cardinality columns (useful for ID detection)
    compact["high_cardinality_columns"] = profile.get("high_cardinality_columns", [])

    # 7.5 Duplicate stats (useful for data quality checks)
    compact["duplicate_stats"] = profile.get("duplicate_stats", {})

    # 7.6 Sampling metadata (helps interpret profile fidelity)
    compact["sampling"] = profile.get("sampling", {})

    # 7.7 Numeric range digest (compact semantic guardrails for planner)
    numeric_ranges_digest = _build_numeric_ranges_digest(profile, max_examples_per_bucket=4)
    if numeric_ranges_digest:
        compact["numeric_ranges_digest"] = numeric_ranges_digest

    # 7.8 Target candidates (contextual evidence, not rules)
    target_candidates = _build_target_candidates(profile)
    if target_candidates:
        compact["target_candidates"] = target_candidates

    # 7.9 Temporal normalization facts (contract-gate safety context)
    temporal_facts = _compact_temporal_normalization_facts(
        profile.get("temporal_normalization_facts")
    )
    if temporal_facts:
        compact["temporal_normalization_facts"] = temporal_facts

    # 8. Column DTypes - Simplify
    dtypes = profile.get("dtypes", {})
    if len(dtypes) > max_cols:
        # Too many columns, summarize
        type_counts = {}
        for col, dtype in dtypes.items():
            t = str(dtype)
            type_counts[t] = type_counts.get(t, 0) + 1
        compact["dtypes_summary"] = type_counts
        compact["dtypes_note"] = f"Total {len(dtypes)} columns. Showing only summary."
    else:
        compact["dtypes"] = dtypes

    return compact


def build_column_metadata_for_strategist(
    data_profile: Dict[str, Any],
    dataset_semantics: Dict[str, Any],
    max_cols: int = 60,
) -> Dict[str, Any]:
    """
    Build a compact column_metadata dict for the Strategist agent.

    Assembles explicit target, column types, per-column statistics, and
    quality flags from already-computed data_profile + dataset_semantics.
    """
    data_profile = data_profile if isinstance(data_profile, dict) else {}
    dataset_semantics = dataset_semantics if isinstance(dataset_semantics, dict) else {}

    columns: List[str] = []
    basic_stats = data_profile.get("basic_stats")
    if isinstance(basic_stats, dict):
        columns = list(basic_stats.get("columns") or [])

    dtypes = data_profile.get("dtypes") or {}
    missingness = data_profile.get("missingness") or {}
    cardinality = data_profile.get("cardinality") or {}
    numeric_summary = data_profile.get("numeric_summary") or {}
    outcome_analysis = data_profile.get("outcome_analysis") or {}

    # --- 1. Target ---
    primary_target = str(dataset_semantics.get("primary_target") or "").strip()
    target_columns = [
        str(col).strip()
        for col in (dataset_semantics.get("target_columns") or dataset_semantics.get("primary_targets") or [])
        if str(col).strip()
    ]
    if primary_target and primary_target not in target_columns:
        target_columns.insert(0, primary_target)
    target_block: Dict[str, Any] = {}
    if primary_target:
        target_block["column"] = primary_target
    if target_columns:
        target_block["columns"] = target_columns[:8]
    if primary_target:
        oa = outcome_analysis.get(primary_target)
        if isinstance(oa, dict):
            inferred = oa.get("inferred_type")
            if inferred:
                target_block["inferred_type"] = str(inferred)
            n_unique = oa.get("n_unique")
            if n_unique is not None:
                target_block["n_unique"] = n_unique
        miss = missingness.get(primary_target)
        if miss is not None:
            target_block["missing_rate"] = round(float(miss), 4)

    # --- 2. Special columns ---
    id_candidates = list(dataset_semantics.get("id_candidates") or [])
    split_cols: List[str] = []
    for sc in (data_profile.get("split_candidates") or []):
        if isinstance(sc, dict):
            col = sc.get("column")
            if col:
                split_cols.append(str(col))
        elif isinstance(sc, str):
            split_cols.append(sc)

    special = set(id_candidates) | set(split_cols) | set(target_columns)

    # --- 3. Column types (grouped) ---
    _LOW_CARDINALITY_THRESHOLD = 15
    type_groups: Dict[str, List[str]] = {
        "numeric": [],
        "low_cardinality": [],
        "high_cardinality": [],
        "constant": [],
    }
    for col in columns:
        if col in special:
            continue
        card_info = cardinality.get(col)
        n_unique = card_info.get("unique") if isinstance(card_info, dict) else None
        if n_unique is not None:
            if n_unique <= 1:
                type_groups["constant"].append(col)
            elif n_unique <= _LOW_CARDINALITY_THRESHOLD:
                type_groups["low_cardinality"].append(col)
            else:
                type_groups["numeric"].append(col)
        else:
            type_groups["numeric"].append(col)

    # Remove empty groups
    type_groups = {k: v for k, v in type_groups.items() if v}

    # --- 4. Per-column stats (capped) ---
    priority_cols: List[str] = []
    for target_col in target_columns:
        if target_col in columns and target_col not in priority_cols:
            priority_cols.append(target_col)
    remaining = [c for c in columns if c not in set(target_columns)]
    remaining.sort(key=lambda c: (-float(missingness.get(c, 0)), c))
    priority_cols.extend(remaining)
    priority_cols = priority_cols[:max_cols]

    col_stats: List[Dict[str, Any]] = []
    for col in priority_cols:
        entry: Dict[str, Any] = {"name": col, "dtype": str(dtypes.get(col, "unknown"))}
        miss = missingness.get(col)
        if miss is not None:
            entry["missing_rate"] = round(float(miss), 4)
        card_info = cardinality.get(col)
        if isinstance(card_info, dict):
            n_unique = card_info.get("unique")
            if n_unique is not None:
                entry["cardinality"] = n_unique
            if isinstance(n_unique, int) and n_unique <= _LOW_CARDINALITY_THRESHOLD:
                top_vals = card_info.get("top_values") or []
                entry["top_values"] = [
                    str(tv.get("value", "")) for tv in top_vals[:5]
                    if isinstance(tv, dict)
                ]
        num = numeric_summary.get(col)
        if isinstance(num, dict) and num:
            for k in ("min", "max", "mean", "std"):
                v = num.get(k)
                if v is not None:
                    try:
                        entry[k] = round(float(v), 2)
                    except (TypeError, ValueError):
                        pass
        col_stats.append(entry)

    # --- 5. Quality flags ---
    quality_flags: Dict[str, Any] = {}
    high_card = data_profile.get("high_cardinality_columns") or []
    if isinstance(high_card, list) and high_card:
        quality_flags["high_cardinality"] = [
            str(item.get("column", item)) if isinstance(item, dict) else str(item)
            for item in high_card
        ]
    constant_cols = data_profile.get("constant_columns") or []
    if isinstance(constant_cols, list) and constant_cols:
        quality_flags["constant"] = constant_cols
    high_missing = [
        {"column": col, "missing_rate": round(float(frac), 4)}
        for col, frac in sorted(missingness.items(), key=lambda x: -float(x[1]))
        if float(frac) > 0.3
    ][:15]
    if high_missing:
        quality_flags["high_missing"] = high_missing

    return {
        "target": target_block,
        "split_column": split_cols[0] if split_cols else None,
        "id_columns": id_candidates or [],
        "column_types": type_groups,
        "column_stats": col_stats,
        "quality_flags": quality_flags,
    }


def _compact_basic_stats_for_llm(basic_stats: Dict[str, Any], max_cols: int = 60) -> Dict[str, Any]:
    """Compact basic_stats to prevent prompt bloat on wide datasets."""
    if not isinstance(basic_stats, dict):
        return {}

    compact_basic: Dict[str, Any] = {
        "n_rows": basic_stats.get("n_rows"),
        "n_cols": basic_stats.get("n_cols"),
    }

    columns = basic_stats.get("columns")
    if not isinstance(columns, list):
        return compact_basic

    col_count = len(columns)
    compact_basic["columns_count"] = col_count
    if col_count <= max_cols:
        compact_basic["columns"] = columns
        return compact_basic

    # Keep deterministic head/tail samples for auditability while staying token-efficient.
    head_count = min(20, col_count)
    tail_count = min(8, max(0, col_count - head_count))
    compact_basic["columns_head"] = columns[:head_count]
    compact_basic["columns_tail"] = columns[-tail_count:] if tail_count > 0 else []
    compact_basic["columns_note"] = (
        f"Column list compacted for LLM context: showing {head_count} head + {tail_count} tail "
        f"out of {col_count} columns."
    )
    return compact_basic


def _build_numeric_ranges_digest(
    profile: Dict[str, Any],
    max_examples_per_bucket: int = 4,
) -> Dict[str, Any]:
    """
    Build compact numeric-range evidence for planner semantic decisions.

    Purpose:
    - Provide range evidence without dumping full numeric_summary.
    - Help planner choose dtype/scaling directives that are compatible with observed data.
    """
    if not isinstance(profile, dict):
        return {}

    numeric_summary = profile.get("numeric_summary")
    if not isinstance(numeric_summary, dict) or not numeric_summary:
        return {}

    buckets: Dict[str, List[str]] = {
        "constant": [],
        "normalized_0_1": [],
        "normalized_neg1_1": [],
        "byte_0_255": [],
        "nonnegative_wide": [],
        "mixed_sign": [],
        "negative_only": [],
    }
    int8_unsafe: List[str] = []

    for col, stats in numeric_summary.items():
        if not isinstance(stats, dict):
            continue
        min_val = stats.get("min")
        max_val = stats.get("max")
        if min_val is None or max_val is None:
            continue
        try:
            lo = float(min_val)
            hi = float(max_val)
        except (TypeError, ValueError):
            continue

        if lo < -128.0 or hi > 127.0:
            int8_unsafe.append(str(col))

        if lo == hi:
            buckets["constant"].append(str(col))
        elif lo >= 0.0 and hi <= 1.01:
            buckets["normalized_0_1"].append(str(col))
        elif lo >= -1.01 and hi <= 1.01:
            buckets["normalized_neg1_1"].append(str(col))
        elif lo >= 0.0 and hi <= 255.5:
            buckets["byte_0_255"].append(str(col))
        elif lo >= 0.0:
            buckets["nonnegative_wide"].append(str(col))
        elif lo < 0.0 and hi > 0.0:
            buckets["mixed_sign"].append(str(col))
        else:
            buckets["negative_only"].append(str(col))

    total_with_ranges = sum(len(v) for v in buckets.values())
    if total_with_ranges == 0:
        return {}

    range_buckets_payload: List[Dict[str, Any]] = []
    for bucket_name in (
        "normalized_0_1",
        "normalized_neg1_1",
        "byte_0_255",
        "nonnegative_wide",
        "mixed_sign",
        "negative_only",
        "constant",
    ):
        cols = buckets.get(bucket_name, [])
        if not cols:
            continue
        range_buckets_payload.append(
            {
                "bucket": bucket_name,
                "count": len(cols),
                "examples": cols[:max_examples_per_bucket],
            }
        )

    digest: Dict[str, Any] = {
        "numeric_columns_with_observed_ranges": total_with_ranges,
        "range_buckets": range_buckets_payload,
        "dtype_guardrails": {
            "signed_int8_unsafe_count": len(int8_unsafe),
            "signed_int8_unsafe_examples": int8_unsafe[:max_examples_per_bucket],
            "conflict_rule": (
                "If strategy wording suggests a dtype that conflicts with observed min/max ranges, "
                "prioritize observed range evidence."
            ),
        },
    }

    # Useful compact signal for mixed-scale datasets.
    has_norm = bool(buckets["normalized_0_1"] or buckets["normalized_neg1_1"])
    has_large = bool(buckets["byte_0_255"] or buckets["nonnegative_wide"] or buckets["mixed_sign"])
    if has_norm and has_large:
        digest["scaling_guardrail"] = (
            "Mixed feature scales detected. If scaling is requested, declare exact scale_columns and avoid "
            "rescaling already normalized columns."
        )

    return digest


def _build_target_candidates(profile: Dict[str, Any], max_candidates: int = 8) -> List[Dict[str, Any]]:
    basic_stats = profile.get("basic_stats", {}) if isinstance(profile.get("basic_stats"), dict) else {}
    columns = basic_stats.get("columns") or []
    if not isinstance(columns, list) or not columns:
        return []

    n_rows = int(basic_stats.get("n_rows") or 0)
    missingness = profile.get("missingness", {}) if isinstance(profile.get("missingness"), dict) else {}
    cardinality = profile.get("cardinality", {}) if isinstance(profile.get("cardinality"), dict) else {}
    dtypes = profile.get("dtypes", {}) if isinstance(profile.get("dtypes"), dict) else {}
    high_card = set()
    high_card_cols = profile.get("high_cardinality_columns")
    if isinstance(high_card_cols, list):
        for item in high_card_cols:
            if isinstance(item, dict) and item.get("column"):
                high_card.add(str(item.get("column")))

    candidates: List[Dict[str, Any]] = []

    for col in columns:
        name = str(col)
        name_lower = name.lower()
        tokens = {tok for tok in name_lower.replace("-", "_").split("_") if tok}
        score = 0.0
        evidence: List[str] = []

        if name_lower in TARGET_TOKEN_HINTS:
            score += 3.0
            evidence.append("name_exact_target_token")
        if tokens & TARGET_TOKEN_HINTS:
            score += 1.5
            evidence.append("name_contains_target_token")
        if "label" in name_lower or "target" in name_lower:
            score += 1.0
            evidence.append("name_contains_label_target")

        id_like = False
        if tokens & ID_TOKEN_HINTS or name_lower in ID_TOKEN_HINTS:
            id_like = True
            score -= 2.0
            evidence.append("name_id_like")

        card_info = cardinality.get(name, {}) if isinstance(cardinality, dict) else {}
        n_unique = card_info.get("unique")
        if isinstance(n_unique, int):
            if n_unique <= 20:
                score += 1.0
                evidence.append("low_cardinality")
            elif n_rows and n_unique < max(20, int(0.5 * n_rows)):
                score += 0.5
                evidence.append("medium_cardinality")
            elif n_rows and n_unique >= int(0.9 * n_rows):
                score -= 1.0
                evidence.append("near_unique")

        null_frac = missingness.get(name)
        if isinstance(null_frac, (int, float)):
            if 0 < null_frac < 0.9:
                score += 0.5
                evidence.append("partial_missingness")

        if name in high_card:
            score -= 0.5
            evidence.append("high_cardinality")

        if score <= 0 and not evidence:
            continue

        unique_ratio = None
        if isinstance(n_unique, int) and n_rows:
            unique_ratio = round(n_unique / n_rows, 4)

        candidates.append(
            {
                "column": name,
                "score": round(score, 2),
                "null_frac": round(float(null_frac), 4) if isinstance(null_frac, (int, float)) else None,
                "n_unique": n_unique if isinstance(n_unique, int) else None,
                "unique_ratio": unique_ratio,
                "dtype_hint": dtypes.get(name),
                "evidence": evidence,
                "id_like_name": id_like,
            }
        )

    candidates.sort(key=lambda x: (x.get("score") or 0), reverse=True)
    return candidates[:max_candidates]


def build_numeric_ranges_summary(dataset_profile: Dict[str, Any], max_examples: int = 10) -> str:
    """
    Build a compact summary of numeric column ranges for Data Engineer context.

    This helps the DE understand the actual scale of the data without making
    assumptions (e.g., pixels could be 0-255 OR already normalized 0-1).

    Groups columns by similar range patterns to keep output concise.

    Args:
        dataset_profile: The dataset_profile.json content (with numeric_summary)
        max_examples: Max example columns to show per range group

    Returns:
        A formatted string summary for LLM context
    """
    numeric_summary = dataset_profile.get("numeric_summary", {})
    if not numeric_summary:
        return ""

    # Categorize columns by their range patterns
    range_groups: Dict[str, List[Dict[str, Any]]] = {
        "normalized_0_1": [],      # min >= 0, max <= 1
        "normalized_neg1_1": [],   # min >= -1, max <= 1
        "percent_0_100": [],       # min >= 0, max <= 100
        "pixel_0_255": [],         # min >= 0, max <= 255 (typical image)
        "unbounded_positive": [],  # min >= 0, no clear upper bound
        "unbounded_mixed": [],     # negative and positive values
        "constant": [],            # min == max
        "other": [],
    }

    for col, stats in numeric_summary.items():
        if not isinstance(stats, dict):
            continue

        min_val = stats.get("min")
        max_val = stats.get("max")

        if min_val is None or max_val is None:
            continue

        try:
            min_val = float(min_val)
            max_val = float(max_val)
        except (TypeError, ValueError):
            continue

        col_info = {
            "col": col,
            "min": min_val,
            "max": max_val,
            "mean": stats.get("mean"),
            "std": stats.get("std"),
        }

        # Categorize based on range
        if min_val == max_val:
            range_groups["constant"].append(col_info)
        elif min_val >= 0 and max_val <= 1.01:
            range_groups["normalized_0_1"].append(col_info)
        elif min_val >= -1.01 and max_val <= 1.01:
            range_groups["normalized_neg1_1"].append(col_info)
        elif min_val >= 0 and max_val <= 100.5:
            range_groups["percent_0_100"].append(col_info)
        elif min_val >= 0 and max_val <= 255.5:
            range_groups["pixel_0_255"].append(col_info)
        elif min_val >= 0:
            range_groups["unbounded_positive"].append(col_info)
        else:
            range_groups["unbounded_mixed"].append(col_info)

    # Build summary text
    lines = ["NUMERIC_RANGES_SUMMARY (from data profile):"]

    # Define friendly descriptions for each group
    group_descriptions = {
        "normalized_0_1": "Already normalized [0, 1]",
        "normalized_neg1_1": "Already normalized [-1, 1]",
        "percent_0_100": "Percent-like scale [0, 100]",
        "pixel_0_255": "Pixel/byte scale [0, 255]",
        "unbounded_positive": "Positive values (unbounded)",
        "unbounded_mixed": "Mixed sign values",
        "constant": "Constant columns (min=max)",
    }

    for group_key, description in group_descriptions.items():
        cols = range_groups.get(group_key, [])
        if not cols:
            continue

        count = len(cols)
        lines.append(f"\n  {description}: {count} columns")

        # Show example columns with their actual ranges
        examples = cols[:max_examples]
        for c in examples:
            mean_str = f", mean={c['mean']:.3f}" if c.get('mean') is not None else ""
            lines.append(f"    - {c['col']}: [{c['min']:.2f}, {c['max']:.2f}]{mean_str}")

        if count > max_examples:
            lines.append(f"    ... and {count - max_examples} more")

    # Add important note for the Data Engineer
    if range_groups["normalized_0_1"]:
        lines.append("\n  NOTE: Some columns are ALREADY in [0,1] range - DO NOT rescale these.")
    if range_groups["pixel_0_255"]:
        lines.append("\n  NOTE: Some columns appear to be in [0,255] range - check if normalization is needed.")

    return "\n".join(lines)
