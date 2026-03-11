import fnmatch
import re
from typing import Any, Dict, List, Set, Tuple


def _normalize_nonempty_str_list(value: Any) -> Tuple[List[str], List[Any]]:
    if value is None:
        return [], []
    if isinstance(value, str):
        text = value.strip()
        return ([text] if text else []), []
    if not isinstance(value, list):
        return [], [value]
    items: List[str] = []
    invalid: List[Any] = []
    for entry in value:
        if isinstance(entry, str):
            text = entry.strip()
            if text:
                items.append(text)
        else:
            invalid.append(entry)
    return list(dict.fromkeys(items)), invalid


def _canonical_reason_token(reason: str) -> str:
    token = re.sub(r"[^0-9a-zA-Z]+", "_", str(reason or "").strip().lower()).strip("_")
    if not token:
        return ""
    if token in {"constant", "zero_variance", "no_variance", "single_value", "single_unique"}:
        return "constant"
    if token in {"all_null", "null_only", "all_missing", "empty_only"}:
        return "all_null"
    if token in {"duplicate", "duplicates", "deduplicate", "dedup"}:
        return "duplicate"
    if token in {"high_null", "high_missing", "missing_threshold", "high_null_fraction"}:
        return "high_null_fraction"
    if token in {"low_information", "near_constant", "low_variance"}:
        return "low_information"
    return token


def _infer_drop_reasons_from_text(*values: Any) -> List[str]:
    text = " ".join(str(value) for value in values if value not in (None, ""))
    normalized = re.sub(r"[^0-9a-zA-Z]+", " ", text).lower()
    reasons: List[str] = []
    if (
        "constant" in normalized
        or "zero variance" in normalized
        or "no variance" in normalized
        or "std dev 0" in normalized
        or "stddev 0" in normalized
        or "variance 0" in normalized
    ):
        reasons.append("constant")
    if "all null" in normalized or "null only" in normalized or "all missing" in normalized:
        reasons.append("all_null")
    if "duplicate" in normalized or "dedup" in normalized:
        reasons.append("duplicate")
    if "high missing" in normalized or "high null" in normalized or "missing threshold" in normalized:
        reasons.append("high_null_fraction")
    if "low information" in normalized or "near constant" in normalized or "low variance" in normalized:
        reasons.append("low_information")
    return list(dict.fromkeys(reasons))


def _normalize_selector_ref_token(value: Any) -> str:
    token = str(value or "").strip().lower()
    token = re.sub(r"[^a-z0-9]+", "_", token).strip("_")
    return token


def _looks_like_selector_reference(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    token = value.strip()
    if not token:
        return False
    low = token.lower()
    if low.startswith(("regex:", "pattern:", "prefix:", "suffix:", "contains:", "selector:")):
        return True
    if "*" in token or "?" in token:
        return True
    if token.startswith("^") or token.endswith("$"):
        return True
    if "\\d" in token or "\\w" in token or "\\s" in token:
        return True
    if any(ch in token for ch in ("[", "]", "(", ")", "{", "}", "|", "+")):
        return True
    if low.endswith(("_features", "_feature_set", "_family")):
        return True
    if low in {"features", "feature_set", "model_features", "all_features"}:
        return True
    return False


def selector_reference_matches_any(
    reference: Any,
    selectors: Any,
) -> bool:
    """
    Determine whether a semantic selector reference token is covered by declared selector objects.

    This allows planner/validator to treat compact family refs (e.g. regex/prefix/selector:name)
    as valid communication between agents on wide schemas.
    """
    ref = str(reference or "").strip()
    if not ref:
        return False
    if not isinstance(selectors, list):
        return False
    selector_dicts = [item for item in selectors if isinstance(item, dict)]
    if not selector_dicts:
        return False

    low_ref = ref.lower()
    norm_ref = _normalize_selector_ref_token(ref)

    # selector:<name|id|family|role> explicit indirection
    if low_ref.startswith("selector:"):
        selector_key = _normalize_selector_ref_token(ref.split(":", 1)[1])
        if not selector_key:
            return False
        for selector in selector_dicts:
            for key in ("name", "id", "family", "role", "selector_hint"):
                if _normalize_selector_ref_token(selector.get(key)) == selector_key:
                    return True
        return False

    # Generic family aliases can map to the declared selector family when present.
    if low_ref.endswith(("_features", "_feature_set", "_family")) or low_ref in {
        "features",
        "feature_set",
        "model_features",
        "all_features",
    }:
        return True

    # Inline selector declarations (regex:/prefix:/suffix:/contains:)
    for prefix, selector_type, selector_key in (
        ("regex:", "regex", "pattern"),
        ("pattern:", "regex", "pattern"),
        ("prefix:", "prefix", "prefix"),
        ("suffix:", "suffix", "suffix"),
        ("contains:", "contains", "value"),
    ):
        if low_ref.startswith(prefix):
            payload = ref[len(prefix) :].strip()
            if not payload:
                return False
            for selector in selector_dicts:
                sel_type = str(selector.get("type") or "").strip().lower()
                if sel_type == "pattern":
                    sel_type = "regex"
                if sel_type != selector_type:
                    continue
                if str(selector.get(selector_key) or "").strip() == payload:
                    return True
            return False

    # Wildcard token can be matched against an equivalent regex selector.
    if "*" in ref or "?" in ref:
        wildcard_regex = fnmatch.translate(ref)
        for selector in selector_dicts:
            sel_type = str(selector.get("type") or "").strip().lower()
            if sel_type in {"regex", "pattern"}:
                pattern = str(selector.get("pattern") or "").strip()
                if pattern and pattern == wildcard_regex:
                    return True

    # Raw regex-like token support (e.g., ^pixel\\d+$).
    if _looks_like_selector_reference(ref):
        for selector in selector_dicts:
            sel_type = str(selector.get("type") or "").strip().lower()
            if sel_type in {"regex", "pattern"}:
                if str(selector.get("pattern") or "").strip() == ref:
                    return True

    # Name/family/role direct references without selector: prefix.
    for selector in selector_dicts:
        for key in ("name", "id", "family", "role"):
            if _normalize_selector_ref_token(selector.get(key)) == norm_ref:
                return True

    return False


def expand_required_feature_selectors(
    selectors: Any,
    candidate_columns: List[str],
) -> Tuple[List[str], List[str]]:
    if selectors is None:
        return [], []
    if not isinstance(selectors, list):
        return [], ["required_feature_selectors must be a list when present."]
    if not candidate_columns:
        return [], []

    candidates = [str(col) for col in candidate_columns if isinstance(col, str) and col.strip()]
    candidate_set = set(candidates)
    expanded: List[str] = []
    issues: List[str] = []

    def _add_many(values: List[str]) -> None:
        for value in values:
            if value in candidate_set and value not in expanded:
                expanded.append(value)

    for idx, selector in enumerate(selectors):
        if not isinstance(selector, dict):
            issues.append(f"required_feature_selectors[{idx}] must be an object.")
            continue
        selector_type = str(selector.get("type") or "").strip().lower()
        if not selector_type:
            issues.append(f"required_feature_selectors[{idx}] missing selector type.")
            continue

        try:
            if selector_type in {"regex", "pattern"}:
                pattern = str(selector.get("pattern") or "").strip()
                if not pattern:
                    issues.append(f"required_feature_selectors[{idx}] missing regex pattern.")
                    continue
                regex = re.compile(pattern, flags=re.IGNORECASE)
                _add_many([col for col in candidates if regex.match(col)])
                continue

            if selector_type == "prefix":
                prefix = str(selector.get("value") or selector.get("prefix") or "").strip()
                if not prefix:
                    issues.append(f"required_feature_selectors[{idx}] missing prefix value.")
                    continue
                _add_many([col for col in candidates if col.lower().startswith(prefix.lower())])
                continue

            if selector_type == "suffix":
                suffix = str(selector.get("value") or selector.get("suffix") or "").strip()
                if not suffix:
                    issues.append(f"required_feature_selectors[{idx}] missing suffix value.")
                    continue
                _add_many([col for col in candidates if col.lower().endswith(suffix.lower())])
                continue

            if selector_type == "contains":
                token = str(selector.get("value") or "").strip()
                if not token:
                    issues.append(f"required_feature_selectors[{idx}] missing contains value.")
                    continue
                _add_many([col for col in candidates if token.lower() in col.lower()])
                continue

            if selector_type == "list":
                values, invalid_values = _normalize_nonempty_str_list(selector.get("columns"))
                if invalid_values:
                    issues.append(f"required_feature_selectors[{idx}] list.columns must be list[str].")
                _add_many([col for col in values if col in candidate_set])
                continue

            if selector_type in {"all_columns_except", "all_numeric_except"}:
                raw_excluded = selector.get("except_columns")
                if raw_excluded is None:
                    raw_excluded = selector.get("value")
                excluded, invalid_values = _normalize_nonempty_str_list(raw_excluded)
                if invalid_values:
                    issues.append(
                        f"required_feature_selectors[{idx}] {selector_type}.except_columns must be list[str]."
                    )
                excluded_set = {col.lower() for col in excluded}
                _add_many([col for col in candidates if col.lower() not in excluded_set])
                continue

            if selector_type == "prefix_numeric_range":
                prefix = str(selector.get("prefix") or "").strip()
                start = selector.get("start")
                end = selector.get("end")
                if not prefix or not isinstance(start, int) or not isinstance(end, int):
                    issues.append(
                        f"required_feature_selectors[{idx}] prefix_numeric_range requires prefix(str), start(int), end(int)."
                    )
                    continue
                lo = min(start, end)
                hi = max(start, end)
                regex = re.compile(rf"^{re.escape(prefix)}(\d+)$", flags=re.IGNORECASE)
                matched: List[str] = []
                for col in candidates:
                    m = regex.match(col)
                    if not m:
                        continue
                    try:
                        pos = int(m.group(1))
                    except Exception:
                        continue
                    if lo <= pos <= hi:
                        matched.append(col)
                _add_many(matched)
                continue

            issues.append(f"required_feature_selectors[{idx}] unsupported selector type '{selector_type}'.")
        except Exception as sel_err:
            issues.append(f"required_feature_selectors[{idx}] expansion error: {sel_err}")

    return expanded, issues


def extract_selector_drop_reasons(
    column_transformations: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    reasons: List[str] = []
    issues: List[str] = []
    transforms = column_transformations if isinstance(column_transformations, dict) else {}

    def _add_reason(raw: Any) -> None:
        token = _canonical_reason_token(str(raw or ""))
        if token and token not in reasons:
            reasons.append(token)

    def _consume_reason_payload(raw: Any, prefix: str) -> None:
        if raw is None:
            return
        if isinstance(raw, str):
            inferred = _infer_drop_reasons_from_text(raw)
            if inferred:
                for token in inferred:
                    _add_reason(token)
            else:
                _add_reason(raw)
            return
        if isinstance(raw, list):
            for idx, item in enumerate(raw):
                if isinstance(item, str):
                    _add_reason(item)
                    continue
                if isinstance(item, dict):
                    for key in ("reason", "name", "type"):
                        if item.get(key):
                            _add_reason(item.get(key))
                    inferred = _infer_drop_reasons_from_text(
                        item.get("criteria"),
                        item.get("condition"),
                        item.get("rationale"),
                        item.get("description"),
                    )
                    for token in inferred:
                        _add_reason(token)
                    continue
                issues.append(f"{prefix}[{idx}] must be string/object.")
            return
        if isinstance(raw, dict):
            for key in ("reasons", "allowed_reasons", "allow_selector_drops_when"):
                if key in raw:
                    _consume_reason_payload(raw.get(key), f"{prefix}.{key}")
            return
        issues.append(f"{prefix} must be string/list/object when present.")

    drop_policy = transforms.get("drop_policy")
    if drop_policy is not None:
        _consume_reason_payload(drop_policy, "column_transformations.drop_policy")

    for alias in ("allow_selector_drops_when", "allowed_reasons", "drop_reasons"):
        if alias in transforms:
            _consume_reason_payload(transforms.get(alias), f"column_transformations.{alias}")

    feature_engineering = transforms.get("feature_engineering")
    if isinstance(feature_engineering, list):
        for idx, item in enumerate(feature_engineering):
            if not isinstance(item, dict):
                issues.append(f"column_transformations.feature_engineering[{idx}] must be object.")
                continue
            action = str(item.get("action") or "").strip().lower()
            if action not in {"drop", "remove", "exclude"}:
                continue
            explicit_reason = item.get("reason") or item.get("drop_reason")
            if explicit_reason:
                _add_reason(explicit_reason)
            inferred = _infer_drop_reasons_from_text(
                item.get("name"),
                item.get("criteria"),
                item.get("condition"),
                item.get("rationale"),
                item.get("description"),
            )
            for token in inferred:
                _add_reason(token)

    return reasons, issues


def collect_manifest_dropped_columns_by_reason(manifest: Dict[str, Any]) -> Dict[str, List[str]]:
    if not isinstance(manifest, dict):
        return {}

    reason_to_columns: Dict[str, Set[str]] = {}

    def _add(reason: str, column: str) -> None:
        canonical_reason = _canonical_reason_token(reason)
        col = str(column or "").strip()
        if not canonical_reason or not col:
            return
        bucket = reason_to_columns.setdefault(canonical_reason, set())
        bucket.add(col)

    def _consume(reason: str, payload: Any) -> None:
        cols, _ = _normalize_nonempty_str_list(payload)
        for col in cols:
            _add(reason, col)
        if isinstance(payload, dict):
            for key, value in payload.items():
                _consume(key, value)
            return
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    col = item.get("column") or item.get("name")
                    item_reason = item.get("reason") or reason
                    if col:
                        _add(str(item_reason or reason), str(col))

    for key, value in manifest.items():
        key_token = _canonical_reason_token(key)
        if not key_token:
            continue

        if key_token in {"constant_columns_dropped", "dropped_constant_columns"}:
            _consume("constant", value)
            continue
        if key_token in {"all_null_columns_dropped", "dropped_all_null_columns"}:
            _consume("all_null", value)
            continue
        if key_token in {"duplicate_columns_dropped", "dropped_duplicate_columns"}:
            _consume("duplicate", value)
            continue

        if key_token.endswith("_columns_dropped"):
            reason = key_token[: -len("_columns_dropped")]
            _consume(reason, value)
            continue
        if key_token.startswith("dropped_") and key_token.endswith("_columns"):
            reason = key_token[len("dropped_") : -len("_columns")]
            _consume(reason, value)
            continue
        if key_token in {"dropped_columns", "columns_dropped"}:
            _consume("dropped", value)
            continue

    return {reason: sorted(list(columns)) for reason, columns in reason_to_columns.items() if columns}


def resolve_required_columns_for_cleaning(
    required_columns: Any,
    required_feature_selectors: Any,
    candidate_columns: List[str],
    column_transformations: Dict[str, Any] | None,
    manifest: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    base_required, _ = _normalize_nonempty_str_list(required_columns)
    selector_required, selector_issues = expand_required_feature_selectors(
        required_feature_selectors,
        candidate_columns or [],
    )
    transforms = column_transformations if isinstance(column_transformations, dict) else {}
    explicit_drop_columns, _ = _normalize_nonempty_str_list(transforms.get("drop_columns"))
    explicit_drop_norm = {col.lower() for col in explicit_drop_columns if col}

    selector_drop_reasons, policy_issues = extract_selector_drop_reasons(transforms)
    policy_dropped_cols_norm: Set[str] = set()
    policy_dropped_cols: Set[str] = set()
    dropped_by_reason = collect_manifest_dropped_columns_by_reason(manifest or {})
    if selector_drop_reasons and dropped_by_reason:
        for reason in selector_drop_reasons:
            for col in dropped_by_reason.get(reason, []):
                if not col:
                    continue
                policy_dropped_cols.add(col)
                policy_dropped_cols_norm.add(col.lower())

    selector_effective = [
        col
        for col in selector_required
        if col.lower() not in explicit_drop_norm and col.lower() not in policy_dropped_cols_norm
    ]
    merged = list(dict.fromkeys(base_required + selector_effective))

    return {
        "required_columns": merged,
        "base_required_columns": base_required,
        "selector_required_columns": selector_required,
        "selector_required_effective_columns": selector_effective,
        "explicit_drop_columns": explicit_drop_columns,
        "selector_drop_reasons": selector_drop_reasons,
        "policy_dropped_selector_columns": sorted(policy_dropped_cols),
        "selector_issues": selector_issues,
        "policy_issues": policy_issues,
        "manifest_dropped_columns_by_reason": dropped_by_reason,
    }
