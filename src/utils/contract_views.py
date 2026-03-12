from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from src.utils.contract_accessors import (
    get_canonical_columns,
    get_clean_dataset_output_path,
    get_clean_manifest_path,
    get_column_roles,
    get_cleaning_gates,
    get_declared_artifacts,
    get_derived_column_names,
    get_outlier_policy,
    get_outcome_columns,
    get_qa_gates,
    get_reviewer_gates,
    get_required_outputs,
    get_task_semantics,
    get_validation_requirements,
)
from src.utils.problem_capabilities import infer_problem_capabilities, resolve_problem_capabilities_from_contract

# Shared helper for decisioning requirements extraction
def _get_decisioning_requirements(contract_full: Dict[str, Any], contract_min: Dict[str, Any]) -> Dict[str, Any]:
    for source in (contract_min, contract_full):
        if not isinstance(source, dict):
            continue
        decisioning = source.get("decisioning_requirements")
        if isinstance(decisioning, dict) and decisioning:
            return {
                "enabled": bool(decisioning.get("enabled")),
                "required": bool(decisioning.get("required")),
                "output": decisioning.get("output", {}),
                "policy_notes": decisioning.get("policy_notes", ""),
            }
    return {"enabled": False, "required": False, "output": {}, "policy_notes": ""}

# contract_full: traceability/strategy; contract_min: binding; views: prompt context.


class DEView(TypedDict, total=False):
    role: str
    task_semantics: Dict[str, Any]
    required_columns: List[str]
    required_feature_selectors: List[Dict[str, Any]]
    optional_passthrough_columns: List[str]
    column_dtype_targets: Dict[str, Dict[str, Any]]
    column_transformations: Dict[str, Any]
    output_path: str
    output_manifest_path: str
    required_columns_path: str
    column_sets_path: str
    output_dialect: Dict[str, Any]
    cleaning_gates: List[Dict[str, Any]]
    data_engineer_runbook: Any
    outlier_policy: Dict[str, Any]
    outlier_report_path: str
    constraints: Dict[str, Any]


class MLView(TypedDict, total=False):
    role: str
    objective_type: str
    task_semantics: Dict[str, Any]
    canonical_columns: List[str]
    derived_features: List[str]
    column_roles: Dict[str, List[str]]
    decision_columns: List[str]
    outcome_columns: List[str]
    audit_only_columns: List[str]
    identifier_columns: List[str]
    allowed_feature_sets: Dict[str, Any]
    forbidden_features: List[str]
    column_dtype_targets: Dict[str, Dict[str, Any]]
    required_outputs: List[str]
    validation_requirements: Dict[str, Any]
    evaluation_spec: Dict[str, Any]
    objective_analysis: Dict[str, Any]
    qa_gates: List[Dict[str, Any]]
    reviewer_gates: List[Any]
    ml_engineer_runbook: Any
    case_rules: Any
    plot_spec: Dict[str, Any]
    artifact_requirements: Dict[str, Any]
    artifact_paths: Dict[str, str]
    cleaning_manifest_path: str
    cleaned_data_path: str
    outlier_policy: Dict[str, Any]
    split_spec: Dict[str, Any]
    n_train_rows: int
    n_test_rows: int
    n_total_rows: int


class ReviewerView(TypedDict, total=False):
    role: str
    objective_type: str
    task_semantics: Dict[str, Any]
    reviewer_gates: List[Any]
    required_outputs: List[str]
    expected_metrics: List[str]
    strategy_summary: str
    verification: Dict[str, Any]


class TranslatorView(TypedDict, total=False):
    role: str
    reporting_policy: Dict[str, Any]
    plot_spec: Dict[str, Any]
    evidence_inventory: List[Dict[str, Any]]
    key_decisions: List[str]
    limitations: List[str]
    constraints: Dict[str, Any]


class ResultsAdvisorView(TypedDict, total=False):
    role: str
    objective_type: str
    reporting_policy: Dict[str, Any]
    evidence_inventory: List[Dict[str, Any]]


class CleaningView(TypedDict, total=False):
    role: str
    task_semantics: Dict[str, Any]
    strategy_title: str
    business_objective: str
    required_columns: List[str]
    required_feature_selectors: List[Dict[str, Any]]
    column_sets_path: str
    column_transformations: Dict[str, Any]
    dialect: Dict[str, Any]
    cleaning_gates: List[Dict[str, Any]]
    column_roles: Dict[str, List[str]]
    allowed_feature_sets: Dict[str, Any]
    outlier_policy: Dict[str, Any]
    outlier_report_path: str


class QAView(TypedDict, total=False):
    role: str
    task_semantics: Dict[str, Any]
    qa_gates: List[Dict[str, Any]]
    artifact_requirements: Dict[str, Any]
    allowed_feature_sets: Dict[str, Any]
    column_roles: Dict[str, List[str]]
    canonical_columns: List[str]
    objective_summary: Dict[str, str]
    reporting_policy: Dict[str, Any]
    n_train_rows: int
    n_test_rows: int
    n_total_rows: int
    split_spec: Dict[str, Any]


_PRESERVE_KEYS = {
    "task_semantics",
    "required_columns",
    "optional_passthrough_columns",
    "required_feature_selectors",
    "column_dtype_targets",
    "required_outputs",
    "column_transformations",
    "drop_columns",
    "scale_columns",
    "drop_policy",
    "feature_engineering",
    "dtype_conversion",
    "forbidden_features",
    "reviewer_gates",
    "qa_gates",
    "cleaning_gates",
    "data_engineer_runbook",
    "evaluation_spec",
    "objective_analysis",
    "ml_engineer_runbook",
    "gates",
    "plot_spec",
    "plots",
    "outlier_policy",
    "split_spec",
    "n_train_rows",
    "n_test_rows",
    "n_total_rows",
}

_IDENTIFIER_TOKENS = {
    "id",
    "uuid",
    "guid",
    "key",
    "codigo",
    "code",
    "cod",
    "identifier",
    "reference",
    "ref",
    "account",
    "entity",
}
_SHORT_IDENTIFIER_TOKENS = {"id", "cod", "ref", "key"}
_LONG_IDENTIFIER_TOKENS = sorted(_IDENTIFIER_TOKENS - _SHORT_IDENTIFIER_TOKENS)
_STRICT_IDENTIFIER_EXACT = {
    "id",
    "uuid",
    "guid",
    "rowid",
    "recordid",
    "row_id",
    "record_id",
    "index",
    "idx",
    "record",
}
_STRICT_IDENTIFIER_PATTERN = re.compile(r"^(row|record)[ _\-]?id$", re.IGNORECASE)
_CANDIDATE_IDENTIFIER_TOKENS = {"key", "ref", "code", "cod"}
_CANDIDATE_SUFFIXES = ("_id", "-id", " id")
_DEFAULT_DE_OUTLIER_REPORT_PATH = "data/outlier_treatment_report.json"


def _coerce_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _coerce_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _first_value(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _coerce_positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        parsed = int(value)
        return parsed if parsed > 0 else None
    if isinstance(value, str):
        token = value.strip()
        if not token:
            return None
        try:
            parsed = int(float(token))
        except Exception:
            return None
        return parsed if parsed > 0 else None
    return None


def _resolve_row_count_hints(
    contract_full: Dict[str, Any],
    contract_min: Dict[str, Any],
) -> Dict[str, int]:
    hints: Dict[str, int] = {}
    train_keys = ("n_train_rows", "train_rows", "n_train", "rows_train")
    test_keys = ("n_test_rows", "test_rows", "n_test", "rows_test")
    total_keys = ("n_total_rows", "total_rows", "n_rows", "row_count", "rows")

    def _scan(source: Any) -> None:
        if not isinstance(source, dict):
            return
        if "n_train_rows" not in hints:
            for key in train_keys:
                parsed = _coerce_positive_int(source.get(key))
                if parsed is not None:
                    hints["n_train_rows"] = parsed
                    break
        if "n_test_rows" not in hints:
            for key in test_keys:
                parsed = _coerce_positive_int(source.get(key))
                if parsed is not None:
                    hints["n_test_rows"] = parsed
                    break
        if "n_total_rows" not in hints:
            for key in total_keys:
                parsed = _coerce_positive_int(source.get(key))
                if parsed is not None:
                    hints["n_total_rows"] = parsed
                    break
        basic_stats = source.get("basic_stats")
        if isinstance(basic_stats, dict):
            if "n_total_rows" not in hints:
                parsed = _coerce_positive_int(
                    basic_stats.get("n_rows") or basic_stats.get("rows") or basic_stats.get("row_count")
                )
                if parsed is not None:
                    hints["n_total_rows"] = parsed
            if "n_train_rows" not in hints:
                parsed = _coerce_positive_int(
                    basic_stats.get("n_train_rows") or basic_stats.get("train_rows")
                )
                if parsed is not None:
                    hints["n_train_rows"] = parsed
            if "n_test_rows" not in hints:
                parsed = _coerce_positive_int(
                    basic_stats.get("n_test_rows") or basic_stats.get("test_rows")
                )
                if parsed is not None:
                    hints["n_test_rows"] = parsed

    def _coerce_ratio(value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            ratio = float(value)
        elif isinstance(value, str):
            token = value.strip()
            if not token:
                return None
            try:
                ratio = float(token)
            except Exception:
                return None
        else:
            return None
        if ratio < 0.0 or ratio > 1.0:
            return None
        return ratio

    def _scan_outcome_counts(source: Any) -> None:
        if not isinstance(source, dict):
            return
        if (
            "n_train_rows" in hints
            and "n_total_rows" in hints
            and "n_test_rows" in hints
        ):
            return
        outcome_analysis = source.get("outcome_analysis")
        if not isinstance(outcome_analysis, dict):
            return
        for entry in outcome_analysis.values():
            if not isinstance(entry, dict):
                continue
            total = _coerce_positive_int(
                entry.get("total_count")
                or entry.get("n_total_rows")
                or entry.get("n_rows")
                or entry.get("row_count")
                or entry.get("rows")
            )
            non_null = _coerce_positive_int(
                entry.get("non_null_count")
                or entry.get("n_non_null")
                or entry.get("non_null_rows")
                or entry.get("labeled_rows")
                or entry.get("train_rows")
            )
            if non_null is None and isinstance(total, int):
                null_frac = _coerce_ratio(entry.get("null_frac"))
                if null_frac is not None:
                    inferred_non_null = int(round(total * (1.0 - null_frac)))
                    if inferred_non_null > 0 and inferred_non_null <= total:
                        non_null = inferred_non_null
            if "n_total_rows" not in hints and isinstance(total, int):
                hints["n_total_rows"] = total
            if "n_train_rows" not in hints and isinstance(non_null, int):
                hints["n_train_rows"] = non_null
            if (
                "n_test_rows" not in hints
                and isinstance(total, int)
                and isinstance(non_null, int)
                and total >= non_null
            ):
                inferred_test = total - non_null
                if inferred_test > 0:
                    hints["n_test_rows"] = inferred_test
            if (
                "n_train_rows" in hints
                and "n_total_rows" in hints
                and "n_test_rows" in hints
            ):
                return

    for source in (contract_min, contract_full):
        _scan(source)
        _scan_outcome_counts(source)
        if isinstance(source, dict):
            _scan(source.get("dataset_profile"))
            _scan(source.get("data_profile"))
            _scan(source.get("evaluation_spec"))
            _scan(source.get("execution_constraints"))
            _scan(source.get("split_spec"))
            _scan_outcome_counts(source.get("dataset_profile"))
            _scan_outcome_counts(source.get("data_profile"))
            _scan_outcome_counts(source.get("evaluation_spec"))
    if (
        "n_total_rows" not in hints
        and "n_train_rows" in hints
        and "n_test_rows" in hints
    ):
        hints["n_total_rows"] = int(hints["n_train_rows"] + hints["n_test_rows"])
    if (
        "n_test_rows" not in hints
        and "n_total_rows" in hints
        and "n_train_rows" in hints
        and hints["n_total_rows"] >= hints["n_train_rows"]
    ):
        hints["n_test_rows"] = int(hints["n_total_rows"] - hints["n_train_rows"])
    if (
        "n_train_rows" not in hints
        and "n_total_rows" in hints
        and "n_test_rows" in hints
        and hints["n_total_rows"] >= hints["n_test_rows"]
    ):
        hints["n_train_rows"] = int(hints["n_total_rows"] - hints["n_test_rows"])
    return hints


def is_identifier_column(col_name: str) -> bool:
    if not col_name:
        return False
    if is_strict_identifier_column(col_name):
        return True
    if is_candidate_identifier_column(col_name):
        return True
    return False


def is_strict_identifier_column(col_name: str) -> bool:
    if not col_name:
        return False
    raw = str(col_name)
    lowered = raw.lower().replace("-", "_")
    normalized = re.sub(r"[^0-9a-zA-Z_]+", "", lowered)
    if normalized in _STRICT_IDENTIFIER_EXACT:
        return True
    if _STRICT_IDENTIFIER_PATTERN.match(raw):
        return True
    tokens = [t for t in re.split(r"[^0-9a-zA-Z]+", lowered) if t]
    camel_tokens = [t.lower() for t in re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\\d+", col_name) if t]
    if any(token in {"uuid", "guid"} for token in (tokens + camel_tokens)):
        return True
    return False


def is_candidate_identifier_column(col_name: str) -> bool:
    if not col_name:
        return False
    if is_strict_identifier_column(col_name):
        return False
    raw = str(col_name)
    lowered = raw.lower()
    normalized = re.sub(r"[^0-9a-zA-Z_]+", "", lowered)
    if normalized.endswith("_id") or normalized.endswith("-id") or normalized.endswith(" id"):
        return True
    tokens = [t for t in re.split(r"[^0-9a-zA-Z]+", lowered) if t]
    camel_tokens = [t.lower() for t in re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\\d+", col_name) if t]
    if any(token in _CANDIDATE_IDENTIFIER_TOKENS for token in (tokens + camel_tokens)):
        return True
    if re.search(r"[A-Za-z]+Id$", raw):
        return True
    return False


def _resolve_required_outputs(contract_min: Dict[str, Any], contract_full: Dict[str, Any]) -> List[str]:
    merged: List[str] = []
    seen: set[str] = set()

    def _extract_path(item: Any) -> str:
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            for key in ("path", "output", "artifact"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    return value
        return ""

    def _add(values: Any) -> None:
        if not isinstance(values, list):
            return
        for item in values:
            if not item:
                continue
            text = _extract_path(item).replace("\\", "/").strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(text)

    _add(contract_min.get("required_outputs"))
    _add(contract_full.get("required_outputs"))
    # Always widen with accessor-derived outputs to include artifact_requirements.required_files/plots
    # while preserving top-level required_outputs precedence and de-duplication.
    _add(get_required_outputs(contract_full))
    _add(get_required_outputs(contract_min))
    return merged


def _resolve_required_columns(contract_min: Dict[str, Any], contract_full: Dict[str, Any]) -> List[str]:
    artifact_reqs = _coerce_dict(contract_min.get("artifact_requirements")) or _coerce_dict(
        contract_full.get("artifact_requirements")
    )
    clean_cfg = _coerce_dict(artifact_reqs.get("clean_dataset"))
    required = clean_cfg.get("required_columns")
    if isinstance(required, list) and required:
        return [str(c) for c in required if c]
    canonical = contract_min.get("canonical_columns")
    if isinstance(canonical, list) and canonical:
        return [str(c) for c in canonical if c]
    canonical = get_canonical_columns(contract_full)
    if canonical:
        return [str(c) for c in canonical if c]
    return []


def _resolve_required_feature_selectors(contract_min: Dict[str, Any], contract_full: Dict[str, Any]) -> List[Dict[str, Any]]:
    artifact_reqs = _coerce_dict(contract_min.get("artifact_requirements")) or _coerce_dict(
        contract_full.get("artifact_requirements")
    )
    clean_cfg = _coerce_dict(artifact_reqs.get("clean_dataset"))
    selectors = clean_cfg.get("required_feature_selectors")
    if not isinstance(selectors, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for item in selectors:
        if not isinstance(item, dict):
            continue
        payload = dict(item)
        sel_type = payload.get("type")
        if isinstance(sel_type, str):
            payload["type"] = sel_type.strip()
        normalized.append(payload)
    return normalized


def _resolve_passthrough_columns(
    contract_min: Dict[str, Any], contract_full: Dict[str, Any], required_columns: List[str]
) -> List[str]:
    allowed_sets = _resolve_allowed_feature_sets(contract_min, contract_full)
    audit_only = allowed_sets.get("audit_only_features")
    if not isinstance(audit_only, list) or not audit_only:
        return []
    required_set = {str(c) for c in required_columns if c}
    passthrough = [str(c) for c in audit_only if c]
    return [c for c in passthrough if c not in required_set]


def _resolve_output_path(
    contract_min: Dict[str, Any],
    contract_full: Dict[str, Any],
    required_outputs: List[str],
) -> Optional[str]:
    combined = contract_full if isinstance(contract_full, dict) else {}
    if isinstance(contract_min, dict) and contract_min:
        combined = {**combined, **contract_min}
    output_path = get_clean_dataset_output_path(combined)
    if output_path:
        return str(output_path)
    for path in required_outputs:
        lower = str(path).lower()
        if "cleaned" in lower and lower.endswith(".csv"):
            return str(path)
    return None


def _resolve_manifest_path(
    contract_min: Dict[str, Any],
    contract_full: Dict[str, Any],
    required_outputs: List[str],
) -> Optional[str]:
    combined = contract_full if isinstance(contract_full, dict) else {}
    if isinstance(contract_min, dict) and contract_min:
        combined = {**combined, **contract_min}
    manifest_path = get_clean_manifest_path(combined)
    if manifest_path:
        return str(manifest_path)
    for path in required_outputs:
        if "cleaning_manifest" in str(path).lower():
            return str(path)
    return None


def _resolve_output_dialect(contract_min: Dict[str, Any], contract_full: Dict[str, Any]) -> Dict[str, Any]:
    dialect = contract_min.get("output_dialect")
    if isinstance(dialect, dict) and dialect:
        return dialect
    dialect = contract_full.get("output_dialect")
    if isinstance(dialect, dict) and dialect:
        return dialect
    return {}


def _resolve_column_transformations(contract_min: Dict[str, Any], contract_full: Dict[str, Any]) -> Dict[str, Any]:
    artifact_reqs = _coerce_dict(contract_min.get("artifact_requirements")) or _coerce_dict(
        contract_full.get("artifact_requirements")
    )
    clean_cfg = _coerce_dict(artifact_reqs.get("clean_dataset"))
    transforms = clean_cfg.get("column_transformations")
    if not isinstance(transforms, dict):
        transforms = {}

    def _collect(alias_keys: List[str]) -> List[str]:
        values: List[str] = []
        for source in (transforms, clean_cfg):
            for key in alias_keys:
                raw = source.get(key)
                if isinstance(raw, str) and raw.strip():
                    values.append(raw.strip())
                elif isinstance(raw, list):
                    values.extend([str(item).strip() for item in raw if isinstance(item, str) and str(item).strip()])
        return list(dict.fromkeys(values))

    drop_columns = _collect(["drop_columns", "remove_columns", "columns_to_drop", "excluded_columns"])
    scale_columns = _collect(
        ["scale_columns", "normalize_columns", "standardize_columns", "rescale_columns"]
    )
    drop_policy = transforms.get("drop_policy")
    if drop_policy is None and "drop_policy" in clean_cfg:
        drop_policy = clean_cfg.get("drop_policy")
    if not (drop_columns or scale_columns or drop_policy is not None):
        return {}

    payload = dict(transforms)
    payload["drop_columns"] = drop_columns
    payload["scale_columns"] = scale_columns
    if drop_policy is not None:
        payload["drop_policy"] = drop_policy
    return payload


def _resolve_column_dtype_targets(contract_min: Dict[str, Any], contract_full: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    for source in (contract_min, contract_full):
        if not isinstance(source, dict):
            continue
        targets = source.get("column_dtype_targets")
        if isinstance(targets, dict) and targets:
            normalized: Dict[str, Dict[str, Any]] = {}
            for key, value in targets.items():
                col = str(key or "").strip()
                if not col or not isinstance(value, dict):
                    continue
                target_dtype = str(value.get("target_dtype") or "").strip()
                if not target_dtype:
                    continue
                payload = {
                    "target_dtype": target_dtype,
                }
                if "nullable" in value:
                    payload["nullable"] = value.get("nullable")
                if "role" in value:
                    payload["role"] = value.get("role")
                if "source" in value:
                    payload["source"] = value.get("source")
                if "matched_count" in value:
                    payload["matched_count"] = value.get("matched_count")
                if isinstance(value.get("matched_sample"), list) and value.get("matched_sample"):
                    payload["matched_sample"] = [
                        str(item) for item in value.get("matched_sample", []) if str(item).strip()
                    ][:20]
                normalized[col] = payload
            if normalized:
                return normalized

    artifact_reqs = _coerce_dict(contract_min.get("artifact_requirements")) or _coerce_dict(
        contract_full.get("artifact_requirements")
    )
    clean_cfg = _coerce_dict(artifact_reqs.get("clean_dataset"))
    targets = clean_cfg.get("column_dtype_targets")
    if isinstance(targets, dict) and targets:
        normalized: Dict[str, Dict[str, Any]] = {}
        for key, value in targets.items():
            col = str(key or "").strip()
            if not col or not isinstance(value, dict):
                continue
            target_dtype = str(value.get("target_dtype") or "").strip()
            if not target_dtype:
                continue
            payload = {"target_dtype": target_dtype}
            if "nullable" in value:
                payload["nullable"] = value.get("nullable")
            if "role" in value:
                payload["role"] = value.get("role")
            if "source" in value:
                payload["source"] = value.get("source")
            normalized[col] = payload
        return normalized
    return {}


def _normalize_artifact_index(entries: Any) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in entries or []:
        if isinstance(item, dict) and item.get("path"):
            normalized.append(
                {
                    "path": str(item.get("path")),
                    "artifact_type": item.get("artifact_type") or item.get("type"),
                }
            )
        elif isinstance(item, str):
            normalized.append({"path": item, "artifact_type": "artifact"})
    return normalized


def _resolve_objective_type(contract_min: Dict[str, Any], contract_full: Dict[str, Any], required_outputs: List[str]) -> str:
    for source in (contract_min, contract_full):
        task_semantics = get_task_semantics(source if isinstance(source, dict) else {})
        if isinstance(task_semantics, dict):
            for key in ("objective_type", "problem_family"):
                value = str(task_semantics.get(key) or "").strip()
                if value and value.lower() != "unknown":
                    return value
    for source in (contract_full, contract_min):
        capabilities = resolve_problem_capabilities_from_contract(source if isinstance(source, dict) else {})
        family = str(capabilities.get("family") or "").strip()
        if family and family != "unknown":
            return family

    # V4.1: Check objective_analysis.problem_type first (primary source)
    for source in (contract_min, contract_full):
        obj_analysis = source.get("objective_analysis") if isinstance(source, dict) else None
        if isinstance(obj_analysis, dict) and obj_analysis.get("problem_type"):
            return str(obj_analysis.get("problem_type"))

    # Fallback: check evaluation_spec.objective_type (V4.1 evaluation spec)
    for source in (contract_min, contract_full):
        eval_spec = source.get("evaluation_spec") if isinstance(source, dict) else None
        if isinstance(eval_spec, dict):
            obj = eval_spec.get("objective_type")
            if obj:
                return str(obj)

    # V4.1: NO fallback to legacy execution_plan - infer from outputs instead
    return _infer_objective_from_outputs(required_outputs)


def _infer_objective_from_outputs(required_outputs: List[str]) -> str:
    capabilities = infer_problem_capabilities(required_outputs=required_outputs or [])
    family = str(capabilities.get("family") or "").strip()
    if family and family != "unknown":
        return family
    tokens = " ".join([str(p).lower() for p in required_outputs or []])
    if "report" in tokens or "summary" in tokens:
        return "descriptive"
    return "unknown"


def _resolve_column_roles(contract_min: Dict[str, Any], contract_full: Dict[str, Any]) -> Dict[str, List[str]]:
    roles_min = get_column_roles(contract_min)
    if roles_min:
        return roles_min
    return get_column_roles(contract_full)


def _extract_roles(roles: Dict[str, List[str]]) -> Dict[str, List[str]]:
    return {
        "decision": _coerce_list(roles.get("decision")),
        "outcome": _coerce_list(roles.get("outcome")),
        "audit_only": _coerce_list(roles.get("post_decision_audit_only") or roles.get("audit_only")),
    }


def _resolve_derived_columns(contract_min: Dict[str, Any], contract_full: Dict[str, Any]) -> List[str]:
    derived: List[str] = []
    for source in (contract_min, contract_full):
        if not isinstance(source, dict):
            continue
        derived.extend(get_derived_column_names(source))
    return list(dict.fromkeys([str(c) for c in derived if c]))


def _resolve_allowed_feature_sets(contract_min: Dict[str, Any], contract_full: Dict[str, Any]) -> Dict[str, Any]:
    allowed = contract_min.get("allowed_feature_sets")
    if isinstance(allowed, dict) and allowed:
        return allowed
    allowed = contract_full.get("allowed_feature_sets")
    if isinstance(allowed, dict) and allowed:
        return allowed
    return {"segmentation_features": [], "model_features": [], "forbidden_features": []}


def _resolve_forbidden_features(allowed_feature_sets: Dict[str, Any]) -> List[str]:
    forbidden = allowed_feature_sets.get("forbidden_features")
    if isinstance(forbidden, list):
        return [str(c) for c in forbidden if c]
    forbidden = allowed_feature_sets.get("forbidden_for_modeling")
    if isinstance(forbidden, list):
        return [str(c) for c in forbidden if c]
    return []


def _resolve_validation_requirements(contract_min: Dict[str, Any], contract_full: Dict[str, Any]) -> Dict[str, Any]:
    validation = contract_min.get("validation_requirements")
    if isinstance(validation, dict) and validation:
        return validation
    eval_spec = contract_min.get("evaluation_spec") if isinstance(contract_min, dict) else None
    if isinstance(eval_spec, dict):
        validation = eval_spec.get("validation_requirements")
        if isinstance(validation, dict) and validation:
            return validation
    validation = get_validation_requirements(contract_full)
    if validation:
        return validation
    eval_spec = contract_full.get("evaluation_spec") if isinstance(contract_full, dict) else None
    if isinstance(eval_spec, dict):
        validation = eval_spec.get("validation_requirements")
        if isinstance(validation, dict) and validation:
            return validation
    return {}


def _resolve_qa_gates(contract_min: Dict[str, Any], contract_full: Dict[str, Any]) -> List[Dict[str, Any]]:
    gates = get_qa_gates(contract_full)
    if gates:
        return gates
    gates = get_qa_gates(contract_min)
    if gates:
        return gates
    def _normalize(raw: Any) -> List[Dict[str, Any]]:
        if not isinstance(raw, list):
            return []
        normalized: List[Dict[str, Any]] = []
        seen: set[str] = set()
        alias_keys = ("name", "id", "gate", "metric", "check", "rule", "title", "label")
        for gate in raw:
            if isinstance(gate, dict):
                name = ""
                for key in alias_keys:
                    value = gate.get(key)
                    if isinstance(value, str) and value.strip():
                        name = value.strip()
                        break
                if not name:
                    continue
                severity = str(gate.get("severity") or "HARD").upper()
                if severity not in {"HARD", "SOFT"}:
                    severity = "HARD"
                params = gate.get("params")
                if not isinstance(params, dict):
                    params = {}
                for param_key in ("metric", "check", "rule", "threshold", "target", "min", "max", "operator", "direction", "condition"):
                    if param_key in gate and param_key not in params:
                        params[param_key] = gate.get(param_key)
                key = str(name).lower()
                if key in seen:
                    continue
                seen.add(key)
                entry: Dict[str, Any] = {"name": str(name), "severity": severity, "params": params}
                for extra_key in ("condition", "evidence_required", "action_if_fail"):
                    if extra_key in gate:
                        entry[extra_key] = gate.get(extra_key)
                normalized.append(entry)
            elif isinstance(gate, str) and gate.strip():
                key = gate.strip().lower()
                if key in seen:
                    continue
                seen.add(key)
                normalized.append({"name": gate.strip(), "severity": "HARD", "params": {}})
        return normalized
    eval_spec = contract_full.get("evaluation_spec") if isinstance(contract_full, dict) else None
    if isinstance(eval_spec, dict):
        gates = _normalize(eval_spec.get("qa_gates"))
        if gates:
            return gates
    eval_spec = contract_min.get("evaluation_spec") if isinstance(contract_min, dict) else None
    if isinstance(eval_spec, dict):
        gates = _normalize(eval_spec.get("qa_gates"))
        if gates:
            return gates
    return []


def _resolve_cleaning_gates(contract_min: Dict[str, Any], contract_full: Dict[str, Any]) -> List[Dict[str, Any]]:
    gates = get_cleaning_gates(contract_full)
    if gates:
        return gates
    gates = get_cleaning_gates(contract_min)
    if gates:
        return gates
    eval_spec = contract_full.get("evaluation_spec") if isinstance(contract_full, dict) else None
    if isinstance(eval_spec, dict):
        raw = eval_spec.get("cleaning_gates")
        if isinstance(raw, list):
            return [g for g in raw if isinstance(g, dict)]
    eval_spec = contract_min.get("evaluation_spec") if isinstance(contract_min, dict) else None
    if isinstance(eval_spec, dict):
        raw = eval_spec.get("cleaning_gates")
        if isinstance(raw, list):
            return [g for g in raw if isinstance(g, dict)]
    return []


def _resolve_data_engineer_runbook(contract_min: Dict[str, Any], contract_full: Dict[str, Any]) -> Any:
    for source in (contract_full, contract_min):
        if not isinstance(source, dict):
            continue
        runbook = source.get("data_engineer_runbook")
        if isinstance(runbook, dict) and runbook:
            return runbook
        if isinstance(runbook, list) and runbook:
            return runbook
        if isinstance(runbook, str) and runbook.strip():
            return runbook.strip()
    return {}


def _resolve_optional_outputs(contract_min: Dict[str, Any], contract_full: Dict[str, Any]) -> List[str]:
    artifact_reqs = _coerce_dict(contract_min.get("artifact_requirements")) or _coerce_dict(
        contract_full.get("artifact_requirements")
    )
    optional_outputs: List[str] = []
    for key in ("optional_files", "optional_outputs"):
        values = artifact_reqs.get(key)
        if isinstance(values, list):
            optional_outputs.extend([str(v) for v in values if v])
    for value in artifact_reqs.values():
        if not isinstance(value, dict):
            continue
        if value.get("optional") is True:
            path = _first_value(value.get("path"), value.get("output_path"), value.get("output"))
            if path:
                optional_outputs.append(str(path))
            expected = value.get("expected")
            if isinstance(expected, list):
                optional_outputs.extend([str(v) for v in expected if v])
    return list(dict.fromkeys([str(v) for v in optional_outputs if v]))


def _resolve_case_rules(contract_full: Dict[str, Any]) -> Any:
    # V4.1: Only use direct case_rules/case_taxonomy keys, no spec_extraction fallback
    for path in (
        ("case_rules",),
        ("case_taxonomy",),
        ("evaluation_spec", "case_taxonomy"),
    ):
        cursor: Any = contract_full
        for key in path:
            if not isinstance(cursor, dict):
                cursor = None
                break
            cursor = cursor.get(key)
        if cursor:
            return cursor
    return None


def _resolve_outlier_policy(contract_min: Dict[str, Any], contract_full: Dict[str, Any]) -> Dict[str, Any]:
    for source in (contract_min, contract_full):
        if not isinstance(source, dict):
            continue
        policy = get_outlier_policy(source)
        if isinstance(policy, dict) and policy:
            normalized = dict(policy)
            enabled = normalized.get("enabled")
            if isinstance(enabled, str):
                enabled = enabled.strip().lower() in {"1", "true", "yes", "on", "enabled"}
            if enabled is None:
                enabled = bool(
                    normalized.get("target_columns")
                    or normalized.get("methods")
                    or normalized.get("treatment")
                )
            normalized["enabled"] = bool(enabled)
            stage = str(normalized.get("apply_stage") or "data_engineer").strip().lower()
            if stage not in {"data_engineer", "ml_engineer", "both"}:
                stage = "data_engineer"
            normalized["apply_stage"] = stage
            report_path = normalized.get("report_path") or normalized.get("output_path")
            if isinstance(report_path, str) and report_path.strip():
                normalized["report_path"] = report_path.strip()
            elif bool(enabled) and stage in {"data_engineer", "both"}:
                normalized["report_path"] = _DEFAULT_DE_OUTLIER_REPORT_PATH
            target_columns = normalized.get("target_columns")
            if isinstance(target_columns, list):
                normalized["target_columns"] = [str(col) for col in target_columns if col]
            strict = normalized.get("strict")
            if isinstance(strict, str):
                strict = strict.strip().lower() in {"1", "true", "yes", "on", "required"}
            if strict is not None:
                normalized["strict"] = bool(strict)
            return normalized
    return {}


def _resolve_reviewer_gates(contract_min: Dict[str, Any], contract_full: Dict[str, Any]) -> List[Any]:
    gates = get_reviewer_gates(contract_full)
    if gates:
        return gates
    gates = get_reviewer_gates(contract_min)
    if gates:
        return gates
    eval_spec = contract_min.get("evaluation_spec")
    if isinstance(eval_spec, dict) and isinstance(eval_spec.get("reviewer_gates"), list):
        normalized = get_reviewer_gates({"reviewer_gates": eval_spec.get("reviewer_gates")})
        if normalized:
            return normalized
    eval_spec = contract_full.get("evaluation_spec") if isinstance(contract_full, dict) else None
    if isinstance(eval_spec, dict) and isinstance(eval_spec.get("reviewer_gates"), list):
        normalized = get_reviewer_gates({"reviewer_gates": eval_spec.get("reviewer_gates")})
        if normalized:
            return normalized
    return []


def _resolve_de_outlier_report_path_from_policy(outlier_policy: Dict[str, Any]) -> str:
    if not isinstance(outlier_policy, dict) or not outlier_policy:
        return ""
    enabled = outlier_policy.get("enabled")
    if isinstance(enabled, str):
        enabled = enabled.strip().lower() in {"1", "true", "yes", "on", "enabled"}
    if enabled is None:
        enabled = bool(
            outlier_policy.get("target_columns")
            or outlier_policy.get("methods")
            or outlier_policy.get("treatment")
        )
    stage = str(outlier_policy.get("apply_stage") or "data_engineer").strip().lower()
    if not bool(enabled) or stage not in {"data_engineer", "both"}:
        return ""
    report_path = outlier_policy.get("report_path") or outlier_policy.get("output_path")
    if isinstance(report_path, str) and report_path.strip():
        return report_path.strip()
    return _DEFAULT_DE_OUTLIER_REPORT_PATH


def _summarize_strategy(contract_full: Dict[str, Any], contract_min: Dict[str, Any], max_chars: int = 180) -> str:
    title = _first_value(contract_full.get("strategy_title"), contract_min.get("strategy_title"))
    objective_type = _resolve_objective_type(contract_min, contract_full, [])
    summary = f"{title or 'Strategy'} | objective={objective_type}"
    return summary[:max_chars]


def _expected_metrics_from_objective(objective_type: str, reviewer_gates: List[Any]) -> List[str]:
    obj = str(objective_type or "").lower()
    if "classif" in obj:
        return ["auc", "f1", "precision", "recall"]
    if "regress" in obj:
        return ["rmse", "mae", "r2"]
    if "forecast" in obj:
        return ["mae", "rmse"]
    if "rank" in obj:
        return ["ndcg", "map"]
    if "segment" in obj or "cluster" in obj:
        return ["silhouette"]
    if reviewer_gates:
        return ["metric_required_by_gate"]
    return []


def _build_min_reporting_policy(artifact_index: List[Dict[str, Any]]) -> Dict[str, Any]:
    artifacts = _normalize_artifact_index(artifact_index)
    artifact_types = {str(item.get("artifact_type") or "").lower() for item in artifacts if isinstance(item, dict)}
    artifact_paths_by_type: Dict[str, str] = {}
    for item in artifacts:
        if not isinstance(item, dict):
            continue
        artifact_type = str(item.get("artifact_type") or "").lower().strip()
        artifact_path = str(item.get("path") or "").strip()
        if artifact_type and artifact_path and artifact_type not in artifact_paths_by_type:
            artifact_paths_by_type[artifact_type] = artifact_path
    slots = []
    sections = ["Executive Decision", "Evidence & Metrics", "Risks & Limitations", "Next Actions"]
    if "metrics" in artifact_types:
        slots.append({"id": "model_metrics", "mode": "required", "sources": [artifact_paths_by_type.get("metrics", "data/metrics.json")]})
    if "predictions" in artifact_types:
        slots.append({"id": "predictions_overview", "mode": "optional", "sources": [artifact_paths_by_type.get("predictions", "data/scored_rows.csv")]})
    if "insights" in artifact_types:
        slots.append({"id": "insights", "mode": "optional", "sources": [artifact_paths_by_type.get("insights", "data/insights.json")]})
    if "report" in artifact_types:
        slots.append({"id": "alignment_check", "mode": "optional", "sources": [artifact_paths_by_type.get("report", "data/alignment_check.json")]})
    return {"sections": sections, "slots": slots}


def _truncate_text(value: str, max_len: int) -> str:
    if not isinstance(value, str) or len(value) <= max_len:
        return value
    return value[: max_len - 14] + "...[TRUNCATED]"


def _cap_plot_spec(plot_spec: Dict[str, Any] | None, max_caption_chars: int = 400) -> Dict[str, Any] | None:
    if not isinstance(plot_spec, dict) or not plot_spec:
        return None
    capped = dict(plot_spec)
    max_plots = plot_spec.get("max_plots")
    max_plots_int: int | None = None
    if isinstance(max_plots, (int, float)):
        max_plots_int = int(max_plots)
    elif isinstance(max_plots, str) and max_plots.strip().isdigit():
        max_plots_int = int(max_plots.strip())
    if max_plots_int is not None:
        capped["max_plots"] = max_plots_int
    plots = plot_spec.get("plots")
    if isinstance(plots, list):
        limit = max_plots_int if max_plots_int is not None and max_plots_int >= 0 else len(plots)
        trimmed = []
        for item in plots[:limit]:
            if not isinstance(item, dict):
                continue
            plot = dict(item)
            caption = plot.get("caption_template")
            if isinstance(caption, str):
                plot["caption_template"] = _truncate_text(caption, max_caption_chars)
            trimmed.append(plot)
        capped["plots"] = trimmed
    return capped


def _resolve_visual_context(
    contract_full: Dict[str, Any],
    contract_min: Dict[str, Any],
) -> Dict[str, Any]:
    artifact_reqs = _coerce_dict(contract_min.get("artifact_requirements")) or _coerce_dict(
        contract_full.get("artifact_requirements")
    )
    visual_reqs = (
        artifact_reqs.get("visual_requirements")
        if isinstance(artifact_reqs.get("visual_requirements"), dict)
        else {}
    )
    visual_items = visual_reqs.get("items") if isinstance(visual_reqs.get("items"), list) else []
    visual_payload: Dict[str, Any] = {
        "enabled": bool(visual_reqs.get("enabled")),
        "required": bool(visual_reqs.get("required")),
        "outputs_dir": visual_reqs.get("outputs_dir") or "static/plots",
        "items": visual_items,
        "notes": visual_reqs.get("notes") or "",
    }
    view_warnings: Dict[str, Any] = {}

    # Single source of truth: artifact_requirements.visual_requirements.plot_spec.
    canonical_plot_spec = _cap_plot_spec(
        visual_reqs.get("plot_spec") if isinstance(visual_reqs, dict) else None
    )

    if canonical_plot_spec is not None:
        visual_payload["enabled"] = bool(canonical_plot_spec.get("enabled", True))
        if not visual_items:
            visual_payload["items"] = _derive_visual_items_from_plot_spec(
                canonical_plot_spec,
                str(visual_payload.get("outputs_dir") or "static/plots"),
            )

    if canonical_plot_spec is not None:
        view_warnings["plot_spec_source"] = "artifact_requirements.visual_requirements.plot_spec"

    return {
        "visual_payload": visual_payload,
        "plot_spec": canonical_plot_spec,
        "view_warnings": view_warnings,
    }


def _slugify_plot_token(value: Any, fallback: str) -> str:
    token = re.sub(r"[^0-9a-zA-Z]+", "_", str(value or "")).strip("_").lower()
    return token or fallback


def _normalize_plot_output_path(raw_path: Any, outputs_dir: str, fallback_id: str) -> str:
    base_dir = str(outputs_dir or "static/plots").replace("\\", "/").rstrip("/")
    if not base_dir:
        base_dir = "static/plots"
    candidate = str(raw_path or "").strip().replace("\\", "/")
    if not candidate:
        candidate = f"{_slugify_plot_token(fallback_id, 'plot')}.png"
    if not os.path.splitext(candidate)[1]:
        candidate = f"{candidate}.png"
    if not os.path.isabs(candidate) and "/" not in candidate:
        candidate = f"{base_dir}/{candidate}"
    return candidate


def _derive_visual_items_from_plot_spec(plot_spec: Dict[str, Any] | None, outputs_dir: str) -> List[Dict[str, Any]]:
    if not isinstance(plot_spec, dict):
        return []
    plots = plot_spec.get("plots")
    if not isinstance(plots, list):
        return []
    items: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for idx, raw in enumerate(plots, start=1):
        if not isinstance(raw, dict):
            continue
        plot_id = raw.get("plot_id") or raw.get("id") or raw.get("name") or raw.get("title") or f"plot_{idx}"
        safe_id = _slugify_plot_token(plot_id, f"plot_{idx}")
        raw_path = (
            raw.get("path")
            or raw.get("output_path")
            or raw.get("output")
            or raw.get("artifact")
            or raw.get("file")
            or raw.get("filename")
            or raw.get("expected_filename")
            or safe_id
        )
        path = _normalize_plot_output_path(raw_path, outputs_dir, safe_id)
        key = path.lower()
        if key in seen:
            continue
        seen.add(key)
        items.append(
            {
                "id": safe_id,
                "purpose": str(raw.get("goal") or raw.get("title") or raw.get("caption_template") or "").strip(),
                "type": str(raw.get("type") or raw.get("plot_type") or "other"),
                "expected_filename": os.path.basename(path),
                "path": path,
                "required": True,
            }
        )
    return items


def _trim_value(
    obj: Any,
    max_str_len: int,
    max_list_items: int,
    preserve_keys: set[str],
    path: List[str],
) -> Any:
    if isinstance(obj, str):
        return _truncate_text(obj, max_str_len)
    if isinstance(obj, list):
        key = path[-1] if path else ""
        if key not in preserve_keys and len(obj) > max_list_items:
            trimmed = obj[:max_list_items]
            trimmed.append(f"...({len(obj)} total)")
            obj = trimmed
        return [
            _trim_value(item, max_str_len, max_list_items, preserve_keys, path + ["[]"])
            for item in obj
        ]
    if isinstance(obj, dict):
        trimmed: Dict[str, Any] = {}
        for key in sorted(obj.keys()):
            trimmed[key] = _trim_value(obj[key], max_str_len, max_list_items, preserve_keys, path + [key])
        return trimmed
    return obj


def trim_to_budget(
    obj: Any,
    max_chars: int,
    max_str_len: int = 1200,
    max_list_items: int = 25,
) -> Any:
    if obj is None:
        return obj
    try:
        max_str_len = int(max_str_len)
    except Exception:
        max_str_len = 1200
    try:
        max_list_items = int(max_list_items)
    except Exception:
        max_list_items = 25
    max_str_len = max(200, max_str_len)
    max_list_items = max(8, max_list_items)
    for _ in range(4):
        trimmed = _trim_value(obj, max_str_len, max_list_items, _PRESERVE_KEYS, [])
        payload = json.dumps(trimmed, ensure_ascii=True, sort_keys=True)
        if len(payload) <= max_chars:
            return trimmed
        max_str_len = max(200, int(max_str_len * 0.7))
        max_list_items = max(8, int(max_list_items * 0.7))
    return trimmed


def build_de_view(
    contract_full: Dict[str, Any] | None,
    contract_min: Dict[str, Any] | None,
    artifact_index: Any,
) -> Dict[str, Any]:
    contract_full = contract_full if isinstance(contract_full, dict) else {}
    contract_min = contract_min if isinstance(contract_min, dict) else {}
    task_semantics = _resolve_task_semantics(contract_min, contract_full)
    required_outputs = _resolve_required_outputs(contract_min, contract_full)
    required_columns = _resolve_required_columns(contract_min, contract_full)
    required_feature_selectors = _resolve_required_feature_selectors(contract_min, contract_full)
    passthrough_columns = _resolve_passthrough_columns(contract_min, contract_full, required_columns)
    column_transformations = _resolve_column_transformations(contract_min, contract_full)
    column_dtype_targets = _resolve_column_dtype_targets({}, contract_full)
    output_path = _resolve_output_path(contract_min, contract_full, required_outputs)
    manifest_path = _resolve_manifest_path(contract_min, contract_full, required_outputs)
    cleaning_gates = _resolve_cleaning_gates(contract_min, contract_full)
    data_engineer_runbook = _resolve_data_engineer_runbook(contract_min, contract_full)
    outlier_policy = _resolve_outlier_policy(contract_min, contract_full)
    view: DEView = {
        "role": "data_engineer",
        "task_semantics": task_semantics,
        "required_columns": required_columns,
        "optional_passthrough_columns": passthrough_columns,
        "output_path": output_path or "",
        "required_columns_path": "data/required_columns.json",
        "column_sets_path": "data/column_sets.json",
        "cleaning_gates": cleaning_gates,
        "data_engineer_runbook": data_engineer_runbook,
        "constraints": {
            "scope": "cleaning_only",
            "hard_constraints": [
                "no_modeling",
                "no_score_fitting",
                "no_prescriptive_tuning",
                "no_analytics",
            ],
        },
    }
    if required_feature_selectors:
        view["required_feature_selectors"] = required_feature_selectors
    if outlier_policy and outlier_policy.get("apply_stage") in {"data_engineer", "both"}:
        view["outlier_policy"] = outlier_policy
        report_path = _resolve_de_outlier_report_path_from_policy(outlier_policy)
        if report_path:
            view["outlier_report_path"] = report_path
    if column_transformations:
        view["column_transformations"] = column_transformations
    if column_dtype_targets:
        view["column_dtype_targets"] = column_dtype_targets
    if manifest_path:
        view["output_manifest_path"] = manifest_path
    output_dialect = _resolve_output_dialect(contract_min, contract_full)
    if output_dialect:
        view["output_dialect"] = output_dialect
    column_sets_summary = contract_min.get("column_sets_summary") or contract_full.get("column_sets_summary")
    if column_sets_summary:
        view["column_sets_summary"] = column_sets_summary
    return trim_to_budget(view, 8000)


def build_ml_view(
    contract_full: Dict[str, Any] | None,
    contract_min: Dict[str, Any] | None,
    artifact_index: Any,
) -> Dict[str, Any]:
    contract_full = contract_full if isinstance(contract_full, dict) else {}
    contract_min = contract_min if isinstance(contract_min, dict) else {}
    required_outputs = _resolve_required_outputs(contract_min, contract_full)
    objective_type = _resolve_objective_type(contract_min, contract_full, required_outputs)
    task_semantics = _resolve_task_semantics(contract_min, contract_full)
    canonical_columns = contract_min.get("canonical_columns")
    if not isinstance(canonical_columns, list):
        canonical_columns = get_canonical_columns(contract_full)
    canonical_columns = [str(c) for c in canonical_columns if c]
    roles_min = get_column_roles(contract_min)
    roles_full = get_column_roles(contract_full)
    allowed_sets_min = contract_min.get("allowed_feature_sets")
    if not isinstance(allowed_sets_min, dict):
        allowed_sets_min = {}
    allowed_sets_full = contract_full.get("allowed_feature_sets")
    if not isinstance(allowed_sets_full, dict):
        allowed_sets_full = {}
    forbidden_min = _resolve_forbidden_features(allowed_sets_min)
    role_sets_min = _extract_roles(roles_min)
    pre_decision_min = _coerce_list(roles_min.get("pre_decision"))
    decision_min = _coerce_list(roles_min.get("decision"))
    outcome_min = _coerce_list(roles_min.get("outcome"))
    canonical_set = set(canonical_columns)
    lax_roles = False
    if not roles_min:
        lax_roles = True
    elif not role_sets_min.get("audit_only") and not forbidden_min and not decision_min and not outcome_min:
        if canonical_set:
            overlap = len(set(pre_decision_min) & canonical_set)
            coverage = overlap / max(1, len(canonical_set))
            if coverage >= 0.9:
                lax_roles = True
        else:
            lax_roles = True
    use_full_roles = bool(roles_full) and lax_roles
    column_roles = roles_full if use_full_roles else roles_min
    if not column_roles:
        column_roles = roles_full or roles_min
    role_sets = _extract_roles(column_roles)
    audit_only_cols = [str(c) for c in role_sets.get("audit_only", []) if c]
    decision_cols = [str(c) for c in role_sets.get("decision", []) if c]
    outcome_cols = [str(c) for c in role_sets.get("outcome", []) if c]
    pre_decision_cols = _coerce_list(column_roles.get("pre_decision"))
    if not pre_decision_cols:
        assigned = set(decision_cols + outcome_cols + audit_only_cols)
        pre_decision_cols = [c for c in canonical_columns if c and c not in assigned]

    def _list_or_none(source: Dict[str, Any], key: str) -> List[str] | None:
        val = source.get(key)
        if isinstance(val, list):
            return [str(c) for c in val if c]
        return None

    full_model = _list_or_none(allowed_sets_full, "model_features")
    full_seg = _list_or_none(allowed_sets_full, "segmentation_features")
    full_forbidden = _list_or_none(allowed_sets_full, "forbidden_for_modeling")
    if full_forbidden is None:
        full_forbidden = _list_or_none(allowed_sets_full, "forbidden_features")
    full_audit = _list_or_none(allowed_sets_full, "audit_only_features")

    min_model = _list_or_none(allowed_sets_min, "model_features")
    min_seg = _list_or_none(allowed_sets_min, "segmentation_features")
    min_forbidden = _list_or_none(allowed_sets_min, "forbidden_features")
    if min_forbidden is None:
        min_forbidden = _list_or_none(allowed_sets_min, "forbidden_for_modeling")
    min_audit = _list_or_none(allowed_sets_min, "audit_only_features")

    model_features = (
        full_model
        if full_model is not None
        else (min_model if min_model is not None else list(dict.fromkeys(pre_decision_cols + decision_cols)))
    )
    segmentation_features = (
        full_seg
        if full_seg is not None
        else (min_seg if min_seg is not None else list(pre_decision_cols))
    )
    forbidden = (
        full_forbidden
        if full_forbidden is not None
        else (min_forbidden if min_forbidden is not None else [])
    )
    audit_only_features = full_audit if full_audit is not None else min_audit
    if audit_only_features is not None:
        audit_only_cols = list(dict.fromkeys([str(c) for c in audit_only_features if c]))

    explicit_allowed = set()
    for candidate_list in (full_model, min_model, full_seg, min_seg):
        if isinstance(candidate_list, list):
            explicit_allowed.update(str(c) for c in candidate_list if c)

    derived_columns = _resolve_derived_columns(contract_min, contract_full)
    derived_set = {str(c) for c in derived_columns if c}
    allowed_noncanonical = set(explicit_allowed) | set(derived_set)

    strict_ids = []
    candidate_ids = []
    seen = set()
    for col in canonical_columns:
        if not col:
            continue
        if col in seen:
            continue
        seen.add(col)
        if is_strict_identifier_column(col):
            strict_ids.append(col)
        elif is_candidate_identifier_column(col):
            candidate_ids.append(col)

    strict_allowed_by_contract = [c for c in strict_ids if c in explicit_allowed]
    candidate_allowed_by_contract = [c for c in candidate_ids if c in explicit_allowed]
    strict_forbidden = [c for c in strict_ids if c not in explicit_allowed]

    forbidden = list(dict.fromkeys([str(c) for c in forbidden if c]))
    forbidden = sorted(dict.fromkeys(forbidden + strict_forbidden + outcome_cols + audit_only_cols))

    def _filter_noncanonical(items: List[str]) -> tuple[List[str], List[str]]:
        filtered = []
        dropped = []
        for val in items:
            if not val:
                continue
            if val in canonical_set or val in allowed_noncanonical:
                filtered.append(str(val))
            else:
                dropped.append(str(val))
        return filtered, dropped

    forbidden_filtered, dropped_forbidden = _filter_noncanonical(forbidden)
    forbidden_set = set(forbidden_filtered)
    model_features_filtered, dropped_model = _filter_noncanonical(model_features)
    segmentation_features_filtered, dropped_segmentation = _filter_noncanonical(segmentation_features)

    view_warnings: Dict[str, Any] = {}
    dropped_noncanonical = {}
    if dropped_model:
        dropped_noncanonical["model_features"] = dropped_model
    if dropped_segmentation:
        dropped_noncanonical["segmentation_features"] = dropped_segmentation
    if dropped_forbidden:
        dropped_noncanonical["forbidden_features"] = dropped_forbidden
    if dropped_noncanonical:
        view_warnings["dropped_noncanonical"] = dropped_noncanonical

    model_features = [c for c in model_features_filtered if c not in forbidden_set]
    segmentation_features = [c for c in segmentation_features_filtered if c not in forbidden_set]
    derived_features = list(
        dict.fromkeys([c for c in derived_columns if c in set(model_features + segmentation_features)])
    )
    final_forbidden = forbidden_filtered
    allowed_sets = {
        "segmentation_features": segmentation_features,
        "model_features": model_features,
        "forbidden_features": final_forbidden,
    }
    if audit_only_features is not None:
        allowed_sets["audit_only_features"] = [str(c) for c in audit_only_features if c]
    validation = _resolve_validation_requirements(contract_min, contract_full)
    case_rules = _resolve_case_rules(contract_full)
    outlier_policy = _resolve_outlier_policy(contract_min, contract_full)
    column_dtype_targets = _resolve_column_dtype_targets({}, contract_full)

    artifact_reqs = _coerce_dict(contract_min.get("artifact_requirements")) or _coerce_dict(
        contract_full.get("artifact_requirements")
    )
    scored_rows_schema: Dict[str, Any] = {}
    if isinstance(artifact_reqs.get("scored_rows_schema"), dict):
        scored_rows_schema = {
            key: artifact_reqs["scored_rows_schema"].get(key)
            for key in (
                "required_columns",
                "recommended_columns",
                "required_any_of_groups",
                "required_any_of_group_severity",
            )
            if artifact_reqs["scored_rows_schema"].get(key) not in (None, [], {})
        }
    required_files_payload: List[str] = []
    required_files = artifact_reqs.get("required_files")
    if isinstance(required_files, list):
        for entry in required_files:
            if not entry:
                continue
            if isinstance(entry, dict):
                path = entry.get("path") or entry.get("output") or entry.get("artifact")
                if path:
                    required_files_payload.append(str(path))
            else:
                required_files_payload.append(str(entry))
    # Filter required_outputs for ML view: use deliverables-by-owner when available
    from src.utils.contract_accessors import get_deliverables_by_owner
    ml_deliverables = get_deliverables_by_owner(contract_full, "ml_engineer")
    if ml_deliverables:
        ml_required_outputs = [d["path"] for d in ml_deliverables if d.get("required")]
    else:
        # Backward compat: fall back to full required_outputs
        ml_required_outputs = list(required_outputs)

    artifact_payload: Dict[str, Any] = {"required_outputs": ml_required_outputs}
    if required_files_payload:
        artifact_payload["required_files"] = required_files_payload
    if scored_rows_schema:
        artifact_payload["scored_rows_schema"] = scored_rows_schema
    file_schemas = artifact_reqs.get("file_schemas")
    if isinstance(file_schemas, dict) and file_schemas:
        artifact_payload["file_schemas"] = file_schemas

    view: MLView = {
        "role": "ml_engineer",
        "objective_type": objective_type,
        "task_semantics": task_semantics,
        "canonical_columns": canonical_columns,
        "derived_features": derived_features,
        "column_roles": column_roles,
        "decision_columns": decision_cols,
        "outcome_columns": outcome_cols,
        "audit_only_columns": audit_only_cols,
        "identifier_columns": strict_ids + candidate_ids,
        "allowed_feature_sets": allowed_sets,
        "forbidden_features": final_forbidden,
        "column_dtype_targets": column_dtype_targets,
        "required_outputs": ml_required_outputs,
        "validation_requirements": validation,
    }
    row_count_hints = _resolve_row_count_hints(contract_full, contract_min)
    if row_count_hints:
        for key in ("n_train_rows", "n_test_rows", "n_total_rows"):
            if key in row_count_hints:
                view[key] = row_count_hints[key]
    split_spec = contract_min.get("split_spec")
    if not isinstance(split_spec, dict) or not split_spec:
        split_spec = contract_full.get("split_spec")
    if isinstance(split_spec, dict) and split_spec:
        view["split_spec"] = split_spec
    column_sets_summary = contract_min.get("column_sets_summary") or contract_full.get("column_sets_summary")
    if column_sets_summary:
        view["column_sets_summary"] = column_sets_summary
    training_rows_rule = contract_min.get("training_rows_rule") or contract_full.get("training_rows_rule")
    scoring_rows_rule = contract_min.get("scoring_rows_rule") or contract_full.get("scoring_rows_rule")
    secondary_scoring_subset = contract_min.get("secondary_scoring_subset") or contract_full.get("secondary_scoring_subset")
    data_partitioning_notes = contract_min.get("data_partitioning_notes") or contract_full.get("data_partitioning_notes")
    if training_rows_rule:
        view["training_rows_rule"] = training_rows_rule
    if scoring_rows_rule:
        view["scoring_rows_rule"] = scoring_rows_rule
    if secondary_scoring_subset:
        view["secondary_scoring_subset"] = secondary_scoring_subset
    if isinstance(data_partitioning_notes, list) and data_partitioning_notes:
        view["data_partitioning_notes"] = data_partitioning_notes
    identifier_policy = {
        "strict_forbidden": strict_forbidden,
        "candidates": candidate_ids,
        "guidance": [
            "Candidate identifiers may be useful categorical features if low-cardinality.",
            "If a candidate identifier is high-cardinality or near-unique, drop it.",
            "Perform a quick uniqueness/cardinality check on a sample before using.",
        ],
    }
    view["identifier_policy"] = identifier_policy
    view["identifier_overrides"] = {
        "strict_allowed_by_contract": strict_allowed_by_contract,
        "candidate_allowed_by_contract": candidate_allowed_by_contract,
    }
    if artifact_payload:
        view["artifact_requirements"] = artifact_payload
    visual_ctx = _resolve_visual_context(contract_full, contract_min)
    visual_payload = visual_ctx.get("visual_payload", {})
    plot_spec = visual_ctx.get("plot_spec")
    visual_warnings = visual_ctx.get("view_warnings") if isinstance(visual_ctx.get("view_warnings"), dict) else {}
    view["decisioning_requirements"] = _get_decisioning_requirements(contract_full, contract_min)
    merged_warnings = dict(view_warnings)
    if visual_warnings:
        merged_warnings.update(visual_warnings)
    if merged_warnings:
        view["view_warnings"] = merged_warnings
    if case_rules is not None:
        view["case_rules"] = case_rules
    if outlier_policy:
        view["outlier_policy"] = outlier_policy
    if plot_spec is not None:
        view["plot_spec"] = plot_spec
    view["visual_requirements"] = visual_payload
    view["decisioning_requirements"] = _get_decisioning_requirements(contract_full, contract_min)
    return trim_to_budget(view, 16000)


def build_reviewer_view(
    contract_full: Dict[str, Any] | None,
    contract_min: Dict[str, Any] | None,
    artifact_index: Any,
) -> Dict[str, Any]:
    contract_full = contract_full if isinstance(contract_full, dict) else {}
    contract_min = contract_min if isinstance(contract_min, dict) else {}
    required_outputs = _resolve_required_outputs(contract_min, contract_full)
    objective_type = _resolve_objective_type(contract_min, contract_full, required_outputs)
    task_semantics = _resolve_task_semantics(contract_min, contract_full)
    reviewer_gates = _resolve_reviewer_gates(contract_min, contract_full)
    expected_metrics = _expected_metrics_from_objective(objective_type, reviewer_gates)
    strategy_summary = _summarize_strategy(contract_full, contract_min)
    view: ReviewerView = {
        "role": "reviewer",
        "objective_type": objective_type,
        "task_semantics": task_semantics,
        "reviewer_gates": reviewer_gates,
        "required_outputs": required_outputs,
        "expected_metrics": expected_metrics,
        "strategy_summary": strategy_summary,
        "verification": {
            "required_outputs": required_outputs,
            "artifact_index_expected": bool(artifact_index),
        },
    }
    view["decisioning_requirements"] = _get_decisioning_requirements(contract_full, contract_min)
    return trim_to_budget(view, 12000)


def build_qa_view(
    contract_full: Dict[str, Any] | None,
    contract_min: Dict[str, Any] | None,
    artifact_index: Any,
) -> Dict[str, Any]:
    contract_full = contract_full if isinstance(contract_full, dict) else {}
    contract_min = contract_min if isinstance(contract_min, dict) else {}
    task_semantics = _resolve_task_semantics(contract_min, contract_full)
    required_outputs = _resolve_required_outputs(contract_min, contract_full)
    optional_outputs = _resolve_optional_outputs(contract_min, contract_full)
    qa_gates = _resolve_qa_gates(contract_min, contract_full)
    allowed_feature_sets = _resolve_allowed_feature_sets(contract_min, contract_full)
    column_roles = _resolve_column_roles(contract_min, contract_full)
    canonical_columns = contract_min.get("canonical_columns")
    if not isinstance(canonical_columns, list):
        canonical_columns = get_canonical_columns(contract_full)
    canonical_columns = [str(c) for c in canonical_columns if c]
    artifact_reqs = _coerce_dict(contract_min.get("artifact_requirements")) or _coerce_dict(
        contract_full.get("artifact_requirements")
    )
    artifact_payload: Dict[str, Any] = {
        "required_outputs": required_outputs,
        "optional_outputs": optional_outputs,
    }
    file_schemas = artifact_reqs.get("file_schemas")
    if isinstance(file_schemas, dict) and file_schemas:
        artifact_payload["file_schemas"] = file_schemas
    objective_summary = {
        "strategy_title": _first_value(contract_full.get("strategy_title"), contract_min.get("strategy_title")) or "",
        "business_objective": _first_value(contract_full.get("business_objective"), contract_min.get("business_objective"))
        or "",
    }
    reporting_policy = contract_full.get("reporting_policy")
    if not isinstance(reporting_policy, dict) or not reporting_policy:
        reporting_policy = contract_min.get("reporting_policy")
    view: QAView = {
        "role": "qa_reviewer",
        "task_semantics": task_semantics,
        "qa_gates": qa_gates,
        "artifact_requirements": artifact_payload,
        "allowed_feature_sets": allowed_feature_sets,
        "column_roles": column_roles,
        "canonical_columns": canonical_columns,
        "objective_summary": objective_summary,
    }
    row_count_hints = _resolve_row_count_hints(contract_full, contract_min)
    if row_count_hints:
        for key in ("n_train_rows", "n_test_rows", "n_total_rows"):
            if key in row_count_hints:
                view[key] = row_count_hints[key]
    split_spec = contract_min.get("split_spec")
    if not isinstance(split_spec, dict) or not split_spec:
        split_spec = contract_full.get("split_spec")
    if isinstance(split_spec, dict) and split_spec:
        view["split_spec"] = split_spec
    if isinstance(reporting_policy, dict) and reporting_policy:
        view["reporting_policy"] = reporting_policy
    view["decisioning_requirements"] = _get_decisioning_requirements(contract_full, contract_min)
    return trim_to_budget(view, 12000)


def build_translator_view(
    contract_full: Dict[str, Any] | None,
    contract_min: Dict[str, Any] | None,
    artifact_index: Any,
    insights: Any = None,
) -> Dict[str, Any]:
    contract_full = contract_full if isinstance(contract_full, dict) else {}
    contract_min = contract_min if isinstance(contract_min, dict) else {}
    policy = contract_full.get("reporting_policy")
    if not isinstance(policy, dict) or not policy:
        policy = contract_min.get("reporting_policy")
    if not isinstance(policy, dict) or not policy:
        policy = _build_min_reporting_policy(_normalize_artifact_index(artifact_index))
    evidence = _normalize_artifact_index(artifact_index)
    objective_type = _resolve_objective_type(contract_min, contract_full, [])
    required_outputs = _resolve_required_outputs(contract_min, contract_full)
    key_decisions = []
    if objective_type:
        key_decisions.append(f"objective_type:{objective_type}")
    if required_outputs:
        key_decisions.append(f"required_outputs:{len(required_outputs)}")
    limitations = []
    risks = contract_full.get("data_risks") if isinstance(contract_full, dict) else None
    if isinstance(risks, list):
        limitations.extend([str(item) for item in risks if item])
    view: TranslatorView = {
        "role": "translator",
        "reporting_policy": policy,
        "evidence_inventory": evidence,
        "key_decisions": key_decisions,
        "limitations": limitations,
        "constraints": {"no_markdown_tables": True, "cite_sources": True},
    }
    visual_ctx = _resolve_visual_context(contract_full, contract_min)
    plot_spec = visual_ctx.get("plot_spec")
    visual_payload = visual_ctx.get("visual_payload", {})
    if plot_spec is not None:
        view["plot_spec"] = plot_spec
    visual_warnings = visual_ctx.get("view_warnings")
    if isinstance(visual_warnings, dict) and visual_warnings:
        view["view_warnings"] = visual_warnings
    view["visual_requirements"] = visual_payload
    view["decisioning_requirements"] = _get_decisioning_requirements(contract_full, contract_min)
    return trim_to_budget(view, 16000)


def build_results_advisor_view(
    contract_full: Dict[str, Any] | None,
    contract_min: Dict[str, Any] | None,
    artifact_index: Any,
) -> Dict[str, Any]:
    contract_full = contract_full if isinstance(contract_full, dict) else {}
    contract_min = contract_min if isinstance(contract_min, dict) else {}
    required_outputs = _resolve_required_outputs(contract_min, contract_full)
    objective_type = _resolve_objective_type(contract_min, contract_full, required_outputs)
    policy = contract_full.get("reporting_policy")
    if not isinstance(policy, dict) or not policy:
        policy = contract_min.get("reporting_policy")
    if not isinstance(policy, dict) or not policy:
        policy = {}
    view: ResultsAdvisorView = {
        "role": "results_advisor",
        "objective_type": objective_type,
        "reporting_policy": policy,
        "evidence_inventory": _normalize_artifact_index(artifact_index),
    }
    return trim_to_budget(view, 12000)


def build_cleaning_view(
    contract_full: Dict[str, Any] | None,
    contract_min: Dict[str, Any] | None,
    artifact_index: Any,
    cleaning_code: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build cleaning reviewer view with optional DE cleaning code for intent verification.

    Args:
        cleaning_code: Optional Python code generated by the Data Engineer. When provided,
                      allows the Reviewer to verify INTENT (e.g., check if /255 division
                      exists in code) rather than guessing from data samples alone.
    """
    contract_full = contract_full if isinstance(contract_full, dict) else {}
    contract_min = contract_min if isinstance(contract_min, dict) else {}
    task_semantics = _resolve_task_semantics(contract_min, contract_full)
    required_columns = _resolve_required_columns(contract_min, contract_full)
    required_feature_selectors = _resolve_required_feature_selectors(contract_min, contract_full)
    column_roles = _resolve_column_roles(contract_min, contract_full)
    allowed_feature_sets = _resolve_allowed_feature_sets(contract_min, contract_full)
    dialect = _resolve_output_dialect(contract_min, contract_full)
    cleaning_gates = _resolve_cleaning_gates(contract_min, contract_full)
    column_transformations = _resolve_column_transformations(contract_min, contract_full)
    outlier_policy = _resolve_outlier_policy(contract_min, contract_full)
    view: CleaningView = {
        "role": "cleaning_reviewer",
        "task_semantics": task_semantics,
        "strategy_title": _first_value(contract_full.get("strategy_title"), contract_min.get("strategy_title")) or "",
        "business_objective": _first_value(
            contract_full.get("business_objective"), contract_min.get("business_objective")
        ) or "",
        "required_columns": required_columns,
        "column_sets_path": "data/column_sets.json",
        "dialect": dialect,
        "cleaning_gates": cleaning_gates,
        "column_roles": column_roles,
        "allowed_feature_sets": allowed_feature_sets,
    }
    if required_feature_selectors:
        view["required_feature_selectors"] = required_feature_selectors
    if outlier_policy and outlier_policy.get("apply_stage") in {"data_engineer", "both"}:
        view["outlier_policy"] = outlier_policy
        report_path = _resolve_de_outlier_report_path_from_policy(outlier_policy)
        if report_path:
            view["outlier_report_path"] = report_path
    if column_transformations:
        view["column_transformations"] = column_transformations
    # Include cleaning code for intent verification (rescale detection, synthetic data check)
    if cleaning_code and isinstance(cleaning_code, str):
        # Truncate if too long to fit in budget
        max_code_len = 8000
        code_to_include = cleaning_code[:max_code_len] if len(cleaning_code) > max_code_len else cleaning_code
        view["cleaning_code"] = code_to_include
    return trim_to_budget(view, 15000)


def _project_decisioning_requirements(contract_full: Dict[str, Any]) -> Dict[str, Any]:
    decisioning = contract_full.get("decisioning_requirements")
    if isinstance(decisioning, dict):
        return {
            "enabled": bool(decisioning.get("enabled")),
            "required": bool(decisioning.get("required")),
            "output": decisioning.get("output", {}),
            "policy_notes": decisioning.get("policy_notes", ""),
        }
    return {"enabled": False, "required": False, "output": {}, "policy_notes": ""}


def _project_objective_type(contract_full: Dict[str, Any]) -> str:
    task_semantics = get_task_semantics(contract_full)
    if isinstance(task_semantics, dict):
        for key in ("objective_type", "problem_family"):
            value = str(task_semantics.get(key) or "").strip()
            if value and value.lower() != "unknown":
                return value
    objective_analysis = contract_full.get("objective_analysis")
    if isinstance(objective_analysis, dict):
        problem_type = str(objective_analysis.get("problem_type") or "").strip()
        if problem_type and problem_type.lower() != "unknown":
            return problem_type
    evaluation_spec = contract_full.get("evaluation_spec")
    if isinstance(evaluation_spec, dict):
        objective_type = str(evaluation_spec.get("objective_type") or "").strip()
        if objective_type and objective_type.lower() != "unknown":
            return objective_type
    capabilities = resolve_problem_capabilities_from_contract(contract_full if isinstance(contract_full, dict) else {})
    family = str(capabilities.get("family") or "").strip()
    if family and family != "unknown":
        return family
    required_outputs = _project_required_outputs(contract_full)
    inferred = _infer_objective_from_outputs(required_outputs)
    return inferred or "unknown"


def _resolve_task_semantics(contract_min: Dict[str, Any], contract_full: Dict[str, Any]) -> Dict[str, Any]:
    for source in (contract_min, contract_full):
        task_semantics = get_task_semantics(source if isinstance(source, dict) else {})
        if isinstance(task_semantics, dict) and task_semantics:
            return task_semantics

    objective_type = _resolve_objective_type(contract_min, contract_full, _resolve_required_outputs(contract_min, contract_full))
    outcome_columns = get_outcome_columns(contract_min if isinstance(contract_min, dict) and contract_min else contract_full)
    primary_target = outcome_columns[0] if outcome_columns else None
    column_roles = get_column_roles(contract_min if isinstance(contract_min, dict) and contract_min else contract_full)
    identifier_columns = []
    if isinstance(column_roles, dict):
        identifier_columns = [
            str(col) for col in (column_roles.get("identifiers") or []) if str(col).strip()
        ]
    return {
        "objective_type": objective_type,
        "problem_family": objective_type,
        "primary_target": primary_target,
        "target_columns": outcome_columns,
        "multi_target": len(outcome_columns) > 1,
        "prediction_unit": {"kind": "row", "identifier_columns": identifier_columns},
    }


def _project_required_outputs(contract_full: Dict[str, Any]) -> List[str]:
    outputs = contract_full.get("required_outputs")
    if not isinstance(outputs, list) or not outputs:
        outputs = get_required_outputs(contract_full)
    normalized: List[str] = []
    seen: set[str] = set()

    def _extract_path(item: Any) -> str:
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            for key in ("path", "output", "artifact"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    return value
        return ""

    for item in outputs:
        if not item:
            continue
        text = _extract_path(item).replace("\\", "/").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(text)
    return normalized


def _project_artifact_requirements(contract_full: Dict[str, Any]) -> Dict[str, Any]:
    artifact_reqs = contract_full.get("artifact_requirements")
    if isinstance(artifact_reqs, dict):
        return artifact_reqs
    return {}


def _project_plot_payload(contract_full: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any] | None]:
    artifact_reqs = _project_artifact_requirements(contract_full)
    visual_requirements = artifact_reqs.get("visual_requirements")
    if not isinstance(visual_requirements, dict):
        visual_requirements = {}
    plot_spec = visual_requirements.get("plot_spec")
    if not isinstance(plot_spec, dict):
        reporting_policy = contract_full.get("reporting_policy")
        if isinstance(reporting_policy, dict):
            rp_plot_spec = reporting_policy.get("plot_spec")
            if isinstance(rp_plot_spec, dict):
                plot_spec = rp_plot_spec
    return visual_requirements, plot_spec if isinstance(plot_spec, dict) else None


def build_contract_views_projection(
    contract_full: Dict[str, Any] | None,
    artifact_index: Any,
    cleaning_code: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Build agent views as pure projection from validated execution contract.

    No fallback inference from auxiliary artifacts and no deterministic synthesis
    beyond safe defaults for missing optional fields.
    """
    contract_full = contract_full if isinstance(contract_full, dict) else {}
    artifact_reqs = _project_artifact_requirements(contract_full)
    required_outputs = _project_required_outputs(contract_full)
    objective_type = _project_objective_type(contract_full)
    task_semantics = _resolve_task_semantics(contract_full, contract_full)
    canonical_columns = [str(c) for c in get_canonical_columns(contract_full) if c]
    column_roles = get_column_roles(contract_full)
    decision_columns = [str(c) for c in get_column_roles(contract_full).get("decision", []) if c]
    outcome_columns = [str(c) for c in get_outcome_columns(contract_full) if c]
    audit_only_columns = [
        str(c)
        for c in (
            column_roles.get("post_decision_audit_only")
            or column_roles.get("audit_only")
            or []
        )
        if c
    ]
    decisioning_requirements = _project_decisioning_requirements(contract_full)
    outlier_policy = _resolve_outlier_policy({}, contract_full)
    visual_requirements, plot_spec = _project_plot_payload(contract_full)
    derived_columns = [str(c) for c in get_derived_column_names(contract_full) if c]
    validation_requirements = contract_full.get("validation_requirements")
    if not isinstance(validation_requirements, dict):
        validation_requirements = {}
    evaluation_spec = contract_full.get("evaluation_spec")
    if not isinstance(evaluation_spec, dict):
        evaluation_spec = {}
    objective_analysis = contract_full.get("objective_analysis")
    if not isinstance(objective_analysis, dict):
        objective_analysis = {}
    ml_engineer_runbook = contract_full.get("ml_engineer_runbook")
    if not isinstance(ml_engineer_runbook, (dict, list, str)):
        ml_engineer_runbook = {}
    allowed_feature_sets = contract_full.get("allowed_feature_sets")
    if not isinstance(allowed_feature_sets, dict):
        allowed_feature_sets = {
            "model_features": [],
            "segmentation_features": [],
            "audit_only_features": [],
            "forbidden_for_modeling": [],
        }
    forbidden_features = allowed_feature_sets.get("forbidden_for_modeling")
    if not isinstance(forbidden_features, list):
        forbidden_features = allowed_feature_sets.get("forbidden_features")
    if not isinstance(forbidden_features, list):
        forbidden_features = []
    forbidden_features = [str(c) for c in forbidden_features if c]
    model_features = allowed_feature_sets.get("model_features")
    if not isinstance(model_features, list):
        model_features = []
    segmentation_features = allowed_feature_sets.get("segmentation_features")
    if not isinstance(segmentation_features, list):
        segmentation_features = []
    clean_cfg = artifact_reqs.get("clean_dataset")
    if not isinstance(clean_cfg, dict):
        clean_cfg = {}
    de_required_columns = clean_cfg.get("required_columns")
    if not isinstance(de_required_columns, list) or not de_required_columns:
        de_required_columns = list(canonical_columns)
    de_required_columns = [str(c) for c in de_required_columns if c]
    de_required_feature_selectors = clean_cfg.get("required_feature_selectors")
    if not isinstance(de_required_feature_selectors, list):
        de_required_feature_selectors = []
    else:
        de_required_feature_selectors = [dict(item) for item in de_required_feature_selectors if isinstance(item, dict)]
    de_passthrough = clean_cfg.get("optional_passthrough_columns")
    if not isinstance(de_passthrough, list):
        de_passthrough = []
    de_passthrough = [str(c) for c in de_passthrough if c]
    column_transformations = clean_cfg.get("column_transformations")
    if not isinstance(column_transformations, dict):
        column_transformations = {}
    if "drop_policy" not in column_transformations and "drop_policy" in clean_cfg:
        column_transformations["drop_policy"] = clean_cfg.get("drop_policy")
    if not isinstance(column_transformations.get("drop_columns"), list):
        column_transformations["drop_columns"] = [
            str(c) for c in (clean_cfg.get("drop_columns") or []) if isinstance(c, str) and str(c).strip()
        ]
    else:
        column_transformations["drop_columns"] = [
            str(c) for c in column_transformations.get("drop_columns", []) if isinstance(c, str) and str(c).strip()
        ]
    if not isinstance(column_transformations.get("scale_columns"), list):
        column_transformations["scale_columns"] = [
            str(c) for c in (clean_cfg.get("scale_columns") or []) if isinstance(c, str) and str(c).strip()
        ]
    else:
        column_transformations["scale_columns"] = [
            str(c) for c in column_transformations.get("scale_columns", []) if isinstance(c, str) and str(c).strip()
        ]
    has_column_transformations = bool(
        column_transformations.get("drop_columns")
        or column_transformations.get("scale_columns")
        or ("drop_policy" in column_transformations and column_transformations.get("drop_policy") is not None)
        or (
            isinstance(column_transformations.get("feature_engineering"), list)
            and bool(column_transformations.get("feature_engineering"))
        )
        or (
            isinstance(column_transformations.get("dtype_conversion"), list)
            and bool(column_transformations.get("dtype_conversion"))
        )
    )

    output_path = _resolve_output_path(contract_full, contract_full, required_outputs) or ""
    manifest_path = _resolve_manifest_path(contract_full, contract_full, required_outputs) or ""
    declared_artifacts = get_declared_artifacts(contract_full)
    artifact_paths = {
        str(item.get("kind") or item.get("path") or f"artifact_{idx}"): str(item.get("path") or "")
        for idx, item in enumerate(declared_artifacts, start=1)
        if isinstance(item, dict) and str(item.get("path") or "").strip()
    }

    required_files: List[str] = []
    for entry in artifact_reqs.get("required_files", []) if isinstance(artifact_reqs.get("required_files"), list) else []:
        if isinstance(entry, dict):
            path = entry.get("path") or entry.get("output") or entry.get("artifact")
            if path:
                required_files.append(str(path))
        elif entry:
            required_files.append(str(entry))
    scored_rows_schema = artifact_reqs.get("scored_rows_schema")
    if not isinstance(scored_rows_schema, dict):
        scored_rows_schema = {}

    # Filter required_outputs for ML view: use deliverables-by-owner when available
    from src.utils.contract_accessors import get_deliverables_by_owner
    ml_deliverables_v41 = get_deliverables_by_owner(contract_full, "ml_engineer")
    if ml_deliverables_v41:
        ml_required_outputs_v41 = [d["path"] for d in ml_deliverables_v41 if d.get("required")]
    else:
        ml_required_outputs_v41 = list(required_outputs)

    artifact_payload: Dict[str, Any] = {"required_outputs": ml_required_outputs_v41}
    if required_files:
        artifact_payload["required_files"] = required_files
    if scored_rows_schema:
        artifact_payload["scored_rows_schema"] = scored_rows_schema
    file_schemas = artifact_reqs.get("file_schemas")
    if isinstance(file_schemas, dict) and file_schemas:
        artifact_payload["file_schemas"] = file_schemas

    column_dtype_targets = _resolve_column_dtype_targets({}, contract_full)

    strategy_title = str(contract_full.get("strategy_title") or "")
    business_objective = str(contract_full.get("business_objective") or "")
    reviewer_gates = get_reviewer_gates(contract_full)
    qa_gates = get_qa_gates(contract_full)
    cleaning_gates = get_cleaning_gates(contract_full)
    data_engineer_runbook = contract_full.get("data_engineer_runbook")
    if not isinstance(data_engineer_runbook, (dict, list, str)):
        data_engineer_runbook = {}
    if isinstance(data_engineer_runbook, str):
        data_engineer_runbook = data_engineer_runbook.strip()
        if not data_engineer_runbook:
            data_engineer_runbook = {}
    reporting_policy = contract_full.get("reporting_policy")
    if not isinstance(reporting_policy, dict):
        reporting_policy = {}
    expected_metrics = []
    primary_metric = validation_requirements.get("primary_metric")
    if isinstance(primary_metric, str) and primary_metric.strip():
        expected_metrics.append(primary_metric.strip())
    metrics_to_report = validation_requirements.get("metrics_to_report")
    if isinstance(metrics_to_report, list):
        for metric in metrics_to_report:
            if metric and str(metric) not in expected_metrics:
                expected_metrics.append(str(metric))
    id_columns = []
    for bucket in ("id", "identifier", "identifiers"):
        values = column_roles.get(bucket)
        if isinstance(values, list):
            id_columns.extend([str(c) for c in values if c])
    id_columns = list(dict.fromkeys(id_columns))

    de_view: DEView = {
        "role": "data_engineer",
        "task_semantics": task_semantics,
        "required_columns": de_required_columns,
        "optional_passthrough_columns": de_passthrough,
        "output_path": output_path,
        "required_columns_path": "data/required_columns.json",
        "column_sets_path": "data/column_sets.json",
        "cleaning_gates": cleaning_gates if isinstance(cleaning_gates, list) else [],
        "data_engineer_runbook": data_engineer_runbook,
        "constraints": {
            "scope": "cleaning_only",
            "hard_constraints": [
                "no_modeling",
                "no_score_fitting",
                "no_prescriptive_tuning",
                "no_analytics",
            ],
        },
    }
    if de_required_feature_selectors:
        de_view["required_feature_selectors"] = de_required_feature_selectors
    if outlier_policy and outlier_policy.get("apply_stage") in {"data_engineer", "both"}:
        de_view["outlier_policy"] = outlier_policy
        report_path = _resolve_de_outlier_report_path_from_policy(outlier_policy)
        if report_path:
            de_view["outlier_report_path"] = report_path
    if has_column_transformations:
        de_view["column_transformations"] = column_transformations
    if column_dtype_targets:
        de_view["column_dtype_targets"] = column_dtype_targets
    if manifest_path:
        de_view["output_manifest_path"] = manifest_path
    output_dialect = contract_full.get("output_dialect")
    if isinstance(output_dialect, dict) and output_dialect:
        de_view["output_dialect"] = output_dialect

    ml_view: MLView = {
        "role": "ml_engineer",
        "objective_type": objective_type,
        "task_semantics": task_semantics,
        "canonical_columns": canonical_columns,
        "derived_features": [c for c in derived_columns if c in set(model_features + segmentation_features)],
        "column_roles": column_roles,
        "decision_columns": decision_columns,
        "outcome_columns": outcome_columns,
        "audit_only_columns": audit_only_columns,
        "identifier_columns": id_columns,
        "allowed_feature_sets": allowed_feature_sets,
        "forbidden_features": forbidden_features,
        "column_dtype_targets": column_dtype_targets,
        "required_outputs": ml_required_outputs_v41,
        "validation_requirements": validation_requirements,
        "evaluation_spec": evaluation_spec,
        "objective_analysis": objective_analysis,
        "qa_gates": qa_gates if isinstance(qa_gates, list) else [],
        "reviewer_gates": reviewer_gates if isinstance(reviewer_gates, list) else [],
        "ml_engineer_runbook": ml_engineer_runbook,
        "artifact_requirements": artifact_payload,
        "cleaned_data_path": output_path,
        "cleaning_manifest_path": manifest_path,
        "decisioning_requirements": decisioning_requirements,
        "visual_requirements": visual_requirements,
    }
    if artifact_paths:
        ml_view["artifact_paths"] = artifact_paths
    if outlier_policy:
        ml_view["outlier_policy"] = outlier_policy
    if plot_spec is not None:
        ml_view["plot_spec"] = plot_spec
    split_spec = contract_full.get("split_spec")
    if isinstance(split_spec, dict) and split_spec:
        ml_view["split_spec"] = split_spec
    case_rules = contract_full.get("case_rules")
    if case_rules is not None:
        ml_view["case_rules"] = case_rules
    for opt_key in ("training_rows_rule", "scoring_rows_rule", "secondary_scoring_subset", "data_partitioning_notes"):
        value = contract_full.get(opt_key)
        if value:
            ml_view[opt_key] = value
    row_count_hints = _resolve_row_count_hints(contract_full, {})
    if row_count_hints:
        for key in ("n_train_rows", "n_test_rows", "n_total_rows"):
            if key in row_count_hints:
                ml_view[key] = row_count_hints[key]

    reviewer_summary_parts = [strategy_title.strip(), objective_type.strip()]
    reviewer_summary = " | ".join([part for part in reviewer_summary_parts if part]) or "contract_based_review"
    reviewer_view: ReviewerView = {
        "role": "reviewer",
        "objective_type": objective_type,
        "task_semantics": task_semantics,
        "reviewer_gates": reviewer_gates if isinstance(reviewer_gates, list) else [],
        "required_outputs": required_outputs,
        "expected_metrics": expected_metrics,
        "strategy_summary": reviewer_summary,
        "verification": {
            "required_outputs": required_outputs,
            "artifact_index_expected": bool(artifact_index),
        },
    }
    reviewer_view["decisioning_requirements"] = decisioning_requirements

    qa_artifact_payload: Dict[str, Any] = {
        "required_outputs": required_outputs,
        "optional_outputs": [str(item) for item in (artifact_reqs.get("optional_outputs") or []) if item],
    }
    file_schemas = artifact_reqs.get("file_schemas")
    if isinstance(file_schemas, dict) and file_schemas:
        qa_artifact_payload["file_schemas"] = file_schemas
    qa_view: QAView = {
        "role": "qa_reviewer",
        "task_semantics": task_semantics,
        "qa_gates": qa_gates if isinstance(qa_gates, list) else [],
        "artifact_requirements": qa_artifact_payload,
        "allowed_feature_sets": allowed_feature_sets,
        "column_roles": column_roles,
        "canonical_columns": canonical_columns,
        "objective_summary": {
            "strategy_title": strategy_title,
            "business_objective": business_objective,
        },
        "decisioning_requirements": decisioning_requirements,
    }
    if row_count_hints:
        for key in ("n_train_rows", "n_test_rows", "n_total_rows"):
            if key in row_count_hints:
                qa_view[key] = row_count_hints[key]
    if isinstance(split_spec, dict) and split_spec:
        qa_view["split_spec"] = split_spec
    if reporting_policy:
        qa_view["reporting_policy"] = reporting_policy

    cleaning_view: CleaningView = {
        "role": "cleaning_reviewer",
        "strategy_title": strategy_title,
        "business_objective": business_objective,
        "required_columns": de_required_columns,
        "column_sets_path": "data/column_sets.json",
        "dialect": output_dialect if isinstance(output_dialect, dict) else {},
        "cleaning_gates": cleaning_gates if isinstance(cleaning_gates, list) else [],
        "column_roles": column_roles,
        "allowed_feature_sets": allowed_feature_sets,
    }
    if de_required_feature_selectors:
        cleaning_view["required_feature_selectors"] = de_required_feature_selectors
    if outlier_policy and outlier_policy.get("apply_stage") in {"data_engineer", "both"}:
        cleaning_view["outlier_policy"] = outlier_policy
        report_path = _resolve_de_outlier_report_path_from_policy(outlier_policy)
        if report_path:
            cleaning_view["outlier_report_path"] = report_path
    if has_column_transformations:
        cleaning_view["column_transformations"] = column_transformations
    if cleaning_code and isinstance(cleaning_code, str):
        max_code_len = 8000
        cleaning_view["cleaning_code"] = cleaning_code[:max_code_len] if len(cleaning_code) > max_code_len else cleaning_code

    evidence = _normalize_artifact_index(artifact_index)
    translator_view: TranslatorView = {
        "role": "translator",
        "reporting_policy": reporting_policy,
        "evidence_inventory": evidence,
        "key_decisions": [
            f"objective_type:{objective_type}",
            f"required_outputs:{len(required_outputs)}",
        ],
        "limitations": [str(item) for item in (contract_full.get("data_risks") or []) if item],
        "constraints": {"no_markdown_tables": True, "cite_sources": True},
        "decisioning_requirements": decisioning_requirements,
        "visual_requirements": visual_requirements,
    }
    if plot_spec is not None:
        translator_view["plot_spec"] = plot_spec

    results_advisor_view: ResultsAdvisorView = {
        "role": "results_advisor",
        "objective_type": objective_type,
        "reporting_policy": reporting_policy,
        "evidence_inventory": evidence,
    }

    return {
        "de_view": trim_to_budget(de_view, 8000),
        "ml_view": trim_to_budget(ml_view, 16000),
        "cleaning_view": trim_to_budget(cleaning_view, 15000),
        "qa_view": trim_to_budget(qa_view, 12000),
        "reviewer_view": trim_to_budget(reviewer_view, 12000),
        "translator_view": trim_to_budget(translator_view, 16000),
        "results_advisor_view": trim_to_budget(results_advisor_view, 12000),
    }


def sanitize_contract_min_for_de(contract_min: Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(contract_min, dict):
        return {}
    sanitized = dict(contract_min)
    sanitized.pop("business_objective", None)
    return sanitized


def persist_views(
    views: Dict[str, Dict[str, Any]],
    base_dir: str = "data",
    run_bundle_dir: Optional[str] = None,
) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    if not views:
        return paths
    rel_dir = os.path.join("contracts", "views")
    out_dir = os.path.join(base_dir, rel_dir)
    os.makedirs(out_dir, exist_ok=True)
    for name, payload in views.items():
        if not payload:
            continue
        filename = f"{name}.json"
        path = os.path.join(out_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        paths[name] = path
        if run_bundle_dir:
            bundle_path = os.path.join(run_bundle_dir, rel_dir, filename)
            os.makedirs(os.path.dirname(bundle_path), exist_ok=True)
            with open(bundle_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
    return paths
