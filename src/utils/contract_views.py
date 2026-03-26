from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, List, Optional


# ── Budget trimming (used by ml_engineer.py) ────────────────────────

_PRESERVE_KEYS = {
    "task_semantics",
    "canonical_columns",
    "column_roles",
    "allowed_feature_sets",
    "model_features",
    "required_columns",
    "optional_passthrough_columns",
    "required_feature_selectors",
    "column_dtype_targets",
    "column_resolution_context",
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


def _truncate_text(value: str, max_len: int) -> str:
    if not isinstance(value, str) or len(value) <= max_len:
        return value
    return value[: max_len - 14] + "...[TRUNCATED]"


def _trim_value(
    obj: Any,
    max_str_len: int,
    max_list_items: int,
    preserve_keys: set[str],
    path: List[str],
) -> Any:
    preserve_subtree = any(segment in preserve_keys for segment in path if segment and segment != "[]")
    if isinstance(obj, str):
        if preserve_subtree:
            return obj
        return _truncate_text(obj, max_str_len)
    if isinstance(obj, list):
        key = path[-1] if path else ""
        if not preserve_subtree and key not in preserve_keys and len(obj) > max_list_items:
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


# ── V5 view construction ────────────────────────────────────────────


def _build_views_v5(contract: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Build agent views by trivial merge from v5 hierarchical contract.

    View formulas:
      de_view            = shared + data_engineer
      ml_view            = shared + ml_engineer
      cleaning_view      = shared + data_engineer + cleaning_reviewer
      qa_view            = shared + ml_engineer   + qa_reviewer
      reviewer_view      = shared + ml_engineer
      translator_view    = shared + business_translator
      results_advisor_view = shared
    """
    shared = contract.get("shared") or {}
    de = contract.get("data_engineer") or {}
    ml = contract.get("ml_engineer") or {}
    cr = contract.get("cleaning_reviewer") or {}
    qa = contract.get("qa_reviewer") or {}
    bt = contract.get("business_translator") or {}

    return {
        "de_view": {**shared, **de, "role": "data_engineer"},
        "ml_view": {**shared, **ml, "role": "ml_engineer"},
        "cleaning_view": {**shared, **de, **cr, "role": "cleaning_reviewer"},
        "qa_view": {**shared, **ml, **qa, "role": "qa_reviewer"},
        "reviewer_view": {**shared, **ml, "role": "reviewer"},
        "translator_view": {**shared, **bt, "role": "translator"},
        "results_advisor_view": {**shared, "role": "results_advisor"},
    }


def build_contract_views_projection(
    contract_full: Dict[str, Any] | None,
    artifact_index: Any = None,
    cleaning_code: Optional[str] = None,
    data_profile: Optional[Dict[str, Any]] = None,
    dataset_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Build agent views as pure projection from validated execution contract.

    V5.0 contracts use trivial merge (shared + agent section).
    """
    contract_full = contract_full if isinstance(contract_full, dict) else {}

    # If the contract was flattened, use the original v5 hierarchy for views.
    v5_original = contract_full.get("_v5_original") if isinstance(contract_full, dict) else None
    if isinstance(v5_original, dict) and str(v5_original.get("contract_version", "")).startswith("5"):
        return _build_views_v5(v5_original)
    if str(contract_full.get("contract_version", "")).startswith("5") and "shared" in contract_full:
        return _build_views_v5(contract_full)

    # No legacy v4 fallback — return empty views with role markers
    return {
        "de_view": {"role": "data_engineer"},
        "ml_view": {"role": "ml_engineer"},
        "cleaning_view": {"role": "cleaning_reviewer"},
        "qa_view": {"role": "qa_reviewer"},
        "reviewer_view": {"role": "reviewer"},
        "translator_view": {"role": "translator"},
        "results_advisor_view": {"role": "results_advisor"},
    }


# ── Persistence ─────────────────────────────────────────────────────


def persist_view_projection_reports(
    reports: Dict[str, Dict[str, Any]],
    base_dir: str = "data",
    run_bundle_dir: Optional[str] = None,
) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    if not reports:
        return paths
    rel_dir = os.path.join("contracts", "view_projection_reports")
    out_dir = os.path.join(base_dir, rel_dir)
    os.makedirs(out_dir, exist_ok=True)
    for name, payload in reports.items():
        if not isinstance(payload, dict) or not payload:
            continue
        filename = f"{name}_coverage.json"
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
