from __future__ import annotations

import copy
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
    get_dataset_artifact_binding,
    get_declared_artifacts,
    get_deliverables_by_owner,
    get_enriched_dataset_output_path,
    get_outlier_policy,
    get_outcome_columns,
    get_qa_gates,
    get_reviewer_gates,
    get_required_outputs,
    get_required_outputs_by_owner,
    get_task_semantics,
    get_validation_requirements,
)
from src.utils.problem_capabilities import infer_problem_capabilities, resolve_problem_capabilities_from_contract
from src.utils.contract_validator import resolve_contract_active_workstreams

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
    scope: str
    active_workstreams: Dict[str, Any]
    future_ml_handoff: Dict[str, Any]
    task_semantics: Dict[str, Any]
    canonical_columns: List[str]
    column_roles: Dict[str, List[str]]
    allowed_feature_sets: Any
    model_features: List[str]
    required_outputs: List[str]
    required_columns: List[str]
    required_feature_selectors: List[Dict[str, Any]]
    optional_passthrough_columns: List[str]
    column_dtype_targets: Dict[str, Dict[str, Any]]
    column_resolution_context: Dict[str, Dict[str, Any]]
    column_resolution_context_path: str
    column_transformations: Dict[str, Any]
    output_path: str
    output_manifest_path: str
    artifact_requirements: Dict[str, Any]
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
    scope: str
    active_workstreams: Dict[str, Any]
    future_ml_handoff: Dict[str, Any]
    objective_type: str
    primary_metric: str
    metric_definition_rule: str
    task_semantics: Dict[str, Any]
    canonical_columns: List[str]
    column_roles: Dict[str, List[str]]
    allowed_feature_sets: Any
    model_features: List[str]
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
    cleaning_manifest_path: str
    cleaned_data_path: str
    outlier_policy: Dict[str, Any]
    split_spec: Dict[str, Any]
    n_train_rows: int
    n_test_rows: int
    n_total_rows: int


class ReviewerView(TypedDict, total=False):
    role: str
    scope: str
    active_workstreams: Dict[str, Any]
    future_ml_handoff: Dict[str, Any]
    objective_type: str
    task_semantics: Dict[str, Any]
    reviewer_gates: List[Any]
    required_outputs: List[str]
    expected_metrics: List[str]
    strategy_summary: str
    verification: Dict[str, Any]


class TranslatorView(TypedDict, total=False):
    role: str
    scope: str
    active_workstreams: Dict[str, Any]
    future_ml_handoff: Dict[str, Any]
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
    scope: str
    active_workstreams: Dict[str, Any]
    future_ml_handoff: Dict[str, Any]
    task_semantics: Dict[str, Any]
    strategy_title: str
    business_objective: str
    canonical_columns: List[str]
    model_features: List[str]
    required_outputs: List[str]
    required_columns: List[str]
    required_feature_selectors: List[Dict[str, Any]]
    column_resolution_context: Dict[str, Dict[str, Any]]
    column_resolution_context_path: str
    column_sets_path: str
    column_transformations: Dict[str, Any]
    artifact_requirements: Dict[str, Any]
    dialect: Dict[str, Any]
    cleaning_gates: List[Dict[str, Any]]
    column_roles: Dict[str, List[str]]
    allowed_feature_sets: Any
    outlier_policy: Dict[str, Any]
    outlier_report_path: str


class QAView(TypedDict, total=False):
    role: str
    scope: str
    active_workstreams: Dict[str, Any]
    future_ml_handoff: Dict[str, Any]
    task_semantics: Dict[str, Any]
    review_subject: str
    subject_required_outputs: List[str]
    qa_required_outputs: List[str]
    artifacts_to_verify: List[str]
    subject_code_path_hint: str
    qa_gates: List[Dict[str, Any]]
    artifact_requirements: Dict[str, Any]
    model_features: List[str]
    allowed_feature_sets: Any
    column_roles: Dict[str, List[str]]
    canonical_columns: List[str]
    objective_summary: Dict[str, str]
    reporting_policy: Dict[str, Any]
    n_train_rows: int
    n_test_rows: int
    n_total_rows: int
    split_spec: Dict[str, Any]


_AGENT_INTERFACE_ALIASES: Dict[str, Tuple[str, ...]] = {
    "data_engineer": ("data_engineer", "de_view"),
    "ml_engineer": ("ml_engineer", "ml_view"),
    "cleaning_reviewer": ("cleaning_reviewer", "cleaning_view"),
    "qa_reviewer": ("qa_reviewer", "qa_view"),
    "reviewer": ("reviewer", "reviewer_view"),
    "translator": ("translator", "translator_view"),
    "results_advisor": ("results_advisor", "results_advisor_view"),
}

_INVARIANT_VIEW_KEYS = {"role", "scope", "active_workstreams", "future_ml_handoff", "task_semantics"}
_CANONICAL_VIEW_KEYS: Dict[str, set[str]] = {
    "data_engineer": {
        "canonical_columns",
        "column_roles",
        "allowed_feature_sets",
        "model_features",
        "required_outputs",
        "required_columns",
        "required_feature_selectors",
        "optional_passthrough_columns",
        "column_dtype_targets",
        "column_resolution_context",
        "column_resolution_context_path",
        "column_transformations",
        "output_path",
        "output_manifest_path",
        "artifact_requirements",
        "cleaning_gates",
        "data_engineer_runbook",
        "outlier_policy",
        "outlier_report_path",
        "output_dialect",
    },
    "ml_engineer": {
        "objective_type",
        "primary_metric",
        "metric_definition_rule",
        "canonical_columns",
        "derived_features",
        "column_roles",
        "decision_columns",
        "outcome_columns",
        "audit_only_columns",
        "identifier_columns",
        "allowed_feature_sets",
        "model_features",
        "forbidden_features",
        "column_dtype_targets",
        "required_outputs",
        "validation_requirements",
        "evaluation_spec",
        "objective_analysis",
        "qa_gates",
        "reviewer_gates",
        "ml_engineer_runbook",
        "artifact_requirements",
        "artifact_paths",
        "cleaning_manifest_path",
        "cleaned_data_path",
        "outlier_policy",
        "split_spec",
        "n_train_rows",
        "n_test_rows",
        "n_total_rows",
        "decisioning_requirements",
        "visual_requirements",
        "plot_spec",
        "case_rules",
        "training_rows_rule",
        "scoring_rows_rule",
        "secondary_scoring_subset",
        "data_partitioning_notes",
    },
    "cleaning_reviewer": {
        "strategy_title",
        "business_objective",
        "canonical_columns",
        "model_features",
        "required_outputs",
        "required_columns",
        "required_feature_selectors",
        "column_resolution_context",
        "column_resolution_context_path",
        "column_transformations",
        "cleaning_gates",
        "column_roles",
        "allowed_feature_sets",
        "outlier_policy",
        "outlier_report_path",
        "dialect",
        "cleaning_code",
    },
    "qa_reviewer": {
        "review_subject",
        "subject_required_outputs",
        "qa_required_outputs",
        "artifacts_to_verify",
        "subject_code_path_hint",
        "qa_gates",
        "artifact_requirements",
        "model_features",
        "allowed_feature_sets",
        "column_roles",
        "canonical_columns",
        "objective_summary",
        "reporting_policy",
        "split_spec",
        "n_train_rows",
        "n_test_rows",
        "n_total_rows",
        "decisioning_requirements",
    },
    "reviewer": {
        "reviewer_gates",
        "required_outputs",
        "expected_metrics",
        "strategy_summary",
        "verification",
        "decisioning_requirements",
    },
    "translator": {
        "reporting_policy",
        "evidence_inventory",
        "decisioning_requirements",
        "visual_requirements",
        "plot_spec",
    },
    "results_advisor": {
        "objective_type",
        "reporting_policy",
        "evidence_inventory",
    },
}
_PROTECTED_SEMANTIC_LIST_KEYS = {
    "required_outputs",
    "cleaning_gates",
    "qa_gates",
    "reviewer_gates",
}

_VIEW_PROJECTION_SPECS: Dict[str, Dict[str, Any]] = {
    "data_engineer": {
        "role": "data_engineer",
        "budget": 8000,
        "constants": {
            "required_columns_path": "data/required_columns.json",
            "column_sets_path": "data/column_sets.json",
        },
        "always_fields": [
            "data_engineer_runbook",
            "optional_passthrough_columns",
            "model_features",
        ],
        "fields": [
            "scope",
            "active_workstreams",
            "future_ml_handoff",
            "task_semantics",
            "canonical_columns",
            "column_roles",
            "allowed_feature_sets",
            "model_features",
            "required_outputs",
            "required_columns",
            "required_feature_selectors",
            "optional_passthrough_columns",
            "column_dtype_targets",
            "column_resolution_context",
            "column_resolution_context_path",
            "column_transformations",
            "artifact_requirements",
            "output_path",
            "output_manifest_path",
            "output_dialect",
            "cleaning_gates",
            "data_engineer_runbook",
            "outlier_policy",
            "outlier_report_path",
            "column_sets_summary",
            "constraints",
        ],
    },
    "ml_engineer": {
        "role": "ml_engineer",
        "budget": 16000,
        "always_fields": [
            "model_features",
            "forbidden_features",
            "validation_requirements",
            "visual_requirements",
        ],
        "fields": [
            "scope",
            "active_workstreams",
            "future_ml_handoff",
            "objective_type",
            "primary_metric",
            "metric_definition_rule",
            "task_semantics",
            "canonical_columns",
            "derived_features",
            "column_roles",
            "decision_columns",
            "outcome_columns",
            "audit_only_columns",
            "identifier_columns",
            "allowed_feature_sets",
            "model_features",
            "forbidden_features",
            "column_dtype_targets",
            "required_outputs",
            "validation_requirements",
            "evaluation_spec",
            "objective_analysis",
            "qa_gates",
            "reviewer_gates",
            "ml_engineer_runbook",
            "case_rules",
            "plot_spec",
            "artifact_requirements",
            "artifact_paths",
            "cleaning_manifest_path",
            "cleaned_data_path",
            "outlier_policy",
            "split_spec",
            "n_train_rows",
            "n_test_rows",
            "n_total_rows",
            "column_sets_summary",
            "training_rows_rule",
            "scoring_rows_rule",
            "secondary_scoring_subset",
            "data_partitioning_notes",
            "identifier_policy",
            "identifier_overrides",
            "decisioning_requirements",
            "visual_requirements",
            "view_warnings",
        ],
    },
    "reviewer": {
        "role": "reviewer",
        "budget": 12000,
        "fields": [
            "scope",
            "active_workstreams",
            "future_ml_handoff",
            "objective_type",
            "task_semantics",
            "reviewer_gates",
            "required_outputs",
            "expected_metrics",
            "strategy_summary",
            "verification",
            "decisioning_requirements",
        ],
    },
    "qa_reviewer": {
        "role": "qa_reviewer",
        "budget": 12000,
        "always_fields": [
            "review_subject",
            "subject_required_outputs",
            "qa_required_outputs",
            "artifacts_to_verify",
            "subject_code_path_hint",
        ],
        "fields": [
            "scope",
            "active_workstreams",
            "future_ml_handoff",
            "task_semantics",
            "review_subject",
            "subject_required_outputs",
            "qa_required_outputs",
            "artifacts_to_verify",
            "subject_code_path_hint",
            "qa_gates",
            "artifact_requirements",
            "model_features",
            "allowed_feature_sets",
            "column_roles",
            "canonical_columns",
            "objective_summary",
            "reporting_policy",
            "split_spec",
            "n_train_rows",
            "n_test_rows",
            "n_total_rows",
            "decisioning_requirements",
        ],
    },
    "cleaning_reviewer": {
        "role": "cleaning_reviewer",
        "budget": 15000,
        "constants": {
            "column_sets_path": "data/column_sets.json",
        },
        "fields": [
            "scope",
            "active_workstreams",
            "future_ml_handoff",
            "task_semantics",
            "strategy_title",
            "business_objective",
            "canonical_columns",
            "model_features",
            "required_outputs",
            "required_columns",
            "required_feature_selectors",
            "column_resolution_context",
            "column_resolution_context_path",
        "column_transformations",
        "artifact_requirements",
        "dialect",
            "cleaning_gates",
            "column_roles",
            "allowed_feature_sets",
            "outlier_policy",
            "outlier_report_path",
            "cleaning_code",
        ],
    },
    "translator": {
        "role": "translator",
        "budget": 16000,
        "fields": [
            "scope",
            "active_workstreams",
            "future_ml_handoff",
            "reporting_policy",
            "plot_spec",
            "evidence_inventory",
            "key_decisions",
            "limitations",
            "constraints",
            "view_warnings",
            "visual_requirements",
            "decisioning_requirements",
        ],
    },
    "results_advisor": {
        "role": "results_advisor",
        "budget": 12000,
        "fields": [
            "objective_type",
            "reporting_policy",
            "evidence_inventory",
        ],
    },
}


def _deep_merge_view_payload(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged: Dict[str, Any] = {str(k): copy.deepcopy(v) for k, v in base.items()}
        for key, value in override.items():
            existing = merged.get(key)
            merged[key] = _deep_merge_view_payload(existing, value)
        return merged
    return copy.deepcopy(override)


def _is_emptyish(value: Any) -> bool:
    return value in (None, "", [], {})


def _build_declared_agent_view(
    interface_key: str,
    projection_context: Dict[str, Any],
    contract_min: Dict[str, Any],
    contract_full: Dict[str, Any],
) -> Dict[str, Any]:
    spec = _VIEW_PROJECTION_SPECS[interface_key]
    view: Dict[str, Any] = {"role": spec["role"]}
    always_fields = set(spec.get("always_fields", []))
    constants = spec.get("constants")
    if isinstance(constants, dict):
        for key, value in constants.items():
            view[key] = copy.deepcopy(value)
    for field in spec.get("fields", []):
        if field not in projection_context:
            continue
        value = projection_context.get(field)
        if field not in always_fields and _is_emptyish(value):
            continue
        view[field] = copy.deepcopy(value)
    return _finalize_agent_view(view, int(spec["budget"]), contract_min, contract_full, interface_key)


def _sequence_identity(item: Any) -> str:
    if isinstance(item, dict):
        for key in ("path", "output_path", "artifact", "file"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return f"path:{value.strip().replace('\\', '/').lower()}"
        for key in ("name", "id", "gate", "intent"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return f"{key}:{value.strip().lower()}"
        try:
            return "json:" + json.dumps(item, sort_keys=True, ensure_ascii=True)
        except Exception:
            return f"repr:{repr(item)}"
    return f"scalar:{str(item).strip().lower()}"


def _merge_additive_list(base: List[Any], override: List[Any]) -> List[Any]:
    merged = [copy.deepcopy(item) for item in base]
    seen = {_sequence_identity(item) for item in merged}
    for item in override:
        identity = _sequence_identity(item)
        if identity in seen:
            continue
        seen.add(identity)
        merged.append(copy.deepcopy(item))
    return merged


def _merge_lossless_value(base: Any, override: Any) -> Any:
    if _is_emptyish(base):
        return copy.deepcopy(override)
    if isinstance(base, dict) and isinstance(override, dict):
        merged: Dict[str, Any] = {str(k): copy.deepcopy(v) for k, v in base.items()}
        for key, value in override.items():
            if key in merged:
                merged[key] = _merge_lossless_value(merged[key], value)
            else:
                merged[key] = copy.deepcopy(value)
        return merged
    if isinstance(base, list) and isinstance(override, list):
        return _merge_additive_list(base, override)
    return copy.deepcopy(base)


def _merge_protected_list(key: str, base: List[Any], override: List[Any]) -> List[Any]:
    if not base:
        return [copy.deepcopy(item) for item in override]
    if key not in _PROTECTED_SEMANTIC_LIST_KEYS:
        return [copy.deepcopy(item) for item in base]
    override_by_identity = {_sequence_identity(item): item for item in override}
    merged: List[Any] = []
    for item in base:
        identity = _sequence_identity(item)
        if identity in override_by_identity and isinstance(item, dict) and isinstance(override_by_identity[identity], dict):
            merged.append(_merge_lossless_value(item, override_by_identity[identity]))
        else:
            merged.append(copy.deepcopy(item))
    return merged


def _merge_agent_interface_payload(
    base_view: Dict[str, Any],
    explicit_interface: Dict[str, Any],
    interface_key: str,
) -> Dict[str, Any]:
    merged: Dict[str, Any] = copy.deepcopy(base_view)
    protected_keys = _CANONICAL_VIEW_KEYS.get(interface_key, set()) | _INVARIANT_VIEW_KEYS
    for key, value in explicit_interface.items():
        existing = merged.get(key)
        if key in protected_keys:
            if isinstance(existing, list) and isinstance(value, list):
                merged[key] = _merge_protected_list(key, existing, value)
            else:
                merged[key] = _merge_lossless_value(existing, value)
            continue
        if isinstance(existing, list) and isinstance(value, list):
            merged[key] = _merge_additive_list(existing, value)
        elif isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge_view_payload(existing, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _resolve_agent_interface_payload(
    contract_min: Dict[str, Any],
    contract_full: Dict[str, Any],
    interface_key: str,
) -> Dict[str, Any]:
    aliases = _AGENT_INTERFACE_ALIASES.get(interface_key, (interface_key,))
    resolved: Dict[str, Any] = {}
    for source in (contract_full, contract_min):
        if not isinstance(source, dict):
            continue
        agent_interfaces = source.get("agent_interfaces")
        if not isinstance(agent_interfaces, dict):
            continue
        for alias in aliases:
            candidate = agent_interfaces.get(alias)
            if isinstance(candidate, dict) and candidate:
                resolved = _deep_merge_view_payload(resolved, candidate)
    return resolved


def _finalize_agent_view(
    view: Dict[str, Any],
    budget: int,
    contract_min: Dict[str, Any],
    contract_full: Dict[str, Any],
    interface_key: str,
) -> Dict[str, Any]:
    base_view = copy.deepcopy(view)
    explicit_interface = _resolve_agent_interface_payload(contract_min, contract_full, interface_key)
    if explicit_interface:
        view = _merge_agent_interface_payload(view, explicit_interface, interface_key)
        for invariant_key in _INVARIANT_VIEW_KEYS:
            if invariant_key in base_view:
                view[invariant_key] = copy.deepcopy(base_view[invariant_key])
    return trim_to_budget(view, budget)


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
_DEFAULT_COLUMN_RESOLUTION_CONTEXT_PATH = "data/column_resolution_context.json"
_NULLISH_TOKENS = {
    "",
    "nan",
    "na",
    "n/a",
    "null",
    "none",
    "missing",
    "unknown",
    "not available",
    "not_applicable",
    "not_applicable",
}


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


def _resolve_artifact_requirements_payload(
    contract_min: Dict[str, Any],
    contract_full: Dict[str, Any],
) -> Dict[str, Any]:
    artifact_reqs = _coerce_dict(contract_min.get("artifact_requirements")) or _coerce_dict(
        contract_full.get("artifact_requirements")
    )
    return artifact_reqs


def _resolve_dataset_artifact_binding(
    contract_min: Dict[str, Any],
    contract_full: Dict[str, Any],
    binding_name: str,
) -> Dict[str, Any]:
    combined = contract_full if isinstance(contract_full, dict) else {}
    if isinstance(contract_min, dict) and contract_min:
        combined = {**combined, **contract_min}
    binding = get_dataset_artifact_binding(combined, binding_name)
    if isinstance(binding, dict):
        return binding
    artifact_reqs = _resolve_artifact_requirements_payload(contract_min, contract_full)
    aliases = [str(binding_name or "").strip()]
    if binding_name == "cleaned_dataset":
        aliases.append("clean_dataset")
    for alias in aliases:
        candidate = _coerce_dict(artifact_reqs.get(alias))
        if candidate:
            return candidate
    return {}


def _project_de_artifact_requirements(
    contract_min: Dict[str, Any],
    contract_full: Dict[str, Any],
) -> Dict[str, Any]:
    artifact_reqs = _resolve_artifact_requirements_payload(contract_min, contract_full)
    payload: Dict[str, Any] = {}
    if isinstance(artifact_reqs.get("cleaned_dataset"), dict):
        payload["cleaned_dataset"] = copy.deepcopy(artifact_reqs.get("cleaned_dataset"))
    elif isinstance(artifact_reqs.get("clean_dataset"), dict):
        payload["clean_dataset"] = copy.deepcopy(artifact_reqs.get("clean_dataset"))
    if isinstance(artifact_reqs.get("enriched_dataset"), dict):
        payload["enriched_dataset"] = copy.deepcopy(artifact_reqs.get("enriched_dataset"))
    schema_binding = _coerce_dict(artifact_reqs.get("schema_binding"))
    if schema_binding:
        payload["schema_binding"] = copy.deepcopy(schema_binding)
    return payload


def _first_value(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _resolve_contract_execution_context(
    contract_min: Dict[str, Any],
    contract_full: Dict[str, Any],
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    source = contract_min if isinstance(contract_min, dict) and contract_min else contract_full
    source = source if isinstance(source, dict) else {}
    scope = str(source.get("scope") or "").strip() or "cleaning_only"
    active_workstreams = resolve_contract_active_workstreams(source)
    future_ml_handoff = source.get("future_ml_handoff")
    if not isinstance(future_ml_handoff, dict):
        future_ml_handoff = {}
    return scope, active_workstreams, future_ml_handoff


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


def _resolve_required_outputs_for_owner(
    contract_min: Dict[str, Any],
    contract_full: Dict[str, Any],
    owner: str,
    fallback_paths: Optional[List[str]] = None,
) -> List[str]:
    resolved: List[str] = []
    seen: set[str] = set()
    for source in (contract_full, contract_min):
        if not isinstance(source, dict):
            continue
        for path in get_required_outputs_by_owner(source, owner):
            normalized = str(path or "").replace("\\", "/").strip()
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            resolved.append(normalized)
    if not resolved:
        for path in fallback_paths or []:
            normalized = str(path or "").replace("\\", "/").strip()
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            resolved.append(normalized)
    return resolved


def _resolve_qa_review_subject(
    contract_min: Dict[str, Any],
    contract_full: Dict[str, Any],
) -> str:
    _, active_workstreams, _ = _resolve_contract_execution_context(contract_min, contract_full)
    de_outputs = _resolve_required_outputs_for_owner(contract_min, contract_full, "data_engineer")
    ml_outputs = _resolve_required_outputs_for_owner(contract_min, contract_full, "ml_engineer")
    model_training = bool(active_workstreams.get("model_training"))
    if model_training and ml_outputs:
        return "ml_engineer"
    if de_outputs:
        return "data_engineer"
    if ml_outputs:
        return "ml_engineer"
    scope = str(_first_value(contract_min.get("scope"), contract_full.get("scope")) or "").strip().lower()
    if scope in {"ml_only", "full_pipeline"} or model_training:
        return "ml_engineer"
    return "data_engineer"


def _default_subject_code_path(review_subject: str) -> str:
    subject = str(review_subject or "").strip().lower()
    if subject == "data_engineer":
        return "artifacts/data_engineer_last.py"
    return "artifacts/ml_engineer_last.py"


def _resolve_required_columns(contract_min: Dict[str, Any], contract_full: Dict[str, Any]) -> List[str]:
    clean_cfg = _resolve_dataset_artifact_binding(contract_min, contract_full, "cleaned_dataset")
    required = clean_cfg.get("required_columns")
    if isinstance(required, list) and required:
        return [str(c) for c in required if c]
    artifact_reqs = _resolve_artifact_requirements_payload(contract_min, contract_full)
    schema_binding = _coerce_dict(artifact_reqs.get("schema_binding"))
    required = schema_binding.get("required_columns")
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
    clean_cfg = _resolve_dataset_artifact_binding(contract_min, contract_full, "cleaned_dataset")
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
    required_set = {str(c) for c in required_columns if c}
    artifact_reqs = _resolve_artifact_requirements_payload(contract_min, contract_full)

    def _normalize(raw: Any) -> List[str]:
        if not isinstance(raw, list):
            return []
        return [str(c) for c in raw if isinstance(c, str) and str(c).strip()]

    def _declared_optional_passthrough() -> tuple[bool, List[str]]:
        clean_cfg = _resolve_dataset_artifact_binding(contract_min, contract_full, "cleaned_dataset")
        if "optional_passthrough_columns" in clean_cfg and isinstance(clean_cfg.get("optional_passthrough_columns"), list):
            return True, _normalize(clean_cfg.get("optional_passthrough_columns"))
        schema_binding = _coerce_dict(artifact_reqs.get("schema_binding"))
        if "optional_passthrough_columns" in schema_binding and isinstance(
            schema_binding.get("optional_passthrough_columns"), list
        ):
            return True, _normalize(schema_binding.get("optional_passthrough_columns"))
        return False, []

    declared_present, declared_passthrough = _declared_optional_passthrough()
    if declared_present:
        return [c for c in declared_passthrough if c not in required_set]

    allowed_sets = _resolve_allowed_feature_sets(contract_min, contract_full)
    audit_only = allowed_sets.get("audit_only_features") if isinstance(allowed_sets, dict) else None
    passthrough = _normalize(audit_only)
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
    clean_cfg = _resolve_dataset_artifact_binding(contract_min, contract_full, "cleaned_dataset")
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

    clean_cfg = _resolve_dataset_artifact_binding(contract_min, contract_full, "cleaned_dataset")
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


def _profile_columns(data_profile: Dict[str, Any], dataset_profile: Dict[str, Any]) -> List[str]:
    for source in (data_profile, dataset_profile):
        if not isinstance(source, dict):
            continue
        basic_stats = source.get("basic_stats")
        if isinstance(basic_stats, dict):
            columns = basic_stats.get("columns")
            if isinstance(columns, list) and columns:
                return [str(col) for col in columns if str(col).strip()]
        columns = source.get("columns")
        if isinstance(columns, list) and columns:
            return [str(col) for col in columns if str(col).strip()]
        column_inventory = source.get("column_inventory")
        if isinstance(column_inventory, list) and column_inventory:
            return [str(col) for col in column_inventory if str(col).strip()]
    return []


def _profile_map(
    data_profile: Dict[str, Any],
    dataset_profile: Dict[str, Any],
    *keys: str,
) -> Dict[str, Any]:
    for source in (data_profile, dataset_profile):
        if not isinstance(source, dict):
            continue
        for key in keys:
            value = source.get(key)
            if isinstance(value, dict) and value:
                return value
    return {}


def _profile_top_values(
    data_profile: Dict[str, Any],
    dataset_profile: Dict[str, Any],
    column: str,
) -> List[Dict[str, Any]]:
    column = str(column or "").strip()
    if not column:
        return []
    for source in (data_profile, dataset_profile):
        if not isinstance(source, dict):
            continue
        cardinality = source.get("cardinality")
        if not isinstance(cardinality, dict):
            continue
        entry = cardinality.get(column)
        if not isinstance(entry, dict):
            continue
        top_values = entry.get("top_values")
        if not isinstance(top_values, list):
            continue
        normalized: List[Dict[str, Any]] = []
        for item in top_values:
            if isinstance(item, dict):
                value = item.get("value")
                if value is None:
                    continue
                normalized.append(
                    {
                        "value": str(value),
                        "count": item.get("count"),
                    }
                )
            elif item is not None:
                normalized.append({"value": str(item), "count": None})
        if normalized:
            return normalized
    return []


def _collect_known_column_mentions(value: Any, known_columns: set[str]) -> List[str]:
    matches: List[str] = []

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            for nested in node.values():
                _walk(nested)
            return
        if isinstance(node, list):
            for nested in node:
                _walk(nested)
            return
        if isinstance(node, str):
            token = node.strip()
            if token and token in known_columns and token not in matches:
                matches.append(token)

    _walk(value)
    return matches


def _value_format_families(raw: str) -> List[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    lowered = text.lower()
    families: List[str] = []

    def _add(name: str) -> None:
        if name not in families:
            families.append(name)

    if lowered in _NULLISH_TOKENS:
        _add("placeholder_token")
    if re.search(r"[$€£¥]", text):
        _add("currency_symbol")
    if re.search(r"\d\s*[kmb]$", lowered):
        _add("magnitude_suffix")
    if "%" in text:
        _add("percent_symbol")
    if re.search(r"^[<>?~]", text) or "?" in text:
        _add("noisy_prefix_or_symbol")
    if re.search(r"^\d{4}-\d{2}-\d{2}$", text):
        _add("iso_date")
    if re.search(r"^\d{4}-\d{2}-\d{2}[ t]\d{1,2}:\d{2}", text.lower()):
        _add("timestamp_with_time")
    if re.search(r"^\d{4}/\d{1,2}/\d{1,2}", text):
        _add("ymd_slash_date")
    if re.search(r"^\d{1,2}/\d{1,2}/\d{4}", text):
        _add("slash_date")
    if re.search(r"^\d{1,2}-\d{1,2}-\d{4}", text):
        _add("dash_date")
    if re.search(r"\d{1,2}:\d{2}", text):
        _add("time_component")
    if re.search(r"^\d{1,3}(\.\d{3})+(,\d+)?$", text.lstrip("?")):
        _add("thousands_dot")
    if re.search(r"^\d{1,3}(,\d{3})+(\.\d+)?$", text.lstrip("?")):
        _add("thousands_comma")
    if re.search(r"\d,\d", text):
        _add("decimal_comma")
    if re.search(r"\d\.\d", text):
        _add("decimal_dot")
    if lowered in {"true", "false", "yes", "no", "y", "n", "0", "1"}:
        _add("boolean_token")
    if "@" in text and "." in text:
        _add("email_like")

    if "slash_date" in families or "dash_date" in families:
        parts = re.split(r"[/\-]", text.split(" ")[0])
        if len(parts) >= 3:
            try:
                first = int(parts[0])
                second = int(parts[1])
            except Exception:
                first = second = -1
            if 0 < first <= 12 and 0 < second <= 12:
                _add("ambiguous_day_month_order")

    return families


def _infer_semantic_kind(
    column: str,
    target_info: Dict[str, Any],
    observed_dtype: str,
    examples: List[str],
) -> str:
    name = str(column or "").strip().lower()
    target_dtype = str(target_info.get("target_dtype") or "").strip().lower()
    families: set[str] = set()
    for example in examples:
        families.update(_value_format_families(example))

    if any(token in target_dtype for token in ("datetime", "timestamp", "date", "time")):
        return "datetime_like"
    if "bool" in target_dtype or families.intersection({"boolean_token"}):
        return "boolean_like"
    if families.intersection({"currency_symbol", "magnitude_suffix"}) or any(
        token in name for token in ("revenue", "amount", "value", "price", "cost", "budget", "contract")
    ):
        return "amount_like"
    if families.intersection({"percent_symbol"}) or any(
        token in name for token in ("rate", "ratio", "pct", "percent")
    ):
        return "rate_like"
    if any(token in target_dtype for token in ("int", "float", "double", "numeric", "number")) or observed_dtype in {
        "float64",
        "int64",
        "int32",
        "float32",
    }:
        if any(token in name for token in ("count", "num_", "qty", "quantity", "visits", "employees")):
            return "count_like"
        return "numeric_like"
    return "categorical_like"


def _preservation_expectation(
    column: str,
    required_columns: List[str],
    passthrough_columns: List[str],
    drop_columns: List[str],
    semantic_kind: str,
) -> str:
    if column in drop_columns:
        return "remove_after_gate_satisfied"
    if column in required_columns:
        return "retain_in_output"
    if column in passthrough_columns:
        return "retain_as_passthrough_if_needed"
    if semantic_kind in {"datetime_like", "amount_like", "rate_like", "count_like", "numeric_like", "boolean_like"}:
        return "recover_if_defensible_flag_if_not"
    return "preserve_or_flag_if_not"


def _build_column_resolution_context(
    contract_min: Dict[str, Any],
    contract_full: Dict[str, Any],
    data_profile: Dict[str, Any],
    dataset_profile: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    profile_columns = _profile_columns(data_profile, dataset_profile)
    known_columns = set(profile_columns)
    required_columns = _resolve_required_columns(contract_min, contract_full)
    passthrough_columns = _resolve_passthrough_columns(contract_min, contract_full, required_columns)
    required_feature_selectors = _resolve_required_feature_selectors(contract_min, contract_full)
    column_dtype_targets = _resolve_column_dtype_targets(contract_min, contract_full)
    cleaning_gates = _resolve_cleaning_gates(contract_min, contract_full)
    column_transformations = _resolve_column_transformations(contract_min, contract_full)
    drop_columns = [str(col) for col in _coerce_list(column_transformations.get("drop_columns")) if str(col).strip()]
    dtypes = _profile_map(data_profile, dataset_profile, "dtypes", "type_hints", "column_types")
    missingness = _profile_map(data_profile, dataset_profile, "missingness", "missing_frac")

    relevant_columns: List[str] = []
    gate_relevance: Dict[str, List[Dict[str, Any]]] = {}

    def _register_column(column: str) -> None:
        col = str(column or "").strip()
        if not col:
            return
        if known_columns and col not in known_columns:
            return
        if col not in relevant_columns:
            relevant_columns.append(col)

    for column in required_columns + passthrough_columns + list(column_dtype_targets.keys()):
        _register_column(column)

    for column in _collect_known_column_mentions(required_feature_selectors, known_columns):
        _register_column(column)
    for column in _collect_known_column_mentions(column_transformations, known_columns):
        _register_column(column)

    for gate in cleaning_gates:
        if not isinstance(gate, dict):
            continue
        gate_name = str(gate.get("name") or "").strip()
        severity = str(gate.get("severity") or "").strip().upper()
        action_type = str(gate.get("action_type") or "").strip().lower()
        gate_columns = _collect_known_column_mentions(gate, known_columns)
        for column in gate_columns:
            _register_column(column)
            gate_relevance.setdefault(column, []).append(
                {
                    "name": gate_name,
                    "severity": severity,
                    "action_type": action_type,
                }
            )

    context: Dict[str, Dict[str, Any]] = {}
    for column in relevant_columns:
        top_values = _profile_top_values(data_profile, dataset_profile, column)
        examples = [str(item.get("value") or "") for item in top_values if str(item.get("value") or "").strip()][:8]
        placeholder_tokens = [
            str(item)
            for item in examples
            if str(item).strip().lower() in _NULLISH_TOKENS and str(item).strip()
        ]
        format_families: List[str] = []
        for example in examples:
            for family in _value_format_families(example):
                if family not in format_families:
                    format_families.append(family)
        target_info = column_dtype_targets.get(column) if isinstance(column_dtype_targets.get(column), dict) else {}
        observed_dtype = str(dtypes.get(column) or "").strip()
        semantic_kind = _infer_semantic_kind(column, target_info, observed_dtype, examples)
        payload: Dict[str, Any] = {
            "semantic_kind": semantic_kind,
            "observed_storage_dtype": observed_dtype or "unknown",
            "observed_format_families": format_families,
            "top_raw_examples": examples,
            "null_or_placeholder_tokens": list(dict.fromkeys(placeholder_tokens)),
            "gate_relevance": gate_relevance.get(column, []),
            "preservation_expectation": _preservation_expectation(
                column,
                required_columns,
                passthrough_columns,
                drop_columns,
                semantic_kind,
            ),
        }
        if target_info:
            payload["target_dtype"] = str(target_info.get("target_dtype") or "").strip()
            if "nullable" in target_info:
                payload["nullable"] = target_info.get("nullable")
        missing_value = missingness.get(column)
        if isinstance(missing_value, (int, float)):
            payload["missingness"] = round(float(missing_value), 4)
        if top_values:
            payload["top_value_counts"] = [
                {
                    "value": str(item.get("value") or ""),
                    "count": item.get("count"),
                }
                for item in top_values[:8]
                if str(item.get("value") or "").strip()
            ]
        context[column] = payload
    return context


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


def _dedupe_string_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    seen: set[str] = set()
    normalized: List[str] = []
    for item in values:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(text)
    return normalized


def _resolve_model_features(contract_min: Dict[str, Any], contract_full: Dict[str, Any]) -> List[str]:
    for source in (contract_min, contract_full):
        if not isinstance(source, dict):
            continue
        if "model_features" in source and isinstance(source.get("model_features"), list):
            return _dedupe_string_list(source.get("model_features"))
        allowed = source.get("allowed_feature_sets")
        if isinstance(allowed, dict) and "model_features" in allowed and isinstance(allowed.get("model_features"), list):
            return _dedupe_string_list(allowed.get("model_features"))
    return []


def _resolve_allowed_feature_sets(contract_min: Dict[str, Any], contract_full: Dict[str, Any]) -> Any:
    for source in (contract_min, contract_full):
        if not isinstance(source, dict) or "allowed_feature_sets" not in source:
            continue
        return copy.deepcopy(source.get("allowed_feature_sets"))
    return {}


def _resolve_forbidden_features(allowed_feature_sets: Any) -> List[str]:
    if not isinstance(allowed_feature_sets, dict):
        return []
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


def build_de_view(
    contract_full: Dict[str, Any] | None,
    contract_min: Dict[str, Any] | None,
    artifact_index: Any,
    data_profile: Optional[Dict[str, Any]] = None,
    dataset_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    contract_full = contract_full if isinstance(contract_full, dict) else {}
    contract_min = contract_min if isinstance(contract_min, dict) else {}
    scope, active_workstreams, future_ml_handoff = _resolve_contract_execution_context(contract_min, contract_full)
    task_semantics = _resolve_task_semantics(contract_min, contract_full)
    canonical_columns = contract_min.get("canonical_columns")
    if not isinstance(canonical_columns, list):
        canonical_columns = get_canonical_columns(contract_full)
    canonical_columns = [str(c) for c in canonical_columns if c]
    column_roles = _resolve_column_roles(contract_min, contract_full)
    allowed_feature_sets = _resolve_allowed_feature_sets(contract_min, contract_full)
    model_features = _resolve_model_features(contract_min, contract_full)
    required_outputs = _resolve_required_outputs(contract_min, contract_full)
    required_columns = _resolve_required_columns(contract_min, contract_full)
    required_feature_selectors = _resolve_required_feature_selectors(contract_min, contract_full)
    passthrough_columns = _resolve_passthrough_columns(contract_min, contract_full, required_columns)
    column_transformations = _resolve_column_transformations(contract_min, contract_full)
    column_dtype_targets = _resolve_column_dtype_targets({}, contract_full)
    column_resolution_context = _build_column_resolution_context(
        contract_min,
        contract_full,
        data_profile if isinstance(data_profile, dict) else {},
        dataset_profile if isinstance(dataset_profile, dict) else {},
    )
    output_path = _resolve_output_path(contract_min, contract_full, required_outputs)
    manifest_path = _resolve_manifest_path(contract_min, contract_full, required_outputs)
    de_required_outputs = _resolve_required_outputs_for_owner(
        contract_min,
        contract_full,
        "data_engineer",
        fallback_paths=[output_path, manifest_path],
    )
    cleaning_gates = _resolve_cleaning_gates(contract_min, contract_full)
    data_engineer_runbook = _resolve_data_engineer_runbook(contract_min, contract_full)
    outlier_policy = _resolve_outlier_policy(contract_min, contract_full)
    report_path = ""
    if outlier_policy and outlier_policy.get("apply_stage") in {"data_engineer", "both"}:
        report_path = _resolve_de_outlier_report_path_from_policy(outlier_policy)
    output_dialect = _resolve_output_dialect(contract_min, contract_full)
    column_sets_summary = contract_min.get("column_sets_summary") or contract_full.get("column_sets_summary")
    dataset_artifact_requirements = _project_de_artifact_requirements(contract_min, contract_full)
    projection_context: Dict[str, Any] = {
        "scope": scope,
        "active_workstreams": active_workstreams,
        "future_ml_handoff": future_ml_handoff,
        "task_semantics": task_semantics,
        "canonical_columns": canonical_columns,
        "column_roles": column_roles,
        "allowed_feature_sets": allowed_feature_sets,
        "model_features": model_features,
        "required_outputs": de_required_outputs,
        "required_columns": required_columns,
        "required_feature_selectors": required_feature_selectors,
        "optional_passthrough_columns": passthrough_columns,
        "column_dtype_targets": column_dtype_targets,
        "column_resolution_context": column_resolution_context,
        "column_resolution_context_path": _DEFAULT_COLUMN_RESOLUTION_CONTEXT_PATH,
        "column_transformations": column_transformations,
        "output_path": output_path or "",
        "output_manifest_path": manifest_path or "",
        "artifact_requirements": dataset_artifact_requirements,
        "output_dialect": output_dialect,
        "cleaning_gates": cleaning_gates,
        "data_engineer_runbook": data_engineer_runbook,
        "outlier_policy": outlier_policy if outlier_policy and outlier_policy.get("apply_stage") in {"data_engineer", "both"} else {},
        "outlier_report_path": report_path,
        "column_sets_summary": column_sets_summary,
        "constraints": {
            "scope": "cleaning_only" if not active_workstreams.get("model_training") else scope,
            "hard_constraints": [
                "no_modeling",
                "no_score_fitting",
                "no_prescriptive_tuning",
                "no_analytics",
            ],
        },
    }
    return _build_declared_agent_view("data_engineer", projection_context, contract_min, contract_full)


def build_ml_view(
    contract_full: Dict[str, Any] | None,
    contract_min: Dict[str, Any] | None,
    artifact_index: Any,
) -> Dict[str, Any]:
    contract_full = contract_full if isinstance(contract_full, dict) else {}
    contract_min = contract_min if isinstance(contract_min, dict) else {}
    scope, active_workstreams, future_ml_handoff = _resolve_contract_execution_context(contract_min, contract_full)
    required_outputs = _resolve_required_outputs(contract_min, contract_full)
    objective_type = _resolve_objective_type(contract_min, contract_full, required_outputs)
    task_semantics = _resolve_task_semantics(contract_min, contract_full)
    canonical_columns = contract_min.get("canonical_columns")
    if not isinstance(canonical_columns, list):
        canonical_columns = get_canonical_columns(contract_full)
    canonical_columns = [str(c) for c in canonical_columns if c]
    column_roles = _resolve_column_roles(contract_min, contract_full)
    allowed_feature_sets = _resolve_allowed_feature_sets(contract_min, contract_full)
    model_features = _resolve_model_features(contract_min, contract_full)
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
    evaluation_spec = contract_min.get("evaluation_spec")
    if not isinstance(evaluation_spec, dict) or not evaluation_spec:
        evaluation_spec = contract_full.get("evaluation_spec")
    if not isinstance(evaluation_spec, dict):
        evaluation_spec = {}
    objective_analysis = contract_min.get("objective_analysis")
    if not isinstance(objective_analysis, dict) or not objective_analysis:
        objective_analysis = contract_full.get("objective_analysis")
    if not isinstance(objective_analysis, dict):
        objective_analysis = {}
    qa_gates = _resolve_qa_gates(contract_min, contract_full)
    reviewer_gates = _resolve_reviewer_gates(contract_min, contract_full)
    ml_engineer_runbook = {}
    for source in (contract_min, contract_full):
        if not isinstance(source, dict):
            continue
        runbook = source.get("ml_engineer_runbook")
        if isinstance(runbook, (dict, list)) and runbook:
            ml_engineer_runbook = runbook
            break
        if isinstance(runbook, str) and runbook.strip():
            ml_engineer_runbook = runbook.strip()
            break
    clean_dataset_cfg = _resolve_dataset_artifact_binding(contract_min, contract_full, "cleaned_dataset")
    cleaned_data_path = str(
        clean_dataset_cfg.get("output_path")
        or clean_dataset_cfg.get("path")
        or ""
    ).strip()
    cleaning_manifest_path = str(
        clean_dataset_cfg.get("output_manifest_path")
        or clean_dataset_cfg.get("manifest_path")
        or clean_dataset_cfg.get("manifest")
        or ""
    ).strip()
    if not cleaned_data_path:
        cleaned_data_path = str(get_clean_dataset_output_path(contract_full) or get_clean_dataset_output_path(contract_min) or "").strip()
    if not cleaning_manifest_path:
        cleaning_manifest_path = str(get_clean_manifest_path(contract_full) or get_clean_manifest_path(contract_min) or "").strip()
    primary_metric = ""
    metric_definition_rule = ""
    for source in (validation, evaluation_spec):
        if not isinstance(source, dict):
            continue
        if not primary_metric:
            primary_metric = str(source.get("primary_metric") or "").strip()
        if not metric_definition_rule:
            metric_definition_rule = str(source.get("metric_definition_rule") or "").strip()

    projection_context: Dict[str, Any] = {
        "scope": scope,
        "active_workstreams": active_workstreams,
        "future_ml_handoff": future_ml_handoff,
        "objective_type": objective_type,
        "primary_metric": primary_metric,
        "metric_definition_rule": metric_definition_rule,
        "task_semantics": task_semantics,
        "canonical_columns": canonical_columns,
        "column_roles": column_roles,
        "column_dtype_targets": column_dtype_targets,
        "required_outputs": ml_required_outputs,
        "validation_requirements": validation,
        "evaluation_spec": evaluation_spec,
        "objective_analysis": objective_analysis,
        "qa_gates": qa_gates,
        "reviewer_gates": reviewer_gates,
        "ml_engineer_runbook": ml_engineer_runbook,
    }
    if _contract_declares_any_path(contract_min, contract_full, ["model_features", "allowed_feature_sets.model_features"]):
        projection_context["model_features"] = model_features
    if _contract_declares_any_path(contract_min, contract_full, ["allowed_feature_sets"]):
        projection_context["allowed_feature_sets"] = allowed_feature_sets
    row_count_hints = _resolve_row_count_hints(contract_full, contract_min)
    if row_count_hints:
        for key in ("n_train_rows", "n_test_rows", "n_total_rows"):
            if key in row_count_hints:
                projection_context[key] = row_count_hints[key]
    split_spec = contract_min.get("split_spec")
    if not isinstance(split_spec, dict) or not split_spec:
        split_spec = contract_full.get("split_spec")
    if isinstance(split_spec, dict) and split_spec:
        projection_context["split_spec"] = split_spec
    column_sets_summary = contract_min.get("column_sets_summary") or contract_full.get("column_sets_summary")
    if column_sets_summary:
        projection_context["column_sets_summary"] = column_sets_summary
    training_rows_rule = contract_min.get("training_rows_rule") or contract_full.get("training_rows_rule")
    scoring_rows_rule = contract_min.get("scoring_rows_rule") or contract_full.get("scoring_rows_rule")
    secondary_scoring_subset = contract_min.get("secondary_scoring_subset") or contract_full.get("secondary_scoring_subset")
    data_partitioning_notes = contract_min.get("data_partitioning_notes") or contract_full.get("data_partitioning_notes")
    if training_rows_rule:
        projection_context["training_rows_rule"] = training_rows_rule
    if scoring_rows_rule:
        projection_context["scoring_rows_rule"] = scoring_rows_rule
    if secondary_scoring_subset:
        projection_context["secondary_scoring_subset"] = secondary_scoring_subset
    if isinstance(data_partitioning_notes, list) and data_partitioning_notes:
        projection_context["data_partitioning_notes"] = data_partitioning_notes
    if artifact_payload:
        projection_context["artifact_requirements"] = artifact_payload
    if cleaned_data_path:
        projection_context["cleaned_data_path"] = cleaned_data_path
    if cleaning_manifest_path:
        projection_context["cleaning_manifest_path"] = cleaning_manifest_path
    visual_ctx = _resolve_visual_context(contract_full, contract_min)
    visual_payload = visual_ctx.get("visual_payload", {})
    plot_spec = visual_ctx.get("plot_spec")
    projection_context["decisioning_requirements"] = _get_decisioning_requirements(contract_full, contract_min)
    if case_rules is not None:
        projection_context["case_rules"] = case_rules
    if outlier_policy:
        projection_context["outlier_policy"] = outlier_policy
    if plot_spec is not None:
        projection_context["plot_spec"] = plot_spec
    projection_context["visual_requirements"] = visual_payload
    return _build_declared_agent_view("ml_engineer", projection_context, contract_min, contract_full)


def build_reviewer_view(
    contract_full: Dict[str, Any] | None,
    contract_min: Dict[str, Any] | None,
    artifact_index: Any,
) -> Dict[str, Any]:
    contract_full = contract_full if isinstance(contract_full, dict) else {}
    contract_min = contract_min if isinstance(contract_min, dict) else {}
    scope, active_workstreams, future_ml_handoff = _resolve_contract_execution_context(contract_min, contract_full)
    required_outputs = _resolve_required_outputs(contract_min, contract_full)
    objective_type = _resolve_objective_type(contract_min, contract_full, required_outputs)
    task_semantics = _resolve_task_semantics(contract_min, contract_full)
    reviewer_gates = _resolve_reviewer_gates(contract_min, contract_full)
    expected_metrics = _expected_metrics_from_objective(objective_type, reviewer_gates)
    strategy_summary = _summarize_strategy(contract_full, contract_min)
    projection_context: Dict[str, Any] = {
        "scope": scope,
        "active_workstreams": active_workstreams,
        "future_ml_handoff": future_ml_handoff,
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
        "decisioning_requirements": _get_decisioning_requirements(contract_full, contract_min),
    }
    return _build_declared_agent_view("reviewer", projection_context, contract_min, contract_full)


def build_qa_view(
    contract_full: Dict[str, Any] | None,
    contract_min: Dict[str, Any] | None,
    artifact_index: Any,
) -> Dict[str, Any]:
    contract_full = contract_full if isinstance(contract_full, dict) else {}
    contract_min = contract_min if isinstance(contract_min, dict) else {}
    scope, active_workstreams, future_ml_handoff = _resolve_contract_execution_context(contract_min, contract_full)
    task_semantics = _resolve_task_semantics(contract_min, contract_full)
    required_outputs = _resolve_required_outputs(contract_min, contract_full)
    optional_outputs = _resolve_optional_outputs(contract_min, contract_full)
    review_subject = _resolve_qa_review_subject(contract_min, contract_full)
    model_features = _resolve_model_features(contract_min, contract_full)
    subject_required_outputs = _resolve_required_outputs_for_owner(
        contract_min,
        contract_full,
        review_subject,
        fallback_paths=required_outputs,
    )
    qa_required_outputs = _resolve_required_outputs_for_owner(contract_min, contract_full, "qa_engineer")
    artifacts_to_verify = list(subject_required_outputs)
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
        "required_outputs": list(artifacts_to_verify),
        "qa_required_outputs": list(qa_required_outputs),
        "optional_outputs": optional_outputs,
    }
    if review_subject == "data_engineer":
        artifact_payload.update(_project_de_artifact_requirements(contract_min, contract_full))
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
    projection_context: Dict[str, Any] = {
        "scope": scope,
        "active_workstreams": active_workstreams,
        "future_ml_handoff": future_ml_handoff,
        "task_semantics": task_semantics,
        "review_subject": review_subject,
        "subject_required_outputs": subject_required_outputs,
        "qa_required_outputs": qa_required_outputs,
        "artifacts_to_verify": artifacts_to_verify,
        "subject_code_path_hint": _default_subject_code_path(review_subject),
        "qa_gates": qa_gates,
        "artifact_requirements": artifact_payload,
        "model_features": model_features,
        "allowed_feature_sets": allowed_feature_sets,
        "column_roles": column_roles,
        "canonical_columns": canonical_columns,
        "objective_summary": objective_summary,
        "decisioning_requirements": _get_decisioning_requirements(contract_full, contract_min),
    }
    row_count_hints = _resolve_row_count_hints(contract_full, contract_min)
    if row_count_hints:
        for key in ("n_train_rows", "n_test_rows", "n_total_rows"):
            if key in row_count_hints:
                projection_context[key] = row_count_hints[key]
    split_spec = contract_min.get("split_spec")
    if not isinstance(split_spec, dict) or not split_spec:
        split_spec = contract_full.get("split_spec")
    if isinstance(split_spec, dict) and split_spec:
        projection_context["split_spec"] = split_spec
    if isinstance(reporting_policy, dict) and reporting_policy:
        projection_context["reporting_policy"] = reporting_policy
    return _build_declared_agent_view("qa_reviewer", projection_context, contract_min, contract_full)


def build_translator_view(
    contract_full: Dict[str, Any] | None,
    contract_min: Dict[str, Any] | None,
    artifact_index: Any,
    insights: Any = None,
) -> Dict[str, Any]:
    contract_full = contract_full if isinstance(contract_full, dict) else {}
    contract_min = contract_min if isinstance(contract_min, dict) else {}
    scope, active_workstreams, future_ml_handoff = _resolve_contract_execution_context(contract_min, contract_full)
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
    projection_context: Dict[str, Any] = {
        "scope": scope,
        "active_workstreams": active_workstreams,
        "future_ml_handoff": future_ml_handoff,
        "reporting_policy": policy,
        "evidence_inventory": evidence,
        "key_decisions": key_decisions,
        "limitations": limitations,
        "constraints": {"no_markdown_tables": True, "cite_sources": True},
        "decisioning_requirements": _get_decisioning_requirements(contract_full, contract_min),
    }
    visual_ctx = _resolve_visual_context(contract_full, contract_min)
    plot_spec = visual_ctx.get("plot_spec")
    visual_payload = visual_ctx.get("visual_payload", {})
    if plot_spec is not None:
        projection_context["plot_spec"] = plot_spec
    visual_warnings = visual_ctx.get("view_warnings")
    if isinstance(visual_warnings, dict) and visual_warnings:
        projection_context["view_warnings"] = visual_warnings
    projection_context["visual_requirements"] = visual_payload
    return _build_declared_agent_view("translator", projection_context, contract_min, contract_full)


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
    projection_context: Dict[str, Any] = {
        "objective_type": objective_type,
        "reporting_policy": policy,
        "evidence_inventory": _normalize_artifact_index(artifact_index),
    }
    return _build_declared_agent_view("results_advisor", projection_context, contract_min, contract_full)


def build_cleaning_view(
    contract_full: Dict[str, Any] | None,
    contract_min: Dict[str, Any] | None,
    artifact_index: Any,
    cleaning_code: Optional[str] = None,
    data_profile: Optional[Dict[str, Any]] = None,
    dataset_profile: Optional[Dict[str, Any]] = None,
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
    scope, active_workstreams, future_ml_handoff = _resolve_contract_execution_context(contract_min, contract_full)
    task_semantics = _resolve_task_semantics(contract_min, contract_full)
    canonical_columns = contract_min.get("canonical_columns")
    if not isinstance(canonical_columns, list):
        canonical_columns = get_canonical_columns(contract_full)
    canonical_columns = [str(c) for c in canonical_columns if c]
    model_features = _resolve_model_features(contract_min, contract_full)
    required_columns = _resolve_required_columns(contract_min, contract_full)
    required_feature_selectors = _resolve_required_feature_selectors(contract_min, contract_full)
    column_resolution_context = _build_column_resolution_context(
        contract_min,
        contract_full,
        data_profile if isinstance(data_profile, dict) else {},
        dataset_profile if isinstance(dataset_profile, dict) else {},
    )
    column_roles = _resolve_column_roles(contract_min, contract_full)
    allowed_feature_sets = _resolve_allowed_feature_sets(contract_min, contract_full)
    dialect = _resolve_output_dialect(contract_min, contract_full)
    cleaning_gates = _resolve_cleaning_gates(contract_min, contract_full)
    column_transformations = _resolve_column_transformations(contract_min, contract_full)
    outlier_policy = _resolve_outlier_policy(contract_min, contract_full)
    all_required_outputs = _resolve_required_outputs(contract_min, contract_full)
    output_path = _resolve_output_path(contract_min, contract_full, all_required_outputs)
    manifest_path = _resolve_manifest_path(contract_min, contract_full, all_required_outputs)
    dataset_artifact_requirements = _project_de_artifact_requirements(contract_min, contract_full)
    cleaning_required_outputs = _resolve_required_outputs_for_owner(
        contract_min,
        contract_full,
        "data_engineer",
        fallback_paths=[output_path, manifest_path],
    )
    report_path = ""
    if outlier_policy and outlier_policy.get("apply_stage") in {"data_engineer", "both"}:
        report_path = _resolve_de_outlier_report_path_from_policy(outlier_policy)
    projection_context: Dict[str, Any] = {
        "scope": scope,
        "active_workstreams": active_workstreams,
        "future_ml_handoff": future_ml_handoff,
        "task_semantics": task_semantics,
        "strategy_title": _first_value(contract_full.get("strategy_title"), contract_min.get("strategy_title")) or "",
        "business_objective": _first_value(
            contract_full.get("business_objective"), contract_min.get("business_objective")
        ) or "",
        "canonical_columns": canonical_columns,
        "model_features": model_features,
        "required_outputs": cleaning_required_outputs,
        "required_columns": required_columns,
        "required_feature_selectors": required_feature_selectors,
        "column_resolution_context": column_resolution_context,
        "column_resolution_context_path": _DEFAULT_COLUMN_RESOLUTION_CONTEXT_PATH,
        "column_transformations": column_transformations,
        "artifact_requirements": dataset_artifact_requirements,
        "dialect": dialect,
        "cleaning_gates": cleaning_gates,
        "column_roles": column_roles,
        "allowed_feature_sets": allowed_feature_sets,
        "outlier_policy": outlier_policy if outlier_policy and outlier_policy.get("apply_stage") in {"data_engineer", "both"} else {},
        "outlier_report_path": report_path,
    }
    # Include cleaning code for intent verification (rescale detection, synthetic data check)
    if cleaning_code and isinstance(cleaning_code, str):
        # Truncate if too long to fit in budget
        max_code_len = 8000
        code_to_include = cleaning_code[:max_code_len] if len(cleaning_code) > max_code_len else cleaning_code
        projection_context["cleaning_code"] = code_to_include
    return _build_declared_agent_view("cleaning_reviewer", projection_context, contract_min, contract_full)


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


def _build_views_v5(contract: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Build agent views for v5.0 hierarchical contracts via trivial merge.

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
    artifact_index: Any,
    cleaning_code: Optional[str] = None,
    data_profile: Optional[Dict[str, Any]] = None,
    dataset_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Build agent views as pure projection from validated execution contract.

    V5.0 contracts use trivial merge (shared + agent section).
    V4.x contracts fall through to legacy field-to-view projection.
    """
    contract_full = contract_full if isinstance(contract_full, dict) else {}

    # ── V5 dispatch: hierarchical merge ──────────────────────────────
    # If the contract was flattened, use the original v5 hierarchy for views.
    v5_original = contract_full.get("_v5_original") if isinstance(contract_full, dict) else None
    if isinstance(v5_original, dict) and str(v5_original.get("contract_version", "")).startswith("5"):
        return _build_views_v5(v5_original)
    if str(contract_full.get("contract_version", "")).startswith("5") and "shared" in contract_full:
        return _build_views_v5(contract_full)

    # ── V4.x legacy path ────────────────────────────────────────────
    artifact_index = artifact_index if isinstance(artifact_index, list) else []
    contract_min = contract_full
    return {
        "de_view": build_de_view(
            contract_full,
            contract_min,
            artifact_index,
            data_profile=data_profile,
            dataset_profile=dataset_profile,
        ),
        "ml_view": build_ml_view(contract_full, contract_min, artifact_index),
        "cleaning_view": build_cleaning_view(
            contract_full,
            contract_min,
            artifact_index,
            cleaning_code=cleaning_code,
            data_profile=data_profile,
            dataset_profile=dataset_profile,
        ),
        "qa_view": build_qa_view(contract_full, contract_min, artifact_index),
        "reviewer_view": build_reviewer_view(contract_full, contract_min, artifact_index),
        "translator_view": build_translator_view(contract_full, contract_min, artifact_index),
        "results_advisor_view": build_results_advisor_view(contract_full, contract_min, artifact_index),
    }


def _contract_declares_path(source: Dict[str, Any], dotted_path: str) -> bool:
    current: Any = source
    for segment in dotted_path.split("."):
        if not isinstance(current, dict) or segment not in current:
            return False
        current = current.get(segment)
    return True


def _contract_declares_any_path(contract_min: Dict[str, Any], contract_full: Dict[str, Any], paths: List[str]) -> bool:
    for source in (contract_min, contract_full):
        if not isinstance(source, dict):
            continue
        for dotted_path in paths:
            if _contract_declares_path(source, dotted_path):
                return True
    return False


def _normalize_text_set(values: Any) -> set[str]:
    return {text.strip().lower() for text in _dedupe_string_list(values)}


def _normalize_path_set(values: Any) -> set[str]:
    normalized: set[str] = set()
    for item in _coerce_list(values):
        text = ""
        if isinstance(item, str):
            text = item
        elif isinstance(item, dict):
            for key in ("path", "output_path", "output", "artifact", "file"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    text = value
                    break
        text = str(text or "").replace("\\", "/").strip()
        if text:
            normalized.add(text.lower())
    return normalized


def _extract_gate_name_set(values: Any) -> set[str]:
    names: set[str] = set()
    if not isinstance(values, list):
        return names
    for item in values:
        if isinstance(item, dict):
            text = str(item.get("name") or item.get("id") or item.get("gate") or "").strip().lower()
            if text:
                names.add(text)
        elif isinstance(item, str):
            text = item.strip().lower()
            if text:
                names.add(text)
    return names


def _normalize_role_map(values: Any) -> Dict[str, set[str]]:
    if not isinstance(values, dict):
        return {}
    normalized: Dict[str, set[str]] = {}
    for key, raw in values.items():
        items = _normalize_text_set(raw)
        if items:
            normalized[str(key)] = items
    return normalized


def _normalize_contract_payload(values: Any) -> Any:
    if isinstance(values, dict):
        return {
            str(key): _normalize_contract_payload(value)
            for key, value in sorted(values.items(), key=lambda item: str(item[0]))
        }
    if isinstance(values, list):
        return [_normalize_contract_payload(item) for item in values]
    if isinstance(values, str):
        return values.strip()
    return values


def _summarize_projection_value(value: Any) -> Dict[str, Any]:
    if isinstance(value, list):
        return {"kind": "list", "count": len(value)}
    if isinstance(value, dict):
        return {"kind": "dict", "keys": sorted([str(k) for k in value.keys()])[:12], "count": len(value)}
    if isinstance(value, str):
        return {"kind": "str", "len": len(value)}
    if value is None:
        return {"kind": "none"}
    return {"kind": type(value).__name__}


def _binding_payload(value: Any, *, declared: bool, source_paths: List[str]) -> Dict[str, Any]:
    return {
        "value": copy.deepcopy(value),
        "declared": bool(declared),
        "source_paths": list(source_paths),
    }


def _resolve_binding_required_outputs_for_owner(
    contract_min: Dict[str, Any],
    contract_full: Dict[str, Any],
    owner: str,
) -> Dict[str, Any]:
    value = _resolve_required_outputs_for_owner(contract_min, contract_full, owner)
    return _binding_payload(value, declared=bool(value), source_paths=[f"required_outputs[owner={owner}]"])


def _resolve_binding_required_columns(contract_min: Dict[str, Any], contract_full: Dict[str, Any]) -> Dict[str, Any]:
    value = _resolve_required_columns(contract_min, contract_full)
    return _binding_payload(
        value,
        declared=bool(value),
        source_paths=[
            "artifact_requirements.cleaned_dataset.required_columns",
            "artifact_requirements.clean_dataset.required_columns",
            "artifact_requirements.schema_binding.required_columns",
            "required_columns",
        ],
    )


def _resolve_binding_optional_passthrough(contract_min: Dict[str, Any], contract_full: Dict[str, Any]) -> Dict[str, Any]:
    value = _resolve_passthrough_columns(contract_min, contract_full, _resolve_required_columns(contract_min, contract_full))
    declared = _contract_declares_any_path(
        contract_min,
        contract_full,
        [
            "artifact_requirements.cleaned_dataset.optional_passthrough_columns",
            "artifact_requirements.clean_dataset.optional_passthrough_columns",
            "artifact_requirements.schema_binding.optional_passthrough_columns",
        ],
    )
    return _binding_payload(
        value,
        declared=declared,
        source_paths=[
            "artifact_requirements.cleaned_dataset.optional_passthrough_columns",
            "artifact_requirements.clean_dataset.optional_passthrough_columns",
            "artifact_requirements.schema_binding.optional_passthrough_columns",
        ],
    )


def _resolve_binding_dataset_artifact_requirements(
    contract_min: Dict[str, Any],
    contract_full: Dict[str, Any],
) -> Dict[str, Any]:
    value = _project_de_artifact_requirements(contract_min, contract_full)
    declared = _contract_declares_any_path(
        contract_min,
        contract_full,
        [
            "artifact_requirements.cleaned_dataset",
            "artifact_requirements.enriched_dataset",
            "artifact_requirements.clean_dataset",
            "artifact_requirements.schema_binding",
        ],
    )
    return _binding_payload(
        value,
        declared=declared,
        source_paths=[
            "artifact_requirements.cleaned_dataset",
            "artifact_requirements.enriched_dataset",
            "artifact_requirements.clean_dataset",
            "artifact_requirements.schema_binding",
        ],
    )


def _resolve_binding_model_features(contract_min: Dict[str, Any], contract_full: Dict[str, Any]) -> Dict[str, Any]:
    value = _resolve_model_features(contract_min, contract_full)
    declared = _contract_declares_any_path(
        contract_min,
        contract_full,
        [
            "model_features",
            "allowed_feature_sets.model_features",
        ],
    )
    return _binding_payload(
        value,
        declared=declared,
        source_paths=["model_features", "allowed_feature_sets.model_features"],
    )


def _resolve_binding_allowed_feature_sets(contract_min: Dict[str, Any], contract_full: Dict[str, Any]) -> Dict[str, Any]:
    value = _resolve_allowed_feature_sets(contract_min, contract_full)
    declared = _contract_declares_any_path(
        contract_min,
        contract_full,
        ["allowed_feature_sets"],
    )
    return _binding_payload(value, declared=declared, source_paths=["allowed_feature_sets"])


def _resolve_binding_column_roles(contract_min: Dict[str, Any], contract_full: Dict[str, Any]) -> Dict[str, Any]:
    value = _resolve_column_roles(contract_min, contract_full)
    return _binding_payload(value, declared=bool(value), source_paths=["column_roles"])


def _resolve_binding_canonical_columns(contract_min: Dict[str, Any], contract_full: Dict[str, Any]) -> Dict[str, Any]:
    value = contract_min.get("canonical_columns")
    if not isinstance(value, list):
        value = get_canonical_columns(contract_full)
    value = [str(column) for column in value if column]
    return _binding_payload(value, declared=bool(value), source_paths=["canonical_columns"])


def _resolve_binding_cleaning_gates(contract_min: Dict[str, Any], contract_full: Dict[str, Any]) -> Dict[str, Any]:
    value = _resolve_cleaning_gates(contract_min, contract_full)
    return _binding_payload(value, declared=bool(value), source_paths=["cleaning_gates"])


def _resolve_binding_qa_gates(contract_min: Dict[str, Any], contract_full: Dict[str, Any]) -> Dict[str, Any]:
    value = _resolve_qa_gates(contract_min, contract_full)
    return _binding_payload(value, declared=bool(value), source_paths=["qa_gates"])


def _resolve_binding_qa_subject_outputs(contract_min: Dict[str, Any], contract_full: Dict[str, Any]) -> Dict[str, Any]:
    review_subject = _resolve_qa_review_subject(contract_min, contract_full)
    value = _resolve_required_outputs_for_owner(contract_min, contract_full, review_subject)
    return _binding_payload(
        value,
        declared=bool(value),
        source_paths=[f"required_outputs[owner={review_subject}]"],
    )


def _resolve_binding_qa_outputs(contract_min: Dict[str, Any], contract_full: Dict[str, Any]) -> Dict[str, Any]:
    value = _resolve_required_outputs_for_owner(contract_min, contract_full, "qa_engineer")
    return _binding_payload(value, declared=bool(value), source_paths=["required_outputs[owner=qa_engineer]"])


def _resolve_binding_reviewer_gates(contract_min: Dict[str, Any], contract_full: Dict[str, Any]) -> Dict[str, Any]:
    value = _resolve_reviewer_gates(contract_min, contract_full)
    return _binding_payload(value, declared=bool(value), source_paths=["reviewer_gates"])


def _resolve_binding_reporting_policy(contract_min: Dict[str, Any], contract_full: Dict[str, Any]) -> Dict[str, Any]:
    value = contract_min.get("reporting_policy")
    if not isinstance(value, dict) or not value:
        value = contract_full.get("reporting_policy")
    return _binding_payload(
        value if isinstance(value, dict) else {},
        declared=_contract_declares_any_path(contract_min, contract_full, ["reporting_policy"]),
        source_paths=["reporting_policy"],
    )


def _resolve_binding_decisioning_requirements(contract_min: Dict[str, Any], contract_full: Dict[str, Any]) -> Dict[str, Any]:
    value = _get_decisioning_requirements(contract_full, contract_min)
    return _binding_payload(
        value,
        declared=_contract_declares_any_path(contract_min, contract_full, ["decisioning_requirements"]),
        source_paths=["decisioning_requirements"],
    )


def _resolve_binding_plot_spec(contract_min: Dict[str, Any], contract_full: Dict[str, Any]) -> Dict[str, Any]:
    visual_requirements, plot_spec = _project_plot_payload(contract_full if isinstance(contract_full, dict) else {})
    if not plot_spec and isinstance(contract_min, dict):
        reporting_policy = contract_min.get("reporting_policy")
        if isinstance(reporting_policy, dict):
            candidate = reporting_policy.get("plot_spec")
            if isinstance(candidate, dict):
                plot_spec = candidate
    return _binding_payload(
        plot_spec if isinstance(plot_spec, dict) else {},
        declared=bool(visual_requirements) or bool(plot_spec),
        source_paths=["artifact_requirements.visual_requirements.plot_spec", "reporting_policy.plot_spec"],
    )


def _resolve_binding_objective_type(contract_min: Dict[str, Any], contract_full: Dict[str, Any]) -> Dict[str, Any]:
    value = _resolve_objective_type(contract_min, contract_full, _resolve_required_outputs(contract_min, contract_full))
    return _binding_payload(
        value if isinstance(value, str) else "",
        declared=bool(value),
        source_paths=["task_semantics.objective_type", "objective_analysis.problem_type", "evaluation_spec.objective_type"],
    )


_VIEW_PROJECTION_BINDINGS: Dict[str, List[Dict[str, Any]]] = {
    "de_view": [
        {"name": "canonical_columns", "mode": "list_subset", "resolver": _resolve_binding_canonical_columns},
        {"name": "column_roles", "mode": "role_map_subset", "resolver": _resolve_binding_column_roles},
        {"name": "allowed_feature_sets", "mode": "feature_sets_subset", "resolver": _resolve_binding_allowed_feature_sets},
        {"name": "model_features", "mode": "list_subset", "resolver": _resolve_binding_model_features},
        {"name": "artifact_requirements", "mode": "dict_contract_equal", "resolver": _resolve_binding_dataset_artifact_requirements},
        {"name": "required_outputs", "mode": "paths_subset", "resolver": lambda cmin, cfull: _resolve_binding_required_outputs_for_owner(cmin, cfull, "data_engineer")},
        {"name": "required_columns", "mode": "list_subset", "resolver": _resolve_binding_required_columns},
        {"name": "optional_passthrough_columns", "mode": "list_subset", "resolver": _resolve_binding_optional_passthrough},
        {"name": "cleaning_gates", "mode": "gates_subset", "resolver": _resolve_binding_cleaning_gates},
    ],
    "cleaning_view": [
        {"name": "canonical_columns", "mode": "list_subset", "resolver": _resolve_binding_canonical_columns},
        {"name": "column_roles", "mode": "role_map_subset", "resolver": _resolve_binding_column_roles},
        {"name": "allowed_feature_sets", "mode": "feature_sets_subset", "resolver": _resolve_binding_allowed_feature_sets},
        {"name": "model_features", "mode": "list_subset", "resolver": _resolve_binding_model_features},
        {"name": "artifact_requirements", "mode": "dict_contract_equal", "resolver": _resolve_binding_dataset_artifact_requirements},
        {"name": "required_outputs", "mode": "paths_subset", "resolver": lambda cmin, cfull: _resolve_binding_required_outputs_for_owner(cmin, cfull, "data_engineer")},
        {"name": "required_columns", "mode": "list_subset", "resolver": _resolve_binding_required_columns},
        {"name": "cleaning_gates", "mode": "gates_subset", "resolver": _resolve_binding_cleaning_gates},
    ],
    "qa_view": [
        {"name": "canonical_columns", "mode": "list_subset", "resolver": _resolve_binding_canonical_columns},
        {"name": "column_roles", "mode": "role_map_subset", "resolver": _resolve_binding_column_roles},
        {"name": "allowed_feature_sets", "mode": "feature_sets_subset", "resolver": _resolve_binding_allowed_feature_sets},
        {"name": "model_features", "mode": "list_subset", "resolver": _resolve_binding_model_features},
        {"name": "subject_required_outputs", "mode": "paths_subset", "resolver": _resolve_binding_qa_subject_outputs},
        {"name": "qa_required_outputs", "mode": "paths_subset", "resolver": _resolve_binding_qa_outputs},
        {"name": "qa_gates", "mode": "gates_subset", "resolver": _resolve_binding_qa_gates},
    ],
    "ml_view": [
        {"name": "canonical_columns", "mode": "list_subset", "resolver": _resolve_binding_canonical_columns},
        {"name": "column_roles", "mode": "role_map_subset", "resolver": _resolve_binding_column_roles},
        {"name": "allowed_feature_sets", "mode": "feature_sets_subset", "resolver": _resolve_binding_allowed_feature_sets},
        {"name": "model_features", "mode": "list_subset", "resolver": _resolve_binding_model_features},
        {"name": "required_outputs", "mode": "paths_subset", "resolver": lambda cmin, cfull: _resolve_binding_required_outputs_for_owner(cmin, cfull, "ml_engineer")},
        {"name": "qa_gates", "mode": "gates_subset", "resolver": _resolve_binding_qa_gates},
        {"name": "reviewer_gates", "mode": "gates_subset", "resolver": _resolve_binding_reviewer_gates},
    ],
    "reviewer_view": [
        {"name": "required_outputs", "mode": "paths_subset", "resolver": lambda cmin, cfull: _resolve_binding_required_outputs_for_owner(cmin, cfull, "ml_engineer")},
        {"name": "reviewer_gates", "mode": "gates_subset", "resolver": _resolve_binding_reviewer_gates},
        {"name": "decisioning_requirements", "mode": "dict_present", "resolver": _resolve_binding_decisioning_requirements},
    ],
    "translator_view": [
        {"name": "reporting_policy", "mode": "dict_present", "resolver": _resolve_binding_reporting_policy},
        {"name": "decisioning_requirements", "mode": "dict_present", "resolver": _resolve_binding_decisioning_requirements},
        {"name": "plot_spec", "mode": "dict_present", "resolver": _resolve_binding_plot_spec},
    ],
    "results_advisor_view": [
        {"name": "objective_type", "mode": "scalar_present", "resolver": _resolve_binding_objective_type},
        {"name": "reporting_policy", "mode": "dict_present", "resolver": _resolve_binding_reporting_policy},
    ],
}


def _binding_matches(expected_payload: Dict[str, Any], actual_value: Any, field_present: bool, mode: str) -> bool:
    declared = bool(expected_payload.get("declared"))
    expected_value = expected_payload.get("value")
    if not declared:
        return True
    if not field_present:
        return False
    if mode == "paths_subset":
        expected = _normalize_path_set(expected_value)
        actual = _normalize_path_set(actual_value)
        return (not expected and field_present) or expected.issubset(actual)
    if mode == "gates_subset":
        expected = _extract_gate_name_set(expected_value)
        actual = _extract_gate_name_set(actual_value)
        return (not expected and field_present) or expected.issubset(actual)
    if mode == "role_map_subset":
        expected = _normalize_role_map(expected_value)
        actual = _normalize_role_map(actual_value)
        if not expected:
            return field_present
        for role, columns in expected.items():
            if not columns.issubset(actual.get(role, set())):
                return False
        return True
    if mode == "feature_sets_subset":
        expected = _normalize_contract_payload(expected_value)
        actual = _normalize_contract_payload(actual_value)
        return expected == actual
    if mode == "dict_contract_equal":
        expected = _normalize_contract_payload(expected_value)
        actual = _normalize_contract_payload(actual_value)
        return expected == actual
    if mode == "dict_present":
        if not field_present:
            return False
        if isinstance(expected_value, dict) and expected_value:
            return isinstance(actual_value, dict) and bool(actual_value)
        return field_present
    if mode == "scalar_present":
        if not field_present:
            return False
        if isinstance(expected_value, str) and expected_value.strip():
            return isinstance(actual_value, str) and bool(actual_value.strip())
        return field_present
    expected = _normalize_text_set(expected_value)
    actual = _normalize_text_set(actual_value)
    return (not expected and field_present) or expected.issubset(actual)


def build_contract_view_projection_reports(
    contract_full: Dict[str, Any] | None,
    views: Dict[str, Dict[str, Any]] | None,
    contract_min: Dict[str, Any] | None = None,
) -> Dict[str, Dict[str, Any]]:
    contract_full = contract_full if isinstance(contract_full, dict) else {}
    contract_min = contract_min if isinstance(contract_min, dict) else contract_full
    views = views if isinstance(views, dict) else {}
    reports: Dict[str, Dict[str, Any]] = {}
    for view_name, bindings in _VIEW_PROJECTION_BINDINGS.items():
        view_payload = views.get(view_name)
        if not isinstance(view_payload, dict) or not view_payload:
            reports[view_name] = {
                "view_name": view_name,
                "accepted": False,
                "required_bindings": [binding["name"] for binding in bindings],
                "missing_bindings": ["view"],
                "field_reports": [],
            }
            continue
        field_reports: List[Dict[str, Any]] = []
        missing_bindings: List[str] = []
        for binding in bindings:
            binding_name = str(binding["name"])
            expected_payload = binding["resolver"](contract_min, contract_full)
            actual_present = binding_name in view_payload
            actual_value = view_payload.get(binding_name)
            accepted = _binding_matches(expected_payload, actual_value, actual_present, str(binding["mode"]))
            field_report = {
                "binding": binding_name,
                "accepted": accepted,
                "declared_by_contract": bool(expected_payload.get("declared")),
                "source_paths": list(expected_payload.get("source_paths") or []),
                "expected_summary": _summarize_projection_value(expected_payload.get("value")),
                "actual_summary": _summarize_projection_value(actual_value),
            }
            field_reports.append(field_report)
            if not accepted:
                missing_bindings.append(binding_name)
        reports[view_name] = {
            "view_name": view_name,
            "accepted": not missing_bindings,
            "required_bindings": [binding["name"] for binding in bindings],
            "missing_bindings": missing_bindings,
            "field_reports": field_reports,
        }
    return reports


def list_view_projection_report_errors(reports: Dict[str, Dict[str, Any]] | None) -> List[str]:
    if not isinstance(reports, dict):
        return []
    errors: List[str] = []
    for view_name, report in reports.items():
        if not isinstance(report, dict):
            continue
        for binding in report.get("missing_bindings") or []:
            name = str(binding).strip().lower()
            if not name:
                continue
            if name == "view":
                errors.append(f"{view_name}_missing")
            else:
                errors.append(f"{view_name}_binding_{name}_missing")
    return errors


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
