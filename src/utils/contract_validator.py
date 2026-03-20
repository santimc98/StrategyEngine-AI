"""
Contract validation utilities.

Ensures contract self-consistency and detects ambiguity between
"outputs as files" vs "outputs as columns".
"""
import os
import re
import copy
from typing import Any, Dict, List, Tuple, Optional

from src.utils.cleaning_contract_semantics import (
    extract_selector_drop_reasons,
    selector_reference_matches_any,
)


# File extensions that indicate a path is an artifact file
FILE_EXTENSIONS = {
    ".csv", ".json", ".md", ".png", ".jpg", ".jpeg", ".pdf",
    ".parquet", ".pkl", ".pickle", ".txt", ".html", ".xlsx", ".xls"
}

# Universal set of metric-like tokens that should NOT appear as required columns in scored_rows_schema
# (metrics belong in metrics.json, not per-row)
METRIC_LIKE_TOKENS = {
    "accuracy", "roc_auc", "auc", "f1", "precision", "recall", "log_loss",
    "rmse", "mae", "mse", "r2", "rmsle", "gini", "normalized_gini",
    "f1_score", "f1-score", "roc-auc", "roc auc", "logloss", "cross_entropy",
    "balanced_accuracy", "cohen_kappa", "matthews_corrcoef", "mcc",
    "mean_squared_error", "mean_absolute_error", "root_mean_squared_error",
}

# Canonical column roles for column_roles normalization
CANONICAL_ROLES = {
    "outcome", "decision", "id", "feature", "timestamp", "group", "weight",
    "ignore", "forbidden", "target", "label", "identifier", "index",
    "segmentation", "audit_only", "protected", "sensitive",
    "pre_decision", "post_decision_audit_only", "unknown", "identifiers", "time_columns",
}

# Strict role-bucket ontology for V4.1 execution contracts.
STRICT_ROLE_BUCKETS = {
    "pre_decision",
    "decision",
    "outcome",
    "post_decision_audit_only",
    "unknown",
    "identifiers",
    "time_columns",
}

CONTRACT_SCOPE_VALUES = {"cleaning_only", "ml_only", "full_pipeline"}
SCOPE_ALIAS_VALUES = {
    "cleaning",
    "clean",
    "clean_only",
    "ml",
    "training",
    "modeling",
    "model_only",
}

ITERATION_POLICY_LIMIT_KEYS = (
    "max_iterations",
    "metric_improvement_max",
    "runtime_fix_max",
    "compliance_bootstrap_max",
)

ITERATION_POLICY_LIMIT_ALIASES = (
    "max_pipeline_iterations",
    "max_model_iterations",
    "max_iter",
    "max_retries",
    "gate_retry_limit",
    "runtime_retry_limit",
    "max_runtime_retries",
)

OPTIMIZATION_POLICY_DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "max_rounds": 8,
    "quick_eval_folds": 2,
    "full_eval_folds": 5,
    "min_delta": 0.0005,
    "patience": 3,
    "allow_model_switch": True,
    "allow_ensemble": True,
    "allow_hpo": True,
    "allow_feature_engineering": True,
    "allow_calibration": True,
}

OPTIMIZATION_POLICY_BOOL_KEYS = (
    "enabled",
    "allow_model_switch",
    "allow_ensemble",
    "allow_hpo",
    "allow_feature_engineering",
    "allow_calibration",
)
OPTIMIZATION_POLICY_INT_KEYS = (
    "max_rounds",
    "quick_eval_folds",
    "full_eval_folds",
    "patience",
)
OPTIMIZATION_POLICY_FLOAT_KEYS = (
    "min_delta",
)

# Role synonym mapping for normalization
ROLE_SYNONYM_MAP = {
    "target": "outcome",
    "label": "outcome",
    "identifier": "id",
    "index": "id",
    "ignored": "ignore",
    "excluded": "forbidden",
    "protected_attribute": "protected",
}


def get_default_optimization_policy() -> Dict[str, Any]:
    return dict(OPTIMIZATION_POLICY_DEFAULTS)


def _coerce_bool_value(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "on"}:
            return True
        if token in {"0", "false", "no", "off"}:
            return False
    return default


def _coerce_int_value(value: Any, default: int, minimum: int) -> int:
    try:
        candidate = int(float(value))
    except Exception:
        candidate = default
    if candidate < minimum:
        return default
    return candidate


def _coerce_float_value(value: Any, default: float, minimum: float) -> float:
    try:
        candidate = float(value)
    except Exception:
        candidate = default
    if candidate < minimum:
        return default
    return candidate


def normalize_optimization_policy(policy: Any) -> Dict[str, Any]:
    defaults = get_default_optimization_policy()
    if not isinstance(policy, dict):
        return defaults
    normalized = dict(defaults)
    for key in OPTIMIZATION_POLICY_BOOL_KEYS:
        normalized[key] = _coerce_bool_value(policy.get(key), defaults[key])
    for key in OPTIMIZATION_POLICY_INT_KEYS:
        minimum = 1 if key != "patience" else 0
        normalized[key] = _coerce_int_value(policy.get(key), defaults[key], minimum)
    for key in OPTIMIZATION_POLICY_FLOAT_KEYS:
        normalized[key] = _coerce_float_value(policy.get(key), defaults[key], 0.0)
    for key, value in policy.items():
        if key in normalized:
            continue
        normalized[key] = value
    return normalized


def _normalize_metric_name(name: str) -> str:
    """
    Normalize a metric name to canonical form.
    
    Rules:
      - lower()
      - replace spaces/hyphens with underscores
      - remove parentheses and common annotations
      - normalize common variants (roc-auc -> roc_auc, rmsle/rmse_log1p -> same token)
    
    Args:
        name: Raw metric name
        
    Returns:
        Normalized metric name string
    """
    if not isinstance(name, str):
        return str(name) if name is not None else ""
    
    # Lowercase
    s = name.lower().strip()
    
    # Remove common parenthetical annotations
    s = re.sub(r"\s*\([^)]*\)\s*", "", s)
    
    # Replace spaces and hyphens with underscores
    s = re.sub(r"[\s\-]+", "_", s)
    
    # Collapse multiple underscores
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    
    # Canonical normalizations
    roc_auc_variants = {"rocauc", "roc_auc", "auc_roc", "auroc", "roc"}
    if s in roc_auc_variants or s == "roc_auc":
        return "roc_auc"

    normalized_gini_variants = {
        "normalized_gini",
        "normalized_gini_coefficient",
        "kaggle_normalized_gini",
    }
    if s in normalized_gini_variants:
        return "normalized_gini"

    gini_variants = {"gini", "gini_coefficient", "gini_score"}
    if s in gini_variants:
        return "gini"
    
    # RMSE log variants
    rmse_log_variants = {"rmsle", "rmse_log", "rmse_log1p", "rmsle_log1p", "log_rmse"}
    if s in rmse_log_variants:
        return "rmsle"
    
    return s


def _is_metric_like_token(name: str) -> bool:
    """
    Check if a column name looks like a metric (should not be in required_columns).
    
    Args:
        name: Column/metric name to check
        
    Returns:
        True if the name looks like a metric
    """
    if not name:
        return False
    normalized = _normalize_metric_name(name)
    return normalized in METRIC_LIKE_TOKENS


def is_probably_path(value: str) -> bool:
    """
    Determine if a string looks like a filesystem path.

    Heuristics:
    - Contains "/" or "\\" or glob "*"
    - Starts with data/, static/, or reports/
    - Ends with a known file extension
    """
    if not isinstance(value, str) or not value.strip():
        return False

    value = value.strip()
    lower = value.lower()

    if "*" in value:
        return True
    if lower.startswith(("data/", "static/", "reports/")):
        return True
    if "/" in value or "\\" in value:
        return True
    _, ext = os.path.splitext(lower)
    if ext in FILE_EXTENSIONS:
        return True
    return False


def is_file_path(value: str) -> bool:
    """
    Determine if a string looks like a file path.

    A file path contains:
    - A "/" or "\\" (path separator), OR
    - A recognized file extension
    """
    if not isinstance(value, str) or not value.strip():
        return False

    value = value.strip()

    # Has a recognized file extension
    _, ext = os.path.splitext(value.lower())
    if ext in FILE_EXTENSIONS:
        return True

    # Contains path separator but looks like a conceptual phrase
    if ("/" in value or "\\" in value) and re.search(r"[\s\(\)\[\]\{\}<>]", value):
        return False

    # Contains path separator
    if "/" in value or "\\" in value:
        return True

    return False


def is_column_name(value: str) -> bool:
    """
    Determine if a string looks like a column name.

    A column name:
    - Does NOT contain "/" or "\\"
    - Does NOT have a file extension
    - Is a valid identifier-like string
    """
    if not isinstance(value, str) or not value.strip():
        return False

    value = value.strip()

    # Contains path separator -> not a column
    if "/" in value or "\\" in value:
        return False

    # Has a file extension -> not a column
    _, ext = os.path.splitext(value.lower())
    if ext in FILE_EXTENSIONS:
        return False

    # Disallow spaces or bracketed annotations in column names for this heuristic
    if re.search(r"[\s\(\)\[\]\{\}<>]", value):
        return False

    # Basic identifier-like check (letters, numbers, underscore, dash, dot)
    if not re.match(r"^[A-Za-z0-9_.-]+$", value):
        return False

    return True


def _extract_required_output_path(item: Any) -> Tuple[str, Optional[str]]:
    """
    Extract a required output path from supported entry formats.

    Supported formats:
      - "data/file.csv" (legacy string)
      - {"path": "data/file.csv", ...} (canonical object)

    Returns:
      (path, error_code) where error_code is None when valid.
    """
    if isinstance(item, str):
        path = item.strip()
        if not path:
            return "", "empty_path"
        return path, None

    if isinstance(item, dict):
        if "path" not in item:
            return "", "missing_path"
        path_value = item.get("path")
        if not isinstance(path_value, str):
            return "", "path_not_string"
        path = path_value.strip()
        if not path:
            return "", "empty_path"
        return path, None

    return "", "invalid_type"


def _normalize_required_output_path(path: str) -> str:
    return str(path or "").replace("\\", "/").strip().lower()


def detect_output_ambiguity(
    required_outputs: List[Any]
) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    """
    Detect and separate ambiguous entries in required_outputs.

    Returns:
        (files, columns, warnings, conceptual_outputs)
        - files: entries that are clearly file paths
        - columns: entries that are clearly column names
        - warnings: list of {"item": ..., "message": ...} for ambiguous cases
        - conceptual_outputs: entries that are non-file, non-column output requests
    """
    files = []
    columns = []
    warnings = []
    conceptual_outputs = []

    for item in required_outputs:
        # Handle both string and dict formats
        if isinstance(item, dict):
            path = item.get("path") or item.get("name") or ""
            desc = item.get("description", "")
        else:
            path = str(item) if item else ""
            desc = ""

        if not path:
            continue

        path_clean = path.strip()

        if is_file_path(path_clean):
            files.append({"path": path_clean, "description": desc})
        elif is_column_name(path_clean):
            # This looks like a column, not a file
            columns.append({"name": path_clean, "description": desc})
            warnings.append({
                "item": path_clean,
                "message": f"'{path_clean}' in required_outputs looks like a column name (no extension/path). Moved to required_columns.",
                "action": "moved_to_columns"
            })
        else:
            conceptual_outputs.append({"name": path_clean, "description": desc})
            warnings.append({
                "item": path_clean,
                "message": f"'{path_clean}' looks like a conceptual output. Moved to reporting_requirements.",
                "action": "moved_to_conceptual_outputs"
            })

    return files, columns, warnings, conceptual_outputs


def normalize_artifact_requirements(
    contract: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[Dict]]:
    """
    Normalize contract to use artifact_requirements structure.

    Converts explicit contract output declarations to:
    - artifact_requirements.required_files (paths)
    - artifact_requirements.scored_rows_schema only when the contract
      explicitly declares it or legacy output columns imply it

    Returns:
        (artifact_requirements, warnings)
    """
    warnings = []

    # Start with existing artifact_requirements if present
    artifact_req = contract.get("artifact_requirements", {})
    if not isinstance(artifact_req, dict):
        artifact_req = {}

    required_files = artifact_req.get("required_files", [])
    if not isinstance(required_files, list):
        required_files = []

    # Normalize mixed types (str vs dict) in required_files
    normalized_files = []
    for item in required_files:
        if isinstance(item, str) and item.strip():
            normalized_files.append({"path": item.strip(), "description": ""})
        elif isinstance(item, dict):
            normalized_files.append(item)
    required_files = normalized_files

    # Preserve declared file_schemas for validation/path resolution, but do not
    # silently promote them to required_files. Required-ness must come from the
    # contract's explicit deliverables/required outputs.
    raw_file_schemas = artifact_req.get("file_schemas", {})
    file_schemas = {}
    if isinstance(raw_file_schemas, dict):
        for raw_path, schema_obj in raw_file_schemas.items():
            if not isinstance(raw_path, str) or not raw_path.strip():
                continue
            normalized_path = raw_path.strip().replace("\\", "/").lstrip("/")
            file_schemas[normalized_path] = schema_obj

    scored_schema_raw = artifact_req.get("scored_rows_schema")
    scored_schema_present = isinstance(scored_schema_raw, dict)
    scored_schema = scored_schema_raw if scored_schema_present else {}

    required_columns = scored_schema.get("required_columns", [])
    if not isinstance(required_columns, list):
        required_columns = []

    # Process legacy required_outputs
    legacy_outputs = contract.get("required_outputs", [])
    if isinstance(legacy_outputs, list) and legacy_outputs:
        files, columns, ambig_warnings, conceptual_outputs = detect_output_ambiguity(legacy_outputs)
        warnings.extend(ambig_warnings)

        # Merge with existing
        existing_paths = {f.get("path", "").lower() for f in required_files}
        for f in files:
            if f["path"].lower() not in existing_paths:
                required_files.append(f)
                existing_paths.add(f["path"].lower())

        existing_cols = {c.get("name", "").lower() if isinstance(c, dict) else str(c).lower() for c in required_columns}
        for c in columns:
            col_name = c.get("name", "")
            if col_name.lower() not in existing_cols:
                required_columns.append(col_name)
                existing_cols.add(col_name.lower())

        if conceptual_outputs:
            reporting_requirements = contract.get("reporting_requirements")
            if not isinstance(reporting_requirements, dict):
                reporting_requirements = {}
            existing_conceptual = reporting_requirements.get("conceptual_outputs")
            if not isinstance(existing_conceptual, list):
                existing_conceptual = []
            existing_lower = {str(item).lower() for item in existing_conceptual}
            for item in conceptual_outputs:
                name = item.get("name") if isinstance(item, dict) else str(item)
                if not name:
                    continue
                if name.lower() in existing_lower:
                    continue
                existing_conceptual.append(name)
                existing_lower.add(name.lower())
            reporting_requirements["conceptual_outputs"] = existing_conceptual
            existing_narrative = reporting_requirements.get("narrative_outputs")
            if not isinstance(existing_narrative, list):
                existing_narrative = []
            existing_narrative_lower = {str(item).lower() for item in existing_narrative}
            for item in existing_conceptual:
                if str(item).lower() in existing_narrative_lower:
                    continue
                existing_narrative.append(item)
                existing_narrative_lower.add(str(item).lower())
            reporting_requirements["narrative_outputs"] = existing_narrative
            contract["reporting_requirements"] = reporting_requirements
            notes = contract.get("notes_for_engineers")
            if not isinstance(notes, list):
                notes = []
            note = "Conceptual outputs requested (non-file): " + ", ".join(existing_conceptual)
            if note not in notes:
                notes.append(note)
            contract["notes_for_engineers"] = notes

    # Build normalized structure
    artifact_requirements = {
        "required_files": required_files,
        "file_schemas": file_schemas
    }
    scored_schema_payload = {
        "required_columns": required_columns,
        "recommended_columns": scored_schema.get("recommended_columns", []),
    }
    if isinstance(scored_schema.get("required_any_of_groups"), list):
        scored_schema_payload["required_any_of_groups"] = scored_schema.get("required_any_of_groups", [])
    if isinstance(scored_schema.get("required_any_of_group_severity"), list):
        scored_schema_payload["required_any_of_group_severity"] = scored_schema.get(
            "required_any_of_group_severity",
            [],
        )
    if (
        scored_schema_present
        or bool(required_columns)
        or bool(scored_schema_payload.get("recommended_columns"))
        or bool(scored_schema_payload.get("required_any_of_groups"))
    ):
        artifact_requirements["scored_rows_schema"] = scored_schema_payload

    # Preserve schema_binding and clean_dataset if present (do not drop optional passthrough info)
    schema_binding = artifact_req.get("schema_binding")
    if isinstance(schema_binding, dict):
        artifact_requirements["schema_binding"] = schema_binding
    clean_dataset = artifact_req.get("clean_dataset")
    if isinstance(clean_dataset, dict):
        artifact_requirements["clean_dataset"] = clean_dataset

    def _extract_path(item: Any) -> str:
        if not item:
            return ""
        if isinstance(item, dict):
            path = item.get("path") or item.get("output") or item.get("artifact")
            return str(path) if path else ""
        return str(item)

    required_plots = artifact_req.get("required_plots", [])
    if not isinstance(required_plots, list):
        required_plots = []
    combined_outputs: List[str] = []
    for source in (required_files, required_plots):
        for entry in source:
            path = _extract_path(entry)
            if path and is_probably_path(path):
                combined_outputs.append(path)

    combined_output_paths = list(dict.fromkeys(combined_outputs))
    required_outputs_current = contract.get("required_outputs")
    required_output_artifacts = contract.get("required_output_artifacts")
    spec_extraction = contract.get("spec_extraction")
    deliverables = spec_extraction.get("deliverables") if isinstance(spec_extraction, dict) else None
    has_rich_parallel = bool(required_output_artifacts) if isinstance(required_output_artifacts, list) else False
    has_deliverables = bool(deliverables) if isinstance(deliverables, list) else False
    has_rich_required_outputs = (
        isinstance(required_outputs_current, list)
        and any(isinstance(item, dict) for item in required_outputs_current)
    )

    # Keep rich metadata payloads untouched; only backfill required_outputs when missing/empty.
    if not isinstance(required_outputs_current, list) or not required_outputs_current:
        contract["required_outputs"] = combined_output_paths
    elif has_rich_required_outputs or has_rich_parallel or has_deliverables:
        # Preserve rich required_outputs and/or parallel rich sources as-is.
        pass
    else:
        # Keep existing list[str] contract interface without forcing conversion from other formats.
        normalized_required_outputs: List[str] = []
        seen_required: set[str] = set()
        for item in required_outputs_current:
            if not isinstance(item, str):
                continue
            path = item.strip()
            if not path or not is_probably_path(path):
                continue
            key = path.lower()
            if key in seen_required:
                continue
            seen_required.add(key)
            normalized_required_outputs.append(path)
        if normalized_required_outputs:
            contract["required_outputs"] = normalized_required_outputs
        elif combined_output_paths:
            contract["required_outputs"] = combined_output_paths

    return artifact_requirements, warnings


def normalize_allowed_feature_sets(
    contract: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Normalize allowed_feature_sets to canonical dict format.
    
    Handles:
      - dict (preferred): normalizes values, unifies synonyms (forbidden, forbidden_features -> forbidden_for_modeling)
      - list[str]: converts to {"model_features": list, ...}
      - list[list[str]]: flattens and converts to dict
      - Other types (None, str, int): returns empty dict with error note
    
    Args:
        contract: The contract dictionary
        
    Returns:
        (normalized_dict, notes) where notes contains repair messages
    """
    notes: List[str] = []
    raw = contract.get("allowed_feature_sets")
    
    # Already a dict - normalize it
    if isinstance(raw, dict):
        normalized: Dict[str, Any] = {}
        
        # Synonym mapping for key unification
        synonym_map = {
            "forbidden": "forbidden_for_modeling",
            "forbidden_features": "forbidden_for_modeling",
        }
        
        for key, value in raw.items():
            # Unify synonyms
            canonical_key = synonym_map.get(key, key)
            if canonical_key != key:
                notes.append(f"Unified '{key}' to '{canonical_key}' in allowed_feature_sets")
            
            # Normalize values to list of unique strings
            if canonical_key == "rationale":
                # Rationale is always kept as a string
                if isinstance(value, str):
                    normalized[canonical_key] = value.strip() if value else ""
                else:
                    normalized[canonical_key] = str(value) if value else ""
            elif isinstance(value, list):
                cleaned = []
                seen = set()
                for item in value:
                    if isinstance(item, str) and item.strip():
                        s = item.strip()
                        if s not in seen:
                            cleaned.append(s)
                            seen.add(s)
                    elif isinstance(item, list):
                        # Flatten nested lists
                        for sub in item:
                            if isinstance(sub, str) and sub.strip():
                                s = sub.strip()
                                if s not in seen:
                                    cleaned.append(s)
                                    seen.add(s)
                normalized[canonical_key] = cleaned
            elif isinstance(value, str):
                # Single string value -> list
                normalized[canonical_key] = [value.strip()] if value.strip() else []
        
        # Ensure required keys exist
        for req_key in ("model_features", "segmentation_features", "audit_only_features", "forbidden_for_modeling"):
            if req_key not in normalized:
                normalized[req_key] = []
        if "rationale" not in normalized:
            normalized["rationale"] = ""
        
        return normalized, notes
    
    # list[str] -> dict
    if isinstance(raw, list):
        flat_features: List[str] = []
        seen = set()
        
        for item in raw:
            if isinstance(item, str) and item.strip():
                s = item.strip()
                if s not in seen:
                    flat_features.append(s)
                    seen.add(s)
            elif isinstance(item, list):
                # list[list[str]] - flatten
                for sub in item:
                    if isinstance(sub, str) and sub.strip():
                        s = sub.strip()
                        if s not in seen:
                            flat_features.append(s)
                            seen.add(s)
        
        if flat_features:
            notes.append(
                f"Normalized allowed_feature_sets from list to dict (model_features={len(flat_features)} features)"
            )
            return {
                "model_features": flat_features,
                "segmentation_features": [],
                "audit_only_features": [],
                "forbidden_for_modeling": [],
                "rationale": "normalized_from_list",
            }, notes
        else:
            notes.append("allowed_feature_sets was an empty list; normalized to empty dict")
            return {
                "model_features": [],
                "segmentation_features": [],
                "audit_only_features": [],
                "forbidden_for_modeling": [],
                "rationale": "normalized_from_empty_list",
            }, notes
    
    # None or unrecognized type
    if raw is None:
        return {
            "model_features": [],
            "segmentation_features": [],
            "audit_only_features": [],
            "forbidden_for_modeling": [],
            "rationale": "",
        }, []
    
    # Invalid type (str, int, etc.)
    notes.append(
        f"allowed_feature_sets had invalid type '{type(raw).__name__}'; cannot normalize"
    )
    return {
        "model_features": [],
        "segmentation_features": [],
        "audit_only_features": [],
        "forbidden_for_modeling": [],
        "rationale": "ERROR: invalid_type_in_source",
    }, notes


def normalize_validation_requirements(
    contract: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Normalize validation_requirements to canonical format.
    
    Rules:
      - Ensure dict type
      - Canonicalize metric names with _normalize_metric_name()
      - Move validation_requirements.metrics to metrics_to_report if latter missing
      - Ensure primary_metric is in metrics_to_report
      - If primary_metric missing, try to extract from qa_gates benchmark_kpi_report
    
    Args:
        contract: The contract dictionary
        
    Returns:
        (normalized_validation_requirements, notes)
    """
    notes: List[str] = []
    raw = contract.get("validation_requirements")
    
    if not isinstance(raw, dict):
        raw = {}
    
    # Deep copy to avoid mutation
    normalized = dict(raw)
    
    # Get or initialize metrics_to_report
    metrics_to_report = normalized.get("metrics_to_report")
    if not isinstance(metrics_to_report, list):
        metrics_to_report = []
    
    # Migration: validation_requirements.metrics -> metrics_to_report
    legacy_metrics = normalized.pop("metrics", None)
    if isinstance(legacy_metrics, list) and legacy_metrics:
        for m in legacy_metrics:
            if isinstance(m, str) and m.strip():
                canonical = _normalize_metric_name(m)
                if canonical not in [_normalize_metric_name(x) for x in metrics_to_report]:
                    metrics_to_report.append(canonical)
        notes.append(
            f"Migrated validation_requirements.metrics ({len(legacy_metrics)} items) to metrics_to_report"
        )
    
    # Canonicalize all metrics in metrics_to_report
    canonical_metrics = []
    seen = set()
    for m in metrics_to_report:
        if isinstance(m, str):
            c = _normalize_metric_name(m)
            if c and c not in seen:
                canonical_metrics.append(c)
                seen.add(c)
    normalized["metrics_to_report"] = canonical_metrics
    
    # Handle primary_metric
    primary_metric = normalized.get("primary_metric")
    if isinstance(primary_metric, str) and primary_metric.strip():
        primary_canonical = _normalize_metric_name(primary_metric)
        normalized["primary_metric"] = primary_canonical
        # Ensure it's in metrics_to_report
        if primary_canonical not in seen:
            canonical_metrics.insert(0, primary_canonical)
            notes.append(f"Added primary_metric '{primary_canonical}' to metrics_to_report")
    else:
        # Try to infer from qa_gates
        qa_gates = contract.get("qa_gates")
        if isinstance(qa_gates, list):
            for gate in qa_gates:
                if isinstance(gate, dict):
                    gate_name = gate.get("name", "").lower()
                    if "benchmark" in gate_name or "kpi" in gate_name or "metric" in gate_name:
                        params = gate.get("params", {})
                        if isinstance(params, dict):
                            metric = params.get("metric") or params.get("primary_metric")
                            if isinstance(metric, str) and metric.strip():
                                inferred = _normalize_metric_name(metric)
                                normalized["primary_metric"] = inferred
                                if inferred not in seen:
                                    canonical_metrics.insert(0, inferred)
                                notes.append(
                                    f"Inferred primary_metric '{inferred}' from qa_gates benchmark/kpi gate"
                                )
                                break
    
    normalized["metrics_to_report"] = canonical_metrics
    return normalized, notes


# =============================================================================
# CONTRACT SCHEMA LINTER (Fix #6)
# =============================================================================


def _normalize_role(role: str) -> str:
    """Normalize a role string to canonical form."""
    if not isinstance(role, str):
        return "feature"
    r = role.strip().lower()
    return ROLE_SYNONYM_MAP.get(r, r)


def _extract_role_from_value(value: Any) -> Tuple[str, bool]:
    """
    Extract role from a column_roles value that may be str or dict.

    Args:
        value: The value associated with a column in column_roles.
               Can be str (the role directly) or dict with role info.

    Returns:
        (role_string, was_extracted_from_dict)
        - role_string: The extracted and normalized role
        - was_extracted_from_dict: True if role was parsed from a dict
    """
    # Direct string role
    if isinstance(value, str):
        return _normalize_role(value), False

    # Dict with role info: {"role": "outcome", ...} or {"column_role": "target", ...}
    if isinstance(value, dict):
        for key in ("role", "column_role", "type", "category"):
            extracted = value.get(key)
            if isinstance(extracted, str) and extracted.strip():
                return _normalize_role(extracted), True
        # Dict but no recognizable role key
        return "feature", True

    # Other types (int, list, etc.) - can't extract
    return "feature", False


def _is_role_bucket_key(key: Any) -> bool:
    if not isinstance(key, str):
        return False
    normalized = re.sub(r"[^a-z0-9]+", "_", key.strip().lower()).strip("_")
    return normalized in {
        "pre_decision",
        "decision",
        "outcome",
        "post_decision_audit_only",
        "audit_only",
        "unknown",
        "identifiers",
        "time_columns",
    }


def _normalize_bucket_role_key(key: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", key.strip().lower()).strip("_")
    if normalized == "audit_only":
        return "post_decision_audit_only"
    return normalized


def _extract_role_columns(role_map: Any, role_name: str) -> List[str]:
    """Return columns for a role from either role->list or column->role formats."""
    if not isinstance(role_map, dict):
        return []
    role_name = _normalize_role(role_name)
    cols: List[str] = []

    # Format A: role -> list[str]
    if all(_is_role_bucket_key(k) for k in role_map.keys()):
        for key, values in role_map.items():
            bucket = _normalize_bucket_role_key(str(key))
            if _normalize_role(bucket) != role_name:
                continue
            if isinstance(values, list):
                cols.extend([str(v) for v in values if isinstance(v, str) and v.strip()])
        return list(dict.fromkeys(cols))

    # Format C: column -> role (string/dict)
    for col, raw_role in role_map.items():
        if not isinstance(col, str) or not col.strip():
            continue
        role_value, _ = _extract_role_from_value(raw_role)
        if _normalize_role(role_value) == role_name:
            cols.append(col.strip())
    return list(dict.fromkeys(cols))


# Supervised metric tokens - if primary_metric matches these, it's supervised learning
SUPERVISED_METRIC_TOKENS = {
    # Classification
    "accuracy", "roc_auc", "auc", "f1", "f1_score", "precision", "recall",
    "log_loss", "logloss", "cross_entropy", "balanced_accuracy",
    "cohen_kappa", "matthews_corrcoef", "mcc", "gini", "normalized_gini",
    # Regression
    "rmse", "mae", "mse", "r2", "rmsle", "mean_squared_error",
    "mean_absolute_error", "root_mean_squared_error",
}


def lint_column_roles(
    contract: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[str]]:
    """
    Lint and normalize column_roles to canonical dict format.

    Accepts:
      - dict (standard): {column: role, ...}
      - list[dict]: [{"column": "X", "role": "outcome"}, ...]
      - list[list/tuple]: [["X", "outcome"], ["Y", "feature"], ...]
      - str/int: error (irreparable)

    Returns:
        (normalized_dict, issues, notes)
        - normalized_dict: role->list (preferred V4.1) or {column: canonical_role}
        - issues: list of {rule, severity, message, item}
        - notes: repair notes for unknowns
    """
    issues: List[Dict[str, Any]] = []
    notes: List[str] = []
    raw = contract.get("column_roles")

    # None -> empty dict (ok)
    if raw is None:
        return {}, [], []

    # Already a dict - normalize roles
    if isinstance(raw, dict):
        # Preferred V4.1 format: role -> list[str]
        if raw and all(_is_role_bucket_key(k) for k in raw.keys()):
            normalized_bucket: Dict[str, List[str]] = {}
            for role_key, cols in raw.items():
                bucket = _normalize_bucket_role_key(str(role_key))
                if not isinstance(cols, list):
                    issues.append({
                        "rule": "contract_schema_lint.column_roles",
                        "severity": "warning",
                        "message": f"Role bucket '{role_key}' has non-list value; ignored",
                        "item": role_key,
                    })
                    continue
                cleaned_cols: List[str] = []
                seen: set[str] = set()
                for col in cols:
                    if not isinstance(col, str) or not col.strip():
                        continue
                    col_clean = col.strip()
                    col_norm = col_clean.lower()
                    if col_norm in seen:
                        continue
                    seen.add(col_norm)
                    cleaned_cols.append(col_clean)
                normalized_bucket[bucket] = cleaned_cols
            return normalized_bucket, issues, notes

        normalized: Dict[str, str] = {}
        for col, role_value in raw.items():
            if not isinstance(col, str) or not col.strip():
                continue
            col_clean = col.strip()

            # Extract role from value (handles both str and dict formats)
            role_norm, was_from_dict = _extract_role_from_value(role_value)

            if was_from_dict:
                # Log that we parsed role from dict
                issues.append({
                    "rule": "contract_schema_lint.column_roles_parsed",
                    "severity": "info",
                    "message": f"Parsed role '{role_norm}' for column '{col_clean}' from dict value",
                    "item": col_clean
                })
                notes.append(f"column_roles: parsed '{role_norm}' for '{col_clean}' from dict")

            # Validate the normalized role
            if role_norm not in CANONICAL_ROLES:
                original_role = role_value if isinstance(role_value, str) else str(role_value)
                issues.append({
                    "rule": "contract_schema_lint.column_roles",
                    "severity": "warning",
                    "message": f"Unknown role '{original_role}' for column '{col_clean}'; keeping as-is",
                    "item": col_clean
                })
                notes.append(f"column_roles: unknown role '{original_role}' for '{col_clean}', kept literal")
                # Keep the normalized version even if not canonical
                if isinstance(role_value, str):
                    role_norm = role_value.strip().lower()

            normalized[col_clean] = role_norm
        return normalized, issues, notes

    # list[dict] format: [{"column": "X", "role": "outcome"}, ...]
    if isinstance(raw, list):
        normalized = {}
        for item in raw:
            if isinstance(item, dict):
                col = item.get("column") or item.get("name") or item.get("col")
                role = item.get("role") or item.get("type") or "feature"
                if isinstance(col, str) and col.strip():
                    col_clean = col.strip()
                    role_norm = _normalize_role(role) if isinstance(role, str) else "feature"
                    if role_norm not in CANONICAL_ROLES:
                        issues.append({
                            "rule": "contract_schema_lint.column_roles",
                            "severity": "warning",
                            "message": f"Unknown role '{role}' for column '{col_clean}'; defaulting to 'feature'",
                            "item": col_clean
                        })
                        notes.append(f"column_roles: unknown role '{role}' for '{col_clean}', defaulted to 'feature'")
                        role_norm = "feature"
                    normalized[col_clean] = role_norm
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                # list[list/tuple] format: [["X", "outcome"], ...]
                col, role = item[0], item[1]
                if isinstance(col, str) and col.strip():
                    col_clean = col.strip()
                    role_norm = _normalize_role(role) if isinstance(role, str) else "feature"
                    if role_norm not in CANONICAL_ROLES:
                        issues.append({
                            "rule": "contract_schema_lint.column_roles",
                            "severity": "warning",
                            "message": f"Unknown role '{role}' for column '{col_clean}'; defaulting to 'feature'",
                            "item": col_clean
                        })
                        notes.append(f"column_roles: unknown role '{role}' for '{col_clean}', defaulted to 'feature'")
                        role_norm = "feature"
                    normalized[col_clean] = role_norm

        if normalized:
            notes.insert(0, f"column_roles: normalized from list format to dict ({len(normalized)} columns)")
            return normalized, issues, notes
        else:
            # Empty list or all items invalid
            notes.append("column_roles: list was empty or all items invalid, returning empty dict")
            return {}, issues, notes

    # Invalid type (str, int, etc.) - irreparable
    issues.append({
        "rule": "contract_schema_lint.column_roles",
        "severity": "fail",
        "message": f"column_roles has invalid type '{type(raw).__name__}'; must be dict or list",
        "item": "column_roles"
    })
    notes.append(f"column_roles: irreparable type '{type(raw).__name__}', returning empty dict")
    return {}, issues, notes


def lint_required_columns(
    required_columns: Any
) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
    """
    Lint scored_rows_schema.required_columns.

    Rules:
      - Must be list[str]
      - Strip, dedupe
      - Remove metric-like tokens
      - Remove path-like values (contains .csv/.json or '/')

    Returns:
        (clean_columns, issues, notes)
    """
    issues: List[Dict[str, Any]] = []
    notes: List[str] = []

    # Handle None
    if required_columns is None:
        return [], [], []

    # str -> [str]
    if isinstance(required_columns, str):
        required_columns = [required_columns]
        notes.append("required_columns: converted single string to list")

    # Not a list at this point -> try to extract strings
    if not isinstance(required_columns, list):
        issues.append({
            "rule": "contract_schema_lint.required_columns",
            "severity": "warning",
            "message": f"required_columns has invalid type '{type(required_columns).__name__}'; returning empty",
            "item": "required_columns"
        })
        notes.append(f"required_columns: invalid type '{type(required_columns).__name__}', returned empty")
        return [], issues, notes

    clean: List[str] = []
    seen: set = set()

    for item in required_columns:
        # Extract string value
        if isinstance(item, dict):
            col_name = item.get("name") or item.get("column") or ""
        elif isinstance(item, str):
            col_name = item
        else:
            # Non-string, non-dict -> skip
            continue

        if not isinstance(col_name, str) or not col_name.strip():
            continue

        col_clean = col_name.strip()
        col_lower = col_clean.lower()

        # Skip duplicates
        if col_lower in seen:
            continue

        # Check for metric-like token
        if _is_metric_like_token(col_clean):
            issues.append({
                "rule": "contract_schema_lint.required_columns",
                "severity": "warning",
                "message": f"Removed metric-like token '{col_clean}' from required_columns; metrics belong in metrics.json",
                "item": col_clean
            })
            notes.append(f"required_columns: removed metric '{col_clean}'")
            continue

        # Check for path-like value
        if is_file_path(col_clean):
            issues.append({
                "rule": "contract_schema_lint.required_columns",
                "severity": "warning",
                "message": f"Removed path-like value '{col_clean}' from required_columns; looks like a file path",
                "item": col_clean
            })
            notes.append(f"required_columns: removed path-like '{col_clean}'")
            continue

        clean.append(col_clean)
        seen.add(col_lower)

    return clean, issues, notes


def lint_allowed_feature_sets_coherence(
    contract: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[str]]:
    """
    Lint allowed_feature_sets for internal coherence.

    Rules:
      - forbidden_for_modeling ∩ model_features != ∅ → auto-repair (remove from model_features)
      - outcome/decision columns in model_features → warning + auto-repair

    Returns:
        (repaired_allowed_feature_sets, issues, notes)
    """
    issues: List[Dict[str, Any]] = []
    notes: List[str] = []

    allowed_sets = contract.get("allowed_feature_sets")
    if not isinstance(allowed_sets, dict):
        return allowed_sets if allowed_sets else {}, [], []

    # Deep copy to avoid mutation issues
    repaired = {k: list(v) if isinstance(v, list) else v for k, v in allowed_sets.items()}

    model_features = repaired.get("model_features", [])
    if not isinstance(model_features, list):
        model_features = []

    forbidden = repaired.get("forbidden_for_modeling", [])
    if not isinstance(forbidden, list):
        forbidden = []

    # Get outcome and decision columns
    column_roles = contract.get("column_roles", {})
    outcome_cols = set(_extract_role_columns(column_roles, "outcome"))
    decision_cols = set(_extract_role_columns(column_roles, "decision"))

    # Also check explicit outcome_columns and decision_columns
    for col in contract.get("outcome_columns", []) or []:
        if isinstance(col, str):
            outcome_cols.add(col)
    for col in contract.get("decision_columns", []) or []:
        if isinstance(col, str):
            decision_cols.add(col)

    # Check 1: forbidden_for_modeling ∩ model_features
    forbidden_set = set(forbidden)
    model_set = set(model_features)
    overlap = forbidden_set & model_set

    if overlap:
        for col in overlap:
            model_features.remove(col)
            issues.append({
                "rule": "contract_schema_lint.feature_set_coherence",
                "severity": "warning",
                "message": f"Column '{col}' in both model_features and forbidden_for_modeling; removed from model_features",
                "item": col
            })
            notes.append(f"feature_set_coherence: removed '{col}' from model_features (was in forbidden)")
        repaired["model_features"] = model_features

    # Check 2: outcome/decision columns in model_features (leakage-by-contract)
    leakage_cols = (outcome_cols | decision_cols) & set(model_features)
    if leakage_cols:
        for col in leakage_cols:
            model_features.remove(col)
            col_type = "outcome" if col in outcome_cols else "decision"
            issues.append({
                "rule": "contract_schema_lint.feature_set_coherence",
                "severity": "warning",
                "message": f"Leakage-by-contract: {col_type} column '{col}' found in model_features; removed",
                "item": col
            })
            notes.append(f"feature_set_coherence: removed {col_type} column '{col}' from model_features (leakage)")
        repaired["model_features"] = model_features

    return repaired, issues, notes


def lint_artifact_requirements_coherence(
    contract: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Lint artifact_requirements for coherence with required_outputs.

    Rules:
      - If scored_rows_schema exists and expects scored_rows file but missing in required_outputs → warning

    Returns:
        (issues, notes)
    """
    issues: List[Dict[str, Any]] = []
    notes: List[str] = []

    artifact_req = contract.get("artifact_requirements")
    if not isinstance(artifact_req, dict):
        return [], []

    scored_schema = artifact_req.get("scored_rows_schema")
    if not isinstance(scored_schema, dict):
        return [], []

    # Check if schema has required_columns (meaning scored_rows is expected)
    required_columns = scored_schema.get("required_columns", [])
    if not required_columns:
        return [], []

    # Check if scored_rows file is in required_files or required_outputs
    required_files = artifact_req.get("required_files", [])
    required_outputs = contract.get("required_outputs", [])

    has_scored_file = False

    # Check in required_files
    for f in required_files:
        path = f.get("path", "") if isinstance(f, dict) else str(f)
        if "scored" in path.lower() and path.endswith(".csv"):
            has_scored_file = True
            break

    # Check in required_outputs
    if not has_scored_file:
        for item in required_outputs:
            path = item.get("path", "") if isinstance(item, dict) else str(item)
            if "scored" in path.lower() and path.endswith(".csv"):
                has_scored_file = True
                break

    if not has_scored_file:
        issues.append({
            "rule": "contract_schema_lint.artifact_coherence",
            "severity": "warning",
            "message": "scored_rows_schema has required_columns but no scored_rows file in required_files/outputs",
            "item": "scored_rows_schema"
        })
        notes.append("artifact_coherence: scored_rows_schema defined but no scored_rows file in outputs")

    return issues, notes


def _is_supervised_contract(contract: Dict[str, Any]) -> bool:
    """
    Determine if the contract implies supervised learning.

    Checks:
      - validation_requirements.primary_metric is a supervised metric
      - validation_requirements.metrics_to_report contains supervised metrics
      - qa_gates contain benchmark/metric gates

    Returns:
        True if supervised learning is implied.
    """
    val_req = contract.get("validation_requirements")
    if isinstance(val_req, dict):
        # Check primary_metric
        primary = val_req.get("primary_metric")
        if isinstance(primary, str):
            primary_norm = _normalize_metric_name(primary)
            if primary_norm in SUPERVISED_METRIC_TOKENS:
                return True

        # Check metrics_to_report
        metrics = val_req.get("metrics_to_report", [])
        if isinstance(metrics, list):
            for m in metrics:
                if isinstance(m, str):
                    m_norm = _normalize_metric_name(m)
                    if m_norm in SUPERVISED_METRIC_TOKENS:
                        return True

    # Check qa_gates for benchmark gates
    qa_gates = contract.get("qa_gates", [])
    if isinstance(qa_gates, list):
        for gate in qa_gates:
            if isinstance(gate, dict):
                gate_name = str(gate.get("name", "")).lower()
                if any(kw in gate_name for kw in ("benchmark", "metric", "kpi")):
                    return True

    return False


def _infer_outcome_from_validation_requirements(contract: Dict[str, Any]) -> List[str]:
    """
    Attempt to infer outcome column from validation_requirements.params.

    Checks common keys like target, label_column, outcome_column.

    Returns:
        List of inferred outcome column names.
    """
    candidates: List[str] = []

    val_req = contract.get("validation_requirements")
    if isinstance(val_req, dict):
        params = val_req.get("params", {})
        if isinstance(params, dict):
            for key in ("target", "target_column", "label_column", "outcome_column", "y_column"):
                val = params.get(key)
                if isinstance(val, str) and val.strip() and val.lower() != "unknown":
                    candidates.append(val.strip())
                elif isinstance(val, list):
                    for v in val:
                        if isinstance(v, str) and v.strip() and v.lower() != "unknown":
                            candidates.append(v.strip())

    # Also check objective_analysis
    obj_analysis = contract.get("objective_analysis")
    if isinstance(obj_analysis, dict):
        for key in ("target_column", "outcome_column", "target", "outcome", "label"):
            val = obj_analysis.get(key)
            if isinstance(val, str) and val.strip() and val.lower() != "unknown":
                if val.strip() not in candidates:
                    candidates.append(val.strip())
            elif isinstance(val, list):
                for v in val:
                    if isinstance(v, str) and v.strip() and v.lower() != "unknown":
                        if v.strip() not in candidates:
                            candidates.append(v.strip())

    return candidates


def lint_outcome_presence_and_coherence(
    contract: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], List[str], bool]:
    """
    Lint outcome column presence and coherence for supervised learning contracts.

    Rules:
      - If supervised and outcome_columns empty → try to infer from column_roles/params
      - If still empty after inference → HARD error (status="error")
      - Ensure outcomes are in canonical_columns (auto-repair)
      - Ensure outcomes are NOT in model_features (auto-repair via forbidden_for_modeling)

    Args:
        contract: The contract dictionary (will be mutated for auto-repair)

    Returns:
        (issues, notes, has_critical_error)
    """
    issues: List[Dict[str, Any]] = []
    notes: List[str] = []
    has_critical_error = False

    # Determine if supervised
    is_supervised = _is_supervised_contract(contract)
    if not is_supervised:
        return issues, notes, has_critical_error

    # Collect outcome candidates
    outcome_candidates: List[str] = []

    # Source 1: column_roles (supports both role->list and column->role)
    column_roles = contract.get("column_roles", {})
    if isinstance(column_roles, dict):
        for col in _extract_role_columns(column_roles, "outcome"):
            if col not in outcome_candidates:
                outcome_candidates.append(col)

    # Source 2: explicit outcome_columns
    explicit_outcomes = contract.get("outcome_columns", [])
    if isinstance(explicit_outcomes, list):
        for oc in explicit_outcomes:
            if isinstance(oc, str) and oc.strip() and oc.lower() != "unknown":
                if oc.strip() not in outcome_candidates:
                    outcome_candidates.append(oc.strip())
    elif isinstance(explicit_outcomes, str) and explicit_outcomes.lower() != "unknown":
        if explicit_outcomes.strip() not in outcome_candidates:
            outcome_candidates.append(explicit_outcomes.strip())

    # Source 3: validation_requirements.params and objective_analysis
    inferred = _infer_outcome_from_validation_requirements(contract)
    for col in inferred:
        if col not in outcome_candidates:
            outcome_candidates.append(col)
            notes.append(f"outcome_coherence: inferred outcome '{col}' from validation_requirements/objective_analysis")

    # Source 4: target_candidates (from data_profile compact)
    if not outcome_candidates:
        target_candidates = contract.get("target_candidates")
        if isinstance(target_candidates, list):
            for item in target_candidates:
                if isinstance(item, dict):
                    col = item.get("column") or item.get("name") or item.get("candidate")
                    if isinstance(col, str) and col.strip():
                        outcome_candidates.append(col.strip())
                        notes.append(f"outcome_coherence: inferred outcome '{col.strip()}' from target_candidates")
                        break

    # If still no outcomes → HARD error for supervised contract
    if not outcome_candidates:
        issues.append({
            "rule": "contract_schema_lint.outcome_required",
            "severity": "error",
            "message": "Supervised learning contract requires outcome_columns but none found or inferable. "
                       "Specify outcome_columns or mark a column with role='outcome' in column_roles.",
            "item": "outcome_columns"
        })
        notes.append("outcome_coherence: CRITICAL - supervised contract has no outcome column")
        has_critical_error = True
        return issues, notes, has_critical_error

    # =========================================================================
    # Auto-repair: Ensure outcomes are properly configured
    # =========================================================================

    # Ensure outcome_columns is set
    contract["outcome_columns"] = outcome_candidates

    # Ensure outcomes are in canonical_columns
    canonical = contract.get("canonical_columns", [])
    if not isinstance(canonical, list):
        canonical = []

    canonical_lower = {c.lower() for c in canonical if isinstance(c, str)}
    added_to_canonical = []
    for oc in outcome_candidates:
        if oc.lower() not in canonical_lower:
            canonical.append(oc)
            canonical_lower.add(oc.lower())
            added_to_canonical.append(oc)

    if added_to_canonical:
        contract["canonical_columns"] = canonical
        issues.append({
            "rule": "contract_schema_lint.outcome_in_canonical",
            "severity": "warning",
            "message": f"Auto-added outcome column(s) {added_to_canonical} to canonical_columns",
            "item": added_to_canonical
        })
        notes.append(f"outcome_coherence: added {added_to_canonical} to canonical_columns")

    # Ensure outcomes are represented in column_roles while preserving existing format.
    if isinstance(column_roles, dict):
        role_bucket_format = bool(column_roles) and all(_is_role_bucket_key(k) for k in column_roles.keys())
        if role_bucket_format:
            outcome_bucket = column_roles.get("outcome")
            if not isinstance(outcome_bucket, list):
                outcome_bucket = []
            seen_outcomes = {str(c).strip().lower() for c in outcome_bucket if isinstance(c, str) and c.strip()}
            for oc in outcome_candidates:
                key = str(oc).strip().lower()
                if key in seen_outcomes:
                    continue
                outcome_bucket.append(oc)
                seen_outcomes.add(key)
                notes.append(f"outcome_coherence: appended '{oc}' to column_roles.outcome")
            column_roles["outcome"] = outcome_bucket
            contract["column_roles"] = column_roles
        else:
            for oc in outcome_candidates:
                current_role = column_roles.get(oc)
                role_value, _ = _extract_role_from_value(current_role)
                if _normalize_role(role_value) == "outcome":
                    continue
                column_roles[oc] = "outcome"
                if current_role:
                    notes.append(f"outcome_coherence: changed column_roles['{oc}'] from '{current_role}' to 'outcome'")
                else:
                    notes.append(f"outcome_coherence: set column_roles['{oc}'] = 'outcome'")
            contract["column_roles"] = column_roles

    # Ensure outcomes are in forbidden_for_modeling (to prevent leakage)
    allowed_sets = contract.get("allowed_feature_sets", {})
    if isinstance(allowed_sets, dict):
        forbidden = allowed_sets.get("forbidden_for_modeling", [])
        if not isinstance(forbidden, list):
            forbidden = []

        forbidden_lower = {f.lower() for f in forbidden if isinstance(f, str)}
        added_to_forbidden = []
        for oc in outcome_candidates:
            if oc.lower() not in forbidden_lower:
                forbidden.append(oc)
                forbidden_lower.add(oc.lower())
                added_to_forbidden.append(oc)

        if added_to_forbidden:
            allowed_sets["forbidden_for_modeling"] = forbidden
            contract["allowed_feature_sets"] = allowed_sets
            notes.append(f"outcome_coherence: added {added_to_forbidden} to forbidden_for_modeling")

    return issues, notes, has_critical_error


def run_contract_schema_linter(
    contract: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[str], bool]:
    """
    Run the full contract schema linter.

    This is the main entry point for Fix #6 + Fix #7 linting.
    Applies:
      A) column_roles normalization
      B) required_columns linting
      C) allowed_feature_sets coherence
      D) artifact_requirements coherence
      E) outcome presence and coherence (Fix #7)

    Args:
        contract: The contract dictionary (will be mutated)

    Returns:
        (contract, all_issues, all_notes, has_critical_error)
        - contract: The mutated contract with normalizations applied
        - all_issues: List of issue dicts with rule, severity, message, item
        - all_notes: List of repair notes for traceability
        - has_critical_error: True if a critical/unrecoverable error was found
    """
    all_issues: List[Dict[str, Any]] = []
    all_notes: List[str] = []
    has_critical_error = False

    # A) Lint column_roles
    normalized_roles, roles_issues, roles_notes = lint_column_roles(contract)
    contract["column_roles"] = normalized_roles
    all_issues.extend(roles_issues)
    all_notes.extend(roles_notes)

    # Check for critical error in column_roles
    for issue in roles_issues:
        if issue.get("severity") == "fail":
            has_critical_error = True

    # B) Lint required_columns in artifact_requirements
    artifact_req = contract.get("artifact_requirements")
    if isinstance(artifact_req, dict):
        scored_schema = artifact_req.get("scored_rows_schema")
        if isinstance(scored_schema, dict):
            raw_cols = scored_schema.get("required_columns")
            clean_cols, cols_issues, cols_notes = lint_required_columns(raw_cols)
            scored_schema["required_columns"] = clean_cols
            all_issues.extend(cols_issues)
            all_notes.extend(cols_notes)

    # C) Lint allowed_feature_sets coherence
    repaired_afs, afs_issues, afs_notes = lint_allowed_feature_sets_coherence(contract)
    if repaired_afs:
        contract["allowed_feature_sets"] = repaired_afs
    all_issues.extend(afs_issues)
    all_notes.extend(afs_notes)

    # D) Lint artifact_requirements coherence
    artifact_issues, artifact_notes = lint_artifact_requirements_coherence(contract)
    all_issues.extend(artifact_issues)
    all_notes.extend(artifact_notes)

    # E) Lint outcome presence and coherence (Fix #7)
    outcome_issues, outcome_notes, outcome_critical = lint_outcome_presence_and_coherence(contract)
    all_issues.extend(outcome_issues)
    all_notes.extend(outcome_notes)
    if outcome_critical:
        has_critical_error = True

    return contract, all_issues, all_notes, has_critical_error


def lint_scored_rows_schema(
    contract: Dict[str, Any]
) -> Tuple[List[str], List[str], List[str]]:
    """
    Lint scored_rows_schema.required_columns to remove metric-like tokens.
    
    Metrics (accuracy, roc_auc, f1, etc.) belong in metrics.json, not per-row columns.
    
    Args:
        contract: The contract dictionary
        
    Returns:
        (clean_required_columns, removed_metric_like, notes)
    """
    notes: List[str] = []
    removed: List[str] = []
    
    artifact_req = contract.get("artifact_requirements")
    if not isinstance(artifact_req, dict):
        return [], [], []
    
    scored_schema = artifact_req.get("scored_rows_schema")
    if not isinstance(scored_schema, dict):
        return [], [], []
    
    required_columns = scored_schema.get("required_columns")
    if not isinstance(required_columns, list):
        return [], [], []
    
    clean_columns: List[str] = []
    for col in required_columns:
        col_name = col.get("name") if isinstance(col, dict) else str(col) if col else ""
        if not col_name:
            continue
        
        if _is_metric_like_token(col_name):
            removed.append(col_name)
            notes.append(
                f"Removed metric-like required column '{col_name}' from scored_rows_schema; "
                "metrics belong in metrics.json"
            )
        else:
            # Preserve original format (str or dict)
            clean_columns.append(col if isinstance(col, dict) else col_name)
    
    return clean_columns, removed, notes


def validate_contract(contract: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate contract for self-consistency and apply strict normalization.

    Rules:
    1) allowed_feature_sets must be subset of (canonical_columns ∪ derived_columns ∪ expandable)
    2) outcome_columns must be in canonical_columns (or explicitly unknown)
    3) decision_columns only if action_space/levers declared
    4) artifact_requirements.required_files must exist as list
    5) Contract schema linting (Fix #6): column_roles, required_columns, coherence checks

    Returns:
        {
            "status": "ok" | "warning" | "error",
            "issues": [...],
            "normalized_artifact_requirements": {...}
        }
    """
    issues = []
    status = "ok"

    if not isinstance(contract, dict):
        return {
            "status": "error",
            "issues": [{"rule": "structure", "severity": "error", "message": "Contract must be a dictionary"}],
            "normalized_artifact_requirements": None
        }

    # Ensure unknowns is a list for traceability
    if not isinstance(contract.get("unknowns"), list):
        contract["unknowns"] = []

    # =========================================================================
    # STRICT NORMALIZATION (auto-repair)
    # =========================================================================
    
    # 1) Normalize allowed_feature_sets
    normalized_afs, afs_notes = normalize_allowed_feature_sets(contract)
    contract["allowed_feature_sets"] = normalized_afs
    
    # Check for critical failure (invalid type that couldn't be normalized)
    afs_has_error = normalized_afs.get("rationale", "").startswith("ERROR:")
    
    for note in afs_notes:
        is_error = "invalid type" in note.lower()
        issues.append({
            "rule": "contract_normalization",
            "severity": "error" if is_error else "warning",
            "message": f"[allowed_feature_sets] {note}",
            "item": "allowed_feature_sets"
        })
        if is_error:
            status = "error"
        elif status == "ok":
            status = "warning"
        
        # Add to unknowns for traceability
        unknowns = contract.get("unknowns")
        if isinstance(unknowns, list):
            unknowns.append(f"Normalized allowed_feature_sets: {note}")
    
    # 2) Normalize validation_requirements
    normalized_val_req, val_req_notes = normalize_validation_requirements(contract)
    contract["validation_requirements"] = normalized_val_req
    
    for note in val_req_notes:
        issues.append({
            "rule": "contract_normalization",
            "severity": "warning",
            "message": f"[validation_requirements] {note}",
            "item": "validation_requirements"
        })
        if status == "ok":
            status = "warning"
        
        # Add to unknowns for traceability
        unknowns = contract.get("unknowns")
        if isinstance(unknowns, list):
            unknowns.append(f"Normalized validation_requirements: {note}")
    
    # 3) Lint scored_rows_schema (remove metric-like columns)
    clean_cols, removed_metrics, lint_notes = lint_scored_rows_schema(contract)
    if removed_metrics:
        # Update the contract
        artifact_req_for_lint = contract.get("artifact_requirements")
        if isinstance(artifact_req_for_lint, dict):
            scored_schema = artifact_req_for_lint.get("scored_rows_schema")
            if isinstance(scored_schema, dict):
                scored_schema["required_columns"] = clean_cols
    
    for note in lint_notes:
        issues.append({
            "rule": "contract_normalization",
            "severity": "warning",
            "message": f"[scored_rows_schema] {note}",
            "item": "scored_rows_schema"
        })
        if status == "ok":
            status = "warning"
        
        # Add to unknowns for traceability
        unknowns = contract.get("unknowns")
        if isinstance(unknowns, list):
            unknowns.append(note)
    
    # Validate allowed_feature_sets is now a proper dict
    if afs_has_error:
        issues.append({
            "rule": "allowed_feature_sets_type",
            "severity": "error",
            "message": "allowed_feature_sets could not be normalized to a valid dict; contract is unusable",
            "item": "allowed_feature_sets"
        })
        status = "error"

    # =========================================================================
    # NORMALIZE ARTIFACT REQUIREMENTS (existing logic)
    # =========================================================================
    
    # Normalize artifact_requirements
    artifact_req, normalization_warnings = normalize_artifact_requirements(contract)
    contract["artifact_requirements"] = artifact_req  # Ensure it's set before linter runs
    for w in normalization_warnings:
        issues.append({
            "rule": "output_ambiguity",
            "severity": "warning",
            "message": w["message"],
            "item": w["item"]
        })
        if status == "ok":
            status = "warning"

    # =========================================================================
    # CONTRACT SCHEMA LINTER (Fix #6 + Fix #7)
    # =========================================================================
    _, linter_issues, linter_notes, linter_critical = run_contract_schema_linter(contract)

    # Process linter issues
    for linter_issue in linter_issues:
        issues.append(linter_issue)
        severity = linter_issue.get("severity", "warning")
        if severity == "fail" or severity == "error":
            status = "error"
        elif status == "ok" and severity != "info":
            status = "warning"

    # Critical error from linter (e.g., missing outcome for supervised contract)
    if linter_critical:
        status = "error"

    # Add linter notes to unknowns for traceability
    unknowns = contract.get("unknowns")
    if isinstance(unknowns, list):
        for note in linter_notes:
            if note not in unknowns:
                unknowns.append(note)

    # Update artifact_req after linter may have modified it
    artifact_req = contract.get("artifact_requirements", artifact_req)

    # Get column sets
    canonical_columns = set(contract.get("canonical_columns", []) or [])
    derived_columns = set(contract.get("derived_columns", []) or [])

    # Get feature selectors expandable columns (placeholder for now)
    selectors = contract.get("feature_selectors", [])
    expandable = set()
    # Feature selectors will be expanded later with actual data

    all_known_columns = canonical_columns | derived_columns | expandable

    # Rule 1: allowed_feature_sets validation (now guaranteed to be a dict after normalization)
    allowed_sets = contract.get("allowed_feature_sets", {})
    if isinstance(allowed_sets, dict):
        for set_name, features in allowed_sets.items():
            # Skip forbidden set and non-feature keys
            if set_name in ("forbidden", "forbidden_for_modeling", "rationale"):
                continue
            if isinstance(features, list):
                for feat in features:
                    if _is_selector_set_alias(feat, selectors):
                        continue
                    if feat not in all_known_columns and not _matches_any_selector(feat, selectors):
                        # Only warn if we have canonical_columns defined
                        if canonical_columns:
                            issues.append({
                                "rule": "feature_set_consistency",
                                "severity": "warning",
                                "message": f"Feature '{feat}' in allowed_feature_sets.{set_name} not in canonical_columns",
                                "item": feat
                            })
                            if status == "ok":
                                status = "warning"

    # Rule 2: outcome_columns validation
    outcome_columns = contract.get("outcome_columns", [])
    if isinstance(outcome_columns, list):
        for oc in outcome_columns:
            if oc and oc not in all_known_columns:
                # Only warn if canonical_columns is defined
                if canonical_columns:
                    issues.append({
                        "rule": "outcome_column_consistency",
                        "severity": "warning",
                        "message": f"Outcome column '{oc}' not in canonical_columns",
                        "item": oc
                    })
                    if status == "ok":
                        status = "warning"

    # Rule 3: decision_columns validation
    decision_columns = contract.get("decision_columns", [])
    has_action_space = bool(contract.get("action_space") or contract.get("levers"))

    if decision_columns and not has_action_space:
        # Check if any evidence of editable levers
        business_objective = contract.get("business_objective", "")
        has_lever_evidence = any(
            kw in str(business_objective).lower()
            for kw in ["price", "precio", "discount", "descuento", "limit", "límite", "offer", "oferta"]
        )
        if not has_lever_evidence:
            issues.append({
                "rule": "decision_columns_without_levers",
                "severity": "warning",
                "message": f"decision_columns={decision_columns} declared but no action_space/levers defined",
                "item": decision_columns
            })
            if status == "ok":
                status = "warning"

    # Rule 4: artifact_requirements.required_files must exist
    req_files = artifact_req.get("required_files", [])
    if not isinstance(req_files, list) or not req_files:
        issues.append({
            "rule": "missing_required_files",
            "severity": "warning",
            "message": "artifact_requirements.required_files is empty or missing",
            "item": None
        })
        if status == "ok":
            status = "warning"

    return {
        "status": status,
        "issues": issues,
        "normalized_artifact_requirements": artifact_req
    }


def _strict_issue(rule: str, severity: str, message: str, item: Any = None) -> Dict[str, Any]:
    return {
        "rule": rule,
        "severity": severity,
        "message": message,
        "item": item,
    }


def _status_from_issues(issues: List[Dict[str, Any]]) -> str:
    has_error = any(str(issue.get("severity", "")).lower() in {"error", "fail"} for issue in issues)
    if has_error:
        return "error"
    has_warning = any(str(issue.get("severity", "")).lower() == "warning" for issue in issues)
    if has_warning:
        return "warning"
    return "ok"


def validate_contract_readonly(contract: Dict[str, Any]) -> Dict[str, Any]:
    """
    Strict contract validation without mutating input payload.

    This validator enforces planner acceptance policy:
    - schema/type integrity for core contract keys
    - single ontology for roles and gates
    - outputs as paths
    - semantic checks from validate_contract(copy)
    """
    issues: List[Dict[str, Any]] = []

    if not isinstance(contract, dict):
        issues.append(
            _strict_issue(
                "contract.structure",
                "error",
                "Contract must be a dictionary.",
                type(contract).__name__,
            )
        )
        return {
            "status": "error",
            "accepted": False,
            "issues": issues,
            "summary": {"error_count": 1, "warning_count": 0},
        }

    required_keys = {
        "contract_version": str,
        "strategy_title": str,
        "business_objective": str,
        "output_dialect": dict,
        "canonical_columns": list,
        "column_roles": dict,
        "allowed_feature_sets": dict,
        "artifact_requirements": dict,
        "required_outputs": list,
        "validation_requirements": dict,
        "qa_gates": list,
        "cleaning_gates": list,
        "reviewer_gates": list,
        "data_engineer_runbook": dict,
        "ml_engineer_runbook": dict,
        "iteration_policy": dict,
    }
    for key, expected_type in required_keys.items():
        value = contract.get(key)
        if value is None:
            issues.append(
                _strict_issue(
                    "contract.required_key_missing",
                    "error",
                    f"Missing required top-level key '{key}'.",
                    key,
                )
            )
            continue
        if not isinstance(value, expected_type):
            issues.append(
                _strict_issue(
                    "contract.required_key_type",
                    "error",
                    f"Key '{key}' must be of type {expected_type.__name__}.",
                    key,
                )
            )
            continue
        if expected_type in (list, dict) and not value:
            issues.append(
                _strict_issue(
                    "contract.required_key_empty",
                    "error",
                    f"Key '{key}' must not be empty.",
                    key,
                )
            )
        if expected_type is str and not str(value).strip():
            issues.append(
                _strict_issue(
                    "contract.required_key_empty",
                    "error",
                    f"Key '{key}' must not be blank.",
                    key,
                )
            )

    contract_version = contract.get("contract_version")
    if str(contract_version) != "4.1":
        issues.append(
            _strict_issue(
                "contract.version",
                "error",
                "contract_version must be exactly '4.1'.",
                contract_version,
            )
        )

    legacy_keys = sorted(
        {
            "spec_extraction",
            "data_requirements",
            "role_runbooks",
            "validations",
            "quality_gates",
            "execution_plan",
            "artifact_schemas",
            "required_columns",
            "feature_availability",
            "decision_variables",
            "availability_summary",
        }
        & set(contract.keys())
    )
    if legacy_keys:
        issues.append(
            _strict_issue(
                "contract.legacy_keys",
                "error",
                "Legacy keys are not allowed in V4.1 contract.",
                legacy_keys,
            )
        )

    role_map = contract.get("column_roles")
    role_values: Dict[str, List[str]] = {}
    if not isinstance(role_map, dict):
        issues.append(
            _strict_issue(
                "contract.column_roles_type",
                "error",
                "column_roles must be a dictionary in role->list format.",
                "column_roles",
            )
        )
    elif not role_map:
        issues.append(
            _strict_issue(
                "contract.column_roles_empty",
                "error",
                "column_roles cannot be empty.",
                "column_roles",
            )
        )
    else:
        non_canonical_keys: List[str] = []
        unknown_roles: List[str] = []
        non_list_roles: List[str] = []
        non_string_columns: List[Any] = []
        blank_columns: List[str] = []

        for raw_role_name, columns in role_map.items():
            role_name = str(raw_role_name).strip()
            role_norm = _normalize_bucket_role_key(role_name)
            if role_name != role_norm:
                non_canonical_keys.append(role_name)
            if role_norm not in STRICT_ROLE_BUCKETS:
                unknown_roles.append(role_name)
                continue
            if not isinstance(columns, list):
                non_list_roles.append(role_name)
                continue

            cleaned_columns: List[str] = []
            for col in columns:
                if not isinstance(col, str):
                    non_string_columns.append(col)
                    continue
                col_clean = col.strip()
                if not col_clean:
                    blank_columns.append(role_name)
                    continue
                cleaned_columns.append(col_clean)
            role_values[role_norm] = cleaned_columns

        if non_canonical_keys:
            issues.append(
                _strict_issue(
                    "contract.role_ontology",
                    "error",
                    "column_roles keys must use canonical bucket names only.",
                    sorted(list(dict.fromkeys(non_canonical_keys))),
                )
            )
        if unknown_roles:
            issues.append(
                _strict_issue(
                    "contract.role_ontology",
                    "error",
                    "column_roles contains unknown role buckets outside the shared ontology.",
                    sorted(list(dict.fromkeys(unknown_roles))),
                )
            )
        if non_list_roles:
            issues.append(
                _strict_issue(
                    "contract.column_roles_format",
                    "error",
                    "column_roles must be role->list[str]; found non-list bucket values.",
                    sorted(list(dict.fromkeys(non_list_roles))),
                )
            )
        if non_string_columns:
            issues.append(
                _strict_issue(
                    "contract.column_roles_format",
                    "error",
                    "column_roles bucket entries must be strings.",
                    non_string_columns[:10],
                )
            )
        if blank_columns:
            issues.append(
                _strict_issue(
                    "contract.column_roles_format",
                    "error",
                    "column_roles bucket entries must not be blank strings.",
                    sorted(list(dict.fromkeys(blank_columns))),
                )
            )

    if role_values and not role_values.get("pre_decision"):
        issues.append(
            _strict_issue(
                "contract.role_ontology",
                "error",
                "column_roles must include non-empty 'pre_decision' bucket.",
                "pre_decision",
            )
        )

    required_outputs = contract.get("required_outputs")
    if isinstance(required_outputs, list):
        if not required_outputs:
            issues.append(
                _strict_issue(
                    "contract.required_outputs",
                    "error",
                    "required_outputs cannot be empty.",
                    "required_outputs",
                )
            )
        seen_paths: set[str] = set()
        duplicate_paths: List[str] = []
        for item in required_outputs:
            path, path_error = _extract_required_output_path(item)
            if path_error == "invalid_type":
                issues.append(
                    _strict_issue(
                        "contract.required_outputs_type",
                        "error",
                        "required_outputs entries must be strings or objects with a non-empty 'path'.",
                        item,
                    )
                )
                continue
            if path_error == "missing_path":
                issues.append(
                    _strict_issue(
                        "contract.required_outputs_path",
                        "error",
                        "required_outputs object entries must include a non-empty string field 'path'.",
                        item,
                    )
                )
                continue
            if path_error == "path_not_string":
                issues.append(
                    _strict_issue(
                        "contract.required_outputs_path",
                        "error",
                        "required_outputs object field 'path' must be a string.",
                        item,
                    )
                )
                continue
            if path_error == "empty_path":
                issues.append(
                    _strict_issue(
                        "contract.required_outputs_path",
                        "error",
                        "required_outputs entries must not contain empty path values.",
                        item,
                    )
                )
                continue
            if not is_file_path(path):
                issues.append(
                    _strict_issue(
                        "contract.required_outputs_path",
                        "error",
                        "required_outputs entries must be file-like paths (not logical labels).",
                        item,
                    )
                )
                continue
            normalized_path = _normalize_required_output_path(path)
            if normalized_path in seen_paths:
                duplicate_paths.append(path)
            else:
                seen_paths.add(normalized_path)
        if duplicate_paths:
            issues.append(
                _strict_issue(
                    "contract.required_outputs_duplicates",
                    "warning",
                    "required_outputs contains duplicate artifact paths.",
                    sorted(list(dict.fromkeys(duplicate_paths))),
                )
            )
    elif required_outputs is not None:
        issues.append(
            _strict_issue(
                "contract.required_outputs_type",
                "error",
                "required_outputs must be a list.",
                type(required_outputs).__name__,
            )
        )

    optimization_policy = contract.get("optimization_policy")
    if optimization_policy is None:
        issues.append(
            _strict_issue(
                "contract.optimization_policy_missing",
                "warning",
                "optimization_policy missing; runtime will use safe optimization defaults.",
                get_default_optimization_policy(),
            )
        )
    elif not isinstance(optimization_policy, dict):
        issues.append(
            _strict_issue(
                "contract.optimization_policy_type",
                "error",
                "optimization_policy must be an object when present.",
                type(optimization_policy).__name__,
            )
        )
    else:
        missing_keys = [key for key in OPTIMIZATION_POLICY_DEFAULTS.keys() if key not in optimization_policy]
        if missing_keys:
            issues.append(
                _strict_issue(
                    "contract.optimization_policy_missing_keys",
                    "warning",
                    "optimization_policy missing keys; runtime will fill defaults.",
                    missing_keys,
                )
            )
        for key in OPTIMIZATION_POLICY_BOOL_KEYS:
            if key not in optimization_policy:
                continue
            value = optimization_policy.get(key)
            if not isinstance(value, (bool, int, float, str)):
                issues.append(
                    _strict_issue(
                        "contract.optimization_policy_value",
                        "error",
                        f"optimization_policy.{key} must be boolean-like.",
                        value,
                    )
                )
        for key in OPTIMIZATION_POLICY_INT_KEYS:
            if key not in optimization_policy:
                continue
            value = optimization_policy.get(key)
            min_value = 0 if key == "patience" else 1
            try:
                numeric = int(float(value))
                if numeric < min_value:
                    raise ValueError("below minimum")
            except Exception:
                issues.append(
                    _strict_issue(
                        "contract.optimization_policy_value",
                        "error",
                        f"optimization_policy.{key} must be an integer >= {min_value}.",
                        value,
                    )
                )
        if "min_delta" in optimization_policy:
            try:
                min_delta = float(optimization_policy.get("min_delta"))
                if min_delta < 0:
                    raise ValueError("below minimum")
            except Exception:
                issues.append(
                    _strict_issue(
                        "contract.optimization_policy_value",
                        "error",
                        "optimization_policy.min_delta must be a number >= 0.",
                        optimization_policy.get("min_delta"),
                    )
                )

    gate_keys = ("qa_gates", "cleaning_gates", "reviewer_gates")
    for gate_key in gate_keys:
        gates = contract.get(gate_key)
        if not isinstance(gates, list):
            issues.append(
                _strict_issue(
                    "contract.gates_type",
                    "error",
                    f"{gate_key} must be a list.",
                    gate_key,
                )
            )
            continue
        if not gates:
            issues.append(
                _strict_issue(
                    "contract.gates_empty",
                    "error",
                    f"{gate_key} cannot be empty.",
                    gate_key,
                )
            )
        for gate in gates:
            if not isinstance(gate, dict):
                issues.append(
                    _strict_issue(
                        "contract.gate_entry_type",
                        "error",
                        f"All {gate_key} entries must be objects.",
                        gate,
                    )
                )
                continue
            gate_name = gate.get("name")
            if not isinstance(gate_name, str) or not gate_name.strip():
                issues.append(
                    _strict_issue(
                        "contract.gate_name",
                        "error",
                        f"Gate in {gate_key} is missing valid name.",
                        gate,
                    )
                )
            severity = str(gate.get("severity") or "").upper()
            if severity not in {"HARD", "SOFT"}:
                issues.append(
                    _strict_issue(
                        "contract.gate_severity",
                        "error",
                        f"Gate '{gate_name or '<unknown>'}' in {gate_key} has invalid severity.",
                        severity,
                    )
                )

    # Semantic/lint checks from current validator logic, but on a deep copy only.
    try:
        semantic_result = validate_contract(copy.deepcopy(contract))
    except Exception as err:
        issues.append(
            _strict_issue(
                "contract.semantic_validation_exception",
                "error",
                f"Semantic validation crashed: {err}",
                "validate_contract",
            )
        )
        semantic_result = {"status": "error", "issues": []}

    semantic_issues = semantic_result.get("issues") if isinstance(semantic_result, dict) else []
    for item in semantic_issues if isinstance(semantic_issues, list) else []:
        if not isinstance(item, dict):
            continue
        severity = str(item.get("severity") or "warning").lower()
        if severity in {"error", "fail"}:
            issues.append(item)

    status = _status_from_issues(issues)
    error_count = sum(1 for issue in issues if str(issue.get("severity", "")).lower() in {"error", "fail"})
    warning_count = sum(1 for issue in issues if str(issue.get("severity", "")).lower() == "warning")
    return {
        "status": status,
        "accepted": status != "error",
        "issues": issues,
        "summary": {
            "error_count": error_count,
            "warning_count": warning_count,
        },
    }


def normalize_contract_scope(scope_value: Any) -> str:
    token = str(scope_value or "").strip().lower()
    if token in CONTRACT_SCOPE_VALUES:
        return token
    if token in {"cleaning", "clean", "clean_only"}:
        return "cleaning_only"
    if token in {"ml", "training", "modeling", "model_only"}:
        return "ml_only"
    return "full_pipeline"


def _gate_has_consumable_name(gate: Dict[str, Any]) -> bool:
    if not isinstance(gate, dict):
        return False
    for key in ("name", "id", "gate", "metric", "check", "rule", "title", "label"):
        value = gate.get(key)
        if isinstance(value, str) and value.strip():
            return True
    return False


def _gate_list_valid(gates: Any) -> bool:
    if not isinstance(gates, list) or not gates:
        return False
    for gate in gates:
        if isinstance(gate, str) and gate.strip():
            continue
        if isinstance(gate, dict) and _gate_has_consumable_name(gate):
            continue
        return False
    return True


def _runbook_present(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(bool(str(item).strip()) for item in value if item is not None)
    if isinstance(value, dict):
        if "steps" in value and isinstance(value.get("steps"), list):
            return any(bool(str(step).strip()) for step in value.get("steps") or [] if step is not None)
        return bool(value)
    return False


_DROP_COLUMN_DIRECTIVE_PATTERN = re.compile(
    r"\b(drop|discard|remove|eliminar|descartar|quitar)\b.{0,32}\b(column|columns|columna|columnas)\b",
    re.IGNORECASE,
)
_SCALE_ACTION_PATTERN = re.compile(
    r"\b("
    r"scale|scaling|rescale|rescaling|"
    r"standardize|standardise|standardization|standardisation|"
    r"normalize|normalise|normalization|normalisation|normalizar|"
    r"minmax|min-max|zscore|z-score|escalar|estandarizar"
    r")\b",
    re.IGNORECASE,
)
_SCALE_TARGET_PATTERN = re.compile(
    r"\b("
    r"column|columns|columna|columnas|"
    r"feature|features|variable|variables|"
    r"predictor|predictors|input|inputs|numeric|numerical"
    r")\b",
    re.IGNORECASE,
)
_SCHEMA_STANDARDIZATION_PATTERN = re.compile(
    r"\b(schema|schema_version|column\s+name|column\s+names|naming|format)\b",
    re.IGNORECASE,
)
_DTYPE_QUALIFIER_PATTERN = re.compile(
    r"\b(dtype|dtypes|data\s*type|data\s*types|tipo\s*de\s*dato|tipos\s*de\s*datos)\b",
    re.IGNORECASE,
)
_ACTION_NEGATION_PATTERN = re.compile(
    r"(do\s+not|don't|must\s+not|never|avoid|forbid|forbidden|prohibido|no\s+debe|no\b|sin\b)",
    re.IGNORECASE,
)
_COLUMN_REF_KEYS = {
    "column",
    "columns",
    "required_column",
    "required_columns",
    "target_column",
    "target_columns",
    "feature",
    "features",
    "field",
    "fields",
    "column_name",
    "column_names",
    "canonical_column",
    "canonical_columns",
}


def _normalize_nonempty_str_list(value: Any) -> Tuple[List[str], List[Any]]:
    if value is None:
        return [], []
    if isinstance(value, str):
        cleaned = value.strip()
        return ([cleaned] if cleaned else []), []
    if not isinstance(value, list):
        return [], [value]
    cleaned: List[str] = []
    invalid: List[Any] = []
    for item in value:
        if isinstance(item, str):
            item_clean = item.strip()
            if item_clean:
                cleaned.append(item_clean)
            continue
        invalid.append(item)
    deduped = list(dict.fromkeys(cleaned))
    return deduped, invalid


def _looks_like_selector_token(value: str) -> bool:
    """
    Heuristic detector for selector-like feature tokens (wildcards/regex patterns).

    These tokens are valid as family hints but are not concrete column names and
    must not be validated as literal columns.
    """
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


def _flatten_runbook_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [_flatten_runbook_text(item) for item in value]
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        parts: List[str] = []
        for key, item in value.items():
            if isinstance(item, (str, list, dict)):
                parts.append(_flatten_runbook_text(item))
            elif item is not None:
                parts.append(str(item))
            if key in {"must", "must_not", "steps", "reasoning_checklist", "validation_checklist"}:
                continue
        return "\n".join(part for part in parts if part)
    if value is None:
        return ""
    return str(value)


def _has_non_negated_action(text: str, pattern: re.Pattern[str]) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    for match in pattern.finditer(text):
        prefix = text[max(0, match.start() - 36):match.start()]
        if _ACTION_NEGATION_PATTERN.search(prefix):
            continue
        return True
    return False


def _has_non_negated_feature_scaling_action(text: str) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    for match in _SCALE_ACTION_PATTERN.finditer(text):
        clause_start = max(
            text.rfind("\n", 0, match.start()),
            text.rfind(".", 0, match.start()),
            text.rfind(";", 0, match.start()),
            text.rfind(":", 0, match.start()),
        )
        clause_start = 0 if clause_start < 0 else clause_start + 1
        clause_end_candidates = [
            idx for idx in (
                text.find("\n", match.end()),
                text.find(".", match.end()),
                text.find(";", match.end()),
                text.find(":", match.end()),
            ) if idx >= 0
        ]
        clause_end = min(clause_end_candidates) if clause_end_candidates else len(text)
        clause = text[clause_start:clause_end]
        prefix = clause[max(0, match.start() - clause_start - 36): match.start() - clause_start]
        if _ACTION_NEGATION_PATTERN.search(prefix):
            continue
        action_start = match.start() - clause_start
        action_end = match.end() - clause_start
        target_matches = list(_SCALE_TARGET_PATTERN.finditer(clause))
        target_is_local = any(
            min(abs(target.start() - action_end), abs(action_start - target.end())) <= 24
            for target in target_matches
        )
        # "dtype normalization" / "data type normalization" is type coercion, not feature scaling.
        if _DTYPE_QUALIFIER_PATTERN.search(clause):
            continue
        if target_is_local:
            return True
        # Ignore schema/format standardization; that is structural cleaning, not feature scaling.
        if _SCHEMA_STANDARDIZATION_PATTERN.search(clause):
            continue
    return False


def _extract_columns_from_value(value: Any) -> List[str]:
    values: List[str] = []
    if isinstance(value, str):
        val = value.strip()
        if val:
            values.append(val)
        return values
    if isinstance(value, list):
        for item in value:
            values.extend(_extract_columns_from_value(item))
        return values
    if isinstance(value, dict):
        for key, item in value.items():
            key_norm = str(key).strip().lower()
            if key_norm in _COLUMN_REF_KEYS:
                values.extend(_extract_columns_from_value(item))
                continue
            if key_norm == "params":
                values.extend(_extract_columns_from_value(item))
        return values
    return values


def _collect_gate_column_refs(gates: Any, hard_only: bool = True) -> List[str]:
    if not isinstance(gates, list):
        return []
    refs: List[str] = []
    for gate in gates:
        if not isinstance(gate, dict):
            continue
        severity = str(gate.get("severity") or "HARD").strip().upper()
        if hard_only and severity != "HARD":
            continue
        refs.extend(_extract_columns_from_value(gate))
    return list(dict.fromkeys([ref for ref in refs if isinstance(ref, str) and ref.strip()]))


def _resolve_cleaning_column_transformations(clean_dataset: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    transform_block = clean_dataset.get("column_transformations")
    issues: List[str] = []
    if transform_block is None:
        transform_block = {}
    if not isinstance(transform_block, dict):
        return {"drop_columns": [], "scale_columns": [], "selector_drop_reasons": [], "criteria_drop_directives": []}, [
            "clean_dataset.column_transformations must be an object when present."
        ]

    def _collect(alias_keys: Tuple[str, ...]) -> Tuple[List[str], List[Any]]:
        collected: List[str] = []
        invalid_items: List[Any] = []
        for source in (transform_block, clean_dataset):
            for key in alias_keys:
                if key not in source:
                    continue
                raw = source.get(key)
                clean, invalid = _normalize_nonempty_str_list(raw)
                collected.extend(clean)
                invalid_items.extend(invalid)
        return list(dict.fromkeys(collected)), invalid_items

    drop_cols, invalid_drop = _collect(
        ("drop_columns", "remove_columns", "columns_to_drop", "excluded_columns")
    )
    scale_cols, invalid_scale = _collect(
        ("scale_columns", "normalize_columns", "standardize_columns", "rescale_columns")
    )
    if invalid_drop:
        issues.append("column_transformations.drop_columns must be list[str].")
    if invalid_scale:
        issues.append("column_transformations.scale_columns must be list[str].")

    transform_payload = dict(transform_block)
    if "drop_policy" not in transform_payload and "drop_policy" in clean_dataset:
        transform_payload["drop_policy"] = clean_dataset.get("drop_policy")
    selector_drop_reasons, selector_drop_issues = extract_selector_drop_reasons(transform_payload)
    for issue in selector_drop_issues:
        issues.append(f"column_transformations.drop_policy: {issue}")

    criteria_drop_directives: List[str] = []
    feature_engineering = transform_block.get("feature_engineering")
    if isinstance(feature_engineering, list):
        for idx, item in enumerate(feature_engineering):
            if not isinstance(item, dict):
                continue
            action = str(item.get("action") or "").strip().lower()
            if action not in {"drop", "remove", "exclude"}:
                continue
            if any(item.get(key) for key in ("criteria", "condition", "rule", "rationale")):
                criteria_drop_directives.append(f"feature_engineering[{idx}]")

    return {
        "drop_columns": drop_cols,
        "scale_columns": scale_cols,
        "selector_drop_reasons": selector_drop_reasons,
        "criteria_drop_directives": criteria_drop_directives,
    }, issues


def _normalize_selector_entry(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Auto-normalise LLM-generated selector variants into the canonical format
    expected by _expand_required_feature_selectors.

    Handles four common deviations:
      1. Nested dict:   {"selector": {"type": "regex", "pattern": "..."}}
         → hoist inner dict, preserve metadata (name/family_id/role).
      2. String short:  {"selector": "regex:^pixel\\d+$"}
         → parse into {"type": "regex", "pattern": "^pixel\\d+$"}.
      3. Wrong key:     {"type": "regex", "selector": "^pixel\\d+$"}
         → move "selector" value into "pattern".
      4. prefix short:  {"selector": "prefix:pixel"}
         → {"type": "prefix", "value": "pixel"}.
    """
    out = dict(raw)

    # ── Case 1: nested dict under "selector" key ──────────────────────
    inner = out.get("selector")
    if isinstance(inner, dict):
        # Hoist inner dict fields to top level (inner wins on conflict)
        metadata = {k: v for k, v in out.items() if k != "selector"}
        out = {**metadata, **inner}
        return out

    # ── Case 2 & 4: string shorthand "type:value" ────────────────────
    if isinstance(inner, str) and ":" in inner:
        parts = inner.split(":", 1)
        stype = parts[0].strip().lower()
        svalue = parts[1].strip() if len(parts) > 1 else ""
        metadata = {k: v for k, v in out.items() if k != "selector"}
        if stype in {"regex", "pattern"}:
            out = {**metadata, "type": "regex", "pattern": svalue}
        elif stype == "prefix":
            out = {**metadata, "type": "prefix", "value": svalue}
        elif stype == "suffix":
            out = {**metadata, "type": "suffix", "value": svalue}
        elif stype == "prefix_numeric_range":
            # Try to parse "pixel0-783" style
            import re as _re
            m = _re.match(r"([A-Za-z_]+)(\d+)-(\d+)$", svalue)
            if m:
                out = {**metadata, "type": "prefix_numeric_range",
                       "prefix": m.group(1), "start": int(m.group(2)), "end": int(m.group(3))}
            else:
                out = {**metadata, "type": stype, "value": svalue}
        else:
            out = {**metadata, "type": stype, "value": svalue}
        return out

    # ── Case 3: type present but pattern stored in "selector" ─────────
    if "type" in out and isinstance(inner, str) and inner.strip():
        stype = str(out.get("type") or "").strip().lower()
        if stype in {"regex", "pattern"} and not out.get("pattern"):
            out["pattern"] = inner
            del out["selector"]
        elif stype in {"prefix", "suffix"} and not out.get("value"):
            out["value"] = inner
            del out["selector"]

    stype = str(out.get("type") or "").strip().lower()
    if stype in {"all_columns_except", "all_numeric_except"}:
        raw_excluded = out.get("except_columns")
        if raw_excluded is None:
            raw_excluded = out.get("value")
        if raw_excluded is None:
            raw_excluded = out.get("columns")
        if isinstance(raw_excluded, str):
            raw_excluded = [raw_excluded]
        if isinstance(raw_excluded, list):
            out["except_columns"] = raw_excluded

    return out


def _expand_required_feature_selectors(
    selectors: Any,
    candidate_columns: List[str],
) -> Tuple[List[str], List[str]]:
    """
    Expand optional clean_dataset.required_feature_selectors into concrete columns.
    """
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
        # ── Auto-normalise LLM format variants ──
        selector = _normalize_selector_entry(selector)
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
                excluded, invalid_values = _normalize_nonempty_str_list(selector.get("except_columns"))
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


def _normalize_column_dtype_targets(raw: Any) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    if raw is None:
        return {}, []
    if not isinstance(raw, dict):
        return {}, ["column_dtype_targets must be an object mapping key -> {target_dtype,...}."]
    normalized: Dict[str, Dict[str, Any]] = {}
    issues: List[str] = []
    for key, value in raw.items():
        col_key = str(key or "").strip()
        if not col_key:
            issues.append("column_dtype_targets contains an empty key.")
            continue
        if not isinstance(value, dict):
            issues.append(f"column_dtype_targets['{col_key}'] must be an object.")
            continue
        target_dtype = str(value.get("target_dtype") or "").strip()
        if not target_dtype:
            issues.append(f"column_dtype_targets['{col_key}'] missing target_dtype.")
            continue
        payload: Dict[str, Any] = {"target_dtype": target_dtype}
        if "nullable" in value:
            payload["nullable"] = value.get("nullable")
        if "role" in value and value.get("role") not in (None, ""):
            payload["role"] = value.get("role")
        if "source" in value and value.get("source") not in (None, ""):
            payload["source"] = value.get("source")
        if "matched_count" in value:
            payload["matched_count"] = value.get("matched_count")
        if isinstance(value.get("matched_sample"), list):
            payload["matched_sample"] = [
                str(item) for item in value.get("matched_sample", []) if str(item).strip()
            ][:20]
        normalized[col_key] = payload
    return normalized, issues


def _selector_matches_column(selector: Dict[str, Any], column: str) -> bool:
    """
    Return True when a selector definition matches a concrete column name.

    This allows semantic coverage checks without requiring a fully enumerated
    canonical column inventory inside the contract.
    """
    if not isinstance(selector, dict) or not isinstance(column, str) or not column:
        return False
    selector_type = str(selector.get("type") or "").strip().lower()
    col = column.strip()
    if not selector_type or not col:
        return False
    try:
        if selector_type in {"regex", "pattern"}:
            pattern = str(selector.get("pattern") or "").strip()
            if not pattern:
                return False
            return re.compile(pattern, flags=re.IGNORECASE).match(col) is not None
        if selector_type == "prefix":
            prefix = str(selector.get("value") or selector.get("prefix") or "").strip()
            return bool(prefix) and col.lower().startswith(prefix.lower())
        if selector_type == "suffix":
            suffix = str(selector.get("value") or selector.get("suffix") or "").strip()
            return bool(suffix) and col.lower().endswith(suffix.lower())
        if selector_type == "contains":
            token = str(selector.get("value") or "").strip()
            return bool(token) and token.lower() in col.lower()
        if selector_type == "list":
            values, _ = _normalize_nonempty_str_list(selector.get("columns"))
            return col in set(values)
        if selector_type in {"all_columns_except", "all_numeric_except"}:
            excluded, _ = _normalize_nonempty_str_list(selector.get("except_columns"))
            excluded_norm = {item.lower() for item in excluded}
            return col.lower() not in excluded_norm
        if selector_type == "prefix_numeric_range":
            prefix = str(selector.get("prefix") or "").strip()
            start = selector.get("start")
            end = selector.get("end")
            if not prefix or not isinstance(start, int) or not isinstance(end, int):
                return False
            m = re.compile(rf"^{re.escape(prefix)}(\d+)$", flags=re.IGNORECASE).match(col)
            if not m:
                return False
            idx = int(m.group(1))
            lo = min(start, end)
            hi = max(start, end)
            return lo <= idx <= hi
    except Exception:
        return False
    return False


def _column_matches_any_selector(column: str, selectors: Any) -> bool:
    if not isinstance(selectors, list):
        return False
    for selector in selectors:
        if _selector_matches_column(selector, column):
            return True
    return False


def _collect_ml_required_columns(contract: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Collect columns the ML stage depends on, using contract-declared semantics only.

    Returns:
      - concrete_columns: concrete column identifiers
      - selector_hints: selector-like tokens (e.g., wildcard/regex) that should
        be represented via clean_dataset.required_feature_selectors
    """
    if not isinstance(contract, dict):
        return [], []

    required: List[str] = []
    selector_hints: List[str] = []

    def _extend(values: Any) -> None:
        cols, _ = _normalize_nonempty_str_list(values)
        for col in cols:
            if _looks_like_selector_token(col):
                if col not in selector_hints:
                    selector_hints.append(col)
                continue
            if col not in required:
                required.append(col)

    column_roles = contract.get("column_roles")
    if isinstance(column_roles, dict):
        # NOTE: "decision" columns are model OUTPUTS (e.g. prob_12h, predicted_price)
        # that do not exist in the input clean dataset.  They must NOT be required
        # in clean_dataset coverage — only pre_decision (features) and outcome
        # (labels) are genuine input-side ML dependencies.
        for key in (
            "pre_decision",
            "outcome",
            "features",
            "feature",
            "target",
            "label",
        ):
            _extend(column_roles.get(key))

    allowed_feature_sets = contract.get("allowed_feature_sets")
    if isinstance(allowed_feature_sets, dict):
        _extend(allowed_feature_sets.get("model_features"))
        _extend(allowed_feature_sets.get("segmentation_features"))

        forbidden, _ = _normalize_nonempty_str_list(
            allowed_feature_sets.get("forbidden_for_modeling")
            or allowed_feature_sets.get("forbidden_features")
        )
        if forbidden:
            forbidden_norm = {col.lower() for col in forbidden}
            required = [col for col in required if col.lower() not in forbidden_norm]

    return required, selector_hints


_MULTI_TARGET_MARKERS = (
    "multi_output",
    "multioutput",
    "multi_target",
    "multitarget",
    "multi_label",
    "multilabel",
    "multi_horizon",
    "multihorizon",
)
_TARGET_NAME_MARKERS = ("target", "label", "outcome", "response", "y_")
_TRAILING_TARGET_BUCKET_RE = re.compile(
    r"(?:[_\-\s]?(?:t\d+|\d+(?:h|hr|hrs|hour|hours|d|day|days|w|week|weeks|m|min|mins|month|months)))+$",
    flags=re.IGNORECASE,
)


def _target_family_signature(value: Any) -> str:
    token = str(value or "").strip().lower()
    if not token:
        return ""
    token = _TRAILING_TARGET_BUCKET_RE.sub("", token)
    token = re.sub(r"[_\-\s]+$", "", token)
    token = re.sub(r"[_\-\s]+", "_", token)
    return token


def _looks_like_target_semantic_column(value: Any) -> bool:
    token = str(value or "").strip().lower()
    if not token:
        return False
    normalized = re.sub(r"[^a-z0-9_]+", "_", token).strip("_")
    if any(marker in normalized for marker in _TARGET_NAME_MARKERS):
        return True
    return normalized.startswith(("y_", "label_", "target_", "outcome_"))


def _has_multi_target_signal(*values: Any) -> bool:
    for value in values:
        if isinstance(value, dict):
            if _has_multi_target_signal(*value.values()):
                return True
            continue
        if isinstance(value, list):
            if _has_multi_target_signal(*value):
                return True
            continue
        text = str(value or "").strip().lower()
        if not text:
            continue
        if any(marker in text for marker in _MULTI_TARGET_MARKERS):
            return True
        if "12h" in text and "24h" in text:
            return True
    return False


def validate_contract_minimal_readonly(
    contract: Dict[str, Any],
    column_inventory: List[str] | None = None,
    steward_semantics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Minimal, scope-driven contract validation.

    Designed for LLM freedom: validates only the stable interface needed by the
    orchestrator and downstream agents, without enforcing a rigid monolithic schema.

    Args:
        contract: The execution contract dict to validate.
        column_inventory: Full column header list from the dataset. When provided,
            required_feature_selectors are expanded against this list instead of
            canonical_columns alone (critical for wide-schema datasets where
            canonical_columns only has anchor columns like ["label", "__split"]).
        steward_semantics: Optional steward-derived semantic hints (e.g., primary_target,
            split_candidates, id_candidates) used for semantic sanity checks.
    """
    issues: List[Dict[str, Any]] = []

    if not isinstance(contract, dict):
        issues.append(
            _strict_issue(
                "contract.structure",
                "error",
                "Contract must be a dictionary.",
                type(contract).__name__,
            )
        )
        return {
            "status": "error",
            "accepted": False,
            "issues": issues,
            "summary": {"error_count": 1, "warning_count": 0},
        }

    steward_semantics = steward_semantics if isinstance(steward_semantics, dict) else {}

    def _collect_target_candidates() -> List[str]:
        candidates: List[str] = []

        def _extend(values: Any) -> None:
            cleaned, _ = _normalize_nonempty_str_list(values)
            for col in cleaned:
                if col not in candidates:
                    candidates.append(col)

        def _extend_from_object(obj: Any, keys: Tuple[str, ...]) -> None:
            if not isinstance(obj, dict):
                return
            for key in keys:
                _extend(obj.get(key))

        _extend(steward_semantics.get("primary_target"))
        _extend(steward_semantics.get("primary_targets"))
        _extend(steward_semantics.get("target_column"))
        _extend(steward_semantics.get("target_columns"))
        _extend(steward_semantics.get("label_column"))
        _extend(steward_semantics.get("label_columns"))

        _extend(contract.get("target_column"))
        _extend(contract.get("target_columns"))

        objective_analysis = contract.get("objective_analysis")
        _extend_from_object(
            objective_analysis,
            (
                "primary_target",
                "primary_targets",
                "target_column",
                "target_columns",
                "label_column",
                "label_columns",
            ),
        )

        evaluation_spec = contract.get("evaluation_spec")
        _extend_from_object(
            evaluation_spec,
            (
                "primary_target",
                "primary_targets",
                "target_column",
                "target_columns",
                "label_column",
                "label_columns",
            ),
        )
        if isinstance(evaluation_spec, dict):
            eval_params = evaluation_spec.get("params")
            _extend_from_object(
                eval_params,
                (
                    "primary_target",
                    "primary_targets",
                    "target_column",
                    "target_columns",
                    "label_column",
                    "label_columns",
                ),
            )

        validation_requirements = contract.get("validation_requirements")
        _extend_from_object(
            validation_requirements,
            (
                "primary_target",
                "primary_targets",
                "target_column",
                "target_columns",
                "label_column",
                "label_columns",
            ),
        )
        if isinstance(validation_requirements, dict):
            vr_params = validation_requirements.get("params")
            _extend_from_object(
                vr_params,
                (
                    "primary_target",
                    "primary_targets",
                    "target_column",
                    "target_columns",
                    "label_column",
                    "label_columns",
                ),
            )

        return candidates

    def _collect_outcome_columns() -> List[str]:
        outcomes: List[str] = []

        def _extend(values: Any) -> None:
            cleaned, _ = _normalize_nonempty_str_list(values)
            for col in cleaned:
                if col not in outcomes:
                    outcomes.append(col)

        _extend(contract.get("outcome_columns"))
        column_roles = contract.get("column_roles")
        if isinstance(column_roles, dict):
            _extend(column_roles.get("outcome"))
            _extend(column_roles.get("target"))
            _extend(column_roles.get("label"))
        return outcomes

    def _normalize_target_value_token(value: Any) -> str:
        if value is None:
            return ""
        text = str(value).strip().strip("\"'")
        if not text:
            return ""
        lowered = text.lower()
        if lowered in {"nan", "none", "null", "na", "n/a", "<na>"}:
            return ""
        try:
            number = float(text)
        except Exception:
            return lowered
        if not (number == number):  # NaN guard
            return ""
        rounded = round(number)
        if abs(number - rounded) < 1e-9:
            return str(int(rounded))
        return str(number)

    def _collect_steward_observed_target_values(target_candidates: List[str]) -> Dict[str, List[str]]:
        observed: Dict[str, List[str]] = {}
        if not isinstance(steward_semantics, dict):
            return observed

        raw_map_candidates = (
            steward_semantics.get("target_observed_values"),
            steward_semantics.get("observed_target_values"),
            steward_semantics.get("target_values_by_column"),
        )
        for raw_map in raw_map_candidates:
            if not isinstance(raw_map, dict):
                continue
            for key, values in raw_map.items():
                key_name = str(key or "").strip()
                if not key_name:
                    continue
                normalized_values, _ = _normalize_nonempty_str_list(values)
                if not normalized_values:
                    continue
                target_list = observed.setdefault(key_name, [])
                for raw_value in normalized_values:
                    token = _normalize_target_value_token(raw_value)
                    if token and token not in target_list:
                        target_list.append(token)

        if not observed and len(target_candidates) == 1:
            fallback_values = steward_semantics.get("target_values")
            normalized_values, _ = _normalize_nonempty_str_list(fallback_values)
            if normalized_values:
                only_target = target_candidates[0]
                observed_values: List[str] = []
                for raw_value in normalized_values:
                    token = _normalize_target_value_token(raw_value)
                    if token and token not in observed_values:
                        observed_values.append(token)
                if observed_values:
                    observed[only_target] = observed_values
        return observed

    def _collect_target_mapping_gates() -> List[Dict[str, Any]]:
        raw_gates = contract.get("cleaning_gates")
        if not isinstance(raw_gates, list):
            return []
        mapping_gates: List[Dict[str, Any]] = []
        for gate in raw_gates:
            if not isinstance(gate, dict):
                continue
            gate_name = (
                gate.get("name")
                or gate.get("id")
                or gate.get("gate")
                or gate.get("rule")
                or gate.get("check")
            )
            gate_norm = re.sub(r"[^a-z0-9]+", "_", str(gate_name or "").strip().lower()).strip("_")
            if not gate_norm:
                continue
            if gate_norm in {"target_mapping_check", "target_mapping", "target_label_mapping", "target_map_check"}:
                mapping_gates.append(gate)
                continue
            if "target" in gate_norm and "mapping" in gate_norm:
                mapping_gates.append(gate)
        return mapping_gates

    def _extract_mapping_keys(gate: Dict[str, Any]) -> List[str]:
        if not isinstance(gate, dict):
            return []
        params = gate.get("params")
        mapping = params.get("mapping") if isinstance(params, dict) else None
        if mapping is None:
            mapping = gate.get("mapping")
        if not isinstance(mapping, dict):
            return []
        keys: List[str] = []
        for raw_key in mapping.keys():
            token = _normalize_target_value_token(raw_key)
            if token and token not in keys:
                keys.append(token)
        return keys

    def _collect_structural_columns() -> List[str]:
        structural: List[str] = []

        def _extend(values: Any) -> None:
            cleaned, _ = _normalize_nonempty_str_list(values)
            for col in cleaned:
                if col not in structural:
                    structural.append(col)

        _extend(steward_semantics.get("split_candidates"))
        _extend(steward_semantics.get("id_candidates"))
        _extend(steward_semantics.get("identifier_columns"))

        column_roles = contract.get("column_roles")
        if isinstance(column_roles, dict):
            _extend(column_roles.get("identifiers"))
            _extend(column_roles.get("id"))
            _extend(column_roles.get("time_columns"))

        evaluation_spec = contract.get("evaluation_spec")
        if isinstance(evaluation_spec, dict):
            _extend(evaluation_spec.get("split_column"))
            _extend(evaluation_spec.get("split_columns"))
            _extend(evaluation_spec.get("fold_column"))

        return structural

    scope_raw = contract.get("scope")
    scope = normalize_contract_scope(scope_raw)
    if scope_raw is None:
        issues.append(
            _strict_issue(
                "contract.scope_missing",
                "error",
                "scope missing; scope is required and must be one of cleaning_only/ml_only/full_pipeline.",
                "scope",
            )
        )
    elif str(scope_raw).strip().lower() not in CONTRACT_SCOPE_VALUES:
        token = str(scope_raw).strip().lower()
        if token in SCOPE_ALIAS_VALUES:
            issues.append(
                _strict_issue(
                    "contract.scope_alias_normalized",
                    "warning",
                    f"Scope alias '{scope_raw}' normalized to '{scope}'.",
                    scope_raw,
                )
            )
        else:
            issues.append(
                _strict_issue(
                    "contract.scope_unknown",
                    "error",
                    f"Unknown scope '{scope_raw}'. Normalized to '{scope}'.",
                    scope_raw,
                )
            )

    strategy_title = contract.get("strategy_title")
    if not isinstance(strategy_title, str) or not strategy_title.strip():
        issues.append(
            _strict_issue(
                "contract.strategy_title",
                "error",
                "strategy_title must be a non-empty string.",
                strategy_title,
            )
        )

    business_objective = contract.get("business_objective")
    if not isinstance(business_objective, str) or not business_objective.strip():
        issues.append(
            _strict_issue(
                "contract.business_objective",
                "error",
                "business_objective must be a non-empty string.",
                business_objective,
            )
        )

    output_dialect = contract.get("output_dialect")
    if not isinstance(output_dialect, dict) or not output_dialect:
        issues.append(
            _strict_issue(
                "contract.output_dialect",
                "warning",
                "output_dialect missing or empty; execution may fall back to runtime dialect detection.",
                output_dialect,
            )
        )

    canonical_columns = contract.get("canonical_columns")
    if not isinstance(canonical_columns, list) or not any(
        isinstance(col, str) and col.strip() for col in (canonical_columns or [])
    ):
        issues.append(
            _strict_issue(
                "contract.canonical_columns",
                "error",
                "canonical_columns must contain at least one non-empty column name.",
                canonical_columns,
            )
        )
    else:
        inventory_cols, _ = _normalize_nonempty_str_list(column_inventory if isinstance(column_inventory, list) else [])
        canonical_cols, _ = _normalize_nonempty_str_list(canonical_columns)
        if inventory_cols and canonical_cols:
            inventory_norm = {col.lower(): col for col in inventory_cols}
            effective_covered = {col.lower() for col in canonical_cols}

            selector_sources: List[Any] = []
            artifact_requirements = contract.get("artifact_requirements")
            if isinstance(artifact_requirements, dict):
                clean_dataset = artifact_requirements.get("clean_dataset")
                if isinstance(clean_dataset, dict):
                    selector_sources.append(clean_dataset.get("required_feature_selectors"))
            selector_sources.append(contract.get("required_feature_selectors"))
            selector_sources.append(contract.get("feature_selectors"))

            for selector_source in selector_sources:
                selector_cols, _ = _expand_required_feature_selectors(selector_source, inventory_cols)
                for col in selector_cols:
                    effective_covered.add(col.lower())

            inventory_set = set(inventory_norm)
            coverage = len(effective_covered & inventory_set) / max(1, len(inventory_set))
            coverage_threshold = 0.80
            if coverage < coverage_threshold:
                missing = [inventory_norm[key] for key in sorted(inventory_set - effective_covered)]
                issues.append(
                    _strict_issue(
                        "contract.canonical_columns_coverage",
                        "error",
                        (
                            "canonical_columns + declared selectors cover only "
                            f"{coverage:.0%} of column_inventory (minimum {coverage_threshold:.0%})."
                        ),
                        {"missing_columns_sample": missing[:25], "missing_count": len(missing)},
                    )
                )

    required_outputs = contract.get("required_outputs")
    if not isinstance(required_outputs, list) or not required_outputs:
        issues.append(
            _strict_issue(
                "contract.required_outputs",
                "error",
                "required_outputs must be a non-empty list of artifact outputs (string paths or objects with 'path').",
                required_outputs,
            )
        )
    else:
        invalid_outputs: List[Dict[str, Any]] = []
        seen_paths: set[str] = set()
        duplicate_paths: List[str] = []
        for output in required_outputs:
            path, path_error = _extract_required_output_path(output)
            if path_error == "invalid_type":
                invalid_outputs.append({"entry": output, "reason": "invalid_type"})
                continue
            if path_error == "missing_path":
                invalid_outputs.append({"entry": output, "reason": "missing_path"})
                continue
            if path_error == "path_not_string":
                invalid_outputs.append({"entry": output, "reason": "path_not_string"})
                continue
            if path_error == "empty_path":
                invalid_outputs.append({"entry": output, "reason": "empty_path"})
                continue
            if not is_file_path(path):
                invalid_outputs.append({"entry": output, "reason": "not_file_like"})
                continue
            normalized_path = _normalize_required_output_path(path)
            if normalized_path in seen_paths:
                duplicate_paths.append(path)
            else:
                seen_paths.add(normalized_path)
        if invalid_outputs:
            issues.append(
                _strict_issue(
                    "contract.required_outputs_path",
                    "error",
                    "required_outputs entries must be file-like paths and objects must include a non-empty string 'path'.",
                    invalid_outputs[:10],
                )
            )
        if duplicate_paths:
            issues.append(
                _strict_issue(
                    "contract.required_outputs_duplicates",
                    "warning",
                    "required_outputs contains duplicate artifact paths.",
                    sorted(list(dict.fromkeys(duplicate_paths))),
                )
            )

    requires_cleaning = scope in {"cleaning_only", "full_pipeline"}
    requires_ml = scope in {"ml_only", "full_pipeline"}
    artifact_requirements = contract.get("artifact_requirements")
    clean_dataset_cfg = {}
    if isinstance(artifact_requirements, dict):
        candidate = artifact_requirements.get("clean_dataset")
        if isinstance(candidate, dict):
            clean_dataset_cfg = candidate
    raw_dtype_targets = contract.get("column_dtype_targets")
    if raw_dtype_targets in (None, {}) and isinstance(clean_dataset_cfg.get("column_dtype_targets"), dict):
        raw_dtype_targets = clean_dataset_cfg.get("column_dtype_targets")
    column_dtype_targets, dtype_issues = _normalize_column_dtype_targets(raw_dtype_targets)
    for msg in dtype_issues:
        issues.append(
            _strict_issue(
                "contract.column_dtype_targets",
                "error",
                msg,
                raw_dtype_targets,
            )
        )
    if (requires_cleaning or requires_ml) and not column_dtype_targets:
        issues.append(
            _strict_issue(
                "contract.column_dtype_targets_missing",
                "error",
                "column_dtype_targets missing/empty for active scope; fail-closed to reduce dtype drift between DE and ML.",
                raw_dtype_targets,
            )
        )

    if requires_cleaning:
        if not _gate_list_valid(contract.get("cleaning_gates")):
            issues.append(
                _strict_issue(
                    "contract.cleaning_gates",
                    "error",
                    "cleaning_gates missing/empty for cleaning scope; fail-closed to protect reviewer alignment.",
                    contract.get("cleaning_gates"),
                )
            )
        if not _runbook_present(contract.get("data_engineer_runbook")):
            issues.append(
                _strict_issue(
                    "contract.data_engineer_runbook",
                    "error",
                    "data_engineer_runbook missing for cleaning scope; fail-closed (DE execution contract incomplete).",
                    contract.get("data_engineer_runbook"),
                )
            )

        clean_dataset = {}
        if isinstance(artifact_requirements, dict):
            candidate = artifact_requirements.get("clean_dataset")
            if isinstance(candidate, dict):
                clean_dataset = candidate
        required_cols, invalid_required_cols = _normalize_nonempty_str_list(
            clean_dataset.get("required_columns")
        )
        passthrough_cols, _ = _normalize_nonempty_str_list(
            clean_dataset.get("optional_passthrough_columns")
        )
        canonical_cols, _ = _normalize_nonempty_str_list(canonical_columns if isinstance(canonical_columns, list) else [])
        # Wide-schema fix: canonical_columns may only contain anchor columns
        # (e.g. ["label","__split"]) while 784 pixel columns are represented by
        # selectors.  Use the full column_inventory when available so that
        # prefix_numeric_range / regex selectors actually resolve.
        _selector_candidates = (
            [str(c) for c in column_inventory if isinstance(c, str) and c.strip()]
            if column_inventory
            else canonical_cols
        )
        required_feature_selectors = clean_dataset.get("required_feature_selectors")
        selector_cols, selector_issues = _expand_required_feature_selectors(
            required_feature_selectors,
            _selector_candidates,
        )
        if selector_issues:
            issues.append(
                _strict_issue(
                    "contract.clean_dataset_required_feature_selectors",
                    "error",
                    "Some required_feature_selectors entries are invalid/unsupported; fail-closed.",
                    selector_issues[:10],
                )
            )
        transforms, transform_shape_issues = _resolve_cleaning_column_transformations(clean_dataset)
        for msg in transform_shape_issues:
            issues.append(
                _strict_issue(
                    "contract.clean_dataset_column_transformations_shape",
                    "error",
                    msg,
                    clean_dataset.get("column_transformations"),
                )
            )
        drop_cols = transforms.get("drop_columns") or []
        scale_cols = transforms.get("scale_columns") or []
        selector_drop_reasons = transforms.get("selector_drop_reasons") or []
        criteria_drop_directives = transforms.get("criteria_drop_directives") or []
        has_declared_selectors = bool(
            isinstance(required_feature_selectors, list)
            and any(isinstance(item, dict) for item in required_feature_selectors)
        )
        if invalid_required_cols:
            issues.append(
                _strict_issue(
                    "contract.clean_dataset_required_columns_type",
                    "error",
                    "artifact_requirements.clean_dataset.required_columns must contain only strings.",
                    invalid_required_cols[:10],
                )
            )

        runbook_text = _flatten_runbook_text(contract.get("data_engineer_runbook"))
        runbook_has_drop_columns = _has_non_negated_action(runbook_text, _DROP_COLUMN_DIRECTIVE_PATTERN)
        runbook_has_scale_columns = _has_non_negated_feature_scaling_action(runbook_text)
        if runbook_has_drop_columns and not (drop_cols or selector_drop_reasons):
            issues.append(
                _strict_issue(
                    "contract.cleaning_transforms_drop_missing",
                    "error",
                    "Runbook describes dropping columns but no structured drop declaration was found. "
                    "Declare clean_dataset.column_transformations.drop_columns and/or "
                    "clean_dataset.column_transformations.drop_policy.",
                    "artifact_requirements.clean_dataset.column_transformations",
                )
            )
        if runbook_has_scale_columns and not scale_cols:
            issues.append(
                _strict_issue(
                    "contract.cleaning_transforms_scale_missing",
                    "error",
                    "Runbook describes column scaling/normalization but clean_dataset.column_transformations.scale_columns is missing.",
                    "artifact_requirements.clean_dataset.column_transformations.scale_columns",
                )
            )
        if has_declared_selectors and criteria_drop_directives and not (drop_cols or selector_drop_reasons):
            issues.append(
                _strict_issue(
                    "contract.clean_dataset_selector_drop_policy_missing",
                    "error",
                    "clean_dataset declares required_feature_selectors plus criteria-based drop directives, "
                    "but no structured selector-drop policy exists. Add column_transformations.drop_policy "
                    "(allow_selector_drops_when) or explicit drop_columns.",
                    {"criteria_drop_directives": criteria_drop_directives[:10]},
                )
            )
        if selector_drop_reasons and not has_declared_selectors:
            issues.append(
                _strict_issue(
                    "contract.clean_dataset_drop_policy_without_selectors",
                    "warning",
                    "column_transformations.drop_policy is present but required_feature_selectors is empty; "
                    "policy will have no effect.",
                    selector_drop_reasons,
                )
            )

        required_norm = {col.lower(): col for col in required_cols}
        passthrough_norm = {col.lower(): col for col in passthrough_cols}
        selector_norm = {col.lower(): col for col in selector_cols}
        drop_norm = {col.lower(): col for col in drop_cols}
        scale_norm = {col.lower(): col for col in scale_cols}

        drop_in_required = [required_norm[key] for key in sorted(set(required_norm) & set(drop_norm))]
        if drop_in_required:
            issues.append(
                _strict_issue(
                    "contract.cleaning_transforms_drop_conflict",
                    "error",
                    "drop_columns cannot include columns declared in clean_dataset.required_columns.",
                    drop_in_required[:20],
                )
            )

        selector_covered_scale = {
            key
            for key in scale_norm
            if _column_matches_any_selector(scale_norm[key], required_feature_selectors)
        }
        selector_semantic_scale = {
            key
            for key in scale_norm
            if selector_reference_matches_any(scale_norm[key], required_feature_selectors)
        }
        scale_allowed_norm = (
            set(required_norm)
            | set(passthrough_norm)
            | set(selector_norm)
            | set(selector_covered_scale)
            | set(selector_semantic_scale)
        )
        scale_outside_required = [scale_norm[key] for key in sorted(set(scale_norm) - scale_allowed_norm)]
        if scale_outside_required:
            issues.append(
                _strict_issue(
                    "contract.cleaning_transforms_scale_conflict",
                    "error",
                    "scale_columns must be covered by clean_dataset.required_columns/optional_passthrough_columns "
                    "or required_feature_selectors (including selector references like regex:/prefix:/selector:<name>).",
                    scale_outside_required[:20],
                )
            )

        # Fail-closed semantic guard:
        # If selector-based drop policy is active, required/passthrough anchors cannot
        # live inside selector coverage, otherwise they become potentially droppable.
        if selector_drop_reasons and has_declared_selectors:
            selector_required_overlap = [
                required_norm[key] for key in sorted(set(required_norm) & set(selector_norm))
            ]
            if selector_required_overlap:
                issues.append(
                    _strict_issue(
                        "contract.clean_dataset_selector_drop_required_conflict",
                        "error",
                        "required_columns overlap required_feature_selectors while selector-drop policy is active; "
                        "required_columns must be non-droppable anchors.",
                        selector_required_overlap[:25],
                    )
                )
            selector_passthrough_overlap = [
                passthrough_norm[key] for key in sorted(set(passthrough_norm) & set(selector_norm))
            ]
            if selector_passthrough_overlap:
                issues.append(
                    _strict_issue(
                        "contract.clean_dataset_selector_drop_passthrough_conflict",
                        "error",
                        "optional_passthrough_columns overlap required_feature_selectors while selector-drop policy is active; "
                        "passthrough anchors must be non-droppable.",
                        selector_passthrough_overlap[:25],
                    )
                )

        if column_dtype_targets:
            anchor_refs = list(dict.fromkeys(required_cols + passthrough_cols))
            role_bucket_names = ("outcome", "decision", "identifiers", "time_columns")
            column_roles = contract.get("column_roles")
            if isinstance(column_roles, dict):
                for bucket in role_bucket_names:
                    values = column_roles.get(bucket)
                    if isinstance(values, list):
                        for col in values:
                            col_name = str(col or "").strip()
                            if col_name and col_name not in anchor_refs:
                                anchor_refs.append(col_name)
            dtype_keys_norm = {str(k).strip().lower() for k in column_dtype_targets.keys() if str(k).strip()}
            selector_present = any(str(k).strip().lower().startswith("selector:") for k in column_dtype_targets.keys())
            uncovered_dtype_anchors = [
                col
                for col in anchor_refs
                if col and str(col).strip().lower() not in dtype_keys_norm
            ]
            if uncovered_dtype_anchors and not selector_present:
                issues.append(
                    _strict_issue(
                        "contract.column_dtype_targets_anchor_missing",
                        "warning",
                        "column_dtype_targets should cover anchor columns (required/passthrough/role anchors) or provide selector-level dtype targets.",
                        uncovered_dtype_anchors[:25],
                    )
                )

        hard_gate_cols = _collect_gate_column_refs(contract.get("cleaning_gates"), hard_only=True)
        hard_gate_norm = {col.lower(): col for col in hard_gate_cols}
        # Wide-schema: HARD gates may reference selector family tokens
        # (e.g. "PIXEL_FEATURES") instead of individual columns.  Treat
        # declared selector names / family_ids as "covered" so they don't
        # trip the contradiction rule.
        _selector_family_ids: set[str] = set()
        if has_declared_selectors and isinstance(required_feature_selectors, list):
            for _sel in required_feature_selectors:
                if isinstance(_sel, dict):
                    for _fkey in ("name", "family_id", "family"):
                        _fval = _sel.get(_fkey)
                        if isinstance(_fval, str) and _fval.strip():
                            _selector_family_ids.add(_fval.strip().lower())
        gate_missing = [
            hard_gate_norm[key]
            for key in sorted(
                set(hard_gate_norm)
                - (set(required_norm) | set(passthrough_norm)
                   | set(selector_norm) | _selector_family_ids)
            )
        ]
        if gate_missing:
            issues.append(
                _strict_issue(
                    "contract.cleaning_gate_required_columns_contradiction",
                    "error",
                    "HARD cleaning gates reference columns outside required_columns/optional_passthrough_columns.",
                    gate_missing[:20],
                )
            )
        gate_drop_conflict = [hard_gate_norm[key] for key in sorted(set(hard_gate_norm) & set(drop_norm))]
        if gate_drop_conflict:
            issues.append(
                _strict_issue(
                    "contract.cleaning_gate_drop_conflict",
                    "error",
                    "HARD cleaning gates cannot target columns also declared in drop_columns.",
                    gate_drop_conflict[:20],
                )
            )
        if selector_drop_reasons and has_declared_selectors:
            gate_selector_conflict = [
                hard_gate_norm[key] for key in sorted(set(hard_gate_norm) & set(selector_norm))
            ]
            if gate_selector_conflict:
                issues.append(
                    _strict_issue(
                        "contract.cleaning_gate_selector_drop_conflict",
                        "error",
                        "HARD cleaning gates cannot depend on selector-covered columns when selector-drop policy is active.",
                        gate_selector_conflict[:25],
                    )
                )

        if requires_ml:
            ml_required_cols, ml_selector_hints = _collect_ml_required_columns(contract)
            if ml_selector_hints:
                if not has_declared_selectors:
                    issues.append(
                        _strict_issue(
                            "contract.clean_dataset_selector_hints_unresolved",
                            "error",
                            "Selector-like ML feature hints are present but clean_dataset.required_feature_selectors "
                            "is empty. Declare selectors explicitly to improve cleaning/ML alignment (fail-closed).",
                            ml_selector_hints[:25],
                        )
                    )
            ml_required_norm = {col.lower(): col for col in ml_required_cols}
            selector_covered_ml = {
                key
                for key, col in ml_required_norm.items()
                if _column_matches_any_selector(col, required_feature_selectors)
            }
            coverage_norm = set(required_norm) | set(passthrough_norm) | set(selector_norm) | set(selector_covered_ml)
            missing_ml_cols = [
                ml_required_norm[key]
                for key in sorted(set(ml_required_norm) - coverage_norm)
            ]
            if missing_ml_cols:
                issues.append(
                    _strict_issue(
                        "contract.clean_dataset_ml_columns_missing",
                        "error",
                        "full_pipeline contract leaves ML-required columns outside clean_dataset coverage "
                        "(required_columns/optional_passthrough_columns/required_feature_selectors).",
                        missing_ml_cols[:25],
                    )
                )
            drop_vs_ml = [
                ml_required_norm[key]
                for key in sorted(set(ml_required_norm) & set(drop_norm))
            ]
            if drop_vs_ml:
                issues.append(
                    _strict_issue(
                        "contract.clean_dataset_drop_ml_columns_conflict",
                        "error",
                        "clean_dataset.column_transformations.drop_columns removes ML-required columns.",
                        drop_vs_ml[:25],
                    )
                )

    if requires_cleaning:
        target_candidates = _collect_target_candidates()
        mapping_gates = _collect_target_mapping_gates()
        observed_targets = _collect_steward_observed_target_values(target_candidates)
        if mapping_gates and observed_targets:
            conflicting_gates: List[Dict[str, Any]] = []
            for target_col, observed_values in observed_targets.items():
                if not observed_values:
                    continue
                if not all(value in {"0", "1"} for value in observed_values):
                    continue
                for gate in mapping_gates:
                    mapping_keys = _extract_mapping_keys(gate)
                    if not mapping_keys:
                        continue
                    if any(value not in {"0", "1"} for value in mapping_keys):
                        gate_name = (
                            gate.get("name")
                            or gate.get("id")
                            or gate.get("gate")
                            or gate.get("rule")
                            or gate.get("check")
                            or "target_mapping_check"
                        )
                        conflicting_gates.append(
                            {
                                "gate": str(gate_name),
                                "target_column": target_col,
                                "mapping_labels": mapping_keys,
                                "observed_values": observed_values,
                            }
                        )
            if conflicting_gates:
                issues.append(
                    _strict_issue(
                        "contract.target_mapping_consistency",
                        "error",
                        "target_mapping_check conflicts with observed numeric-binary target values; remove label remapping or align with observed domain.",
                        conflicting_gates[:10],
                    )
                )

    if requires_ml:
        target_candidates = _collect_target_candidates()
        outcome_columns = _collect_outcome_columns()
        if target_candidates and outcome_columns:
            target_norm = {col.lower() for col in target_candidates if col}
            if _has_multi_target_signal(
                contract.get("evaluation_spec"),
                contract.get("objective_analysis"),
                contract.get("business_objective"),
                outcome_columns,
                target_candidates,
                steward_semantics.get("notes") if isinstance(steward_semantics, dict) else None,
            ):
                target_families = {
                    _target_family_signature(col)
                    for col in target_candidates
                    if _target_family_signature(col)
                }
                for col in outcome_columns:
                    if not _looks_like_target_semantic_column(col):
                        continue
                    family = _target_family_signature(col)
                    if family and (not target_families or family in target_families):
                        target_norm.add(col.lower())
            unexpected_outcomes = [
                col for col in outcome_columns
                if col and col.lower() not in target_norm
            ]
            if unexpected_outcomes:
                steward_target_candidates: List[str] = []
                for key in ("primary_target", "primary_targets", "target_column", "target_columns"):
                    steward_targets, _ = _normalize_nonempty_str_list(steward_semantics.get(key))
                    for col in steward_targets:
                        if col not in steward_target_candidates:
                            steward_target_candidates.append(col)
                severity = "error" if steward_target_candidates or len(unexpected_outcomes) >= 3 else "warning"
                issues.append(
                    _strict_issue(
                        "contract.outcome_columns_sanity",
                        severity,
                        "outcome columns include non-target fields; outcomes must contain only target column(s).",
                        {
                            "unexpected_outcomes": unexpected_outcomes[:25],
                            "target_candidates": target_candidates[:25],
                        },
                    )
                )

        allowed_feature_sets = contract.get("allowed_feature_sets")
        model_features: List[str] = []
        if isinstance(allowed_feature_sets, dict):
            model_features, _ = _normalize_nonempty_str_list(allowed_feature_sets.get("model_features"))
        if not model_features:
            column_roles = contract.get("column_roles")
            if isinstance(column_roles, dict):
                model_features, _ = _normalize_nonempty_str_list(column_roles.get("pre_decision"))

        if model_features:
            structural_norm = {
                col.lower()
                for col in _collect_structural_columns()
                if isinstance(col, str) and col.strip()
            }
            useful_features = [
                feat
                for feat in model_features
                if _looks_like_selector_token(feat) or feat.lower() not in structural_norm
            ]
            if not useful_features:
                issues.append(
                    _strict_issue(
                        "contract.model_features_empty",
                        "error",
                        "model_features contains only structural columns (split/id/time). No useful modeling features remain.",
                        {"model_features": model_features[:25], "structural_columns": sorted(structural_norm)[:25]},
                    )
                )
        else:
            issues.append(
                _strict_issue(
                    "contract.model_features_empty",
                    "error",
                    "model_features is missing/empty; declare at least one useful modeling feature.",
                    {"model_features": model_features},
                )
            )

        evaluation_spec = contract.get("evaluation_spec")
        if not isinstance(evaluation_spec, dict) or not evaluation_spec:
            issues.append(
                _strict_issue(
                    "contract.evaluation_spec",
                    "error",
                    "evaluation_spec missing/empty for ML scope; fail-closed (ML objective context incomplete).",
                    evaluation_spec,
                )
            )
        if not _gate_list_valid(contract.get("qa_gates")):
            issues.append(
                _strict_issue(
                    "contract.qa_gates",
                    "error",
                    "qa_gates missing/empty for ML scope; fail-closed to protect QA review quality.",
                    contract.get("qa_gates"),
                )
            )
        if not _gate_list_valid(contract.get("reviewer_gates")):
            issues.append(
                _strict_issue(
                    "contract.reviewer_gates",
                    "error",
                    "reviewer_gates missing/empty for ML scope; fail-closed to protect review-board alignment.",
                    contract.get("reviewer_gates"),
                )
            )
        if not isinstance(contract.get("validation_requirements"), dict) or not contract.get("validation_requirements"):
            issues.append(
                _strict_issue(
                    "contract.validation_requirements",
                    "error",
                    "validation_requirements missing/empty for ML scope; fail-closed (benchmark traceability required).",
                    contract.get("validation_requirements"),
                )
            )
        if not _runbook_present(contract.get("ml_engineer_runbook")):
            issues.append(
                _strict_issue(
                    "contract.ml_engineer_runbook",
                    "error",
                    "ml_engineer_runbook missing for ML scope; fail-closed (ML execution contract incomplete).",
                    contract.get("ml_engineer_runbook"),
                )
            )

    # Contract-to-view executability checks (fail-closed): if the contract is accepted,
    # projected runtime views must be executable without additional deterministic repairs.
    try:
        from src.utils.contract_views import build_contract_views_projection

        projected_views = build_contract_views_projection(contract, artifact_index=[])

        if requires_cleaning:
            de_view = projected_views.get("de_view") if isinstance(projected_views, dict) else None
            if not isinstance(de_view, dict) or not de_view:
                issues.append(
                    _strict_issue(
                        "contract.de_view_missing",
                        "error",
                        "de_view could not be projected from contract.",
                        "de_view",
                    )
                )
            else:
                de_output_path = str(de_view.get("output_path") or "").strip()
                de_manifest_path = str(
                    de_view.get("output_manifest_path") or de_view.get("manifest_path") or ""
                ).strip()
                if not de_output_path or not is_file_path(de_output_path):
                    issues.append(
                        _strict_issue(
                            "contract.de_view_output_path",
                            "error",
                            "de_view.output_path must be a valid artifact file path.",
                            de_output_path or None,
                        )
                    )
                if not de_manifest_path or not is_file_path(de_manifest_path):
                    issues.append(
                        _strict_issue(
                            "contract.de_view_manifest_path",
                            "error",
                            "de_view.output_manifest_path must be a valid artifact file path.",
                            de_manifest_path or None,
                        )
                    )
                required_cols = de_view.get("required_columns")
                if not isinstance(required_cols, list) or not any(
                    isinstance(col, str) and col.strip() for col in required_cols
                ):
                    issues.append(
                        _strict_issue(
                            "contract.de_view_required_columns",
                            "error",
                            "de_view.required_columns must be a non-empty list.",
                            required_cols,
                        )
                    )
                de_cleaning_gates = de_view.get("cleaning_gates")
                if not _gate_list_valid(de_cleaning_gates):
                    issues.append(
                        _strict_issue(
                            "contract.de_view_cleaning_gates",
                            "error",
                            "de_view.cleaning_gates must be a non-empty list of consumable gate objects.",
                            de_cleaning_gates,
                        )
                    )
                de_runbook = de_view.get("data_engineer_runbook")
                if not _runbook_present(de_runbook):
                    issues.append(
                        _strict_issue(
                            "contract.de_view_data_engineer_runbook",
                            "error",
                            "de_view.data_engineer_runbook must be present and non-empty.",
                            de_runbook,
                        )
                    )

        if requires_ml:
            ml_view = projected_views.get("ml_view") if isinstance(projected_views, dict) else None
            reviewer_view = projected_views.get("reviewer_view") if isinstance(projected_views, dict) else None
            qa_view = projected_views.get("qa_view") if isinstance(projected_views, dict) else None

            if not isinstance(ml_view, dict) or not ml_view:
                issues.append(
                    _strict_issue(
                        "contract.ml_view_missing",
                        "error",
                        "ml_view could not be projected from contract.",
                        "ml_view",
                    )
                )
            else:
                objective_type = str(ml_view.get("objective_type") or "").strip().lower()
                if not objective_type or objective_type == "unknown":
                    issues.append(
                        _strict_issue(
                            "contract.ml_view_objective_type",
                            "error",
                            "ml_view.objective_type is unknown; set objective_analysis.problem_type or evaluation_spec.objective_type.",
                            objective_type or None,
                        )
                    )
                column_roles = ml_view.get("column_roles")
                if not isinstance(column_roles, dict) or not column_roles:
                    issues.append(
                        _strict_issue(
                            "contract.ml_view_column_roles",
                            "error",
                            "ml_view.column_roles must be a non-empty role mapping.",
                            column_roles,
                        )
                    )

            if not isinstance(reviewer_view, dict) or not reviewer_view:
                issues.append(
                    _strict_issue(
                        "contract.reviewer_view_missing",
                        "error",
                        "reviewer_view could not be projected from contract.",
                        "reviewer_view",
                    )
                )
            else:
                reviewer_gates = reviewer_view.get("reviewer_gates")
                if not isinstance(reviewer_gates, list) or not reviewer_gates:
                    issues.append(
                        _strict_issue(
                            "contract.reviewer_view_gates",
                            "error",
                            "reviewer_view.reviewer_gates is empty after projection (fail-closed).",
                            reviewer_gates,
                        )
                    )

            if not isinstance(qa_view, dict) or not qa_view:
                issues.append(
                    _strict_issue(
                        "contract.qa_view_missing",
                        "error",
                        "qa_view could not be projected from contract.",
                        "qa_view",
                    )
                )
            else:
                qa_gates = qa_view.get("qa_gates")
                if not isinstance(qa_gates, list) or not qa_gates:
                    issues.append(
                        _strict_issue(
                            "contract.qa_view_gates",
                            "error",
                            "qa_view.qa_gates is empty after projection (fail-closed).",
                            qa_gates,
                        )
                    )
    except Exception as projection_err:
        issues.append(
            _strict_issue(
                "contract.view_projection_exception",
                "error",
                "Failed to validate projected contract views.",
                str(projection_err),
            )
        )

    iteration_policy = contract.get("iteration_policy")
    if not isinstance(iteration_policy, dict) or not iteration_policy:
        issues.append(
            _strict_issue(
                "contract.iteration_policy",
                "warning",
                "iteration_policy missing/empty; runtime will use default iteration behavior.",
                iteration_policy,
            )
        )
    else:
        limit_keys = list(ITERATION_POLICY_LIMIT_KEYS) + list(ITERATION_POLICY_LIMIT_ALIASES)
        policy_has_limit = False
        canonical_present = any(key in iteration_policy for key in ITERATION_POLICY_LIMIT_KEYS)
        alias_present = any(key in iteration_policy for key in ITERATION_POLICY_LIMIT_ALIASES)
        for key in limit_keys:
            if key not in iteration_policy:
                continue
            value = iteration_policy.get(key)
            try:
                if float(value) >= 1:
                    policy_has_limit = True
                    break
            except Exception:
                continue
        if not policy_has_limit:
            issues.append(
                _strict_issue(
                    "contract.iteration_policy_limits",
                    "warning",
                    "iteration_policy has no numeric iteration limit >= 1; runtime will use defaults.",
                    iteration_policy,
                )
            )
        elif alias_present and not canonical_present:
            issues.append(
                _strict_issue(
                    "contract.iteration_policy_alias",
                    "warning",
                    "iteration_policy uses non-canonical limit keys; accepted with runtime alias mapping.",
                    [key for key in ITERATION_POLICY_LIMIT_ALIASES if key in iteration_policy],
                )
            )

    optimization_policy = contract.get("optimization_policy")
    if optimization_policy is None:
        issues.append(
            _strict_issue(
                "contract.optimization_policy_missing",
                "warning",
                "optimization_policy missing; runtime will use safe optimization defaults.",
                get_default_optimization_policy(),
            )
        )
    elif not isinstance(optimization_policy, dict):
        issues.append(
            _strict_issue(
                "contract.optimization_policy_type",
                "error",
                "optimization_policy must be an object when present.",
                type(optimization_policy).__name__,
            )
        )
    else:
        missing_opt_keys = [key for key in OPTIMIZATION_POLICY_DEFAULTS.keys() if key not in optimization_policy]
        if missing_opt_keys:
            issues.append(
                _strict_issue(
                    "contract.optimization_policy_missing_keys",
                    "warning",
                    "optimization_policy missing keys; runtime will fill defaults.",
                    missing_opt_keys,
                )
            )
        for key in OPTIMIZATION_POLICY_INT_KEYS:
            if key not in optimization_policy:
                continue
            min_value = 0 if key == "patience" else 1
            try:
                numeric = int(float(optimization_policy.get(key)))
                if numeric < min_value:
                    raise ValueError("below minimum")
            except Exception:
                issues.append(
                    _strict_issue(
                        "contract.optimization_policy_value",
                        "error",
                        f"optimization_policy.{key} must be an integer >= {min_value}.",
                        optimization_policy.get(key),
                    )
                )
        if "min_delta" in optimization_policy:
            try:
                min_delta = float(optimization_policy.get("min_delta"))
                if min_delta < 0:
                    raise ValueError("below minimum")
            except Exception:
                issues.append(
                    _strict_issue(
                        "contract.optimization_policy_value",
                        "error",
                        "optimization_policy.min_delta must be a number >= 0.",
                        optimization_policy.get("min_delta"),
                    )
                )

    status = _status_from_issues(issues)
    error_count = sum(1 for issue in issues if str(issue.get("severity", "")).lower() in {"error", "fail"})
    warning_count = sum(1 for issue in issues if str(issue.get("severity", "")).lower() == "warning")
    return {
        "status": status,
        "accepted": status != "error",
        "issues": issues,
        "summary": {
            "error_count": error_count,
            "warning_count": warning_count,
            "scope": scope,
        },
    }


def _matches_any_selector(feature: str, selectors: List[Dict]) -> bool:
    """Check if a feature matches any feature selector."""
    if not selectors:
        return False

    for sel in selectors:
        sel_type = sel.get("type", "")
        if sel_type == "regex":
            pattern = sel.get("pattern", "")
            if pattern and re.match(pattern, feature):
                return True
        elif sel_type == "prefix":
            prefix = sel.get("value", "")
            if prefix and feature.startswith(prefix):
                return True

    return False


def _is_selector_set_alias(feature: str, selectors: List[Dict]) -> bool:
    """
    Accept compact set aliases (e.g., SET_1) when selector metadata exists.
    This avoids false positives for wide datasets represented via grouped selectors.
    """
    if not feature or not selectors:
        return False
    if not isinstance(feature, str):
        return False
    token = feature.strip()
    if not token:
        return False
    return bool(re.fullmatch(r"SET_\d+", token, flags=re.IGNORECASE))
