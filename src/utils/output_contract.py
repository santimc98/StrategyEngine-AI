import glob
import json
import os
import re
from typing import List, Dict, Any, Tuple, Optional

from src.utils.metric_eval import resolve_metric_value


def get_csv_dialect(work_dir: str = ".") -> Dict[str, Any]:
    """
    Read CSV dialect settings from cleaning_manifest.json.

    Args:
        work_dir: Working directory where data/cleaning_manifest.json may exist

    Returns:
        {"sep": str, "encoding": str} with defaults if manifest not found
    """
    defaults = {"sep": ",", "encoding": "utf-8"}

    manifest_path = os.path.join(work_dir, "data", "cleaning_manifest.json")
    if not os.path.exists(manifest_path):
        return defaults

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        output_dialect = manifest.get("output_dialect", {})
        if not isinstance(output_dialect, dict):
            return defaults

        return {
            "sep": output_dialect.get("sep", defaults["sep"]),
            "encoding": output_dialect.get("encoding", defaults["encoding"]),
        }
    except Exception:
        return defaults


def _split_deliverables(deliverables: List[Any]) -> Tuple[List[str], List[str]]:
    def _is_probably_path(value: str) -> bool:
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
        if ext in {".csv", ".json", ".md", ".png", ".jpg", ".jpeg", ".pdf", ".txt", ".html", ".xlsx", ".xls", ".parquet", ".pkl", ".pickle"}:
            return True
        return False

    required: List[str] = []
    optional: List[str] = []
    for item in deliverables or []:
        if isinstance(item, dict):
            path = item.get("path") or item.get("output") or item.get("artifact")
            if not path:
                continue
            if not _is_probably_path(str(path)):
                continue
            is_required = item.get("required")
            if is_required is None:
                is_required = True
            if is_required:
                required.append(path)
            else:
                optional.append(path)
        else:
            path = str(item)
            if not path:
                continue

            # CRITICAL FIX: Handle stringified dicts like "{'path': 'data/scored_rows.csv', ...}"
            # These come from execution_contract when artifact schemas are included in required_outputs
            # Extract just the path value and ignore schema metadata
            if path.startswith("{") and "'path':" in path:
                import ast
                try:
                    # Parse as Python literal to extract path
                    parsed = ast.literal_eval(path)
                    if isinstance(parsed, dict) and "path" in parsed:
                        path = parsed["path"]
                except Exception:
                    # If parsing fails, skip this entry (it's malformed)
                    continue

            if path and _is_probably_path(path):
                required.append(path)
    return required, optional


def _check_paths(paths: List[str]) -> Tuple[List[str], List[str]]:
    present: List[str] = []
    missing: List[str] = []
    for pattern in paths or []:
        try:
            if any(char in pattern for char in ["*", "?", "["]):
                matches = glob.glob(pattern)
                if matches:
                    present.extend(matches)
                else:
                    missing.append(pattern)
            else:
                if os.path.exists(pattern):
                    present.append(pattern)
                else:
                    missing.append(pattern)
        except Exception:
            missing.append(pattern)
    return present, missing


def _coerce_float_maybe(value: Any) -> Optional[float]:
    try:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value or "").strip()
        if not text:
            return None
        return float(text)
    except Exception:
        return None


def _normalize_rel_path(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text


def _extract_json_artifact_from_evidence(value: Any) -> Tuple[str, str]:
    text = _normalize_rel_path(value)
    lower = text.lower()
    marker = ".json"
    idx = lower.find(marker)
    if idx < 0:
        return "", ""
    artifact = text[: idx + len(marker)]
    metric_path = text[idx + len(marker) :].lstrip(".:/# ")
    return artifact, metric_path


def _load_json_artifact(path: str, work_dir: str) -> Tuple[Dict[str, Any], str]:
    rel = _normalize_rel_path(path)
    if not rel:
        return {}, ""
    abs_path = rel if os.path.isabs(rel) else os.path.join(work_dir or ".", rel)
    try:
        with open(abs_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return (payload if isinstance(payload, dict) else {}), abs_path
    except Exception:
        return {}, abs_path


def _collect_contract_qa_gates(contract: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(contract, dict):
        return []
    gates: List[Dict[str, Any]] = []

    def _extend(value: Any) -> None:
        if not isinstance(value, list):
            return
        for item in value:
            if isinstance(item, dict) and item:
                gates.append(item)

    _extend(contract.get("qa_gates"))
    for section_name in ("ml_engineer", "qa_reviewer", "reviewer"):
        section = contract.get(section_name)
        if isinstance(section, dict):
            _extend(section.get("qa_gates"))
            _extend(section.get("reviewer_gates"))
    evaluation_spec = contract.get("evaluation_spec") if isinstance(contract.get("evaluation_spec"), dict) else {}
    _extend(evaluation_spec.get("qa_gates"))
    return gates


def _gate_metric_name(gate: Dict[str, Any]) -> str:
    params = gate.get("params") if isinstance(gate.get("params"), dict) else {}
    metric = str(
        params.get("metric")
        or params.get("field")
        or params.get("metric_name")
        or gate.get("metric")
        or gate.get("field")
        or ""
    ).strip()
    if metric:
        return metric
    _artifact, evidence_metric = _extract_json_artifact_from_evidence(
        gate.get("evidence_source") or params.get("evidence_source")
    )
    return evidence_metric


def _gate_thresholds(gate: Dict[str, Any]) -> Dict[str, Optional[float] | str]:
    params = gate.get("params") if isinstance(gate.get("params"), dict) else {}
    min_value = _coerce_float_maybe(params.get("min_value"))
    if min_value is None:
        min_value = _coerce_float_maybe(params.get("min"))
    max_value = _coerce_float_maybe(params.get("max_value"))
    if max_value is None:
        max_value = _coerce_float_maybe(params.get("max"))
    threshold = _coerce_float_maybe(params.get("threshold"))
    if threshold is None:
        threshold = _coerce_float_maybe(params.get("target"))
    operator = str(params.get("operator") or "").strip()
    return {
        "min_value": min_value,
        "max_value": max_value,
        "threshold": threshold,
        "operator": operator,
    }


def _resolve_gate_artifact_and_metric(gate: Dict[str, Any]) -> Tuple[str, str]:
    params = gate.get("params") if isinstance(gate.get("params"), dict) else {}
    artifact = _normalize_rel_path(
        gate.get("applies_to_artifact")
        or params.get("artifact_path")
        or params.get("path")
    )
    metric = _gate_metric_name(gate)
    evidence_artifact, evidence_metric = _extract_json_artifact_from_evidence(
        gate.get("evidence_source") or params.get("evidence_source")
    )
    if not artifact and evidence_artifact:
        artifact = evidence_artifact
    if not metric and evidence_metric:
        metric = evidence_metric
    return artifact, metric


def _compare_numeric_gate(value: float, thresholds: Dict[str, Optional[float] | str]) -> Tuple[bool, List[str]]:
    failures: List[str] = []
    min_value = thresholds.get("min_value")
    max_value = thresholds.get("max_value")
    threshold = thresholds.get("threshold")
    operator = str(thresholds.get("operator") or "").strip()
    if isinstance(min_value, (int, float)) and value < float(min_value):
        failures.append(f"value {value:.6g} is below min_value {float(min_value):.6g}")
    if isinstance(max_value, (int, float)) and value > float(max_value):
        failures.append(f"value {value:.6g} is above max_value {float(max_value):.6g}")
    if isinstance(threshold, (int, float)) and operator:
        if operator == ">" and not value > float(threshold):
            failures.append(f"value {value:.6g} is not > {float(threshold):.6g}")
        elif operator == ">=" and not value >= float(threshold):
            failures.append(f"value {value:.6g} is not >= {float(threshold):.6g}")
        elif operator == "<" and not value < float(threshold):
            failures.append(f"value {value:.6g} is not < {float(threshold):.6g}")
        elif operator == "<=" and not value <= float(threshold):
            failures.append(f"value {value:.6g} is not <= {float(threshold):.6g}")
        elif operator == "==" and not value == float(threshold):
            failures.append(f"value {value:.6g} is not == {float(threshold):.6g}")
    return not failures, failures


_METRIC_THRESHOLD_STOPWORDS = {
    "max",
    "min",
    "value",
    "threshold",
    "target",
    "pct",
    "percent",
    "percentage",
    "rate",
    "ratio",
    "metric",
    "allowed",
    "tolerance",
    "tol",
    "limit",
}


def _metric_token_set(value: Any) -> set[str]:
    return {
        token
        for token in re.split(r"[^a-z0-9]+", str(value or "").lower())
        if token and token not in _METRIC_THRESHOLD_STOPWORDS
    }


def _extract_metric_candidates_from_evidence(value: Any) -> List[str]:
    _artifact, metric_text = _extract_json_artifact_from_evidence(value)
    if not metric_text:
        return []
    metrics: List[str] = []
    for part in re.split(r"[,;]", metric_text):
        match = re.match(
            r"\s*([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)",
            str(part or ""),
        )
        if not match:
            continue
        metric = match.group(1).strip(". ")
        if metric and metric.lower() not in {"for", "and", "or"} and metric not in metrics:
            metrics.append(metric)
    return metrics


def _operator_from_threshold_key(key: Any) -> str:
    token = str(key or "").strip().lower()
    if token.startswith(("max_", "upper_", "lte_", "le_")):
        return "<="
    if token.startswith(("min_", "lower_", "gte_", "ge_")):
        return ">="
    return ""


def _thresholds_from_check(
    check: Dict[str, Any],
    gate_params: Dict[str, Any],
) -> Dict[str, Optional[float] | str]:
    min_value = _coerce_float_maybe(check.get("min_value"))
    if min_value is None:
        min_value = _coerce_float_maybe(check.get("min"))
    max_value = _coerce_float_maybe(check.get("max_value"))
    if max_value is None:
        max_value = _coerce_float_maybe(check.get("max"))
    threshold = _coerce_float_maybe(check.get("threshold"))
    if threshold is None:
        threshold = _coerce_float_maybe(check.get("target"))
    threshold_param = str(check.get("threshold_param") or "").strip()
    if threshold is None and threshold_param:
        threshold = _coerce_float_maybe(gate_params.get(threshold_param))
    operator = str(check.get("operator") or "").strip()
    if not operator and threshold_param:
        operator = _operator_from_threshold_key(threshold_param)
    return {
        "min_value": min_value,
        "max_value": max_value,
        "threshold": threshold,
        "operator": operator,
        "threshold_param": threshold_param,
    }


def _matching_metric_for_threshold_param(param_key: str, metrics: List[str]) -> str:
    param_tokens = _metric_token_set(param_key)
    if not param_tokens:
        return ""
    best_metric = ""
    best_overlap = -1
    for metric in metrics:
        metric_tokens = _metric_token_set(metric)
        if not metric_tokens:
            continue
        overlap = len(param_tokens & metric_tokens)
        if param_tokens.issubset(metric_tokens):
            return metric
        if overlap > best_overlap and overlap >= max(1, len(param_tokens) - 1):
            best_metric = metric
            best_overlap = overlap
    return best_metric


def expand_numeric_gate_checks(
    gate: Dict[str, Any],
    *,
    primary_metric_name: str = "",
) -> List[Dict[str, Any]]:
    """Expand a QA gate into artifact-backed numeric metric checks.

    Supports explicit ``params.metric_checks`` and a contract-backward-compatible
    fallback for composite gates encoded as evidence_source plus max_/min_
    params, e.g. max_entity_drift_pct -> eligibility_drift_entity_pct.
    """
    if not isinstance(gate, dict):
        return []
    params = gate.get("params") if isinstance(gate.get("params"), dict) else {}
    gate_name = str(gate.get("name") or gate.get("id") or "").strip()
    severity = str(gate.get("severity") or "HARD").strip().upper()
    if severity not in {"HARD", "SOFT"}:
        severity = "HARD"
    base_artifact, evidence_metric = _resolve_gate_artifact_and_metric(gate)
    evidence_source = gate.get("evidence_source") or params.get("evidence_source")

    checks: List[Dict[str, Any]] = []

    def _append_check(metric: str, thresholds: Dict[str, Any], artifact: str | None = None) -> None:
        metric = str(metric or "").strip()
        artifact_path = _normalize_rel_path(artifact or base_artifact)
        if not metric or not artifact_path or not artifact_path.lower().endswith(".json"):
            return
        has_threshold = any(
            isinstance(thresholds.get(key), (int, float))
            for key in ("min_value", "max_value", "threshold")
        )
        if not has_threshold:
            return
        item = {
            "name": gate_name or metric,
            "severity": severity,
            "artifact_path": artifact_path,
            "metric": metric,
            "min_value": thresholds.get("min_value"),
            "max_value": thresholds.get("max_value"),
            "threshold": thresholds.get("threshold"),
            "operator": thresholds.get("operator"),
            "threshold_param": thresholds.get("threshold_param"),
            "evidence_source": evidence_source,
        }
        key = (
            item["name"],
            item["artifact_path"],
            item["metric"],
            item.get("operator"),
            item.get("threshold"),
            item.get("min_value"),
            item.get("max_value"),
        )
        existing_keys = {
            (
                existing.get("name"),
                existing.get("artifact_path"),
                existing.get("metric"),
                existing.get("operator"),
                existing.get("threshold"),
                existing.get("min_value"),
                existing.get("max_value"),
            )
            for existing in checks
        }
        if key not in existing_keys:
            checks.append(item)

    raw_metric_checks = params.get("metric_checks")
    if not isinstance(raw_metric_checks, list):
        raw_metric_checks = gate.get("metric_checks")
    if isinstance(raw_metric_checks, list):
        for raw_check in raw_metric_checks:
            if not isinstance(raw_check, dict):
                continue
            metric = str(
                raw_check.get("metric")
                or raw_check.get("field")
                or raw_check.get("metric_name")
                or ""
            ).strip()
            thresholds = _thresholds_from_check(raw_check, params)
            artifact = raw_check.get("artifact_path") or raw_check.get("applies_to_artifact")
            _append_check(metric, thresholds, str(artifact or ""))

    thresholds = _gate_thresholds(gate)
    metric = _gate_metric_name(gate) or evidence_metric or primary_metric_name
    _append_check(metric, thresholds)

    evidence_metrics = _extract_metric_candidates_from_evidence(evidence_source)
    if evidence_metrics:
        for key, value in params.items():
            key_text = str(key or "").strip()
            if key_text in {
                "min",
                "max",
                "min_value",
                "max_value",
                "threshold",
                "target",
                "metric",
                "field",
                "metric_name",
            }:
                continue
            threshold_value = _coerce_float_maybe(value)
            if threshold_value is None:
                continue
            operator = _operator_from_threshold_key(key_text)
            if not operator:
                continue
            matched_metric = _matching_metric_for_threshold_param(key_text, evidence_metrics)
            if not matched_metric:
                continue
            _append_check(
                matched_metric,
                {
                    "min_value": None,
                    "max_value": None,
                    "threshold": float(threshold_value),
                    "operator": operator,
                    "threshold_param": key_text,
                },
            )
    return checks


def evaluate_numeric_qa_gates(contract: Dict[str, Any], work_dir: str = ".") -> Dict[str, Any]:
    """Evaluate artifact-backed numeric QA gates against produced JSON artifacts."""
    checked: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    for gate in _collect_contract_qa_gates(contract if isinstance(contract, dict) else {}):
        if not isinstance(gate, dict):
            continue
        for check_index, check in enumerate(expand_numeric_gate_checks(gate)):
            artifact = str(check.get("artifact_path") or "").strip()
            metric = str(check.get("metric") or "").strip()
            severity = str(check.get("severity") or "HARD").strip().upper()
            payload, abs_path = _load_json_artifact(artifact, work_dir)
            resolved = resolve_metric_value(payload, metric) if payload else {}
            value = _coerce_float_maybe(resolved.get("value") if isinstance(resolved, dict) else None)
            item = {
                "name": str(check.get("name") or metric).strip() or metric,
                "severity": severity,
                "artifact_path": artifact,
                "metric": metric,
                "check_index": check_index,
                "value": value,
                "min_value": check.get("min_value"),
                "max_value": check.get("max_value"),
                "threshold": check.get("threshold"),
                "operator": check.get("operator"),
                "threshold_param": check.get("threshold_param"),
                "matched_key": resolved.get("matched_key") if isinstance(resolved, dict) else None,
                "evidence_source": check.get("evidence_source"),
                "abs_path": abs_path,
            }
            if value is None:
                item["status"] = "missing_evidence"
                item["passed"] = False
                item["detail"] = f"Could not resolve numeric metric '{metric}' from {artifact}."
            else:
                passed, reasons = _compare_numeric_gate(float(value), check)
                item["status"] = "pass" if passed else "fail"
                item["passed"] = bool(passed)
                item["detail"] = "; ".join(reasons) if reasons else "numeric gate satisfied"
            checked.append(item)
            if item.get("passed") is False:
                if severity == "HARD":
                    failures.append(item)
                else:
                    warnings.append(item)
    return {
        "checked": checked,
        "failures": failures,
        "warnings": warnings,
        "summary": (
            f"Numeric QA gates checked: {len(checked)}; "
            f"hard_failures={len(failures)}; warnings={len(warnings)}"
        ),
    }


def check_required_outputs(required_outputs: List[Any]) -> Dict[str, object]:
    """
    Best-effort validation of required outputs.
    Supports glob patterns (e.g., static/plots/*.png).
    Returns dict with present, missing, summary.
    Never raises.
    """
    required, optional = _split_deliverables(required_outputs or [])
    present_required, missing_required = _check_paths(required)
    present_optional, missing_optional = _check_paths(optional)
    present = present_required + present_optional
    summary = (
        f"Present: {len(present)}; Missing required: {len(missing_required)}; "
        f"Missing optional: {len(missing_optional)}"
    )
    return {
        "present": present,
        "missing": missing_required,
        "missing_optional": missing_optional,
        "summary": summary,
    }


def check_required_outputs_for_owner(
    required_outputs: List[Any], owner: str
) -> Dict[str, object]:
    """Validate required outputs filtered by owner.

    Filters the list to only items belonging to *owner* before delegating to
    check_required_outputs(). Items without an explicit 'owner' field are
    assigned one via path-based inference.
    """
    from src.utils.contract_accessors import _infer_owner_from_path

    filtered: List[Any] = []
    for item in required_outputs or []:
        if isinstance(item, dict):
            path = item.get("path") or ""
            item_owner = item.get("owner") or _infer_owner_from_path(path)
        elif isinstance(item, str):
            item_owner = _infer_owner_from_path(item)
        else:
            continue
        if item_owner == owner:
            filtered.append(item)
    return check_required_outputs(filtered)


def check_scored_rows_schema(
    scored_rows_path: str,
    required_columns: List[str],
    required_any_of_groups: Optional[List[List[str]]] = None,
    required_any_of_group_severity: Optional[List[str]] = None,
    dialect: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Check that scored_rows.csv exists and contains required columns.
    Supports optional any-of groups for semantic column matching with severity.

    Args:
        scored_rows_path: Path to scored_rows.csv
        required_columns: List of required column names (legacy)
        required_any_of_groups: List of groups, each group is a list of column names.
                                At least one column from each group must be present.
        required_any_of_group_severity: List of severity strings ("fail" or "warning") for each group.
                                        Defaults to all "fail" except first group => "warning".
        dialect: Optional {"sep": str, "encoding": str} from get_csv_dialect()

    Returns:
        {
            "exists": bool,
            "present_columns": [...],
            "missing_columns": [...],
            "missing_any_of_groups": [...],
            "matched_any_of_groups": [...],
            "missing_any_of_groups_with_severity": [{"group": [...], "severity": "..."}],
            "summary": str
        }
    """
    if not os.path.exists(scored_rows_path):
        return {
            "exists": False,
            "present_columns": [],
            "missing_columns": required_columns,
            "missing_any_of_groups": required_any_of_groups or [],
            "matched_any_of_groups": [],
            "missing_any_of_groups_with_severity": [],
            "summary": f"File not found: {scored_rows_path}",
        }

    # Use dialect settings or defaults
    sep = dialect.get("sep", ",") if dialect else ","
    encoding = dialect.get("encoding", "utf-8") if dialect else "utf-8"

    try:
        import pandas as pd
        # Only read header to check columns, respecting dialect
        df_header = pd.read_csv(scored_rows_path, nrows=0, sep=sep, encoding=encoding)
        actual_cols = list(df_header.columns)
        actual_norm = {col.strip().lower(): col for col in actual_cols}

        # Check required_columns (legacy, case-insensitive)
        present = []
        missing = []
        for col in required_columns:
            if col.lower() in actual_norm:
                present.append(col)
            else:
                missing.append(col)

        # Default severity: all "fail" except first group => "warning"
        if required_any_of_groups and not required_any_of_group_severity:
            required_any_of_group_severity = ["warning"] + ["fail"] * (len(required_any_of_groups) - 1)

        # Check any-of groups with severity
        missing_groups = []
        matched_groups = []
        missing_with_severity = []
        if required_any_of_groups:
            for idx, group in enumerate(required_any_of_groups):
                if not group or not isinstance(group, list):
                    continue
                group_norm = [g.strip().lower() for g in group if g]
                matches = [actual_norm.get(g) for g in group_norm if g in actual_norm]
                if not matches:
                    missing_groups.append(group)
                    severity = required_any_of_group_severity[idx] if idx < len(required_any_of_group_severity) else "fail"
                    missing_with_severity.append({"group": group, "severity": severity})
                else:
                    matched_groups.append({"group": group, "matched": matches})

        summary = f"Columns - Present: {len(present)}/{len(required_columns)}"
        if missing:
            summary += f"; Missing: {', '.join(missing)}"
        if missing_groups:
            summary += f"; Missing groups: {len(missing_groups)}"

        return {
            "exists": True,
            "present_columns": present,
            "missing_columns": missing,
            "missing_any_of_groups": missing_groups,
            "matched_any_of_groups": matched_groups,
            "missing_any_of_groups_with_severity": missing_with_severity,
            "summary": summary,
        }
    except Exception as e:
        return {
            "exists": True,
            "present_columns": [],
            "missing_columns": required_columns,
            "missing_any_of_groups": required_any_of_groups or [],
            "matched_any_of_groups": [],
            "missing_any_of_groups_with_severity": [],
            "summary": f"Error reading file: {e}",
        }


def _coerce_positive_int(value: Any) -> Optional[int]:
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


def _count_csv_rows(
    csv_path: str,
    dialect: Optional[Dict[str, str]] = None,
) -> int:
    import pandas as pd

    sep = dialect.get("sep", ",") if isinstance(dialect, dict) else ","
    encoding = dialect.get("encoding", "utf-8") if isinstance(dialect, dict) else "utf-8"
    total = 0
    # Chunked read keeps memory bounded for large artifacts.
    for chunk in pd.read_csv(csv_path, sep=sep, encoding=encoding, chunksize=200_000):
        total += int(len(chunk))
    return int(total)


def _split_contract_rule_clauses(rule: str) -> List[str]:
    text = str(rule or "").strip()
    if not text:
        return []
    protected = re.sub(
        r"(?is)\bBETWEEN\s+'[^']+'\s+AND\s+'[^']+'",
        lambda m: m.group(0).replace(" AND ", " __BETWEEN_AND__ "),
        text,
    )
    clauses = re.split(r"(?i)\s+AND\s+", protected)
    return [clause.replace("__BETWEEN_AND__", " AND ").strip() for clause in clauses if clause.strip()]


def _coerce_rule_literal(series: "pd.Series", raw_value: str):
    import pandas as pd

    value = str(raw_value).strip()
    if value.startswith("'") and value.endswith("'"):
        value = value[1:-1]

    parsed_int = _coerce_positive_int(value)
    if parsed_int is not None and not re.search(r"[-:/T]", value):
        return pd.to_numeric(series, errors="coerce"), float(parsed_int)

    if re.search(r"\d{4}-\d{2}-\d{2}", value):
        return pd.to_datetime(series, errors="coerce"), pd.Timestamp(value)

    return series.astype(str).str.strip(), str(value)


def _evaluate_contract_rule(df, rule: str) -> Tuple[Optional["pd.Series"], List[str], List[str]]:
    import pandas as pd

    clauses = _split_contract_rule_clauses(rule)
    if not clauses:
        return None, [], ["empty_rule"]

    mask = pd.Series(True, index=df.index)
    referenced_cols: List[str] = []
    errors: List[str] = []

    for clause in clauses:
        between_match = re.match(
            r"(?is)^([A-Za-z_][A-Za-z0-9_]*)\s+BETWEEN\s+'([^']+)'\s+AND\s+'([^']+)'$",
            clause,
        )
        if between_match:
            col, low_raw, high_raw = between_match.groups()
            referenced_cols.append(col)
            if col not in df.columns:
                errors.append(f"missing_column:{col}")
                continue
            series = pd.to_datetime(df[col], errors="coerce")
            mask &= series.between(pd.Timestamp(low_raw), pd.Timestamp(high_raw), inclusive="both")
            continue

        null_match = re.match(r"(?is)^([A-Za-z_][A-Za-z0-9_]*)\s+IS\s+(NOT\s+)?NULL$", clause)
        if null_match:
            col, not_token = null_match.groups()
            referenced_cols.append(col)
            if col not in df.columns:
                errors.append(f"missing_column:{col}")
                continue
            mask &= df[col].notna() if not_token else df[col].isna()
            continue

        compare_match = re.match(
            r"(?is)^([A-Za-z_][A-Za-z0-9_]*)\s*(<=|>=|=|<|>)\s*('[^']*'|[-+]?\d+(?:\.\d+)?)$",
            clause,
        )
        if compare_match:
            col, operator, raw_value = compare_match.groups()
            referenced_cols.append(col)
            if col not in df.columns:
                errors.append(f"missing_column:{col}")
                continue
            lhs, rhs = _coerce_rule_literal(df[col], raw_value)
            if operator == "<=":
                mask &= lhs <= rhs
            elif operator == ">=":
                mask &= lhs >= rhs
            elif operator == "<":
                mask &= lhs < rhs
            elif operator == ">":
                mask &= lhs > rhs
            else:
                mask &= lhs == rhs
            continue

        errors.append(f"unsupported_clause:{clause}")

    if errors:
        return None, referenced_cols, errors
    return mask, referenced_cols, []


def _resolve_contract_subset_validation(
    contract: Dict[str, Any],
    artifact_requirements: Dict[str, Any],
    work_dir: str,
    dialect: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    import pandas as pd
    from src.utils.contract_accessors import (
        get_artifact_requirements,
        get_cleaned_dataset_output_path,
        get_declared_artifact_path_by_intent,
        normalize_artifact_path,
    )

    artifact_requirements = dict(artifact_requirements) if isinstance(artifact_requirements, dict) else {}
    hydrated_file_schemas = dict(artifact_requirements.get("file_schemas") or {})

    checked: List[Dict[str, Any]] = []
    mismatches: List[Dict[str, Any]] = []
    mismatch_seen = set()
    errors: List[Dict[str, Any]] = []

    if not isinstance(contract, dict):
        return hydrated_file_schemas, {
            "checked": checked,
            "mismatches": mismatches,
            "errors": errors,
            "summary": "Subset selector checks skipped: invalid contract",
        }

    top_artifacts = get_artifact_requirements(contract)
    full_archive_cfg = top_artifacts.get("full_archive_dataset") if isinstance(top_artifacts, dict) else {}
    task_semantics = contract.get("task_semantics") if isinstance(contract.get("task_semantics"), dict) else {}
    split_spec = contract.get("split_spec") if isinstance(contract.get("split_spec"), dict) else {}
    source_rel = (
        get_declared_artifact_path_by_intent(contract, "full_fidelity_archive", owner="data_engineer", required_only=True)
        or (full_archive_cfg.get("output_path") if isinstance(full_archive_cfg, dict) else "")
        or get_cleaned_dataset_output_path(contract)
    )
    source_rel = normalize_artifact_path(source_rel)
    source_abs = os.path.join(work_dir, source_rel) if source_rel else ""
    source_df = None
    if source_abs and os.path.exists(source_abs):
        sep = dialect.get("sep", ",") if isinstance(dialect, dict) else ","
        encoding = dialect.get("encoding", "utf-8") if isinstance(dialect, dict) else "utf-8"
        source_df = pd.read_csv(source_abs, sep=sep, encoding=encoding)

    checks = [
        {
            "intent": "holdout_predictions",
            "rule": str(
                contract.get("scoring_rows_rule_secondary")
                or task_semantics.get("scoring_rows_rule_secondary")
                or split_spec.get("scoring_rows_rule_secondary")
                or ""
            ).strip(),
        },
        {
            "intent": "scoring_output",
            "rule": str(
                contract.get("scoring_rows_rule_primary")
                or task_semantics.get("scoring_rows_rule_primary")
                or split_spec.get("scoring_rows_rule_primary")
                or ""
            ).strip(),
        },
    ]

    for item in checks:
        intent = item["intent"]
        rule = item["rule"]
        if not rule:
            continue

        output_rel = get_declared_artifact_path_by_intent(contract, intent, owner="ml_engineer", required_only=True)
        output_rel = normalize_artifact_path(output_rel)
        if not output_rel:
            continue
        output_abs = os.path.join(work_dir, output_rel)

        expected_count = None
        referenced_cols: List[str] = []
        if source_df is not None:
            source_mask, referenced_cols, source_errors = _evaluate_contract_rule(source_df, rule)
            if source_errors:
                errors.append(
                    {
                        "path": output_rel,
                        "intent": intent,
                        "rule": rule,
                        "source_path": source_rel,
                        "errors": source_errors,
                    }
                )
            elif source_mask is not None:
                expected_count = int(source_mask.sum())

        schema_obj = hydrated_file_schemas.get(output_rel)
        if not isinstance(schema_obj, dict):
            schema_obj = {}
        if expected_count is not None and _coerce_positive_int(schema_obj.get("expected_row_count")) is None:
            schema_obj["expected_row_count"] = int(expected_count)
            schema_obj.setdefault("selector_rule", rule)
            schema_obj.setdefault("selector_intent", intent)
            hydrated_file_schemas[output_rel] = schema_obj

        if not os.path.exists(output_abs):
            continue

        actual_count = _count_csv_rows(output_abs, dialect=dialect)
        payload: Dict[str, Any] = {
            "path": output_rel,
            "intent": intent,
            "rule": rule,
            "actual_row_count": actual_count,
            "expected_row_count": expected_count,
            "matches": expected_count is None or actual_count == expected_count,
        }

        def _record_mismatch(kind: str) -> None:
            key = (output_rel, intent, kind)
            if key in mismatch_seen:
                return
            mismatch_seen.add(key)
            mismatch_payload = dict(payload)
            mismatch_payload["mismatch_kind"] = kind
            mismatches.append(mismatch_payload)

        sep = dialect.get("sep", ",") if isinstance(dialect, dict) else ","
        encoding = dialect.get("encoding", "utf-8") if isinstance(dialect, dict) else "utf-8"
        output_df = pd.read_csv(output_abs, sep=sep, encoding=encoding)
        output_mask, output_referenced, output_errors = _evaluate_contract_rule(output_df, rule)
        selector_cols_present = bool(output_referenced or referenced_cols) and all(
            col in output_df.columns for col in (output_referenced or referenced_cols)
        )
        payload["selector_columns_present"] = selector_cols_present

        if output_errors and not any(err.startswith("missing_column:") for err in output_errors):
            payload["selector_errors"] = output_errors
            errors.append(dict(payload))
        elif selector_cols_present and output_mask is not None:
            matches_rule = bool(output_mask.all())
            payload["all_rows_match_rule"] = matches_rule
            payload["rows_matching_rule"] = int(output_mask.sum())
            if not matches_rule:
                _record_mismatch("selector_rule")

        if expected_count is not None and actual_count != expected_count:
            _record_mismatch("row_count")

        checked.append(payload)

    summary = (
        f"Subset selector checks: {len(checked)}; "
        f"Mismatches: {len(mismatches)}; Errors: {len(errors)}"
    )
    return hydrated_file_schemas, {
        "checked": checked,
        "mismatches": mismatches,
        "errors": errors,
        "summary": summary,
    }


def check_csv_row_counts(
    file_schemas: Dict[str, Any],
    work_dir: str = ".",
    dialect: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Validate optional expected_row_count for CSV artifacts declared in file_schemas.

    Args:
        file_schemas: artifact_requirements.file_schemas mapping
        work_dir: Base directory for relative artifact paths
        dialect: Optional CSV dialect settings

    Returns:
        {
            "checked": [...],
            "mismatches": [...],
            "errors": [...],
            "summary": str
        }
    """
    checked: List[Dict[str, Any]] = []
    mismatches: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    if not isinstance(file_schemas, dict):
        return {
            "checked": checked,
            "mismatches": mismatches,
            "errors": errors,
            "summary": "Row count checks skipped: invalid file_schemas",
        }

    for raw_path, schema in file_schemas.items():
        path = str(raw_path or "").strip()
        if not path or not path.lower().endswith(".csv"):
            continue
        if not isinstance(schema, dict):
            continue
        expected = _coerce_positive_int(schema.get("expected_row_count"))
        if expected is None:
            continue

        artifact_path = os.path.join(work_dir, path)
        if not os.path.exists(artifact_path):
            # Missing files are already surfaced by required_files checks.
            continue

        try:
            actual = _count_csv_rows(artifact_path, dialect=dialect)
            payload = {
                "path": path,
                "expected_row_count": expected,
                "actual_row_count": actual,
                "matches": bool(actual == expected),
            }
            checked.append(payload)
            if actual != expected:
                mismatches.append(payload)
        except Exception as err:
            errors.append({"path": path, "expected_row_count": expected, "error": str(err)})

    summary = (
        f"Row counts checked: {len(checked)}; "
        f"Mismatches: {len(mismatches)}; Errors: {len(errors)}"
    )
    return {
        "checked": checked,
        "mismatches": mismatches,
        "errors": errors,
        "summary": summary,
    }


def check_artifact_requirements(
    artifact_requirements: Dict[str, Any],
    work_dir: str = ".",
    contract: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    P1.4: Check all artifact requirements including files and scored_rows schema.

    Args:
        artifact_requirements: Normalized artifact requirements dict
        work_dir: Working directory for file paths

    Returns:
        {
            "status": "ok" | "warning" | "error",
            "files_report": {...},
            "scored_rows_report": {...},
            "summary": str
        }
    """
    if not isinstance(artifact_requirements, dict):
        return {
            "status": "error",
            "files_report": {},
            "scored_rows_report": {},
            "row_count_report": {},
            "selector_report": {},
            "summary": "Invalid artifact_requirements",
        }

    # Check required files
    required_files = artifact_requirements.get("required_files", [])
    file_paths = []
    for f in required_files:
        if isinstance(f, dict):
            path = f.get("path", "")
        else:
            path = str(f) if f else ""
        if path:
            file_paths.append(os.path.join(work_dir, path))

    files_report = check_required_outputs(file_paths)
    file_schemas = artifact_requirements.get("file_schemas", {})
    dialect = get_csv_dialect(work_dir)
    selector_report = {
        "checked": [],
        "mismatches": [],
        "errors": [],
        "summary": "Subset selector checks skipped: no contract context",
    }
    if isinstance(contract, dict):
        file_schemas, selector_report = _resolve_contract_subset_validation(
            contract,
            artifact_requirements,
            work_dir=work_dir,
            dialect=dialect,
        )
    row_count_report = check_csv_row_counts(file_schemas, work_dir=work_dir, dialect=dialect)

    # Check scored_rows schema
    scored_schema = artifact_requirements.get("scored_rows_schema", {})
    if not isinstance(scored_schema, dict):
        scored_schema = {}
    required_columns = scored_schema.get("required_columns", [])
    required_any_of_groups = scored_schema.get("required_any_of_groups")
    required_any_of_group_severity = scored_schema.get("required_any_of_group_severity")

    # Find scored_rows file
    scored_rows_path = None
    for f in required_files:
        path = f.get("path", "") if isinstance(f, dict) else str(f)
        if "scored" in path.lower() and path.endswith(".csv"):
            scored_rows_path = os.path.join(work_dir, path)
            break

    if scored_rows_path:
        scored_rows_report = check_scored_rows_schema(
            scored_rows_path,
            required_columns,
            required_any_of_groups,
            required_any_of_group_severity,
            dialect
        )
    else:
        scored_rows_report = {
            "exists": False,
            "applicable": False,
            "present_columns": [],
            "missing_columns": [],
            "missing_any_of_groups": [],
            "matched_any_of_groups": [],
            "missing_any_of_groups_with_severity": [],
            "summary": "Not applicable (no scored_rows artifact declared in required_files)",
        }

    # Determine overall status
    if files_report.get("missing"):
        status = "error"
    elif selector_report.get("mismatches") or selector_report.get("errors"):
        status = "error"
    elif row_count_report.get("mismatches") or row_count_report.get("errors"):
        status = "error"
    elif scored_rows_report.get("applicable", True) and scored_rows_report.get("missing_columns"):
        status = "error"
    else:
        # Check any-of groups with severity
        missing_with_severity = (
            scored_rows_report.get("missing_any_of_groups_with_severity", [])
            if scored_rows_report.get("applicable", True)
            else []
        )
        has_fail_severity = any(m["severity"] == "fail" for m in missing_with_severity)
        if has_fail_severity:
            status = "error"
        elif missing_with_severity:
            status = "warning"
        else:
            status = "ok"

    summary = (
        f"Files: {files_report.get('summary', 'N/A')}; "
        f"Rows: {row_count_report.get('summary', 'N/A')}; "
        f"Scored rows: {scored_rows_report.get('summary', 'N/A')}; "
        f"Subset selectors: {selector_report.get('summary', 'N/A')}"
    )

    return {
        "status": status,
        "files_report": files_report,
        "row_count_report": row_count_report,
        "selector_report": selector_report,
        "scored_rows_report": scored_rows_report,
        "summary": summary,
    }


def build_output_contract_report(
    contract: Dict[str, Any],
    work_dir: str = ".",
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a comprehensive output contract report combining:
      - required outputs presence check (backward compatible)
      - artifact requirements schema validation (new)

    This is the unified entry point for output_contract_report.json generation.

    Args:
        contract: The execution contract (V4.1)
        work_dir: Working directory for file checks
        reason: Optional abort/failure reason

    Returns:
        {
            # Backward compatible keys
            "present": [...],
            "missing": [...],
            "missing_optional": [...],
            "summary": str,
            # New keys
            "artifact_requirements_report": {...},
            "overall_status": "ok" | "warning" | "error",
            # Optional
            "reason": str (if provided)
        }
    """
    # Import here to avoid circular imports
    from src.utils.contract_accessors import get_required_outputs, get_artifact_requirements

    # 1) Check required outputs presence (backward compatible)
    required_outputs = get_required_outputs(contract) if isinstance(contract, dict) else []
    required_outputs_report = check_required_outputs(required_outputs)

    # 2) Check artifact requirements with schema validation
    artifact_req = get_artifact_requirements(contract) if isinstance(contract, dict) else {}
    artifact_report = check_artifact_requirements(artifact_req, work_dir=work_dir, contract=contract)

    # 3) Evaluate artifact-backed numeric QA gates. These gates are contract
    # compliance, not advisory reviewer opinion: if a HARD numeric gate fails
    # against a produced artifact, the output contract is not satisfied.
    qa_gate_report = evaluate_numeric_qa_gates(contract if isinstance(contract, dict) else {}, work_dir=work_dir)

    # 4) Build unified report
    report: Dict[str, Any] = {
        # Backward compatible keys
        "present": required_outputs_report.get("present", []),
        "missing": required_outputs_report.get("missing", []),
        "missing_optional": required_outputs_report.get("missing_optional", []),
        "summary": required_outputs_report.get("summary", ""),
        # New keys
        "artifact_requirements_report": artifact_report,
        "qa_gate_results": qa_gate_report,
    }

    # 5) Derive overall_status
    # Error if: artifact_report has error OR missing files in required outputs
    # OR artifact-backed HARD numeric gates fail.
    if artifact_report.get("status") == "error" or report["missing"] or qa_gate_report.get("failures"):
        overall_status = "error"
    elif artifact_report.get("status") == "warning" or qa_gate_report.get("warnings"):
        overall_status = "warning"
    else:
        overall_status = "ok"

    report["overall_status"] = overall_status
    if qa_gate_report.get("failures"):
        report["qa_gate_failures"] = qa_gate_report.get("failures") or []
        report["hard_failures"] = [
            str(item.get("name") or item.get("metric") or "qa_gate_failure")
            for item in (qa_gate_report.get("failures") or [])
            if isinstance(item, dict)
        ]
    if qa_gate_report.get("warnings"):
        report["qa_gate_warnings"] = qa_gate_report.get("warnings") or []
    if qa_gate_report.get("checked"):
        report["summary"] = (
            f"{report.get('summary') or ''} {qa_gate_report.get('summary') or ''}"
        ).strip()

    # 6) Include reason if provided
    if reason:
        report["reason"] = reason

    return report
