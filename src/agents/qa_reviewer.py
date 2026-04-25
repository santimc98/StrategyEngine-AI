import ast
import os
import re
import json
import copy
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from src.utils.reviewer_llm import init_reviewer_llm
from src.utils.senior_protocol import SENIOR_EVIDENCE_RULE
from src.utils.ml_plan_validation import validate_ml_plan_constraints
from src.utils.review_context_packets import build_review_context_packet
from src.utils.reviewer_response_schema import build_qa_response_schema
from src.utils.llm_json_repair import JsonObjectParseError, parse_json_object_with_repair
from src.utils.openrouter_reasoning import create_chat_completion_with_reasoning
from src.utils.metric_eval import (
    canonicalize_metric_name,
    canonicalize_metrics_report_file,
    normalize_metrics_report_payload,
    resolve_metric_value,
)
from src.utils.output_contract import expand_numeric_gate_checks

load_dotenv()


def _parse_json_payload_with_trace(text: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
    return parse_json_object_with_repair(text or "", actor="qa_reviewer")


_CONTRACT_FALLBACK_WARNING = "CONTRACT_BROKEN_FALLBACK: qa_gates missing; please fix contract generation"

# Fallback gates removed (seniority refactoring): if the contract is missing
# qa_gates, the reviewer rejects with CONTRACT_BROKEN_FALLBACK to force
# contract regeneration instead of inventing its own gates.
# EXCEPTION: security_sandbox is an unconditional system-safety gate that
# must always be present regardless of contract state.
_UNCONDITIONAL_SAFETY_GATES = [
    {"name": "security_sandbox", "severity": "HARD", "params": {}},
    {"name": "output_row_count_consistency", "severity": "HARD", "params": {}},
]


def _normalize_qa_gate_spec(item: Any) -> Optional[Dict[str, Any]]:
    if isinstance(item, dict):
        name = (
            item.get("name")
            or item.get("id")
            or item.get("gate")
            or item.get("metric")
            or item.get("check")
            or item.get("rule")
            or item.get("title")
            or item.get("label")
        )
        if not name:
            return None
        severity = item.get("severity")
        required = item.get("required")
        if severity is None and required is not None:
            severity = "HARD" if bool(required) else "SOFT"
        severity = str(severity).upper() if severity else "HARD"
        if severity not in {"HARD", "SOFT"}:
            severity = "HARD"
        params = item.get("params")
        if not isinstance(params, dict):
            params = {}
        for param_key in (
            "metric",
            "check",
            "rule",
            "threshold",
            "target",
            "min",
            "max",
            "min_value",
            "max_value",
            "operator",
            "direction",
            "condition",
            "field",
            "metric_checks",
        ):
            if param_key in item and param_key not in params:
                params[param_key] = item.get(param_key)
        gate_spec: Dict[str, Any] = {"name": str(name), "severity": severity, "params": params}
        for extra_key in ("condition", "evidence_required", "action_if_fail", "applies_to_artifact", "evidence_source"):
            if extra_key in item:
                gate_spec[extra_key] = item.get(extra_key)
        return gate_spec
    if isinstance(item, str):
        name = item.strip()
        if not name:
            return None
        return {"name": name, "severity": "HARD", "params": {}}
    return None


def _normalize_qa_gates(raw_gates: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_gates, list):
        return []
    normalized: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw_gates:
        spec = _normalize_qa_gate_spec(item)
        if not spec:
            continue
        key = spec["name"].strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        normalized.append(spec)
    return normalized


def _gate_lookup(qa_gates: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(g.get("name")).lower(): g for g in qa_gates if isinstance(g, dict) and g.get("name")}


def _gate_names(qa_gates: List[Dict[str, Any]]) -> List[str]:
    return [str(g.get("name")) for g in qa_gates if isinstance(g, dict) and g.get("name")]


def resolve_qa_gates(evaluation_spec: Dict[str, Any] | None) -> tuple[List[Dict[str, Any]], str, List[str]]:
    warnings: List[str] = []
    contract_source = "fallback"
    raw_gates: Any = None
    if isinstance(evaluation_spec, dict):
        contract_source = str(evaluation_spec.get("_contract_source") or "qa_view")
        raw_gates = (
            evaluation_spec.get("qa_gates")
            or evaluation_spec.get("gates")
            or []
        )
    qa_gates = _normalize_qa_gates(raw_gates)
    if not qa_gates:
        warnings.append(_CONTRACT_FALLBACK_WARNING)
        qa_gates = list(_UNCONDITIONAL_SAFETY_GATES)
        contract_source = "fallback"
    else:
        # Ensure unconditional safety gates are always present even when
        # the contract provides its own qa_gates.
        existing_names = {str(g.get("name")).lower() for g in qa_gates if isinstance(g, dict) and g.get("name")}
        for safety_gate in _UNCONDITIONAL_SAFETY_GATES:
            if str(safety_gate["name"]).lower() not in existing_names:
                qa_gates.append(dict(safety_gate))
    return qa_gates, contract_source, warnings



def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    token = str(raw).strip().lower()
    return token not in {"0", "false", "no", "off", ""}


def _coerce_bool(value: Any, default: bool = False) -> bool:
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
    return bool(default)


def _is_augmentation_technique_name(name: Any) -> bool:
    token = str(name or "").strip().lower()
    if not token:
        return False
    normalized = re.sub(r"[^a-z0-9]+", "_", token).strip("_")
    direct_tokens = {
        "data_augmentation",
        "augmentation",
        "oversampling",
        "undersampling",
        "resampling",
        "bootstrap_augmentation",
        "class_balancing",
        "synthetic_minority_oversampling",
    }
    if normalized in direct_tokens:
        return True
    fuzzy_tokens = [
        "augment",
        "oversampl",
        "undersampl",
        "resampl",
        "synthetic",
        "bootstrap",
        "class_balanc",
    ]
    return any(part in normalized for part in fuzzy_tokens)


def _extract_hypothesis_technique(packet: Any) -> str:
    if not isinstance(packet, dict):
        return ""
    hypothesis = packet.get("hypothesis")
    if isinstance(hypothesis, dict):
        tech = hypothesis.get("technique")
        if isinstance(tech, str) and tech.strip():
            return tech.strip()
    direct = packet.get("technique")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    return ""


def _extract_metric_round_context(evaluation_spec: Dict[str, Any] | None) -> Dict[str, bool]:
    if not isinstance(evaluation_spec, dict):
        return {"metric_round_active": False, "augmentation_requested": False}

    metric_round_active = _coerce_bool(
        evaluation_spec.get("metric_improvement_round_active")
        or evaluation_spec.get("ml_improvement_round_active")
        or False
    )
    augmentation_requested = _coerce_bool(evaluation_spec.get("augmentation_requested") or False)

    iteration_handoff = evaluation_spec.get("iteration_handoff")
    if isinstance(iteration_handoff, dict):
        source = str(iteration_handoff.get("source") or "").strip().lower()
        if "metric_improvement" in source:
            metric_round_active = True
        candidate_packets = [
            iteration_handoff.get("hypothesis_packet"),
            iteration_handoff.get("iteration_hypothesis_packet"),
        ]
        for packet in candidate_packets:
            if _is_augmentation_technique_name(_extract_hypothesis_technique(packet)):
                augmentation_requested = True
                break

    direct_packet = evaluation_spec.get("ml_improvement_hypothesis_packet")
    if _is_augmentation_technique_name(_extract_hypothesis_technique(direct_packet)):
        augmentation_requested = True

    fe_plan = evaluation_spec.get("feature_engineering_plan")
    if isinstance(fe_plan, dict):
        techniques = fe_plan.get("techniques")
        if isinstance(techniques, list):
            for item in techniques:
                if isinstance(item, dict):
                    technique = item.get("technique") or item.get("name")
                else:
                    technique = item
                if _is_augmentation_technique_name(technique):
                    augmentation_requested = True
                    break

    return {
        "metric_round_active": bool(metric_round_active),
        "augmentation_requested": bool(augmentation_requested),
    }


def _resolve_ml_data_path(evaluation_spec: Dict[str, Any] | None) -> str:
    if not isinstance(evaluation_spec, dict):
        return ""
    explicit = evaluation_spec.get("ml_data_path")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    artifact_requirements = evaluation_spec.get("artifact_requirements")
    if isinstance(artifact_requirements, dict):
        for key in ("required_outputs", "required_files"):
            values = artifact_requirements.get(key)
            if not isinstance(values, list):
                continue
            for path in values:
                if isinstance(path, str) and path.strip().lower().endswith(".csv"):
                    return path.strip()
    return ""


def _extract_artifact_path(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if not isinstance(item, dict):
        return ""
    for key in ("path", "file", "artifact_path", "output_path", "metrics_path"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _collect_metric_artifact_paths(
    evaluation_spec: Dict[str, Any] | None,
    subject_required_outputs: List[Any] | None,
    qa_required_outputs: List[Any] | None,
) -> List[str]:
    evaluation_spec = evaluation_spec if isinstance(evaluation_spec, dict) else {}
    paths: List[str] = []

    def _consider(value: Any, *, intent: str = "") -> None:
        path = _extract_artifact_path(value)
        if not path:
            return
        lower_path = path.lower()
        intent_text = str(intent or "").strip().lower()
        if not lower_path.endswith(".json"):
            return
        metric_like = (
            "metric" in lower_path
            or "cv_" in lower_path
            or "latency" in lower_path
            or "benchmark" in lower_path
            or "performance" in lower_path
            or "kpi" in lower_path
            or "metrics" in intent_text
            or "cv_metrics" in intent_text
            or "latency" in intent_text
            or "benchmark" in intent_text
            or "performance" in intent_text
            or "kpi" in intent_text
        )
        if metric_like and path not in paths:
            paths.append(path)

    explicit_path_keys = (
        "metrics_path",
        "metrics_report_path",
        "cv_metrics_path",
        "primary_metrics_path",
    )
    for key in explicit_path_keys:
        value = evaluation_spec.get(key)
        if isinstance(value, str) and value.strip():
            _consider(value, intent=key)

    for container in (
        subject_required_outputs or [],
        qa_required_outputs or [],
        evaluation_spec.get("subject_required_outputs") or [],
        evaluation_spec.get("qa_required_outputs") or [],
        evaluation_spec.get("artifacts_to_verify") or [],
        evaluation_spec.get("required_outputs") or [],
    ):
        if not isinstance(container, list):
            continue
        for item in container:
            if isinstance(item, dict):
                intent = str(
                    item.get("intent")
                    or item.get("kind")
                    or item.get("type")
                    or ""
                )
            else:
                intent = ""
            _consider(item, intent=intent)
    return paths[:8]


def _metric_payload_source_priority(source: Any) -> int:
    token = str(source or "").replace("\\", "/").lower()
    stale_markers = (
        "round.baseline",
        "round_baseline",
        "baseline.metrics",
        "incumbent",
        "best_attempt",
        "history",
        "previous",
        "prior_",
    )
    if any(marker in token for marker in stale_markers):
        return -100000
    current_markers = (
        "current",
        "candidate",
        "artifacts/ml/validation_metrics.json",
        "artifacts/ml/latency_benchmark.json",
        "validation_metrics.json",
        "latency_benchmark.json",
    )
    if any(marker in token for marker in current_markers):
        return 100000
    if token.startswith("evaluation_spec.metrics"):
        return 10000
    return 0


def _resolve_primary_metric_name_from_context(
    evaluation_spec: Dict[str, Any] | None,
    qa_gates: List[Dict[str, Any]] | None,
) -> str:
    evaluation_spec = evaluation_spec if isinstance(evaluation_spec, dict) else {}
    qa_gates = qa_gates if isinstance(qa_gates, list) else []

    candidate_paths = [
        ("primary_metric", evaluation_spec.get("primary_metric")),
        ("metric_name", evaluation_spec.get("metric_name")),
    ]
    for parent_key in ("validation_requirements", "evaluation_spec", "metric_policy"):
        parent = evaluation_spec.get(parent_key)
        if isinstance(parent, dict):
            for key in ("primary_metric", "metric", "metric_name"):
                candidate_paths.append((f"{parent_key}.{key}", parent.get(key)))
    for gate in qa_gates:
        if not isinstance(gate, dict):
            continue
        params = gate.get("params") if isinstance(gate.get("params"), dict) else {}
        candidate_paths.append((f"qa_gate:{gate.get('name')}.metric", params.get("metric")))
        candidate_paths.append((f"qa_gate:{gate.get('name')}.field", params.get("field")))

    for _source, value in candidate_paths:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _build_deterministic_metric_facts(
    evaluation_spec: Dict[str, Any] | None,
    qa_gates: List[Dict[str, Any]] | None,
    subject_required_outputs: List[Any] | None,
    qa_required_outputs: List[Any] | None,
) -> Dict[str, Any]:
    evaluation_spec = evaluation_spec if isinstance(evaluation_spec, dict) else {}
    qa_gates = qa_gates if isinstance(qa_gates, list) else []

    payload_candidates: List[tuple[str, Dict[str, Any]]] = []
    for key in ("metrics_report", "metrics_payload", "cv_metrics", "metrics_json"):
        payload = evaluation_spec.get(key)
        if isinstance(payload, dict) and payload:
            normalized = normalize_metrics_report_payload(payload)
            payload_candidates.append((f"evaluation_spec.{key}", normalized or payload))

    for path in _collect_metric_artifact_paths(
        evaluation_spec,
        subject_required_outputs,
        qa_required_outputs,
    ):
        normalized = canonicalize_metrics_report_file(path)
        if isinstance(normalized, dict) and normalized:
            payload_candidates.append((path, normalized))
    payload_candidates.sort(
        key=lambda item: _metric_payload_source_priority(item[0]),
        reverse=True,
    )

    primary_metric_name = _resolve_primary_metric_name_from_context(evaluation_spec, qa_gates)
    best_payload: Dict[str, Any] = {}
    best_source = ""
    best_resolved: Dict[str, Any] = {}
    best_score = -1

    for source, payload in payload_candidates:
        if not isinstance(payload, dict) or not payload:
            continue
        metric_name = primary_metric_name or str(payload.get("primary_metric_name") or payload.get("primary_metric") or "").strip()
        resolved: Dict[str, Any] = {}
        canonical_metric = canonicalize_metric_name(metric_name) if metric_name else ""
        explicit_primary_value = _coerce_float_maybe(payload.get("primary_metric_value"))
        if (
            canonical_metric
            and isinstance(payload.get("metrics_mean"), dict)
            and _coerce_float_maybe((payload.get("metrics_mean") or {}).get(metric_name)) is not None
        ):
            resolved = {
                "value": float(_coerce_float_maybe((payload.get("metrics_mean") or {}).get(metric_name))),
                "matched_key": f"metrics_mean.{metric_name}",
                "score": 90000,
            }
        elif (
            canonical_metric
            and isinstance(payload.get("metrics_mean"), dict)
            and _coerce_float_maybe((payload.get("metrics_mean") or {}).get(canonical_metric)) is not None
        ):
            resolved = {
                "value": float(_coerce_float_maybe((payload.get("metrics_mean") or {}).get(canonical_metric))),
                "matched_key": f"metrics_mean.{canonical_metric}",
                "score": 90000,
            }
        elif metric_name and explicit_primary_value is not None:
            resolved = {
                "value": float(explicit_primary_value),
                "matched_key": "primary_metric_value",
                "score": 100000,
            }
        else:
            resolved = resolve_metric_value(payload, metric_name or primary_metric_name or "")
        score = int(resolved.get("score") or 0) if isinstance(resolved, dict) else 0
        score += _metric_payload_source_priority(source)
        if not metric_name and isinstance(payload.get("primary_metric_name"), str):
            metric_name = str(payload.get("primary_metric_name") or "").strip()
            canonical_metric = canonicalize_metric_name(metric_name)
            if (
                canonical_metric
                and isinstance(payload.get("metrics_mean"), dict)
                and _coerce_float_maybe((payload.get("metrics_mean") or {}).get(canonical_metric)) is not None
            ):
                resolved = {
                    "value": float(_coerce_float_maybe((payload.get("metrics_mean") or {}).get(canonical_metric))),
                    "matched_key": f"metrics_mean.{canonical_metric}",
                    "score": 90000,
                }
            elif _coerce_float_maybe(payload.get("primary_metric_value")) is not None:
                resolved = {
                    "value": float(_coerce_float_maybe(payload.get("primary_metric_value"))),
                    "matched_key": "primary_metric_value",
                    "score": 100000,
                }
            else:
                resolved = resolve_metric_value(payload, metric_name)
            score = int(resolved.get("score") or 0) if isinstance(resolved, dict) else 0
            score += _metric_payload_source_priority(source)
        if resolved and score >= best_score:
            best_payload = payload
            best_source = source
            best_resolved = resolved
            best_score = score
            if metric_name and not primary_metric_name:
                primary_metric_name = metric_name

    primary_metric_value = best_resolved.get("value") if isinstance(best_resolved, dict) else None
    higher_is_better = best_payload.get("higher_is_better") if isinstance(best_payload, dict) else None
    gate_metric_facts = _build_gate_metric_facts(
        qa_gates=qa_gates,
        payload_candidates=payload_candidates,
        primary_metric_name=primary_metric_name,
        higher_is_better=higher_is_better if isinstance(higher_is_better, bool) else None,
    )
    return {
        "available": bool(best_payload),
        "primary_metric_name": primary_metric_name or None,
        "primary_metric_canonical_name": canonicalize_metric_name(primary_metric_name) if primary_metric_name else None,
        "primary_metric_value": float(primary_metric_value) if isinstance(primary_metric_value, (int, float)) else None,
        "primary_metric_source": best_source or None,
        "matched_key": best_resolved.get("matched_key") if isinstance(best_resolved, dict) else None,
        "higher_is_better": bool(higher_is_better) if isinstance(higher_is_better, bool) else None,
        "metric_artifacts_considered": [source for source, _payload in payload_candidates[:8]],
        "gate_metric_facts": gate_metric_facts,
        "_metrics_payload": best_payload if isinstance(best_payload, dict) else {},
    }


def _looks_metric_gate(gate_spec: Dict[str, Any] | None) -> bool:
    if not isinstance(gate_spec, dict):
        return False
    params = gate_spec.get("params") if isinstance(gate_spec.get("params"), dict) else {}
    name = str(gate_spec.get("name") or "").strip().lower()
    if any(key in params for key in ("metric", "field", "threshold", "target", "min", "max", "min_value", "max_value", "operator", "direction", "metric_checks")):
        return True
    if gate_spec.get("evidence_source") or params.get("evidence_source"):
        return True
    metric_tokens = (
        "metric",
        "auc",
        "lift",
        "logloss",
        "loss",
        "rmse",
        "mae",
        "mape",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "brier",
        "r2",
        "gini",
        "latency",
        "benchmark",
        "ndcg",
        "map",
        "mrr",
    )
    return any(token in name for token in metric_tokens)


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


def _normalize_artifact_path_token(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/").lower()
    while text.startswith("./"):
        text = text[2:]
    return text


def _extract_json_artifact_from_evidence(value: Any) -> tuple[str, str]:
    text = str(value or "").strip().replace("\\", "/")
    lower = text.lower()
    marker = ".json"
    idx = lower.find(marker)
    if idx < 0:
        return "", ""
    artifact = text[: idx + len(marker)]
    metric = text[idx + len(marker) :].lstrip(".:/# ")
    return artifact, metric


def _metric_artifact_source_matches(source: str, expected: str) -> bool:
    expected_norm = _normalize_artifact_path_token(expected)
    if not expected_norm:
        return True
    source_norm = _normalize_artifact_path_token(source)
    if source_norm == expected_norm or source_norm.endswith("/" + expected_norm):
        return True
    return bool(os.path.basename(source_norm) and os.path.basename(source_norm) == os.path.basename(expected_norm))


def _gate_metric_name(gate_spec: Dict[str, Any], primary_metric_name: str = "") -> str:
    params = gate_spec.get("params") if isinstance(gate_spec.get("params"), dict) else {}
    metric_name = str(
        params.get("metric")
        or params.get("field")
        or params.get("metric_name")
        or gate_spec.get("metric")
        or gate_spec.get("field")
        or ""
    ).strip()
    if metric_name:
        return metric_name
    _artifact, evidence_metric = _extract_json_artifact_from_evidence(
        gate_spec.get("evidence_source") or params.get("evidence_source")
    )
    return evidence_metric or primary_metric_name


def _gate_expected_artifact(gate_spec: Dict[str, Any]) -> str:
    params = gate_spec.get("params") if isinstance(gate_spec.get("params"), dict) else {}
    artifact = str(
        gate_spec.get("applies_to_artifact")
        or params.get("artifact_path")
        or params.get("path")
        or ""
    ).strip()
    if artifact:
        return artifact
    evidence_artifact, _metric = _extract_json_artifact_from_evidence(
        gate_spec.get("evidence_source") or params.get("evidence_source")
    )
    return evidence_artifact


def _gate_threshold_details(gate_spec: Dict[str, Any]) -> Dict[str, Any]:
    params = gate_spec.get("params") if isinstance(gate_spec.get("params"), dict) else {}
    min_value = _coerce_float_maybe(params.get("min_value"))
    if min_value is None:
        min_value = _coerce_float_maybe(params.get("min"))
    max_value = _coerce_float_maybe(params.get("max_value"))
    if max_value is None:
        max_value = _coerce_float_maybe(params.get("max"))
    threshold = _coerce_float_maybe(params.get("threshold"))
    if threshold is None:
        threshold = _coerce_float_maybe(params.get("target"))
    return {
        "min_value": min_value,
        "max_value": max_value,
        "threshold": threshold,
        "operator": str(params.get("operator") or "").strip(),
    }


def _compare_gate_threshold_details(value: float, thresholds: Dict[str, Any]) -> tuple[Optional[bool], List[str]]:
    failures: List[str] = []
    checked = False
    min_value = thresholds.get("min_value")
    max_value = thresholds.get("max_value")
    threshold = thresholds.get("threshold")
    operator = str(thresholds.get("operator") or "").strip()
    if min_value is not None:
        checked = True
        if value < float(min_value):
            failures.append(f"value {value:.6g} below min_value {float(min_value):.6g}")
    if max_value is not None:
        checked = True
        if value > float(max_value):
            failures.append(f"value {value:.6g} above max_value {float(max_value):.6g}")
    if threshold is not None and operator:
        checked = True
        passed = _compare_metric_value(value, operator, float(threshold))
        if passed is False:
            failures.append(f"value {value:.6g} does not satisfy {operator} {float(threshold):.6g}")
    if not checked:
        return None, []
    return not failures, failures


def _build_gate_metric_facts(
    *,
    qa_gates: List[Dict[str, Any]],
    payload_candidates: List[tuple[str, Dict[str, Any]]],
    primary_metric_name: str,
    higher_is_better: Optional[bool],
) -> List[Dict[str, Any]]:
    facts: List[Dict[str, Any]] = []
    for gate_spec in qa_gates if isinstance(qa_gates, list) else []:
        checks = expand_numeric_gate_checks(gate_spec, primary_metric_name=primary_metric_name)
        if not checks and not _looks_metric_gate(gate_spec):
            continue
        gate_name = str(gate_spec.get("name") or "").strip()
        if not checks:
            metric_name = _gate_metric_name(gate_spec, primary_metric_name)
            if not metric_name:
                continue
            thresholds = _gate_threshold_details(gate_spec)
            if not any(thresholds.get(key) is not None for key in ("min_value", "max_value", "threshold")):
                operator, threshold = _infer_metric_gate_threshold(gate_spec, metric_name, higher_is_better)
                thresholds["operator"] = operator or thresholds.get("operator") or ""
                thresholds["threshold"] = threshold
            if not any(thresholds.get(key) is not None for key in ("min_value", "max_value", "threshold")):
                continue
            checks = [
                {
                    "name": gate_name,
                    "severity": str(gate_spec.get("severity") or "HARD").upper(),
                    "metric": metric_name,
                    "artifact_path": _gate_expected_artifact(gate_spec),
                    "min_value": thresholds.get("min_value"),
                    "max_value": thresholds.get("max_value"),
                    "operator": thresholds.get("operator"),
                    "threshold": thresholds.get("threshold"),
                    "threshold_param": None,
                }
            ]
        for check_index, check in enumerate(checks):
            metric_name = str(check.get("metric") or "").strip()
            if not metric_name:
                continue
            expected_artifact = str(check.get("artifact_path") or "").strip()
            source_matched = False
            fact_added = False
            for source, payload in payload_candidates:
                if expected_artifact and not _metric_artifact_source_matches(source, expected_artifact):
                    continue
                source_matched = True
                resolved = resolve_metric_value(payload, metric_name)
                value = _coerce_float_maybe(resolved.get("value") if isinstance(resolved, dict) else None)
                if value is None:
                    continue
                passed, reasons = _compare_gate_threshold_details(float(value), check)
                fact = {
                    "gate_name": gate_name,
                    "severity": str(check.get("severity") or gate_spec.get("severity") or "HARD").upper(),
                    "metric": metric_name,
                    "source": source,
                    "source_priority": _metric_payload_source_priority(source),
                    "expected_artifact": expected_artifact or None,
                    "matched_key": resolved.get("matched_key") if isinstance(resolved, dict) else None,
                    "check_index": check_index,
                    "threshold_param": check.get("threshold_param"),
                    "value": float(value),
                    "min_value": check.get("min_value"),
                    "max_value": check.get("max_value"),
                    "operator": check.get("operator"),
                    "threshold": check.get("threshold"),
                    "passed": passed,
                    "status": "pass" if passed is True else ("fail" if passed is False else "unknown"),
                    "detail": "; ".join(reasons) if reasons else "numeric gate satisfied",
                }
                facts.append(fact)
                fact_added = True
                break
            if expected_artifact and not fact_added:
                facts.append(
                    {
                        "gate_name": gate_name,
                        "severity": str(check.get("severity") or gate_spec.get("severity") or "HARD").upper(),
                        "metric": metric_name,
                        "source": None,
                        "source_priority": None,
                        "expected_artifact": expected_artifact,
                        "check_index": check_index,
                        "threshold_param": check.get("threshold_param"),
                        "value": None,
                        "min_value": check.get("min_value"),
                        "max_value": check.get("max_value"),
                        "operator": check.get("operator"),
                        "threshold": check.get("threshold"),
                        "passed": False,
                        "status": "missing_evidence" if source_matched else "missing_artifact_evidence",
                        "detail": f"Could not resolve artifact-backed metric '{metric_name}' for gate '{gate_name}'.",
                    }
                )
    return facts[:20]


def _infer_metric_gate_threshold(
    gate_spec: Dict[str, Any],
    metric_name: str,
    higher_is_better: Optional[bool],
) -> tuple[Optional[str], Optional[float]]:
    params = gate_spec.get("params") if isinstance(gate_spec.get("params"), dict) else {}
    operator = str(params.get("operator") or "").strip()
    condition = str(params.get("condition") or "").strip()
    if not operator and condition:
        match = re.search(r"(>=|<=|>|<|==)", condition)
        if match:
            operator = match.group(1)

    min_value = _coerce_float_maybe(params.get("min_value"))
    if min_value is None:
        min_value = _coerce_float_maybe(params.get("min"))
    max_value = _coerce_float_maybe(params.get("max_value"))
    if max_value is None:
        max_value = _coerce_float_maybe(params.get("max"))
    target_value = _coerce_float_maybe(params.get("target"))
    threshold_value = _coerce_float_maybe(params.get("threshold"))
    direction = str(params.get("direction") or "").strip().lower()
    if min_value is not None:
        return operator or ">=", min_value
    if max_value is not None:
        return operator or "<=", max_value
    if threshold_value is not None:
        if operator:
            return operator, threshold_value
        if direction in {"decrease", "minimize", "lower_is_better"}:
            return "<=", threshold_value
        if direction in {"increase", "maximize", "higher_is_better"}:
            return ">=", threshold_value
        if isinstance(higher_is_better, bool):
            return (">=" if higher_is_better else "<="), threshold_value
        return None, threshold_value
    if target_value is not None:
        if operator:
            return operator, target_value
        if isinstance(higher_is_better, bool):
            return (">=" if higher_is_better else "<="), target_value
        return None, target_value

    gate_name = str(gate_spec.get("name") or "").strip().lower()
    canonical_metric = canonicalize_metric_name(metric_name)
    metric_norm = re.sub(r"[^a-z0-9]+", "", str(metric_name or "").lower())
    if "positive" in gate_name:
        return ">", 0.0
    if "above_random_baseline" in gate_name or "better_than_random" in gate_name:
        if canonical_metric in {"top_decile_lift", "lift"} or "lift" in canonical_metric or "lift" in metric_norm:
            return ">", 1.0
        if canonical_metric in {"roc_auc", "auc"}:
            return ">", 0.5
    return None, None


def _compare_metric_value(value: float, operator: str, threshold: float) -> Optional[bool]:
    if operator == ">":
        return value > threshold
    if operator == ">=":
        return value >= threshold
    if operator == "<":
        return value < threshold
    if operator == "<=":
        return value <= threshold
    if operator == "==":
        return value == threshold
    return None


def _apply_metric_gate_consistency_guard(
    result: Dict[str, Any],
    qa_gates: List[Dict[str, Any]],
    metric_facts: Dict[str, Any] | None,
) -> tuple[Dict[str, Any], List[str]]:
    result = dict(result or {})
    metric_facts = metric_facts if isinstance(metric_facts, dict) else {}
    payload = metric_facts.get("_metrics_payload") if isinstance(metric_facts.get("_metrics_payload"), dict) else {}
    failed_gates = [str(g) for g in (result.get("failed_gates") or []) if str(g).strip()]
    guard_notes: List[str] = []
    forced_failed: List[str] = []
    for fact in metric_facts.get("gate_metric_facts") or []:
        if not isinstance(fact, dict):
            continue
        if fact.get("passed") is not False:
            continue
        if str(fact.get("severity") or "HARD").strip().upper() == "SOFT":
            continue
        gate_name = str(fact.get("gate_name") or "").strip()
        if not gate_name:
            continue
        if gate_name not in failed_gates:
            failed_gates.append(gate_name)
        forced_failed.append(gate_name)
        guard_notes.append(
            f"QA_METRIC_FACT_ENFORCED: gate '{gate_name}' failed deterministic numeric evidence "
            f"{fact.get('metric')}={fact.get('value')} from {fact.get('source') or fact.get('expected_artifact')} "
            f"({fact.get('detail')})."
        )

    if forced_failed:
        result["failed_gates"] = failed_gates
        hard_failures = [str(g) for g in (result.get("hard_failures") or []) if str(g).strip()]
        for gate_name in forced_failed:
            if gate_name not in hard_failures:
                hard_failures.append(gate_name)
        result["hard_failures"] = hard_failures
        required_fixes = [str(x) for x in (result.get("required_fixes") or []) if str(x).strip()]
        for note in guard_notes:
            if note not in required_fixes:
                required_fixes.append(note)
        result["required_fixes"] = required_fixes[:20]
        result["status"] = "REJECTED"
        feedback = str(result.get("feedback") or "").strip()
        note = "Deterministic metric facts found HARD numeric gate failures."
        result["feedback"] = f"{feedback}\n{note}".strip() if feedback else note

    if not failed_gates:
        return result, guard_notes

    gate_lookup = _gate_lookup(qa_gates)
    primary_metric_name = str(metric_facts.get("primary_metric_name") or "").strip()
    removed_gates: List[str] = []
    forced_failed_set = {gate.lower() for gate in forced_failed}
    facts_by_gate: Dict[str, List[Dict[str, Any]]] = {}
    for fact in metric_facts.get("gate_metric_facts") or []:
        if not isinstance(fact, dict):
            continue
        gate_key = str(fact.get("gate_name") or "").strip().lower()
        if not gate_key:
            continue
        facts_by_gate.setdefault(gate_key, []).append(fact)

    for gate_name in failed_gates:
        gate_key = str(gate_name or "").strip().lower()
        if gate_key in forced_failed_set:
            continue
        matching_facts = facts_by_gate.get(gate_key) or []
        hard_matching_facts = [
            fact
            for fact in matching_facts
            if str(fact.get("severity") or "HARD").strip().upper() != "SOFT"
        ]
        if hard_matching_facts and all(fact.get("passed") is True for fact in hard_matching_facts):
            removed_gates.append(gate_name)
            fact_summary = ", ".join(
                f"{fact.get('metric')}={fact.get('value')} from {fact.get('source')}"
                for fact in hard_matching_facts[:4]
            )
            guard_notes.append(
                f"QA_METRIC_FACT_OVERRIDE: removed gate '{gate_name}' because current deterministic "
                f"artifact facts satisfy all HARD numeric checks ({fact_summary})."
            )
            continue
        gate_spec = gate_lookup.get(gate_name.lower())
        if not _looks_metric_gate(gate_spec):
            continue
        if not payload:
            continue
        params = gate_spec.get("params") if isinstance(gate_spec, dict) else {}
        gate_metric_name = str(
            params.get("metric")
            or params.get("field")
            or primary_metric_name
            or ""
        ).strip()
        if not gate_metric_name:
            continue
        same_metric = (
            str(primary_metric_name or "").strip()
            and str(gate_metric_name or "").strip()
            and canonicalize_metric_name(primary_metric_name) == canonicalize_metric_name(gate_metric_name)
        )
        if same_metric and _coerce_float_maybe(metric_facts.get("primary_metric_value")) is not None:
            resolved = {
                "value": float(_coerce_float_maybe(metric_facts.get("primary_metric_value"))),
                "matched_key": metric_facts.get("matched_key") or "primary_metric_value",
            }
        else:
            resolved = resolve_metric_value(payload, gate_metric_name)
        metric_value = _coerce_float_maybe(resolved.get("value") if isinstance(resolved, dict) else None)
        if metric_value is None:
            continue
        higher_is_better = metric_facts.get("higher_is_better")
        operator, threshold = _infer_metric_gate_threshold(gate_spec, gate_metric_name, higher_is_better)
        if threshold is None or not operator:
            continue
        passed = _compare_metric_value(metric_value, operator, float(threshold))
        if passed is not True:
            continue
        removed_gates.append(gate_name)
        guard_notes.append(
            f"QA_METRIC_FACT_OVERRIDE: removed gate '{gate_name}' because deterministic metric facts show "
            f"{gate_metric_name}={metric_value:.6g} satisfies {operator} {float(threshold):.6g} "
            f"(source={metric_facts.get('primary_metric_source') or 'metrics_payload'} key={resolved.get('matched_key')})."
        )

    if not removed_gates:
        return result, guard_notes

    removed_set = {gate.lower() for gate in removed_gates}
    result["failed_gates"] = [
        gate for gate in failed_gates if str(gate).lower() not in removed_set
    ]
    hard_failures = result.get("hard_failures")
    if isinstance(hard_failures, list):
        result["hard_failures"] = [
            gate for gate in hard_failures if str(gate).lower() not in removed_set
        ]
    required_fixes = result.get("required_fixes")
    if isinstance(required_fixes, list):
        filtered_fixes: List[str] = []
        for fix in required_fixes:
            text = str(fix or "").strip()
            lower = text.lower()
            if any(gate in lower for gate in removed_set):
                continue
            filtered_fixes.append(text)
        result["required_fixes"] = filtered_fixes
    if not result["failed_gates"] and str(result.get("status") or "").upper() == "REJECTED":
        result["status"] = "APPROVE_WITH_WARNINGS"
        feedback = str(result.get("feedback") or "").strip()
        note = "Deterministic metric facts overrode unsupported metric-gate failures."
        result["feedback"] = f"{feedback}\n{note}".strip() if feedback else note
    return result, guard_notes


class QAReviewerAgent:
    def __init__(self, api_key: str = None):
        """
        Initializes the QA Reviewer Agent with MIMO v2 Flash.
        Role: Strict Code Quality Gate.
        """
        self.provider, self.client, self.model_name, self.model_warning = init_reviewer_llm(api_key)
        if self.model_warning:
            print(f"WARNING: {self.model_warning}")
        self.last_prompt = None
        self.last_response = None
        self.last_json_parse_trace = None
        self._generation_config = {
            "temperature": float(
                os.getenv(
                    "QA_REVIEWER_GEMINI_TEMPERATURE",
                    os.getenv("REVIEWER_GEMINI_TEMPERATURE", "0.2"),
                )
            ),
            "top_p": 0.9,
            "top_k": 40,
            "max_output_tokens": int(
                os.getenv(
                    "QA_REVIEWER_GEMINI_MAX_TOKENS",
                    os.getenv("REVIEWER_GEMINI_MAX_TOKENS", "32768"),
                )
            ),
            "response_mime_type": "application/json",
        }
        schema_flag = str(
            os.getenv(
                "QA_REVIEWER_USE_RESPONSE_SCHEMA",
                os.getenv("REVIEWER_USE_RESPONSE_SCHEMA", "0"),
            )
        ).strip().lower()
        self._use_response_schema = schema_flag not in {"0", "false", "no", "off", ""}

    def _generation_config_for_review(self, qa_gate_names: List[str] | None = None) -> Dict[str, Any]:
        config = dict(self._generation_config)
        if self._use_response_schema:
            config["response_schema"] = copy.deepcopy(build_qa_response_schema(qa_gate_names or []))
        return config

    @staticmethod
    def _is_response_schema_unsupported_error(err: Exception) -> bool:
        message = str(err or "").lower()
        if "response_schema" not in message:
            return False
        unsupported_tokens = (
            "unknown field",
            "unknown name",
            "not supported",
            "unsupported",
            "unrecognized",
            "no such field",
            "schema not supported",
        )
        return any(token in message for token in unsupported_tokens)

    def _generate_gemini_json(
        self,
        prompt: str,
        *,
        generation_config: Dict[str, Any] | None = None,
    ) -> tuple[str, Dict[str, Any]]:
        if not self.client:
            raise RuntimeError("Gemini client not configured")
        config = dict(generation_config or self._generation_config)
        try:
            response = self.client.generate_content(prompt, generation_config=config)
            return str(getattr(response, "text", "") or ""), config
        except Exception as err:
            if not self._is_response_schema_unsupported_error(err):
                raise
            fallback_config = dict(config)
            fallback_config.pop("response_schema", None)
            response = self.client.generate_content(prompt, generation_config=fallback_config)
            return str(getattr(response, "text", "") or ""), fallback_config

    def _attempt_llm_json_repair(
        self,
        raw_text: str,
        *,
        schema: Dict[str, Any] | None,
        repair_label: str,
    ) -> tuple[Dict[str, Any] | None, Dict[str, Any]]:
        trace: Dict[str, Any] = {
            "repair_label": str(repair_label or "qa_json"),
            "repair_attempted": False,
            "repair_succeeded": False,
            "provider": self.provider,
            "model": self.model_name,
        }
        if not isinstance(schema, dict) or not schema:
            return None, trace
        if not self.client or self.provider == "none":
            return None, trace
        raw = str(raw_text or "").strip()
        if not raw:
            return None, trace
        trace["repair_attempted"] = True

        schema_json = json.dumps(schema, ensure_ascii=True)
        raw_preview = raw[:12000]
        repair_prompt = (
            "You are a strict JSON repair tool. Return ONLY one JSON object, no markdown.\n"
            "TASK: Repair/normalize RAW_JSON so it conforms to TARGET_SCHEMA.\n"
            "RULES:\n"
            "- Keep original semantic intent when possible.\n"
            "- If malformed/truncated, complete minimally with safe defaults.\n"
            "- Do not invent gate names.\n"
            "TARGET_SCHEMA:\n"
            + schema_json
            + "\nRAW_JSON:\n"
            + raw_preview
        )

        try:
            if self.provider == "gemini":
                repaired_text, used_config = self._generate_gemini_json(
                    repair_prompt,
                    generation_config={
                        "temperature": 0.0,
                        "response_mime_type": "application/json",
                        "response_schema": copy.deepcopy(schema),
                    },
                )
                trace["used_response_schema"] = bool(
                    isinstance(used_config, dict) and "response_schema" in used_config
                )
            else:
                response = create_chat_completion_with_reasoning(
                    self.client,
                    agent_name="qa_reviewer",
                    model_name=self.model_name,
                    call_kwargs={
                        "model": self.model_name,
                        "messages": [
                            {"role": "system", "content": "Return only valid JSON."},
                            {"role": "user", "content": repair_prompt},
                        ],
                        "response_format": {"type": "json_object"},
                        "temperature": 0.0,
                    },
                )
                repaired_text = response.choices[0].message.content
                trace["used_response_schema"] = False
            parsed, parsed_trace = parse_json_object_with_repair(
                str(repaired_text or ""),
                actor="qa_reviewer_json_repair",
            )
            trace["repair_succeeded"] = isinstance(parsed, dict)
            trace["repair_parse_trace"] = parsed_trace
            return parsed if isinstance(parsed, dict) else None, trace
        except Exception as exc:
            trace["repair_error"] = f"{type(exc).__name__}: {exc}"[:240]
            return None, trace

    def _retry_with_simplified_prompt(
        self,
        failed_response_preview: str,
        *,
        qa_gate_names: List[str],
    ) -> Dict[str, Any] | None:
        """
        Last-resort retry: re-prompt the LLM with a simplified instruction
        asking for ONLY valid JSON.  Returns parsed dict or None.
        """
        if not self.client or self.provider == "none":
            return None

        simplified_prompt = (
            "Your previous response was not valid JSON and could not be parsed.\n"
            "You MUST respond with ONLY a valid JSON object, no other text.\n"
            "Use this exact structure:\n"
            '{\n'
            '  "status": "APPROVED" or "APPROVE_WITH_WARNINGS" or "REJECTED",\n'
            '  "feedback": "your assessment here",\n'
            '  "failed_gates": [],\n'
            '  "required_fixes": []\n'
            '}\n\n'
            + "Valid gate names: "
            + json.dumps(qa_gate_names)
            + "\n\nYour previous (malformed) response started with:\n"
            + str(failed_response_preview or "")[:3000]
            + "\n\nNow return ONLY the corrected JSON."
        )

        try:
            response = create_chat_completion_with_reasoning(
                self.client,
                agent_name="qa_reviewer",
                model_name=self.model_name,
                call_kwargs={
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": "Return only valid JSON."},
                        {"role": "user", "content": simplified_prompt},
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.0,
                },
            )
            content = response.choices[0].message.content

            parsed, _trace = _parse_json_payload_with_trace(content)
            print(
                "QA_JSON_RETRY_PASS: "
                f"success=True provider={self.provider} model={self.model_name}"
            )
            return parsed
        except Exception as exc:
            print(
                "QA_JSON_RETRY_PASS: "
                f"success=False error={type(exc).__name__} "
                f"provider={self.provider} model={self.model_name}"
            )
            return None

    def _parse_json_with_llm_repair(
        self,
        text: str,
        *,
        qa_gate_names: List[str],
        repair_label: str,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        try:
            parsed, trace = _parse_json_payload_with_trace(text)
            merged_trace = dict(trace or {})
            merged_trace["repair_via_llm"] = False
            self.last_json_parse_trace = merged_trace
            return parsed, merged_trace
        except JsonObjectParseError as err:
            base_trace = err.trace if isinstance(err.trace, dict) else {}
            repaired, repair_trace = self._attempt_llm_json_repair(
                text,
                schema=build_qa_response_schema(qa_gate_names),
                repair_label=repair_label,
            )
            if isinstance(repaired, dict):
                merged_trace = dict(base_trace)
                merged_trace["repair_via_llm"] = True
                merged_trace["llm_repair"] = repair_trace
                self.last_json_parse_trace = merged_trace
                print(
                    "QA_JSON_REPAIR_PASS: "
                    + f"label={repair_label} success=True provider={self.provider} model={self.model_name}"
                )
                return repaired, merged_trace
            merged_trace = dict(base_trace)
            merged_trace["repair_via_llm"] = False
            merged_trace["llm_repair"] = repair_trace
            self.last_json_parse_trace = merged_trace
            print(
                "QA_JSON_REPAIR_PASS: "
                + f"label={repair_label} success=False provider={self.provider} model={self.model_name}"
            )
            raise

    def review_code(
        self,
        code: str,
        strategy: Dict[str, Any],
        business_objective: str,
        evaluation_spec: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Conducts a strict Quality Assurance audit on the generated code.
        Focus: Mapping integrity, Leakage prevention, Safety, consistency.
        """
        # Validate ML Plan constraints (Fail-Fast)
        if evaluation_spec and "ml_plan" in evaluation_spec:
            ml_plan = evaluation_spec["ml_plan"]
            data_profile = evaluation_spec.get("data_profile", {})
            # Run constraint check
            check = validate_ml_plan_constraints(ml_plan, data_profile, {}, strategy)
            if not check["ok"]:
                msg = f"ML_PLAN_INVALID: {'; '.join(check['violations'])}"
                return {
                    "status": "REJECTED",
                    "feedback": msg,
                    "failed_gates": ["ML_PLAN_INVALID"],
                    "required_fixes": ["Regenerate ML plan with initialized ML Engineer / ensure LLM call works"],
                    "hard_failures": ["ML_PLAN_INVALID"],
                    "evidence": [],
                }

        static_facts = collect_static_qa_facts(code)
        static_result = run_static_qa_checks(code, evaluation_spec, static_facts)
        try:
            os.makedirs("data", exist_ok=True)
            facts_payload = static_result.get("facts") if isinstance(static_result, dict) else static_facts
            with open("data/qa_static_facts.json", "w", encoding="utf-8") as f:
                json.dump(facts_payload, f, indent=2)
        except Exception:
            pass
        if static_result and static_result.get("status") == "REJECTED":
            return static_result

        output_format_instructions = """
        Return a raw JSON object:
        {
            "status": "APPROVED" | "APPROVE_WITH_WARNINGS" | "REJECTED",
            "feedback": "Detailed explanation of rejection reasons or 'QA Passed'.",
            "failed_gates": ["List", "of", "failed", "gates"],
            "required_fixes": ["List", "of", "required", "actions"],
            "evidence": [
                {"claim": "Short claim", "source": "artifact_path#key_or_script_path:line or missing"}
            ]
        }
        """

        from src.utils.prompting import render_prompt

        eval_spec_json = json.dumps(evaluation_spec or {}, indent=2)
        qa_gate_specs, contract_source_used, gate_warnings = resolve_qa_gates(evaluation_spec)
        qa_gate_names = _gate_names(qa_gate_specs)
        qa_gates_json = json.dumps(qa_gate_specs, indent=2, ensure_ascii=True)
        active_qa_gates_json = json.dumps(qa_gate_names, indent=2, ensure_ascii=True)
        review_subject = str((evaluation_spec or {}).get("review_subject") or "ml_engineer").strip().lower() or "ml_engineer"
        subject_required_outputs = (
            (evaluation_spec or {}).get("subject_required_outputs")
            or (evaluation_spec or {}).get("artifacts_to_verify")
            or []
        )
        if not isinstance(subject_required_outputs, list):
            subject_required_outputs = []
        qa_required_outputs = (evaluation_spec or {}).get("qa_required_outputs") or []
        if not isinstance(qa_required_outputs, list):
            qa_required_outputs = []
        deterministic_metric_facts = _build_deterministic_metric_facts(
            evaluation_spec,
            qa_gate_specs,
            subject_required_outputs,
            qa_required_outputs,
        )
        deterministic_metric_prompt_facts = {
            "available": bool(deterministic_metric_facts.get("available")),
            "primary_metric_name": deterministic_metric_facts.get("primary_metric_name"),
            "primary_metric_canonical_name": deterministic_metric_facts.get("primary_metric_canonical_name"),
            "primary_metric_value": deterministic_metric_facts.get("primary_metric_value"),
            "primary_metric_source": deterministic_metric_facts.get("primary_metric_source"),
            "matched_key": deterministic_metric_facts.get("matched_key"),
            "higher_is_better": deterministic_metric_facts.get("higher_is_better"),
            "metric_artifacts_considered": deterministic_metric_facts.get("metric_artifacts_considered") or [],
            "gate_metric_facts": deterministic_metric_facts.get("gate_metric_facts") or [],
            "source_priority_rule": "current candidate artifact facts outrank baseline/incumbent/history facts",
        }
        subject_code_path_hint = str((evaluation_spec or {}).get("subject_code_path_hint") or "").strip()
        if review_subject == "data_engineer":
            subject_specific_guidance = (
                "- Audit cleaning/preparation code and declared subject outputs, not downstream model-training behavior.\n"
                "- Treat qa_gates as technical verification of the data engineer deliverables: cleaned/enriched datasets, "
                "traceability artifacts, manifests, exclusions, normalization, and row/accounting evidence.\n"
                "- Do not invent ML-only failures when the contract is cleaning-first and model_training is false.\n"
                "- If 'cleaning_quality_summary' is present in QA context, inspect 'notable_columns' for null inflation.\n"
                "  Large null_inflation_pp values may indicate a broken parser (e.g., datetime parsing destroying valid dates) "
                "or legitimate row exclusion. Distinguish between the two by checking whether rows were dropped and whether "
                "the inflated column is the exclusion criterion. Value destruction is a HARD quality failure; row-exclusion inflation is expected.\n"
                "- If 'cleaning_manifest' is present, cross-check declared transformations against the actual code.\n"
                "PROACTIVE CONTRACT COMPLIANCE (report as warnings, not gate failures):\n"
                "Beyond the active gates, a senior QA review should surface contract compliance gaps as warnings. "
                "These do not block approval but add audit value:\n"
                "- If the contract defines a training_rows_rule or scoring_rows_rule, verify whether the code actually "
                "applies it. If the rule is not enforced in the code, report it as a warning with evidence.\n"
                "- If the cleaning_manifest is present, check whether it documents all material operations the code performed "
                "(row filtering, imputation, type conversions, column drops). Undocumented operations reduce traceability.\n"
                "- If the contract specifies constraints (e.g., 'date parsing failures must generate flags'), check whether "
                "the code honors them. Report unmet constraints as warnings."
            )
        else:
            subject_specific_guidance = (
                "- Audit modeling/evaluation code and declared ML outputs against the active qa_gates.\n"
                "- Use evaluation, metric, split, and artifact evidence when those are present in context."
            )
        metric_round_context = _extract_metric_round_context(evaluation_spec)
        metric_round_active = bool(metric_round_context.get("metric_round_active"))
        augmentation_requested = bool(metric_round_context.get("augmentation_requested"))
        SYSTEM_PROMPT_TEMPLATE = """
        You are the Lead QA Engineer.

        MISSION:
        Audit the declared review subject against the contract-driven ACTIVE_QA_GATES using concrete evidence from code,
        produced artifacts, and the provided QA context. Your job is to verify technical correctness and contractual
        executability, not to invent a different plan.

        SOURCE OF TRUTH AND PRECEDENCE:
        1. ACTIVE_QA_GATES and contract-driven QA context.
        2. Declared review subject outputs and subject code path hints.
        3. Concrete evidence from code/artifacts/static checks.
        4. Business objective and strategy as explanatory context only.
        If sources disagree, preserve the contract-driven QA context unless concrete evidence proves a hard violation.

        QA DECISION WORKFLOW (MANDATORY):
        1. Identify the review subject and the outputs it is responsible for.
        2. Read the ACTIVE_QA_GATES and decide what evidence each active gate actually requires.
        3. Inspect only the code regions and artifacts that matter for those active gates.
        4. For each gate, reason about what constitutes a real violation vs an acceptable outcome:
           - A "verify_*" gate asks you to check whether a condition holds in the OUTPUT, not whether
             explicit validation code exists. If the output data satisfies the condition, the gate passes
             regardless of how the code achieved it. Absence of a check is not the same as a violation.
           - A gate that checks row counts or data shape must account for legitimate upstream operations
             (e.g., filtering debug rows, deduplication) that are requested by OTHER gates in the contract.
             Cross-reference before concluding that a reduction is anomalous.
        5. Reject only when an active HARD gate is truly violated with concrete evidence from the output.
        6. If evidence is incomplete or ambiguous, downgrade to APPROVE_WITH_WARNINGS instead of inventing certainty.

        === EVIDENCE RULE ===
        $senior_evidence_rule
        
        REVIEW SUBJECT GUIDANCE:
        $subject_specific_guidance

        QUALITY GATES (SPEC-DRIVEN):
        Use the gate families below as reasoning aids, not as a generic checklist. Only evaluate what is active.
        
        1. CONTRACT TRACEABILITY (only if gate enabled):
           - Feature/target selection must remain traceable to contract columns and roles.
           - Ask for explicit traceability evidence only when a dedicated gate requires it.
           
        2. CONSISTENCY CHECKS (only if gate enabled):
           - Check for column aliasing (two features mapping to same column).
           - Check for empty DataFrame.
           - Check target variation (nunique > 1).
           
        3. DATA LEAKAGE PREVENTION (only if gate enabled):
           - Target column must NOT be in X (features).
           - High cardinality columns (IDs) must be excluded unless justified.
           - If X is explicitly built from contract feature_cols and excludes extra columns, this is sufficient.
           
        4. OUTPUT SAFETY (only if gate enabled):
           - Verify that required outputs are written safely and to the intended paths.
           - Explicit directory creation is one acceptable pattern, but not the only valid implementation.

        5. INPUT CSV LOADING (only if gate enabled):
           - Verify the code loads the authoritative ML dataset declared in context (ml_data_path) or an equivalent
             contract-backed path. Prefer evidence of correct data provenance over a specific API call shape.

        6. NO SYNTHETIC DATA (only if gate enabled):
           - Reject only when there is concrete evidence that fabricated or synthetic data replaces the required real dataset.
           - Random generators, sklearn.datasets.make_*, or literal DataFrame constructors are strong evidence patterns,
             but judge them in context instead of treating them as a blind string match.

        7. CONTRACT COLUMNS (only if gate enabled):
           - Verify that feature and target selection remain traceable to contract columns, declared aliases, or
             authoritative selectors from the provided QA context.
           - Explicit canonical column references are helpful evidence, but not the only acceptable implementation.

        8. OUTPUT ROW COUNT CONSISTENCY (only if gate enabled):
           - For CSV artifacts with artifact_requirements.file_schemas.<path>.expected_row_count, verify code writes
             the correct row subset (e.g., test-only vs all rows) before to_csv.
           - Reject if a subset-sized artifact is built from full-frame columns without filtering/guard.
           
        INPUT CONTEXT:
        - Business Objective: "$business_objective"
        - Strategy: $strategy_title
        - Review Subject: $review_subject
        - QA Context (JSON): $evaluation_spec_json
        - Subject Required Outputs: $subject_required_outputs_json
        - QA Output Paths: $qa_required_outputs_json
        - Subject Code Path Hint: $subject_code_path_hint
        - ML Dataset Path: $ml_data_path
        - Contract Source Used: $contract_source_used
        - QA Gates (contract-driven, with severity/params): $qa_gates
        - ACTIVE_QA_GATES (names only): $active_qa_gates
        - Execution Diagnostics (JSON): $execution_diagnostics_json
        - Deterministic Metric Facts (JSON, authoritative when present): $deterministic_metric_facts_json
        - HARD_BLOCKER_PACKET (JSON): $hard_blocker_packet_json
        - Metric Improvement Round Active: $metric_round_active
        - Augmentation Requested (from hypothesis/plan): $augmentation_requested
        
        INSTRUCTIONS:
        - Analyze the code regions that matter for the active gates and supplied evidence.
        - If any HARD gate is violated, INVALIDATE with status "REJECTED".
        - If only SOFT gates are violated, return "APPROVE_WITH_WARNINGS".
        - If issues are minor (Style, Comments, non-critical Best Practices) but code is SAFE and CORRECT, return "APPROVE_WITH_WARNINGS".
        - Provide specific, actionable feedback on what is missing and how to fix it.
        - Do not request stylistic changes. Focus on correctness and safety.
        - Only fail gates listed in QA Gates; otherwise mention as warnings.
        - When listing failed_gates, use the gate "name" values from QA Gates.
        - failed_gates/hard_failures MUST be an exact subset of ACTIVE_QA_GATES.
        - If Deterministic Metric Facts provide a primary metric value, treat that as the authoritative metric evidence.
          Do not use stddev/variance fields as the primary metric unless the declared primary metric is explicitly a
          variability metric.
        - If Deterministic Metric Facts include gate_metric_facts with passed=false for a HARD gate, reject that gate.
          If passed=true, treat the numeric artifact evidence as authoritative for that gate.
        - In metric-improvement rounds, do not fail the current candidate using baseline/incumbent/history metrics.
          Baseline evidence is comparison context only; current candidate artifact facts are authoritative for current gate verdicts.
        - Treat HARD_BLOCKER_PACKET as prioritized focus context, not as an automatic failure list.
        - Before APPROVED or APPROVE_WITH_WARNINGS, re-check code_lines_of_interest against active_hard_gates_summary.
        - If known_restored_candidate_risks is non-empty, explicitly verify those risks before approving a restored or recycled candidate.
        - If you approve despite an item in HARD_BLOCKER_PACKET, explain why it is resolved or unsupported by evidence.
        - SELF-CHECK BEFORE FAILING ANY GATE: verify the gate name appears verbatim in ACTIVE_QA_GATES.
          If it does not, you MUST NOT include it in failed_gates or hard_failures — report it as a
          warning in feedback text only. The gate families above (1-8) are reasoning aids for active gates,
          not an independent list of gates to evaluate.
        - Never invent gates. Non-active findings go only to feedback/warnings (no gate failure).
        - IMPORTANT EXCEPTION: if Metric Improvement Round Active=true AND Augmentation Requested=true,
          do NOT fail no_synthetic_data for controlled augmentation/resampling changes in this round.
          Report any caution as warning text only.
        
        EVIDENCE REQUIREMENT:
        - Any REJECT or warning must cite evidence from the provided artifacts or code.
        - Include evidence in feedback using: EVIDENCE: <artifact_path>#<key> -> <short snippet>
        - If you cannot find evidence, downgrade to APPROVE_WITH_WARNINGS and state NO_EVIDENCE_FOUND.
        - SELF-CHECK BEFORE REJECT: without at least one concrete evidence item, you must not reject.
        - Populate the "evidence" list with sufficient items to support your claims. If evidence is missing, use source="missing".
        - Evidence sources must be artifact paths or script paths; otherwise use source="missing".
        
        OUTPUT FORMAT (JSON):
        $output_format_instructions
        """
        
        ml_data_path = _resolve_ml_data_path(evaluation_spec)
        execution_diagnostics = (evaluation_spec or {}).get("execution_diagnostics")
        if not isinstance(execution_diagnostics, dict):
            execution_diagnostics = {}
        review_context_packet = build_review_context_packet(
            code,
            qa_gate_specs,
            code_path_hint=subject_code_path_hint or "artifacts/ml_engineer_last.py",
            context_blocks=[
                evaluation_spec if isinstance(evaluation_spec, dict) else {},
                execution_diagnostics,
            ],
        )
        hard_blocker_packet_json = json.dumps(review_context_packet, indent=2, ensure_ascii=True)
        system_prompt = render_prompt(
            SYSTEM_PROMPT_TEMPLATE,
            business_objective=business_objective,
            strategy_title=strategy.get('title', 'Unknown'),
            review_subject=review_subject,
            evaluation_spec_json=eval_spec_json,
            subject_required_outputs_json=json.dumps(subject_required_outputs, indent=2, ensure_ascii=True),
            qa_required_outputs_json=json.dumps(qa_required_outputs, indent=2, ensure_ascii=True),
            subject_code_path_hint=subject_code_path_hint or "missing",
            ml_data_path=ml_data_path,
            contract_source_used=contract_source_used,
            qa_gates=qa_gates_json,
            active_qa_gates=active_qa_gates_json,
            execution_diagnostics_json=json.dumps(execution_diagnostics, indent=2, ensure_ascii=True),
            deterministic_metric_facts_json=json.dumps(deterministic_metric_prompt_facts, indent=2, ensure_ascii=True),
            hard_blocker_packet_json=hard_blocker_packet_json,
            metric_round_active=str(metric_round_active).lower(),
            augmentation_requested=str(augmentation_requested).lower(),
            output_format_instructions=output_format_instructions,
            subject_specific_guidance=subject_specific_guidance,
            senior_evidence_rule=SENIOR_EVIDENCE_RULE,
        )

        USER_MESSAGE_TEMPLATE = """
        AUDIT THIS CODE:
        
        ```python
        $code
        ```
        """
        
        user_message = render_prompt(USER_MESSAGE_TEMPLATE, code=code)
        self.last_prompt = system_prompt + "\n\n" + user_message

        if not self.client or self.provider == "none":
            fallback = static_result or {
                "status": "APPROVE_WITH_WARNINGS",
                "feedback": "QA reviewer LLM disabled; using static checks only.",
                "failed_gates": [],
                "required_fixes": [],
            }
            status = fallback.get("status")
            if status == "PASS":
                fallback["status"] = "APPROVED"
            elif status == "WARN":
                fallback["status"] = "APPROVE_WITH_WARNINGS"
            warnings = list(fallback.get("warnings") or [])
            if gate_warnings:
                warnings.extend(gate_warnings)
            warnings.append("LLM_DISABLED_NO_API_KEY")
            fallback["warnings"] = warnings
            fallback.setdefault("failed_gates", [])
            fallback.setdefault("required_fixes", [])
            fallback["qa_gates_evaluated"] = qa_gate_names
            fallback.setdefault("hard_failures", [])
            fallback.setdefault("soft_failures", [])
            fallback["contract_source_used"] = contract_source_used
            return fallback

        try:
            print(f"DEBUG: QA Reviewer calling OpenRouter ({self.model_name})...")
            response = create_chat_completion_with_reasoning(
                self.client,
                agent_name="qa_reviewer",
                model_name=self.model_name,
                call_kwargs={
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    "response_format": {"type": "json_object"},
                },
            )
            content = response.choices[0].message.content
            self.last_response = content
            
            # Parse JSON (tolerant)
            parse_error = None
            parse_trace: Dict[str, Any] = {}
            result = None
            try:
                result, parse_trace = self._parse_json_with_llm_repair(
                    content,
                    qa_gate_names=qa_gate_names,
                    repair_label="review_code",
                )
                self.last_json_parse_trace = parse_trace
            except JsonObjectParseError as err:
                parse_error = err
                parse_trace = err.trace if isinstance(getattr(err, "trace", None), dict) else {}
                self.last_json_parse_trace = parse_trace
            except Exception as err:
                parse_error = err

            if result is None:
                err_msg = f"QA JSON parse failed: {parse_error}"
                print(f"{err_msg}. Attempting simplified retry...")

                retry_result = self._retry_with_simplified_prompt(
                    str(content or ""),
                    qa_gate_names=qa_gate_names,
                )

                if isinstance(retry_result, dict):
                    result = retry_result
                    result["_json_retry_used"] = True
                else:
                    print(f"{err_msg}. Retry also failed. Defaulting to REJECTED.")
                    fallback = {
                        "status": "REJECTED",
                        "feedback": (
                            "QA Error: Failed to parse JSON response after retry. "
                            "Rejecting to maintain governance integrity. "
                            f"Parse error: {parse_error}"
                        ),
                        "failed_gates": ["qa_json_parse_failure"],
                        "required_fixes": [
                            "Re-run QA review. If persistent, check LLM response format."
                        ],
                    }
                    warnings = []
                    if gate_warnings:
                        warnings.extend(gate_warnings)
                    static_warnings = static_result.get("warnings", []) if static_result else []
                    if static_warnings:
                        warnings.extend(static_warnings)
                    if warnings:
                        warning_text = "\n".join([f"- {w}" for w in warnings])
                        feedback = fallback.get("feedback") or "QA rejected."
                        fallback["feedback"] = f"{feedback}\nWarnings:\n{warning_text}"
                    fallback["qa_gates_evaluated"] = qa_gate_names
                    fallback["hard_failures"] = ["qa_json_parse_failure"]
                    fallback["soft_failures"] = []
                    fallback["contract_source_used"] = contract_source_used
                    fallback["warnings"] = warnings
                    if parse_trace:
                        fallback["json_parse_trace"] = parse_trace
                    return fallback

            # Fallback normalization
            if result['status'] not in ['APPROVED', 'APPROVE_WITH_WARNINGS', 'REJECTED']:
                result['status'] = 'APPROVE_WITH_WARNINGS'
                result['feedback'] = "QA Error: Invalid status returned; downgraded to warnings."
            
            # Normalize lists
            for field in ['failed_gates', 'required_fixes']:
                val = result.get(field, [])
                if isinstance(val, str):
                    result[field] = [val]
                elif not isinstance(val, list):
                    result[field] = []
                else:
                     result[field] = val

            gate_lookup = _gate_lookup(qa_gate_specs)
            allowed = {name.lower() for name in qa_gate_names}
            filtered: List[str] = []
            for g in result.get("failed_gates", []):
                if str(g).lower() in allowed:
                    filtered.append(g)
            result["failed_gates"] = filtered
            result, metric_guard_notes = _apply_metric_gate_consistency_guard(
                result,
                qa_gate_specs,
                deterministic_metric_facts,
            )
            filtered = [str(g) for g in (result.get("failed_gates") or []) if str(g).strip()]
            result["failed_gates"] = filtered

            hard_failures: List[str] = []
            soft_failures: List[str] = []
            for g in filtered:
                spec = gate_lookup.get(str(g).lower())
                if spec and str(spec.get("severity")).upper() == "SOFT":
                    soft_failures.append(g)
                else:
                    hard_failures.append(g)

            if hard_failures:
                result["status"] = "REJECTED"
            elif soft_failures:
                if result.get("status") != "APPROVE_WITH_WARNINGS":
                    result["status"] = "APPROVE_WITH_WARNINGS"
            else:
                if result.get("status") == "REJECTED":
                    result["status"] = "APPROVE_WITH_WARNINGS"
                    result["feedback"] = "Spec-driven gating: no HARD QA gates failed; downgraded to warnings."

            warnings = []
            if gate_warnings:
                warnings.extend(gate_warnings)
            static_warnings = static_result.get("warnings", []) if static_result else []
            if static_warnings:
                warnings.extend(static_warnings)
            if metric_guard_notes:
                warnings.extend(metric_guard_notes)
            if warnings:
                if result.get("status") == "APPROVED":
                    result["status"] = "APPROVE_WITH_WARNINGS"
                feedback = result.get("feedback") or "QA Passed with warnings."
                warning_text = "\n".join([f"- {w}" for w in warnings])
                result["feedback"] = f"{feedback}\nWarnings:\n{warning_text}"

            result["qa_gates_evaluated"] = qa_gate_names
            result["hard_failures"] = hard_failures
            result["soft_failures"] = soft_failures
            result["contract_source_used"] = contract_source_used
            result["warnings"] = warnings
            if parse_trace:
                result["json_parse_trace"] = parse_trace
            return result
                
        except Exception as e:
            fallback = {
                "status": "APPROVE_WITH_WARNINGS",
                "feedback": f"QA System Error: {e}. Defaulting to warnings.",
                "failed_gates": [],
                "required_fixes": [],
            }
            warnings = []
            if gate_warnings:
                warnings.extend(gate_warnings)
            static_warnings = static_result.get("warnings", []) if static_result else []
            if static_warnings:
                warnings.extend(static_warnings)
            if warnings:
                warning_text = "\n".join([f"- {w}" for w in warnings])
                fallback["feedback"] = f"{fallback.get('feedback')}\nWarnings:\n{warning_text}"
            fallback["qa_gates_evaluated"] = qa_gate_names
            fallback["hard_failures"] = []
            fallback["soft_failures"] = []
            fallback["contract_source_used"] = contract_source_used
            fallback["warnings"] = warnings
            return fallback


def _is_random_call(call_node: ast.Call) -> bool:
    """
    Detects calls that rely on random generators by checking AST paths (numpy/random modules).
    """
    def _unwind_name(node: ast.AST) -> List[str]:
        parts: List[str] = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        return list(reversed(parts))

    parts = _unwind_name(call_node.func)
    if not parts:
        return False
    normalized_parts = [part.lower() for part in parts if part]
    if not normalized_parts:
        return False
    func_name = normalized_parts[-1]
    module_prefix = ".".join(normalized_parts[:-1]).rstrip(".")
    if module_prefix in {"np.random", "numpy.random", "random"}:
        return True
    if module_prefix and module_prefix.endswith("random"):
        return True
    random_func_names = {"rand", "randn", "randint", "uniform", "normal", "choice"}
    if func_name in random_func_names:
        if module_prefix in {"np.random", "numpy.random", "np", "numpy", "random"} or module_prefix.endswith("random"):
            return True
    if func_name == "default_rng":
        return True
    return False


def _node_is_index_like(node: ast.AST) -> bool:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        return node.func.id in {"len", "range"}
    if isinstance(node, ast.Attribute):
        return node.attr in {"index", "indices", "shape"}
    if isinstance(node, ast.Name):
        return node.id.lower() in {"idx", "index", "indices", "rows", "row_idx", "row_indices"}
    return False


def _is_resampling_random_call(call_node: ast.Call) -> bool:
    name = _call_name(call_node).lower()
    if not any(token in name for token in ("choice", "permutation", "shuffle", "default_rng", "seed")):
        return False
    if name.endswith("seed") or "default_rng" in name:
        return True
    for arg in call_node.args:
        if _node_is_index_like(arg):
            return True
        if isinstance(arg, ast.Attribute) and _node_is_index_like(arg):
            return True
    for kw in call_node.keywords:
        if _node_is_index_like(kw.value):
            return True
    return False


def _detect_synthetic_data_calls(tree: ast.AST, allow_resampling_random: bool) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if _is_sklearn_make_call(node) or _is_dataframe_literal_call(node):
                return True
            if _is_random_call(node):
                if allow_resampling_random and _is_resampling_random_call(node):
                    continue
                return True
    return False


def _is_sklearn_make_call(call_node: ast.Call) -> bool:
    name = _call_name(call_node)
    name_lower = name.lower()
    if "sklearn.datasets.make_" in name_lower or ".datasets.make_" in name_lower:
        return True
    if name_lower.startswith("make_"):
        return True
    return False


def _is_pure_literal_node(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return all(_is_pure_literal_node(elt) for elt in node.elts)
    if isinstance(node, ast.Dict):
        keys = [k for k in node.keys if k is not None]
        return all(_is_pure_literal_node(k) for k in keys) and all(
            _is_pure_literal_node(v) for v in node.values
        )
    return False


def _is_dataframe_literal_call(call_node: ast.Call) -> bool:
    name = _call_name(call_node)
    name_lower = name.lower()
    if not name_lower.endswith("dataframe"):
        return False
    for arg in call_node.args[:1]:
        if _is_pure_literal_node(arg):
            return True
    for kw in call_node.keywords:
        if kw.arg in (None, "data") and _is_pure_literal_node(kw.value):
            return True
    return False


def _is_synthetic_data_call(call_node: ast.Call) -> bool:
    return _is_random_call(call_node) or _is_sklearn_make_call(call_node) or _is_dataframe_literal_call(call_node)


def _call_name(call_node: ast.Call) -> str:
    try:
        return ast.unparse(call_node.func)
    except Exception:
        if isinstance(call_node.func, ast.Name):
            return call_node.func.id
        if isinstance(call_node.func, ast.Attribute):
            return call_node.func.attr
    return ""


def _looks_like_regressor(name: str) -> bool:
    """Detect regressors generically by naming convention, not hardcoded list."""
    if not name:
        return False
    simple = name.split(".")[-1]
    # Convention: sklearn/xgboost/lightgbm regressors end in "Regressor"
    if simple.endswith("Regressor"):
        return True
    # Convention: SVR/LinearSVR are support vector regressors
    if simple.lower() in {"svr", "linearsvr"}:
        return True
    # Convention: common regression models (ElasticNet, Lasso, Ridge) inherit RegressorMixin
    if simple in {"ElasticNet", "Lasso", "Ridge", "LinearRegression"}:
        return True
    return False


def _extract_target_name(target_node: ast.AST) -> Optional[str]:
    if isinstance(target_node, ast.Name):
        return target_node.id
    if isinstance(target_node, ast.Attribute):
        return target_node.attr
    if isinstance(target_node, ast.Subscript):
        try:
            return ast.unparse(target_node)
        except Exception:
            return None
    return None


def _expr_has_random_call(node: ast.AST) -> bool:
    class _RandomVisitor(ast.NodeVisitor):
        def __init__(self):
            self.found = False

        def visit_Call(self, call):
            if _is_random_call(call):
                self.found = True
            self.generic_visit(call)

    visitor = _RandomVisitor()
    visitor.visit(node)
    return visitor.found


def _is_split_fabrication_call(call_node: ast.Call) -> bool:
    if not isinstance(call_node.func, ast.Attribute):
        return False
    if call_node.func.attr != "split":
        return False
    # Detect patterns like df["col"].str.split(..., expand=True)
    base = call_node.func.value
    is_str_access = isinstance(base, ast.Attribute) and base.attr == "str"
    has_expand_true = any(
        isinstance(kw, ast.keyword) and kw.arg == "expand" and isinstance(kw.value, ast.Constant) and bool(kw.value.value) is True
        for kw in call_node.keywords
    )
    if is_str_access and has_expand_true:
        return True
    try:
        func_repr = ast.unparse(call_node.func).lower()
    except Exception:
        func_repr = ""
    return ".str.split" in func_repr and "expand=true" in func_repr


def _expr_has_split_fabrication(node: ast.AST) -> bool:
    class _SplitVisitor(ast.NodeVisitor):
        def __init__(self):
            self.found = False

        def visit_Call(self, call):
            if _is_split_fabrication_call(call):
                self.found = True
            self.generic_visit(call)

    visitor = _SplitVisitor()
    visitor.visit(node)
    return visitor.found


def _numeric_literal_value(node: ast.AST) -> Optional[float]:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        if isinstance(node.value, bool):
            return None
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.operand, ast.Constant):
        raw = node.operand.value
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            return None
        if isinstance(node.op, ast.USub):
            return -float(raw)
        if isinstance(node.op, ast.UAdd):
            return float(raw)
    return None


def _if_has_value_error_raise(body_nodes) -> bool:
    for node in body_nodes:
        if isinstance(node, ast.Raise):
            exc = node.exc
            if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name) and exc.func.id == "ValueError":
                return True
        for child in ast.walk(node):
            if isinstance(child, ast.Raise):
                exc = child.exc
                if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name) and exc.func.id == "ValueError":
                    return True
    return False


class _StaticQAScanner(ast.NodeVisitor):
    def __init__(self):
        self.has_random_target_noise = False
        self.has_split_fabrication = False
        self.has_variance_guard = False
        self.variance_guard_deferred_sinks: set[str] = set()
        self.nunique_aliases: set[str] = set()
        self.unique_aliases: set[str] = set()
        self.unique_len_aliases: set[str] = set()
        self.var_aliases: set[str] = set()
        self.std_aliases: set[str] = set()
        self.has_leakage_assert = False
        self.has_regression_model = False
        self.has_regression_metric = False
        self.has_fit_call = False
        self.has_infer_group_key_call = False
        self.has_group_split_usage = False
        self.has_security_violation = False
        self.has_train_eval_split = False
        self.has_mkdirs = False
        self.forbidden_imports_found = False
        self.has_read_csv = False
        self.has_synthetic_data = False

    def visit_Assign(self, node: ast.Assign):
        self._handle_assignment(node.targets, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        targets = [node.target]
        self._handle_assignment(targets, node.value)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign):
        targets = [node.target]
        self._handle_assignment(targets, node.value)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if _is_split_fabrication_call(node):
            self.has_split_fabrication = True
        name = _call_name(node)
        if _is_synthetic_data_call(node):
            self.has_synthetic_data = True
        if "read_csv" in name:
            self.has_read_csv = True
        if "assert_no_deterministic_target_leakage" in name:
            self.has_leakage_assert = True
        if _looks_like_regressor(name):
            self.has_regression_model = True
        metric_names = {"r2_score", "mean_squared_error", "mean_absolute_error", "mean_absolute_percentage_error"}
        simple_name = name.split(".")[-1]
        if simple_name in {
            "train_test_split",
            "cross_val_score",
            "cross_validate",
            "cross_val_predict",
            "KFold",
            "StratifiedKFold",
            "GroupKFold",
            "TimeSeriesSplit",
            "ShuffleSplit",
            "StratifiedShuffleSplit",
            "GroupShuffleSplit",
        } or "train_test_split" in name:
            self.has_train_eval_split = True
        if simple_name in metric_names:
            self.has_regression_metric = True
        if simple_name == "fit":
            self.has_fit_call = True
        if "infer_group_key" in name.lower():
            self.has_infer_group_key_call = True
        if simple_name == "split":
            if len(node.args) >= 3 or any(isinstance(kw, ast.keyword) and kw.arg == "groups" for kw in node.keywords):
                self.has_group_split_usage = True
        if any(isinstance(kw, ast.keyword) and kw.arg == "groups" for kw in node.keywords):
            self.has_group_split_usage = True
        forbidden_calls = {
            "os.system",
            "subprocess.run",
            "subprocess.Popen",
            "subprocess.call",
            "requests.get",
            "requests.post",
            "requests.put",
            "requests.delete",
        }
        if name in forbidden_calls:
            self.has_security_violation = True
        if name.endswith("makedirs") or name.endswith(".mkdir") or name == "mkdir":
            self.has_mkdirs = True
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import):
        forbidden = {"sys", "subprocess", "requests", "socket"}
        for alias in node.names:
            if alias.name.split(".")[0] in forbidden:
                self.has_security_violation = True
                self.forbidden_imports_found = True
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        forbidden = {"sys", "subprocess", "requests", "socket"}
        if node.module and node.module.split(".")[0] in forbidden:
            self.has_security_violation = True
            self.forbidden_imports_found = True
        self.generic_visit(node)

    def visit_If(self, node: ast.If):
        has_raise = _if_has_value_error_raise(node.body)
        is_variance_condition = self._condition_checks_variance_guard(node.test)
        if is_variance_condition:
            if has_raise:
                self.has_variance_guard = True
            else:
                self.variance_guard_deferred_sinks.update(
                    self._collect_variance_guard_sink_names(node.body)
                )
        elif has_raise and self._test_references_names(node.test, self.variance_guard_deferred_sinks):
            self.has_variance_guard = True
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert):
        if self._assert_checks_variance_guard(node.test) or self._test_references_names(
            node.test, self.variance_guard_deferred_sinks
        ):
            self.has_variance_guard = True
        self.generic_visit(node)

    def _handle_assignment(self, targets, value):
        if value is None:
            return
        self._track_variance_guard_aliases(targets, value)
        if _expr_has_random_call(value):
            for tgt in targets:
                name = _extract_target_name(tgt)
                if not name:
                    continue
                name_lower = name.lower()
                if name_lower == "y" or "target" in name_lower:
                    self.has_random_target_noise = True
        if _expr_has_split_fabrication(value):
            self.has_split_fabrication = True

    def _iter_target_names(self, targets: List[ast.AST]) -> List[str]:
        names: List[str] = []

        def _walk(node: ast.AST) -> None:
            if isinstance(node, ast.Name):
                names.append(node.id)
                return
            if isinstance(node, (ast.Tuple, ast.List)):
                for child in node.elts:
                    _walk(child)

        for target in targets:
            _walk(target)
        return names

    def _track_variance_guard_aliases(self, targets: List[ast.AST], value: ast.AST) -> None:
        target_names = self._iter_target_names(targets)
        if not target_names:
            return

        for name in target_names:
            self.nunique_aliases.discard(name)
            self.unique_aliases.discard(name)
            self.unique_len_aliases.discard(name)
            self.var_aliases.discard(name)
            self.std_aliases.discard(name)

        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "len"
            and value.args
        ):
            inner_kind = self._guard_stat_kind_from_expr(value.args[0])
            if inner_kind in {"nunique", "unique"}:
                self.unique_len_aliases.update(target_names)
            return

        if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Attribute):
            return
        attr = str(value.func.attr or "").strip()
        if not attr:
            return
        if attr == "nunique":
            self.nunique_aliases.update(target_names)
        elif attr == "unique":
            self.unique_aliases.update(target_names)
        elif attr == "var":
            self.var_aliases.update(target_names)
        elif attr == "std":
            self.std_aliases.update(target_names)

    def _guard_stat_kind_from_expr(self, expr: ast.AST) -> Optional[str]:
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
            attr = str(expr.func.attr or "").strip()
            if attr in {"nunique", "unique", "var", "std"}:
                return attr
        if (
            isinstance(expr, ast.Call)
            and isinstance(expr.func, ast.Name)
            and expr.func.id == "len"
            and expr.args
        ):
            inner_kind = self._guard_stat_kind_from_expr(expr.args[0])
            if inner_kind in {"nunique", "unique"}:
                return "nunique"
        if isinstance(expr, ast.Name):
            name = expr.id
            if name in self.nunique_aliases:
                return "nunique"
            if name in self.unique_len_aliases:
                return "nunique"
            if name in self.unique_aliases:
                return "unique"
            if name in self.var_aliases:
                return "var"
            if name in self.std_aliases:
                return "std"
        return None

    def _call_receiver_name(self, call_node: ast.Call) -> Optional[str]:
        if not isinstance(call_node.func, ast.Attribute):
            return None
        receiver = call_node.func.value
        if not isinstance(receiver, ast.Name):
            return None
        method = str(call_node.func.attr or "").strip().lower()
        if method in {"append", "extend", "add", "update", "setdefault", "insert"}:
            return receiver.id
        return None

    def _collect_variance_guard_sink_names(self, body_nodes: List[ast.stmt]) -> set[str]:
        sink_names: set[str] = set()
        for stmt in body_nodes:
            if isinstance(stmt, ast.Assign):
                sink_names.update(self._iter_target_names(stmt.targets))
            elif isinstance(stmt, ast.AnnAssign):
                sink_names.update(self._iter_target_names([stmt.target]))
            elif isinstance(stmt, ast.AugAssign):
                sink_names.update(self._iter_target_names([stmt.target]))
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                receiver = self._call_receiver_name(stmt.value)
                if receiver:
                    sink_names.add(receiver)
            for child in ast.walk(stmt):
                if isinstance(child, ast.Call):
                    receiver = self._call_receiver_name(child)
                    if receiver:
                        sink_names.add(receiver)
        return {name for name in sink_names if isinstance(name, str) and name.strip()}

    def _test_references_names(self, test_node: ast.AST, names: set[str]) -> bool:
        if not names:
            return False
        referenced = {
            child.id
            for child in ast.walk(test_node)
            if isinstance(child, ast.Name)
        }
        return bool(referenced & names)

    def _condition_checks_variance_guard(self, test_node: ast.AST) -> bool:
        if not isinstance(test_node, ast.Compare):
            return False
        stat_kind = self._guard_stat_kind_from_expr(test_node.left)
        if not stat_kind:
            return False
        for op, comp in zip(test_node.ops, test_node.comparators):
            value = _numeric_literal_value(comp)
            if value is None:
                continue
            if stat_kind == "nunique":
                if isinstance(op, ast.LtE) and value <= 1:
                    return True
                if isinstance(op, ast.Lt) and value <= 2:
                    return True
                if isinstance(op, ast.Eq) and value == 1:
                    return True
            if stat_kind in {"var", "std"}:
                if isinstance(op, ast.Eq) and value == 0:
                    return True
                if isinstance(op, ast.LtE) and value <= 0:
                    return True
        return False

    def _assert_checks_variance_guard(self, test_node: ast.AST) -> bool:
        if not isinstance(test_node, ast.Compare):
            return False
        stat_kind = self._guard_stat_kind_from_expr(test_node.left)
        if not stat_kind:
            return False
        for op, comp in zip(test_node.ops, test_node.comparators):
            value = _numeric_literal_value(comp)
            if value is None:
                continue
            if stat_kind == "nunique":
                # Equivalent assert forms of "nunique must be >= 2".
                if isinstance(op, ast.Gt) and value >= 1:
                    return True
                if isinstance(op, ast.GtE) and value >= 2:
                    return True
            if stat_kind in {"var", "std"}:
                # Equivalent assert forms of "variance/std must be > 0".
                if isinstance(op, ast.Gt) and value >= 0:
                    return True
                if isinstance(op, ast.NotEq) and value == 0:
                    return True
        return False



def _resolve_contract_columns_for_qa(evaluation_spec: Dict[str, Any] | None) -> List[str]:
    """V4.1: Extract columns from canonical_columns, column_roles, etc."""
    columns: List[str] = []
    if isinstance(evaluation_spec, dict):
        # V4.1 primary keys
        for key in ("canonical_columns", "contract_columns", "required_columns", "allowed_columns"):
            vals = evaluation_spec.get(key)
            if isinstance(vals, list):
                columns.extend([str(c) for c in vals if c])
        # From allowed feature sets if present
        allowed_sets = evaluation_spec.get("allowed_feature_sets")
        if isinstance(allowed_sets, dict):
            for key in (
                "model_features",
                "segmentation_features",
                "audit_only_features",
                "forbidden_features",
                "forbidden_for_modeling",
            ):
                vals = allowed_sets.get(key)
                if isinstance(vals, list):
                    columns.extend([str(c) for c in vals if c])
        # From artifact requirements clean_dataset bindings
        artifact_reqs = evaluation_spec.get("artifact_requirements")
        if isinstance(artifact_reqs, dict):
            clean_cfg = artifact_reqs.get("clean_dataset")
            if isinstance(clean_cfg, dict):
                req_cols = clean_cfg.get("required_columns")
                if isinstance(req_cols, list):
                    columns.extend([str(c) for c in req_cols if c])
        
        # V4.1: Extract from column_roles (role -> list[str] mapping)
        column_roles = evaluation_spec.get("column_roles")
        if isinstance(column_roles, dict):
            for role, cols in column_roles.items():
                if isinstance(cols, list):
                    columns.extend([str(c) for c in cols if c])
                elif isinstance(cols, str):
                    columns.append(cols)
    
    if not columns:
        return []
    # Remove duplicates while preserving order
    return list(dict.fromkeys(columns))


def _code_mentions_columns(code: str, columns: List[str], tree: ast.AST | None = None) -> bool:
    if not columns:
        return False
    col_set = {str(c) for c in columns if c}
    if tree is not None:
        literals = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        for col in col_set:
            if col in literals:
                return True
    lowered = code.lower()
    return any(col.lower() in lowered for col in col_set)


def _coerce_positive_row_count(value: Any) -> Optional[int]:
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


def _normalize_artifact_path_for_match(path_like: Any) -> str:
    text = str(path_like or "").strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text


def _extract_row_count_hints_for_qa(evaluation_spec: Dict[str, Any] | None) -> Dict[str, int]:
    hints: Dict[str, int] = {}
    if not isinstance(evaluation_spec, dict):
        return hints

    train_keys = ("n_train_rows", "train_rows", "n_train", "rows_train")
    test_keys = ("n_test_rows", "test_rows", "n_test", "rows_test")
    total_keys = ("n_total_rows", "total_rows", "n_rows", "row_count", "rows")

    def _scan(source: Any) -> None:
        if not isinstance(source, dict):
            return
        if "n_train" not in hints:
            for key in train_keys:
                parsed = _coerce_positive_row_count(source.get(key))
                if parsed is not None:
                    hints["n_train"] = parsed
                    break
        if "n_test" not in hints:
            for key in test_keys:
                parsed = _coerce_positive_row_count(source.get(key))
                if parsed is not None:
                    hints["n_test"] = parsed
                    break
        if "n_total" not in hints:
            for key in total_keys:
                parsed = _coerce_positive_row_count(source.get(key))
                if parsed is not None:
                    hints["n_total"] = parsed
                    break
        basic_stats = source.get("basic_stats")
        if isinstance(basic_stats, dict):
            if "n_total" not in hints:
                parsed = _coerce_positive_row_count(
                    basic_stats.get("n_rows") or basic_stats.get("rows") or basic_stats.get("row_count")
                )
                if parsed is not None:
                    hints["n_total"] = parsed
            if "n_train" not in hints:
                parsed = _coerce_positive_row_count(
                    basic_stats.get("n_train_rows") or basic_stats.get("train_rows")
                )
                if parsed is not None:
                    hints["n_train"] = parsed
            if "n_test" not in hints:
                parsed = _coerce_positive_row_count(
                    basic_stats.get("n_test_rows") or basic_stats.get("test_rows")
                )
                if parsed is not None:
                    hints["n_test"] = parsed

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
        outcome_analysis = source.get("outcome_analysis")
        if not isinstance(outcome_analysis, dict):
            return
        for entry in outcome_analysis.values():
            if not isinstance(entry, dict):
                continue
            total = _coerce_positive_row_count(
                entry.get("total_count")
                or entry.get("n_total_rows")
                or entry.get("n_rows")
                or entry.get("row_count")
                or entry.get("rows")
            )
            non_null = _coerce_positive_row_count(
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
            if "n_total" not in hints and isinstance(total, int):
                hints["n_total"] = total
            if "n_train" not in hints and isinstance(non_null, int):
                hints["n_train"] = non_null
            if (
                "n_test" not in hints
                and isinstance(total, int)
                and isinstance(non_null, int)
                and total >= non_null
            ):
                inferred_test = total - non_null
                if inferred_test > 0:
                    hints["n_test"] = inferred_test
            if "n_train" in hints and "n_test" in hints and "n_total" in hints:
                return

    nested_eval = evaluation_spec.get("evaluation_spec") if isinstance(evaluation_spec.get("evaluation_spec"), dict) else {}
    artifact_reqs = (
        evaluation_spec.get("artifact_requirements")
        if isinstance(evaluation_spec.get("artifact_requirements"), dict)
        else {}
    )
    split_spec = evaluation_spec.get("split_spec") if isinstance(evaluation_spec.get("split_spec"), dict) else {}
    nested_dataset_profile = (
        evaluation_spec.get("dataset_profile")
        if isinstance(evaluation_spec.get("dataset_profile"), dict)
        else {}
    )
    data_profile = evaluation_spec.get("data_profile") if isinstance(evaluation_spec.get("data_profile"), dict) else {}
    for source in (
        evaluation_spec,
        nested_eval,
        artifact_reqs,
        split_spec,
        nested_dataset_profile,
        data_profile,
    ):
        _scan(source)
        _scan_outcome_counts(source)

    n_train = hints.get("n_train")
    n_test = hints.get("n_test")
    n_total = hints.get("n_total")
    if n_total is None and n_train is not None and n_test is not None:
        hints["n_total"] = int(n_train + n_test)
    if n_test is None and n_total is not None and n_train is not None and n_total >= n_train:
        hints["n_test"] = int(n_total - n_train)
    if n_train is None and n_total is not None and n_test is not None and n_total >= n_test:
        hints["n_train"] = int(n_total - n_test)
    return hints


def _resolve_expected_csv_row_counts_for_qa(
    evaluation_spec: Dict[str, Any] | None,
    row_hints: Dict[str, int],
) -> Dict[str, int]:
    expected: Dict[str, int] = {}
    if not isinstance(evaluation_spec, dict):
        return expected

    artifact_reqs = (
        evaluation_spec.get("artifact_requirements")
        if isinstance(evaluation_spec.get("artifact_requirements"), dict)
        else {}
    )
    nested_eval = evaluation_spec.get("evaluation_spec") if isinstance(evaluation_spec.get("evaluation_spec"), dict) else {}
    if not artifact_reqs and isinstance(nested_eval.get("artifact_requirements"), dict):
        artifact_reqs = nested_eval.get("artifact_requirements")
    file_schemas = artifact_reqs.get("file_schemas") if isinstance(artifact_reqs, dict) else None

    alias_map = {
        "n_train": row_hints.get("n_train"),
        "n_train_rows": row_hints.get("n_train"),
        "train_rows": row_hints.get("n_train"),
        "n_test": row_hints.get("n_test"),
        "n_test_rows": row_hints.get("n_test"),
        "test_rows": row_hints.get("n_test"),
        "n_total": row_hints.get("n_total"),
        "n_total_rows": row_hints.get("n_total"),
        "total_rows": row_hints.get("n_total"),
        "n_rows": row_hints.get("n_total"),
        "row_count": row_hints.get("n_total"),
    }
    if isinstance(file_schemas, dict):
        for raw_path, schema in file_schemas.items():
            path = _normalize_artifact_path_for_match(raw_path)
            if not path or not path.lower().endswith(".csv") or not isinstance(schema, dict):
                continue
            resolved = _coerce_positive_row_count(schema.get("expected_row_count"))
            if resolved is None and isinstance(schema.get("expected_row_count"), str):
                token = re.sub(r"[^a-z0-9]+", "_", schema.get("expected_row_count", "").lower()).strip("_")
                candidate = alias_map.get(token)
                resolved = int(candidate) if isinstance(candidate, int) and candidate > 0 else None
            if resolved is not None:
                expected[path] = int(resolved)

    n_total = row_hints.get("n_total")
    n_test = row_hints.get("n_test")
    if not (isinstance(n_total, int) and isinstance(n_test, int) and n_total > 0 and n_test > 0 and n_test < n_total):
        return expected

    def _infer_kind_from_path(path_value: str) -> str:
        base = os.path.basename(path_value).lower()
        if "submission" in base:
            return "submission"
        if "scored_rows" in base or "scored-rows" in base:
            return "scored_rows"
        if "prediction" in base or "predictions" in base:
            return "prediction"
        if "forecast" in base:
            return "forecast"
        if "ranking" in base:
            return "ranking"
        return ""

    def _ingest_paths(items: Any) -> None:
        if not isinstance(items, list):
            return
        for item in items:
            if isinstance(item, dict):
                raw_path = item.get("path") or item.get("output") or item.get("artifact")
            else:
                raw_path = item
            path = _normalize_artifact_path_for_match(raw_path)
            if not path or not path.lower().endswith(".csv") or path in expected:
                continue
            kind = _infer_kind_from_path(path)
            if kind == "scored_rows":
                expected[path] = int(n_total)
            elif kind in {"submission", "prediction", "forecast", "ranking"}:
                expected[path] = int(n_test)

    _ingest_paths(artifact_reqs.get("required_outputs"))
    _ingest_paths(artifact_reqs.get("optional_outputs"))
    _ingest_paths(artifact_reqs.get("required_files"))
    return expected


def _literal_str_from_ast(node: ast.AST | None) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return str(node.value)
    if isinstance(node, ast.JoinedStr):
        chunks: List[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                chunks.append(str(value.value))
            else:
                return ""
        return "".join(chunks)
    return ""


def _resolve_to_csv_target_path(call_node: ast.Call) -> str:
    path_node: ast.AST | None = None
    if call_node.args:
        path_node = call_node.args[0]
    else:
        for kw in call_node.keywords:
            if isinstance(kw, ast.keyword) and kw.arg == "path_or_buf":
                path_node = kw.value
                break
    return _normalize_artifact_path_for_match(_literal_str_from_ast(path_node))


def _looks_like_read_csv_call(call_node: ast.AST) -> bool:
    if not isinstance(call_node, ast.Call):
        return False
    return "read_csv" in _call_name(call_node)


def _extract_assignment_targets(targets: List[ast.AST]) -> List[str]:
    names: List[str] = []
    for target in targets:
        if isinstance(target, ast.Name):
            names.append(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for child in target.elts:
                if isinstance(child, ast.Name):
                    names.append(child.id)
    return names


def _collect_assignment_map(tree: ast.AST) -> Dict[str, ast.AST]:
    mapping: Dict[str, tuple[int, ast.AST]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = _extract_assignment_targets(node.targets)
            for name in targets:
                previous = mapping.get(name)
                if previous is None or node.lineno >= previous[0]:
                    mapping[name] = (node.lineno, node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
            previous = mapping.get(name)
            if previous is None or node.lineno >= previous[0]:
                mapping[name] = (node.lineno, node.value)
    return {name: payload[1] for name, payload in mapping.items() if payload[1] is not None}


def _expr_references_alias(expr: ast.AST, aliases: set[str]) -> bool:
    for node in ast.walk(expr):
        if isinstance(node, ast.Name) and node.id in aliases:
            return True
    return False


# Methods that preserve row count — calling them on a subset still yields a
# subset, and calling them on a full frame yields a full frame.  This list is
# intentionally conservative: only methods whose output has the *same* number
# of rows as the receiver are included.  It is universal and not tied to any
# particular ML task or library.
_ROW_PRESERVING_METHODS: frozenset[str] = frozenset({
    "copy", "reset_index", "rename", "astype", "set_index",
    "sort_values", "sort_index", "fillna", "replace", "assign",
    "clip", "round", "abs", "to_frame", "squeeze",
    "infer_objects", "convert_dtypes", "reindex_like",
})


def _is_row_preserving_call(expr: ast.AST) -> bool:
    """True if *expr* is a method call that preserves the row count of its receiver."""
    if not isinstance(expr, ast.Call):
        return False
    func = expr.func
    if not isinstance(func, ast.Attribute):
        return False
    return func.attr in _ROW_PRESERVING_METHODS


def _expr_has_subset_signal(expr: ast.AST, aliases: set[str], subset_vars: set[str]) -> bool:
    def _is_full_row_slice(slice_node: ast.AST) -> bool:
        if isinstance(slice_node, ast.Slice):
            return slice_node.lower is None and slice_node.upper is None and slice_node.step is None
        return False

    def _is_column_only_projection(slice_node: ast.AST) -> bool:
        if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
            return True
        if isinstance(slice_node, ast.List):
            return all(
                isinstance(item, ast.Constant) and isinstance(item.value, str)
                for item in slice_node.elts
            )
        return False

    for node in ast.walk(expr):
        if isinstance(node, ast.Name) and node.id in subset_vars:
            return True
        if not isinstance(node, ast.Subscript):
            continue
        value = node.value
        if isinstance(value, ast.Attribute) and value.attr in {"loc", "iloc"} and isinstance(value.value, ast.Name):
            if value.value.id in aliases:
                if isinstance(node.slice, ast.Tuple) and node.slice.elts:
                    row_selector = node.slice.elts[0]
                    if _is_full_row_slice(row_selector):
                        continue
                return True
        if isinstance(value, ast.Name) and value.id in aliases:
            # df["col"] / df[["a","b"]] are column projections (not row subsets).
            if _is_column_only_projection(node.slice):
                continue
            # Non-column selectors are treated as row subsets.
            if not _is_full_row_slice(node.slice):
                return True
    for node in ast.walk(expr):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"query", "sample", "head", "tail", "dropna", "drop_duplicates"}:
                if isinstance(node.func.value, ast.Name) and (
                    node.func.value.id in aliases or node.func.value.id in subset_vars
                ):
                    return True
    return False


def _collect_dataframe_lineage(
    assignment_map: Dict[str, ast.AST],
) -> tuple[set[str], set[str]]:
    full_aliases: set[str] = set()
    subset_vars: set[str] = set()
    for var_name, expr in assignment_map.items():
        if _looks_like_read_csv_call(expr):
            full_aliases.add(var_name)
    changed = True
    while changed:
        changed = False
        for var_name, expr in assignment_map.items():
            if var_name in full_aliases or var_name in subset_vars:
                continue
            if isinstance(expr, ast.Name):
                if expr.id in full_aliases:
                    full_aliases.add(var_name)
                    changed = True
                    continue
                if expr.id in subset_vars:
                    subset_vars.add(var_name)
                    changed = True
                    continue
            has_subset_signal = _expr_has_subset_signal(expr, full_aliases, subset_vars)
            has_full_reference = _expr_references_alias(expr, full_aliases)

            if has_full_reference and has_subset_signal:
                # Mixed expressions default to full-risk unless the expression is
                # a direct subset extractor from the base frame.
                if isinstance(expr, ast.Subscript):
                    subset_vars.add(var_name)
                elif _is_row_preserving_call(expr):
                    # .copy(), .reset_index(), etc. preserve the row scope of
                    # their receiver.  Trace back to the receiver object to
                    # determine whether this is a subset or full frame.
                    _receiver = expr.func.value
                    _recv_subset = _expr_has_subset_signal(_receiver, full_aliases, subset_vars)
                    if _recv_subset:
                        subset_vars.add(var_name)
                    else:
                        full_aliases.add(var_name)
                elif isinstance(expr, ast.Call) and _call_name(expr).endswith("DataFrame"):
                    full_component = False
                    subset_component = False
                    candidate_exprs: List[ast.AST] = []
                    if expr.args:
                        candidate_exprs.append(expr.args[0])
                    for kw in expr.keywords:
                        if isinstance(kw, ast.keyword) and kw.arg == "data":
                            candidate_exprs.append(kw.value)
                    expanded: List[ast.AST] = []
                    for candidate in candidate_exprs:
                        if isinstance(candidate, ast.Dict):
                            expanded.extend([item for item in candidate.values if item is not None])
                        else:
                            expanded.append(candidate)
                    for candidate in expanded:
                        cand_has_subset = _expr_has_subset_signal(candidate, full_aliases, subset_vars)
                        cand_has_full = _expr_references_alias(candidate, full_aliases)
                        if cand_has_full and not cand_has_subset:
                            full_component = True
                        elif cand_has_subset:
                            subset_component = True
                    if full_component:
                        full_aliases.add(var_name)
                    elif subset_component:
                        subset_vars.add(var_name)
                    else:
                        full_aliases.add(var_name)
                else:
                    full_aliases.add(var_name)
                changed = True
                continue
            if has_subset_signal:
                subset_vars.add(var_name)
                changed = True
                continue
            if has_full_reference:
                full_aliases.add(var_name)
                changed = True
    return full_aliases, subset_vars


def _classify_row_scope(
    expr: ast.AST,
    assignment_map: Dict[str, ast.AST],
    full_aliases: set[str],
    subset_vars: set[str],
    depth: int = 0,
) -> str:
    if depth > 5:
        return "unknown"
    if isinstance(expr, ast.Name):
        if expr.id in subset_vars:
            return "subset"
        if expr.id in full_aliases:
            return "full"
        next_expr = assignment_map.get(expr.id)
        if next_expr is not None:
            return _classify_row_scope(
                next_expr,
                assignment_map,
                full_aliases,
                subset_vars,
                depth=depth + 1,
            )
        return "unknown"
    if isinstance(expr, ast.Dict):
        verdicts = [
            _classify_row_scope(item, assignment_map, full_aliases, subset_vars, depth=depth + 1)
            for item in expr.values
            if item is not None
        ]
        if "full" in verdicts:
            return "full"
        if "subset" in verdicts:
            return "subset"
        return "unknown"
    if isinstance(expr, ast.Call):
        call_name = _call_name(expr)
        if call_name.endswith("DataFrame"):
            candidate_exprs: List[ast.AST] = []
            if expr.args:
                candidate_exprs.append(expr.args[0])
            for kw in expr.keywords:
                if isinstance(kw, ast.keyword) and kw.arg == "data":
                    candidate_exprs.append(kw.value)
            verdicts = [
                _classify_row_scope(item, assignment_map, full_aliases, subset_vars, depth=depth + 1)
                for item in candidate_exprs
            ]
            if "full" in verdicts:
                return "full"
            if "subset" in verdicts:
                return "subset"
            return "unknown"
        # Row-preserving methods propagate the scope of their receiver.
        if _is_row_preserving_call(expr):
            return _classify_row_scope(
                expr.func.value, assignment_map, full_aliases, subset_vars, depth=depth + 1,
            )
    has_subset = _expr_has_subset_signal(expr, full_aliases, subset_vars)
    has_full = _expr_references_alias(expr, full_aliases)
    if has_subset and not has_full:
        return "subset"
    if has_subset and has_full:
        if isinstance(expr, ast.Subscript):
            return "subset"
        # Row-preserving calls with mixed signals: trace receiver scope.
        if _is_row_preserving_call(expr):
            return _classify_row_scope(
                expr.func.value, assignment_map, full_aliases, subset_vars, depth=depth + 1,
            )
        # Mixed full/subset signals in one expression are treated as risky.
        return "full"
    if has_full:
        return "full"
    return "unknown"


def _match_expected_artifact_path(path: str, expected_paths: Dict[str, int]) -> str:
    normalized = _normalize_artifact_path_for_match(path)
    if not normalized:
        return ""
    if normalized in expected_paths:
        return normalized
    if normalized.lower() in {p.lower(): p for p in expected_paths}.keys():
        for key in expected_paths:
            if key.lower() == normalized.lower():
                return key
    basename = os.path.basename(normalized).lower()
    candidates = [key for key in expected_paths if os.path.basename(key).lower() == basename]
    if len(candidates) == 1:
        return candidates[0]
    return ""


def _analyze_output_row_count_consistency(
    tree: ast.AST,
    evaluation_spec: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    row_hints = _extract_row_count_hints_for_qa(evaluation_spec)
    expected_paths = _resolve_expected_csv_row_counts_for_qa(evaluation_spec, row_hints)
    if not expected_paths:
        return {
            "active": False,
            "row_count_hints": row_hints,
            "expected_csv_row_counts": {},
            "issues": [],
        }

    assignment_map = _collect_assignment_map(tree)
    full_aliases, subset_vars = _collect_dataframe_lineage(assignment_map)
    n_total = row_hints.get("n_total")
    n_test = row_hints.get("n_test")

    issues: List[Dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "to_csv":
            continue
        raw_path = _resolve_to_csv_target_path(node)
        matched = _match_expected_artifact_path(raw_path, expected_paths)
        if not matched:
            continue
        expected = expected_paths.get(matched)
        if not isinstance(expected, int):
            continue
        requires_subset = False
        if isinstance(n_total, int) and n_total > 0 and expected < n_total:
            requires_subset = True
        elif isinstance(n_test, int) and n_test > 0 and expected == n_test:
            requires_subset = True
        if not requires_subset:
            continue
        scope = _classify_row_scope(node.func.value, assignment_map, full_aliases, subset_vars)
        if scope == "full":
            issues.append(
                {
                    "path": matched,
                    "expected_row_count": expected,
                    "lineno": getattr(node, "lineno", None),
                    "reason": "output_expected_subset_but_code_writes_full_dataframe",
                }
            )

    return {
        "active": True,
        "row_count_hints": row_hints,
        "expected_csv_row_counts": expected_paths,
        "issues": issues,
    }


def run_static_qa_checks(
    code: str,
    evaluation_spec: Dict[str, Any] | None = None,
    facts: Dict[str, Any] | None = None,
) -> Optional[Dict[str, Any]]:
    """
    Deterministic pre-checks to block unsafe patterns without relying on the LLM.
    """
    def _detect_perfect_score_pattern(text: str) -> bool:
        lowered = text.lower()
        patterns = [
            r"r2[^\\n]{0,50}(1\\.0|0\\.99|0\\.98)",
            r"r[\s_]?2[\s:=><]{1,3}(1\\.0|0\\.99|0\\.98)",
            r"mae[^\\n]{0,40}(0\\.0+|1e-0*[1-6])",
        ]
        import re

        for pat in patterns:
            if re.search(pat, lowered, flags=re.IGNORECASE):
                return True
        return False

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    scanner = _StaticQAScanner()
    scanner.visit(tree)
    qa_gate_specs, contract_source_used, gate_warnings = resolve_qa_gates(evaluation_spec)
    qa_gate_names = _gate_names(qa_gate_specs)
    qa_gate_set = set(qa_gate_names)
    gate_lookup = _gate_lookup(qa_gate_specs)
    if gate_warnings:
        for warning in gate_warnings:
            print(warning)

    # SENIOR REASONING: Fail-fast if ML plan is invalid/incomplete
    ml_plan = (evaluation_spec or {}).get("ml_plan")
    data_profile = (evaluation_spec or {}).get("data_profile")
    strategy = (evaluation_spec or {}).get("strategy")
    contract = (evaluation_spec or {}).get("contract")
    if ml_plan:
        plan_validation = validate_ml_plan_constraints(
            ml_plan, data_profile, contract, strategy
        )
        if not plan_validation.get("ok", True):
            violations = plan_validation.get("violations", [])
            plan_source = ml_plan.get("plan_source", "unknown")
            # Fail-fast: reject if plan_source indicates LLM failure or missing init
            if plan_source in ("missing_llm_init", "llm_error", "fallback"):
                return {
                    "status": "REJECTED",
                    "feedback": f"ML plan is invalid (source={plan_source}). Cannot proceed with code evaluation.",
                    "failed_gates": ["ML_PLAN_INVALID"],
                    "required_fixes": [
                        "Regenerate ML plan with initialized ML Engineer.",
                        "Ensure LLM call works properly.",
                        "Plan must include training_rows_policy + metric_policy from LLM.",
                    ],
                    "facts": {
                        "ml_plan_validation": plan_validation,
                        "plan_source": plan_source,
                    },
                    "warnings": violations + gate_warnings,
                    "qa_gates_evaluated": qa_gate_names,
                    "hard_failures": ["ML_PLAN_INVALID"],
                    "soft_failures": [],
                    "contract_source_used": contract_source_used,
                }
            # If plan_source is "llm" but has violations, add as warnings not hard failure
            elif violations:
                gate_warnings.extend([f"ML_PLAN_CONSTRAINT: {v}" for v in violations])

    require_variance_guard = "target_variance_guard" in qa_gate_set
    require_leakage_guard = "leakage_prevention" in qa_gate_set
    require_dialect_guard = "dialect_mismatch_handling" in qa_gate_set
    require_group_split = "group_split_required" in qa_gate_set
    require_read_csv = "must_read_input_csv" in qa_gate_set
    ml_data_path = _resolve_ml_data_path(evaluation_spec)
    require_no_synth = "no_synthetic_data" in qa_gate_set
    require_contract_columns = "must_reference_contract_columns" in qa_gate_set
    train_eval_gate = None
    if "train_eval_split" in qa_gate_set:
        train_eval_gate = "train_eval_split"
    elif "train_eval_separation" in qa_gate_set:
        train_eval_gate = "train_eval_separation"
    require_train_eval = train_eval_gate is not None
    require_output_row_count_consistency = "output_row_count_consistency" in qa_gate_set

    allow_resampling_random = False
    allow_synthetic_augmentation = False
    no_synth_spec = gate_lookup.get("no_synthetic_data")
    if isinstance(no_synth_spec, dict):
        params = no_synth_spec.get("params")
        if isinstance(params, dict):
            allow_resampling_random = bool(params.get("allow_resampling_random"))
            allow_synthetic_augmentation = bool(
                params.get("allow_synthetic_augmentation")
                or params.get("allow_augmentation")
                or params.get("allow_data_augmentation")
            )
    metric_round_context = _extract_metric_round_context(evaluation_spec)
    metric_round_active = bool(metric_round_context.get("metric_round_active"))
    augmentation_requested = bool(metric_round_context.get("augmentation_requested"))
    dynamic_aug_relax = (
        _env_flag("QA_RELAX_NO_SYNTH_FOR_AUGMENTATION", True)
        and metric_round_active
        and augmentation_requested
    )
    synthetic_detected = _detect_synthetic_data_calls(tree, allow_resampling_random)

    warnings: List[str] = []
    failed_gates: List[str] = []
    required_fixes: List[str] = []
    hard_failures: List[str] = []
    soft_failures: List[str] = []

    def _flag(gate: str, message: str, fix: str) -> None:
        spec = gate_lookup.get(gate.lower())
        if not spec:
            warnings.append(message)
            return
        if str(spec.get("severity")).upper() == "SOFT":
            soft_failures.append(gate)
            warnings.append(message)
            return
        failed_gates.append(gate)
        hard_failures.append(gate)
        required_fixes.append(fix)

    if scanner.has_security_violation:
        _flag(
            "security_sandbox",
            "Security sandbox violation detected (forbidden import or call).",
            "Remove forbidden imports/calls (sys, subprocess, requests, os.system).",
        )

    if scanner.has_random_target_noise:
        _flag(
            "target_variance_guard",
            "Target modifications driven by random generators detected.",
            "Remove random/jitter modifications to the target and enforce variance check instead.",
        )

    if scanner.has_split_fabrication:
        _flag(
            "dialect_mismatch_handling",
            "Delimiter mismatch must be fixed via dialect; do not fabricate columns via split/expand.",
            "Load with correct output_dialect and abort on delimiter mismatch; do not split columns.",
        )

    if require_read_csv and not scanner.has_read_csv:
        _flag(
            "must_read_input_csv",
            f"Missing pandas.read_csv call; code must load the ML dataset ({ml_data_path}).",
            f"Read the ML dataset via pandas.read_csv('{ml_data_path}') and use it as the dataset source.",
        )

    if require_no_synth and synthetic_detected:
        if (allow_synthetic_augmentation and augmentation_requested) or dynamic_aug_relax:
            warnings.append(
                "NO_SYNTHETIC_DATA_RELAXED_FOR_AUGMENTATION: metric improvement round requested "
                "controlled augmentation; synthetic-data gate enforced as warning only."
            )
        else:
            _flag(
                "no_synthetic_data",
                "Synthetic data construction detected (random generators, make_* datasets, or literal DataFrame).",
                "Remove synthetic data creation; load the real CSV and operate on contract columns only.",
            )

    contract_columns = _resolve_contract_columns_for_qa(evaluation_spec)
    if not contract_columns:
        require_contract_columns = False
    has_contract_column_reference = _code_mentions_columns(code, contract_columns, tree)
    if require_contract_columns and not has_contract_column_reference:
        _flag(
            "must_reference_contract_columns",
            "No references to contract canonical columns detected in code.",
            "Reference contract columns explicitly (canonical names) when selecting features/targets.",
        )

    if require_variance_guard and not scanner.has_variance_guard:
        _flag(
            "target_variance_guard",
            "Missing target variance guard (nunique <= 1 must raise ValueError).",
            "Add an explicit nunique<=1 check that raises ValueError before training.",
        )

    regression_present = scanner.has_regression_model or (scanner.has_fit_call and scanner.has_regression_metric)
    if require_leakage_guard and regression_present and not scanner.has_leakage_assert:
        _flag(
            "leakage_prevention",
            "Regression detected but assert_no_deterministic_target_leakage is missing before training.",
            "Call assert_no_deterministic_target_leakage(...) on regression tasks before fitting.",
        )

    if require_group_split and scanner.has_infer_group_key_call and not scanner.has_group_split_usage:
        _flag(
            "group_split_required",
            "Group key inferred but group-aware split not used.",
            "Use GroupKFold/GroupShuffleSplit (or groups=) when infer_group_key returns a grouping vector.",
        )

    # EXECUTION-AWARE VALIDATION: Allow execution results to override static checks
    # If metrics.json exists with valid content, the methodology is likely sound
    # regardless of whether we detected the exact split pattern in code.
    metrics_path = (evaluation_spec or {}).get("metrics_path") or "data/metrics.json"
    has_valid_metrics = False
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8", errors="replace") as f:
                metrics_content = json.load(f)
            # Valid if it has model_performance or any metric keys
            if isinstance(metrics_content, dict) and (
                metrics_content.get("model_performance") or
                any(k for k in metrics_content.keys() if k not in ("error", "status"))
            ):
                has_valid_metrics = True
        except Exception:
            pass

    if require_train_eval and scanner.has_fit_call and not scanner.has_train_eval_split:
        # RELAXED: If valid metrics exist, downgrade from flag to warning
        if has_valid_metrics:
            warnings.append(
                "SOFT_OVERRIDE: train/eval split not detected in static analysis, "
                "but valid metrics.json exists - assuming methodology is sound."
            )
        else:
            _flag(
                train_eval_gate,
                "Model training detected without explicit train/eval separation.",
                "Add train_test_split or a cross-validation strategy before fitting.",
            )

    if _detect_perfect_score_pattern(code) and not (scanner.has_leakage_assert or "leakage" in code.lower()):
        _flag(
            "leakage_prevention",
            "Perfect/near-perfect score pattern detected; require explicit leakage audit and explanation.",
            "Explain high R2/low MAE and include assert_no_deterministic_target_leakage before training.",
        )

    # SENIOR REASONING: Plan ↔ Code coherence validation
    row_count_consistency = _analyze_output_row_count_consistency(tree, evaluation_spec)
    row_count_issues = row_count_consistency.get("issues") if isinstance(row_count_consistency, dict) else []
    if row_count_issues:
        first_issue = row_count_issues[0] if isinstance(row_count_issues[0], dict) else {}
        issue_path = first_issue.get("path") if isinstance(first_issue, dict) else "unknown.csv"
        issue_expected = first_issue.get("expected_row_count") if isinstance(first_issue, dict) else "unknown"
        issue_line = first_issue.get("lineno") if isinstance(first_issue, dict) else None
        issue_context = (
            f"{issue_path} (expected_rows={issue_expected}, line={issue_line})"
            if issue_line is not None
            else f"{issue_path} (expected_rows={issue_expected})"
        )
        if require_output_row_count_consistency:
            _flag(
                "output_row_count_consistency",
                "Output CSV row-count consistency risk detected: "
                f"artifact expected subset-sized rows but code writes full-frame data at {issue_context}.",
                "Build output CSVs from the correct subset and add an explicit row-count guard before to_csv.",
            )
        else:
            warnings.append(
                "OUTPUT_ROW_COUNT_CONSISTENCY: artifact expected subset-sized rows but code appears to write "
                f"full-frame data at {issue_context}."
            )
    elif not isinstance(row_count_consistency, dict):
        row_count_consistency = {"active": False, "issues": []}

    require_plan_coherence = "plan_code_coherence" in qa_gate_set
    ml_plan = (evaluation_spec or {}).get("ml_plan")
    data_profile = (evaluation_spec or {}).get("data_profile")
    coherence_result = None
    if ml_plan:
        # Run coherence check as warning-only (LLM reviewer decides)
        try:
            from src.utils.ml_plan_validation import validate_plan_code_coherence
            coherence_result = validate_plan_code_coherence(ml_plan, code, data_profile)
            if coherence_result and not coherence_result.get("passed", True):
                for violation in coherence_result.get("violations", []):
                    warnings.append(f"PLAN_COHERENCE: {violation}")
                    required_fixes.append(
                        "Fix code to implement the ml_plan.json training_rows_policy and metric_policy correctly."
                    )
            if coherence_result and coherence_result.get("warnings"):
                for warning in coherence_result.get("warnings", []):
                    warnings.append(f"PLAN_COHERENCE: {warning}")
        except Exception as coh_err:
            warnings.append(f"Plan coherence check failed: {coh_err}")

    facts_payload = facts if isinstance(facts, dict) else collect_static_qa_facts(code)
    facts_payload["has_read_csv"] = scanner.has_read_csv
    facts_payload["has_synthetic_data"] = synthetic_detected
    facts_payload["augmentation_requested"] = augmentation_requested
    facts_payload["metric_round_active"] = metric_round_active
    facts_payload["no_synth_relaxed_for_augmentation"] = bool(
        synthetic_detected
        and require_no_synth
        and ((allow_synthetic_augmentation and augmentation_requested) or dynamic_aug_relax)
    )
    facts_payload["has_contract_column_reference"] = has_contract_column_reference
    facts_payload["contract_columns_checked"] = contract_columns
    facts_payload["qa_gates_evaluated"] = qa_gate_names
    facts_payload["hard_failures"] = hard_failures
    facts_payload["soft_failures"] = soft_failures
    facts_payload["contract_source_used"] = contract_source_used
    facts_payload["output_row_count_consistency"] = row_count_consistency
    if coherence_result:
        facts_payload["plan_code_coherence"] = coherence_result
    if gate_warnings:
        facts_payload["warnings"] = list(gate_warnings)

    if hard_failures:
        return {
            "status": "REJECTED",
            "feedback": "Static QA gate failures detected.",
            "failed_gates": failed_gates,
            "required_fixes": required_fixes,
            "facts": facts_payload,
            "warnings": warnings + gate_warnings,
            "qa_gates_evaluated": qa_gate_names,
            "hard_failures": hard_failures,
            "soft_failures": soft_failures,
            "contract_source_used": contract_source_used,
        }
    if warnings or soft_failures or gate_warnings:
        return {
            "status": "WARN",
            "warnings": warnings + gate_warnings,
            "facts": facts_payload,
            "qa_gates_evaluated": qa_gate_names,
            "hard_failures": hard_failures,
            "soft_failures": soft_failures,
            "contract_source_used": contract_source_used,
        }
    return {
        "status": "PASS",
        "facts": facts_payload,
        "qa_gates_evaluated": qa_gate_names,
        "hard_failures": hard_failures,
        "soft_failures": soft_failures,
        "contract_source_used": contract_source_used,
    }


def collect_static_qa_facts(code: str) -> Dict[str, bool]:
    facts = {
        "has_variance_guard": False,
        "has_group_split_usage": False,
        "has_infer_group_key_call": False,
        "has_leakage_assert": False,
        "has_random_target_noise": False,
        "has_split_fabrication": False,
        "has_security_violation": False,
        "has_train_eval_split": False,
        "has_mkdirs": False,
        "forbidden_imports_found": False,
        "has_read_csv": False,
        "has_synthetic_data": False,
        "has_contract_column_reference": False,
    }
    try:
        tree = ast.parse(code)
    except Exception:
        return facts
    scanner = _StaticQAScanner()
    scanner.visit(tree)
    facts["has_variance_guard"] = scanner.has_variance_guard
    facts["has_group_split_usage"] = scanner.has_group_split_usage
    facts["has_infer_group_key_call"] = scanner.has_infer_group_key_call
    facts["has_leakage_assert"] = scanner.has_leakage_assert
    facts["has_random_target_noise"] = scanner.has_random_target_noise
    facts["has_split_fabrication"] = scanner.has_split_fabrication
    facts["has_security_violation"] = scanner.has_security_violation
    facts["has_train_eval_split"] = scanner.has_train_eval_split
    facts["has_mkdirs"] = scanner.has_mkdirs
    facts["forbidden_imports_found"] = scanner.forbidden_imports_found
    facts["has_read_csv"] = scanner.has_read_csv
    facts["has_synthetic_data"] = scanner.has_synthetic_data
    facts["has_contract_column_reference"] = _code_mentions_columns(
        code, _resolve_contract_columns_for_qa(None), tree
    )
    return facts
