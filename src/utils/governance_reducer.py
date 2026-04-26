"""
Governance Reducer: Deterministic verdict computation for run outcomes.

This module provides a unified, deterministic reducer that derives a single
global verdict from multiple compliance sources:
- output_contract_report.json (overall_status, artifact_requirements_report)
- state (review_verdict, gate_context, hard_failures)
- integrity_audit_report.json (critical issues)

RULES:
- overall_status ∈ {"ok", "warning", "error"}
- If overall_status == "error" → run_outcome = "NO_GO"
- Else if ceiling_detected or observational_only or overall_status == "warning" → "GO_WITH_LIMITATIONS"
- Else → "GO"
"""

import math
from typing import Any, Dict, List, Set


# Status values that indicate rejection/failure
_REJECTED_STATUSES: Set[str] = {"REJECTED", "FAIL", "CRASH", "ERROR"}


def _normalize_status(status: Any) -> str:
    """Normalize status string to uppercase, stripped."""
    if status is None:
        return ""
    return str(status).strip().upper()


def _is_rejected_status(status: Any) -> bool:
    """Check if a status value indicates rejection."""
    return _normalize_status(status) in _REJECTED_STATUSES


def _safe_list(value: Any) -> List[Any]:
    """Safely convert to list."""
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _dedupe_reasons(reasons: List[str]) -> List[str]:
    """Deduplicate reasons while preserving order."""
    seen: Set[str] = set()
    result: List[str] = []
    for r in reasons:
        if r and r not in seen:
            seen.add(r)
            result.append(r)
    return result


def _normalize_gate_name(name: Any) -> str:
    return str(name or "").strip().lower()


_PERFORMANCE_ARTIFACT_TOKENS = {
    "metric",
    "metrics",
    "evaluation",
    "eval",
    "validation",
    "performance",
    "benchmark",
    "portfolio",
    "inference",
    "cv_",
    "model",
    "scorecard",
}

_PERFORMANCE_METRIC_TOKENS = {
    "accuracy",
    "auc",
    "roc",
    "pr_auc",
    "f1",
    "precision",
    "recall",
    "qwk",
    "kappa",
    "mae",
    "mape",
    "rmse",
    "r2",
    "lift",
    "gini",
    "log_loss",
    "brier",
    "ndcg",
    "map",
    "mrr",
    "latency",
    "throughput",
    "coverage",
    "portfolio",
    "calibration",
}

_NON_PERFORMANCE_METRIC_TOKENS = {
    "row_count",
    "rows",
    "missing",
    "null",
    "nan",
    "schema",
    "column",
    "columns",
    "leakage",
    "forbidden",
    "required",
    "artifact",
    "runtime",
    "security",
    "integrity",
}


def _coerce_float_maybe(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            if math.isfinite(float(value)):
                return float(value)
        except Exception:
            return None
    try:
        text = str(value).strip()
        if not text:
            return None
        parsed = float(text)
        if math.isfinite(parsed):
            return parsed
    except Exception:
        return None
    return None


def is_performance_threshold_failure(failure: Any) -> bool:
    """Return True only for evidence-backed numeric target misses.

    These are not runtime/compliance defects: the artifact exists and contains a
    real metric value, but that value has not reached a business/model target
    yet. Missing evidence, schema issues, data quality gates, and integrity
    gates remain blocking elsewhere.
    """
    if not isinstance(failure, dict):
        return False
    if failure.get("passed") is not False:
        return False
    if str(failure.get("status") or "").strip().lower() == "missing_evidence":
        return False
    if _coerce_float_maybe(failure.get("value")) is None:
        return False
    threshold_keys = ("min_value", "max_value", "threshold")
    if not any(_coerce_float_maybe(failure.get(key)) is not None for key in threshold_keys):
        return False

    metric = str(failure.get("metric") or failure.get("name") or "").strip().lower()
    artifact = str(failure.get("artifact_path") or failure.get("evidence_source") or "").strip().lower()
    detail = str(failure.get("detail") or "").strip().lower()
    semantic_haystack = f"{metric} {detail}"
    if any(token in semantic_haystack for token in _NON_PERFORMANCE_METRIC_TOKENS):
        return False
    metric_like = any(token in metric for token in _PERFORMANCE_METRIC_TOKENS)
    artifact_like = any(token in artifact for token in _PERFORMANCE_ARTIFACT_TOKENS)
    if not metric_like and not artifact_like:
        return False
    if artifact and ".json" not in artifact:
        return False
    return True


def classify_output_contract_performance_thresholds(
    output_contract_report: Dict[str, Any] | None,
) -> Dict[str, Any]:
    """Split output-contract QA failures into blocking vs optimization targets."""
    report = output_contract_report if isinstance(output_contract_report, dict) else {}
    qa_results = report.get("qa_gate_results") if isinstance(report.get("qa_gate_results"), dict) else {}
    raw_failures = qa_results.get("failures") if isinstance(qa_results.get("failures"), list) else report.get("qa_gate_failures")
    failures = [item for item in (raw_failures or []) if item]
    performance_gaps = [item for item in failures if is_performance_threshold_failure(item)]
    blocking_failures = [item for item in failures if item not in performance_gaps]

    artifact_report = report.get("artifact_requirements_report")
    artifact_status = (
        str(artifact_report.get("status") or "").strip().lower()
        if isinstance(artifact_report, dict)
        else ""
    )
    missing = [str(item) for item in (report.get("missing") or []) if str(item).strip()]
    only_performance_threshold_failures = bool(
        failures
        and performance_gaps
        and not blocking_failures
        and not missing
        and artifact_status != "error"
    )
    return {
        "performance_threshold_gaps": performance_gaps,
        "blocking_qa_gate_failures": blocking_failures,
        "only_performance_threshold_failures": only_performance_threshold_failures,
    }


def _extract_hard_gate_names(contract: Dict[str, Any]) -> Set[str]:
    contract = contract if isinstance(contract, dict) else {}
    hard_names: Set[str] = set()
    gate_blocks = [
        contract.get("cleaning_gates"),
        contract.get("qa_gates"),
        contract.get("reviewer_gates"),
    ]
    eval_spec = contract.get("evaluation_spec")
    if isinstance(eval_spec, dict):
        gate_blocks.extend([eval_spec.get("qa_gates"), eval_spec.get("gates")])

    for block in gate_blocks:
        if not isinstance(block, list):
            continue
        for gate in block:
            if isinstance(gate, dict):
                name = gate.get("name") or gate.get("id") or gate.get("gate")
                if not name:
                    continue
                severity = str(gate.get("severity") or "").strip().upper()
                required = gate.get("required")
                is_hard = severity == "HARD" or (severity == "" and required is not False)
                if is_hard:
                    hard_names.add(_normalize_gate_name(name))
            elif isinstance(gate, str):
                hard_names.add(_normalize_gate_name(gate))
    return hard_names


def _segmentation_requirement_enabled(contract: Dict[str, Any]) -> bool:
    eval_spec = contract.get("evaluation_spec")
    if not isinstance(eval_spec, dict):
        return False
    seg_obj = eval_spec.get("segmentation")
    if isinstance(seg_obj, dict):
        if bool(seg_obj.get("required")):
            return True
        features = seg_obj.get("features")
        if isinstance(features, list) and any(str(x).strip() for x in features):
            return True
    seg_required = eval_spec.get("segmentation_required")
    if isinstance(seg_required, list) and any(str(x).strip() for x in seg_required):
        return True
    return False


def _decisioning_requirement_enabled(contract: Dict[str, Any]) -> bool:
    req = contract.get("decisioning_requirements")
    if not isinstance(req, dict):
        return False
    enabled = req.get("enabled")
    required = req.get("required")
    if required is True:
        return True
    if enabled is True and required is not False:
        return True
    return False


def _is_contract_hard_gate_failed(
    gate_name: str,
    hard_gate_names: Set[str],
    segmentation_required: bool,
    decisioning_required: bool,
) -> bool:
    gate_key = _normalize_gate_name(gate_name)
    if not gate_key:
        return False
    if gate_key in hard_gate_names:
        return True
    if segmentation_required and "segmentation" in gate_key:
        return True
    if decisioning_required and any(
        token in gate_key for token in ("decisioning", "operational", "uncertainty", "review_rule")
    ):
        return True
    return False


def compute_governance_verdict(
    output_contract_report: Dict[str, Any],
    state: Dict[str, Any],
    contract: Dict[str, Any] = None,
    integrity_report: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Deterministic reducer that computes a unified governance verdict.

    This function aggregates signals from multiple sources to derive a single
    overall_status and run_outcome. It is deterministic (no network calls,
    no dataset-specific heuristics).

    Args:
        output_contract_report: The output_contract_report.json content
        state: The current pipeline state (contains review_verdict, gate_context, hard_failures)
        integrity_report: Optional integrity_audit_report.json content

    Returns:
        {
            "overall_status": "ok" | "warning" | "error",
            "hard_failures": [...],
            "failed_gates": [...],
            "reasons": [...],
        }
    """
    output_contract_report = output_contract_report if isinstance(output_contract_report, dict) else {}
    state = state if isinstance(state, dict) else {}
    integrity_report = integrity_report if isinstance(integrity_report, dict) else {}
    contract = contract if isinstance(contract, dict) else {}

    hard_failures: List[str] = []
    failed_gates: List[str] = []
    reasons: List[str] = []
    overall_status = "ok"

    # =========================================================================
    # Source A: output_contract_report.json
    # =========================================================================

    # A1: Check overall_status from output_contract_report
    performance_thresholds = classify_output_contract_performance_thresholds(output_contract_report)
    oc_overall_status = _normalize_status(output_contract_report.get("overall_status")).lower()
    if oc_overall_status == "error":
        if performance_thresholds.get("only_performance_threshold_failures"):
            overall_status = "warning"
            reasons.append("output_contract.performance_threshold_gaps")
        else:
            overall_status = "error"
            reasons.append("output_contract_report.overall_status=error")
            hard_failures.append("output_contract_compliance_error")
    elif oc_overall_status == "warning" and overall_status != "error":
        overall_status = "warning"
        reasons.append("output_contract_report.overall_status=warning")

    # A2: Check artifact_requirements_report.status
    artifact_report = output_contract_report.get("artifact_requirements_report")
    if isinstance(artifact_report, dict):
        artifact_status = _normalize_status(artifact_report.get("status")).lower()
        if artifact_status == "error":
            overall_status = "error"
            reasons.append("artifact_requirements_report.status=error")
            hard_failures.append("artifact_requirements_error")
        elif artifact_status == "warning" and overall_status != "error":
            overall_status = "warning"
            reasons.append("artifact_requirements_report.status=warning")

    # A3: Check missing files
    missing_files = _safe_list(output_contract_report.get("missing"))
    if missing_files:
        overall_status = "error"
        reasons.append(f"output_contract.missing={missing_files}")
        hard_failures.append("output_contract_missing_files")
        failed_gates.append("output_contract_missing")

    # =========================================================================
    # Source B: state + gate_context
    # =========================================================================

    # B1: Check review_verdict
    review_verdict = state.get("last_successful_review_verdict") or state.get("review_verdict")
    if _is_rejected_status(review_verdict):
        overall_status = "error"
        reasons.append(f"review_verdict={review_verdict}")
        hard_failures.append(f"review_{_normalize_status(review_verdict).lower()}")
    elif _normalize_status(review_verdict) == "APPROVE_WITH_WARNINGS" and overall_status == "ok":
        overall_status = "warning"
        reasons.append("review_verdict=APPROVE_WITH_WARNINGS")

    # B2: Check gate_context status
    gate_context = state.get("last_successful_gate_context") or state.get("last_gate_context")
    if isinstance(gate_context, dict):
        gc_status = gate_context.get("status")
        if _is_rejected_status(gc_status):
            overall_status = "error"
            reasons.append(f"gate_context.status={gc_status}")
            hard_failures.append(f"gate_{_normalize_status(gc_status).lower()}")
        elif _normalize_status(gc_status) == "APPROVE_WITH_WARNINGS" and overall_status == "ok":
            overall_status = "warning"
            reasons.append("gate_context.status=APPROVE_WITH_WARNINGS")

        # B3: Collect failed_gates from gate_context
        gc_failed_gates = _safe_list(gate_context.get("failed_gates"))
        for gate in gc_failed_gates:
            if gate and str(gate) not in failed_gates:
                failed_gates.append(str(gate))

        # B4: Collect hard_failures from gate_context (if present)
        gc_hard_failures = _safe_list(gate_context.get("hard_failures"))
        for hf in gc_hard_failures:
            if hf and str(hf) not in hard_failures:
                hard_failures.append(str(hf))
                overall_status = "error"
                reasons.append(f"gate_context.hard_failure={hf}")

    # B5: Check pipeline_aborted_reason
    pipeline_aborted = state.get("pipeline_aborted_reason")
    if pipeline_aborted:
        overall_status = "error"
        reasons.append(f"pipeline_aborted:{pipeline_aborted}")
        hard_failures.append(f"pipeline_aborted:{pipeline_aborted}")
        failed_gates.append(f"pipeline_aborted:{pipeline_aborted}")

    # B6: Check data_engineer_failed
    if state.get("data_engineer_failed"):
        overall_status = "error"
        reasons.append("data_engineer_failed=True")
        hard_failures.append("data_engineer_failed")

    # =========================================================================
    # Source C: QA/static/cleaning hard_failures from state
    # =========================================================================

    # C1: Accumulated hard_failures from state (if present)
    state_hard_failures = _safe_list(state.get("hard_failures"))
    for hf in state_hard_failures:
        if hf and str(hf) not in hard_failures:
            hard_failures.append(str(hf))
            overall_status = "error"
            reasons.append(f"state.hard_failure={hf}")

    # C2: Check QA and cleaning reviewer results stored in state
    for key in ["qa_last_result", "cleaning_last_result"]:
        result = state.get(key)
        if isinstance(result, dict):
            result_status = _normalize_status(result.get("status"))
            if result_status == "REJECTED":
                source = key.replace("_last_result", "")
                reasons.append(f"{source}.status=REJECTED")
                result_hard_failures = _safe_list(result.get("hard_failures"))
                for hf in result_hard_failures:
                    if hf and str(hf) not in hard_failures:
                        hard_failures.append(str(hf))
                        overall_status = "error"
                        reasons.append(f"{source}.hard_failure={hf}")

    # =========================================================================
    # Source D: integrity_audit_report.json
    # =========================================================================
    integrity_issues = integrity_report.get("issues", []) if isinstance(integrity_report, dict) else []
    integrity_critical_count = 0
    for issue in integrity_issues:
        if isinstance(issue, dict):
            severity = str(issue.get("severity", "")).strip().lower()
            if severity == "critical":
                integrity_critical_count += 1
    if integrity_critical_count > 0:
        overall_status = "error"
        reasons.append(f"integrity_critical_count={integrity_critical_count}")
        hard_failures.append("integrity_critical")
        failed_gates.append("integrity_critical")

    # E: Failed gates severity reduction.
    # - Any hard gate failure -> overall_status=error.
    # - Any remaining failed gate -> at least warning (never "ok" with unresolved gates).
    failed_gates = _dedupe_reasons([str(g) for g in failed_gates if g])
    if failed_gates:
        hard_gate_names = _extract_hard_gate_names(contract)
        segmentation_required = _segmentation_requirement_enabled(contract)
        decisioning_required = _decisioning_requirement_enabled(contract)
        hard_failed = [
            gate
            for gate in failed_gates
            if _is_contract_hard_gate_failed(
                gate_name=gate,
                hard_gate_names=hard_gate_names,
                segmentation_required=segmentation_required,
                decisioning_required=decisioning_required,
            )
        ]
        if hard_failed:
            overall_status = "error"
            reasons.append(f"hard_failed_gates={hard_failed}")
            for gate in hard_failed:
                hard_failures.append(f"hard_gate_failed:{gate}")
        elif overall_status == "ok":
            overall_status = "warning"
            reasons.append("failed_gates_present")

    return {
        "overall_status": overall_status,
        "hard_failures": _dedupe_reasons(hard_failures),
        "failed_gates": failed_gates,
        "reasons": _dedupe_reasons(reasons),
    }


def derive_run_outcome(
    governance_verdict: Dict[str, Any],
    ceiling_detected: bool = False,
    observational_only: bool = False,
) -> str:
    """
    Derive the final run_outcome from governance verdict and additional signals.

    Rules:
    - If overall_status == "error" → "NO_GO"
    - Else if ceiling_detected or observational_only or overall_status == "warning" → "GO_WITH_LIMITATIONS"
    - Else → "GO"

    Args:
        governance_verdict: Output from compute_governance_verdict()
        ceiling_detected: Whether metric ceiling was detected
        observational_only: Whether counterfactual_policy is "observational_only"

    Returns:
        "GO" | "GO_WITH_LIMITATIONS" | "NO_GO"
    """
    overall_status = governance_verdict.get("overall_status", "ok")

    if overall_status == "error":
        return "NO_GO"

    if ceiling_detected or observational_only or overall_status == "warning":
        return "GO_WITH_LIMITATIONS"

    return "GO"


def merge_hard_failures_into_state(
    state: Dict[str, Any],
    new_hard_failures: List[str],
) -> Dict[str, Any]:
    """
    Merge new hard_failures into state accumulator without duplicates.

    This helper ensures hard_failures are accumulated across pipeline stages.

    Args:
        state: Current pipeline state
        new_hard_failures: List of new hard failures to add

    Returns:
        Updated state with merged hard_failures
    """
    existing = _safe_list(state.get("hard_failures"))
    merged = list(existing)
    for hf in new_hard_failures:
        if hf and str(hf) not in merged:
            merged.append(str(hf))
    state["hard_failures"] = merged
    return state


def enrich_gate_context_with_hard_failures(
    gate_context: Dict[str, Any],
    reviewer_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Enrich gate_context with hard_failures from reviewer result.

    This ensures hard_failures are included in gate_context when present
    in QA/cleaning reviewer outputs.

    Args:
        gate_context: The gate context dict to enrich
        reviewer_result: The reviewer result containing potential hard_failures

    Returns:
        Enriched gate_context
    """
    if not isinstance(gate_context, dict):
        gate_context = {}
    if not isinstance(reviewer_result, dict):
        return gate_context

    hard_failures = _safe_list(reviewer_result.get("hard_failures"))
    if hard_failures:
        gate_context["hard_failures"] = hard_failures

    return gate_context
