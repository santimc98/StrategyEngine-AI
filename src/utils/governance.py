import json
import os
from datetime import datetime
from typing import Any, Dict, List

from src.utils.json_sanitize import dump_json
from src.utils.governance_reducer import (
    compute_governance_verdict,
    derive_run_outcome,
)


def _safe_load_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _load_metrics_report(state: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Load metrics from the first available canonical metrics artifact path."""
    state_obj = state if isinstance(state, dict) else {}
    state_metrics = state_obj.get("metrics_report")
    if isinstance(state_metrics, dict) and state_metrics:
        return dict(state_metrics)

    metrics_snapshot = state_obj.get("metrics_artifact_snapshot")
    if isinstance(metrics_snapshot, dict):
        snapshot_payload = metrics_snapshot.get("metrics_payload")
        if isinstance(snapshot_payload, dict) and snapshot_payload:
            return dict(snapshot_payload)

    loop_state = state_obj.get("metric_loop_state")
    if isinstance(loop_state, dict):
        final_entry = loop_state.get("final") if isinstance(loop_state.get("final"), dict) else {}
        incumbent_entry = loop_state.get("incumbent") if isinstance(loop_state.get("incumbent"), dict) else {}
        for entry in (final_entry, incumbent_entry):
            payload = entry.get("metrics_payload") if isinstance(entry.get("metrics_payload"), dict) else {}
            if payload:
                return dict(payload)

    candidates = [
        "data/metrics.json",
        "reports/evaluation_metrics.json",
        "data/evaluation_metrics.json",
        "reports/model_evaluation_metrics.json",
        "data/model_evaluation_metrics.json",
    ]
    for path in candidates:
        payload = _safe_load_json(path)
        if isinstance(payload, dict) and payload:
            return payload
    return {}


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False


def _flatten_metrics(obj: Any, prefix: str = "", out: Dict[str, float] | None = None) -> Dict[str, float]:
    if out is None:
        out = {}
    if not isinstance(obj, dict):
        return out
    for key, value in obj.items():
        metric_key = f"{prefix}{key}" if prefix else str(key)
        if _is_number(value):
            out[metric_key] = float(value)
        elif isinstance(value, dict):
            _flatten_metrics(value, f"{metric_key}.", out)
    return out


def _merge_explicit_primary_metric(metric_pool: Dict[str, float], metrics_report: Dict[str, Any]) -> Dict[str, float]:
    if not isinstance(metric_pool, dict):
        metric_pool = {}
    if not isinstance(metrics_report, dict):
        return metric_pool

    primary_name = str(metrics_report.get("primary_metric_name") or "").strip()
    primary_value = metrics_report.get("primary_metric_value")
    if primary_name and _is_number(primary_value):
        metric_pool.setdefault(primary_name, float(primary_value))

    model_perf = metrics_report.get("model_performance")
    if isinstance(model_perf, dict):
        model_primary_name = str(model_perf.get("primary_metric_name") or "").strip()
        model_primary_value = model_perf.get("primary_metric_value")
        if model_primary_name and _is_number(model_primary_value):
            metric_pool.setdefault(model_primary_name, float(model_primary_value))

    return metric_pool


def _metric_higher_is_better(name: str) -> bool:
    key = str(name or "").lower()
    if any(token in key for token in ["loss", "error", "mae", "rmse", "mse", "mape", "smape", "logloss", "brier"]):
        return False
    return True


def _normalize_metric_token(name: Any) -> str:
    return "".join(ch for ch in str(name or "").lower() if ch.isalnum())


def _extract_metric_value_by_name(metric_pool: Dict[str, float], metric_name: Any) -> Any:
    if not isinstance(metric_pool, dict):
        return None
    target = _normalize_metric_token(metric_name)
    if not target:
        return None
    exact_key = None
    fuzzy_key = None
    for key in metric_pool.keys():
        normalized = _normalize_metric_token(key)
        if not normalized:
            continue
        if normalized == target:
            exact_key = key
            break
        if target in normalized and fuzzy_key is None:
            fuzzy_key = key
    selected = exact_key or fuzzy_key
    if selected is None:
        return None
    return metric_pool.get(selected)


def _extract_baseline_vs_model(metric_pool: Dict[str, float]) -> Dict[str, Any]:
    baseline_vs_model = []
    baseline_keys = [k for k in metric_pool.keys() if any(tok in k.lower() for tok in ["baseline", "dummy", "naive", "null"])]
    for base_key in baseline_keys:
        base_val = metric_pool.get(base_key)
        norm = base_key.lower()
        for prefix in ["baseline_", "dummy_", "naive_", "null_"]:
            if norm.startswith(prefix):
                model_key = base_key[len(prefix):]
                break
        else:
            model_key = base_key.replace("baseline.", "", 1)
        model_val = metric_pool.get(model_key)
        if model_val is None:
            continue
        higher_is_better = _metric_higher_is_better(model_key)
        delta = (model_val - base_val) if higher_is_better else (base_val - model_val)
        baseline_vs_model.append(
            {
                "metric": model_key,
                "baseline": base_val,
                "model": model_val,
                "delta": delta,
                "higher_is_better": higher_is_better,
            }
        )
    return {"pairs": baseline_vs_model, "metric_pool": metric_pool}


def _detect_metric_ceiling(
    baseline_vs_model: Dict[str, Any],
    data_adequacy: Dict[str, Any],
    thresholds: Dict[str, float],
) -> Dict[str, Any]:
    ceiling_detected = False
    reason = None
    pairs = baseline_vs_model.get("pairs", []) if isinstance(baseline_vs_model, dict) else []
    metric_pool = baseline_vs_model.get("metric_pool", {}) if isinstance(baseline_vs_model, dict) else {}
    for pair in pairs:
        metric = str(pair.get("metric", "")).lower()
        delta = pair.get("delta")
        if delta is None:
            continue
        if "auc" in metric:
            if delta < thresholds["auc"]:
                ceiling_detected = True
                reason = "low_signal"
        if "f1" in metric:
            if delta < thresholds["f1"]:
                ceiling_detected = True
                reason = "low_signal"
        if "r2" in metric:
            if delta < thresholds["r2"]:
                ceiling_detected = True
                reason = "low_signal"
        if any(tok in metric for tok in ["mae", "rmse", "mape"]):
            if delta < thresholds["error"]:
                ceiling_detected = True
                reason = "low_signal"
    if isinstance(data_adequacy, dict):
        reasons = data_adequacy.get("reasons", []) or []
        reason_tags = {str(item).split(":", 1)[0] for item in reasons if item}
        stale_pipeline_abort = bool(metric_pool) and reason_tags == {"pipeline_aborted_before_metrics"}
        if data_adequacy.get("status") in {"data_limited", "insufficient_signal"} and not stale_pipeline_abort:
            ceiling_detected = True
            reason = reason or "low_signal"
        if any("high_dimensionality_low_sample" in r for r in reasons):
            ceiling_detected = True
            reason = "small_n"
    for key, value in metric_pool.items():
        if "cv_std" in str(key).lower() and _is_number(value) and value >= thresholds["cv_std"]:
            ceiling_detected = True
            reason = "high_variance_cv"
            break
    return {"metric_ceiling_detected": ceiling_detected, "ceiling_reason": reason}


def _collect_contract_views(base_dir: str = "data") -> Dict[str, Any]:
    views_dir = os.path.join(base_dir, "contracts", "views")
    view_names = [
        "de_view",
        "ml_view",
        "cleaning_view",
        "qa_view",
        "reviewer_view",
        "translator_view",
        "results_advisor_view",
    ]
    paths = {}
    present = []
    for name in view_names:
        path = os.path.join(views_dir, f"{name}.json")
        if os.path.exists(path):
            paths[name] = path
            present.append(name)
    return {"paths": paths, "present": present}


def _packet_has_no_findings(packet: Any) -> bool | None:
    """
    Return:
    - True when packet is explicitly non-blocking (no failed/hard findings)
    - False when packet has explicit findings
    - None when packet is unavailable/unknown
    """
    if not isinstance(packet, dict):
        return None
    status = str(packet.get("status") or "").strip().upper()
    failed = [str(x) for x in (packet.get("failed_gates") or []) if x]
    hard = [str(x) for x in (packet.get("hard_failures") or []) if x]
    if failed or hard:
        return False
    if status in {"REJECTED", "NEEDS_IMPROVEMENT", "FAIL", "FAILED", "ERROR", "CRASH"}:
        return False
    return True


def _normalize_failed_gates_for_summary(state: Dict[str, Any], failed_gates: List[Any]) -> List[str]:
    """
    Keep deterministic failed gates, but drop broad board labels that are not backed
    by reviewer packet evidence (e.g., "qa_gates" with empty qa_last_result findings).
    """
    raw = [str(item).strip() for item in (failed_gates or []) if str(item).strip()]
    if not raw:
        return []

    qa_clean = _packet_has_no_findings(state.get("qa_last_result"))
    reviewer_clean = _packet_has_no_findings(state.get("reviewer_last_result"))
    results_clean = _packet_has_no_findings(state.get("results_last_result"))

    drop_if_clean = {
        "qa_gates": qa_clean,
        "reviewer_alignment": reviewer_clean,
        "results_quality": results_clean,
    }

    normalized: List[str] = []
    for gate in raw:
        key = gate.lower()
        should_drop = drop_if_clean.get(key)
        if should_drop is True:
            continue
        if gate not in normalized:
            normalized.append(gate)
    return normalized


def build_governance_report(state: Dict[str, Any]) -> Dict[str, Any]:
    contract = _safe_load_json("data/execution_contract.json") or state.get("execution_contract", {})
    output_contract = _safe_load_json("data/output_contract_report.json")
    case_alignment = _safe_load_json("data/case_alignment_report.json")
    alignment_check = _safe_load_json("data/alignment_check.json")
    integrity_report = _safe_load_json("data/integrity_audit_report.json")
    integrity = _safe_load_json("data/integrity_audit_report.json")

    issues = integrity_report.get("issues", []) if isinstance(integrity_report, dict) else []
    severity_counts = {}
    for issue in issues:
        sev = str(issue.get("severity", "unknown"))
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    review_verdict = state.get("last_successful_review_verdict") or state.get("review_verdict")
    gate_context = state.get("last_successful_gate_context") or state.get("last_gate_context")

    return {
        "run_id": state.get("run_id"),
        "timestamp": datetime.utcnow().isoformat(),
        "strategy_title": contract.get("strategy_title", ""),
        "business_objective": contract.get("business_objective", ""),
        "review_verdict": review_verdict,
        "last_gate_context": gate_context,
        "output_contract": output_contract,
        "case_alignment": case_alignment,
        "alignment_check": alignment_check,
        "integrity_issues_summary": severity_counts,
        "budget_counters": state.get("budget_counters", {}),
        "run_budget": state.get("run_budget", {}),
        "data_risks": contract.get("data_risks", []),
    }


def write_governance_report(state: Dict[str, Any], path: str = "data/governance_report.json") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    report = build_governance_report(state)
    try:
        dump_json(path, report)
    except Exception:
        pass


def build_run_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build run summary with unified governance verdict.

    Uses the governance reducer to derive a deterministic overall_status
    and run_outcome from multiple compliance sources.
    """
    case_alignment = _safe_load_json("data/case_alignment_report.json")
    output_contract = _safe_load_json("data/output_contract_report.json")
    data_adequacy = _safe_load_json("data/data_adequacy_report.json")
    alignment_check = _safe_load_json("data/alignment_check.json")
    integrity = _safe_load_json("data/integrity_audit_report.json")

    # Status from state (for backward compatibility)
    status = state.get("last_successful_review_verdict") or state.get("review_verdict") or "UNKNOWN"

    # Collect failed_gates from multiple sources (legacy logic preserved)
    failed_gates: List[str] = []
    gate_context = state.get("last_successful_gate_context") or state.get("last_gate_context")
    if isinstance(gate_context, dict):
        failed_gates = list(gate_context.get("failed_gates", []) or [])

    if isinstance(case_alignment, dict) and case_alignment.get("status") == "FAIL":
        failed_gates.extend(case_alignment.get("failures", []))

    if isinstance(output_contract, dict) and output_contract.get("missing"):
        failed_gates.append("output_contract_missing")

    pipeline_aborted = state.get("pipeline_aborted_reason")
    if pipeline_aborted:
        failed_gates.append(f"pipeline_aborted:{pipeline_aborted}")

    # Integrity critical count (preserved for backward compatibility)
    integrity_issues = integrity.get("issues", []) if isinstance(integrity, dict) else []
    integrity_critical_count = sum(
        1
        for issue in integrity_issues
        if str(issue.get("severity", "")).strip().lower() == "critical"
    )
    if integrity_critical_count > 0:
        failed_gates.append("integrity_critical")

    # Data adequacy summary (preserved)
    adequacy_summary = {}
    if isinstance(data_adequacy, dict):
        alignment = data_adequacy.get("quality_gates_alignment", {}) if isinstance(data_adequacy, dict) else {}
        alignment_summary = {}
        if isinstance(alignment, dict) and alignment:
            mapped = alignment.get("mapped_gates", {}) if isinstance(alignment, dict) else {}
            unmapped = alignment.get("unmapped_gates", {}) if isinstance(alignment, dict) else {}
            alignment_summary = {
                "status": alignment.get("status"),
                "mapped_gate_count": len(mapped) if isinstance(mapped, dict) else 0,
                "unmapped_gate_count": len(unmapped) if isinstance(unmapped, dict) else 0,
            }
        adequacy_summary = {
            "status": data_adequacy.get("status"),
            "reasons": data_adequacy.get("reasons", []),
            "recommendations": data_adequacy.get("recommendations", []),
            "consecutive_data_limited": data_adequacy.get("consecutive_data_limited"),
            "data_limited_threshold": data_adequacy.get("data_limited_threshold"),
            "threshold_reached": data_adequacy.get("threshold_reached"),
            "quality_gates_alignment": alignment_summary,
        }

    # Warnings (preserved)
    warnings: List[str] = []
    if state.get("qa_budget_exceeded"):
        warnings.append("QA_INCOMPLETE: QA budget exceeded; QA audit skipped.")

    # Contract and metrics for ceiling detection (preserved)
    contract = _safe_load_json("data/execution_contract.json") or state.get("execution_contract", {})
    metrics_report = _load_metrics_report(state)
    weights_report = _safe_load_json("data/weights.json")
    metric_pool: Dict[str, float] = {}
    metric_pool.update(_flatten_metrics(metrics_report))
    metric_pool = _merge_explicit_primary_metric(metric_pool, metrics_report)
    metric_pool.update(_flatten_metrics(weights_report))
    baseline_vs_model = _extract_baseline_vs_model(metric_pool)
    thresholds = {
        "auc": float(os.getenv("CEILING_DELTA_AUC", "0.02")),
        "f1": float(os.getenv("CEILING_DELTA_F1", "0.03")),
        "r2": float(os.getenv("CEILING_DELTA_R2", "0.02")),
        "error": float(os.getenv("CEILING_DELTA_ERROR", "0.03")),
        "cv_std": float(os.getenv("CEILING_CV_STD", "0.05")),
    }
    ceiling_info = _detect_metric_ceiling(baseline_vs_model, data_adequacy, thresholds)

    # =========================================================================
    # NEW: Unified Governance Verdict via Reducer
    # =========================================================================
    governance_verdict = compute_governance_verdict(
        output_contract_report=output_contract,
        state=state,
        contract=contract,
        integrity_report=integrity,
    )

    # Merge hard_failures and reasons from reducer
    reducer_hard_failures = governance_verdict.get("hard_failures", [])
    reducer_failed_gates = governance_verdict.get("failed_gates", [])
    reducer_reasons = governance_verdict.get("reasons", [])
    overall_status_global = governance_verdict.get("overall_status", "ok")

    # Merge reducer failed_gates into legacy failed_gates
    for gate in reducer_failed_gates:
        if gate and gate not in failed_gates:
            failed_gates.append(gate)

    # Canonicalize broad failed-area labels against reviewer packet evidence.
    failed_gates = _normalize_failed_gates_for_summary(state, failed_gates)

    # Counterfactual policy check
    counterfactual_policy = ""
    if isinstance(contract, dict):
        counterfactual_policy = str(contract.get("counterfactual_policy") or "")
    observational_only = counterfactual_policy == "observational_only"
    ceiling_detected = bool(ceiling_info.get("metric_ceiling_detected"))

    # =========================================================================
    # Run Outcome Derivation (unified via reducer)
    # =========================================================================
    run_outcome = derive_run_outcome(
        governance_verdict=governance_verdict,
        ceiling_detected=ceiling_detected,
        observational_only=observational_only,
    )

    # =========================================================================
    # LEGACY FALLBACK: Preserve existing critical token detection as secondary check
    # This ensures no regression while transitioning to reducer-based logic
    # =========================================================================
    failed_gates_lower = [str(item).lower() for item in failed_gates if item]
    critical_tokens = [
        "synthetic",
        "leakage",
        "security",
        "output_contract_missing",
        "dataframe_literal_overwrite",
        "unknown_columns_referenced",
        "df_column_assignment_forbidden",
    ]
    critical_hit = any(any(tok in gate for tok in critical_tokens) for gate in failed_gates_lower)
    output_missing = bool(isinstance(output_contract, dict) and output_contract.get("missing"))
    integrity_flagged = integrity_critical_count > 0

    # Secondary NO_GO check (legacy signals that might not be in reducer yet)
    if (
        run_outcome != "NO_GO"
        and (
            output_missing
            or critical_hit
            or integrity_flagged
            or status in {"REJECTED", "FAIL", "CRASH"}
            or pipeline_aborted
            or state.get("data_engineer_failed")
        )
    ):
        run_outcome = "NO_GO"

    review_board_payload = state.get("review_board_verdict")
    if not isinstance(review_board_payload, dict):
        loaded_board = _safe_load_json("data/review_board_verdict.json")
        review_board_payload = loaded_board if isinstance(loaded_board, dict) else {}
    metric_round_finalization = (
        review_board_payload.get("metric_round_finalization")
        if isinstance(review_board_payload, dict)
        and isinstance(review_board_payload.get("metric_round_finalization"), dict)
        else {}
    )
    metric_improvement_summary: Dict[str, Any] = {}
    if metric_round_finalization:
        metric_name = metric_round_finalization.get("metric_name")
        metric_improvement_summary = {
            "kept": metric_round_finalization.get("kept") or state.get("ml_improvement_kept"),
            "metric_name": metric_name,
            "baseline_metric": metric_round_finalization.get("baseline_metric"),
            "candidate_metric": metric_round_finalization.get("candidate_metric"),
            "final_metric_reported": metric_round_finalization.get("final_metric"),
            "final_metric_artifact": _extract_metric_value_by_name(metric_pool, metric_name),
            "reason": metric_round_finalization.get("force_finalize_reason") or "",
        }

    return {
        "run_id": state.get("run_id"),
        "status": status,
        "run_outcome": run_outcome,
        "failed_gates": list(dict.fromkeys(failed_gates)),
        "warnings": warnings,
        "budget_counters": state.get("budget_counters", {}),
        "data_adequacy": adequacy_summary,
        "metric_ceiling_detected": ceiling_detected,
        "ceiling_reason": ceiling_info.get("ceiling_reason"),
        "baseline_vs_model": baseline_vs_model.get("pairs", []),
        "metrics": {
            "baseline_vs_model": baseline_vs_model.get("pairs", []),
            "metric_pool_size": len(metric_pool),
        },
        "alignment_check": {
            "status": alignment_check.get("status"),
            "failure_mode": alignment_check.get("failure_mode"),
            "summary": alignment_check.get("summary"),
        } if isinstance(alignment_check, dict) and alignment_check else {},
        "integrity_critical_count": integrity_critical_count,
        "contract_views": _collect_contract_views(),
        # NEW: Governance verdict fields for traceability
        "overall_status_global": overall_status_global,
        "hard_failures": reducer_hard_failures,
        "governance_reasons": reducer_reasons,
        "metric_improvement": metric_improvement_summary,
    }
