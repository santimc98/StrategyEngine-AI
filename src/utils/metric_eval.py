from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _norm_token(text: Any) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "", str(text or "").lower())


def flatten_numeric_metrics(payload: Any, prefix: str = "") -> List[Tuple[str, float]]:
    items: List[Tuple[str, float]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            next_key = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, (int, float)):
                items.append((next_key, float(value)))
            elif isinstance(value, dict):
                items.extend(flatten_numeric_metrics(value, next_key))
            elif isinstance(value, list) and value:
                # Compute mean of numeric lists (e.g. fold scores like
                # cv_results.roc_auc: [0.91, 0.92, 0.90]) and expose
                # as a scalar so metric extraction can find it.
                nums = [float(v) for v in value if isinstance(v, (int, float))]
                if nums and len(nums) == len(value):
                    mean_val = sum(nums) / len(nums)
                    items.append((f"{next_key}_mean", mean_val))
    return items


def extract_primary_metric(metrics_json: Dict[str, Any], metric_name: str) -> Optional[float]:
    if not isinstance(metrics_json, dict):
        return None
    metric = str(metric_name or "").strip()
    preferred_keys: List[str] = []
    if metric:
        preferred_keys.extend(
            [
                metric,
                f"cv_{metric}",
                f"{metric}_mean",
                f"mean_{metric}",
                f"primary_{metric}",
            ]
        )
    preferred_keys.extend(["primary_metric_value", "metric_value", "score", "value", "cv_mean"])
    preferred_norm = [_norm_token(key) for key in preferred_keys if str(key).strip()]

    flat = flatten_numeric_metrics(metrics_json)
    for key, value in flat:
        token = _norm_token(key)
        if any(pref and pref == token for pref in preferred_norm):
            return float(value)
    for key, value in flat:
        token = _norm_token(key)
        if any(pref and pref in token for pref in preferred_norm):
            return float(value)
    return None


def extract_stability_signals(critique_packet: Dict[str, Any] | None) -> Dict[str, Any]:
    critique_packet = critique_packet if isinstance(critique_packet, dict) else {}
    validation = (
        critique_packet.get("validation_signals")
        if isinstance(critique_packet.get("validation_signals"), dict)
        else {}
    )
    cv_block = validation.get("cv") if isinstance(validation.get("cv"), dict) else {}
    error_modes = critique_packet.get("error_modes") if isinstance(critique_packet.get("error_modes"), list) else []
    error_mode_ids = {
        str(item.get("id") or "").strip().lower()
        for item in error_modes
        if isinstance(item, dict) and str(item.get("id") or "").strip()
    }
    cv_std = _coerce_float(cv_block.get("cv_std"))
    if cv_std is None:
        cv_std = _coerce_float(validation.get("cv_std"))
    gap = _coerce_float(validation.get("generalization_gap"))
    variance_level = str(cv_block.get("variance_level") or "").strip().lower()
    return {
        "cv_std": abs(float(cv_std)) if cv_std is not None else None,
        "generalization_gap_abs": abs(float(gap)) if gap is not None else None,
        "variance_level": variance_level,
        "error_mode_ids": sorted(list(error_mode_ids)),
    }


def stability_ok(
    critique_packet: Dict[str, Any] | None,
    *,
    max_cv_std: Optional[float] = None,
    max_generalization_gap: Optional[float] = 0.02,
    disallow_high_variance: bool = True,
) -> bool:
    signals = extract_stability_signals(critique_packet)
    error_mode_ids = set(signals.get("error_mode_ids") or [])
    if "fold_instability" in error_mode_ids or "generalization_gap_high" in error_mode_ids:
        return False
    if disallow_high_variance and str(signals.get("variance_level") or "").lower() == "high":
        return False
    cv_std = _coerce_float(signals.get("cv_std"))
    if max_cv_std is not None and cv_std is not None and cv_std > float(max_cv_std):
        return False
    gap = _coerce_float(signals.get("generalization_gap_abs"))
    if max_generalization_gap is not None and gap is not None and gap > float(max_generalization_gap):
        return False
    return True


def select_incumbent(
    candidates: List[Dict[str, Any]] | None,
    *,
    higher_is_better: bool,
    min_delta: float = 0.0,
) -> Dict[str, Any]:
    rows = [row for row in (candidates or []) if isinstance(row, dict)]
    if not rows:
        return {"selected_label": "none", "reason": "no_candidates", "candidates": []}

    normalized: List[Dict[str, Any]] = []
    baseline_metric: float | None = None
    for idx, row in enumerate(rows):
        label = str(row.get("label") or f"candidate_{idx}")
        metric_value = _coerce_float(row.get("metric_value"))
        if baseline_metric is None and str(label).lower() in {"baseline", "incumbent"}:
            baseline_metric = metric_value
        stable = bool(row.get("stability_ok", True))
        cv_std = _coerce_float(row.get("cv_std"))
        gap = _coerce_float(row.get("generalization_gap_abs"))
        cost = _coerce_float(row.get("cost"))
        normalized.append(
            {
                "label": label,
                "metric_value": metric_value,
                "stability_ok": stable,
                "cv_std": abs(float(cv_std)) if cv_std is not None else None,
                "generalization_gap_abs": abs(float(gap)) if gap is not None else None,
                "cost": max(0.0, float(cost)) if cost is not None else 0.0,
                "_idx": idx,
            }
        )

    if baseline_metric is None and normalized:
        baseline_metric = _coerce_float(normalized[0].get("metric_value"))

    scored: List[Dict[str, Any]] = []
    for row in normalized:
        metric_value = _coerce_float(row.get("metric_value"))
        if metric_value is None:
            continue
        if baseline_metric is None:
            delta = 0.0
        else:
            delta = float(metric_value) - float(baseline_metric)
        signed_delta = float(delta) if higher_is_better else -float(delta)
        meets_delta = bool(signed_delta >= float(min_delta))
        row_scored = dict(row)
        row_scored["signed_delta"] = signed_delta
        row_scored["meets_min_delta"] = meets_delta
        row_scored["eligible"] = bool(row_scored.get("stability_ok")) and bool(meets_delta)
        scored.append(row_scored)

    if not scored:
        return {"selected_label": "none", "reason": "no_numeric_metric", "candidates": normalized}

    def _rank_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
        metric_value = _coerce_float(row.get("metric_value"))
        metric_rank = -float(metric_value) if higher_is_better and metric_value is not None else (
            float(metric_value) if metric_value is not None else float("inf")
        )
        cv_std = _coerce_float(row.get("cv_std"))
        gap = _coerce_float(row.get("generalization_gap_abs"))
        cost = _coerce_float(row.get("cost"))
        return (
            0 if bool(row.get("eligible")) else 1,
            metric_rank,
            float(cv_std) if cv_std is not None else float("inf"),
            float(gap) if gap is not None else float("inf"),
            float(cost) if cost is not None else float("inf"),
            str(row.get("label") or ""),
            int(row.get("_idx") or 0),
        )

    winner = sorted(scored, key=_rank_key)[0]
    return {
        "selected_label": str(winner.get("label") or "none"),
        "reason": "eligible_best" if bool(winner.get("eligible")) else "best_available",
        "baseline_metric": baseline_metric,
        "min_delta": float(min_delta),
        "higher_is_better": bool(higher_is_better),
        "winner": winner,
        "candidates": scored,
    }

