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


_METRIC_CANONICAL_ALIASES: Dict[str, Tuple[str, ...]] = {
    "roc_auc": (
        "roc_auc",
        "roc-auc",
        "roc auc",
        "auc_roc",
        "auc-roc",
        "auc roc",
        "auc",
    ),
    "pr_auc": (
        "pr_auc",
        "pr-auc",
        "pr auc",
        "average_precision",
        "average precision",
        "avg_precision",
        "ap",
    ),
    "logloss": (
        "logloss",
        "log_loss",
        "log-loss",
        "binary_logloss",
        "cross_entropy",
        "cross-entropy",
        "crossentropy",
        "neg_log_loss",
        "nll",
    ),
    "accuracy": ("accuracy", "acc"),
    "f1": ("f1", "f1_score", "f1-score"),
    "precision": ("precision", "ppv"),
    "recall": ("recall", "tpr", "sensitivity"),
    "rmse": ("rmse", "root_mean_squared_error"),
    "rmsle": ("rmsle", "root_mean_squared_log_error"),
    "mae": ("mae", "mean_absolute_error", "l1"),
    "mape": ("mape", "mean_absolute_percentage_error"),
    "smape": ("smape", "symmetric_mean_absolute_percentage_error"),
    "r2": ("r2", "r_squared", "rsquared", "coefficient_of_determination"),
    "gini": ("gini", "normalized_gini"),
    "ndcg": ("ndcg",),
    "map": ("map", "mean_average_precision"),
    "mrr": ("mrr", "mean_reciprocal_rank"),
    "spearman": ("spearman", "spearmanr"),
    "kendall": ("kendall", "kendalltau"),
}

_PREFERRED_METRIC_KEY_TOKENS: Dict[str, int] = {
    "overall": 18,
    "oof": 16,
    "global": 12,
    "primary": 10,
    "leaderboard": 8,
    "mean": 7,
    "avg": 7,
    "average": 7,
    "cv": 6,
    "score": 2,
}

_DISFAVORED_METRIC_KEY_TOKENS: Dict[str, int] = {
    "std": 30,
    "stdev": 30,
    "stddev": 30,
    "stderr": 30,
    "variance": 30,
    "var": 22,
    "ci": 20,
    "lower": 18,
    "upper": 18,
    "fold": 14,
    "folds": 14,
    "count": 28,
    "counts": 28,
    "rows": 26,
    "nrows": 26,
    "num": 14,
    "seconds": 10,
    "time": 10,
    "latency": 10,
}


def _metric_tokens(text: Any) -> List[str]:
    raw_tokens = [tok for tok in re.split(r"[^0-9a-zA-Z]+", str(text or "").lower()) if tok]
    tokens: List[str] = []
    seen = set()
    for token in raw_tokens:
        variants = [token]
        if token.endswith("s") and len(token) > 3:
            variants.append(token[:-1])
        if token.endswith("ies") and len(token) > 4:
            variants.append(token[:-3] + "y")
        for variant in variants:
            cleaned = variant.strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            tokens.append(cleaned)
    return tokens


def canonicalize_metric_name(metric_name: str) -> str:
    norm = _norm_token(metric_name)
    if not norm:
        return ""

    token_set = set(_metric_tokens(metric_name))

    if "rmsle" in norm:
        return "rmsle"
    if "rmse" in norm:
        return "rmse"
    if "logloss" in norm or "crossentropy" in norm:
        return "logloss"
    if "smape" in norm:
        return "smape"
    if "mape" in norm:
        return "mape"
    if "mae" in norm:
        return "mae"
    if norm in {"r2", "rsquared"} or "rsquared" in norm or "r_squared" in str(metric_name).lower():
        return "r2"
    if "spearman" in norm:
        return "spearman"
    if "kendall" in norm:
        return "kendall"
    if "ndcg" in norm:
        return "ndcg"
    if norm == "map" or "meanaverageprecision" in norm:
        return "map"
    if "mrr" in norm or "meanreciprocalrank" in norm:
        return "mrr"
    if "gini" in norm:
        return "gini"
    if "accuracy" in norm or norm.endswith("acc"):
        return "accuracy"
    if re.fullmatch(r"f1(score)?", norm):
        return "f1"
    if "precision" in norm or norm == "ppv":
        return "precision"
    if "recall" in norm or "sensitivity" in norm or norm == "tpr":
        return "recall"
    if "averageprecision" in norm or ("pr" in token_set and "auc" in token_set):
        return "pr_auc"
    if "auc" in norm:
        return "roc_auc"

    for canonical, aliases in _METRIC_CANONICAL_ALIASES.items():
        normalized_aliases = {_norm_token(alias) for alias in aliases}
        if norm in normalized_aliases:
            return canonical
    return norm


def _metric_aliases(metric_name: str) -> List[str]:
    aliases: List[str] = []
    seen = set()

    def _add(value: Any) -> None:
        norm = _norm_token(value)
        if not norm or norm in seen:
            return
        seen.add(norm)
        aliases.append(norm)

    _add(metric_name)
    canonical = canonicalize_metric_name(metric_name)
    if canonical:
        _add(canonical)
        for alias in _METRIC_CANONICAL_ALIASES.get(canonical, ()):
            _add(alias)
    for token in _metric_tokens(metric_name):
        _add(token)
    return aliases


def _score_metric_candidate(metric_name: str, key: str) -> int | None:
    query_norm = _norm_token(metric_name)
    if not query_norm:
        return None

    key_norm = _norm_token(key)
    if not key_norm:
        return None

    query_canonical = canonicalize_metric_name(metric_name)
    key_canonical = canonicalize_metric_name(key)
    aliases = _metric_aliases(metric_name)
    key_tokens = set(_metric_tokens(key))

    if query_canonical and key_canonical and query_canonical != key_canonical:
        if query_norm != key_norm and query_norm not in key_norm and key_norm not in query_norm:
            return None

    score = 0
    if key_norm == query_norm:
        score += 160
    if query_canonical and key_canonical and query_canonical == key_canonical:
        score += 120
    for alias in aliases:
        if alias == key_norm:
            score += 80
        elif alias in key_tokens:
            score += 65
        elif alias and alias in key_norm:
            score += 45

    query_tokens = set(_metric_tokens(metric_name))
    if query_tokens:
        overlap = len(query_tokens & key_tokens)
        if overlap:
            score += overlap * 12

    for token, weight in _PREFERRED_METRIC_KEY_TOKENS.items():
        if token in key_tokens or token in key_norm:
            score += int(weight)
    for token, weight in _DISFAVORED_METRIC_KEY_TOKENS.items():
        if token in key_tokens or token in key_norm:
            score -= int(weight)

    if score <= 0:
        return None
    return score


def _metric_names_match(requested_metric: str, candidate_metric: str) -> bool:
    requested = str(requested_metric or "").strip()
    candidate = str(candidate_metric or "").strip()
    if not requested or not candidate:
        return False
    requested_norm = _norm_token(requested)
    candidate_norm = _norm_token(candidate)
    if requested_norm and requested_norm == candidate_norm:
        return True
    requested_canonical = canonicalize_metric_name(requested)
    candidate_canonical = canonicalize_metric_name(candidate)
    if requested_canonical and candidate_canonical and requested_canonical == candidate_canonical:
        return True
    return False


def _resolve_explicit_primary_metric(metrics_json: Dict[str, Any], metric_name: str) -> Dict[str, Any]:
    if not isinstance(metrics_json, dict):
        return {}

    explicit_candidates: List[Tuple[str, str, Any]] = []

    top_level_metric = metrics_json.get("primary_metric")
    if isinstance(top_level_metric, dict):
        explicit_candidates.append(
            (
                "primary_metric",
                str(
                    top_level_metric.get("name")
                    or top_level_metric.get("metric")
                    or top_level_metric.get("id")
                    or metric_name
                ).strip(),
                top_level_metric.get("value"),
            )
        )
    else:
        explicit_candidates.append(
            (
                "primary_metric_value",
                str(metrics_json.get("primary_metric_name") or top_level_metric or metric_name).strip(),
                metrics_json.get("primary_metric_value"),
            )
        )

    model_perf = metrics_json.get("model_performance")
    if isinstance(model_perf, dict):
        model_perf_primary = model_perf.get("primary_metric")
        if isinstance(model_perf_primary, dict):
            explicit_candidates.append(
                (
                    "model_performance.primary_metric",
                    str(
                        model_perf_primary.get("name")
                        or model_perf_primary.get("metric")
                        or model_perf_primary.get("id")
                        or metric_name
                    ).strip(),
                    model_perf_primary.get("value"),
                )
            )
        explicit_candidates.append(
            (
                "model_performance.primary_metric_value",
                str(model_perf.get("primary_metric_name") or model_perf_primary or metric_name).strip(),
                model_perf.get("primary_metric_value"),
            )
        )

    for matched_key, candidate_name, raw_value in explicit_candidates:
        value = _coerce_float(raw_value)
        if value is None:
            continue
        if metric_name and candidate_name and not _metric_names_match(metric_name, candidate_name):
            continue
        chosen_name = candidate_name or str(metric_name or "").strip() or matched_key
        return {
            "metric_name": chosen_name,
            "canonical_name": canonicalize_metric_name(chosen_name),
            "matched_key": matched_key,
            "value": float(value),
            "score": 100000,
        }

    return {}


def resolve_metric_value(metrics_json: Dict[str, Any], metric_name: str) -> Dict[str, Any]:
    if not isinstance(metrics_json, dict):
        return {}

    explicit_primary = _resolve_explicit_primary_metric(metrics_json, metric_name)
    if explicit_primary:
        return explicit_primary

    flat = flatten_numeric_metrics(metrics_json)
    best_key: str | None = None
    best_value: float | None = None
    best_score: int | None = None

    for key, value in flat:
        score = _score_metric_candidate(metric_name, key)
        if score is None:
            continue
        if best_score is None or score > best_score or (score == best_score and len(key) < len(best_key or key)):
            best_key = str(key)
            best_value = float(value)
            best_score = int(score)

    if best_key is None or best_value is None:
        return {}

    return {
        "metric_name": str(metric_name or "").strip() or best_key,
        "canonical_name": canonicalize_metric_name(metric_name or best_key),
        "matched_key": best_key,
        "value": float(best_value),
        "score": int(best_score or 0),
    }


def flatten_numeric_metrics(payload: Any, prefix: str = "") -> List[Tuple[str, float]]:
    items: List[Tuple[str, float]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            next_key = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, (int, float)):
                items.append((next_key, float(value)))
            elif isinstance(value, str):
                coerced = _coerce_float(value)
                if coerced is not None:
                    items.append((next_key, float(coerced)))
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
    resolved = resolve_metric_value(metrics_json, metric_name)
    value = resolved.get("value") if isinstance(resolved, dict) else None
    return float(value) if value is not None else None


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
