import re
from typing import Any, Dict, Iterable, List, Tuple


_UNKNOWN_FAMILY = "unknown"

_FAMILY_PROFILES: Dict[str, Dict[str, str]] = {
    "classification": {
        "target_semantics": "class_label",
        "metric_family": "classification",
        "output_mode": "predictions",
    },
    "regression": {
        "target_semantics": "continuous_value",
        "metric_family": "regression",
        "output_mode": "predictions",
    },
    "forecasting": {
        "target_semantics": "time_series_target",
        "metric_family": "forecasting",
        "output_mode": "forecast",
    },
    "survival_analysis": {
        "target_semantics": "time_to_event",
        "metric_family": "survival",
        "output_mode": "risk_scores",
    },
    "ranking": {
        "target_semantics": "ordered_priority",
        "metric_family": "ranking",
        "output_mode": "ranking_scores",
    },
    "clustering": {
        "target_semantics": "group_membership",
        "metric_family": "clustering",
        "output_mode": "segments",
    },
    "optimization": {
        "target_semantics": "decision_variable",
        "metric_family": "optimization",
        "output_mode": "decisions",
    },
    "descriptive": {
        "target_semantics": "descriptive_only",
        "metric_family": "generic",
        "output_mode": "report",
    },
    _UNKNOWN_FAMILY: {
        "target_semantics": "unknown",
        "metric_family": "generic",
        "output_mode": "generic",
    },
}

_ALIAS_MAP: Dict[str, str] = {
    "classification": "classification",
    "classifier": "classification",
    "binary": "classification",
    "multiclass": "classification",
    "multi_class": "classification",
    "categorical_prediction": "classification",
    "regression": "regression",
    "regress": "regression",
    "continuous": "regression",
    "numeric_prediction": "regression",
    "forecasting": "forecasting",
    "forecast": "forecasting",
    "time_series": "forecasting",
    "timeseries": "forecasting",
    "time-series": "forecasting",
    "survival_analysis": "survival_analysis",
    "survival": "survival_analysis",
    "time_to_event": "survival_analysis",
    "time-to-event": "survival_analysis",
    "hazard": "survival_analysis",
    "censoring": "survival_analysis",
    "ranking": "ranking",
    "rank": "ranking",
    "prioritization": "ranking",
    "prioritisation": "ranking",
    "triage": "ranking",
    "targeting": "ranking",
    "recommendation": "ranking",
    "recommendations": "ranking",
    "clustering": "clustering",
    "cluster": "clustering",
    "segmentation": "clustering",
    "optimization": "optimization",
    "optimisation": "optimization",
    "optimize": "optimization",
    "optimise": "optimization",
    "prescriptive": "optimization",
    "decisioning": "optimization",
    "descriptive": "descriptive",
    "analytics": "descriptive",
    "eda": "descriptive",
    "reporting": "descriptive",
}

_TEXT_PATTERNS: List[tuple[str, str]] = [
    ("survival_analysis", r"\bsurvival\b|\btime[-_\s]?to[-_\s]?event\b|\bhazard\b|\bcensor"),
    ("forecasting", r"\bforecast|\btime[-_\s]?series\b|\btimeseries\b|\bbacktest\b|\bhorizon\b"),
    ("ranking", r"\branking\b|\brank\b|\bprioriti|\btriage\b|\btargeting\b|\brecommend"),
    ("clustering", r"\bcluster|\bsegmentation\b|\bsegment\b"),
    ("optimization", r"\boptimi[sz]|\bprescriptive\b|\ballocation\b|\bpricing\b|\bdecision"),
    ("classification", r"\bclassif|\bbinary\b|\bmulticlass\b|\bclass label\b"),
    ("regression", r"\bregress|\bcontinuous\b|\bnumeric prediction\b"),
    ("descriptive", r"\bdescriptive\b|\beda\b|\breporting\b|\binsight"),
]

_METRIC_PATTERNS: List[tuple[str, tuple[str, ...]]] = [
    ("survival_analysis", ("concordanceindex", "concordance", "integratedbrierscore", "ibs", "uncensored", "censored")),
    ("ranking", ("ndcg", "map", "mrr", "hitrate", "precisionatk", "recallatk", "kendall", "spearman")),
    ("forecasting", ("pinball", "coverage", "wmape", "smape", "mase")),
    ("classification", ("rocauc", "auc", "f1", "precision", "recall", "balancedaccuracy", "logloss", "averageprecision", "prauc")),
    ("regression", ("rmse", "mse", "mae", "r2", "rmsle", "mape", "smape")),
    ("clustering", ("silhouette", "daviesbouldin", "calinskiharabasz", "ari", "nmi")),
]

_OUTPUT_PATTERNS: List[tuple[str, tuple[str, ...]]] = [
    ("survival_analysis", ("survival", "hazard", "kaplan", "cox")),
    ("forecasting", ("forecast", "backtest")),
    ("ranking", ("ranking", "recommend", "priority")),
    ("clustering", ("cluster", "segment")),
    ("optimization", ("optimization", "allocation", "policy", "decision", "pricing")),
]

_METRIC_FAMILY_PATTERNS: List[tuple[str, tuple[str, ...]]] = [
    ("survival", ("concordanceindex", "concordance", "integratedbrierscore", "ibs", "maeuncensored", "censored")),
    ("forecasting", ("pinball", "coverage", "wmape", "smape", "mase", "wape")),
    ("ranking", ("ndcg", "map", "mrr", "hitrate", "precisionatk", "recallatk", "kendall", "spearman", "gini")),
    (
        "classification",
        ("rocauc", "auc", "f1", "precision", "recall", "balancedaccuracy", "logloss", "averageprecision", "prauc", "brier"),
    ),
    ("regression", ("rmse", "mse", "mae", "r2", "rmsle", "mape", "smape", "medae")),
    ("clustering", ("silhouette", "daviesbouldin", "calinskiharabasz", "ari", "nmi")),
    (
        "optimization",
        ("objectivevalue", "profit", "revenue", "cost", "uplift", "regret", "constraintviolation", "netvalue"),
    ),
]

_METRIC_PREFERENCE_TOKENS: Dict[str, Tuple[str, ...]] = {
    "classification": ("f1", "roc_auc", "auc", "pr_auc", "average_precision", "balanced_accuracy", "accuracy", "precision", "recall", "log_loss"),
    "regression": ("mae", "rmse", "mse", "mape", "smape", "r2"),
    "forecasting": ("wmape", "smape", "mase", "mae", "rmse", "pinball", "coverage"),
    "ranking": ("ndcg", "map", "mrr", "spearman", "kendall", "gini"),
    "survival": ("concordance_index", "c_index", "concordance", "integrated_brier_score", "ibs", "mae_uncensored"),
    "clustering": ("silhouette", "davies_bouldin", "calinski_harabasz", "ari", "nmi"),
    "optimization": ("objective_value", "net_value", "profit", "revenue", "cost", "uplift", "regret", "constraint_violation"),
    "generic": (),
}

_BASELINE_REQUIRED_FAMILIES = {"classification", "regression", "forecasting"}
_PRIMARY_METRIC_REQUIRED_FAMILIES = {
    "classification",
    "regression",
    "forecasting",
    "ranking",
    "survival_analysis",
    "clustering",
    "optimization",
}


def _normalize_token(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = text.replace("/", " ").replace("-", "_")
    return re.sub(r"[^a-z0-9_]+", "_", text).strip("_")


def normalize_problem_family(value: Any) -> str:
    token = _normalize_token(value)
    if not token:
        return _UNKNOWN_FAMILY
    if token in _ALIAS_MAP:
        return _ALIAS_MAP[token]
    for alias, family in _ALIAS_MAP.items():
        if alias and alias in token:
            return family
    return _UNKNOWN_FAMILY


def _family_from_any(value: Any) -> str:
    if isinstance(value, dict) and value.get("family"):
        return normalize_problem_family(value.get("family"))
    return normalize_problem_family(value)


def _normalized_metric_tokens(values: Iterable[Any] | None) -> List[str]:
    out: List[str] = []
    for value in values or []:
        token = _normalize_token(value)
        if token:
            out.append(token)
    return out


def normalize_metric_family(value: Any) -> str:
    token = _normalize_token(value)
    if not token:
        return "generic"
    for family, patterns in _METRIC_FAMILY_PATTERNS:
        if token == family or any(pattern == token for pattern in patterns):
            return family
    if token in {"survival_analysis", "survival"}:
        return "survival"
    if token in {"time_series", "timeseries"}:
        return "forecasting"
    if token in {"generic", "unknown", "other"}:
        return "generic"
    return token


def _first_known_family(candidates: Iterable[Any]) -> str:
    for candidate in candidates:
        family = normalize_problem_family(candidate)
        if family != _UNKNOWN_FAMILY:
            return family
    return _UNKNOWN_FAMILY


def _infer_family_from_text(texts: Iterable[Any]) -> str:
    combined = " ".join(str(text or "") for text in texts if str(text or "").strip()).lower()
    if not combined:
        return _UNKNOWN_FAMILY
    for family, pattern in _TEXT_PATTERNS:
        if re.search(pattern, combined):
            return family
    return _UNKNOWN_FAMILY


def _infer_family_from_metrics(metric_values: Iterable[Any]) -> str:
    metrics = _normalized_metric_tokens(metric_values)
    for family, patterns in _METRIC_PATTERNS:
        for metric in metrics:
            if any(pattern in metric for pattern in patterns):
                return family
    return _UNKNOWN_FAMILY


def _infer_family_from_outputs(required_outputs: Iterable[Any] | None) -> str:
    joined = " ".join(str(path or "") for path in (required_outputs or [])).lower()
    if not joined:
        return _UNKNOWN_FAMILY
    for family, tokens in _OUTPUT_PATTERNS:
        if any(token in joined for token in tokens):
            return family
    return _UNKNOWN_FAMILY


def metric_family_for_metric(metric_name: Any) -> str:
    token = _normalize_token(metric_name)
    if not token:
        return "generic"
    compact = token.replace("_", "")
    for family, patterns in _METRIC_FAMILY_PATTERNS:
        if any(pattern in token or pattern in compact for pattern in patterns):
            return family
    return "generic"


def metric_higher_is_better(metric_name: Any) -> bool:
    token = _normalize_token(metric_name)
    if not token:
        return True
    if any(pattern in token for pattern in ("loss", "error", "mae", "rmse", "mse", "mape", "smape", "brier", "regret", "violation")):
        return False
    return True


def metric_preference_tokens(metric_family: Any) -> Tuple[str, ...]:
    family = normalize_metric_family(metric_family)
    return _METRIC_PREFERENCE_TOKENS.get(family, ())


def problem_metric_families(value: Any) -> Tuple[str, ...]:
    family = _family_from_any(value)
    if family == "classification":
        return ("classification",)
    if family == "regression":
        return ("regression",)
    if family == "forecasting":
        return ("forecasting", "regression")
    if family == "survival_analysis":
        return ("survival", "ranking", "regression")
    if family == "ranking":
        return ("ranking",)
    if family == "clustering":
        return ("clustering",)
    if family == "optimization":
        return ("optimization", "regression", "ranking")
    return ("generic",)


def problem_requires_primary_metric(value: Any) -> bool:
    family = _family_from_any(value)
    return family in _PRIMARY_METRIC_REQUIRED_FAMILIES


def problem_prefers_baseline_metric(value: Any) -> bool:
    family = _family_from_any(value)
    return family in _BASELINE_REQUIRED_FAMILIES


def infer_problem_capabilities(
    *,
    objective_text: str = "",
    objective_type: Any = None,
    problem_type: Any = None,
    evaluation_spec: Dict[str, Any] | None = None,
    validation_requirements: Dict[str, Any] | None = None,
    required_outputs: Iterable[Any] | None = None,
    strategy: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    eval_spec = evaluation_spec if isinstance(evaluation_spec, dict) else {}
    validation = validation_requirements if isinstance(validation_requirements, dict) else {}
    strategy_payload = strategy if isinstance(strategy, dict) else {}

    explicit_candidates = [
        problem_type,
        eval_spec.get("problem_type"),
        objective_type,
        eval_spec.get("objective_type"),
        strategy_payload.get("analysis_type"),
        strategy_payload.get("problem_type"),
    ]
    family = _first_known_family(explicit_candidates)

    metrics = list(validation.get("metrics_to_report") or [])
    if validation.get("primary_metric"):
        metrics.append(validation.get("primary_metric"))
    if eval_spec.get("primary_metric"):
        metrics.append(eval_spec.get("primary_metric"))
    metrics.extend(eval_spec.get("metrics_to_report") or [])

    if family == _UNKNOWN_FAMILY:
        family = _infer_family_from_metrics(metrics)
    if family == _UNKNOWN_FAMILY:
        family = _infer_family_from_outputs(required_outputs)
    if family == _UNKNOWN_FAMILY:
        family = _infer_family_from_text(
            [
                objective_text,
                strategy_payload.get("title"),
                strategy_payload.get("objective"),
                strategy_payload.get("description"),
                eval_spec.get("target_type"),
                eval_spec.get("survival_time_col"),
                eval_spec.get("survival_event_col"),
            ]
        )

    profile = dict(_FAMILY_PROFILES.get(family, _FAMILY_PROFILES[_UNKNOWN_FAMILY]))
    return {
        "family": family,
        "target_semantics": profile.get("target_semantics", "unknown"),
        "metric_family": profile.get("metric_family", "generic"),
        "output_mode": profile.get("output_mode", "generic"),
    }


def resolve_problem_capabilities_from_contract(
    contract: Dict[str, Any] | None,
    *,
    objective_text: str = "",
    strategy: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    payload = contract if isinstance(contract, dict) else {}
    return infer_problem_capabilities(
        objective_text=objective_text or str(payload.get("business_objective") or ""),
        objective_type=payload.get("objective_type"),
        problem_type=(payload.get("objective_analysis") or {}).get("problem_type")
        if isinstance(payload.get("objective_analysis"), dict)
        else None,
        evaluation_spec=payload.get("evaluation_spec") if isinstance(payload.get("evaluation_spec"), dict) else {},
        validation_requirements=payload.get("validation_requirements")
        if isinstance(payload.get("validation_requirements"), dict)
        else {},
        required_outputs=payload.get("required_outputs") if isinstance(payload.get("required_outputs"), list) else [],
        strategy=strategy,
    )


def is_problem_family(value: Any, *families: str) -> bool:
    if isinstance(value, dict) and value.get("family"):
        family = str(value.get("family"))
    else:
        family = normalize_problem_family(value)
    normalized_families = {normalize_problem_family(item) for item in families if str(item or "").strip()}
    return family in normalized_families
