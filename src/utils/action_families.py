from __future__ import annotations

import re
from typing import Any, Dict, List


ACTION_FAMILIES: List[str] = [
    "feature_engineering",
    "model_family_switch",
    "hyperparameter_search",
    "ensemble_or_stacking",
    "calibration",
    "class_imbalance",
    "loss_objective_adjustment",
]


_FAMILY_KEYWORDS: Dict[str, List[str]] = {
    "feature_engineering": [
        "feature",
        "interaction",
        "encoding",
        "binning",
        "transform",
        "derived",
        "aggregation",
        "polynomial",
        "target encoding",
        "hashing",
    ],
    "model_family_switch": [
        "model switch",
        "model family",
        "algorithm swap",
        "switch model",
        "change model",
        "new model",
    ],
    "hyperparameter_search": [
        "hyperparameter",
        "hpo",
        "grid search",
        "random search",
        "bayesian",
        "optuna",
        "tuning",
    ],
    "ensemble_or_stacking": [
        "ensemble",
        "stacking",
        "blend",
        "blending",
        "voting",
        "bagging",
    ],
    "calibration": [
        "calibration",
        "calibrate",
        "isotonic",
        "platt",
        "sigmoid",
        "threshold tuning",
    ],
    "class_imbalance": [
        "imbalance",
        "class weight",
        "smote",
        "oversample",
        "undersample",
        "resampling",
        "focal",
    ],
    "loss_objective_adjustment": [
        "loss",
        "objective",
        "custom objective",
        "cost-sensitive",
        "asymmetric",
        "margin",
    ],
}


_ACTION_FAMILY_GUIDANCE: Dict[str, List[str]] = {
    "feature_engineering": [
        "Apply transforms inside train/CV boundaries to avoid leakage.",
        "Keep feature generation deterministic and reproducible.",
        "Touch only feature construction/selectors; preserve artifact outputs.",
    ],
    "model_family_switch": [
        "Switch model family only if hypothesis explicitly requests it.",
        "Preserve training split/CV protocol and all required outputs.",
        "Keep preprocessing and persistence contracts stable.",
    ],
    "hyperparameter_search": [
        "Constrain search space and iterations to current runtime budget.",
        "Avoid broad replanning; only tune parameters tied to hypothesis.",
        "Keep model/data pipeline structure unchanged.",
    ],
    "ensemble_or_stacking": [
        "Build ensemble as an additive patch over the working baseline.",
        "Preserve baseline model path as fallback in the script.",
        "Keep output schema/paths exactly as contract requires.",
    ],
    "calibration": [
        "Add calibration post-fit without changing data split policy.",
        "Preserve core model training flow and contract outputs.",
        "Do not change problem framing or target semantics.",
    ],
    "class_imbalance": [
        "Apply class imbalance handling only in training folds.",
        "Avoid leakage from global resampling before split/CV.",
        "Preserve prediction/export interfaces and required artifacts.",
    ],
    "loss_objective_adjustment": [
        "Adjust loss/objective minimally, keeping model pipeline stable.",
        "Maintain metric computation and output contracts unchanged.",
        "Preserve calibration/export behavior unless hypothesis says otherwise.",
    ],
}


def normalize_action_family(value: Any) -> str:
    token = str(value or "").strip().lower()
    return token if token in ACTION_FAMILIES else "feature_engineering"


def classify_action_family(hypothesis_packet: Dict[str, Any] | None) -> str:
    packet = hypothesis_packet if isinstance(hypothesis_packet, dict) else {}
    hypothesis = packet.get("hypothesis") if isinstance(packet.get("hypothesis"), dict) else {}

    explicit_candidates = [
        hypothesis.get("action_family"),
        hypothesis.get("family"),
        packet.get("action_family"),
        packet.get("family"),
    ]
    for candidate in explicit_candidates:
        normalized = normalize_action_family(candidate)
        if normalized != "feature_engineering" or str(candidate or "").strip().lower() == "feature_engineering":
            return normalized

    parts: List[str] = []
    for field in (
        hypothesis.get("technique"),
        hypothesis.get("objective"),
        packet.get("action"),
    ):
        if field:
            parts.append(str(field))
    params = hypothesis.get("params") if isinstance(hypothesis.get("params"), dict) else {}
    if params:
        parts.append(" ".join([str(k) for k in params.keys()]))
        parts.append(" ".join([str(v) for v in params.values() if isinstance(v, str)]))

    blob = " ".join(parts).strip().lower()
    if not blob:
        return "feature_engineering"
    blob = re.sub(r"\s+", " ", blob)

    for family in ACTION_FAMILIES:
        keywords = _FAMILY_KEYWORDS.get(family, [])
        if any(keyword in blob for keyword in keywords):
            return family

    return "feature_engineering"


def get_action_family_guidance(action_family: str) -> List[str]:
    normalized = normalize_action_family(action_family)
    guidance = _ACTION_FAMILY_GUIDANCE.get(normalized, _ACTION_FAMILY_GUIDANCE["feature_engineering"])
    return [str(item) for item in guidance]
