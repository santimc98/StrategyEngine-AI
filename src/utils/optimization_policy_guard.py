from __future__ import annotations

import copy
import json
import re
from typing import Any, Dict, List, Tuple

from src.utils.contract_validator import normalize_optimization_policy


_POLICY_BOOL_FIELDS = (
    "allow_model_switch",
    "allow_ensemble",
    "allow_hpo",
    "allow_feature_engineering",
    "allow_calibration",
)


def resolve_optimization_policy(source: Any) -> Dict[str, Any]:
    """Resolve optimization_policy from a contract, state/context, or direct policy."""
    if not isinstance(source, dict):
        return normalize_optimization_policy({})
    if any(key in source for key in _POLICY_BOOL_FIELDS) or "enabled" in source:
        return normalize_optimization_policy(source)
    contract = source.get("contract") if isinstance(source.get("contract"), dict) else source
    policy = contract.get("optimization_policy") if isinstance(contract, dict) else None
    if not isinstance(policy, dict) and isinstance(contract, dict):
        agents = contract.get("agents") if isinstance(contract.get("agents"), dict) else {}
        ml = agents.get("ml_engineer") if isinstance(agents.get("ml_engineer"), dict) else {}
        policy = ml.get("optimization_policy") if isinstance(ml.get("optimization_policy"), dict) else None
    return normalize_optimization_policy(policy)


def compact_optimization_policy_constraints(policy: Any) -> Dict[str, Any]:
    normalized = resolve_optimization_policy(policy)
    return {
        "enabled": bool(normalized.get("enabled")),
        "allow_model_switch": bool(normalized.get("allow_model_switch")),
        "allow_ensemble": bool(normalized.get("allow_ensemble")),
        "allow_hpo": bool(normalized.get("allow_hpo")),
        "allow_feature_engineering": bool(normalized.get("allow_feature_engineering")),
        "allow_calibration": bool(normalized.get("allow_calibration")),
    }


def _action_blob(action: Dict[str, Any]) -> str:
    parts: List[str] = []
    for key in (
        "technique",
        "action_family",
        "family",
        "objective",
    ):
        value = action.get(key)
        if value is not None:
            parts.append(str(value))
    params = action.get("params")
    if not isinstance(params, dict):
        params = action.get("concrete_params")
    if isinstance(params, dict):
        parts.append(json.dumps(params, sort_keys=True, default=str))
    blob = " ".join(parts).lower()
    return re.sub(r"[^a-z0-9_+.-]+", " ", blob)


def classify_optimization_action(action: Any) -> List[str]:
    if not isinstance(action, dict):
        return []
    blob = _action_blob(action)
    blob_words = blob.replace("_", " ")
    families: set[str] = set()
    action_family = str(action.get("action_family") or action.get("family") or "").strip().lower()
    if action_family == "ensemble_or_stacking":
        families.add("ensemble")
    if action_family == "feature_engineering":
        families.add("feature_engineering")
    if action_family == "hyperparameter_search":
        families.add("hpo")
    if action_family == "calibration":
        families.add("calibration")
    if action_family == "model_family_switch":
        families.add("model_switch")

    if re.search(r"\b(ensemble|ensembling|stacking|stacked|voting|blend|blending)\b", blob_words):
        families.add("ensemble")
    if re.search(r"\b(hpo|optuna|hyperparameter|grid search|random search|bayesian|search space|n trials)\b", blob_words):
        families.add("hpo")
    if re.search(r"\b(calibration|calibrated|isotonic|platt|temperature)\b", blob_words):
        families.add("calibration")
    if re.search(r"\b(model switch|model family|algorithm swap|switch model|change model|new model|fallback to)\b", blob_words):
        families.add("model_switch")
    if re.search(
        r"\b(feature engineering|target encoding|frequency encoding|rare category|"
        r"missing indicators|interaction|interactions|derived|polynomial|binning|"
        r"quantile binning|hashing|encoding scheme)\b",
        blob_words,
    ):
        families.add("feature_engineering")
    return sorted(families)


def optimization_policy_violations(action: Any, policy: Any) -> List[str]:
    normalized = resolve_optimization_policy(policy)
    families = classify_optimization_action(action)
    violations: List[str] = []
    if "ensemble" in families and not bool(normalized.get("allow_ensemble")):
        violations.append("allow_ensemble=false")
    if "feature_engineering" in families and not bool(normalized.get("allow_feature_engineering")):
        violations.append("allow_feature_engineering=false")
    if "hpo" in families and not bool(normalized.get("allow_hpo")):
        violations.append("allow_hpo=false")
    if "calibration" in families and not bool(normalized.get("allow_calibration")):
        violations.append("allow_calibration=false")
    if "model_switch" in families and not bool(normalized.get("allow_model_switch")):
        violations.append("allow_model_switch=false")
    return violations


def is_optimization_action_allowed(action: Any, policy: Any) -> bool:
    return not optimization_policy_violations(action, policy)


def filter_optimization_actions(actions: Any, policy: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not isinstance(actions, list):
        return [], []
    allowed: List[Dict[str, Any]] = []
    blocked: List[Dict[str, Any]] = []
    for item in actions:
        if not isinstance(item, dict):
            continue
        violations = optimization_policy_violations(item, policy)
        if violations:
            blocked.append(
                {
                    "technique": str(item.get("technique") or item.get("name") or "").strip(),
                    "action_family": str(item.get("action_family") or item.get("family") or "").strip(),
                    "violations": violations,
                }
            )
            continue
        allowed.append(copy.deepcopy(item))
    return allowed, blocked


def hypothesis_action_from_packet(packet: Any) -> Dict[str, Any]:
    if not isinstance(packet, dict):
        return {}
    hypothesis = packet.get("hypothesis") if isinstance(packet.get("hypothesis"), dict) else {}
    params = hypothesis.get("params") if isinstance(hypothesis.get("params"), dict) else {}
    action: Dict[str, Any] = {
        "technique": hypothesis.get("technique") or packet.get("technique"),
        "action_family": hypothesis.get("action_family") or packet.get("action_family"),
        "objective": hypothesis.get("objective"),
        "params": params,
    }
    if "blueprint_params" in hypothesis and isinstance(hypothesis.get("blueprint_params"), dict):
        action["concrete_params"] = hypothesis.get("blueprint_params")
    if hypothesis.get("code_change_hint"):
        action["code_change_hint"] = hypothesis.get("code_change_hint")
    return action
