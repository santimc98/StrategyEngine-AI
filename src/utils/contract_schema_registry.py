import copy
from typing import Any, Dict, List


CONTRACT_FIELD_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "column_dtype_targets": {
        "type": "dict[str, dict]",
        "required_keys_per_entry": ["target_dtype"],
        "example": {
            "age": {"target_dtype": "float64", "nullable": False},
            "sex": {"target_dtype": "string"},
        },
        "common_errors": [
            "entry uses key 'type' instead of 'target_dtype'",
            "entry uses key 'dtype' instead of 'target_dtype'",
            "entry is a plain string dtype instead of object",
        ],
    },
    "required_feature_selectors": {
        "type": "list[dict]",
        "required_keys_per_entry": ["type"],
        "example": [
            {"type": "prefix", "value": "feature_"},
            {"type": "regex", "pattern": "^pixel_\\d+$"},
        ],
        "common_errors": [
            "selector is object but missing 'type'",
            "selector provided as shorthand string",
        ],
    },
    "gate_list": {
        "type": "list[dict]",
        "required_keys_per_entry": ["name", "severity", "params"],
        "example": [
            {"name": "no_nulls_target", "severity": "HARD", "params": {"column": "target"}},
        ],
        "common_errors": [
            "gate missing consumable identifier",
            "gate provided as plain string",
        ],
    },
}


RULE_TO_REPAIR_ACTION: Dict[str, str] = {
    "contract.column_dtype_targets": (
        "Each column_dtype_targets entry must be an object with key 'target_dtype' (NOT 'type'). "
        "Example: {\"target_dtype\": \"float64\", \"nullable\": false}. "
        "Replace all {\"type\": X} with {\"target_dtype\": X}."
    ),
    "contract.clean_dataset_required_feature_selectors": (
        "required_feature_selectors must be list[object], and each selector object must include key 'type'. "
        "Valid selector types include regex/prefix/suffix/contains/list/all_columns_except/prefix_numeric_range."
    ),
    "contract.cleaning_gates": (
        "Gate lists must use executable objects: {\"name\": str, \"severity\": \"HARD\"|\"SOFT\", \"params\": object}."
    ),
    "contract.qa_gates": (
        "Gate lists must use executable objects: {\"name\": str, \"severity\": \"HARD\"|\"SOFT\", \"params\": object}."
    ),
    "contract.reviewer_gates": (
        "Gate lists must use executable objects: {\"name\": str, \"severity\": \"HARD\"|\"SOFT\", \"params\": object}."
    ),
}


def build_contract_schema_examples_text() -> str:
    dtype_example = CONTRACT_FIELD_SCHEMAS["column_dtype_targets"]["example"]
    selector_example = CONTRACT_FIELD_SCHEMAS["required_feature_selectors"]["example"]
    gate_example = CONTRACT_FIELD_SCHEMAS["gate_list"]["example"]
    return (
        "Schema examples for non-obvious fields (copy key names exactly):\n"
        f"- column_dtype_targets: {dtype_example}\n"
        f"- required_feature_selectors: {selector_example}\n"
        f"- gate_list_example (applies to cleaning_gates/qa_gates/reviewer_gates): {gate_example}"
    )


def get_contract_schema_repair_action(rule: str) -> str:
    return RULE_TO_REPAIR_ACTION.get(str(rule or "").strip(), "")


def _infer_selector_type(selector: Dict[str, Any]) -> str:
    if not isinstance(selector, dict):
        return ""
    if isinstance(selector.get("type"), str) and selector.get("type").strip():
        return str(selector.get("type")).strip().lower()
    if selector.get("pattern") or selector.get("regex"):
        return "regex"
    if selector.get("prefix"):
        return "prefix"
    if selector.get("suffix"):
        return "suffix"
    if selector.get("contains"):
        return "contains"
    if isinstance(selector.get("columns"), list):
        return "list"
    if isinstance(selector.get("except_columns"), list):
        return "all_columns_except"
    if selector.get("start") is not None and selector.get("end") is not None and selector.get("prefix"):
        return "prefix_numeric_range"
    return ""


def _repair_column_dtype_targets(contract: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(contract, dict):
        return contract
    targets = contract.get("column_dtype_targets")
    if not isinstance(targets, dict):
        return contract
    repaired: Dict[str, Dict[str, Any]] = {}
    for raw_col, raw_spec in targets.items():
        col = str(raw_col or "").strip()
        if not col:
            continue
        if isinstance(raw_spec, str):
            repaired[col] = {"target_dtype": raw_spec.strip() or "preserve"}
            continue
        if not isinstance(raw_spec, dict):
            repaired[col] = {"target_dtype": "preserve"}
            continue
        spec = dict(raw_spec)
        if not str(spec.get("target_dtype") or "").strip():
            for alias in ("type", "dtype", "data_type", "targetType"):
                alias_value = spec.get(alias)
                if isinstance(alias_value, str) and alias_value.strip():
                    spec["target_dtype"] = alias_value.strip()
                    if alias != "target_dtype":
                        spec.pop(alias, None)
                    break
        if not str(spec.get("target_dtype") or "").strip():
            spec["target_dtype"] = "preserve"
        repaired[col] = spec
    contract["column_dtype_targets"] = repaired
    return contract


def _repair_required_feature_selectors(contract: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(contract, dict):
        return contract
    artifact_requirements = contract.get("artifact_requirements")
    cleaned_dataset = None
    if isinstance(artifact_requirements, dict):
        candidate = artifact_requirements.get("cleaned_dataset")
        if not isinstance(candidate, dict):
            candidate = artifact_requirements.get("clean_dataset")
        if isinstance(candidate, dict):
            cleaned_dataset = candidate
    if not isinstance(cleaned_dataset, dict):
        return contract
    selectors = cleaned_dataset.get("required_feature_selectors")
    if selectors is None:
        return contract
    if isinstance(selectors, dict):
        selectors = [selectors]
    elif isinstance(selectors, str):
        selectors = [{"selector": selectors}]
    if not isinstance(selectors, list):
        return contract
    normalized: List[Dict[str, Any]] = []
    for item in selectors:
        if isinstance(item, str):
            item = {"selector": item}
        if not isinstance(item, dict):
            continue
        selector = dict(item)
        if isinstance(selector.get("selector"), str) and ":" in selector.get("selector"):
            head, tail = selector.get("selector").split(":", 1)
            head = head.strip().lower()
            tail = tail.strip()
            if head in {"regex", "pattern"}:
                selector = {"type": "regex", "pattern": tail}
            elif head == "prefix":
                selector = {"type": "prefix", "value": tail}
            elif head == "suffix":
                selector = {"type": "suffix", "value": tail}
            elif head == "contains":
                selector = {"type": "contains", "value": tail}
        selector_type = _infer_selector_type(selector)
        if not selector_type:
            continue
        selector["type"] = selector_type
        if selector_type == "regex":
            pattern = selector.get("pattern") or selector.get("value") or selector.get("regex")
            if isinstance(pattern, str) and pattern.strip():
                selector["pattern"] = pattern.strip()
                selector.pop("value", None)
                selector.pop("regex", None)
            else:
                continue
        if selector_type in {"prefix", "suffix", "contains"}:
            value = selector.get("value") or selector.get(selector_type)
            if isinstance(value, str) and value.strip():
                selector["value"] = value.strip()
            else:
                continue
        normalized.append(selector)
    cleaned_dataset["required_feature_selectors"] = normalized
    return contract


def _repair_gate_lists(contract: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(contract, dict):
        return contract
    for key in ("cleaning_gates", "qa_gates", "reviewer_gates"):
        gates = contract.get(key)
        if not isinstance(gates, list):
            continue
        normalized = []
        for idx, gate in enumerate(gates):
            if isinstance(gate, str):
                name = gate.strip()
                if not name:
                    continue
                normalized.append({"name": name, "severity": "HARD", "params": {}})
                continue
            if not isinstance(gate, dict):
                continue
            payload = copy.deepcopy(gate)
            name = ""
            for alias in ("name", "id", "gate", "metric", "check", "rule", "title", "label"):
                alias_value = payload.get(alias)
                if isinstance(alias_value, str) and alias_value.strip():
                    name = alias_value.strip()
                    break
            if not name:
                name = f"{key}_{idx + 1}"
            severity = str(payload.get("severity") or payload.get("level") or "HARD").strip().upper()
            if severity not in {"HARD", "SOFT"}:
                severity = "HARD"
            params = payload.get("params")
            if not isinstance(params, dict):
                params = {}
            for semantic_key in ("metric", "check", "rule", "threshold", "condition"):
                if semantic_key in payload and semantic_key not in params:
                    params[semantic_key] = payload.get(semantic_key)
            gate_obj = {"name": name, "severity": severity, "params": params}
            for semantic_key in (
                "condition",
                "evidence_required",
                "action_if_fail",
                "action_type",
                "column_phase",
                "final_state",
                "applies_to",
                "phase_scope",
                "active_in_phases",
                "phases",
                "review_phases",
                "stage_scope",
            ):
                if semantic_key in payload and payload.get(semantic_key) not in (None, ""):
                    gate_obj[semantic_key] = payload.get(semantic_key)
            normalized.append(gate_obj)
        if normalized:
            contract[key] = normalized
    return contract


def apply_contract_schema_registry_repairs(contract: Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(contract, dict):
        return {}
    repaired = copy.deepcopy(contract)
    repaired = _repair_column_dtype_targets(repaired)
    repaired = _repair_required_feature_selectors(repaired)
    repaired = _repair_gate_lists(repaired)
    return repaired
