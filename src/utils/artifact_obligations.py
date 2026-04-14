from __future__ import annotations

import copy
from typing import Any, Dict, List, Tuple


_DATA_ENGINEER_BINDING_ALIASES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("cleaned_dataset", ("cleaned_dataset", "clean_dataset")),
    ("enriched_dataset", ("enriched_dataset",)),
    ("schema_binding", ("schema_binding",)),
)


def _coerce_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _find_binding(
    artifact_requirements: Dict[str, Any],
    canonical_name: str,
    aliases: Tuple[str, ...],
) -> Tuple[str, Dict[str, Any]]:
    for alias in aliases:
        candidate = artifact_requirements.get(alias)
        if isinstance(candidate, dict):
            return alias, copy.deepcopy(candidate)
    return "", {}


def _collect_field_source_paths(
    payload: Dict[str, Any],
    *,
    contract_prefix: str,
    relative_prefix: str = "",
    out: Dict[str, str] | None = None,
) -> Dict[str, str]:
    result = out if isinstance(out, dict) else {}
    if not isinstance(payload, dict):
        return result
    for key, value in payload.items():
        key_token = str(key)
        rel_path = f"{relative_prefix}.{key_token}" if relative_prefix else key_token
        abs_path = f"{contract_prefix}.{key_token}"
        result[rel_path] = abs_path
        if isinstance(value, dict):
            _collect_field_source_paths(
                value,
                contract_prefix=abs_path,
                relative_prefix=rel_path,
                out=result,
            )
    return result


def build_data_engineer_artifact_obligations(contract: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a lossless extraction of DE-relevant artifact bindings already declared
    in the execution contract.

    This helper must not invent new semantics. Every emitted obligation is either:
    - a literal copy of a contract binding; or
    - a traceability field pointing back to the source contract path.
    """
    contract = contract if isinstance(contract, dict) else {}
    artifact_requirements = _coerce_dict(contract.get("artifact_requirements"))
    if not artifact_requirements:
        return {}

    bindings: List[Dict[str, Any]] = []
    emitted_keys: set[str] = set()
    for canonical_name, aliases in _DATA_ENGINEER_BINDING_ALIASES:
        binding_key, binding_payload = _find_binding(artifact_requirements, canonical_name, aliases)
        if not binding_payload:
            continue
        emitted_keys.add(binding_key)
        source_contract_path = f"artifact_requirements.{binding_key}"
        bindings.append(
            {
                "binding_name": canonical_name,
                "binding_contract_key": binding_key,
                "source_contract_path": source_contract_path,
                "declared_binding": binding_payload,
                "field_source_paths": _collect_field_source_paths(
                    binding_payload,
                    contract_prefix=source_contract_path,
                ),
            }
        )

    for binding_key, binding_payload in artifact_requirements.items():
        if binding_key in emitted_keys or not isinstance(binding_payload, dict) or not binding_payload:
            continue
        source_contract_path = f"artifact_requirements.{binding_key}"
        bindings.append(
            {
                "binding_name": str(binding_key),
                "binding_contract_key": str(binding_key),
                "source_contract_path": source_contract_path,
                "declared_binding": copy.deepcopy(binding_payload),
                "field_source_paths": _collect_field_source_paths(
                    binding_payload,
                    contract_prefix=source_contract_path,
                ),
            }
        )

    if not bindings:
        return {}

    return {
        "role": "data_engineer",
        "artifact_requirements_source_path": "artifact_requirements",
        "artifact_bindings": bindings,
    }
