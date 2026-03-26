import json
import os
from typing import Any, Dict, List, Optional

from src.utils.contract_accessors import get_outcome_columns


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
    return value if isinstance(value, list) else []


def _unique_strings(items: List[Any]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if not item:
            continue
        text = str(item)
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _extract_contract(state: Dict[str, Any]) -> Dict[str, Any]:
    snapshot = state.get("execution_contract_snapshot")
    if isinstance(snapshot, dict):
        snap_contract = snapshot.get("execution_contract")
        if isinstance(snap_contract, dict) and snap_contract:
            return snap_contract
    contract = _as_dict(state.get("execution_contract"))
    if contract:
        return contract
    # Fallback: contract may only exist on disk (not in state dict)
    try:
        with open(os.path.join("data", "execution_contract.json"), "r", encoding="utf-8") as f:
            disk_contract = json.load(f)
        if isinstance(disk_contract, dict) and disk_contract:
            return disk_contract
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return {}


def _extract_objective_type(contract: Dict[str, Any], state: Dict[str, Any]) -> Optional[str]:
    eval_spec = _as_dict(contract.get("evaluation_spec"))
    obj_analysis = _as_dict(contract.get("objective_analysis"))
    strategy_spec = _as_dict(state.get("strategy_spec"))
    return (
        eval_spec.get("objective_type")
        or obj_analysis.get("problem_type")
        or strategy_spec.get("objective_type")
        or contract.get("objective_type")
    )


def _extract_target_columns(contract: Dict[str, Any]) -> List[str]:
    targets: List[Any] = []

    # V4.1 canonical resolver first (handles multiple column_roles formats safely).
    try:
        targets.extend(get_outcome_columns(contract))
    except Exception:
        pass

    # Contract-explicit target hints (LLM/contract derived).
    if contract.get("target_column"):
        targets.append(contract.get("target_column"))
    targets.extend(_as_list(contract.get("target_columns")))

    eval_spec = _as_dict(contract.get("evaluation_spec"))
    if eval_spec.get("target_column"):
        targets.append(eval_spec.get("target_column"))
    targets.extend(_as_list(eval_spec.get("outcome_columns")))
    return _unique_strings(targets)


def _extract_required_outputs(contract: Dict[str, Any]) -> List[str]:
    required = _as_list(contract.get("required_outputs"))
    if required:
        return _unique_strings(required)
    artifact_reqs = _as_dict(contract.get("artifact_requirements"))
    files = _as_list(artifact_reqs.get("required_files"))
    normalized: List[str] = []
    for entry in files:
        if not entry:
            continue
        if isinstance(entry, dict):
            path = entry.get("path") or entry.get("output") or entry.get("artifact")
            if path:
                normalized.append(str(path))
            continue
        normalized.append(str(entry))
    return _unique_strings(normalized)


def _extract_contract_columns(contract: Dict[str, Any], max_cols: int = 40) -> Dict[str, Any]:
    columns_list = [str(c) for c in _as_list(contract.get("canonical_columns")) if c]
    return {
        "n_cols": len(columns_list) if columns_list else None,
        "sample": columns_list[:max_cols],
    }


def build_run_facts_pack(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a compact, deterministic snapshot of run facts for agent context.
    """
    state = state or {}
    contract = _extract_contract(state)
    dialect = {
        "sep": state.get("csv_sep"),
        "decimal": state.get("csv_decimal"),
        "encoding": state.get("csv_encoding"),
    }
    column_info = _extract_contract_columns(contract)
    dataset_scale_hints = state.get("dataset_scale_hints")
    if not isinstance(dataset_scale_hints, dict):
        dataset_scale_hints = {}

    run_facts = {
        "run_id": state.get("run_id"),
        "input_csv_path": state.get("csv_path"),
        "contract_source": state.get("execution_contract_source") or "execution_contract",
        "contract_signature": state.get("execution_contract_signature"),
        "contract_min_signature": None,
        "contract_version": contract.get("contract_version"),
        "dialect": dialect,
        "dataset_scale_hints": dataset_scale_hints or None,
        "objective_type": _extract_objective_type(contract, state),
        "target_columns": _extract_target_columns(contract),
        "required_outputs": _extract_required_outputs(contract),
        "visual_requirements": _as_dict(_as_dict(contract.get("artifact_requirements")).get("visual_requirements"))
        or None,
        "decisioning_requirements": _as_dict(contract.get("decisioning_requirements")) or None,
        "column_inventory": column_info,
        "iteration": {
            "iteration_count": state.get("iteration_count"),
            "data_engineer_attempt": state.get("data_engineer_attempt"),
            "ml_engineer_attempt": state.get("ml_engineer_attempt"),
            "reviewer_iteration": state.get("reviewer_iteration"),
        },
    }
    return run_facts


def format_run_facts_block(run_facts: Dict[str, Any], max_chars: int = 4000) -> str:
    payload = json.dumps(run_facts or {}, indent=2, sort_keys=True, ensure_ascii=True)
    block = (
        "=== RUN_FACTS_PACK_JSON (read-only) ===\n"
        + payload
        + "\n=== END RUN_FACTS_PACK_JSON ==="
    )
    if len(block) <= max_chars:
        return block

    head_len = max(0, int(max_chars * 0.6) - 20)
    tail_len = max(0, max_chars - head_len - 30)
    head = block[:head_len]
    tail = block[-tail_len:] if tail_len else ""
    return f"{head}\n...(truncated)...\n{tail}"
