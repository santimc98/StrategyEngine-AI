import json
from pathlib import Path

from src.agents.execution_planner import _build_semantic_guard_validation, _repair_common_json_damage
from src.utils.contract_validator import validate_contract_minimal_readonly


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_run_json(run_id: str, filename: str) -> dict:
    path = _REPO_ROOT / "runs" / run_id / "agents" / "execution_planner" / filename
    return json.loads(path.read_text(encoding="utf-8"))


def test_replay_daa319ab_semantic_guard_accepts_materialized_required_outputs():
    semantic_core = _load_run_json("daa319ab", "semantic_core.json")
    contract_raw = _load_run_json("daa319ab", "contract_raw.json")

    result = _build_semantic_guard_validation(semantic_core, contract_raw)

    assert result.get("accepted") is True
    rules = {issue.get("rule") for issue in (result.get("issues") or []) if isinstance(issue, dict)}
    assert "semantic_guard.required_outputs_dropped" not in rules


def test_replay_deb6da59_semantic_guard_accepts_filename_like_required_outputs_materialized_to_paths():
    semantic_core = _load_run_json("deb6da59", "semantic_core.json")
    contract_raw = _load_run_json("deb6da59", "contract_raw.json")

    result = _build_semantic_guard_validation(semantic_core, contract_raw)

    assert result.get("accepted") is True
    rules = {issue.get("rule") for issue in (result.get("issues") or []) if isinstance(issue, dict)}
    assert "semantic_guard.required_outputs_dropped" not in rules


def test_replay_2a1e9a11_truncated_contract_response_is_structurally_recoverable():
    response_path = _REPO_ROOT / "runs" / "2a1e9a11" / "agents" / "execution_planner" / "response_attempt_1.txt"
    semantic_core = _load_run_json("2a1e9a11", "semantic_core.json")
    raw_text = response_path.read_text(encoding="utf-8")

    repaired = _repair_common_json_damage(raw_text)
    parsed = json.loads(repaired)
    result = _build_semantic_guard_validation(semantic_core, parsed)

    assert isinstance(parsed, dict)
    assert parsed.get("scope") == "cleaning_only"
    assert result.get("accepted") is True


def test_replay_e60616ac_length_truncated_contract_response_is_structurally_recoverable():
    response_path = _REPO_ROOT / "runs" / "e60616ac" / "agents" / "execution_planner" / "response_attempt_1.txt"
    semantic_core = _load_run_json("e60616ac", "semantic_core.json")
    raw_text = response_path.read_text(encoding="utf-8")

    repaired = _repair_common_json_damage(raw_text)
    parsed = json.loads(repaired)
    result = _build_semantic_guard_validation(semantic_core, parsed)

    assert isinstance(parsed, dict)
    assert parsed.get("scope") == "cleaning_only"
    assert result.get("accepted") is True
    rules = {issue.get("rule") for issue in (result.get("issues") or []) if isinstance(issue, dict)}
    assert "semantic_guard.required_outputs_dropped" not in rules


def test_replay_6b664652_validator_accepts_processing_only_passthrough_columns_used_in_hard_parse_gates():
    contract_raw = _load_run_json("6b664652", "contract_raw.json")

    result = validate_contract_minimal_readonly(
        contract_raw,
        column_inventory=contract_raw.get("canonical_columns"),
    )

    assert result.get("accepted") is True
    rules = {issue.get("rule") for issue in (result.get("issues") or []) if isinstance(issue, dict)}
    assert "contract.cleaning_gate_drop_conflict" not in rules


def test_replay_97929fa9_validator_accepts_zero_optimization_rounds_for_disabled_cleaning_only_policy():
    contract_raw = _load_run_json("97929fa9", "contract_raw.json")

    result = validate_contract_minimal_readonly(
        contract_raw,
        column_inventory=contract_raw.get("canonical_columns"),
    )

    assert result.get("accepted") is True
    rules = {issue.get("rule") for issue in (result.get("issues") or []) if isinstance(issue, dict)}
    assert "contract.optimization_policy_value" not in rules


