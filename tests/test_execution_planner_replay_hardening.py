import json
from pathlib import Path

from src.agents.execution_planner import _build_semantic_guard_validation, _repair_common_json_damage
from src.utils.contract_validator import validate_contract_minimal_readonly
from src.utils.contract_views import build_contract_views_projection


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


def test_replay_6f2993d7_views_preserve_top_level_required_outputs_and_cleaning_gates():
    contract_raw = _load_run_json("6f2993d7", "contract_raw.json")

    projected = build_contract_views_projection(contract_raw, artifact_index=[])
    de_view = projected.get("de_view") or {}
    cleaning_view = projected.get("cleaning_view") or {}

    top_required_outputs = contract_raw.get("required_outputs") or []
    top_cleaning_gates = contract_raw.get("cleaning_gates") or []
    de_required_outputs = de_view.get("required_outputs") or []
    cleaning_gates = cleaning_view.get("cleaning_gates") or []

    assert len(de_required_outputs) == len(top_required_outputs)
    assert len(cleaning_gates) == len(top_cleaning_gates)

    def _extract_paths(items: list) -> set[str]:
        paths: set[str] = set()
        for item in items:
            if isinstance(item, dict) and isinstance(item.get("path"), str):
                paths.add(item["path"])
            elif isinstance(item, str):
                paths.add(item)
        return paths

    top_paths = _extract_paths(top_required_outputs)
    de_paths = _extract_paths(de_required_outputs)
    assert de_paths == top_paths

    top_gate_names = {
        item.get("name")
        for item in top_cleaning_gates
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    cleaning_gate_names = {
        item.get("name")
        for item in cleaning_gates
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    assert cleaning_gate_names == top_gate_names


def test_replay_ccc6cbf5_de_view_preserves_declared_optional_passthrough_columns():
    contract_raw = _load_run_json("ccc6cbf5", "contract_raw.json")

    projected = build_contract_views_projection(contract_raw, artifact_index=[])
    de_view = projected.get("de_view") or {}
    artifact_reqs = (contract_raw.get("artifact_requirements") or {}).get("clean_dataset") or {}
    expected_optional = artifact_reqs.get("optional_passthrough_columns") or []

    assert de_view.get("optional_passthrough_columns") == expected_optional
