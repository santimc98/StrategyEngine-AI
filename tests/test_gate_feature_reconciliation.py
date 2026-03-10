"""Tests that qa_gates forbidden columns are excluded from model_features.

Bug context (run c77c8288):
  The LLM correctly classified `event` as auxiliary and created a qa_gate
  leakage_prevention_auxiliary with forbidden_at_inference=["event"].
  But the deterministic fallback populated model_features from all
  pre_decision columns, including `event`.  The ML Engineer used `event`
  as a model feature (as instructed by the contract) and QA then rejected
  it for violating its own gate — a self-contradictory contract.

  Fix: cross-check qa_gates.*.params.forbidden_at_inference against
  model_features and remove contradictions before emitting the contract.
"""
import pytest


def _build_minimal_contract_snippet(
    model_features: list[str],
    qa_gates: list[dict],
) -> tuple[list[str], set[str]]:
    """Simulate the gate-aware feature reconciliation logic.

    Returns (filtered_model_features, gate_forbidden_set).
    """
    _gate_forbidden_at_inference: set[str] = set()
    for gate in (qa_gates if isinstance(qa_gates, list) else []):
        if not isinstance(gate, dict):
            continue
        params = gate.get("params")
        if not isinstance(params, dict):
            continue
        for key in ("forbidden_at_inference", "forbidden_columns", "excluded_columns"):
            forbidden_list = params.get(key)
            if isinstance(forbidden_list, list):
                _gate_forbidden_at_inference.update(str(c) for c in forbidden_list if c)
    if _gate_forbidden_at_inference:
        model_features = [c for c in model_features if c not in _gate_forbidden_at_inference]
    return model_features, _gate_forbidden_at_inference


class TestGateFeatureReconciliation:

    def test_removes_forbidden_at_inference_from_model_features(self):
        """The exact scenario from run c77c8288."""
        features = ["event", "area_first_ha", "area_growth_abs_0_5h"]
        gates = [
            {
                "name": "leakage_prevention_auxiliary",
                "severity": "HARD",
                "params": {
                    "forbidden_at_inference": ["time_to_hit_hours", "event"],
                },
            },
        ]
        result, forbidden = _build_minimal_contract_snippet(features, gates)
        assert "event" not in result
        assert "time_to_hit_hours" not in result  # wasn't there to begin with
        assert "area_first_ha" in result
        assert "area_growth_abs_0_5h" in result

    def test_handles_forbidden_columns_param_key(self):
        """Some gates use forbidden_columns instead of forbidden_at_inference."""
        features = ["col_a", "col_b", "col_c"]
        gates = [
            {"name": "some_gate", "severity": "HARD", "params": {"forbidden_columns": ["col_b"]}},
        ]
        result, _ = _build_minimal_contract_snippet(features, gates)
        assert result == ["col_a", "col_c"]

    def test_handles_excluded_columns_param_key(self):
        features = ["x", "y", "z"]
        gates = [
            {"name": "gate", "severity": "SOFT", "params": {"excluded_columns": ["y", "z"]}},
        ]
        result, _ = _build_minimal_contract_snippet(features, gates)
        assert result == ["x"]

    def test_no_removal_when_no_forbidden_params(self):
        features = ["a", "b", "c"]
        gates = [
            {"name": "schema_check", "severity": "HARD", "params": {"expected_columns": ["a", "b"]}},
        ]
        result, forbidden = _build_minimal_contract_snippet(features, gates)
        assert result == ["a", "b", "c"]
        assert forbidden == set()

    def test_no_removal_when_no_gates(self):
        features = ["a", "b"]
        result, _ = _build_minimal_contract_snippet(features, [])
        assert result == ["a", "b"]

    def test_multiple_gates_accumulate_forbidden(self):
        """Multiple gates may each forbid different columns."""
        features = ["a", "b", "c", "d"]
        gates = [
            {"name": "gate1", "severity": "HARD", "params": {"forbidden_at_inference": ["a"]}},
            {"name": "gate2", "severity": "SOFT", "params": {"forbidden_columns": ["c"]}},
        ]
        result, forbidden = _build_minimal_contract_snippet(features, gates)
        assert result == ["b", "d"]
        assert forbidden == {"a", "c"}

    def test_robust_to_malformed_gates(self):
        """Gracefully handles None, missing params, etc."""
        features = ["x", "y"]
        gates = [
            None,
            {"name": "no_params"},
            {"name": "empty_params", "params": {}},
            {"name": "string_params", "params": "invalid"},
            {"name": "ok", "params": {"forbidden_at_inference": ["x"]}},
        ]
        result, _ = _build_minimal_contract_snippet(features, gates)
        assert result == ["y"]
