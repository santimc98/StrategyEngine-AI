from typing import Any, Dict

import pytest

from src.utils.contract_response_schema import (
    EXECUTION_CONTRACT_CANONICAL_SCHEMA,
    EXECUTION_CONTRACT_V42_MIN_SCHEMA,
    OPTIMIZATION_POLICY_MIN_SCHEMA,
)


def _minimal_contract_base() -> Dict[str, Any]:
    return {
        "contract_version": "4.1",
        "scope": "full_pipeline",
        "strategy_title": "Baseline Strategy",
        "business_objective": "Predict churn",
        "output_dialect": {"sep": ",", "decimal": ".", "encoding": "utf-8"},
        "canonical_columns": ["id", "feature_a", "target"],
        "column_roles": {
            "pre_decision": ["feature_a"],
            "decision": [],
            "outcome": ["target"],
            "post_decision_audit_only": [],
            "unknown": [],
            "identifiers": ["id"],
            "time_columns": [],
        },
        "allowed_feature_sets": {
            "segmentation_features": [],
            "model_features": ["feature_a"],
            "forbidden_features": [],
            "audit_only_features": [],
        },
        "active_workstreams": {
            "cleaning": True,
            "feature_engineering": True,
            "model_training": False,
        },
        "artifact_requirements": {},
        "required_outputs": ["data/metrics.json"],
        "iteration_policy": {"max_iterations": 2},
    }


def test_v42_schema_exposes_optimization_policy_shape():
    props = EXECUTION_CONTRACT_V42_MIN_SCHEMA.get("properties") or {}
    assert "optimization_policy" in props
    opt = props.get("optimization_policy") or {}
    assert isinstance(opt, dict)
    assert "required" in opt
    assert "max_rounds" in (opt.get("properties") or {})
    assert OPTIMIZATION_POLICY_MIN_SCHEMA.get("type") == "object"
    opt_props = OPTIMIZATION_POLICY_MIN_SCHEMA.get("properties") or {}
    assert (opt_props.get("max_rounds") or {}).get("minimum") == 0
    assert (opt_props.get("quick_eval_folds") or {}).get("minimum") == 0
    assert (opt_props.get("full_eval_folds") or {}).get("minimum") == 0


def test_v42_schema_remains_backward_compatible_with_v41_contract():
    try:
        from jsonschema import Draft7Validator
    except Exception:
        pytest.skip("jsonschema is not available")

    contract = _minimal_contract_base()
    errors = sorted(Draft7Validator(EXECUTION_CONTRACT_V42_MIN_SCHEMA).iter_errors(contract), key=lambda e: list(e.path))
    assert errors == []


def test_canonical_schema_exposes_agent_interfaces_for_explicit_downstream_contracts():
    props = EXECUTION_CONTRACT_CANONICAL_SCHEMA.get("properties") or {}
    agent_interfaces = props.get("agent_interfaces") or {}

    assert isinstance(agent_interfaces, dict)
    interface_props = agent_interfaces.get("properties") or {}
    assert "data_engineer" in interface_props
    assert "ml_engineer" in interface_props
    assert "reviewer" in interface_props


def test_canonical_schema_allows_required_outputs_with_intent_and_path():
    props = EXECUTION_CONTRACT_CANONICAL_SCHEMA.get("properties") or {}
    required_outputs = props.get("required_outputs") or {}
    items = required_outputs.get("items") or {}
    any_of = items.get("anyOf") or []

    object_variants = [variant for variant in any_of if isinstance(variant, dict) and variant.get("type") == "object"]
    assert object_variants
    object_schema = object_variants[0]
    assert "path" in (object_schema.get("properties") or {})
    assert "intent" in (object_schema.get("properties") or {})


def test_canonical_schema_allows_gate_action_type_extensions():
    props = EXECUTION_CONTRACT_CANONICAL_SCHEMA.get("properties") or {}
    cleaning_gates = props.get("cleaning_gates") or {}
    items = cleaning_gates.get("items") or {}
    any_of = items.get("anyOf") or []

    object_variants = [variant for variant in any_of if isinstance(variant, dict) and variant.get("type") == "object"]
    assert object_variants
    object_schema = object_variants[0]
    gate_props = object_schema.get("properties") or {}
    assert "action_type" in gate_props
    assert "column_phase" in gate_props
    assert "final_state" in gate_props
