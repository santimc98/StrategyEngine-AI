from typing import Any, Dict

import pytest

from src.utils.contract_response_schema import (
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


def test_v42_schema_remains_backward_compatible_with_v41_contract():
    try:
        from jsonschema import Draft7Validator
    except Exception:
        pytest.skip("jsonschema is not available")

    contract = _minimal_contract_base()
    errors = sorted(Draft7Validator(EXECUTION_CONTRACT_V42_MIN_SCHEMA).iter_errors(contract), key=lambda e: list(e.path))
    assert errors == []

