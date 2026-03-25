from typing import Any, Dict

import pytest

from src.utils.contract_response_schema import (
    EXECUTION_SEMANTIC_CORE_SCHEMA,
    OPTIMIZATION_POLICY_MIN_SCHEMA,
    EXECUTION_CONTRACT_V5_CANONICAL_REQUIRED_KEYS,
    V5_AGENT_SECTION_KEYS,
)


def test_optimization_policy_schema_shape():
    assert OPTIMIZATION_POLICY_MIN_SCHEMA.get("type") == "object"
    opt_props = OPTIMIZATION_POLICY_MIN_SCHEMA.get("properties") or {}
    assert (opt_props.get("max_rounds") or {}).get("minimum") == 0
    assert (opt_props.get("quick_eval_folds") or {}).get("minimum") == 0
    assert (opt_props.get("full_eval_folds") or {}).get("minimum") == 0


def test_semantic_core_schema_requires_key_fields():
    required = EXECUTION_SEMANTIC_CORE_SCHEMA.get("required") or []
    assert "scope" in required
    assert "canonical_columns" in required
    assert "column_roles" in required
    assert "task_semantics" in required
    assert "active_workstreams" in required


def test_semantic_core_schema_allows_gate_action_type_extensions():
    props = EXECUTION_SEMANTIC_CORE_SCHEMA.get("properties") or {}
    cleaning_gates = props.get("cleaning_gates") or {}
    items = cleaning_gates.get("items") or {}
    any_of = items.get("anyOf") or []
    object_variants = [v for v in any_of if isinstance(v, dict) and v.get("type") == "object"]
    assert object_variants
    gate_props = object_variants[0].get("properties") or {}
    assert "action_type" in gate_props
    assert "column_phase" in gate_props
    assert "final_state" in gate_props


def test_v5_canonical_required_keys():
    assert "contract_version" in EXECUTION_CONTRACT_V5_CANONICAL_REQUIRED_KEYS
    assert "shared" in EXECUTION_CONTRACT_V5_CANONICAL_REQUIRED_KEYS
    assert "data_engineer" in EXECUTION_CONTRACT_V5_CANONICAL_REQUIRED_KEYS
    assert "ml_engineer" in EXECUTION_CONTRACT_V5_CANONICAL_REQUIRED_KEYS


def test_v5_agent_section_keys():
    assert "data_engineer" in V5_AGENT_SECTION_KEYS
    assert "ml_engineer" in V5_AGENT_SECTION_KEYS
    assert "cleaning_reviewer" in V5_AGENT_SECTION_KEYS
    assert "qa_reviewer" in V5_AGENT_SECTION_KEYS
    assert "business_translator" in V5_AGENT_SECTION_KEYS
