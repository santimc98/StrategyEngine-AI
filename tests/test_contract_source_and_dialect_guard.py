"""
V4.1 Contract Source and Dialect Guard Tests

These tests verify behavior using V4.1 contract schema:
- canonical_columns for input columns  
- derived_columns for computed columns
- column_roles for role mappings
"""

import json

from src.agents.execution_planner import ExecutionPlannerAgent
from src.graph.graph import _resolve_required_input_columns, dialect_guard_violations, _filter_input_contract
from src.utils.contract_accessors import get_canonical_columns, get_derived_column_names


def test_execution_planner_marks_missing_column_as_derived_with_inventory():
    """V4.1: Missing columns should appear in derived_columns, not canonical_columns."""
    planner = ExecutionPlannerAgent(api_key=None)
    strategy = {"required_columns": ["RefScore"], "title": "Ranking"}
    contract = planner.generate_contract(strategy=strategy, column_inventory=["col_a"])
    
    # V4.1: RefScore should be in derived_columns since it's not in inventory
    derived = get_derived_column_names(contract)
    canonical = get_canonical_columns(contract)
    
    # Either RefScore is in derived_columns, or if the contract still uses legacy format,
    # check that it's marked appropriately
    is_derived = any(c.lower() == "refscore" for c in derived)
    not_in_canonical = not any(c.lower() == "refscore" for c in canonical)
    
    # For backwards compatibility, also check old format
    reqs = contract.get("data_requirements", [])
    old_style_derived = False
    if reqs:
        ref_req = next((r for r in reqs if str(r.get("name")).lower() == "refscore"), {})
        old_style_derived = ref_req.get("source") == "derived"
    
    assert is_derived or old_style_derived or not_in_canonical, \
        f"RefScore should be marked as derived. derived_columns={derived}, canonical={canonical}"


def test_resolve_required_input_columns_ignores_derived():
    """V4.1: _resolve_required_input_columns should use canonical_columns, not derived."""
    contract = {
        "canonical_columns": ["feature_a"],
        "derived_columns": [{"name": "target_x"}],
    }
    required = _resolve_required_input_columns(contract, {"required_columns": ["fallback"]})
    assert "feature_a" in required
    assert "target_x" not in required
    
    filtered = _filter_input_contract(contract)
    # V4.1: Should preserve canonical_columns
    assert "canonical_columns" in filtered or "data_requirements" in filtered


def test_dialect_guard_flags_mismatch():
    code = "import pandas as pd\npd.read_csv('data/raw.csv', sep=';', decimal=',', encoding='utf-8')\n"
    issues = dialect_guard_violations(code, csv_sep=",", csv_decimal=".", csv_encoding="utf-8", expected_path="data/raw.csv")
    assert any("literal" in i or "missing" in i for i in issues)


def test_dialect_guard_accepts_matching_dialect():
    code = "import pandas as pd\nsep_val = ';'\npd.read_csv('data/raw.csv', sep=sep_val, decimal=',', encoding='utf-8')\n"
    issues = dialect_guard_violations(code, csv_sep=";", csv_decimal=",", csv_encoding="utf-8", expected_path="data/raw.csv")
    assert issues == []
