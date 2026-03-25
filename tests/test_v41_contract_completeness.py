
import pytest
from src.utils.contract_accessors import strip_legacy_keys, assert_no_legacy_keys, LEGACY_KEYS

def test_legacy_keys_definitions():
    """Ensure newly banned keys are in LEGACY_KEYS."""
    required = {"required_columns", "feature_availability", "decision_variables"}
    missing = required - LEGACY_KEYS
    assert not missing, f"LEGACY_KEYS missing: {missing}"

def test_strip_legacy_keys_removes_required_columns():
    """Test that strip_legacy_keys removes 'required_columns'."""
    contract = {
        "contract_version": "4.1",
        "required_columns": ["col1", "col2"],
        "canonical_columns": ["col1", "col2"]
    }
    cleaned = strip_legacy_keys(contract)
    assert "required_columns" not in cleaned
    assert "canonical_columns" in cleaned

def test_assert_no_legacy_keys_detects_violation():
    """Test that assert_no_legacy_keys raises AssertionError for 'required_columns'."""
    contract = {
        "contract_version": "4.1",
        "required_columns": ["col1"]
    }
    with pytest.raises(ValueError) as exc:
        assert_no_legacy_keys(contract, where="test")
    assert "Legacy keys found" in str(exc.value) or "legacy keys found" in str(exc.value).lower()

def test_graph_py_clean_of_legacy_reads():
    """Ensure src/graph/graph.py does not contain contract.get('required_columns')."""
    import os
    path = os.path.abspath("src/graph/graph.py")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Check for specific banned pattern
    assert 'contract.get("required_columns")' not in content, "Found legacy contract.get('required_columns') in graph.py"
    assert "contract.get('required_columns')" not in content, "Found legacy contract.get('required_columns') in graph.py"

def test_ml_engineer_py_clean_of_legacy_reads():
    """Ensure src/agents/ml_engineer.py does not contain legacy keys."""
    import os
    path = os.path.abspath("src/agents/ml_engineer.py")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    
    assert 'feature_availability' not in content, "Found 'feature_availability' in ml_engineer.py (should be removed)"
    assert 'decision_variables' not in content, "Found 'decision_variables' in ml_engineer.py (should be removed)"
