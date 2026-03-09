"""
Tests for validate_decision_variable_isolation in ml_validation.py

V4.1: Tests use column_roles in role->list format, no feature_availability.
"""
import pytest

from src.utils.ml_validation import validate_decision_variable_isolation


class TestPriceOptimizationException:
    """Test that price optimization correctly allows decision variable in features."""

    def test_price_optimization_allows_decision_var_in_features(self):
        """
        Case A: Price optimization with column_roles V4.1 format.
        Decision variable in features is ALLOWED for price optimization.
        """
        code = """
import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('data.csv')
features = ['segment', 'price', 'region']  # price is decision variable
X = df[features]
y = df['converted']
model = LogisticRegression()
model.fit(X, y)
"""
        # V4.1 contract with column_roles in role->list format
        contract = {
            "column_roles": {
                "decision": ["price"],
                "outcome": ["converted"],
                "pre_decision": ["segment", "region"],
            },
            "objective_analysis": {
                "problem_type": "optimization",
                "decision_variable": "price",
                "success_criteria": "maximize revenue by optimizing price"
            }
        }

        result = validate_decision_variable_isolation(code, contract)

        assert result["passed"] is True
        assert result["error_message"] == ""
        assert result["violated_variables"] == []

    def test_price_optimization_with_elasticity_criteria(self):
        """Price optimization with elasticity in success_criteria passes."""
        code = """
features = ['price', 'customer_tier']
X = df[features]
"""
        contract = {
            "column_roles": {"decision": ["price"]},
            "objective_analysis": {
                "problem_type": "optimization",
                "decision_variable": "price",
                "success_criteria": "model price elasticity for conversion probability"
            }
        }

        result = validate_decision_variable_isolation(code, contract)

        assert result["passed"] is True


class TestNonPriceOptimizationViolation:
    """Test that non-price optimization detects decision variable violations."""

    def test_resource_allocation_detects_decision_var_violation(self):
        """
        Case B: Resource allocation optimization.
        Decision variable in features is a VIOLATION.
        """
        code = """
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('data.csv')
features = ['customer_value', 'allocation_amount', 'history']  # allocation_amount is decision
X = df[features]
y = df['success']
model = RandomForestClassifier()
model.fit(X, y)
"""
        # V4.1 contract - non-price optimization
        contract = {
            "column_roles": {
                "decision": ["allocation_amount"],
                "outcome": ["success"],
                "pre_decision": ["customer_value", "history"],
            },
            "objective_analysis": {
                "problem_type": "optimization",
                "decision_variable": "allocation_amount",
                "success_criteria": "maximize success rate through resource allocation"
            }
        }

        result = validate_decision_variable_isolation(code, contract)

        assert result["passed"] is False
        assert "allocation_amount" in result["violated_variables"]
        assert "CAUSAL_VIOLATION" in result["error_message"]

    def test_prescriptive_problem_alias_still_detects_decision_var_violation(self):
        code = """
features = ['customer_value', 'allocation_amount']
X = df[features]
"""
        contract = {
            "business_objective": "Use prescriptive decisioning to maximize success",
            "column_roles": {
                "decision": ["allocation_amount"],
                "pre_decision": ["customer_value"],
            },
            "objective_analysis": {
                "problem_type": "decisioning",
                "success_criteria": "maximize success rate through resource allocation",
            },
        }

        result = validate_decision_variable_isolation(code, contract)

        assert result["passed"] is False
        assert "allocation_amount" in result["violated_variables"]

    def test_optimization_without_decision_var_in_features_passes(self):
        """Optimization that correctly excludes decision variable passes."""
        code = """
features = ['customer_value', 'history', 'segment']
X = df[features]
y = df['success']
"""
        contract = {
            "column_roles": {
                "decision": ["allocation_amount"],
                "outcome": ["success"],
                "pre_decision": ["customer_value", "history", "segment"],
            },
            "objective_analysis": {
                "problem_type": "optimization",
                "decision_variable": "allocation_amount",
                "success_criteria": "maximize success rate"
            }
        }

        result = validate_decision_variable_isolation(code, contract)

        assert result["passed"] is True
        assert result["violated_variables"] == []


class TestNonOptimizationSkip:
    """Test that non-optimization problems skip this validation."""

    def test_prediction_problem_skips_check(self):
        """Prediction problems skip decision variable isolation check."""
        code = """
features = ['feature1', 'feature2', 'decision_col']
X = df[features]
"""
        contract = {
            "column_roles": {"decision": ["decision_col"]},
            "objective_analysis": {
                "problem_type": "prediction",
                "success_criteria": "predict churn"
            }
        }

        result = validate_decision_variable_isolation(code, contract)

        assert result["passed"] is True

    def test_clustering_problem_skips_check(self):
        """Clustering problems skip decision variable isolation check."""
        code = """
features = ['a', 'b', 'c']
"""
        contract = {
            "objective_analysis": {
                "problem_type": "clustering"
            }
        }

        result = validate_decision_variable_isolation(code, contract)

        assert result["passed"] is True


class TestV41ColumnRolesFormat:
    """Test that V4.1 column_roles format (role->list) is correctly handled."""

    def test_column_roles_role_to_list_format(self):
        """V4.1 format: column_roles = {"decision": ["col1", "col2"], ...}"""
        code = """
model_features = ['pre_col', 'decision_var']
X = df[model_features]
"""
        contract = {
            "column_roles": {
                "decision": ["decision_var"],
                "pre_decision": ["pre_col"],
            },
            "objective_analysis": {
                "problem_type": "optimization",
                "success_criteria": "minimize cost"
            }
        }

        result = validate_decision_variable_isolation(code, contract)

        assert result["passed"] is False
        assert "decision_var" in result["violated_variables"]

    def test_multiple_decision_variables(self):
        """Multiple decision variables detected."""
        code = """
features = ['feature1', 'price', 'discount']
X = df[features]
"""
        contract = {
            "column_roles": {
                "decision": ["price", "discount"],
                "pre_decision": ["feature1"],
            },
            "objective_analysis": {
                "problem_type": "optimization",
                "success_criteria": "minimize cost allocation"
            }
        }

        result = validate_decision_variable_isolation(code, contract)

        assert result["passed"] is False
        assert "price" in result["violated_variables"]
        assert "discount" in result["violated_variables"]


class TestDecisionColumnsTopLevel:
    """Test that decision_columns top-level key is also supported."""

    def test_decision_columns_top_level_key(self):
        """decision_columns as top-level key (V4.1) is detected."""
        code = """
features = ['a', 'b', 'decision_col']
"""
        contract = {
            "decision_columns": ["decision_col"],
            "objective_analysis": {
                "problem_type": "optimization",
                "success_criteria": "minimize waste"
            }
        }

        result = validate_decision_variable_isolation(code, contract)

        assert result["passed"] is False
        assert "decision_col" in result["violated_variables"]


class TestNoDecisionVariables:
    """Test behavior when no decision variables are defined."""

    def test_no_decision_vars_passes(self):
        """If no decision variables defined, validation passes."""
        code = """
features = ['a', 'b', 'c']
X = df[features]
"""
        contract = {
            "column_roles": {
                "pre_decision": ["a", "b", "c"],
            },
            "objective_analysis": {
                "problem_type": "optimization",
                "success_criteria": "maximize throughput"
            }
        }

        result = validate_decision_variable_isolation(code, contract)

        assert result["passed"] is True
        assert result["violated_variables"] == []


class TestNoLegacyFeatureAvailability:
    """Ensure no fallback to legacy feature_availability."""

    def test_feature_availability_ignored(self):
        """
        V4.1: feature_availability is NOT used even if present.
        Only column_roles and decision_columns are used.
        """
        code = """
features = ['a', 'legacy_decision']
X = df[features]
"""
        # Contract with ONLY legacy feature_availability (no column_roles)
        contract = {
            "feature_availability": [
                {"column": "legacy_decision", "availability": "decision"},
                {"column": "a", "availability": "pre_decision"},
            ],
            "objective_analysis": {
                "problem_type": "optimization",
                "success_criteria": "minimize cost"
            }
        }

        result = validate_decision_variable_isolation(code, contract)

        # Should PASS because V4.1 ignores feature_availability
        # (no decision_columns or column_roles["decision"] defined)
        assert result["passed"] is True
        assert result["violated_variables"] == []
