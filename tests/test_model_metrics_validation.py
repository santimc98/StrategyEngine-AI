import pytest
from src.utils.ml_validation import validate_model_metrics_consistency

def test_validate_metrics_success_baseline():
    """Test success case: baseline model is correctly reported."""
    metrics = {
        "model_performance": {
            "best_model_name": "LogisticRegression",
            "best_model_auc": 0.85,
            "baseline_auc": 0.85
        }
    }
    result = validate_model_metrics_consistency(metrics)
    assert result["passed"] is True
    assert result["error_message"] == ""

def test_validate_metrics_failure_baseline_inconsistent():
    """Test failure: best_model_name is baseline but AUC differs from baseline_auc."""
    metrics = {
        "model_performance": {
            "best_model_name": "DummyClassifier",
            "best_model_auc": 0.90,
            "baseline_auc": 0.50
        }
    }
    result = validate_model_metrics_consistency(metrics)
    assert result["passed"] is False
    assert "Inconsistency: Selected best model is a baseline" in result["error_message"]

def test_validate_metrics_success_advanced():
    """Test success case: advanced model is better than baseline."""
    metrics = {
        "model_performance": {
            "best_model_name": "XGBClassifier",
            "best_model_auc": 0.92,
            "baseline_auc": 0.85
        }
    }
    result = validate_model_metrics_consistency(metrics)
    assert result["passed"] is True

def test_validate_metrics_failure_advanced_worse_than_baseline():
    """Test failure: advanced model is significantly worse than baseline."""
    metrics = {
        "model_performance": {
            "best_model_name": "RandomForestRegressor",
            "best_model_r2": 0.60,
            "baseline_r2": 0.80
        }
    }
    # Note: the function supports r2 as fallback for r2
    result = validate_model_metrics_consistency(metrics)
    assert result["passed"] is False
    assert "significantly worse than baseline" in result["error_message"]

def test_validate_metrics_missing_perf():
    """Test robustness when model_performance is missing."""
    metrics = {}
    result = validate_model_metrics_consistency(metrics)
    assert result["passed"] is True

def test_validate_metrics_none_values():
    """Test robustness with None values."""
    metrics = {
        "model_performance": {
            "best_model_name": None,
            "best_model_auc": 0.85,
            "baseline_auc": 0.85
        }
    }
    result = validate_model_metrics_consistency(metrics)
    assert result["passed"] is True


def test_validate_metrics_supports_primary_metric_consistency_for_survival():
    metrics = {
        "model_performance": {
            "best_model_name": "CoxPH",
            "primary_metric": "concordance_index",
            "primary_metric_value": 0.73,
            "baseline_concordance_index": 0.68,
        }
    }

    result = validate_model_metrics_consistency(metrics)

    assert result["passed"] is True
    assert result["details"]["metric_name"] == "concordance_index"


def test_validate_metrics_uses_direction_for_lower_is_better_primary_metric():
    metrics = {
        "model_performance": {
            "best_model_name": "ElasticNet",
            "primary_metric": "mae",
            "primary_metric_value": 1.25,
            "baseline_mae": 0.8,
        }
    }

    result = validate_model_metrics_consistency(metrics)

    assert result["passed"] is False
    assert "significantly worse than baseline metric" in result["error_message"]
