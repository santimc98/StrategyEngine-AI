from src.agents.execution_planner import ExecutionPlannerAgent, _ensure_optimization_policy
from src.utils.contract_validator import normalize_optimization_policy


def test_ensure_optimization_policy_backfills_defaults():
    contract = {"strategy_title": "x"}
    out = _ensure_optimization_policy(contract)
    policy = out.get("optimization_policy") or {}
    assert policy.get("enabled") is True
    assert policy.get("max_rounds") == 8
    assert policy.get("quick_eval_folds") == 2
    assert policy.get("full_eval_folds") == 5
    assert float(policy.get("min_delta")) == 0.0005
    assert policy.get("patience") == 3


def test_execution_planner_fallback_contract_without_api_key():
    """Without an API key the planner returns an empty fallback.
    Post-migration: no auto-fill of operational sections."""
    planner = ExecutionPlannerAgent(api_key=None)
    contract = planner.generate_contract(
        strategy={"title": "Baseline"},
        business_objective="Predict churn",
        column_inventory=["id", "feature_a", "target"],
    )
    # With no LLM client, contract is essentially empty
    assert isinstance(contract, dict)


def test_normalize_optimization_policy_preserves_zero_rounds_for_disabled_cleaning_only_policy():
    policy = normalize_optimization_policy(
        {
            "enabled": False,
            "max_rounds": 0,
            "quick_eval_folds": 0,
            "full_eval_folds": 0,
            "min_delta": 0,
            "patience": 0,
            "allow_model_switch": False,
            "allow_ensemble": False,
            "allow_hpo": False,
            "allow_feature_engineering": True,
            "allow_calibration": False,
        }
    )

    assert policy.get("enabled") is False
    assert policy.get("max_rounds") == 0
    assert policy.get("quick_eval_folds") == 0
    assert policy.get("full_eval_folds") == 0


def test_normalize_optimization_policy_normalizes_direction_and_tie_breakers():
    policy = normalize_optimization_policy(
        {
            "enabled": True,
            "optimization_direction": "lower_is_better",
            "tie_breakers": [
                {"metric": "cv_std", "order": "minimise", "reason": "Prefer lower variance."},
                "generalization_gap_abs",
            ],
        }
    )

    assert policy.get("optimization_direction") == "minimize"
    assert policy.get("tie_breakers") == [
        {"field": "cv_std", "direction": "minimize", "reason": "Prefer lower variance."},
        {"field": "generalization_gap_abs", "direction": "unspecified"},
    ]


def test_ensure_optimization_policy_lifts_semantics_from_validation_and_evaluation_spec():
    contract = {
        "validation_requirements": {
            "primary_metric": "mae",
            "optimization_direction": "minimize",
            "tie_breakers": [{"field": "cv_std", "direction": "minimize"}],
        },
        "evaluation_spec": {"primary_metric": "mae"},
        "optimization_policy": {
            "enabled": True,
            "max_rounds": 4,
            "quick_eval_folds": 2,
            "full_eval_folds": 5,
            "min_delta": 0.001,
            "patience": 2,
            "allow_model_switch": True,
            "allow_ensemble": False,
            "allow_hpo": True,
            "allow_feature_engineering": True,
            "allow_calibration": False,
        },
    }

    out = _ensure_optimization_policy(contract)
    policy = out.get("optimization_policy") or {}

    assert policy.get("optimization_direction") == "minimize"
    assert policy.get("tie_breakers") == [{"field": "cv_std", "direction": "minimize"}]

