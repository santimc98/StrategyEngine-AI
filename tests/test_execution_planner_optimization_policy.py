from src.agents.execution_planner import ExecutionPlannerAgent, _ensure_optimization_policy


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

