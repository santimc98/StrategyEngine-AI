from unittest.mock import MagicMock

from src.agents.execution_planner import ExecutionPlannerAgent, parse_derive_from_expression


def test_parse_derive_from_expression_simple():
    parsed = parse_derive_from_expression("CurrentPhase == 'Contract'")
    assert parsed.get("column") == "CurrentPhase"
    assert parsed.get("positive_values") == ["Contract"]


def test_canonical_columns_exclude_derived_targets_and_segments():
    planner = ExecutionPlannerAgent(api_key=None)
    strategy = {
        "required_columns": ["CurrentPhase", "Amount"],
        "analysis_type": "predictive",
        "title": "Segmented Conversion",
    }
    business_objective = "Segment accounts and predict conversion success."
    data_summary = "Column Types:\n- Categorical/Boolean: CurrentPhase\n- Numerical: Amount\n"
    contract = planner.generate_contract(
        strategy=strategy,
        data_summary=data_summary,
        business_objective=business_objective,
        column_inventory=["CurrentPhase", "Amount"],
    )
    canonical = contract.get("canonical_columns") or []
    assert "is_success" not in canonical
    assert "cluster_id" not in canonical

    evaluation_spec = planner.generate_evaluation_spec(
        strategy=strategy,
        contract=contract,
        data_summary=data_summary,
        business_objective=business_objective,
        column_inventory=["CurrentPhase", "Amount"],
    )
    spec_canonical = evaluation_spec.get("canonical_columns") or []
    assert "is_success" not in spec_canonical
    assert "cluster_id" not in spec_canonical


def test_invalid_llm_contract_is_not_replaced_by_deterministic_scaffold(monkeypatch):
    monkeypatch.delenv("EXECUTION_PLANNER_SECTION_FIRST", raising=False)
    monkeypatch.delenv("EXECUTION_PLANNER_PROGRESSIVE_MODE", raising=False)

    planner = ExecutionPlannerAgent(api_key="mock_key")
    response = MagicMock()
    response.text = (
        '{"scope":"full_pipeline",'
        '"objective_analysis":{"problem_type":"classification"},'
        '"evaluation_spec":{"objective_type":"classification"}}'
    )
    response.candidates = []
    response.usage_metadata = None
    planner.client = MagicMock()
    planner.client.generate_content.return_value = response

    contract = planner.generate_contract(
        strategy={"required_columns": ["id", "feature", "target"], "title": "No scaffold override"},
        business_objective="Predict target.",
        column_inventory=["id", "feature", "target"],
    )

    assert isinstance(contract, dict)
    assert contract.get("canonical_columns") == []
    diagnostics = planner.last_contract_diagnostics or {}
    summary = diagnostics.get("summary") or {}
    assert summary.get("accepted") is False
