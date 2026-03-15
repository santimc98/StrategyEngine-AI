from unittest.mock import MagicMock

from src.agents.execution_planner import (
    ExecutionPlannerAgent,
    _apply_planner_structural_support,
    parse_derive_from_expression,
)


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


def test_planner_structural_support_projects_clean_dataset_from_canonical_contract():
    contract = {
        "scope": "full_pipeline",
        "canonical_columns": ["event_id", "__split", "feature_a", "target"],
        "column_roles": {
            "pre_decision": ["feature_a"],
            "outcome": ["target"],
            "identifiers": ["event_id"],
        },
        "allowed_feature_sets": {
            "model_features": ["feature_a"],
            "segmentation_features": [],
            "forbidden_features": ["target"],
            "audit_only_features": ["__split"],
        },
        "required_outputs": [
            "artifacts/clean/clean_dataset.csv",
            "artifacts/clean/clean_dataset_manifest.json",
            "artifacts/ml/submission.csv",
        ],
    }

    supported = _apply_planner_structural_support(contract)
    clean_dataset = ((supported.get("artifact_requirements") or {}).get("clean_dataset") or {})

    assert clean_dataset.get("output_path") == "artifacts/clean/clean_dataset.csv"
    assert clean_dataset.get("output_manifest_path") == "artifacts/clean/clean_dataset_manifest.json"
    assert set(clean_dataset.get("required_columns") or []) >= {"event_id", "__split", "feature_a", "target"}
