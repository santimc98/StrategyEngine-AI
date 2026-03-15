import json
from unittest.mock import MagicMock

from src.agents.execution_planner import ExecutionPlannerAgent


def test_execution_planner_progressive_mode_applies_llm_patch(monkeypatch):
    monkeypatch.setenv("EXECUTION_PLANNER_PROGRESSIVE_MODE", "1")
    monkeypatch.setenv("EXECUTION_PLANNER_SECTION_FIRST", "0")
    monkeypatch.setenv("EXECUTION_PLANNER_PROGRESSIVE_ROUNDS", "1")

    planner = ExecutionPlannerAgent(api_key="mock_key")

    patch_payload = {
        "feature_engineering_plan": {
            "derived_columns": [
                {
                    "name": "is_success",
                    "source_column": "status",
                    "derivation_type": "rule_from_outcome",
                    "positive_values": ["won"],
                }
            ]
        },
        "qa_gates": [
            {"name": "benchmark_kpi_report", "severity": "HARD", "params": {"metric": "accuracy"}}
        ],
        "reviewer_gates": [
            {"name": "strategy_followed", "severity": "HARD", "params": {}}
        ],
        "objective_analysis": {"problem_type": "classification"},
        "evaluation_spec": {"objective_type": "binary_classification"},
    }

    mock_resp = MagicMock()
    mock_resp.text = json.dumps(patch_payload)
    mock_resp.candidates = []
    mock_resp.usage_metadata = None
    planner.client = MagicMock()
    planner.client.generate_content.return_value = mock_resp

    contract = planner.generate_contract(
        strategy={"required_columns": ["status", "amount"], "title": "Progressive Test"},
        business_objective="Predict win probability.",
        column_inventory=["status", "amount", "id"],
    )

    assert isinstance(contract, dict) and contract
    assert "feature_engineering_plan" in contract
    derived = contract.get("feature_engineering_plan", {}).get("derived_columns", [])
    assert any(isinstance(item, dict) and item.get("name") == "is_success" for item in derived)
    assert isinstance(planner.last_planner_diag, list)


def test_execution_planner_progressive_mode_keeps_feature_engineering_tasks(monkeypatch):
    monkeypatch.setenv("EXECUTION_PLANNER_PROGRESSIVE_MODE", "1")
    monkeypatch.setenv("EXECUTION_PLANNER_SECTION_FIRST", "0")
    monkeypatch.setenv("EXECUTION_PLANNER_PROGRESSIVE_ROUNDS", "1")

    planner = ExecutionPlannerAgent(api_key="mock_key")

    patch_payload = {
        "feature_engineering_tasks": [
            {
                "technique": "interaction",
                "input_columns": ["amount", "discount"],
                "output_column_name": "amount_x_discount",
                "rationale": "Capture nonlinear pricing effects.",
            }
        ],
        "objective_analysis": {"problem_type": "regression"},
        "evaluation_spec": {"objective_type": "regression"},
    }

    mock_resp = MagicMock()
    mock_resp.text = json.dumps(patch_payload)
    mock_resp.candidates = []
    mock_resp.usage_metadata = None
    planner.client = MagicMock()
    planner.client.generate_content.return_value = mock_resp

    contract = planner.generate_contract(
        strategy={"required_columns": ["amount", "discount"], "title": "Progressive FE Tasks"},
        business_objective="Predict revenue.",
        column_inventory=["amount", "discount", "id"],
    )

    assert isinstance(contract, dict) and contract
    tasks = contract.get("feature_engineering_tasks")
    assert isinstance(tasks, list) and tasks
    assert any(
        isinstance(item, dict) and item.get("output_column_name") == "amount_x_discount"
        for item in tasks
    )
