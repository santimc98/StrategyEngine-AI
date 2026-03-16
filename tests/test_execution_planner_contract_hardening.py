import json

from unittest.mock import MagicMock

from src.agents.execution_planner import (
    _build_patch_transport_validation,
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


def test_planner_structural_support_projects_missing_ml_operational_sections_from_canonical_contract():
    contract = {
        "scope": "full_pipeline",
        "strategy_title": "Binary Churn",
        "business_objective": "Predict churn probability for each customer.",
        "output_dialect": {"sep": ",", "decimal": ".", "encoding": "utf-8"},
        "canonical_columns": ["customer_id", "__split", "feature_a", "churned"],
        "required_outputs": ["artifacts/ml/submission.csv"],
        "column_roles": {
            "pre_decision": ["feature_a"],
            "decision": [],
            "outcome": ["churned"],
            "post_decision_audit_only": [],
            "unknown": [],
            "identifiers": ["customer_id"],
            "time_columns": [],
        },
        "allowed_feature_sets": {
            "model_features": ["feature_a"],
            "segmentation_features": [],
            "forbidden_features": ["churned"],
            "audit_only_features": ["__split"],
        },
        "task_semantics": {
            "problem_family": "classification",
            "objective_type": "binary_classification",
            "primary_target": "churned",
            "target_columns": ["churned"],
            "partitioning": {"split_column": "__split"},
            "output_schema": {
                "prediction_artifact": "artifacts/ml/submission.csv",
                "required_prediction_columns": ["prob_churned"],
            },
        },
        "artifact_requirements": {},
        "cleaning_gates": [{"name": "no_null_target", "severity": "HARD", "params": {"column": "churned"}}],
        "qa_gates": [{"name": "metrics_present", "severity": "HARD", "params": {}}],
        "reviewer_gates": [{"name": "strategy_followed", "severity": "HARD", "params": {}}],
        "data_engineer_runbook": "clean and preserve split column",
        "ml_engineer_runbook": "train binary classifier",
    }

    supported = _apply_planner_structural_support(contract)

    assert (supported.get("evaluation_spec") or {}).get("objective_type") == "binary_classification"
    assert (supported.get("validation_requirements") or {}).get("primary_metric") == "log_loss"
    assert (supported.get("validation_requirements") or {}).get("method") == "holdout"
    assert (supported.get("iteration_policy") or {}).get("max_iterations") >= 1
    dtype_targets = supported.get("column_dtype_targets") or {}
    assert "__split" in dtype_targets
    assert "churned" in dtype_targets


def test_execution_planner_patch_transport_validation_rejects_empty_changes():
    result = _build_patch_transport_validation({"changes": {}})

    assert result.get("accepted") is False
    issues = result.get("issues") or []
    assert any(issue.get("rule") == "contract.patch_payload_trivial" for issue in issues if isinstance(issue, dict))


def test_execution_planner_quality_repair_switches_to_incremental_patch(monkeypatch):
    planner = ExecutionPlannerAgent(api_key="mock_key")
    planner.client = object()
    planner._build_model_client = lambda _model_name: object()

    calls = []
    responses = [
        {
            "contract_version": "4.2",
            "scope": "full_pipeline",
            "strategy_title": "Incremental Repair",
            "business_objective": "Predict churn probability.",
            "output_dialect": {"sep": ",", "decimal": ".", "encoding": "utf-8"},
            "canonical_columns": ["customer_id", "__split", "feature_a", "churned"],
            "required_outputs": ["artifacts/ml/submission.csv"],
            "column_roles": {
                "pre_decision": ["feature_a"],
                "decision": [],
                "outcome": ["churned"],
                "post_decision_audit_only": [],
                "unknown": [],
                "identifiers": ["customer_id"],
                "time_columns": [],
            },
            "allowed_feature_sets": {
                "model_features": ["feature_a"],
                "segmentation_features": [],
                "forbidden_features": ["churned"],
                "audit_only_features": ["__split"],
            },
            "task_semantics": {
                "problem_family": "classification",
                "objective_type": "binary_classification",
                "primary_target": "churned",
                "target_columns": ["churned"],
                "partitioning": {"split_column": "__split"},
                "output_schema": {
                    "prediction_artifact": "artifacts/ml/submission.csv",
                    "required_prediction_columns": ["prob_churned"],
                },
            },
            "artifact_requirements": {},
            "cleaning_gates": [{"name": "no_null_target", "severity": "HARD", "params": {"column": "churned"}}],
            "reviewer_gates": [{"name": "strategy_followed", "severity": "HARD", "params": {}}],
            "data_engineer_runbook": "clean",
            "ml_engineer_runbook": "train",
        },
        {
            "changes": {
                "qa_gates": [{"name": "metrics_present", "severity": "HARD", "params": {}}],
            }
        },
    ]

    def _fake_generate(_client, _prompt, output_token_floor=1024, *, model_name=None, tool_mode="contract"):
        class _Resp:
            def __init__(self, text):
                self.text = text
                self.candidates = []
                self.usage_metadata = None

        calls.append(tool_mode)
        payload = responses[min(len(calls) - 1, len(responses) - 1)]
        return _Resp(json.dumps(payload)), {"max_output_tokens": output_token_floor, "model_name": model_name}

    planner._generate_content_with_budget = _fake_generate

    contract = planner.generate_contract(
        strategy={"required_columns": ["customer_id", "__split", "feature_a", "churned"], "title": "Incremental Repair"},
        business_objective="Predict churn probability.",
        column_inventory=["customer_id", "__split", "feature_a", "churned"],
    )

    assert calls[:2] == ["contract", "patch"]
    assert (contract.get("qa_gates") or [])[0]["name"] == "metrics_present"


def test_empty_tool_payload_is_classified_as_transport_failure(monkeypatch):
    planner = ExecutionPlannerAgent(api_key="mock_key")
    response = MagicMock()
    response.text = "{}"
    response.candidates = []
    response.usage_metadata = None
    planner.client = MagicMock()
    planner.client.generate_content.return_value = response

    contract = planner.generate_contract(
        strategy={"required_columns": ["id", "feature", "target"], "title": "Transport failure"},
        business_objective="Predict target.",
        column_inventory=["id", "feature", "target"],
    )

    assert contract == {}
    diagnostics = planner.last_contract_diagnostics or {}
    transport = diagnostics.get("transport_validation") or {}
    assert transport.get("accepted") is False
    issues = transport.get("issues") or []
    assert any(issue.get("rule") == "contract.transport_payload_empty" for issue in issues if isinstance(issue, dict))
