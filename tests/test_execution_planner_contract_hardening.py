import json
from types import SimpleNamespace

from unittest.mock import MagicMock

from src.agents.execution_planner import (
    _apply_validation_adjudication,
    _build_patch_transport_validation,
    _repair_common_json_damage,
    _build_semantic_guard_validation,
    ExecutionPlannerAgent,
    _apply_planner_structural_support,
    parse_derive_from_expression,
)
from src.utils.contract_accessors import get_cleaning_gates


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
    # Post-migration: no auto-fill of empty sections; LLM output passes through
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

    # Post-migration: path resolution is still applied
    assert clean_dataset.get("output_path") == "artifacts/clean/clean_dataset.csv"
    assert clean_dataset.get("output_manifest_path") == "artifacts/clean/clean_dataset_manifest.json"
    # required_columns no longer auto-projected — LLM must provide them


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

    # Post-migration: operational sections are NOT auto-projected.
    # The LLM must generate evaluation_spec, validation_requirements, etc.
    # Structural support only applies path resolution and schema coercion.
    assert supported.get("scope") == "full_pipeline"
    assert supported.get("strategy_title") == "Binary Churn"
    # Schema coercion normalizes optimization_policy
    assert isinstance(supported.get("optimization_policy"), dict)


def test_planner_structural_support_projects_mean_multi_horizon_primary_metric_from_canonical_semantics():
    contract = {
        "scope": "full_pipeline",
        "strategy_title": "Multi-horizon wildfire risk",
        "business_objective": "Predict risk within 12h, 24h, 48h and 72h for each event.",
        "canonical_columns": ["event_id", "__split", "feature_a", "label_12h", "label_24h", "label_48h", "label_72h"],
        "required_outputs": ["artifacts/ml/cv_metrics.json", "artifacts/submission/submission.csv"],
        "column_roles": {
            "pre_decision": ["feature_a"],
            "decision": [],
            "outcome": ["label_12h", "label_24h", "label_48h", "label_72h"],
            "post_decision_audit_only": [],
            "unknown": [],
            "identifiers": ["event_id"],
            "time_columns": [],
        },
        "allowed_feature_sets": {
            "model_features": ["feature_a"],
            "segmentation_features": [],
            "forbidden_features": ["label_12h", "label_24h", "label_48h", "label_72h"],
            "audit_only_features": ["__split"],
        },
        "task_semantics": {
            "problem_family": "probabilistic multi-horizon supervised classification",
            "objective_type": "predictive",
            "primary_target": "label_12h",
            "target_columns": ["label_12h", "label_24h", "label_48h", "label_72h"],
            "multi_target": True,
            "partitioning": {"split_column": "__split"},
            "output_schema": {
                "prediction_artifact": "artifacts/submission/submission.csv",
                "required_prediction_columns": ["prob_12h", "prob_24h", "prob_48h", "prob_72h"],
            },
        },
        "qa_gates": [
            {
                "name": "metric_selection",
                "severity": "HARD",
                "params": {"rule": "optimize model selection on mean multi-horizon log loss"},
            }
        ],
        "reviewer_gates": [
            {
                "name": "review_metrics",
                "severity": "HARD",
                "params": {"rule": "review OOF mean multi-horizon log loss and per-horizon metrics"},
            }
        ],
        "data_engineer_runbook": "Preserve split and labels.",
        "ml_engineer_runbook": (
            "Select models using aggregated out-of-fold mean multi-horizon log loss and keep submission schema exact."
        ),
    }

    supported = _apply_planner_structural_support(contract)

    # Post-migration: evaluation_spec and validation_requirements are NOT
    # auto-projected. The LLM generates them. Structural support preserves
    # existing contract content and applies only coercion.
    assert supported.get("scope") == "full_pipeline"
    assert supported.get("strategy_title") == "Multi-horizon wildfire risk"
    # Gate normalization is applied via schema registry
    qa_gates = supported.get("qa_gates") or []
    assert len(qa_gates) == 1
    assert qa_gates[0].get("name") == "metric_selection"


def test_planner_structural_support_derives_cleaning_scope_from_active_workstreams():
    contract = {
        "scope": "full_pipeline",
        "active_workstreams": {
            "cleaning": True,
            "feature_engineering": True,
            "model_training": False,
        },
        "canonical_columns": ["lead_id", "created_at", "target"],
        "required_outputs": ["artifacts/clean_dataset.csv", "artifacts/cleaning_manifest.json"],
        "column_roles": {
            "pre_decision": ["created_at"],
            "decision": [],
            "outcome": ["target"],
            "post_decision_audit_only": [],
            "unknown": [],
            "identifiers": ["lead_id"],
            "time_columns": ["created_at"],
        },
        "allowed_feature_sets": {
            "model_features": ["created_at"],
            "segmentation_features": [],
            "forbidden_features": [],
            "audit_only_features": [],
        },
    }

    supported = _apply_planner_structural_support(contract)

    assert supported.get("scope") == "cleaning_only"
    assert (supported.get("active_workstreams") or {}).get("model_training") is False


def test_planner_structural_support_preserves_gate_semantic_extensions():
    contract = {
        "scope": "cleaning_only",
        "cleaning_gates": [
            {
                "name": "segregate_pii_and_leakage",
                "severity": "HARD",
                "action_type": "drop",
                "column_phase": "transform",
                "final_state": "removed",
                "params": {"target_columns": ["email", "phone"]},
            }
        ],
    }

    supported = _apply_planner_structural_support(contract)
    gate = (supported.get("cleaning_gates") or [])[0]

    assert gate.get("action_type") == "drop"
    assert gate.get("column_phase") == "transform"
    assert gate.get("final_state") == "removed"


def test_gate_accessors_preserve_gate_semantic_extensions():
    contract = {
        "cleaning_gates": [
            {
                "name": "segregate_pii_and_leakage",
                "severity": "HARD",
                "action_type": "drop",
                "column_phase": "transform",
                "final_state": "removed",
                "params": {"target_columns": ["email", "phone"]},
            }
        ]
    }

    gate = get_cleaning_gates(contract)[0]

    assert gate.get("action_type") == "drop"
    assert gate.get("column_phase") == "transform"
    assert gate.get("final_state") == "removed"


def test_execution_planner_patch_transport_validation_rejects_empty_changes():
    result = _build_patch_transport_validation({"changes": {}})

    assert result.get("accepted") is False
    issues = result.get("issues") or []
    assert any(issue.get("rule") == "contract.patch_payload_trivial" for issue in issues if isinstance(issue, dict))


def test_semantic_guard_rejects_compiled_contract_that_changes_authoritative_workstreams():
    semantic_core = {
        "scope": "cleaning_only",
        "active_workstreams": {
            "cleaning": True,
            "feature_engineering": True,
            "model_training": False,
        },
        "task_semantics": {
            "problem_family": "classification",
            "objective_type": "binary_classification",
            "primary_target": "target",
            "target_columns": ["target"],
        },
        "model_features": ["feature_a"],
        "required_outputs": ["data/clean/dataset_clean.csv"],
        "column_roles": {
            "pre_decision": ["feature_a"],
            "decision": [],
            "outcome": ["target"],
            "post_decision_audit_only": [],
            "unknown": [],
            "identifiers": ["id"],
            "time_columns": [],
        },
    }
    compiled = {
        "scope": "full_pipeline",
        "active_workstreams": {
            "cleaning": True,
            "feature_engineering": True,
            "model_training": True,
        },
        "task_semantics": {
            "problem_family": "classification",
            "objective_type": "binary_classification",
            "primary_target": "target",
            "target_columns": ["target"],
        },
        "model_features": ["feature_a"],
        "required_outputs": ["data/clean/dataset_clean.csv"],
        "column_roles": semantic_core["column_roles"],
    }

    result = _build_semantic_guard_validation(semantic_core, compiled)

    assert result.get("accepted") is False
    issues = result.get("issues") or []
    rules = {issue.get("rule") for issue in issues if isinstance(issue, dict)}
    assert "semantic_guard.scope_changed" in rules
    assert "semantic_guard.active_workstreams_changed" in rules


def test_semantic_guard_accepts_compatible_scope_projection_from_semantic_workstreams():
    semantic_core = {
        "scope": "data_preparation",
        "active_workstreams": {
            "data_cleaning": True,
            "feature_engineering": True,
            "model_training": False,
        },
        "task_semantics": {
            "problem_family": "descriptive",
            "objective_type": "descriptive",
            "primary_target": "target",
            "target_columns": ["target"],
        },
        "model_features": ["feature_a"],
        "required_outputs": ["data/clean/dataset_clean.csv"],
        "column_roles": {
            "pre_decision": ["feature_a"],
            "decision": [],
            "outcome": ["target"],
            "post_decision_audit_only": [],
            "unknown": [],
            "identifiers": ["id"],
            "time_columns": [],
        },
    }
    compiled = {
        "scope": "cleaning_only",
        "active_workstreams": {
            "cleaning": True,
            "feature_engineering": True,
            "model_training": False,
        },
        "task_semantics": semantic_core["task_semantics"],
        "model_features": ["feature_a"],
        "required_outputs": ["data/clean/dataset_clean.csv"],
        "column_roles": semantic_core["column_roles"],
    }

    result = _build_semantic_guard_validation(semantic_core, compiled)

    assert result.get("accepted") is True
    issues = result.get("issues") or []
    rules = {issue.get("rule") for issue in issues if isinstance(issue, dict)}
    assert "semantic_guard.scope_changed" not in rules
    assert "semantic_guard.active_workstreams_changed" not in rules


def test_semantic_guard_accepts_materialized_required_outputs_with_intent_mapping():
    semantic_core = {
        "scope": "cleaning_only",
        "active_workstreams": {
            "cleaning": True,
            "feature_engineering": True,
            "model_training": False,
        },
        "task_semantics": {
            "problem_family": "data_preparation",
            "objective_type": "descriptive",
            "primary_target": "target_future",
            "target_columns": ["target_future"],
        },
        "model_features": ["feature_a"],
        "required_outputs": ["cleaned_dataset", "quality_report"],
        "column_roles": {
            "pre_decision": ["feature_a"],
            "decision": [],
            "outcome": ["target_future"],
            "post_decision_audit_only": [],
            "unknown": [],
            "identifiers": ["id"],
            "time_columns": [],
        },
    }
    compiled = {
        "scope": "cleaning_only",
        "active_workstreams": semantic_core["active_workstreams"],
        "task_semantics": semantic_core["task_semantics"],
        "model_features": ["feature_a"],
        "required_outputs": [
            {
                "intent": "cleaned_dataset",
                "path": "artifacts/clean/dataset_cleaned.csv",
                "required": True,
            },
            {
                "intent": "quality_report",
                "path": "artifacts/report/quality_report.json",
                "required": True,
            },
        ],
        "column_roles": semantic_core["column_roles"],
    }

    result = _build_semantic_guard_validation(semantic_core, compiled)

    assert result.get("accepted") is True
    rules = {issue.get("rule") for issue in (result.get("issues") or []) if isinstance(issue, dict)}
    assert "semantic_guard.required_outputs_dropped" not in rules


def test_semantic_guard_accepts_semantic_required_outputs_materialized_from_artifact_and_intent():
    semantic_core = {
        "scope": "data_preparation",
        "active_workstreams": {
            "data_cleaning": True,
            "feature_engineering": True,
            "model_training": False,
        },
        "task_semantics": {
            "problem_family": "data_preparation",
            "objective_type": "descriptive",
            "primary_target": "target_future",
            "target_columns": ["target_future"],
        },
        "model_features": ["feature_a"],
        "required_outputs": [
            {
                "artifact": "dataset_cleaned",
                "intent": "A fully traceable, standardized dataset containing all original columns.",
            },
            {
                "artifact": "data_quality_report",
                "intent": "A comprehensive audit report detailing missingness and transformations.",
            },
        ],
        "column_roles": {
            "pre_decision": ["feature_a"],
            "decision": [],
            "outcome": ["target_future"],
            "post_decision_audit_only": [],
            "unknown": [],
            "identifiers": ["id"],
            "time_columns": [],
        },
    }
    compiled = {
        "scope": "cleaning_only",
        "active_workstreams": {
            "cleaning": True,
            "feature_engineering": True,
            "model_training": False,
        },
        "task_semantics": semantic_core["task_semantics"],
        "model_features": ["feature_a"],
        "required_outputs": [
            {
                "intent": "A fully traceable, standardized dataset containing all original columns.",
                "path": "artifacts/clean/dataset_cleaned.csv",
                "required": True,
                "owner": "data_engineer",
                "kind": "dataset",
                "description": "dataset_cleaned",
            },
            {
                "intent": "A comprehensive audit report detailing missingness and transformations.",
                "path": "artifacts/report/data_quality_report.json",
                "required": True,
                "owner": "data_engineer",
                "kind": "report",
                "description": "data_quality_report",
            },
        ],
        "column_roles": semantic_core["column_roles"],
    }

    result = _build_semantic_guard_validation(semantic_core, compiled)

    assert result.get("accepted") is True
    rules = {issue.get("rule") for issue in (result.get("issues") or []) if isinstance(issue, dict)}
    assert "semantic_guard.required_outputs_dropped" not in rules


def test_semantic_guard_accepts_filename_like_semantic_required_outputs_materialized_to_artifact_paths():
    semantic_core = {
        "scope": "cleaning_only",
        "active_workstreams": {
            "data_cleaning": True,
            "feature_engineering": True,
            "model_training": False,
        },
        "task_semantics": {
            "problem_family": "data_preparation",
            "objective_type": "descriptive",
            "primary_target": "target_future",
            "target_columns": ["target_future"],
        },
        "model_features": ["feature_a"],
        "required_outputs": ["dataset_cleaned.csv", "dataset_enriched.csv"],
        "column_roles": {
            "pre_decision": ["feature_a"],
            "decision": [],
            "outcome": ["target_future"],
            "post_decision_audit_only": [],
            "unknown": [],
            "identifiers": ["id"],
            "time_columns": [],
        },
    }
    compiled = {
        "scope": "cleaning_only",
        "active_workstreams": {
            "cleaning": True,
            "feature_engineering": True,
            "model_training": False,
        },
        "task_semantics": semantic_core["task_semantics"],
        "model_features": ["feature_a"],
        "required_outputs": [
            {
                "intent": "dataset_cleaned",
                "path": "artifacts/clean/dataset_cleaned.csv",
                "required": True,
                "owner": "data_engineer",
            },
            {
                "intent": "dataset_enriched",
                "path": "artifacts/clean/dataset_enriched.csv",
                "required": True,
                "owner": "data_engineer",
            },
        ],
        "column_roles": semantic_core["column_roles"],
    }

    result = _build_semantic_guard_validation(semantic_core, compiled)

    assert result.get("accepted") is True
    rules = {issue.get("rule") for issue in (result.get("issues") or []) if isinstance(issue, dict)}
    assert "semantic_guard.required_outputs_dropped" not in rules


def test_semantic_guard_accepts_semantic_required_outputs_with_artifact_only_alias():
    semantic_core = {
        "scope": "data_preparation",
        "active_workstreams": {
            "data_cleaning": True,
            "feature_engineering": True,
            "model_training": False,
        },
        "task_semantics": {
            "problem_family": "data_preparation",
            "objective_type": "descriptive",
            "primary_target": "target_future",
            "target_columns": ["target_future"],
        },
        "model_features": ["feature_a"],
        "required_outputs": [
            {"artifact": "dataset_enriched"},
            {"artifact": "exclusion_registry"},
        ],
        "column_roles": {
            "pre_decision": ["feature_a"],
            "decision": [],
            "outcome": ["target_future"],
            "post_decision_audit_only": [],
            "unknown": [],
            "identifiers": ["id"],
            "time_columns": [],
        },
    }
    compiled = {
        "scope": "cleaning_only",
        "active_workstreams": {
            "cleaning": True,
            "feature_engineering": True,
            "model_training": False,
        },
        "task_semantics": semantic_core["task_semantics"],
        "model_features": ["feature_a"],
        "required_outputs": [
            {
                "path": "artifacts/clean/dataset_enriched.csv",
                "required": True,
                "owner": "data_engineer",
                "kind": "dataset",
                "description": "dataset_enriched",
            },
            {
                "path": "artifacts/report/exclusion_registry.json",
                "required": True,
                "owner": "data_engineer",
                "kind": "report",
                "description": "exclusion_registry",
            },
        ],
        "column_roles": semantic_core["column_roles"],
    }

    result = _build_semantic_guard_validation(semantic_core, compiled)

    assert result.get("accepted") is True
    rules = {issue.get("rule") for issue in (result.get("issues") or []) if isinstance(issue, dict)}
    assert "semantic_guard.required_outputs_dropped" not in rules


def test_apply_validation_adjudication_can_clear_ambiguous_required_output_issue():
    validation_result = {
        "status": "error",
        "accepted": False,
        "issues": [
            {
                "severity": "error",
                "rule": "semantic_guard.required_outputs_dropped",
                "message": "Compiled contract did not preserve semantic output intent.",
                "item": {"missing_semantic_outputs": ["cleaned_dataset"]},
                "adjudicable": True,
            }
        ],
        "summary": {"error_count": 1, "warning_count": 0},
    }
    adjudication = {
        "issue_verdicts": [
            {
                "issue_index": 0,
                "decision": "clear",
                "reason": "The compiled output materializes cleaned_dataset as a concrete file artifact.",
            }
        ]
    }

    result = _apply_validation_adjudication(validation_result, adjudication)

    assert result.get("accepted") is True
    assert result.get("status") == "ok"
    assert result.get("issues") == []
    assert (result.get("summary") or {}).get("adjudicated") is True


def test_repair_common_json_damage_closes_truncated_terminal_string_and_object():
    raw = (
        '{\n'
        '  "scope": "cleaning_only",\n'
        '  "optimization_policy": {\n'
        '    "primary_objective": "maximize quality",\n'
        '    "fallback_strategy": "keep original and flag parse'
    )

    repaired = _repair_common_json_damage(raw)
    parsed = json.loads(repaired)

    assert parsed["scope"] == "cleaning_only"
    assert parsed["optimization_policy"]["fallback_strategy"] == "keep original and flag parse"


def test_execution_planner_is_one_shot_and_preserves_first_invalid_compile(monkeypatch):
    planner = ExecutionPlannerAgent(api_key="mock_key")
    planner.client = object()
    planner._build_model_client = lambda _model_name: object()
    planner.model_chain = ["model_a", "model_b"]

    calls = []
    responses = [
        {
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
            },
            "active_workstreams": {
                "cleaning": True,
                "feature_engineering": True,
                "model_training": True,
            },
            "model_features": ["feature_a"],
            "cleaning_gates": [{"name": "no_null_target", "severity": "HARD", "params": {"column": "churned"}}],
            "qa_gates": [{"name": "metrics_present", "severity": "HARD", "params": {}}],
            "reviewer_gates": [{"name": "strategy_followed", "severity": "HARD", "params": {}}],
            "data_engineer_runbook": "clean",
            "optimization_policy": {
                "enabled": False,
                "max_rounds": 1,
                "quick_eval_folds": 1,
                "full_eval_folds": 1,
                "min_delta": 0,
                "patience": 1,
                "allow_model_switch": False,
                "allow_ensemble": False,
                "allow_hpo": False,
                "allow_feature_engineering": False,
                "allow_calibration": False,
            },
        },
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
    ]

    def _fake_generate(_client, _prompt, output_token_floor=1024, *, model_name=None, tool_mode="contract"):
        class _Resp:
            def __init__(self, text):
                self.text = text
                self.candidates = []
                self.usage_metadata = None

        calls.append((tool_mode, model_name))
        if len(calls) > len(responses):
            raise AssertionError("execution_planner performed an unexpected retry")
        payload = responses[len(calls) - 1]
        return _Resp(json.dumps(payload)), {"max_output_tokens": output_token_floor, "model_name": model_name}

    planner._generate_content_with_budget = _fake_generate

    contract = planner.generate_contract(
        strategy={"required_columns": ["customer_id", "__split", "feature_a", "churned"], "title": "Incremental Repair"},
        business_objective="Predict churn probability.",
        column_inventory=["customer_id", "__split", "feature_a", "churned"],
    )

    assert calls == [("semantic", "model_a"), ("contract", "model_a")]
    diagnostics = planner.last_contract_diagnostics or {}
    summary = diagnostics.get("summary") or {}
    assert summary.get("accepted") is False
    clean_dataset = (contract.get("artifact_requirements") or {}).get("clean_dataset") or {}
    assert clean_dataset == {}


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
    assert any(issue.get("rule") == "semantic_core.transport_payload_empty" for issue in issues if isinstance(issue, dict))


def test_execution_planner_extracts_contract_text_from_raw_openrouter_body_when_sdk_content_is_empty():
    planner = ExecutionPlannerAgent(api_key="mock_key")
    response = SimpleNamespace(
        candidates=[],
        text="",
        choices=[SimpleNamespace(message=SimpleNamespace(content="", reasoning=None))],
        _codex_raw_body=json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": '{"scope":"cleaning_only","strategy_title":"x"}',
                                }
                            ]
                        }
                    }
                ]
            }
        ),
    )

    extracted = planner._extract_openai_response_text(response)

    assert extracted == '{"scope":"cleaning_only","strategy_title":"x"}'


def test_empty_completion_with_tokens_is_classified_as_transport_empty_completion(monkeypatch):
    planner = ExecutionPlannerAgent(api_key="mock_key")
    response = SimpleNamespace(
        text="",
        candidates=[],
        choices=[SimpleNamespace(message=SimpleNamespace(content=""), finish_reason="stop")],
        usage=SimpleNamespace(completion_tokens=1234, prompt_tokens=100),
    )
    planner.client = MagicMock()
    planner.client.generate_content.return_value = response

    contract = planner.generate_contract(
        strategy={"required_columns": ["id", "feature", "target"], "title": "Empty completion"},
        business_objective="Predict target.",
        column_inventory=["id", "feature", "target"],
    )

    assert contract == {}
    diagnostics = planner.last_contract_diagnostics or {}
    transport = diagnostics.get("transport_validation") or {}
    assert transport.get("accepted") is False
    issues = transport.get("issues") or []
    assert any(issue.get("rule") == "semantic_core.transport_empty_completion" for issue in issues if isinstance(issue, dict))
