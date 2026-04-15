import json
from types import SimpleNamespace

from unittest.mock import MagicMock

from src.agents.execution_planner import (
    _apply_validation_adjudication,
    _build_patch_transport_validation,
    _repair_common_json_damage,
    _build_semantic_guard_validation,
    _infer_strategy_audit_only_columns,
    _reconcile_compiled_task_semantics,
    _reconcile_compiled_feature_surfaces,
    _reconcile_semantic_core_with_dataset_semantics,
    ExecutionPlannerAgent,
    _apply_planner_structural_support,
    parse_derive_from_expression,
)
from src.utils.contract_validator import validate_contract_minimal_readonly
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


def test_planner_structural_support_separates_cleaned_and_enriched_dataset_bindings():
    contract = {
        "scope": "cleaning_only",
        "canonical_columns": ["lead_id", "feature_a", "feature_b", "target"],
        "column_roles": {
            "pre_decision": ["feature_a", "feature_b"],
            "outcome": ["target"],
            "identifiers": ["lead_id"],
        },
        "allowed_feature_sets": {
            "model_features": ["feature_a", "feature_b"],
            "segmentation_features": [],
            "forbidden_features": [],
            "audit_only_features": ["lead_id"],
        },
        "future_ml_handoff": {"enabled": True, "target_columns": ["target"]},
        "required_outputs": [
            {"intent": "cleaned_dataset", "path": "artifacts/clean/dataset_cleaned.csv", "owner": "data_engineer"},
            {"intent": "enriched_dataset", "path": "artifacts/clean/dataset_enriched.csv", "owner": "data_engineer"},
            {"path": "artifacts/clean/cleaning_manifest.json", "owner": "data_engineer"},
        ],
        "artifact_requirements": {},
    }

    supported = _apply_planner_structural_support(contract)
    artifact_requirements = supported.get("artifact_requirements") or {}

    assert (artifact_requirements.get("cleaned_dataset") or {}).get("output_path") == "artifacts/clean/dataset_cleaned.csv"
    assert (artifact_requirements.get("enriched_dataset") or {}).get("output_path") == "artifacts/clean/dataset_enriched.csv"
    assert (artifact_requirements.get("enriched_dataset") or {}).get("required_columns") == ["feature_a", "feature_b", "target"]


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


def test_semantic_guard_accepts_required_outputs_materialized_via_semantic_overlap():
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
            "primary_target": "ref_score",
            "target_columns": ["ref_score"],
        },
        "model_features": ["feature_a"],
        "required_outputs": [
            "cleaned_scoring_dataset",
            "ranked_scoring_output_for_FE_2025_10_31",
            "optimized_weight_spec",
        ],
        "column_roles": {
            "pre_decision": ["feature_a"],
            "decision": [],
            "outcome": ["ref_score"],
            "post_decision_audit_only": [],
            "unknown": [],
            "identifiers": ["case_id"],
            "time_columns": ["FE"],
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
                "intent": "cleaned_dataset",
                "path": "artifacts/clean/scoring_dataset_cleaned.csv",
                "required": True,
                "owner": "data_engineer",
            },
            {
                "intent": "ranked_scoring_output",
                "path": "artifacts/ml/final_scores_2025_10_31.csv",
                "required": True,
                "owner": "ml_engineer",
            },
            {
                "intent": "optimized_weight_spec",
                "path": "artifacts/ml/weights.json",
                "required": True,
                "owner": "ml_engineer",
            },
        ],
        "column_roles": semantic_core["column_roles"],
    }

    result = _build_semantic_guard_validation(semantic_core, compiled)

    assert result.get("accepted") is True
    rules = {issue.get("rule") for issue in (result.get("issues") or []) if isinstance(issue, dict)}
    assert "semantic_guard.required_outputs_dropped" not in rules


def test_semantic_guard_keeps_ambiguous_semantic_required_outputs_as_nonblocking_warning():
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
            "primary_target": "ref_score",
            "target_columns": ["ref_score"],
        },
        "model_features": ["feature_a"],
        "required_outputs": [
            "case_summary_table",
            "risk_register_and_bucket_improvement_note",
        ],
        "column_roles": {
            "pre_decision": ["feature_a"],
            "decision": [],
            "outcome": ["ref_score"],
            "post_decision_audit_only": [],
            "unknown": [],
            "identifiers": ["case_id"],
            "time_columns": ["FE"],
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
                "intent": "cleaned_dataset",
                "path": "artifacts/clean/scoring_dataset_cleaned.csv",
                "required": True,
            }
        ],
        "column_roles": semantic_core["column_roles"],
    }

    result = _build_semantic_guard_validation(semantic_core, compiled)

    assert result.get("accepted") is True
    matching_issues = [
        issue
        for issue in (result.get("issues") or [])
        if isinstance(issue, dict) and issue.get("rule") == "semantic_guard.required_outputs_dropped"
    ]
    assert matching_issues
    assert all(str(issue.get("severity") or "").lower() == "warning" for issue in matching_issues)
    assert all(issue.get("adjudicable") is True for issue in matching_issues)


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


def test_contract_compiler_retries_after_truncated_transport_response(monkeypatch):
    planner = ExecutionPlannerAgent(api_key="mock_key")
    planner.client = object()
    planner._build_model_client = lambda _model_name: object()

    semantic_payload = {
        "scope": "full_pipeline",
        "strategy_title": "Retry truncation",
        "business_objective": "Predict target.",
        "output_dialect": {"sep": ",", "decimal": ".", "encoding": "utf-8"},
        "canonical_columns": ["id", "feature", "target"],
        "allowed_feature_sets": {
            "model_features": ["feature"],
            "segmentation_features": [],
            "forbidden_features": ["target"],
            "audit_only_features": ["id"],
        },
        "task_semantics": {
            "problem_family": "classification",
            "objective_type": "binary_classification",
            "primary_target": "target",
            "target_columns": ["target"],
            "prediction_unit": "row",
        },
        "active_workstreams": {"data_engineering": True, "model_training": True, "review": True},
        "required_outputs": ["artifacts/ml/predictions.csv"],
        "column_roles": {
            "pre_decision": ["feature"],
            "decision": [],
            "outcome": ["target"],
            "post_decision_audit_only": [],
            "identifiers": ["id"],
            "time_columns": [],
            "unknown": [],
        },
        "model_features": ["feature"],
        "cleaning_gates": [{"name": "cleaned_dataset_present", "severity": "HARD", "params": {}}],
        "qa_gates": [{"name": "metrics_present", "severity": "HARD", "params": {}}],
        "reviewer_gates": [{"name": "strategy_followed", "severity": "HARD", "params": {}}],
        "data_engineer_runbook": {"steps": ["load", "clean", "persist"]},
        "optimization_policy": {"primary_objective": "maximize roc_auc"},
    }
    compiled_contract = {
        "contract_version": "5.0",
        "scope": "full_pipeline",
        "strategy_title": "Retry truncation",
        "business_objective": "Predict target.",
        "output_dialect": {"sep": ",", "decimal": ".", "encoding": "utf-8"},
        "canonical_columns": ["id", "feature", "target"],
        "active_workstreams": {"data_engineering": True, "model_training": True, "review": True},
        "allowed_feature_sets": semantic_payload["allowed_feature_sets"],
        "task_semantics": semantic_payload["task_semantics"],
        "column_roles": semantic_payload["column_roles"],
        "required_outputs": [{"path": "artifacts/ml/predictions.csv", "owner": "ml_engineer", "required": True}],
        "model_features": ["feature"],
        "cleaning_gates": [],
        "qa_gates": [],
        "reviewer_gates": [],
        "data_engineer_runbook": {"steps": ["load", "clean", "persist"]},
        "optimization_policy": {"primary_objective": "maximize roc_auc"},
        "shared": {
            "column_dtype_targets": {"id": {"target_dtype": "object"}, "target": {"target_dtype": "int64"}},
            "optimization_policy": {"primary_objective": "maximize roc_auc"},
            "iteration_policy": {"max_iterations": 1, "metric_improvement_max": 0, "runtime_fix_max": 0, "compliance_bootstrap_max": 0},
        },
        "data_engineer": {
            "artifact_requirements": {
                "cleaned_dataset": {
                    "output_path": "artifacts/clean/dataset_cleaned.csv",
                    "output_manifest_path": "artifacts/clean/cleaning_manifest.json",
                    "required_columns": ["id", "feature", "target"],
                }
            }
        },
        "ml_engineer": {
            "artifact_requirements": {},
            "evaluation_spec": {"primary_metric": "roc_auc"},
            "validation_requirements": {"primary_metric": "roc_auc"},
        },
        "reviewer": {"reviewer_gates": []},
        "qa_reviewer": {"qa_gates": []},
        "reporting_policy": {},
    }

    calls = []

    def _response(text, *, finish_reason="stop", completion_tokens=11):
        return SimpleNamespace(
            text=text,
            candidates=[],
            choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish_reason)],
            usage=SimpleNamespace(completion_tokens=completion_tokens, prompt_tokens=17),
        )

    def _fake_generate(_client, prompt, output_token_floor=1024, *, model_name=None, tool_mode="contract"):
        calls.append({"tool_mode": tool_mode, "prompt": prompt})
        if tool_mode == "semantic":
            return _response(json.dumps(semantic_payload)), {"max_output_tokens": output_token_floor}
        if len([call for call in calls if call["tool_mode"] == "contract"]) == 1:
            return _response("", finish_reason="length", completion_tokens=32768), {"max_output_tokens": output_token_floor}
        return _response(json.dumps(compiled_contract)), {"max_output_tokens": output_token_floor}

    planner._generate_content_with_budget = _fake_generate

    contract = planner.generate_contract(
        strategy={"required_columns": ["id", "feature", "target"], "title": "Retry truncation"},
        business_objective="Predict target.",
        column_inventory=["id", "feature", "target"],
    )

    assert contract.get("strategy_title") == "Retry truncation"
    assert [call["tool_mode"] for call in calls] == ["semantic", "contract", "contract"]
    assert "TRANSPORT RETRY FROM PREVIOUS ATTEMPT" in calls[-1]["prompt"]
    diagnostics = planner.last_planner_diag or []
    assert any(
        "contract.transport_truncated" in (entry.get("transport_issue_rules") or [])
        for entry in diagnostics
        if isinstance(entry, dict)
    )


def test_execution_planner_records_llm_call_trace_for_main_stages(monkeypatch):
    planner = ExecutionPlannerAgent(api_key="mock_key")
    planner.client = object()
    planner._build_model_client = lambda _model_name: object()

    semantic_payload = {
        "scope": "full_pipeline",
        "strategy_title": "Duplicate outputs",
        "business_objective": "Predict target.",
        "output_dialect": {"sep": ",", "decimal": ".", "encoding": "utf-8"},
        "canonical_columns": ["id", "feature", "target"],
        "allowed_feature_sets": {
            "model_features": ["feature"],
            "segmentation_features": [],
            "forbidden_features": ["target"],
            "audit_only_features": ["id"],
        },
        "task_semantics": {
            "problem_family": "classification",
            "objective_type": "binary_classification",
            "primary_target": "target",
            "target_columns": ["target"],
            "prediction_unit": "row",
        },
        "active_workstreams": {
            "data_engineering": True,
            "model_training": True,
            "review": True,
        },
        "required_outputs": ["artifacts/ml/predictions.csv", "artifacts/ml/predictions.csv"],
        "column_roles": {
            "pre_decision": ["feature"],
            "decision": [],
            "outcome": ["target"],
            "post_decision_audit_only": [],
            "identifiers": ["id"],
            "time_columns": [],
            "unknown": [],
        },
        "model_features": ["feature"],
        "cleaning_gates": [],
        "qa_gates": [],
        "reviewer_gates": [],
        "data_engineer_runbook": {"steps": ["load", "clean", "persist"]},
        "optimization_policy": {"primary_objective": "maximize roc_auc"},
    }
    compiled_contract = {
        "contract_version": "5.0",
        "scope": "full_pipeline",
        "strategy_title": "Duplicate outputs",
        "business_objective": "Predict target.",
        "output_dialect": {"sep": ",", "decimal": ".", "encoding": "utf-8"},
        "canonical_columns": ["id", "feature", "target"],
        "active_workstreams": {
            "data_engineering": True,
            "model_training": True,
            "review": True,
        },
        "allowed_feature_sets": {
            "model_features": ["feature"],
            "segmentation_features": [],
            "forbidden_features": ["target"],
            "audit_only_features": ["id"],
        },
        "task_semantics": {
            "problem_family": "classification",
            "objective_type": "binary_classification",
            "primary_target": "target",
            "target_columns": ["target"],
            "prediction_unit": "row",
        },
        "column_roles": {
            "pre_decision": ["feature"],
            "decision": [],
            "outcome": ["target"],
            "post_decision_audit_only": [],
            "identifiers": ["id"],
            "time_columns": [],
            "unknown": [],
        },
        "required_outputs": [
            {"path": "artifacts/ml/predictions.csv", "owner": "ml_engineer", "required": True},
            {"path": "artifacts/ml/predictions.csv", "owner": "ml_engineer", "required": True},
        ],
        "model_features": ["feature"],
        "cleaning_gates": [],
        "qa_gates": [],
        "reviewer_gates": [],
        "data_engineer_runbook": {"steps": ["load", "clean", "persist"]},
        "optimization_policy": {"primary_objective": "maximize roc_auc"},
        "shared": {
            "column_dtype_targets": {"id": {"target_dtype": "object"}, "target": {"target_dtype": "float64"}},
            "optimization_policy": {"primary_objective": "maximize roc_auc"},
            "iteration_policy": {"max_iterations": 1, "metric_improvement_max": 0, "runtime_fix_max": 0, "compliance_bootstrap_max": 0},
        },
        "data_engineer": {
            "artifact_requirements": {
                "cleaned_dataset": {
                    "output_path": "artifacts/clean/dataset_cleaned.csv",
                    "output_manifest_path": "artifacts/clean/cleaning_manifest.json",
                    "required_columns": ["id", "feature", "target"],
                }
            }
        },
        "ml_engineer": {
            "artifact_requirements": {},
            "evaluation_spec": {"primary_metric": "roc_auc"},
            "validation_requirements": {"primary_metric": "roc_auc"},
        },
        "reviewer": {"reviewer_gates": []},
        "qa_reviewer": {"qa_gates": []},
        "reporting_policy": {},
    }

    calls = []

    def _fake_generate(_client, prompt, output_token_floor=1024, *, model_name=None, tool_mode="contract"):
        if "AMBIGUOUS_ISSUES_JSON" in prompt:
            payload = {"issue_verdicts": [{"issue_index": 1, "decision": "clear", "reason": "duplicate output path is acceptable here"}]}
        elif tool_mode == "semantic":
            payload = semantic_payload
        else:
            payload = compiled_contract
        response = SimpleNamespace(
            text=json.dumps(payload),
            candidates=[],
            usage_metadata={"completion_tokens": 11, "prompt_tokens": 17},
        )
        calls.append({"tool_mode": tool_mode, "model_name": model_name, "is_adjudicator": "AMBIGUOUS_ISSUES_JSON" in prompt})
        return response, {"max_output_tokens": output_token_floor, "model_name": model_name}

    planner._generate_content_with_budget = _fake_generate

    planner.generate_contract(
        strategy={"required_columns": ["id", "feature", "target"], "title": "Duplicate outputs"},
        business_objective="Predict target.",
        column_inventory=["id", "feature", "target"],
    )

    llm_trace = planner.last_llm_call_trace or []
    assert llm_trace
    assert llm_trace[0].get("stage") == "semantic_core"
    assert all(entry.get("prompt_fingerprint") for entry in llm_trace)


def test_semantic_guard_allows_pre_decision_role_refinement_into_structural_buckets():
    semantic_core = {
        "model_features": ["Importe Norm", "RIIM10 Norm"],
        "column_roles": {
            "pre_decision": [
                "EntityId",
                "CodPartidaAbierta",
                "FE",
                "FV",
                "DaysToDue",
                "Sector",
                "Importe Norm",
                "RIIM10 Norm",
            ],
            "operational_dependencies": ["EntityId", "CodPartidaAbierta", "FE"],
            "outcome": ["Score"],
        }
    }
    compiled_contract = {
        "column_roles": {
            "pre_decision": ["Importe Norm", "RIIM10 Norm"],
            "identifiers": ["EntityId", "CodPartidaAbierta"],
            "time_columns": ["FE", "FV"],
            "unknown": ["Sector", "DaysToDue"],
            "outcome": ["Score"],
        }
    }

    result = _build_semantic_guard_validation(semantic_core, compiled_contract)

    assert not any(
        issue.get("rule") == "semantic_guard.column_roles_changed"
        for issue in (result.get("issues") or [])
        if isinstance(issue, dict)
    )


def test_semantic_guard_allows_non_target_outcomes_reclassified_as_audit_forbidden():
    semantic_core = {
        "task_semantics": {
            "primary_target": "churn_60d",
            "target_columns": ["churn_60d"],
        },
        "model_features": ["arr_current"],
        "column_roles": {
            "pre_decision": ["arr_current"],
            "outcome": ["churn_60d", "cancelled_at", "final_account_status"],
        },
    }
    compiled_contract = {
        "task_semantics": {
            "primary_target": "churn_60d",
            "target_columns": ["churn_60d"],
        },
        "model_features": ["arr_current"],
        "column_roles": {
            "pre_decision": ["arr_current"],
            "outcome": ["churn_60d"],
            "post_decision_audit_only": ["cancelled_at", "final_account_status"],
        },
        "allowed_feature_sets": {
            "model_features": ["arr_current"],
            "audit_only_features": ["cancelled_at", "final_account_status"],
            "forbidden_features": ["cancelled_at", "final_account_status"],
        },
        "ml_engineer": {
            "qa_gates": [
                {
                    "name": "no_leakage_columns_in_feature_matrix",
                    "params": {"forbidden_columns": ["cancelled_at", "final_account_status"]},
                }
            ]
        },
    }

    result = _build_semantic_guard_validation(semantic_core, compiled_contract)

    assert result.get("accepted") is True
    assert not any(
        issue.get("rule") == "semantic_guard.column_roles_changed"
        for issue in (result.get("issues") or [])
        if isinstance(issue, dict)
    )


def test_semantic_guard_rejects_target_reclassified_as_audit_only():
    semantic_core = {
        "task_semantics": {
            "primary_target": "churn_60d",
            "target_columns": ["churn_60d"],
        },
        "model_features": ["arr_current"],
        "column_roles": {
            "pre_decision": ["arr_current"],
            "outcome": ["churn_60d"],
        },
    }
    compiled_contract = {
        "task_semantics": {
            "primary_target": "churn_60d",
            "target_columns": ["churn_60d"],
        },
        "model_features": ["arr_current"],
        "column_roles": {
            "pre_decision": ["arr_current"],
            "outcome": [],
            "post_decision_audit_only": ["churn_60d"],
        },
        "allowed_feature_sets": {
            "model_features": ["arr_current"],
            "audit_only_features": ["churn_60d"],
            "forbidden_features": ["churn_60d"],
        },
    }

    result = _build_semantic_guard_validation(semantic_core, compiled_contract)

    assert result.get("accepted") is False
    issues = [issue for issue in (result.get("issues") or []) if isinstance(issue, dict)]
    assert any(
        issue.get("rule") == "semantic_guard.column_roles_changed"
        and ((issue.get("item") or {}).get("missing") == ["churn_60d"])
        for issue in issues
    )


def test_semantic_guard_rejects_outcome_reclassification_when_target_semantics_missing():
    semantic_core = {
        "model_features": ["arr_current"],
        "column_roles": {
            "pre_decision": ["arr_current"],
            "outcome": ["label_or_proxy"],
        },
    }
    compiled_contract = {
        "model_features": ["arr_current"],
        "column_roles": {
            "pre_decision": ["arr_current"],
            "outcome": [],
            "post_decision_audit_only": ["label_or_proxy"],
        },
        "allowed_feature_sets": {
            "model_features": ["arr_current"],
            "audit_only_features": ["label_or_proxy"],
            "forbidden_features": ["label_or_proxy"],
        },
    }

    result = _build_semantic_guard_validation(semantic_core, compiled_contract)

    assert result.get("accepted") is False
    issues = [issue for issue in (result.get("issues") or []) if isinstance(issue, dict)]
    assert any(
        issue.get("rule") == "semantic_guard.column_roles_changed"
        and ((issue.get("item") or {}).get("missing") == ["label_or_proxy"])
        for issue in issues
    )


def test_reconcile_compiled_feature_surfaces_preserves_semantic_model_features_and_promotes_derived_extras():
    semantic_core = {
        "model_features": ["region", "country", "industry"],
    }
    compiled_contract = {
        "model_features": ["region", "country", "industry", "account_age_days", "days_since_last_qbr"],
        "allowed_feature_sets": {
            "model_features": ["region", "country", "industry"],
            "derived_temporal_features": ["account_age_days", "days_since_last_qbr"],
            "audit_only_features": ["account_created_at"],
        },
        "feature_engineering_plan": {
            "derived_columns": ["account_age_days", "days_since_last_qbr"],
        },
    }

    repaired = _reconcile_compiled_feature_surfaces(compiled_contract, semantic_core)

    assert repaired.get("model_features") == ["region", "country", "industry"]
    assert set(repaired.get("derived_columns") or []) == {"account_age_days", "days_since_last_qbr"}

    validation = _build_semantic_guard_validation(semantic_core, repaired)
    rules = {issue.get("rule") for issue in (validation.get("issues") or []) if isinstance(issue, dict)}
    assert "semantic_guard.model_features_changed" not in rules


def test_reconcile_semantic_core_promotes_authoritative_dataset_semantics_row_rules():
    semantic_core = {
        "task_semantics": {
            "primary_target": "churn_60d",
            "training_rows_rule": "churn_60d IS NOT NULL",
            "scoring_rows_rule": "churn_60d IS NULL",
        }
    }
    data_profile = {
        "dataset_semantics": {
            "primary_target": "churn_60d",
            "training_rows_rule": "churn_60d IS NOT NULL AND snapshot_month_end <= '2025-10-31'",
            "scoring_rows_rule_primary": "churn_60d IS NULL",
            "scoring_rows_rule_secondary": "churn_60d IS NOT NULL AND snapshot_month_end > '2025-10-31'",
            "split_candidates": ["snapshot_month_end"],
            "id_candidates": ["account_id"],
        }
    }

    repaired = _reconcile_semantic_core_with_dataset_semantics(semantic_core, data_profile)
    task_semantics = repaired.get("task_semantics") or {}

    assert task_semantics.get("training_rows_rule") == "churn_60d IS NOT NULL AND snapshot_month_end <= '2025-10-31'"
    assert task_semantics.get("scoring_rows_rule") == "churn_60d IS NULL"
    assert task_semantics.get("scoring_rows_rule_secondary") == "churn_60d IS NOT NULL AND snapshot_month_end > '2025-10-31'"
    assert task_semantics.get("temporal_ordering_column") == "snapshot_month_end"
    assert task_semantics.get("entity_identifier") == "account_id"


def test_reconcile_compiled_task_semantics_preserves_semantic_partition_rules_and_cleaning_gate_params():
    semantic_core = {
        "task_semantics": {
            "primary_target": "churn_60d",
            "training_rows_rule": "churn_60d IS NOT NULL AND snapshot_month_end <= '2025-10-31'",
            "scoring_rows_rule": "churn_60d IS NULL",
            "scoring_rows_rule_secondary": "churn_60d IS NOT NULL AND snapshot_month_end > '2025-10-31'",
        }
    }
    compiled_contract = {
        "contract_version": "5.0",
        "shared": {
            "task_semantics": {
                "primary_target": "churn_60d",
                "training_rows_rule": "churn_60d IS NOT NULL",
                "scoring_rows_rule": "churn_60d IS NULL",
            },
        },
        "data_engineer": {
            "cleaning_gates": [
                {
                    "name": "enforce_cohort_partition",
                    "severity": "HARD",
                    "params": {
                        "training_rows_rule": "churn_60d IS NOT NULL",
                        "scoring_rows_rule": "churn_60d IS NULL",
                        "scoring_rows_rule_secondary": "",
                    },
                }
            ],
            "artifact_requirements": {
                "cleaned_dataset": {
                    "cohort_split_column": {
                        "values": {
                            "train": "churn_60d IS NOT NULL",
                            "score_primary": "churn_60d IS NULL",
                            "score_secondary": "",
                        }
                    }
                }
            },
        },
    }

    repaired = _reconcile_compiled_task_semantics(compiled_contract, semantic_core)
    task_semantics = (repaired.get("shared") or {}).get("task_semantics") or {}
    gate_params = (((repaired.get("data_engineer") or {}).get("cleaning_gates") or [])[0] or {}).get("params") or {}
    cohort_values = ((((repaired.get("data_engineer") or {}).get("artifact_requirements") or {}).get("cleaned_dataset") or {}).get("cohort_split_column") or {}).get("values") or {}

    assert task_semantics.get("training_rows_rule") == "churn_60d IS NOT NULL AND snapshot_month_end <= '2025-10-31'"
    assert gate_params.get("training_rows_rule") == "churn_60d IS NOT NULL AND snapshot_month_end <= '2025-10-31'"
    assert gate_params.get("scoring_rows_rule_secondary") == "churn_60d IS NOT NULL AND snapshot_month_end > '2025-10-31'"
    assert cohort_values.get("train") == "churn_60d IS NOT NULL AND snapshot_month_end <= '2025-10-31'"
    assert cohort_values.get("score_secondary") == "churn_60d IS NOT NULL AND snapshot_month_end > '2025-10-31'"


def test_infer_strategy_audit_only_columns_from_reasoning_text():
    strategy = {
        "objective_reasoning": (
            "CSM owner should be excluded from the initial model and evaluated separately as a stratification variable. "
            "risk_flag_internal is reporting-only."
        )
    }

    inferred = _infer_strategy_audit_only_columns(
        strategy,
        ["account_id", "csm_owner", "risk_flag_internal", "arr_current"],
    )

    assert inferred == ["csm_owner", "risk_flag_internal"]


def test_reconcile_compiled_feature_surfaces_preserves_semantic_audit_only_exclusions():
    semantic_core = {
        "allowed_feature_sets": {
            "audit_only_features": ["csm_owner"],
            "forbidden_features": ["risk_flag_internal"],
        },
        "column_roles": {
            "post_decision_audit_only": ["csm_owner"],
        },
    }
    compiled_contract = {
        "model_features": ["arr_current", "csm_owner", "risk_flag_internal"],
        "allowed_feature_sets": {
            "model_features": ["arr_current", "csm_owner", "risk_flag_internal"],
            "audit_only_features": [],
            "forbidden_features": [],
        },
    }

    repaired = _reconcile_compiled_feature_surfaces(compiled_contract, semantic_core)
    validation = _build_semantic_guard_validation(semantic_core, repaired)
    rules = {issue.get("rule") for issue in (validation.get("issues") or []) if isinstance(issue, dict)}

    assert repaired.get("model_features") == ["arr_current"]
    allowed = repaired.get("allowed_feature_sets") or {}
    assert allowed.get("audit_only_features") == ["csm_owner"]
    assert allowed.get("forbidden_features") == ["risk_flag_internal"]
    assert "semantic_guard.audit_only_reintroduced" not in rules
    assert "semantic_guard.forbidden_features_reintroduced" not in rules


def test_contract_validation_accepts_explicit_optimization_direction_and_tie_breakers():
    planner = ExecutionPlannerAgent(api_key=None)
    contract = {
        "contract_version": "5.0",
        "scope": "full_pipeline",
        "strategy_title": "Metric-aware optimization",
        "business_objective": "Predict target while preferring stable challengers.",
        "output_dialect": {"sep": ",", "decimal": ".", "encoding": "utf-8"},
        "canonical_columns": ["id", "feature_a", "target"],
        "column_roles": {
            "pre_decision": ["feature_a"],
            "decision": [],
            "outcome": ["target"],
            "post_decision_audit_only": [],
            "unknown": [],
            "identifiers": ["id"],
            "time_columns": [],
        },
        "allowed_feature_sets": {
            "model_features": ["feature_a"],
            "segmentation_features": [],
            "forbidden_features": ["target"],
            "audit_only_features": ["id"],
        },
        "task_semantics": {
            "problem_family": "regression",
            "objective_type": "regression",
            "primary_target": "target",
            "target_columns": ["target"],
        },
        "active_workstreams": {"cleaning": True, "feature_engineering": True, "model_training": True},
        "model_features": ["feature_a"],
        "required_outputs": [{"path": "artifacts/ml/cv_metrics.json", "required": True}],
        "cleaning_gates": [],
        "qa_gates": [],
        "reviewer_gates": [],
        "data_engineer_runbook": {"steps": ["load"]},
        "ml_engineer_runbook": {"steps": ["train"]},
        "evaluation_spec": {
            "objective_type": "regression",
            "primary_target": "target",
            "primary_metric": "mae",
            "label_columns": ["target"],
        },
        "validation_requirements": {
            "method": "cross_validation",
            "primary_metric": "mae",
            "metrics_to_report": ["mae"],
        },
        "optimization_policy": {
            "enabled": True,
            "max_rounds": 4,
            "quick_eval_folds": 2,
            "full_eval_folds": 5,
            "min_delta": 0.001,
            "patience": 2,
            "optimization_direction": "minimize",
            "tie_breakers": [
                {"field": "cv_std", "direction": "minimize", "reason": "Prefer more stable challengers."},
                {"field": "generalization_gap_abs", "direction": "minimize"},
            ],
            "allow_model_switch": True,
            "allow_ensemble": False,
            "allow_hpo": True,
            "allow_feature_engineering": True,
            "allow_calibration": False,
        },
    }

    diagnostics = validate_contract_minimal_readonly(contract)

    issues = diagnostics.get("issues") or []
    assert not any(
        issue.get("rule") == "contract.optimization_policy_value"
        and str(issue.get("severity") or "").lower() in {"error", "fail"}
        for issue in issues
        if isinstance(issue, dict)
    )
