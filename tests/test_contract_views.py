import json
from pathlib import Path

from src.utils.contract_views import (
    build_de_view,
    build_cleaning_view,
    build_contract_views_projection,
    build_contract_view_projection_reports,
    build_ml_view,
    build_qa_view,
    build_reviewer_view,
    build_translator_view,
    list_view_projection_report_errors,
    persist_views,
    trim_to_budget,
)


def _load_fixture(name: str):
    base = Path(__file__).parent / "fixtures" / name
    with base.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_de_view_excludes_prohibited_fields():
    contract_full = _load_fixture("contract_full_small.json")
    contract_min = _load_fixture("contract_min_small.json")
    artifact_index = _load_fixture("artifact_index_small.json")
    de_view = build_de_view(contract_full, contract_min, artifact_index)
    payload = json.dumps(de_view, ensure_ascii=True)
    assert "strategy_rationale" not in de_view
    assert "case_rules" not in de_view
    assert "weights" not in payload
    assert "optimization" not in payload
    assert de_view.get("required_columns")
    assert isinstance(de_view.get("cleaning_gates"), list)
    assert "data_engineer_runbook" in de_view


def test_de_view_preserves_declared_optional_passthrough_columns_from_clean_dataset():
    contract = {
        "canonical_columns": ["lead_id", "feature_a", "target"],
        "artifact_requirements": {
            "clean_dataset": {
                "required_columns": ["lead_id", "feature_a", "target"],
                "optional_passthrough_columns": ["raw_event_ts", "status_text"],
            }
        },
        "allowed_feature_sets": {
            "audit_only_features": ["audit_col"],
        },
        "cleaning_gates": [{"name": "parse_event_ts", "severity": "HARD"}],
        "required_outputs": [
            {"path": "artifacts/clean/dataset_cleaned.csv", "owner": "data_engineer"},
            {"path": "artifacts/clean/cleaning_manifest.json", "owner": "data_engineer"},
        ],
    }
    de_view = build_de_view(contract, contract, artifact_index=[])
    assert de_view.get("optional_passthrough_columns") == ["raw_event_ts", "status_text"]


def test_de_view_respects_explicit_empty_optional_passthrough_columns_without_falling_back():
    contract = {
        "canonical_columns": ["lead_id", "feature_a", "target"],
        "artifact_requirements": {
            "clean_dataset": {
                "required_columns": ["lead_id", "feature_a", "target"],
                "optional_passthrough_columns": [],
            }
        },
        "allowed_feature_sets": {
            "audit_only_features": ["audit_col"],
        },
        "cleaning_gates": [{"name": "gate_a", "severity": "HARD"}],
        "required_outputs": [
            {"path": "artifacts/clean/dataset_cleaned.csv", "owner": "data_engineer"},
            {"path": "artifacts/clean/cleaning_manifest.json", "owner": "data_engineer"},
        ],
    }
    de_view = build_de_view(contract, contract, artifact_index=[])
    assert de_view.get("optional_passthrough_columns") == []


def test_de_view_projects_cleaned_and_enriched_artifact_bindings_without_collapsing_them():
    contract = {
        "scope": "cleaning_only",
        "canonical_columns": ["lead_id", "feature_a", "feature_b", "target"],
        "model_features": ["feature_a", "feature_b"],
        "future_ml_handoff": {"enabled": True, "target_columns": ["target"]},
        "column_roles": {
            "pre_decision": ["feature_a", "feature_b"],
            "identifiers": ["lead_id"],
            "outcome": ["target"],
        },
        "artifact_requirements": {
            "cleaned_dataset": {
                "required_columns": ["lead_id", "feature_a", "feature_b", "target"],
                "optional_passthrough_columns": ["raw_notes"],
                "output_path": "artifacts/clean/dataset_cleaned.csv",
                "output_manifest_path": "artifacts/clean/cleaning_manifest.json",
            },
            "enriched_dataset": {
                "required_columns": ["feature_a", "feature_b", "target"],
                "output_path": "artifacts/clean/dataset_enriched.csv",
            },
        },
        "cleaning_gates": [{"name": "gate_a", "severity": "HARD"}],
        "required_outputs": [
            {"intent": "cleaned_dataset", "path": "artifacts/clean/dataset_cleaned.csv", "owner": "data_engineer"},
            {"intent": "enriched_dataset", "path": "artifacts/clean/dataset_enriched.csv", "owner": "data_engineer"},
            {"path": "artifacts/clean/cleaning_manifest.json", "owner": "data_engineer"},
        ],
    }

    de_view = build_de_view(contract, contract, artifact_index=[])

    assert de_view.get("output_path") == "artifacts/clean/dataset_cleaned.csv"
    assert de_view.get("required_columns") == ["lead_id", "feature_a", "feature_b", "target"]
    artifact_reqs = de_view.get("artifact_requirements") or {}
    assert (artifact_reqs.get("cleaned_dataset") or {}).get("optional_passthrough_columns") == ["raw_notes"]
    assert (artifact_reqs.get("enriched_dataset") or {}).get("required_columns") == ["feature_a", "feature_b", "target"]


def test_de_view_preserves_raw_allowed_feature_sets_from_contract():
    contract = {
        "canonical_columns": ["lead_id", "feature_a", "feature_b", "target"],
        "model_features": ["feature_a", "feature_b"],
        "allowed_feature_sets": ["commercial_activity", "future_modeling_subset"],
        "column_roles": {
            "pre_decision": ["lead_id", "feature_a", "feature_b"],
            "identifiers": ["lead_id"],
            "outcome": ["target"],
        },
        "artifact_requirements": {
            "clean_dataset": {
                "required_columns": ["lead_id", "feature_a", "feature_b", "target"],
                "optional_passthrough_columns": [],
            }
        },
        "cleaning_gates": [{"name": "gate_a", "severity": "HARD"}],
        "required_outputs": [
            {"path": "artifacts/clean/dataset_cleaned.csv", "owner": "data_engineer"},
            {"path": "artifacts/clean/dataset_enriched.csv", "owner": "data_engineer"},
            {"path": "artifacts/clean/cleaning_manifest.json", "owner": "data_engineer"},
        ],
    }

    de_view = build_de_view(contract, contract, artifact_index=[])

    assert de_view.get("model_features") == ["feature_a", "feature_b"]
    assert de_view.get("column_roles", {}).get("identifiers") == ["lead_id"]
    assert de_view.get("allowed_feature_sets") == ["commercial_activity", "future_modeling_subset"]


def test_projection_reports_flag_missing_contract_critical_bindings_for_de_view():
    contract = {
        "canonical_columns": ["lead_id", "feature_a", "target"],
        "model_features": ["feature_a"],
        "allowed_feature_sets": ["future_modeling_subset"],
        "column_roles": {
            "pre_decision": ["lead_id", "feature_a"],
            "identifiers": ["lead_id"],
            "outcome": ["target"],
        },
        "artifact_requirements": {
            "clean_dataset": {
                "required_columns": ["lead_id", "feature_a", "target"],
                "optional_passthrough_columns": [],
            }
        },
        "cleaning_gates": [{"name": "gate_a", "severity": "HARD"}],
        "required_outputs": [
            {"path": "artifacts/clean/dataset_cleaned.csv", "owner": "data_engineer"},
            {"path": "artifacts/clean/dataset_enriched.csv", "owner": "data_engineer"},
            {"path": "artifacts/clean/cleaning_manifest.json", "owner": "data_engineer"},
        ],
    }
    de_view = build_de_view(contract, contract, artifact_index=[])
    de_view.pop("model_features", None)

    reports = build_contract_view_projection_reports(contract, {"de_view": de_view}, contract_min=contract)

    assert "model_features" in (reports.get("de_view", {}).get("missing_bindings") or [])
    assert "de_view_binding_model_features_missing" in list_view_projection_report_errors(reports)


def test_projection_reports_flag_missing_dataset_artifact_requirements_binding_for_de_view():
    contract = {
        "canonical_columns": ["lead_id", "feature_a", "target"],
        "model_features": ["feature_a"],
        "allowed_feature_sets": ["future_modeling_subset"],
        "column_roles": {
            "pre_decision": ["lead_id", "feature_a"],
            "identifiers": ["lead_id"],
            "outcome": ["target"],
        },
        "artifact_requirements": {
            "cleaned_dataset": {
                "required_columns": ["lead_id", "feature_a", "target"],
                "output_path": "artifacts/clean/dataset_cleaned.csv",
                "output_manifest_path": "artifacts/clean/cleaning_manifest.json",
            },
            "enriched_dataset": {
                "required_columns": ["feature_a", "target"],
                "output_path": "artifacts/clean/dataset_enriched.csv",
            },
        },
        "cleaning_gates": [{"name": "gate_a", "severity": "HARD"}],
        "required_outputs": [
            {"intent": "cleaned_dataset", "path": "artifacts/clean/dataset_cleaned.csv", "owner": "data_engineer"},
            {"intent": "enriched_dataset", "path": "artifacts/clean/dataset_enriched.csv", "owner": "data_engineer"},
            {"path": "artifacts/clean/cleaning_manifest.json", "owner": "data_engineer"},
        ],
    }

    de_view = build_de_view(contract, contract, artifact_index=[])
    de_view.pop("artifact_requirements", None)

    reports = build_contract_view_projection_reports(contract, {"de_view": de_view}, contract_min=contract)

    assert "artifact_requirements" in (reports.get("de_view", {}).get("missing_bindings") or [])


def test_projection_reports_require_exact_allowed_feature_sets_payload():
    contract = {
        "canonical_columns": ["feature_a", "feature_b", "target"],
        "model_features": ["feature_a"],
        "allowed_feature_sets": [
            {"family": "commercial_activity", "intent": "allowed"},
            {"family": "administrative", "intent": "exclude"},
        ],
        "column_roles": {"pre_decision": ["feature_a", "feature_b"], "outcome": ["target"]},
        "required_outputs": [
            {"path": "artifacts/clean/dataset_cleaned.csv", "owner": "data_engineer"},
            {"path": "artifacts/clean/dataset_enriched.csv", "owner": "data_engineer"},
        ],
    }

    de_view = build_de_view(contract, contract, artifact_index=[])
    mutated_view = dict(de_view)
    mutated_view["allowed_feature_sets"] = [{"family": "commercial_activity", "intent": "allowed"}]

    reports = build_contract_view_projection_reports(contract, {"de_view": mutated_view}, contract_min=contract)

    assert "allowed_feature_sets" in (reports.get("de_view", {}).get("missing_bindings") or [])


def test_projection_reports_accept_explicit_empty_optional_passthrough_columns():
    contract = {
        "canonical_columns": ["lead_id", "feature_a", "target"],
        "artifact_requirements": {
            "clean_dataset": {
                "required_columns": ["lead_id", "feature_a", "target"],
                "optional_passthrough_columns": [],
            }
        },
        "cleaning_gates": [{"name": "gate_a", "severity": "HARD"}],
        "required_outputs": [
            {"path": "artifacts/clean/dataset_cleaned.csv", "owner": "data_engineer"},
            {"path": "artifacts/clean/cleaning_manifest.json", "owner": "data_engineer"},
        ],
    }

    de_view = build_de_view(contract, contract, artifact_index=[])
    reports = build_contract_view_projection_reports(contract, {"de_view": de_view}, contract_min=contract)

    assert "optional_passthrough_columns" not in (reports.get("de_view", {}).get("missing_bindings") or [])


def test_ml_view_includes_required_fields():
    contract_full = _load_fixture("contract_full_small.json")
    contract_min = _load_fixture("contract_min_small.json")
    artifact_index = _load_fixture("artifact_index_small.json")
    ml_view = build_ml_view(contract_full, contract_min, artifact_index)
    assert ml_view.get("required_outputs")
    assert isinstance(ml_view.get("column_roles"), dict)
    assert "allowed_feature_sets" in ml_view
    assert ml_view.get("model_features") == ["col_a"]
    assert ml_view.get("objective_type")
    decisioning = ml_view.get("decisioning_requirements", {})
    assert isinstance(decisioning, dict)
    assert decisioning.get("enabled") is False


def test_projection_ml_view_includes_contract_context_blocks():
    contract = {
        "scope": "full_pipeline",
        "canonical_columns": ["feature_a", "target"],
        "column_roles": {
            "pre_decision": ["feature_a"],
            "outcome": ["target"],
        },
        "required_outputs": ["data/metrics.json", "data/scored_rows.csv"],
        "qa_gates": [{"name": "benchmark_kpi_report", "severity": "HARD"}],
        "reviewer_gates": [{"name": "runtime_success", "severity": "HARD"}],
        "evaluation_spec": {"objective_type": "predictive", "requires_target": True},
        "objective_analysis": {"problem_type": "predictive"},
        "ml_engineer_runbook": {"steps": ["train", "validate"]},
        "validation_requirements": {"primary_metric": "accuracy"},
        "artifact_requirements": {"clean_dataset": {"output_path": "data/cleaned_data.csv", "manifest_path": "data/cleaning_manifest.json"}},
        "task_semantics": {
            "problem_family": "classification",
            "objective_type": "classification",
            "primary_target": "target",
            "target_columns": ["target"],
            "multi_target": False,
        },
    }
    projected = build_contract_views_projection(contract, artifact_index=[])
    ml_view = projected.get("ml_view") or {}

    assert isinstance(ml_view.get("evaluation_spec"), dict)
    assert isinstance(ml_view.get("objective_analysis"), dict)
    assert isinstance(ml_view.get("qa_gates"), list)
    assert isinstance(ml_view.get("reviewer_gates"), list)
    assert isinstance(ml_view.get("ml_engineer_runbook"), dict)
    assert (ml_view.get("task_semantics") or {}).get("primary_target") == "target"


def test_ml_view_exposes_primary_metric_and_metric_rule_from_contract():
    contract = {
        "scope": "full_pipeline",
        "canonical_columns": ["feature_a", "label_12h", "label_24h"],
        "column_roles": {
            "pre_decision": ["feature_a"],
            "outcome": ["label_12h", "label_24h"],
        },
        "required_outputs": ["artifacts/ml/cv_metrics.json", "artifacts/submission/submission.csv"],
        "evaluation_spec": {
            "objective_type": "predictive",
            "primary_metric": "mean_multi_horizon_log_loss",
            "metric_definition_rule": "Use a simple arithmetic mean unless the contract explicitly provides weights.",
        },
        "validation_requirements": {
            "primary_metric": "mean_multi_horizon_log_loss",
            "metric_definition_rule": "Use a simple arithmetic mean unless the contract explicitly provides weights.",
        },
        "task_semantics": {
            "problem_family": "multi_output_classification",
            "objective_type": "multi_output_classification",
            "primary_target": "label_12h",
            "target_columns": ["label_12h", "label_24h"],
            "multi_target": True,
        },
    }

    projected = build_contract_views_projection(contract, artifact_index=[])
    ml_view = projected.get("ml_view") or {}

    assert ml_view.get("primary_metric") == "mean_multi_horizon_log_loss"
    assert ml_view.get("metric_definition_rule") == (
        "Use a simple arithmetic mean unless the contract explicitly provides weights."
    )


def test_projection_uses_task_semantics_objective_when_objective_analysis_is_unknown():
    contract = {
        "scope": "full_pipeline",
        "canonical_columns": ["feature_a", "label_12h", "label_24h"],
        "column_roles": {
            "pre_decision": ["feature_a"],
            "outcome": ["label_12h", "label_24h"],
        },
        "objective_analysis": {"problem_type": "unknown"},
        "evaluation_spec": {"objective_type": "unknown"},
        "required_outputs": ["data/submission.csv"],
        "task_semantics": {
            "problem_family": "multi_output_classification",
            "objective_type": "multi_output_classification",
            "primary_target": "label_12h",
            "target_columns": ["label_12h", "label_24h"],
            "multi_target": True,
        },
    }

    projected = build_contract_views_projection(contract, artifact_index=[])

    assert (projected.get("ml_view") or {}).get("objective_type") == "multi_output_classification"
    assert (projected.get("reviewer_view") or {}).get("objective_type") == "multi_output_classification"


def test_projection_de_view_includes_gates_and_runbook():
    contract = {
        "scope": "cleaning_only",
        "active_workstreams": {
            "cleaning": True,
            "feature_engineering": True,
            "model_training": False,
        },
        "future_ml_handoff": {
            "enabled": True,
            "primary_target": "target_future",
        },
        "cleaning_gates": [{"name": "required_columns_present", "severity": "HARD", "params": {}}],
        "data_engineer_runbook": {"steps": ["load", "clean", "persist"]},
        "artifact_requirements": {
            "clean_dataset": {
                "required_columns": ["id", "feature_a"],
                "output_path": "data/cleaned_data.csv",
                "manifest_path": "data/cleaning_manifest.json",
            }
        },
    }
    projected = build_contract_views_projection(contract, artifact_index=[])
    de_view = projected.get("de_view") or {}

    assert isinstance(de_view.get("cleaning_gates"), list)
    assert de_view.get("cleaning_gates")
    assert isinstance(de_view.get("data_engineer_runbook"), dict)
    assert de_view.get("data_engineer_runbook")
    assert (de_view.get("active_workstreams") or {}).get("model_training") is False
    assert (de_view.get("future_ml_handoff") or {}).get("primary_target") == "target_future"


def test_projection_prefers_explicit_agent_interfaces_over_legacy_inference():
    contract = {
        "scope": "cleaning_only",
        "strategy_title": "CRM prep",
        "business_objective": "Prepare a clean CRM dataset for future modeling.",
        "canonical_columns": ["lead_id", "feature_a", "target"],
        "column_roles": {
            "pre_decision": ["feature_a"],
            "decision": [],
            "outcome": ["target"],
            "post_decision_audit_only": [],
            "unknown": [],
            "identifiers": ["lead_id"],
            "time_columns": [],
        },
        "allowed_feature_sets": {
            "model_features": ["feature_a"],
            "segmentation_features": ["feature_a"],
            "forbidden_features": [],
            "audit_only_features": [],
        },
        "required_outputs": ["artifacts/clean/default_clean.csv"],
        "artifact_requirements": {
            "clean_dataset": {
                "required_columns": ["lead_id", "feature_a", "target"],
                "output_path": "artifacts/clean/default_clean.csv",
                "output_manifest_path": "artifacts/clean/default_manifest.json",
            }
        },
        "cleaning_gates": [{"name": "legacy_gate", "severity": "HARD", "params": {}}],
        "agent_interfaces": {
            "data_engineer": {
                "required_columns": ["lead_id", "feature_a"],
                "output_path": "artifacts/clean/interface_clean.csv",
                "output_manifest_path": "artifacts/clean/interface_manifest.json",
                "cleaning_gates": [{"name": "interface_gate", "severity": "SOFT", "params": {}}],
            },
            "translator": {
                "key_decisions": ["interface:key_decision"],
                "constraints": {"cite_sources": True, "no_markdown_tables": True, "tone": "executive"},
            },
        },
    }

    projected = build_contract_views_projection(contract, artifact_index=[])
    de_view = projected.get("de_view") or {}
    translator_view = projected.get("translator_view") or {}

    assert de_view.get("output_path") == "artifacts/clean/default_clean.csv"
    assert de_view.get("output_manifest_path") == "artifacts/clean/default_manifest.json"
    assert de_view.get("required_columns") == ["lead_id", "feature_a", "target"]
    cleaning_gate_names = {gate.get("name") for gate in (de_view.get("cleaning_gates") or []) if isinstance(gate, dict)}
    assert "legacy_gate" in cleaning_gate_names
    assert "interface:key_decision" in (translator_view.get("key_decisions") or [])
    assert (translator_view.get("constraints") or {}).get("tone") == "executive"


def test_build_ml_view_prefers_explicit_ml_engineer_interface():
    contract_full = {
        "scope": "full_pipeline",
        "canonical_columns": ["feature_a", "target"],
        "column_roles": {
            "pre_decision": ["feature_a"],
            "decision": [],
            "outcome": ["target"],
            "post_decision_audit_only": [],
            "unknown": [],
            "identifiers": [],
            "time_columns": [],
        },
        "allowed_feature_sets": {
            "model_features": ["feature_a"],
            "segmentation_features": ["feature_a"],
            "forbidden_features": [],
            "audit_only_features": [],
        },
        "required_outputs": ["artifacts/ml/default_metrics.json"],
        "validation_requirements": {"primary_metric": "log_loss"},
        "agent_interfaces": {
            "ml_engineer": {
                "required_outputs": ["artifacts/ml/interface_metrics.json"],
                "primary_metric": "mean_multi_horizon_log_loss",
                "metric_definition_rule": "Use the explicit interface metric.",
            }
        },
    }

    ml_view = build_ml_view(contract_full, {}, [])

    assert ml_view.get("required_outputs") == ["artifacts/ml/default_metrics.json"]
    assert ml_view.get("primary_metric") == "log_loss"
    assert ml_view.get("metric_definition_rule") == "Use the explicit interface metric."


def test_projection_preserves_top_level_de_outputs_when_interface_is_partial():
    contract = {
        "scope": "cleaning_only",
        "required_outputs": [
            {"path": "artifacts/clean/dataset_cleaned.csv", "intent": "cleaned_dataset", "owner": "data_engineer"},
            {"path": "artifacts/clean/cleaning_manifest.json", "intent": "cleaning_manifest", "owner": "data_engineer"},
            {"path": "artifacts/clean/dataset_enriched.csv", "intent": "enriched_dataset", "owner": "data_engineer"},
        ],
        "artifact_requirements": {
            "clean_dataset": {
                "required_columns": ["lead_id", "feature_a", "target"],
                "output_path": "artifacts/clean/dataset_cleaned.csv",
                "output_manifest_path": "artifacts/clean/cleaning_manifest.json",
            }
        },
        "cleaning_gates": [
            {"name": "required_columns_present", "severity": "HARD", "params": {}},
            {"name": "standardize_dates", "severity": "HARD", "params": {}},
        ],
        "data_engineer_runbook": {"steps": ["load", "clean", "persist"]},
        "agent_interfaces": {
            "data_engineer": {
                "required_outputs": ["artifacts/clean/dataset_cleaned.csv"],
                "cleaning_gates": [{"name": "required_columns_present", "severity": "HARD", "params": {}}],
            }
        },
    }

    projected = build_contract_views_projection(contract, artifact_index=[])
    de_view = projected.get("de_view") or {}
    cleaning_view = projected.get("cleaning_view") or {}

    assert set(de_view.get("required_outputs") or []) == {
        "artifacts/clean/dataset_cleaned.csv",
        "artifacts/clean/cleaning_manifest.json",
        "artifacts/clean/dataset_enriched.csv",
    }
    assert set(cleaning_view.get("required_outputs") or []) == {
        "artifacts/clean/dataset_cleaned.csv",
        "artifacts/clean/cleaning_manifest.json",
        "artifacts/clean/dataset_enriched.csv",
    }
    assert {gate.get("name") for gate in (de_view.get("cleaning_gates") or []) if isinstance(gate, dict)} == {
        "required_columns_present",
        "standardize_dates",
    }


def test_projection_ml_view_preserves_runbook_list_shape():
    contract = {
        "scope": "full_pipeline",
        "canonical_columns": ["feature_a", "target"],
        "column_roles": {
            "pre_decision": ["feature_a"],
            "outcome": ["target"],
        },
        "required_outputs": ["data/metrics.json"],
        "ml_engineer_runbook": [
            {"step": "train_baseline"},
            {"step": "evaluate_cv"},
        ],
    }
    projected = build_contract_views_projection(contract, artifact_index=[])
    ml_view = projected.get("ml_view") or {}
    runbook = ml_view.get("ml_engineer_runbook")
    assert isinstance(runbook, list)
    assert runbook and runbook[0].get("step") == "train_baseline"


def test_projection_ml_view_preserves_runbook_string_shape():
    contract = {
        "scope": "full_pipeline",
        "canonical_columns": ["feature_a", "target"],
        "column_roles": {
            "pre_decision": ["feature_a"],
            "outcome": ["target"],
        },
        "required_outputs": ["data/metrics.json"],
        "ml_engineer_runbook": "Parse target_json, extract survival labels, and train a discrete-time hazard baseline.",
    }
    projected = build_contract_views_projection(contract, artifact_index=[])
    ml_view = projected.get("ml_view") or {}
    runbook = ml_view.get("ml_engineer_runbook")
    assert isinstance(runbook, str)
    assert "discrete-time hazard" in runbook


def test_projection_propagates_outlier_policy_to_relevant_views():
    contract = {
        "scope": "full_pipeline",
        "strategy_title": "Robust modeling with outlier handling",
        "business_objective": "Predict target robustly",
        "canonical_columns": ["feature_a", "target"],
        "column_roles": {
            "pre_decision": ["feature_a"],
            "outcome": ["target"],
        },
        "cleaning_gates": [{"name": "required_columns_present", "severity": "HARD", "params": {}}],
        "qa_gates": [{"name": "benchmark_kpi_report", "severity": "HARD", "params": {}}],
        "reviewer_gates": [{"name": "runtime_success", "severity": "HARD", "params": {}}],
        "validation_requirements": {"primary_metric": "rmse"},
        "artifact_requirements": {
            "clean_dataset": {
                "required_columns": ["feature_a", "target"],
                "output_path": "data/cleaned_data.csv",
                "output_manifest_path": "data/cleaning_manifest.json",
            }
        },
        "required_outputs": ["data/metrics.json", "data/scored_rows.csv"],
        "outlier_policy": {
            "enabled": True,
            "apply_stage": "data_engineer",
            "target_columns": ["feature_a"],
            "report_path": "data/outlier_treatment_report.json",
            "strict": True,
        },
    }

    projected = build_contract_views_projection(contract, artifact_index=[])
    de_view = projected.get("de_view") or {}
    cleaning_view = projected.get("cleaning_view") or {}
    ml_view = projected.get("ml_view") or {}

    assert isinstance(de_view.get("outlier_policy"), dict)
    assert de_view.get("outlier_report_path") == "data/outlier_treatment_report.json"
    assert isinstance(cleaning_view.get("outlier_policy"), dict)
    assert cleaning_view.get("outlier_report_path") == "data/outlier_treatment_report.json"
    assert isinstance(ml_view.get("outlier_policy"), dict)


def test_projection_builds_column_resolution_context_from_profile_evidence():
    contract = {
        "scope": "cleaning_only",
        "artifact_requirements": {
            "clean_dataset": {
                "required_columns": ["created_at", "annual_revenue"],
                "output_path": "data/cleaned_data.csv",
                "output_manifest_path": "data/cleaning_manifest.json",
            }
        },
        "cleaning_gates": [
            {
                "name": "robust_date_parsing_with_invalid_flagging",
                "severity": "HARD",
                "action_type": "parse",
                "params": {"target_columns": ["created_at"]},
            },
            {
                "name": "normalize_numeric_ranges_and_amounts",
                "severity": "HARD",
                "action_type": "standardize",
                "params": {"target_columns": ["annual_revenue"]},
            },
        ],
        "column_dtype_targets": {
            "created_at": {"target_dtype": "datetime"},
            "annual_revenue": {"target_dtype": "float64"},
        },
    }
    data_profile = {
        "basic_stats": {"columns": ["created_at", "annual_revenue"]},
        "dtypes": {"created_at": "object", "annual_revenue": "object"},
        "missingness": {"created_at": 0.12, "annual_revenue": 0.21},
        "cardinality": {
            "created_at": {
                "top_values": [
                    {"value": "2025-07-08", "count": 5},
                    {"value": "27/06/2025", "count": 3},
                    {"value": "not_a_date", "count": 1},
                ]
            },
            "annual_revenue": {
                "top_values": [
                    {"value": "$350k", "count": 2},
                    {"value": "0.1M", "count": 1},
                    {"value": "unknown", "count": 1},
                ]
            },
        },
    }

    projected = build_contract_views_projection(contract, artifact_index=[], data_profile=data_profile)
    de_view = projected.get("de_view") or {}
    cleaning_view = projected.get("cleaning_view") or {}
    context = de_view.get("column_resolution_context") or {}

    assert de_view.get("column_resolution_context_path") == "data/column_resolution_context.json"
    assert cleaning_view.get("column_resolution_context_path") == "data/column_resolution_context.json"
    assert set(context) >= {"created_at", "annual_revenue"}
    assert context["created_at"]["semantic_kind"] == "datetime_like"
    assert "iso_date" in (context["created_at"].get("observed_format_families") or [])
    assert "slash_date" in (context["created_at"].get("observed_format_families") or [])
    assert context["annual_revenue"]["semantic_kind"] == "amount_like"
    assert "currency_symbol" in (context["annual_revenue"].get("observed_format_families") or [])
    assert "magnitude_suffix" in (context["annual_revenue"].get("observed_format_families") or [])
    assert context["annual_revenue"]["preservation_expectation"] == "retain_in_output"
    assert (cleaning_view.get("column_resolution_context") or {}) == context


def test_ml_view_includes_scored_rows_schema():
    contract_min = {
        "canonical_columns": ["id", "feature_a"],
        "column_roles": {"pre_decision": ["id", "feature_a"]},
        "allowed_feature_sets": {
            "model_features": ["feature_a"],
            "segmentation_features": ["feature_a"],
            "forbidden_features": [],
        },
        "artifact_requirements": {
            "required_files": [{"path": "data/scored_rows.csv"}],
            "scored_rows_schema": {
                "required_columns": ["id", "prediction"],
                "required_any_of_groups": [["prediction", "probability"]],
                "required_any_of_group_severity": ["fail"],
            },
        },
        "required_outputs": ["data/scored_rows.csv"],
    }
    ml_view = build_ml_view({}, contract_min, [])
    artifact_requirements = ml_view.get("artifact_requirements") or {}
    scored_schema = artifact_requirements.get("scored_rows_schema") or {}
    assert scored_schema.get("required_columns") == ["id", "prediction"]
    assert scored_schema.get("required_any_of_groups")


def test_ml_view_includes_plot_spec_from_visual_requirements():
    contract_full = {
        "canonical_columns": ["col_a"],
        "column_roles": {"pre_decision": ["col_a"]},
        "allowed_feature_sets": {
            "model_features": ["col_a"],
            "segmentation_features": ["col_a"],
            "forbidden_features": [],
        },
        "artifact_requirements": {
            "visual_requirements": {
                "enabled": True,
                "required": True,
                "outputs_dir": "static/plots",
                "items": [],
                "plot_spec": {
                    "enabled": True,
                    "max_plots": 1,
                    "plots": [{"id": "plot_1", "caption_template": "Example"}],
                },
            }
        },
    }
    contract_min = {
        "canonical_columns": ["col_a"],
        "column_roles": {"pre_decision": ["col_a"]},
        "allowed_feature_sets": {
            "model_features": ["col_a"],
            "segmentation_features": ["col_a"],
            "forbidden_features": [],
        },
    }
    ml_view = build_ml_view(contract_full, contract_min, [])
    plot_spec = ml_view.get("plot_spec") or {}
    assert plot_spec.get("plots")


def test_ml_view_derives_visual_items_from_plot_spec_when_missing():
    contract_full = {
        "canonical_columns": ["col_a"],
        "column_roles": {"pre_decision": ["col_a"]},
        "allowed_feature_sets": {
            "model_features": ["col_a"],
            "segmentation_features": ["col_a"],
            "forbidden_features": [],
        },
        "artifact_requirements": {
            "visual_requirements": {
                "enabled": True,
                "required": True,
                "outputs_dir": "static/plots",
                "items": [],
                "plot_spec": {
                    "enabled": True,
                    "plots": [{"plot_id": "confidence_distribution"}],
                },
            }
        },
    }
    contract_min = {}

    ml_view = build_ml_view(contract_full, contract_min, [])
    visual = ml_view.get("visual_requirements") or {}
    items = visual.get("items") or []
    assert items
    assert items[0].get("id") == "confidence_distribution"
    assert items[0].get("path") == "static/plots/confidence_distribution.png"


def test_ml_view_prefers_visual_requirements_plot_spec_over_reporting_policy():
    contract_full = {
        "canonical_columns": ["col_a"],
        "column_roles": {"pre_decision": ["col_a"]},
        "allowed_feature_sets": {
            "model_features": ["col_a"],
            "segmentation_features": ["col_a"],
            "forbidden_features": [],
        },
        "reporting_policy": {
            "plot_spec": {
                "enabled": True,
                "plots": [{"plot_id": "policy_plot"}],
            }
        },
    }
    contract_min = {
        "canonical_columns": ["col_a"],
        "column_roles": {"pre_decision": ["col_a"]},
        "allowed_feature_sets": {
            "model_features": ["col_a"],
            "segmentation_features": ["col_a"],
            "forbidden_features": [],
        },
        "artifact_requirements": {
            "visual_requirements": {
                "enabled": True,
                "required": True,
                "outputs_dir": "static/plots",
                "items": [],
                "plot_spec": {
                    "enabled": True,
                    "plots": [{"plot_id": "canonical_plot"}],
                },
            }
        },
    }

    ml_view = build_ml_view(contract_full, contract_min, [])
    plot_spec = ml_view.get("plot_spec") or {}
    plots = plot_spec.get("plots") or []
    assert plots
    assert plots[0].get("plot_id") == "canonical_plot"
    assert "view_warnings" not in ml_view


def test_ml_view_required_outputs_merge_contract_min_and_full():
    contract_full = {
        "artifact_requirements": {
            "required_files": [{"path": "static/plots/confidence_distribution.png"}],
        }
    }
    contract_min = {
        "canonical_columns": ["col_a"],
        "column_roles": {"pre_decision": ["col_a"]},
        "allowed_feature_sets": {
            "model_features": ["col_a"],
            "segmentation_features": ["col_a"],
            "forbidden_features": [],
        },
        "required_outputs": ["data/metrics.json"],
    }

    ml_view = build_ml_view(contract_full, contract_min, [])
    outputs = ml_view.get("required_outputs") or []
    assert "data/metrics.json" in outputs
    assert "static/plots/confidence_distribution.png" in outputs


def test_ml_and_reviewer_views_extract_paths_from_required_outputs_dict_entries():
    contract_min = {
        "canonical_columns": ["feature_a", "target"],
        "column_roles": {
            "pre_decision": ["feature_a"],
            "outcome": ["target"],
        },
        "allowed_feature_sets": {
            "model_features": ["feature_a"],
            "segmentation_features": [],
            "forbidden_features": [],
        },
        "required_outputs": [
            {"path": "data/metrics.json", "required": True},
            {"path": "data/scored_rows.csv", "required": True},
        ],
        "qa_gates": [{"name": "benchmark_metric", "severity": "HARD"}],
        "reviewer_gates": [{"name": "runtime_success", "severity": "HARD"}],
    }

    ml_view = build_ml_view({}, contract_min, [])
    reviewer_view = build_reviewer_view({}, contract_min, [])

    for view in (ml_view, reviewer_view):
        outputs = view.get("required_outputs") or []
        assert all(isinstance(path, str) for path in outputs)
        assert "data/metrics.json" in outputs
        assert "data/scored_rows.csv" in outputs
        assert not any("{'path':" in path for path in outputs)


def test_projection_views_keep_required_outputs_as_paths_with_rich_metadata_present():
    contract = {
        "scope": "full_pipeline",
        "canonical_columns": ["feature_a", "target"],
        "column_roles": {
            "pre_decision": ["feature_a"],
            "outcome": ["target"],
        },
        "required_outputs": ["data/metrics.json", "data/scored_rows.csv"],
        "required_output_artifacts": [
            {"path": "data/metrics.json", "required": True, "owner": "ml_engineer", "kind": "metrics"},
            {"path": "data/scored_rows.csv", "required": True, "owner": "ml_engineer", "kind": "predictions"},
        ],
        "spec_extraction": {
            "deliverables": [
                {"path": "data/metrics.json", "required": True, "owner": "ml_engineer", "kind": "metrics"},
                {"path": "data/scored_rows.csv", "required": True, "owner": "ml_engineer", "kind": "predictions"},
            ]
        },
        "qa_gates": [{"name": "benchmark_metric", "severity": "HARD"}],
        "reviewer_gates": [{"name": "runtime_success", "severity": "HARD"}],
        "validation_requirements": {"primary_metric": "accuracy"},
        "ml_engineer_runbook": {"steps": ["train", "validate"]},
        "artifact_requirements": {
            "clean_dataset": {
                "output_path": "data/cleaned_data.csv",
                "manifest_path": "data/cleaning_manifest.json",
            }
        },
    }

    projected = build_contract_views_projection(contract, artifact_index=[])
    for view_key in ("ml_view", "reviewer_view"):
        outputs = (projected.get(view_key) or {}).get("required_outputs") or []
        assert all(isinstance(path, str) for path in outputs)
        assert "data/metrics.json" in outputs
        assert "data/scored_rows.csv" in outputs
        assert not any("{'path':" in path for path in outputs)


def test_translator_view_includes_decisioning_requirements():
    contract_full = {
        "strategy_title": "Decision Strategy",
        "business_objective": "Prioritize top cases",
        "canonical_columns": ["id", "feature_a"],
        "decisioning_requirements": {
            "enabled": True,
            "required": True,
            "output": {
                "file": "data/scored_rows.csv",
                "required_columns": [
                    {"name": "priority_score", "role": "score", "type": "numeric"},
                ],
            },
            "policy_notes": "Rank cases by priority_score.",
        },
    }
    contract_min = {
        "strategy_title": "Decision Strategy",
        "business_objective": "Prioritize top cases",
        "canonical_columns": ["id", "feature_a"],
    }
    translator_view = build_translator_view(contract_full, contract_min, [])
    decisioning = translator_view.get("decisioning_requirements", {})
    assert decisioning.get("enabled") is True
    assert decisioning.get("required") is True


def test_ml_view_exposes_declared_cleaning_manifest_path():
    contract = {
        "strategy_title": "Custom Paths",
        "business_objective": "Respect declared artifact locations",
        "canonical_columns": ["entity_id", "feature_a", "label"],
        "column_roles": {"pre_decision": ["entity_id", "feature_a"], "target": ["label"], "id": ["entity_id"]},
        "allowed_feature_sets": {"model_features": ["feature_a"], "segmentation_features": [], "forbidden_for_modeling": []},
        "validation_requirements": {"primary_metric": "logloss"},
        "required_outputs": ["outputs/final/submission.csv", "artifacts/manifests/custom_clean_manifest.json"],
        "artifact_requirements": {
            "clean_dataset": {
                "output_path": "prepared/cleaned_features.csv",
                "output_manifest_path": "artifacts/manifests/custom_clean_manifest.json",
            },
            "required_files": [{"path": "outputs/final/submission.csv"}],
            "file_schemas": {"outputs/final/submission.csv": {"expected_row_count": 95}},
        },
    }

    projected = build_contract_views_projection(contract, artifact_index=[])
    ml_view = projected.get("ml_view") or {}

    assert ml_view.get("cleaned_data_path") == "prepared/cleaned_features.csv"
    assert ml_view.get("cleaning_manifest_path") == "artifacts/manifests/custom_clean_manifest.json"


def test_ml_view_preserves_explicit_allowed_feature_sets_without_role_heuristics():
    contract_min = {
        "canonical_columns": ["feature_a", "target", "audit_col", "entity_id"],
        "column_roles": {
            "pre_decision": ["feature_a", "target", "audit_col", "entity_id"]
        },
        "allowed_feature_sets": {
            "model_features": ["feature_a", "target", "audit_col", "entity_id"],
            "segmentation_features": ["feature_a", "entity_id"],
            "forbidden_features": [],
        },
    }
    contract_full = {
        "canonical_columns": ["feature_a", "target", "audit_col", "entity_id"],
        "column_roles": {
            "pre_decision": ["feature_a", "entity_id"],
            "outcome": ["target"],
            "post_decision_audit_only": ["audit_col"],
        },
    }
    ml_view = build_ml_view(contract_full, contract_min, [])
    assert ml_view.get("column_roles") == contract_min["column_roles"]
    assert ml_view.get("allowed_feature_sets") == contract_min["allowed_feature_sets"]


def test_ml_view_does_not_inject_identifier_metadata():
    contract_min = {
        "canonical_columns": ["EntityId", "feature_a"],
        "column_roles": {
            "pre_decision": ["EntityId", "feature_a"]
        },
        "allowed_feature_sets": {
            "model_features": ["EntityId", "feature_a"],
            "segmentation_features": ["EntityId", "feature_a"],
            "forbidden_features": [],
        },
    }
    ml_view = build_ml_view({}, contract_min, [])
    assert "identifier_columns" not in ml_view
    assert "identifier_overrides" not in ml_view
    assert ml_view.get("allowed_feature_sets") == contract_min["allowed_feature_sets"]


def test_ml_view_does_not_rewrite_forbidden_features_from_min():
    contract_min = {
        "canonical_columns": ["feature_a", "audit_col", "EntityId"],
        "column_roles": {
            "pre_decision": ["feature_a", "audit_col", "EntityId"],
            "post_decision_audit_only": ["audit_col"],
        },
        "allowed_feature_sets": {
            "model_features": ["feature_a", "audit_col", "EntityId"],
            "segmentation_features": ["feature_a", "EntityId"],
            "forbidden_features": ["audit_col"],
        },
    }
    ml_view = build_ml_view({}, contract_min, [])
    assert "forbidden_features" not in ml_view
    assert ml_view.get("allowed_feature_sets") == contract_min["allowed_feature_sets"]


def test_ml_view_preserves_declared_allowed_feature_sets_without_expansion():
    contract_full = {
        "canonical_columns": ["feature_a", "feature_b", "audit_col", "EntityId"],
        "allowed_feature_sets": {
            "model_features": ["feature_a", "audit_col", "EntityId"],
            "segmentation_features": ["feature_b", "EntityId"],
            "forbidden_for_modeling": ["audit_col"],
            "audit_only_features": ["audit_col"],
        },
    }
    contract_min = {
        "canonical_columns": ["feature_a", "feature_b", "audit_col", "EntityId"],
        "allowed_feature_sets": {
            "model_features": ["feature_b"],
            "segmentation_features": ["feature_a"],
            "forbidden_features": [],
        },
    }
    ml_view = build_ml_view(contract_full, contract_min, [])
    assert ml_view.get("allowed_feature_sets") == contract_min["allowed_feature_sets"]
    assert ml_view.get("model_features") == ["feature_b"]
    assert "audit_only_columns" not in ml_view
    assert "identifier_columns" not in ml_view


def test_ml_view_preserves_explicit_empty_model_features_when_declared():
    contract_min = {
        "canonical_columns": ["feature_a", "target"],
        "column_roles": {"future_modeling_features": ["feature_a"], "future_target": ["target"]},
        "allowed_feature_sets": ["future_modeling_features"],
        "model_features": [],
        "qa_gates": [{"name": "qa_gate"}],
        "reviewer_gates": [{"name": "review_gate"}],
    }

    ml_view = build_ml_view({}, contract_min, [])

    assert "model_features" in ml_view
    assert ml_view.get("model_features") == []


def test_projection_reports_accept_ml_view_when_contract_declares_model_features():
    contract = {
        "canonical_columns": ["feature_a", "target"],
        "column_roles": {"future_modeling_features": ["feature_a"], "future_target": ["target"]},
        "allowed_feature_sets": ["future_modeling_features"],
        "model_features": ["feature_a"],
        "qa_gates": [{"name": "qa_gate"}],
        "reviewer_gates": [{"name": "review_gate"}],
    }

    projected = build_contract_views_projection(contract, artifact_index=[])
    reports = build_contract_view_projection_reports(contract, {"ml_view": projected.get("ml_view") or {}}, contract_min=contract)

    assert (projected.get("ml_view") or {}).get("model_features") == ["feature_a"]
    assert (reports.get("ml_view") or {}).get("accepted") is True
    assert "model_features" not in ((reports.get("ml_view") or {}).get("missing_bindings") or [])


def test_reviewer_view_contains_gates_and_outputs():
    contract_full = _load_fixture("contract_full_small.json")
    contract_min = _load_fixture("contract_min_small.json")
    artifact_index = _load_fixture("artifact_index_small.json")
    reviewer_view = build_reviewer_view(contract_full, contract_min, artifact_index)
    assert reviewer_view.get("reviewer_gates")
    assert reviewer_view.get("required_outputs")


def test_qa_view_contains_gates_and_requirements():
    contract_full = _load_fixture("contract_full_small.json")
    contract_min = _load_fixture("contract_min_small.json")
    artifact_index = _load_fixture("artifact_index_small.json")
    qa_view = build_qa_view(contract_full, contract_min, artifact_index)
    assert qa_view.get("qa_gates")
    assert qa_view.get("artifact_requirements", {}).get("required_outputs")
    assert qa_view.get("review_subject") in {"data_engineer", "ml_engineer"}
    assert qa_view.get("subject_required_outputs")
    assert qa_view.get("qa_required_outputs") is not None
    assert qa_view.get("column_roles")
    assert qa_view.get("allowed_feature_sets")


def test_qa_view_targets_data_engineer_for_cleaning_only_contract():
    contract_full = {
        "scope": "cleaning_only",
        "active_workstreams": {"data_cleaning": True, "feature_engineering": True, "model_training": False},
        "required_outputs": [
            {"path": "artifacts/clean/dataset_cleaned.csv", "owner": "data_engineer"},
            {"path": "artifacts/clean/dataset_enriched.csv", "owner": "data_engineer"},
            {"path": "artifacts/qa/data_validation_results.json", "owner": "qa_engineer"},
        ],
        "qa_gates": [{"name": "verify_exclusions", "severity": "HARD"}],
        "column_roles": {"features": ["feature_a"]},
        "canonical_columns": ["feature_a"],
    }
    qa_view = build_qa_view(contract_full, contract_full, artifact_index=[])
    assert qa_view.get("review_subject") == "data_engineer"
    assert qa_view.get("subject_required_outputs") == [
        "artifacts/clean/dataset_cleaned.csv",
        "artifacts/clean/dataset_enriched.csv",
    ]
    assert qa_view.get("qa_required_outputs") == ["artifacts/qa/data_validation_results.json"]


def test_qa_view_targets_ml_engineer_for_training_contract():
    contract_full = {
        "scope": "ml_only",
        "active_workstreams": {"data_cleaning": False, "feature_engineering": False, "model_training": True},
        "required_outputs": [
            {"path": "artifacts/ml/model_metrics.json", "owner": "ml_engineer"},
            {"path": "artifacts/ml/scored_rows.csv", "owner": "ml_engineer"},
            {"path": "artifacts/qa/model_validation_results.json", "owner": "qa_engineer"},
        ],
        "qa_gates": [{"name": "target_mapping_check", "severity": "HARD"}],
        "column_roles": {"features": ["feature_a"], "outcome": ["target"]},
        "canonical_columns": ["feature_a", "target"],
    }
    qa_view = build_qa_view(contract_full, contract_full, artifact_index=[])
    assert qa_view.get("review_subject") == "ml_engineer"
    assert qa_view.get("subject_required_outputs") == [
        "artifacts/ml/model_metrics.json",
        "artifacts/ml/scored_rows.csv",
    ]
    assert qa_view.get("qa_required_outputs") == ["artifacts/qa/model_validation_results.json"]


def test_qa_view_carries_row_count_hints_when_available():
    contract_full = _load_fixture("contract_full_small.json")
    contract_min = _load_fixture("contract_min_small.json")
    contract_full["dataset_profile"] = {
        "n_train_rows": 100,
        "n_test_rows": 40,
    }
    qa_view = build_qa_view(contract_full, contract_min, artifact_index=[])
    assert qa_view.get("n_train_rows") == 100
    assert qa_view.get("n_test_rows") == 40
    assert qa_view.get("n_total_rows") == 140


def test_ml_and_qa_views_propagate_split_spec_and_file_schemas():
    contract_full = {
        "canonical_columns": ["id", "feature_a", "target", "is_train"],
        "column_roles": {"pre_decision": ["id", "feature_a", "is_train"], "outcome": ["target"]},
        "required_outputs": ["data/submission.csv", "data/scored_rows.csv"],
        "artifact_requirements": {
            "required_files": [{"path": "data/submission.csv"}, {"path": "data/scored_rows.csv"}],
            "file_schemas": {
                "data/submission.csv": {"expected_row_count": 270000},
                "data/scored_rows.csv": {"expected_row_count": 900000},
            },
        },
        "split_spec": {
            "status": "resolved",
            "split_column": "is_train",
            "training_rows_policy": "only_rows_with_label",
            "n_train_rows": 630000,
            "n_test_rows": 270000,
            "n_total_rows": 900000,
        },
        "data_profile": {
            "outcome_analysis": {
                "target": {"non_null_count": 630000, "total_count": 900000},
            }
        },
    }
    contract_min = {
        "canonical_columns": ["id", "feature_a", "target", "is_train"],
        "column_roles": {"pre_decision": ["id", "feature_a", "is_train"], "outcome": ["target"]},
        "allowed_feature_sets": {
            "model_features": ["feature_a"],
            "segmentation_features": ["feature_a"],
            "forbidden_features": ["target"],
        },
        "artifact_requirements": contract_full["artifact_requirements"],
        "required_outputs": ["data/submission.csv", "data/scored_rows.csv"],
    }

    ml_view = build_ml_view(contract_full, contract_min, [])
    qa_view = build_qa_view(contract_full, contract_min, [])

    ml_artifacts = ml_view.get("artifact_requirements") or {}
    assert (ml_artifacts.get("file_schemas") or {}).get("data/submission.csv", {}).get("expected_row_count") == 270000
    assert (qa_view.get("artifact_requirements") or {}).get("file_schemas", {}).get("data/scored_rows.csv", {}).get("expected_row_count") == 900000
    assert (ml_view.get("split_spec") or {}).get("status") == "resolved"
    assert (qa_view.get("split_spec") or {}).get("split_column") == "is_train"
    assert ml_view.get("n_train_rows") == 630000
    assert qa_view.get("n_test_rows") == 270000


def test_translator_view_contains_policy_and_inventory():
    contract_full = _load_fixture("contract_full_small.json")
    contract_min = _load_fixture("contract_min_small.json")
    artifact_index = _load_fixture("artifact_index_small.json")
    translator_view = build_translator_view(contract_full, contract_min, artifact_index)
    assert translator_view.get("reporting_policy")
    assert translator_view.get("evidence_inventory")


def test_trim_to_budget_preserves_required_fields():
    payload = {
        "required_columns": ["a", "b"],
        "required_outputs": ["data/metrics.json"],
        "forbidden_features": ["x"],
        "gates": ["gate_a"],
        "long_text": "x" * 5000,
        "items": list(range(200)),
    }
    trimmed = trim_to_budget(payload, max_chars=900)
    assert len(json.dumps(trimmed, ensure_ascii=True)) <= 900
    assert trimmed.get("required_columns") == ["a", "b"]
    assert trimmed.get("required_outputs") == ["data/metrics.json"]
    assert trimmed.get("forbidden_features") == ["x"]
    assert trimmed.get("gates") == ["gate_a"]


def test_trim_to_budget_preserves_nested_contractual_subtrees_without_placeholder_truncation():
    payload = {
        "column_roles": {
            "future_modeling_features": [f"feature_{idx}" for idx in range(40)],
            "administrative_exclude": ["internal_debug_flag", "legacy_import_batch"],
        },
        "allowed_feature_sets": {
            "model_features": [f"feature_{idx}" for idx in range(40)],
            "named_sets": [f"family_{idx}" for idx in range(20)],
        },
        "non_contractual_notes": [f"note_{idx}" for idx in range(200)],
    }

    trimmed = trim_to_budget(payload, max_chars=600)

    assert trimmed.get("column_roles", {}).get("future_modeling_features") == [f"feature_{idx}" for idx in range(40)]
    assert trimmed.get("allowed_feature_sets", {}).get("named_sets") == [f"family_{idx}" for idx in range(20)]
    assert "...(40 total)" not in json.dumps(trimmed, ensure_ascii=True)


def test_trim_to_budget_accepts_optional_limits():
    payload = {
        "long_text": "x" * 8000,
        "items": list(range(500)),
    }
    trimmed = trim_to_budget(
        payload,
        max_chars=700,
        max_str_len=280,
        max_list_items=9,
    )
    assert len(json.dumps(trimmed, ensure_ascii=True)) <= 700
    assert isinstance(trimmed, dict)


def test_persist_views_writes_files(tmp_path):
    views = {
        "de_view": {"role": "data_engineer"},
        "ml_view": {"role": "ml_engineer"},
        "cleaning_view": {"role": "cleaning_reviewer"},
        "qa_view": {"role": "qa_reviewer"},
        "reviewer_view": {"role": "reviewer"},
        "translator_view": {"role": "translator"},
    }
    paths = persist_views(views, base_dir=str(tmp_path))
    assert (tmp_path / "contracts" / "views" / "de_view.json").exists()
    assert (tmp_path / "contracts" / "views" / "ml_view.json").exists()
    assert (tmp_path / "contracts" / "views" / "cleaning_view.json").exists()
    assert (tmp_path / "contracts" / "views" / "qa_view.json").exists()
    assert paths.get("de_view")


def test_cleaning_view_contains_gates_and_requirements():
    contract_full = _load_fixture("contract_full_small.json")
    contract_min = _load_fixture("contract_min_small.json")
    artifact_index = _load_fixture("artifact_index_small.json")
    cleaning_view = build_cleaning_view(contract_full, contract_min, artifact_index)
    assert cleaning_view.get("required_columns")
    assert cleaning_view.get("cleaning_gates")
    assert cleaning_view.get("dialect")
