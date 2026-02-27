import json
from pathlib import Path

from src.utils.contract_views import (
    build_de_view,
    build_cleaning_view,
    build_contract_views_projection,
    build_ml_view,
    build_qa_view,
    build_reviewer_view,
    build_translator_view,
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


def test_ml_view_includes_required_fields():
    contract_full = _load_fixture("contract_full_small.json")
    contract_min = _load_fixture("contract_min_small.json")
    artifact_index = _load_fixture("artifact_index_small.json")
    ml_view = build_ml_view(contract_full, contract_min, artifact_index)
    assert ml_view.get("required_outputs")
    assert "forbidden_features" in ml_view
    assert isinstance(ml_view.get("forbidden_features"), list)
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
    }
    projected = build_contract_views_projection(contract, artifact_index=[])
    ml_view = projected.get("ml_view") or {}

    assert isinstance(ml_view.get("evaluation_spec"), dict)
    assert isinstance(ml_view.get("objective_analysis"), dict)
    assert isinstance(ml_view.get("qa_gates"), list)
    assert isinstance(ml_view.get("reviewer_gates"), list)
    assert isinstance(ml_view.get("ml_engineer_runbook"), dict)


def test_projection_de_view_includes_gates_and_runbook():
    contract = {
        "scope": "cleaning_only",
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
    warnings = ml_view.get("view_warnings") or {}
    assert warnings.get("plot_spec_source") == "artifact_requirements.visual_requirements.plot_spec"


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


def test_ml_view_inherits_roles_when_min_lax():
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
    forbidden = set(ml_view.get("forbidden_features") or [])
    assert "audit_col" in forbidden
    assert "target" in forbidden
    assert "audit_col" not in (ml_view.get("allowed_feature_sets", {}).get("model_features") or [])
    assert "target" not in (ml_view.get("allowed_feature_sets", {}).get("model_features") or [])


def test_ml_view_excludes_identifier_columns():
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
    assert "EntityId" in (ml_view.get("identifier_columns") or [])
    identifier_overrides = ml_view.get("identifier_overrides", {})
    assert "EntityId" in (identifier_overrides.get("candidate_allowed_by_contract") or [])
    assert "EntityId" in (ml_view.get("allowed_feature_sets", {}).get("model_features") or [])


def test_ml_view_preserves_forbidden_features_from_min():
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
    forbidden = set(ml_view.get("forbidden_features") or [])
    assert forbidden == {"audit_col"}
    assert "feature_a" in (ml_view.get("allowed_feature_sets", {}).get("model_features") or [])
    assert "audit_col" not in (ml_view.get("allowed_feature_sets", {}).get("model_features") or [])
    identifier_overrides = ml_view.get("identifier_overrides", {})
    assert "EntityId" in (identifier_overrides.get("candidate_allowed_by_contract") or [])


def test_ml_view_prefers_full_allowed_feature_sets():
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
    allowed = ml_view.get("allowed_feature_sets") or {}
    assert allowed.get("model_features") == ["feature_a", "EntityId"]
    assert allowed.get("segmentation_features") == ["feature_b", "EntityId"]
    assert ml_view.get("audit_only_columns") == ["audit_col"]
    assert "EntityId" in (ml_view.get("identifier_columns") or [])
    identifier_overrides = ml_view.get("identifier_overrides", {})
    assert "EntityId" in (identifier_overrides.get("candidate_allowed_by_contract") or [])


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
    assert qa_view.get("column_roles")
    assert qa_view.get("allowed_feature_sets")


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
