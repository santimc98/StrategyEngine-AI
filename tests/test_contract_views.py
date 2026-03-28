import json
from pathlib import Path

from src.utils.contract_views import (
    build_contract_views_projection,
    persist_views,
    trim_to_budget,
)


def _v5_contract():
    return {
        "contract_version": "5.0",
        "shared": {
            "scope": "full_pipeline",
            "strategy_title": "Test strategy",
            "business_objective": "Test objective",
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
                "segmentation_features": [],
                "model_features": ["feature_a"],
                "forbidden_features": [],
                "audit_only_features": [],
            },
            "task_semantics": {
                "problem_family": "binary_classification",
                "objective_type": "binary_classification",
                "primary_target": "target",
            },
            "active_workstreams": {
                "cleaning": True,
                "feature_engineering": False,
                "model_training": True,
            },
            "model_features": ["feature_a"],
            "column_dtype_targets": {
                "id": {"target_dtype": "int64", "nullable": False, "role": "identifiers"},
            },
            "iteration_policy": {
                "max_iterations": 6,
                "metric_improvement_max": 4,
                "runtime_fix_max": 3,
                "compliance_bootstrap_max": 2,
            },
        },
        "data_engineer": {
            "required_outputs": [
                {"intent": "cleaned_dataset", "path": "artifacts/clean/dataset_cleaned.csv", "required": True},
            ],
            "cleaning_gates": [{"name": "no_leakage", "severity": "HARD", "params": {}}],
            "runbook": {"objectives": ["Clean the data"]},
            "artifact_requirements": {
                "cleaned_dataset": {
                    "output_path": "artifacts/clean/dataset_cleaned.csv",
                    "output_manifest_path": "artifacts/clean/cleaning_manifest.json",
                },
            },
        },
        "ml_engineer": {
            "required_outputs": [
                {"intent": "submission", "path": "artifacts/ml/submission.csv", "required": True},
            ],
            "qa_gates": [{"name": "qa_shape", "severity": "HARD", "params": {}}],
            "reviewer_gates": [{"name": "rev_metric", "severity": "SOFT", "params": {}}],
            "evaluation_spec": {"primary_metric": "log_loss"},
            "validation_requirements": {"method": "stratified_cv"},
        },
        "cleaning_reviewer": {
            "focus_areas": ["leakage prevention"],
        },
        "qa_reviewer": {
            "review_subject": "ml_engineer",
            "artifacts_to_verify": ["artifacts/ml/submission.csv"],
        },
        "business_translator": {
            "reporting_policy": {"audience": "ops_team"},
        },
    }


def test_v5_views_have_correct_roles():
    views = build_contract_views_projection(_v5_contract())
    assert views["de_view"]["role"] == "data_engineer"
    assert views["ml_view"]["role"] == "ml_engineer"
    assert views["cleaning_view"]["role"] == "cleaning_reviewer"
    assert views["qa_view"]["role"] == "qa_reviewer"
    assert views["reviewer_view"]["role"] == "reviewer"
    assert views["translator_view"]["role"] == "translator"
    assert views["results_advisor_view"]["role"] == "results_advisor"


def test_v5_de_view_has_shared_plus_de_fields():
    views = build_contract_views_projection(_v5_contract())
    de = views["de_view"]
    assert de["scope"] == "full_pipeline"
    assert de["cleaning_gates"]
    assert de["artifact_requirements"]
    assert de["canonical_columns"] == ["id", "feature_a", "target"]


def test_v5_ml_view_has_shared_plus_ml_fields():
    views = build_contract_views_projection(_v5_contract())
    ml = views["ml_view"]
    assert ml["scope"] == "full_pipeline"
    assert ml["qa_gates"]
    assert ml["reviewer_gates"]
    assert ml["evaluation_spec"]["primary_metric"] == "log_loss"


def test_v5_ml_view_normalizes_task_target_from_evaluation_spec():
    contract = _v5_contract()
    contract["shared"]["task_semantics"] = {
        "problem_family": "ranking_calibration",
        "objective_type": "ranking_calibration",
        "primary_target": "Score",
        "target_columns": ["Score"],
    }
    contract["ml_engineer"]["evaluation_spec"] = {
        "primary_metric": "mean_absolute_error",
        "primary_target": "ref_score",
        "label_columns": ["ref_score"],
    }

    views = build_contract_views_projection(contract)

    assert views["ml_view"]["task_semantics"]["primary_target"] == "ref_score"
    assert views["ml_view"]["task_semantics"]["target_columns"] == ["ref_score"]
    assert views["reviewer_view"]["task_semantics"]["primary_target"] == "ref_score"
    assert views["qa_view"]["task_semantics"]["primary_target"] == "ref_score"


def test_v5_cleaning_view_merges_de_and_cleaning_reviewer():
    views = build_contract_views_projection(_v5_contract())
    cv = views["cleaning_view"]
    assert cv["cleaning_gates"]
    assert cv["focus_areas"] == ["leakage prevention"]


def test_v5_qa_view_merges_ml_and_qa_reviewer():
    views = build_contract_views_projection(_v5_contract())
    qa = views["qa_view"]
    assert qa["qa_gates"]
    assert qa["review_subject"] == "ml_engineer"


def test_v5_translator_view_has_reporting_policy():
    views = build_contract_views_projection(_v5_contract())
    tr = views["translator_view"]
    assert tr["reporting_policy"]["audience"] == "ops_team"


def test_v5_results_advisor_view_has_shared_only():
    views = build_contract_views_projection(_v5_contract())
    ra = views["results_advisor_view"]
    assert ra["scope"] == "full_pipeline"
    assert "cleaning_gates" not in ra
    assert "qa_gates" not in ra


def test_v5_flattened_contract_uses_v5_original():
    contract = _v5_contract()
    flat = dict(contract.get("shared") or {})
    flat["contract_version"] = "4.1"
    flat["_v5_original"] = contract
    views = build_contract_views_projection(flat)
    assert views["de_view"]["cleaning_gates"]
    assert views["ml_view"]["evaluation_spec"]


def test_v5_all_contract_fields_in_union():
    contract = _v5_contract()
    views = build_contract_views_projection(contract)
    all_contract_fields = set()
    for section_key in ["shared", "data_engineer", "ml_engineer", "cleaning_reviewer", "qa_reviewer", "business_translator"]:
        section = contract.get(section_key)
        if isinstance(section, dict):
            all_contract_fields.update(section.keys())
    all_view_fields = set()
    for view in views.values():
        all_view_fields.update(view.keys())
    missing = all_contract_fields - all_view_fields
    assert not missing, f"Fields missing from views: {missing}"


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
