from src.utils.contract_validator import validate_contract_minimal_readonly


def _base_full_pipeline_contract():
    return {
        "scope": "full_pipeline",
        "strategy_title": "Risk Scoring",
        "business_objective": "Predict risk and support operational decisions.",
        "output_dialect": {"sep": ",", "decimal": ".", "encoding": "utf-8"},
        "canonical_columns": ["id", "feature_a", "target"],
        "column_roles": {
            "pre_decision": ["feature_a"],
            "decision": [],
            "outcome": ["target"],
            "post_decision_audit_only": [],
            "unknown": [],
        },
        "artifact_requirements": {
            "clean_dataset": {
                "required_columns": ["id", "feature_a", "target"],
                "output_path": "data/cleaned_data.csv",
                "output_manifest_path": "data/cleaning_manifest.json",
            },
            "required_files": [
                {"path": "data/cleaned_data.csv"},
                {"path": "data/cleaning_manifest.json"},
                {"path": "reports/performance_metrics.json"},
                {"path": "outputs/risk_scores_and_decisions.csv"},
            ],
        },
        "required_outputs": [
            "data/cleaned_data.csv",
            "data/cleaning_manifest.json",
            "reports/performance_metrics.json",
            "outputs/risk_scores_and_decisions.csv",
        ],
        "column_dtype_targets": {
            "id": {"target_dtype": "string"},
            "feature_a": {"target_dtype": "float64"},
            "target": {"target_dtype": "float64", "nullable": True},
        },
        "cleaning_gates": ["schema_integrity"],
        "data_engineer_runbook": {"steps": ["load", "clean", "persist"]},
        "qa_gates": ["metric_threshold"],
        "reviewer_gates": ["artifact_completeness"],
        "validation_requirements": {"primary_metric": "normalized_gini", "method": "stratified_kfold"},
        "evaluation_spec": {"objective_type": "classification", "primary_metric": "normalized_gini"},
        "ml_engineer_runbook": {"steps": ["train", "evaluate", "persist"]},
        "objective_analysis": {"problem_type": "prediction"},
        "iteration_policy": {"max_iterations": 2},
    }


def test_validate_contract_minimal_readonly_rejects_missing_de_manifest_path():
    contract = _base_full_pipeline_contract()
    clean_dataset = contract["artifact_requirements"]["clean_dataset"]
    clean_dataset.pop("output_manifest_path", None)
    contract["required_outputs"] = [
        path for path in contract.get("required_outputs", []) if "manifest" not in str(path).lower()
    ]
    contract["artifact_requirements"]["required_files"] = [
        item
        for item in contract["artifact_requirements"].get("required_files", [])
        if "manifest" not in str(item.get("path", "")).lower()
    ]

    result = validate_contract_minimal_readonly(contract)

    assert result.get("accepted") is False
    rules = {str(issue.get("rule")) for issue in result.get("issues", []) if isinstance(issue, dict)}
    assert "contract.de_view_manifest_path" in rules


def test_validate_contract_minimal_readonly_rejects_unknown_ml_objective():
    contract = _base_full_pipeline_contract()
    contract.pop("objective_analysis", None)
    contract.pop("evaluation_spec", None)
    contract["required_outputs"] = [
        "data/cleaned_data.csv",
        "data/cleaning_manifest.json",
        "reports/performance_metrics.json",
        "artifacts/model_bundle.bin",
    ]
    contract["artifact_requirements"]["required_files"] = [
        {"path": path} for path in contract["required_outputs"]
    ]

    result = validate_contract_minimal_readonly(contract)

    assert result.get("accepted") is False
    rules = {str(issue.get("rule")) for issue in result.get("issues", []) if isinstance(issue, dict)}
    assert {"contract.ml_view_objective_type", "contract.evaluation_spec"} & rules


def test_validate_contract_minimal_readonly_accepts_executable_views_contract():
    contract = _base_full_pipeline_contract()

    result = validate_contract_minimal_readonly(contract)

    assert result.get("accepted") is True
    assert str(result.get("status")).lower() in {"ok", "warning"}


def test_validate_contract_minimal_readonly_accepts_required_outputs_objects():
    contract = _base_full_pipeline_contract()
    contract["required_outputs"] = [
        {"path": path, "required": True, "owner": "ml_engineer"}
        for path in contract["required_outputs"]
    ]

    result = validate_contract_minimal_readonly(contract)

    assert result.get("accepted") is True
    assert str(result.get("status")).lower() in {"ok", "warning"}


def test_validate_contract_minimal_readonly_rejects_required_outputs_object_without_path():
    contract = _base_full_pipeline_contract()
    contract["required_outputs"] = [
        {"output": "data/cleaned_data.csv"},
        {"path": "data/cleaning_manifest.json", "required": True},
        {"path": "reports/performance_metrics.json", "required": True},
        {"path": "outputs/risk_scores_and_decisions.csv", "required": True},
    ]

    result = validate_contract_minimal_readonly(contract)

    assert result.get("accepted") is False
    rules = {str(issue.get("rule")) for issue in result.get("issues", []) if isinstance(issue, dict)}
    assert "contract.required_outputs_path" in rules


def test_validate_contract_minimal_readonly_allows_missing_iteration_policy_with_warning():
    contract = _base_full_pipeline_contract()
    contract.pop("iteration_policy", None)

    result = validate_contract_minimal_readonly(contract)

    assert result.get("accepted") is True
    rules = {str(issue.get("rule")) for issue in result.get("issues", []) if isinstance(issue, dict)}
    assert "contract.iteration_policy" in rules


def test_validate_contract_minimal_readonly_allows_missing_optimization_policy_with_warning():
    contract = _base_full_pipeline_contract()
    contract.pop("optimization_policy", None)

    result = validate_contract_minimal_readonly(contract)

    assert result.get("accepted") is True
    rules = {str(issue.get("rule")) for issue in result.get("issues", []) if isinstance(issue, dict)}
    assert "contract.optimization_policy_missing" in rules


def test_validate_contract_minimal_readonly_rejects_invalid_optimization_policy_type():
    contract = _base_full_pipeline_contract()
    contract["optimization_policy"] = "enabled=true"

    result = validate_contract_minimal_readonly(contract)

    assert result.get("accepted") is False
    rules = {str(issue.get("rule")) for issue in result.get("issues", []) if isinstance(issue, dict)}
    assert "contract.optimization_policy_type" in rules


def test_validate_contract_minimal_readonly_rejects_missing_evaluation_spec_for_ml_scope():
    contract = _base_full_pipeline_contract()
    contract.pop("evaluation_spec", None)

    result = validate_contract_minimal_readonly(contract)

    assert result.get("accepted") is False
    rules = {str(issue.get("rule")) for issue in result.get("issues", []) if isinstance(issue, dict)}
    assert "contract.evaluation_spec" in rules


def test_validate_contract_minimal_readonly_accepts_iteration_policy_alias_keys():
    contract = _base_full_pipeline_contract()
    contract["iteration_policy"] = {
        "max_pipeline_iterations": 3,
        "gate_retry_limit": 2,
    }

    result = validate_contract_minimal_readonly(contract)

    assert result.get("accepted") is True
    rules = {str(issue.get("rule")) for issue in result.get("issues", []) if isinstance(issue, dict)}
    assert "contract.iteration_policy_limits" not in rules


def test_validate_contract_minimal_readonly_accepts_max_retries_alias():
    contract = _base_full_pipeline_contract()
    contract["iteration_policy"] = {
        "max_retries": 4,
    }

    result = validate_contract_minimal_readonly(contract)

    assert result.get("accepted") is True
    rules = {str(issue.get("rule")) for issue in result.get("issues", []) if isinstance(issue, dict)}
    assert "contract.iteration_policy_limits" not in rules


def test_validate_contract_minimal_readonly_accepts_alias_gate_objects():
    contract = _base_full_pipeline_contract()
    contract["qa_gates"] = [
        {"metric": "roc_auc", "severity": "HARD", "threshold": 0.8},
        {"name": "stability_check", "severity": "SOFT"},
    ]
    contract["reviewer_gates"] = [
        {"check": "artifact_completeness", "severity": "HARD"},
    ]

    result = validate_contract_minimal_readonly(contract)

    assert result.get("accepted") is True
    rules = {str(issue.get("rule")) for issue in result.get("issues", []) if isinstance(issue, dict)}
    assert "contract.qa_view_gates" not in rules
    assert "contract.reviewer_view_gates" not in rules


def test_validate_contract_minimal_readonly_rejects_unresolved_selector_hints():
    contract = _base_full_pipeline_contract()
    contract["canonical_columns"] = ["label", "__split", "pixel0", "pixel1"]
    contract["column_roles"] = {
        "pre_decision": ["pixel*"],
        "decision": [],
        "outcome": ["label"],
        "post_decision_audit_only": [],
        "unknown": [],
    }
    clean_dataset = contract["artifact_requirements"]["clean_dataset"]
    clean_dataset["required_columns"] = ["label", "__split"]
    clean_dataset.pop("required_feature_selectors", None)

    result = validate_contract_minimal_readonly(contract)

    assert result.get("accepted") is False
    rules = {str(issue.get("rule")) for issue in result.get("issues", []) if isinstance(issue, dict)}
    assert "contract.clean_dataset_selector_hints_unresolved" in rules


def test_validate_contract_minimal_readonly_rejects_selector_drop_required_overlap():
    contract = _base_full_pipeline_contract()
    contract["canonical_columns"] = ["label", "__split", "pixel0", "pixel1"]
    contract["column_roles"] = {
        "pre_decision": ["pixel0", "pixel1"],
        "decision": [],
        "outcome": ["label"],
        "post_decision_audit_only": [],
        "unknown": [],
    }
    clean_dataset = contract["artifact_requirements"]["clean_dataset"]
    clean_dataset["required_columns"] = ["label", "__split", "pixel0"]
    clean_dataset["required_feature_selectors"] = [{"type": "regex", "pattern": "^pixel\\d+$"}]
    clean_dataset["column_transformations"] = {
        "drop_policy": {"allow_selector_drops_when": ["constant"]},
    }

    result = validate_contract_minimal_readonly(contract)

    assert result.get("accepted") is False
    rules = {str(issue.get("rule")) for issue in result.get("issues", []) if isinstance(issue, dict)}
    assert "contract.clean_dataset_selector_drop_required_conflict" in rules


def test_validate_contract_minimal_readonly_rejects_selector_drop_passthrough_overlap():
    contract = _base_full_pipeline_contract()
    contract["canonical_columns"] = ["label", "__split", "pixel0", "pixel1"]
    contract["column_roles"] = {
        "pre_decision": ["pixel0", "pixel1"],
        "decision": [],
        "outcome": ["label"],
        "post_decision_audit_only": [],
        "unknown": [],
    }
    clean_dataset = contract["artifact_requirements"]["clean_dataset"]
    clean_dataset["required_columns"] = ["label", "__split"]
    clean_dataset["optional_passthrough_columns"] = ["pixel0"]
    clean_dataset["required_feature_selectors"] = [{"type": "regex", "pattern": "^pixel\\d+$"}]
    clean_dataset["column_transformations"] = {
        "drop_policy": {"allow_selector_drops_when": ["constant"]},
    }

    result = validate_contract_minimal_readonly(contract)

    assert result.get("accepted") is False
    rules = {str(issue.get("rule")) for issue in result.get("issues", []) if isinstance(issue, dict)}
    assert "contract.clean_dataset_selector_drop_passthrough_conflict" in rules


def test_validate_contract_minimal_readonly_rejects_selector_drop_hard_gate_overlap():
    contract = _base_full_pipeline_contract()
    contract["canonical_columns"] = ["label", "__split", "pixel0", "pixel1"]
    contract["column_roles"] = {
        "pre_decision": ["pixel0", "pixel1"],
        "decision": [],
        "outcome": ["label"],
        "post_decision_audit_only": [],
        "unknown": [],
    }
    clean_dataset = contract["artifact_requirements"]["clean_dataset"]
    clean_dataset["required_columns"] = ["label", "__split"]
    clean_dataset["required_feature_selectors"] = [{"type": "regex", "pattern": "^pixel\\d+$"}]
    clean_dataset["column_transformations"] = {
        "drop_policy": {"allow_selector_drops_when": ["constant"]},
    }
    contract["cleaning_gates"] = [
        {
            "name": "pixel_not_null",
            "severity": "HARD",
            "params": {"columns": ["pixel0"]},
        }
    ]

    result = validate_contract_minimal_readonly(contract)

    assert result.get("accepted") is False
    rules = {str(issue.get("rule")) for issue in result.get("issues", []) if isinstance(issue, dict)}
    assert "contract.cleaning_gate_selector_drop_conflict" in rules


def test_validate_contract_minimal_readonly_accepts_scale_selector_reference():
    contract = _base_full_pipeline_contract()
    contract["canonical_columns"] = ["id", "feature_a", "feature_b", "target"]
    contract["column_roles"] = {
        "pre_decision": ["feature_a", "feature_b"],
        "decision": [],
        "outcome": ["target"],
        "post_decision_audit_only": [],
        "unknown": [],
    }
    clean_dataset = contract["artifact_requirements"]["clean_dataset"]
    clean_dataset["required_columns"] = ["id", "feature_a", "feature_b", "target"]
    clean_dataset["required_feature_selectors"] = [
        {"type": "regex", "pattern": "^feature_[ab]$", "name": "model_features"}
    ]
    clean_dataset["column_transformations"] = {
        "scale_columns": ["regex:^feature_[ab]$"],
    }

    result = validate_contract_minimal_readonly(contract)

    assert result.get("accepted") is True
    rules = {str(issue.get("rule")) for issue in result.get("issues", []) if isinstance(issue, dict)}
    assert "contract.cleaning_transforms_scale_conflict" not in rules


def test_validate_contract_minimal_readonly_rejects_unresolved_scale_selector_reference():
    contract = _base_full_pipeline_contract()
    contract["canonical_columns"] = ["id", "feature_a", "feature_b", "target"]
    contract["column_roles"] = {
        "pre_decision": ["feature_a", "feature_b"],
        "decision": [],
        "outcome": ["target"],
        "post_decision_audit_only": [],
        "unknown": [],
    }
    clean_dataset = contract["artifact_requirements"]["clean_dataset"]
    clean_dataset["required_columns"] = ["id", "feature_a", "feature_b", "target"]
    clean_dataset["required_feature_selectors"] = [
        {"type": "regex", "pattern": "^feature_[ab]$", "name": "model_features"}
    ]
    clean_dataset["column_transformations"] = {
        "scale_columns": ["selector:unknown_family"],
    }

    result = validate_contract_minimal_readonly(contract)

    assert result.get("accepted") is False
    rules = {str(issue.get("rule")) for issue in result.get("issues", []) if isinstance(issue, dict)}
    assert "contract.cleaning_transforms_scale_conflict" in rules


def test_validate_contract_minimal_readonly_rejects_low_canonical_coverage_without_selectors():
    contract = _base_full_pipeline_contract()
    contract["canonical_columns"] = ["id", "target"]
    inventory = ["id", "target", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"]

    result = validate_contract_minimal_readonly(contract, column_inventory=inventory)

    assert result.get("accepted") is False
    rules = {str(issue.get("rule")) for issue in result.get("issues", []) if isinstance(issue, dict)}
    assert "contract.canonical_columns_coverage" in rules


def test_validate_contract_minimal_readonly_allows_low_canonical_coverage_when_selectors_cover_inventory():
    contract = _base_full_pipeline_contract()
    contract["canonical_columns"] = ["label", "__split"]
    contract["column_roles"] = {
        "pre_decision": ["pixel*"],
        "decision": [],
        "outcome": ["label"],
        "post_decision_audit_only": [],
        "unknown": [],
    }
    clean_dataset = contract["artifact_requirements"]["clean_dataset"]
    clean_dataset["required_columns"] = ["label", "__split"]
    clean_dataset["required_feature_selectors"] = [{"type": "regex", "pattern": "^pixel\\d+$"}]
    inventory = ["label", "__split", "pixel0", "pixel1", "pixel2", "pixel3", "pixel4"]

    result = validate_contract_minimal_readonly(contract, column_inventory=inventory)

    rules = {str(issue.get("rule")) for issue in result.get("issues", []) if isinstance(issue, dict)}
    assert "contract.canonical_columns_coverage" not in rules


def test_validate_contract_minimal_readonly_rejects_unexpected_outcome_columns_against_steward_targets():
    contract = _base_full_pipeline_contract()
    contract["outcome_columns"] = ["target", "feature_a"]
    contract["column_roles"]["outcome"] = ["target", "feature_a"]

    result = validate_contract_minimal_readonly(
        contract,
        steward_semantics={"primary_target": "target", "split_candidates": [], "id_candidates": ["id"]},
    )

    assert result.get("accepted") is False
    rules = {str(issue.get("rule")) for issue in result.get("issues", []) if isinstance(issue, dict)}
    assert "contract.outcome_columns_sanity" in rules


def test_validate_contract_minimal_readonly_rejects_model_features_with_only_structural_columns():
    contract = _base_full_pipeline_contract()
    contract["canonical_columns"] = ["id", "__split", "target"]
    contract["column_roles"] = {
        "pre_decision": ["id", "__split"],
        "decision": [],
        "outcome": ["target"],
        "post_decision_audit_only": [],
        "identifiers": ["id", "__split"],
    }
    contract["allowed_feature_sets"] = {"model_features": ["id", "__split"]}

    result = validate_contract_minimal_readonly(
        contract,
        steward_semantics={"primary_target": "target", "split_candidates": ["__split"], "id_candidates": ["id"]},
    )

    assert result.get("accepted") is False
    rules = {str(issue.get("rule")) for issue in result.get("issues", []) if isinstance(issue, dict)}
    assert "contract.model_features_empty" in rules


def test_validate_contract_minimal_readonly_rejects_conflicting_target_mapping_against_observed_numeric_values():
    contract = _base_full_pipeline_contract()
    contract["cleaning_gates"] = [
        {
            "name": "target_mapping_check",
            "severity": "HARD",
            "params": {"mapping": {"Presence": 1, "Absence": 0}},
        }
    ]

    result = validate_contract_minimal_readonly(
        contract,
        steward_semantics={
            "primary_target": "target",
            "target_observed_values": {"target": ["0.0", "1.0", "nan"]},
        },
    )

    assert result.get("accepted") is False
    rules = {str(issue.get("rule")) for issue in result.get("issues", []) if isinstance(issue, dict)}
    assert "contract.target_mapping_consistency" in rules


def test_validate_contract_minimal_readonly_allows_target_mapping_when_observed_labels_are_textual():
    contract = _base_full_pipeline_contract()
    contract["cleaning_gates"] = [
        {
            "name": "target_mapping_check",
            "severity": "HARD",
            "params": {"mapping": {"Presence": 1, "Absence": 0}},
        }
    ]

    result = validate_contract_minimal_readonly(
        contract,
        steward_semantics={
            "primary_target": "target",
            "target_observed_values": {"target": ["Presence", "Absence"]},
        },
    )

    rules = {str(issue.get("rule")) for issue in result.get("issues", []) if isinstance(issue, dict)}
    assert "contract.target_mapping_consistency" not in rules


def test_validate_contract_minimal_readonly_allows_multi_output_outcomes_with_anchor_primary_target():
    contract = _base_full_pipeline_contract()
    contract["business_objective"] = (
        "Predecir probabilidades de impacto a 12h, 24h, 48h y 72h "
        "para el archivo oficial de submission."
    )
    contract["canonical_columns"] = [
        "event_id",
        "__split",
        "label_12h",
        "label_24h",
        "label_48h",
        "label_72h",
        "feature_a",
        "feature_b",
    ]
    labels = ["label_12h", "label_24h", "label_48h", "label_72h"]
    contract["column_roles"] = {
        "pre_decision": ["feature_a", "feature_b"],
        "decision": [],
        "outcome": labels,
        "post_decision_audit_only": [],
        "identifiers": ["event_id"],
        "unknown": [],
    }
    contract["outcome_columns"] = labels
    contract["target_column"] = "label_24h"
    contract["evaluation_spec"] = {
        "objective_type": "multi_output_classification",
        "primary_metric": "average_roc_auc",
    }
    contract["allowed_feature_sets"] = {
        "model_features": ["feature_a", "feature_b"],
        "segmentation_features": ["feature_a", "feature_b"],
        "forbidden_features": labels,
        "audit_only_features": [],
    }
    clean_dataset = contract["artifact_requirements"]["clean_dataset"]
    clean_dataset["required_columns"] = ["event_id", "__split", *labels]
    clean_dataset["required_feature_selectors"] = [
        {
            "type": "all_numeric_except",
            "value": ["event_id", "__split", *labels],
        }
    ]
    contract["column_dtype_targets"] = {
        "event_id": {"target_dtype": "int64", "nullable": False},
        "__split": {"target_dtype": "string", "nullable": False},
        "label_12h": {"target_dtype": "float64", "nullable": True},
        "label_24h": {"target_dtype": "float64", "nullable": True},
        "label_48h": {"target_dtype": "float64", "nullable": True},
        "label_72h": {"target_dtype": "float64", "nullable": True},
    }

    result = validate_contract_minimal_readonly(
        contract,
        column_inventory=contract["canonical_columns"],
        steward_semantics={
            "primary_target": "label_24h",
            "notes": ["The analytical task requires predicting all four label horizons."],
            "split_candidates": ["__split"],
            "id_candidates": ["event_id"],
        },
    )

    rules = {str(issue.get("rule")) for issue in result.get("issues", []) if isinstance(issue, dict)}
    assert "contract.outcome_columns_sanity" not in rules
    assert "contract.clean_dataset_ml_columns_missing" not in rules


def test_validate_contract_minimal_readonly_uses_evaluation_spec_when_objective_analysis_is_unknown():
    contract = _base_full_pipeline_contract()
    contract["objective_analysis"] = {"problem_type": "unknown"}
    contract["evaluation_spec"] = {
        "objective_type": "multi_output_classification",
        "primary_metric": "average_roc_auc",
    }
    contract["canonical_columns"] = ["id", "feature_a", "target"]

    result = validate_contract_minimal_readonly(contract)

    rules = {str(issue.get("rule")) for issue in result.get("issues", []) if isinstance(issue, dict)}
    assert "contract.ml_view_objective_type" not in rules
