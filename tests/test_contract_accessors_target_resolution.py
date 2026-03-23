from src.utils.contract_accessors import (
    get_clean_dataset_output_path,
    get_dataset_artifact_binding,
    get_enriched_dataset_output_path,
    get_clean_manifest_path,
    get_declared_artifact_path,
    get_declared_file_schema,
    get_outcome_columns,
)


def test_get_outcome_columns_reads_validation_requirements_target_column():
    contract = {
        "validation_requirements": {
            "target_column": "target",
        }
    }
    assert get_outcome_columns(contract) == ["target"]


def test_get_outcome_columns_prefers_explicit_outcome_columns():
    contract = {
        "outcome_columns": ["claim_flag"],
        "validation_requirements": {
            "target_column": "target",
        },
    }
    assert get_outcome_columns(contract) == ["claim_flag"]


def test_get_outcome_columns_reads_task_semantics_targets():
    contract = {
        "task_semantics": {
            "primary_target": "label_24h",
            "target_columns": ["label_12h", "label_24h", "label_48h", "label_72h"],
        }
    }

    assert get_outcome_columns(contract) == ["label_12h", "label_24h", "label_48h", "label_72h"]


def test_declared_artifact_resolution_preserves_custom_paths():
    contract = {
        "required_outputs": ["artifacts/submission_bundle/submission.csv"],
        "spec_extraction": {
            "deliverables": [
                {
                    "path": "artifacts/submission_bundle/submission.csv",
                    "required": True,
                    "owner": "ml_engineer",
                    "kind": "submission",
                }
            ]
        },
        "artifact_requirements": {
            "clean_dataset": {
                "output_path": "prepared/clean_dataset.csv",
                "output_manifest_path": "artifacts/manifests/clean_manifest.json",
            },
            "file_schemas": {
                "artifacts/submission_bundle/submission.csv": {"expected_row_count": 95}
            },
        },
    }

    assert get_clean_manifest_path(contract) == "artifacts/manifests/clean_manifest.json"
    assert (
        get_declared_artifact_path(contract, "submission.csv", kind="submission")
        == "artifacts/submission_bundle/submission.csv"
    )
    assert get_declared_file_schema(contract, "submission.csv").get("expected_row_count") == 95


def test_declared_artifact_resolution_treats_evaluation_report_as_metrics_artifact():
    contract = {
        "required_outputs": ["reports/evaluation_report.json"],
        "artifact_requirements": {
            "required_files": [{"path": "reports/evaluation_report.json"}],
        },
    }

    assert (
        get_declared_artifact_path(contract, "metrics.json", kind="metrics")
        == "reports/evaluation_report.json"
    )


def test_declared_artifact_resolution_treats_evaluation_summary_as_metrics_artifact():
    contract = {
        "required_outputs": ["artifacts/ml/evaluation_summary.json"],
        "artifact_requirements": {
            "required_files": [{"path": "artifacts/ml/evaluation_summary.json"}],
        },
    }

    assert (
        get_declared_artifact_path(contract, "metrics.json", kind="metrics")
        == "artifacts/ml/evaluation_summary.json"
    )


def test_declared_artifact_resolution_treats_cv_metrics_as_metrics_artifact():
    contract = {
        "required_outputs": ["artifacts/ml/cv_metrics.json"],
        "artifact_requirements": {
            "required_files": [{"path": "artifacts/ml/cv_metrics.json"}],
        },
    }

    assert (
        get_declared_artifact_path(contract, "metrics.json", kind="metrics")
        == "artifacts/ml/cv_metrics.json"
    )


def test_clean_artifact_resolution_handles_clean_dataset_aliases_from_required_outputs():
    contract = {
        "required_outputs": [
            "artifacts/clean/clean_dataset.csv",
            "artifacts/clean/clean_dataset_manifest.json",
        ]
    }

    assert get_clean_dataset_output_path(contract) == "artifacts/clean/clean_dataset.csv"
    assert get_clean_manifest_path(contract) == "artifacts/clean/clean_dataset_manifest.json"


def test_dataset_artifact_binding_prefers_explicit_cleaned_and_enriched_contract_bindings():
    contract = {
        "required_outputs": [
            {"intent": "cleaned_dataset", "path": "artifacts/clean/cleaned_dataset.csv", "owner": "data_engineer"},
            {"intent": "enriched_dataset", "path": "artifacts/clean/enriched_dataset.csv", "owner": "data_engineer"},
        ],
        "artifact_requirements": {
            "cleaned_dataset": {
                "required_columns": ["id", "feature_a", "target"],
                "output_path": "artifacts/clean/cleaned_dataset.csv",
                "output_manifest_path": "artifacts/clean/cleaning_manifest.json",
            },
            "enriched_dataset": {
                "required_columns": ["feature_a", "target"],
                "output_path": "artifacts/clean/enriched_dataset.csv",
            },
        },
    }

    assert get_dataset_artifact_binding(contract, "cleaned_dataset").get("output_path") == "artifacts/clean/cleaned_dataset.csv"
    assert get_clean_dataset_output_path(contract) == "artifacts/clean/cleaned_dataset.csv"
    assert get_clean_manifest_path(contract) == "artifacts/clean/cleaning_manifest.json"
    assert get_enriched_dataset_output_path(contract) == "artifacts/clean/enriched_dataset.csv"
