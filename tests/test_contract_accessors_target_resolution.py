from src.utils.contract_accessors import (
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
