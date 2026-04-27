from src.utils.background_worker import _has_execution_failure


def test_has_execution_failure_detects_runtime_markers() -> None:
    assert _has_execution_failure(
        {
            "execution_output": "TIMEOUT: Script exceeded 7200s limit",
            "execution_error": False,
            "sandbox_failed": False,
        }
    ) is True


def test_has_execution_failure_ignores_clean_success_payload() -> None:
    assert _has_execution_failure(
        {
            "execution_output": "Artifacts generated successfully",
            "execution_error": False,
            "sandbox_failed": False,
            "output_contract_report": {"overall_status": "ok", "missing": []},
        }
    ) is False


def test_has_execution_failure_detects_cleaning_failure_preview() -> None:
    assert _has_execution_failure(
        {
            "execution_output": "",
            "execution_error": False,
            "sandbox_failed": False,
            "cleaned_data_preview": "Error: Cleaning Failed",
        }
    ) is True


def test_has_execution_failure_detects_output_contract_error() -> None:
    assert _has_execution_failure(
        {
            "execution_output": "Script completed but required outputs were missing",
            "execution_error": False,
            "sandbox_failed": False,
            "output_contract_report": {
                "overall_status": "error",
                "missing": ["artifacts/clean/dataset_clean_with_features.csv"],
            },
        }
    ) is True


def test_has_execution_failure_does_not_treat_threshold_only_contract_gap_as_runtime_error() -> None:
    threshold_failure = {
        "name": "qwk_minimum",
        "severity": "HARD",
        "artifact_path": "artifacts/ml/evaluation_report.json",
        "metric": "quadratic_weighted_kappa",
        "value": 0.774,
        "min_value": 0.8,
        "status": "fail",
        "passed": False,
    }

    assert _has_execution_failure(
        {
            "execution_output": "Script completed and metrics were written",
            "execution_error": False,
            "sandbox_failed": False,
            "output_contract_report": {
                "overall_status": "error",
                "missing": [],
                "artifact_requirements_report": {"status": "ok"},
                "qa_gate_results": {"failures": [threshold_failure], "warnings": []},
            },
        }
    ) is False
