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
