import json
import os

from src.utils import run_status


def test_write_worker_input_persists_sandbox_config(tmp_path, monkeypatch):
    monkeypatch.setattr(run_status, "RUNS_DIR", str(tmp_path))

    run_status.write_worker_input(
        "abc12345",
        __file__,
        "Objetivo",
        sandbox_config={"provider": "gcp", "settings": {"endpoint": "https://sandbox.example.com"}},
    )

    input_path = os.path.join(str(tmp_path), "abc12345", "worker_input.json")
    with open(input_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload["business_objective"] == "Objetivo"
    assert payload["sandbox_config"] == {
        "provider": "remote",
        "settings": {"endpoint": "https://sandbox.example.com"},
    }


def test_write_worker_input_preserves_execution_backend_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(run_status, "RUNS_DIR", str(tmp_path))

    run_status.write_worker_input(
        "backend-run",
        __file__,
        "Objetivo",
        sandbox_config={
            "provider": "local",
            "settings": {
                "execution_backend": {
                    "mode": "cloudrun",
                    "cloudrun_enabled": True,
                    "job": "corp-heavy",
                    "region": "europe-southwest1",
                    "bucket": "corp-bucket",
                }
            },
        },
    )

    input_path = os.path.join(str(tmp_path), "backend-run", "worker_input.json")
    with open(input_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload["sandbox_config"]["settings"]["execution_backend"] == {
        "mode": "cloudrun",
        "cloudrun_enabled": True,
        "job": "corp-heavy",
        "region": "europe-southwest1",
        "bucket": "corp-bucket",
    }


def test_write_final_state_prefers_review_board_verdict_and_keeps_governance_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(run_status, "RUNS_DIR", str(tmp_path))

    run_status.write_final_state(
        "run-final-state",
        {
            "run_id": "run-final-state",
            "review_verdict": "APPROVED",
            "review_board_verdict": {
                "status": "REJECTED",
                "final_review_verdict": "NEEDS_IMPROVEMENT",
            },
            "run_outcome": "NO_GO",
            "overall_status_global": "error",
            "hard_failures": ["runtime_failure"],
        },
    )

    final_state_path = os.path.join(str(tmp_path), "run-final-state", "worker_final_state.json")
    with open(final_state_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload["review_verdict"] == "NEEDS_IMPROVEMENT"
    assert payload["run_outcome"] == "NO_GO"
    assert payload["overall_status_global"] == "error"
    assert payload["hard_failures"] == ["runtime_failure"]
