"""
Tests for the manifest/artifact roundtrip flow during ML code execution.

NOTE: Code execution uses the Heavy Runner (Cloud Run / Local Runner)
architecture via the sandbox gateway abstraction. These tests verify
the heavy runner request construction and contract handling.
"""
import os
import json
import pytest
from unittest.mock import patch

from src.graph.graph import execute_code


def _make_heavy_result(*, ok=True, downloaded=None, error=None):
    """Build a minimal heavy_result dict that execute_code expects."""
    return {
        "status": "SUCCEEDED" if ok else "FAILED",
        "status_ok": ok,
        "job_failed_raw": not ok,
        "output_uri": "gs://test/outputs/",
        "dataset_uri": "gs://test/datasets/",
        "downloaded": downloaded or {},
        "missing_artifacts": [],
        "gcs_listing": [],
        "error": error,
        "error_raw": None,
        "status_arbitration": "ok" if ok else "error",
        "gcloud_flag": None,
    }


def test_manifest_roundtrip_upload(tmp_path, monkeypatch):
    """Verify that execute_code correctly submits the heavy runner request
    with required dependencies and outputs, and returns without error."""
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    (tmp_path / "data" / "cleaned_data.csv").write_text("a,target\n1,0\n2,1\n", encoding="utf-8")

    manifest = {"dialect": {"sep": ",", "decimal": ".", "encoding": "utf-8"}}
    (tmp_path / "data" / "cleaning_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    state = {
        "generated_code": "import pandas as pd\ndf = pd.read_csv('data/cleaned_data.csv')\nprint('ok')",
        "execution_output": "",
        "execution_attempt": 0,
        "run_id": "testrun-manifest",
        "ml_data_path": str(tmp_path / "data" / "cleaned_data.csv"),
        "execution_contract": {
            "required_outputs": ["data/cleaned_data.csv", "data/metrics.json"],
            "required_dependencies": [],
            "outcome_columns": ["target"],
            "evaluation_spec": {"target_column": "target"},
        },
    }

    downloaded = {
        "data/metrics.json": str(tmp_path / "data" / "metrics.json"),
    }
    (tmp_path / "data" / "metrics.json").write_text('{"rmse": 0.5}', encoding="utf-8")

    launched_requests = []

    def capture_launch(**kwargs):
        launched_requests.append(kwargs.get("request", {}))
        return _make_heavy_result(ok=True, downloaded=downloaded)

    with patch("src.graph.graph._get_execution_runtime_mode", return_value="cloudrun"), \
         patch("src.graph.graph._get_heavy_runner_config", return_value={"job": "j", "bucket": "b", "region": "r"}), \
         patch("src.graph.graph.launch_heavy_runner_job", side_effect=capture_launch), \
         patch("src.graph.graph.scan_code_safety", return_value=(True, [])):

        result = execute_code(state)

    # The heavy runner should have been called
    assert len(launched_requests) > 0, "Heavy runner should have been launched"
    request = launched_requests[0]

    # Required outputs should be in the request
    assert "data/metrics.json" in request.get("required_outputs", [])

    # Target column should be resolved
    assert request.get("target_col") == "target"

    # No error
    assert not result.get("error_message", "").startswith("HEAVY_RUNNER: target column missing")


def test_manifest_patching_logic(tmp_path, monkeypatch):
    """Verify that the execution contract required_outputs are included in the heavy runner request."""
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    (tmp_path / "data" / "cleaned_data.csv").write_text("a,target\n1,0\n2,1\n", encoding="utf-8")

    manifest = {"dialect": {"sep": ",", "decimal": ".", "encoding": "utf-8"}}
    (tmp_path / "data" / "cleaning_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    state = {
        "generated_code": "import pandas as pd\ndf = pd.read_csv('data/cleaned_data.csv')\nprint('ok')",
        "execution_output": "",
        "execution_attempt": 0,
        "run_id": "testrun-manifest-patch",
        "ml_data_path": str(tmp_path / "data" / "cleaned_data.csv"),
        "execution_contract": {
            "required_outputs": [
                "data/cleaned_data.csv",
                "data/metrics.json",
                "data/cleaning_manifest.json",
            ],
            "required_dependencies": [],
            "outcome_columns": ["target"],
            "evaluation_spec": {"target_column": "target"},
        },
    }

    launched_requests = []

    def capture_launch(**kwargs):
        launched_requests.append(kwargs.get("request", {}))
        return _make_heavy_result(ok=True, downloaded={})

    with patch("src.graph.graph._get_execution_runtime_mode", return_value="cloudrun"), \
         patch("src.graph.graph._get_heavy_runner_config", return_value={"job": "j", "bucket": "b", "region": "r"}), \
         patch("src.graph.graph.launch_heavy_runner_job", side_effect=capture_launch), \
         patch("src.graph.graph.scan_code_safety", return_value=(True, [])):

        execute_code(state)

    assert len(launched_requests) > 0, "Heavy runner should have been launched"
    required_in_request = launched_requests[0].get("required_outputs", [])
    assert "data/metrics.json" in required_in_request
    assert "data/cleaning_manifest.json" not in required_in_request
    assert "data/cleaned_data.csv" not in required_in_request
