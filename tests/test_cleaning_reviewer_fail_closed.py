import json
import os
from unittest.mock import patch

import pytest

from src.graph.graph import run_data_engineer


@pytest.fixture
def tmp_workdir(tmp_path, monkeypatch):
    old_cwd = os.getcwd()
    monkeypatch.chdir(tmp_path)
    yield tmp_path
    os.chdir(old_cwd)


class _ExplodingReviewer:
    def review_cleaning(self, *args, **kwargs):
        raise RuntimeError("boom")


def _mock_de_heavy_success(*args, **kwargs):
    os.makedirs("data", exist_ok=True)
    with open("data/cleaned_data.csv", "wb") as f:
        f.write(b"a,b,target\n1,2,3\n4,5,9\n")
    with open("data/cleaning_manifest.json", "wb") as f:
        f.write(
            json.dumps(
                {"output_dialect": {"sep": ",", "decimal": ".", "encoding": "utf-8"}}
            ).encode("utf-8")
        )
    return {
        "ok": True,
        "downloaded": {
            "data/cleaned_data.csv": "data/cleaned_data.csv",
            "data/cleaning_manifest.json": "data/cleaning_manifest.json",
        },
    }


def test_cleaning_reviewer_failure_fail_closed(tmp_workdir, monkeypatch):
    raw_path = tmp_workdir / "raw.csv"
    raw_path.write_text("col1,col2,target\n1,2,3\n4,5,9\n", encoding="utf-8")

    monkeypatch.setattr("src.graph.graph.cleaning_reviewer", _ExplodingReviewer())

    with patch("src.graph.graph._get_heavy_runner_config", return_value={"job": "j", "bucket": "b", "region": "r"}), \
         patch("src.graph.graph._execute_data_engineer_via_heavy_runner", side_effect=_mock_de_heavy_success), \
         patch("src.graph.graph.scan_code_safety", return_value=(True, [])), \
         patch("src.graph.graph.data_engineer.generate_cleaning_plan", return_value={"plan_source": "test"}), \
         patch("src.graph.graph.data_engineer.generate_cleaning_script", return_value="print('clean')"), \
         patch("src.graph.graph.data_engineer_preflight", return_value=[]), \
         patch.dict(os.environ, {"DEEPSEEK_API_KEY": "dummy", "GOOGLE_API_KEY": "dummy"}):

        state = {
            "selected_strategy": {"title": "t", "analysis_type": "regression", "required_columns": ["a", "b", "target"]},
            "business_objective": "",
            "csv_path": str(raw_path),
            "csv_encoding": "utf-8",
            "csv_sep": ",",
            "csv_decimal": ".",
            "data_summary": "",
            "leakage_audit_summary": "",
            "execution_contract": {
                "contract_version": "4.1",
                "cleaning_gates": [{"name": "required_columns_present", "severity": "HARD", "params": {}}],
                "data_engineer_runbook": {"steps": ["load", "clean", "persist"]},
            },
            "execution_contract_diagnostics": {
                "validation": {"accepted": True, "status": "ok"},
                "summary": {"accepted": True},
            },
            "de_view": {
                "required_columns": ["a", "b", "target"],
                "output_path": "data/cleaned_data.csv",
                "output_manifest_path": "data/cleaning_manifest.json",
                "cleaning_gates": [{"name": "required_columns_present", "severity": "HARD", "params": {}}],
                "data_engineer_runbook": {"steps": ["load", "clean", "persist"]},
            },
            "cleaning_view": {
                "cleaning_gates": [],
                "required_columns": [],
                "dialect": {"sep": ",", "decimal": ".", "encoding": "utf-8"},
            },
        }

        result = run_data_engineer(state)

        assert "cleaning_reviewer_failed" == result.get("pipeline_aborted_reason")
        assert result.get("data_engineer_failed") is True
        assert "Cleaning reviewer failed" in result.get("error_message", "")
