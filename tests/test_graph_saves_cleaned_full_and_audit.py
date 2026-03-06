import json
import os

import pytest
from unittest.mock import patch

from src.graph.graph import run_data_engineer


@pytest.fixture
def tmp_workdir(tmp_path, monkeypatch):
    old_cwd = os.getcwd()
    monkeypatch.chdir(tmp_path)
    yield tmp_path
    os.chdir(old_cwd)
def _run_data_engineer_with_local_runner(
    *,
    state,
    cleaned_bytes,
    manifest_bytes,
    mock_cleaning_result,
    mock_audit=None,
):
    def _fake_heavy_runner(**kwargs):
        os.makedirs("data", exist_ok=True)
        with open("data/cleaned_data.csv", "wb") as f_clean:
            f_clean.write(cleaned_bytes)
        with open("data/cleaning_manifest.json", "wb") as f_manifest:
            f_manifest.write(manifest_bytes)
        return {
            "ok": True,
            "unavailable": False,
            "error_details": "",
            "outlier_report_path": "",
        }

    patchers = [
        patch("src.graph.graph.scan_code_safety", return_value=(True, [])),
        patch("src.graph.graph._get_execution_runtime_mode", return_value="local"),
        patch("src.graph.graph._get_heavy_runner_config", return_value={"bucket": "local", "job": "local"}),
        patch("src.graph.graph._execute_data_engineer_via_heavy_runner", side_effect=_fake_heavy_runner),
        patch("src.graph.graph.data_engineer.generate_cleaning_script", return_value="print('clean')"),
        patch("src.graph.graph.cleaning_reviewer.review_cleaning", return_value=mock_cleaning_result),
        patch.dict(os.environ, {"DEEPSEEK_API_KEY": "dummy", "GOOGLE_API_KEY": "dummy"}),
    ]
    if mock_audit is not None:
        patchers.append(
            patch("src.graph.graph.run_unsupervised_numeric_relation_audit", return_value=mock_audit)
        )

    with patchers[0], patchers[1], patchers[2], patchers[3], patchers[4], patchers[5], patchers[6]:
        if mock_audit is not None:
            with patchers[7]:
                return run_data_engineer(state)
        return run_data_engineer(state)


def test_graph_saves_cleaned_full_and_audit(tmp_workdir, monkeypatch):
    raw_path = tmp_workdir / "raw.csv"
    raw_path.write_text("col1,col2,target\n1,2,3\n4,5,9\n", encoding="utf-8")

    cleaned_bytes = b"a,b,target\n1,2,3\n4,5,9\n"
    manifest_bytes = json.dumps(
        {"output_dialect": {"sep": ",", "decimal": ".", "encoding": "utf-8"}}
    ).encode("utf-8")

    # V4.1: Mock cleaning reviewer to pass (contract-strict mode rejects without cleaning_gates)
    mock_cleaning_result = {
        "status": "APPROVED",
        "feedback": "Test mock approval",
        "failed_checks": [],
        "required_fixes": [],
        "warnings": [],
        "hard_failures": [],
        "soft_failures": [],
        "contract_source_used": "cleaning_view",
    }
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
            "cleaning_gates": [{"name": "required_columns_present", "severity": "HARD", "params": {}}],
            "required_columns": ["a", "b", "target"],
            "dialect": {"sep": ",", "decimal": ".", "encoding": "utf-8"},
        },
    }

    result = _run_data_engineer_with_local_runner(
        state=state,
        cleaned_bytes=cleaned_bytes,
        manifest_bytes=manifest_bytes,
        mock_cleaning_result=mock_cleaning_result,
    )

    assert os.path.exists("data/cleaned_full.csv")
    assert os.path.exists("data/leakage_audit.json")
    assert "leakage_audit_summary" in result


def test_graph_leakage_summary_includes_risk_flags(tmp_workdir, monkeypatch):
    raw_path = tmp_workdir / "raw.csv"
    raw_path.write_text("col1,col2,target\n1,2,3\n4,5,9\n", encoding="utf-8")

    cleaned_bytes = b"a,b,target\n1,2,3\n4,5,9\n"
    manifest_bytes = json.dumps(
        {"output_dialect": {"sep": ",", "decimal": ".", "encoding": "utf-8"}}
    ).encode("utf-8")

    mock_cleaning_result = {
        "status": "APPROVED",
        "feedback": "Test mock approval",
        "failed_checks": [],
        "required_fixes": [],
        "warnings": [],
        "hard_failures": [],
        "soft_failures": [],
        "contract_source_used": "cleaning_view",
    }
    mock_audit = {
        "relations": [],
        "risk_flags": [
            {
                "type": "suspicious_name",
                "columns": ["target_proxy"],
                "support_frac": 1.0,
            }
        ],
        "scanned_columns": ["a", "b", "target"],
        "rows": 2,
    }
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
            "cleaning_gates": [{"name": "required_columns_present", "severity": "HARD", "params": {}}],
            "required_columns": ["a", "b", "target"],
            "dialect": {"sep": ",", "decimal": ".", "encoding": "utf-8"},
        },
    }

    result = _run_data_engineer_with_local_runner(
        state=state,
        cleaned_bytes=cleaned_bytes,
        manifest_bytes=manifest_bytes,
        mock_cleaning_result=mock_cleaning_result,
        mock_audit=mock_audit,
    )

    assert "suspicious_name:target_proxy" in result.get("leakage_audit_summary", "")
