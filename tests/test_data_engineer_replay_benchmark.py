import json
from pathlib import Path

from src.utils.data_engineer_replay_benchmark import (
    DataEngineerReplayCase,
    _run_quality_checks,
    execute_script_for_case,
    load_data_engineer_replay_case,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _build_minimal_case(tmp_path: Path) -> DataEngineerReplayCase:
    csv_path = tmp_path / "input.csv"
    csv_path.write_text("feature_a,target\n1,0\n2,1\n", encoding="utf-8")
    return DataEngineerReplayCase(
        run_id="synthetic-case",
        repo_root=REPO_ROOT,
        run_dir=tmp_path,
        csv_path=csv_path,
        business_objective="benchmark",
        csv_encoding="utf-8",
        csv_sep=",",
        csv_decimal=".",
        strategy={"target_columns": ["target"]},
        execution_contract={"model_features": ["feature_a"], "target_columns": ["target"]},
        de_view={"required_columns": ["feature_a", "target"]},
        data_audit="audit",
        required_output_paths=[
            "artifacts/clean/leads_cleaned.csv",
            "artifacts/clean/leads_enriched_features.csv",
            "artifacts/clean/cleaning_manifest.json",
        ],
        model_features=["feature_a"],
        target_columns=["target"],
        prompt_path=None,
        baseline_script_path=None,
        baseline_error_path=None,
        dataset_profile_path=None,
        worker_input_path=None,
        de_context_path=None,
        quality_checks=[],
    )


def test_load_data_engineer_replay_case_8ec99856():
    case = load_data_engineer_replay_case("8ec99856", repo_root=REPO_ROOT)

    assert case.run_id == "8ec99856"
    assert case.csv_path.exists()
    assert "converted_to_opportunity_90d" in case.target_columns
    assert "annual_revenue" in case.model_features
    assert "artifacts/clean/leads_enriched_features.csv" in case.required_output_paths
    assert case.de_view.get("model_features")


def test_execute_script_for_case_reports_success(tmp_path):
    case = _build_minimal_case(tmp_path)
    script = """
import json
import os
import pandas as pd

os.makedirs("artifacts/clean", exist_ok=True)
df = pd.read_csv("data/raw.csv")
df.to_csv("artifacts/clean/leads_cleaned.csv", index=False)
df[["feature_a", "target"]].to_csv("artifacts/clean/leads_enriched_features.csv", index=False)
with open("artifacts/clean/cleaning_manifest.json", "w", encoding="utf-8") as f:
    json.dump({"ok": True}, f)
"""
    result = execute_script_for_case(case, workspace_dir=tmp_path / "workspace_success", script_text=script)

    assert result["success"] is True
    assert result["required_outputs_missing_count"] == 0
    assert result["enriched_schema_exact_match"] is True


def test_execute_script_for_case_reports_failure(tmp_path):
    case = _build_minimal_case(tmp_path)
    script = """
raise RuntimeError("boom")
"""
    result = execute_script_for_case(case, workspace_dir=tmp_path / "workspace_failure", script_text=script)

    assert result["success"] is False
    assert result["returncode"] != 0
    assert result["required_outputs_missing_count"] == 3


def test_data_engineer_benchmark_cases_inventory_references_existing_artifacts():
    inventory_path = REPO_ROOT / "tests" / "data_engineer_benchmark_cases.json"
    payload = json.loads(inventory_path.read_text(encoding="utf-8"))

    assert payload.get("version") == 1
    cases = payload.get("cases")
    assert isinstance(cases, list) and cases

    for case in cases:
        assert case.get("status") == "benchmark_ready"
        required_files = case.get("required_files")
        assert isinstance(required_files, list) and required_files
        for rel_path in required_files:
            assert (REPO_ROOT / rel_path).exists(), rel_path


def test_quality_check_max_null_inflation(tmp_path):
    case = _build_minimal_case(tmp_path)
    case.quality_checks = [
        {
            "type": "max_null_inflation",
            "report_path": "artifacts/reports/quality_audit_report.json",
            "columns": ["created_at"],
            "max_increase": 0.20,
        }
    ]
    workspace = tmp_path / "workspace_quality_dates"
    (workspace / "data").mkdir(parents=True, exist_ok=True)
    (workspace / "artifacts" / "reports").mkdir(parents=True, exist_ok=True)
    (workspace / "data" / "dataset_profile.json").write_text(
        json.dumps({"missing_frac": {"created_at": 0.10}}),
        encoding="utf-8",
    )
    (workspace / "artifacts" / "reports" / "quality_audit_report.json").write_text(
        json.dumps({"null_rates_after_cleaning": {"created_at": 0.45}}),
        encoding="utf-8",
    )
    results = _run_quality_checks(case, workspace)
    assert results[0]["passed"] is False
    assert results[0]["violations"][0]["column"] == "created_at"


def test_quality_check_no_all_placeholder_dedup_drops(tmp_path):
    case = _build_minimal_case(tmp_path)
    case.quality_checks = [
        {
            "type": "no_all_placeholder_dedup_drops",
            "csv_path": "artifacts/clean/deduplication_decisions.csv",
            "decision_column": "decision",
            "keys_column": "dedup_keys",
            "drop_token": "DROPPED",
            "placeholder_tokens": ["", "nan", "<na>"],
        }
    ]
    workspace = tmp_path / "workspace_quality_dedup"
    (workspace / "artifacts" / "clean").mkdir(parents=True, exist_ok=True)
    (workspace / "artifacts" / "clean" / "deduplication_decisions.csv").write_text(
        "lead_id,decision,dedup_keys\n"
        "L1,DUPLICATE_DROPPED,\"{'email': 'nan', 'company_name': 'nan'}\"\n",
        encoding="utf-8",
    )
    results = _run_quality_checks(case, workspace)
    assert results[0]["passed"] is False
    assert results[0]["violations"][0]["lead_id"] == "L1"
