import json
from pathlib import Path

import pandas as pd
import pytest

from src.agents.cleaning_reviewer import (
    _build_facts,
    _build_gates_contract,
    _build_llm_prompt,
    _merge_cleaning_gates,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_run_8ef7d68e_attempt_1_prompt_contains_primary_audit_evidence():
    run_dir = REPO_ROOT / "runs" / "8ef7d68e"
    if not run_dir.exists():
        pytest.skip("recorded run 8ef7d68e is not available in this checkout")

    attempt_output = run_dir / "sandbox" / "local_runner" / "data_engineer" / "attempt_1" / "output"
    cleaned_csv = attempt_output / "artifacts" / "clean" / "account_snapshot_ml_ready.csv"
    manifest_path = attempt_output / "artifacts" / "clean" / "cleaning_manifest.json"
    outlier_report_path = attempt_output / "data" / "outlier_treatment_report.json"
    cleaning_view_path = run_dir / "work" / "data" / "contracts" / "views" / "cleaning_view.json"
    obligations_path = run_dir / "work" / "data" / "contracts" / "support" / "data_engineer_artifact_obligations.json"
    script_path = run_dir / "agents" / "data_engineer" / "iteration_1" / "script.py"

    for path in (cleaned_csv, manifest_path, outlier_report_path, cleaning_view_path, script_path):
        assert path.exists(), f"missing replay artifact: {path}"

    cleaning_view = _load_json(cleaning_view_path)
    artifact_obligations = _load_json(obligations_path) if obligations_path.exists() else {}
    manifest = _load_json(manifest_path)
    outlier_report = _load_json(outlier_report_path)
    dialect = manifest.get("output_dialect") or {"sep": ",", "decimal": ".", "encoding": "utf-8"}
    cleaned_sample = pd.read_csv(cleaned_csv, dtype=str, sep=dialect["sep"], nrows=20)
    cleaned_header = list(cleaned_sample.columns)
    gates, contract_source, warnings = _merge_cleaning_gates(cleaning_view)
    assert warnings == []

    facts = _build_facts(
        cleaned_header=cleaned_header,
        required_columns=cleaning_view["artifact_requirements"]["cleaned_dataset"]["required_columns"],
        manifest=manifest,
        sample_str=cleaned_sample,
        sample_infer=cleaned_sample,
        raw_sample=None,
        gates=gates,
        column_roles=cleaning_view.get("column_roles") or {},
        dataset_profile={},
        outlier_policy=cleaning_view.get("outlier_policy") or {},
        outlier_report=outlier_report,
        outlier_report_path=str(outlier_report_path),
        cleaned_csv_path=str(cleaned_csv),
        cleaning_manifest_path=str(manifest_path),
        dialect=dialect,
        cleaning_view=cleaning_view,
        artifact_obligations=artifact_obligations,
    )
    gates_contract = _build_gates_contract(gates, cleaning_view)
    prompt, payload = _build_llm_prompt(
        gates=gates,
        required_columns=cleaning_view["artifact_requirements"]["cleaned_dataset"]["required_columns"],
        dialect=dialect,
        column_roles=cleaning_view.get("column_roles") or {},
        facts=facts,
        deterministic_gate_results=[],
        contract_source_used=contract_source,
        cleaning_code=script_path.read_text(encoding="utf-8"),
        artifact_obligations=artifact_obligations,
        gates_contract=gates_contract,
    )

    assert "SENIOR DATA QUALITY AUDITOR" in prompt
    assert len(payload["cleaning_code"]) > 40000
    assert "TRUNCATED_MIDDLE" not in payload["cleaning_code"]
    assert payload["facts"]["manifest_deduplication"]["key"] == ["account_id", "snapshot_month_end"]
    statuses = payload["facts"]["manifest_cleaning_gates_status"]
    for gate in cleaning_view["cleaning_gates"]:
        assert gate["name"] in statuses
    scoring = payload["facts"]["secondary_outputs"]["artifacts/clean/account_snapshot_scoring_cohort.csv"]
    assert scoring["sample_rows"]
    assert scoring["row_count"] == manifest["partition_counts"]["scoring_rows"]
    assert payload["facts"]["outlier_report_raw"] == outlier_report
