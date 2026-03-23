import json
import os

from src.graph.graph import _persist_output_contract_report


def test_output_contract_report_is_error_when_execution_contract_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)

    report = _persist_output_contract_report(
        {"execution_contract": {}},
        reason="execution_contract_invalid",
        path="data/output_contract_report.json",
    )

    assert report["overall_status"] == "error"
    assert report["reason"] == "execution_contract_invalid"
    assert "Contract unavailable" in report["summary"]

    with open("data/output_contract_report.json", "r", encoding="utf-8") as handle:
        persisted = json.load(handle)
    assert persisted["overall_status"] == "error"
    assert persisted["reason"] == "execution_contract_invalid"
