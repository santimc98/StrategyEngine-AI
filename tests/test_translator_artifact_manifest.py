import json
import os
import csv
import shutil
from pathlib import Path

from src.agents.business_translator import BusinessTranslatorAgent


class _EchoModel:
    def generate_content(self, prompt):
        class _Resp:
            def __init__(self, text):
                self.text = text

        return _Resp(prompt)


def test_translator_builds_artifact_manifest_and_html_tables(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)

    with open(os.path.join("data", "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"accuracy": 0.92}, f)

    with open(os.path.join("data", "execution_contract.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "required_outputs": ["data/metrics.json", "data/scored_rows.csv"],
                "artifact_requirements": {
                    "required_files": [{"path": "data/metrics.json"}, {"path": "data/scored_rows.csv"}]
                },
            },
            f,
        )

    with open(os.path.join("data", "output_contract_report.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "overall_status": "error",
                "present": ["data/metrics.json"],
                "missing": ["data/scored_rows.csv"],
                "artifact_requirements_report": {
                    "status": "error",
                    "files_report": {
                        "present": ["data/metrics.json"],
                        "missing": ["data/scored_rows.csv"],
                    },
                },
            },
            f,
        )

    with open(os.path.join("data", "produced_artifact_index.json"), "w", encoding="utf-8") as f:
        json.dump([{"path": "data/metrics.json", "artifact_type": "metrics"}], f)

    with open(os.path.join("data", "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"run_outcome": "GO_WITH_LIMITATIONS"}, f)

    agent = BusinessTranslatorAgent(api_key="dummy_key")
    agent.model = _EchoModel()
    report = agent.generate_report({"execution_output": "ok", "business_objective": "Objetivo"})

    assert os.path.exists(os.path.join("data", "report_artifact_manifest.json"))
    assert os.path.exists(os.path.join("data", "report_visual_tables.json"))

    with open(os.path.join("data", "report_artifact_manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["summary"]["required_total"] >= 2
    assert manifest["summary"]["required_missing"] >= 1
    assert any(item.get("path") == "data/metrics.json" for item in manifest.get("items", []))

    assert "artifact_inventory_table_html" not in report
    with open(os.path.join("data", "report_visual_tables.json"), "r", encoding="utf-8") as f:
        visual_tables = json.load(f)
    assert "exec-table artifact-inventory" in visual_tables["artifact_inventory_table_html"]
    assert "exec-table artifact-compliance" in visual_tables["artifact_compliance_table_html"]


def test_translator_manifest_profiles_csv_dimensions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)

    with open(os.path.join("data", "scored_rows.csv"), "w", encoding="utf-8") as f:
        f.write("id,prediction,score\n1,1,0.93\n2,0,0.11\n")

    with open(os.path.join("data", "execution_contract.json"), "w", encoding="utf-8") as f:
        json.dump({"required_outputs": ["data/scored_rows.csv"]}, f)

    with open(os.path.join("data", "produced_artifact_index.json"), "w", encoding="utf-8") as f:
        json.dump([{"path": "data/scored_rows.csv", "artifact_type": "predictions"}], f)

    agent = BusinessTranslatorAgent(api_key="dummy_key")
    agent.model = _EchoModel()
    _ = agent.generate_report({"execution_output": "ok", "business_objective": "Objetivo"})

    with open(os.path.join("data", "report_artifact_manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)

    scored = next(item for item in manifest["items"] if item["path"] == "data/scored_rows.csv")
    assert scored["present"] is True
    assert scored["row_count"] == 2
    assert scored["column_count"] == 3


def test_translator_manifest_csv_row_count_handles_multiline_cells(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)

    with open(os.path.join("data", "scored_rows.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "comment_text", "score"])
        writer.writerow([1, "line one\nline two", 0.93])
        writer.writerow([2, "single line", 0.11])

    with open(os.path.join("data", "execution_contract.json"), "w", encoding="utf-8") as f:
        json.dump({"required_outputs": ["data/scored_rows.csv"]}, f)

    with open(os.path.join("data", "produced_artifact_index.json"), "w", encoding="utf-8") as f:
        json.dump([{"path": "data/scored_rows.csv", "artifact_type": "predictions"}], f)

    agent = BusinessTranslatorAgent(api_key="dummy_key")
    agent.model = _EchoModel()
    _ = agent.generate_report({"execution_output": "ok", "business_objective": "Objetivo"})

    with open(os.path.join("data", "report_artifact_manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)

    scored = next(item for item in manifest["items"] if item["path"] == "data/scored_rows.csv")
    assert scored["present"] is True
    assert scored["row_count"] == 2


def test_translator_manifest_handles_rich_required_outputs_without_stringifying_dicts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)

    with open(os.path.join("data", "dataset_limpio.csv"), "w", encoding="utf-8") as f:
        f.write("id,value\n1,10\n")

    with open(os.path.join("data", "execution_contract.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "required_outputs": [
                    {
                        "intent": "dataset_limpio_csv",
                        "path": "data/dataset_limpio.csv",
                        "owner": "data_engineer",
                        "required": True,
                    },
                    {
                        "intent": "data_quality_report",
                        "path": "data/data_quality_report.json",
                        "owner": "data_engineer",
                        "required": True,
                    },
                ],
                "artifact_requirements": {
                    "required_files": [
                        {"path": "data/dataset_limpio.csv"},
                        {"path": "data/data_quality_report.json"},
                    ]
                },
            },
            f,
        )

    with open(os.path.join("data", "output_contract_report.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "overall_status": "error",
                "present": ["data/dataset_limpio.csv"],
                "missing": ["data/data_quality_report.json"],
                "artifact_requirements_report": {
                    "status": "error",
                    "files_report": {
                        "present": ["data/dataset_limpio.csv"],
                        "missing": ["data/data_quality_report.json"],
                    },
                },
            },
            f,
        )

    with open(os.path.join("data", "produced_artifact_index.json"), "w", encoding="utf-8") as f:
        json.dump([{"path": "data/dataset_limpio.csv", "artifact_type": "dataset"}], f)

    with open(os.path.join("data", "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"run_outcome": "NO_GO"}, f)

    agent = BusinessTranslatorAgent(api_key="dummy_key")
    agent.model = _EchoModel()
    _ = agent.generate_report({"execution_output": "ok", "business_objective": "Objetivo"})

    with open(os.path.join("data", "report_artifact_manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["summary"]["required_total"] == 2
    assert manifest["summary"]["required_missing"] == 1
    paths = [item.get("path") for item in manifest.get("items", [])]
    assert "data/dataset_limpio.csv" in paths
    assert "data/data_quality_report.json" in paths
    assert not any(path and path.startswith("{") for path in paths)


def test_replay_c946b64d_manifest_deduplicates_rich_required_outputs_from_real_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    os.makedirs(os.path.join("artifacts", "clean"), exist_ok=True)

    repo_root = Path(__file__).resolve().parents[1]
    run_root = repo_root / "runs" / "c946b64d"

    shutil.copy(run_root / "work" / "data" / "execution_contract_raw.json", tmp_path / "data" / "execution_contract.json")
    shutil.copy(run_root / "work" / "data" / "output_contract_report.json", tmp_path / "data" / "output_contract_report.json")
    shutil.copy(run_root / "work" / "data" / "produced_artifact_index.json", tmp_path / "data" / "produced_artifact_index.json")
    shutil.copy(run_root / "work" / "data" / "run_summary.json", tmp_path / "data" / "run_summary.json")
    shutil.copy(
        run_root / "work" / "artifacts" / "clean" / "dataset_enriquecido.csv",
        tmp_path / "artifacts" / "clean" / "dataset_enriquecido.csv",
    )

    agent = BusinessTranslatorAgent(api_key="dummy_key")
    agent.model = _EchoModel()
    _ = agent.generate_report({"execution_output": "ok", "business_objective": "Objetivo"})

    with open(os.path.join("data", "report_artifact_manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["summary"]["required_total"] == 9
    assert manifest["summary"]["required_present"] == 1
    assert manifest["summary"]["required_missing"] == 8
    paths = [item.get("path") for item in manifest.get("items", [])]
    assert "artifacts/clean/dataset_enriquecido.csv" in paths
    assert not any(path and str(path).startswith("{") for path in paths)


def test_translator_manifest_treats_glob_required_plot_output_as_present_when_files_exist(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs(os.path.join("data"), exist_ok=True)
    os.makedirs(os.path.join("static", "plots"), exist_ok=True)

    with open(os.path.join("static", "plots", "cv_folds.png"), "wb") as f:
        f.write(b"png")

    with open(os.path.join("data", "execution_contract.json"), "w", encoding="utf-8") as f:
        json.dump({"required_outputs": [{"path": "static/plots/*.png", "required": True}]}, f)

    with open(os.path.join("data", "output_contract_report.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "overall_status": "error",
                "present": [],
                "missing": ["static/plots/*.png"],
            },
            f,
        )

    with open(os.path.join("data", "produced_artifact_index.json"), "w", encoding="utf-8") as f:
        json.dump(
            [{"path": "static/plots/cv_folds.png", "artifact_type": "plot"}],
            f,
        )

    with open(os.path.join("data", "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"run_outcome": "GO_WITH_LIMITATIONS"}, f)

    agent = BusinessTranslatorAgent(api_key="dummy_key")
    agent.model = _EchoModel()
    _ = agent.generate_report({"execution_output": "ok", "business_objective": "Objetivo"})

    with open(os.path.join("data", "report_artifact_manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)

    wildcard_item = next(item for item in manifest["items"] if item["path"] == "static/plots/*.png")
    assert wildcard_item["present"] is True
    assert wildcard_item["status"] == "ok"
    assert "static/plots/cv_folds.png" in wildcard_item.get("matched_paths", [])
    assert manifest["summary"]["required_missing"] == 0
