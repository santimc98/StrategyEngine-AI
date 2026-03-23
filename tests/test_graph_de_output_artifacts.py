import json
from pathlib import Path
import os

from src.graph.graph import _resolve_de_output_artifacts, _persist_runner_artifact_index


def test_resolve_de_output_artifacts_covers_all_owned_contract_outputs():
    state = {
        "execution_contract": {
            "required_outputs": [
                {"path": "artifacts/clean/dataset_limpio.csv", "owner": "data_engineer", "required": True},
                {"path": "artifacts/clean/dataset_enriquecido.csv", "owner": "data_engineer", "required": True},
                {"path": "artifacts/reports/data_quality_report.json", "owner": "data_engineer", "required": True},
                {"path": "artifacts/reports/transformation_log.json", "owner": "data_engineer", "required": True},
                {"path": "artifacts/model/model_metrics.json", "owner": "ml_engineer", "required": True},
            ]
        },
        "de_view": {
            "output_path": "artifacts/clean/dataset_enriquecido.csv",
            "output_manifest_path": "artifacts/clean/cleaning_manifest.json",
            "required_outputs": [
                "artifacts/clean/dataset_limpio.csv",
                "artifacts/clean/dataset_enriquecido.csv",
                "artifacts/reports/data_quality_report.json",
                "artifacts/reports/transformation_log.json",
                "artifacts/clean/cleaning_manifest.json",
            ],
        },
    }

    output_path, manifest_path, required_artifacts = _resolve_de_output_artifacts(state)

    assert output_path == "artifacts/clean/dataset_enriquecido.csv"
    assert manifest_path == "artifacts/clean/cleaning_manifest.json"
    assert required_artifacts == [
        "artifacts/clean/dataset_limpio.csv",
        "artifacts/clean/dataset_enriquecido.csv",
        "artifacts/reports/data_quality_report.json",
        "artifacts/reports/transformation_log.json",
        "artifacts/clean/cleaning_manifest.json",
    ]


def test_replay_c946b64d_resolve_de_output_artifacts_covers_all_owned_outputs_from_contract_and_view():
    repo_root = Path(__file__).resolve().parents[1]
    contract = json.loads(
        (repo_root / "runs" / "c946b64d" / "work" / "data" / "execution_contract_raw.json").read_text(
            encoding="utf-8"
        )
    )
    de_view = json.loads(
        (repo_root / "runs" / "c946b64d" / "work" / "data" / "contracts" / "views" / "de_view.json").read_text(
            encoding="utf-8"
        )
    )

    output_path, manifest_path, required_artifacts = _resolve_de_output_artifacts(
        {"execution_contract": contract, "de_view": de_view}
    )

    assert output_path == "artifacts/clean/dataset_enriquecido.csv"
    assert manifest_path == "artifacts/clean/cleaning_manifest.json"
    assert required_artifacts == [
        "artifacts/clean/dataset_limpio.csv",
        "artifacts/clean/dataset_enriquecido.csv",
        "artifacts/reports/data_quality_report.json",
        "artifacts/reports/column_role_matrix.json",
        "artifacts/reports/leakage_risk_register.json",
        "artifacts/logs/transformation_log.json",
        "artifacts/clean/deduplication_decisions.csv",
        "artifacts/reports/data_dictionary_cleaned.json",
        "artifacts/clean/quality_flags_dataset.csv",
        "artifacts/clean/cleaning_manifest.json",
    ]


def test_persist_runner_artifact_index_keeps_downloaded_outputs_on_partial_failure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    state = {
        "execution_contract": {
            "required_outputs": [
                {"path": "artifacts/clean/clean_dataset.csv", "owner": "data_engineer", "required": True},
                {"path": "artifacts/clean/deduplication_decisions.csv", "owner": "data_engineer", "required": True},
                {"path": "artifacts/clean/cleaning_manifest.json", "owner": "data_engineer", "required": True},
            ]
        }
    }
    runner_result = {
        "downloaded": {
            "artifacts/clean/clean_dataset.csv": "artifacts/clean/clean_dataset.csv",
            "artifacts/clean/cleaning_manifest.json": "artifacts/clean/cleaning_manifest.json",
        },
        "error": {
            "error": "missing_required_outputs",
            "missing": ["artifacts/clean/deduplication_decisions.csv"],
            "uploaded": [
                "artifacts/clean/clean_dataset.csv",
                "artifacts/clean/cleaning_manifest.json",
            ],
            "present": ["artifacts/clean/clean_dataset.csv"],
        },
    }

    merged = _persist_runner_artifact_index(state, runner_result, contract=state["execution_contract"])

    paths = {item["path"] for item in merged}
    assert "artifacts/clean/clean_dataset.csv" in paths
    assert "artifacts/clean/cleaning_manifest.json" in paths
    assert "artifacts/clean/deduplication_decisions.csv" not in paths
    persisted = json.loads(Path("data/produced_artifact_index.json").read_text(encoding="utf-8"))
    persisted_paths = {item["path"] for item in persisted}
    assert paths == persisted_paths
