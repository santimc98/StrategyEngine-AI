import json
import os

import pytest

from src.graph import graph as graph_mod


@pytest.fixture
def tmp_workdir(tmp_path, monkeypatch):
    old_cwd = os.getcwd()
    monkeypatch.chdir(tmp_path)
    yield tmp_path
    os.chdir(old_cwd)


def test_artifact_gate_dialect_prefers_state(tmp_workdir):
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "cleaning_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump({"output_dialect": {"sep": ";", "decimal": ",", "encoding": "latin-1"}}, handle)

    state = {"csv_sep": "\t", "csv_decimal": ".", "csv_encoding": "utf-8"}
    contract = {"output_dialect": {"sep": ";", "decimal": ",", "encoding": "latin-1"}}

    dialect = graph_mod._resolve_artifact_gate_dialect(state, contract)
    assert dialect["sep"] == "\t"
    assert dialect["decimal"] == "."
    assert dialect["encoding"] == "utf-8"


def test_artifact_gate_dialect_uses_declared_manifest_path(tmp_workdir):
    manifest_path = tmp_workdir / "artifacts" / "manifests" / "custom_clean_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"output_dialect": {"sep": ";", "decimal": ",", "encoding": "latin-1"}}),
        encoding="utf-8",
    )
    contract = {
        "artifact_requirements": {
            "clean_dataset": {
                "output_path": "artifacts/features/custom_cleaned.csv",
                "output_manifest_path": "artifacts/manifests/custom_clean_manifest.json",
            }
        }
    }

    dialect = graph_mod._resolve_artifact_gate_dialect({}, contract)
    assert dialect["sep"] == ";"
    assert dialect["decimal"] == ","
    assert dialect["encoding"] == "latin-1"


def test_case_alignment_skip_reason_uses_declared_scored_rows_path(tmp_workdir):
    manifest_path = tmp_workdir / "artifacts" / "manifests" / "custom_clean_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"output_dialect": {"sep": ";", "decimal": ".", "encoding": "utf-8"}}), encoding="utf-8")

    scored_rows_path = tmp_workdir / "artifacts" / "outputs" / "scored_rows.csv"
    scored_rows_path.parent.mkdir(parents=True, exist_ok=True)
    scored_rows_path.write_text("segment_name;pred_score\nA;0.9\nB;0.2\n", encoding="utf-8")

    contract = {
        "required_outputs": ["artifacts/outputs/scored_rows.csv"],
        "artifact_requirements": {
            "clean_dataset": {
                "output_manifest_path": "artifacts/manifests/custom_clean_manifest.json",
            }
        },
    }

    skip_reason = graph_mod._case_alignment_skip_reason(contract, {})
    assert skip_reason == ""
