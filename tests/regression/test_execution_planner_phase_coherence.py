import json
from pathlib import Path

import pytest

from src.agents.execution_planner import (
    _normalize_cleaning_gates,
    _normalize_qa_gates,
    _validate_gate_phase_coherence,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_fa341d69_contract_flags_phase_coherence_violations():
    run_dir = REPO_ROOT / "runs" / "fa341d69"
    contract_path = run_dir / "agents" / "execution_planner" / "contract_full.json"
    if not contract_path.exists():
        pytest.skip("recorded run fa341d69 is not available")

    contract = _load_json(contract_path)
    result = _validate_gate_phase_coherence(contract)

    assert result["status"] == "violations"
    flagged_names = {violation["gate_name"] for violation in result["violations"]}
    assert "target_excluded_from_features" in flagged_names
    assert "leakage_columns_absent_from_features" in flagged_names

    problem_kinds = {
        violation["kind"]
        for violation in result["violations"]
        if violation["gate_name"]
        in {"target_excluded_from_features", "leakage_columns_absent_from_features"}
    }
    assert problem_kinds & {"structural_contradiction", "phase_mismatch"}


def test_gate_normalizers_preserve_binding_fields():
    raw = {
        "name": "no_leakage_in_model",
        "severity": "HARD",
        "params": {"forbidden_columns": ["x"]},
        "applies_to_artifact": "artifacts/ml/model.pkl",
        "evaluated_by": "qa_reviewer",
        "evidence_source": "model.feature_names_in_",
        "phase_reasoning": "Property of trained model, not cleaned CSV.",
    }

    qa_gate = _normalize_qa_gates([raw])[0]
    cleaning_gate = _normalize_cleaning_gates([raw])[0]

    for gate in (qa_gate, cleaning_gate):
        assert gate["applies_to_artifact"] == "artifacts/ml/model.pkl"
        assert gate["evaluated_by"] == "qa_reviewer"
        assert gate["evidence_source"] == "model.feature_names_in_"
        assert gate["phase_reasoning"].startswith("Property")


def test_well_formed_contract_passes_phase_coherence():
    contract = {
        "contract_version": "5.0",
        "data_engineer": {
            "required_outputs": [
                {
                    "intent": "primary_deliverable",
                    "path": "artifacts/clean/x.csv",
                    "primary": True,
                    "kind": "dataset",
                },
                {
                    "intent": "cleaning_manifest",
                    "path": "artifacts/clean/cleaning_manifest.json",
                    "kind": "manifest",
                },
            ],
            "artifact_requirements": {
                "cleaned_dataset": {
                    "output_path": "artifacts/clean/x.csv",
                    "required_columns": ["id", "a", "b", "target"],
                }
            },
            "cleaning_gates": [
                {
                    "name": "required_columns_present",
                    "severity": "HARD",
                    "params": {},
                    "applies_to_artifact": "artifacts/clean/x.csv",
                    "evaluated_by": "cleaning_reviewer",
                    "evidence_source": "cleaned CSV header",
                    "phase_reasoning": "Column presence is observable at cleaning time.",
                }
            ],
        },
        "ml_engineer": {
            "required_outputs": [
                {
                    "intent": "model_artifact",
                    "path": "artifacts/ml/model.pkl",
                    "kind": "model",
                },
                {
                    "intent": "validation_metrics",
                    "path": "artifacts/ml/validation_metrics.json",
                    "kind": "metrics",
                },
            ],
            "qa_gates": [
                {
                    "name": "no_leakage_in_trained_model",
                    "severity": "HARD",
                    "params": {"forbidden_columns": ["leak_col"]},
                    "applies_to_artifact": "artifacts/ml/model.pkl",
                    "evaluated_by": "qa_reviewer",
                    "evidence_source": "model.feature_names_in_",
                    "phase_reasoning": "Property of the trained model feature set.",
                }
            ],
            "reviewer_gates": [
                {
                    "name": "metric_report_present",
                    "severity": "SOFT",
                    "params": {},
                    "applies_to_artifact": "validation_metrics",
                    "evaluated_by": "qa_reviewer",
                    "evidence_source": "validation_metrics.json",
                    "phase_reasoning": "Metric reporting is observable after ML validation.",
                }
            ],
        },
    }

    result = _validate_gate_phase_coherence(contract)
    assert result["status"] == "ok", result["violations"]
