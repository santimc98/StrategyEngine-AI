from src.utils.contract_validator import normalize_artifact_requirements
from src.utils.contract_accessors import get_required_outputs
from src.utils.output_contract import check_required_outputs


def test_normalize_artifact_requirements_filters_conceptual_outputs() -> None:
    contract = {"required_outputs": ["data/metrics.json", "Priority ranking"]}

    normalize_artifact_requirements(contract)

    required_outputs = contract.get("required_outputs", [])
    assert "Priority ranking" not in required_outputs
    assert "data/metrics.json" in required_outputs
    reporting = contract.get("reporting_requirements", {})
    narrative = reporting.get("narrative_outputs", [])
    assert "Priority ranking" in narrative


def test_get_required_outputs_ignores_concepts() -> None:
    contract = {"required_outputs": ["Priority ranking", "data/metrics.json"]}

    outputs = get_required_outputs(contract)

    assert "data/metrics.json" in outputs
    assert "Priority ranking" not in outputs


def test_get_required_outputs_extracts_dict_path_only() -> None:
    contract = {
        "required_outputs": [
            {"path": "data/metrics.json", "id": "metrics_artifact"},
            {"id": "should_not_be_path"},
        ]
    }

    outputs = get_required_outputs(contract)

    assert "data/metrics.json" in outputs
    assert "should_not_be_path" not in outputs
    assert not any("{'path':" in path for path in outputs)


def test_get_required_outputs_includes_visualization_required_plots() -> None:
    contract = {
        "required_outputs": ["data/metrics.json"],
        "visualization_requirements": {
            "required": True,
            "required_plots": [{"name": "reliability_curve"}],
            "outputs_dir": "static/plots",
        },
    }

    outputs = get_required_outputs(contract)

    assert "data/metrics.json" in outputs
    assert "static/plots/reliability_curve.png" in outputs


def test_output_contract_ignores_concepts(tmp_path) -> None:
    metrics_path = tmp_path / "data" / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text("{}", encoding="utf-8")

    report = check_required_outputs([str(metrics_path), "Priority ranking"])

    assert "Priority ranking" not in report.get("missing", [])


def test_get_required_outputs_skips_inactive_ml_owner_for_cleaning_only_scope() -> None:
    contract = {
        "scope": "cleaning_only",
        "active_workstreams": {
            "cleaning": True,
            "feature_engineering": True,
            "model_training": False,
        },
        "required_outputs": [
            {"path": "artifacts/clean/clean_base.csv", "owner": "data_engineer", "required": True},
            {"path": "artifacts/ml/handoff_note.json", "owner": "ml_engineer", "required": True},
        ],
    }

    outputs = get_required_outputs(contract)

    assert "artifacts/clean/clean_base.csv" in outputs
    assert "artifacts/ml/handoff_note.json" not in outputs
