from src.graph.graph import _validate_projected_views_for_execution


def _base_cleaning_views():
    return {
        "de_view": {
            "output_path": "data/cleaned_data.csv",
            "output_manifest_path": "data/cleaning_manifest.json",
            "required_outputs": ["data/cleaned_data.csv", "data/cleaning_manifest.json"],
            "required_columns": ["id", "feature_a"],
            "cleaning_gates": [{"name": "required_columns_present", "severity": "HARD", "params": {}}],
            "data_engineer_runbook": {"steps": ["load", "clean", "persist"]},
        },
        "cleaning_view": {
            "required_outputs": ["data/cleaned_data.csv", "data/cleaning_manifest.json"],
            "cleaning_gates": [{"name": "required_columns_present", "severity": "HARD", "params": {}}],
        },
    }


def test_view_projection_rejects_missing_cleaning_gates():
    contract = {"scope": "cleaning_only", "cleaning_gates": [{"name": "required_columns_present", "severity": "HARD"}]}
    views = _base_cleaning_views()
    views["cleaning_view"] = {"cleaning_gates": []}

    errors = _validate_projected_views_for_execution(contract, views)

    assert "cleaning_view_cleaning_gates_empty" in errors


def test_view_projection_accepts_non_empty_cleaning_gates():
    contract = {
        "scope": "cleaning_only",
        "cleaning_gates": [{"name": "required_columns_present", "severity": "HARD"}],
        "artifact_requirements": {
            "clean_dataset": {
                "output_path": "data/cleaned_data.csv",
                "output_manifest_path": "data/cleaning_manifest.json",
            }
        },
    }
    views = _base_cleaning_views()

    errors = _validate_projected_views_for_execution(contract, views)

    assert "cleaning_view_cleaning_gates_empty" not in errors
    assert "de_view_cleaning_gates_empty" not in errors
    assert "de_view_data_engineer_runbook_missing" not in errors


def test_view_projection_rejects_missing_de_view_runbook():
    contract = {"scope": "cleaning_only", "cleaning_gates": [{"name": "required_columns_present", "severity": "HARD"}]}
    views = _base_cleaning_views()
    views["de_view"]["data_engineer_runbook"] = {}

    errors = _validate_projected_views_for_execution(contract, views)

    assert "de_view_data_engineer_runbook_missing" in errors


def test_view_projection_rejects_incomplete_de_required_outputs_coverage():
    contract = {
        "scope": "cleaning_only",
        "required_outputs": [
            {"path": "data/cleaned_data.csv", "owner": "data_engineer"},
            {"path": "data/cleaning_manifest.json", "owner": "data_engineer"},
            {"path": "data/dataset_enriched.csv", "owner": "data_engineer"},
        ],
        "cleaning_gates": [{"name": "required_columns_present", "severity": "HARD"}],
    }
    views = _base_cleaning_views()

    errors = _validate_projected_views_for_execution(contract, views)

    assert "de_view_required_outputs_incomplete" in errors
    assert "cleaning_view_required_outputs_incomplete" in errors


def test_view_projection_rejects_incomplete_cleaning_gate_coverage():
    contract = {
        "scope": "cleaning_only",
        "cleaning_gates": [
            {"name": "required_columns_present", "severity": "HARD"},
            {"name": "standardize_dates", "severity": "HARD"},
        ],
        "artifact_requirements": {
            "clean_dataset": {
                "output_path": "data/cleaned_data.csv",
                "output_manifest_path": "data/cleaning_manifest.json",
            }
        },
    }
    views = _base_cleaning_views()

    errors = _validate_projected_views_for_execution(contract, views)

    assert "de_view_cleaning_gates_incomplete" in errors
    assert "cleaning_view_cleaning_gates_incomplete" in errors
