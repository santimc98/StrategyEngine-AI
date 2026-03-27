from pathlib import Path

from src.graph.graph import (
    _build_contract_operational_dependency_mismatch,
    _resolve_required_input_columns,
    check_engineer_success,
    check_execution_status,
)
from src.utils.contract_validator import collect_contract_operational_dependency_columns


def _sample_contract() -> dict:
    return {
        "canonical_columns": [
            "1stYearAmount",
            "Size",
            "CurrentPhase",
            "Validation",
            "DateOfClose",
        ],
        "column_roles": {
            "pre_decision": ["Size"],
            "outcome": ["1stYearAmount"],
            "identifiers": [],
            "time_columns": ["DateOfClose"],
        },
        "task_semantics": {
            "problem_family": "regression",
            "objective_type": "regression",
            "primary_target": "1stYearAmount",
            "target_columns": ["1stYearAmount"],
            "training_row_filter": "CurrentPhase == 'Contract' AND Validation == 'Valid'",
        },
        "evaluation_spec": {
            "objective_type": "regression",
            "primary_target": "1stYearAmount",
            "split_column": "DateOfClose",
        },
        "artifact_requirements": {
            "cleaned_dataset": {
                "required_columns": ["1stYearAmount", "Size", "DateOfClose"],
            }
        },
    }


def test_collect_contract_operational_dependency_columns_extracts_filters_and_split():
    deps = collect_contract_operational_dependency_columns(_sample_contract())

    assert deps["task_semantics_filters"] == ["CurrentPhase", "Validation"]
    assert deps["split_columns"] == ["DateOfClose"]
    assert deps["all"] == ["CurrentPhase", "Validation", "DateOfClose"]


def test_resolve_required_input_columns_supports_cleaned_dataset_alias():
    required = _resolve_required_input_columns(
        {"artifact_requirements": {"cleaned_dataset": {"required_columns": ["feature_a", "target"]}}},
        {"required_columns": ["fallback"]},
    )

    assert required == ["feature_a", "target"]


def test_contract_operational_dependency_mismatch_requires_missing_and_uncovered():
    report = _build_contract_operational_dependency_mismatch(
        _sample_contract(),
        cleaned_header=["1stYearAmount", "Size", "DateOfClose"],
    )

    assert report["should_replan"] is True
    assert report["actionable_dependencies"] == ["CurrentPhase", "Validation"]


def test_contract_operational_dependency_mismatch_avoids_false_positive_when_passthrough_covers_columns():
    contract = _sample_contract()
    contract["artifact_requirements"]["cleaned_dataset"]["optional_passthrough_columns"] = [
        "CurrentPhase",
        "Validation",
    ]

    report = _build_contract_operational_dependency_mismatch(
        contract,
        cleaned_header=["1stYearAmount", "Size", "DateOfClose"],
    )

    assert report["should_replan"] is False
    assert report["uncovered_by_contract"] == []


def test_check_engineer_success_can_route_back_to_planner():
    state = {
        "error_message": "PLANNER_CONTRACT_MISMATCH",
        "planner_repair_required": True,
    }

    assert check_engineer_success(state) == "replan"


def test_check_execution_status_replans_on_runtime_contract_mismatch(tmp_path: Path):
    cleaned_path = tmp_path / "dataset_cleaned.csv"
    cleaned_path.write_text("1stYearAmount,Size,DateOfClose\n100,SMB,2025-01-01\n", encoding="utf-8")

    state = {
        "execution_output": (
            "Traceback (most recent call last):\n"
            "ValueError: Missing required input columns: ['CurrentPhase', 'Validation']\n"
        ),
        "ml_data_path": str(cleaned_path),
        "csv_sep": ",",
        "csv_encoding": "utf-8",
        "execution_contract": _sample_contract(),
    }

    assert check_execution_status(state) == "replan_contract"
    assert state["planner_repair_required"] is True
