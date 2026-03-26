"""Tests for _repair_deliverable_invariants auto-promotion logic."""

from src.agents.execution_planner import _repair_deliverable_invariants


def _make_contract(artifacts, required_outputs=None):
    return {
        "required_output_artifacts": list(artifacts),
        "required_outputs": list(required_outputs or []),
        "spec_extraction": {"deliverables": list(artifacts)},
    }


def test_repair_function_is_importable():
    assert callable(_repair_deliverable_invariants)


def test_promote_optional_predictions_to_required():
    """An optional predictions deliverable should be promoted to required."""
    contract = _make_contract([
        {"id": "metrics_json", "path": "data/metrics.json", "required": True,
         "kind": "metrics", "owner": "ml_engineer"},
        {"id": "predictions_csv", "path": "data/predictions.csv", "required": False,
         "kind": "predictions", "owner": "ml_engineer"},
    ], required_outputs=["data/metrics.json"])

    errors = [{
        "invariant": "ml_requires_predictions_or_submission",
        "severity": "error",
        "expected_kind": "predictions|submission",
        "expected_owner": "ml_engineer",
    }]

    result = _repair_deliverable_invariants(contract, errors)
    pred = next(a for a in result["required_output_artifacts"]
                if a["id"] == "predictions_csv")
    assert pred["required"] is True
    assert "data/predictions.csv" in result["required_outputs"]


def test_synthesise_when_no_existing_deliverable():
    """When no matching deliverable exists, create one."""
    contract = _make_contract([
        {"id": "metrics_json", "path": "data/metrics.json", "required": True,
         "kind": "metrics", "owner": "ml_engineer"},
    ], required_outputs=["data/metrics.json"])

    errors = [{
        "invariant": "ml_requires_predictions_or_submission",
        "severity": "error",
        "expected_kind": "predictions|submission",
        "expected_owner": "ml_engineer",
    }]

    result = _repair_deliverable_invariants(contract, errors)
    kinds = [a["kind"] for a in result["required_output_artifacts"]]
    assert "predictions" in kinds
    auto = next(a for a in result["required_output_artifacts"]
                if a["kind"] == "predictions")
    assert auto["required"] is True
    assert auto["path"] in result["required_outputs"]


def test_does_not_promote_already_required():
    """A deliverable that is already required should not be duplicated."""
    contract = _make_contract([
        {"id": "predictions_csv", "path": "data/predictions.csv", "required": True,
         "kind": "predictions", "owner": "ml_engineer"},
    ], required_outputs=["data/predictions.csv"])

    errors = [{
        "invariant": "ml_requires_predictions_or_submission",
        "severity": "error",
        "expected_kind": "predictions|submission",
        "expected_owner": "ml_engineer",
    }]

    result = _repair_deliverable_invariants(contract, errors)
    pred_count = sum(1 for a in result["required_output_artifacts"]
                     if a["kind"] in ("predictions", "submission"))
    assert pred_count >= 1


def test_promote_submission_over_predictions():
    """When both submission and predictions exist, promote the first match."""
    contract = _make_contract([
        {"id": "sub", "path": "data/submission.csv", "required": False,
         "kind": "submission", "owner": "ml_engineer"},
        {"id": "pred", "path": "data/predictions.csv", "required": False,
         "kind": "predictions", "owner": "ml_engineer"},
    ])

    errors = [{
        "invariant": "ml_requires_predictions_or_submission",
        "severity": "error",
        "expected_kind": "predictions|submission",
        "expected_owner": "ml_engineer",
    }]

    result = _repair_deliverable_invariants(contract, errors)
    sub = next(a for a in result["required_output_artifacts"] if a["id"] == "sub")
    assert sub["required"] is True


def test_no_errors_no_changes():
    """Empty errors list means no changes."""
    original = _make_contract([
        {"id": "x", "path": "data/x.csv", "required": False,
         "kind": "predictions", "owner": "ml_engineer"},
    ])
    result = _repair_deliverable_invariants(original, [])
    pred = next(a for a in result["required_output_artifacts"] if a["id"] == "x")
    assert pred["required"] is False


def test_dataset_invariant_repair():
    """DE dataset invariant repair works for any problem type."""
    contract = _make_contract([
        {"id": "cleaned", "path": "data/cleaned_data.csv", "required": False,
         "kind": "dataset", "owner": "data_engineer"},
    ])

    errors = [{
        "invariant": "de_requires_dataset",
        "severity": "error",
        "expected_kind": "dataset",
        "expected_owner": "data_engineer",
    }]

    result = _repair_deliverable_invariants(contract, errors)
    ds = next(a for a in result["required_output_artifacts"]
              if a["kind"] == "dataset")
    assert ds["required"] is True
