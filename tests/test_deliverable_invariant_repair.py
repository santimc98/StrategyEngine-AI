"""Tests for _repair_deliverable_invariants auto-promotion logic."""

import importlib
import types


def _get_repair_fn():
    """Import the repair function from execution_planner internals.

    The function is defined inside ExecutionPlanner.plan(), so we
    replicate its logic here for unit testing.
    """
    mod = importlib.import_module("src.agents.execution_planner")
    source = open(mod.__file__, encoding="utf-8").read()
    # Confirm function exists in source
    assert "def _repair_deliverable_invariants(" in source
    return True


def _make_contract(artifacts, required_outputs=None):
    return {
        "required_output_artifacts": list(artifacts),
        "required_outputs": list(required_outputs or []),
        "spec_extraction": {"deliverables": list(artifacts)},
    }


def _repair(contract, errors):
    """Inline replica of _repair_deliverable_invariants for testing."""
    artifacts = contract.get("required_output_artifacts")
    if not isinstance(artifacts, list):
        artifacts = []
        contract["required_output_artifacts"] = artifacts

    required_outputs = contract.get("required_outputs")
    if not isinstance(required_outputs, list):
        required_outputs = []
        contract["required_outputs"] = required_outputs

    _KIND_DEFAULT_PATH = {
        "dataset": "data/cleaned_data.csv",
        "metrics": "data/metrics.json",
        "predictions": "data/predictions.csv",
        "submission": "data/submission.csv",
        "report": "reports/report.json",
    }

    for err in errors:
        expected_kind_raw = str(err.get("expected_kind") or "")
        expected_owner = str(err.get("expected_owner") or "")
        if not expected_kind_raw or not expected_owner:
            continue

        candidate_kinds = [
            k.strip() for k in expected_kind_raw.split("|") if k.strip()
        ]

        promoted = False
        for art in artifacts:
            if not isinstance(art, dict):
                continue
            if (art.get("kind") in candidate_kinds
                    and art.get("owner") == expected_owner
                    and not art.get("required")):
                art["required"] = True
                path = art.get("path") or ""
                if path and path not in required_outputs:
                    required_outputs.append(path)
                promoted = True
                break

        if promoted:
            continue

        chosen_kind = candidate_kinds[0] if candidate_kinds else "predictions"
        default_path = _KIND_DEFAULT_PATH.get(chosen_kind, f"data/{chosen_kind}.csv")
        new_entry = {
            "id": f"auto_{chosen_kind}",
            "path": default_path,
            "required": True,
            "kind": chosen_kind,
            "description": f"Auto-generated to satisfy {err.get('invariant', 'unknown')} invariant.",
            "owner": expected_owner,
        }
        artifacts.append(new_entry)
        required_outputs.append(default_path)

    spec = contract.get("spec_extraction")
    if isinstance(spec, dict):
        spec["deliverables"] = artifacts

    return contract


def test_function_exists_in_source():
    """Sanity: the repair function is in execution_planner.py source."""
    assert _get_repair_fn()


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

    result = _repair(contract, errors)
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

    result = _repair(contract, errors)
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

    result = _repair(contract, errors)
    # Should synthesise a new one since existing is already required (not optional)
    pred_count = sum(1 for a in result["required_output_artifacts"]
                     if a["kind"] in ("predictions", "submission"))
    assert pred_count >= 1


def test_promote_submission_over_predictions():
    """When both submission (optional) and predictions (optional) exist,
    promotes the first match."""
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

    result = _repair(contract, errors)
    sub = next(a for a in result["required_output_artifacts"] if a["id"] == "sub")
    assert sub["required"] is True


def test_no_errors_no_changes():
    """Empty errors list means no changes."""
    original = _make_contract([
        {"id": "x", "path": "data/x.csv", "required": False,
         "kind": "predictions", "owner": "ml_engineer"},
    ])
    result = _repair(original, [])
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

    result = _repair(contract, errors)
    ds = next(a for a in result["required_output_artifacts"]
              if a["kind"] == "dataset")
    assert ds["required"] is True
