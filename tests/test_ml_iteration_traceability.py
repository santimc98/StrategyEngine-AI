import json
from pathlib import Path

from src.graph import graph as graph_mod
from src.utils.run_bundle import init_run_bundle


def test_append_ml_iteration_journal_uses_run_bundle_dir(tmp_path, monkeypatch) -> None:
    run_id = "trace_bundle_run"
    run_dir = init_run_bundle(
        run_id,
        state={},
        base_dir=str(tmp_path / "runs"),
        enable_tee=False,
    )

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(workspace)

    entry = {
        "iteration_id": 3,
        "stage": "review_complete",
        "reviewer_verdict": "NEEDS_IMPROVEMENT",
        "qa_verdict": "APPROVE_WITH_WARNINGS",
        "outputs_missing": [],
        "metric_round": {
            "round_id": 1,
            "action": "APPLY",
            "technique": "missing_indicators",
            "signature": "hyp_x",
            "delta": 0.0004,
            "kept": "baseline",
            "reason": "delta_below_threshold | baseline_restored",
        },
    }
    written_ids = graph_mod._append_ml_iteration_journal(run_id, entry, [])

    expected_journal = Path(run_dir) / "report" / "governance" / "ml_iteration_journal.jsonl"
    assert expected_journal.exists()
    assert "3:review_complete" in written_ids

    wrong_journal = workspace / "runs" / run_id / "report" / "governance" / "ml_iteration_journal.jsonl"
    assert not wrong_journal.exists()

    trace_summary = Path(run_dir) / "report" / "governance" / "ml_iteration_trace_summary.json"
    assert trace_summary.exists()
    payload = json.loads(trace_summary.read_text(encoding="utf-8"))
    assert payload.get("entries_count") == 1
    assert payload.get("stages_count", {}).get("review_complete") == 1
    assert payload.get("metric_rounds_count") == 1
    assert payload.get("metric_rounds", [{}])[0].get("technique") == "missing_indicators"

    events_path = Path(run_dir) / "events.jsonl"
    assert events_path.exists()
    events_text = events_path.read_text(encoding="utf-8")
    assert "ml_iteration_trace" in events_text
