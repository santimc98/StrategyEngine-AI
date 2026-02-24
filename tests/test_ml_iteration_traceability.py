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
    assert any(item.startswith("3:review_complete") for item in written_ids)

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


def test_append_ml_iteration_journal_keeps_multiple_metric_rounds_same_iteration(tmp_path) -> None:
    run_id = "trace_rounds_same_iteration"
    run_dir = init_run_bundle(
        run_id,
        state={},
        base_dir=str(tmp_path / "runs"),
        enable_tee=False,
    )
    entry_r1 = {
        "iteration_id": 3,
        "stage": "review_complete",
        "reviewer_verdict": "APPROVED",
        "qa_verdict": "APPROVED",
        "outputs_missing": [],
        "metric_round": {
            "round_id": 1,
            "action": "APPLY",
            "technique": "missing_indicators",
            "signature": "hyp_r1",
            "delta": -0.0001,
            "kept": "baseline",
            "reason": "delta_below_threshold",
        },
    }
    entry_r2 = {
        "iteration_id": 3,
        "stage": "review_complete",
        "reviewer_verdict": "APPROVED",
        "qa_verdict": "APPROVED",
        "outputs_missing": [],
        "metric_round": {
            "round_id": 2,
            "action": "APPLY",
            "technique": "rare_category_grouping",
            "signature": "hyp_r2",
            "delta": 0.0007,
            "kept": "improved",
            "reason": "candidate_selected",
        },
    }
    written_ids = graph_mod._append_ml_iteration_journal(run_id, entry_r1, [])
    written_ids = graph_mod._append_ml_iteration_journal(run_id, entry_r2, written_ids)

    journal_path = Path(run_dir) / "report" / "governance" / "ml_iteration_journal.jsonl"
    rows = [line for line in journal_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 2
    assert "3:review_complete:r1" in written_ids
    assert "3:review_complete:r2" in written_ids

    summary_path = Path(run_dir) / "report" / "governance" / "ml_iteration_trace_summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload.get("metric_rounds_count") == 2
    rounds = payload.get("metric_rounds") if isinstance(payload.get("metric_rounds"), list) else []
    assert [item.get("round_id") for item in rounds] == [1, 2]


def test_refresh_ml_iteration_trace_summary_fills_missing_rounds_from_state(tmp_path) -> None:
    run_id = "trace_round_summary_refresh"
    run_dir = init_run_bundle(
        run_id,
        state={},
        base_dir=str(tmp_path / "runs"),
        enable_tee=False,
    )
    entry = {
        "iteration_id": 2,
        "stage": "review_complete",
        "reviewer_verdict": "APPROVED",
        "qa_verdict": "APPROVED",
        "outputs_missing": [],
        "metric_round": {
            "round_id": 1,
            "action": "APPLY",
            "technique": "missing_indicators",
            "signature": "hyp_r1",
            "delta": -0.0001,
            "kept": "baseline",
            "reason": "baseline_restored",
        },
    }
    graph_mod._append_ml_iteration_journal(run_id, entry, [])

    state = {
        "iteration_count": 1,
        "review_verdict": "APPROVED",
        "qa_last_result": {"status": "APPROVED"},
        "ml_improvement_round_history": [
            {
                "round_id": 1,
                "delta": -0.0001,
                "kept": "baseline",
                "reason": "baseline_restored",
                "hypothesis": {"action": "APPLY", "technique": "missing_indicators", "signature": "hyp_r1"},
            },
            {
                "round_id": 2,
                "delta": 0.0008,
                "kept": "improved",
                "reason": "candidate_selected",
                "hypothesis": {"action": "APPLY", "technique": "rare_category_grouping", "signature": "hyp_r2"},
            },
        ],
    }
    graph_mod._refresh_ml_iteration_trace_summary(run_id, state)

    summary_path = Path(run_dir) / "report" / "governance" / "ml_iteration_trace_summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload.get("metric_rounds_count") == 2
    rounds = payload.get("metric_rounds") if isinstance(payload.get("metric_rounds"), list) else []
    assert [item.get("round_id") for item in rounds] == [1, 2]
