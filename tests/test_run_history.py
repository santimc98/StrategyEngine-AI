import json
from pathlib import Path

from src.utils import run_history


def test_extract_run_summary_prefers_review_board_verdict_from_worker_final_state(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "run123"
    run_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(run_history, "_RUNS_DIR", str(runs_dir))

    (run_dir / "worker_status.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "started_at": 1,
                "metric_name": "Gini",
                "metric_value": "0.28",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "worker_final_state.json").write_text(
        json.dumps(
            {
                "review_verdict": "APPROVED",
                "review_board_verdict": {
                    "status": "REJECTED",
                    "final_review_verdict": "NEEDS_IMPROVEMENT",
                },
            }
        ),
        encoding="utf-8",
    )

    summary = run_history._extract_run_summary("run123", str(run_dir))

    assert summary is not None
    assert summary["verdict"] == "NEEDS_IMPROVEMENT"
