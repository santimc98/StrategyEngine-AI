import json
import os

from src.utils.governance import build_run_summary


def test_build_run_summary_tracks_metric_improvement_with_artifact_metric(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)

    with open("data/review_board_verdict.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "metric_round_finalization": {
                    "metric_name": "cv_roc_auc",
                    "kept": "improved",
                    "baseline_metric": 0.8011,
                    "candidate_metric": 0.8030,
                    "final_metric": 0.8030,
                    "force_finalize_reason": "",
                }
            },
            handle,
            indent=2,
        )

    with open("data/metrics.json", "w", encoding="utf-8") as handle:
        json.dump({"cv_roc_auc": 0.8030}, handle, indent=2)

    summary = build_run_summary({"review_verdict": "APPROVED", "ml_improvement_kept": "improved"})
    metric_improvement = summary.get("metric_improvement", {})

    assert metric_improvement.get("kept") == "improved"
    assert metric_improvement.get("metric_name") == "cv_roc_auc"
    assert metric_improvement.get("final_metric_artifact") == 0.803
