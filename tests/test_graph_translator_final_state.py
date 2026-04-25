from src.graph.graph import _build_translator_state_delta


def test_build_translator_state_delta_promotes_canonical_summary_fields():
    state = {
        "run_id": "run12345",
        "review_verdict": "NEEDS_IMPROVEMENT",
        "run_outcome": "NO_GO",
        "ml_improvement_kept": "baseline",
        "review_board_verdict": {
            "status": "NEEDS_IMPROVEMENT",
            "final_review_verdict": "NEEDS_IMPROVEMENT",
        },
    }
    report_state = {
        "review_board_verdict": {
            "status": "APPROVED",
            "final_review_verdict": "APPROVED",
        },
        "final_report": "# Executive Report",
        "pdf_path": "C:/tmp/final_report.pdf",
        "primary_metric_name": "pct_corporations_within_1_level",
    }
    summary = {
        "status": "APPROVED",
        "run_outcome": "GO_WITH_LIMITATIONS",
        "overall_status_global": "ok",
        "hard_failures": [],
        "failed_gates": [],
        "metric_improvement": {"kept": "best_attempt"},
    }

    result = _build_translator_state_delta(state, report_state, summary)

    assert result["review_verdict"] == "APPROVED"
    assert result["run_outcome"] == "GO_WITH_LIMITATIONS"
    assert result["ml_improvement_kept"] == "best_attempt"
    assert result["review_board_verdict"]["status"] == "APPROVED"
    assert result["final_report"] == "# Executive Report"
    assert result["pdf_path"] == "C:/tmp/final_report.pdf"
    assert state["review_verdict"] == "NEEDS_IMPROVEMENT"
    assert report_state["review_board_verdict"]["status"] == "APPROVED"
