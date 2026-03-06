from src.graph.graph import (
    _build_ml_iteration_journal_entry,
    _build_ml_iteration_memory_block,
)


def test_journal_entry_prioritizes_structured_patch_actions_over_noisy_reason_actions():
    state = {
        "generated_code": "print('baseline')\n",
        "iteration_count": 3,
    }

    entry = _build_ml_iteration_journal_entry(
        state=state,
        preflight_issues=[],
        runtime_error=None,
        outputs_present=["data/metrics.json"],
        outputs_missing=[],
        reviewer_verdict="APPROVE_WITH_WARNINGS",
        reviewer_reasons=["leakage"],
        qa_verdict="REJECTED",
        qa_reasons=["leakage"],
        next_actions=["Exclude post-outcome features and document leakage prevention."],
        stage="review_complete",
        reviewer_packet={
            "status": "REJECTED",
            "failed_gates": ["strategy_followed"],
            "required_fixes": ["Replace Logistic Regression with a boosting ensemble."],
        },
        qa_packet={"status": "APPROVE_WITH_WARNINGS"},
        iteration_handoff={
            "quality_focus": {
                "failed_gates": ["strategy_followed"],
                "required_fixes": [
                    "Replace Logistic Regression with a boosting ensemble.",
                    "Implement a Stacking ensemble.",
                ],
            },
            "patch_objectives": [
                "Resolve failed gates: strategy_followed",
                "Replace Logistic Regression with a boosting ensemble.",
                "Implement a Stacking ensemble.",
            ],
        },
    )

    assert entry["quality_failed_gates"] == ["strategy_followed"]
    assert entry["next_actions"][:2] == [
        "Replace Logistic Regression with a boosting ensemble.",
        "Implement a Stacking ensemble.",
    ]
    assert "Exclude post-outcome features and document leakage prevention." not in entry["next_actions"]


def test_iteration_memory_block_prefers_structured_blockers_and_actions():
    entries = [
        {
            "iteration_id": 1,
            "reviewer_verdict": "APPROVE_WITH_WARNINGS",
            "qa_verdict": "REJECTED",
            "reviewer_reasons": ["leakage"],
            "qa_reasons": ["alignment"],
            "quality_failed_gates": ["strategy_followed"],
            "quality_hard_failures": ["strategy_followed"],
            "next_actions": [
                "Replace Logistic Regression with a boosting ensemble.",
                "Implement a Stacking ensemble.",
            ],
            "iteration_diagnostics": {"n_rows": 254655},
        },
        {
            "iteration_id": 2,
            "reviewer_verdict": "APPROVE_WITH_WARNINGS",
            "qa_verdict": "REJECTED",
            "reviewer_reasons": ["leakage"],
            "qa_reasons": ["alignment"],
            "quality_failed_gates": ["strategy_followed"],
            "quality_hard_failures": ["strategy_followed"],
            "next_actions": [
                "Replace Logistic Regression with a boosting ensemble.",
                "Implement a Stacking ensemble.",
            ],
            "iteration_diagnostics": {"n_rows": 254655},
        },
    ]

    block = _build_ml_iteration_memory_block(entries, max_chars=1600)

    assert "quality_gate:strategy_followed" in block
    assert "Repeated blockers" in block
    assert "NEXT: Replace Logistic Regression with a boosting ensemble." in block
    assert "Exclude post-outcome features" not in block
    assert "leakage (x2)" not in block
