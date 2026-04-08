from src.utils.review_context_packets import build_review_context_packet


def test_build_review_context_packet_surfaces_restored_risks_and_code_lines() -> None:
    code = "\n".join(
        [
            "if int(holdout_mask.sum()) < 400:",
            "    raise ValueError('holdout too small')",
            "scored_output['churn_risk_score'] = baseline_model.predict_proba(X_score)[:, 1]",
        ]
    )
    packet = build_review_context_packet(
        code,
        [
            {
                "name": "temporal_validation_credibility",
                "severity": "HARD",
                "params": {"min_rows": 1000},
            },
            {
                "name": "scoring_output_primary_model",
                "severity": "HARD",
                "params": {"model": "primary_model"},
            },
        ],
        code_path_hint="artifacts/ml_engineer_last.py",
        context_blocks=[
            {
                "review_history_context": {
                    "best_attempt_restored_recently": True,
                    "feedback_history_tail": [
                        "BEST_ATTEMPT_RESTORED[result_evaluator]: restored attempt 2 as authoritative state after a later degraded execution."
                    ],
                    "last_gate_context": {
                        "failed_gates": ["temporal_validation_credibility"],
                        "required_fixes": [
                            "Do not lower the holdout credibility threshold below 1000 rows.",
                            "Use primary_model.predict_proba for scoring.",
                        ],
                    },
                }
            }
        ],
    )

    assert packet["active_hard_gates_summary"]
    assert packet["known_candidate_risks"]
    assert packet["known_restored_candidate_risks"]
    lines = packet["code_lines_of_interest"]
    assert any("baseline_model.predict_proba" in item["snippet"] for item in lines)
    assert any("< 400" in item["snippet"] for item in lines)
