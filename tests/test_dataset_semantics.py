from src.utils.dataset_semantics import summarize_dataset_semantics


def test_summarize_dataset_semantics_surfaces_target_validation_status():
    summary = summarize_dataset_semantics(
        {
            "primary_target": "score_label",
            "target_status": "invalid",
            "recommended_primary_target": "renewal_probability",
            "target_status_reason": "score_label is 96% missing and behaves like a post-decision audit field",
            "target_analysis": {
                "primary_target": "score_label",
                "target_null_frac_exact": 0.96,
                "target_missing_count_exact": 960,
                "target_total_count_exact": 1000,
            },
            "split_candidates": ["partition_flag"],
            "id_candidates": ["customer_id"],
        },
        {
            "training_rows_rule": "rows where score_label is not missing",
            "scoring_rows_rule_primary": "all rows",
        },
    )

    assert "target_status: invalid" in summary
    assert "recommended_primary_target: renewal_probability" in summary
    assert "target_status_reason: score_label is 96% missing" in summary
