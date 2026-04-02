from src.utils.dataset_semantics import summarize_dataset_semantics, build_target_lineage_summary


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


def test_build_target_lineage_summary_preserves_preliminary_validated_and_final_targets():
    lineage = build_target_lineage_summary(
        {
            "summary": "## Target Variable Decision\n**Recommended primary target: `won_90d`**",
        },
        {
            "primary_target": "pipeline_amount_90d",
            "target_status": "questioned",
            "recommended_primary_target": "pipeline_amount_90d",
            "target_status_reason": "Zero inflation requires extra care.",
        },
        {
            "task_semantics": {
                "primary_target": "won_90d",
            }
        },
    )

    assert lineage["preliminary_steward_target"] == "won_90d"
    assert lineage["validated_steward_target"] == "pipeline_amount_90d"
    assert lineage["final_contract_target"] == "won_90d"
    assert lineage["preliminary_summary_conflicts_with_validated_semantics"] is True
    assert lineage["contract_differs_from_validated_steward"] is True
    assert lineage["contract_matches_preliminary_steward"] is True
    assert lineage["lineage_status"] == "diverged"
