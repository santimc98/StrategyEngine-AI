from src.utils.data_atlas import resolve_steward_target_reconsideration_candidate


def test_reconsideration_candidate_is_selected_for_invalid_target():
    result = resolve_steward_target_reconsideration_candidate(
        current_target="bad_target",
        steward_context_quality={
            "target_status": "invalid",
            "reasons": ["primary_target_invalid"],
            "recommended_primary_target": "good_target",
            "target_status_reason": "bad_target is 98% missing",
        },
        dataset_semantics={"primary_target": "bad_target"},
        header_cols=["bad_target", "good_target", "customer_id"],
    )

    assert result["should_retry"] is True
    assert result["candidate"] == "good_target"
    assert result["reason"] == "recommended_primary_target_available"


def test_reconsideration_candidate_is_skipped_when_target_is_not_invalid():
    result = resolve_steward_target_reconsideration_candidate(
        current_target="label",
        steward_context_quality={
            "target_status": "questioned",
            "reasons": [],
            "recommended_primary_target": "better_label",
            "target_status_reason": "label has moderate missingness",
        },
        dataset_semantics={"primary_target": "label"},
        header_cols=["label", "better_label"],
    )

    assert result["should_retry"] is False
    assert result["reason"] == "target_not_invalid"
