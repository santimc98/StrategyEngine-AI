from src.utils.cleaning_contract_semantics import expand_required_feature_selectors


def test_expand_required_feature_selectors_supports_all_numeric_except_value_alias():
    columns = ["event_id", "__split", "label_12h", "feature_a", "feature_b"]

    expanded, issues = expand_required_feature_selectors(
        [
            {
                "type": "all_numeric_except",
                "value": ["event_id", "__split", "label_12h"],
            }
        ],
        columns,
    )

    assert issues == []
    assert expanded == ["feature_a", "feature_b"]
