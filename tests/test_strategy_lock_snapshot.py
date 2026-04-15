from src.graph import graph as graph_mod


def _state_with_feature_families(feature_families):
    return {
        "selected_strategy": {
            "id": "temporal_churn",
            "title": "Temporal Churn-60d Prioritization Pipeline",
        },
        "execution_contract": {
            "contract_version": "4.1",
            "canonical_columns": ["login_days_14d", "account_id", "arr_current"],
            "decision_columns": ["account_id"],
            "outcome_columns": ["churn_60d"],
            "allowed_feature_sets": {
                "model_features": ["login_days_14d", "arr_current"],
                "forbidden_features": ["days_to_churn", "churn_reason"],
                "audit_only_features": ["final_account_status"],
                "feature_families": feature_families,
            },
        },
    }


def test_strategy_snapshot_handles_rich_feature_family_metadata() -> None:
    state = _state_with_feature_families(
        [
            {
                "family": "usage_engagement",
                "columns": ["login_days_14d", "product_adoption_score"],
            },
            {
                "family": "financial_health",
                "columns": ["arr_current", "invoice_overdue_days"],
            },
        ]
    )

    snapshot = graph_mod._capture_strategy_snapshot(state)

    assert snapshot["forbidden_features"] == ["churn_reason", "days_to_churn"]
    assert snapshot["allowed_feature_sets"]["model_features"] == ["arr_current", "login_days_14d"]
    assert len(snapshot["allowed_feature_sets"]["feature_families"]) == 2


def test_strategy_snapshot_normalizes_feature_families_without_false_drift() -> None:
    state_a = _state_with_feature_families(
        [
            {"family": "usage_engagement", "columns": ["product_adoption_score", "login_days_14d"]},
            {"family": "financial_health", "columns": ["invoice_overdue_days", "arr_current"]},
        ]
    )
    state_b = _state_with_feature_families(
        [
            {"family": "financial_health", "columns": ["arr_current", "invoice_overdue_days"]},
            {"family": "usage_engagement", "columns": ["login_days_14d", "product_adoption_score"]},
        ]
    )

    snapshot = graph_mod._capture_strategy_snapshot(state_a)
    ok, details = graph_mod._validate_strategy_lock(
        {**state_b, "strategy_lock_snapshot": snapshot}
    )

    assert ok is True
    assert details["reason"] == "no_drift"


def test_derive_forbidden_supports_contract_forbidden_features_key() -> None:
    allowed = {
        "forbidden_features": ["future_outcome", "leakage_score"],
        "feature_families": [{"family": "safe", "columns": ["arr_current"]}],
    }

    assert graph_mod._derive_forbidden_from_allowed(allowed, []) == [
        "future_outcome",
        "leakage_score",
    ]
