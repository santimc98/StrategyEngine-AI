from src.graph import graph as graph_mod


def test_resolve_total_iteration_limit_defaults_to_legacy_cap_without_policy() -> None:
    state = {}
    limit = graph_mod._resolve_total_iteration_limit(state, None, {})
    assert limit == 6


def test_resolve_total_iteration_limit_respects_explicit_total_cap() -> None:
    state = {}
    contract = {
        "iteration_policy": {
            "total_iteration_max": 4,
            "compliance_bootstrap_max": 10,
            "metric_improvement_max": 10,
            "runtime_fix_max": 10,
        }
    }
    policy = graph_mod._get_iteration_policy({"execution_contract": contract})
    assert policy is not None
    limit = graph_mod._resolve_total_iteration_limit(state, policy, contract)
    assert limit == 4


def test_resolve_total_iteration_limit_derives_from_phase_budgets() -> None:
    state = {"max_runtime_fix_attempts": 2}
    contract = {
        "iteration_policy": {
            "compliance_bootstrap_max": 2,
            "metric_improvement_rounds": 1,
            "runtime_fix_max": 2,
        }
    }
    policy = graph_mod._get_iteration_policy({"execution_contract": contract})
    assert policy is not None
    limit = graph_mod._resolve_total_iteration_limit(state, policy, contract)
    assert limit == 5


def test_check_evaluation_stops_when_total_iteration_budget_reached() -> None:
    state = {
        "iteration_count": 5,
        "review_verdict": "NEEDS_IMPROVEMENT",
        "execution_contract": {
            "iteration_policy": {
                "compliance_bootstrap_max": 2,
                "metric_improvement_rounds": 1,
                "runtime_fix_max": 2,
            }
        },
        "primary_metric_snapshot": {
            "primary_metric_name": "roc_auc",
            "primary_metric_value": 0.71,
            "baseline_value": 0.72,
        },
    }
    route = graph_mod.check_evaluation(state)
    assert route == "approved"
    assert state.get("stop_reason") == "BUDGET"


def test_check_evaluation_allows_metric_round_continue_even_when_total_limit_reached(monkeypatch) -> None:
    def _fake_bootstrap(state, contract):
        state["ml_improvement_round_active"] = True
        state["ml_improvement_round_count"] = 2
        state["ml_improvement_rounds_allowed"] = 3
        return True

    monkeypatch.setattr(graph_mod, "_bootstrap_metric_improvement_round", _fake_bootstrap)

    state = {
        "iteration_count": 5,
        "review_verdict": "APPROVED",
        "metric_improvement_nodes_managed": False,
        "ml_improvement_continue": True,
        "ml_improvement_round_active": False,
        "execution_contract": {
            "iteration_policy": {
                "compliance_bootstrap_max": 2,
                "metric_improvement_rounds": 1,
                "runtime_fix_max": 2,
            }
        },
        "primary_metric_snapshot": {
            "primary_metric_name": "roc_auc",
            "primary_metric_value": 0.71,
            "baseline_value": 0.72,
        },
    }

    route = graph_mod.check_evaluation(state)
    assert route == "retry"
