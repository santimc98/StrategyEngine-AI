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


def test_finalize_route_allows_metric_round_continue_even_when_total_limit_reached() -> None:
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

    route = graph_mod.check_finalize_metric_improvement_route(state)
    assert route == "bootstrap_improvement_round"


def test_check_evaluation_uses_expensive_cycle_count_for_total_budget() -> None:
    state = {
        "iteration_count": 1,
        "expensive_cycle_count": 5,
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


def test_check_execution_status_stops_on_repeated_runtime_root_cause() -> None:
    state = {
        "execution_output": "Traceback (most recent call last)\nValueError: bad split",
        "attempt_ledger": [
            {
                "phase": "runtime",
                "outcome": "retry_fix",
                "root_cause": "valueerror: bad split",
                "source": "prepare_runtime_fix",
                "execution_attempt": 1,
            }
        ],
        "runtime_fix_count": 1,
        "feedback_history": [],
    }
    route = graph_mod.check_execution_status(state)
    assert route == "failed_runtime"
    assert state.get("runtime_fix_terminal") is True
    assert state.get("runtime_fix_terminal_reason") == "repeated_root_cause"


def test_check_evaluation_stops_on_repeated_compliance_root_cause() -> None:
    state = {
        "review_verdict": "NEEDS_IMPROVEMENT",
        "last_iteration_type": "compliance",
        "feedback_history": [],
        "last_gate_context": {
            "failed_gates": ["strategy_followed"],
            "required_fixes": ["Replace baseline model family."],
        },
        "attempt_ledger": [
            {
                "phase": "compliance",
                "outcome": "needs_improvement",
                "root_cause": "failed_gate:strategy_followed",
                "source": "run_result_evaluator",
                "execution_attempt": 1,
            },
            {
                "phase": "compliance",
                "outcome": "needs_improvement",
                "root_cause": "failed_gate:strategy_followed",
                "source": "run_result_evaluator",
                "execution_attempt": 2,
            },
        ],
        "execution_attempt": 3,
        "primary_metric_snapshot": {
            "primary_metric_name": "roc_auc",
            "primary_metric_value": 0.71,
            "baseline_value": 0.72,
        },
    }
    route = graph_mod.check_evaluation(state)
    assert route == "approved"
    assert state.get("stop_reason") == "ROOT_CAUSE_REPEAT"
