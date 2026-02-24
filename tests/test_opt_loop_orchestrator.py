from src.graph import graph as graph_mod
from src.utils.metric_eval import select_incumbent


def test_check_review_board_opt_route_bootstraps_when_round_not_active() -> None:
    state = {
        "ml_improvement_round_active": False,
        "review_verdict": "APPROVED",
    }
    assert graph_mod.check_review_board_opt_route(state) == "bootstrap_opt_round"


def test_check_review_board_opt_route_full_eval_when_round_active() -> None:
    state = {
        "ml_improvement_round_active": True,
        "review_verdict": "APPROVED",
    }
    assert graph_mod.check_review_board_opt_route(state) == "run_full_eval"


def test_check_finalize_opt_loop_route_respects_round_budget_guard() -> None:
    state = {
        "iteration_count": 3,
        "review_verdict": "APPROVED",
        "ml_improvement_continue": True,
        "ml_improvement_round_active": False,
        "ml_improvement_round_count": 2,
        "ml_improvement_rounds_allowed": 2,
        "ml_improvement_patience": 2,
        "ml_improvement_no_improve_streak": 0,
        "primary_metric_snapshot": {
            "primary_metric_name": "roc_auc",
            "primary_metric_value": 0.8,
            "baseline_value": 0.8,
        },
    }
    route = graph_mod.check_finalize_opt_loop_route(state)
    assert route != "bootstrap_opt_round"


def test_check_finalize_opt_loop_route_continues_within_budget() -> None:
    state = {
        "iteration_count": 3,
        "review_verdict": "APPROVED",
        "ml_improvement_continue": True,
        "ml_improvement_round_active": False,
        "ml_improvement_round_count": 1,
        "ml_improvement_rounds_allowed": 3,
        "ml_improvement_patience": 2,
        "ml_improvement_no_improve_streak": 0,
        "primary_metric_snapshot": {
            "primary_metric_name": "roc_auc",
            "primary_metric_value": 0.8,
            "baseline_value": 0.8,
        },
    }
    assert graph_mod.check_finalize_opt_loop_route(state) == "bootstrap_opt_round"


def test_select_incumbent_is_deterministic_on_ties() -> None:
    candidates = [
        {
            "label": "candidate_b",
            "metric_value": 0.801,
            "stability_ok": True,
            "cv_std": 0.01,
            "generalization_gap_abs": 0.005,
            "cost": 2.0,
        },
        {
            "label": "candidate_a",
            "metric_value": 0.801,
            "stability_ok": True,
            "cv_std": 0.01,
            "generalization_gap_abs": 0.005,
            "cost": 2.0,
        },
    ]
    first = select_incumbent(candidates, higher_is_better=True, min_delta=0.0)
    second = select_incumbent(candidates, higher_is_better=True, min_delta=0.0)
    assert first.get("selected_label") == second.get("selected_label") == "candidate_a"

