from src.agents.execution_planner import build_execution_plan
from src.agents.results_advisor import ResultsAdvisorAgent
from src.utils.contract_views import _resolve_objective_type
from src.utils.problem_capabilities import (
    infer_problem_capabilities,
    metric_family_for_metric,
    metric_preference_tokens,
    problem_metric_families,
    resolve_problem_capabilities_from_contract,
)


def test_infer_problem_capabilities_detects_survival_from_problem_type():
    caps = infer_problem_capabilities(
        problem_type="survival_analysis",
        evaluation_spec={
            "survival_time_col": "time_to_hit_hours",
            "survival_event_col": "event",
        },
        validation_requirements={
            "primary_metric": "concordance_index",
            "metrics_to_report": ["integrated_brier_score"],
        },
    )

    assert caps["family"] == "survival_analysis"
    assert caps["metric_family"] == "survival"
    assert caps["output_mode"] == "risk_scores"


def test_resolve_problem_capabilities_from_contract_uses_metrics_and_outputs():
    contract = {
        "business_objective": "Estimate time-to-event under censoring",
        "evaluation_spec": {"objective_type": "predictive"},
        "validation_requirements": {
            "primary_metric": "concordance_index",
            "metrics_to_report": ["integrated_brier_score", "mae_uncensored"],
        },
        "required_outputs": ["data/metrics.json", "data/scored_rows.csv"],
    }

    caps = resolve_problem_capabilities_from_contract(contract)

    assert caps["family"] == "survival_analysis"


def test_build_execution_plan_adds_survival_outputs():
    plan = build_execution_plan("survival analysis / time-to-event prediction", {})
    output_types = {item["artifact_type"] for item in plan["outputs"]}

    assert plan["objective_type"] == "survival_analysis"
    assert "predictions" in output_types
    assert "calibration" in output_types


def test_contract_views_resolve_objective_type_uses_capabilities():
    contract_full = {
        "business_objective": "Model survival risk over time",
        "evaluation_spec": {"objective_type": "predictive"},
        "validation_requirements": {"primary_metric": "concordance_index"},
        "required_outputs": ["data/metrics.json", "data/scored_rows.csv"],
    }

    objective_type = _resolve_objective_type({}, contract_full, contract_full["required_outputs"])

    assert objective_type == "survival_analysis"


def test_contract_views_output_only_fallback_uses_capability_family():
    objective_type = _resolve_objective_type({}, {}, ["data/cluster_assignments.csv"])

    assert objective_type == "clustering"


def test_results_advisor_metric_priority_supports_survival():
    agent = ResultsAdvisorAgent.__new__(ResultsAdvisorAgent)

    priority = agent._objective_metric_priority("survival_analysis")

    assert priority[0] == "concordance_index"
    assert "integrated_brier_score" in priority


def test_metric_family_helpers_cover_survival_and_optimization():
    assert metric_family_for_metric("concordance_index") == "survival"
    assert metric_family_for_metric("objective_value") == "optimization"
    assert "regression" in problem_metric_families("forecasting")
    assert "concordance_index" in metric_preference_tokens("survival")
