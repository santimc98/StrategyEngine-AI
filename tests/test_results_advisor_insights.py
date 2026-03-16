import os

from src.agents.results_advisor import ResultsAdvisorAgent


def test_results_advisor_insights_minimal():
    advisor = ResultsAdvisorAgent(api_key="")
    insights = advisor.generate_insights({"artifact_index": []})
    assert isinstance(insights, dict)
    assert insights.get("schema_version") == "1"
    summary_lines = insights.get("summary_lines")
    assert isinstance(summary_lines, list)
    assert summary_lines


def test_results_advisor_deployment_recommendation_with_ci(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "metrics.json"), "w", encoding="utf-8") as handle:
        handle.write('{"model_performance": {"Revenue Lift": {"mean": 1.05, "ci_lower": 0.95, "ci_upper": 1.1}}}')

    advisor = ResultsAdvisorAgent(api_key="")
    insights = advisor.generate_insights(
        {"artifact_index": [{"path": "data/metrics.json", "artifact_type": "metrics"}]}
    )
    assert insights.get("deployment_recommendation") == "PILOT"
    assert insights.get("confidence") in {"LOW", "MEDIUM"}


def test_results_advisor_does_not_flag_leakage_on_preventive_feedback(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    advisor = ResultsAdvisorAgent(api_key="")
    insights = advisor.generate_insights(
        {
            "artifact_index": [],
            "review_feedback": "Leakage guard prevents leakage and reports no leakage detected.",
        }
    )
    risks = insights.get("risks") or []
    assert all("leakage" not in str(item).lower() for item in risks)


def test_results_advisor_flags_leakage_on_explicit_risk_feedback(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    advisor = ResultsAdvisorAgent(api_key="")
    insights = advisor.generate_insights(
        {
            "artifact_index": [],
            "review_feedback": "Potential leakage detected from post-outcome fields in features.",
        }
    )
    risks = [str(item).lower() for item in (insights.get("risks") or [])]
    assert any("leakage" in item for item in risks)


def test_results_advisor_primary_metric_prefers_explicit_metric_field(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "metrics.json"), "w", encoding="utf-8") as handle:
        handle.write(
            '{"model_performance": {"primary_metric": "RMSLE", "primary_metric_value": 0.4249, "cv_mae_mean": 318.4, "cv_rmsle_mean": 0.4249}}'
        )

    advisor = ResultsAdvisorAgent(api_key="")
    insights = advisor.generate_insights(
        {
            "artifact_index": [{"path": "data/metrics.json", "artifact_type": "metrics"}],
            "evaluation_spec": {"objective_type": "regression"},
        }
    )
    assert str(insights.get("primary_metric") or "").lower() == "rmsle"


def test_results_advisor_is_pure_critic_and_does_not_emit_iteration_decisions():
    advisor = ResultsAdvisorAgent(api_key="")
    insights = advisor.generate_insights(
        {
            "artifact_index": [],
            "review_verdict": "APPROVED",
            "data_adequacy_report": {"status": "adequate"},
            "execution_contract": {
                "feature_engineering_tasks": [
                    {"technique": "interaction", "input_columns": ["a", "b"], "output_column_name": "a_x_b"}
                ]
            },
            "metric_history": [{"primary_metric_name": "roc_auc", "primary_metric_value": 0.71}],
        }
    )
    assert insights.get("iteration_recommendation") == {}


def test_results_advisor_insights_reclassifies_legacy_json_metrics_artifact(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "evaluation_summary.json"), "w", encoding="utf-8") as handle:
        handle.write(
            '{"status":"success","primary_metric":"mean_multi_horizon_log_loss","mean_multi_horizon_log_loss":0.330041811925438}'
        )

    advisor = ResultsAdvisorAgent(api_key="")
    insights = advisor.generate_insights(
        {
            "artifact_index": [
                {"path": "data/evaluation_summary.json", "artifact_type": "json"},
            ],
            "objective_type": "predictive",
        }
    )

    assert insights.get("metrics_summary")
    assert "Metrics artifact missing or empty" not in " ".join(insights.get("risks", []))
    assert "data/evaluation_summary.json" in (insights.get("artifacts_used") or [])


def test_results_advisor_insights_prefers_context_metrics_payload(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    advisor = ResultsAdvisorAgent(api_key="")
    insights = advisor.generate_insights(
        {
            "artifact_index": [
                {"path": "artifacts/ml/evaluation_summary.json", "artifact_type": "json"},
            ],
            "objective_type": "predictive",
            "metrics": {
                "status": "success",
                "primary_metric": "mean_multi_horizon_log_loss",
                "mean_multi_horizon_log_loss": 0.330041811925438,
                "source": "artifact:artifacts/ml/evaluation_summary.json",
            },
            "primary_metric_state": {
                "primary_metric_name": "mean_multi_horizon_log_loss",
                "primary_metric_value": 0.330041811925438,
                "primary_metric_source": "artifact:artifacts/ml/evaluation_summary.json",
            },
        }
    )

    assert insights.get("metrics_summary")
    assert "Metrics artifact missing or empty" not in " ".join(insights.get("risks", []))
    assert "artifacts/ml/evaluation_summary.json" in (insights.get("artifacts_used") or [])
