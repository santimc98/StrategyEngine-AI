import os

from src.agents.results_advisor import ResultsAdvisorAgent


class _CaptureChatCompletions:
    def __init__(self, *, should_raise: Exception | None = None):
        self.should_raise = should_raise
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.should_raise:
            raise self.should_raise

        class _Msg:
            content = (
                '{"packet_type":"advisor_critique_packet","packet_version":"1.0",'
                '"run_id":"r1","iteration":1,"timestamp_utc":"2026-03-28T00:00:00Z",'
                '"primary_metric_name":"mae","higher_is_better":false,'
                '"metric_comparison":{"baseline_value":1.0,"candidate_value":0.9,"delta_abs":-0.1,"delta_rel":-0.1,'
                '"min_delta_required":0.0005,"meets_min_delta":true},'
                '"validation_signals":{"validation_mode":"unknown"},'
                '"error_modes":[],"risk_flags":[],"active_gates_context":[],'
                '"analysis_summary":"Candidate meets min delta.","strictly_no_code_advice":true}'
            )

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        return _Resp()


class _CaptureClient:
    def __init__(self, *, should_raise: Exception | None = None):
        self.chat = type("Chat", (), {"completions": _CaptureChatCompletions(should_raise=should_raise)})()


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


def test_results_advisor_critique_openrouter_call_uses_configured_max_tokens():
    advisor = ResultsAdvisorAgent(api_key="")
    advisor.critique_mode = "llm"
    advisor.critique_provider = "openrouter"
    advisor.critique_model_name = "openai/gpt-5.4"
    advisor._generation_config["max_tokens"] = 8192
    client = _CaptureClient()
    advisor.critique_client = client

    packet = advisor.generate_critique_packet(
        {
            "run_id": "r1",
            "iteration": 1,
            "phase": "candidate_review",
            "primary_metric_name": "mae",
            "higher_is_better": False,
            "min_delta": 0.0005,
            "baseline_metrics": {"primary_metric_value": 1.0},
            "candidate_metrics": {"primary_metric_value": 0.9},
        }
    )

    calls = client.chat.completions.calls
    assert packet.get("packet_type") == "advisor_critique_packet"
    assert calls
    assert calls[0]["max_tokens"] == 8192


def test_results_advisor_transport_failure_does_not_report_schema_errors():
    advisor = ResultsAdvisorAgent(api_key="")
    advisor.critique_mode = "llm"
    advisor.critique_provider = "openrouter"
    advisor.critique_model_name = "openai/gpt-5.4"
    advisor._generation_config["max_tokens"] = 8192
    advisor.critique_client = _CaptureClient(should_raise=RuntimeError("402 credits exceeded"))

    packet = advisor.generate_critique_packet(
        {
            "run_id": "r1",
            "iteration": 1,
            "phase": "candidate_review",
            "primary_metric_name": "mae",
            "higher_is_better": False,
            "min_delta": 0.0005,
            "baseline_metrics": {"primary_metric_value": 1.0},
            "candidate_metrics": {"primary_metric_value": 0.9},
        }
    )

    assert packet == {}
    assert advisor.last_critique_meta.get("source") == "llm_transport_failure"
    assert "validation_errors" not in advisor.last_critique_meta
    assert advisor.last_critique_meta.get("llm_error", {}).get("stage") == "api_call_exception"
