from src.agents.ml_engineer import MLEngineerAgent


class _FakeOpenAI:
    def __init__(self, api_key=None, base_url=None, timeout=None, default_headers=None):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.default_headers = default_headers


def test_editor_mode_prompt_includes_structured_feedback_json(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-openrouter")
    monkeypatch.setattr("src.agents.ml_engineer.OpenAI", _FakeOpenAI)

    def _fake_call_chat_with_fallback(client, messages, models, call_kwargs=None, logger=None, context_tag=None):
        return {"dummy": True}, models[0]

    monkeypatch.setattr("src.agents.ml_engineer.call_chat_with_fallback", _fake_call_chat_with_fallback)
    monkeypatch.setattr(
        "src.agents.ml_engineer.extract_response_text",
        lambda response: "import json\nprint('ok')\n",
    )

    agent = MLEngineerAgent()
    _ = agent.generate_code(
        strategy={"title": "Test Strategy", "analysis_type": "predictive", "required_columns": []},
        data_path="data/cleaned_data.csv",
        feedback_history=["legacy reviewer text"],
        previous_code="print('previous')\n",
        gate_context={
            "source": "reviewer",
            "status": "REJECTED",
            "feedback": "legacy reviewer text",
            "feedback_json": {
                "version": "v1",
                "status": "REJECTED",
                "failed_gates": ["submission_format_validation"],
                "required_fixes": ["Write required outputs at exact paths."],
            },
            "failed_gates": ["submission_format_validation"],
            "required_fixes": ["Write required outputs at exact paths."],
        },
        iteration_handoff={"mode": "patch"},
        execution_contract={"required_outputs": ["data/metrics.json"], "canonical_columns": []},
        ml_view={"required_outputs": ["data/metrics.json"]},
        editor_mode=True,
    )

    prompt = str(agent.last_prompt or "")
    assert "LATEST_ITERATION_FEEDBACK_JSON" in prompt
    assert "submission_format_validation" in prompt


def test_metric_optimization_editor_prompt_uses_optimization_template(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-openrouter")
    monkeypatch.setattr("src.agents.ml_engineer.OpenAI", _FakeOpenAI)

    def _fake_call_chat_with_fallback(client, messages, models, call_kwargs=None, logger=None, context_tag=None):
        return {"dummy": True}, models[0]

    monkeypatch.setattr("src.agents.ml_engineer.call_chat_with_fallback", _fake_call_chat_with_fallback)
    monkeypatch.setattr(
        "src.agents.ml_engineer.extract_response_text",
        lambda response: "import json\nprint('ok')\n",
    )

    agent = MLEngineerAgent()
    _ = agent.generate_code(
        strategy={"title": "Test Strategy", "analysis_type": "predictive", "required_columns": []},
        data_path="data/cleaned_data.csv",
        feedback_history=["# IMPROVEMENT_ROUND\nRESULTS_ADVISOR_FEEDBACK: apply one FE hypothesis."],
        previous_code="print('baseline')\n",
        gate_context={
            "source": "metric_improvement_optimizer",
            "status": "OPTIMIZATION_REQUIRED",
            "feedback": "Optimization round active.",
            "failed_gates": [],
            "required_fixes": ["Apply hypothesis with material edits."],
        },
        iteration_handoff={
            "mode": "optimize",
            "source": "actor_critic_metric_improvement",
            "optimization_focus": {
                "round_id": 1,
                "rounds_allowed": 1,
                "primary_metric_name": "roc_auc",
                "baseline_metric": 0.80,
                "min_delta": 0.0005,
                "higher_is_better": True,
                "feature_engineering_plan": {"techniques": [{"technique": "missing_indicators"}]},
            },
            "editor_constraints": {
                "must_apply_hypothesis": True,
                "forbid_noop": True,
                "patch_intensity": "aggressive",
            },
            "critic_packet": {"analysis_summary": "Baseline stable, no gain yet."},
            "hypothesis_packet": {
                "action": "APPLY",
                "hypothesis": {"technique": "missing_indicators", "target_columns": ["ALL_NUMERIC"]},
            },
        },
        execution_contract={"required_outputs": ["data/metrics.json"], "canonical_columns": []},
        ml_view={"required_outputs": ["data/metrics.json"]},
        editor_mode=True,
    )

    prompt = str(agent.last_prompt or "")
    assert "MODE: CODE_EDITOR_MODE_OPTIMIZATION" in prompt
    assert "OPTIMIZATION TARGET:" in prompt
    assert "FEATURE ENGINEERING PLAN (contract):" in prompt
    assert "HYPOTHESIS_PACKET_JSON" in prompt
