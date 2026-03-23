import os

os.environ.setdefault("DEEPSEEK_API_KEY", "dummy")
os.environ.setdefault("GOOGLE_API_KEY", "dummy")
os.environ.setdefault("SANDBOX_PROVIDER", "local")
os.environ.setdefault("OPENROUTER_API_KEY", "dummy-openrouter")

from src.agents.ml_engineer import MLEngineerAgent
from src.graph.graph import run_engineer


class _FakeOpenAI:
    def __init__(self, api_key=None, base_url=None, timeout=None, default_headers=None):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.default_headers = default_headers


def test_optimization_mode_prompt_includes_action_family_and_invariants(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-openrouter")
    monkeypatch.setattr("src.agents.ml_engineer.OpenAI", _FakeOpenAI)

    def _fake_call_chat_with_fallback(client, messages, models, call_kwargs=None, logger=None, context_tag=None):
        return {"dummy": True}, models[0]

    monkeypatch.setattr("src.agents.ml_engineer.call_chat_with_fallback", _fake_call_chat_with_fallback)
    monkeypatch.setattr(
        "src.agents.ml_engineer.extract_response_text",
        lambda response: "import json\nprint('optimized')\n",
    )

    agent = MLEngineerAgent()
    _ = agent.generate_code(
        strategy={"title": "Test Strategy", "analysis_type": "predictive", "required_columns": []},
        data_path="data/cleaned_data.csv",
        previous_code="print('baseline')\n",
        feedback_history=["# IMPROVEMENT_ROUND"],
        gate_context={
            "source": "metric_improvement_optimizer",
            "status": "OPTIMIZATION_REQUIRED",
            "feedback": "Apply hypothesis with minimal patch.",
        },
        iteration_handoff={
            "mode": "optimize",
            "source": "actor_critic_metric_improvement",
            "optimization_focus": {
                "round_id": 1,
                "rounds_allowed": 3,
                "primary_metric_name": "roc_auc",
                "baseline_metric": 0.8,
                "min_delta": 0.0005,
                "higher_is_better": True,
            },
            "optimization_context": {
                "policy": {"phase": "explore"},
                "contract_lock": {"required_outputs": ["data/metrics.json", "data/submission.csv"]},
                "experiment_tracker_recent": [{"signature": "hyp_a", "delta": 0.0003}],
            },
            "hypothesis_packet": {
                "action": "APPLY",
                "hypothesis": {"technique": "stacking blend with calibrated meta model"},
            },
            "editor_constraints": {"must_apply_hypothesis": True, "forbid_noop": True},
        },
        execution_contract={"required_outputs": ["data/metrics.json"], "canonical_columns": []},
        ml_view={"required_outputs": ["data/metrics.json"]},
        editor_mode=True,
    )

    prompt = str(agent.last_prompt or "")
    assert (
        "MODE: METRIC_IMPROVEMENT" in prompt
        or "metric-improvement round" in prompt
    )
    assert "incumbent script" in prompt
    assert "cheapest valid change" in prompt


def test_run_engineer_optimization_mode_skips_baseline_plan(monkeypatch):
    calls = {"plan_called": False}

    def _fake_plan(*args, **kwargs):
        calls["plan_called"] = True
        raise AssertionError("generate_ml_plan should not run in optimization mode")

    def _fake_generate_code(*, editor_mode=False, previous_code=None, **kwargs):
        calls["editor_mode"] = editor_mode
        calls["previous_code"] = previous_code
        return "print('optimized-edit')"

    monkeypatch.setattr("src.graph.graph.ml_engineer.generate_ml_plan", _fake_plan, raising=True)
    monkeypatch.setattr("src.graph.graph.ml_engineer.generate_code", _fake_generate_code, raising=True)
    monkeypatch.setattr(
        "src.graph.graph.load_recent_memory",
        lambda run_id, k=5: [{"iter": 1, "attempt": 1, "event": "candidate_evaluated"}],
        raising=True,
    )

    state = {
        "run_id": "unit_opt_mode",
        "selected_strategy": {"title": "Strategy", "analysis_type": "predictive", "required_columns": []},
        "feedback_history": [],
        "data_summary": "",
        "business_objective": "",
        "csv_encoding": "utf-8",
        "csv_sep": ",",
        "csv_decimal": ".",
        "generated_code": "print('baseline')",
        "last_generated_code": "print('baseline')",
        "last_gate_context": {
            "source": "metric_improvement_optimizer",
            "status": "OPTIMIZATION_REQUIRED",
            "feedback": "Apply structured hypothesis.",
            "failed_gates": [],
            "required_fixes": ["Apply one action-family hypothesis."],
        },
        "iteration_handoff": {
            "mode": "optimize",
            "source": "actor_critic_metric_improvement",
            "hypothesis_packet": {
                "action": "APPLY",
                "hypothesis": {"technique": "bounded hyperparameter search"},
            },
            "optimization_context": {
                "contract_lock": {"required_outputs": ["data/metrics.json"]},
                "experiment_tracker_recent": [{"signature": "hyp_01"}],
            },
            "editor_constraints": {"must_apply_hypothesis": True, "forbid_noop": True},
        },
        "ml_improvement_round_active": True,
        "iteration_count": 0,
        "execution_contract": {"required_outputs": ["data/metrics.json"], "canonical_columns": []},
    }

    result = run_engineer(state)

    assert calls["plan_called"] is False
    assert calls.get("editor_mode") is True
    assert calls.get("previous_code") == "print('baseline')"
    assert result.get("generated_code") == "print('optimized-edit')"
