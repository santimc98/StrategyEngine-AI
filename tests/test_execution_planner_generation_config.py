from src.agents.execution_planner import ExecutionPlannerAgent


def test_execution_planner_sets_max_output_tokens_floor(monkeypatch):
    monkeypatch.delenv("EXECUTION_PLANNER_MAX_OUTPUT_TOKENS", raising=False)
    agent = ExecutionPlannerAgent(api_key=None)

    assert int(agent._generation_config.get("max_output_tokens", 0)) >= 4000


def test_execution_planner_generation_config_keeps_stable_defaults(monkeypatch):
    monkeypatch.delenv("EXECUTION_PLANNER_MAX_OUTPUT_TOKENS", raising=False)
    agent = ExecutionPlannerAgent(api_key=None)

    assert agent._generation_config["temperature"] == 0.0
    assert agent._generation_config["top_p"] == 0.9
    assert agent._generation_config["top_k"] == 40


def test_execution_planner_max_output_tokens_env_below_floor_is_clamped(monkeypatch):
    monkeypatch.setenv("EXECUTION_PLANNER_MAX_OUTPUT_TOKENS", "1024")
    agent = ExecutionPlannerAgent(api_key=None)

    assert int(agent._generation_config.get("max_output_tokens", 0)) == 4000


def test_execution_planner_dynamic_budget_respects_context_window(monkeypatch):
    monkeypatch.setenv("EXECUTION_PLANNER_CONTEXT_WINDOW_TOKENS", "9000")
    monkeypatch.setenv("EXECUTION_PLANNER_MAX_OUTPUT_TOKENS", "5000")
    agent = ExecutionPlannerAgent(api_key=None)
    prompt = "x" * 24000  # ~6000 prompt tokens with current estimator

    cfg = agent._generation_config_for_prompt(prompt)

    assert int(cfg.get("max_output_tokens", 0)) >= 1024
    assert int(cfg.get("max_output_tokens", 0)) <= int(agent._generation_config.get("max_output_tokens", 0))


def test_execution_planner_generation_config_includes_response_schema_when_enabled(monkeypatch):
    monkeypatch.setenv("EXECUTION_PLANNER_USE_RESPONSE_SCHEMA", "1")
    agent = ExecutionPlannerAgent(api_key=None)

    cfg = agent._generation_config_for_prompt("short prompt")

    assert "response_schema" in cfg
    schema = cfg.get("response_schema") or {}
    assert schema.get("type") == "object"
    assert "required_outputs" in ((schema.get("properties") or {}).keys())
    assert "optimization_policy" in ((schema.get("properties") or {}).keys())


def test_execution_planner_generate_content_retries_without_response_schema(monkeypatch):
    monkeypatch.setenv("EXECUTION_PLANNER_USE_RESPONSE_SCHEMA", "1")
    agent = ExecutionPlannerAgent(api_key=None)

    class _FakeModel:
        def __init__(self):
            self.calls = []

        def generate_content(self, prompt, generation_config=None):
            self.calls.append({"prompt": prompt, "generation_config": generation_config})
            if isinstance(generation_config, dict) and "response_schema" in generation_config:
                raise ValueError("Unknown field response_schema")
            return type("_Resp", (), {"text": "{}"})()

    fake_model = _FakeModel()
    response, used_config = agent._generate_content_with_budget(fake_model, "prompt")

    assert getattr(response, "text", "") == "{}"
    assert len(fake_model.calls) == 2
    first_cfg = fake_model.calls[0].get("generation_config") or {}
    second_cfg = fake_model.calls[1].get("generation_config") or {}
    assert "response_schema" in first_cfg
    assert "response_schema" not in second_cfg
    assert "response_schema" not in used_config
    assert used_config.get("response_mime_type") == "application/json"
