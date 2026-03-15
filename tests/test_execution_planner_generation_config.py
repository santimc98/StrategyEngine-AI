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
    assert "contract" in ((schema.get("properties") or {}).keys())
    contract_schema = (schema.get("properties") or {}).get("contract") or {}
    assert contract_schema.get("type") == "object"
    assert contract_schema.get("additionalProperties") is True


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


def test_execution_planner_defaults_to_openai_function_calling_stack(monkeypatch):
    monkeypatch.delenv("EXECUTION_PLANNER_PROVIDER", raising=False)
    monkeypatch.delenv("EXECUTION_PLANNER_PRIMARY_MODEL", raising=False)
    monkeypatch.delenv("EXECUTION_PLANNER_MODEL", raising=False)

    agent = ExecutionPlannerAgent(api_key=None)

    assert agent.provider == "openrouter"
    assert agent.model_name == "openai/gpt-5.4"


def test_execution_planner_forces_openrouter_even_if_provider_env_requests_openai(monkeypatch):
    monkeypatch.setenv("EXECUTION_PLANNER_PROVIDER", "openai")

    agent = ExecutionPlannerAgent(api_key=None)

    assert agent.provider == "openrouter"
    assert agent.base_url == "https://openrouter.ai/api/v1"


def test_execution_planner_extracts_tool_call_arguments():
    agent = ExecutionPlannerAgent(api_key=None)

    function_obj = type("_Fn", (), {"arguments": '{"scope":"full_pipeline"}'})()
    tool_call = type("_ToolCall", (), {"function": function_obj})()
    message = type("_Msg", (), {"tool_calls": [tool_call], "content": None})()
    choice = type("_Choice", (), {"message": message})()
    response = type("_Resp", (), {"choices": [choice]})()

    assert agent._extract_openai_response_text(response) == '{"scope":"full_pipeline"}'


def test_execution_planner_unwraps_transport_payload():
    agent = ExecutionPlannerAgent(api_key=None)

    payload = {"contract": {"scope": "full_pipeline", "required_outputs": ["data/submission.csv"]}}

    assert agent._extract_openai_response_text is not None
    from src.agents.execution_planner import _unwrap_execution_contract_transport

    assert _unwrap_execution_contract_transport(payload) == payload["contract"]
