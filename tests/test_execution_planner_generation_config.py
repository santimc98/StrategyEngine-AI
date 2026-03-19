from src.agents.execution_planner import ExecutionPlannerAgent
from src.utils.contract_response_schema import EXECUTION_CONTRACT_CANONICAL_REQUIRED_KEYS


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
    agent = ExecutionPlannerAgent(api_key=None)

    cfg = agent._generation_config_for_prompt("short prompt")

    assert "response_schema" not in cfg
    assert cfg.get("response_mime_type") == "application/json"


def test_execution_planner_canonical_schema_required_surface_is_semantic_first():
    assert "task_semantics" in EXECUTION_CONTRACT_CANONICAL_REQUIRED_KEYS
    assert "artifact_requirements" in EXECUTION_CONTRACT_CANONICAL_REQUIRED_KEYS
    # Post-migration: these are now required (LLM must generate them, no auto-projection)
    assert "evaluation_spec" in EXECUTION_CONTRACT_CANONICAL_REQUIRED_KEYS
    assert "validation_requirements" in EXECUTION_CONTRACT_CANONICAL_REQUIRED_KEYS
    assert "iteration_policy" in EXECUTION_CONTRACT_CANONICAL_REQUIRED_KEYS
    assert "column_dtype_targets" in EXECUTION_CONTRACT_CANONICAL_REQUIRED_KEYS


def test_execution_planner_generate_content_retries_without_response_schema(monkeypatch):
    agent = ExecutionPlannerAgent(api_key=None)

    class _FakeModel:
        def __init__(self):
            self.calls = []

        def generate_content(self, prompt, generation_config=None):
            self.calls.append({"prompt": prompt, "generation_config": generation_config})
            return type("_Resp", (), {"text": "{}"})()

    fake_model = _FakeModel()
    response, used_config = agent._generate_content_with_budget(fake_model, "prompt")

    assert getattr(response, "text", "") == "{}"
    assert len(fake_model.calls) == 1
    first_cfg = fake_model.calls[0].get("generation_config") or {}
    assert "tools" in first_cfg
    assert "tool_config" in first_cfg
    assert "response_mime_type" not in first_cfg
    assert "response_schema" not in first_cfg
    assert "tools" in used_config
    assert "tool_config" in used_config


def test_execution_planner_defaults_to_google_function_calling_stack(monkeypatch):
    monkeypatch.delenv("EXECUTION_PLANNER_PRIMARY_MODEL", raising=False)
    monkeypatch.delenv("EXECUTION_PLANNER_MODEL", raising=False)

    agent = ExecutionPlannerAgent(api_key=None)

    assert agent.provider == "google"
    assert agent.model_name == "gemini-3.1-pro-preview"


def test_execution_planner_uses_google_api_key_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")

    captured = {}

    class _FakeAdapter:
        def __init__(self, api_key, model_name):
            captured["api_key"] = api_key
            captured["model_name"] = model_name

    monkeypatch.setattr(
        "src.agents.execution_planner._GeminiGenerateContentAdapter",
        _FakeAdapter,
    )

    agent = ExecutionPlannerAgent()

    assert agent.provider == "google"
    assert captured.get("api_key") == "test-google-key"
    assert captured.get("model_name") == "gemini-3.1-pro-preview"


def test_execution_planner_extracts_tool_call_arguments():
    agent = ExecutionPlannerAgent(api_key=None)

    function_call = type("_FnCall", (), {"args": {"scope": "full_pipeline"}})()
    response = type("_Resp", (), {"function_calls": [function_call]})()

    assert agent._extract_openai_response_text(response) == '{"scope": "full_pipeline"}'


def test_execution_planner_unwraps_transport_payload():
    agent = ExecutionPlannerAgent(api_key=None)

    payload = {"contract": {"scope": "full_pipeline", "required_outputs": ["data/submission.csv"]}}

    assert agent._extract_openai_response_text is not None
    from src.agents.execution_planner import _unwrap_execution_contract_transport

    assert _unwrap_execution_contract_transport(payload) == payload["contract"]


def test_execution_planner_transport_validation_rejects_empty_payload():
    from src.agents.execution_planner import _build_transport_validation

    result = _build_transport_validation({})

    assert result.get("accepted") is False
    issues = result.get("issues") or []
    assert any(issue.get("rule") == "contract.transport_payload_empty" for issue in issues if isinstance(issue, dict))
