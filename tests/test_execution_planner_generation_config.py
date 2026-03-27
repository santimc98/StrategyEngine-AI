from types import SimpleNamespace

from src.agents.execution_planner import ExecutionPlannerAgent, _OpenRouterAdapter
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
    assert "iteration_policy" in EXECUTION_CONTRACT_CANONICAL_REQUIRED_KEYS
    assert "column_dtype_targets" in EXECUTION_CONTRACT_CANONICAL_REQUIRED_KEYS
    # These remain conditional capability sections, not universal top-level requirements.
    assert "evaluation_spec" not in EXECUTION_CONTRACT_CANONICAL_REQUIRED_KEYS
    assert "validation_requirements" not in EXECUTION_CONTRACT_CANONICAL_REQUIRED_KEYS


def test_execution_planner_generate_content_uses_plain_json_generation_without_tool_calling(monkeypatch):
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
    assert "tools" not in first_cfg
    assert "tool_config" not in first_cfg
    assert first_cfg.get("response_mime_type") == "application/json"
    assert "response_schema" not in first_cfg
    assert "tools" not in used_config
    assert "tool_config" not in used_config
    assert used_config.get("response_mime_type") == "application/json"


def test_execution_planner_defaults_to_openrouter_json_generation_stack(monkeypatch):
    monkeypatch.delenv("EXECUTION_PLANNER_PRIMARY_MODEL", raising=False)
    monkeypatch.delenv("EXECUTION_PLANNER_MODEL", raising=False)
    monkeypatch.delenv("EXECUTION_PLANNER_COMPILER_MODEL", raising=False)

    agent = ExecutionPlannerAgent(api_key=None)

    assert agent.provider == "openrouter"
    assert agent.model_name == "google/gemini-3.1-pro-preview"
    assert agent.compiler_model_name == "google/gemini-3-flash-preview"


def test_execution_planner_allows_explicit_compiler_model_override(monkeypatch):
    monkeypatch.setenv("EXECUTION_PLANNER_COMPILER_MODEL", "google/gemini-3.1-pro-preview")

    agent = ExecutionPlannerAgent(api_key=None)

    assert agent.compiler_model_name == "google/gemini-3.1-pro-preview"


def test_execution_planner_uses_openrouter_api_key_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")

    captured = {}

    class _FakeAdapter:
        def __init__(self, api_key, model_name):
            captured["api_key"] = api_key
            captured["model_name"] = model_name

    monkeypatch.setattr(
        "src.agents.execution_planner._OpenRouterAdapter",
        _FakeAdapter,
    )

    agent = ExecutionPlannerAgent()

    assert agent.provider == "openrouter"
    assert captured.get("api_key") == "test-openrouter-key"
    assert captured.get("model_name") == "google/gemini-3.1-pro-preview"


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


def test_openrouter_adapter_uses_single_standard_transport_call_by_default(monkeypatch):
    monkeypatch.delenv("EXECUTION_PLANNER_CAPTURE_RAW_RESPONSE", raising=False)
    monkeypatch.delenv("EXECUTION_PLANNER_TRANSPORT_MAX_RETRIES", raising=False)

    class _FakeCreate:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))],
                usage=SimpleNamespace(completion_tokens=5, prompt_tokens=7),
                _request_id="req_standard_123",
            )

    create_api = _FakeCreate()
    raw_api = SimpleNamespace(create=lambda **kwargs: (_ for _ in ()).throw(AssertionError("raw transport should not be used")))
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create_api.create, with_raw_response=raw_api)))

    adapter = _OpenRouterAdapter(api_key="test", model_name="model")
    adapter._client = fake_client

    response = adapter.generate_content("hello", {"max_output_tokens": 123})

    assert len(create_api.calls) == 1
    assert getattr(response, "_codex_transport_mode", None) == "standard"
    assert getattr(response, "_codex_transport_max_retries", None) == 0
    assert getattr(response, "_codex_request_id", None) == "req_standard_123"


def test_openrouter_adapter_raw_capture_mode_makes_single_raw_request(monkeypatch):
    monkeypatch.setenv("EXECUTION_PLANNER_CAPTURE_RAW_RESPONSE", "1")
    monkeypatch.delenv("EXECUTION_PLANNER_TRANSPORT_MAX_RETRIES", raising=False)

    class _FakeRawResponse:
        def __init__(self):
            self.http_response = SimpleNamespace(
                text='{"choices":[{"message":{"content":"{\\"ok\\": true}"}}]}',
                headers={"x-request-id": "req_raw_456"},
            )

        def parse(self):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))],
                usage=SimpleNamespace(completion_tokens=5, prompt_tokens=7),
            )

    class _RawAPI:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return _FakeRawResponse()

    raw_api = _RawAPI()
    standard_calls = []
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: standard_calls.append(kwargs),
                with_raw_response=raw_api,
            )
        )
    )

    adapter = _OpenRouterAdapter(api_key="test", model_name="model")
    adapter._client = fake_client

    response = adapter.generate_content("hello", {"max_output_tokens": 123})

    assert len(raw_api.calls) == 1
    assert standard_calls == []
    assert getattr(response, "_codex_transport_mode", None) == "with_raw_response"
    assert isinstance(getattr(response, "_codex_raw_body", None), str)
    assert getattr(response, "_codex_transport_max_retries", None) == 0
    assert getattr(response, "_codex_request_id", None) == "req_raw_456"


def test_openrouter_adapter_disables_sdk_retries_by_default(monkeypatch):
    monkeypatch.delenv("EXECUTION_PLANNER_TRANSPORT_MAX_RETRIES", raising=False)

    captured = {}

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.chat = SimpleNamespace(completions=SimpleNamespace())

    monkeypatch.setattr("src.agents.execution_planner.OpenAI", _FakeOpenAI)

    adapter = _OpenRouterAdapter(api_key="test-key", model_name="model-name")

    assert adapter.transport_max_retries == 0
    assert captured.get("max_retries") == 0


def test_openrouter_adapter_respects_explicit_transport_retry_override(monkeypatch):
    monkeypatch.setenv("EXECUTION_PLANNER_TRANSPORT_MAX_RETRIES", "1")

    captured = {}

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.chat = SimpleNamespace(completions=SimpleNamespace())

    monkeypatch.setattr("src.agents.execution_planner.OpenAI", _FakeOpenAI)

    adapter = _OpenRouterAdapter(api_key="test-key", model_name="model-name")

    assert adapter.transport_max_retries == 1
    assert captured.get("max_retries") == 1
