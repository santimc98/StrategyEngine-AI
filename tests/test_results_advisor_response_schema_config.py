from src.agents.results_advisor import ResultsAdvisorAgent
from src.agents import results_advisor as results_advisor_mod


def test_results_advisor_generation_config_includes_response_schema_when_enabled(monkeypatch):
    monkeypatch.setenv("RESULTS_ADVISOR_USE_RESPONSE_SCHEMA", "1")
    agent = ResultsAdvisorAgent(api_key="")

    cfg = agent._generation_config_for_critique()

    assert "response_schema" in cfg
    schema = cfg.get("response_schema") or {}
    assert schema.get("type") == "object"
    assert "metric_comparison" in ((schema.get("properties") or {}).keys())


def test_results_advisor_generate_gemini_retries_without_response_schema(monkeypatch):
    monkeypatch.setenv("RESULTS_ADVISOR_USE_RESPONSE_SCHEMA", "1")
    agent = ResultsAdvisorAgent(api_key="")

    class _FakeGemini:
        def __init__(self):
            self.calls = []

        def generate_content(self, prompt, generation_config=None):
            self.calls.append({"prompt": prompt, "generation_config": generation_config})
            if isinstance(generation_config, dict) and "response_schema" in generation_config:
                raise ValueError("Unknown field response_schema")
            return type("_Resp", (), {"text": "{}"})()

    fake = _FakeGemini()
    agent.fe_provider = "gemini"
    agent.fe_client = fake

    content, used_config = agent._generate_gemini_json(
        "prompt",
        generation_config=agent._generation_config_for_critique(),
    )

    assert content == "{}"
    assert len(fake.calls) == 2
    first_cfg = fake.calls[0].get("generation_config") or {}
    second_cfg = fake.calls[1].get("generation_config") or {}
    assert "response_schema" in first_cfg
    assert "response_schema" not in second_cfg
    assert "response_schema" not in used_config


def test_results_advisor_critique_defaults_to_openrouter_gpt54(monkeypatch):
    captured = {}

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.delenv("RESULTS_ADVISOR_CRITIQUE_MODEL", raising=False)
    monkeypatch.setattr(results_advisor_mod, "OpenAI", _FakeOpenAI)

    agent = ResultsAdvisorAgent()

    assert agent.critique_mode == "llm"
    assert agent.critique_provider == "openrouter"
    assert agent.critique_model_name == "openai/gpt-5.4"
    assert captured.get("base_url") == "https://openrouter.ai/api/v1"
