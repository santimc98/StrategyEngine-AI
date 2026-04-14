from src.agents.ml_engineer import MLEngineerAgent


class FakeOpenAI:
    def __init__(self, api_key=None, base_url=None, timeout=None, **kwargs):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout


def test_ml_engineer_forces_openrouter_provider(monkeypatch):
    monkeypatch.setenv("ML_ENGINEER_PROVIDER", "zai")
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-openrouter")
    monkeypatch.setenv("OPENROUTER_ML_PRIMARY_MODEL", "moonshotai/kimi-k2.5")
    monkeypatch.setenv("OPENROUTER_ML_FALLBACK_MODEL", "minimax/minimax-m2.5")
    monkeypatch.delenv("OPENROUTER_ML_PLAN_MODEL", raising=False)
    monkeypatch.delenv("OPENROUTER_ML_EDITOR_MODEL", raising=False)
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    monkeypatch.delenv("GLM_API_KEY", raising=False)
    monkeypatch.setattr("src.agents.ml_engineer.OpenAI", FakeOpenAI)

    agent = MLEngineerAgent()

    assert agent.provider == "openrouter"
    assert agent.client.base_url == "https://openrouter.ai/api/v1"
    assert agent.model_name == "moonshotai/kimi-k2.5"
    assert agent.fallback_model_name == "minimax/minimax-m2.5"
    assert agent.plan_model_name == "moonshotai/kimi-k2.5"
    assert agent.editor_model_name == "moonshotai/kimi-k2.5"


def test_ml_engineer_editor_model_defaults_to_gpt54_mini(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-openrouter")
    monkeypatch.setenv("OPENROUTER_ML_PRIMARY_MODEL", "openai/gpt-5.4")
    monkeypatch.setenv("OPENROUTER_ML_FALLBACK_MODEL", "minimax/minimax-m2.5")
    monkeypatch.delenv("OPENROUTER_ML_PLAN_MODEL", raising=False)
    monkeypatch.delenv("OPENROUTER_ML_EDITOR_MODEL", raising=False)
    monkeypatch.setattr("src.agents.ml_engineer.OpenAI", FakeOpenAI)

    agent = MLEngineerAgent()

    assert agent.model_name == "openai/gpt-5.4"
    assert agent.plan_model_name == "openai/gpt-5.4"
    assert agent.editor_model_name == "openai/gpt-5.4-mini"


def test_ml_engineer_editor_model_honors_explicit_override(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-openrouter")
    monkeypatch.setenv("OPENROUTER_ML_PRIMARY_MODEL", "openai/gpt-5.4")
    monkeypatch.setenv("OPENROUTER_ML_FALLBACK_MODEL", "minimax/minimax-m2.5")
    monkeypatch.delenv("OPENROUTER_ML_PLAN_MODEL", raising=False)
    monkeypatch.setenv("OPENROUTER_ML_EDITOR_MODEL", "minimax/minimax-m2.5")
    monkeypatch.setattr("src.agents.ml_engineer.OpenAI", FakeOpenAI)

    agent = MLEngineerAgent()

    assert agent.model_name == "openai/gpt-5.4"
    assert agent.editor_model_name == "minimax/minimax-m2.5"


def test_ml_engineer_plan_model_has_independent_routing(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-openrouter")
    monkeypatch.setenv("OPENROUTER_ML_PLAN_MODEL", "google/gemini-3.1-pro-preview")
    monkeypatch.setenv("OPENROUTER_ML_PRIMARY_MODEL", "anthropic/claude-opus-4.6")
    monkeypatch.setenv("OPENROUTER_ML_FALLBACK_MODEL", "anthropic/claude-sonnet-4.6")
    monkeypatch.setattr("src.agents.ml_engineer.OpenAI", FakeOpenAI)

    captured = {}

    def fake_call_chat_with_fallback(client, messages, model_chain, call_kwargs, logger, context_tag):
        captured["model_chain"] = list(model_chain)
        captured["context_tag"] = context_tag
        return object(), model_chain[0]

    monkeypatch.setattr("src.agents.ml_engineer.call_chat_with_fallback", fake_call_chat_with_fallback)
    monkeypatch.setattr("src.agents.ml_engineer.extract_response_text", lambda _response: "{}")

    agent = MLEngineerAgent()
    content = agent._execute_llm_call("system", "user")

    assert content == "{}"
    assert captured["context_tag"] == "ml_engineer_plan"
    assert captured["model_chain"] == [
        "google/gemini-3.1-pro-preview",
        "anthropic/claude-opus-4.6",
        "anthropic/claude-sonnet-4.6",
    ]
