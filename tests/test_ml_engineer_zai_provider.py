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
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    monkeypatch.delenv("GLM_API_KEY", raising=False)
    monkeypatch.setattr("src.agents.ml_engineer.OpenAI", FakeOpenAI)

    agent = MLEngineerAgent()

    assert agent.provider == "openrouter"
    assert agent.client.base_url == "https://openrouter.ai/api/v1"
    assert agent.model_name == "moonshotai/kimi-k2.5"
    assert agent.fallback_model_name == "minimax/minimax-m2.5"
    assert agent.editor_model_name == "moonshotai/kimi-k2.5"


def test_ml_engineer_editor_model_follows_primary(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-openrouter")
    monkeypatch.setenv("OPENROUTER_ML_PRIMARY_MODEL", "openai/gpt-5.4")
    monkeypatch.setenv("OPENROUTER_ML_FALLBACK_MODEL", "minimax/minimax-m2.5")
    monkeypatch.setenv("OPENROUTER_ML_EDITOR_MODEL", "minimax/minimax-m2.5")
    monkeypatch.setattr("src.agents.ml_engineer.OpenAI", FakeOpenAI)

    agent = MLEngineerAgent()

    assert agent.model_name == "openai/gpt-5.4"
    assert agent.editor_model_name == "openai/gpt-5.4"
