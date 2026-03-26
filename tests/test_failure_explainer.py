from src.agents.failure_explainer import FailureExplainerAgent


def test_failure_explainer_returns_empty_without_code_or_error(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    agent = FailureExplainerAgent(api_key=None)

    assert agent.explain_data_engineer_failure("", "boom", {}) == ""
    assert agent.explain_data_engineer_failure("print('x')", "", {}) == ""
    assert agent.explain_ml_failure("", "boom", {}) == ""
    assert agent.explain_ml_failure("print('x')", "", {}) == ""


def test_failure_explainer_data_prompt_uses_llm(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    agent = FailureExplainerAgent(api_key=None)
    agent._client = object()
    agent._model_name = "dummy"
    captured = {}

    def _fake_call(prompt: str) -> str:
        captured["prompt"] = prompt
        return "WHERE: clean\nWHY: mismatch\nFIX: align rows"

    agent._call_llm = _fake_call
    result = agent.explain_data_engineer_failure(
        "print('clean')",
        "ValueError: length mismatch",
        {"step": "impute"},
    )

    assert result == "WHERE: clean\nWHY: mismatch\nFIX: align rows"
    prompt = captured.get("prompt") or ""
    assert "senior debugging assistant" in prompt
    assert "generated Python cleaning code" in prompt
    assert "ValueError: length mismatch" in prompt
    assert "'step': 'impute'" in prompt


def test_failure_explainer_ml_prompt_falls_back_on_llm_error(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    agent = FailureExplainerAgent(api_key=None)
    agent._client = object()
    agent._model_name = "dummy"

    def _fake_call(_prompt: str) -> str:
        raise RuntimeError("boom")

    agent._call_llm = _fake_call
    result = agent.explain_ml_failure(
        "print('train')",
        "RuntimeError: boom",
        {"phase": "fit"},
    )

    assert result.startswith("Automated diagnosis unavailable.")
    assert "RuntimeError: boom" in result


def test_failure_explainer_truncates_large_inputs(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    agent = FailureExplainerAgent(api_key=None)
    agent._client = object()
    agent._model_name = "dummy"
    captured = {}

    def _fake_call(prompt: str) -> str:
        captured["prompt"] = prompt
        return "WHERE: fit\nWHY: overflow\nFIX: inspect feature pipeline"

    agent._call_llm = _fake_call
    code = "A" * 7000
    error = "B" * 5000
    context = {"payload": "C" * 2500}

    agent.explain_ml_failure(code, error, context)

    prompt = captured.get("prompt") or ""
    assert "...[truncated]..." in prompt
    assert "senior ML debugging assistant" in prompt
