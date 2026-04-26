from src.agents.failure_explainer import FailureExplainerAgent


def _assert_contains_all(text: str, *needles: str) -> None:
    for needle in needles:
        assert needle in text


def _assert_contains_terms(text: str, *terms: str) -> None:
    lowered = text.lower()
    for term in terms:
        assert term.lower() in lowered


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
    _assert_contains_all(prompt, "ValueError: length mismatch", "'step': 'impute'")
    _assert_contains_terms(prompt, "senior", "debugging assistant", "cleaning code")


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
    _assert_contains_all(prompt, "...[truncated]...")
    _assert_contains_terms(prompt, "senior", "ml", "debugging assistant")


def test_failure_explainer_injects_boolean_quantile_runtime_hint(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    agent = FailureExplainerAgent(api_key=None)
    agent._client = object()
    agent._model_name = "dummy"
    captured = {}

    def _fake_call(prompt: str) -> str:
        captured["prompt"] = prompt
        return "WHERE: drift\nWHY: bool quantile\nFIX: cast bool"

    agent._call_llm = _fake_call
    error = (
        "TypeError: numpy boolean subtract, the `-` operator, is not supported\n"
        "File pandas/core/series.py, line 2901, in quantile"
    )

    agent.explain_ml_failure(
        "valid_num = num.dropna(); valid_num.quantile(0.25)",
        error,
        {"phase": "runtime_repair"},
    )

    prompt = captured.get("prompt") or ""
    _assert_contains_all(prompt, "SYSTEM_DETECTED_RUNTIME_FACTS", "empty-series guard alone is insufficient")
    _assert_contains_terms(prompt, "boolean", "coerce to numeric", "cast bool")


def test_failure_explainer_fallback_distinguishes_boolean_quantile(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    agent = FailureExplainerAgent(api_key=None)

    result = agent.explain_ml_failure(
        "print('x')",
        "TypeError: numpy boolean subtract while computing Series.quantile",
        {},
    )

    _assert_contains_terms(result, "boolean", "quantile", "empty-series guard alone is insufficient")
