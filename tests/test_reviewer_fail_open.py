from src.agents.reviewer import ReviewerAgent
from types import SimpleNamespace


def test_reviewer_fail_closed(monkeypatch):
    monkeypatch.setenv("MIMO_API_KEY", "test-key")
    reviewer = ReviewerAgent(api_key="test-key")

    class DummyCompletions:
        def create(self, *args, **kwargs):
            raise RuntimeError("boom")

    class DummyChat:
        completions = DummyCompletions()

    class DummyClient:
        chat = DummyChat()

    reviewer.client = DummyClient()
    result = reviewer.review_code("print('ok')")
    assert result.get("status") == "REJECTED"
    assert "Reviewer unavailable" in result.get("feedback", "")
    assert "LLM_REVIEW_UNAVAILABLE" in (result.get("hard_failures") or [])


def test_reviewer_evaluate_results_parses_wrapped_json(monkeypatch):
    monkeypatch.setenv("MIMO_API_KEY", "test-key")
    reviewer = ReviewerAgent(api_key="test-key")

    content = (
        "Sure, here is the evaluation.\n"
        "```json\n"
        "{\n"
        '  "status": "APPROVED",\n'
        '  "feedback": "looks good",\n'
        '  "failed_gates": [],\n'
        '  "required_fixes": [],\n'
        '  "retry_worth_it": false\n'
        "}\n"
        "```\n"
        "Done."
    )

    class DummyCompletions:
        def create(self, *args, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
            )

    reviewer.provider = "openai"
    reviewer.client = SimpleNamespace(chat=SimpleNamespace(completions=DummyCompletions()))

    result = reviewer.evaluate_results(
        execution_output="execution finished successfully",
        business_objective="test objective",
        strategy_context="test strategy",
        evaluation_spec={},
    )
    assert result.get("status") == "APPROVED"
    assert result.get("feedback") == "looks good"


def test_reviewer_prompt_is_context_relative_not_recipe_driven():
    reviewer = ReviewerAgent(api_key=None)

    reviewer.review_code(
        "print('ok')",
        business_objective="Predecir el valor del contrato con validacion fiable",
        strategy_context="Usar el contexto de estrategia provisto por la run",
        evaluation_spec={"reviewer_gates": []},
    )

    prompt = reviewer.last_prompt or ""
    assert "METHOD FIT & EXECUTION RELIABILITY" in prompt
    assert "generic preferred methodology from memory" in prompt
    assert "Baseline Check:" not in prompt
    assert '"Black Box" Neural Net is bad' not in prompt
