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


def test_reviewer_prompt_includes_hard_blocker_packet_for_restored_candidates():
    reviewer = ReviewerAgent(api_key=None)

    reviewer.review_code(
        "\n".join(
            [
                "if int(holdout_mask.sum()) < 400:",
                "    raise ValueError('holdout too small')",
                "scored_output['churn_risk_score'] = baseline_model.predict_proba(X_score)[:, 1]",
            ]
        ),
        business_objective="Mantener credibilidad del holdout y scoring operativo",
        strategy_context="Usar el contexto de la run",
        evaluation_spec={
            "reviewer_gates": [
                {
                    "name": "temporal_validation_credibility",
                    "severity": "HARD",
                    "params": {"min_rows": 1000},
                }
            ],
            "review_history_context": {
                "best_attempt_restored_recently": True,
                "feedback_history_tail": [
                    "BEST_ATTEMPT_RESTORED[result_evaluator]: restored attempt 2 as authoritative state after a later degraded execution."
                ],
                "last_gate_context": {
                    "failed_gates": ["temporal_validation_credibility"],
                    "required_fixes": [
                        "Use primary_model.predict_proba for scoring.",
                    ],
                },
            },
        },
        reviewer_view={"subject_code_path_hint": "artifacts/ml_engineer_last.py"},
    )

    prompt = reviewer.last_prompt or ""
    assert "HARD_BLOCKER_PACKET" in prompt
    assert "best_attempt_restored_recently" in prompt
    assert "baseline_model.predict_proba" in prompt
