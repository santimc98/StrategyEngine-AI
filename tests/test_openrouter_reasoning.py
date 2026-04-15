from types import SimpleNamespace

import pytest

from src.utils import openrouter_reasoning
from src.utils.openrouter_reasoning import (
    apply_reasoning_to_call_kwargs,
    build_openrouter_reasoning,
    create_chat_completion_with_reasoning,
    model_supports_openrouter_reasoning,
)
from src.utils.llm_fallback import call_chat_with_fallback


@pytest.fixture(autouse=True)
def clean_reasoning_env(monkeypatch):
    for key in (
        "OPENROUTER_REASONING_ENABLED",
        "OPENROUTER_REASONING_EFFORT",
        "OPENROUTER_REASONING_EXCLUDE",
        "OPENROUTER_REASONING_FORCE",
        "OPENROUTER_REASONING_ENABLED_DATA_ENGINEER",
        "OPENROUTER_REASONING_EFFORT_DATA_ENGINEER",
        "OPENROUTER_REASONING_EXCLUDE_DATA_ENGINEER",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(openrouter_reasoning, "_OVERRIDES_CACHE", {})


def test_defaults_apply_xhigh_reasoning_to_openai_reasoning_models():
    payload = build_openrouter_reasoning(
        agent_name="data_engineer",
        model_name="openai/gpt-5.4",
    )

    assert payload == {"effort": "xhigh", "exclude": True}


def test_gemini_and_claude_reasoning_models_use_same_openrouter_shape():
    assert model_supports_openrouter_reasoning("google/gemini-3.1-pro-preview")
    assert model_supports_openrouter_reasoning("anthropic/claude-sonnet-4.5")

    gemini_payload = build_openrouter_reasoning(
        agent_name="execution_planner",
        model_name="google/gemini-3.1-pro-preview",
    )
    claude_payload = build_openrouter_reasoning(
        agent_name="reviewer",
        model_name="anthropic/claude-sonnet-4.5",
    )

    assert gemini_payload == {"effort": "xhigh", "exclude": True}
    assert claude_payload == {"effort": "xhigh", "exclude": True}


def test_non_reasoning_model_skips_reasoning_by_default():
    assert build_openrouter_reasoning(
        agent_name="translator",
        model_name="openai/gpt-4o",
    ) == {}


def test_apply_reasoning_merges_existing_extra_body_without_overwrite():
    kwargs = apply_reasoning_to_call_kwargs(
        {"extra_body": {"provider": {"order": ["OpenAI"]}}},
        agent_name="ml_engineer",
        model_name="openai/gpt-5.4",
    )

    assert kwargs["_codex_reasoning_applied"] is True
    assert kwargs["extra_body"]["provider"] == {"order": ["OpenAI"]}
    assert kwargs["extra_body"]["reasoning"] == {"effort": "xhigh", "exclude": True}


def test_agent_env_override_can_disable_reasoning(monkeypatch):
    monkeypatch.setenv("OPENROUTER_REASONING_ENABLED_DATA_ENGINEER", "false")

    assert build_openrouter_reasoning(
        agent_name="data_engineer",
        model_name="openai/gpt-5.4",
    ) == {}


def test_create_chat_completion_retries_without_reasoning_on_provider_rejection():
    class FakeCompletions:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                raise ValueError("400 unsupported reasoning parameter")
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="ok"),
                    )
                ]
            )

    completions = FakeCompletions()
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    response = create_chat_completion_with_reasoning(
        fake_client,
        agent_name="data_engineer",
        model_name="openai/gpt-5.4",
        call_kwargs={"model": "openai/gpt-5.4", "messages": []},
    )

    assert response.choices[0].message.content == "ok"
    assert completions.calls[0]["extra_body"]["reasoning"] == {
        "effort": "xhigh",
        "exclude": True,
    }
    assert "extra_body" not in completions.calls[1]


def test_llm_fallback_applies_reasoning_to_model_chain_calls():
    class FakeCompletions:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="ok"),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(completion_tokens=1, prompt_tokens=1),
            )

    completions = FakeCompletions()
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    response, used_model = call_chat_with_fallback(
        fake_client,
        [{"role": "user", "content": "hello"}],
        ["openai/gpt-5.4"],
        call_kwargs={"temperature": 0.0},
        logger=None,
        context_tag="data_engineer",
    )

    assert used_model == "openai/gpt-5.4"
    assert response.choices[0].message.content == "ok"
    assert completions.calls[0]["extra_body"]["reasoning"] == {
        "effort": "xhigh",
        "exclude": True,
    }
