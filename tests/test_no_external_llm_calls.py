import os

import pytest


def _external_llm_tests_allowed() -> bool:
    return os.getenv("ALLOW_EXTERNAL_LLM_TESTS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def test_pytest_blocks_openai_compatible_chat_completion_calls():
    if _external_llm_tests_allowed():
        pytest.skip("external LLM integration runs explicitly enabled")

    from openai.resources.chat.completions import Completions

    with pytest.raises(AssertionError, match="External LLM call blocked during pytest"):
        Completions.create(
            None,
            model="z-ai/glm-5",
            messages=[{"role": "user", "content": "ping"}],
        )


def test_pytest_blocks_llm_http_transport_calls():
    if _external_llm_tests_allowed():
        pytest.skip("external LLM integration runs explicitly enabled")

    import httpx

    with httpx.Client() as client:
        request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
        with pytest.raises(AssertionError, match="External LLM HTTP request blocked during pytest"):
            client.send(request)
