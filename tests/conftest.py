import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_GRAPH_SINGLETON_METHOD_SHADOWS = {
    "data_engineer": {"generate_cleaning_script"},
    "ml_engineer": {"generate_ml_plan", "generate_code"},
    "cleaning_reviewer": {"review_cleaning"},
    "qa_reviewer": {"review_code"},
    "reviewer": {"review_code"},
}

_ALLOW_EXTERNAL_LLM_TESTS_ENV = "ALLOW_EXTERNAL_LLM_TESTS"
_EXTERNAL_LLM_HOST_MARKERS = (
    "openrouter.ai",
    "api.openai.com",
    "generativelanguage.googleapis.com",
    "aiplatform.googleapis.com",
)
_PYTEST_DUMMY_LLM_KEYS = {
    "OPENROUTER_API_KEY": "pytest-dummy-openrouter",
    "GOOGLE_API_KEY": "pytest-dummy-google",
}


def _external_llm_tests_allowed() -> bool:
    return str(os.environ.get(_ALLOW_EXTERNAL_LLM_TESTS_ENV, "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _install_test_llm_env_defaults() -> None:
    if _external_llm_tests_allowed():
        return
    for key, value in _PYTEST_DUMMY_LLM_KEYS.items():
        os.environ.setdefault(key, value)


def _blocked_external_llm_call(*_args, **kwargs):
    model = kwargs.get("model")
    raise AssertionError(
        "External LLM call blocked during pytest. "
        f"model={model!r}. Mock the client or set "
        f"{_ALLOW_EXTERNAL_LLM_TESTS_ENV}=1 for an explicit integration run."
    )


async def _blocked_external_llm_call_async(*args, **kwargs):
    return _blocked_external_llm_call(*args, **kwargs)


def _blocked_gemini_generation(*_args, **_kwargs):
    raise AssertionError(
        "External Gemini generation blocked during pytest. "
        "Mock google.generativeai.GenerativeModel or set "
        f"{_ALLOW_EXTERNAL_LLM_TESTS_ENV}=1 for an explicit integration run."
    )


class _BlockedGeminiModel:
    def __init__(self, *args, **kwargs):
        self.model_name = kwargs.get("model_name") or (args[0] if args else None)

    def generate_content(self, *args, **kwargs):
        return _blocked_gemini_generation(*args, **kwargs)


def _llm_host_in_url(url: object) -> bool:
    text = str(url or "").lower()
    return any(marker in text for marker in _EXTERNAL_LLM_HOST_MARKERS)


def _install_external_llm_guard() -> None:
    if _external_llm_tests_allowed():
        return

    # OpenAI SDK path used by OpenRouter-compatible clients.
    for module_name in (
        "openai.resources.chat.completions.completions",
        "openai.resources.chat.completions",
    ):
        try:
            module = __import__(module_name, fromlist=["Completions", "AsyncCompletions"])
        except Exception:
            continue
        completions = getattr(module, "Completions", None)
        async_completions = getattr(module, "AsyncCompletions", None)
        if (
            completions is not None
            and getattr(completions, "create", None) is not _blocked_external_llm_call
        ):
            setattr(completions, "create", _blocked_external_llm_call)
        if (
            async_completions is not None
            and getattr(async_completions, "create", None) is not _blocked_external_llm_call_async
        ):
            setattr(async_completions, "create", _blocked_external_llm_call_async)

    # Gemini SDK direct path used by steward/execution-planner fallbacks.
    try:
        import google.generativeai as genai  # type: ignore

        model_cls = getattr(genai, "GenerativeModel", None)
        if model_cls is not None:
            setattr(genai, "GenerativeModel", _BlockedGeminiModel)
    except Exception:
        pass

    # Transport-level safety net for paths that bypass the SDK helpers.
    try:
        import httpx

        original_send = httpx.Client.send
        original_async_send = httpx.AsyncClient.send

        def guarded_send(self, request, *args, **kwargs):
            if _llm_host_in_url(getattr(request, "url", "")):
                raise AssertionError(
                    f"External LLM HTTP request blocked during pytest: {request.url}"
                )
            return original_send(self, request, *args, **kwargs)

        async def guarded_async_send(self, request, *args, **kwargs):
            if _llm_host_in_url(getattr(request, "url", "")):
                raise AssertionError(
                    f"External LLM HTTP request blocked during pytest: {request.url}"
                )
            return await original_async_send(self, request, *args, **kwargs)

        if getattr(httpx.Client.send, "__name__", "") != "guarded_send":
            httpx.Client.send = guarded_send
        if getattr(httpx.AsyncClient.send, "__name__", "") != "guarded_async_send":
            httpx.AsyncClient.send = guarded_async_send
    except Exception:
        pass

    try:
        import requests

        original_request = requests.sessions.Session.request

        def guarded_request(self, method, url, *args, **kwargs):
            if _llm_host_in_url(url):
                raise AssertionError(f"External LLM HTTP request blocked during pytest: {url}")
            return original_request(self, method, url, *args, **kwargs)

        if getattr(requests.sessions.Session.request, "__name__", "") != "guarded_request":
            requests.sessions.Session.request = guarded_request
    except Exception:
        pass


_install_test_llm_env_defaults()
_install_external_llm_guard()


def _clear_graph_singleton_method_shadows() -> None:
    graph_module = sys.modules.get("src.graph.graph")
    if graph_module is None:
        return
    for singleton_name, method_names in _GRAPH_SINGLETON_METHOD_SHADOWS.items():
        singleton = getattr(graph_module, singleton_name, None)
        if singleton is None or not hasattr(singleton, "__dict__"):
            continue
        for method_name in method_names:
            shadow = singleton.__dict__.get(method_name)
            if callable(shadow) and hasattr(type(singleton), method_name):
                singleton.__dict__.pop(method_name, None)


@pytest.fixture(autouse=True)
def _restore_graph_singleton_method_descriptors():
    """
    Undo method shadowing on graph-level singleton instances after tests patch
    attributes like src.graph.graph.ml_engineer.generate_code directly.
    """
    _clear_graph_singleton_method_shadows()
    yield
    _clear_graph_singleton_method_shadows()
