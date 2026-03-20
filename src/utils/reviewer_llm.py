from __future__ import annotations

import os
from typing import Any, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

_PLACEHOLDER_KEYS = {"dummy", "test", "placeholder", "changeme"}


def _is_placeholder_key(value: Optional[str]) -> bool:
    if not value:
        return True
    return value.strip().lower() in _PLACEHOLDER_KEYS


def init_reviewer_llm(
    api_key: Optional[str] = None,
) -> Tuple[str, Any, Optional[str], Optional[str]]:
    """
    Returns (provider, client, model_name, warning).
    All reviewers route through OpenRouter using the OpenAI-compatible API.
    provider is always "openrouter" or "none".
    """
    openrouter_key = api_key if (api_key and not _is_placeholder_key(api_key)) else os.getenv("OPENROUTER_API_KEY")
    if openrouter_key and not _is_placeholder_key(openrouter_key):
        from openai import OpenAI

        model_name = os.getenv("REVIEWER_MODEL", "google/gemini-3-flash-preview")
        client = OpenAI(api_key=openrouter_key, base_url="https://openrouter.ai/api/v1")
        return "openrouter", client, model_name, None

    return "none", None, None, "No reviewer LLM API key configured (OPENROUTER_API_KEY)."
