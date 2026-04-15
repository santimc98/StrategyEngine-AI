"""OpenRouter reasoning controls shared by all agent LLM calls.

OpenRouter exposes a provider-normalized `reasoning` object through the
OpenAI-compatible SDK via `extra_body`. Keep this logic centralized so agents
can opt into deeper reasoning without each call site knowing provider details.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict

from src.utils.paths import PROJECT_ROOT

_ALLOWED_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}
_DEFAULT_REASONING_EFFORT = "xhigh"
_DEFAULT_EXCLUDE_REASONING = True
_OVERRIDES_CACHE: Dict[str, Any] | None = None


def _truthy(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    text = str(value).strip().lower()
    if not text:
        return default
    return text in {"1", "true", "yes", "on", "y"}


def _normalize_agent_name(agent_name: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(agent_name or "default").strip().lower()).strip("_") or "default"


def _agent_env_suffix(agent_name: str) -> str:
    return _normalize_agent_name(agent_name).upper()


def _load_reasoning_overrides() -> Dict[str, Any]:
    global _OVERRIDES_CACHE
    if _OVERRIDES_CACHE is not None:
        return _OVERRIDES_CACHE
    path = os.path.join(PROJECT_ROOT, "data", "agent_reasoning_overrides.json")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        _OVERRIDES_CACHE = payload if isinstance(payload, dict) else {}
    except Exception:
        _OVERRIDES_CACHE = {}
    return _OVERRIDES_CACHE


def _override_for_agent(agent_name: str) -> Dict[str, Any]:
    overrides = _load_reasoning_overrides()
    value = overrides.get(_normalize_agent_name(agent_name))
    if value is None:
        value = overrides.get(str(agent_name or ""))
    if isinstance(value, str):
        return {"effort": value}
    return value if isinstance(value, dict) else {}


def model_supports_openrouter_reasoning(model_name: Any) -> bool:
    """Best-effort allowlist for models whose OpenRouter routes support reasoning."""
    if _truthy(os.getenv("OPENROUTER_REASONING_FORCE"), default=False):
        return True
    model = str(model_name or "").strip().lower()
    if not model:
        return False
    if model.startswith("openai/"):
        return any(token in model for token in ("gpt-5", "o1", "o3", "o4"))
    if model.startswith("google/"):
        return any(token in model for token in ("gemini-3", "gemini-2.5", "thinking"))
    if model.startswith("anthropic/"):
        return bool(re.search(r"claude.*(3\.7|4)", model))
    if model.startswith("x-ai/") or model.startswith("xai/"):
        return "grok" in model
    if model.startswith("deepseek/"):
        return any(token in model for token in ("r1", "reasoner", "v3.2"))
    if model.startswith("qwen/") or model.startswith("alibaba/"):
        return any(token in model for token in ("qwen3", "thinking"))
    if model.startswith("moonshotai/") or model.startswith("moonshot/"):
        return "kimi" in model
    if model.startswith("minimax/"):
        return True
    if model.startswith("z-ai/") or model.startswith("zai/"):
        return any(token in model for token in ("glm-4.5", "glm-5"))
    return False


def _resolve_reasoning_effort(agent_name: str) -> str:
    suffix = _agent_env_suffix(agent_name)
    override = _override_for_agent(agent_name)
    raw = (
        os.getenv(f"OPENROUTER_REASONING_EFFORT_{suffix}")
        or override.get("effort")
        or os.getenv("OPENROUTER_REASONING_EFFORT")
        or _DEFAULT_REASONING_EFFORT
    )
    effort = str(raw or _DEFAULT_REASONING_EFFORT).strip().lower()
    return effort if effort in _ALLOWED_EFFORTS else _DEFAULT_REASONING_EFFORT


def _resolve_reasoning_enabled(agent_name: str) -> bool:
    suffix = _agent_env_suffix(agent_name)
    override = _override_for_agent(agent_name)
    if os.getenv(f"OPENROUTER_REASONING_ENABLED_{suffix}") is not None:
        return _truthy(os.getenv(f"OPENROUTER_REASONING_ENABLED_{suffix}"), default=True)
    if "enabled" in override:
        return _truthy(override.get("enabled"), default=True)
    return _truthy(os.getenv("OPENROUTER_REASONING_ENABLED"), default=True)


def _resolve_reasoning_exclude(agent_name: str) -> bool:
    suffix = _agent_env_suffix(agent_name)
    override = _override_for_agent(agent_name)
    if os.getenv(f"OPENROUTER_REASONING_EXCLUDE_{suffix}") is not None:
        return _truthy(os.getenv(f"OPENROUTER_REASONING_EXCLUDE_{suffix}"), default=_DEFAULT_EXCLUDE_REASONING)
    if "exclude" in override:
        return _truthy(override.get("exclude"), default=_DEFAULT_EXCLUDE_REASONING)
    return _truthy(os.getenv("OPENROUTER_REASONING_EXCLUDE"), default=_DEFAULT_EXCLUDE_REASONING)


def build_openrouter_reasoning(
    *,
    agent_name: str,
    model_name: Any,
) -> Dict[str, Any]:
    if not _resolve_reasoning_enabled(agent_name):
        return {}
    if not model_supports_openrouter_reasoning(model_name):
        return {}
    effort = _resolve_reasoning_effort(agent_name)
    if effort == "none":
        return {"effort": "none", "exclude": True}
    return {
        "effort": effort,
        "exclude": _resolve_reasoning_exclude(agent_name),
    }


def apply_reasoning_to_call_kwargs(
    call_kwargs: Dict[str, Any] | None,
    *,
    agent_name: str,
    model_name: Any,
) -> Dict[str, Any]:
    kwargs = dict(call_kwargs or {})
    reasoning = build_openrouter_reasoning(agent_name=agent_name, model_name=model_name)
    if not reasoning:
        return kwargs
    extra_body = kwargs.get("extra_body")
    if not isinstance(extra_body, dict):
        extra_body = {}
    else:
        extra_body = dict(extra_body)
    extra_body.setdefault("reasoning", reasoning)
    kwargs["extra_body"] = extra_body
    kwargs["_codex_reasoning_applied"] = True
    return kwargs


def strip_reasoning_from_call_kwargs(call_kwargs: Dict[str, Any] | None) -> Dict[str, Any]:
    kwargs = dict(call_kwargs or {})
    extra_body = kwargs.get("extra_body")
    if isinstance(extra_body, dict) and "reasoning" in extra_body:
        extra_body = dict(extra_body)
        extra_body.pop("reasoning", None)
        if extra_body:
            kwargs["extra_body"] = extra_body
        else:
            kwargs.pop("extra_body", None)
    kwargs.pop("_codex_reasoning_applied", None)
    return kwargs


def is_reasoning_parameter_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    if "reasoning" not in text and "thinking" not in text and "extra_body" not in text:
        return False
    markers = (
        "unsupported",
        "unknown",
        "invalid",
        "not supported",
        "unrecognized",
        "unexpected",
        "bad request",
        "400",
    )
    return any(marker in text for marker in markers)


def create_chat_completion_with_reasoning(
    client: Any,
    *,
    agent_name: str,
    model_name: Any,
    call_kwargs: Dict[str, Any],
) -> Any:
    enriched = apply_reasoning_to_call_kwargs(
        call_kwargs,
        agent_name=agent_name,
        model_name=model_name,
    )
    reasoning_applied = bool(enriched.pop("_codex_reasoning_applied", False))
    try:
        return client.chat.completions.create(**enriched)
    except Exception as exc:
        if reasoning_applied and is_reasoning_parameter_error(exc):
            return client.chat.completions.create(**strip_reasoning_from_call_kwargs(enriched))
        raise
