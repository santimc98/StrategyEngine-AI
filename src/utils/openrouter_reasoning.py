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
_DEFAULT_REASONING_EFFORT = "medium"
_DEFAULT_EXCLUDE_REASONING = True
_DEFAULT_MAX_TOKENS = 32768
_MAX_TOKEN_FALLBACK_STEPS = (49152, 32768, 24576, 16384, 8192, 4096, 2048)
_AGENT_MAX_TOKEN_DEFAULTS = {
    # --- HIGH reasoning: need headroom for thinking + full output ---
    "strategist": 49152,            # high effort; creative strategy design
    "strategist_generate": 49152,   # high effort; strategy candidate generation
    "execution_planner": 49152,     # high effort; semantic reasoning
    "ml_engineer_plan": 49152,      # high effort; ML architecture planning
    # --- MEDIUM reasoning: moderate thinking + structured output ---
    "steward": 32768,               # medium effort; data audit
    "steward_semantics": 32768,     # medium effort; data semantics
    "data_engineer": 49152,         # medium effort; code gen needs headroom (27K+ prompts)
    "model_analyst": 32768,         # medium effort; analysis + hypotheses
    "reviewer": 32768,              # medium effort; code review + gates
    "qa_reviewer": 32768,           # medium effort; compliance review
    "cleaning_reviewer": 32768,     # medium effort; cleaning review
    "failure_explainer": 32768,     # medium effort; causal diagnosis
    # --- LOW/NONE reasoning: full budget goes to output content ---
    "ml_engineer": 49152,           # low effort; long code output, heavy prompts
    "translator": 32768,            # low effort; long executive report
    "review_board": 32768,          # low effort; synthesis of reviews
    "results_advisor": 32768,       # low effort; interpretation
    "results_advisor_critique": 32768,  # low effort; evaluation
    "ml_engineer_editor": 32768,    # low effort; code editing
    "data_engineer_editor": 32768,  # low effort; code editing
    "results_advisor_llm": 32768,   # none; mechanical support
    "translator_repair": 32768,     # none; format repair
    # --- DISABLED reasoning: pure translation, no thinking overhead ---
    "execution_planner_compiler": 65536,  # disabled; full budget for large JSON contract
}
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


def _coerce_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except Exception:
        return None
    return parsed if parsed > 0 else None


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


def _resolve_output_budget_enabled(agent_name: str) -> bool:
    suffix = _agent_env_suffix(agent_name)
    if os.getenv(f"OPENROUTER_MAX_TOKENS_ENABLED_{suffix}") is not None:
        return _truthy(os.getenv(f"OPENROUTER_MAX_TOKENS_ENABLED_{suffix}"), default=True)
    return _truthy(os.getenv("OPENROUTER_MAX_TOKENS_ENABLED"), default=True)


def _resolve_output_budget(agent_name: str) -> int:
    normalized = _normalize_agent_name(agent_name)
    suffix = _agent_env_suffix(agent_name)
    candidates = (
        os.getenv(f"OPENROUTER_MAX_TOKENS_{suffix}"),
        os.getenv(f"{suffix}_MAX_TOKENS"),
        os.getenv("OPENROUTER_MAX_TOKENS_DEFAULT"),
        _AGENT_MAX_TOKEN_DEFAULTS.get(normalized),
        _DEFAULT_MAX_TOKENS,
    )
    for candidate in candidates:
        parsed = _coerce_positive_int(candidate)
        if parsed:
            return parsed
    return _DEFAULT_MAX_TOKENS


def _resolve_output_budget_floor_enabled(agent_name: str) -> bool:
    suffix = _agent_env_suffix(agent_name)
    if os.getenv(f"OPENROUTER_MAX_TOKENS_FLOOR_{suffix}") is not None:
        return _truthy(os.getenv(f"OPENROUTER_MAX_TOKENS_FLOOR_{suffix}"), default=True)
    return _truthy(os.getenv("OPENROUTER_MAX_TOKENS_FLOOR"), default=False)


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
    if _resolve_output_budget_enabled(agent_name):
        desired_max_tokens = _resolve_output_budget(agent_name)
        current_max_tokens = _coerce_positive_int(kwargs.get("max_tokens"))
        if current_max_tokens is None or (
            _resolve_output_budget_floor_enabled(agent_name)
            and current_max_tokens < desired_max_tokens
        ):
            kwargs["max_tokens"] = desired_max_tokens
            kwargs["_codex_max_tokens_applied"] = True
            kwargs["_codex_max_tokens_requested"] = desired_max_tokens
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


def pop_internal_call_markers(call_kwargs: Dict[str, Any] | None) -> Dict[str, Any]:
    kwargs = dict(call_kwargs or {})
    for key in (
        "_codex_reasoning_applied",
        "_codex_max_tokens_applied",
        "_codex_max_tokens_requested",
    ):
        kwargs.pop(key, None)
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
    return pop_internal_call_markers(kwargs)


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


def is_token_budget_parameter_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    if not any(token in text for token in ("token", "max_tokens", "max output", "context", "length")):
        return False
    markers = (
        "maximum",
        "exceed",
        "exceeds",
        "too large",
        "too many",
        "less than",
        "invalid",
        "unsupported",
        "context length",
        "max_tokens",
        "max output",
        "400",
    )
    return any(marker in text for marker in markers)


def lower_token_budget_in_call_kwargs(call_kwargs: Dict[str, Any] | None) -> Dict[str, Any] | None:
    kwargs = dict(call_kwargs or {})
    current = _coerce_positive_int(kwargs.get("max_tokens"))
    if current is None:
        return None
    for candidate in _MAX_TOKEN_FALLBACK_STEPS:
        if candidate < current:
            lowered = dict(kwargs)
            lowered["max_tokens"] = candidate
            return lowered
    return None


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
    enriched = pop_internal_call_markers(enriched)
    reasoning_stripped = False
    attempted_token_budgets: set[int] = set()
    while True:
        try:
            return client.chat.completions.create(**enriched)
        except Exception as exc:
            if reasoning_applied and not reasoning_stripped and is_reasoning_parameter_error(exc):
                enriched = strip_reasoning_from_call_kwargs(enriched)
                reasoning_stripped = True
                continue
            if is_token_budget_parameter_error(exc):
                current_tokens = _coerce_positive_int(enriched.get("max_tokens"))
                if current_tokens is not None:
                    if current_tokens in attempted_token_budgets:
                        raise
                    attempted_token_budgets.add(current_tokens)
                lowered = lower_token_budget_in_call_kwargs(enriched)
                if lowered is not None:
                    enriched = lowered
                    continue
            raise
