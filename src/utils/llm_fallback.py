import logging
from typing import Iterable, Tuple, Any, Dict, List

from src.utils.openrouter_reasoning import (
    create_chat_completion_with_reasoning,
)


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        parts = [_coerce_text(item) for item in value]
        parts = [part for part in parts if part]
        return "\n".join(parts).strip()
    if isinstance(value, dict):
        parts: List[str] = []
        for key in ("text", "content", "output_text", "reasoning", "reasoning_content", "value"):
            text = _coerce_text(value.get(key))
            if text:
                parts.append(text)
        if parts:
            return "\n".join(parts).strip()
        return ""
    for attr in ("text", "content", "output_text", "reasoning", "reasoning_content", "value"):
        if hasattr(value, attr):
            text = _coerce_text(getattr(value, attr))
            if text:
                return text
    return ""


def extract_response_text(response: Any) -> str:
    if response is None:
        return ""
    choices = getattr(response, "choices", None)
    if not choices:
        return ""
    first = choices[0]
    msg = getattr(first, "message", None)
    if msg is None:
        return ""

    # Some providers return structured content blocks instead of plain strings.
    content = _coerce_text(getattr(msg, "content", None))
    if content:
        return content

    # Fallback for providers exposing reasoning in separate fields.
    reasoning = _coerce_text(getattr(msg, "reasoning", None))
    if reasoning:
        return reasoning
    reasoning_content = _coerce_text(getattr(msg, "reasoning_content", None))
    if reasoning_content:
        return reasoning_content
    return ""


def _response_has_content(response: Any) -> bool:
    if extract_response_text(response):
        return True
    choices = getattr(response, "choices", None)
    if not choices:
        return False
    first = choices[0]
    msg = getattr(first, "message", None)
    if msg is None:
        return False
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        return True
    return False


def _empty_completion_detail(response: Any) -> str:
    try:
        choices = getattr(response, "choices", None) or []
        finish_reason = getattr(choices[0], "finish_reason", None) if choices else None
    except Exception:
        finish_reason = None
    usage = getattr(response, "usage", None)
    completion_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None
    prompt_tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
    detail = f"finish_reason={finish_reason} completion_tokens={completion_tokens} prompt_tokens={prompt_tokens}"
    return detail


def call_chat_with_fallback(
    llm_client: Any,
    messages: List[Dict[str, str]],
    model_chain: Iterable[str],
    *,
    call_kwargs: Dict[str, Any],
    logger: logging.Logger | None,
    context_tag: str,
) -> Tuple[Any, str]:
    last_exc: Exception | None = None
    for model in model_chain:
        if not model:
            continue
        try:
            response = create_chat_completion_with_reasoning(
                llm_client,
                agent_name=context_tag,
                model_name=model,
                call_kwargs={
                    "model": model,
                    "messages": messages,
                    **(call_kwargs or {}),
                },
            )
            if not _response_has_content(response):
                detail = _empty_completion_detail(response)
                raise ValueError(f"EMPTY_COMPLETION {detail}")
            return response, model
        except Exception as exc:  # pragma: no cover - safety net
            last_exc = exc
            if logger:
                logger.warning(
                    "LLM_FALLBACK_WARNING context=%s model=%s error=%s message=%s",
                    context_tag,
                    model,
                    type(exc).__name__,
                    str(exc)[:200],
                )
            else:
                print(
                    f"LLM_FALLBACK_WARNING context={context_tag} model={model} "
                    f"error={type(exc).__name__} message={str(exc)[:200]}"
                )
            continue
    if last_exc is not None:
        raise last_exc
    raise ValueError("No models provided for fallback.")
