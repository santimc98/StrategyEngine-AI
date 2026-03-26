import os
from typing import Any, Dict

from dotenv import load_dotenv

from src.utils.retries import call_with_retries

load_dotenv()


class FailureExplainerAgent:
    """
    Explains runtime failures using code + traceback + context.
    Returns a short, plain-text diagnosis to feed back into the next attempt.
    Uses OpenRouter as LLM provider.
    """

    def __init__(self, api_key: Any = None):
        self._client = None
        self._model_name = None

        openrouter_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if openrouter_key:
            try:
                from openai import OpenAI

                self._model_name = os.getenv(
                    "FAILURE_EXPLAINER_MODEL", "google/gemini-3-flash-preview"
                )
                self._client = OpenAI(
                    api_key=openrouter_key,
                    base_url="https://openrouter.ai/api/v1",
                )
            except Exception:
                self._client = None

    def _call_llm(self, prompt: str) -> str:
        """Call OpenRouter and return the text response."""
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2048,
        )
        return (response.choices[0].message.content or "").strip()

    def _explain_failure(
        self,
        code: str,
        error_details: str,
        context: Dict[str, Any] | None,
        *,
        prompt_prefix: str,
    ) -> str:
        if not code or not error_details:
            return ""
        if not self._client:
            return self._fallback(error_details)

        ctx = context or {}
        code_snippet = self._truncate(code, 6000)
        error_snippet = self._truncate(error_details, 4000)
        context_snippet = self._truncate(str(ctx), 2000)
        prompt = (
            prompt_prefix
            + "\n\nCODE:\n"
            + code_snippet
            + "\n\nERROR:\n"
            + error_snippet
            + "\n\nCONTEXT:\n"
            + context_snippet
            + "\n"
        )

        try:
            content = call_with_retries(lambda: self._call_llm(prompt), max_retries=2)
        except Exception:
            return self._fallback(error_details)

        return (content or "").strip()

    def explain_data_engineer_failure(
        self,
        code: str,
        error_details: str,
        context: Dict[str, Any] | None = None,
    ) -> str:
        return self._explain_failure(
            code,
            error_details,
            context,
            prompt_prefix=(
                "You are a senior debugging assistant. "
                "Given the generated Python cleaning code, the traceback/error, and context, "
                "explain why the failure happened and how to fix it. "
                "Return concise plain text (3-6 short lines). "
                "Use this format with short lines: "
                "WHERE: <location or step>, WHY: <root cause>, FIX: <specific change>. "
                "Prioritize the earliest root cause, not just the final exception. "
                "Be concrete about the coding invariant that was violated (shape/length mismatch, "
                "missing columns, incorrect file path, stale artifacts, wrong import, etc.). "
                "If uncertain, propose a minimal diagnostic check to confirm the cause. "
                "Do NOT include code. Do NOT restate the full traceback."
            ),
        )

    def explain_ml_failure(
        self,
        code: str,
        error_details: str,
        context: Dict[str, Any] | None = None,
    ) -> str:
        return self._explain_failure(
            code,
            error_details,
            context,
            prompt_prefix=(
                "You are a senior ML debugging assistant. "
                "Given the generated ML Python code, the runtime error output, and context, "
                "explain why the failure happened and how to fix it. "
                "Return concise plain text (3-6 short lines). "
                "Use this format with short lines: "
                "WHERE: <location or step>, WHY: <root cause>, FIX: <specific change>. "
                "Prioritize the earliest root cause, not just the final exception. "
                "Name the violated invariant (mismatched shapes/lengths, pipeline refit side effects, "
                "missing column, wrong import, wrong file path, or derived field not created). "
                "If multiple errors appear, address the first causal one. "
                "If uncertain, propose a minimal diagnostic check to confirm the cause. "
                "Do NOT include code. Do NOT restate the full traceback."
            ),
        )

    def _fallback(self, error_details: str) -> str:
        if not error_details:
            return ""
        return f"Automated diagnosis unavailable. Raw error summary: {error_details[:500]}"

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        if not text:
            return ""
        if len(text) <= limit:
            return text
        head = text[: limit // 2]
        tail = text[-(limit // 2) :]
        return f"{head}\n...[truncated]...\n{tail}"
