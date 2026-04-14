"""Central model-routing defaults for enterprise/demo runs.

The map is intentionally slot-based: prompts stay generic, while each agent
subtask can be routed to the model family best suited to that responsibility.
Runtime settings and environment variables can still override every slot.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, Mapping


RECOMMENDED_AGENT_MODEL_DEFAULTS: Mapping[str, str] = {
    "steward": "google/gemini-3.1-pro-preview",
    "steward_semantics": "google/gemini-3.1-pro-preview",
    "strategist": "anthropic/claude-opus-4.6",
    "strategist_fallback": "anthropic/claude-sonnet-4.6",
    "execution_planner": "anthropic/claude-opus-4.6",
    "execution_planner_compiler": "anthropic/claude-sonnet-4.6",
    "data_engineer": "anthropic/claude-opus-4.6",
    "data_engineer_editor": "anthropic/claude-sonnet-4.6",
    "data_engineer_fallback": "anthropic/claude-sonnet-4.6",
    "ml_engineer_plan": "google/gemini-3.1-pro-preview",
    "ml_engineer": "anthropic/claude-opus-4.6",
    "ml_engineer_editor": "anthropic/claude-sonnet-4.6",
    "ml_engineer_fallback": "anthropic/claude-sonnet-4.6",
    "model_analyst": "anthropic/claude-opus-4.6",
    "cleaning_reviewer": "google/gemini-3.1-pro-preview",
    "reviewer": "anthropic/claude-sonnet-4.6",
    "qa_reviewer": "anthropic/claude-sonnet-4.6",
    "review_board": "anthropic/claude-opus-4.6",
    "translator": "anthropic/claude-opus-4.6",
    "translator_repair": "openai/gpt-5.4-mini",
    "results_advisor": "anthropic/claude-sonnet-4.6",
    "results_advisor_critique": "openai/gpt-5.4",
    "results_advisor_llm": "anthropic/claude-sonnet-4.6",
    "failure_explainer": "openai/gpt-5.4-mini",
}


MODEL_SLOT_ENV_VARS: Mapping[str, Iterable[str]] = {
    "steward": ("STEWARD_MODEL",),
    "steward_semantics": ("STEWARD_SEMANTICS_MODEL",),
    "strategist": ("STRATEGIST_MODEL", "OPENROUTER_STRATEGIST_PRIMARY_MODEL"),
    "strategist_fallback": ("STRATEGIST_FALLBACK_MODEL", "OPENROUTER_STRATEGIST_FALLBACK_MODEL"),
    "execution_planner": ("EXECUTION_PLANNER_PRIMARY_MODEL", "EXECUTION_PLANNER_MODEL"),
    "execution_planner_compiler": ("EXECUTION_PLANNER_COMPILER_MODEL",),
    "data_engineer": ("OPENROUTER_DE_PRIMARY_MODEL",),
    "data_engineer_editor": ("OPENROUTER_DE_EDITOR_MODEL",),
    "data_engineer_fallback": ("OPENROUTER_DE_FALLBACK_MODEL",),
    "ml_engineer_plan": ("OPENROUTER_ML_PLAN_MODEL",),
    "ml_engineer": ("OPENROUTER_ML_PRIMARY_MODEL",),
    "ml_engineer_editor": ("OPENROUTER_ML_EDITOR_MODEL",),
    "ml_engineer_fallback": ("OPENROUTER_ML_FALLBACK_MODEL",),
    "model_analyst": ("OPENROUTER_MODEL_ANALYST_PRIMARY_MODEL",),
    "cleaning_reviewer": ("CLEANING_REVIEWER_MODEL", "REVIEWER_MODEL"),
    "reviewer": ("REVIEWER_MODEL",),
    "qa_reviewer": ("QA_REVIEWER_MODEL", "REVIEWER_MODEL"),
    "review_board": ("REVIEW_BOARD_MODEL", "REVIEWER_MODEL"),
    "translator": ("TRANSLATOR_MODEL",),
    "translator_repair": ("TRANSLATOR_REPAIR_MODEL",),
    "results_advisor": ("RESULTS_ADVISOR_MODEL",),
    "results_advisor_critique": ("RESULTS_ADVISOR_CRITIQUE_MODEL",),
    "results_advisor_llm": ("RESULTS_ADVISOR_LLM_MODEL",),
    "failure_explainer": ("FAILURE_EXPLAINER_MODEL",),
}


def get_recommended_agent_model_defaults() -> Dict[str, str]:
    return {key: str(value).strip() for key, value in RECOMMENDED_AGENT_MODEL_DEFAULTS.items() if str(value).strip()}


def get_env_agent_model_overrides() -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    for slot, env_names in MODEL_SLOT_ENV_VARS.items():
        for env_name in env_names:
            value = str(os.getenv(env_name) or "").strip()
            if value:
                overrides[slot] = value
                break
    return overrides


def build_initial_agent_model_defaults() -> Dict[str, str]:
    """Recommended routing, with explicit environment configuration taking precedence."""

    defaults = get_recommended_agent_model_defaults()
    defaults.update(get_env_agent_model_overrides())
    return defaults
