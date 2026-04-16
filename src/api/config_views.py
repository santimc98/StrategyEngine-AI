from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils.paths import DATA_DIR as _DATA_DIR
from src.utils.sandbox_config import (
    get_execution_backend_config,
    load_sandbox_config,
    normalize_sandbox_config,
    save_sandbox_config,
)
from src.utils.sandbox_provider import (
    get_sandbox_provider_spec,
    is_sandbox_provider_available,
    list_sandbox_providers,
    test_sandbox_provider_connectivity,
)
from src.utils.model_routing_defaults import get_recommended_agent_model_defaults

MODEL_SETTING_SPECS: List[Dict[str, str]] = [
    {"key": "steward", "label": "Steward", "section": "primary"},
    {"key": "strategist", "label": "Strategist", "section": "primary"},
    {"key": "execution_planner", "label": "Execution Planner", "section": "primary"},
    {"key": "data_engineer", "label": "Data Engineer", "section": "primary"},
    {"key": "ml_engineer", "label": "ML Engineer", "section": "primary"},
    {"key": "model_analyst", "label": "Model Analyst", "section": "primary"},
    {"key": "cleaning_reviewer", "label": "Cleaning Reviewer", "section": "primary"},
    {"key": "reviewer", "label": "Reviewer", "section": "primary"},
    {"key": "qa_reviewer", "label": "QA Reviewer", "section": "primary"},
    {"key": "review_board", "label": "Review Board", "section": "primary"},
    {"key": "translator", "label": "Business Translator", "section": "primary"},
    {"key": "results_advisor", "label": "Results Advisor", "section": "primary"},
    {"key": "failure_explainer", "label": "Failure Explainer", "section": "primary"},
    {"key": "steward_semantics", "label": "Steward Semantics", "section": "advanced"},
    {"key": "strategist_fallback", "label": "Strategist Fallback", "section": "advanced"},
    {"key": "execution_planner_compiler", "label": "Execution Planner Compiler", "section": "advanced"},
    {"key": "data_engineer_plan", "label": "Data Engineer Plan", "section": "advanced"},
    {"key": "data_engineer_editor", "label": "Data Engineer Editor", "section": "advanced"},
    {"key": "data_engineer_fallback", "label": "Data Engineer Fallback", "section": "advanced"},
    {"key": "ml_engineer_plan", "label": "ML Engineer Plan", "section": "advanced"},
    {"key": "ml_engineer_editor", "label": "ML Engineer Editor", "section": "advanced"},
    {"key": "ml_engineer_fallback", "label": "ML Engineer Fallback", "section": "advanced"},
    {"key": "translator_repair", "label": "Business Translator Repair", "section": "advanced"},
    {"key": "results_advisor_critique", "label": "Results Advisor Critique", "section": "advanced"},
    {"key": "results_advisor_llm", "label": "Results Advisor LLM", "section": "advanced"},
]

MODEL_SETTING_LABELS: Dict[str, str] = {
    spec["key"]: spec["label"] for spec in MODEL_SETTING_SPECS
}
PRIMARY_MODEL_KEYS: List[str] = [
    spec["key"] for spec in MODEL_SETTING_SPECS if spec["section"] == "primary"
]
ADVANCED_MODEL_KEYS: List[str] = [
    spec["key"] for spec in MODEL_SETTING_SPECS if spec["section"] == "advanced"
]

MODEL_PRESET_OPTIONS: List[Tuple[str, str]] = [
    ("z-ai/glm-5", "GLM-5"),
    ("moonshotai/kimi-k2.5", "Kimi K2.5"),
    ("minimax/minimax-m2.5", "Minimax M-2.5"),
    ("minimax/minimax-m2.7", "Minimax M-2.7"),
    ("deepseek/deepseek-chat-v3.2", "DeepSeek V3.2"),
    ("anthropic/claude-opus-4.6", "Claude Opus 4.6"),
    ("anthropic/claude-sonnet-4.6", "Claude Sonnet 4.6"),
    ("openai/chatgpt-5.2", "ChatGPT 5.2"),
    ("openai/gpt-5.3-codex", "GPT-5.3 Codex"),
    ("openai/gpt-5.4-mini", "GPT-5.4 Mini"),
    ("openai/gpt-5.4-nano", "GPT-5.4 Nano"),
    ("openai/gpt-5.4", "GPT-5.4"),
    ("google/gemini-3-flash-preview", "Gemini 3 Flash Preview"),
    ("google/gemini-3.1-pro-preview", "Gemini 3.1 Pro Preview"),
]

CUSTOM_MODEL_OPTION = "__custom_model__"
MODEL_OVERRIDES_PATH = Path(_DATA_DIR) / "agent_model_overrides.json"


def _load_runtime_model_hooks():
    try:
        from src.graph.graph import get_runtime_agent_models, set_runtime_agent_models

        return get_runtime_agent_models, set_runtime_agent_models, None
    except Exception as exc:
        return None, None, str(exc)


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _sanitize_agent_model_map(raw: Any) -> Dict[str, str]:
    cleaned: Dict[str, str] = {}
    if not isinstance(raw, dict):
        return cleaned
    for agent_key in MODEL_SETTING_LABELS:
        value = str(raw.get(agent_key) or "").strip()
        if value:
            cleaned[agent_key] = value
    return cleaned


def _merge_agent_model_maps(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, str]:
    merged: Dict[str, str] = {}
    base_map = _sanitize_agent_model_map(base)
    overrides_map = _sanitize_agent_model_map(overrides)
    for agent_key in MODEL_SETTING_LABELS:
        merged[agent_key] = overrides_map.get(agent_key) or base_map.get(agent_key) or ""
    return merged


def load_agent_model_overrides() -> Dict[str, str]:
    return _sanitize_agent_model_map(_load_json(MODEL_OVERRIDES_PATH))


def save_agent_model_overrides(overrides: Dict[str, Any]) -> Dict[str, str]:
    payload = _sanitize_agent_model_map(overrides)
    os.makedirs(MODEL_OVERRIDES_PATH.parent, exist_ok=True)
    with MODEL_OVERRIDES_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return payload


def _sandbox_status_summary(config: Dict[str, Any]) -> Dict[str, str]:
    spec = get_sandbox_provider_spec(config.get("provider"))
    if is_sandbox_provider_available(spec.name):
        detail = "Disponible"
        color = "#a6e3a1"
        severity = "ok"
    else:
        detail = "Pendiente de backend"
        color = "#f9e2af"
        severity = "warning"
    return {
        "label": spec.label,
        "detail": detail,
        "color": color,
        "severity": severity,
    }


def _execution_backend_status_summary(config: Dict[str, Any]) -> Dict[str, str]:
    backend = get_execution_backend_config(config)
    mode = str(backend.get("mode") or "cloudrun").strip().lower() or "cloudrun"
    if mode == "local":
        return {
            "label": "Local Runner",
            "detail": "Activo",
            "color": "#a6e3a1",
            "severity": "ok",
        }
    enabled = bool(backend.get("cloudrun_enabled"))
    has_required = all(str(backend.get(key) or "").strip() for key in ("job", "region", "bucket"))
    if enabled and has_required:
        detail = "Configurado"
        color = "#a6e3a1"
        severity = "ok"
    elif enabled or has_required:
        detail = "Incompleto"
        color = "#f9e2af"
        severity = "warning"
    else:
        detail = "Sin configurar"
        color = "#f38ba8"
        severity = "error"
    return {
        "label": "Cloud Run",
        "detail": detail,
        "color": color,
        "severity": severity,
    }


def get_model_settings_view() -> Dict[str, Any]:
    get_models, _, bootstrap_error = _load_runtime_model_hooks()
    base_models = _sanitize_agent_model_map(get_models()) if callable(get_models) else {}
    persisted = load_agent_model_overrides()
    recommended = _sanitize_agent_model_map(get_recommended_agent_model_defaults())
    effective = _merge_agent_model_maps(base_models, persisted)
    return {
        "runtime_available": callable(get_models),
        "bootstrap_error": bootstrap_error,
        "presets": [{"id": model_id, "label": label} for model_id, label in MODEL_PRESET_OPTIONS],
        "custom_option": CUSTOM_MODEL_OPTION,
        "agents": MODEL_SETTING_SPECS,
        "recommended_models": recommended,
        "base_models": base_models,
        "persisted_models": persisted,
        "effective_models": effective,
        "primary_models": {key: effective.get(key, "") for key in PRIMARY_MODEL_KEYS if effective.get(key)},
        "advanced_models": {key: effective.get(key, "") for key in ADVANCED_MODEL_KEYS if effective.get(key)},
    }


def apply_and_persist_model_settings(models: Dict[str, Any]) -> Dict[str, Any]:
    payload = _sanitize_agent_model_map(models)
    get_models, set_models, bootstrap_error = _load_runtime_model_hooks()
    if callable(set_models):
        base_models = _sanitize_agent_model_map(get_models())
        merged = _merge_agent_model_maps(base_models, payload)
        applied = _sanitize_agent_model_map(set_models(merged))
        persisted = save_agent_model_overrides(_merge_agent_model_maps(base_models, applied))
        return {
            "runtime_available": True,
            "bootstrap_error": None,
            "saved_models": persisted,
            "effective_models": _merge_agent_model_maps(base_models, persisted),
        }

    persisted = save_agent_model_overrides(payload)
    return {
        "runtime_available": False,
        "bootstrap_error": bootstrap_error,
        "saved_models": persisted,
        "effective_models": persisted,
    }


def reset_model_settings() -> Dict[str, Any]:
    get_models, set_models, bootstrap_error = _load_runtime_model_hooks()
    if callable(get_models) and callable(set_models):
        base_models = _sanitize_agent_model_map(get_models())
        applied = _sanitize_agent_model_map(set_models(base_models))
        persisted = save_agent_model_overrides(_merge_agent_model_maps(base_models, applied))
        return {
            "runtime_available": True,
            "bootstrap_error": None,
            "saved_models": persisted,
            "effective_models": _merge_agent_model_maps(base_models, persisted),
        }

    if MODEL_OVERRIDES_PATH.exists():
        MODEL_OVERRIDES_PATH.unlink()
    return {
        "runtime_available": False,
        "bootstrap_error": bootstrap_error,
        "saved_models": {},
        "effective_models": {},
    }


def get_sandbox_settings_view() -> Dict[str, Any]:
    config = load_sandbox_config()
    normalized = normalize_sandbox_config(config)
    provider = str(normalized.get("provider") or "local").strip().lower() or "local"
    provider_spec = get_sandbox_provider_spec(provider)
    provider_status = _sandbox_status_summary(normalized)
    backend_status = _execution_backend_status_summary(normalized)
    connectivity_ok, connectivity_message = test_sandbox_provider_connectivity(
        provider,
        normalized.get("settings") if isinstance(normalized.get("settings"), dict) else {},
    )

    providers_payload = []
    for spec in list_sandbox_providers():
        providers_payload.append(
            {
                "name": spec.name,
                "label": spec.label,
                "description": spec.description,
                "implemented": spec.implemented,
                "available": is_sandbox_provider_available(spec.name),
                "config_fields": [
                    {
                        "key": field.key,
                        "label": field.label,
                        "description": field.description,
                        "placeholder": field.placeholder,
                        "secret": field.secret,
                        "required": field.required,
                    }
                    for field in spec.config_fields
                ],
            }
        )

    return {
        "config": normalized,
        "provider": provider,
        "provider_spec": {
            "name": provider_spec.name,
            "label": provider_spec.label,
            "description": provider_spec.description,
            "implemented": provider_spec.implemented,
        },
        "provider_status": provider_status,
        "execution_backend": get_execution_backend_config(normalized),
        "execution_backend_status": backend_status,
        "provider_connectivity": {
            "ok": bool(connectivity_ok),
            "message": connectivity_message,
        },
        "providers": providers_payload,
    }


def save_sandbox_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    normalized = normalize_sandbox_config(config)
    save_sandbox_config(normalized)
    return get_sandbox_settings_view()
