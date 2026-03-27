import streamlit as st
import pandas as pd
import os
import json
import subprocess
import sys
import time
import glob
import signal
import threading
import io
import uuid as _uuid_mod
import zipfile
from datetime import datetime
from typing import Any, Dict, List, Tuple

# Ensure src is in path
APP_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(APP_ROOT)

from src.utils.paths import PROJECT_ROOT as _PROJECT_ROOT_PATHS

from src.graph.graph import (
    app_graph,
    request_abort,
    clear_abort,
    get_runtime_agent_models,
    set_runtime_agent_models,
)
from src.utils.run_workspace import recover_orphaned_workspace_cwd
from src.utils.run_status import (
    get_active_run_id as _get_active_run_id,
    is_process_alive as _is_process_alive,
    kill_worker as _kill_worker,
    read_final_state as _read_final_state,
    read_log_entries as _read_log_entries,
    read_status as _read_run_status,
    request_run_abort as _request_run_abort,
    write_worker_input as _write_worker_input,
)
from src.utils.api_keys_store import (
    API_KEY_REGISTRY,
    apply_keys_to_env,
    load_keys as _load_api_keys,
    mask_key as _mask_api_key,
    save_keys as _save_api_keys,
    test_key_connectivity as _test_api_key,
)
from src.utils.sandbox_config import (
    get_execution_backend_config as _get_execution_backend_config,
    load_sandbox_config as _load_sandbox_config,
    mask_sandbox_secret as _mask_sandbox_secret,
    merge_execution_backend_config as _merge_execution_backend_config,
    normalize_sandbox_config as _normalize_sandbox_config,
    normalize_execution_backend_config as _normalize_execution_backend_config,
    save_sandbox_config as _save_sandbox_config,
)
import src.utils.sandbox_provider as _sandbox_provider
from src.utils.run_history import list_runs as _list_runs, load_run_result as _load_run_result

# Auto-heal cwd when prior run crashed inside runs/<run_id>/work.
recover_orphaned_workspace_cwd(project_root=APP_ROOT)
try:
    os.chdir(APP_ROOT)
except Exception as cwd_err:
    print(f"APP_CWD_WARNING: {cwd_err}")

# Apply stored API keys to environment on startup
apply_keys_to_env()

_SIGNAL_HANDLER_INSTALLED = False

_get_sandbox_provider_spec = getattr(
    _sandbox_provider,
    "get_sandbox_provider_spec",
    lambda provider=None: type(
        "SandboxProviderSpecFallback",
        (),
        {
            "name": str(provider or "local").strip().lower() or "local",
            "label": str(provider or "local").strip().title() or "Local",
            "description": "Metadata de sandbox no disponible en este despliegue.",
            "implemented": False,
            "config_fields": (),
        },
    )(),
)
_is_sandbox_provider_available = getattr(
    _sandbox_provider,
    "is_sandbox_provider_available",
    lambda provider=None: str(provider or "local").strip().lower() in {"", "local", "default"},
)
_list_sandbox_providers = getattr(
    _sandbox_provider,
    "list_sandbox_providers",
    lambda: [_get_sandbox_provider_spec("local")],
)
_test_sandbox_provider_connectivity = getattr(
    _sandbox_provider,
    "test_sandbox_provider_connectivity",
    lambda provider=None, settings=None: (
        _is_sandbox_provider_available(provider),
        "Sandbox local disponible" if _is_sandbox_provider_available(provider) else "Provider no disponible",
    ),
)

MODEL_SETTING_SPECS: List[Dict[str, str]] = [
    {"key": "steward", "label": "Steward", "section": "primary"},
    {"key": "strategist", "label": "Strategist", "section": "primary"},
    {"key": "execution_planner", "label": "Execution Planner", "section": "primary"},
    {"key": "data_engineer", "label": "Data Engineer", "section": "primary"},
    {"key": "ml_engineer", "label": "ML Engineer", "section": "primary"},
    {"key": "cleaning_reviewer", "label": "Cleaning Reviewer", "section": "primary"},
    {"key": "reviewer", "label": "Reviewer", "section": "primary"},
    {"key": "qa_reviewer", "label": "QA Reviewer", "section": "primary"},
    {"key": "review_board", "label": "Review Board", "section": "primary"},
    {"key": "translator", "label": "Business Translator", "section": "primary"},
    {"key": "results_advisor", "label": "Results Advisor", "section": "primary"},
    {"key": "failure_explainer", "label": "Failure Explainer", "section": "primary"},
    {"key": "strategist_fallback", "label": "Strategist Fallback", "section": "advanced"},
    {"key": "execution_planner_compiler", "label": "Execution Planner Compiler", "section": "advanced"},
    {"key": "data_engineer_fallback", "label": "Data Engineer Fallback", "section": "advanced"},
    {"key": "ml_engineer_editor", "label": "ML Engineer Editor", "section": "advanced"},
    {"key": "ml_engineer_fallback", "label": "ML Engineer Fallback", "section": "advanced"},
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
    ("openai/chatgpt-5.2", "ChatGPT 5.2"),
    ("openai/gpt-5.3-codex", "GPT-5.3 Codex"),
    ("openai/gpt-5.4-mini", "GPT-5.4 Mini"),
    ("openai/gpt-5.4-nano", "GPT-5.4 Nano"),
    ("openai/gpt-5.4", "GPT-5.4"),
    ("google/gemini-3-flash-preview", "Gemini 3 Flash Preview"),
    ("google/gemini-3.1-pro-preview", "Gemini 3.1 Pro Preview"),
]
CUSTOM_MODEL_OPTION = "__custom_model__"
MODEL_OVERRIDES_PATH = os.path.join(APP_ROOT, "data", "agent_model_overrides.json")
MODEL_PRESET_LABELS: Dict[str, str] = {model_id: label for model_id, label in MODEL_PRESET_OPTIONS}

def _load_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _is_glob_pattern(path: str) -> bool:
    return any(ch in path for ch in ("*", "?", "["))


def _build_artifact_roots(result: Dict[str, Any]) -> List[str]:
    roots: List[str] = [APP_ROOT]

    for key in ("work_dir_abs", "work_dir"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            roots.append(value.strip())

    run_id = str(result.get("run_id") or "").strip()
    if run_id:
        roots.extend(
            [
                os.path.join(APP_ROOT, "runs", run_id, "work"),
                os.path.join(APP_ROOT, "runs", run_id, "artifacts"),
                os.path.join(APP_ROOT, "runs", run_id, "report"),
                os.path.join(APP_ROOT, "runs", run_id, "sandbox"),
            ]
        )

    unique_roots: List[str] = []
    seen = set()
    for root in roots:
        try:
            abs_root = os.path.abspath(root)
        except Exception:
            continue
        if abs_root in seen:
            continue
        seen.add(abs_root)
        unique_roots.append(abs_root)
    return unique_roots


def _resolve_ml_artifact_files(output_report: Dict[str, Any], result: Dict[str, Any]) -> List[Tuple[str, str]]:
    present_outputs = output_report.get("present", []) if isinstance(output_report, dict) else []
    if not present_outputs and isinstance(result.get("artifact_paths"), list):
        present_outputs = [p for p in result.get("artifact_paths", []) if isinstance(p, str)]

    roots = _build_artifact_roots(result if isinstance(result, dict) else {})
    resolved: List[Tuple[str, str]] = []
    seen_files = set()

    def _register(path: str, arcname_hint: str) -> None:
        try:
            abs_path = os.path.abspath(path)
        except Exception:
            return
        if not os.path.isfile(abs_path):
            return
        if abs_path in seen_files:
            return
        seen_files.add(abs_path)
        arcname = str(arcname_hint or os.path.basename(abs_path)).replace("\\", "/").lstrip("./")
        if not arcname or arcname.startswith("../"):
            arcname = os.path.basename(abs_path)
        resolved.append((abs_path, arcname))

    for raw in present_outputs:
        if not isinstance(raw, str) or not raw.strip():
            continue
        rel_path = raw.strip()
        rel_norm = rel_path.replace("\\", "/")

        if os.path.isabs(rel_path):
            if _is_glob_pattern(rel_path):
                for match in glob.glob(rel_path):
                    _register(match, os.path.basename(match))
            else:
                _register(rel_path, os.path.basename(rel_path))
            continue

        matched = False
        for root in roots:
            candidate = os.path.join(root, rel_path)
            if _is_glob_pattern(rel_path):
                matches = glob.glob(candidate)
                if matches:
                    matched = True
                for match in matches:
                    try:
                        arcname = os.path.relpath(match, root)
                    except Exception:
                        arcname = os.path.basename(match)
                    _register(match, arcname)
            else:
                if os.path.isfile(candidate):
                    matched = True
                    _register(candidate, rel_norm)

        if not matched:
            if _is_glob_pattern(rel_path):
                for match in glob.glob(rel_path):
                    _register(match, os.path.basename(match))
            else:
                _register(rel_path, rel_norm)

    return resolved


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


def _load_agent_model_overrides() -> Dict[str, str]:
    return _sanitize_agent_model_map(_load_json(MODEL_OVERRIDES_PATH))


def _save_agent_model_overrides(overrides: Dict[str, Any]) -> None:
    payload = _sanitize_agent_model_map(overrides)
    os.makedirs(os.path.dirname(MODEL_OVERRIDES_PATH), exist_ok=True)
    with open(MODEL_OVERRIDES_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _model_option_label(option_id: str) -> str:
    if option_id == CUSTOM_MODEL_OPTION:
        return "Personalizado (ID de modelo)"
    return MODEL_PRESET_LABELS.get(option_id, option_id)


def _init_runtime_model_settings() -> None:
    if "base_agent_models" not in st.session_state:
        st.session_state["base_agent_models"] = _sanitize_agent_model_map(get_runtime_agent_models())

    if "agent_model_overrides" not in st.session_state:
        persisted_overrides = _load_agent_model_overrides()
        st.session_state["agent_model_overrides"] = _merge_agent_model_maps(
            st.session_state["base_agent_models"],
            persisted_overrides,
        )

    if "show_model_settings" not in st.session_state:
        st.session_state["show_model_settings"] = False

    applied_models = _sanitize_agent_model_map(
        set_runtime_agent_models(st.session_state.get("agent_model_overrides", {}))
    )
    st.session_state["agent_model_overrides"] = _merge_agent_model_maps(
        st.session_state.get("base_agent_models", {}),
        applied_models,
    )


def _sandbox_status_summary(config: Dict[str, Any]) -> tuple[str, str, str]:
    spec = _get_sandbox_provider_spec(config.get("provider"))
    if _is_sandbox_provider_available(spec.name):
        color = "#a6e3a1"
        detail = "Disponible"
    else:
        color = "#f9e2af"
        detail = "Pendiente de backend"
    return spec.label, detail, color


def _execution_backend_status_summary(config: Dict[str, Any]) -> tuple[str, str, str]:
    backend = _get_execution_backend_config(config)
    mode = str(backend.get("mode") or "cloudrun").strip().lower() or "cloudrun"
    if mode == "local":
        return "Local Runner", "Activo", "#a6e3a1"
    enabled = bool(backend.get("cloudrun_enabled"))
    has_required = all(str(backend.get(key) or "").strip() for key in ("job", "region", "bucket"))
    if enabled and has_required:
        return "Cloud Run", "Configurado", "#a6e3a1"
    if enabled or has_required:
        return "Cloud Run", "Incompleto", "#f9e2af"
    return "Cloud Run", "Sin configurar", "#f38ba8"


def _handle_shutdown(signum, frame):
    request_abort(f"signal={signum}")
    raise KeyboardInterrupt

def _install_signal_handlers():
    global _SIGNAL_HANDLER_INSTALLED
    if _SIGNAL_HANDLER_INSTALLED:
        return
    if threading.current_thread() is not threading.main_thread():
        return
    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)
    _SIGNAL_HANDLER_INSTALLED = True

_install_signal_handlers()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="StrategyEngine AI",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

_init_runtime_model_settings()

# ---------------------------------------------------------------------------
# Professional CSS Design System
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ---------- Base & Variables ---------- */
:root {
    --bg-dark: #0e1117;
    --bg-content: #fafbfc;
    --accent: #4F8BF9;
    --accent-dark: #3a6fd8;
    --success: #28a745;
    --warning: #f0ad4e;
    --danger: #dc3545;
    --text-primary: #1a1a2e;
    --text-secondary: #6c757d;
    --card-bg: #ffffff;
    --card-border: #e9ecef;
    --radius: 12px;
    --shadow: 0 2px 12px rgba(0,0,0,.08);
    --shadow-hover: 0 4px 20px rgba(0,0,0,.12);
}

/* ---------- Hide Streamlit defaults ---------- */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent;}

/* ---------- Typography ---------- */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0e1117 0%, #161b22 100%);
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li,
section[data-testid="stSidebar"] .stMarkdown label,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] h5,
section[data-testid="stSidebar"] h6 {
    color: #e6edf3 !important;
}
section[data-testid="stSidebar"] .stTextArea label,
section[data-testid="stSidebar"] .stFileUploader label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stNumberInput label,
section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: #ffffff !important;
}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label p,
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label span {
    color: #8b949e !important;
}

/* ---------- Cards ---------- */
.card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: var(--radius);
    padding: 1.5rem;
    box-shadow: var(--shadow);
    margin-bottom: 1rem;
    transition: box-shadow 0.2s ease;
}
.card:hover {
    box-shadow: var(--shadow-hover);
}
.card-header {
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-secondary);
    margin-bottom: 0.5rem;
}
.card-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--text-primary);
}

/* ---------- Status Badges ---------- */
.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 50px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}
.badge-success { background: #d4edda; color: #155724; }
.badge-progress { background: #cce5ff; color: #004085; }
.badge-warning { background: #fff3cd; color: #856404; }
.badge-error { background: #f8d7da; color: #721c24; }

/* ---------- Pipeline Steps ---------- */
.pipeline-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 0.25rem;
    padding: 1rem 0;
    flex-wrap: wrap;
}
.pipeline-step {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.35rem;
    flex: 1;
    min-width: 80px;
}
.step-icon {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    border: 2px solid #dee2e6;
    background: #f8f9fa;
    color: #adb5bd;
    transition: all 0.3s ease;
}
.step-icon.active {
    border-color: var(--accent);
    background: var(--accent);
    color: white;
    animation: pulse 1.5s infinite;
}
.step-icon.completed {
    border-color: var(--success);
    background: var(--success);
    color: white;
}
.step-label {
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--text-secondary);
    text-align: center;
}
.step-label.active { color: var(--accent); }
.step-label.completed { color: var(--success); }

@keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(79,139,249,.5); }
    70%  { box-shadow: 0 0 0 10px rgba(79,139,249,0); }
    100% { box-shadow: 0 0 0 0 rgba(79,139,249,0); }
}

/* ---------- Activity Log ---------- */
.activity-log {
    background: #1e1e2e;
    border-radius: var(--radius);
    padding: 1rem 1.25rem;
    max-height: 300px;
    overflow-y: auto;
    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
    font-size: 0.82rem;
    line-height: 1.7;
    color: #cdd6f4;
}
.log-entry { margin-bottom: 0.2rem; }
.log-time { color: #6c7086; }
.log-agent { color: #89b4fa; font-weight: 600; }
.log-ok { color: #a6e3a1; }
.log-warn { color: #f9e2af; }

/* ---------- Metric Pills ---------- */
.metric-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: #f0f4ff;
    border: 1px solid #d0daf5;
    border-radius: 50px;
    padding: 0.3rem 0.85rem;
    font-size: 0.8rem;
    font-weight: 500;
    color: #3a5ba0;
    margin-right: 0.5rem;
    margin-bottom: 0.5rem;
}

/* ---------- Hero Section ---------- */
.hero {
    text-align: center;
    padding: 3rem 1rem 2rem;
}
.hero h1 {
    font-size: 2.5rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
}
.hero-gradient {
    background: linear-gradient(135deg, var(--accent) 0%, #7c3aed 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-subtitle {
    font-size: 1.1rem;
    color: var(--text-secondary);
    line-height: 1.6;
    text-align: center;
}

/* ---------- Feature Cards ---------- */
.feature-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: var(--radius);
    padding: 1.75rem;
    text-align: center;
    box-shadow: var(--shadow);
    height: 100%;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.feature-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-hover);
}
.feature-icon {
    font-size: 2.2rem;
    margin-bottom: 0.75rem;
}
.feature-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
}
.feature-desc {
    font-size: 0.88rem;
    color: var(--text-secondary);
    line-height: 1.5;
}

/* ---------- Steps ---------- */
.steps-container {
    display: flex;
    justify-content: center;
    gap: 2rem;
    margin: 2rem 0;
    flex-wrap: wrap;
}
.step-item {
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    max-width: 220px;
}
.step-number {
    background: var(--accent);
    color: white;
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.9rem;
    flex-shrink: 0;
}
.step-text {
    font-size: 0.9rem;
    color: var(--text-secondary);
    line-height: 1.5;
}
.step-text strong {
    color: var(--text-primary);
    display: block;
    margin-bottom: 0.15rem;
}

/* ---------- Results Banner ---------- */
.result-banner {
    border-radius: var(--radius);
    padding: 1.25rem 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.result-banner.success {
    background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    border: 1px solid #b1dfbb;
}
.result-banner.error {
    background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
    border: 1px solid #f1b0b7;
}
.result-banner-icon { font-size: 1.5rem; }
.result-banner-text {
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--text-primary);
}

/* ---------- Winner Card ---------- */
.winner-card {
    background: linear-gradient(135deg, #f0fff4 0%, #e6ffed 100%);
    border: 2px solid var(--success);
    border-radius: var(--radius);
    padding: 1.25rem 1.5rem;
    margin: 1rem 0;
}

/* ---------- Console Output ---------- */
.console-output {
    background: #1e1e2e;
    border-radius: var(--radius);
    padding: 1rem 1.25rem;
    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
    font-size: 0.8rem;
    line-height: 1.6;
    color: #cdd6f4;
    max-height: 500px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-wrap: break-word;
}

/* ---------- Download Buttons ---------- */
.stDownloadButton > button {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dark) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.5rem !important;
    font-weight: 600 !important;
    transition: opacity 0.2s ease !important;
}
.stDownloadButton > button:hover {
    opacity: 0.9 !important;
}

/* ---------- Start Button Override ---------- */
section[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dark) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    padding: 0.65rem 1rem !important;
    font-size: 1rem !important;
    width: 100% !important;
    transition: opacity 0.2s ease !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    opacity: 0.9 !important;
}

/* ---------- Sidebar Settings Panel ---------- */
.sidebar-settings-panel {
    background: linear-gradient(160deg, rgba(79,139,249,0.12) 0%, rgba(8,17,28,0.35) 100%);
    border: 1px solid rgba(79,139,249,0.35);
    border-radius: 12px;
    padding: 0.9rem 0.9rem 0.4rem;
    margin: 0.6rem 0 0.9rem;
}
.sidebar-settings-panel .ssp-title {
    color: #e6edf3;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 0.45rem;
}
.sidebar-settings-panel .ssp-desc {
    color: #b6c2cf;
    font-size: 0.78rem;
    margin-bottom: 0.75rem;
    line-height: 1.4;
}

/* ---------- Fade-in animation ---------- */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.fade-in { animation: fadeIn 0.4s ease-out; }

/* ---------- Progress Header ---------- */
.progress-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem 1.25rem;
    background: linear-gradient(135deg, #f8f9ff 0%, #f0f4ff 100%);
    border: 1px solid #d0daf5;
    border-radius: var(--radius);
    margin-bottom: 0.75rem;
}
.progress-timer {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--text-primary);
}
.progress-timer-icon {
    font-size: 1.1rem;
}
.progress-pct {
    font-size: 1.3rem;
    font-weight: 800;
    color: var(--accent);
}
.progress-stage {
    font-size: 0.8rem;
    color: var(--text-secondary);
    font-weight: 500;
}

/* ---------- Iteration Badge ---------- */
.iter-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
    border: 1px solid #3d3d5c;
    border-radius: 8px;
    padding: 0.4rem 0.85rem;
    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
    font-size: 0.78rem;
    color: #cdd6f4;
    margin-top: 0.5rem;
}
.iter-badge-label { color: #6c7086; }
.iter-badge-value { color: #89b4fa; font-weight: 700; }
.iter-badge-metric { color: #a6e3a1; font-weight: 600; }
.iter-badge-sep { color: #45475a; }

/* ---------- Sidebar Run Status ---------- */
.sidebar-run-status {
    background: linear-gradient(135deg, rgba(79,139,249,0.1) 0%, rgba(124,58,237,0.1) 100%);
    border: 1px solid rgba(79,139,249,0.25);
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
}
.sidebar-run-status .srs-title {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #8b949e;
    margin-bottom: 0.6rem;
}
.sidebar-run-status .srs-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.25rem 0;
    font-size: 0.8rem;
}
.sidebar-run-status .srs-label { color: #8b949e; }
.sidebar-run-status .srs-value { color: #e6edf3; font-weight: 600; }
.sidebar-run-status .srs-step {
    color: #89b4fa;
    font-weight: 700;
    font-size: 0.85rem;
}
.sidebar-run-status .srs-timer {
    color: #a6e3a1;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
}

/* ---------- Chat Input Area (ChatGPT / Claude style) ---------- */
.chat-input-container {
    max-width: 740px;
    margin: 0 auto;
    padding: 0 1rem;
}
.chat-input-box {
    background: var(--card-bg);
    border: 1.5px solid var(--card-border);
    border-radius: 16px;
    padding: 1.25rem 1.5rem 1rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
.chat-input-box:focus-within {
    border-color: var(--accent);
    box-shadow: 0 2px 20px rgba(79,139,249,0.15);
}
.chat-input-box .stTextArea textarea {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
    font-size: 1rem !important;
    line-height: 1.6 !important;
    resize: none !important;
    color: var(--text-primary) !important;
}
.chat-input-box .stTextArea > div > div { border: none !important; }
.chat-input-box .stTextArea label { display: none !important; }
.chat-attach-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 0.5rem;
    padding-top: 0.75rem;
    border-top: 1px solid var(--card-border);
}
.file-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: linear-gradient(135deg, rgba(79,139,249,0.12) 0%, rgba(124,58,237,0.08) 100%);
    border: 1px solid rgba(79,139,249,0.3);
    border-radius: 20px;
    padding: 0.3rem 0.9rem;
    font-size: 0.82rem;
    color: var(--accent);
    font-weight: 600;
}
.file-chip-placeholder {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    color: var(--text-secondary);
    font-size: 0.82rem;
}
.chat-start-btn button {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dark) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 24px !important;
    font-weight: 700 !important;
    padding: 0.6rem 2rem !important;
    font-size: 0.95rem !important;
    transition: opacity 0.2s ease, transform 0.15s ease !important;
}
.chat-start-btn button:hover {
    opacity: 0.9 !important;
    transform: scale(1.02) !important;
}

/* ---------- Sidebar Run History ---------- */
.sidebar-run-item {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px;
    padding: 0.55rem 0.7rem;
    margin-bottom: 0.4rem;
    cursor: pointer;
    transition: background 0.15s ease;
}
.sidebar-run-item:hover {
    background: rgba(79,139,249,0.1);
}
.sidebar-run-title {
    font-size: 0.78rem;
    font-weight: 600;
    color: #e6edf3;
    line-height: 1.35;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.sidebar-run-meta {
    font-size: 0.68rem;
    color: #8b949e;
    margin-top: 0.2rem;
    display: flex;
    gap: 0.5rem;
    align-items: center;
    flex-wrap: wrap;
}
.sidebar-run-metric {
    color: #a6e3a1;
    font-weight: 600;
}
.sidebar-run-objective {
    font-size: 0.66rem;
    color: #7d8590;
    margin-top: 0.2rem;
    line-height: 1.3;
    font-style: italic;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}

/* ---------- Footer ---------- */
.footer {
    text-align: center;
    padding: 2rem 0 1rem;
    color: var(--text-secondary);
    font-size: 0.78rem;
    border-top: 1px solid var(--card-border);
    margin-top: 3rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header (with home button when viewing results)
# ---------------------------------------------------------------------------
_header_col1, _header_col2 = st.columns([5, 1])
with _header_col1:
    st.markdown("""
    <div style="padding: 0.5rem 0 0.25rem;">
        <h1 style="margin:0; font-size:2rem; font-weight:800;">
            <span class="hero-gradient">StrategyEngine AI</span>
        </h1>
        <p style="margin:0.25rem 0 0; color:#6c757d; font-size:0.95rem;">
            Inteligencia Artificial Aut&oacute;noma para Decisiones de Negocio
        </p>
    </div>
    """, unsafe_allow_html=True)
with _header_col2:
    if st.session_state.get("analysis_complete"):
        if st.button("\U0001f3e0 Inicio", key="header_home_btn", use_container_width=True):
            st.session_state["analysis_complete"] = False
            st.session_state["analysis_result"] = None
            st.session_state["dismissed_latest_run"] = True
            st.session_state.pop("viewing_run_id", None)
            st.rerun()
st.markdown("---")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 0.5rem 0 1rem;">
        <span style="font-size:1.6rem; font-weight:800;" class="hero-gradient">StrategyEngine AI</span>
        <br>
        <span style="font-size:0.72rem; color:#8b949e; letter-spacing:0.05em;">v2.0 &bull; Plataforma IA Empresarial</span>
    </div>
    """, unsafe_allow_html=True)

    if st.button("⚙ Ajustes de Modelos", key="toggle_model_settings", use_container_width=True):
        st.session_state["show_model_settings"] = not bool(st.session_state.get("show_model_settings"))

    active_models = _merge_agent_model_maps(
        st.session_state.get("base_agent_models", {}),
        st.session_state.get("agent_model_overrides", {}),
    )
    primary_model_lines = "".join(
        f"<div><strong>{MODEL_SETTING_LABELS[agent_key]}:</strong> {active_models.get(agent_key, 'N/A')}</div>"
        for agent_key in PRIMARY_MODEL_KEYS
        if str(active_models.get(agent_key) or "").strip()
    )
    advanced_model_lines = "".join(
        f"<div><strong>{MODEL_SETTING_LABELS[agent_key]}:</strong> {active_models.get(agent_key, 'N/A')}</div>"
        for agent_key in ADVANCED_MODEL_KEYS
        if str(active_models.get(agent_key) or "").strip()
    )
    active_model_lines = f"<div><strong>Principales</strong></div>{primary_model_lines}"
    if advanced_model_lines:
        active_model_lines += f"<div style='margin-top:0.45rem;'><strong>Avanzados</strong></div>{advanced_model_lines}"
    st.markdown(
        f"""
        <div class="sidebar-settings-panel">
            <div class="ssp-title">Modelos Activos</div>
            <div class="ssp-desc">{active_model_lines}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.get("show_model_settings"):
        st.markdown(
            """
            <div class="sidebar-settings-panel">
                <div class="ssp-title">Configuraci&oacute;n de Modelos del Runtime</div>
                <div class="ssp-desc">La UI es la fuente de verdad del runtime: los cambios se persisten y el worker los carga al arrancar, sin tocar archivos internos.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        pending_models: Dict[str, str] = {}
        preset_ids = [model_id for model_id, _ in MODEL_PRESET_OPTIONS]
        selector_options = preset_ids + [CUSTOM_MODEL_OPTION]

        def _render_model_controls(model_keys: List[str]) -> None:
            for agent_key in model_keys:
                agent_label = MODEL_SETTING_LABELS[agent_key]
                current_model = str(active_models.get(agent_key) or "").strip()
                current_is_preset = current_model in MODEL_PRESET_LABELS
                default_option = current_model if current_is_preset else CUSTOM_MODEL_OPTION
                if default_option not in selector_options:
                    default_option = CUSTOM_MODEL_OPTION
                selected_option = st.selectbox(
                    f"Modelo para {agent_label}",
                    selector_options,
                    index=selector_options.index(default_option),
                    format_func=_model_option_label,
                    key=f"model_option_{agent_key}",
                )
                if selected_option == CUSTOM_MODEL_OPTION:
                    custom_default = "" if current_is_preset else current_model
                    custom_model = st.text_input(
                        f"ID del modelo ({agent_label})",
                        value=custom_default,
                        placeholder="proveedor/nombre-modelo",
                        key=f"model_custom_{agent_key}",
                    )
                    pending_models[agent_key] = str(custom_model or "").strip()
                else:
                    pending_models[agent_key] = selected_option

        st.markdown("**Agentes Principales**")
        _render_model_controls(PRIMARY_MODEL_KEYS)

        with st.expander("Slots avanzados y secundarios", expanded=False):
            st.caption(
                "Incluye compiladores, editores y fallbacks. "
                "Solo toca estos slots si quieres un routing distinto al estandar."
            )
            _render_model_controls(ADVANCED_MODEL_KEYS)

        apply_col, reset_col = st.columns(2)
        with apply_col:
            apply_models_btn = st.button("Guardar y aplicar", key="apply_agent_models", use_container_width=True)
        with reset_col:
            reset_models_btn = st.button("Restablecer", key="reset_agent_models", use_container_width=True)

        if apply_models_btn:
            missing_agents = [MODEL_SETTING_LABELS[k] for k, v in pending_models.items() if not str(v or "").strip()]
            if missing_agents:
                st.error(f"Falta seleccionar un modelo para: {', '.join(missing_agents)}")
            else:
                merged_models = _merge_agent_model_maps(
                    st.session_state.get("base_agent_models", {}),
                    pending_models,
                )
                applied_models = _sanitize_agent_model_map(set_runtime_agent_models(merged_models))
                st.session_state["agent_model_overrides"] = _merge_agent_model_maps(
                    st.session_state.get("base_agent_models", {}),
                    applied_models,
                )
                _save_agent_model_overrides(st.session_state["agent_model_overrides"])
                st.success("Configuraci\u00f3n aplicada correctamente.")
                st.rerun()

        if reset_models_btn:
            default_models = _merge_agent_model_maps({}, st.session_state.get("base_agent_models", {}))
            applied_models = _sanitize_agent_model_map(set_runtime_agent_models(default_models))
            st.session_state["agent_model_overrides"] = _merge_agent_model_maps(default_models, applied_models)
            _save_agent_model_overrides(st.session_state["agent_model_overrides"])
            for agent_key in MODEL_SETTING_LABELS:
                option_key = f"model_option_{agent_key}"
                custom_key = f"model_custom_{agent_key}"
                if option_key in st.session_state:
                    del st.session_state[option_key]
                if custom_key in st.session_state:
                    del st.session_state[custom_key]
            st.success("Modelos restablecidos a valores por defecto.")
            st.rerun()

    st.markdown("---")

    # ---- API Keys Panel ----
    if "show_api_keys" not in st.session_state:
        st.session_state["show_api_keys"] = False

    if st.button("\U0001f511 Claves API", key="toggle_api_keys", use_container_width=True):
        st.session_state["show_api_keys"] = not bool(st.session_state.get("show_api_keys"))

    # Show summary of configured keys
    stored_keys = _load_api_keys()
    configured_count = sum(1 for reg in API_KEY_REGISTRY if stored_keys.get(reg["env_var"]))
    required_count = sum(1 for reg in API_KEY_REGISTRY if reg.get("required"))
    total_count = len(API_KEY_REGISTRY)

    if configured_count >= required_count:
        key_status_color = "#a6e3a1"
        key_status_text = f"{configured_count}/{total_count} configuradas"
    elif configured_count > 0:
        key_status_color = "#f9e2af"
        key_status_text = f"{configured_count}/{total_count} configuradas"
    else:
        key_status_color = "#f38ba8"
        key_status_text = "Sin configurar"

    st.markdown(
        f"""
        <div class="sidebar-settings-panel">
            <div class="ssp-title">Claves API</div>
            <div class="ssp-desc" style="color:{key_status_color};">{key_status_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.get("show_api_keys"):
        st.markdown(
            """
            <div class="sidebar-settings-panel">
                <div class="ssp-title">Configuraci&oacute;n de Claves API</div>
                <div class="ssp-desc">Introduce las claves de los proveedores de IA. Se almacenan de forma segura en tu m&aacute;quina.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        pending_keys: Dict[str, str] = {}
        for reg in API_KEY_REGISTRY:
            env_var = reg["env_var"]
            label = reg["label"]
            desc = reg["description"]
            required = reg.get("required", False)
            existing = stored_keys.get(env_var, "")
            req_badge = " *" if required else ""

            st.markdown(
                f'<span style="color:#e6edf3; font-size:0.82rem; font-weight:600;">'
                f'{label}{req_badge}</span>'
                f'<br><span style="color:#6c7086; font-size:0.72rem;">{desc}</span>',
                unsafe_allow_html=True,
            )
            new_value = st.text_input(
                f"Clave {label}",
                value=existing,
                type="password",
                placeholder=reg.get("placeholder", ""),
                key=f"api_key_{env_var}",
                label_visibility="collapsed",
            )
            pending_keys[env_var] = new_value

            # Show connection status inline
            if existing:
                st.markdown(
                    f'<span style="color:#a6e3a1; font-size:0.7rem;">\u2713 {_mask_api_key(existing)}</span>',
                    unsafe_allow_html=True,
                )

        col_save, col_test = st.columns(2)
        with col_save:
            if st.button("Guardar claves", key="save_api_keys", use_container_width=True):
                _save_api_keys(pending_keys)
                apply_keys_to_env(pending_keys)
                st.success("Claves guardadas.")
                st.rerun()

        with col_test:
            if st.button("Verificar", key="test_api_keys", use_container_width=True):
                for reg in API_KEY_REGISTRY:
                    env_var = reg["env_var"]
                    value = pending_keys.get(env_var, "").strip()
                    if not value:
                        continue
                    ok, msg = _test_api_key(env_var, value)
                    if ok:
                        st.success(f"{reg['label']}: {msg}")
                    else:
                        st.error(f"{reg['label']}: {msg}")

    st.markdown("---")

    if "show_sandbox_settings" not in st.session_state:
        st.session_state["show_sandbox_settings"] = False
    if "show_execution_backend_settings" not in st.session_state:
        st.session_state["show_execution_backend_settings"] = False

    if st.button("Sandbox de ejecucion", key="toggle_sandbox_settings", use_container_width=True):
        st.session_state["show_sandbox_settings"] = not bool(st.session_state.get("show_sandbox_settings"))

    stored_sandbox_config = _load_sandbox_config()
    sandbox_provider_specs = _list_sandbox_providers()
    sandbox_label, sandbox_status_text, sandbox_status_color = _sandbox_status_summary(stored_sandbox_config)
    st.markdown(
        f"""
        <div class="sidebar-settings-panel">
            <div class="ssp-title">Sandbox</div>
            <div class="ssp-desc">
                <div><strong>Provider:</strong> {sandbox_label}</div>
                <div style="color:{sandbox_status_color};">{sandbox_status_text}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.get("show_sandbox_settings"):
        st.markdown(
            """
            <div class="sidebar-settings-panel">
                <div class="ssp-title">Configuraci&oacute;n de Sandbox</div>
                <div class="ssp-desc">Elige si las runs se ejecutan en local o en un backend remoto registrado por tu empresa.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        sandbox_options = [spec.name for spec in sandbox_provider_specs]
        current_provider = stored_sandbox_config.get("provider", "local")
        if current_provider not in sandbox_options:
            sandbox_options.append(current_provider)
        selected_provider = st.selectbox(
            "Proveedor de sandbox",
            sandbox_options,
            index=sandbox_options.index(current_provider),
            format_func=lambda name: (
                f"{_get_sandbox_provider_spec(name).label} "
                f"({'disponible' if _is_sandbox_provider_available(name) else 'pendiente de backend'})"
            ),
            key="sandbox_provider_select",
        )
        selected_spec = _get_sandbox_provider_spec(selected_provider)
        stored_settings = (
            stored_sandbox_config.get("settings", {})
            if selected_provider == current_provider and isinstance(stored_sandbox_config.get("settings"), dict)
            else {}
        )
        pending_sandbox_settings: Dict[str, Any] = {}

        st.markdown(
            f'<span style="color:#6c7086; font-size:0.72rem;">{selected_spec.description}</span>',
            unsafe_allow_html=True,
        )
        if not _is_sandbox_provider_available(selected_provider):
            st.warning(
                "Este provider todavia no tiene backend registrado en este despliegue. "
                "Puedes guardar la configuracion, pero la app bloqueara la run hasta que exista una implementacion real."
            )

        for field in selected_spec.config_fields:
            existing_value = str(stored_settings.get(field.key) or "")
            st.markdown(
                f'<span style="color:#e6edf3; font-size:0.82rem; font-weight:600;">{field.label}'
                f'{" *" if field.required else ""}</span>'
                f'<br><span style="color:#6c7086; font-size:0.72rem;">{field.description}</span>',
                unsafe_allow_html=True,
            )
            widget_value = st.text_input(
                field.label,
                value=existing_value,
                type="password" if field.secret else "default",
                placeholder=field.placeholder,
                key=f"sandbox_field_{selected_provider}_{field.key}",
                label_visibility="collapsed",
            )
            pending_sandbox_settings[field.key] = widget_value
            if field.secret and existing_value:
                st.markdown(
                    f'<span style="color:#a6e3a1; font-size:0.7rem;">\u2713 {_mask_sandbox_secret(existing_value)}</span>',
                    unsafe_allow_html=True,
                )

        sandbox_save_col, sandbox_test_col = st.columns(2)
        candidate_sandbox_config = _merge_execution_backend_config(
            _normalize_sandbox_config({"provider": selected_provider, "settings": pending_sandbox_settings}),
            _get_execution_backend_config(stored_sandbox_config, include_env_fallback=False),
        )

        with sandbox_save_col:
            if st.button("Guardar sandbox", key="save_sandbox_settings", use_container_width=True):
                missing_fields = [
                    field.label
                    for field in selected_spec.config_fields
                    if field.required and not str(candidate_sandbox_config.get("settings", {}).get(field.key) or "").strip()
                ]
                if missing_fields:
                    st.error(f"Faltan campos obligatorios: {', '.join(missing_fields)}")
                else:
                    _save_sandbox_config(candidate_sandbox_config)
                    st.success("Configuracion de sandbox guardada.")
                    st.rerun()

        with sandbox_test_col:
            if st.button("Verificar sandbox", key="test_sandbox_settings", use_container_width=True):
                ok, msg = _test_sandbox_provider_connectivity(
                    selected_provider,
                    candidate_sandbox_config.get("settings", {}),
                )
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)

    if st.button("Backend de ejecucion", key="toggle_execution_backend_settings", use_container_width=True):
        st.session_state["show_execution_backend_settings"] = not bool(
            st.session_state.get("show_execution_backend_settings")
        )

    execution_backend_label, execution_backend_status, execution_backend_color = _execution_backend_status_summary(
        stored_sandbox_config
    )
    st.markdown(
        f"""
        <div class="sidebar-settings-panel">
            <div class="ssp-title">Backend de ejecucion</div>
            <div class="ssp-desc">
                <div><strong>Motor:</strong> {execution_backend_label}</div>
                <div style="color:{execution_backend_color};">{execution_backend_status}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.get("show_execution_backend_settings"):
        st.markdown(
            """
            <div class="sidebar-settings-panel">
                <div class="ssp-title">Configuraci&oacute;n de Backend</div>
                <div class="ssp-desc">Controla desde aqu&iacute; c&oacute;mo se ejecutan los scripts: runner local o Cloud Run corporativo. Esta configuraci&oacute;n acompa&ntilde;a a cada run y evita depender del archivo .env.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        stored_backend_config = _get_execution_backend_config(stored_sandbox_config)
        backend_mode = st.selectbox(
            "Modo de ejecucion",
            ["local", "cloudrun"],
            index=0 if str(stored_backend_config.get("mode") or "cloudrun").strip().lower() == "local" else 1,
            format_func=lambda value: "Local Runner" if value == "local" else "Cloud Run",
            key="execution_backend_mode",
        )

        cloudrun_enabled = st.checkbox(
            "Activar backend Cloud Run",
            value=bool(stored_backend_config.get("cloudrun_enabled", False)),
            key="execution_backend_cloudrun_enabled",
        )
        de_cloudrun_enabled = st.checkbox(
            "Permitir heavy runner para Data Engineer",
            value=bool(stored_backend_config.get("data_engineer_cloudrun_enabled", True)),
            key="execution_backend_de_cloudrun_enabled",
        )

        backend_field_defaults = {
            "job": str(stored_backend_config.get("job") or ""),
            "region": str(stored_backend_config.get("region") or ""),
            "bucket": str(stored_backend_config.get("bucket") or ""),
            "project": str(stored_backend_config.get("project") or ""),
            "gcloud_bin": str(stored_backend_config.get("gcloud_bin") or ""),
            "gsutil_bin": str(stored_backend_config.get("gsutil_bin") or ""),
            "input_prefix": str(stored_backend_config.get("input_prefix") or ""),
            "output_prefix": str(stored_backend_config.get("output_prefix") or ""),
            "dataset_prefix": str(stored_backend_config.get("dataset_prefix") or ""),
            "script_timeout_seconds": str(stored_backend_config.get("script_timeout_seconds") or ""),
            "local_script_timeout_seconds": str(stored_backend_config.get("local_script_timeout_seconds") or ""),
            "script_timeout_min_seconds": str(stored_backend_config.get("script_timeout_min_seconds") or ""),
            "script_timeout_max_seconds": str(stored_backend_config.get("script_timeout_max_seconds") or ""),
            "local_script_timeout_min_seconds": str(stored_backend_config.get("local_script_timeout_min_seconds") or ""),
            "local_script_timeout_max_seconds": str(stored_backend_config.get("local_script_timeout_max_seconds") or ""),
            "timeout_margin_multiplier": str(stored_backend_config.get("timeout_margin_multiplier") or ""),
            "timeout_margin_seconds": str(stored_backend_config.get("timeout_margin_seconds") or ""),
            "default_cpu": str(stored_backend_config.get("default_cpu") or ""),
            "default_memory_gb": str(stored_backend_config.get("default_memory_gb") or ""),
            "cpu_hint": str(stored_backend_config.get("cpu_hint") or ""),
            "memory_gb_hint": str(stored_backend_config.get("memory_gb_hint") or ""),
            "model_type": str(stored_backend_config.get("model_type") or ""),
        }
        model_params_default = stored_backend_config.get("model_params")
        if isinstance(model_params_default, dict):
            model_params_default_text = json.dumps(model_params_default, ensure_ascii=False, indent=2)
        else:
            model_params_default_text = str(model_params_default or "")

        if backend_mode == "cloudrun":
            st.markdown(
                '<span style="color:#e6edf3; font-size:0.82rem; font-weight:600;">Cloud Run Job *</span>'
                '<br><span style="color:#6c7086; font-size:0.72rem;">Nombre del job corporativo que ejecuta el heavy runner.</span>',
                unsafe_allow_html=True,
            )
            backend_field_defaults["job"] = st.text_input(
                "Cloud Run Job",
                value=backend_field_defaults["job"],
                key="execution_backend_job",
                label_visibility="collapsed",
            )
            st.markdown(
                '<span style="color:#e6edf3; font-size:0.82rem; font-weight:600;">Region *</span>'
                '<br><span style="color:#6c7086; font-size:0.72rem;">Regi&oacute;n donde vive el job de Cloud Run.</span>',
                unsafe_allow_html=True,
            )
            backend_field_defaults["region"] = st.text_input(
                "Region",
                value=backend_field_defaults["region"],
                key="execution_backend_region",
                label_visibility="collapsed",
            )
            st.markdown(
                '<span style="color:#e6edf3; font-size:0.82rem; font-weight:600;">Bucket GCS *</span>'
                '<br><span style="color:#6c7086; font-size:0.72rem;">Bucket usado para subir requests, datasets y artifacts de ejecuci&oacute;n.</span>',
                unsafe_allow_html=True,
            )
            backend_field_defaults["bucket"] = st.text_input(
                "Bucket GCS",
                value=backend_field_defaults["bucket"],
                key="execution_backend_bucket",
                label_visibility="collapsed",
            )
            st.markdown(
                '<span style="color:#e6edf3; font-size:0.82rem; font-weight:600;">Project</span>'
                '<br><span style="color:#6c7086; font-size:0.72rem;">ID de proyecto de Google Cloud si el job no usa el proyecto por defecto del CLI.</span>',
                unsafe_allow_html=True,
            )
            backend_field_defaults["project"] = st.text_input(
                "Project",
                value=backend_field_defaults["project"],
                key="execution_backend_project",
                label_visibility="collapsed",
            )
        else:
            st.info("Las runs se resolveran en Local Runner. Cloud Run queda guardado como configuracion corporativa opcional.")

        with st.expander("Opciones avanzadas del backend"):
            backend_field_defaults["gcloud_bin"] = st.text_input(
                "Ruta gcloud",
                value=backend_field_defaults["gcloud_bin"],
                placeholder="Opcional. Ej: C:\\...\\gcloud.cmd",
                key="execution_backend_gcloud_bin",
            )
            backend_field_defaults["gsutil_bin"] = st.text_input(
                "Ruta gsutil",
                value=backend_field_defaults["gsutil_bin"],
                placeholder="Opcional. Ej: C:\\...\\gsutil.cmd",
                key="execution_backend_gsutil_bin",
            )
            backend_field_defaults["input_prefix"] = st.text_input(
                "Input prefix",
                value=backend_field_defaults["input_prefix"],
                placeholder="inputs",
                key="execution_backend_input_prefix",
            )
            backend_field_defaults["output_prefix"] = st.text_input(
                "Output prefix",
                value=backend_field_defaults["output_prefix"],
                placeholder="outputs",
                key="execution_backend_output_prefix",
            )
            backend_field_defaults["dataset_prefix"] = st.text_input(
                "Dataset prefix",
                value=backend_field_defaults["dataset_prefix"],
                placeholder="datasets",
                key="execution_backend_dataset_prefix",
            )
            backend_field_defaults["script_timeout_seconds"] = st.text_input(
                "Timeout script Cloud Run (s)",
                value=backend_field_defaults["script_timeout_seconds"],
                key="execution_backend_script_timeout_seconds",
            )
            backend_field_defaults["local_script_timeout_seconds"] = st.text_input(
                "Timeout script local (s)",
                value=backend_field_defaults["local_script_timeout_seconds"],
                key="execution_backend_local_script_timeout_seconds",
            )
            backend_field_defaults["script_timeout_min_seconds"] = st.text_input(
                "Min timeout Cloud Run (s)",
                value=backend_field_defaults["script_timeout_min_seconds"],
                key="execution_backend_script_timeout_min_seconds",
            )
            backend_field_defaults["script_timeout_max_seconds"] = st.text_input(
                "Max timeout Cloud Run (s)",
                value=backend_field_defaults["script_timeout_max_seconds"],
                key="execution_backend_script_timeout_max_seconds",
            )
            backend_field_defaults["local_script_timeout_min_seconds"] = st.text_input(
                "Min timeout local (s)",
                value=backend_field_defaults["local_script_timeout_min_seconds"],
                key="execution_backend_local_script_timeout_min_seconds",
            )
            backend_field_defaults["local_script_timeout_max_seconds"] = st.text_input(
                "Max timeout local (s)",
                value=backend_field_defaults["local_script_timeout_max_seconds"],
                key="execution_backend_local_script_timeout_max_seconds",
            )
            backend_field_defaults["timeout_margin_multiplier"] = st.text_input(
                "Multiplicador de margen",
                value=backend_field_defaults["timeout_margin_multiplier"],
                key="execution_backend_timeout_margin_multiplier",
            )
            backend_field_defaults["timeout_margin_seconds"] = st.text_input(
                "Margen fijo extra (s)",
                value=backend_field_defaults["timeout_margin_seconds"],
                key="execution_backend_timeout_margin_seconds",
            )
            backend_field_defaults["default_cpu"] = st.text_input(
                "CPU por defecto Cloud Run",
                value=backend_field_defaults["default_cpu"],
                key="execution_backend_default_cpu",
            )
            backend_field_defaults["default_memory_gb"] = st.text_input(
                "Memoria por defecto Cloud Run",
                value=backend_field_defaults["default_memory_gb"],
                placeholder="Ej: 32Gi",
                key="execution_backend_default_memory_gb",
            )
            backend_field_defaults["cpu_hint"] = st.text_input(
                "CPU hint",
                value=backend_field_defaults["cpu_hint"],
                key="execution_backend_cpu_hint",
            )
            backend_field_defaults["memory_gb_hint"] = st.text_input(
                "Memory hint",
                value=backend_field_defaults["memory_gb_hint"],
                placeholder="Ej: 16Gi",
                key="execution_backend_memory_gb_hint",
            )
            backend_field_defaults["model_type"] = st.text_input(
                "Model type del heavy runner",
                value=backend_field_defaults["model_type"],
                key="execution_backend_model_type",
            )
            model_params_default_text = st.text_area(
                "Model params (JSON opcional)",
                value=model_params_default_text,
                height=120,
                key="execution_backend_model_params",
            )
            force_cloudrun = st.checkbox(
                "Forzar Cloud Run para ML",
                value=bool(stored_backend_config.get("force_cloudrun", False)),
                key="execution_backend_force_cloudrun",
            )
            force_de_cloudrun = st.checkbox(
                "Forzar Cloud Run para Data Engineer",
                value=bool(stored_backend_config.get("force_data_engineer_cloudrun", False)),
                key="execution_backend_force_de_cloudrun",
            )
            safe_mode = st.checkbox(
                "Safe mode del heavy runner",
                value=bool(stored_backend_config.get("safe_mode", False)),
                key="execution_backend_safe_mode",
            )
            float32 = st.checkbox(
                "Forzar float32 en heavy runner",
                value=bool(stored_backend_config.get("float32", False)),
                key="execution_backend_float32",
            )

        candidate_execution_backend = _normalize_execution_backend_config(
            {
                "mode": backend_mode,
                "cloudrun_enabled": cloudrun_enabled,
                "data_engineer_cloudrun_enabled": de_cloudrun_enabled,
                "force_cloudrun": force_cloudrun,
                "force_data_engineer_cloudrun": force_de_cloudrun,
                "safe_mode": safe_mode,
                "float32": float32,
                "job": backend_field_defaults["job"],
                "region": backend_field_defaults["region"],
                "bucket": backend_field_defaults["bucket"],
                "project": backend_field_defaults["project"],
                "gcloud_bin": backend_field_defaults["gcloud_bin"],
                "gsutil_bin": backend_field_defaults["gsutil_bin"],
                "input_prefix": backend_field_defaults["input_prefix"],
                "output_prefix": backend_field_defaults["output_prefix"],
                "dataset_prefix": backend_field_defaults["dataset_prefix"],
                "script_timeout_seconds": backend_field_defaults["script_timeout_seconds"],
                "local_script_timeout_seconds": backend_field_defaults["local_script_timeout_seconds"],
                "script_timeout_min_seconds": backend_field_defaults["script_timeout_min_seconds"],
                "script_timeout_max_seconds": backend_field_defaults["script_timeout_max_seconds"],
                "local_script_timeout_min_seconds": backend_field_defaults["local_script_timeout_min_seconds"],
                "local_script_timeout_max_seconds": backend_field_defaults["local_script_timeout_max_seconds"],
                "timeout_margin_multiplier": backend_field_defaults["timeout_margin_multiplier"],
                "timeout_margin_seconds": backend_field_defaults["timeout_margin_seconds"],
                "default_cpu": backend_field_defaults["default_cpu"],
                "default_memory_gb": backend_field_defaults["default_memory_gb"],
                "cpu_hint": backend_field_defaults["cpu_hint"],
                "memory_gb_hint": backend_field_defaults["memory_gb_hint"],
                "model_type": backend_field_defaults["model_type"],
                "model_params": model_params_default_text,
            }
        )

        backend_save_col, backend_preview_col = st.columns(2)
        with backend_save_col:
            if st.button("Guardar backend", key="save_execution_backend_settings", use_container_width=True):
                must_validate_cloudrun = backend_mode == "cloudrun" and (
                    cloudrun_enabled
                    or any(
                        str(candidate_execution_backend.get(key) or "").strip()
                        for key in ("job", "region", "bucket")
                    )
                )
                missing_backend_fields: List[str] = []
                if must_validate_cloudrun:
                    for key, label in (("job", "Cloud Run Job"), ("region", "Region"), ("bucket", "Bucket GCS")):
                        if not str(candidate_execution_backend.get(key) or "").strip():
                            missing_backend_fields.append(label)
                if missing_backend_fields:
                    st.error(f"Faltan campos obligatorios del backend: {', '.join(missing_backend_fields)}")
                else:
                    merged_backend_config = _merge_execution_backend_config(
                        stored_sandbox_config,
                        candidate_execution_backend,
                    )
                    _save_sandbox_config(merged_backend_config)
                    st.success("Configuracion de backend guardada.")
                    st.rerun()

        with backend_preview_col:
            if st.button("Ver resumen backend", key="preview_execution_backend_settings", use_container_width=True):
                st.json(candidate_execution_backend)

    # ---- CRM Connectors (alternative data source) ----
    _show_crm = st.session_state.get("show_crm_panel", False)
    if st.button("Conectar CRM", key="toggle_crm_panel", use_container_width=True):
        st.session_state["show_crm_panel"] = not _show_crm
        _show_crm = not _show_crm

    if _show_crm:
        crm_source = st.radio(
            "CRM", ["Salesforce", "HubSpot", "Dynamics 365"], label_visibility="collapsed",
        )
        if crm_source == "Salesforce":
            sf_auth_mode = st.selectbox("Modo de autenticaci\u00f3n", ["Token API", "OAuth (Access Token)"], key="sf_auth_mode")
            if sf_auth_mode == "Token API":
                sf_username = st.text_input("Username", key="sf_username")
                sf_password = st.text_input("Password", type="password", key="sf_password")
                sf_token = st.text_input("Security Token", type="password", key="sf_security_token")
                sf_connect = st.button("Conectar a Salesforce", key="sf_connect")
                if sf_connect and sf_username and sf_password and sf_token:
                    try:
                        from src.connectors.salesforce_connector import SalesforceConnector
                        connector = SalesforceConnector()
                        connector.authenticate({"mode": "token", "username": sf_username, "password": sf_password, "security_token": sf_token})
                        st.session_state["crm_connector"] = connector
                        st.session_state["crm_authenticated"] = True
                        st.session_state["crm_objects"] = connector.list_objects()
                    except Exception as exc:
                        st.error(f"Error: {exc}")
                        st.session_state["crm_authenticated"] = False
            else:
                sf_access_token = st.text_input("Access Token", type="password", key="sf_oauth_token")
                sf_instance_url = st.text_input("Instance URL", placeholder="https://your-instance.salesforce.com", key="sf_instance_url")
                sf_connect_oauth = st.button("Conectar a Salesforce", key="sf_connect_oauth")
                if sf_connect_oauth and sf_access_token and sf_instance_url:
                    try:
                        from src.connectors.salesforce_connector import SalesforceConnector
                        connector = SalesforceConnector()
                        connector.authenticate({"mode": "oauth", "access_token": sf_access_token, "instance_url": sf_instance_url})
                        st.session_state["crm_connector"] = connector
                        st.session_state["crm_authenticated"] = True
                        st.session_state["crm_objects"] = connector.list_objects()
                    except Exception as exc:
                        st.error(f"Error: {exc}")
                        st.session_state["crm_authenticated"] = False
            if st.session_state.get("crm_authenticated") and type(st.session_state.get("crm_connector")).__name__ == "SalesforceConnector":
                st.markdown('<span class="badge badge-success">Conectado a Salesforce</span>', unsafe_allow_html=True)
                crm_objects = st.session_state.get("crm_objects", [])
                if crm_objects:
                    obj_labels = [f"{o['label']} ({o['name']})" for o in crm_objects]
                    selected_idx = st.selectbox("Tabla CRM", range(len(obj_labels)), format_func=lambda i: obj_labels[i], key="sf_obj_select")
                    max_recs = st.number_input("M\u00e1x. registros", min_value=100, max_value=50000, value=10000, step=500, key="sf_max_recs")
                    if st.button("Extraer Datos", key="sf_fetch"):
                        selected_obj = crm_objects[selected_idx]["name"]
                        try:
                            connector = st.session_state["crm_connector"]
                            df_crm = connector.fetch_object_data(selected_obj, max_records=int(max_recs))
                            if df_crm.empty:
                                st.warning(f"El objeto '{selected_obj}' no contiene datos.")
                            else:
                                os.makedirs("data", exist_ok=True)
                                crm_csv = os.path.join("data", f"crm_{selected_obj.lower()}.csv")
                                df_crm.to_csv(crm_csv, index=False, encoding="utf-8")
                                st.session_state["crm_data_path"] = crm_csv
                                st.session_state["crm_preview_df"] = df_crm
                                st.markdown(f'<span class="metric-pill">{len(df_crm):,} registros extra\u00eddos</span>', unsafe_allow_html=True)
                        except Exception as exc:
                            st.error(f"Error al obtener los datos: {exc}")
                if st.session_state.get("crm_data_path") and st.session_state.get("crm_preview_df") is not None:
                    st.markdown(f'<span class="metric-pill">{len(st.session_state["crm_preview_df"]):,} registros listos</span>', unsafe_allow_html=True)

        elif crm_source == "HubSpot":
            hs_auth_mode = st.selectbox("Modo de autenticaci\u00f3n", ["Private App Token", "OAuth (Access Token)"], key="hs_auth_mode")
            hs_token = st.text_input("Token", type="password", key="hs_token")
            if st.button("Conectar a HubSpot", key="hs_connect"):
                if hs_token:
                    try:
                        from src.connectors.hubspot_connector import HubSpotConnector
                        connector = HubSpotConnector()
                        connector.authenticate({"access_token": hs_token})
                        st.session_state["crm_connector"] = connector
                        st.session_state["crm_authenticated"] = True
                        st.session_state["crm_objects"] = connector.list_objects()
                    except Exception as exc:
                        st.error(f"Error: {exc}")
                        st.session_state["crm_authenticated"] = False
            if st.session_state.get("crm_authenticated") and type(st.session_state.get("crm_connector")).__name__ == "HubSpotConnector":
                st.markdown('<span class="badge badge-success">Conectado a HubSpot</span>', unsafe_allow_html=True)
                crm_objects = st.session_state.get("crm_objects", [])
                if crm_objects:
                    obj_labels = [f"{o['label']} ({o['name']})" for o in crm_objects]
                    selected_idx = st.selectbox("Tabla CRM", range(len(obj_labels)), format_func=lambda i: obj_labels[i], key="hs_obj_select")
                    max_recs = st.number_input("M\u00e1x. registros", min_value=100, max_value=50000, value=10000, step=500, key="hs_max_recs")
                    if st.button("Extraer Datos", key="hs_fetch"):
                        selected_obj = crm_objects[selected_idx]["name"]
                        try:
                            connector = st.session_state["crm_connector"]
                            df_crm = connector.fetch_object_data(selected_obj, max_records=int(max_recs))
                            if df_crm.empty:
                                st.warning(f"El objeto '{selected_obj}' no contiene datos.")
                            else:
                                os.makedirs("data", exist_ok=True)
                                crm_csv = os.path.join("data", f"crm_{selected_obj.lower()}.csv")
                                df_crm.to_csv(crm_csv, index=False, encoding="utf-8")
                                st.session_state["crm_data_path"] = crm_csv
                                st.session_state["crm_preview_df"] = df_crm
                                st.markdown(f'<span class="metric-pill">{len(df_crm):,} registros extra\u00eddos</span>', unsafe_allow_html=True)
                        except Exception as exc:
                            st.error(f"Error al obtener los datos: {exc}")
                if st.session_state.get("crm_data_path") and st.session_state.get("crm_preview_df") is not None:
                    st.markdown(f'<span class="metric-pill">{len(st.session_state["crm_preview_df"]):,} registros listos</span>', unsafe_allow_html=True)

        elif crm_source == "Dynamics 365":
            dy_crm_url = st.text_input("URL de la organizaci\u00f3n", placeholder="https://miorg.crm.dynamics.com", key="dy_crm_url")
            dy_tenant = st.text_input("Tenant ID", key="dy_tenant_id")
            dy_client_id = st.text_input("Client ID", key="dy_client_id")
            dy_secret = st.text_input("Client Secret", type="password", key="dy_client_secret")
            if st.button("Conectar a Dynamics 365", key="dy_connect"):
                if dy_crm_url and dy_tenant and dy_client_id and dy_secret:
                    try:
                        from src.connectors.dynamics_connector import DynamicsConnector
                        connector = DynamicsConnector()
                        connector.authenticate({
                            "crm_url": dy_crm_url,
                            "tenant_id": dy_tenant,
                            "client_id": dy_client_id,
                            "client_secret": dy_secret,
                        })
                        st.session_state["crm_connector"] = connector
                        st.session_state["crm_authenticated"] = True
                        st.session_state["crm_objects"] = connector.list_objects()
                    except Exception as exc:
                        st.error(f"Error: {exc}")
                        st.session_state["crm_authenticated"] = False
                else:
                    st.warning("Completa todos los campos.")
            if st.session_state.get("crm_authenticated") and type(st.session_state.get("crm_connector")).__name__ == "DynamicsConnector":
                st.markdown('<span class="badge badge-success">Conectado a Dynamics 365</span>', unsafe_allow_html=True)
                crm_objects = st.session_state.get("crm_objects", [])
                if crm_objects:
                    obj_labels = [f"{o['label']} ({o['name']})" for o in crm_objects]
                    selected_idx = st.selectbox("Entidad", range(len(obj_labels)), format_func=lambda i: obj_labels[i], key="dy_obj_select")
                    max_recs = st.number_input("M\u00e1x. registros", min_value=100, max_value=50000, value=10000, step=500, key="dy_max_recs")
                    if st.button("Extraer Datos", key="dy_fetch"):
                        selected_obj = crm_objects[selected_idx]["name"]
                        try:
                            connector = st.session_state["crm_connector"]
                            df_crm = connector.fetch_object_data(selected_obj, max_records=int(max_recs))
                            if df_crm.empty:
                                st.warning(f"La entidad '{selected_obj}' no contiene datos.")
                            else:
                                os.makedirs("data", exist_ok=True)
                                crm_csv = os.path.join("data", f"crm_{selected_obj.lower()}.csv")
                                df_crm.to_csv(crm_csv, index=False, encoding="utf-8")
                                st.session_state["crm_data_path"] = crm_csv
                                st.session_state["crm_preview_df"] = df_crm
                                st.markdown(f'<span class="metric-pill">{len(df_crm):,} registros extra\u00eddos</span>', unsafe_allow_html=True)
                        except Exception as exc:
                            st.error(f"Error al obtener los datos: {exc}")
                if st.session_state.get("crm_data_path") and st.session_state.get("crm_preview_df") is not None:
                    st.markdown(f'<span class="metric-pill">{len(st.session_state["crm_preview_df"]):,} registros listos</span>', unsafe_allow_html=True)

    # ---- Run History in Sidebar ----
    st.markdown("---")
    _sidebar_runs = _list_runs(limit=10)
    if _sidebar_runs:
        st.markdown(
            '<div style="font-size:0.72rem; font-weight:700; text-transform:uppercase; '
            'letter-spacing:0.07em; color:#8b949e; margin-bottom:0.5rem;">Historial</div>',
            unsafe_allow_html=True,
        )
        _STATUS_ICONS_SIDEBAR = {
            "complete": "\u2705", "error": "\u274c", "aborted": "\u26a0\ufe0f",
        }
        for _sr in _sidebar_runs:
            _sr_icon = _STATUS_ICONS_SIDEBAR.get(_sr["status"], "\u2753")
            _sr_strategy = _sr.get("strategy") or ""
            _sr_title = _sr_strategy if _sr_strategy else _sr["run_id"]
            _sr_metric = ""
            if _sr["metric_value"]:
                _sr_metric = f'<span class="sidebar-run-metric">{_sr["metric_name"]}: {_sr["metric_value"]}</span>'
            _sr_objective = _sr.get("business_objective") or ""
            _sr_objective_html = ""
            if _sr_objective:
                _sr_obj_escaped = _sr_objective.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                _sr_objective_html = f'<div class="sidebar-run-objective">{_sr_obj_escaped}</div>'
            st.markdown(
                f'<div class="sidebar-run-item">'
                f'<div class="sidebar-run-title">{_sr_icon} {_sr_title}</div>'
                f'{_sr_objective_html}'
                f'<div class="sidebar-run-meta">'
                f'<span>{_sr["run_id"]}</span>'
                f'<span>{_sr.get("started_str") or ""}</span>'
                f'<span>{_sr.get("elapsed") or ""}</span>'
                f'{_sr_metric}'
                f'</div></div>',
                unsafe_allow_html=True,
            )
            _sr_btn_cols = st.columns([1, 1] if _sr["status"] == "complete" else [1])
            if _sr["status"] == "complete":
                with _sr_btn_cols[0]:
                    if st.button("Ver", key=f"view_run_{_sr['run_id']}", use_container_width=True):
                        _loaded = _load_run_result(_sr["run_id"])
                        if _loaded:
                            st.session_state["analysis_result"] = _loaded
                            st.session_state["analysis_complete"] = True
                            st.session_state["viewing_run_id"] = _sr["run_id"]
                            st.session_state.pop("pdf_binary", None)
                            st.rerun()
                        else:
                            st.error("No se pudieron cargar los resultados.")
            _reuse_col = _sr_btn_cols[-1] if _sr["status"] == "complete" else _sr_btn_cols[0]
            if _sr_objective:
                with _reuse_col:
                    if st.button("Reutilizar", key=f"reuse_run_{_sr['run_id']}", use_container_width=True):
                        st.session_state["main_business_objective"] = _sr_objective
                        st.session_state.pop("analysis_complete", None)
                        st.session_state.pop("analysis_result", None)
                        st.session_state.pop("viewing_run_id", None)
                        st.rerun()

# ---------------------------------------------------------------------------
# Session State init
# ---------------------------------------------------------------------------
if "analysis_complete" not in st.session_state:
    st.session_state["analysis_complete"] = False
if "analysis_result" not in st.session_state:
    st.session_state["analysis_result"] = None

# ---------------------------------------------------------------------------
# Background worker reconnection
# ---------------------------------------------------------------------------
def _get_latest_run_id() -> str | None:
    try:
        with open(os.path.join("runs", "latest", "run_id.txt"), "r") as f:
            return f.read().strip() or None
    except FileNotFoundError:
        return None

if not st.session_state.get("analysis_complete"):
    _reconnect_run = _get_active_run_id()
    if _reconnect_run:
        st.session_state["active_run_id"] = _reconnect_run
    elif not st.session_state.get("active_run_id"):
        # Check if latest run completed while session was disconnected.
        # Skip if user explicitly dismissed the results (clicked "Nuevo an\u00e1lisis").
        if not st.session_state.get("dismissed_latest_run"):
            _latest = _get_latest_run_id()
            if _latest:
                _latest_status = _read_run_status(_latest)
                if _latest_status and _latest_status.get("status") == "complete":
                    _final = _read_final_state(_latest)
                    if _final:
                        st.session_state["analysis_result"] = _final
                        st.session_state["analysis_complete"] = True

# ---------------------------------------------------------------------------
# Welcome Screen — ChatGPT / Claude style centered input
# ---------------------------------------------------------------------------
uploaded_file = None
business_objective = ""
start_btn = False

if not st.session_state.get("analysis_complete"):
    # Hero
    st.markdown("""
    <div class="hero fade-in" style="padding: 2.5rem 1rem 1rem;">
        <h1><span class="hero-gradient">Inteligencia de Negocio Aut&oacute;noma</span></h1>
        <p class="hero-subtitle">
            Define tu objetivo, adjunta un dataset y deja que el equipo de agentes IA
            audite, dise&ntilde;e estrategias, construya modelos y genere un informe ejecutivo.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ---- Centered chat-like input area ----
    st.markdown('<div class="chat-input-container fade-in">', unsafe_allow_html=True)
    st.markdown('<div class="chat-input-box">', unsafe_allow_html=True)

    business_objective = st.text_area(
        "Objetivo de negocio",
        placeholder="Describe el objetivo que deseas lograr con tus datos.\nEj: Reducir el churn de clientes en un 10% identificando los factores clave de abandono...",
        height=120,
        label_visibility="collapsed",
        key="main_business_objective",
    )

    st.markdown('</div>', unsafe_allow_html=True)  # close chat-input-box

    # Attachment row + start button
    _attach_col, _start_col = st.columns([3, 1])
    with _attach_col:
        uploaded_file = st.file_uploader(
            "Adjuntar CSV",
            type=["csv", "xlsx", "xls"],
            label_visibility="collapsed",
            key="main_file_upload",
        )
        if uploaded_file is not None:
            file_size = uploaded_file.size
            if file_size < 1024:
                size_str = f"{file_size} B"
            elif file_size < 1024 * 1024:
                size_str = f"{file_size / 1024:.1f} KB"
            else:
                size_str = f"{file_size / (1024*1024):.1f} MB"
            st.markdown(
                f'<div class="file-chip">&#128206; {uploaded_file.name} &mdash; {size_str}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="file-chip-placeholder">&#128206; Adjunta un archivo CSV o Excel (obligatorio)</div>',
                unsafe_allow_html=True,
            )
    with _start_col:
        st.markdown('<div class="chat-start-btn">', unsafe_allow_html=True)
        start_btn = st.button("Iniciar", use_container_width=True, key="main_start_btn")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # close chat-input-container

    # Feature cards (compact)
    st.markdown("")
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        st.markdown("""
        <div class="feature-card fade-in">
            <div class="feature-icon">&#128269;</div>
            <div class="feature-title">Auditor&iacute;a de Datos</div>
            <div class="feature-desc">An&aacute;lisis autom&aacute;tico de calidad, integridad y distribuci&oacute;n de tus datos.</div>
        </div>
        """, unsafe_allow_html=True)
    with col_f2:
        st.markdown("""
        <div class="feature-card fade-in">
            <div class="feature-icon">&#127919;</div>
            <div class="feature-title">Estrategia IA</div>
            <div class="feature-desc">Razonamiento estrat&eacute;gico senior para dise&ntilde;ar el mejor enfoque para tus datos.</div>
        </div>
        """, unsafe_allow_html=True)
    with col_f3:
        st.markdown("""
        <div class="feature-card fade-in">
            <div class="feature-icon">&#9881;&#65039;</div>
            <div class="feature-title">ML Automatizado</div>
            <div class="feature-desc">Entrenamiento iterativo de modelos y generaci&oacute;n de informes ejecutivos.</div>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Resolve data_path from any source
# ---------------------------------------------------------------------------
data_path = None

if uploaded_file is not None:
    os.makedirs("data", exist_ok=True)
    temp_path = os.path.join("data", uploaded_file.name)
    with open(temp_path, "wb") as f:
        uploaded_file.seek(0)
        while chunk := uploaded_file.read(8 * 1024 * 1024):
            f.write(chunk)

    ext = os.path.splitext(uploaded_file.name)[1].lower()
    if ext in ('.xlsx', '.xls'):
        from src.connectors.excel_converter import convert_to_csv
        data_path = convert_to_csv(temp_path)
        st.info(f"Archivo Excel convertido a CSV: {os.path.basename(data_path)}")
    else:
        data_path = temp_path

if data_path is None:
    data_path = st.session_state.get("crm_data_path")

# ---------------------------------------------------------------------------
# Unified preview for any data source
# ---------------------------------------------------------------------------
if data_path is not None:
    if not st.session_state["analysis_complete"] and not start_btn:
        try:
            df_preview = pd.read_csv(data_path, nrows=50)
        except Exception:
            df_preview = None

        # Fallback: try CSV with semicolon separator
        if df_preview is None or (df_preview is not None and len(df_preview.columns) <= 1):
            try:
                df_preview = pd.read_csv(data_path, sep=';', nrows=50)
            except Exception:
                df_preview = None

        if df_preview is not None and len(df_preview.columns) > 1:
            n_rows, n_cols = df_preview.shape
            dtypes_summary = df_preview.dtypes.value_counts()
            dtype_parts = [f"{count} {str(dtype)}" for dtype, count in dtypes_summary.items()]

            st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
            st.markdown("#### Vista Previa del Dataset")

            pills_html = (
                f'<span class="metric-pill">{n_rows:,} filas</span>'
                f'<span class="metric-pill">{n_cols} columnas</span>'
            )
            for part in dtype_parts:
                pills_html += f'<span class="metric-pill">{part}</span>'
            st.markdown(pills_html, unsafe_allow_html=True)

            st.dataframe(df_preview.head(10), use_container_width=True, height=300)
            st.markdown('</div>', unsafe_allow_html=True)
        elif df_preview is not None:
            st.warning("El dataset parece tener solo una columna. Verifica el formato y separador del archivo.")

# ---------------------------------------------------------------------------
# Pipeline steps definition (for visual tracker)
# ---------------------------------------------------------------------------
PIPELINE_STEPS = [
    ("steward",            "Auditor",      "&#128270;"),
    ("strategist",         "Estratega",    "&#129504;"),
    ("execution_planner",  "Planner",      "&#128220;"),
    ("data_engineer",      "Ing. Datos",   "&#128295;"),
    ("engineer",           "Ing. ML",      "&#9881;"),
    ("evaluate_results",   "Revisor",      "&#128269;"),
    ("review_board",       "Calidad",      "&#9989;"),
    ("translator",         "Informe",      "&#128202;"),
]

def _render_pipeline(completed_steps: set, active_step: str | None = None, iter_info: str = ""):
    """Render the visual pipeline tracker with optional iteration info."""
    parts = []
    for key, label, icon in PIPELINE_STEPS:
        if key in completed_steps:
            cls_icon = "completed"
            cls_label = "completed"
        elif key == active_step:
            cls_icon = "active"
            cls_label = "active"
        else:
            cls_icon = ""
            cls_label = ""
        parts.append(
            f'<div class="pipeline-step">'
            f'  <div class="step-icon {cls_icon}">{icon}</div>'
            f'  <div class="step-label {cls_label}">{label}</div>'
            f'</div>'
        )
    html = '<div class="pipeline-container">' + "".join(parts) + '</div>'
    if iter_info:
        html += iter_info
    return html

def _fmt_elapsed(seconds: float) -> str:
    """Format elapsed seconds as MM:SS or HH:MM:SS."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

# Progress weight per step (cumulative %)
_STEP_PROGRESS = {
    "steward": 10,
    "strategist": 22,
    "execution_planner": 32,
    "data_engineer": 48,
    "engineer": 60,       # first ML iteration start
    "evaluate_results": 78,
    "review_board": 82,
    "translator": 94,
    "generate_pdf": 100,
}

# Friendly stage names for the progress header
_STAGE_NAMES = {
    "steward": "Auditando datos",
    "strategist": "Generando estrategia",
    "execution_planner": "Planificando ejecución",
    "data_engineer": "Procesando datos",
    "engineer": "Entrenando modelo ML",
    "evaluate_results": "Evaluando resultados",
    "review_board": "Revisando calidad",
    "translator": "Generando informe",
    None: "Completado",
}

def _start_terminal_log_tail(run_id: str) -> None:
    """Stream worker stdout.log to Streamlit's terminal (tail -f style).

    Runs as a daemon thread — if Streamlit dies, the thread dies silently
    but the worker process keeps running independently (stdout goes to file).
    """
    log_path = os.path.join("runs", run_id, "worker_stdout.log")

    def _tail(tail_log_path, tail_run_id):
        try:
            for _ in range(30):
                if os.path.exists(tail_log_path):
                    break
                time.sleep(0.5)
            pid = 0
            for _ in range(20):
                status = _read_run_status(tail_run_id)
                if isinstance(status, dict) and status.get("pid"):
                    pid = int(status["pid"])
                    break
                time.sleep(0.5)
            with open(tail_log_path, "r", encoding="utf-8", errors="replace") as f:
                while True:
                    line = f.readline()
                    if line:
                        sys.stderr.write(line)
                        sys.stderr.flush()
                    else:
                        if pid and not _is_process_alive(pid):
                            for remaining in f:
                                sys.stderr.write(remaining)
                                sys.stderr.flush()
                            break
                        time.sleep(0.3)
        except Exception:
            pass

    threading.Thread(target=_tail, args=(log_path, run_id), daemon=True).start()


# ---------------------------------------------------------------------------
# Background Worker Polling UI
# ---------------------------------------------------------------------------
def _run_polling_ui(run_id: str) -> None:
    """Poll background worker status files and update Streamlit UI in real time."""
    progress_header_placeholder = st.empty()
    abort_col1, abort_col2, abort_col3 = st.columns([3, 1, 3])
    with abort_col2:
        if st.button("Cancelar ejecuci\u00f3n", type="secondary", use_container_width=True):
            _request_run_abort(run_id)
            _kill_worker(run_id)
            st.session_state.pop("active_run_id", None)
            st.warning("Ejecuci\u00f3n cancelada.")
            time.sleep(1)
            st.rerun()
            return
    progress_bar = st.progress(0)
    pipeline_placeholder = st.empty()
    log_placeholder = st.empty()
    sidebar_status_placeholder = st.sidebar.empty()

    log_lines_read = 0
    log_html_entries: list[str] = []

    while True:
        status = _read_run_status(run_id)
        if not status:
            time.sleep(2)
            continue

        # Read new log entries
        new_logs = _read_log_entries(run_id, after_line=log_lines_read)
        for entry in new_logs:
            ts = entry.get("ts", "")
            agent = entry.get("agent", "")
            msg = entry.get("msg", "")
            level = entry.get("level", "info")
            cls = {"ok": "log-ok", "warn": "log-warn", "info": ""}.get(level, "")
            log_html_entries.append(
                f'<div class="log-entry">'
                f'<span class="log-time">[{ts}]</span> '
                f'<span class="log-agent">{agent}</span> '
                f'<span class="{cls}">{msg}</span>'
                f'</div>'
            )
        log_lines_read += len(new_logs)

        # Current state from worker
        stage = status.get("stage")
        stage_name = status.get("stage_name") or _STAGE_NAMES.get(stage, stage or "Procesando")
        progress = status.get("progress", 0)
        iteration = status.get("iteration", 0)
        max_iterations = status.get("max_iterations", 6)
        metric_name = status.get("metric_name", "")
        metric_value = status.get("metric_value", "")
        completed_steps_set = set(status.get("completed_steps", []))
        started_at = status.get("started_at", time.time())

        elapsed_str = _fmt_elapsed(time.time() - started_at)

        # --- Update UI elements ---
        progress_header_placeholder.markdown(f"""
        <div class="progress-header">
            <div class="progress-timer">
                <span class="progress-timer-icon">&#9202;</span>
                <span>{elapsed_str}</span>
            </div>
            <div style="text-align:center;">
                <div class="progress-stage">{stage_name}</div>
            </div>
            <div class="progress-pct">{progress}%</div>
        </div>
        """, unsafe_allow_html=True)

        progress_bar.progress(min(progress, 100))

        # Iteration badge
        iter_info = ""
        if iteration >= 1:
            parts = [
                f'<span class="iter-badge-label">Iteraci\u00f3n</span>',
                f'<span class="iter-badge-value">{iteration}/{max_iterations}</span>',
            ]
            if metric_value:
                parts.append(f'<span class="iter-badge-sep">|</span>')
                parts.append(f'<span class="iter-badge-metric">{metric_name}: {metric_value}</span>')
            iter_info = '<div class="iter-badge">' + " ".join(parts) + '</div>'

        pipeline_placeholder.markdown(
            '<div class="card">'
            + _render_pipeline(completed_steps_set, stage, iter_info)
            + '</div>',
            unsafe_allow_html=True
        )

        log_placeholder.markdown(
            '<div class="activity-log">' + "\n".join(log_html_entries) + '</div>',
            unsafe_allow_html=True
        )

        # Sidebar run status
        iter_display = f"{iteration}/{max_iterations}" if iteration > 0 else "--"
        metric_display = f"{metric_name}: {metric_value}" if metric_value else "--"
        sidebar_status_placeholder.markdown(f"""
        <div class="sidebar-run-status">
            <div class="srs-title">Ejecuci&oacute;n en Curso</div>
            <div class="srs-row">
                <span class="srs-label">Etapa</span>
                <span class="srs-step">{stage_name}</span>
            </div>
            <div class="srs-row">
                <span class="srs-label">Progreso</span>
                <span class="srs-value">{progress}%</span>
            </div>
            <div class="srs-row">
                <span class="srs-label">Tiempo</span>
                <span class="srs-timer">{elapsed_str}</span>
            </div>
            <div class="srs-row">
                <span class="srs-label">Iteraci&oacute;n ML</span>
                <span class="srs-value">{iter_display}</span>
            </div>
            <div class="srs-row">
                <span class="srs-label">Metrica</span>
                <span class="srs-value">{metric_display}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # --- Check for completion ---
        run_status = status.get("status", "running")
        if run_status == "complete":
            final = _read_final_state(run_id)
            if final:
                st.session_state["analysis_result"] = final
                st.session_state["analysis_complete"] = True
            st.session_state.pop("active_run_id", None)
            sidebar_status_placeholder.empty()
            time.sleep(0.5)
            st.rerun()
            return

        if run_status in ("error", "aborted"):
            if run_status == "aborted":
                st.warning("Ejecuci\u00f3n cancelada por el usuario.")
            else:
                error = status.get("error", "Error desconocido")
                st.error(f"Error en la ejecuci\u00f3n: {error}")
            final = _read_final_state(run_id)
            if final:
                st.session_state["analysis_result"] = final
                st.session_state["analysis_complete"] = True
            st.session_state.pop("active_run_id", None)
            sidebar_status_placeholder.empty()
            if st.session_state.get("analysis_complete"):
                time.sleep(0.5)
                st.rerun()
            return

        # Check if worker process died unexpectedly
        pid = status.get("pid")
        if pid and not _is_process_alive(pid):
            st.error("El proceso de ejecuci\u00f3n ha terminado inesperadamente.")
            final = _read_final_state(run_id)
            if final:
                st.session_state["analysis_result"] = final
                st.session_state["analysis_complete"] = True
            st.session_state.pop("active_run_id", None)
            sidebar_status_placeholder.empty()
            if st.session_state.get("analysis_complete"):
                time.sleep(0.5)
                st.rerun()
            return

        time.sleep(3)

# ---------------------------------------------------------------------------
# Start Analysis / Resume Active Run
# ---------------------------------------------------------------------------
if st.session_state.get("active_run_id") and not st.session_state.get("analysis_complete"):
    # Resume polling an active background worker (survives session reconnect)
    _start_terminal_log_tail(st.session_state["active_run_id"])
    _run_polling_ui(st.session_state["active_run_id"])

elif start_btn:
    if data_path is None:
        st.error("Adjunta un archivo CSV o conecta un CRM antes de iniciar.")
    elif not business_objective:
        st.error("Define un objetivo de negocio antes de iniciar el an\u00e1lisis.")
    else:
        sandbox_config = _load_sandbox_config()
        sandbox_provider = str(sandbox_config.get("provider") or "local").strip().lower() or "local"
        sandbox_spec = _get_sandbox_provider_spec(sandbox_provider)
        if not _is_sandbox_provider_available(sandbox_provider):
            st.error(
                f"El sandbox seleccionado ({sandbox_spec.label}) no tiene backend registrado en este despliegue. "
                "Selecciona Local o instala el provider remoto antes de lanzar la run."
            )
            st.stop()

        st.session_state["analysis_complete"] = False
        st.session_state["analysis_result"] = None
        st.session_state.pop("dismissed_latest_run", None)
        st.session_state.pop("viewing_run_id", None)
        clear_abort()

        # Kill any previously running worker before starting a new one
        prev_run = _get_active_run_id()
        if prev_run:
            _request_run_abort(prev_run)
            _kill_worker(prev_run)

        if os.path.exists("static/plots"):
            files = glob.glob("static/plots/*")
            for f in files:
                os.remove(f)

        # Generate run_id and write worker input
        run_id = _uuid_mod.uuid4().hex[:8]
        _write_worker_input(run_id, data_path, business_objective, sandbox_config=sandbox_config)

        # Launch background worker as independent subprocess
        worker_log_dir = os.path.join("runs", run_id)
        os.makedirs(worker_log_dir, exist_ok=True)
        worker_stdout_path = os.path.join(worker_log_dir, "worker_stdout.log")
        worker_stdout = open(worker_stdout_path, "w", encoding="utf-8")
        subprocess.Popen(
            [sys.executable, "-m", "src.utils.background_worker", run_id],
            cwd=APP_ROOT,
            stdout=worker_stdout,
            stderr=subprocess.STDOUT,
            # Detach from parent so it survives Streamlit session death
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        worker_stdout.close()  # child process has inherited the handle

        _start_terminal_log_tail(run_id)

        st.session_state["active_run_id"] = run_id
        _run_polling_ui(run_id)

# ---------------------------------------------------------------------------
# Results Dashboard
# ---------------------------------------------------------------------------
if st.session_state.get("analysis_complete") and st.session_state.get("analysis_result"):
    result = st.session_state["analysis_result"]

    # Back to home button — always visible in results dashboard
    viewing_run = st.session_state.get("viewing_run_id")
    run_id_display = viewing_run or result.get("run_id", "")

    col_back, col_run_label = st.columns([1, 5])
    with col_back:
        if st.button("\u2190 Nuevo an\u00e1lisis", key="back_to_home", use_container_width=True):
            st.session_state["analysis_complete"] = False
            st.session_state["analysis_result"] = None
            st.session_state["dismissed_latest_run"] = True
            st.session_state.pop("viewing_run_id", None)
            st.rerun()
    with col_run_label:
        if run_id_display:
            st.markdown(
                f'<span style="color:var(--text-secondary); font-size:0.9rem;">'
                f'Ejecuci\u00f3n: <strong>{run_id_display}</strong></span>',
                unsafe_allow_html=True,
            )

    # Success Banner — use board's authoritative verdict when available
    _board_pld = result.get("review_board_verdict")
    verdict = (
        (_board_pld.get("final_review_verdict") if isinstance(_board_pld, dict) else None)
        or result.get("review_verdict")
        or "APPROVED"
    )
    if verdict in ("REJECTED", "FAIL", "CRASH"):
        st.markdown("""
        <div class="result-banner error fade-in">
            <div class="result-banner-icon">&#10060;</div>
            <div class="result-banner-text">An&aacute;lisis finalizado con rechazo del revisor</div>
        </div>
        """, unsafe_allow_html=True)
    elif verdict == "NEEDS_IMPROVEMENT":
        st.markdown("""
        <div class="result-banner error fade-in">
            <div class="result-banner-icon">&#9888;&#65039;</div>
            <div class="result-banner-text">An&aacute;lisis completado con observaciones del revisor</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-banner success fade-in">
            <div class="result-banner-icon">&#9989;</div>
            <div class="result-banner-text">An&aacute;lisis completado exitosamente</div>
        </div>
        """, unsafe_allow_html=True)

    # Summary metric cards
    iteration_count = result.get('iteration_count', result.get('current_iteration', 'N/A'))
    selected_strat = result.get('selected_strategy', {})
    strat_title = selected_strat.get('title', 'N/A') if isinstance(selected_strat, dict) else 'N/A'

    # Map verdict and gate labels to Spanish
    _VERDICT_LABELS = {
        "APPROVED": "Aprobado", "APPROVE_WITH_WARNINGS": "Aprobado con Avisos",
        "NEEDS_IMPROVEMENT": "Necesita Mejoras", "REJECTED": "Rechazado",
        "FAIL": "Fallido", "CRASH": "Error Cr\u00edtico",
    }
    _GATE_LABELS = {"PASSED": "Superado", "FAILED": "No Superado"}

    # Estimate API cost from budget counters
    _counters = result.get('budget_counters', {})
    if not isinstance(_counters, dict):
        _counters = {}
    _total_api_calls = sum(
        int(_counters.get(k, 0))
        for k in ("de_calls", "ml_calls", "reviewer_calls", "qa_calls", "execution_calls")
    )
    # Fixed agents that always run once: steward, strategist, planner, translator
    _fixed_calls = 4
    _total_llm_calls = _total_api_calls + _fixed_calls
    # Rough cost estimate: ~$0.05 per LLM call average (mix of cheap/expensive models)
    _est_cost = _total_llm_calls * 0.05
    _cost_str = f"~${_est_cost:.2f}"
    _cost_color = "var(--success)" if _est_cost < 1.5 else "var(--warning)" if _est_cost < 3.0 else "var(--danger)"

    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    with mc1:
        st.markdown(f"""
        <div class="card fade-in" style="text-align:center;">
            <div class="card-header">Estrategia</div>
            <div style="font-size:0.95rem; font-weight:700; color:var(--text-primary);">{strat_title}</div>
        </div>
        """, unsafe_allow_html=True)
    with mc2:
        st.markdown(f"""
        <div class="card fade-in" style="text-align:center;">
            <div class="card-header">Iteraciones ML</div>
            <div class="card-value">{iteration_count}</div>
        </div>
        """, unsafe_allow_html=True)
    with mc3:
        rv = (
            (_board_pld.get("final_review_verdict") if isinstance(_board_pld, dict) else None)
            or result.get("review_verdict")
            or "N/A"
        )
        rv_label = _VERDICT_LABELS.get(rv, rv)
        badge_cls = "badge-success" if rv in ("APPROVED", "APPROVE_WITH_WARNINGS") else "badge-warning"
        st.markdown(f"""
        <div class="card fade-in" style="text-align:center;">
            <div class="card-header">Veredicto</div>
            <div style="margin-top:0.5rem;"><span class="badge {badge_cls}">{rv_label}</span></div>
        </div>
        """, unsafe_allow_html=True)
    with mc4:
        raw_gate = result.get('gate_status', 'N/A')
        gate_label = _GATE_LABELS.get(raw_gate, raw_gate)
        st.markdown(f"""
        <div class="card fade-in" style="text-align:center;">
            <div class="card-header">Control de Calidad</div>
            <div style="font-size:0.95rem; font-weight:700; color:var(--text-primary);">{gate_label}</div>
        </div>
        """, unsafe_allow_html=True)
    with mc5:
        st.markdown(f"""
        <div class="card fade-in" style="text-align:center;">
            <div class="card-header">Coste Estimado</div>
            <div style="font-size:1.1rem; font-weight:700; color:{_cost_color};">{_cost_str}</div>
            <div style="font-size:0.7rem; color:var(--text-secondary);">{_total_llm_calls} llamadas API</div>
        </div>
        """, unsafe_allow_html=True)

    # Tabs
    tab_init, tab1, tab2, tab_plan, tab_de, tab3, tab4 = st.tabs([
        "Estado Inicial",
        "Auditor\u00eda de Datos",
        "Estrategia",
        "Plan de Ejecuci\u00f3n",
        "Ingenier\u00eda de Datos",
        "Modelo ML",
        "Informe Ejecutivo"
    ])

    # --- Tab 0: Initial State ---
    with tab_init:
        st.markdown("#### Objetivo de Negocio")
        _biz_obj = result.get("business_objective", "")
        if _biz_obj:
            st.markdown(f'<div class="card fade-in" style="white-space:pre-wrap;">{_biz_obj}</div>',
                        unsafe_allow_html=True)
        else:
            st.info("No se registr\u00f3 un objetivo de negocio para esta ejecuci\u00f3n.")

        st.markdown("#### Vista Previa del Dataset")
        _csv_path = result.get("csv_path", "")
        if _csv_path and os.path.isfile(_csv_path):
            try:
                _init_df = pd.read_csv(_csv_path, nrows=10)
                _init_shape = pd.read_csv(_csv_path, nrows=0)
                # Show dataset dimensions
                import csv as _csv_mod
                _total_rows = sum(1 for _ in open(_csv_path, encoding="utf-8", errors="ignore")) - 1
                _total_cols = len(_init_shape.columns)
                _dim_cols = st.columns(3)
                with _dim_cols[0]:
                    st.markdown(f'<div class="card fade-in" style="text-align:center;">'
                                f'<div class="card-header">Filas</div>'
                                f'<div class="card-value">{_total_rows:,}</div>'
                                f'</div>', unsafe_allow_html=True)
                with _dim_cols[1]:
                    st.markdown(f'<div class="card fade-in" style="text-align:center;">'
                                f'<div class="card-header">Columnas</div>'
                                f'<div class="card-value">{_total_cols}</div>'
                                f'</div>', unsafe_allow_html=True)
                with _dim_cols[2]:
                    _fname = os.path.basename(_csv_path)
                    st.markdown(f'<div class="card fade-in" style="text-align:center;">'
                                f'<div class="card-header">Archivo</div>'
                                f'<div style="font-size:0.85rem;font-weight:700;">{_fname}</div>'
                                f'</div>', unsafe_allow_html=True)
                st.dataframe(_init_df, use_container_width=True)
            except Exception as _csv_err:
                st.warning(f"No se pudo cargar el CSV: {_csv_err}")
        elif _csv_path:
            st.warning(f"El archivo CSV ya no est\u00e1 disponible en: {_csv_path}")
        else:
            st.info("No se registr\u00f3 la ruta del dataset para esta ejecuci\u00f3n.")

    # --- Tab 1: Data Audit ---
    with tab1:
        st.markdown("#### Auditor\u00eda de Datos")
        data_summary = result.get('data_summary', '')
        if data_summary:
            st.markdown(f'<div class="card fade-in">{data_summary}</div>', unsafe_allow_html=True)
        else:
            st.info("No se gener\u00f3 un resumen de auditor\u00eda para esta ejecuci\u00f3n.")

    # --- Tab 2: Strategy ---
    with tab2:
        st.markdown("#### Plan Estrat\u00e9gico")
        strategies = result.get('strategies', {})

        if isinstance(strategies, dict) and 'strategies' in strategies:
            for i, strat in enumerate(strategies['strategies'], 1):
                with st.expander(f"Estrategia {i}: {strat.get('title')}", expanded=(i == 1)):
                    st.write(f"**Hip\u00f3tesis:** {strat.get('hypothesis')}")
                    st.write(f"**Dificultad estimada:** {strat.get('estimated_difficulty')}")
                    st.write(f"**Razonamiento:** {strat.get('reasoning')}")
        else:
            st.json(strategies)

        selected = result.get('selected_strategy', {})

        if selected:
            st.markdown(f"""
            <div class="winner-card fade-in">
                <strong>Estrategia Seleccionada:</strong> {selected.get('title', 'N/A')}<br>
                <span style="color:var(--text-secondary);">{result.get('selection_reason', '')}</span>
            </div>
            """, unsafe_allow_html=True)


    # --- Tab: Execution Plan ---
    with tab_plan:
        st.markdown("#### Plan de Ejecuci\u00f3n")
        contract = result.get('execution_contract', {})
        if isinstance(contract, dict) and contract:
            # Summary cards
            _plan_cols = st.columns(4)
            with _plan_cols[0]:
                _pt = contract.get('problem_type', 'N/A')
                st.markdown(f'<div class="card fade-in" style="text-align:center;">'
                            f'<div class="card-header">Tipo de Problema</div>'
                            f'<div style="font-size:0.95rem;font-weight:700;">{_pt}</div>'
                            f'</div>', unsafe_allow_html=True)
            with _plan_cols[1]:
                _eval = contract.get('evaluation_spec', {}) if isinstance(contract.get('evaluation_spec'), dict) else {}
                _pm = _eval.get('primary_metric', contract.get('primary_metric', 'N/A'))
                st.markdown(f'<div class="card fade-in" style="text-align:center;">'
                            f'<div class="card-header">M\u00e9trica Principal</div>'
                            f'<div style="font-size:0.95rem;font-weight:700;">{_pm}</div>'
                            f'</div>', unsafe_allow_html=True)
            with _plan_cols[2]:
                _deps = contract.get('required_dependencies', [])
                _deps_str = ', '.join(_deps[:6]) if isinstance(_deps, list) and _deps else 'Est\u00e1ndar'
                st.markdown(f'<div class="card fade-in" style="text-align:center;">'
                            f'<div class="card-header">Dependencias</div>'
                            f'<div style="font-size:0.85rem;">{_deps_str}</div>'
                            f'</div>', unsafe_allow_html=True)
            with _plan_cols[3]:
                _vis = contract.get('artifact_requirements', {})
                _vis = _vis.get('visual_requirements', {}) if isinstance(_vis, dict) else {}
                _vis_items = _vis.get('items', []) if isinstance(_vis, dict) else []
                _vis_count = len(_vis_items) if isinstance(_vis_items, list) else 0
                st.markdown(f'<div class="card fade-in" style="text-align:center;">'
                            f'<div class="card-header">Visualizaciones</div>'
                            f'<div style="font-size:0.95rem;font-weight:700;">{_vis_count} plots</div>'
                            f'</div>', unsafe_allow_html=True)

            # Runbook
            _runbook = contract.get('runbook', '')
            if _runbook:
                with st.expander("Runbook de ejecuci\u00f3n", expanded=True):
                    st.markdown(_runbook)

            # Artifact requirements
            _art_req = contract.get('artifact_requirements', {})
            if isinstance(_art_req, dict) and _art_req:
                with st.expander("Requisitos de artefactos"):
                    st.json(_art_req)

            # Full contract JSON (collapsed)
            with st.expander("Contrato completo (JSON)"):
                st.json(contract)
        else:
            st.info("No se gener\u00f3 un plan de ejecuci\u00f3n para esta run.")

    # --- Tab: Data Engineering ---
    with tab_de:
        st.markdown("#### Ingenier\u00eda de Datos")

        code = result.get('cleaning_code', '')
        preview = result.get('cleaned_data_preview', '')

        col_de_code, col_de_preview = st.columns(2)

        with col_de_code:
            st.markdown("**Script de Procesamiento Generado**")
            if code:
                st.code(code, language='python')
            else:
                st.info("No se gener\u00f3 c\u00f3digo de procesamiento en esta ejecuci\u00f3n.")

        with col_de_preview:
            st.markdown("**Vista Previa de Datos Procesados**")
            if isinstance(preview, str) and preview.strip().startswith('{'):
                try:
                    from io import StringIO
                    st.dataframe(pd.read_json(StringIO(preview), orient='split'), use_container_width=True)
                except Exception:
                    st.write(preview)
            elif preview:
                st.write(preview)
            else:
                st.info("No hay vista previa disponible.")

    # --- Tab 4: ML Model ---
    with tab3:
        st.markdown("#### Modelo ML")

        col_code, col_out = st.columns(2)

        with col_code:
            st.markdown("**C\u00f3digo del Modelo**")
            ml_code = result.get('generated_code', '')
            if ml_code.strip() == "# Generation Failed":
                ml_code = result.get('last_generated_code', ml_code)
            if ml_code:
                st.code(ml_code, language='python')
            else:
                st.info("No se gener\u00f3 c\u00f3digo de modelo en esta ejecuci\u00f3n.")

        with col_out:
            st.markdown("**Registro de Ejecuci\u00f3n**")
            ml_output = result.get('execution_output', '')
            last_ok = result.get('last_successful_execution_output')
            if "BUDGET_EXCEEDED" in str(ml_output) and last_ok:
                ml_output = f"{ml_output}\n\n--- \u00daltima ejecuci\u00f3n exitosa ---\n{last_ok}"
            import html as html_mod
            escaped_output = html_mod.escape(str(ml_output))
            if escaped_output.strip():
                st.markdown(f'<div class="console-output">{escaped_output}</div>', unsafe_allow_html=True)
            else:
                st.info("No hay registro de ejecuci\u00f3n disponible.")

    # --- Tab 5: Executive Report ---
    with tab4:
        st.markdown("#### Informe Ejecutivo")
        final_report = result.get('final_report', '')
        if final_report:
            # Render report with inline images: split on ![alt](path) and
            # render each image via st.image() so they appear inline.
            import re as _re
            _img_pattern = _re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
            _report_run_id = st.session_state.get("viewing_run_id") or result.get("run_id")
            _last_end = 0
            for _m in _img_pattern.finditer(final_report):
                # Render markdown text before this image
                _text_before = final_report[_last_end:_m.start()].strip()
                if _text_before:
                    st.markdown(_text_before)
                # Resolve image path from run dir or cwd
                _img_alt = _m.group(1)
                _img_rel = _m.group(2)
                _img_resolved = None
                if _report_run_id:
                    _candidate = os.path.join("runs", str(_report_run_id), "work", _img_rel)
                    if os.path.isfile(_candidate):
                        _img_resolved = _candidate
                if not _img_resolved and os.path.isfile(_img_rel):
                    _img_resolved = _img_rel
                if _img_resolved:
                    st.image(_img_resolved, caption=_img_alt or None, use_container_width=True)
                _last_end = _m.end()
            # Render remaining text after last image (or full report if no images)
            _remainder = final_report[_last_end:].strip()
            if _remainder:
                st.markdown(_remainder)
        else:
            st.info("No se gener\u00f3 un informe ejecutivo para esta ejecuci\u00f3n.")

        # Plot gallery — resolve from run directory first, fallback to cwd
        _plot_run_id = st.session_state.get("viewing_run_id") or result.get("run_id")
        _plot_roots = []
        if _plot_run_id:
            _plot_roots.append(os.path.join("runs", str(_plot_run_id), "work", "static", "plots"))
        _plot_roots.append("static/plots")
        plots = []
        for _pr in _plot_roots:
            plots = glob.glob(os.path.join(_pr, "*.png"))
            if plots:
                break
        if plots:
            st.markdown("#### Gr\u00e1ficos y Visualizaciones")
            cols = st.columns(min(len(plots), 3))
            for i, plot_path in enumerate(plots):
                with cols[i % len(cols)]:
                    st.image(plot_path, caption=os.path.basename(plot_path), use_container_width=True)

        # Downloads section
        st.markdown("---")
        st.markdown("#### Descargar Resultados")
        dl_col1, dl_col2 = st.columns(2)

        # PDF Download
        if 'pdf_binary' not in st.session_state:
            pdf_path = result.get('pdf_path')
            if pdf_path and os.path.exists(pdf_path):
                try:
                    with open(pdf_path, "rb") as pdf_file:
                        st.session_state['pdf_binary'] = pdf_file.read()
                except Exception as e:
                    st.warning(f"No se pudo cargar el PDF: {e}")

        with dl_col1:
            if 'pdf_binary' in st.session_state:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M')
                st.download_button(
                    label="Descargar Informe PDF",
                    data=st.session_state['pdf_binary'],
                    file_name=f"Informe_Ejecutivo_{timestamp}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

        # ML Artifacts ZIP
        output_report = result.get("output_contract_report")
        if not isinstance(output_report, dict):
            # Try run-specific path first, then CWD fallback
            _dl_run_id = st.session_state.get("viewing_run_id") or result.get("run_id")
            if _dl_run_id:
                output_report = _load_json(os.path.join("runs", str(_dl_run_id), "work", "data", "output_contract_report.json"))
            if not isinstance(output_report, dict):
                output_report = _load_json("data/output_contract_report.json") or {}
        present_files = _resolve_ml_artifact_files(output_report, result if isinstance(result, dict) else {})

        with dl_col2:
            if present_files:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    for file_path, arcname in present_files:
                        zf.write(file_path, arcname=arcname)
                zip_buffer.seek(0)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M')
                st.download_button(
                    label="Descargar Artefactos ML (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name=f"Entregables_ML_{timestamp}.zip",
                    mime="application/zip",
                    use_container_width=True
                )
            else:
                st.info("No se encontraron artefactos ML en esta ejecuci\u00f3n.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("""
<div class="footer">
    <span>&copy; 2026 StrategyEngine AI &mdash; Plataforma de IA Multi-Agente</span>
</div>
""", unsafe_allow_html=True)
