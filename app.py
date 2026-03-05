import streamlit as st
import pandas as pd
import os
import json
import sys
import time
import glob
import signal
import threading
import io
import zipfile
from datetime import datetime
from typing import Any, Dict, List, Tuple

# Ensure src is in path
APP_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(APP_ROOT)

from src.graph.graph import (
    app_graph,
    request_abort,
    clear_abort,
    get_runtime_agent_models,
    set_runtime_agent_models,
)
from src.utils.run_workspace import recover_orphaned_workspace_cwd

# Auto-heal cwd when prior run crashed inside runs/<run_id>/work.
recover_orphaned_workspace_cwd(project_root=APP_ROOT)
try:
    os.chdir(APP_ROOT)
except Exception as cwd_err:
    print(f"APP_CWD_WARNING: {cwd_err}")

_SIGNAL_HANDLER_INSTALLED = False

AGENT_MODEL_LABELS: Dict[str, str] = {
    "strategist": "Strategist",
    "data_engineer": "Data Engineer",
    "ml_engineer": "ML Engineer",
}
MODEL_PRESET_OPTIONS: List[Tuple[str, str]] = [
    ("z-ai/glm-5", "GLM-5"),
    ("moonshotai/kimi-k2.5", "Kimi K2.5"),
    ("minimax/minimax-m2.5", "Minimax M-2.5"),
    ("deepseek/deepseek-chat-v3.2", "DeepSeek V3.2"),
    ("anthropic/claude-opus-4.6", "Claude Opus 4.6"),
    ("openai/chatgpt-5.2", "ChatGPT 5.2"),
    ("openai/gpt-5.3-codex", "GPT-5.3 Codex"),
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
    for agent_key in AGENT_MODEL_LABELS:
        value = str(raw.get(agent_key) or "").strip()
        if value:
            cleaned[agent_key] = value
    return cleaned


def _merge_agent_model_maps(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, str]:
    merged: Dict[str, str] = {}
    base_map = _sanitize_agent_model_map(base)
    overrides_map = _sanitize_agent_model_map(overrides)
    for agent_key in AGENT_MODEL_LABELS:
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
        return "Personalizado (ID OpenRouter)"
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
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div style="padding: 0.5rem 0 0.25rem;">
    <h1 style="margin:0; font-size:2rem; font-weight:800;">
        <span class="hero-gradient">StrategyEngine AI</span>
    </h1>
    <p style="margin:0.25rem 0 0; color:#6c757d; font-size:0.95rem;">
        Plataforma de Inteligencia de Negocio Autonoma &mdash; Multi-Agent AI
    </p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 0.5rem 0 1rem;">
        <span style="font-size:1.6rem; font-weight:800;" class="hero-gradient">StrategyEngine AI</span>
        <br>
        <span style="font-size:0.72rem; color:#8b949e; letter-spacing:0.05em;">v2.0 &bull; Enterprise AI Platform</span>
    </div>
    """, unsafe_allow_html=True)

    if st.button("⚙ Ajustes de modelos", key="toggle_model_settings", use_container_width=True):
        st.session_state["show_model_settings"] = not bool(st.session_state.get("show_model_settings"))

    active_models = _merge_agent_model_maps(
        st.session_state.get("base_agent_models", {}),
        st.session_state.get("agent_model_overrides", {}),
    )
    active_model_lines = "".join(
        f"<div><strong>{AGENT_MODEL_LABELS[agent_key]}:</strong> {active_models.get(agent_key, 'N/A')}</div>"
        for agent_key in AGENT_MODEL_LABELS
    )
    st.markdown(
        f"""
        <div class="sidebar-settings-panel">
            <div class="ssp-title">Modelos activos</div>
            <div class="ssp-desc">{active_model_lines}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.get("show_model_settings"):
        st.markdown(
            """
            <div class="sidebar-settings-panel">
                <div class="ssp-title">Configuracion de modelos por agente</div>
                <div class="ssp-desc">Selecciona el modelo principal para cada agente (ruteado por OpenRouter).</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        pending_models: Dict[str, str] = {}
        preset_ids = [model_id for model_id, _ in MODEL_PRESET_OPTIONS]
        selector_options = preset_ids + [CUSTOM_MODEL_OPTION]

        for agent_key, agent_label in AGENT_MODEL_LABELS.items():
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
                    f"ID OpenRouter ({agent_label})",
                    value=custom_default,
                    placeholder="proveedor/modelo",
                    key=f"model_custom_{agent_key}",
                )
                pending_models[agent_key] = str(custom_model or "").strip()
            else:
                pending_models[agent_key] = selected_option

        apply_col, reset_col = st.columns(2)
        with apply_col:
            apply_models_btn = st.button("Guardar y aplicar", key="apply_agent_models", use_container_width=True)
        with reset_col:
            reset_models_btn = st.button("Restablecer", key="reset_agent_models", use_container_width=True)

        if apply_models_btn:
            missing_agents = [AGENT_MODEL_LABELS[k] for k, v in pending_models.items() if not str(v or "").strip()]
            if missing_agents:
                st.error(f"Falta definir modelo para: {', '.join(missing_agents)}")
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
                st.success("Configuracion aplicada.")
                st.rerun()

        if reset_models_btn:
            default_models = _merge_agent_model_maps({}, st.session_state.get("base_agent_models", {}))
            applied_models = _sanitize_agent_model_map(set_runtime_agent_models(default_models))
            st.session_state["agent_model_overrides"] = _merge_agent_model_maps(default_models, applied_models)
            _save_agent_model_overrides(st.session_state["agent_model_overrides"])
            for agent_key in AGENT_MODEL_LABELS:
                option_key = f"model_option_{agent_key}"
                custom_key = f"model_custom_{agent_key}"
                if option_key in st.session_state:
                    del st.session_state[option_key]
                if custom_key in st.session_state:
                    del st.session_state[custom_key]
            st.success("Modelos restablecidos.")
            st.rerun()

    st.markdown("---")
    st.markdown("##### Fuente de Datos")

    data_source = st.radio(
        "Selecciona la fuente de datos",
        ["Archivo Local", "Salesforce", "HubSpot"],
        label_visibility="collapsed",
    )

    uploaded_file = None

    # ---- Archivo Local ----
    if data_source == "Archivo Local":
        uploaded_file = st.file_uploader("Cargar archivo CSV o Excel", type=["csv", "xlsx", "xls"])
        if uploaded_file is not None:
            file_size = uploaded_file.size
            if file_size < 1024:
                size_str = f"{file_size} B"
            elif file_size < 1024 * 1024:
                size_str = f"{file_size / 1024:.1f} KB"
            else:
                size_str = f"{file_size / (1024*1024):.1f} MB"
            st.markdown(f'<span class="metric-pill">{uploaded_file.name} &mdash; {size_str}</span>', unsafe_allow_html=True)

    # ---- Salesforce ----
    elif data_source == "Salesforce":
        sf_auth_mode = st.selectbox("Modo de autenticacion", ["Token API", "OAuth (Access Token)"], key="sf_auth_mode")

        if sf_auth_mode == "Token API":
            sf_username = st.text_input("Username", key="sf_username")
            sf_password = st.text_input("Password", type="password", key="sf_password")
            sf_token = st.text_input("Security Token", type="password", key="sf_security_token")
            sf_connect = st.button("Conectar a Salesforce", key="sf_connect")

            if sf_connect and sf_username and sf_password and sf_token:
                try:
                    from src.connectors.salesforce_connector import SalesforceConnector
                    connector = SalesforceConnector()
                    connector.authenticate({
                        "mode": "token",
                        "username": sf_username,
                        "password": sf_password,
                        "security_token": sf_token,
                    })
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
                    connector.authenticate({
                        "mode": "oauth",
                        "access_token": sf_access_token,
                        "instance_url": sf_instance_url,
                    })
                    st.session_state["crm_connector"] = connector
                    st.session_state["crm_authenticated"] = True
                    st.session_state["crm_objects"] = connector.list_objects()
                except Exception as exc:
                    st.error(f"Error: {exc}")
                    st.session_state["crm_authenticated"] = False

        # Object selection & data fetch (shared for both SF auth modes)
        if st.session_state.get("crm_authenticated") and type(st.session_state.get("crm_connector")).__name__ == "SalesforceConnector":
            st.markdown('<span class="badge badge-success">Conectado a Salesforce</span>', unsafe_allow_html=True)
            crm_objects = st.session_state.get("crm_objects", [])
            if crm_objects:
                obj_labels = [f"{o['label']} ({o['name']})" for o in crm_objects]
                selected_idx = st.selectbox("Objeto CRM", range(len(obj_labels)), format_func=lambda i: obj_labels[i], key="sf_obj_select")
                max_recs = st.number_input("Max registros", min_value=100, max_value=50000, value=10000, step=500, key="sf_max_recs")
                fetch_btn = st.button("Extraer Datos", key="sf_fetch")

                if fetch_btn:
                    selected_obj = crm_objects[selected_idx]["name"]
                    try:
                        connector = st.session_state["crm_connector"]
                        df_crm = connector.fetch_object_data(selected_obj, max_records=int(max_recs))
                        if df_crm.empty:
                            st.warning(f"El objeto '{selected_obj}' no contiene registros.")
                        else:
                            os.makedirs("data", exist_ok=True)
                            crm_csv = os.path.join("data", f"crm_{selected_obj.lower()}.csv")
                            df_crm.to_csv(crm_csv, index=False, encoding="utf-8")
                            st.session_state["crm_data_path"] = crm_csv
                            st.session_state["crm_preview_df"] = df_crm
                            st.markdown(f'<span class="metric-pill">{len(df_crm):,} registros extraidos</span>', unsafe_allow_html=True)
                    except Exception as exc:
                        st.error(f"Error al extraer datos: {exc}")

            if st.session_state.get("crm_data_path"):
                preview_df = st.session_state.get("crm_preview_df")
                if preview_df is not None:
                    st.markdown(f'<span class="metric-pill">{len(preview_df):,} registros listos</span>', unsafe_allow_html=True)

    # ---- HubSpot ----
    elif data_source == "HubSpot":
        hs_auth_mode = st.selectbox("Modo de autenticacion", ["Private App Token", "OAuth (Access Token)"], key="hs_auth_mode")
        hs_token = st.text_input("Token", type="password", key="hs_token")
        hs_connect = st.button("Conectar a HubSpot", key="hs_connect")

        if hs_connect and hs_token:
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
                selected_idx = st.selectbox("Objeto CRM", range(len(obj_labels)), format_func=lambda i: obj_labels[i], key="hs_obj_select")
                max_recs = st.number_input("Max registros", min_value=100, max_value=50000, value=10000, step=500, key="hs_max_recs")
                fetch_btn = st.button("Extraer Datos", key="hs_fetch")

                if fetch_btn:
                    selected_obj = crm_objects[selected_idx]["name"]
                    try:
                        connector = st.session_state["crm_connector"]
                        df_crm = connector.fetch_object_data(selected_obj, max_records=int(max_recs))
                        if df_crm.empty:
                            st.warning(f"El objeto '{selected_obj}' no contiene registros.")
                        else:
                            os.makedirs("data", exist_ok=True)
                            crm_csv = os.path.join("data", f"crm_{selected_obj.lower()}.csv")
                            df_crm.to_csv(crm_csv, index=False, encoding="utf-8")
                            st.session_state["crm_data_path"] = crm_csv
                            st.session_state["crm_preview_df"] = df_crm
                            st.markdown(f'<span class="metric-pill">{len(df_crm):,} registros extraidos</span>', unsafe_allow_html=True)
                    except Exception as exc:
                        st.error(f"Error al extraer datos: {exc}")

            if st.session_state.get("crm_data_path"):
                preview_df = st.session_state.get("crm_preview_df")
                if preview_df is not None:
                    st.markdown(f'<span class="metric-pill">{len(preview_df):,} registros listos</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("##### Objetivo de Negocio")

    business_objective = st.text_area(
        "Describe el objetivo que deseas lograr con tus datos",
        placeholder="Ej: Reducir el churn de clientes en un 10% identificando los factores clave de abandono...",
        height=130,
        label_visibility="collapsed"
    )

    st.markdown("")  # spacer
    start_btn = st.button("Iniciar Analisis", use_container_width=True)

# ---------------------------------------------------------------------------
# Session State init
# ---------------------------------------------------------------------------
if "analysis_complete" not in st.session_state:
    st.session_state["analysis_complete"] = False
if "analysis_result" not in st.session_state:
    st.session_state["analysis_result"] = None

# ---------------------------------------------------------------------------
# Resolve data_path from any source
# ---------------------------------------------------------------------------
data_path = None

if data_source == "Archivo Local" and uploaded_file is not None:
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

elif data_source in ("Salesforce", "HubSpot"):
    data_path = st.session_state.get("crm_data_path")

# ---------------------------------------------------------------------------
# Welcome Screen (no data loaded and no results)
# ---------------------------------------------------------------------------
if data_path is None and not st.session_state.get("analysis_complete"):
    st.markdown("""
    <div class="hero fade-in">
        <h1><span class="hero-gradient">Inteligencia de Negocio Autonoma</span></h1>
        <p class="hero-subtitle">
            Sube tus datos, define un objetivo y deja que nuestro equipo de agentes IA
            audite, diseñe estrategias, construya modelos y genere un reporte ejecutivo&nbsp;&mdash; todo en minutos.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        st.markdown("""
        <div class="feature-card fade-in">
            <div class="feature-icon">&#128269;</div>
            <div class="feature-title">Auditoria de Datos</div>
            <div class="feature-desc">Analisis automatico de calidad, integridad y distribucion de tus datos con recomendaciones accionables.</div>
        </div>
        """, unsafe_allow_html=True)
    with col_f2:
        st.markdown("""
        <div class="feature-card fade-in">
            <div class="feature-icon">&#127919;</div>
            <div class="feature-title">Estrategia IA</div>
            <div class="feature-desc">Generacion y evaluacion de multiples estrategias analiticas con deliberacion experta para seleccionar la optima.</div>
        </div>
        """, unsafe_allow_html=True)
    with col_f3:
        st.markdown("""
        <div class="feature-card fade-in">
            <div class="feature-icon">&#9881;&#65039;</div>
            <div class="feature-title">ML Automatizado</div>
            <div class="feature-desc">Ingenieria de datos, entrenamiento de modelos y evaluacion iterativa con generacion de reportes ejecutivos.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="steps-container fade-in">
        <div class="step-item">
            <div class="step-number">1</div>
            <div class="step-text"><strong>Sube tus datos</strong>Carga un archivo CSV/Excel o conecta tu CRM desde el panel lateral.</div>
        </div>
        <div class="step-item">
            <div class="step-number">2</div>
            <div class="step-text"><strong>Define tu objetivo</strong>Describe que quieres lograr con tus datos.</div>
        </div>
        <div class="step-item">
            <div class="step-number">3</div>
            <div class="step-text"><strong>Obtiene resultados</strong>Recibe un reporte ejecutivo con insights accionables.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

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
            st.warning("El dataset parece tener solo una columna. Verifica el formato del archivo.")

# ---------------------------------------------------------------------------
# Pipeline steps definition (for visual tracker)
# ---------------------------------------------------------------------------
PIPELINE_STEPS = [
    ("steward",        "Steward",    "&#128270;"),
    ("strategist",     "Strategist", "&#129504;"),
    ("domain_expert",  "Expert",     "&#127942;"),
    ("data_engineer",  "Data Eng",   "&#128295;"),
    ("engineer",       "ML Eng",     "&#9881;"),
    ("evaluate_results","Reviewer",  "&#128269;"),
    ("translator",     "Report",     "&#128202;"),
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
    "steward": 12,
    "strategist": 22,
    "domain_expert": 34,
    "data_engineer": 48,
    "engineer": 60,       # first ML iteration start
    "evaluate_results": 82,
    "translator": 94,
    "generate_pdf": 100,
}

# Friendly stage names for the progress header
_STAGE_NAMES = {
    "steward": "Auditando datos",
    "strategist": "Generando estrategias",
    "domain_expert": "Deliberacion experta",
    "data_engineer": "Limpieza de datos",
    "engineer": "Entrenando modelo ML",
    "evaluate_results": "Evaluando resultados",
    "translator": "Generando reporte",
    None: "Completado",
}

# ---------------------------------------------------------------------------
# Start Analysis
# ---------------------------------------------------------------------------
if start_btn:
    if data_path is None:
        st.sidebar.error("Por favor carga datos: sube un archivo o conecta un CRM.")
    elif not business_objective:
        st.sidebar.error("Por favor define un objetivo de negocio.")
    else:
        st.session_state["analysis_complete"] = False
        st.session_state["analysis_result"] = None
        clear_abort()

        if os.path.exists("static/plots"):
            files = glob.glob("static/plots/*")
            for f in files:
                os.remove(f)

        try:
            # --- UI placeholders ---
            progress_header_placeholder = st.empty()
            progress_bar = st.progress(0)
            pipeline_placeholder = st.empty()
            log_placeholder = st.empty()

            # --- Sidebar status placeholders ---
            sidebar_status_placeholder = st.sidebar.empty()

            # --- Tracking state ---
            completed_steps: set = set()
            active_step: str | None = "steward"
            log_entries: list[str] = []
            run_start = time.time()
            current_progress: int = 0
            ml_iteration: int = 0
            ml_max_iterations: int = 6
            best_metric_name: str = ""
            best_metric_value: str = ""
            current_metric_value: str = ""

            def add_log(agent: str, message: str, level: str = "info"):
                ts = datetime.now().strftime("%H:%M:%S")
                cls = {"ok": "log-ok", "warn": "log-warn", "info": ""}.get(level, "")
                log_entries.append(
                    f'<div class="log-entry">'
                    f'<span class="log-time">[{ts}]</span> '
                    f'<span class="log-agent">{agent}</span> '
                    f'<span class="{cls}">{message}</span>'
                    f'</div>'
                )

            def _build_iter_badge() -> str:
                """Build HTML for the ML iteration badge."""
                if ml_iteration < 1:
                    return ""
                parts = [
                    f'<span class="iter-badge-label">Iteracion</span>',
                    f'<span class="iter-badge-value">{ml_iteration}/{ml_max_iterations}</span>',
                ]
                if current_metric_value:
                    parts.append(f'<span class="iter-badge-sep">|</span>')
                    parts.append(f'<span class="iter-badge-metric">{best_metric_name}: {current_metric_value}</span>')
                return '<div class="iter-badge">' + " ".join(parts) + '</div>'

            def refresh_ui():
                elapsed = time.time() - run_start
                elapsed_str = _fmt_elapsed(elapsed)
                stage_name = _STAGE_NAMES.get(active_step, active_step or "Procesando")

                # Progress header with timer and percentage
                progress_header_placeholder.markdown(f"""
                <div class="progress-header">
                    <div class="progress-timer">
                        <span class="progress-timer-icon">&#9202;</span>
                        <span>{elapsed_str}</span>
                    </div>
                    <div style="text-align:center;">
                        <div class="progress-stage">{stage_name}</div>
                    </div>
                    <div class="progress-pct">{current_progress}%</div>
                </div>
                """, unsafe_allow_html=True)

                # Progress bar
                progress_bar.progress(min(current_progress, 100))

                # Pipeline tracker with iteration badge
                pipeline_placeholder.markdown(
                    '<div class="card">'
                    + _render_pipeline(completed_steps, active_step, _build_iter_badge())
                    + '</div>',
                    unsafe_allow_html=True
                )

                # Activity log
                log_placeholder.markdown(
                    '<div class="activity-log">' + "\n".join(log_entries) + '</div>',
                    unsafe_allow_html=True
                )

                # Sidebar run status
                iter_display = f"{ml_iteration}/{ml_max_iterations}" if ml_iteration > 0 else "--"
                metric_display = f"{best_metric_name}: {current_metric_value}" if current_metric_value else "--"
                sidebar_status_placeholder.markdown(f"""
                <div class="sidebar-run-status">
                    <div class="srs-title">Ejecucion en Curso</div>
                    <div class="srs-row">
                        <span class="srs-label">Etapa</span>
                        <span class="srs-step">{stage_name}</span>
                    </div>
                    <div class="srs-row">
                        <span class="srs-label">Progreso</span>
                        <span class="srs-value">{current_progress}%</span>
                    </div>
                    <div class="srs-row">
                        <span class="srs-label">Tiempo</span>
                        <span class="srs-timer">{elapsed_str}</span>
                    </div>
                    <div class="srs-row">
                        <span class="srs-label">Iteracion ML</span>
                        <span class="srs-value">{iter_display}</span>
                    </div>
                    <div class="srs-row">
                        <span class="srs-label">Metrica</span>
                        <span class="srs-value">{metric_display}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            add_log("Sistema", "Iniciando pipeline de analisis...", "info")
            add_log("Data Steward", "Analizando calidad e integridad de datos...", "info")
            refresh_ui()

            initial_state = {
                "csv_path": data_path,
                "business_objective": business_objective
            }

            final_state = initial_state.copy()

            for event in app_graph.stream(initial_state, config={"recursion_limit": 250}):
                if event is None:
                    continue

                for key, value in event.items():
                    if value is not None:
                        final_state.update(value)

                if 'steward' in event:
                    completed_steps.add("steward")
                    active_step = "strategist"
                    current_progress = _STEP_PROGRESS["steward"]
                    # Extract data summary stats
                    summary = final_state.get('data_summary', '')
                    add_log("Data Steward", "Auditoria de calidad completada.", "ok")
                    add_log("Strategist", "Generando 3 estrategias de alto impacto...", "info")

                elif 'strategist' in event:
                    completed_steps.add("strategist")
                    active_step = "domain_expert"
                    current_progress = _STEP_PROGRESS["strategist"]
                    # Extract strategy titles for detailed logging
                    strategies = final_state.get('strategies', {})
                    if isinstance(strategies, dict) and 'strategies' in strategies:
                        strat_list = strategies['strategies']
                        titles = [s.get('title', '?') for s in strat_list[:3]]
                        add_log("Strategist", f"{len(strat_list)} estrategias generadas:", "ok")
                        for i, t in enumerate(titles, 1):
                            add_log("Strategist", f"  {i}. {t}", "info")
                    else:
                        add_log("Strategist", "Estrategias generadas.", "ok")
                    add_log("Domain Expert", "Evaluando y puntuando cada estrategia...", "info")

                elif 'domain_expert' in event:
                    completed_steps.add("domain_expert")
                    active_step = "data_engineer"
                    current_progress = _STEP_PROGRESS["domain_expert"]
                    selected = final_state.get('selected_strategy', {})
                    reviews = final_state.get('domain_expert_reviews', [])
                    # Log each strategy score
                    if reviews:
                        for rev in reviews:
                            score = rev.get('score', '?')
                            title = rev.get('title', '?')
                            add_log("Domain Expert", f"  {title} — {score}/10", "info")
                    sel_title = selected.get('title', 'N/A') if isinstance(selected, dict) else 'N/A'
                    add_log("Domain Expert", f"Estrategia ganadora: {sel_title}", "ok")
                    add_log("Data Engineer", "Ejecutando script de limpieza y estandarizacion...", "info")

                elif 'data_engineer' in event:
                    completed_steps.add("data_engineer")
                    active_step = "engineer"
                    current_progress = _STEP_PROGRESS["data_engineer"]
                    add_log("Data Engineer", "Dataset limpiado y estandarizado.", "ok")
                    # Detect iteration policy from contract
                    contract = final_state.get('execution_contract', {})
                    if isinstance(contract, dict):
                        iter_policy = contract.get('iteration_policy', {})
                        if isinstance(iter_policy, dict):
                            limit = iter_policy.get('max_iterations', iter_policy.get('limit', ml_max_iterations))
                            if isinstance(limit, (int, float)) and limit >= 1:
                                ml_max_iterations = int(limit)
                        metric = contract.get('primary_metric', contract.get('metric', ''))
                        if metric:
                            best_metric_name = str(metric).upper()
                    if not best_metric_name:
                        best_metric_name = "Metric"
                    ml_iteration = 1
                    add_log("ML Engineer", f"Generando codigo — Iteracion 1/{ml_max_iterations}...", "info")

                elif 'engineer' in event:
                    # ML code generation event
                    iteration = final_state.get('current_iteration', ml_iteration)
                    if isinstance(iteration, int) and iteration > ml_iteration:
                        ml_iteration = iteration
                    add_log("ML Engineer", f"Codigo generado (Iteracion {ml_iteration}). Ejecutando...", "info")
                    # Interpolate progress between data_engineer (48%) and evaluate_results (82%)
                    ml_range = _STEP_PROGRESS["evaluate_results"] - _STEP_PROGRESS["data_engineer"]
                    iter_progress = min(ml_iteration / ml_max_iterations, 1.0)
                    current_progress = _STEP_PROGRESS["data_engineer"] + int(ml_range * iter_progress * 0.7)

                elif 'execute_code' in event:
                    # Extract metric from execution output
                    exec_output = str(final_state.get('execution_output', ''))
                    import re
                    # Try to find common metric patterns in output
                    for pattern in [
                        r'(?:RMSLE|rmsle)[:\s=]+([0-9]+\.?[0-9]*)',
                        r'(?:RMSE|rmse)[:\s=]+([0-9]+\.?[0-9]*)',
                        r'(?:MAE|mae)[:\s=]+([0-9]+\.?[0-9]*)',
                        r'(?:AUC|auc)[:\s=]+([0-9]+\.?[0-9]*)',
                        r'(?:F1|f1)[:\s=]+([0-9]+\.?[0-9]*)',
                        r'(?:Accuracy|accuracy)[:\s=]+([0-9]+\.?[0-9]*)',
                        r'(?:R2|r2|R-squared)[:\s=]+([0-9]+\.?[0-9]*)',
                    ]:
                        match = re.search(pattern, exec_output, re.IGNORECASE)
                        if match:
                            current_metric_value = match.group(1)
                            # Auto-detect metric name if generic
                            if best_metric_name == "Metric":
                                name_match = re.search(r'([A-Za-z0-9_-]+)[:\s=]+' + re.escape(current_metric_value), exec_output)
                                if name_match:
                                    best_metric_name = name_match.group(1).upper()
                            break
                    metric_str = f" — {best_metric_name}: {current_metric_value}" if current_metric_value else ""
                    add_log("ML Engineer", f"Ejecucion completada (Iteracion {ml_iteration}){metric_str}.", "ok")
                    add_log("Reviewer", "Evaluando resultados vs. objetivo de negocio...", "info")
                    active_step = "evaluate_results"

                elif 'evaluate_results' in event:
                    verdict = final_state.get('review_verdict', 'APPROVED')
                    if verdict == "NEEDS_IMPROVEMENT":
                        feedback = final_state.get('execution_feedback', '')
                        # Truncate long feedback for the log
                        if len(feedback) > 200:
                            feedback = feedback[:200] + "..."
                        add_log("Reviewer", f"Requiere mejoras: {feedback}", "warn")
                        ml_iteration += 1
                        if ml_iteration <= ml_max_iterations:
                            add_log("ML Engineer", f"Refinando modelo — Iteracion {ml_iteration}/{ml_max_iterations}...", "info")
                        active_step = "engineer"
                        # Update progress for retry iterations
                        ml_range = _STEP_PROGRESS["evaluate_results"] - _STEP_PROGRESS["data_engineer"]
                        iter_progress = min(ml_iteration / ml_max_iterations, 1.0)
                        current_progress = _STEP_PROGRESS["data_engineer"] + int(ml_range * iter_progress)
                    else:
                        completed_steps.add("engineer")
                        completed_steps.add("evaluate_results")
                        active_step = "translator"
                        current_progress = _STEP_PROGRESS["evaluate_results"]
                        metric_str = f" — {best_metric_name}: {current_metric_value}" if current_metric_value else ""
                        add_log("Reviewer", f"Resultados aprobados{metric_str}.", "ok")
                        add_log("Translator", "Generando informe ejecutivo...", "info")

                elif 'retry_handler' in event:
                    pass

                elif 'translator' in event:
                    completed_steps.add("translator")
                    active_step = None
                    current_progress = _STEP_PROGRESS["translator"]
                    add_log("Translator", "Reporte ejecutivo generado.", "ok")

                elif 'generate_pdf' in event:
                    current_progress = _STEP_PROGRESS["generate_pdf"]
                    add_log("Sistema", "PDF final generado.", "ok")

                refresh_ui()

            st.session_state["analysis_result"] = final_state
            st.session_state["analysis_complete"] = True

            # Final pipeline: all complete
            completed_steps = {s[0] for s in PIPELINE_STEPS}
            active_step = None
            current_progress = 100
            elapsed_total = _fmt_elapsed(time.time() - run_start)
            add_log("Sistema", f"Pipeline completado en {elapsed_total}.", "ok")
            refresh_ui()

            time.sleep(0.5)
            # Clear sidebar status before rerun
            sidebar_status_placeholder.empty()
            st.rerun()

        except Exception as e:
            st.error(f"Ocurrio un error critico: {e}")
            st.exception(e)

# ---------------------------------------------------------------------------
# Results Dashboard
# ---------------------------------------------------------------------------
if st.session_state.get("analysis_complete") and st.session_state.get("analysis_result"):
    result = st.session_state["analysis_result"]

    # Success Banner
    verdict = result.get('review_verdict', 'APPROVED')
    if verdict == "NEEDS_IMPROVEMENT":
        st.markdown("""
        <div class="result-banner error fade-in">
            <div class="result-banner-icon">&#9888;&#65039;</div>
            <div class="result-banner-text">Analisis completado con observaciones del revisor</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-banner success fade-in">
            <div class="result-banner-icon">&#9989;</div>
            <div class="result-banner-text">Analisis completado exitosamente</div>
        </div>
        """, unsafe_allow_html=True)

    # Summary metric cards
    iteration_count = result.get('iteration_count', result.get('current_iteration', 'N/A'))
    selected_strat = result.get('selected_strategy', {})
    strat_title = selected_strat.get('title', 'N/A') if isinstance(selected_strat, dict) else 'N/A'

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.markdown(f"""
        <div class="card fade-in" style="text-align:center;">
            <div class="card-header">Estrategia</div>
            <div style="font-size:1rem; font-weight:700; color:var(--text-primary);">{strat_title}</div>
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
        rv = result.get('review_verdict', 'N/A')
        badge_cls = "badge-success" if rv == "APPROVED" else "badge-warning"
        st.markdown(f"""
        <div class="card fade-in" style="text-align:center;">
            <div class="card-header">Veredicto</div>
            <div style="margin-top:0.5rem;"><span class="badge {badge_cls}">{rv}</span></div>
        </div>
        """, unsafe_allow_html=True)
    with mc4:
        gate_status = "Pass" if result.get('gate_status', '') == 'PASSED' else result.get('gate_status', 'N/A')
        st.markdown(f"""
        <div class="card fade-in" style="text-align:center;">
            <div class="card-header">Gate Status</div>
            <div style="font-size:1rem; font-weight:700; color:var(--text-primary);">{gate_status}</div>
        </div>
        """, unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab_de, tab3, tab4 = st.tabs([
        "Auditoria de Datos",
        "Estrategia",
        "Ingenieria de Datos",
        "ML Engineer",
        "Reporte Ejecutivo"
    ])

    # --- Tab 1: Data Audit ---
    with tab1:
        st.markdown("#### Auditoria de Datos")
        data_summary = result.get('data_summary', 'No disponible')
        st.markdown(f'<div class="card fade-in">{data_summary}</div>', unsafe_allow_html=True)

    # --- Tab 2: Strategy ---
    with tab2:
        st.markdown("#### Plan Estrategico")
        strategies = result.get('strategies', {})

        if isinstance(strategies, dict) and 'strategies' in strategies:
            for i, strat in enumerate(strategies['strategies'], 1):
                with st.expander(f"Estrategia {i}: {strat.get('title')}", expanded=(i == 1)):
                    st.write(f"**Hipotesis:** {strat.get('hypothesis')}")
                    st.write(f"**Dificultad:** {strat.get('estimated_difficulty')}")
                    st.write(f"**Razonamiento:** {strat.get('reasoning')}")
        else:
            st.json(strategies)

        selected = result.get('selected_strategy', {})
        reviews = result.get('domain_expert_reviews', [])

        if selected:
            st.markdown(f"""
            <div class="winner-card fade-in">
                <strong>Estrategia Ganadora:</strong> {selected.get('title', 'N/A')}<br>
                <span style="color:var(--text-secondary);">{result.get('selection_reason', 'N/A')}</span>
            </div>
            """, unsafe_allow_html=True)

        if reviews:
            st.markdown("#### Deliberacion del Experto")
            for rev in reviews:
                score = rev.get('score', 'N/A')
                badge_cls = "badge-success" if isinstance(score, (int, float)) and score >= 7 else "badge-warning"
                with st.expander(f"{rev.get('title')} — Score: {score}/10"):
                    st.write(f"**Razonamiento:** {rev.get('reasoning')}")
                    st.write(f"**Riesgos:** {rev.get('risks')}")
                    st.write(f"**Recomendacion:** {rev.get('recommendation')}")

    # --- Tab 3: Data Engineering ---
    with tab_de:
        st.markdown("#### Ingenieria de Datos")

        code = result.get('cleaning_code', '# No code available')
        preview = result.get('cleaned_data_preview', 'No preview available')

        col_de_code, col_de_preview = st.columns(2)

        with col_de_code:
            st.markdown("**Script de Limpieza Generado**")
            st.code(code, language='python')

        with col_de_preview:
            st.markdown("**Vista Previa (Cleaned Data)**")
            if isinstance(preview, str) and preview.strip().startswith('{'):
                try:
                    from io import StringIO
                    st.dataframe(pd.read_json(StringIO(preview), orient='split'), use_container_width=True)
                except Exception as e:
                    st.write(f"Cannot render dataframe: {e}")
                    st.write(preview)
            else:
                st.write(preview)

    # --- Tab 4: ML Engineer ---
    with tab3:
        st.markdown("#### ML Engineer")

        col_code, col_out = st.columns(2)

        with col_code:
            st.markdown("**Codigo Generado**")
            ml_code = result.get('generated_code', '# No code')
            if ml_code.strip() == "# Generation Failed":
                ml_code = result.get('last_generated_code', ml_code)
            st.code(ml_code, language='python')

        with col_out:
            st.markdown("**Salida de Consola**")
            ml_output = result.get('execution_output', '')
            last_ok = result.get('last_successful_execution_output')
            if "BUDGET_EXCEEDED" in str(ml_output) and last_ok:
                ml_output = f"{ml_output}\n\n--- Last successful execution output ---\n{last_ok}"
            # Render in dark console style
            import html as html_mod
            escaped_output = html_mod.escape(str(ml_output))
            st.markdown(f'<div class="console-output">{escaped_output}</div>', unsafe_allow_html=True)

    # --- Tab 5: Executive Report ---
    with tab4:
        st.markdown("#### Informe Ejecutivo")
        st.markdown(result.get('final_report', 'No disponible'))

        # Plot gallery
        plots = glob.glob("static/plots/*.png")
        if plots:
            st.markdown("#### Visualizaciones")
            cols = st.columns(min(len(plots), 3))
            for i, plot_path in enumerate(plots):
                with cols[i % len(cols)]:
                    st.image(plot_path, caption=os.path.basename(plot_path), use_container_width=True)

        # Downloads section
        st.markdown("---")
        st.markdown("#### Descargas")
        dl_col1, dl_col2 = st.columns(2)

        # PDF Download
        if 'pdf_binary' not in st.session_state:
            pdf_path = result.get('pdf_path')
            if pdf_path and os.path.exists(pdf_path):
                try:
                    with open(pdf_path, "rb") as pdf_file:
                        st.session_state['pdf_binary'] = pdf_file.read()
                except Exception as e:
                    st.warning(f"Could not reload PDF: {e}")

        with dl_col1:
            if 'pdf_binary' in st.session_state:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M')
                st.download_button(
                    label="Descargar Reporte PDF",
                    data=st.session_state['pdf_binary'],
                    file_name=f"Reporte_Ejecutivo_{timestamp}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

        # ML Artifacts ZIP
        output_report = result.get("output_contract_report")
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
                    label="Descargar Entregables ML (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name=f"Entregables_ML_{timestamp}.zip",
                    mime="application/zip",
                    use_container_width=True
                )
            else:
                st.info("No se encontraron entregables ML en esta ejecucion.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("""
<div class="footer">
    <span>&copy; 2025 StrategyEngine AI &mdash; Powered by Multi-Agent AI</span>
</div>
""", unsafe_allow_html=True)
