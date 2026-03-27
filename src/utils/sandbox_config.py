"""Encrypted local storage for sandbox runtime configuration."""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, Optional

from src.utils.api_keys_store import _DATA_DIR, _decrypt, _encrypt
from src.utils.sandbox_provider import resolve_sandbox_provider_name

_DEFAULT_STORE_PATH = os.path.join(_DATA_DIR, "sandbox_config.enc")
_EXECUTION_BACKEND_BOOL_FIELDS = {
    "cloudrun_enabled",
    "data_engineer_cloudrun_enabled",
    "force_cloudrun",
    "force_data_engineer_cloudrun",
    "float32",
    "safe_mode",
}
_EXECUTION_BACKEND_INT_FIELDS = {
    "script_timeout_seconds",
    "local_script_timeout_seconds",
    "timeout_margin_seconds",
    "script_timeout_min_seconds",
    "script_timeout_max_seconds",
    "local_script_timeout_min_seconds",
    "local_script_timeout_max_seconds",
    "default_cpu",
    "cpu_hint",
}
_EXECUTION_BACKEND_FLOAT_FIELDS = {
    "timeout_margin_multiplier",
}
_EXECUTION_BACKEND_TEXT_FIELDS = {
    "job",
    "region",
    "bucket",
    "project",
    "input_prefix",
    "output_prefix",
    "dataset_prefix",
    "gcloud_bin",
    "gsutil_bin",
    "default_memory_gb",
    "memory_gb_hint",
    "model_type",
}


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    if not text:
        return None
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    coerced = _coerce_bool(raw)
    return default if coerced is None else coerced


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _clean_positive_int(value: Any) -> Optional[int]:
    try:
        parsed = int(str(value).strip())
    except Exception:
        return None
    return parsed if parsed > 0 else None


def _clean_positive_float(value: Any) -> Optional[float]:
    try:
        parsed = float(str(value).strip())
    except Exception:
        return None
    return parsed if parsed > 0 else None


def normalize_execution_backend_config(raw: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Normalize the persisted execution backend payload stored under sandbox settings."""

    payload = raw if isinstance(raw, dict) else {}
    nested = payload.get("execution_backend")
    if isinstance(nested, dict):
        payload = nested

    normalized: Dict[str, Any] = {}

    mode = str(
        payload.get("mode")
        or payload.get("runtime_mode")
        or ""
    ).strip().lower()
    if mode in {"local", "cloudrun"}:
        normalized["mode"] = mode

    for key in _EXECUTION_BACKEND_BOOL_FIELDS:
        if key not in payload or payload.get(key) is None:
            continue
        coerced = _coerce_bool(payload.get(key))
        if coerced is not None:
            normalized[key] = coerced

    for key in _EXECUTION_BACKEND_INT_FIELDS:
        if key not in payload or payload.get(key) is None:
            continue
        parsed = _clean_positive_int(payload.get(key))
        if parsed is not None:
            normalized[key] = parsed

    for key in _EXECUTION_BACKEND_FLOAT_FIELDS:
        if key not in payload or payload.get(key) is None:
            continue
        parsed = _clean_positive_float(payload.get(key))
        if parsed is not None:
            normalized[key] = parsed

    for key in _EXECUTION_BACKEND_TEXT_FIELDS:
        if key not in payload or payload.get(key) is None:
            continue
        text = _clean_text(payload.get(key))
        if text:
            normalized[key] = text

    model_params = payload.get("model_params")
    if isinstance(model_params, dict):
        cleaned_params = {
            str(k).strip(): v
            for k, v in model_params.items()
            if str(k or "").strip() and v is not None
        }
        if cleaned_params:
            normalized["model_params"] = cleaned_params
    elif model_params is not None:
        text = _clean_text(model_params)
        if text:
            normalized["model_params"] = text

    return normalized


def get_execution_backend_config(
    config: Optional[Dict[str, Any]] = None,
    *,
    include_env_fallback: bool = True,
) -> Dict[str, Any]:
    """Return effective execution backend config, optionally merging env fallbacks."""

    sandbox_config = normalize_sandbox_config(config or {})
    settings = sandbox_config.get("settings")
    raw_backend = settings.get("execution_backend") if isinstance(settings, dict) else {}
    normalized = normalize_execution_backend_config(raw_backend if isinstance(raw_backend, dict) else {})
    if not include_env_fallback:
        return normalized

    effective = dict(normalized)

    if "mode" not in effective:
        mode = str(
            os.getenv("RUN_EXECUTION_MODE")
            or os.getenv("EXECUTION_RUNTIME_MODE")
            or os.getenv("CODE_EXECUTION_RUNTIME")
            or "cloudrun"
        ).strip().lower()
        effective["mode"] = mode if mode in {"local", "cloudrun"} else "cloudrun"

    env_bool_defaults = {
        "cloudrun_enabled": _env_flag("HEAVY_RUNNER_ENABLED", False),
        "data_engineer_cloudrun_enabled": _env_flag("HEAVY_RUNNER_DE_ENABLED", True),
        "force_cloudrun": _env_flag("HEAVY_RUNNER_FORCE", False),
        "force_data_engineer_cloudrun": _env_flag("HEAVY_RUNNER_FORCE_DE", False),
        "float32": _env_flag("HEAVY_RUNNER_FLOAT32", False),
        "safe_mode": _env_flag("HEAVY_RUNNER_SAFE_MODE", False),
    }
    for key, value in env_bool_defaults.items():
        effective.setdefault(key, value)

    env_text_defaults = {
        "job": os.getenv("HEAVY_RUNNER_JOB"),
        "region": os.getenv("HEAVY_RUNNER_REGION"),
        "bucket": os.getenv("HEAVY_RUNNER_BUCKET"),
        "project": os.getenv("HEAVY_RUNNER_PROJECT"),
        "input_prefix": os.getenv("HEAVY_RUNNER_INPUT_PREFIX"),
        "output_prefix": os.getenv("HEAVY_RUNNER_OUTPUT_PREFIX"),
        "dataset_prefix": os.getenv("HEAVY_RUNNER_DATASET_PREFIX"),
        "gcloud_bin": os.getenv("HEAVY_RUNNER_GCLOUD_BIN"),
        "gsutil_bin": os.getenv("HEAVY_RUNNER_GSUTIL_BIN"),
        "default_memory_gb": os.getenv("HEAVY_RUNNER_DEFAULT_MEMORY_GB") or os.getenv("HEAVY_RUNNER_DEFAULT_MEMORY"),
        "memory_gb_hint": (
            os.getenv("HEAVY_RUNNER_MEMORY_GB_HINT")
            or os.getenv("HEAVY_RUNNER_MEMORY_GB")
            or os.getenv("HEAVY_RUNNER_MEMORY")
            or os.getenv("CLOUDRUN_JOB_MEMORY")
            or os.getenv("CLOUD_RUN_MEMORY")
        ),
        "model_type": os.getenv("HEAVY_RUNNER_MODEL_TYPE"),
    }
    for key, value in env_text_defaults.items():
        text = _clean_text(value)
        if text and key not in effective:
            effective[key] = text

    env_int_defaults = {
        "script_timeout_seconds": (
            os.getenv("HEAVY_RUNNER_SCRIPT_TIMEOUT_SECONDS")
            if effective.get("mode") != "local"
            else (os.getenv("LOCAL_RUNNER_SCRIPT_TIMEOUT_SECONDS") or os.getenv("HEAVY_RUNNER_SCRIPT_TIMEOUT_SECONDS"))
        ),
        "local_script_timeout_seconds": os.getenv("LOCAL_RUNNER_SCRIPT_TIMEOUT_SECONDS"),
        "timeout_margin_seconds": os.getenv("HEAVY_RUNNER_TIMEOUT_MARGIN_SECONDS"),
        "script_timeout_min_seconds": os.getenv("HEAVY_RUNNER_SCRIPT_TIMEOUT_MIN_SECONDS"),
        "script_timeout_max_seconds": os.getenv("HEAVY_RUNNER_SCRIPT_TIMEOUT_MAX_SECONDS"),
        "local_script_timeout_min_seconds": os.getenv("LOCAL_RUNNER_SCRIPT_TIMEOUT_MIN_SECONDS"),
        "local_script_timeout_max_seconds": os.getenv("LOCAL_RUNNER_SCRIPT_TIMEOUT_MAX_SECONDS"),
        "default_cpu": os.getenv("HEAVY_RUNNER_DEFAULT_CPU"),
        "cpu_hint": (
            os.getenv("HEAVY_RUNNER_CPU_HINT")
            or os.getenv("HEAVY_RUNNER_CPU")
            or os.getenv("CLOUDRUN_JOB_CPU")
            or os.getenv("CLOUD_RUN_CPU")
        ),
    }
    for key, value in env_int_defaults.items():
        if key in effective:
            continue
        parsed = _clean_positive_int(value)
        if parsed is not None:
            effective[key] = parsed

    if "timeout_margin_multiplier" not in effective:
        parsed = _clean_positive_float(os.getenv("HEAVY_RUNNER_TIMEOUT_MARGIN_MULTIPLIER"))
        if parsed is not None:
            effective["timeout_margin_multiplier"] = parsed

    if "model_params" not in effective:
        model_params = _clean_text(os.getenv("HEAVY_RUNNER_MODEL_PARAMS"))
        if model_params:
            effective["model_params"] = model_params

    return effective


def merge_execution_backend_config(
    config: Optional[Dict[str, Any]],
    execution_backend: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Return a sandbox config payload preserving provider settings and replacing backend settings."""

    normalized = normalize_sandbox_config(config or {})
    settings = dict(normalized.get("settings") or {})
    backend_payload = normalize_execution_backend_config(execution_backend or {})
    if backend_payload:
        settings["execution_backend"] = backend_payload
    else:
        settings.pop("execution_backend", None)
    normalized["settings"] = settings
    return normalized


def normalize_sandbox_config(raw: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return a stable provider/settings payload."""

    payload = raw if isinstance(raw, dict) else {}
    provider = str(
        payload.get("provider")
        or payload.get("sandbox_provider")
        or os.getenv("SANDBOX_PROVIDER", "local")
        or "local"
    ).strip().lower() or "local"
    provider = resolve_sandbox_provider_name(provider)

    settings_raw = payload.get("settings")
    if not isinstance(settings_raw, dict):
        settings_raw = payload.get("config")
    if not isinstance(settings_raw, dict):
        settings_raw = {
            key: value
            for key, value in payload.items()
            if key not in {"provider", "sandbox_provider", "settings", "config"}
        }

    settings: Dict[str, Any] = {}
    for key, value in settings_raw.items():
        clean_key = str(key or "").strip()
        if not clean_key or value is None:
            continue
        if clean_key == "execution_backend":
            backend_payload = normalize_execution_backend_config(value if isinstance(value, dict) else {})
            if backend_payload:
                settings[clean_key] = backend_payload
            continue
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                settings[clean_key] = stripped
            continue
        if isinstance(value, (bool, int, float)):
            settings[clean_key] = value
            continue
        if isinstance(value, dict):
            nested = {str(nk).strip(): nv for nk, nv in value.items() if str(nk or "").strip() and nv is not None}
            if nested:
                settings[clean_key] = nested
            continue
        if isinstance(value, list):
            cleaned_list = [item for item in value if item is not None and str(item).strip()]
            if cleaned_list:
                settings[clean_key] = cleaned_list
            continue
        as_text = str(value).strip()
        if as_text:
            settings[clean_key] = as_text

    return {
        "provider": provider,
        "settings": settings,
    }


def load_sandbox_config(store_path: str = _DEFAULT_STORE_PATH) -> Dict[str, Any]:
    """Load persisted sandbox configuration or return the local default."""

    if not os.path.isfile(store_path):
        return normalize_sandbox_config({})
    try:
        with open(store_path, "r", encoding="utf-8") as handle:
            token = handle.read().strip()
        if not token:
            return normalize_sandbox_config({})
        raw = _decrypt(token)
        data = json.loads(raw)
    except Exception:
        return normalize_sandbox_config({})
    return normalize_sandbox_config(data)


def save_sandbox_config(config: Optional[Dict[str, Any]], store_path: str = _DEFAULT_STORE_PATH) -> Dict[str, Any]:
    """Persist sandbox configuration to the encrypted local store."""

    normalized = normalize_sandbox_config(config or {})
    payload = json.dumps(normalized, ensure_ascii=False)
    encrypted = _encrypt(payload)
    os.makedirs(os.path.dirname(store_path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(store_path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(encrypted)
        os.replace(tmp_path, store_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return normalized


def mask_sandbox_secret(value: str) -> str:
    """Return a masked representation of a secret sandbox field."""

    secret = str(value or "").strip()
    if not secret:
        return ""
    if len(secret) <= 8:
        return "*" * len(secret)
    return secret[:4] + ("*" * (len(secret) - 8)) + secret[-4:]
