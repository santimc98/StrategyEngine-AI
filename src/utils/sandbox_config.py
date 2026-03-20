"""Encrypted local storage for sandbox runtime configuration."""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, Optional

from src.utils.api_keys_store import _DATA_DIR, _decrypt, _encrypt
from src.utils.sandbox_provider import resolve_sandbox_provider_name

_DEFAULT_STORE_PATH = os.path.join(_DATA_DIR, "sandbox_config.enc")


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
