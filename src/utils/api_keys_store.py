"""Secure local storage for API keys.

Keys are stored in a JSON file encrypted with a machine-derived key using Fernet
symmetric encryption. If the cryptography library is unavailable, falls back to
base64 obfuscation (not truly secure, but prevents plaintext on disk).

The store file lives at ``data/api_keys.enc`` by default.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import platform
import tempfile
from typing import Any, Dict, List, Optional

from src.utils.paths import DATA_DIR as _DATA_DIR

_DEFAULT_STORE_PATH = os.path.join(_DATA_DIR, "api_keys.enc")

# ---------------------------------------------------------------------------
# Key derivation (machine-specific, deterministic)
# ---------------------------------------------------------------------------

def _derive_machine_key() -> bytes:
    """Derive a 32-byte key from stable machine identifiers."""
    parts = [
        platform.node(),
        os.getenv("USERNAME") or os.getenv("USER") or "default",
        platform.platform(),
        "strategyengine-ai-salt-2026",
    ]
    raw = ":".join(parts).encode("utf-8")
    digest = hashlib.sha256(raw).digest()
    return base64.urlsafe_b64encode(digest)


def _get_fernet():
    """Return a Fernet instance or None if cryptography is unavailable."""
    try:
        from cryptography.fernet import Fernet
        return Fernet(_derive_machine_key())
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Encrypt / Decrypt helpers
# ---------------------------------------------------------------------------

def _encrypt(payload: str) -> str:
    fernet = _get_fernet()
    if fernet:
        return fernet.encrypt(payload.encode("utf-8")).decode("ascii")
    return base64.b64encode(payload.encode("utf-8")).decode("ascii")


def _decrypt(token: str) -> str:
    fernet = _get_fernet()
    if fernet:
        try:
            return fernet.decrypt(token.encode("ascii")).decode("utf-8")
        except Exception:
            # Fallback: might be base64 from before cryptography was installed
            try:
                return base64.b64decode(token.encode("ascii")).decode("utf-8")
            except Exception:
                return "{}"
    try:
        return base64.b64decode(token.encode("ascii")).decode("utf-8")
    except Exception:
        return "{}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Registry of supported API keys with metadata for the UI
API_KEY_REGISTRY: List[Dict[str, str]] = [
    {
        "env_var": "GOOGLE_API_KEY",
        "label": "Google Gemini",
        "description": "Auditor de Datos, Traductor, Experto de Dominio",
        "placeholder": "AIza...",
        "required": True,
    },
    {
        "env_var": "OPENROUTER_API_KEY",
        "label": "OpenRouter",
        "description": "Estratega, Ing. Datos, Ing. ML, Analista, Planificador",
        "placeholder": "sk-or-...",
        "required": True,
    },
    {
        "env_var": "MIMO_API_KEY",
        "label": "MIMO / ZAI",
        "description": "Asesor de Resultados, Revisor de Limpieza (opcional)",
        "placeholder": "",
        "required": False,
    },
]


def load_keys(store_path: str = _DEFAULT_STORE_PATH) -> Dict[str, str]:
    """Load all stored API keys. Returns {env_var: value}."""
    if not os.path.isfile(store_path):
        return {}
    try:
        with open(store_path, "r", encoding="utf-8") as f:
            token = f.read().strip()
        if not token:
            return {}
        raw = _decrypt(token)
        data = json.loads(raw)
        return {k: v for k, v in data.items() if isinstance(v, str) and v.strip()}
    except Exception:
        return {}


def save_keys(keys: Dict[str, str], store_path: str = _DEFAULT_STORE_PATH) -> None:
    """Persist API keys to encrypted store."""
    cleaned = {k: v.strip() for k, v in keys.items() if isinstance(v, str) and v.strip()}
    payload = json.dumps(cleaned, ensure_ascii=False)
    encrypted = _encrypt(payload)
    os.makedirs(os.path.dirname(store_path), exist_ok=True)
    # Atomic write
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(store_path), suffix=".tmp"
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(encrypted)
        os.replace(tmp_path, store_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def apply_keys_to_env(keys: Optional[Dict[str, str]] = None,
                      store_path: str = _DEFAULT_STORE_PATH) -> Dict[str, str]:
    """Load keys from store (or use provided dict) and set them in os.environ.

    Returns the applied keys dict.
    """
    if keys is None:
        keys = load_keys(store_path)
    for env_var, value in keys.items():
        if value.strip():
            os.environ[env_var] = value.strip()
    return keys


def mask_key(value: str) -> str:
    """Return a masked version of an API key for display."""
    v = value.strip()
    if len(v) <= 8:
        return "\u2022" * len(v)
    return v[:4] + "\u2022" * (len(v) - 8) + v[-4:]


def test_key_connectivity(env_var: str, value: str) -> tuple[bool, str]:
    """Quick connectivity test for a given API key.

    Returns (success: bool, message: str).
    """
    value = value.strip()
    if not value:
        return False, "Clave vac\u00eda"

    if env_var == "GOOGLE_API_KEY":
        try:
            import google.generativeai as genai
            genai.configure(api_key=value)
            models = genai.list_models()
            model_list = [m.name for m in models]
            if model_list:
                return True, f"Conectado ({len(model_list)} modelos disponibles)"
            return True, "Conectado"
        except Exception as e:
            return False, f"Error: {str(e)[:80]}"

    elif env_var == "OPENROUTER_API_KEY":
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=value,
                base_url="https://openrouter.ai/api/v1",
            )
            # Simple models list call
            models = client.models.list()
            return True, "Conectado a OpenRouter"
        except Exception as e:
            return False, f"Error: {str(e)[:80]}"

    elif env_var == "MIMO_API_KEY":
        if len(value) > 10:
            return True, "Formato v\u00e1lido"
        return False, "Clave demasiado corta"

    return False, "Proveedor no reconocido"
