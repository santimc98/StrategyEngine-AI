from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from src.connectors.base import CRMAuthError, CRMConnectionError, CRMRateLimitError
from src.connectors.dynamics_connector import DynamicsConnector
from src.connectors.hubspot_connector import HubSpotConnector
from src.connectors.salesforce_connector import SalesforceConnector
from src.utils.api_keys_store import (
    API_KEY_REGISTRY,
    apply_keys_to_env,
    load_keys,
    mask_key,
    save_keys,
    test_key_connectivity,
)
from src.utils.paths import DATA_DIR


def _connector_registry() -> Dict[str, Dict[str, Any]]:
    return {
        "salesforce": {
            "label": "Salesforce",
            "connector_class": SalesforceConnector,
            "auth_modes": [
                {
                    "id": "token",
                    "label": "Token API",
                    "fields": [
                        {"key": "username", "label": "Username", "secret": False, "required": True},
                        {"key": "password", "label": "Password", "secret": True, "required": True},
                        {"key": "security_token", "label": "Security Token", "secret": True, "required": True},
                    ],
                },
                {
                    "id": "oauth",
                    "label": "OAuth (Access Token)",
                    "fields": [
                        {"key": "access_token", "label": "Access Token", "secret": True, "required": True},
                        {"key": "instance_url", "label": "Instance URL", "secret": False, "required": True},
                    ],
                },
            ],
        },
        "hubspot": {
            "label": "HubSpot",
            "connector_class": HubSpotConnector,
            "auth_modes": [
                {
                    "id": "token",
                    "label": "Private App Token / OAuth Token",
                    "fields": [
                        {"key": "access_token", "label": "Token", "secret": True, "required": True},
                    ],
                }
            ],
        },
        "dynamics365": {
            "label": "Dynamics 365",
            "connector_class": DynamicsConnector,
            "auth_modes": [
                {
                    "id": "client_credentials",
                    "label": "Client Credentials",
                    "fields": [
                        {"key": "crm_url", "label": "URL de la organización", "secret": False, "required": True},
                        {"key": "tenant_id", "label": "Tenant ID", "secret": False, "required": True},
                        {"key": "client_id", "label": "Client ID", "secret": False, "required": True},
                        {"key": "client_secret", "label": "Client Secret", "secret": True, "required": True},
                    ],
                }
            ],
        },
    }


def get_api_key_status_view() -> Dict[str, Any]:
    stored_keys = load_keys()
    items = []
    configured_count = 0
    required_count = 0
    for reg in API_KEY_REGISTRY:
        env_var = reg["env_var"]
        required = bool(reg.get("required"))
        value = str(stored_keys.get(env_var) or "").strip()
        configured = bool(value)
        configured_count += 1 if configured else 0
        required_count += 1 if required else 0
        items.append(
            {
                "env_var": env_var,
                "label": reg["label"],
                "description": reg["description"],
                "placeholder": reg.get("placeholder", ""),
                "required": required,
                "configured": configured,
                "masked_value": mask_key(value) if configured else "",
            }
        )
    return {
        "items": items,
        "summary": {
            "configured_count": configured_count,
            "required_count": required_count,
            "total_count": len(API_KEY_REGISTRY),
            "all_required_configured": configured_count >= required_count,
        },
    }


def save_api_keys_view(keys: Dict[str, Any]) -> Dict[str, Any]:
    normalized = {str(k): str(v or "").strip() for k, v in (keys or {}).items()}
    merged = load_keys()
    for key, value in normalized.items():
        if value:
            merged[key] = value
    save_keys(merged)
    apply_keys_to_env(merged)
    return get_api_key_status_view()


def test_api_key_view(env_var: str, value: str | None = None) -> Dict[str, Any]:
    candidate = str(value or "").strip()
    if not candidate:
        candidate = str(load_keys().get(env_var) or "").strip()
    if not candidate:
        return {"env_var": env_var, "ok": False, "message": "Clave vacía"}
    ok, message = test_key_connectivity(env_var, candidate)
    return {"env_var": env_var, "ok": bool(ok), "message": message}


def list_connector_specs() -> Dict[str, Any]:
    items = []
    for key, spec in _connector_registry().items():
        items.append(
            {
                "id": key,
                "label": spec["label"],
                "auth_modes": spec["auth_modes"],
            }
        )
    return {"items": items, "count": len(items)}


def _get_connector_spec(connector_id: str) -> Dict[str, Any]:
    registry = _connector_registry()
    spec = registry.get(str(connector_id or "").strip().lower())
    if not spec:
        raise ValueError(f"Unknown connector '{connector_id}'")
    return spec


def _build_connector(connector_id: str, credentials: Dict[str, Any]):
    spec = _get_connector_spec(connector_id)
    connector = spec["connector_class"]()
    connector.authenticate(credentials or {})
    return connector, spec


def _normalize_preview(df: pd.DataFrame, preview_rows: int = 25) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    preview_df = df.head(preview_rows).copy()
    preview_df = preview_df.where(pd.notna(preview_df), None)
    return preview_df.to_dict(orient="records")


def test_connector_connection(connector_id: str, credentials: Dict[str, Any]) -> Dict[str, Any]:
    try:
        connector, spec = _build_connector(connector_id, credentials)
        ok = bool(connector.test_connection())
        return {
            "connector": connector_id,
            "label": spec["label"],
            "ok": ok,
            "message": "Conexión OK" if ok else "Conexión no verificada",
        }
    except (CRMAuthError, CRMConnectionError, CRMRateLimitError, ValueError) as exc:
        return {
            "connector": connector_id,
            "label": _get_connector_spec(connector_id)["label"] if str(connector_id).lower() in _connector_registry() else connector_id,
            "ok": False,
            "message": str(exc),
        }


def list_connector_objects(connector_id: str, credentials: Dict[str, Any]) -> Dict[str, Any]:
    connector, spec = _build_connector(connector_id, credentials)
    objects = connector.list_objects()
    return {
        "connector": connector_id,
        "label": spec["label"],
        "items": objects,
        "count": len(objects),
    }


def fetch_connector_data(
    connector_id: str,
    credentials: Dict[str, Any],
    object_name: str,
    *,
    max_records: int = 10000,
    preview_rows: int = 25,
    save_to_data: bool = True,
) -> Dict[str, Any]:
    connector, spec = _build_connector(connector_id, credentials)
    df = connector.fetch_object_data(object_name, max_records=max_records)
    csv_path = None
    if save_to_data and not df.empty:
        os.makedirs(DATA_DIR, exist_ok=True)
        safe_name = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in object_name.lower())
        csv_path = str(Path(DATA_DIR) / f"crm_{safe_name}.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8")
    return {
        "connector": connector_id,
        "label": spec["label"],
        "object_name": object_name,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": [str(col) for col in df.columns.tolist()],
        "preview": _normalize_preview(df, preview_rows=preview_rows),
        "csv_path": csv_path,
    }
