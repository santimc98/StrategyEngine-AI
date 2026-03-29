import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from src.api import config_views
from src.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_get_model_settings_returns_effective_runtime_view(client, monkeypatch):
    monkeypatch.setattr(
        config_views,
        "_load_runtime_model_hooks",
        lambda: (
            lambda: {"steward": "openai/gpt-5.4", "translator": "google/gemini-3-flash-preview"},
            lambda payload: payload,
            None,
        ),
    )
    monkeypatch.setattr(
        config_views,
        "load_agent_model_overrides",
        lambda: {"translator": "anthropic/claude-opus-4.6"},
    )

    response = client.get("/config/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["runtime_available"] is True
    assert payload["base_models"]["steward"] == "openai/gpt-5.4"
    assert payload["effective_models"]["translator"] == "anthropic/claude-opus-4.6"
    assert any(agent["key"] == "ml_engineer" for agent in payload["agents"])


def test_put_model_settings_applies_and_persists_models(client, monkeypatch):
    saved = {}

    monkeypatch.setattr(
        config_views,
        "_load_runtime_model_hooks",
        lambda: (
            lambda: {"steward": "openai/gpt-5.4"},
            lambda payload: payload,
            None,
        ),
    )
    monkeypatch.setattr(
        config_views,
        "save_agent_model_overrides",
        lambda payload: saved.setdefault("payload", payload) or payload,
    )

    response = client.put(
        "/config/models",
        json={"models": {"steward": "openai/gpt-5.4-mini", "translator": "anthropic/claude-opus-4.6"}},
    )

    assert response.status_code == 200
    assert saved["payload"]["steward"] == "openai/gpt-5.4-mini"
    assert response.json()["effective_models"]["translator"] == "anthropic/claude-opus-4.6"


def test_reset_model_settings_clears_overrides_when_runtime_unavailable(client, monkeypatch, tmp_path):
    overrides_path = tmp_path / "agent_model_overrides.json"
    overrides_path.write_text(json.dumps({"steward": "x"}), encoding="utf-8")

    monkeypatch.setattr(config_views, "MODEL_OVERRIDES_PATH", overrides_path)
    monkeypatch.setattr(config_views, "_load_runtime_model_hooks", lambda: (None, None, "bootstrap failed"))

    response = client.post("/config/models/reset")

    assert response.status_code == 200
    assert response.json()["runtime_available"] is False
    assert response.json()["saved_models"] == {}
    assert not overrides_path.exists()


def test_get_sandbox_settings_returns_provider_and_backend_status(client, monkeypatch):
    monkeypatch.setattr(
        config_views,
        "load_sandbox_config",
        lambda: {
            "provider": "remote",
            "settings": {
                "endpoint": "https://sandbox.example.com",
                "execution_backend": {
                    "mode": "cloudrun",
                    "cloudrun_enabled": True,
                    "job": "corp-heavy",
                    "region": "europe-southwest1",
                    "bucket": "corp-bucket",
                },
            },
        },
    )
    monkeypatch.setattr(
        config_views,
        "list_sandbox_providers",
        lambda: [
            type(
                "Spec",
                (),
                {
                    "name": "local",
                    "label": "Local",
                    "description": "Local",
                    "implemented": True,
                    "config_fields": (),
                },
            )(),
            type(
                "Spec",
                (),
                {
                    "name": "remote",
                    "label": "Gateway remoto",
                    "description": "Remote",
                    "implemented": True,
                    "config_fields": (),
                },
            )(),
        ],
    )
    monkeypatch.setattr(config_views, "is_sandbox_provider_available", lambda provider: True)
    monkeypatch.setattr(
        config_views,
        "get_sandbox_provider_spec",
        lambda provider: type(
            "Spec",
            (),
            {
                "name": "remote",
                "label": "Gateway remoto",
                "description": "Remote",
                "implemented": True,
                "config_fields": (),
            },
        )(),
    )
    monkeypatch.setattr(config_views, "test_sandbox_provider_connectivity", lambda provider, settings: (True, "ok"))

    response = client.get("/config/sandbox")

    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == "remote"
    assert payload["provider_status"]["severity"] == "ok"
    assert payload["execution_backend_status"]["detail"] == "Configurado"
    assert payload["provider_connectivity"]["ok"] is True


def test_put_sandbox_settings_normalizes_and_returns_view(client, monkeypatch):
    saved = {}

    monkeypatch.setattr(
        config_views,
        "save_sandbox_config",
        lambda config: saved.setdefault("config", config) or config,
    )
    monkeypatch.setattr(
        config_views,
        "get_sandbox_settings_view",
        lambda: {"provider": "local", "config": {"provider": "local", "settings": {}}},
    )

    response = client.put(
        "/config/sandbox",
        json={
            "config": {
                "provider": "local",
                "settings": {"execution_backend": {"mode": "local"}},
            }
        },
    )

    assert response.status_code == 200
    assert saved["config"]["provider"] == "local"
    assert response.json()["provider"] == "local"
