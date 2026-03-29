import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from src.api import integration_views
from src.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_get_api_keys_returns_masked_status(client, monkeypatch):
    monkeypatch.setattr(
        integration_views,
        "load_keys",
        lambda: {"OPENROUTER_API_KEY": "sk-or-1234567890"},
    )

    response = client.get("/config/api-keys")

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["configured_count"] == 1
    assert payload["items"][0]["configured"] is True
    assert "1234567890" not in payload["items"][0]["masked_value"]


def test_put_api_keys_saves_and_returns_status(client, monkeypatch):
    saved = {}
    monkeypatch.setattr(
        integration_views,
        "load_keys",
        lambda: {"EXISTING_KEY": "keep-me"},
    )
    monkeypatch.setattr(
        integration_views,
        "save_keys",
        lambda keys: saved.setdefault("keys", keys),
    )
    monkeypatch.setattr(
        integration_views,
        "apply_keys_to_env",
        lambda keys: keys,
    )
    monkeypatch.setattr(
        integration_views,
        "get_api_key_status_view",
        lambda: {"summary": {"configured_count": 1}, "items": []},
    )

    response = client.put(
        "/config/api-keys",
        json={"keys": {"OPENROUTER_API_KEY": "sk-or-abc"}},
    )

    assert response.status_code == 200
    assert saved["keys"]["OPENROUTER_API_KEY"] == "sk-or-abc"
    assert saved["keys"]["EXISTING_KEY"] == "keep-me"
    assert response.json()["summary"]["configured_count"] == 1


def test_post_api_key_test_uses_specific_value(client, monkeypatch):
    monkeypatch.setattr(
        integration_views,
        "test_key_connectivity",
        lambda env_var, value: (True, f"ok:{env_var}:{value}"),
    )

    response = client.post(
        "/config/api-keys/test",
        json={"env_var": "OPENROUTER_API_KEY", "value": "sk-or-test"},
    )

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert response.json()["message"] == "ok:OPENROUTER_API_KEY:sk-or-test"


class _FakeConnector:
    def authenticate(self, credentials):
        self.credentials = credentials
        return True

    def test_connection(self):
        return True

    def list_objects(self):
        return [{"name": "deals", "label": "Deals"}]

    def fetch_object_data(self, object_name, max_records=10000):
        import pandas as pd

        return pd.DataFrame(
            [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}][:max_records]
        )


def test_get_connectors_catalog_returns_items(client, monkeypatch):
    monkeypatch.setattr(
        integration_views,
        "_connector_registry",
        lambda: {
            "fake": {
                "label": "Fake CRM",
                "connector_class": _FakeConnector,
                "auth_modes": [{"id": "token", "label": "Token", "fields": []}],
            }
        },
    )

    response = client.get("/integrations/connectors")

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert payload["items"][0]["id"] == "fake"


def test_connector_test_endpoint_reports_success(client, monkeypatch):
    monkeypatch.setattr(
        integration_views,
        "_connector_registry",
        lambda: {
            "fake": {
                "label": "Fake CRM",
                "connector_class": _FakeConnector,
                "auth_modes": [],
            }
        },
    )

    response = client.post("/integrations/connectors/fake/test", json={"credentials": {"token": "x"}})

    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_connector_objects_endpoint_returns_catalog(client, monkeypatch):
    monkeypatch.setattr(
        integration_views,
        "_connector_registry",
        lambda: {
            "fake": {
                "label": "Fake CRM",
                "connector_class": _FakeConnector,
                "auth_modes": [],
            }
        },
    )

    response = client.post("/integrations/connectors/fake/objects", json={"credentials": {"token": "x"}})

    assert response.status_code == 200
    assert response.json()["count"] == 1
    assert response.json()["items"][0]["name"] == "deals"


def test_connector_fetch_endpoint_returns_preview_and_csv_path(client, monkeypatch, tmp_path):
    monkeypatch.setattr(
        integration_views,
        "_connector_registry",
        lambda: {
            "fake": {
                "label": "Fake CRM",
                "connector_class": _FakeConnector,
                "auth_modes": [],
            }
        },
    )
    monkeypatch.setattr(integration_views, "DATA_DIR", str(tmp_path))

    response = client.post(
        "/integrations/connectors/fake/fetch",
        json={
            "credentials": {"token": "x"},
            "object_name": "deals",
            "max_records": 2,
            "preview_rows": 1,
            "save_to_data": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["row_count"] == 2
    assert payload["column_count"] == 2
    assert len(payload["preview"]) == 1
    assert payload["csv_path"].endswith("crm_deals.csv")
