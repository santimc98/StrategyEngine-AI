import os

from src.utils.sandbox_config import (
    get_execution_backend_config,
    load_sandbox_config,
    merge_execution_backend_config,
    normalize_sandbox_config,
    normalize_execution_backend_config,
    save_sandbox_config,
)


def test_normalize_sandbox_config_defaults_to_local():
    config = normalize_sandbox_config(None)

    assert config["provider"] == "local"
    assert config["settings"] == {}


def test_normalize_sandbox_config_accepts_flat_shape():
    config = normalize_sandbox_config(
        {
            "provider": "gcp",
            "endpoint": "https://sandbox.example.com",
            "api_key": " secret ",
        }
    )

    assert config["provider"] == "remote"
    assert config["settings"] == {
        "endpoint": "https://sandbox.example.com",
        "api_key": "secret",
    }


def test_save_and_load_sandbox_config_roundtrip(tmp_path):
    store_path = os.path.join(tmp_path, "sandbox_config.enc")
    expected = {
        "provider": "remote",
        "settings": {
            "endpoint": "https://azure.example.com",
            "api_key": "abc123",
            "location": "westeurope",
        },
    }

    save_sandbox_config(expected, store_path=store_path)
    loaded = load_sandbox_config(store_path=store_path)

    assert loaded == expected


def test_normalize_sandbox_config_preserves_execution_backend_payload():
    config = normalize_sandbox_config(
        {
            "provider": "local",
            "settings": {
                "execution_backend": {
                    "mode": "cloudrun",
                    "cloudrun_enabled": "true",
                    "job": "corp-heavy",
                    "region": "europe-southwest1",
                    "bucket": "corp-bucket",
                    "script_timeout_seconds": "3600",
                }
            },
        }
    )

    assert config["settings"]["execution_backend"] == {
        "mode": "cloudrun",
        "cloudrun_enabled": True,
        "job": "corp-heavy",
        "region": "europe-southwest1",
        "bucket": "corp-bucket",
        "script_timeout_seconds": 3600,
    }


def test_get_execution_backend_config_merges_env_fallback(monkeypatch):
    monkeypatch.setenv("RUN_EXECUTION_MODE", "cloudrun")
    monkeypatch.setenv("HEAVY_RUNNER_ENABLED", "1")
    monkeypatch.setenv("HEAVY_RUNNER_JOB", "env-job")
    monkeypatch.setenv("HEAVY_RUNNER_REGION", "env-region")
    monkeypatch.setenv("HEAVY_RUNNER_BUCKET", "env-bucket")

    config = get_execution_backend_config({"provider": "local", "settings": {}})

    assert config["mode"] == "cloudrun"
    assert config["cloudrun_enabled"] is True
    assert config["job"] == "env-job"
    assert config["region"] == "env-region"
    assert config["bucket"] == "env-bucket"


def test_merge_execution_backend_config_preserves_provider_settings():
    merged = merge_execution_backend_config(
        {
            "provider": "remote",
            "settings": {
                "endpoint": "https://sandbox.example.com",
                "api_key": "secret",
            },
        },
        {
            "mode": "local",
            "local_script_timeout_seconds": "2400",
        },
    )

    assert merged["provider"] == "remote"
    assert merged["settings"]["endpoint"] == "https://sandbox.example.com"
    assert merged["settings"]["api_key"] == "secret"
    assert merged["settings"]["execution_backend"] == {
        "mode": "local",
        "local_script_timeout_seconds": 2400,
    }


def test_normalize_execution_backend_config_keeps_model_params_dict():
    normalized = normalize_execution_backend_config(
        {
            "mode": "cloudrun",
            "model_params": {"max_depth": 8, "n_estimators": 500},
        }
    )

    assert normalized["model_params"] == {"max_depth": 8, "n_estimators": 500}
