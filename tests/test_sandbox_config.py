import os

from src.utils.sandbox_config import (
    load_sandbox_config,
    normalize_sandbox_config,
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
