import os

os.environ.setdefault("DEEPSEEK_API_KEY", "dummy")
os.environ.setdefault("GOOGLE_API_KEY", "dummy")
os.environ.setdefault("SANDBOX_PROVIDER", "local")
os.environ.setdefault("OPENROUTER_API_KEY", "dummy-openrouter")

from src.graph.graph import _get_execution_runtime_mode, _get_heavy_runner_config


def test_graph_execution_backend_prefers_state_config_over_env(monkeypatch):
    monkeypatch.setenv("RUN_EXECUTION_MODE", "local")
    monkeypatch.setenv("HEAVY_RUNNER_ENABLED", "0")
    monkeypatch.setenv("HEAVY_RUNNER_JOB", "env-job")
    monkeypatch.setenv("HEAVY_RUNNER_REGION", "env-region")
    monkeypatch.setenv("HEAVY_RUNNER_BUCKET", "env-bucket")
    monkeypatch.delenv("HEAVY_RUNNER_GCLOUD_BIN", raising=False)
    monkeypatch.delenv("HEAVY_RUNNER_GSUTIL_BIN", raising=False)

    state = {
        "sandbox_config": {
            "provider": "local",
            "settings": {
                "execution_backend": {
                    "mode": "cloudrun",
                    "cloudrun_enabled": True,
                    "job": "ui-job",
                    "region": "ui-region",
                    "bucket": "ui-bucket",
                    "project": "ui-project",
                }
            },
        }
    }

    assert _get_execution_runtime_mode(state) == "cloudrun"
    assert _get_heavy_runner_config(state) == {
        "mode": "cloudrun",
        "job": "ui-job",
        "region": "ui-region",
        "bucket": "ui-bucket",
        "project": "ui-project",
        "input_prefix": "inputs",
        "output_prefix": "outputs",
        "dataset_prefix": "datasets",
        "gcloud_bin": None,
        "gsutil_bin": None,
        "script_timeout_seconds": None,
    }


def test_graph_execution_backend_local_timeout_from_state(monkeypatch):
    monkeypatch.delenv("LOCAL_RUNNER_SCRIPT_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("HEAVY_RUNNER_SCRIPT_TIMEOUT_SECONDS", raising=False)

    state = {
        "sandbox_config": {
            "provider": "local",
            "settings": {
                "execution_backend": {
                    "mode": "local",
                    "local_script_timeout_seconds": 1800,
                }
            },
        }
    }

    assert _get_execution_runtime_mode(state) == "local"
    assert _get_heavy_runner_config(state)["script_timeout_seconds"] == 1800
