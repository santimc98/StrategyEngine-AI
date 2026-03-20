import json
import os

from src.utils import run_status


def test_write_worker_input_persists_sandbox_config(tmp_path, monkeypatch):
    monkeypatch.setattr(run_status, "RUNS_DIR", str(tmp_path))

    run_status.write_worker_input(
        "abc12345",
        __file__,
        "Objetivo",
        sandbox_config={"provider": "gcp", "settings": {"endpoint": "https://sandbox.example.com"}},
    )

    input_path = os.path.join(str(tmp_path), "abc12345", "worker_input.json")
    with open(input_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload["business_objective"] == "Objetivo"
    assert payload["sandbox_config"] == {
        "provider": "remote",
        "settings": {"endpoint": "https://sandbox.example.com"},
    }
