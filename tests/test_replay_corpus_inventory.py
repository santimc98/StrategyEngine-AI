import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = REPO_ROOT / "tests" / "replay_corpus_inventory.json"


def test_replay_corpus_inventory_references_existing_run_artifacts():
    payload = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))

    assert payload.get("version") == 1
    tracks = payload.get("tracks")
    assert isinstance(tracks, list) and tracks

    allowed_status = {
        "replay_covered",
        "component_test_covered",
        "inventory_only",
        "fixed_by_prompt_refactor",
        "prompt_hardened_follow_up",
    }
    allowed_severity = {"S1", "S2", "A1", "A2", "AMB"}

    for track in tracks:
        assert isinstance(track, dict)
        assert isinstance(track.get("name"), str) and track["name"].strip()
        runs = track.get("runs")
        assert isinstance(runs, list) and runs

        for entry in runs:
            assert isinstance(entry, dict)
            assert entry.get("severity") in allowed_severity
            assert entry.get("status") in allowed_status
            assert isinstance(entry.get("run_id"), str) and entry["run_id"].strip()
            assert isinstance(entry.get("failure_class"), str) and entry["failure_class"].strip()
            required_files = entry.get("required_files")
            assert isinstance(required_files, list) and required_files
            for rel_path in required_files:
                path = REPO_ROOT / rel_path
                assert path.exists(), f"Missing replay inventory artifact: {path}"
