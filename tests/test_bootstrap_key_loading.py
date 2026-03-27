from pathlib import Path


def test_app_applies_stored_keys_before_graph_import():
    content = Path("app.py").read_text(encoding="utf-8")

    apply_idx = content.index("apply_keys_to_env()")
    graph_import_idx = content.index("from src.graph.graph import (")

    assert apply_idx < graph_import_idx


def test_background_worker_applies_stored_keys_before_graph_import():
    content = Path("src/utils/background_worker.py").read_text(encoding="utf-8")

    apply_idx = content.index("apply_keys_to_env()")
    graph_import_idx = content.index("from src.graph.graph import app_graph")

    assert apply_idx < graph_import_idx
