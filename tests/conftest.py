import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_GRAPH_SINGLETON_METHOD_SHADOWS = {
    "data_engineer": {"generate_cleaning_script"},
    "ml_engineer": {"generate_ml_plan", "generate_code"},
    "cleaning_reviewer": {"review_cleaning"},
    "qa_reviewer": {"review_code"},
    "reviewer": {"review_code"},
}


def _clear_graph_singleton_method_shadows() -> None:
    graph_module = sys.modules.get("src.graph.graph")
    if graph_module is None:
        return
    for singleton_name, method_names in _GRAPH_SINGLETON_METHOD_SHADOWS.items():
        singleton = getattr(graph_module, singleton_name, None)
        if singleton is None or not hasattr(singleton, "__dict__"):
            continue
        for method_name in method_names:
            shadow = singleton.__dict__.get(method_name)
            if callable(shadow) and hasattr(type(singleton), method_name):
                singleton.__dict__.pop(method_name, None)


@pytest.fixture(autouse=True)
def _restore_graph_singleton_method_descriptors():
    """
    Undo method shadowing on graph-level singleton instances after tests patch
    attributes like src.graph.graph.ml_engineer.generate_code directly.
    """
    _clear_graph_singleton_method_shadows()
    yield
    _clear_graph_singleton_method_shadows()
