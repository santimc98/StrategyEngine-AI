import ast

from src.agents.ml_engineer import MLEngineerAgent
from src.utils.ml_engineer_memory import append_memory, load_recent_memory


def test_ml_engineer_memory_jsonl_roundtrip(tmp_path):
    run_id = "run_memory_test"
    append_memory(
        run_id,
        {"iter": 1, "attempt": 1, "event": "runtime_error", "phase": "training"},
        base_dir=str(tmp_path),
    )
    append_memory(
        run_id,
        {"iter": 2, "attempt": 2, "event": "qa_reject", "phase": "persistence"},
        base_dir=str(tmp_path),
    )
    append_memory(
        run_id,
        {"iter": 3, "attempt": 3, "event": "attempt_success", "phase": "training"},
        base_dir=str(tmp_path),
    )

    memory_path = tmp_path / run_id / "work" / "memory" / "ml_engineer_memory.jsonl"
    assert memory_path.exists()

    rows = [line for line in memory_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 3

    recent = load_recent_memory(run_id, k=2, base_dir=str(tmp_path))
    assert [item.get("iter") for item in recent] == [2, 3]
    assert recent[-1].get("event") == "attempt_success"


def test_universal_prologue_and_json_default_injection():
    agent = MLEngineerAgent.__new__(MLEngineerAgent)
    raw_code = """
import json

def write_metrics():
    payload = {"metric": 0.9}
    with open("data/metrics.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
""".strip()

    patched = agent._apply_universal_script_guards(
        raw_code,
        csv_sep=";",
        csv_decimal=",",
        csv_encoding="latin-1",
    )

    assert "os.makedirs" in patched and "exist_ok=True" in patched
    assert "def json_default(obj):" in patched

    tree = ast.parse(patched)
    sep_value = None
    decimal_value = None
    encoding_value = None
    has_makedirs = False
    has_default_kw = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == "sep" and isinstance(node.value, ast.Constant):
                sep_value = node.value.value
            if node.targets[0].id == "decimal" and isinstance(node.value, ast.Constant):
                decimal_value = node.value.value
            if node.targets[0].id == "encoding" and isinstance(node.value, ast.Constant):
                encoding_value = node.value.value
        if not isinstance(node, ast.Call):
            continue
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "os"
            and node.func.attr == "makedirs"
        ):
            if len(node.args) >= 1 and isinstance(node.args[0], ast.Constant) and node.args[0].value == "data":
                has_makedirs = True
        if not isinstance(node.func, ast.Attribute):
            continue
        if not isinstance(node.func.value, ast.Name):
            continue
        if node.func.value.id != "json" or node.func.attr != "dump":
            continue
        for kw in node.keywords:
            if kw.arg == "default" and isinstance(kw.value, ast.Name) and kw.value.id == "_json_default":
                has_default_kw = True
                break
    assert has_makedirs is True
    assert sep_value == ";"
    assert decimal_value == ","
    assert encoding_value == "latin-1"
    assert has_default_kw is True
