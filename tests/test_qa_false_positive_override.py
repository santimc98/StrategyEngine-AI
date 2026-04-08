import types

from src.agents.qa_reviewer import (
    _apply_metric_gate_consistency_guard,
    _build_deterministic_metric_facts,
)
from src.graph import graph as graph_module
from src.graph.graph import run_qa_reviewer, qa_reviewer


def test_qa_override_variance_false_positive(monkeypatch):
    code = """
import pandas as pd
df = pd.DataFrame({"y":[1,2,3]})
y = df["y"]
if y.nunique() <= 1:
    raise ValueError("Target has no variance; cannot train meaningful model.")
print("Mapping Summary:", {"target": "y", "features": []})
"""

    def fake_review(c, s, b):
        return {
            "status": "REJECTED",
            "feedback": "Missing target variance guard",
            "failed_gates": ["TARGET_VARIANCE"],
            "required_fixes": ["Add variance guard"],
        }

    def fake_static(*args, **kwargs):
        return {"status": "PASS", "facts": {}}

    monkeypatch.setattr(qa_reviewer, "review_code", fake_review)
    monkeypatch.setattr(graph_module, "run_static_qa_checks", fake_static)
    state = {
        "generated_code": code,
        "selected_strategy": {},
        "business_objective": "",
        "feedback_history": [],
        "qa_reject_streak": 0,
    }
    result = run_qa_reviewer(state)
    assert result["review_verdict"] in ("APPROVED", "APPROVE_WITH_WARNINGS")
    assert "QA_LLM_NONBLOCKING_WARNING" in result["feedback_history"][-1]


def test_qa_fail_safe_preserves_gate_context(monkeypatch):
    def fake_review(c, s, b):
        return {
            "status": "REJECTED",
            "feedback": "Fail",
            "failed_gates": ["TARGET_VARIANCE"],
            "required_fixes": [],
        }

    def fake_static(*args, **kwargs):
        return {"status": "PASS", "facts": {}}

    monkeypatch.setattr(qa_reviewer, "review_code", fake_review)
    monkeypatch.setattr(graph_module, "run_static_qa_checks", fake_static)
    state = {
        "generated_code": "print('ok')",
        "selected_strategy": {},
        "business_objective": "",
        "feedback_history": [],
        "qa_reject_streak": 5,
    }
    result = run_qa_reviewer(state)
    assert result["review_verdict"] in ("APPROVED", "APPROVE_WITH_WARNINGS")
    assert result.get("last_gate_context", {}).get("source") == "qa_reviewer"


def test_build_deterministic_metric_facts_prefers_primary_metric_mean_over_std(tmp_path):
    metrics_path = tmp_path / "cv_metrics.json"
    metrics_path.write_text(
        """{
  "primary_metric_name": "top_decile_lift",
  "metrics_mean": {"top_decile_lift": 9.923888},
  "metrics_std": {"top_decile_lift": 0.0747},
  "higher_is_better": true
}""",
        encoding="utf-8",
    )

    facts = _build_deterministic_metric_facts(
        {
            "primary_metric": "top_decile_lift",
            "qa_required_outputs": [{"path": str(metrics_path), "intent": "cv_metrics"}],
        },
        [{"name": "metric_above_random_baseline", "severity": "HARD", "params": {"metric": "top_decile_lift"}}],
        [],
        [{"path": str(metrics_path), "intent": "cv_metrics"}],
    )

    assert facts["available"] is True
    assert facts["primary_metric_name"] == "top_decile_lift"
    assert facts["primary_metric_value"] == 9.923888
    assert facts["matched_key"] in {"primary_metric_value", "metrics_mean.top_decile_lift", "mean_top_decile_lift"}


def test_metric_gate_consistency_guard_removes_false_positive_metric_reject():
    qa_gates = [
        {
            "name": "metric_above_random_baseline",
            "severity": "HARD",
            "params": {"metric": "top_decile_lift"},
        }
    ]
    metric_facts = {
        "available": True,
        "primary_metric_name": "top_decile_lift",
        "primary_metric_value": 9.923888,
        "primary_metric_source": "artifacts/ml/cv_metrics.json",
        "higher_is_better": True,
        "_metrics_payload": {
            "primary_metric_name": "top_decile_lift",
            "metrics_mean": {"top_decile_lift": 9.923888},
            "metrics_std": {"top_decile_lift": 0.0747},
            "higher_is_better": True,
        },
    }

    adjusted, notes = _apply_metric_gate_consistency_guard(
        {
            "status": "REJECTED",
            "feedback": "Primary metric is below random baseline.",
            "failed_gates": ["metric_above_random_baseline"],
            "required_fixes": ["metric_above_random_baseline: improve top_decile_lift."],
        },
        qa_gates,
        metric_facts,
    )

    assert adjusted["status"] == "APPROVE_WITH_WARNINGS"
    assert adjusted["failed_gates"] == []
    assert notes
    assert "QA_METRIC_FACT_OVERRIDE" in notes[0]
