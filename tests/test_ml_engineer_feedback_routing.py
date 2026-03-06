import os

from src.agents.ml_engineer import MLEngineerAgent


os.environ.setdefault("OPENROUTER_API_KEY", "dummy-openrouter")


def _agent() -> MLEngineerAgent:
    return MLEngineerAgent.__new__(MLEngineerAgent)


def test_synthetic_fallback_detector_ignores_local_make_helpers():
    agent = _agent()
    code = """
def make_plots():
    return None

def main():
    make_plots()
"""

    assert agent._detect_forbidden_input_fallback(code, "data/cleaned_data.csv") == []


def test_synthetic_fallback_detector_flags_sklearn_make_generators():
    agent = _agent()
    code = """
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=100, n_features=4, random_state=42)
"""

    reasons = agent._detect_forbidden_input_fallback(code, "data/cleaned_data.csv")

    assert "forbidden_sklearn_make_call:make_classification" in reasons


def test_editor_phase_prioritizes_training_when_strategy_and_persistence_signals_coexist():
    agent = _agent()
    handoff = {
        "quality_focus": {
            "failed_gates": ["strategy_followed"],
            "required_fixes": [
                "Replace Logistic Regression with a boosting ensemble.",
                "Resolve scored_rows_rowcount_mismatch in submission outputs.",
            ],
        }
    }

    phase = agent._classify_editor_phase(
        gate_context={},
        handoff_payload=handoff,
        feedback_text=(
            "Execution diagnostics reported scored_rows_rowcount_mismatch, but the "
            "reviewer requires replacing Logistic Regression with boosting and stacking."
        ),
    )

    assert phase == "training"


def test_editor_phase_stays_persistence_for_output_only_repairs():
    agent = _agent()
    handoff = {
        "quality_focus": {
            "failed_gates": ["output_contract"],
            "required_fixes": ["Write missing artifact data/metrics.json at the exact path."],
        }
    }

    phase = agent._classify_editor_phase(
        gate_context={},
        handoff_payload=handoff,
        feedback_text="Missing artifact data/metrics.json. Fix persistence and to_csv output paths.",
    )

    assert phase == "persistence"


def test_guardrail_repair_context_preserves_active_patch_intent():
    agent = _agent()
    handoff = {
        "patch_objectives": [
            "Replace Logistic Regression with a boosting ensemble.",
            "Implement stacking.",
        ],
        "must_preserve": ["Preserve artifact generation for data/metrics.json"],
        "quality_focus": {
            "required_fixes": ["Implement K-fold target encoding."],
        },
        "feedback": {
            "reviewer": "Replace Logistic Regression with boosting and stacking.",
            "qa": "Keep leakage guards intact.",
        },
    }
    gate_context = {
        "feedback": "Reviewer requires boosting plus stacking.",
        "required_fixes": ["Implement K-fold target encoding."],
    }

    context = agent._build_guardrail_repair_context(
        handoff_payload=handoff,
        gate_context=gate_context,
        feedback_history=["Previous run still used Logistic Regression."],
    )

    assert "ACTIVE_PATCH_OBJECTIVES" in context
    assert "Replace Logistic Regression with a boosting ensemble." in context
    assert "MUST_PRESERVE" in context
    assert "Preserve artifact generation for data/metrics.json" in context
    assert "ACTIVE_REVIEW_FEEDBACK" in context


def test_editor_phase_prioritizes_runtime_repair_over_metric_optimization():
    agent = _agent()
    handoff = {
        "mode": "optimize",
        "repair_policy": {"repair_first": True, "primary_focus": "runtime"},
        "contract_focus": {"missing_outputs": ["data/metrics.json"]},
        "quality_focus": {"failed_gates": ["runtime_failure"], "required_fixes": ["Reduce runtime cost."]},
        "editor_constraints": {"must_apply_hypothesis": False},
    }
    gate_context = {
        "runtime_error": {"type": "timeout", "summary": "TIMEOUT: Script exceeded 7200s limit"},
        "failed_gates": ["runtime_failure"],
    }

    phase = agent._classify_editor_phase(
        gate_context=gate_context,
        handoff_payload=handoff,
        feedback_text="TIMEOUT: Script exceeded 7200s limit while generating data/metrics.json",
    )

    assert phase == "runtime_repair"
    assert agent._is_metric_optimization_context(gate_context, handoff) is False
