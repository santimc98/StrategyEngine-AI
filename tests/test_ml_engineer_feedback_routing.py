import os

from src.agents.ml_engineer import MLEngineerAgent


os.environ.setdefault("OPENROUTER_API_KEY", "dummy-openrouter")


def _agent() -> MLEngineerAgent:
    return MLEngineerAgent.__new__(MLEngineerAgent)


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


def test_repair_first_metric_round_keeps_optimization_context_active():
    agent = _agent()
    handoff = {
        "mode": "patch",
        "source": "result_evaluator_repair_first",
        "repair_policy": {"repair_first": True, "primary_focus": "runtime"},
        "retry_context": {"error_type": "shape_or_dtype", "repair_focus": "runtime"},
        "optimization_lane": {
            "active": True,
            "resume_after_repair": True,
            "repair_first": True,
            "repair_focus": "runtime",
            "source": "result_evaluator_repair_first",
            "active_technique": "out_of_fold_probability_calibration_per_horizon",
        },
        "optimization_context": {
            "metric_snapshot": {"primary_metric_name": "logloss", "baseline_metric": 0.38},
            "contract_lock": {"required_outputs": ["data/metrics.json"]},
        },
        "hypothesis_packet": {
            "action": "APPLY",
            "hypothesis": {"technique": "out_of_fold_probability_calibration_per_horizon"},
        },
        "editor_constraints": {"must_apply_hypothesis": False},
    }
    gate_context = {
        "runtime_error": {"type": "shape_or_dtype", "summary": "ValueError: inconsistent samples"},
        "failed_gates": ["runtime_failure"],
    }

    phase = agent._classify_editor_phase(
        gate_context=gate_context,
        handoff_payload=handoff,
        feedback_text="ValueError: inconsistent samples during calibration.",
    )

    assert phase == "runtime_repair"
    assert agent._is_metric_optimization_context(gate_context, handoff) is True


class _FakeOpenAI:
    def __init__(self, api_key=None, base_url=None, timeout=None, default_headers=None):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.default_headers = default_headers


def test_generate_code_preserves_llm_output_without_system_rewrite(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-openrouter")
    monkeypatch.setattr("src.agents.ml_engineer.OpenAI", _FakeOpenAI)

    llm_script = (
        "DATA_PATH = 'wrong.csv'\n"
        "SUBMISSION_DATA_PATH = 'data/submission.csv'\n"
        "if __name__ == '__main__':\n"
        "    print('ok')\n"
    )

    def _fake_call_chat_with_fallback(client, messages, models, call_kwargs=None, logger=None, context_tag=None):
        return {"dummy": True}, models[0]

    monkeypatch.setattr("src.agents.ml_engineer.call_chat_with_fallback", _fake_call_chat_with_fallback)
    monkeypatch.setattr(
        "src.agents.ml_engineer.extract_response_text",
        lambda response: llm_script,
    )

    agent = MLEngineerAgent()
    agent._check_script_completeness = lambda code, required_outputs: []

    code = agent.generate_code(
        strategy={"title": "Base Strategy", "analysis_type": "predictive", "required_columns": []},
        data_path="data/cleaned_data.csv",
        execution_contract={"required_outputs": ["data/submission.csv"], "canonical_columns": []},
        ml_view={"required_outputs": ["data/submission.csv"]},
    )

    assert code.strip() == llm_script.strip()
    assert "data/submission.csv" in code
    assert "wrong.csv" in code
    assert [entry.get("stage") for entry in agent.last_prompt_trace] == ["build_generation"]
