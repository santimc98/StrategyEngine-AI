from src.agents.ml_engineer import MLEngineerAgent


def _assert_contains_all(text: str, *needles: str) -> None:
    for needle in needles:
        assert needle in text


def _assert_contains_terms(text: str, *terms: str) -> None:
    lowered = text.lower()
    for term in terms:
        assert term.lower() in lowered


class FakeOpenAI:
    def __init__(self, api_key=None, base_url=None, timeout=None, default_headers=None):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.default_headers = default_headers


def test_incomplete_reprompt_context_has_contract_and_outputs(monkeypatch):
    monkeypatch.setenv("ML_ENGINEER_PROVIDER", "zai")
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-openrouter")
    monkeypatch.setattr("src.agents.ml_engineer.OpenAI", FakeOpenAI)

    agent = MLEngineerAgent()
    contract = {
        "canonical_columns": [f"col_{i}" for i in range(300)],
        "required_outputs": ["data/metrics.json", "data/alignment_check.json"],
        "artifact_requirements": {
            "required_files": [
                {"path": "data/scored_rows.csv", "allowed_name_patterns": ["^segment_.*"]}
            ]
        },
    }
    context = agent._build_incomplete_reprompt_context(
        execution_contract=contract,
        required_outputs=contract["required_outputs"],
        iteration_memory_block="ITERATION_MEMORY_CONTEXT: last attempt failed",
        iteration_memory=[],
        feedback_history=["REVIEWER FEEDBACK: add baseline metrics"],
        gate_context={"feedback": "QA TEAM FEEDBACK: fix outputs"},
    )
    _assert_contains_all(
        context,
        "EXECUTION_CONTRACT_CONTEXT",
        "REQUIRED OUTPUTS",
        "column_list_reference",
        "data/metrics.json",
        "data/alignment_check.json",
        "data/scored_rows.csv",
    )
    assert len(context) > 1000
    assert "..." not in context


def test_reprompt_context_includes_critical_errors(monkeypatch):
    monkeypatch.setenv("ML_ENGINEER_PROVIDER", "zai")
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-openrouter")
    monkeypatch.setattr("src.agents.ml_engineer.OpenAI", FakeOpenAI)

    agent = MLEngineerAgent()
    iteration_memory = [
        {
            "iteration_id": 1,
            "reviewer_reasons": ["Security violation"],
            "next_actions": ["Remove OS imports"],
        }
    ]
    gate_context = {
        "failed_gates": ["QA_CODE_AUDIT"],
        "feedback": "Price used as feature in optimization",
        "required_fixes": ["Remove price from MODEL_FEATURES"],
    }

    context = agent._build_incomplete_reprompt_context(
        execution_contract={},
        required_outputs=[],
        iteration_memory_block="",
        iteration_memory=iteration_memory,
        feedback_history=[],
        gate_context=gate_context,
    )

    _assert_contains_all(
        context,
        "QA_CODE_AUDIT",
        "Price used as feature in optimization",
        "Remove price from MODEL_FEATURES",
        "Security violation",
        "Remove OS imports",
    )
    _assert_contains_terms(
        context,
        "critical errors from previous attempts",
        "attempt 2",
        "attempt 1",
        "rejected",
        "error type",
        "root cause",
        "required fix",
    )
