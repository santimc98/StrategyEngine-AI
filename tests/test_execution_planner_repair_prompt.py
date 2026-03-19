from unittest.mock import MagicMock

from src.agents.execution_planner import (
    ExecutionPlannerAgent,
    MINIMAL_CONTRACT_COMPILER_PROMPT,
    _compress_text_preserve_ends,
)


def test_compress_text_preserve_ends_keeps_tail():
    head = "HEAD"
    tail = "Tail KPI Accuracy"
    middle = "x" * 200
    text = head + middle + tail
    compressed = _compress_text_preserve_ends(text, max_chars=60, head=20, tail=20)
    assert "HEAD" in compressed
    assert "Accuracy" in compressed
    assert "..." in compressed
    assert len(compressed) <= 60 + len("\n...\n")


def test_minimal_contract_prompt_defines_column_roles_semantics():
    prompt = MINIMAL_CONTRACT_COMPILER_PROMPT
    assert "Role definitions (ML execution context)" in prompt
    assert "outcome MUST contain only the target" in prompt


def test_minimal_contract_prompt_declares_phased_compilation():
    prompt = MINIMAL_CONTRACT_COMPILER_PROMPT
    assert "Phased contract compilation protocol" in prompt
    assert "Phase 1 FACTS_EXTRACTOR" in prompt
    assert "Phase 4 VALIDATOR_REPAIR" in prompt


def test_minimal_contract_prompt_enforces_semantic_closure():
    prompt = MINIMAL_CONTRACT_COMPILER_PROMPT
    assert "Phase 2B SEMANTIC_CLOSURE" in prompt
    assert "Do not leave critical meaning stranded" in prompt
    assert "Semantic closure rules:" in prompt
    assert "must be reflected in both" in prompt


def test_minimal_contract_prompt_marks_ml_sections_as_required_not_recommended():
    prompt = MINIMAL_CONTRACT_COMPILER_PROMPT
    assert "evaluation_spec: REQUIRED finalized section for ML scopes" in prompt
    assert "validation_requirements: REQUIRED finalized section for ML scopes" in prompt
    assert "column_dtype_targets MUST include anchor columns for ML scopes" in prompt


def test_execution_planner_main_prompt_no_longer_includes_deterministic_scaffold(monkeypatch):
    monkeypatch.delenv("EXECUTION_PLANNER_SECTION_FIRST", raising=False)
    monkeypatch.delenv("EXECUTION_PLANNER_PROGRESSIVE_MODE", raising=False)

    planner = ExecutionPlannerAgent(api_key="mock_key")
    response = MagicMock()
    response.text = (
        '{"scope":"full_pipeline","strategy_title":"Test","business_objective":"Predict",'
        '"canonical_columns":["id","feature","target"],'
        '"column_roles":{"pre_decision":["feature"],"decision":[],"outcome":["target"],'
        '"post_decision_audit_only":[],"identifiers":["id"],"unknown":[]},'
        '"artifact_requirements":{"clean_dataset":{"output_path":"data/cleaned_data.csv",'
        '"output_manifest_path":"data/cleaning_manifest.json","required_columns":["id","feature","target"]},'
        '"required_files":[{"path":"data/submission.csv"}]},'
        '"required_outputs":["data/submission.csv"],'
        '"objective_analysis":{"problem_type":"classification"},'
        '"evaluation_spec":{"objective_type":"classification"},'
        '"validation_requirements":{"primary_metric":"roc_auc"},'
        '"qa_gates":[{"name":"metrics_present","severity":"HARD","params":{}}],'
        '"reviewer_gates":[{"name":"strategy_followed","severity":"HARD","params":{}}],'
        '"data_engineer_runbook":"clean",'
        '"ml_engineer_runbook":"train",'
        '"iteration_policy":{"max_retries":2},'
        '"optimization_policy":{"enabled":true,"max_rounds":1},'
        '"column_dtype_targets":{"id":{"target_dtype":"int64"}}}'
    )
    response.candidates = []
    response.usage_metadata = None
    planner.client = MagicMock()
    planner.client.generate_content.return_value = response

    planner.generate_contract(
        strategy={"required_columns": ["id", "feature", "target"], "title": "Prompt test"},
        business_objective="Predict target.",
        column_inventory=["id", "feature", "target"],
    )

    assert "deterministic_contract_scaffold" not in (planner.last_prompt or "")
