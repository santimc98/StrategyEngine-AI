"""
Tests for execution planner prompt properties.

These verify semantic invariants of the prompts — not exact phrases,
but structural properties that must hold regardless of prompt rewrites.
"""

import json

from unittest.mock import MagicMock

from src.agents.execution_planner import (
    ExecutionPlannerAgent,
    SEMANTIC_EXECUTION_PLANNER_PROMPT,
    MINIMAL_CONTRACT_COMPILER_PROMPT,
    _compress_text_preserve_ends,
)


# ── Utility function tests (stable) ──────────────────────────────────────────


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


# ── Task A (Semantic Planner) prompt invariants ───────────────────────────────


class TestSemanticPlannerPromptProperties:
    """Properties the Task A prompt must always have."""

    def test_declares_mission(self):
        assert "MISSION" in SEMANTIC_EXECUTION_PLANNER_PROMPT

    def test_produces_semantic_core_json(self):
        prompt = SEMANTIC_EXECUTION_PLANNER_PROMPT.lower()
        assert "semantic_core" in prompt or "semantic core" in prompt

    def test_mentions_canonical_columns(self):
        assert "canonical_columns" in SEMANTIC_EXECUTION_PLANNER_PROMPT

    def test_mentions_column_roles(self):
        assert "column_roles" in SEMANTIC_EXECUTION_PLANNER_PROMPT

    def test_mentions_workstreams_or_scope(self):
        prompt = SEMANTIC_EXECUTION_PLANNER_PROMPT.lower()
        assert "workstream" in prompt or "scope" in prompt

    def test_mentions_gates(self):
        prompt = SEMANTIC_EXECUTION_PLANNER_PROMPT
        assert "cleaning_gates" in prompt or "gates" in prompt.lower()

    def test_mentions_runbook(self):
        prompt = SEMANTIC_EXECUTION_PLANNER_PROMPT
        assert "runbook" in prompt.lower()

    def test_mentions_model_features(self):
        assert "model_features" in SEMANTIC_EXECUTION_PLANNER_PROMPT

    def test_has_output_section(self):
        assert "OUTPUT" in SEMANTIC_EXECUTION_PLANNER_PROMPT

    def test_warns_against_inventing_columns(self):
        prompt = SEMANTIC_EXECUTION_PLANNER_PROMPT.lower()
        assert "invent" in prompt or "never add columns" in prompt or "only from" in prompt


# ── Task B (Contract Compiler) prompt invariants ─────────────────────────────


class TestContractCompilerPromptProperties:
    """Properties the Task B prompt must always have."""

    def test_declares_mission(self):
        assert "MISSION" in MINIMAL_CONTRACT_COMPILER_PROMPT

    def test_references_semantic_core_as_authority(self):
        prompt = MINIMAL_CONTRACT_COMPILER_PROMPT.lower()
        assert "semantic_core" in prompt or "semantic core" in prompt

    def test_mentions_compilation(self):
        prompt = MINIMAL_CONTRACT_COMPILER_PROMPT.lower()
        assert "compil" in prompt  # compile, compilation, compiler

    def test_mentions_artifact_requirements(self):
        assert "artifact_requirements" in MINIMAL_CONTRACT_COMPILER_PROMPT

    def test_mentions_column_dtype_targets(self):
        assert "column_dtype_targets" in MINIMAL_CONTRACT_COMPILER_PROMPT

    def test_mentions_required_outputs(self):
        assert "required_outputs" in MINIMAL_CONTRACT_COMPILER_PROMPT

    def test_mentions_evaluation_spec(self):
        assert "evaluation_spec" in MINIMAL_CONTRACT_COMPILER_PROMPT

    def test_mentions_validation_requirements(self):
        assert "validation_requirements" in MINIMAL_CONTRACT_COMPILER_PROMPT

    def test_mentions_iteration_policy(self):
        assert "iteration_policy" in MINIMAL_CONTRACT_COMPILER_PROMPT

    def test_mentions_optimization_policy(self):
        assert "optimization_policy" in MINIMAL_CONTRACT_COMPILER_PROMPT

    def test_mentions_output_manifest_path(self):
        assert "output_manifest_path" in MINIMAL_CONTRACT_COMPILER_PROMPT

    def test_references_schema_examples(self):
        prompt = MINIMAL_CONTRACT_COMPILER_PROMPT.lower()
        assert "schema" in prompt or "examples" in prompt


# ── Integration tests (mock LLM) ─────────────────────────────────────────────


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


def test_contract_compile_prompt_uses_clean_support_context(monkeypatch):
    monkeypatch.delenv("EXECUTION_PLANNER_SECTION_FIRST", raising=False)
    monkeypatch.delenv("EXECUTION_PLANNER_PROGRESSIVE_MODE", raising=False)

    planner = ExecutionPlannerAgent(api_key="mock_key")
    planner.client = object()
    planner._build_model_client = lambda _model_name: object()

    semantic_payload = {
        "scope": "cleaning_only",
        "strategy_title": "CRM prep",
        "business_objective": "Prepare data for future churn modeling without training now.",
        "output_dialect": {"sep": ",", "decimal": ".", "encoding": "utf-8"},
        "canonical_columns": ["lead_id", "created_at", "feature_a", "target"],
        "required_outputs": ["dataset_cleaned.csv", "dataset_enriched.csv"],
        "column_roles": {
            "pre_decision": ["feature_a"],
            "decision": [],
            "outcome": ["target"],
            "post_decision_audit_only": [],
            "unknown": [],
            "identifiers": ["lead_id"],
            "time_columns": ["created_at"],
        },
        "allowed_feature_sets": {
            "segmentation_features": [],
            "model_features": ["feature_a"],
            "forbidden_features": [],
            "audit_only_features": [],
        },
        "task_semantics": {
            "problem_family": "data_preparation",
            "objective_type": "descriptive",
            "primary_target": "target",
            "target_columns": ["target"],
        },
        "active_workstreams": {
            "cleaning": True,
            "feature_engineering": True,
            "model_training": False,
            "review": True,
            "translation": False,
        },
        "future_ml_handoff": {
            "enabled": True,
            "primary_target": "target",
            "target_columns": ["target"],
            "readiness_goal": "ready for later ML",
            "notes": "defer training",
        },
        "model_features": ["feature_a"],
        "cleaning_gates": ["parse dates"],
        "qa_gates": ["no leakage"],
        "reviewer_gates": ["traceability"],
        "data_engineer_runbook": ["clean", "enrich"],
        "optimization_policy": {
            "enabled": False,
            "max_rounds": 1,
            "quick_eval_folds": 1,
            "full_eval_folds": 1,
            "min_delta": 0,
            "patience": 1,
            "allow_model_switch": False,
            "allow_ensemble": False,
            "allow_hpo": False,
            "allow_feature_engineering": True,
            "allow_calibration": False,
        },
    }
    compile_payload = {
        "contract_version": "4.2",
        **semantic_payload,
        "artifact_requirements": {},
        "column_dtype_targets": {},
        "iteration_policy": {
            "max_iterations": 3,
            "metric_improvement_max": 0,
            "runtime_fix_max": 3,
            "compliance_bootstrap_max": 2,
        },
    }

    calls = []

    def _fake_generate(_client, _prompt, output_token_floor=1024, *, model_name=None, tool_mode="contract"):
        response = MagicMock()
        response.candidates = []
        response.usage_metadata = None
        response.text = json.dumps(semantic_payload if tool_mode == "semantic" else compile_payload)
        calls.append(tool_mode)
        return response, {"model_name": model_name, "max_output_tokens": output_token_floor}

    planner._generate_content_with_budget = _fake_generate

    planner.generate_contract(
        strategy={
            "required_columns": ["lead_id", "created_at", "feature_a", "target"],
            "title": "Prompt test",
            "objective_type": "descriptive",
        },
        business_objective="Prepare data and features for future target prediction without training now.",
        column_inventory=["lead_id", "created_at", "feature_a", "target"],
        data_summary="target_candidates:\n- country\n- target\n",
        data_profile={"dtypes": {"lead_id": "object", "created_at": "object", "feature_a": "float64", "target": "float64"}},
    )

    assert calls == ["semantic", "contract"]
    prompt = planner.last_prompt or ""
    assert "SUPPORT_CONTEXT" in prompt
