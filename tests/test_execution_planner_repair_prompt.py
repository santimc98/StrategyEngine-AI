import json

from unittest.mock import MagicMock

from src.agents.execution_planner import (
    ExecutionPlannerAgent,
    SEMANTIC_EXECUTION_PLANNER_PROMPT,
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


def test_minimal_contract_prompt_declares_active_workstreams_and_future_handoff():
    prompt = MINIMAL_CONTRACT_COMPILER_PROMPT
    assert "active_workstreams" in prompt
    assert "future_ml_handoff" in prompt
    assert "Do NOT set model_training=true merely because a future target exists." in prompt
    assert "Never choose full_pipeline only because a future predictive target exists." in prompt


def test_semantic_execution_planner_prompt_focuses_on_run_intent_before_compilation():
    prompt = SEMANTIC_EXECUTION_PLANNER_PROMPT
    assert "Produce ONE semantic_core JSON object" in prompt
    assert "Do not emit artifact_requirements" in prompt
    assert "A future target does NOT imply model_training=true." in prompt
    assert "MISSION" in prompt
    assert "SOURCE OF TRUTH AND PRECEDENCE" in prompt
    assert "SEMANTIC PLANNING WORKFLOW (MANDATORY)" in prompt


def test_semantic_execution_planner_prompt_enforces_model_feature_closure_for_future_handoff():
    prompt = SEMANTIC_EXECUTION_PLANNER_PROMPT
    assert "Do not leave model_features empty merely because the current run is not training a model." in prompt
    assert "allowed_feature_sets may describe conceptual families" in prompt
    assert "model_features must name explicit columns" in prompt
    assert "reduce the future modeling handoff/readiness claim instead of emitting an empty model_features list" in prompt


def test_contract_compiler_prompt_treats_semantic_core_as_authority():
    prompt = MINIMAL_CONTRACT_COMPILER_PROMPT
    assert "SEMANTIC_CORE_AUTHORITY_JSON is the authoritative semantic source of truth" in prompt
    assert "Never override SEMANTIC_CORE_AUTHORITY_JSON with SUPPORT_CONTEXT." in prompt
    assert "SOURCE OF TRUTH AND PRECEDENCE" in prompt
    assert "Your job is to COMPILE the contract" in prompt


def test_contract_compiler_prompt_focuses_on_minimal_operational_layer():
    prompt = MINIMAL_CONTRACT_COMPILER_PROMPT
    assert "Compile only the minimal operational layer needed to make the contract executable." in prompt
    assert "Preserve these semantic sections VERBATIM" in prompt
    assert "do NOT invent training, CV, or benchmark sections" in prompt


def test_contract_compiler_prompt_declares_senior_compilation_workflow():
    prompt = MINIMAL_CONTRACT_COMPILER_PROMPT
    assert "SENIOR COMPILATION WORKFLOW (MANDATORY)" in prompt
    assert "Identify what the semantic core already decided and preserve it." in prompt
    assert "Build the smallest executable contract that preserves the semantic core" in prompt


def test_contract_compiler_prompt_explains_processing_only_dependencies_via_optional_passthrough():
    prompt = MINIMAL_CONTRACT_COMPILER_PROMPT
    assert "optional_passthrough_columns: list[str] for processing-only dependencies." in prompt
    assert "Do not force every transform-only dependency into required_columns" in prompt
    assert "If a HARD cleaning gate or transform references a column" in prompt
    assert "must be covered by" in prompt
    assert "optional_passthrough_columns instead of silently dropping it" in prompt


def test_contract_compiler_prompt_requires_explicit_agent_interfaces():
    prompt = MINIMAL_CONTRACT_COMPILER_PROMPT
    assert "agent_interfaces" in prompt
    assert "thin adapters" in prompt
    assert "Each block should expose only what that agent needs" in prompt


def test_contract_compiler_prompt_treats_agent_interfaces_as_optional_thin_deltas():
    prompt = MINIMAL_CONTRACT_COMPILER_PROMPT
    assert "agent_interfaces is OPTIONAL" in prompt
    assert "thin deltas" in prompt
    assert "Never mirror full gate lists" in prompt


def test_contract_compiler_prompt_allows_required_output_intent_materialization():
    prompt = MINIMAL_CONTRACT_COMPILER_PROMPT
    assert "required_outputs semantically" in prompt
    assert "materialize them as required_outputs objects with path + intent" in prompt


def test_contract_compiler_prompt_requests_gate_action_semantics():
    prompt = MINIMAL_CONTRACT_COMPILER_PROMPT
    assert 'action_type: one of ["drop", "parse", "coerce", "impute", "standardize", "derive", "check"]' in prompt
    assert "final_state: one of" in prompt


def test_contract_compiler_prompt_treats_translator_as_lightweight_and_excludes_failure_explainer():
    prompt = MINIMAL_CONTRACT_COMPILER_PROMPT
    assert "translator and results_advisor are OPTIONAL light interfaces" in prompt
    assert "Do NOT create any failure_explainer interface." in prompt


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
    assert "resolved_target:" not in prompt
    assert "downstream_consumer_interface:" not in prompt
    assert "evidence_policy:" not in prompt
