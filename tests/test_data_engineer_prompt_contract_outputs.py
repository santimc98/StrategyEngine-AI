from types import SimpleNamespace
from unittest.mock import patch

from src.agents.data_engineer import DataEngineerAgent


def _mock_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
            )
        ]
    )


def test_data_engineer_prompt_prioritizes_owned_required_outputs_from_contract():
    agent = DataEngineerAgent(api_key="fake")
    execution_contract = {
        "scope": "cleaning_only",
        "future_ml_handoff": {"enabled": True, "future_target": "converted_to_opportunity_90d"},
        "required_outputs": [
            {
                "intent": "dataset_clean",
                "path": "artifacts/clean/dataset_clean.csv",
                "owner": "data_engineer",
                "required": True,
                "kind": "dataset",
            },
            {
                "intent": "dataset_enriched",
                "path": "artifacts/clean/dataset_enriched.csv",
                "owner": "data_engineer",
                "required": True,
                "kind": "dataset",
            },
            {
                "intent": "data_dictionary",
                "path": "artifacts/clean/data_dictionary.json",
                "owner": "data_engineer",
                "required": True,
                "kind": "metadata",
            },
            {
                "intent": "decision_log",
                "path": "artifacts/clean/decision_log.json",
                "owner": "data_engineer",
                "required": True,
                "kind": "metadata",
            },
        ],
    }
    de_view = {
        "required_columns": ["lead_id", "converted_to_opportunity_90d"],
        "output_path": "artifacts/clean/dataset_enriched.csv",
        "output_manifest_path": "artifacts/clean/cleaning_manifest.json",
        "required_outputs": [
            "artifacts/clean/dataset_clean.csv",
            "artifacts/clean/dataset_enriched.csv",
            "artifacts/clean/data_dictionary.json",
            "artifacts/clean/decision_log.json",
        ],
        "cleaning_gates": [],
        "data_engineer_runbook": {"steps": ["clean", "persist"]},
    }

    with patch(
        "src.agents.data_engineer.call_chat_with_fallback",
        return_value=(_mock_response("print('ok')"), "mock/model"),
    ):
        code = agent.generate_cleaning_script(
            data_audit="audit",
            strategy={"required_columns": ["lead_id", "converted_to_opportunity_90d"]},
            input_path="data/raw.csv",
            business_objective="prepare cleaned and enriched artifacts for future ML handoff",
            execution_contract=execution_contract,
            de_view=de_view,
        )

    assert "print('ok')" in code
    prompt = agent.last_prompt or ""
    assert "DATA_ENGINEER_REQUIRED_OUTPUTS_CONTEXT" in prompt
    assert "artifacts/clean/dataset_clean.csv" in prompt
    assert "artifacts/clean/data_dictionary.json" in prompt
    assert "artifacts/clean/decision_log.json" in prompt
    assert "Write every required output you own in DATA_ENGINEER_REQUIRED_OUTPUTS_CONTEXT." in prompt
    assert "cleaned output + manifest" not in prompt
    assert "No model training occurs in THIS run." in prompt


def test_data_engineer_prompt_uses_contract_outputs_as_deliverable_closure_not_only_anchors():
    agent = DataEngineerAgent(api_key="fake")
    execution_contract = {
        "scope": "full_pipeline",
        "required_outputs": [
            {"path": "artifacts/clean/dataset_clean.csv", "owner": "data_engineer", "required": True},
            {"path": "artifacts/clean/data_dictionary.json", "owner": "data_engineer", "required": True},
        ],
    }
    de_view = {
        "required_columns": ["id"],
        "output_path": "artifacts/clean/dataset_clean.csv",
        "output_manifest_path": "artifacts/clean/cleaning_manifest.json",
        "cleaning_gates": [],
        "data_engineer_runbook": {"steps": ["clean", "document"]},
    }

    with patch(
        "src.agents.data_engineer.call_chat_with_fallback",
        return_value=(_mock_response("print('ok')"), "mock/model"),
    ):
        agent.generate_cleaning_script(
            data_audit="audit",
            strategy={"required_columns": ["id"]},
            input_path="data/raw.csv",
            execution_contract=execution_contract,
            de_view=de_view,
        )

    prompt = agent.last_prompt or ""
    assert "Do not collapse a multi-artifact contract into one CSV + manifest pair." in prompt
    assert "primary anchors" in prompt
    assert "artifacts/clean/data_dictionary.json" in prompt


def test_data_engineer_prompt_frames_dedup_as_identity_resolution_from_context():
    agent = DataEngineerAgent(api_key="fake")
    execution_contract = {
        "scope": "cleaning_only",
        "required_outputs": [
            {"path": "artifacts/clean/dataset_limpio.csv", "owner": "data_engineer", "required": True},
            {"path": "artifacts/clean/deduplication_decisions.csv", "owner": "data_engineer", "required": True},
        ],
    }
    de_view = {
        "required_columns": ["lead_id", "email", "company_name"],
        "output_path": "artifacts/clean/dataset_limpio.csv",
        "output_manifest_path": "artifacts/clean/cleaning_manifest.json",
        "required_outputs": [
            "artifacts/clean/dataset_limpio.csv",
            "artifacts/clean/deduplication_decisions.csv",
        ],
        "cleaning_gates": [
            {
                "name": "conservative_deduplication_on_contact_and_company",
                "severity": "HARD",
                "action_type": "derive",
                "params": {"subset": ["first_name", "last_name", "email", "company_name"]},
            }
        ],
        "data_engineer_runbook": {"steps": ["standardize", "deduplicate", "persist"]},
    }

    with patch(
        "src.agents.data_engineer.call_chat_with_fallback",
        return_value=(_mock_response("print('ok')"), "mock/model"),
    ):
        agent.generate_cleaning_script(
            data_audit="audit",
            strategy={"required_columns": ["lead_id", "email", "company_name"]},
            input_path="data/raw.csv",
            execution_contract=execution_contract,
            de_view=de_view,
        )

    prompt = agent.last_prompt or ""
    assert "IDENTITY RESOLUTION (WHEN DEDUPLICATION IS IN SCOPE)" in prompt
    assert "Decide which signals are strong, medium, or weak identity evidence for" in prompt
    assert "Treat nulls, placeholders, and missing contact fields as absence of" in prompt
    assert "Do not collapse rows just because a composite key can be mechanically" in prompt
    assert "If the context only supports soft duplicate suspicion, prefer flags/logs" in prompt


def test_data_engineer_prompt_requires_empty_required_artifacts_to_be_materialized():
    agent = DataEngineerAgent(api_key="fake")
    execution_contract = {
        "scope": "cleaning_only",
        "required_outputs": [
            {"path": "artifacts/clean/dataset_limpio.csv", "owner": "data_engineer", "required": True},
            {"path": "artifacts/clean/deduplication_decisions.csv", "owner": "data_engineer", "required": True},
            {"path": "artifacts/audit/leakage_risk_register.json", "owner": "data_engineer", "required": True},
        ],
    }
    de_view = {
        "scope": "cleaning_only",
        "required_columns": ["lead_id"],
        "output_path": "artifacts/clean/dataset_limpio.csv",
        "output_manifest_path": "artifacts/clean/cleaning_manifest.json",
        "required_outputs": [
            "artifacts/clean/dataset_limpio.csv",
            "artifacts/clean/deduplication_decisions.csv",
            "artifacts/audit/leakage_risk_register.json",
        ],
        "cleaning_gates": [],
        "data_engineer_runbook": {"steps": ["clean", "persist"]},
    }

    with patch(
        "src.agents.data_engineer.call_chat_with_fallback",
        return_value=(_mock_response("print('ok')"), "mock/model"),
    ):
        agent.generate_cleaning_script(
            data_audit="audit",
            strategy={"required_columns": ["lead_id"]},
            input_path="data/raw.csv",
            execution_contract=execution_contract,
            de_view=de_view,
        )

    prompt = agent.last_prompt or ""
    assert "still materialize a schema-valid empty artifact" in prompt
    assert "write the empty artifact" in prompt
    assert "\"materialization_policy\": \"required_even_if_empty\"" in prompt


def test_data_engineer_prompt_frames_date_and_numeric_cleaning_as_format_resolution():
    agent = DataEngineerAgent(api_key="fake")
    execution_contract = {
        "scope": "cleaning_only",
        "required_outputs": [
            {"path": "artifacts/clean/dataset_limpio.csv", "owner": "data_engineer", "required": True},
            {"path": "artifacts/reports/data_quality_report.json", "owner": "data_engineer", "required": True},
        ],
    }
    de_view = {
        "required_columns": ["created_at", "last_activity_at", "annual_revenue", "email_open_rate_90d"],
        "output_path": "artifacts/clean/dataset_limpio.csv",
        "output_manifest_path": "artifacts/clean/cleaning_manifest.json",
        "required_outputs": [
            "artifacts/clean/dataset_limpio.csv",
            "artifacts/reports/data_quality_report.json",
        ],
        "cleaning_gates": [
            {
                "name": "robust_date_parsing_with_invalid_flagging",
                "severity": "HARD",
                "action_type": "parse",
                "params": {"target_columns": ["created_at", "last_activity_at"]},
            },
            {
                "name": "normalize_numeric_ranges_and_amounts",
                "severity": "HARD",
                "action_type": "standardize",
                "params": {"target_columns": ["annual_revenue", "email_open_rate_90d"]},
            },
        ],
        "column_dtype_targets": {
            "created_at": {"target_dtype": "datetime"},
            "last_activity_at": {"target_dtype": "datetime"},
            "annual_revenue": {"target_dtype": "float64"},
            "email_open_rate_90d": {"target_dtype": "float64"},
        },
        "data_engineer_runbook": {"steps": ["parse dates", "normalize numerics", "persist"]},
    }

    with patch(
        "src.agents.data_engineer.call_chat_with_fallback",
        return_value=(_mock_response("print('ok')"), "mock/model"),
    ):
        agent.generate_cleaning_script(
            data_audit="audit",
            strategy={"required_columns": ["created_at", "annual_revenue"]},
            input_path="data/raw.csv",
            execution_contract=execution_contract,
            de_view=de_view,
        )

    prompt = agent.last_prompt or ""
    assert "FORMAT RESOLUTION (BEFORE final casting)" in prompt
    assert "format families in THIS dataset" in prompt
    assert "Do not assume one parser or one locale is enough" in prompt
    assert "salvages defensible values" in prompt
    assert "coercing unresolved strings to null" in prompt
    assert "final null inflation is" in prompt
    assert "observed raw quality" in prompt


def test_data_engineer_prompt_treats_temporal_completeness_as_explicit_contractual_requirement():
    agent = DataEngineerAgent(api_key="fake")
    execution_contract = {
        "scope": "cleaning_only",
        "required_outputs": [
            {"path": "artifacts/clean/dataset_cleaned.csv", "owner": "data_engineer", "required": True},
        ],
    }
    de_view = {
        "scope": "cleaning_only",
        "required_columns": ["created_at", "lead_id"],
        "output_path": "artifacts/clean/dataset_cleaned.csv",
        "output_manifest_path": "artifacts/clean/cleaning_manifest.json",
        "required_outputs": ["artifacts/clean/dataset_cleaned.csv"],
        "cleaning_gates": [
            {
                "name": "robust_date_parsing_with_invalid_flagging",
                "severity": "HARD",
                "action_type": "parse",
                "params": {"target_columns": ["created_at"]},
            }
        ],
        "column_dtype_targets": {
            "created_at": {"target_dtype": "datetime"},
        },
        "data_engineer_runbook": {"steps": ["parse dates", "flag unresolved values", "persist"]},
    }

    with patch(
        "src.agents.data_engineer.call_chat_with_fallback",
        return_value=(_mock_response("print('ok')"), "mock/model"),
    ):
        agent.generate_cleaning_script(
            data_audit="audit",
            strategy={"required_columns": ["created_at", "lead_id"]},
            input_path="data/raw.csv",
            execution_contract=execution_contract,
            de_view=de_view,
        )

    prompt = agent.last_prompt or ""
    assert 'Do not infer a hard "no nulls after parsing" requirement for temporal' in prompt
    assert "explicitly requires complete recoverability" in prompt
    assert 'Do not convert "required for downstream use" into "must be fully non-null"' in prompt
    assert "A parsed datetime column is not automatically a hard completeness gate." in prompt


def test_data_engineer_prompt_derives_cleaning_scope_from_de_view_context():
    agent = DataEngineerAgent(api_key="fake")
    execution_contract = {
        "future_ml_handoff": {"enabled": True, "future_target": "converted_to_opportunity_90d"},
        "required_outputs": [
            {"path": "artifacts/clean/dataset_limpio.csv", "owner": "data_engineer", "required": True},
        ],
    }
    de_view = {
        "scope": "cleaning_only",
        "active_workstreams": {
            "cleaning": True,
            "feature_engineering": True,
            "model_training": False,
        },
        "required_columns": ["lead_id", "annual_revenue"],
        "output_path": "artifacts/clean/dataset_limpio.csv",
        "output_manifest_path": "artifacts/clean/cleaning_manifest.json",
        "required_outputs": ["artifacts/clean/dataset_limpio.csv"],
        "cleaning_gates": [],
    }

    with patch(
        "src.agents.data_engineer.call_chat_with_fallback",
        return_value=(_mock_response("print('ok')"), "mock/model"),
    ):
        agent.generate_cleaning_script(
            data_audit="audit",
            strategy={"required_columns": ["lead_id", "annual_revenue"]},
            input_path="data/raw.csv",
            execution_contract=execution_contract,
            de_view=de_view,
        )

    prompt = agent.last_prompt or ""
    assert "PIPELINE SCOPE: CLEANING_ONLY" in prompt
    assert "PIPELINE SCOPE: FULL_PIPELINE" not in prompt


def test_data_engineer_prompt_includes_column_resolution_context_as_support_evidence():
    agent = DataEngineerAgent(api_key="fake")
    execution_contract = {
        "scope": "cleaning_only",
        "required_outputs": [
            {"path": "artifacts/clean/dataset_limpio.csv", "owner": "data_engineer", "required": True},
        ],
    }
    de_view = {
        "required_columns": ["created_at", "annual_revenue"],
        "output_path": "artifacts/clean/dataset_limpio.csv",
        "output_manifest_path": "artifacts/clean/cleaning_manifest.json",
        "required_outputs": ["artifacts/clean/dataset_limpio.csv"],
        "cleaning_gates": [],
        "column_dtype_targets": {
            "created_at": {"target_dtype": "datetime"},
            "annual_revenue": {"target_dtype": "float64"},
        },
        "column_resolution_context": {
            "created_at": {
                "semantic_kind": "datetime_like",
                "observed_format_families": ["iso_date", "slash_date"],
                "top_raw_examples": ["2025-07-08", "27/06/2025", "not_a_date"],
            },
            "annual_revenue": {
                "semantic_kind": "amount_like",
                "observed_format_families": ["currency_symbol", "magnitude_suffix"],
                "top_raw_examples": ["$350k", "0.1M", "unknown"],
            },
        },
        "data_engineer_runbook": {"steps": ["parse dates", "normalize numerics", "persist"]},
    }

    with patch(
        "src.agents.data_engineer.call_chat_with_fallback",
        return_value=(_mock_response("print('ok')"), "mock/model"),
    ):
        agent.generate_cleaning_script(
            data_audit="audit",
            strategy={"required_columns": ["created_at", "annual_revenue"]},
            input_path="data/raw.csv",
            execution_contract=execution_contract,
            de_view=de_view,
        )

    prompt = agent.last_prompt or ""
    assert "COLUMN_RESOLUTION_CONTEXT" in prompt
    assert "raw formats, placeholders" in prompt
    assert "Use COLUMN_RESOLUTION_CONTEXT first" in prompt
    assert "\"annual_revenue\"" in prompt
    assert "\"currency_symbol\"" in prompt


def test_data_engineer_prompt_includes_artifact_obligations_as_contract_extraction_context():
    agent = DataEngineerAgent(api_key="fake")
    execution_contract = {
        "scope": "cleaning_only",
        "artifact_requirements": {
            "cleaned_dataset": {
                "output_path": "artifacts/clean/dataset_cleaned.csv",
                "output_manifest_path": "artifacts/clean/cleaning_manifest.json",
                "required_columns": ["lead_id", "created_at"],
                "optional_passthrough_columns": ["raw_event_ts"],
                "column_transformations": {"drop_columns": ["internal_debug_flag"]},
            },
            "enriched_dataset": {
                "output_path": "artifacts/clean/dataset_enriched.csv",
                "required_columns": ["created_at", "score_target"],
            },
        },
        "required_outputs": [
            {"path": "artifacts/clean/dataset_cleaned.csv", "owner": "data_engineer", "required": True},
            {"path": "artifacts/clean/dataset_enriched.csv", "owner": "data_engineer", "required": True},
        ],
    }
    de_view = {
        "required_columns": ["lead_id", "created_at"],
        "output_path": "artifacts/clean/dataset_cleaned.csv",
        "output_manifest_path": "artifacts/clean/cleaning_manifest.json",
        "required_outputs": [
            "artifacts/clean/dataset_cleaned.csv",
            "artifacts/clean/dataset_enriched.csv",
        ],
        "cleaning_gates": [],
    }

    with patch(
        "src.agents.data_engineer.call_chat_with_fallback",
        return_value=(_mock_response("print('ok')"), "mock/model"),
    ):
        agent.generate_cleaning_script(
            data_audit="audit",
            strategy={"required_columns": ["lead_id", "created_at"]},
            input_path="data/raw.csv",
            execution_contract=execution_contract,
            de_view=de_view,
        )

    prompt = agent.last_prompt or ""
    assert "ARTIFACT_OBLIGATIONS_CONTEXT" in prompt
    assert "lossless extraction of artifact bindings already declared in the contract" in prompt
    assert "\"binding_name\": \"cleaned_dataset\"" in prompt
    assert "artifact_requirements.cleaned_dataset.output_path" in prompt


def test_data_engineer_repair_mode_uses_previous_script_patch_context():
    agent = DataEngineerAgent(api_key="fake")
    execution_contract = {
        "scope": "cleaning_only",
        "required_outputs": [
            {"path": "artifacts/clean/dataset_cleaned.csv", "owner": "data_engineer", "required": True},
        ],
    }
    de_view = {
        "scope": "cleaning_only",
        "required_columns": ["lead_id", "created_at"],
        "output_path": "artifacts/clean/dataset_cleaned.csv",
        "output_manifest_path": "artifacts/clean/cleaning_manifest.json",
        "required_outputs": ["artifacts/clean/dataset_cleaned.csv"],
        "cleaning_gates": [],
        "data_engineer_runbook": {"steps": ["clean", "persist"]},
    }
    previous_code = (
        "import pandas as pd\n"
        "df = pd.read_csv('data/raw.csv')\n"
        "raise ValueError('boom')\n"
    )
    feedback_record = {
        "agent": "data_engineer",
        "source": "runtime_retry",
        "status": "REJECTED",
        "iteration": 2,
        "feedback": "Runtime retry required after sandbox failure.",
        "failed_gates": ["runtime_failure"],
        "required_fixes": ["Fix the failing pandas operation without rewriting the whole script."],
        "hard_failures": ["runtime_failure"],
        "runtime_error_tail": "ValueError: boom",
    }

    with patch(
        "src.agents.data_engineer.call_chat_with_fallback",
        return_value=(_mock_response("print('patched')"), "mock/model"),
    ):
        code = agent.generate_cleaning_script(
            data_audit="RUNTIME_ERROR_CONTEXT:\nValueError: boom",
            strategy={"required_columns": ["lead_id", "created_at"]},
            input_path="data/raw.csv",
            execution_contract=execution_contract,
            de_view=de_view,
            repair_mode=True,
            previous_code=previous_code,
            feedback_record=feedback_record,
        )

    prompt = agent.last_prompt or ""
    assert "MODE: REPAIR_EDITOR" in prompt
    assert "PREVIOUS_SCRIPT_BODY_TO_PATCH" in prompt
    assert "Do not regenerate from zero" in prompt
    assert "Fix the failing pandas operation without rewriting the whole script." in prompt
    assert "json.dumps = _safe_dumps_json" not in prompt
    assert "print('patched')" in code


def test_data_engineer_repair_mode_promotes_failure_explainer_fix_and_compacts_error_context():
    agent = DataEngineerAgent(api_key="fake")
    execution_contract = {
        "scope": "cleaning_only",
        "required_outputs": [
            {"path": "artifacts/clean/dataset_cleaned.csv", "owner": "data_engineer", "required": True},
        ],
    }
    de_view = {
        "scope": "cleaning_only",
        "required_columns": ["lead_id", "created_at"],
        "output_path": "artifacts/clean/dataset_cleaned.csv",
        "output_manifest_path": "artifacts/clean/cleaning_manifest.json",
        "required_outputs": ["artifacts/clean/dataset_cleaned.csv"],
        "cleaning_gates": [],
        "data_engineer_runbook": {"steps": ["clean", "persist"]},
    }
    previous_code = "import pandas as pd\ndf = pd.read_csv('data/raw.csv')\nraise KeyError('boom')\n"
    feedback_record = {
        "agent": "data_engineer",
        "source": "runtime_retry",
        "status": "REJECTED",
        "iteration": 2,
        "feedback": "Runtime retry required after sandbox failure.",
        "failed_gates": ["runtime_failure"],
        "required_fixes": [],
        "hard_failures": ["runtime_failure"],
        "runtime_error_tail": "KeyError: boom",
    }
    data_audit = "\n".join(
        [
            "RUNTIME_ERROR_CONTEXT:",
            "KeyError: boom",
            "",
            "TRACEBACK_TAIL_20:",
            "Traceback line 1",
            "Traceback line 2",
            "",
            "LLM_FAILURE_EXPLANATION:",
            "WHERE: Inside the deduplication stage.",
            "WHY: The retained frame lost the grouping column before the selection step.",
            "FIX: Preserve _dedup_group_key on the retained rows before selecting decision columns.",
            "DIAGNOSTIC: Print kept.columns before the failing selection.",
            "",
            "LATEST_ITERATION_FEEDBACK_RECORD_JSON:",
            '{"agent":"data_engineer","required_fixes":[]}',
            "",
            "ITERATION_FEEDBACK_CONTEXT:",
            "Very long stale narrative that should not appear inside REPAIR_ERROR_CONTEXT.",
        ]
    )

    with patch(
        "src.agents.data_engineer.call_chat_with_fallback",
        return_value=(_mock_response("print('patched')"), "mock/model"),
    ):
        code = agent.generate_cleaning_script(
            data_audit=data_audit,
            strategy={"required_columns": ["lead_id", "created_at"]},
            input_path="data/raw.csv",
            execution_contract=execution_contract,
            de_view=de_view,
            repair_mode=True,
            previous_code=previous_code,
            feedback_record=feedback_record,
        )

    prompt = agent.last_prompt or ""
    assert "Preserve _dedup_group_key on the retained rows before selecting decision columns." in prompt
    repair_context = prompt.split("REPAIR_ERROR_CONTEXT:", 1)[1].split("PREVIOUS_SCRIPT_BODY_TO_PATCH:", 1)[0]
    assert "LLM_FAILURE_EXPLANATION:" in repair_context
    assert "LATEST_ITERATION_FEEDBACK_RECORD_JSON:" not in repair_context
    assert "ITERATION_FEEDBACK_CONTEXT:" not in repair_context
    assert "Runtime retry required after sandbox failure." in repair_context
    assert "print('patched')" in code
