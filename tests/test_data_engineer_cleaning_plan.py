import json
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


def test_data_engineer_generates_structured_cleaning_plan_from_context(tmp_path):
    csv_path = tmp_path / "raw.csv"
    csv_path.write_text(
        "account_id,snapshot_month_end,arr_current\n"
        "A1,2025-01-31,\"EUR 279,981\"\n"
        "A2,31/01/2025,0.03m\n",
        encoding="utf-8",
    )
    agent = DataEngineerAgent(api_key="fake")
    execution_contract = {
        "scope": "full_pipeline",
        "required_outputs": [
            {"path": "artifacts/clean/accounts.csv", "owner": "data_engineer", "required": True},
            {"path": "artifacts/clean/cleaning_manifest.json", "owner": "data_engineer", "required": True},
        ],
    }
    de_view = {
        "required_columns": ["account_id", "snapshot_month_end", "arr_current"],
        "output_path": "artifacts/clean/accounts.csv",
        "output_manifest_path": "artifacts/clean/cleaning_manifest.json",
        "cleaning_gates": [
            {
                "name": "robust_date_and_amount_parsing",
                "severity": "HARD",
                "params": {"target_columns": ["snapshot_month_end", "arr_current"]},
            }
        ],
    }
    llm_payload = {
        "plan_version": "data_cleaning_plan_v1",
        "objective_summary": "Clean temporal and ARR fields while preserving all owned outputs.",
        "owned_deliverables": ["artifacts/clean/accounts.csv", "artifacts/clean/cleaning_manifest.json"],
        "operation_order": ["load", "parse_dates", "normalize_arr", "write_outputs"],
        "gate_feasibility_review": [
            {
                "gate_name": "robust_date_and_amount_parsing",
                "status": "requires_implementation",
                "evidence": "mixed date and numeric formats in sample",
                "implementation_note": "Use staged parsing and manifest gate status.",
            }
        ],
    }

    plan = agent.generate_cleaning_plan(
        data_audit="DATA AUDIT",
        strategy={"title": "Temporal churn scoring"},
        input_path="data/raw.csv",
        prompt_input_path=str(csv_path),
        execution_contract=execution_contract,
        de_view=de_view,
        llm_call=lambda _system, _user: json.dumps(llm_payload),
    )

    assert plan["plan_source"] == "llm"
    assert plan["operation_order"] == ["load", "parse_dates", "normalize_arr", "write_outputs"]
    assert plan["gate_feasibility_review"][0]["gate_name"] == "robust_date_and_amount_parsing"
    prompt = agent.last_plan_prompt or ""
    assert "Senior Data Cleaning Architect" in prompt
    assert "DATA_SAMPLE_CONTEXT" in prompt
    assert "EUR 279,981" in prompt
    assert "robust_date_and_amount_parsing" in prompt


def test_data_engineer_plan_call_uses_independent_context_tag_and_model_chain(monkeypatch):
    agent = DataEngineerAgent(api_key="fake")
    agent.plan_model_name = "google/gemini-3.1-pro-preview"
    agent.model_name = "anthropic/claude-opus-4.6"
    agent.fallback_model_name = "anthropic/claude-sonnet-4.6"
    captured = {}

    def fake_call_chat_with_fallback(client, messages, model_chain, call_kwargs, logger, context_tag):
        captured["model_chain"] = list(model_chain)
        captured["context_tag"] = context_tag
        captured["temperature"] = call_kwargs["temperature"]
        return _mock_response("{}"), model_chain[0]

    monkeypatch.setattr("src.agents.data_engineer.call_chat_with_fallback", fake_call_chat_with_fallback)

    assert agent._execute_cleaning_plan_llm_call("system", "user") == "{}"
    assert captured["context_tag"] == "data_engineer_plan"
    assert captured["temperature"] == 0.1
    assert captured["model_chain"] == [
        "google/gemini-3.1-pro-preview",
        "anthropic/claude-opus-4.6",
        "anthropic/claude-sonnet-4.6",
    ]


def test_cleaning_script_prompt_includes_plan_as_repair_memory():
    agent = DataEngineerAgent(api_key="fake")
    data_cleaning_plan = {
        "plan_version": "data_cleaning_plan_v1",
        "operation_order": ["load", "parse_snapshot_month_end", "write_outputs"],
        "gate_feasibility_review": [
            {
                "gate_name": "snapshot_month_end_parse_coverage",
                "status": "requires_implementation",
                "implementation_note": "Use staged date parsing.",
            }
        ],
    }

    with patch(
        "src.agents.data_engineer.call_chat_with_fallback",
        return_value=(_mock_response("print('patched')"), "mock/model"),
    ):
        agent.generate_cleaning_script(
            data_audit="RUNTIME_ERROR_CONTEXT:\nValueError: snapshot_month_end parse failed",
            strategy={"required_columns": ["account_id", "snapshot_month_end"]},
            input_path="data/raw.csv",
            execution_contract={
                "required_outputs": [
                    {"path": "artifacts/clean/accounts.csv", "owner": "data_engineer", "required": True}
                ]
            },
            de_view={
                "required_columns": ["account_id", "snapshot_month_end"],
                "output_path": "artifacts/clean/accounts.csv",
                "output_manifest_path": "artifacts/clean/manifest.json",
                "cleaning_gates": [],
            },
            repair_mode=True,
            previous_code="import pandas as pd\nprint('prev')\n",
            feedback_record={
                "agent": "data_engineer",
                "status": "REJECTED",
                "required_fixes": ["Patch snapshot_month_end parser only."],
                "runtime_error_tail": "ValueError: snapshot_month_end parse failed",
            },
            data_cleaning_plan=data_cleaning_plan,
        )

    prompt = agent.last_prompt or ""
    assert "DATA_CLEANING_PLAN_CONTEXT" in prompt
    assert "parse_snapshot_month_end" in prompt
    assert "Use DATA_CLEANING_PLAN_CONTEXT as memory of the initial architecture" in prompt
    assert "STRUCTURED_REPAIR_GROUND_TRUTH_JSON" in prompt
