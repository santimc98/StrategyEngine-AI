from src.agents.ml_engineer import MLEngineerAgent


def _assert_contains_all(text: str, *needles: str) -> None:
    for needle in needles:
        assert needle in text


def _assert_contains_terms(text: str, *terms: str) -> None:
    lowered = text.lower()
    for term in terms:
        assert term.lower() in lowered


def test_ml_engineer_prompt_renders_decisioning_and_visual_context():
    agent = MLEngineerAgent.__new__(MLEngineerAgent)
    ml_view = {
        "decisioning_requirements": {
            "enabled": True,
            "policy_notes": "Use rank tiers.",
            "output": {"required_columns": [{"name": "priority_rank"}]},
        },
        "visual_requirements": {
            "enabled": True,
            "items": [{"id": "dist"}],
        },
    }
    template = (
        "DECISIONING REQUIREMENTS CONTEXT:\n$decisioning_requirements_context\n"
        "DECISIONING COLUMNS:\n$decisioning_columns_text\n"
        "VISUAL REQUIREMENTS:\n$visual_requirements_context\n"
    )
    prompt = agent._build_system_prompt(template, {}, ml_view=ml_view, execution_contract={})
    assert "$decisioning_requirements_context" not in prompt
    assert "$visual_requirements_context" not in prompt
    _assert_contains_all(
        prompt,
        "DECISIONING REQUIREMENTS CONTEXT",
        "VISUAL REQUIREMENTS",
        "priority_rank",
        "dist",
    )
    _assert_contains_terms(prompt, "rank tiers")


def test_ml_engineer_artifact_schema_block_includes_expected_row_count():
    agent = MLEngineerAgent.__new__(MLEngineerAgent)
    contract = {
        "artifact_requirements": {
            "file_schemas": {
                "data/submission.csv": {
                    "required_columns": ["id", "prediction"],
                    "expected_row_count": 270000,
                }
            }
        }
    }
    block = agent._render_artifact_schema_block(contract, {})
    _assert_contains_all(block, "data/submission.csv", "id", "prediction")
    _assert_contains_terms(block, "artifact", "expected_row_count", "270,000")


def test_ml_engineer_partitioning_context_renders_expected_row_hints():
    agent = MLEngineerAgent.__new__(MLEngineerAgent)
    contract = {
        "evaluation_spec": {"n_train_rows": 630000, "n_test_rows": 270000},
        "artifact_requirements": {
            "file_schemas": {
                "data/submission.csv": {"expected_row_count": 270000},
                "data/scored_rows.csv": {"expected_row_count": 900000},
            }
        },
    }
    ml_plan = {
        "training_rows_policy": "custom",
        "split_column": "is_train",
        "train_filter": {
            "type": "custom_rule",
            "rule": "(is_train == 1) & (target.notnull())",
        },
    }
    context = agent._build_data_partitioning_context(contract, {}, ml_plan)
    _assert_contains_all(
        context,
        "DATA PARTITIONING CONTEXT",
        "900,000",
        "630,000",
        "270,000",
        "data/submission.csv",
        "(is_train == 1) & (target.notnull())",
    )
    _assert_contains_terms(context, "training rows", "test/scoring rows", "train filter rule")


def test_ml_engineer_partitioning_context_renders_rules_without_row_counts():
    agent = MLEngineerAgent.__new__(MLEngineerAgent)
    contract = {
        "split_spec": {
            "status": "resolved",
            "split_column": "is_train",
            "training_rows_rule": "rows where target is not missing",
            "scoring_rows_rule": "rows where target is missing",
            "training_rows_policy": "only_rows_with_label",
            "train_filter": {"type": "label_not_null", "column": "target"},
        }
    }
    context = agent._build_data_partitioning_context(contract, {}, {})
    _assert_contains_all(context, "DATA PARTITIONING CONTEXT", "resolved", "target")
    _assert_contains_terms(context, "split resolution status", "train filter rule", "not null")
