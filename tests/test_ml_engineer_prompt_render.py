from src.agents.ml_engineer import MLEngineerAgent


class _FakeOpenAI:
    def __init__(self, api_key=None, base_url=None, timeout=None, default_headers=None):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.default_headers = default_headers


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


def test_ml_engineer_compacts_cleaned_ml_fact_packet_for_prompt():
    agent = MLEngineerAgent.__new__(MLEngineerAgent)
    packet = agent._compact_cleaned_ml_fact_packet_for_prompt(
        {
            "row_count_total": 7447,
            "target_column": "churn_60d",
            "rows_labeled_target": 6667,
            "rows_unlabeled_target": 780,
            "expected_scoring_row_count": 780,
            "split_value_counts": [
                {"split_value": "train", "rows": 6261},
                {"split_value": "holdout", "rows": 406},
                {"split_value": "scoring", "rows": 780},
            ],
            "temporal_fold_feasibility": {
                "temporal_ordering_column": "snapshot_month_end",
                "total_positive_rows": 230,
                "windows_with_zero_positives": 1,
                "candidate_folds_with_single_class_training": 1,
                "candidate_rolling_origin_folds": [
                    {"validation_window": "2025-01-31"},
                    {"validation_window": "2025-02-28"},
                ],
                "recommended_guardrail": "verify each train/validation fold before fit",
            },
            "feature_readiness": {
                "scope": "allowed_feature_sets.model_features",
                "candidate_feature_count": 3,
                "buckets": {
                    "numeric_ready": {"count": 1, "sample_columns": ["arr_current"]},
                    "categorical_ready": {"count": 1, "sample_columns": ["region"]},
                    "boolean_ready": {"count": 1, "sample_columns": ["executive_sponsor_present"]},
                },
            },
            "validation_relevant_facts": {
                "validation_method": "temporal_holdout_with_cv",
                "primary_metric": "pr_auc",
            },
        }
    )

    assert packet["expected_scoring_row_count"] == 780
    assert packet["temporal_fold_feasibility"]["total_positive_rows"] == 230
    assert packet["temporal_fold_feasibility"]["candidate_rolling_origin_folds"][0]["validation_window"] == "2025-01-31"
    assert packet["feature_readiness"]["candidate_feature_count"] == 3
    assert packet["feature_readiness"]["buckets"]["numeric_ready"]["sample_columns"] == ["arr_current"]
    assert packet["validation_relevant_facts"]["validation_method"] == "temporal_holdout_with_cv"


def test_generate_code_prompt_includes_cleaned_ml_fact_packet(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-openrouter")
    monkeypatch.setattr("src.agents.ml_engineer.OpenAI", _FakeOpenAI)

    def _fake_call_chat_with_fallback(client, messages, models, call_kwargs=None, logger=None, context_tag=None):
        return {"dummy": True}, models[0]

    monkeypatch.setattr("src.agents.ml_engineer.call_chat_with_fallback", _fake_call_chat_with_fallback)
    monkeypatch.setattr(
        "src.agents.ml_engineer.extract_response_text",
        lambda response: "import json\nprint('ok')\n",
    )

    agent = MLEngineerAgent()
    _ = agent.generate_code(
        strategy={"title": "Churn Strategy", "analysis_type": "predictive", "required_columns": []},
        data_path="data/cleaned_data.csv",
        execution_contract={
            "required_outputs": ["artifacts/ml/cv_metrics.json"],
            "evaluation_spec": {"primary_metric": "pr_auc"},
            "validation_requirements": {"method": "temporal_holdout_with_cv"},
        },
        ml_view={"required_outputs": ["artifacts/ml/cv_metrics.json"]},
        cleaned_data_summary_min={
            "row_count": 7447,
            "column_count": 34,
            "cleaned_ml_fact_packet": {
                "row_count_total": 7447,
                "target_column": "churn_60d",
                "rows_labeled_target": 6667,
                "rows_unlabeled_target": 780,
                "expected_scoring_row_count": 780,
                "split_value_counts": [
                    {"split_value": "train", "rows": 6261},
                    {"split_value": "holdout", "rows": 406},
                    {"split_value": "scoring", "rows": 780},
                ],
                "validation_relevant_facts": {
                    "validation_method": "temporal_holdout_with_cv",
                    "primary_metric": "pr_auc",
                },
            },
        },
    )

    prompt = str(agent.last_prompt or "")
    _assert_contains_all(
        prompt,
        "Cleaned ML Fact Packet",
        "expected_scoring_row_count",
        "temporal_holdout_with_cv",
        "pr_auc",
        "7447",
        "780",
        "derive expected",
        "scoring/test counts from the authoritative row filter",
    )
