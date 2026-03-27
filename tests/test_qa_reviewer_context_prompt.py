from src.agents.qa_reviewer import QAReviewerAgent


def test_qa_reviewer_prompt_uses_context_first_structure_for_data_engineer_review():
    agent = QAReviewerAgent(api_key=None)
    agent.review_code(
        "print('clean')",
        {"title": "Cleaning only"},
        "Audit cleaned outputs",
        evaluation_spec={
            "review_subject": "data_engineer",
            "subject_required_outputs": ["artifacts/clean/dataset_cleaned.csv"],
            "qa_required_outputs": ["artifacts/qa/data_validation_results.json"],
            "subject_code_path_hint": "artifacts/data_engineer_last.py",
            "qa_gates": [{"name": "verify_exclusions", "severity": "HARD"}],
        },
    )
    prompt = agent.last_prompt or ""
    assert "MISSION:" in prompt
    assert "SOURCE OF TRUTH AND PRECEDENCE:" in prompt
    assert "QA DECISION WORKFLOW (MANDATORY):" in prompt
    assert "Review Subject: data_engineer" in prompt
    assert "Subject Required Outputs" in prompt


def test_qa_reviewer_prompt_uses_evidence_patterns_instead_of_code_shape_mandates():
    agent = QAReviewerAgent(api_key=None)
    agent.review_code(
        "print('clean')",
        {"title": "ML audit"},
        "Audit cleaned outputs",
        evaluation_spec={
            "review_subject": "ml_engineer",
            "subject_required_outputs": ["artifacts/ml/cv_metrics.json"],
            "qa_required_outputs": ["artifacts/qa/qa_report.json"],
            "subject_code_path_hint": "artifacts/ml_engineer_last.py",
            "qa_gates": [
                {"name": "input_csv_loading", "severity": "HARD"},
                {"name": "no_synthetic_data", "severity": "HARD"},
                {"name": "contract_columns", "severity": "SOFT"},
            ],
        },
    )
    prompt = agent.last_prompt or ""
    assert "correct data provenance over a specific API call shape" in prompt
    assert "judge them in context instead of treating them as a blind string match" in prompt
    assert "not the only acceptable implementation" in prompt
    assert "The code MUST call pandas.read_csv" not in prompt
    assert "The code MUST NOT fabricate datasets" not in prompt
    assert "The code MUST reference canonical contract columns explicitly" not in prompt
