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
