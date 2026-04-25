from src.agents.qa_reviewer import (
    QAReviewerAgent,
    _apply_metric_gate_consistency_guard,
    _build_deterministic_metric_facts,
)


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


def test_qa_reviewer_prompt_includes_hard_blocker_packet_for_restored_candidates():
    agent = QAReviewerAgent(api_key=None)
    agent.review_code(
        "\n".join(
            [
                "if int(holdout_mask.sum()) < 400:",
                "    raise ValueError('holdout too small')",
                "scored_output['churn_risk_score'] = baseline_model.predict_proba(X_score)[:, 1]",
            ]
        ),
        {"title": "ML audit"},
        "Audit restored candidate",
        evaluation_spec={
            "review_subject": "ml_engineer",
            "subject_code_path_hint": "artifacts/ml_engineer_last.py",
            "qa_gates": [
                {
                    "name": "temporal_validation_credibility",
                    "severity": "HARD",
                    "params": {"min_rows": 1000},
                },
                {
                    "name": "scoring_output_primary_model",
                    "severity": "HARD",
                    "params": {"model": "primary_model"},
                },
            ],
            "review_history_context": {
                "best_attempt_restored_recently": True,
                "feedback_history_tail": [
                    "BEST_ATTEMPT_RESTORED[result_evaluator]: restored attempt 2 as authoritative state after a later degraded execution."
                ],
                "last_gate_context": {
                    "failed_gates": ["temporal_validation_credibility"],
                    "required_fixes": [
                        "Use primary_model.predict_proba for scoring.",
                    ],
                },
            },
        },
    )
    prompt = agent.last_prompt or ""
    assert "HARD_BLOCKER_PACKET" in prompt
    assert "best_attempt_restored_recently" in prompt
    assert "baseline_model.predict_proba" in prompt


def test_qa_reviewer_enforces_artifact_backed_hard_numeric_gate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    metrics_dir = tmp_path / "artifacts" / "ml"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "latency_benchmark.json").write_text(
        '{"ms_per_1000_debtors": 135.54}',
        encoding="utf-8",
    )
    qa_gates = [
        {
            "name": "inference_latency_within_spec",
            "severity": "HARD",
            "applies_to_artifact": "artifacts/ml/latency_benchmark.json",
            "params": {
                "metric": "ms_per_1000_debtors",
                "min_value": 10,
                "max_value": 30,
            },
        }
    ]

    facts = _build_deterministic_metric_facts(
        evaluation_spec={},
        qa_gates=qa_gates,
        subject_required_outputs=[
            {"path": "artifacts/ml/latency_benchmark.json", "intent": "latency_benchmark"}
        ],
        qa_required_outputs=[],
    )

    assert facts["gate_metric_facts"][0]["passed"] is False
    result, notes = _apply_metric_gate_consistency_guard(
        {"status": "APPROVED", "failed_gates": [], "hard_failures": []},
        qa_gates,
        facts,
    )
    assert result["status"] == "REJECTED"
    assert result["failed_gates"] == ["inference_latency_within_spec"]
    assert result["hard_failures"] == ["inference_latency_within_spec"]
    assert notes and "QA_METRIC_FACT_ENFORCED" in notes[0]


def test_qa_reviewer_composite_gate_prefers_current_artifact_over_baseline_history(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    metrics_dir = tmp_path / "artifacts" / "ml"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "validation_metrics.json").write_text(
        '{"eligibility_drift_entity_pct": 0.0008, "eligibility_drift_ventas_pct": 0.01, "eligibility_drift_saldo_pct": 0.02}',
        encoding="utf-8",
    )
    qa_gates = [
        {
            "name": "eligibility_drift_within_tolerance",
            "severity": "HARD",
            "applies_to_artifact": "artifacts/ml/validation_metrics.json",
            "evidence_source": (
                "validation_metrics.json.eligibility_drift_entity_pct, "
                "eligibility_drift_ventas_pct, eligibility_drift_saldo_pct for confirmation month"
            ),
            "params": {
                "max_entity_drift_pct": 0.10,
                "max_ventas_drift_pct": 0.10,
                "max_saldo_drift_pct": 0.10,
            },
        }
    ]

    facts = _build_deterministic_metric_facts(
        evaluation_spec={
            "metrics_payload": {
                "eligibility_drift_entity_pct": 0.1066,
                "eligibility_drift_ventas_pct": 0.1066,
                "eligibility_drift_saldo_pct": 0.1066,
            }
        },
        qa_gates=qa_gates,
        subject_required_outputs=[
            {"path": "artifacts/ml/validation_metrics.json", "intent": "validation_metrics"}
        ],
        qa_required_outputs=[],
    )

    gate_facts = facts["gate_metric_facts"]
    assert len(gate_facts) == 3
    assert all(fact["passed"] is True for fact in gate_facts)
    assert all(fact["source"] == "artifacts/ml/validation_metrics.json" for fact in gate_facts)

    result, notes = _apply_metric_gate_consistency_guard(
        {
            "status": "REJECTED",
            "failed_gates": ["eligibility_drift_within_tolerance"],
            "hard_failures": [],
            "required_fixes": ["stale baseline said eligibility drift failed"],
        },
        qa_gates,
        facts,
    )
    assert result["status"] == "APPROVE_WITH_WARNINGS"
    assert result["failed_gates"] == []
    assert any("QA_METRIC_FACT_OVERRIDE" in note for note in notes)


def test_qa_reviewer_prompt_includes_gate_metric_facts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    metrics_dir = tmp_path / "artifacts" / "ml"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "latency_benchmark.json").write_text(
        '{"ms_per_1000_debtors": 20.0}',
        encoding="utf-8",
    )
    agent = QAReviewerAgent(api_key=None)
    agent.review_code(
        "print('model')",
        {"title": "ML audit"},
        "Audit ML outputs",
        evaluation_spec={
            "review_subject": "ml_engineer",
            "subject_required_outputs": [
                {"path": "artifacts/ml/latency_benchmark.json", "intent": "latency_benchmark"}
            ],
            "qa_gates": [
                {
                    "name": "inference_latency_within_spec",
                    "severity": "HARD",
                    "applies_to_artifact": "artifacts/ml/latency_benchmark.json",
                    "params": {"metric": "ms_per_1000_debtors", "max_value": 30},
                }
            ],
        },
    )
    prompt = agent.last_prompt or ""
    assert "gate_metric_facts" in prompt
    assert "current candidate artifact facts outrank baseline" in prompt
