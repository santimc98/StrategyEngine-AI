import json
import os

from src.agents.business_translator import (
    BusinessTranslatorAgent,
    _build_final_incumbent_state,
    _build_governance_contradiction_packet,
    _build_report_artifact_manifest,
    _build_translator_evidence_ledger,
    _ensure_evidence_section,
    _build_metric_progress_summary,
    _score_report_quality,
    _sanitize_review_board_verdict_for_translator,
    _summarize_scored_row_semantics,
    _summarize_ml_engineer_change_summary,
    _validate_report,
    _validate_report_structure,
)


class _EchoModel:
    def generate_content(self, prompt):
        class _Resp:
            def __init__(self, text):
                self.text = text
        return _Resp(prompt)


class _SequencedModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def generate_content(self, prompt):
        self.prompts.append(prompt)
        text = self.responses.pop(0) if self.responses else self.prompts[-1]

        class _Resp:
            def __init__(self, text):
                self.text = text

        return _Resp(text)


def test_translator_evidence_ledger_extracts_metric_and_gate_sources(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "artifacts" / "ml").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    with open(tmp_path / "artifacts" / "ml" / "evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump({"holdout": {"qwk": 0.7742043498751316}}, f)

    ledger = _build_translator_evidence_ledger(
        run_summary={"run_outcome": "GO_WITH_LIMITATIONS", "overall_status_global": "warning"},
        review_board_verdict={
            "final_review_verdict": "APPROVE_WITH_WARNINGS",
            "performance_threshold_gaps": [
                {
                    "name": "individual_ordinal_performance",
                    "metric": "holdout.accuracy_exact",
                    "value": 0.42145044956032013,
                    "operator": ">=",
                    "threshold": 0.65,
                    "status": "fail",
                }
            ],
        },
        output_contract_report={"missing": [], "qa_gate_results": {"checked": []}},
        metrics_payload={"holdout": {"qwk": 0.7742043498751316}},
        data_adequacy_report={"status": "ok", "reasons": []},
        manifest={"summary": {"required_present": 14, "required_total": 14, "required_missing": 0}},
        plot_summaries=None,
        final_incumbent_state={},
    )

    sources = [fact["source"] for fact in ledger["facts"]]
    assert "data/run_summary.json#/run_outcome" in sources
    assert any(source.endswith("#/holdout/qwk") for source in sources)
    assert any("performance_threshold_gaps/0" in source for source in sources)


def test_evidence_section_auto_grounds_missing_claim_from_ledger():
    ledger = {
        "facts": [
            {
                "fact_id": "metric_1",
                "claim": "Metric holdout.qwk is 0.774204.",
                "source": "artifacts/ml/evaluation_report.json#/holdout/qwk",
                "value": 0.774204,
            }
        ]
    }
    report = """# Reporte

## Decisión Ejecutiva

GO_WITH_LIMITATIONS.

## Riesgos

El resumen de holdout reporta qwk=0.7742.

## Evidencia usada

evidence:
{claim: "El resumen de holdout reporta qwk=0.7742.", source: "missing"}
"""

    normalized = _ensure_evidence_section(
        report,
        ["artifacts/ml/evaluation_report.json"],
        evidence_ledger=ledger,
    )

    assert 'source: "artifacts/ml/evaluation_report.json#/holdout/qwk"' in normalized


def test_validate_report_allows_missing_claim_when_ledger_can_ground_it():
    report = """## Decisión Ejecutiva

GO_WITH_LIMITATIONS

## Riesgos

El resumen de holdout reporta qwk=0.7742.

## Evidencia usada

evidence:
{claim: "El resumen de holdout reporta qwk=0.7742.", source: "missing"}
"""

    validation = _validate_report(
        content=report,
        expected_decision="GO_WITH_LIMITATIONS",
        facts_context=[],
        metrics_payload={},
        plots=[],
        expected_language="es",
        evidence_ledger={
            "facts": [
                {
                    "fact_id": "metric_1",
                    "claim": "Metric holdout.qwk is 0.774204.",
                    "source": "artifacts/ml/evaluation_report.json#/holdout/qwk",
                    "value": 0.774204,
                }
            ]
        },
    )

    assert "unsupported_evidence_claims" not in validation["critical_issues"]


class _CaptureCompletions:
    def __init__(self, captured):
        self.captured = captured

    def create(self, **kwargs):
        self.captured.update(kwargs)

        class _Message:
            content = "ok"

        class _Choice:
            message = _Message()
            finish_reason = "stop"

        class _Response:
            choices = [_Choice()]

        return _Response()


class _TranslatorRepairCaptureClient:
    def __init__(self, captured):
        self.chat = type("Chat", (), {"completions": _CaptureCompletions(captured)})()


def test_translator_repair_call_can_use_dedicated_model_slot():
    captured = {}
    agent = BusinessTranslatorAgent.__new__(BusinessTranslatorAgent)
    agent.client = _TranslatorRepairCaptureClient(captured)
    agent.model_name = "anthropic/claude-opus-4.6"
    agent.repair_model_name = "openai/gpt-5.4-mini"
    agent._max_tokens = 1024

    content = agent._call_llm("repair this", model_name=agent.repair_model_name)

    assert content == "ok"
    assert captured["model"] == "openai/gpt-5.4-mini"


def test_report_artifact_manifest_uses_authoritative_run_summary_governance(tmp_path):
    artifact_path = tmp_path / "predictions.csv"
    artifact_path.write_text("id,score\n1,0.8\n", encoding="utf-8")

    manifest = _build_report_artifact_manifest(
        artifact_index=[{"path": str(artifact_path), "artifact_type": "predictions"}],
        required_outputs=[str(artifact_path)],
        output_contract_report={"overall_status": "ok", "missing": []},
        review_verdict="APPROVED",
        gate_context={"failed_gates": [], "required_fixes": []},
        run_summary={
            "status": "NEEDS_IMPROVEMENT",
            "run_outcome": "GO_WITH_LIMITATIONS",
            "governance_reasons": ["qa_review=warning"],
            "hard_failures": ["candidate_rejected"],
        },
        run_id="run12345",
    )

    snapshot = manifest["governance_snapshot"]
    assert snapshot["review_verdict"] == "NEEDS_IMPROVEMENT"
    assert snapshot["review_verdict_source"] == "run_summary.status"
    assert snapshot["run_outcome"] == "NO_GO"
    assert snapshot["run_outcome_source"] == "derived_from_review_verdict"
    assert snapshot["governance_reasons"] == ["qa_review=warning"]
    assert snapshot["hard_failures"] == ["candidate_rejected"]


class _CaptureChatCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)

        class _Msg:
            content = '{"title":"Executive Report","blocks":[],"evidence":[]}'

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        return _Resp()


class _CaptureClient:
    def __init__(self):
        self.chat = type("Chat", (), {"completions": _CaptureChatCompletions()})()


def test_translator_self_check_instructions_present(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "insights.json"), "w", encoding="utf-8") as f:
        json.dump({}, f)

    agent = BusinessTranslatorAgent(api_key="dummy_key")
    agent.model = _EchoModel()
    report = agent.generate_report(
        {"execution_output": "OK", "business_objective": "Objetivo de prueba"}
    )
    assert "Slot Coverage" in report
    assert "reporting_policy" in report


def test_translator_prompt_declares_source_of_truth_and_authoritative_outcome(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "insights.json"), "w", encoding="utf-8") as f:
        json.dump({}, f)
    with open(os.path.join("data", "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"run_outcome": "NO_GO"}, f)

    agent = BusinessTranslatorAgent(api_key="dummy_key")
    agent.model = _EchoModel()
    report = agent.generate_report(
        {"execution_output": "OK", "business_objective": "Objetivo de prueba"}
    )

    assert "=== SOURCE OF TRUTH AND PRECEDENCE ===" in report
    assert "The authoritative executive outcome for this report is: NO_GO" in report


def test_translator_prompt_preserves_target_lineage_when_steward_and_contract_diverge(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "insights.json"), "w", encoding="utf-8") as f:
        json.dump({}, f)
    with open(os.path.join("data", "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"run_outcome": "GO_WITH_LIMITATIONS"}, f)
    with open(os.path.join("data", "steward_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": "## Target Variable Decision\n**Recommended primary target: `won_90d`**",
            },
            f,
        )
    with open(os.path.join("data", "dataset_semantics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "primary_target": "pipeline_amount_90d",
                "target_status": "questioned",
                "recommended_primary_target": "pipeline_amount_90d",
                "target_status_reason": "Zero inflation requires extra care.",
            },
            f,
        )
    with open(os.path.join("data", "execution_contract.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "task_semantics": {
                    "primary_target": "won_90d",
                }
            },
            f,
        )

    agent = BusinessTranslatorAgent(api_key="dummy_key")
    agent.model = _EchoModel()
    prompt = agent.generate_report(
        {"execution_output": "OK", "business_objective": "Rank leads by expected commercial value."}
    )

    assert '"target_lineage": {' in prompt
    assert '"preliminary_steward_target": "won_90d"' in prompt
    assert '"validated_steward_target": "pipeline_amount_90d"' in prompt
    assert '"final_contract_target": "won_90d"' in prompt
    assert '"summary_excerpt_scope": "preliminary_steward_assessment"' in prompt


def test_translator_openrouter_call_uses_configured_max_tokens(monkeypatch):
    monkeypatch.setenv("TRANSLATOR_MAX_TOKENS", "8000")
    agent = BusinessTranslatorAgent(api_key="dummy_key")
    client = _CaptureClient()
    agent.client = client

    payload = agent._call_llm("Return JSON only.")

    assert payload
    calls = client.chat.completions.calls
    assert calls
    assert calls[0]["max_tokens"] == 8000


def test_translator_prompt_keeps_layout_flexible_without_legacy_markdown_scaffold(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "insights.json"), "w", encoding="utf-8") as f:
        json.dump({}, f)
    with open(os.path.join("data", "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"run_outcome": "NO_GO"}, f)

    agent = BusinessTranslatorAgent(api_key="dummy_key")
    agent.model = _EchoModel()
    agent.generate_report(
        {"execution_output": "OK", "business_objective": "Objetivo de prueba"}
    )

    prompt = agent.last_prompt or ""
    assert "Make the authoritative executive decision and its rationale clear early." in prompt
    assert "LEGACY MARKDOWN GUIDANCE BELOW IS DEPRECATED" not in prompt
    assert "Executive decision with clear rationale (always first)" not in prompt


def test_translator_prompt_encourages_fact_inference_and_recommendation_distinction(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "insights.json"), "w", encoding="utf-8") as f:
        json.dump({}, f)
    with open(os.path.join("data", "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"run_outcome": "GO_WITH_LIMITATIONS"}, f)

    agent = BusinessTranslatorAgent(api_key="dummy_key")
    agent.model = _EchoModel()
    prompt = agent.generate_report(
        {"execution_output": "OK", "business_objective": "Prioritize invoices based on calibrated collection risk."}
    )

    assert "Distinguish clearly between supported facts, cautious inference, and recommended action." in prompt
    assert "If you recommend a timeline, threshold, governance gate, or rollout policy" in prompt
    assert "Do not present inferred rollout policies, exact remediation windows, or governance gates as established facts" in prompt


def test_translator_prompt_separates_final_incumbent_from_rejected_challenger(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    os.makedirs(os.path.join("artifacts", "ml"), exist_ok=True)
    os.makedirs(os.path.join("static", "plots"), exist_ok=True)

    with open(os.path.join("data", "insights.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics_summary": [
                    {"metric": "mean_mae", "value": 1489.6658896856218},
                    {"metric": "std_mae", "value": 451.32639733203274},
                ]
            },
            f,
        )
    with open(os.path.join("artifacts", "ml", "cv_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "mean_mae": 1410.2794030184425,
                "std_mae": 150.89153798429797,
                "model_performance": {
                    "mean_mae": 1410.2794030184425,
                    "std_mae": 150.89153798429797,
                },
            },
            f,
        )
    with open(os.path.join("data", "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"run_outcome": "GO_WITH_LIMITATIONS"}, f)

    state = {
        "execution_output": "OK",
        "business_objective": "Objetivo de prueba",
        "primary_metric_state": {
            "primary_metric_name": "MAE",
            "primary_metric_value": 1410.2794030184425,
        },
        "ml_improvement_round_history": [
            {
                "round_id": 1,
                "baseline_metric": 2088.3858989698074,
                "candidate_metric": 1564.31726834445,
                "kept": "improved",
                "hypothesis": {"label": "catboost"},
            },
            {
                "round_id": 2,
                "baseline_metric": 1564.31726834445,
                "candidate_metric": 1410.2794030184425,
                "kept": "improved",
                "hypothesis": {"label": "log_target"},
            },
            {
                "round_id": 3,
                "baseline_metric": 1410.2794030184425,
                "candidate_metric": 1489.6658896856218,
                "kept": "baseline",
                "hypothesis": {"label": "optuna_challenger"},
            },
        ],
        "plot_summaries": [
            {
                "filename": "cv_folds.png",
                "title": "Cross-validation fold performance",
                "facts": [
                    "5-fold CV MAE mean=1489.665890",
                    "5-fold CV MAE std=451.326397",
                ],
            }
        ],
    }

    agent = BusinessTranslatorAgent(api_key="dummy_key")
    agent.model = _EchoModel()
    prompt = agent.generate_report(state, plots=["static/plots/cv_folds.png"])

    assert "Metrics: No data available." not in prompt
    assert '"metric": "MAE", "value": 1410.2794030184425, "source": "canonical_primary_metric"' in prompt
    assert "Rejected challenger from round 3" in prompt
    assert "Metric Progress Summary:" in prompt
    assert '"baseline_start": 2088.3858989698074' in prompt
    assert '"improvement_history_scope": "historical_progress_only"' in prompt
    assert '"selected_incumbent_metric": 1410.2794030184425' in prompt
    assert '"final_incumbent":' not in prompt


def test_validate_report_flags_overconfident_operational_claims():
    report = """
## Executive Decision
GO_WITH_LIMITATIONS

Deployment is approved for a controlled pilot with a defined 30-day remediation roadmap.

## Risks
Residual risk remains in edge cases.

## Evidence Used
evidence:
{claim: "Confirmed artifact present: data/run_summary.json", source: "data/run_summary.json"}
- data/run_summary.json
""".strip()

    validation = _validate_report(
        content=report,
        expected_decision="GO_WITH_LIMITATIONS",
        facts_context=[],
        metrics_payload={},
        plots=[],
        expected_language="en",
    )

    assert validation["reasoning_warnings"]
    assert "30-day remediation roadmap" in validation["reasoning_warnings"][0]
    assert "overconfident_operational_claims" in validation["context_warnings"]
    assert _score_report_quality(validation) < 100


def test_metric_progress_summary_tracks_rejected_metric_gains_separately():
    summary = _build_metric_progress_summary(
        {
            "round_history": [
                {
                    "round_id": 1,
                    "baseline_metric": 8.679845364749,
                    "candidate_metric": 0.0005509950045063333,
                    "kept": "baseline",
                    "metric_improved": True,
                    "governance_approved": False,
                    "hypothesis": {"label": "piecewise_linear_case_model"},
                }
            ]
        },
        "mean_absolute_error",
        8.679845364749,
    )

    assert summary is not None
    assert summary["accepted_rounds"] == []
    assert summary["rejected_rounds"][0]["metric_improved"] is True
    assert summary["rejected_rounds"][0]["governance_approved"] is False
    assert summary["rejected_after_metric_improvement"][0]["candidate_metric"] == 0.0005509950045063333


def test_translator_prompt_prefers_run_summary_data_adequacy_over_stale_report(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "insights.json"), "w", encoding="utf-8") as f:
        json.dump({}, f)
    with open(os.path.join("data", "data_adequacy_report.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "status": "insufficient_signal",
                "reasons": ["pipeline_aborted_before_metrics"],
                "recommendations": ["Investigate metrics pipeline"],
                "signals": {"raw_status": "stale"},
            },
            f,
        )
    with open(os.path.join("data", "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_outcome": "GO_WITH_LIMITATIONS",
                "data_adequacy": {
                    "status": "ok",
                    "reasons": [],
                    "recommendations": [],
                    "quality_gates_alignment": {"status": "partial"},
                },
            },
            f,
        )

    agent = BusinessTranslatorAgent(api_key="dummy_key")
    agent.model = _EchoModel()
    prompt = agent.generate_report(
        {
            "execution_output": "OK",
            "business_objective": "Prioritize invoices based on calibrated collection risk.",
        }
    )

    assert '"status": "ok"' in prompt
    assert '"pipeline_aborted_before_metrics"' not in prompt

    with open(os.path.join("data", "report_visual_tables.json"), "r", encoding="utf-8") as f:
        tables = json.load(f)
    assert "Data Adequacy Status</td><td>ok</td>" in tables.get("kpi_snapshot_table_html", "")


def test_translator_prompt_preserves_metric_round_governance_flags(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "insights.json"), "w", encoding="utf-8") as f:
        json.dump({}, f)
    with open(os.path.join("data", "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"run_outcome": "GO_WITH_LIMITATIONS"}, f)

    agent = BusinessTranslatorAgent(api_key="dummy_key")
    agent.model = _EchoModel()
    prompt = agent.generate_report(
        {
            "execution_output": "OK",
            "business_objective": "Prioritize invoices based on calibrated collection risk.",
            "primary_metric_state": {
                "primary_metric_name": "ordinal_alignment_and_mae",
                "primary_metric_value": 0.5855711331482671,
            },
            "ml_improvement_round_history": [
                {
                    "round_id": 1,
                    "baseline_metric": 1.531444082519,
                    "candidate_metric": 0.5855711331482671,
                    "kept": "improved",
                    "hypothesis": {"label": "piecewise_monotonic_case_offset_calibration"},
                    "metric_improved": True,
                    "governance_approved": True,
                    "approved": True,
                },
                {
                    "round_id": 2,
                    "baseline_metric": 0.5855711331482671,
                    "candidate_metric": 58.660708782334915,
                    "kept": "baseline",
                    "hypothesis": {"label": "constrained_case_aware_weight_optimization"},
                    "metric_improved": False,
                    "governance_approved": False,
                    "approved": False,
                },
            ],
        }
    )

    assert '"metric_improved": true' in prompt
    assert '"governance_approved": true' in prompt
    assert '"governance_approved": false' in prompt


def test_sanitized_review_board_verdict_prefers_final_incumbent_summary():
    payload = _sanitize_review_board_verdict_for_translator(
        {
            "summary": "Recovered to PR-AUC 0.2373.",
            "candidate_assessment_status": "APPROVED",
            "metric_round_finalization": {
                "metric_name": "pr_auc",
                "kept": "baseline",
                "baseline_metric": 0.25671148099991203,
                "candidate_metric": 0.23731575773380917,
                "final_metric": 0.25671148099991203,
                "metric_improved": False,
                "governance_approved": True,
                "approved": True,
            },
            "deterministic_facts": {
                "metrics": {
                    "primary": {
                        "name": "pr_auc",
                        "value": 0.23731575773380917,
                    }
                }
            },
        },
        "pr_auc",
        0.25671148099991203,
    )

    assert payload["summary"].startswith("The challenger passed governance review but did not improve pr_auc")
    assert payload["deterministic_facts"]["metrics"]["primary"]["value"] == 0.25671148099991203
    assert payload["deterministic_facts"]["metrics"]["primary"]["candidate_value"] == 0.23731575773380917


def test_translator_kpi_snapshot_uses_canonical_final_metric_only(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    os.makedirs(os.path.join("artifacts", "ml"), exist_ok=True)

    with open(os.path.join("data", "insights.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics_summary": [
                    {"metric": "mae_all", "value": 0.05851280962074632},
                ]
            },
            f,
        )
    with open(os.path.join("artifacts", "ml", "cv_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "mae_all": 0.05851280962074632,
                "model_performance": {
                    "mae_all": 0.05851280962074632,
                },
            },
            f,
        )
    with open(os.path.join("data", "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"run_outcome": "GO_WITH_LIMITATIONS"}, f)

    agent = BusinessTranslatorAgent(api_key="dummy_key")
    agent.model = _EchoModel()
    agent.generate_report(
        {
            "execution_output": "OK",
            "business_objective": "Pricing objective in English",
            "primary_metric_state": {
                "primary_metric_name": "violation_reduction",
                "primary_metric_value": 3.0,
            },
            "data_adequacy_report": {"status": "sufficient_signal"},
        }
    )

    with open(os.path.join("data", "report_visual_tables.json"), "r", encoding="utf-8") as f:
        tables = json.load(f)
    kpi_html = tables.get("kpi_snapshot_table_html", "")
    prompt = agent.last_prompt or ""

    assert "metric:violation_reduction" in kpi_html
    assert "3" in kpi_html
    assert "mae_all" not in kpi_html
    assert '"metric": "violation_reduction", "value": 3.0, "source": "canonical_primary_metric"' in prompt


def test_translator_prompt_includes_cleaning_progress_summary(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "insights.json"), "w", encoding="utf-8") as f:
        json.dump({}, f)
    with open(os.path.join("data", "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"run_outcome": "GO_WITH_LIMITATIONS"}, f)
    with open(os.path.join("data", "cleaning_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "output_dialect": {"sep": ";", "decimal": ",", "encoding": "utf-8"},
                "row_counts": {"input": 352, "output": 352},
                "conversions": [
                    "Parsed 1stYearAmount from currency-like strings to float64",
                    "Parsed Debtors from numeric-like strings to Float64",
                    "Trimmed whitespace on required categorical fields",
                ],
                "cleaning_gates_status": {
                    "target_is_numeric": "PASSED",
                    "routing_column_valid": "PASSED",
                },
            },
            f,
        )

    agent = BusinessTranslatorAgent(api_key="dummy_key")
    agent.model = _EchoModel()
    prompt = agent.generate_report(
        {
            "execution_output": "OK",
            "business_objective": "Objetivo de prueba",
            "selected_strategy": {
                "title": "Modelo prescriptivo",
                "hypothesis": "La limpieza debe rescatar variables monetarias y de perfil para soportar el modelado.",
            },
        }
    )

    assert "Cleaning Progress Summary:" in prompt
    assert '"rows_before": 352' in prompt
    assert '"rows_after": 352' in prompt
    assert "Parsed 1stYearAmount from currency-like strings to float64" in prompt
    assert '"passed_gates": ["target_is_numeric", "routing_column_valid"]' in prompt


def test_translator_prompt_includes_deterministic_eda_fact_pack(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "insights.json"), "w", encoding="utf-8") as f:
        json.dump({}, f)
    with open(os.path.join("data", "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"run_outcome": "GO_WITH_LIMITATIONS"}, f)
    with open(os.path.join("data", "cleaning_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "row_counts": {"input": 20096, "output": 20096},
                "conversions": [
                    "Parsed Score to numeric float",
                    "Parsed Importe to numeric float",
                    "Validated Score FEC to Rango FEC using observed mapping family with mismatch diagnostics",
                ],
                "cleaning_gates_status": {
                    "numeric_parsing_integrity": "PASSED",
                    "score_fec_mapping_consistency": "WARNING_481_mismatches",
                },
            },
            f,
        )
    with open(os.path.join("data", "data_profile.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "dtypes": {
                    "FechaObs": "datetime64",
                    "Score": "float64",
                    "Sector": "object",
                    "CodPartidaAbierta": "object",
                },
                "missingness_top30": {
                    "Sector": 0.2828,
                    "TipoSegmento": 0.2581,
                    "Score": 0.0,
                },
                "constant_columns": ["FechaObs"],
                "high_cardinality_columns": [
                    {"column": "CodPartidaAbierta", "n_unique": 20077, "unique_ratio": 0.9991}
                ],
                "leakage_flags": [
                    {"column": "Score FEC", "reason": "name_contains_outcome:score", "severity": "SOFT"}
                ],
            },
            f,
        )
    with open(os.path.join("data", "dataset_semantics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "primary_target": "Score",
                "split_candidates": ["FE", "Sector"],
                "notes": [
                    "Sector and TipoSegmento have material missingness and require explicit handling.",
                    "FechaObs is constant and should be excluded from modeling.",
                ],
            },
            f,
        )

    agent = BusinessTranslatorAgent(api_key="dummy_key")
    agent.model = _EchoModel()
    prompt = agent.generate_report(
        {
            "execution_output": "OK",
            "business_objective": "Prioritize invoices by calibrated collection score.",
        }
    )

    assert "EDA Fact Pack:" in prompt
    assert '"top_missing_columns": [{"column": "Sector", "missing_frac": 0.2828}' in prompt
    assert '"high_cardinality_columns": [{"column": "CodPartidaAbierta", "n_unique": 20077, "unique_ratio": 0.9991}]' in prompt
    assert '"quality_flags": ["score_fec_mapping_consistency=WARNING_481_mismatches", "leakage_flag:Score FEC: name_contains_outcome:score [SOFT]", "constant_columns=FechaObs"]' in prompt
    assert '"semantic_notes": ["Sector and TipoSegmento have material missingness and require explicit handling.", "FechaObs is constant and should be excluded from modeling."]' in prompt

    with open(os.path.join("data", "eda_fact_pack.json"), "r", encoding="utf-8") as f:
        eda_fact_pack = json.load(f)
    assert eda_fact_pack["row_retention"]["rows_after"] == 20096
    assert eda_fact_pack["numeric_profile"]["numeric_columns"] == 1


def test_translator_enriches_generic_eda_plot_summaries_from_fact_pack(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "insights.json"), "w", encoding="utf-8") as f:
        json.dump({}, f)
    with open(os.path.join("data", "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"run_outcome": "GO_WITH_LIMITATIONS"}, f)
    with open(os.path.join("data", "cleaning_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"row_counts": {"input": 100, "output": 100}}, f)
    with open(os.path.join("data", "data_profile.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "dtypes": {
                    "FechaObs": "datetime64",
                    "Score": "float64",
                    "Sector": "object",
                    "Importe": "float64",
                },
                "missingness_top30": {"Sector": 0.2828, "TipoSegmento": 0.2581},
                "constant_columns": ["FechaObs"],
                "high_cardinality_columns": [
                    {"column": "CodPartidaAbierta", "n_unique": 999, "unique_ratio": 0.999}
                ],
            },
            f,
        )
    with open(os.path.join("data", "dataset_semantics.json"), "w", encoding="utf-8") as f:
        json.dump({"primary_target": "Score"}, f)

    agent = BusinessTranslatorAgent(api_key="dummy_key")
    agent.model = _EchoModel()
    prompt = agent.generate_report(
        {
            "execution_output": "OK",
            "business_objective": "Prioritize invoices by calibrated collection score.",
        },
        plots=[
            "static/plots/missing_values.png",
            "static/plots/numeric_distributions.png",
        ],
    )

    assert "Missing values overview" in prompt
    assert "Sector missing=28.3%" in prompt
    assert "Numeric distributions overview" in prompt
    assert "numeric_columns=2" in prompt
    assert "constant_columns=1 (FechaObs)" in prompt


def test_translator_loads_eda_fact_inputs_from_work_data_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs(os.path.join("work", "data"), exist_ok=True)
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "insights.json"), "w", encoding="utf-8") as f:
        json.dump({}, f)
    with open(os.path.join("data", "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"run_outcome": "GO_WITH_LIMITATIONS"}, f)
    with open(os.path.join("work", "data", "data_profile.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "dtypes": {"Score": "float64", "Sector": "object"},
                "missingness_top30": {"Sector": 0.31},
            },
            f,
        )
    with open(os.path.join("work", "data", "dataset_semantics.json"), "w", encoding="utf-8") as f:
        json.dump({"primary_target": "Score", "notes": ["Sector requires explicit missing handling."]}, f)

    agent = BusinessTranslatorAgent(api_key="dummy_key")
    agent.model = _EchoModel()
    prompt = agent.generate_report(
        {
            "execution_output": "OK",
            "business_objective": "Prioritize invoices by calibrated collection score.",
        },
        plots=["static/plots/missing_values.png"],
    )

    assert "EDA Fact Pack:" in prompt
    assert '"top_missing_columns": [{"column": "Sector", "missing_frac": 0.31}]' in prompt
    assert '"primary_target": "Score"' in prompt


def test_translator_prompt_includes_engineering_change_summaries(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "insights.json"), "w", encoding="utf-8") as f:
        json.dump({}, f)
    with open(os.path.join("data", "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"run_outcome": "GO_WITH_LIMITATIONS"}, f)
    with open(os.path.join("data", "review_board_verdict.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": "The final challenger was rejected because it violated case coverage constraints.",
                "required_actions": [
                    "Restore full case coverage before rollout.",
                    "Keep the approved incumbent until governance blockers are resolved.",
                ],
            },
            f,
        )
    with open(os.path.join("data", "cleaning_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "row_counts": {"input": 352, "output": 352},
                "conversions": [
                    "Parsed 1stYearAmount from currency-like strings to float64",
                    "Parsed Debtors from numeric-like strings to Float64",
                ],
                "cleaning_gates_status": {"target_is_numeric": "PASSED"},
                "contract_conflicts_resolved": ["Relaxed a strict mapping gate using observed valid families."],
                "notes": ["No deduplication was performed because identity rules were absent."],
            },
            f,
        )
    with open(os.path.join("data", "data_profile.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "dtypes": {"FechaObs": "datetime64", "Debtors": "float64", "Sector": "object"},
                "missingness_top30": {"Sector": 0.22},
                "constant_columns": ["FechaObs"],
            },
            f,
        )
    with open(os.path.join("data", "dataset_semantics.json"), "w", encoding="utf-8") as f:
        json.dump({"primary_target": "1stYearAmount"}, f)

    agent = BusinessTranslatorAgent(api_key="dummy_key")
    agent.model = _EchoModel()
    prompt = agent.generate_report(
        {
            "execution_output": "OK",
            "business_objective": "Build a pricing recommendation model.",
            "primary_metric_state": {
                "primary_metric_name": "MAE",
                "primary_metric_value": 1410.2794030184425,
            },
            "ml_improvement_round_history": [
                {
                    "round_id": 1,
                    "baseline_metric": 2088.3858989698074,
                    "candidate_metric": 1564.31726834445,
                    "kept": "improved",
                    "hypothesis": {"label": "catboost"},
                    "metric_improved": True,
                    "governance_approved": True,
                },
                {
                    "round_id": 2,
                    "baseline_metric": 1564.31726834445,
                    "candidate_metric": 1489.6658896856218,
                    "kept": "baseline",
                    "hypothesis": {"label": "optuna_challenger"},
                    "metric_improved": True,
                    "governance_approved": False,
                },
            ],
        }
    )

    assert "Data Engineer Change Summary:" in prompt
    assert "ML Engineer Change Summary:" in prompt
    assert "Run Causal Impact Summary:" in prompt
    assert "ENGINEERING IMPACT" in prompt
    assert "Business Objective Summary:" in prompt
    assert "Do not mention agents as workflow theater." in prompt
    assert '"accepted_interventions": ["Parsed 1stYearAmount from currency-like strings to float64", "Parsed Debtors from numeric-like strings to Float64"]' in prompt
    assert '"accepted_improvements": [{"round_id": 1, "hypothesis_label": "catboost"' in prompt
    assert '"rejected_after_metric_improvement": [{"round_id": 2, "hypothesis_label": "optuna_challenger"' in prompt
    assert "numerically improved challenger(s) were rejected by governance" in prompt

    with open(os.path.join("data", "data_engineer_change_summary.json"), "r", encoding="utf-8") as f:
        data_summary = json.load(f)
    with open(os.path.join("data", "ml_engineer_change_summary.json"), "r", encoding="utf-8") as f:
        ml_summary = json.load(f)
    with open(os.path.join("data", "run_causal_impact_summary.json"), "r", encoding="utf-8") as f:
        causal_summary = json.load(f)

    assert data_summary["gates_cleared"] == ["target_is_numeric"]
    assert ml_summary["current_incumbent_basis"] == "last_accepted_improvement"
    assert causal_summary["executive_decision_label"] == "GO_WITH_LIMITATIONS"


def test_translator_recovers_ml_history_from_persisted_metric_loop_state(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "insights.json"), "w", encoding="utf-8") as f:
        json.dump({}, f)
    with open(os.path.join("data", "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"run_outcome": "GO_WITH_LIMITATIONS"}, f)
    with open(os.path.join("data", "metric_loop_state.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "schema_version": "v1",
                "round": {"round_id": 2, "rounds_allowed": 3, "no_improve_streak": 1, "patience": 2},
                "incumbent": {"metric_value": 0.585571447601806},
                "best_observed": {"metric_value": 0.585571447601806, "label": "candidate"},
                "round_history": [
                    {
                        "round_id": 1,
                        "baseline_metric": 1.5314284471774324,
                        "candidate_metric": 0.585571447601806,
                        "kept": "improved",
                        "metric_improved": True,
                        "governance_approved": True,
                        "hypothesis": {"technique": "static_reference_buckets", "label": "static_reference_buckets"},
                    },
                    {
                        "round_id": 2,
                        "baseline_metric": 0.585571447601806,
                        "candidate_metric": 55.81055555555555,
                        "kept": "baseline",
                        "metric_improved": False,
                        "governance_approved": False,
                        "hypothesis": {"technique": "aggressive_bucket_override", "label": "aggressive_bucket_override"},
                    },
                ],
            },
            f,
        )

    agent = BusinessTranslatorAgent(api_key="dummy_key")
    agent.model = _EchoModel()
    prompt = agent.generate_report(
        {
            "execution_output": "OK",
            "business_objective": "Improve calibrated ranking quality while preserving business ordering constraints.",
            "primary_metric_state": {
                "primary_metric_name": "Mean absolute deviation from reference score",
                "primary_metric_value": 0.585571447601806,
            },
        }
    )

    assert '"accepted_improvements": [{"round_id": 1, "hypothesis_label": "static_reference_buckets"' in prompt
    assert '"rejected_experiments": [{"round_id": 2, "hypothesis_label": "aggressive_bucket_override"' in prompt
    assert '"rounds_attempted": 2' in prompt


def test_ml_engineer_change_summary_falls_back_to_approved_baseline():
    final_incumbent_state = _build_final_incumbent_state(
        executive_decision_label="GO_WITH_LIMITATIONS",
        run_outcome_token="GO_WITH_LIMITATIONS",
        review_board_verdict={
            "deterministic_facts": {
                "runtime": {"status": "OK"},
                "output_contract": {"overall_status": "ok", "missing_required_artifacts": []},
                "pipeline": {"review_verdict_before_board": "APPROVE_WITH_WARNINGS"},
            }
        },
        output_contract_report={"overall_status": "ok", "missing": []},
        slot_payloads={
            "predictions_overview": {
                "row_count": 1933,
                "columns": ["account_id", "snapshot_month_end", "churn_risk_score"],
            }
        },
        canonical_metric_name="top_decile_lift",
        canonical_metric_value=9.933567924760364,
        run_summary={"run_outcome": "GO_WITH_LIMITATIONS", "data_adequacy": {"status": "warning"}},
    )

    summary = _summarize_ml_engineer_change_summary(
        metric_loop_context={},
        canonical_metric_name="top_decile_lift",
        canonical_metric_value=9.933567924760364,
        review_board_verdict={"summary": "Final incumbent accepted with warnings."},
        final_incumbent_state=final_incumbent_state,
    )

    assert summary is not None
    assert summary["current_incumbent_basis"] == "approved_baseline"
    assert summary["baseline_incumbent"]["predictions_rows"] == 1933
    assert summary["selected_incumbent_metric"] == 9.933567924760364


def test_translator_prompt_includes_final_incumbent_and_governance_contradiction_packets(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    os.makedirs(os.path.join("artifacts", "ml"), exist_ok=True)
    with open(os.path.join("data", "insights.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "slot_payloads": {
                    "predictions_overview": {
                        "row_count": 1933,
                        "columns": ["account_id", "snapshot_month_end", "churn_risk_score"],
                    }
                }
            },
            f,
        )
    with open(os.path.join("data", "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_outcome": "GO_WITH_LIMITATIONS",
                "data_adequacy": {
                    "status": "insufficient_signal",
                    "reasons": ["pipeline_aborted_before_metrics"],
                    "recommendations": [],
                },
            },
            f,
        )
    with open(os.path.join("data", "output_contract_report.json"), "w", encoding="utf-8") as f:
        json.dump({"overall_status": "ok", "missing": []}, f)
    with open(os.path.join("data", "review_board_verdict.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": "Missing operational artifacts and scoring CSV despite strong metric performance.",
                "deterministic_facts": {
                    "runtime": {"status": "OK"},
                    "output_contract": {"overall_status": "ok", "missing_required_artifacts": []},
                    "pipeline": {"review_verdict_before_board": "APPROVE_WITH_WARNINGS"},
                },
            },
            f,
        )
    with open(os.path.join("data", "ml_review_stack.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "result_evaluator": {
                    "feedback": "The pipeline was aborted before metrics and is failing to produce the required operational artifacts (Scoring CSV)."
                }
            },
            f,
        )
    with open(os.path.join("artifacts", "ml", "cv_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"top_decile_lift": 9.933567924760364}, f)

    agent = BusinessTranslatorAgent(api_key="dummy_key")
    agent.model = _EchoModel()
    prompt = agent.generate_report(
        {
            "execution_output": "OK",
            "business_objective": "Priorizar churn B2B.",
            "primary_metric_state": {
                "primary_metric_name": "top_decile_lift",
                "primary_metric_value": 9.933567924760364,
            },
        }
    )

    assert '"final_incumbent_state": {' in prompt
    assert '"governance_contradiction_packet": {' in prompt
    assert '"has_contradictions": true' in prompt
    assert '"current_incumbent_basis": "approved_baseline"' in prompt
    packet_path = os.path.join("data", "governance_contradiction_packet.json")
    assert os.path.getsize(packet_path) > 2
    with open(packet_path, "r", encoding="utf-8") as f:
        persisted_packet = json.load(f)
    assert persisted_packet["has_contradictions"] is True


def test_validate_report_flags_contradicted_current_state_claims():
    validation = _validate_report(
        content="""
# Decisión Ejecutiva

GO_WITH_LIMITATIONS

El pipeline fue interrumpido antes de completar la generación de métricas y no pudo generar algunos artefactos de salida.

## Evidencia
- [source: data/run_summary.json -> run_outcome]
""",
        expected_decision="GO_WITH_LIMITATIONS",
        facts_context=[],
        metrics_payload={},
        plots=[],
        expected_language="es",
        governance_contradiction_packet={
            "contradictions": [
                {"id": "pipeline_aborted_before_metrics_stale"},
                {"id": "missing_required_outputs_stale"},
            ]
        },
    )

    assert "contradicted_current_state_claims" in validation["critical_issues"]
    assert validation["contradicted_current_state_claims"]
    assert _score_report_quality(validation) < 90


def test_translator_structure_validation_accepts_risk_semantics_without_heading():
    report = """
# Reporte Ejecutivo

NO_GO por falta de confianza en la evidencia disponible.

## Hallazgos Clave

La señal principal es insuficiente para producción y mantiene un riesgo operativo alto, por lo que requiere una nueva iteración controlada.

## Evidencia usada

evidence:
{claim: "La conclusión procede del resumen de la run", source: "data/run_summary.json -> run_outcome"}
"""

    issues = _validate_report_structure(report, expected_language="es")

    assert "missing_decision_section" not in issues
    assert "missing_risks_section" not in issues


def test_translator_scored_row_semantics_detects_inverse_risk_percentile():
    rows = [
        {"churn_probability": "0.99", "risk_percentile": "1.0", "priority_tier": "P1"},
        {"churn_probability": "0.80", "risk_percentile": "5.0", "priority_tier": "P1"},
        {"churn_probability": "0.20", "risk_percentile": "75.0", "priority_tier": "P3"},
        {"churn_probability": "0.01", "risk_percentile": "99.0", "priority_tier": "P4"},
    ]

    semantics = _summarize_scored_row_semantics(
        rows,
        ["churn_probability", "risk_percentile", "priority_tier"],
        ".",
    )

    assert semantics["direction"] == "lower_rank_value_means_higher_risk"
    assert semantics["recommended_sort"][0] == {"column": "churn_probability", "order": "descending"}
    assert semantics["recommended_sort"][1] == {"column": "risk_percentile", "order": "ascending"}
    assert "do not describe higher risk_percentile as higher risk" in semantics["guidance"]


def test_translator_structure_validation_flags_language_mix_when_english_expected():
    report = """
# Executive Report

## Decisión Ejecutiva

The final decision is GO_WITH_LIMITATIONS because evidence remains partial.

## Riesgos

Residual data-quality blockers still require supervised rollout.

## Evidence Used

evidence:
{claim: "The conclusion comes from the run summary", source: "data/run_summary.json -> run_outcome"}
"""

    issues = _validate_report_structure(report, expected_language="en")

    assert "possible_language_mix" in issues


def test_translator_quality_score_penalizes_decision_discrepancy_context():
    score = _score_report_quality(
        {
            "structure_issues": [],
            "decision_issue": [],
            "unverified_metrics": [],
            "unsupported_evidence_claims": [],
            "invalid_plots": [],
            "context_warnings": ["decision_discrepancy_authoritative_vs_derived"],
            "decision_discrepancy": {
                "authoritative_decision": "NO_GO",
                "derived_decision": "GO_WITH_LIMITATIONS",
                "run_outcome": "NO_GO",
            },
        }
    )

    assert score < 100


def test_translator_fallback_does_not_claim_missing_declared_artifacts_as_present(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)

    with open(os.path.join("data", "insights.json"), "w", encoding="utf-8") as f:
        json.dump({}, f)
    with open(os.path.join("data", "execution_contract.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "required_outputs": [
                    {"path": "artifacts/clean/dataset_cleaned.csv", "owner": "data_engineer", "required": True},
                    {"path": "artifacts/reports/quality_audit_report.json", "owner": "data_engineer", "required": True},
                ]
            },
            f,
        )
    with open(os.path.join("data", "output_contract_report.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "overall_status": "error",
                "present": [],
                "missing": [
                    "artifacts/clean/dataset_cleaned.csv",
                    "artifacts/reports/quality_audit_report.json",
                ],
            },
            f,
        )
    with open(os.path.join("data", "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"run_outcome": "NO_GO"}, f)
    with open(os.path.join("data", "produced_artifact_index.json"), "w", encoding="utf-8") as f:
        json.dump([], f)

    agent = BusinessTranslatorAgent(api_key="dummy_key")
    agent.model = _EchoModel()
    report = agent.generate_report(
        {"execution_output": "DE failed before producing artifacts", "business_objective": "Objetivo de prueba"}
    )

    assert "Confirmed artifact present: artifacts/clean/dataset_cleaned.csv" not in report
    assert 'source: "artifacts/clean/dataset_cleaned.csv"' not in report
    assert "- missing" in report


def test_translator_prompt_loads_cleaning_manifest_from_work_artifact_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    work_dir = tmp_path / "work"
    (work_dir / "artifacts" / "clean").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    with open(tmp_path / "data" / "insights.json", "w", encoding="utf-8") as f:
        json.dump({}, f)
    with open(tmp_path / "data" / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump({"run_outcome": "GO_WITH_LIMITATIONS"}, f)
    with open(work_dir / "artifacts" / "clean" / "cleaning_manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "row_counts": {"input": 120, "output": 118},
                "conversions": ["Parsed Amount to float", "Normalized urgency buckets"],
                "cleaning_gates_status": {"target_is_numeric": "PASSED"},
                "output_dialect": {"sep": ";", "decimal": ","},
            },
            f,
        )

    agent = BusinessTranslatorAgent(api_key="dummy_key")
    agent.model = _EchoModel()
    prompt = agent.generate_report(
        {
            "execution_output": "OK",
            "business_objective": "Prioritize invoices based on calibrated collection risk.",
            "work_dir": str(work_dir),
        }
    )

    assert "Cleaning Progress Summary:" in prompt
    assert '"rows_before": 120' in prompt
    assert '"rows_after": 118' in prompt
    assert "Parsed Amount to float" in prompt


def test_translator_structured_repair_stays_in_json_mode_and_keeps_english_headings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TRANSLATOR_TWO_PASS_ENABLED", "0")
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "static" / "plots").mkdir(parents=True, exist_ok=True)
    with open(tmp_path / "data" / "insights.json", "w", encoding="utf-8") as f:
        json.dump({}, f)
    with open(tmp_path / "data" / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump({"run_outcome": "GO_WITH_LIMITATIONS"}, f)
    with open(tmp_path / "static" / "plots" / "plot_summaries.json", "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "filename": "cv_folds.png",
                    "title": "Cross-validation stability",
                    "facts": ["mean_mae: 1410.28", "std_mae: 150.89"],
                }
            ],
            f,
        )

    model = _SequencedModel(
        [
            "# Executive Report\n\n## Decisión Ejecutiva\n\nGO_WITH_LIMITATIONS\n",
            json.dumps(
                {
                    "title": "Executive Report: Pricing Run",
                    "blocks": [
                        {"type": "heading", "level": 1, "text": "Executive Report: Pricing Run"},
                        {"type": "heading", "level": 2, "text": "Executive Decision"},
                        {
                            "type": "paragraph",
                            "text": "The final outcome is GO_WITH_LIMITATIONS because the incumbent is usable but still requires controlled rollout.",
                        },
                        {
                            "type": "artifact",
                            "artifact_key": "chart_1",
                            "lead_in": "The stability chart supports the final risk posture.",
                            "analysis": ["Fold dispersion is moderate rather than destabilizing.", "That supports a limited rollout instead of a full block."],
                        },
                        {"type": "heading", "level": 2, "text": "Risks"},
                        {"type": "bullet_list", "items": ["Further monitoring is required before broader deployment."]},
                    ],
                    "evidence": [{"claim": "The final decision comes from the run outcome", "source": "data/run_summary.json -> run_outcome"}],
                },
                ensure_ascii=False,
            ),
        ]
    )
    agent = BusinessTranslatorAgent(api_key="dummy_key")
    agent.model = model
    report = agent.generate_report(
        {
            "execution_output": "OK",
            "business_objective": "Optimize invoice pricing recommendations for margin and win rate.",
        },
        plots=["static/plots/cv_folds.png"],
    )

    assert isinstance(agent.last_report_blocks, list)
    assert "## Evidence Used" in report
    assert "## Evidencia usada" not in report
    assert "![Cross-validation stability](static/plots/cv_folds.png)" in report
    assert any("Return ONLY valid JSON" in prompt for prompt in model.prompts[1:])
