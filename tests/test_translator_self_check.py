import json
import os

from src.agents.business_translator import (
    BusinessTranslatorAgent,
    _score_report_quality,
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
