"""
Tests for output_contract_report compliance checking.

Tests that output_contract_report includes:
  - Backward compatible keys (present, missing, missing_optional, summary)
  - New artifact_requirements_report with schema validation
  - overall_status derivation based on file presence and schema compliance
"""
import json
import os
import tempfile
import pytest
from src.utils.output_contract import (
    check_artifact_requirements,
    check_scored_rows_schema,
    build_output_contract_report,
)


class TestCheckScoredRowsSchema:
    """Test scored_rows.csv schema validation."""

    def test_all_columns_present(self, tmp_path):
        """All required columns present -> no missing columns."""
        # Create scored_rows.csv with required columns
        csv_path = tmp_path / "scored_rows.csv"
        csv_path.write_text("Id,prediction,explanation\n1,0.5,test\n")

        result = check_scored_rows_schema(
            str(csv_path),
            required_columns=["Id", "prediction", "explanation"],
        )

        assert result["exists"] is True
        assert result["missing_columns"] == []
        assert set(result["present_columns"]) == {"Id", "prediction", "explanation"}

    def test_missing_column(self, tmp_path):
        """Missing required column -> appears in missing_columns."""
        csv_path = tmp_path / "scored_rows.csv"
        csv_path.write_text("Id,prediction\n1,0.5\n")

        result = check_scored_rows_schema(
            str(csv_path),
            required_columns=["Id", "prediction", "explanation"],
        )

        assert result["exists"] is True
        assert "explanation" in result["missing_columns"]
        assert "Id" in result["present_columns"]
        assert "prediction" in result["present_columns"]

    def test_case_insensitive(self, tmp_path):
        """Column matching is case-insensitive."""
        csv_path = tmp_path / "scored_rows.csv"
        csv_path.write_text("ID,PREDICTION\n1,0.5\n")

        result = check_scored_rows_schema(
            str(csv_path),
            required_columns=["Id", "prediction"],
        )

        assert result["missing_columns"] == []

    def test_file_not_found(self, tmp_path):
        """Missing file -> exists=False."""
        result = check_scored_rows_schema(
            str(tmp_path / "nonexistent.csv"),
            required_columns=["Id"],
        )

        assert result["exists"] is False
        assert result["missing_columns"] == ["Id"]

    def test_dialect_semicolon_separator(self, tmp_path):
        """CSV with semicolon separator parsed correctly."""
        csv_path = tmp_path / "scored_rows.csv"
        csv_path.write_text("Id;prediction;explanation\n1;0.5;test\n")

        result = check_scored_rows_schema(
            str(csv_path),
            required_columns=["Id", "prediction", "explanation"],
            dialect={"sep": ";", "encoding": "utf-8"},
        )

        assert result["exists"] is True
        assert result["missing_columns"] == []


class TestCheckArtifactRequirements:
    """Test check_artifact_requirements function."""

    def test_all_files_present_schema_ok(self, tmp_path):
        """All files present and schema valid -> status=ok."""
        # Create required files
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "metrics.json").write_text('{"accuracy": 0.9}')
        (data_dir / "scored_rows.csv").write_text("Id,prediction\n1,0.5\n")
        (data_dir / "cleaning_manifest.json").write_text('{"output_dialect": {"sep": ",", "encoding": "utf-8"}}')

        artifact_reqs = {
            "required_files": [
                {"path": "data/metrics.json"},
                {"path": "data/scored_rows.csv"},
            ],
            "scored_rows_schema": {
                "required_columns": ["Id", "prediction"],
            },
        }

        result = check_artifact_requirements(artifact_reqs, work_dir=str(tmp_path))

        assert result["status"] == "ok"
        assert result["files_report"]["missing"] == []
        assert result["scored_rows_report"]["missing_columns"] == []

    def test_missing_file_status_error(self, tmp_path):
        """Missing required file -> status=error."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        # Only create one file
        (data_dir / "metrics.json").write_text('{"accuracy": 0.9}')

        artifact_reqs = {
            "required_files": [
                {"path": "data/metrics.json"},
                {"path": "data/scored_rows.csv"},  # This doesn't exist
            ],
        }

        result = check_artifact_requirements(artifact_reqs, work_dir=str(tmp_path))

        assert result["status"] == "error"
        assert any("scored_rows" in m for m in result["files_report"]["missing"])

    def test_missing_column_status_error(self, tmp_path):
        """Missing required column in scored_rows -> status=error."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "scored_rows.csv").write_text("Id,prediction\n1,0.5\n")
        (data_dir / "cleaning_manifest.json").write_text('{"output_dialect": {"sep": ",", "encoding": "utf-8"}}')

        artifact_reqs = {
            "required_files": [
                {"path": "data/scored_rows.csv"},
            ],
            "scored_rows_schema": {
                "required_columns": ["Id", "prediction", "explanation"],
            },
        }

        result = check_artifact_requirements(artifact_reqs, work_dir=str(tmp_path))

        assert result["status"] == "error"
        assert "explanation" in result["scored_rows_report"]["missing_columns"]

    def test_row_count_mismatch_status_error(self, tmp_path):
        """Row count mismatch in file_schemas.expected_row_count -> status=error."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "submission.csv").write_text("id,score\n1,0.5\n2,0.6\n3,0.7\n")

        artifact_reqs = {
            "required_files": [{"path": "data/submission.csv"}],
            "file_schemas": {
                "data/submission.csv": {"expected_row_count": 2},
            },
        }

        result = check_artifact_requirements(artifact_reqs, work_dir=str(tmp_path))

        assert result["status"] == "error"
        assert len(result.get("row_count_report", {}).get("mismatches", [])) == 1


class TestBuildOutputContractReport:
    """Test build_output_contract_report unified helper."""

    def test_backward_compatible_keys(self, tmp_path):
        """Report includes backward compatible keys."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "metrics.json").write_text('{}')

        contract = {
            "required_outputs": ["data/metrics.json"],
            "artifact_requirements": {
                "required_files": [{"path": "data/metrics.json"}],
            },
        }

        # Change to temp directory for the test
        original_cwd = os.getcwd()
        os.chdir(str(tmp_path))
        try:
            report = build_output_contract_report(contract, work_dir=".")
        finally:
            os.chdir(original_cwd)

        # Backward compatible keys
        assert "present" in report
        assert "missing" in report
        assert "missing_optional" in report
        assert "summary" in report
        # New keys
        assert "artifact_requirements_report" in report
        assert "overall_status" in report

    def test_overall_status_ok(self, tmp_path):
        """All requirements met -> overall_status=ok."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "metrics.json").write_text('{}')
        (data_dir / "scored_rows.csv").write_text("Id,prediction\n1,0.5\n")
        (data_dir / "cleaning_manifest.json").write_text('{"output_dialect": {}}')

        contract = {
            "required_outputs": ["data/metrics.json", "data/scored_rows.csv"],
            "artifact_requirements": {
                "required_files": [
                    {"path": "data/metrics.json"},
                    {"path": "data/scored_rows.csv"},
                ],
                "scored_rows_schema": {
                    "required_columns": ["Id", "prediction"],
                },
            },
        }

        original_cwd = os.getcwd()
        os.chdir(str(tmp_path))
        try:
            report = build_output_contract_report(contract, work_dir=".")
        finally:
            os.chdir(original_cwd)

        assert report["overall_status"] == "ok"
        assert report["artifact_requirements_report"]["status"] == "ok"

    def test_hard_numeric_qa_gate_failure_marks_output_contract_error(self, tmp_path):
        """Artifact-backed HARD numeric gates are contract compliance, not advisory text."""
        metrics_dir = tmp_path / "artifacts" / "ml"
        metrics_dir.mkdir(parents=True)
        (metrics_dir / "latency_benchmark.json").write_text(
            '{"ms_per_1000_debtors": 135.54}',
            encoding="utf-8",
        )

        contract = {
            "required_outputs": ["artifacts/ml/latency_benchmark.json"],
            "ml_engineer": {
                "required_outputs": [
                    {
                        "path": "artifacts/ml/latency_benchmark.json",
                        "intent": "latency_benchmark",
                    }
                ],
                "qa_gates": [
                    {
                        "name": "inference_latency_within_spec",
                        "severity": "HARD",
                        "applies_to_artifact": "artifacts/ml/latency_benchmark.json",
                        "evidence_source": "artifacts/ml/latency_benchmark.json.ms_per_1000_debtors",
                        "params": {
                            "metric": "ms_per_1000_debtors",
                            "min_value": 10,
                            "max_value": 30,
                        },
                    }
                ],
            },
            "artifact_requirements": {
                "required_files": [{"path": "artifacts/ml/latency_benchmark.json"}],
            },
        }

        original_cwd = os.getcwd()
        os.chdir(str(tmp_path))
        try:
            report = build_output_contract_report(contract, work_dir=".")
        finally:
            os.chdir(original_cwd)

        assert report["overall_status"] == "error"
        failures = report["qa_gate_results"]["failures"]
        assert failures[0]["name"] == "inference_latency_within_spec"
        assert failures[0]["value"] == 135.54

    def test_hard_numeric_qa_gate_pass_keeps_output_contract_ok(self, tmp_path):
        metrics_dir = tmp_path / "artifacts" / "ml"
        metrics_dir.mkdir(parents=True)
        (metrics_dir / "latency_benchmark.json").write_text(
            '{"ms_per_1000_debtors": 20.0}',
            encoding="utf-8",
        )
        contract = {
            "required_outputs": ["artifacts/ml/latency_benchmark.json"],
            "ml_engineer": {
                "qa_gates": [
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
                ],
            },
            "artifact_requirements": {
                "required_files": [{"path": "artifacts/ml/latency_benchmark.json"}],
            },
        }

        original_cwd = os.getcwd()
        os.chdir(str(tmp_path))
        try:
            report = build_output_contract_report(contract, work_dir=".")
        finally:
            os.chdir(original_cwd)

        assert report["overall_status"] == "ok"
        assert report["qa_gate_results"]["checked"][0]["passed"] is True

    def test_composite_numeric_qa_gate_uses_metric_checks(self, tmp_path):
        metrics_dir = tmp_path / "artifacts" / "ml"
        metrics_dir.mkdir(parents=True)
        (metrics_dir / "validation_metrics.json").write_text(
            json.dumps(
                {
                    "eligibility_drift_entity_pct": 0.02,
                    "eligibility_drift_ventas_pct": 0.12,
                    "eligibility_drift_saldo_pct": 0.03,
                }
            ),
            encoding="utf-8",
        )
        contract = {
            "required_outputs": ["artifacts/ml/validation_metrics.json"],
            "ml_engineer": {
                "qa_gates": [
                    {
                        "name": "eligibility_drift_within_tolerance",
                        "severity": "HARD",
                        "applies_to_artifact": "artifacts/ml/validation_metrics.json",
                        "params": {
                            "max_entity_drift_pct": 0.10,
                            "max_ventas_drift_pct": 0.10,
                            "max_saldo_drift_pct": 0.10,
                            "metric_checks": [
                                {
                                    "metric": "eligibility_drift_entity_pct",
                                    "operator": "<=",
                                    "threshold_param": "max_entity_drift_pct",
                                },
                                {
                                    "metric": "eligibility_drift_ventas_pct",
                                    "operator": "<=",
                                    "threshold_param": "max_ventas_drift_pct",
                                },
                                {
                                    "metric": "eligibility_drift_saldo_pct",
                                    "operator": "<=",
                                    "threshold_param": "max_saldo_drift_pct",
                                },
                            ],
                        },
                    }
                ]
            },
            "artifact_requirements": {
                "required_files": [{"path": "artifacts/ml/validation_metrics.json"}],
            },
        }

        original_cwd = os.getcwd()
        os.chdir(str(tmp_path))
        try:
            report = build_output_contract_report(contract, work_dir=".")
        finally:
            os.chdir(original_cwd)

        checked = report["qa_gate_results"]["checked"]
        failures = report["qa_gate_results"]["failures"]
        assert len(checked) == 3
        assert report["overall_status"] == "error"
        assert failures[0]["metric"] == "eligibility_drift_ventas_pct"
        assert failures[0]["threshold_param"] == "max_ventas_drift_pct"

    def test_composite_numeric_qa_gate_infers_legacy_threshold_params(self, tmp_path):
        metrics_dir = tmp_path / "artifacts" / "ml"
        metrics_dir.mkdir(parents=True)
        (metrics_dir / "validation_metrics.json").write_text(
            json.dumps(
                {
                    "eligibility_drift_entity_pct": 0.02,
                    "eligibility_drift_ventas_pct": 0.03,
                    "eligibility_drift_saldo_pct": 0.04,
                }
            ),
            encoding="utf-8",
        )
        contract = {
            "required_outputs": ["artifacts/ml/validation_metrics.json"],
            "ml_engineer": {
                "qa_gates": [
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
            },
            "artifact_requirements": {
                "required_files": [{"path": "artifacts/ml/validation_metrics.json"}],
            },
        }

        original_cwd = os.getcwd()
        os.chdir(str(tmp_path))
        try:
            report = build_output_contract_report(contract, work_dir=".")
        finally:
            os.chdir(original_cwd)

        assert report["overall_status"] == "ok"
        assert {item["metric"] for item in report["qa_gate_results"]["checked"]} == {
            "eligibility_drift_entity_pct",
            "eligibility_drift_ventas_pct",
            "eligibility_drift_saldo_pct",
        }

    def test_overall_status_error_missing_file(self, tmp_path):
        """Missing required file -> overall_status=error."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        # Don't create scored_rows.csv

        contract = {
            "required_outputs": ["data/scored_rows.csv"],
            "artifact_requirements": {
                "required_files": [{"path": "data/scored_rows.csv"}],
            },
        }

        original_cwd = os.getcwd()
        os.chdir(str(tmp_path))
        try:
            report = build_output_contract_report(contract, work_dir=".")
        finally:
            os.chdir(original_cwd)

        assert report["overall_status"] == "error"
        assert "scored_rows" in str(report["missing"])

    def test_overall_status_error_missing_required_visual_plot(self, tmp_path):
        """Missing required plot declared in visualization_requirements -> overall_status=error."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "metrics.json").write_text("{}")

        contract = {
            "required_outputs": ["data/metrics.json"],
            "artifact_requirements": {
                "required_files": [{"path": "data/metrics.json"}],
            },
            "visualization_requirements": {
                "required": True,
                "required_plots": [{"name": "confidence_distribution"}],
                "outputs_dir": "static/plots",
            },
        }

        original_cwd = os.getcwd()
        os.chdir(str(tmp_path))
        try:
            report = build_output_contract_report(contract, work_dir=".")
        finally:
            os.chdir(original_cwd)

        assert report["overall_status"] == "error"
        assert "static/plots/confidence_distribution.png" in report["missing"]

    def test_overall_status_error_missing_column(self, tmp_path):
        """Missing required column -> overall_status=error."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "scored_rows.csv").write_text("Id,prediction\n1,0.5\n")
        (data_dir / "cleaning_manifest.json").write_text('{"output_dialect": {}}')

        contract = {
            "required_outputs": ["data/scored_rows.csv"],
            "artifact_requirements": {
                "required_files": [{"path": "data/scored_rows.csv"}],
                "scored_rows_schema": {
                    "required_columns": ["Id", "prediction", "missing_col"],
                },
            },
        }

        original_cwd = os.getcwd()
        os.chdir(str(tmp_path))
        try:
            report = build_output_contract_report(contract, work_dir=".")
        finally:
            os.chdir(original_cwd)

        assert report["overall_status"] == "error"
        assert report["artifact_requirements_report"]["status"] == "error"
        assert "missing_col" in report["artifact_requirements_report"]["scored_rows_report"]["missing_columns"]

    def test_reason_included(self, tmp_path):
        """Reason is included in report when provided."""
        contract = {}

        report = build_output_contract_report(contract, work_dir=".", reason="test_abort")

        assert report["reason"] == "test_abort"

    def test_empty_contract_graceful(self):
        """Empty contract doesn't crash."""
        report = build_output_contract_report({}, work_dir=".")

        assert "present" in report
        assert "missing" in report
        assert "overall_status" in report

    def test_cleaning_only_report_ignores_inactive_ml_required_outputs(self, tmp_path):
        data_dir = tmp_path / "artifacts" / "clean"
        data_dir.mkdir(parents=True)
        (data_dir / "clean_base.csv").write_text("id\n1\n", encoding="utf-8")

        contract = {
            "scope": "cleaning_only",
            "active_workstreams": {
                "cleaning": True,
                "feature_engineering": True,
                "model_training": False,
            },
            "required_outputs": [
                {"path": "artifacts/clean/clean_base.csv", "owner": "data_engineer", "required": True},
                {"path": "artifacts/ml/handoff_note.json", "owner": "ml_engineer", "required": True},
            ],
        }

        original_cwd = os.getcwd()
        os.chdir(str(tmp_path))
        try:
            report = build_output_contract_report(contract, work_dir=".")
        finally:
            os.chdir(original_cwd)

        assert report["missing"] == []
        assert report["overall_status"] == "ok"

    def test_empty_scoring_output_errors_when_primary_scoring_cohort_exists(self, tmp_path):
        clean_dir = tmp_path / "artifacts" / "clean"
        ml_dir = tmp_path / "artifacts" / "ml"
        data_dir = tmp_path / "data"
        clean_dir.mkdir(parents=True)
        ml_dir.mkdir(parents=True)
        data_dir.mkdir(parents=True)

        (clean_dir / "leads_full_archive.csv").write_text(
            "\n".join(
                [
                    "lead_id,created_at",
                    "L1,2025-06-15",
                    "L2,2025-07-02",
                    "L3,2025-07-20",
                    "L4,2024-12-01",
                ]
            ),
            encoding="utf-8",
        )
        (ml_dir / "leads_scored.csv").write_text(
            "lead_id,win_probability,score_decile,created_at\n",
            encoding="utf-8",
        )
        (data_dir / "cleaning_manifest.json").write_text('{"output_dialect": {"sep": ",", "encoding": "utf-8"}}')

        contract = {
            "scoring_rows_rule_primary": "created_at >= '2025-07-01'",
            "required_outputs": [
                {
                    "path": "artifacts/clean/leads_full_archive.csv",
                    "owner": "data_engineer",
                    "required": True,
                    "intent": "full_fidelity_archive",
                    "kind": "dataset",
                },
                {
                    "path": "artifacts/ml/leads_scored.csv",
                    "owner": "ml_engineer",
                    "required": True,
                    "intent": "scoring_output",
                    "kind": "predictions",
                },
            ],
            "artifact_requirements": {
                "required_files": [
                    {"path": "artifacts/clean/leads_full_archive.csv"},
                    {"path": "artifacts/ml/leads_scored.csv"},
                ],
                "full_archive_dataset": {"output_path": "artifacts/clean/leads_full_archive.csv"},
                "scored_rows_schema": {
                    "required_columns": ["lead_id", "win_probability", "score_decile"],
                },
            },
        }

        original_cwd = os.getcwd()
        os.chdir(str(tmp_path))
        try:
            report = build_output_contract_report(contract, work_dir=".")
        finally:
            os.chdir(original_cwd)

        assert report["overall_status"] == "error"
        artifact_report = report["artifact_requirements_report"]
        assert artifact_report["status"] == "error"
        mismatches = artifact_report["row_count_report"]["mismatches"]
        assert len(mismatches) == 1
        assert mismatches[0]["path"] == "artifacts/ml/leads_scored.csv"
        assert mismatches[0]["expected_row_count"] == 2
        assert mismatches[0]["actual_row_count"] == 0

    def test_holdout_predictions_must_match_secondary_selector_rule(self, tmp_path):
        clean_dir = tmp_path / "artifacts" / "clean"
        ml_dir = tmp_path / "artifacts" / "ml"
        data_dir = tmp_path / "data"
        clean_dir.mkdir(parents=True)
        ml_dir.mkdir(parents=True)
        data_dir.mkdir(parents=True)

        (clean_dir / "leads_full_archive.csv").write_text(
            "\n".join(
                [
                    "lead_id,created_at",
                    "L1,2025-01-10",
                    "L2,2025-03-05",
                    "L3,2024-11-01",
                    "L4,2025-08-01",
                ]
            ),
            encoding="utf-8",
        )
        (ml_dir / "holdout_predictions.csv").write_text(
            "\n".join(
                [
                    "lead_id,created_at,win_probability,score_decile",
                    "L3,2024-11-01,0.20,4",
                    "L3B,2024-12-20,0.15,5",
                ]
            ),
            encoding="utf-8",
        )
        (data_dir / "cleaning_manifest.json").write_text('{"output_dialect": {"sep": ",", "encoding": "utf-8"}}')

        contract = {
            "scoring_rows_rule_secondary": "created_at BETWEEN '2025-01-01' AND '2025-06-30'",
            "required_outputs": [
                {
                    "path": "artifacts/clean/leads_full_archive.csv",
                    "owner": "data_engineer",
                    "required": True,
                    "intent": "full_fidelity_archive",
                    "kind": "dataset",
                },
                {
                    "path": "artifacts/ml/holdout_predictions.csv",
                    "owner": "ml_engineer",
                    "required": True,
                    "intent": "holdout_predictions",
                    "kind": "predictions",
                },
            ],
            "artifact_requirements": {
                "required_files": [
                    {"path": "artifacts/clean/leads_full_archive.csv"},
                    {"path": "artifacts/ml/holdout_predictions.csv"},
                ],
                "full_archive_dataset": {"output_path": "artifacts/clean/leads_full_archive.csv"},
            },
        }

        original_cwd = os.getcwd()
        os.chdir(str(tmp_path))
        try:
            report = build_output_contract_report(contract, work_dir=".")
        finally:
            os.chdir(original_cwd)

        assert report["overall_status"] == "error"
        selector_report = report["artifact_requirements_report"]["selector_report"]
        assert len(selector_report["mismatches"]) == 1
        assert selector_report["mismatches"][0]["path"] == "artifacts/ml/holdout_predictions.csv"
        assert selector_report["mismatches"][0]["all_rows_match_rule"] is False


class TestRealWorldScenarios:
    """Test realistic scenarios that previously failed."""

    def test_metric_columns_not_in_required_columns_after_normalization(self, tmp_path):
        """
        After FIX #2, metric-like columns shouldn't be in scored_rows_schema.
        This tests that a properly cleaned schema validates correctly.
        """
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # scored_rows.csv has actual row data, NOT metrics
        (data_dir / "scored_rows.csv").write_text(
            "Id,Survived,prediction,probability\n1,1,1,0.8\n2,0,0,0.3\n"
        )
        (data_dir / "cleaning_manifest.json").write_text(
            '{"output_dialect": {"sep": ",", "encoding": "utf-8"}}'
        )
        (data_dir / "metrics.json").write_text(
            '{"accuracy": 0.85, "roc_auc": 0.9}'
        )

        # Contract with proper schema (no metric columns like Accuracy, ROC-AUC)
        contract = {
            "artifact_requirements": {
                "required_files": [
                    {"path": "data/scored_rows.csv"},
                    {"path": "data/metrics.json"},
                ],
                "scored_rows_schema": {
                    # Proper columns, NOT metrics
                    "required_columns": ["Id", "prediction"],
                },
            },
        }

        result = check_artifact_requirements(
            contract["artifact_requirements"],
            work_dir=str(tmp_path),
        )

        assert result["status"] == "ok"
        assert result["scored_rows_report"]["missing_columns"] == []
