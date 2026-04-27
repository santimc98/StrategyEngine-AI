"""
Tests for governance_reducer.py

Tests the deterministic reducer that derives unified governance verdicts.
"""

import json
import os

import pytest

from src.utils.governance_reducer import (
    compute_governance_verdict,
    derive_run_outcome,
    merge_hard_failures_into_state,
    enrich_gate_context_with_hard_failures,
)
from src.utils.governance import build_run_summary


class TestComputeGovernanceVerdict:
    """Tests for compute_governance_verdict function."""

    def test_all_ok_returns_ok_status(self):
        """When all sources are ok, overall_status should be ok."""
        verdict = compute_governance_verdict(
            output_contract_report={"overall_status": "ok", "missing": []},
            state={"review_verdict": "APPROVED"},
            integrity_report={"issues": []},
        )
        assert verdict["overall_status"] == "ok"
        assert verdict["hard_failures"] == []
        assert verdict["reasons"] == []

    def test_output_contract_error_triggers_error(self):
        """When output_contract_report.overall_status is error, overall_status should be error."""
        verdict = compute_governance_verdict(
            output_contract_report={"overall_status": "error", "missing": []},
            state={"review_verdict": "APPROVED"},
            integrity_report={"issues": []},
        )
        assert verdict["overall_status"] == "error"
        assert "output_contract_compliance_error" in verdict["hard_failures"]
        assert any("overall_status=error" in r for r in verdict["reasons"])

    def test_output_contract_warning_triggers_warning(self):
        """When output_contract_report.overall_status is warning, overall_status should be warning."""
        verdict = compute_governance_verdict(
            output_contract_report={"overall_status": "warning", "missing": []},
            state={"review_verdict": "APPROVED"},
            integrity_report={"issues": []},
        )
        assert verdict["overall_status"] == "warning"
        assert any("overall_status=warning" in r for r in verdict["reasons"])

    def test_artifact_requirements_error_triggers_error(self):
        """When artifact_requirements_report.status is error, overall_status should be error."""
        verdict = compute_governance_verdict(
            output_contract_report={
                "overall_status": "ok",
                "missing": [],
                "artifact_requirements_report": {"status": "error"},
            },
            state={"review_verdict": "APPROVED"},
            integrity_report={"issues": []},
        )
        assert verdict["overall_status"] == "error"
        assert "artifact_requirements_error" in verdict["hard_failures"]

    def test_artifact_requirements_warning_triggers_warning(self):
        """When artifact_requirements_report.status is warning, overall_status should be warning."""
        verdict = compute_governance_verdict(
            output_contract_report={
                "overall_status": "ok",
                "missing": [],
                "artifact_requirements_report": {"status": "warning"},
            },
            state={"review_verdict": "APPROVED"},
            integrity_report={"issues": []},
        )
        assert verdict["overall_status"] == "warning"
        assert any("artifact_requirements_report.status=warning" in r for r in verdict["reasons"])

    def test_missing_files_triggers_error(self):
        """When output_contract has missing files, overall_status should be error."""
        verdict = compute_governance_verdict(
            output_contract_report={"overall_status": "ok", "missing": ["data/scored_rows.csv"]},
            state={"review_verdict": "APPROVED"},
            integrity_report={"issues": []},
        )
        assert verdict["overall_status"] == "error"
        assert "output_contract_missing_files" in verdict["hard_failures"]
        assert "output_contract_missing" in verdict["failed_gates"]

    def test_rejected_review_verdict_triggers_error(self):
        """When review_verdict is REJECTED, overall_status should be error."""
        verdict = compute_governance_verdict(
            output_contract_report={"overall_status": "ok", "missing": []},
            state={"review_verdict": "REJECTED"},
            integrity_report={"issues": []},
        )
        assert verdict["overall_status"] == "error"
        assert "review_rejected" in verdict["hard_failures"]

    def test_fail_status_in_gate_context_triggers_error(self):
        """When gate_context.status is FAIL, overall_status should be error."""
        verdict = compute_governance_verdict(
            output_contract_report={"overall_status": "ok", "missing": []},
            state={
                "review_verdict": "APPROVED",
                "last_gate_context": {"status": "FAIL", "failed_gates": ["some_gate"]},
            },
            integrity_report={"issues": []},
        )
        assert verdict["overall_status"] == "error"
        assert "gate_fail" in verdict["hard_failures"]
        assert "some_gate" in verdict["failed_gates"]

    def test_hard_failures_in_gate_context_triggers_error(self):
        """When gate_context contains hard_failures, overall_status should be error."""
        verdict = compute_governance_verdict(
            output_contract_report={"overall_status": "ok", "missing": []},
            state={
                "review_verdict": "APPROVED",
                "last_gate_context": {
                    "status": "APPROVED",
                    "hard_failures": ["security_sandbox"],
                },
            },
            integrity_report={"issues": []},
        )
        assert verdict["overall_status"] == "error"
        assert "security_sandbox" in verdict["hard_failures"]

    def test_pipeline_aborted_triggers_error(self):
        """When pipeline_aborted_reason is set, overall_status should be error."""
        verdict = compute_governance_verdict(
            output_contract_report={"overall_status": "ok", "missing": []},
            state={
                "review_verdict": "APPROVED",
                "pipeline_aborted_reason": "data_engineer_preflight_failed",
            },
            integrity_report={"issues": []},
        )
        assert verdict["overall_status"] == "error"
        assert any("pipeline_aborted" in hf for hf in verdict["hard_failures"])

    def test_data_engineer_failed_triggers_error(self):
        """When data_engineer_failed is True, overall_status should be error."""
        verdict = compute_governance_verdict(
            output_contract_report={"overall_status": "ok", "missing": []},
            state={"review_verdict": "APPROVED", "data_engineer_failed": True},
            integrity_report={"issues": []},
        )
        assert verdict["overall_status"] == "error"
        assert "data_engineer_failed" in verdict["hard_failures"]

    def test_state_hard_failures_triggers_error(self):
        """When state contains accumulated hard_failures, overall_status should be error."""
        verdict = compute_governance_verdict(
            output_contract_report={"overall_status": "ok", "missing": []},
            state={
                "review_verdict": "APPROVED",
                "hard_failures": ["ML_PLAN_INVALID", "no_synthetic_data"],
            },
            integrity_report={"issues": []},
        )
        assert verdict["overall_status"] == "error"
        assert "ML_PLAN_INVALID" in verdict["hard_failures"]
        assert "no_synthetic_data" in verdict["hard_failures"]

    def test_integrity_critical_triggers_error(self):
        """When integrity report has critical issues, overall_status should be error."""
        verdict = compute_governance_verdict(
            output_contract_report={"overall_status": "ok", "missing": []},
            state={"review_verdict": "APPROVED"},
            integrity_report={
                "issues": [
                    {"type": "MISSING_COLUMN", "severity": "critical"},
                    {"type": "OPTIONAL", "severity": "warning"},
                ]
            },
        )
        assert verdict["overall_status"] == "error"
        assert "integrity_critical" in verdict["hard_failures"]

    def test_integrity_warning_does_not_trigger_error(self):
        """When integrity report has only warning issues, overall_status should remain ok."""
        verdict = compute_governance_verdict(
            output_contract_report={"overall_status": "ok", "missing": []},
            state={"review_verdict": "APPROVED"},
            integrity_report={
                "issues": [{"type": "OPTIONAL", "severity": "warning"}]
            },
        )
        assert verdict["overall_status"] == "ok"
        assert "integrity_critical" not in verdict["hard_failures"]

    def test_error_takes_precedence_over_warning(self):
        """Error status takes precedence over warning."""
        verdict = compute_governance_verdict(
            output_contract_report={
                "overall_status": "warning",
                "missing": [],
                "artifact_requirements_report": {"status": "error"},
            },
            state={"review_verdict": "APPROVED"},
            integrity_report={"issues": []},
        )
        assert verdict["overall_status"] == "error"

    def test_threshold_gap_qa_rejection_is_warning_not_hard_failure(self):
        """Measured metric-threshold misses should not make the final governance NO_GO."""
        threshold_failure = {
            "name": "eligibility_preservation_by_corporation",
            "severity": "HARD",
            "artifact_path": "artifacts/ml/evaluation_report.json",
            "metric": "eligibility_debtor_count_dev_max_pct",
            "value": 0.25,
            "threshold": 0.1,
            "operator": "<=",
            "status": "fail",
            "passed": False,
        }
        verdict = compute_governance_verdict(
            output_contract_report={
                "overall_status": "error",
                "missing": [],
                "artifact_requirements_report": {"status": "ok"},
                "qa_gate_results": {"failures": [threshold_failure], "warnings": []},
            },
            state={
                "review_verdict": "APPROVE_WITH_WARNINGS",
                "review_board_verdict": {
                    "performance_threshold_policy": "optimization_target_not_baseline_blocker",
                    "performance_threshold_gaps": [threshold_failure],
                },
                "last_gate_context": {
                    "status": "APPROVE_WITH_WARNINGS",
                    "failed_gates": ["eligibility_preservation_by_corporation"],
                    "hard_failures": ["eligibility_preservation_by_corporation"],
                },
                "qa_last_result": {
                    "status": "REJECTED",
                    "failed_gates": ["eligibility_preservation_by_corporation"],
                    "hard_failures": ["eligibility_preservation_by_corporation"],
                },
            },
            contract={
                "qa_gates": [
                    {"name": "eligibility_preservation_by_corporation", "severity": "HARD"}
                ]
            },
            integrity_report={"issues": []},
        )

        assert verdict["overall_status"] == "warning"
        assert "eligibility_preservation_by_corporation" not in verdict["hard_failures"]
        assert not any(str(item).startswith("hard_gate_failed") for item in verdict["hard_failures"])


class TestDeriveRunOutcome:
    """Tests for derive_run_outcome function."""

    def test_error_status_returns_no_go(self):
        """When overall_status is error, run_outcome should be NO_GO."""
        verdict = {"overall_status": "error", "hard_failures": ["test"]}
        assert derive_run_outcome(verdict) == "NO_GO"

    def test_warning_status_returns_go_with_limitations(self):
        """When overall_status is warning, run_outcome should be GO_WITH_LIMITATIONS."""
        verdict = {"overall_status": "warning", "hard_failures": []}
        assert derive_run_outcome(verdict) == "GO_WITH_LIMITATIONS"

    def test_ceiling_detected_returns_go_with_limitations(self):
        """When ceiling_detected is True, run_outcome should be GO_WITH_LIMITATIONS."""
        verdict = {"overall_status": "ok", "hard_failures": []}
        assert derive_run_outcome(verdict, ceiling_detected=True) == "GO_WITH_LIMITATIONS"

    def test_observational_only_returns_go_with_limitations(self):
        """When observational_only is True, run_outcome should be GO_WITH_LIMITATIONS."""
        verdict = {"overall_status": "ok", "hard_failures": []}
        assert derive_run_outcome(verdict, observational_only=True) == "GO_WITH_LIMITATIONS"

    def test_all_ok_returns_go(self):
        """When everything is ok, run_outcome should be GO."""
        verdict = {"overall_status": "ok", "hard_failures": []}
        assert derive_run_outcome(verdict) == "GO"

    def test_error_takes_precedence_over_ceiling(self):
        """Error status takes precedence over ceiling_detected."""
        verdict = {"overall_status": "error", "hard_failures": ["test"]}
        assert derive_run_outcome(verdict, ceiling_detected=True) == "NO_GO"


class TestMergeHardFailuresIntoState:
    """Tests for merge_hard_failures_into_state function."""

    def test_merge_new_hard_failures(self):
        """New hard failures should be added to state."""
        state = {"hard_failures": ["existing"]}
        result = merge_hard_failures_into_state(state, ["new1", "new2"])
        assert "existing" in result["hard_failures"]
        assert "new1" in result["hard_failures"]
        assert "new2" in result["hard_failures"]

    def test_no_duplicates(self):
        """Duplicate hard failures should not be added."""
        state = {"hard_failures": ["existing"]}
        result = merge_hard_failures_into_state(state, ["existing", "new"])
        assert result["hard_failures"].count("existing") == 1
        assert "new" in result["hard_failures"]

    def test_empty_state(self):
        """Should work with empty state."""
        state = {}
        result = merge_hard_failures_into_state(state, ["new"])
        assert result["hard_failures"] == ["new"]


class TestEnrichGateContextWithHardFailures:
    """Tests for enrich_gate_context_with_hard_failures function."""

    def test_enriches_gate_context(self):
        """Should add hard_failures from reviewer result to gate_context."""
        gate_context = {"status": "REJECTED", "failed_gates": ["gate1"]}
        reviewer_result = {"hard_failures": ["hf1", "hf2"]}
        result = enrich_gate_context_with_hard_failures(gate_context, reviewer_result)
        assert result["hard_failures"] == ["hf1", "hf2"]
        assert result["status"] == "REJECTED"

    def test_no_hard_failures_in_result(self):
        """Should not modify gate_context if no hard_failures in reviewer result."""
        gate_context = {"status": "APPROVED"}
        reviewer_result = {"status": "APPROVED"}
        result = enrich_gate_context_with_hard_failures(gate_context, reviewer_result)
        assert "hard_failures" not in result or result.get("hard_failures") == []


class TestBuildRunSummaryIntegration:
    """Integration tests for build_run_summary with reducer."""

    def test_output_contract_missing_columns_forces_no_go(self, tmp_path, monkeypatch):
        """
        Case 1: output_contract has missing_columns (overall_status="error" but missing==[])
        Should result in NO_GO.
        """
        monkeypatch.chdir(tmp_path)
        os.makedirs("data", exist_ok=True)

        # output_contract_report with error status (schema validation failed)
        with open("data/output_contract_report.json", "w", encoding="utf-8") as f:
            json.dump({
                "overall_status": "error",
                "missing": [],  # No missing files, but schema validation failed
                "artifact_requirements_report": {
                    "status": "error",
                    "scored_rows_report": {"missing_columns": ["target"]}
                }
            }, f)

        with open("data/metrics.json", "w", encoding="utf-8") as f:
            json.dump({"auc": 0.8}, f)

        summary = build_run_summary({"review_verdict": "APPROVED"})
        assert summary["run_outcome"] == "NO_GO"
        assert summary["overall_status_global"] == "error"
        assert any("artifact_requirements" in hf for hf in summary["hard_failures"])

    def test_artifact_requirements_warning_returns_go_with_limitations(self, tmp_path, monkeypatch):
        """
        Case 2: artifact_requirements_report.status="warning"
        Should result in GO_WITH_LIMITATIONS.
        """
        monkeypatch.chdir(tmp_path)
        os.makedirs("data", exist_ok=True)

        with open("data/output_contract_report.json", "w", encoding="utf-8") as f:
            json.dump({
                "overall_status": "warning",
                "missing": [],
                "artifact_requirements_report": {"status": "warning"}
            }, f)

        with open("data/metrics.json", "w", encoding="utf-8") as f:
            json.dump({"auc": 0.8}, f)

        summary = build_run_summary({"review_verdict": "APPROVED"})
        assert summary["run_outcome"] == "GO_WITH_LIMITATIONS"
        assert summary["overall_status_global"] == "warning"

    def test_hard_failures_present_forces_no_go(self, tmp_path, monkeypatch):
        """
        Case 3: hard_failures present in state
        Should result in NO_GO.
        """
        monkeypatch.chdir(tmp_path)
        os.makedirs("data", exist_ok=True)

        with open("data/output_contract_report.json", "w", encoding="utf-8") as f:
            json.dump({"overall_status": "ok", "missing": []}, f)

        with open("data/metrics.json", "w", encoding="utf-8") as f:
            json.dump({"auc": 0.8}, f)

        summary = build_run_summary({
            "review_verdict": "APPROVED",
            "hard_failures": ["security_sandbox", "no_synthetic_data"],
        })
        assert summary["run_outcome"] == "NO_GO"
        assert summary["overall_status_global"] == "error"
        assert "security_sandbox" in summary["hard_failures"]
        assert "no_synthetic_data" in summary["hard_failures"]

    def test_rejected_status_forces_no_go(self, tmp_path, monkeypatch):
        """
        Case 4: status REJECTED although outputs ok
        Should result in NO_GO.
        """
        monkeypatch.chdir(tmp_path)
        os.makedirs("data", exist_ok=True)

        with open("data/output_contract_report.json", "w", encoding="utf-8") as f:
            json.dump({"overall_status": "ok", "missing": []}, f)

        with open("data/metrics.json", "w", encoding="utf-8") as f:
            json.dump({"auc": 0.8}, f)

        summary = build_run_summary({"review_verdict": "REJECTED"})
        assert summary["run_outcome"] == "NO_GO"
        assert summary["overall_status_global"] == "error"

    def test_all_ok_returns_go(self, tmp_path, monkeypatch):
        """
        Case 5: Everything ok
        Should result in GO.
        """
        monkeypatch.chdir(tmp_path)
        os.makedirs("data", exist_ok=True)

        with open("data/output_contract_report.json", "w", encoding="utf-8") as f:
            json.dump({"overall_status": "ok", "missing": []}, f)

        with open("data/metrics.json", "w", encoding="utf-8") as f:
            json.dump({"auc": 0.8}, f)

        with open("data/integrity_audit_report.json", "w", encoding="utf-8") as f:
            json.dump({"issues": []}, f)

        summary = build_run_summary({"review_verdict": "APPROVED"})
        assert summary["run_outcome"] == "GO"
        assert summary["overall_status_global"] == "ok"
        assert summary["hard_failures"] == []

    def test_gate_context_hard_failures_forces_no_go(self, tmp_path, monkeypatch):
        """
        Hard failures from gate_context should force NO_GO.
        """
        monkeypatch.chdir(tmp_path)
        os.makedirs("data", exist_ok=True)

        with open("data/output_contract_report.json", "w", encoding="utf-8") as f:
            json.dump({"overall_status": "ok", "missing": []}, f)

        with open("data/metrics.json", "w", encoding="utf-8") as f:
            json.dump({"auc": 0.8}, f)

        summary = build_run_summary({
            "review_verdict": "APPROVED",
            "last_gate_context": {
                "status": "REJECTED",
                "hard_failures": ["ML_PLAN_INVALID"],
                "failed_gates": ["ML_PLAN_INVALID"],
            }
        })
        assert summary["run_outcome"] == "NO_GO"
        assert "ML_PLAN_INVALID" in summary["hard_failures"]

    def test_backward_compatibility_with_existing_fields(self, tmp_path, monkeypatch):
        """
        Ensure backward compatibility: existing fields are preserved.
        """
        monkeypatch.chdir(tmp_path)
        os.makedirs("data", exist_ok=True)

        with open("data/output_contract_report.json", "w", encoding="utf-8") as f:
            json.dump({"overall_status": "ok", "missing": []}, f)

        with open("data/metrics.json", "w", encoding="utf-8") as f:
            json.dump({"auc": 0.8}, f)

        with open("data/integrity_audit_report.json", "w", encoding="utf-8") as f:
            json.dump({"issues": [{"severity": "warning"}]}, f)

        summary = build_run_summary({
            "run_id": "test-123",
            "review_verdict": "APPROVED",
            "budget_counters": {"qa": 2},
        })

        # Check backward compatible fields exist
        assert "run_id" in summary
        assert "status" in summary
        assert "run_outcome" in summary
        assert "failed_gates" in summary
        assert "warnings" in summary
        assert "budget_counters" in summary
        assert "metric_ceiling_detected" in summary
        assert "integrity_critical_count" in summary

        # Check new fields exist
        assert "overall_status_global" in summary
        assert "hard_failures" in summary
        assert "governance_reasons" in summary
