"""
Tests for V4.1 contract-strict mode in CleaningReviewerAgent.

The Cleaning Reviewer MUST reject cleaning if cleaning_gates are missing from
the cleaning_view. This ensures the pipeline fails fast and triggers contract
regeneration rather than silently proceeding with fallback gates.
"""
import json
import os
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest

from src.agents.cleaning_reviewer import (
    CleaningReviewerAgent,
    _build_llm_prompt,
    _enforce_contract_strict_rejection,
    _merge_cleaning_gates,
    _CONTRACT_MISSING_CLEANING_GATES,
)


class TestContractStrictMode:
    """Test suite for contract-strict mode behavior."""

    def test_llm_prompt_uses_senior_precedence_and_review_workflow(self):
        prompt, payload = _build_llm_prompt(
            gates=[{"name": "required_columns_present", "severity": "HARD", "params": {}}],
            required_columns=["col_a"],
            dialect={"sep": ",", "decimal": ".", "encoding": "utf-8"},
            column_roles={"identifiers": ["col_a"]},
            facts={"row_count": 3},
            deterministic_gate_results=[{"name": "required_columns_present", "passed": True}],
            contract_source_used="cleaning_view",
            context_pack="sample context",
            cleaning_code="df = df.copy()",
            dataset_profile={"basic_stats": {"n_rows": 3}},
            column_resolution_context={
                "col_a": {
                    "semantic_kind": "datetime_like",
                    "observed_format_families": ["iso_date", "slash_date"],
                }
            },
        )

        assert "MISSION" in prompt
        assert "SOURCE OF TRUTH AND PRECEDENCE" in prompt
        assert "REVIEW DECISION WORKFLOW (MANDATORY)" in prompt
        assert "deterministic_gate_results are supporting evidence only" in prompt
        assert "GUIDANCE, NOT A SUBSTITUTE FOR REASONING" in prompt
        assert "column_resolution_context" in prompt
        assert payload["contract_source_used"] == "cleaning_view"
        assert "column_resolution_context" in payload

    def test_merge_cleaning_gates_returns_fallback_source_when_empty(self):
        """When cleaning_gates is missing/empty, source should be 'fallback'."""
        # Empty cleaning_view
        view_empty: Dict[str, Any] = {}
        gates, source, warnings = _merge_cleaning_gates(view_empty)

        assert source == "fallback"
        assert len(warnings) > 0
        assert any("CONTRACT_BROKEN_FALLBACK" in w for w in warnings)

        # cleaning_view with empty list
        view_empty_list: Dict[str, Any] = {"cleaning_gates": []}
        gates2, source2, warnings2 = _merge_cleaning_gates(view_empty_list)

        assert source2 == "fallback"

    def test_merge_cleaning_gates_returns_cleaning_view_source_when_present(self):
        """When cleaning_gates is present, source should be 'cleaning_view'."""
        view_with_gates: Dict[str, Any] = {
            "cleaning_gates": [
                {"name": "required_columns_present", "severity": "HARD", "params": {}}
            ]
        }
        gates, source, warnings = _merge_cleaning_gates(view_with_gates)

        assert source == "cleaning_view"
        assert len(gates) > 0
        # Should not have CONTRACT_BROKEN_FALLBACK warning
        assert not any("CONTRACT_BROKEN_FALLBACK" in w for w in warnings)

    def test_enforce_contract_strict_rejection_forces_rejected_on_fallback(self):
        """When contract_source_used is 'fallback', result MUST be REJECTED."""
        # Simulate a result that would otherwise be APPROVED
        passing_result: Dict[str, Any] = {
            "status": "APPROVED",
            "feedback": "All gates passed.",
            "failed_checks": [],
            "required_fixes": [],
            "hard_failures": [],
            "soft_failures": [],
            "contract_source_used": "fallback",
        }

        enforced = _enforce_contract_strict_rejection(passing_result)

        assert enforced["status"] == "REJECTED"
        assert _CONTRACT_MISSING_CLEANING_GATES in enforced["hard_failures"]
        assert _CONTRACT_MISSING_CLEANING_GATES in enforced["failed_checks"]
        assert any("Regenerate Execution Contract" in fix for fix in enforced["required_fixes"])
        assert "CONTRACT INCOMPLETE" in enforced["feedback"]

    def test_enforce_contract_strict_rejection_does_not_change_cleaning_view_source(self):
        """When contract_source_used is 'cleaning_view', result should not be modified."""
        passing_result: Dict[str, Any] = {
            "status": "APPROVED",
            "feedback": "All gates passed.",
            "failed_checks": [],
            "required_fixes": [],
            "hard_failures": [],
            "soft_failures": [],
            "contract_source_used": "cleaning_view",
        }

        enforced = _enforce_contract_strict_rejection(passing_result)

        assert enforced["status"] == "APPROVED"
        assert _CONTRACT_MISSING_CLEANING_GATES not in enforced.get("hard_failures", [])


class TestCleaningReviewerContractStrict:
    """Integration tests for CleaningReviewerAgent with contract-strict mode."""

    @pytest.fixture
    def temp_cleaned_csv(self, tmp_path: Path) -> str:
        """Create a minimal cleaned CSV file."""
        csv_path = tmp_path / "cleaned_data.csv"
        df = pd.DataFrame({"col_a": [1, 2, 3], "col_b": ["x", "y", "z"]})
        df.to_csv(csv_path, index=False)
        return str(csv_path)

    @pytest.fixture
    def temp_manifest(self, tmp_path: Path) -> str:
        """Create a minimal cleaning manifest."""
        manifest_path = tmp_path / "cleaning_manifest.json"
        manifest = {
            "original_row_count": 3,
            "cleaned_row_count": 3,
            "output_dialect": {"sep": ",", "decimal": ".", "encoding": "utf-8"},
        }
        manifest_path.write_text(json.dumps(manifest))
        return str(manifest_path)

    def test_review_cleaning_rejects_when_cleaning_gates_missing(
        self, temp_cleaned_csv: str, temp_manifest: str, monkeypatch
    ):
        """
        Case A: cleaning_view without cleaning_gates => REJECTED + CONTRACT_MISSING_CLEANING_GATES.
        """
        # Disable LLM for deterministic test
        monkeypatch.setenv("CLEANING_REVIEWER_PROVIDER", "none")

        agent = CleaningReviewerAgent(api_key=None)

        # cleaning_view WITHOUT cleaning_gates
        context = {
            "cleaning_view": {
                "required_columns": ["col_a", "col_b"],
                # NO cleaning_gates
            },
            "cleaned_csv_path": temp_cleaned_csv,
            "cleaning_manifest_path": temp_manifest,
        }

        result = agent.review_cleaning(context)

        assert result["status"] == "REJECTED"
        assert _CONTRACT_MISSING_CLEANING_GATES in result.get("hard_failures", [])
        assert _CONTRACT_MISSING_CLEANING_GATES in result.get("failed_checks", [])
        assert result.get("contract_source_used") == "fallback"
        assert any("Regenerate" in fix for fix in result.get("required_fixes", []))

    def test_review_cleaning_can_approve_when_cleaning_gates_present(
        self, temp_cleaned_csv: str, temp_manifest: str, monkeypatch
    ):
        """
        Case B: cleaning_view with cleaning_gates => can be APPROVED/APPROVE_WITH_WARNINGS.
        """
        # Disable LLM for deterministic test
        monkeypatch.setenv("CLEANING_REVIEWER_PROVIDER", "none")

        agent = CleaningReviewerAgent(api_key=None)

        # cleaning_view WITH cleaning_gates (minimal set)
        context = {
            "cleaning_view": {
                "required_columns": ["col_a", "col_b"],
                "cleaning_gates": [
                    {"name": "required_columns_present", "severity": "HARD", "params": {}},
                ],
            },
            "cleaned_csv_path": temp_cleaned_csv,
            "cleaning_manifest_path": temp_manifest,
        }

        result = agent.review_cleaning(context)

        # Should NOT be rejected due to missing contract
        assert result.get("contract_source_used") == "cleaning_view"
        assert _CONTRACT_MISSING_CLEANING_GATES not in result.get("hard_failures", [])
        # Status depends on actual gate evaluation, but not CONTRACT_MISSING
        assert result["status"] in {"APPROVED", "APPROVE_WITH_WARNINGS", "REJECTED"}


class TestLegacyContextRemoval:
    """Test that legacy required_columns fallback is removed."""

    def test_parse_legacy_context_does_not_copy_required_columns(self):
        """V4.1: _parse_legacy_context should NOT copy required_columns from context."""
        from src.agents.cleaning_reviewer import _parse_legacy_context

        context = {
            "required_columns": ["legacy_col_a", "legacy_col_b"],
            "cleaning_view": {},
        }

        parsed = _parse_legacy_context(context)
        cleaning_view = parsed["cleaning_view"]

        # required_columns should NOT be copied from legacy context
        assert "required_columns" not in cleaning_view or cleaning_view.get("required_columns") != ["legacy_col_a", "legacy_col_b"]
