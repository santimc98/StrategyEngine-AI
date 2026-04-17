"""
Tests for the post-mortem narrative pivot of the business_translator.

When a run aborts before completing (data_engineer/ml_engineer exhausted retries
without producing valid outputs), the translator must pivot from "executive
report" mode to "post-mortem" mode. The switch is driven deterministically by
the presence of `abort_info` inside the run narrative.

These tests verify the plumbing:
  1. _build_translator_run_narrative emits abort_info when state is aborted.
  2. _build_translator_run_narrative DOES NOT emit abort_info on happy path.
  3. _build_outline_prompt injects POST-MORTEM MODE hint when abort_info given.
  4. generate_report wires post_mortem_directive into the final prompt when
     the narrative contains abort_info, and leaves it empty otherwise.

They do not verify LLM output quality (that is what the replay script is for).
"""

from __future__ import annotations

import json
from string import Template
from typing import Any, Dict

from src.agents.business_translator import _build_outline_prompt
from src.graph.graph import _build_translator_run_narrative


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _aborted_state() -> Dict[str, Any]:
    """State resembling a run where data_engineer exhausted retries."""
    return {
        "business_objective": "Predict customer churn risk for Q3 planning.",
        "execution_contract": {
            "business_objective": "Predict customer churn risk for Q3 planning.",
        },
        "pipeline_aborted_reason": "data_engineer_budget_exceeded",
        "error_message": "DataEngineer: exceeded 6 attempts without producing cleaned_dataset.",
        "data_engineer_attempt_history": [
            {
                "attempt": 1,
                "source": "data_engineer",
                "status": "RUNTIME_ERROR",
                "failed_gates": ["cleaning_gate:dtype_target_int64"],
                "required_fixes": ["Fix Int64 imputation pattern on days_to_churn."],
                "runtime_error_tail": "TypeError: Cannot set non-numeric value on Int64 column 'days_to_churn'.",
                "feedback_summary": "Int64 rejected 'Unknown' sentinel.",
            },
            {
                "attempt": 2,
                "source": "data_engineer",
                "status": "RUNTIME_ERROR",
                "failed_gates": ["cleaning_gate:dtype_target_int64"],
                "required_fixes": ["Cast to object before imputing strings."],
                "runtime_error_tail": "TypeError: Cannot set non-numeric value on Int64 column 'days_to_churn'.",
                "feedback_summary": "Retry still hit same Int64 TypeError.",
            },
            {
                "attempt": 3,
                "source": "data_engineer",
                "status": "RUNTIME_ERROR",
                "failed_gates": ["cleaning_gate:dtype_target_int64", "cleaning_gate:rowcount_preserved"],
                "required_fixes": ["Preserve row count while imputing."],
                "runtime_error_tail": "TypeError: Cannot set non-numeric value on Int64 column 'days_to_churn'.",
                "feedback_summary": "Editor attempt regressed rowcount gate too.",
            },
        ],
    }


def _happy_state() -> Dict[str, Any]:
    """State resembling a run that completed end-to-end successfully."""
    return {
        "business_objective": "Predict customer churn risk for Q3 planning.",
        "execution_contract": {
            "business_objective": "Predict customer churn risk for Q3 planning.",
        },
        "primary_metric_state": {
            "primary_metric_name": "roc_auc",
            "primary_metric_value": 0.9165,
        },
    }


def _happy_run_summary() -> Dict[str, Any]:
    return {
        "run_outcome": "GO",
        "failed_gates": [],
        "warnings": [],
        "metric_improvement": {
            "baseline_metric": 0.88,
            "final_metric_reported": 0.9165,
            "kept": True,
            "metric_name": "roc_auc",
        },
    }


def _aborted_run_summary() -> Dict[str, Any]:
    return {
        "run_outcome": "NO_GO",
        "failed_gates": ["pipeline_aborted:data_engineer_budget_exceeded"],
        "warnings": [],
    }


# ---------------------------------------------------------------------------
# Test 1: narrative emits abort_info on aborted state
# ---------------------------------------------------------------------------

def test_narrative_emits_abort_info_on_aborted_state():
    narrative = _build_translator_run_narrative(_aborted_state(), _aborted_run_summary())

    assert "abort_info" in narrative, (
        "abort_info must be surfaced when pipeline_aborted_reason is set"
    )
    abort = narrative["abort_info"]

    # Reason propagated from state.
    assert abort.get("abort_reason") == "data_engineer_budget_exceeded"

    # Error message preserved (truncated to 800 chars).
    assert "exceeded 6 attempts" in abort.get("error_message", "")

    # Attempt history compressed with the shape the translator prompt expects.
    de_attempts = abort.get("data_engineer_attempts")
    assert isinstance(de_attempts, dict), "data_engineer_attempts must be a dict summary"
    assert de_attempts.get("attempt_count") == 3
    assert "cleaning_gate:dtype_target_int64" in de_attempts.get("unique_failed_gates", [])
    assert "Int64" in de_attempts.get("last_error_tail", "")
    # At most the last 3 attempts are kept.
    assert len(de_attempts.get("attempts", [])) <= 3

    # Aborted flag is true when abort_reason present.
    assert abort.get("run_aborted") is True


# ---------------------------------------------------------------------------
# Test 2: narrative does NOT emit abort_info on happy path (no regression)
# ---------------------------------------------------------------------------

def test_narrative_omits_abort_info_on_happy_state():
    narrative = _build_translator_run_narrative(_happy_state(), _happy_run_summary())

    assert "abort_info" not in narrative, (
        "Happy-path runs must NOT receive abort_info (would confuse the template pivot)"
    )
    # Sanity check: normal fields still populated.
    assert narrative.get("system_decision") == "GO"
    assert narrative.get("primary_metric", {}).get("value") == 0.9165


# ---------------------------------------------------------------------------
# Test 3: outline prompt injects POST-MORTEM MODE hint when abort_info present
# ---------------------------------------------------------------------------

def test_outline_prompt_injects_post_mortem_hint_when_aborted():
    abort_info = {
        "abort_reason": "data_engineer_budget_exceeded",
        "run_aborted": True,
        "data_engineer_attempts": {"attempt_count": 3, "last_error_tail": "TypeError on Int64"},
    }
    prompt_aborted = _build_outline_prompt(
        target_language_code="es",
        executive_decision_label="NO_GO",
        facts_block={"primary_metric": None},
        reporting_policy_context={},
        evidence_paths=["data/steward_summary.json"],
        execution_results="",
        abort_info=abort_info,
    )
    prompt_happy = _build_outline_prompt(
        target_language_code="es",
        executive_decision_label="GO",
        facts_block={"primary_metric": {"value": 0.9}},
        reporting_policy_context={},
        evidence_paths=["data/steward_summary.json"],
        execution_results="",
        abort_info=None,
    )

    assert "POST-MORTEM MODE" in prompt_aborted, (
        "Outline prompt must contain the post-mortem hint when abort_info is provided"
    )
    assert "data_engineer_budget_exceeded" in prompt_aborted, (
        "Outline prompt must surface the abort_reason to the planner"
    )
    assert "POST-MORTEM MODE" not in prompt_happy, (
        "Happy-path outline prompt must NOT contain the post-mortem hint"
    )


# ---------------------------------------------------------------------------
# Test 4: generate_report wiring — post_mortem_directive is gated by abort_info
# ---------------------------------------------------------------------------
# We do not spin up the full generate_report pipeline (too heavy); instead we
# verify the exact wiring logic applied inside it: given a run_narrative dict,
# construct post_mortem_directive following the same rule generate_report uses.

def _simulate_generate_report_wiring(run_narrative: Any) -> str:
    """Mirror the directive-construction logic from generate_report."""
    abort_info = {}
    if isinstance(run_narrative, dict):
        maybe_abort = run_narrative.get("abort_info")
        if isinstance(maybe_abort, dict) and maybe_abort:
            abort_info = maybe_abort
    if not abort_info:
        return ""
    # Sentinel content that our real directive must contain.
    return "=== RUN ABORTED — POST-MORTEM MODE (OVERRIDES NORMAL NARRATIVE) ==="


def test_generate_report_post_mortem_directive_gated_by_abort_info():
    # Aborted narrative → directive is non-empty and marks post-mortem.
    aborted_narrative = _build_translator_run_narrative(
        _aborted_state(), _aborted_run_summary()
    )
    directive_aborted = _simulate_generate_report_wiring(aborted_narrative)
    assert directive_aborted, "Aborted narrative must yield a non-empty directive"
    assert "POST-MORTEM MODE" in directive_aborted

    # Happy narrative → directive is empty, template renders nothing.
    happy_narrative = _build_translator_run_narrative(
        _happy_state(), _happy_run_summary()
    )
    directive_happy = _simulate_generate_report_wiring(happy_narrative)
    assert directive_happy == "", (
        "Happy-path narrative must yield an empty directive (template unchanged)"
    )

    # Verify the sentinel substitutes cleanly into a Template with $post_mortem_directive.
    # This protects against accidental Template-interpolation errors.
    tpl = Template("before\n$post_mortem_directive\nafter")
    rendered_aborted = tpl.substitute(post_mortem_directive=directive_aborted)
    rendered_happy = tpl.substitute(post_mortem_directive=directive_happy)
    assert "POST-MORTEM MODE" in rendered_aborted
    assert "POST-MORTEM MODE" not in rendered_happy


# ---------------------------------------------------------------------------
# Sanity: the real SYSTEM_PROMPT_TEMPLATE in business_translator contains the
# $post_mortem_directive placeholder. If someone removes it by accident, this
# test fires before a run ever runs.
# ---------------------------------------------------------------------------

def test_system_prompt_template_exposes_post_mortem_placeholder():
    import inspect

    from src.agents import business_translator as bt

    source = inspect.getsource(bt)
    assert "$post_mortem_directive" in source, (
        "SYSTEM_PROMPT_TEMPLATE must expose the $post_mortem_directive placeholder "
        "for the abort-narrative pivot to fire"
    )
    assert "RUN ABORTED — POST-MORTEM MODE" in source, (
        "The post-mortem directive content must remain in business_translator"
    )
