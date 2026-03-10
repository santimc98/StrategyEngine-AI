"""Tests that _build_review_board_facts reports correct runtime_status after
a successful retry following a runtime failure.

Bug context (run c77c8288):
  After a runtime error in attempt N, the next attempt (N+1) succeeds.
  However _build_review_board_facts still reported FAILED_RUNTIME because
  _resolve_authoritative_runtime_text found residual traceback text in
  stale state fields (last_gate_context.traceback, heavy_runner_error_context).
  This caused the review board to reject iterations that both reviewer
  and QA had approved, preventing the metric improvement loop from ever
  starting.
"""
import pytest
from unittest.mock import patch


def _make_state(
    execution_error=False,
    sandbox_failed=False,
    execution_output="HEAVY_RUNNER: status=success reason=local_runner_mode",
    last_runtime_error_tail=None,
    heavy_runner_error_context=None,
    last_gate_context=None,
    runtime_fix_terminal=False,
    runtime_fix_count=0,
    extra=None,
):
    state = {
        "execution_error": execution_error,
        "sandbox_failed": sandbox_failed,
        "execution_output": execution_output,
        "last_runtime_error_tail": last_runtime_error_tail,
        "heavy_runner_error_context": heavy_runner_error_context,
        "last_gate_context": last_gate_context or {},
        "runtime_fix_terminal": runtime_fix_terminal,
        "runtime_fix_count": runtime_fix_count,
        "execution_contract": {"contract_version": "4.1"},
        "output_contract_report": {"overall_status": "ok"},
    }
    if extra:
        state.update(extra)
    return state


@pytest.fixture(autouse=True)
def _patch_json_load():
    """Stub out _load_json_safe so the test never touches the filesystem."""
    with patch(
        "src.graph.graph._load_json_safe",
        return_value={},
    ):
        yield


class TestReviewBoardRuntimeStatusAfterRetry:
    """The critical scenario: execution succeeds but stale error fields linger."""

    def test_ok_when_execution_clean_despite_stale_gate_context(self):
        """Residual traceback in last_gate_context must NOT cause FAILED_RUNTIME
        when execution_error is False and execution_output is clean."""
        from src.graph.graph import _build_review_board_facts

        state = _make_state(
            execution_error=False,
            sandbox_failed=False,
            execution_output="HEAVY_RUNNER: status=success reason=local_runner_mode",
            last_gate_context={
                "traceback": "Traceback (most recent call last):\n  File ...\nKeyError: 'prob_12h'",
                "runtime_error": {"summary": "KeyError during monotonic correction"},
                "execution_output_tail": "Traceback (most recent call last):\n...",
            },
        )
        facts = _build_review_board_facts(state)
        assert facts["runtime"]["status"] == "OK", (
            "Clean execution must produce OK runtime status even with stale "
            "gate_context traceback from a previous failed attempt"
        )

    def test_ok_when_execution_clean_despite_stale_heavy_runner_error(self):
        """Residual heavy_runner_error_context must NOT cause FAILED_RUNTIME."""
        from src.graph.graph import _build_review_board_facts

        state = _make_state(
            execution_error=False,
            sandbox_failed=False,
            heavy_runner_error_context={
                "error": "Script failed with exit_code=1",
                "traceback": "Traceback (most recent call last):\n...",
            },
        )
        facts = _build_review_board_facts(state)
        assert facts["runtime"]["status"] == "OK"

    def test_ok_when_execution_clean_despite_stale_runtime_error_tail(self):
        """Residual last_runtime_error_tail must NOT cause FAILED_RUNTIME
        when _clear_runtime_blockers was not called (defensive)."""
        from src.graph.graph import _build_review_board_facts

        state = _make_state(
            execution_error=False,
            sandbox_failed=False,
            last_runtime_error_tail="Traceback (most recent call last):\n...",
        )
        facts = _build_review_board_facts(state)
        assert facts["runtime"]["status"] == "OK"

    def test_failed_when_execution_error_is_true(self):
        """When execution_error is True, FAILED_RUNTIME is correct."""
        from src.graph.graph import _build_review_board_facts

        state = _make_state(
            execution_error=True,
            execution_output="Traceback (most recent call last):\n  File ...\nRuntimeError: boom",
            last_runtime_error_tail="RuntimeError: boom",
        )
        facts = _build_review_board_facts(state)
        assert facts["runtime"]["status"] == "FAILED_RUNTIME"

    def test_failed_when_sandbox_failed(self):
        """When sandbox_failed is True, FAILED_RUNTIME is correct."""
        from src.graph.graph import _build_review_board_facts

        state = _make_state(
            sandbox_failed=True,
            execution_output="Sandbox Execution Failed",
        )
        facts = _build_review_board_facts(state)
        assert facts["runtime"]["status"] == "FAILED_RUNTIME"

    def test_failed_when_execution_output_has_traceback(self):
        """When the current execution_output itself has a Traceback marker,
        FAILED_RUNTIME is correct (this is a real failure, not stale)."""
        from src.graph.graph import _build_review_board_facts

        state = _make_state(
            execution_error=False,
            execution_output="Traceback (most recent call last):\n  File ...\nKeyError: 'x'",
        )
        facts = _build_review_board_facts(state)
        assert facts["runtime"]["status"] == "FAILED_RUNTIME"


class TestClearRuntimeBlockers:
    """Ensure _clear_runtime_blockers cleans all residual error fields."""

    def test_clears_heavy_runner_error_context(self):
        from src.graph.graph import _clear_runtime_blockers

        cleared = _clear_runtime_blockers()
        assert "heavy_runner_error_context" in cleared
        assert cleared["heavy_runner_error_context"] is None

    def test_clears_core_fields(self):
        from src.graph.graph import _clear_runtime_blockers

        cleared = _clear_runtime_blockers()
        assert cleared["execution_error"] is False
        assert cleared["sandbox_failed"] is False
        assert cleared["runtime_fix_terminal"] is False
        assert cleared["last_runtime_error_tail"] is None
