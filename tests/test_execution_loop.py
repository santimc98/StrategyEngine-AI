
import pytest
from unittest.mock import MagicMock, patch
from src.graph.graph import check_execution_status, check_engineer_success, AgentState

def test_check_execution_status_retry():
    state = {
        "execution_output": "Traceback (most recent call last):\nValueError: bad",
        "execution_attempt": 1
    }
    assert check_execution_status(state) == "retry_fix"

def test_check_execution_status_evaluate():
    state = {
        "execution_output": "Success",
        "execution_attempt": 1
    }
    assert check_execution_status(state) == "evaluate"

def test_check_execution_status_max_retries():
    state = {
        "execution_output": "Traceback (most recent call last):\nValueError: bad",
        "execution_attempt": 4,
        "runtime_fix_count": 3,
        "max_runtime_fix_attempts": 3,
    }
    assert check_execution_status(state) == "failed_runtime"


def test_check_execution_status_sandbox_retry_until_limit_then_evaluate():
    retry_state = {
        "execution_output": "Sandbox Execution Failed: peer closed connection without sending complete message body",
        "sandbox_failed": True,
        "sandbox_retry_count": 1,
        "max_sandbox_retries": 2,
    }
    assert check_execution_status(retry_state) == "retry_sandbox"

    exhausted_state = {
        "execution_output": "Sandbox Execution Failed: peer closed connection without sending complete message body",
        "sandbox_failed": True,
        "sandbox_retry_count": 2,
        "max_sandbox_retries": 2,
    }
    assert check_execution_status(exhausted_state) == "evaluate"


def test_check_engineer_success_retries_host_crash_within_limit():
    state = {
        "error_message": "CRITICAL: ML Engineer crashed in host: TypeError",
        "ml_engineer_host_crash": True,
        "ml_engineer_host_crash_count": 1,
        "max_ml_engineer_host_retries": 1,
    }
    assert check_engineer_success(state) == "retry_host_crash"


def test_check_engineer_success_fails_after_host_crash_retry_limit():
    state = {
        "error_message": "CRITICAL: ML Engineer crashed in host: TypeError",
        "ml_engineer_host_crash": True,
        "ml_engineer_host_crash_count": 2,
        "max_ml_engineer_host_retries": 1,
    }
    assert check_engineer_success(state) == "failed"


def test_check_engineer_success_routes_to_metric_round_finalize_on_budget_exceeded():
    state = {
        "error_message": "BUDGET_EXCEEDED: ML Engineer exceeded 6/6",
        "ml_improvement_round_active": True,
    }
    assert check_engineer_success(state) == "finalize_metric_round"
