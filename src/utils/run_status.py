"""Shared status protocol between background worker and Streamlit UI.

The worker writes status/log files; Streamlit polls them.
All writes are atomic (write-to-temp + os.replace) to prevent partial reads.
"""

import json
import os
import tempfile
import time
from datetime import datetime
from typing import Any, Dict, List, Optional


# Absolute paths — immune to os.chdir() inside the graph workspace
from src.utils.paths import PROJECT_ROOT as _PROJECT_ROOT, RUNS_DIR
from src.utils.sandbox_config import normalize_sandbox_config


def _status_path(run_id: str) -> str:
    return os.path.join(RUNS_DIR, run_id, "worker_status.json")


def _log_path(run_id: str) -> str:
    return os.path.join(RUNS_DIR, run_id, "worker_log.jsonl")


def _final_state_path(run_id: str) -> str:
    return os.path.join(RUNS_DIR, run_id, "worker_final_state.json")


def _atomic_write_json(path: str, data: Dict[str, Any]) -> None:
    """Write JSON atomically: write to temp file then replace.

    On Windows, os.replace can fail with PermissionError when another
    process (e.g. Streamlit polling) holds the target file open.  We
    retry a few times with a short sleep to ride out the lock.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dir_name = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        for attempt in range(5):
            try:
                os.replace(tmp_path, path)
                return
            except PermissionError:
                if attempt < 4:
                    time.sleep(0.1 * (attempt + 1))
                else:
                    raise
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Writer API (used by background_worker.py)
# ---------------------------------------------------------------------------

def write_status(
    run_id: str,
    *,
    status: str = "running",
    stage: Optional[str] = None,
    stage_name: Optional[str] = None,
    progress: int = 0,
    iteration: int = 0,
    max_iterations: int = 6,
    metric_name: str = "",
    metric_value: str = "",
    completed_steps: Optional[List[str]] = None,
    error: Optional[str] = None,
    pid: Optional[int] = None,
    started_at: Optional[float] = None,
) -> None:
    data = {
        "pid": pid or os.getpid(),
        "status": status,
        "stage": stage,
        "stage_name": stage_name or "",
        "progress": progress,
        "iteration": iteration,
        "max_iterations": max_iterations,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "completed_steps": completed_steps or [],
        "error": error,
        "started_at": started_at or time.time(),
        "updated_at": time.time(),
    }
    _atomic_write_json(_status_path(run_id), data)


def append_log(run_id: str, agent: str, message: str, level: str = "info") -> None:
    path = _log_path(run_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    entry = {
        "ts": datetime.now().strftime("%H:%M:%S"),
        "agent": agent,
        "msg": message,
        "level": level,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _resolve_authoritative_review_verdict(state: Dict[str, Any]) -> str:
    board_payload = state.get("review_board_verdict")
    if isinstance(board_payload, dict):
        verdict = str(
            board_payload.get("final_review_verdict")
            or board_payload.get("status")
            or ""
        ).strip()
        if verdict:
            return verdict
    return str(state.get("review_verdict") or "").strip()


def write_final_state(run_id: str, state: Dict[str, Any]) -> None:
    """Serialize the final graph state for the results dashboard."""
    # Only keep JSON-serializable fields that the UI needs.
    # P3 fix: expanded whitelist — previous version lost observability fields
    # that the dashboard and governance summary depend on.
    view_state = dict(state if isinstance(state, dict) else {})
    authoritative_verdict = _resolve_authoritative_review_verdict(view_state)
    if authoritative_verdict:
        view_state["review_verdict"] = authoritative_verdict
    keys_to_keep = [
        # Core identification
        "run_id", "business_objective", "csv_path",
        # Review & governance
        "review_verdict", "last_successful_review_verdict",
        "review_board_verdict", "gate_status",
        "run_outcome", "overall_status_global", "hard_failures", "failed_gates",
        "budget_counters", "iteration_count", "current_iteration",
        # Strategy
        "selected_strategy", "strategies", "selection_reason",
        "data_summary", "domain_expert_reviews",
        # Execution contract
        "execution_contract",
        # Data Engineer
        "cleaning_code", "cleaned_data_preview",
        # ML Engineer
        "generated_code", "last_generated_code",
        "execution_output", "last_successful_execution_output",
        # Report & artifacts
        "final_report", "pdf_path", "output_contract_report",
        "execution_feedback", "artifact_paths",
        "work_dir", "work_dir_abs",
        # Pipeline status
        "data_engineer_failed", "pipeline_aborted_reason",
        "ml_improvement_kept", "stop_reason",
    ]
    serializable = {}
    for k in keys_to_keep:
        v = view_state.get(k)
        if v is not None:
            try:
                json.dumps(v, ensure_ascii=False)
                serializable[k] = v
            except (TypeError, ValueError):
                serializable[k] = str(v)
    _atomic_write_json(_final_state_path(run_id), serializable)


def write_worker_input(
    run_id: str,
    csv_path: str,
    business_objective: str,
    sandbox_config: Optional[Dict[str, Any]] = None,
) -> str:
    """Write the input parameters for the worker to read."""
    run_dir = os.path.join(RUNS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    input_path = os.path.join(run_dir, "worker_input.json")
    _atomic_write_json(input_path, {
        "csv_path": os.path.abspath(csv_path),
        "business_objective": business_objective,
        "sandbox_config": normalize_sandbox_config(sandbox_config),
    })
    return input_path


# ---------------------------------------------------------------------------
# Reader API (used by app.py / Streamlit)
# ---------------------------------------------------------------------------

def read_status(run_id: str) -> Optional[Dict[str, Any]]:
    path = _status_path(run_id)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def read_log_entries(run_id: str, after_line: int = 0) -> List[Dict[str, Any]]:
    """Read log entries, optionally skipping the first `after_line` lines."""
    path = _log_path(run_id)
    entries = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < after_line:
                    continue
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except FileNotFoundError:
        pass
    return entries


def read_final_state(run_id: str) -> Optional[Dict[str, Any]]:
    path = _final_state_path(run_id)
    try:
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    # Enrich with business_objective and csv_path from worker_input
    try:
        run_dir = os.path.join(RUNS_DIR, run_id)
        wi_path = os.path.join(run_dir, "worker_input.json")
        with open(wi_path, "r", encoding="utf-8") as f:
            worker_input = json.load(f)
        if isinstance(worker_input, dict):
            if not state.get("business_objective"):
                state["business_objective"] = worker_input.get("business_objective", "")
            if not state.get("csv_path"):
                state["csv_path"] = worker_input.get("csv_path", "")
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return state


def get_active_run_id() -> Optional[str]:
    """Check if there's an active (running) worker."""
    latest_path = os.path.join(RUNS_DIR, "latest", "run_id.txt")
    try:
        with open(latest_path, "r", encoding="utf-8") as f:
            run_id = f.read().strip()
    except FileNotFoundError:
        return None
    if not run_id:
        return None

    status = read_status(run_id)
    if not status:
        return None
    if status.get("status") != "running":
        return None

    # Verify the worker process is still alive
    pid = status.get("pid")
    if pid and is_process_alive(pid):
        return run_id
    return None


def _abort_flag_path(run_id: str) -> str:
    return os.path.join(RUNS_DIR, run_id, "abort_requested")


def request_run_abort(run_id: str) -> None:
    """Signal the background worker to abort by creating a flag file."""
    path = _abort_flag_path(run_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(time.time()))


def is_run_abort_requested(run_id: str) -> bool:
    """Check if an abort has been requested for this run."""
    return os.path.exists(_abort_flag_path(run_id))


def kill_worker(run_id: str) -> bool:
    """Kill the worker process for a run. Returns True if killed."""
    status = read_status(run_id)
    if not status:
        return False
    pid = status.get("pid")
    if not pid:
        return False
    if not is_process_alive(pid):
        return False
    try:
        import psutil
        proc = psutil.Process(pid)
        proc.terminate()
        proc.wait(timeout=5)
        return True
    except ImportError:
        pass
    except Exception:
        pass
    # Fallback: OS kill
    try:
        if os.name == "nt":
            import subprocess as _sp
            _sp.run(["taskkill", "/F", "/PID", str(pid)],
                    capture_output=True, timeout=5)
            return True
        else:
            import signal as _sig
            os.kill(pid, _sig.SIGTERM)
            return True
    except Exception:
        return False


def is_process_alive(pid: int) -> bool:
    """Check if a process with the given PID is still running."""
    try:
        import psutil
        return psutil.pid_exists(pid) and psutil.Process(pid).is_running()
    except ImportError:
        pass
    # Fallback: OS-level check
    try:
        if os.name == "nt":
            import subprocess
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True, timeout=5,
            )
            return str(pid) in result.stdout
        else:
            os.kill(pid, 0)
            return True
    except (OSError, subprocess.TimeoutExpired):
        return False
