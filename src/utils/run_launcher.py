from __future__ import annotations

import os
import subprocess
import sys
import uuid
from typing import Any, Dict, Optional

from src.utils.paths import PROJECT_ROOT, run_dir
from src.utils.run_status import (
    get_active_run_id,
    kill_worker,
    request_run_abort,
    write_worker_input,
)
from src.utils.run_storage import init_run_dir


class ActiveRunConflictError(RuntimeError):
    def __init__(self, active_run_id: str):
        super().__init__(f"An active run is already running: {active_run_id}")
        self.active_run_id = active_run_id


def start_background_run(
    csv_path: str,
    business_objective: str,
    sandbox_config: Optional[Dict[str, Any]] = None,
    *,
    replace_active_run: bool = False,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Start a background worker run without depending on Streamlit.

    This is the API-friendly launcher that the future frontend should call
    through a backend endpoint. It preserves the current worker protocol and
    reuses the same run bundle layout as the Streamlit UI.
    """
    active_run_id = get_active_run_id()
    replaced_run_id: Optional[str] = None
    if active_run_id:
        if replace_active_run:
            request_run_abort(active_run_id)
            kill_worker(active_run_id)
            replaced_run_id = active_run_id
        else:
            raise ActiveRunConflictError(active_run_id)

    resolved_run_id = run_id or uuid.uuid4().hex[:8]
    init_run_dir(resolved_run_id)
    write_worker_input(
        resolved_run_id,
        csv_path,
        business_objective,
        sandbox_config=sandbox_config,
    )

    worker_log_dir = run_dir(resolved_run_id)
    os.makedirs(worker_log_dir, exist_ok=True)
    worker_stdout_path = os.path.join(worker_log_dir, "worker_stdout.log")
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0

    worker_stdout = open(worker_stdout_path, "w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "src.utils.background_worker", resolved_run_id],
            cwd=PROJECT_ROOT,
            stdout=worker_stdout,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )
    finally:
        worker_stdout.close()

    return {
        "run_id": resolved_run_id,
        "pid": process.pid,
        "worker_stdout_path": worker_stdout_path,
        "replaced_run_id": replaced_run_id,
    }
