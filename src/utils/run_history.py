"""Run history utilities — scan completed runs and extract summary metadata."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

_RUNS_DIR = os.path.join(
    os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "runs",
)


def _load_json_safe(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _extract_run_summary(run_id: str, run_dir: str) -> Optional[Dict[str, Any]]:
    """Extract summary metadata from a single run directory."""
    status_path = os.path.join(run_dir, "worker_status.json")
    status = _load_json_safe(status_path)
    if not status:
        return None

    run_status = status.get("status", "unknown")
    if run_status not in ("complete", "error", "aborted"):
        # Skip in-progress runs
        pid = status.get("pid")
        if pid:
            return None

    started_at = status.get("started_at")
    started_str = ""
    if isinstance(started_at, (int, float)) and started_at > 0:
        try:
            started_str = datetime.fromtimestamp(started_at).strftime("%d/%m/%Y %H:%M")
        except Exception:
            pass

    # Try to extract metrics from final state
    final_state = _load_json_safe(os.path.join(run_dir, "worker_final_state.json"))
    strategy_title = ""
    metric_name = ""
    metric_value = ""
    iteration_count = 0
    review_verdict = ""

    if isinstance(final_state, dict):
        selected = final_state.get("selected_strategy")
        if isinstance(selected, dict):
            strategy_title = selected.get("title", "")
        iteration_count = final_state.get("iteration_count", 0) or 0
        review_verdict = final_state.get("review_verdict", "")

    # Try metric from status (last reported during run)
    metric_name = status.get("metric_name", "")
    metric_value = status.get("metric_value", "")

    # Try to get elapsed time
    elapsed = ""
    if isinstance(started_at, (int, float)) and started_at > 0:
        ended = status.get("ended_at")
        if isinstance(ended, (int, float)) and ended > started_at:
            secs = int(ended - started_at)
        else:
            # Estimate from file modification time
            try:
                mtime = os.path.getmtime(status_path)
                secs = int(mtime - started_at)
            except Exception:
                secs = 0
        if secs > 0:
            m, s = divmod(secs, 60)
            h, m = divmod(m, 60)
            elapsed = f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

    return {
        "run_id": run_id,
        "status": run_status,
        "started_at": started_at or 0,
        "started_str": started_str,
        "elapsed": elapsed,
        "strategy": strategy_title,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "iterations": iteration_count,
        "verdict": review_verdict,
    }


def list_runs(runs_dir: str = _RUNS_DIR, limit: int = 20) -> List[Dict[str, Any]]:
    """List recent runs sorted by start time (newest first)."""
    if not os.path.isdir(runs_dir):
        return []

    summaries: List[Dict[str, Any]] = []
    try:
        entries = os.listdir(runs_dir)
    except OSError:
        return []

    for entry in entries:
        if entry in ("latest", "archive", ".gitkeep"):
            continue
        run_dir = os.path.join(runs_dir, entry)
        if not os.path.isdir(run_dir):
            continue
        summary = _extract_run_summary(entry, run_dir)
        if summary:
            summaries.append(summary)

    summaries.sort(key=lambda r: r.get("started_at", 0), reverse=True)
    return summaries[:limit]


def load_run_result(run_id: str, runs_dir: str = _RUNS_DIR) -> Optional[Dict[str, Any]]:
    """Load the full final state of a specific run."""
    run_dir = os.path.join(runs_dir, run_id)
    return _load_json_safe(os.path.join(run_dir, "worker_final_state.json"))
