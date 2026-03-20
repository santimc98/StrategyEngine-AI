"""Single source of truth for all project paths.

Every module that needs to resolve project directories MUST import from here
instead of computing its own root. This guarantees consistency regardless of
the current working directory, the entry point (Streamlit, background worker,
tests, or a third-party integration), or the deployment environment.

Enterprise deployments can override any root directory via environment
variables without touching application code:

    PROJECT_ROOT  – override the project root (default: auto-detected)
    RUNS_DIR      – override the runs storage directory
    DATA_DIR      – override the shared data directory
"""

import os


def _detect_project_root() -> str:
    """Compute project root from this file's location (src/utils/paths.py → ../..)."""
    return os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    )


# ---------------------------------------------------------------------------
# Core paths — all absolute, all immune to os.chdir()
# ---------------------------------------------------------------------------

PROJECT_ROOT: str = os.path.abspath(
    os.environ.get("PROJECT_ROOT") or _detect_project_root()
)

RUNS_DIR: str = os.path.abspath(
    os.environ.get("RUNS_DIR") or os.path.join(PROJECT_ROOT, "runs")
)

DATA_DIR: str = os.path.abspath(
    os.environ.get("DATA_DIR") or os.path.join(PROJECT_ROOT, "data")
)

STATIC_DIR: str = os.path.join(PROJECT_ROOT, "static")

ARCHIVE_DIR: str = os.path.join(RUNS_DIR, "archive")

LATEST_DIR: str = os.path.join(RUNS_DIR, "latest")


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def run_dir(run_id: str) -> str:
    """Absolute path to a specific run's bundle directory."""
    return os.path.join(RUNS_DIR, run_id)


def run_work_dir(run_id: str) -> str:
    """Absolute path to a run's isolated workspace."""
    return os.path.join(RUNS_DIR, run_id, "work")


def data_file(filename: str) -> str:
    """Absolute path to a file in the shared data/ directory."""
    return os.path.join(DATA_DIR, filename)
