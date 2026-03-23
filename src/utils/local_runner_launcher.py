import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional


class LocalRunnerLaunchError(RuntimeError):
    pass


def _normalize_rel_path(path: Any) -> str:
    value = str(path or "").strip()
    if not value:
        return ""
    return value.lstrip("/").replace("\\", "/")


def _normalize_scope_segment(value: Any) -> str:
    raw = str(value or "").strip().strip("/")
    if not raw:
        return ""
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", raw)
    slug = slug.strip("._-")
    return slug.lower()


def _copy_file(src: str, dst: str) -> None:
    if os.path.isdir(src):
        # Directory artifact (e.g., "artifacts/ml/models/"): copy entire tree
        if os.path.exists(dst):
            shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)
        return
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    shutil.copy2(src, dst)


def _load_json_if_exists(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {"raw": payload}
    except Exception:
        return None


def _list_local_files(root: str) -> list[str]:
    if not os.path.isdir(root):
        return []
    found: list[str] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root).replace("\\", "/")
            found.append(rel)
    return sorted(found)


def _ensure_trailing_sep(path: str) -> str:
    if not path:
        return path
    return path if path.endswith(os.sep) else path + os.sep


# ---------------------------------------------------------------------------
# P0 FIX 4: Large-file extensions whose intermediate outputs should be cleaned
# after a successful attempt to avoid multi-GB duplication across attempts.
# ---------------------------------------------------------------------------
_LARGE_DATA_EXTENSIONS: set = {".csv", ".parquet", ".xlsx", ".xls", ".pkl", ".pickle", ".h5", ".hdf5"}
_CLEANUP_SIZE_THRESHOLD: int = 10 * 1024 * 1024  # 10 MB


def _cleanup_previous_attempts(stage_dir: str, current_attempt: int) -> None:
    """Remove large data files from previous attempts' output directories.

    Only deletes files with data extensions above the size threshold.
    Keeps small files (status.json, error.json, request.json, logs) for debugging.
    """
    if current_attempt < 1 or not os.path.isdir(stage_dir):
        return
    for prev in range(current_attempt):
        prev_output = os.path.join(stage_dir, f"attempt_{prev}", "output")
        if not os.path.isdir(prev_output):
            continue
        for dirpath, _, filenames in os.walk(prev_output):
            for name in filenames:
                ext = os.path.splitext(name)[1].lower()
                if ext not in _LARGE_DATA_EXTENSIONS:
                    continue
                full = os.path.join(dirpath, name)
                try:
                    if os.path.getsize(full) > _CLEANUP_SIZE_THRESHOLD:
                        os.remove(full)
                except Exception:
                    pass


def launch_local_runner_job(
    *,
    run_id: str,
    request: Dict[str, Any],
    dataset_path: str,
    bucket: str,
    job: str,
    region: str,
    input_prefix: str = "inputs",
    output_prefix: str = "outputs",
    dataset_prefix: str = "datasets",
    project: Optional[str] = None,
    download_map: Optional[Dict[str, str]] = None,
    wait: bool = True,
    code_text: Optional[str] = None,
    support_files: Optional[list[Dict[str, str]]] = None,
    data_path: Optional[str] = None,
    required_artifacts: Optional[list[str]] = None,
    attempt_id: Optional[int] = None,
    stage_namespace: Optional[str] = None,
) -> Dict[str, Any]:
    _ = (bucket, job, region, input_prefix, output_prefix, dataset_prefix, project, wait)
    if not dataset_path or not os.path.exists(dataset_path):
        raise LocalRunnerLaunchError("dataset_path missing or does not exist for local runner")

    stage_scope = _normalize_scope_segment(stage_namespace) or "generic"
    attempt_num = 0
    try:
        if attempt_id is not None:
            attempt_num = max(0, int(attempt_id))
    except Exception:
        attempt_num = 0

    attempt_scope = f"attempt_{attempt_num}" if attempt_num > 0 else "attempt_0"
    # P0 FIX: Resolve from PROJECT ROOT, not cwd.
    # When cwd is runs/<run_id>/work/, the old relative path created a
    # self-referencing loop: runs/<id>/work/runs/<id>/sandbox/... (14+ GB).
    # Using the same project-root derivation as heavy_train.py (line 140).
    _project_root = str(Path(__file__).resolve().parents[2])
    base_dir = os.path.join(
        _project_root, "runs", str(run_id or "unknown"), "sandbox", "local_runner", stage_scope, attempt_scope
    )
    input_dir = os.path.join(base_dir, "input")
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    request_payload = dict(request or {})
    request_payload["dataset_uri"] = os.path.abspath(dataset_path)

    if code_text is not None:
        if not data_path:
            raise LocalRunnerLaunchError("data_path is required when code_text is provided")
        code_uri = os.path.abspath(os.path.join(input_dir, "ml_script.py"))
        with open(code_uri, "w", encoding="utf-8") as f_code:
            f_code.write(code_text or "")
        request_payload["code_uri"] = code_uri
        request_payload["data_path"] = data_path

        uploaded_support: list[Dict[str, str]] = []
        for item in support_files or []:
            if not isinstance(item, dict):
                continue
            local_path = str(item.get("local_path") or "").strip()
            rel_path = _normalize_rel_path(item.get("path"))
            if not local_path or not rel_path or not os.path.exists(local_path):
                continue
            support_uri = os.path.abspath(os.path.join(input_dir, "support", rel_path))
            _copy_file(local_path, support_uri)
            uploaded_support.append({"uri": support_uri, "path": rel_path})
        if uploaded_support:
            request_payload["support_files"] = uploaded_support

    request_payload["output_uri"] = _ensure_trailing_sep(os.path.abspath(output_dir))
    input_uri = os.path.abspath(os.path.join(input_dir, "request.json"))
    with open(input_uri, "w", encoding="utf-8") as f_req:
        json.dump(request_payload, f_req, indent=2, ensure_ascii=True)

    repo_root = Path(__file__).resolve().parents[2]
    runner_script = repo_root / "cloudrun" / "heavy_runner" / "heavy_train.py"
    if not runner_script.exists():
        raise LocalRunnerLaunchError(f"Local runner script not found: {runner_script}")

    env = dict(os.environ)
    env["INPUT_URI"] = input_uri
    env["OUTPUT_URI"] = request_payload["output_uri"]

    try:
        proc = subprocess.run(
            [sys.executable, str(runner_script)],
            cwd=str(repo_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except Exception as exc:
        raise LocalRunnerLaunchError(f"Local runner execution failed to start: {exc}") from exc

    job_failed = proc.returncode != 0
    job_error_msg = ""
    if job_failed:
        stderr_tail = str(proc.stderr or "")[-3000:]
        stdout_tail = str(proc.stdout or "")[-3000:]
        details = stderr_tail or stdout_tail
        job_error_msg = f"local_runner_exit_code={proc.returncode}"
        if details:
            job_error_msg += f" | tail={details}"

    downloaded: Dict[str, str] = {}
    output_root = request_payload["output_uri"]
    if download_map:
        for filename, local_path in download_map.items():
            rel_name = _normalize_rel_path(filename)
            if not rel_name:
                continue
            src = os.path.join(output_root, rel_name)
            if not os.path.exists(src):
                continue
            _copy_file(src, local_path)
            downloaded[rel_name] = local_path

    for artifact in required_artifacts or []:
        rel_name = _normalize_rel_path(artifact)
        if not rel_name or rel_name in downloaded:
            continue
        src = os.path.join(output_root, rel_name)
        if not os.path.exists(src):
            continue
        local_path = rel_name
        if isinstance(download_map, dict):
            local_path = str(download_map.get(rel_name) or local_path)
        _copy_file(src, local_path)
        downloaded[rel_name] = local_path

    plots_root = os.path.join(output_root, "static", "plots")
    if os.path.isdir(plots_root):
        for name in os.listdir(plots_root):
            src = os.path.join(plots_root, name)
            if not os.path.isfile(src):
                continue
            rel_path = f"static/plots/{name}"
            if rel_path in downloaded:
                continue
            _copy_file(src, rel_path)
            downloaded[rel_path] = rel_path

    status_payload = _load_json_if_exists(os.path.join(output_root, "status.json"))
    status_ok = bool(isinstance(status_payload, dict) and status_payload.get("ok"))
    error_payload = _load_json_if_exists(os.path.join(output_root, "error.json"))

    missing_artifacts: list[str] = []
    for artifact in required_artifacts or []:
        rel_name = _normalize_rel_path(artifact)
        if rel_name and rel_name not in downloaded:
            missing_artifacts.append(rel_name)
    required_artifacts_missing = bool(missing_artifacts)

    if required_artifacts_missing and not error_payload:
        error_payload = {
            "error": "required_artifacts_missing",
            "missing": list(missing_artifacts),
            "required": [
                _normalize_rel_path(path)
                for path in (required_artifacts or [])
                if _normalize_rel_path(path)
            ],
            "downloaded": list(downloaded.keys()),
        }

    local_listing = _list_local_files(output_root)
    if job_failed and not error_payload:
        error_payload = {
            "error": "job_execution_failed",
            "job_error": job_error_msg,
            "downloaded": list(downloaded.keys()),
            "local_listing": local_listing,
            "message": f"Local runner execution failed: {job_error_msg[:500]}",
        }

    raw_job_failed = job_failed
    raw_job_error = job_error_msg
    raw_error_payload = error_payload
    status_arbitration: Dict[str, Any] | None = None

    # Defence-in-depth: when the runner reports ok=true, trust its verdict even
    # if the launcher detects missing artifacts (they may belong to a different
    # pipeline stage, e.g. data/cleaned_data.csv is a DE artifact).
    if status_ok and (job_failed or error_payload):
        status_arbitration = {
            "applied": True,
            "reason": "status_ok_override",
            "ignored_error_payload": bool(error_payload),
            "ignored_job_failure": bool(job_failed),
            "missing_artifacts_at_arbitration": list(missing_artifacts),
        }
        job_failed = False
        job_error_msg = ""
        error_payload = None

    has_error = bool(error_payload) or job_failed

    # P0 FIX 4: After a successful attempt, clean large data files from
    # previous attempts to avoid multi-GB duplication (e.g. scored_data.csv
    # at 1.25 GB × N attempts).
    if not has_error and attempt_num > 0:
        stage_dir = os.path.join(
            _project_root, "runs", str(run_id or "unknown"),
            "sandbox", "local_runner", stage_scope,
        )
        _cleanup_previous_attempts(stage_dir, attempt_num)

    return {
        "status": "error" if has_error else "success",
        "input_uri": input_uri,
        "output_uri": output_root,
        "dataset_uri": request_payload.get("dataset_uri"),
        "downloaded": downloaded,
        "missing_artifacts": missing_artifacts,
        "gcs_listing": local_listing,
        "job_stdout": proc.stdout,
        "job_stderr": proc.stderr,
        "job_failed": job_failed,
        "job_error": job_error_msg if job_failed else None,
        "job_failed_raw": raw_job_failed,
        "job_error_raw": raw_job_error,
        "gcloud_flag": "local_runner",
        "gcloud_bin": "local",
        "gsutil_bin": "local",
        "error": error_payload,
        "error_raw": raw_error_payload,
        "status_payload": status_payload,
        "status_ok": status_ok,
        "status_arbitration": status_arbitration,
        "required_artifacts_missing": required_artifacts_missing,
        "attempt_id": attempt_num if attempt_num > 0 else None,
        "stage_namespace": stage_scope or None,
    }
