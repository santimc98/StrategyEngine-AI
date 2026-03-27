import json
import os
import re
import shutil
import subprocess
import tempfile
from typing import Any, Dict, Optional, Tuple


class CloudRunLaunchError(RuntimeError):
    pass


def _resolve_cli_override(binary: str, env_var: str) -> str:
    override = os.getenv(env_var)
    if not override:
        return binary
    override = override.strip().strip('"')
    if os.path.isdir(override):
        candidates = [os.path.join(override, binary)]
        if os.name == "nt":
            candidates.insert(0, os.path.join(override, f"{binary}.cmd"))
            candidates.insert(1, os.path.join(override, f"{binary}.exe"))
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        os.environ["PATH"] = override + os.pathsep + os.environ.get("PATH", "")
        return binary
    if os.path.exists(override):
        return override
    return override


def _ensure_cli(binary: str) -> None:
    has_sep = os.path.sep in binary or (os.path.altsep and os.path.altsep in binary)
    if os.path.isabs(binary) or has_sep:
        if not os.path.exists(binary):
            raise CloudRunLaunchError(f"Required CLI not found: {binary}")
        return
    if not shutil.which(binary):
        raise CloudRunLaunchError(f"Required CLI not found: {binary}")


def _run_cmd(args: list[str], timeout_s: Optional[int] = None) -> Tuple[str, str]:
    proc = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if proc.returncode != 0:
        raise CloudRunLaunchError(
            f"Command failed ({proc.returncode}): {' '.join(args)}\n"
            f"STDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
        )
    return proc.stdout, proc.stderr


def _run_gcloud_job_execute(
    *,
    gcloud_bin: str,
    job: str,
    region: str,
    env_vars: str,
    project: Optional[str],
    wait: bool,
) -> Tuple[str, str, str]:
    cmd = [gcloud_bin, "run", "jobs", "execute", job, "--region", region, "--update-env-vars", env_vars]
    if project:
        cmd.extend(["--project", project])
    if wait:
        cmd.append("--wait")
    try:
        stdout, stderr = _run_cmd(cmd)
        return stdout, stderr, "update-env-vars"
    except CloudRunLaunchError as exc:
        err_text = str(exc)
        if "unrecognized arguments" not in err_text or "--update-env-vars" not in err_text:
            raise
    cmd = [gcloud_bin, "run", "jobs", "execute", job, "--region", region, "--set-env-vars", env_vars]
    if project:
        cmd.extend(["--project", project])
    if wait:
        cmd.append("--wait")
    stdout, stderr = _run_cmd(cmd)
    return stdout, stderr, "set-env-vars"


def _gsutil_exists(uri: str, gsutil_bin: str) -> bool:
    _ensure_cli(gsutil_bin)
    proc = subprocess.run(
        [gsutil_bin, "-q", "stat", uri],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return proc.returncode == 0


def _gsutil_cp(src: str, dest: str, gsutil_bin: str) -> None:
    _ensure_cli(gsutil_bin)
    _run_cmd([gsutil_bin, "-q", "cp", src, dest])


def _gsutil_cat(uri: str, gsutil_bin: str) -> str:
    _ensure_cli(gsutil_bin)
    stdout, _ = _run_cmd([gsutil_bin, "cat", uri])
    return stdout


def _gsutil_ls(uri: str, gsutil_bin: str) -> list[str]:
    """List files at a GCS URI. Returns empty list on error or if nothing found."""
    _ensure_cli(gsutil_bin)
    try:
        proc = subprocess.run(
            [gsutil_bin, "ls", uri],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
            check=False,
        )
        if proc.returncode != 0:
            return []
        return [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    except Exception:
        return []


def _upload_json_to_gcs(payload: Dict[str, Any], uri: str, gsutil_bin: str) -> None:
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", encoding="utf-8") as tmp:
        json.dump(payload, tmp, indent=2, ensure_ascii=True)
        tmp_path = tmp.name
    try:
        _gsutil_cp(tmp_path, uri, gsutil_bin)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _upload_text_to_gcs(text: str, uri: str, gsutil_bin: str) -> None:
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".py", encoding="utf-8") as tmp:
        tmp.write(text or "")
        tmp_path = tmp.name
    try:
        _gsutil_cp(tmp_path, uri, gsutil_bin)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _normalize_prefix(prefix: str) -> str:
    prefix = str(prefix or "").strip().strip("/")
    return prefix


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


def launch_heavy_runner_job(
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
    gcloud_bin: Optional[str] = None,
    gsutil_bin: Optional[str] = None,
) -> Dict[str, Any]:
    gcloud_bin = str(gcloud_bin or "").strip() or _resolve_cli_override("gcloud", "HEAVY_RUNNER_GCLOUD_BIN")
    gsutil_bin = str(gsutil_bin or "").strip() or _resolve_cli_override("gsutil", "HEAVY_RUNNER_GSUTIL_BIN")
    _ensure_cli(gcloud_bin)
    _ensure_cli(gsutil_bin)

    input_prefix = _normalize_prefix(input_prefix)
    output_prefix = _normalize_prefix(output_prefix)
    dataset_prefix = _normalize_prefix(dataset_prefix)
    run_scope = str(run_id or "unknown").strip().strip("/")
    stage_scope = _normalize_scope_segment(stage_namespace)
    if stage_scope:
        run_scope = f"{run_scope}/{stage_scope}"
    attempt_num = 0
    try:
        if attempt_id is not None:
            attempt_num = max(0, int(attempt_id))
    except Exception:
        attempt_num = 0
    if attempt_num > 0:
        run_scope = f"{run_scope}/attempt_{attempt_num}"

    dataset_uri = request.get("dataset_uri")
    if not dataset_uri or not str(dataset_uri).startswith("gs://"):
        if not dataset_path or not os.path.exists(dataset_path):
            raise CloudRunLaunchError("dataset_path missing or does not exist for heavy runner upload")
        dataset_name = os.path.basename(dataset_path)
        dataset_uri = f"gs://{bucket}/{dataset_prefix}/{run_scope}/{dataset_name}"
        _gsutil_cp(dataset_path, dataset_uri, gsutil_bin)
    request["dataset_uri"] = dataset_uri

    if code_text is not None:
        if not data_path:
            raise CloudRunLaunchError("data_path is required when code_text is provided")
        code_uri = f"gs://{bucket}/{input_prefix}/{run_scope}/ml_script.py"
        _upload_text_to_gcs(code_text, code_uri, gsutil_bin)
        request["code_uri"] = code_uri
        request["data_path"] = data_path

        uploaded_support = []
        if support_files:
            for item in support_files:
                if not isinstance(item, dict):
                    continue
                local_path = item.get("local_path")
                rel_path = item.get("path")
                if not local_path or not rel_path:
                    continue
                if not os.path.exists(local_path):
                    continue
                rel_path = str(rel_path).lstrip("/").replace("\\", "/")
                support_uri = f"gs://{bucket}/{input_prefix}/{run_scope}/support/{rel_path}"
                _gsutil_cp(local_path, support_uri, gsutil_bin)
                uploaded_support.append({"uri": support_uri, "path": rel_path})
        if uploaded_support:
            request["support_files"] = uploaded_support

    if attempt_num > 0:
        input_uri = f"gs://{bucket}/{input_prefix}/{run_scope}/request.json"
    else:
        input_uri = f"gs://{bucket}/{input_prefix}/{run_id}.json"
    output_uri = f"gs://{bucket}/{output_prefix}/{run_scope}/"
    request["output_uri"] = output_uri
    _upload_json_to_gcs(request, input_uri, gsutil_bin)

    env_vars = f"INPUT_URI={input_uri},OUTPUT_URI={output_uri}"

    # Execute job - capture failure but still attempt downloads
    job_failed = False
    job_error_msg = ""
    stdout = ""
    stderr = ""
    flag_used = ""
    try:
        stdout, stderr, flag_used = _run_gcloud_job_execute(
            gcloud_bin=gcloud_bin,
            job=job,
            region=region,
            env_vars=env_vars,
            project=project,
            wait=wait,
        )
    except CloudRunLaunchError as exc:
        # Job execution failed, but outputs may still exist in GCS
        # Attempt to download them for error analysis
        job_failed = True
        job_error_msg = str(exc)

    downloaded: Dict[str, str] = {}
    if download_map:
        for filename, local_path in download_map.items():
            rel_name = _normalize_rel_path(filename)
            if not rel_name:
                continue
            remote_path = output_uri + rel_name
            if not _gsutil_exists(remote_path, gsutil_bin):
                continue
            os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
            _gsutil_cp(remote_path, local_path, gsutil_bin)
            downloaded[rel_name] = local_path

    # Always attempt to fetch required artifacts even when they are absent from download_map.
    if required_artifacts:
        for artifact in required_artifacts:
            rel_name = _normalize_rel_path(artifact)
            if not rel_name or rel_name in downloaded:
                continue
            remote_path = output_uri + rel_name
            if not _gsutil_exists(remote_path, gsutil_bin):
                continue
            local_path = rel_name
            if isinstance(download_map, dict):
                local_path = str(download_map.get(rel_name) or local_path)
            os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
            _gsutil_cp(remote_path, local_path, gsutil_bin)
            downloaded[rel_name] = local_path

    # Always attempt to pull plots if present
    plot_listing = _gsutil_ls(output_uri + "static/plots/*", gsutil_bin)
    for uri in plot_listing:
        if not uri or uri.endswith("/"):
            continue
        rel_path = uri.replace(output_uri, "")
        if not rel_path or rel_path in downloaded:
            continue
        local_path = rel_path
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        _gsutil_cp(uri, local_path, gsutil_bin)
        downloaded[rel_path] = local_path

    status_payload: Optional[Dict[str, Any]] = None
    status_uri = output_uri + "status.json"
    if _gsutil_exists(status_uri, gsutil_bin):
        raw_status = ""
        try:
            raw_status = _gsutil_cat(status_uri, gsutil_bin)
            status_payload = json.loads(raw_status)
        except Exception:
            status_payload = {"ok": False, "error": "Failed to parse status.json"}
            if raw_status:
                status_payload["raw"] = raw_status
    status_ok = bool(isinstance(status_payload, dict) and status_payload.get("ok"))

    error_payload: Optional[Dict[str, Any]] = None
    error_uri = output_uri + "error.json"
    if _gsutil_exists(error_uri, gsutil_bin):
        raw_error = ""
        try:
            raw_error = _gsutil_cat(error_uri, gsutil_bin)
            error_payload = json.loads(raw_error)
        except Exception:
            error_payload = {"error": "Failed to parse error.json"}
            if raw_error:
                error_payload["raw"] = raw_error

    # Check for missing required artifacts
    missing_artifacts: list[str] = []
    if required_artifacts:
        for artifact in required_artifacts:
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

    # Build diagnostic info if job failed or artifacts are missing
    gcs_listing: list[str] = []

    # If job failed, include that in error_payload
    if job_failed and not error_payload:
        gcs_listing = _gsutil_ls(output_uri, gsutil_bin)
        error_payload = {
            "error": "job_execution_failed",
            "job_error": job_error_msg,
            "downloaded": list(downloaded.keys()),
            "gcs_listing": gcs_listing,
            "message": f"Cloud Run Job execution failed: {job_error_msg[:500]}",
        }

    raw_job_failed = job_failed
    raw_job_error = job_error_msg
    raw_error_payload = error_payload
    status_arbitration: Dict[str, Any] | None = None

    # Arbitration: if required outputs are present and status.json says ok,
    # prefer the success contract even if stale error markers exist.
    if status_ok and not missing_artifacts and (job_failed or error_payload):
        status_arbitration = {
            "applied": True,
            "reason": "status_ok_with_required_artifacts",
            "ignored_error_payload": bool(error_payload),
            "ignored_job_failure": bool(job_failed),
        }
        job_failed = False
        job_error_msg = ""
        error_payload = None

    # Determine overall status
    has_error = bool(error_payload) or job_failed

    return {
        "status": "error" if has_error else "success",
        "input_uri": input_uri,
        "output_uri": output_uri,
        "dataset_uri": dataset_uri,
        "downloaded": downloaded,
        "missing_artifacts": missing_artifacts,
        "gcs_listing": gcs_listing,
        "job_stdout": stdout,
        "job_stderr": stderr,
        "job_failed": job_failed,
        "job_error": job_error_msg if job_failed else None,
        "job_failed_raw": raw_job_failed,
        "job_error_raw": raw_job_error,
        "gcloud_flag": flag_used,
        "gcloud_bin": gcloud_bin,
        "gsutil_bin": gsutil_bin,
        "error": error_payload,
        "error_raw": raw_error_payload,
        "status_payload": status_payload,
        "status_ok": status_ok,
        "status_arbitration": status_arbitration,
        "required_artifacts_missing": required_artifacts_missing,
        "attempt_id": attempt_num if attempt_num > 0 else None,
        "stage_namespace": stage_scope or None,
    }
