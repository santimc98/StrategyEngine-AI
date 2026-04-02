from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse

from src.api import config_views
from src.api import integration_views
from src.api import run_views
from src.utils import run_history, run_launcher, run_status
from src.utils.paths import DATA_DIR, PROJECT_ROOT, RUNS_DIR, run_dir

app = FastAPI(
    title="StrategyEngine AI API",
    version="0.1.0",
    summary="Backend API for runs, status polling and result retrieval.",
)


class RunCreateRequest(BaseModel):
    csv_path: str = Field(..., description="Absolute or relative path to the CSV dataset.")
    business_objective: str = Field(..., description="Business objective for the run.")
    sandbox_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional sandbox/runtime configuration.",
    )
    replace_active_run: bool = Field(
        default=False,
        description="Abort and replace the current active run if one exists.",
    )


class RunAbortRequest(BaseModel):
    force_kill: bool = Field(
        default=False,
        description="Kill the worker process after requesting abort.",
    )


class ModelSettingsRequest(BaseModel):
    models: Dict[str, Any] = Field(
        default_factory=dict,
        description="Runtime model settings keyed by agent slot.",
    )


class SandboxSettingsRequest(BaseModel):
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Sandbox provider and execution backend configuration.",
    )


class ApiKeysRequest(BaseModel):
    keys: Dict[str, Any] = Field(
        default_factory=dict,
        description="API keys keyed by env var.",
    )


class ApiKeyTestRequest(BaseModel):
    env_var: str = Field(..., description="API key env var to test.")
    value: Optional[str] = Field(default=None, description="Optional override value to test.")


class ConnectorCredentialsRequest(BaseModel):
    credentials: Dict[str, Any] = Field(
        default_factory=dict,
        description="Connector-specific credentials.",
    )


class ConnectorFetchRequest(BaseModel):
    credentials: Dict[str, Any] = Field(
        default_factory=dict,
        description="Connector-specific credentials.",
    )
    object_name: str = Field(..., description="CRM object/entity name to fetch.")
    max_records: int = Field(default=10000, ge=1, le=50000)
    preview_rows: int = Field(default=25, ge=1, le=200)
    save_to_data: bool = Field(default=True)


def _load_json_safe(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _worker_input_path(run_id: str) -> str:
    return os.path.join(run_dir(run_id), "worker_input.json")


def _manifest_path(run_id: str) -> str:
    return os.path.join(run_dir(run_id), "run_manifest.json")


def _run_exists(run_id: str) -> bool:
    return os.path.isdir(run_dir(run_id))


def _sanitize_upload_filename(filename: str) -> str:
    candidate = Path(str(filename or "").strip()).name
    if not candidate:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing filename")
    if Path(candidate).suffix.lower() != ".csv":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only CSV uploads are supported")
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(candidate).stem).strip("._-") or "dataset"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    return f"{stem}_{timestamp}.csv"


def _build_run_detail(run_id: str) -> Optional[Dict[str, Any]]:
    if not _run_exists(run_id):
        return None

    worker_input = _load_json_safe(_worker_input_path(run_id))
    manifest = _load_json_safe(_manifest_path(run_id))
    status_payload = run_status.read_status(run_id)
    final_state = run_status.read_final_state(run_id)

    detail: Dict[str, Any] = {
        "run_id": run_id,
        "has_final_state": final_state is not None,
        "input": worker_input or {},
        "manifest": manifest or {},
        "status": status_payload or {},
        "result": final_state or {},
    }
    return detail


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "strategyengine-api",
        "project_root": PROJECT_ROOT,
        "runs_dir": RUNS_DIR,
    }


@app.get("/runs/active")
def get_active_run() -> Dict[str, Any]:
    active_run_id = run_status.get_active_run_id()
    if not active_run_id:
        return {"active_run_id": None, "status": None}
    return {
        "active_run_id": active_run_id,
        "status": run_status.read_status(active_run_id),
    }


@app.get("/runs")
def list_runs(limit: int = Query(default=20, ge=1, le=200)) -> Dict[str, Any]:
    items = run_history.list_runs(runs_dir=RUNS_DIR, limit=limit)
    return {"items": items, "count": len(items)}


@app.get("/runs/{run_id}")
def get_run_detail(run_id: str) -> Dict[str, Any]:
    detail = _build_run_detail(run_id)
    if detail is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    return detail


@app.get("/runs/{run_id}/status")
def get_run_status(run_id: str) -> Dict[str, Any]:
    if not _run_exists(run_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    payload = run_status.read_status(run_id)
    if payload is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run status not available")
    return payload


@app.get("/runs/{run_id}/logs")
def get_run_logs(run_id: str, after_line: int = Query(default=0, ge=0)) -> Dict[str, Any]:
    if not _run_exists(run_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    entries = run_status.read_log_entries(run_id, after_line=after_line)
    return {
        "run_id": run_id,
        "after_line": after_line,
        "next_after_line": after_line + len(entries),
        "entries": entries,
    }


@app.get("/runs/{run_id}/activity")
def get_run_activity(run_id: str, after_line: int = Query(default=0, ge=0)) -> Dict[str, Any]:
    if not _run_exists(run_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    return run_views.list_run_activity(run_id, after_line=after_line)


@app.post("/runs", status_code=status.HTTP_201_CREATED)
def create_run(payload: RunCreateRequest) -> Dict[str, Any]:
    csv_path = os.path.abspath(payload.csv_path)
    business_objective = str(payload.business_objective or "").strip()
    if not os.path.exists(csv_path):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="csv_path does not exist",
        )
    if not business_objective:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="business_objective is required",
        )

    try:
        launched = run_launcher.start_background_run(
            csv_path=csv_path,
            business_objective=business_objective,
            sandbox_config=payload.sandbox_config,
            replace_active_run=payload.replace_active_run,
        )
    except run_launcher.ActiveRunConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": "An active run already exists",
                "active_run_id": exc.active_run_id,
            },
        ) from exc

    return {
        "run_id": launched["run_id"],
        "pid": launched["pid"],
        "replaced_run_id": launched["replaced_run_id"],
        "worker_stdout_path": launched["worker_stdout_path"],
    }


@app.post("/datasets/upload", status_code=status.HTTP_201_CREATED)
async def upload_dataset(request: Request) -> Dict[str, Any]:
    filename = _sanitize_upload_filename(request.headers.get("x-filename") or "")
    content = await request.body()
    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty upload body")

    uploads_dir = os.path.join(DATA_DIR, "uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    csv_path = os.path.abspath(os.path.join(uploads_dir, filename))
    with open(csv_path, "wb") as handle:
        handle.write(content)

    return {
        "filename": filename,
        "csv_path": csv_path,
        "size_bytes": len(content),
    }


@app.post("/runs/{run_id}/abort")
def abort_run(run_id: str, payload: RunAbortRequest) -> Dict[str, Any]:
    if not _run_exists(run_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")

    run_status.request_run_abort(run_id)
    worker_killed = False
    if payload.force_kill:
        worker_killed = run_status.kill_worker(run_id)

    return {
        "run_id": run_id,
        "abort_requested": True,
        "worker_killed": worker_killed,
    }


@app.get("/runs/{run_id}/report")
def get_run_report(run_id: str) -> Dict[str, Any]:
    if not _run_exists(run_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    return run_views.get_run_report_payload(run_id)


@app.get("/runs/{run_id}/report/blocks")
def get_run_report_blocks(run_id: str) -> Dict[str, Any]:
    if not _run_exists(run_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    return {
        "run_id": run_id,
        "blocks": run_views.get_run_report_payload(run_id)["blocks"],
    }


@app.get("/runs/{run_id}/report/plots")
def get_run_report_plots(run_id: str) -> Dict[str, Any]:
    if not _run_exists(run_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    plots = run_views.list_report_plots(run_id)
    return {
        "run_id": run_id,
        "items": plots,
        "count": len(plots),
    }


@app.get("/runs/{run_id}/report/plots/{filename}")
def get_run_report_plot_image(run_id: str, filename: str):
    if not _run_exists(run_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    plot_path = run_views.get_plot_file_path(run_id, filename)
    if plot_path is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Plot not found")
    return FileResponse(plot_path, media_type="image/png", filename=plot_path.name)


@app.get("/runs/{run_id}/report/pdf")
def get_run_report_pdf(run_id: str):
    if not _run_exists(run_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    pdf_path = run_views.get_pdf_path(run_id)
    if pdf_path is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="PDF not found")
    return FileResponse(pdf_path, media_type="application/pdf", filename=pdf_path.name)


@app.get("/runs/{run_id}/artifacts/manifest")
def get_run_artifact_manifest(run_id: str) -> Dict[str, Any]:
    if not _run_exists(run_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    manifest = run_views.get_artifact_manifest(run_id)
    return {
        "run_id": run_id,
        **manifest,
    }


@app.get("/runs/{run_id}/artifacts/zip")
def download_run_artifacts_zip(run_id: str):
    """Download a ZIP archive of the run's artifacts, plots, report, and contracts."""
    if not _run_exists(run_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    zip_path = run_views.build_artifacts_zip(run_id)
    if zip_path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No artifacts found for this run",
        )
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"run_{run_id}_artifacts.zip",
        background=None,  # keep file until response is sent
    )


@app.get("/config/models")
def get_model_settings() -> Dict[str, Any]:
    return config_views.get_model_settings_view()


@app.put("/config/models")
def put_model_settings(payload: ModelSettingsRequest) -> Dict[str, Any]:
    return config_views.apply_and_persist_model_settings(payload.models)


@app.post("/config/models/reset")
def reset_model_settings() -> Dict[str, Any]:
    return config_views.reset_model_settings()


@app.get("/config/sandbox")
def get_sandbox_settings() -> Dict[str, Any]:
    return config_views.get_sandbox_settings_view()


@app.put("/config/sandbox")
def put_sandbox_settings(payload: SandboxSettingsRequest) -> Dict[str, Any]:
    return config_views.save_sandbox_settings(payload.config)


@app.get("/config/api-keys")
def get_api_key_status() -> Dict[str, Any]:
    return integration_views.get_api_key_status_view()


@app.put("/config/api-keys")
def put_api_keys(payload: ApiKeysRequest) -> Dict[str, Any]:
    return integration_views.save_api_keys_view(payload.keys)


@app.post("/config/api-keys/test")
def post_api_key_test(payload: ApiKeyTestRequest) -> Dict[str, Any]:
    return integration_views.test_api_key_view(payload.env_var, payload.value)


@app.get("/integrations/connectors")
def get_connectors_catalog() -> Dict[str, Any]:
    return integration_views.list_connector_specs()


@app.post("/integrations/connectors/{connector_id}/test")
def post_connector_test(connector_id: str, payload: ConnectorCredentialsRequest) -> Dict[str, Any]:
    return integration_views.test_connector_connection(connector_id, payload.credentials)


@app.post("/integrations/connectors/{connector_id}/objects")
def post_connector_objects(connector_id: str, payload: ConnectorCredentialsRequest) -> Dict[str, Any]:
    try:
        return integration_views.list_connector_objects(connector_id, payload.credentials)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@app.post("/integrations/connectors/{connector_id}/fetch")
def post_connector_fetch(connector_id: str, payload: ConnectorFetchRequest) -> Dict[str, Any]:
    try:
        return integration_views.fetch_connector_data(
            connector_id,
            payload.credentials,
            payload.object_name,
            max_records=payload.max_records,
            preview_rows=payload.preview_rows,
            save_to_data=payload.save_to_data,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
