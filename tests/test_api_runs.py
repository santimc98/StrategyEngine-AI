import json
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from src.api import main as api_main
from src.api import run_views


@pytest.fixture
def client():
    return TestClient(api_main.app)


def _patch_run_paths(monkeypatch, runs_dir):
    monkeypatch.setattr(api_main, "RUNS_DIR", str(runs_dir))
    monkeypatch.setattr(api_main.run_status, "RUNS_DIR", str(runs_dir))
    monkeypatch.setattr(api_main.run_history, "_RUNS_DIR", str(runs_dir))
    monkeypatch.setattr(api_main, "run_dir", lambda run_id: str(runs_dir / run_id))
    monkeypatch.setattr(run_views, "run_dir", lambda run_id: str(runs_dir / run_id))
    monkeypatch.setattr(run_views, "read_final_state", lambda run_id: api_main.run_status.read_final_state(run_id))


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "strategyengine-api"


def test_list_runs_endpoint_uses_run_history(client, monkeypatch):
    monkeypatch.setattr(
        api_main.run_history,
        "list_runs",
        lambda runs_dir, limit: [{"run_id": "abc12345", "status": "complete"}],
    )

    response = client.get("/runs?limit=10")

    assert response.status_code == 200
    assert response.json() == {
        "items": [{"run_id": "abc12345", "status": "complete"}],
        "count": 1,
    }


def test_get_run_detail_returns_input_status_and_result(client, monkeypatch, tmp_path):
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "abc12345"
    run_dir.mkdir(parents=True)

    _patch_run_paths(monkeypatch, runs_dir)

    (run_dir / "worker_input.json").write_text(
        json.dumps({"csv_path": "C:/tmp/data.csv", "business_objective": "Objetivo"}),
        encoding="utf-8",
    )
    (run_dir / "worker_status.json").write_text(
        json.dumps({"status": "running", "stage": "engineer"}),
        encoding="utf-8",
    )
    (run_dir / "worker_final_state.json").write_text(
        json.dumps({"review_verdict": "APPROVED", "iteration_count": 2}),
        encoding="utf-8",
    )

    response = client.get("/runs/abc12345")

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == "abc12345"
    assert payload["input"]["business_objective"] == "Objetivo"
    assert payload["status"]["stage"] == "engineer"
    assert payload["result"]["review_verdict"] == "APPROVED"


def test_create_run_returns_409_when_active_run_exists(client, monkeypatch, tmp_path):
    csv_path = tmp_path / "dataset.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")

    def _raise_conflict(**kwargs):
        raise api_main.run_launcher.ActiveRunConflictError("active123")

    monkeypatch.setattr(api_main.run_launcher, "start_background_run", _raise_conflict)

    response = client.post(
        "/runs",
        json={
            "csv_path": str(csv_path),
            "business_objective": "Probar run",
            "replace_active_run": False,
        },
    )

    assert response.status_code == 409
    assert response.json()["detail"]["active_run_id"] == "active123"


def test_create_run_launches_background_worker(client, monkeypatch, tmp_path):
    csv_path = tmp_path / "dataset.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")

    monkeypatch.setattr(
        api_main.run_launcher,
        "start_background_run",
        lambda **kwargs: {
            "run_id": "newrun01",
            "pid": 4321,
            "replaced_run_id": None,
            "worker_stdout_path": "C:/tmp/worker_stdout.log",
        },
    )

    response = client.post(
        "/runs",
        json={
            "csv_path": str(csv_path),
            "business_objective": "Construir ranking",
        },
    )

    assert response.status_code == 201
    assert response.json()["run_id"] == "newrun01"


def test_upload_dataset_persists_csv_to_data_dir(client, monkeypatch, tmp_path):
    monkeypatch.setattr(api_main, "DATA_DIR", str(tmp_path))

    response = client.post(
        "/datasets/upload",
        headers={"x-filename": "ventas.csv", "content-type": "text/csv"},
        content=b"a,b\n1,2\n",
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["filename"].endswith(".csv")
    assert payload["size_bytes"] == len(b"a,b\n1,2\n")
    assert Path(payload["csv_path"]).exists()


def test_abort_run_requests_abort_and_optional_kill(client, monkeypatch, tmp_path):
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "abc12345"
    run_dir.mkdir(parents=True)

    _patch_run_paths(monkeypatch, runs_dir)

    requested = {"abort": False, "kill": False}
    monkeypatch.setattr(
        api_main.run_status,
        "request_run_abort",
        lambda run_id: requested.__setitem__("abort", True),
    )
    monkeypatch.setattr(
        api_main.run_status,
        "kill_worker",
        lambda run_id: requested.__setitem__("kill", True) or True,
    )

    response = client.post("/runs/abc12345/abort", json={"force_kill": True})

    assert response.status_code == 200
    assert requested == {"abort": True, "kill": True}
    assert response.json()["worker_killed"] is True


def test_report_endpoint_returns_curated_payload(client, monkeypatch, tmp_path):
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "abc12345"
    work_data = run_dir / "work" / "data"
    plots_dir = run_dir / "work" / "static" / "plots"
    report_dir = run_dir / "report"
    work_data.mkdir(parents=True)
    plots_dir.mkdir(parents=True)
    report_dir.mkdir(parents=True)

    _patch_run_paths(monkeypatch, runs_dir)

    (run_dir / "worker_final_state.json").write_text(
        json.dumps(
            {
                "run_id": "abc12345",
                "review_verdict": "APPROVED",
                "final_report": "# Reporte\n\nTexto",
                "pdf_path": str(report_dir / "final_report.pdf"),
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "worker_input.json").write_text(
        json.dumps({"business_objective": "Objetivo", "csv_path": "C:/tmp/data.csv"}),
        encoding="utf-8",
    )
    (work_data / "run_summary.json").write_text(
        json.dumps({"run_outcome": "GO", "data_adequacy": {"status": "ok"}}),
        encoding="utf-8",
    )
    (work_data / "final_report_blocks.json").write_text(
        json.dumps(
            [
                {
                    "type": "artifact",
                    "artifact_type": "chart",
                    "path": "static/plots/ml_score_distribution_by_case.png",
                }
            ]
        ),
        encoding="utf-8",
    )
    (work_data / "report_artifact_manifest.json").write_text(
        json.dumps({"summary": {"required_total": 4, "required_missing": 0}}),
        encoding="utf-8",
    )
    (work_data / "report_visual_tables.json").write_text(
        json.dumps({"kpi_snapshot_table_html": "<table></table>"}),
        encoding="utf-8",
    )
    (plots_dir / "plot_summaries.json").write_text(
        json.dumps(
            [
                {
                    "filename": "ml_score_distribution_by_case.png",
                    "title": "Distribución ML por caso",
                    "facts": ["cases=20"],
                }
            ]
        ),
        encoding="utf-8",
    )
    (plots_dir / "ml_score_distribution_by_case.png").write_bytes(b"png")
    (report_dir / "final_report.pdf").write_bytes(b"%PDF-1.4\n")

    response = client.get("/runs/abc12345/report")

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == "abc12345"
    assert payload["pdf_available"] is True
    assert payload["pdf_url"] == "/runs/abc12345/report/pdf"
    assert payload["artifact_manifest_summary"]["required_total"] == 4
    assert payload["plots"][0]["filename"] == "ml_score_distribution_by_case.png"
    assert payload["plots"][0]["referenced_in_report"] is True


def test_report_endpoint_prefers_canonical_run_summary_over_stale_manifest_snapshot(client, monkeypatch, tmp_path):
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "canon1234"
    work_data = run_dir / "work" / "data"
    report_dir = run_dir / "report"
    work_data.mkdir(parents=True)
    report_dir.mkdir(parents=True)

    _patch_run_paths(monkeypatch, runs_dir)

    (run_dir / "worker_final_state.json").write_text(
        json.dumps(
            {
                "run_id": "canon1234",
                "review_verdict": "NEEDS_IMPROVEMENT",
                "final_report": "# Reporte\n\nTexto",
                "pdf_path": str(report_dir / "final_report.pdf"),
            }
        ),
        encoding="utf-8",
    )
    (work_data / "run_summary.json").write_text(
        json.dumps({"status": "NEEDS_IMPROVEMENT", "run_outcome": "GO_WITH_LIMITATIONS"}),
        encoding="utf-8",
    )
    (work_data / "final_incumbent_state.json").write_text(
        json.dumps({"authoritative_decision": "GO_WITH_LIMITATIONS", "run_outcome": "GO_WITH_LIMITATIONS"}),
        encoding="utf-8",
    )
    (work_data / "report_artifact_manifest.json").write_text(
        json.dumps(
            {
                "summary": {"required_total": 3, "required_missing": 0},
                "governance_snapshot": {
                    "review_verdict": "APPROVED",
                    "run_outcome": "GO_WITH_LIMITATIONS",
                    "failed_gates": [],
                },
            }
        ),
        encoding="utf-8",
    )
    (report_dir / "final_report.pdf").write_bytes(b"%PDF-1.4\n")

    response = client.get("/runs/canon1234/report")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "NEEDS_IMPROVEMENT"
    assert payload["run_outcome"] == "NO_GO"
    assert payload["run_summary"]["status"] == "NEEDS_IMPROVEMENT"
    assert payload["governance_snapshot"]["review_verdict"] == "NEEDS_IMPROVEMENT"
    assert payload["governance_snapshot"]["review_verdict_source"] == "run_summary.status"
    assert payload["governance_snapshot"]["run_outcome"] == "NO_GO"
    assert payload["governance_snapshot"]["run_outcome_source"] == "derived_from_review_verdict"


def test_report_plots_endpoint_returns_curated_order(client, monkeypatch, tmp_path):
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "runplots1"
    work_data = run_dir / "work" / "data"
    plots_dir = run_dir / "work" / "static" / "plots"
    work_data.mkdir(parents=True)
    plots_dir.mkdir(parents=True)

    _patch_run_paths(monkeypatch, runs_dir)

    (work_data / "final_report_blocks.json").write_text(
        json.dumps(
            [
                {
                    "type": "artifact",
                    "artifact_type": "chart",
                    "path": "static/plots/first.png",
                }
            ]
        ),
        encoding="utf-8",
    )
    (plots_dir / "plot_summaries.json").write_text(
        json.dumps(
            [
                {"filename": "second.png", "title": "Second", "facts": ["f=2"]},
                {"filename": "first.png", "title": "First", "facts": ["f=1"]},
            ]
        ),
        encoding="utf-8",
    )
    (plots_dir / "first.png").write_bytes(b"png")
    (plots_dir / "second.png").write_bytes(b"png")

    response = client.get("/runs/runplots1/report/plots")

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 2
    assert payload["items"][0]["filename"] == "first.png"
    assert payload["items"][0]["referenced_in_report"] is True
    assert payload["items"][1]["filename"] == "second.png"


def test_report_plot_image_endpoint_serves_png(client, monkeypatch, tmp_path):
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "runimg01"
    plots_dir = run_dir / "work" / "static" / "plots"
    plots_dir.mkdir(parents=True)

    _patch_run_paths(monkeypatch, runs_dir)
    (plots_dir / "chart.png").write_bytes(b"png")

    response = client.get("/runs/runimg01/report/plots/chart.png")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content == b"png"


def test_artifact_manifest_endpoint_returns_manifest(client, monkeypatch, tmp_path):
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "runmanifest1"
    work_data = run_dir / "work" / "data"
    work_data.mkdir(parents=True)

    _patch_run_paths(monkeypatch, runs_dir)
    (work_data / "report_artifact_manifest.json").write_text(
        json.dumps({"summary": {"required_total": 6}, "items": [{"path": "x"}]}),
        encoding="utf-8",
    )

    response = client.get("/runs/runmanifest1/artifacts/manifest")

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == "runmanifest1"
    assert payload["summary"]["required_total"] == 6


def test_run_activity_endpoint_returns_curated_internal_events(client, monkeypatch, tmp_path):
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "runactivity1"
    run_dir.mkdir(parents=True)

    _patch_run_paths(monkeypatch, runs_dir)

    (run_dir / "worker_status.json").write_text(
        json.dumps({"status": "running", "stage_name": "Data Engineer", "progress": 42, "iteration": 2}),
        encoding="utf-8",
    )
    (run_dir / "events.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": "2026-04-02T10:07:24.411781+00:00",
                        "event": "run_init",
                        "payload": {"csv_path": "C:/tmp/demo.csv", "dataset_fingerprint": "abc123def456"},
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "timestamp": "2026-04-02T10:19:50.700079+00:00",
                        "event": "heavy_runner_request",
                        "payload": {"mode": "data_engineer_cleaning", "attempt_id": 2},
                    },
                    ensure_ascii=False,
                ),
            ]
        ),
        encoding="utf-8",
    )

    response = client.get("/runs/runactivity1/activity")

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == "runactivity1"
    assert len(payload["entries"]) == 2
    assert payload["entries"][0]["agent"] == "Sistema"
    assert "demo.csv" in payload["entries"][0]["msg"]
    assert payload["entries"][1]["agent"] == "Runner"
    assert "runner" in payload["entries"][1]["msg"].lower()


def test_run_activity_endpoint_supports_incremental_cursor(client, monkeypatch, tmp_path):
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "runactivity2"
    run_dir.mkdir(parents=True)

    _patch_run_paths(monkeypatch, runs_dir)

    (run_dir / "worker_status.json").write_text(json.dumps({"status": "running"}), encoding="utf-8")
    (run_dir / "events.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"timestamp": "2026-04-02T10:00:00+00:00", "event": "run_init", "payload": {}}),
                json.dumps({"timestamp": "2026-04-02T10:01:00+00:00", "event": "steward_start", "payload": {}}),
                json.dumps({"timestamp": "2026-04-02T10:02:00+00:00", "event": "steward_complete", "payload": {}}),
            ]
        ),
        encoding="utf-8",
    )

    response = client.get("/runs/runactivity2/activity?after_line=1")

    assert response.status_code == 200
    payload = response.json()
    # After skipping line 0 (run_init), we get steward_start and steward_complete
    assert len(payload["entries"]) == 2
    assert payload["entries"][0]["agent"] == "Data Steward"
    assert payload["entries"][1]["agent"] == "Data Steward"
    assert payload["after_line"] == 1
    assert payload["next_after_line"] == 3
