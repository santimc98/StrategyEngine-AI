import os
from pathlib import Path

from src.utils.run_storage import (
    init_run_dir,
    finalize_run,
    apply_retention,
    clean_workspace_outputs,
)


def test_latest_is_overwritten(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runs_dir = str(tmp_path / "runs")
    latest = tmp_path / "runs" / "latest"
    latest.mkdir(parents=True, exist_ok=True)
    dummy = latest / "dummy.txt"
    dummy.write_text("stale", encoding="utf-8")
    run_dir = init_run_dir("abc123", base_dir=runs_dir, started_at="2025-01-01T00:00:00")
    assert not dummy.exists()
    assert (latest / "run_id.txt").exists()
    assert (Path(run_dir) / "run_manifest.json").exists()


def test_archive_on_fail(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runs_dir = str(tmp_path / "runs")
    run_dir = init_run_dir("fail123", base_dir=runs_dir, started_at="2025-01-01T00:00:00")
    (Path(run_dir) / "dummy.txt").write_text("x", encoding="utf-8")
    finalize_run("fail123", status_final="FAIL", state={}, runs_dir=runs_dir)
    archive = tmp_path / "runs" / "archive" / "run_fail123.zip"
    assert archive.exists()


def test_no_archive_on_pass(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runs_dir = str(tmp_path / "runs")
    init_run_dir("pass123", base_dir=runs_dir, started_at="2025-01-01T00:00:00")
    finalize_run("pass123", status_final="PASS", state={}, runs_dir=runs_dir)
    archive_dir = tmp_path / "runs" / "archive"
    if archive_dir.exists():
        assert not any(archive_dir.glob("run_pass123.zip"))


def test_retention_keeps_last_n(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    archive_dir = tmp_path / "runs" / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    zips = []
    for idx in range(7):
        path = archive_dir / f"run_{idx}.zip"
        path.write_bytes(b"zip")
        zips.append(path)
    for idx, path in enumerate(zips):
        os.utime(path, (path.stat().st_atime, path.stat().st_mtime + idx))
    apply_retention(keep_last=5, archive_dir=str(archive_dir))
    remaining = list(archive_dir.glob("*.zip"))
    assert len(remaining) == 5


def test_clean_workspace_outputs_removes_cleaned_outputs_preserves_dataset_memory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    keep_path = data_dir / "dataset_memory.json"
    keep_path.write_text("{}", encoding="utf-8")
    remove_paths = [
        data_dir / "cleaned_data.csv",
        data_dir / "cleaned_full.csv",
        data_dir / "cleaning_manifest.json",
        data_dir / "dataset_profile.json",
        data_dir / "metrics.json",
        data_dir / "scored_rows.csv",
        data_dir / "alignment_check.json",
        data_dir / "output_contract_report.json",
    ]
    for path in remove_paths:
        path.write_text("x", encoding="utf-8")

    clean_workspace_outputs()

    assert keep_path.exists()
    assert all(not path.exists() for path in remove_paths)
