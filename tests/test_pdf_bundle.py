import time
from pathlib import Path

from src.graph.graph import generate_pdf_artifact
from src.utils.run_bundle import init_run_bundle


def test_generate_pdf_copies_to_run_bundle(tmp_path, monkeypatch):
    run_id = "runpdf"
    run_dir = init_run_bundle(run_id, {}, base_dir=str(tmp_path / "runs"), enable_tee=False)
    work_dir = Path(run_dir) / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(work_dir)

    def _fake_convert(_content: str, output_filename: str, base_dir=None) -> bool:
        Path(output_filename).write_text("pdf", encoding="utf-8")
        return True

    monkeypatch.setattr("src.graph.graph.convert_report_to_pdf", _fake_convert)

    state = {
        "run_id": run_id,
        "run_bundle_dir": run_dir,
        "work_dir": str(work_dir),
        "work_dir_abs": str(work_dir),
        "run_start_epoch": time.time(),
        "final_report": "Report",
    }
    result = generate_pdf_artifact(state)
    assert result.get("pdf_path")
    assert (Path(run_dir) / "report" / "final_report.pdf").exists()
    assert not (work_dir / "runs" / run_id / "report" / "final_report.pdf").exists()
