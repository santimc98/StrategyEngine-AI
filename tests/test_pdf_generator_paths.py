from pathlib import Path

from src.utils.pdf_generator import resolve_image_path, _prepare_exec_tables_for_pdf


def test_resolve_image_path_falls_back_to_artifacts(tmp_path: Path) -> None:
    plots_dir = tmp_path / "artifacts" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_dir / "plot.png"
    plot_path.write_bytes(b"test")

    resolved = resolve_image_path("static/plots/plot.png", str(tmp_path))

    assert resolved == str(plot_path)


def test_prepare_exec_tables_for_pdf_inserts_breaks_in_long_cell_tokens() -> None:
    html = (
        '<table class="exec-table artifact-inventory"><tbody>'
        '<tr><td>artifacts/ml/scored_output_2025_10_31.csv</td>'
        '<td><span class="status-badge">PRESENT_OPTIONAL</span></td></tr>'
        '</tbody></table>'
    )

    rendered = _prepare_exec_tables_for_pdf(html)

    assert "artifacts/<br/>ml/<br/>scored_<br/>output_<br/>2025_<br/>10_<br/>31.<br/>csv" in rendered
    assert "PRESENT_<br/>OPTIONAL" in rendered
