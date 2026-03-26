from pathlib import Path

from src.utils.pdf_generator import convert_report_to_pdf


_MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
    b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


def test_pdf_fallback_plots(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    plots_dir = Path("static/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    (plots_dir / "test_fallback_plot.png").write_bytes(_MINIMAL_PNG)

    output_pdf = Path("test_output.pdf")
    success = convert_report_to_pdf(
        "# Test Report\n\nNo manual images here.",
        str(output_pdf),
    )

    assert success, "convert_report_to_pdf returned False"
    assert output_pdf.exists(), "PDF output file was not created"

    pdf_bytes = output_pdf.read_bytes()
    assert b"/Subtype /Image" in pdf_bytes, "Fallback image was not embedded in the PDF"
