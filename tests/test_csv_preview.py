import json
import os

from src.utils.csv_preview import load_csv_preview, resolve_preview_dialect


def test_csv_preview_handles_semicolon_dialect_without_parser_error(tmp_path):
    csv_path = tmp_path / "clientes.csv"
    csv_path.write_text(
        "Size;Debtors;Sector\n"
        "23351746,0;54,0;Industria\n"
        "20600000,0;270,0;Servicios\n",
        encoding="utf-8",
    )

    payload = load_csv_preview(str(csv_path), max_rows=10)

    assert payload
    assert payload["col_count"] == 3
    assert payload["row_count_total"] == 2
    assert list(payload["df"].columns) == ["Size", "Debtors", "Sector"]


def test_csv_preview_prefers_adjacent_cleaning_manifest_dialect(tmp_path):
    clean_dir = tmp_path / "artifacts" / "clean"
    clean_dir.mkdir(parents=True)
    (clean_dir / "cleaning_manifest.json").write_text(
        json.dumps({"output_dialect": {"sep": ";", "decimal": ",", "encoding": "utf-8"}}),
        encoding="utf-8",
    )
    csv_path = clean_dir / "dataset_cleaned.csv"
    csv_path.write_text(
        "a;b\n"
        "1;2\n"
        "3;4\n",
        encoding="utf-8",
    )

    dialect = resolve_preview_dialect(str(csv_path))
    payload = load_csv_preview(str(csv_path), max_rows=10)

    assert dialect["sep"] == ";"
    assert payload["dialect_used"]["sep"] == ";"
    assert payload["col_count"] == 2
