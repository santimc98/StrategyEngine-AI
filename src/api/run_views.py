from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.utils.paths import run_dir
from src.utils.run_status import read_final_state


def _load_json_safe(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _load_json_list_safe(path: Path) -> Optional[List[Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, list) else None
    except Exception:
        return None


def _slug_to_title(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"[_\-]+", " ", stem).strip()
    return stem[:1].upper() + stem[1:] if stem else filename


def _run_root(run_id: str) -> Path:
    return Path(run_dir(run_id))


def _run_work_dir(run_id: str) -> Path:
    return _run_root(run_id) / "work"


def _run_data_dir(run_id: str) -> Path:
    return _run_work_dir(run_id) / "data"


def _run_plots_dir(run_id: str) -> Path:
    return _run_work_dir(run_id) / "static" / "plots"


def _run_report_dir(run_id: str) -> Path:
    return _run_root(run_id) / "report"


def _resolve_pdf_path(run_id: str) -> Optional[Path]:
    final_state = read_final_state(run_id) or {}
    pdf_path = final_state.get("pdf_path")
    if isinstance(pdf_path, str) and pdf_path.strip():
        candidate = Path(pdf_path)
        if candidate.exists():
            return candidate

    candidates = [
        _run_report_dir(run_id) / "final_report.pdf",
        _run_report_dir(run_id) / "final_report_b4fca022.pdf",
        _run_work_dir(run_id) / "final_report_b4fca022.pdf",
        _run_work_dir(run_id) / "runs" / run_id / "report" / "final_report.pdf",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _get_report_blocks(run_id: str) -> List[Dict[str, Any]]:
    path = _run_data_dir(run_id) / "final_report_blocks.json"
    blocks = _load_json_list_safe(path)
    if isinstance(blocks, list):
        return [b for b in blocks if isinstance(b, dict)]

    final_state = read_final_state(run_id) or {}
    blocks = final_state.get("final_report_blocks")
    if isinstance(blocks, list):
        return [b for b in blocks if isinstance(b, dict)]
    return []


def _get_report_markdown(run_id: str) -> str:
    summary_path = _run_data_dir(run_id) / "executive_summary.md"
    if summary_path.exists():
        try:
            return summary_path.read_text(encoding="utf-8")
        except Exception:
            pass

    final_state = read_final_state(run_id) or {}
    report = final_state.get("final_report")
    return str(report or "")


def _get_plot_summaries(run_id: str) -> List[Dict[str, Any]]:
    path = _run_plots_dir(run_id) / "plot_summaries.json"
    payload = _load_json_list_safe(path)
    if isinstance(payload, list):
        return [entry for entry in payload if isinstance(entry, dict)]
    return []


def _report_chart_order(blocks: Iterable[Dict[str, Any]]) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for block in blocks:
        if str(block.get("artifact_type") or "").strip().lower() != "chart":
            continue
        rel_path = str(block.get("path") or "").replace("\\", "/").strip()
        if not rel_path:
            continue
        filename = Path(rel_path).name
        if filename and filename not in seen:
            seen.add(filename)
            ordered.append(filename)
    return ordered


def list_report_plots(run_id: str) -> List[Dict[str, Any]]:
    plots_dir = _run_plots_dir(run_id)
    if not plots_dir.exists():
        return []

    blocks = _get_report_blocks(run_id)
    report_order = _report_chart_order(blocks)
    summaries = _get_plot_summaries(run_id)
    summary_by_file = {
        str(entry.get("filename") or "").strip(): entry
        for entry in summaries
        if str(entry.get("filename") or "").strip()
    }
    disk_files = sorted(
        [path.name for path in plots_dir.glob("*.png") if path.is_file()],
        key=str.lower,
    )

    ordered_names: List[str] = []
    seen = set()
    for name in report_order + [n for n in summary_by_file.keys()] + disk_files:
        if name and name not in seen and (plots_dir / name).exists():
            seen.add(name)
            ordered_names.append(name)

    items: List[Dict[str, Any]] = []
    for index, filename in enumerate(ordered_names, start=1):
        summary = summary_by_file.get(filename, {})
        title = str(summary.get("title") or "").strip() or _slug_to_title(filename)
        facts = summary.get("facts")
        if not isinstance(facts, list):
            facts = []
        items.append(
            {
                "filename": filename,
                "title": title,
                "facts": [str(fact) for fact in facts if str(fact).strip()],
                "referenced_in_report": filename in report_order,
                "order": index,
                "image_url": f"/runs/{run_id}/report/plots/{filename}",
            }
        )
    return items


def get_plot_file_path(run_id: str, filename: str) -> Optional[Path]:
    safe_name = Path(filename).name
    if not safe_name or safe_name != filename:
        return None
    candidate = _run_plots_dir(run_id) / safe_name
    return candidate if candidate.exists() else None


def get_artifact_manifest(run_id: str) -> Dict[str, Any]:
    manifest_path = _run_data_dir(run_id) / "report_artifact_manifest.json"
    payload = _load_json_safe(manifest_path) or {}
    return payload


def get_report_visual_tables(run_id: str) -> Dict[str, Any]:
    tables_path = _run_data_dir(run_id) / "report_visual_tables.json"
    payload = _load_json_safe(tables_path) or {}
    return payload


def get_run_report_payload(run_id: str) -> Dict[str, Any]:
    final_state = read_final_state(run_id) or {}
    run_summary = _load_json_safe(_run_data_dir(run_id) / "run_summary.json") or {}
    blocks = _get_report_blocks(run_id)
    markdown = _get_report_markdown(run_id)
    artifact_manifest = get_artifact_manifest(run_id)
    visual_tables = get_report_visual_tables(run_id)
    pdf_path = _resolve_pdf_path(run_id)

    return {
        "run_id": run_id,
        "status": run_summary.get("status") or final_state.get("review_verdict"),
        "run_outcome": run_summary.get("run_outcome"),
        "markdown": markdown,
        "blocks": blocks,
        "pdf_available": pdf_path is not None,
        "pdf_url": f"/runs/{run_id}/report/pdf" if pdf_path is not None else None,
        "plots": list_report_plots(run_id),
        "artifact_manifest_summary": (artifact_manifest.get("summary") or {}),
        "run_summary": run_summary,
        "visual_tables": visual_tables,
    }


def get_pdf_path(run_id: str) -> Optional[Path]:
    return _resolve_pdf_path(run_id)
