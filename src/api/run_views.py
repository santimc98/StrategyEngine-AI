from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.utils.paths import run_dir
from src.utils.run_status import read_event_entries, read_final_state, read_status


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


def _event_title(event_name: str) -> str:
    title = re.sub(r"[_\-]+", " ", str(event_name or "").strip()).strip()
    return title[:1].upper() + title[1:] if title else "Evento"


def _event_phase(event_name: str, payload: Dict[str, Any]) -> str:
    event = str(event_name or "").strip().lower()
    step = str(payload.get("step") or "").strip().lower()
    if step:
        return step
    for token in (
        "steward",
        "strategist",
        "execution_planner",
        "planner",
        "data_engineer",
        "ml_engineer",
        "translator",
        "review_board",
        "results_advisor",
        "reviewer",
        "qa",
    ):
        if token in event:
            return token
    if event.startswith("heavy_runner"):
        return "runtime"
    if event.startswith("run_"):
        return "run"
    return "pipeline"


def _compact_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "sí" if value else "no"
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.4g}"
    if isinstance(value, (int,)):
        return str(value)
    text = str(value).strip()
    return text[:160] + "…" if len(text) > 160 else text


def _activity_summary(event_name: str, payload: Dict[str, Any]) -> Tuple[str, List[Dict[str, str]], str]:
    event = str(event_name or "").strip().lower()
    phase = _event_phase(event, payload)
    details: List[Dict[str, str]] = []
    level = "info"

    if event == "run_init":
        csv_path = Path(str(payload.get("csv_path") or "")).name
        if csv_path:
            details.append({"label": "CSV", "value": csv_path})
        if payload.get("dataset_fingerprint"):
            details.append({"label": "Fingerprint", "value": str(payload.get("dataset_fingerprint"))[:12]})
        return "La run ha arrancado y ya tiene dataset y objetivo cargados.", details, level

    if event.endswith("_start"):
        attempt = payload.get("attempt_id") or payload.get("ml_engineer_attempt")
        if attempt is not None:
            details.append({"label": "Intento", "value": _compact_value(attempt)})
        iteration = payload.get("iteration")
        if iteration is not None:
            details.append({"label": "Iteración", "value": _compact_value(iteration)})
        return f"{_event_title(phase)} ha comenzado su trabajo.", details, level

    if event == "steward_complete":
        target_status = _compact_value(payload.get("target_status") or "confirmed")
        recommended = _compact_value(payload.get("recommended_primary_target"))
        details.append({"label": "Target status", "value": target_status})
        if recommended:
            details.append({"label": "Target sugerido", "value": recommended})
        return "El steward terminó el análisis semántico del dataset.", details, level

    if event == "execution_planner_complete":
        outputs = payload.get("required_outputs") if isinstance(payload.get("required_outputs"), list) else []
        details.append({"label": "Outputs", "value": str(len(outputs))})
        return "El planner compiló el contrato ejecutable de la run.", details, "success"

    if event == "data_profile_built":
        if payload.get("n_rows") is not None:
            details.append({"label": "Filas", "value": _compact_value(payload.get("n_rows"))})
        if payload.get("n_cols") is not None:
            details.append({"label": "Columnas", "value": _compact_value(payload.get("n_cols"))})
        return "Se generó el perfilado base del dataset.", details, level

    if event.endswith("_view_context"):
        if payload.get("length") is not None:
            details.append({"label": "Contexto", "value": f"{_compact_value(payload.get('length'))} chars"})
        return f"Se preparó el contexto de {_event_title(phase)}.", details, level

    if event == "data_engineer_backend_selection":
        details.append({"label": "Backend", "value": _compact_value(payload.get("backend") or "unknown")})
        scale = payload.get("dataset_scale") if isinstance(payload.get("dataset_scale"), dict) else {}
        if scale.get("scale"):
            details.append({"label": "Escala", "value": _compact_value(scale.get("scale"))})
        return "El sistema eligió cómo ejecutar la limpieza en runtime.", details, level

    if event == "auto_fix_applied":
        fixes = payload.get("fixes") if isinstance(payload.get("fixes"), list) else []
        details.append({"label": "Intento", "value": _compact_value(payload.get("attempt"))})
        details.append({"label": "Autofixes", "value": str(len(fixes))})
        return "Se aplicaron autofixes de runtime antes de ejecutar el script.", details, "warning"

    if event == "heavy_runner_request":
        details.append({"label": "Modo", "value": _compact_value(payload.get("mode"))})
        details.append({"label": "Intento", "value": _compact_value(payload.get("attempt_id"))})
        return "El sistema envió el trabajo al runner pesado.", details, level

    if event == "heavy_runner_start":
        details.append({"label": "Paso", "value": _compact_value(payload.get("step"))})
        return "El runner está ejecutando el trabajo solicitado.", details, level

    if event == "heavy_runner_complete":
        step = _compact_value(payload.get("step"))
        status = _compact_value(payload.get("status"))
        downloaded = payload.get("downloaded") if isinstance(payload.get("downloaded"), list) else []
        if step:
            details.append({"label": "Paso", "value": step})
        if status:
            details.append({"label": "Estado", "value": status})
        if downloaded:
            details.append({"label": "Artifacts", "value": str(len(downloaded))})
        level = "success" if status == "success" else "warning"
        return "El runner terminó y devolvió los outputs de esa fase.", details, level

    if event == "cleaned_data_summary_min_built":
        details.append({"label": "Filas", "value": _compact_value(payload.get("row_count"))})
        details.append({"label": "Columnas", "value": _compact_value(payload.get("column_count"))})
        return "Se consolidó el resumen mínimo de la limpieza entregada.", details, "success"

    if event.endswith("_complete"):
        rows = payload.get("rows")
        cols = payload.get("columns")
        if rows is not None:
            details.append({"label": "Filas", "value": _compact_value(rows)})
        if cols is not None:
            details.append({"label": "Columnas", "value": _compact_value(cols)})
        plots = payload.get("plots_count")
        if plots is not None:
            details.append({"label": "Plots", "value": _compact_value(plots)})
        return f"{_event_title(phase)} terminó su fase.", details, "success"

    if event == "ml_plan_generated":
        metric_policy = payload.get("metric_policy") if isinstance(payload.get("metric_policy"), dict) else {}
        cv_policy = payload.get("cv_policy") if isinstance(payload.get("cv_policy"), dict) else {}
        if metric_policy.get("primary_metric"):
            details.append({"label": "Métrica", "value": _compact_value(metric_policy.get("primary_metric"))})
        if cv_policy.get("strategy"):
            details.append({"label": "CV", "value": _compact_value(cv_policy.get("strategy"))})
        return "El ML Engineer definió el plan de entrenamiento y validación.", details, level

    if event == "steward_target_reconsideration_start":
        details.append({"label": "Target sugerido", "value": _compact_value(payload.get("recommended_primary_target"))})
        return "Se abrió una reconsideración automática del target.", details, "warning"

    if event == "steward_target_reconsideration_complete":
        accepted = bool(payload.get("accepted"))
        details.append({"label": "Aceptada", "value": "sí" if accepted else "no"})
        if payload.get("final_primary_target"):
            details.append({"label": "Target final", "value": _compact_value(payload.get("final_primary_target"))})
        return "La reconsideración del target ya se resolvió.", details, "success" if accepted else "warning"

    if event == "review_board_complete":
        if payload.get("final_review_verdict"):
            details.append({"label": "Veredicto", "value": _compact_value(payload.get("final_review_verdict"))})
        return "El review board cerró la decisión de gobernanza.", details, "success"

    summary_parts: List[str] = []
    for key in ("status", "reason", "strategy", "step"):
        value = _compact_value(payload.get(key))
        if value:
            summary_parts.append(f"{key}={value}")
    summary = " | ".join(summary_parts) if summary_parts else "Evento interno registrado."
    return summary, details, level


def list_run_activity(run_id: str, after_line: int = 0) -> Dict[str, Any]:
    raw_entries = read_event_entries(run_id, after_line=after_line)
    entries: List[Dict[str, Any]] = []
    for entry in raw_entries:
        event_name = str(entry.get("event") or entry.get("type") or entry.get("name") or "").strip()
        payload = entry.get("payload") if isinstance(entry.get("payload"), dict) else {}
        summary, details, level = _activity_summary(event_name, payload)
        phase = _event_phase(event_name, payload)
        entries.append(
            {
                "index": int(entry.get("_line") or 0),
                "ts": entry.get("timestamp") or entry.get("ts"),
                "event": event_name,
                "phase": phase,
                "title": _event_title(event_name),
                "summary": summary,
                "details": details,
                "level": level,
            }
        )

    status_payload = read_status(run_id) or {}
    latest = entries[-1] if entries else None
    snapshot = {
        "current_stage": str(status_payload.get("stage_name") or status_payload.get("stage") or ""),
        "status": str(status_payload.get("status") or ""),
        "progress": status_payload.get("progress"),
        "iteration": status_payload.get("iteration"),
        "latest_title": latest.get("title") if latest else "",
        "latest_summary": latest.get("summary") if latest else "",
        "latest_phase": latest.get("phase") if latest else "",
        "latest_ts": latest.get("ts") if latest else "",
    }
    return {
        "run_id": run_id,
        "after_line": after_line,
        "next_after_line": after_line + len(raw_entries),
        "entries": entries,
        "snapshot": snapshot,
    }


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
    raw: Optional[str] = None
    if summary_path.exists():
        try:
            raw = summary_path.read_text(encoding="utf-8")
        except Exception:
            pass

    if not raw:
        final_state = read_final_state(run_id) or {}
        raw = str(final_state.get("final_report") or "")

    # ── Safety net: detect raw JSON stored as "markdown" and rescue it ──
    if raw and _looks_like_raw_json_report(raw):
        rescued = _rescue_markdown_from_raw_json(raw)
        if rescued:
            return rescued
    return raw or ""


def _looks_like_raw_json_report(text: str) -> bool:
    """Return True if *text* appears to be a raw JSON report block
    rather than rendered markdown."""
    stripped = text.strip()
    if stripped.startswith("```json"):
        stripped = re.sub(r"^```json\s*", "", stripped, flags=re.IGNORECASE)
    return stripped.startswith("{") and '"blocks"' in stripped[:500]


def _rescue_markdown_from_raw_json(raw_text: str) -> Optional[str]:
    """Best-effort extraction of readable markdown from truncated JSON report.

    Mirrors the logic in business_translator._rescue_markdown_from_raw_json
    but is kept standalone so the API layer has zero coupling to the agent.
    """
    if not raw_text:
        return None

    # Attempt to parse — with truncation repair (pass raw text so
    # re-start boundaries like ```json\n{ are still detectable).
    payload = _try_parse_truncated_json(raw_text)
    if not payload or not isinstance(payload.get("blocks"), list):
        return None

    lines: list = []
    title = str(payload.get("title") or "").strip()
    if title:
        lines.append(f"# {title}\n")
    for block in payload["blocks"]:
        if not isinstance(block, dict):
            continue
        btype = str(block.get("type") or "").strip().lower()
        if btype == "heading":
            level = int(block.get("level") or 2)
            lines.append(f"\n{'#' * level} {block.get('text', '')}\n")
        elif btype == "paragraph":
            lines.append(f"\n{block.get('text', '')}\n")
        elif btype == "bullet_list":
            for item in (block.get("items") or []):
                lines.append(f"- {item}")
            lines.append("")
        elif btype == "numbered_list":
            for idx, item in enumerate(block.get("items") or [], 1):
                lines.append(f"{idx}. {item}")
            lines.append("")
        elif btype == "artifact":
            lead_in = block.get("lead_in") or ""
            if lead_in:
                lines.append(f"\n{lead_in}\n")
            for item in (block.get("analysis") or []):
                lines.append(f"- {item}")
            lines.append("")
        elif btype == "markdown":
            lines.append(f"\n{block.get('content', '')}\n")
    return "\n".join(lines) if lines else None


def _try_parse_truncated_json(text: str) -> Optional[Dict[str, Any]]:
    """Parse JSON text, repairing truncation if necessary.

    Handles:
    - Clean JSON
    - JSON wrapped in ```json … ``` fences
    - Truncated JSON with multiple concatenated LLM attempts
    """
    if not text:
        return None

    # ── Fast path: whole text is valid JSON ──
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    # ── Strip leading fence, find first '{' ──
    start_idx = text.find("{")
    if start_idx == -1:
        return None

    fragment = text[start_idx:]

    # Cut at second JSON attempt: look for ```json\n{ which signals
    # a new truncated retry from the LLM.
    re_start = re.search(r"```\s*json\s*\n\s*\{", fragment[1:], flags=re.IGNORECASE)
    if re_start:
        fragment = fragment[: re_start.start() + 1]

    # ── String-aware walk ──
    stack: list = []  # 'o' / 'a'
    in_str = False
    escape = False
    last_good = 0
    for i, ch in enumerate(fragment):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            last_good = i
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            stack.append("o")
        elif ch == "[":
            stack.append("a")
        elif ch == "}":
            if stack and stack[-1] == "o":
                stack.pop()
            # Check if outer object is now closed
            if not stack:
                candidate = fragment[: i + 1]
                try:
                    parsed = json.loads(candidate)
                    return parsed if isinstance(parsed, dict) else None
                except Exception:
                    return None
        elif ch == "]":
            if stack and stack[-1] == "a":
                stack.pop()
        if ch.strip():
            last_good = i

    # ── Truncation repair: close everything still open ──
    repaired = fragment[: last_good + 1]
    if in_str:
        repaired = repaired.rstrip("\\") + '"'
    repaired = repaired.rstrip().rstrip(",")
    for item in reversed(stack):
        repaired += "]" if item == "a" else "}"
    try:
        parsed = json.loads(repaired)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


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


def build_artifacts_zip(run_id: str) -> Optional[Path]:
    """Create a temporary ZIP of the run's key artifacts and return the path."""
    import tempfile
    import zipfile

    work_dir = _run_work_dir(run_id)
    report_dir = _run_report_dir(run_id)

    # Collect files worth including in the zip
    targets: list[tuple[Path, str]] = []  # (absolute_path, archive_name)

    # 1. Artifacts directory (CSVs, manifests, dictionaries)
    artifacts_dir = work_dir / "artifacts"
    if artifacts_dir.is_dir():
        for fpath in artifacts_dir.rglob("*"):
            if fpath.is_file():
                arc_name = f"artifacts/{fpath.relative_to(artifacts_dir)}"
                targets.append((fpath, arc_name))

    # 2. Plots
    plots_dir = _run_plots_dir(run_id)
    if plots_dir.is_dir():
        for fpath in plots_dir.rglob("*"):
            if fpath.is_file():
                arc_name = f"plots/{fpath.name}"
                targets.append((fpath, arc_name))

    # 3. PDF report
    pdf_path = _resolve_pdf_path(run_id)
    if pdf_path and pdf_path.exists():
        targets.append((pdf_path, f"report/{pdf_path.name}"))

    # 4. Contracts
    contracts_dir = _run_root(run_id) / "contracts"
    if contracts_dir.is_dir():
        for fpath in contracts_dir.rglob("*"):
            if fpath.is_file():
                arc_name = f"contracts/{fpath.relative_to(contracts_dir)}"
                targets.append((fpath, arc_name))

    if not targets:
        return None

    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=".zip", prefix=f"run_{run_id}_"
    )
    tmp.close()
    zip_path = Path(tmp.name)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for abs_path, arc_name in targets:
            try:
                zf.write(abs_path, arc_name)
            except Exception:
                continue  # skip unreadable files
    return zip_path


def get_report_visual_tables(run_id: str) -> Dict[str, Any]:
    tables_path = _run_data_dir(run_id) / "report_visual_tables.json"
    payload = _load_json_safe(tables_path) or {}
    return payload


def get_run_report_payload(run_id: str) -> Dict[str, Any]:
    final_state = read_final_state(run_id) or {}
    run_summary = _load_json_safe(_run_data_dir(run_id) / "run_summary.json") or {}
    blocks = _get_report_blocks(run_id)
    markdown = _get_report_markdown(run_id)

    # ── If blocks are empty but the source was raw JSON, build blocks
    #    from the rescued payload so the frontend renders with formatting ──
    if not blocks:
        raw_source = None
        summary_path = _run_data_dir(run_id) / "executive_summary.md"
        if summary_path.exists():
            try:
                raw_source = summary_path.read_text(encoding="utf-8")
            except Exception:
                pass
        if not raw_source:
            raw_source = str((read_final_state(run_id) or {}).get("final_report") or "")
        if raw_source and _looks_like_raw_json_report(raw_source):
            payload = _try_parse_truncated_json(raw_source)
            if payload and isinstance(payload.get("blocks"), list):
                blocks = [b for b in payload["blocks"] if isinstance(b, dict)]

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
