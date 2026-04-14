from __future__ import annotations

import json
import os
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from src.utils.paths import run_dir
from src.utils.run_status import read_event_entries, read_final_state


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


def _fmt(value: Any) -> str:
    """Compact display for a payload value."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else f"{value:.4g}"
    if isinstance(value, int):
        return str(value)
    text = str(value).strip()
    return (text[:120] + "...") if len(text) > 120 else text


# ---------------------------------------------------------------------------
# Agent label lookup for events
# ---------------------------------------------------------------------------

_PHASE_AGENT: Dict[str, str] = {
    "steward": "Data Steward",
    "strategist": "Strategist",
    "execution_planner": "Planner",
    "planner": "Planner",
    "data_engineer": "Data Engineer",
    "ml_engineer": "ML Engineer",
    "translator": "Translator",
    "review_board": "Review Board",
    "results_advisor": "Results Advisor",
    "cleaning_reviewer": "Cleaning Reviewer",
    "reviewer": "Reviewer",
    "qa": "QA Reviewer",
    "runtime": "Runner",
    "run": "Sistema",
    "pipeline": "Sistema",
}


def _agent_for_event(event: str, payload: Dict[str, Any]) -> str:
    """Derive a human-readable agent label from an event name."""
    step = str(payload.get("step") or "").strip().lower()
    if step and step in _PHASE_AGENT:
        return _PHASE_AGENT[step]
    ev = event.lower()
    for token, label in _PHASE_AGENT.items():
        if token in ev:
            return label
    if ev.startswith("heavy_runner"):
        return "Runner"
    return "Sistema"


# ---------------------------------------------------------------------------
# Event → single log line  (returns None to skip noisy events)
# ---------------------------------------------------------------------------

def _event_to_log_line(
    event_name: str, payload: Dict[str, Any],
) -> Optional[Dict[str, str]]:
    """Convert a structured event into {agent, msg, level} or None to skip."""
    ev = event_name.strip().lower()

    # Skip noisy / internal-only events
    if ev.endswith("_view_context") or ev == "encoding_guard":
        return None

    agent = _agent_for_event(ev, payload)
    level = "info"

    # -- run lifecycle --
    if ev == "run_init":
        csv_name = Path(str(payload.get("csv_path") or "")).name or "dataset"
        return {"agent": "Sistema", "msg": f"Pipeline iniciado — {csv_name}", "level": "ok"}

    if ev == "run_finalize":
        return {"agent": "Sistema", "msg": "Pipeline finalizado.", "level": "ok"}

    # -- steward --
    if ev == "steward_start":
        return {"agent": "Data Steward", "msg": "Analizando semantica del dataset...", "level": "info"}

    if ev == "steward_complete":
        target = _fmt(payload.get("target_status") or "confirmed")
        rec = _fmt(payload.get("recommended_primary_target"))
        extra = f" (recomendado: {rec})" if rec else ""
        return {"agent": "Data Steward", "msg": f"Analisis completo — target {target}{extra}", "level": "ok"}

    if ev == "steward_target_reconsideration_start":
        rec = _fmt(payload.get("recommended_primary_target"))
        return {"agent": "Data Steward", "msg": f"Reconsideracion de target iniciada — sugerido: {rec}", "level": "warn"}

    if ev == "steward_target_reconsideration_complete":
        accepted = bool(payload.get("accepted"))
        final_t = _fmt(payload.get("final_primary_target"))
        tag = "aceptada" if accepted else "rechazada"
        extra = f" — target final: {final_t}" if final_t else ""
        return {"agent": "Data Steward", "msg": f"Reconsideracion {tag}{extra}", "level": "ok" if accepted else "warn"}

    # -- strategist --
    if ev == "strategist_start":
        return {"agent": "Strategist", "msg": "Formulando estrategia de analisis...", "level": "info"}

    if ev == "strategist_complete":
        title = _fmt(payload.get("selected_title"))
        atype = _fmt(payload.get("analysis_type"))
        extra = f" — {atype}" if atype else ""
        return {
            "agent": "Strategist",
            "msg": f"Estrategia seleccionada: {title}{extra}" if title else "Estrategia seleccionada",
            "level": "ok",
        }

    # -- planner --
    if ev == "execution_planner_start":
        strat = _fmt(payload.get("strategy"))
        return {"agent": "Planner", "msg": f"Compilando contrato — {strat}" if strat else "Compilando contrato de ejecucion...", "level": "info"}

    if ev == "execution_planner_complete":
        outputs = payload.get("required_outputs") if isinstance(payload.get("required_outputs"), list) else []
        return {"agent": "Planner", "msg": f"Contrato listo — {len(outputs)} outputs definidos", "level": "ok"}

    # -- profiling --
    if ev == "data_profile_built":
        n_r = _fmt(payload.get("n_rows"))
        n_c = _fmt(payload.get("n_cols"))
        parts = []
        if n_r:
            parts.append(f"{n_r} filas")
        if n_c:
            parts.append(f"{n_c} columnas")
        desc = " — " + ", ".join(parts) if parts else ""
        return {"agent": "Sistema", "msg": f"Dataset perfilado{desc}", "level": "info"}

    # -- data engineer --
    if ev == "data_engineer_start":
        attempt = payload.get("attempt_id") or payload.get("iteration")
        tag = f" (intento {_fmt(attempt)})" if attempt is not None else ""
        return {"agent": "Data Engineer", "msg": f"Generando script de limpieza{tag}...", "level": "info"}

    if ev == "data_engineer_backend_selection":
        backend = _fmt(payload.get("backend") or "unknown")
        scale_d = payload.get("dataset_scale") if isinstance(payload.get("dataset_scale"), dict) else {}
        scale = _fmt(scale_d.get("scale")) if scale_d else ""
        extra = f", escala {scale}" if scale else ""
        return {"agent": "Data Engineer", "msg": f"Backend seleccionado: {backend}{extra}", "level": "info"}

    if ev == "auto_fix_applied":
        fixes = payload.get("fixes") if isinstance(payload.get("fixes"), list) else []
        attempt = _fmt(payload.get("attempt"))
        return {"agent": "Data Engineer", "msg": f"Auto-fix aplicado (intento {attempt}): {len(fixes)} correcciones", "level": "warn"}

    if ev == "data_engineer_runtime_retry":
        reason = _fmt(payload.get("reason") or payload.get("error") or "error de runtime")
        return {"agent": "Data Engineer", "msg": f"Reintentando — {reason}", "level": "warn"}

    if ev == "data_engineer_dialect_autopatched":
        return {"agent": "Data Engineer", "msg": "Patch de dialecto aplicado automaticamente", "level": "info"}

    if ev == "cleaned_data_summary_min_built":
        rows = _fmt(payload.get("row_count"))
        cols = _fmt(payload.get("column_count"))
        return {"agent": "Data Engineer", "msg": f"Limpieza consolidada — {rows} filas, {cols} columnas", "level": "ok"}

    # -- cleaning reviewer --
    if ev == "cleaning_reviewer_start":
        attempt = _fmt(payload.get("attempt"))
        tag = f" (intento {attempt})" if attempt else ""
        return {"agent": "Cleaning Reviewer", "msg": f"Revisando calidad de limpieza{tag}...", "level": "info"}

    if ev == "cleaning_reviewer_complete":
        st = _fmt(payload.get("status"))
        lvl = "ok" if st in ("APPROVED", "APPROVE_WITH_WARNINGS") else "warn"
        return {"agent": "Cleaning Reviewer", "msg": f"Revision completada — {st}" if st else "Revision completada", "level": lvl}

    if ev == "cleaning_reviewer_runtime_start":
        return {"agent": "Cleaning Reviewer", "msg": "Revisando fallo de ejecucion...", "level": "info"}

    if ev == "cleaning_reviewer_runtime_complete":
        st = _fmt(payload.get("status"))
        return {"agent": "Cleaning Reviewer", "msg": f"Revision de fallo completada — {st}" if st else "Revision de fallo completada", "level": "warn"}

    if ev == "cleaning_reviewer_runtime_failed":
        return {"agent": "Cleaning Reviewer", "msg": "Revision de fallo no pudo completarse", "level": "error"}

    # -- runner --
    if ev == "heavy_runner_request":
        mode = _fmt(payload.get("mode"))
        return {"agent": "Runner", "msg": f"Enviando script al runner ({mode})", "level": "info"}

    if ev == "heavy_runner_start":
        step = _fmt(payload.get("step"))
        return {"agent": "Runner", "msg": f"Ejecutando {step}..." if step else "Ejecutando script...", "level": "info"}

    if ev == "heavy_runner_complete":
        st = _fmt(payload.get("status"))
        downloaded = payload.get("downloaded") if isinstance(payload.get("downloaded"), list) else []
        extra = f" — {len(downloaded)} artifacts" if downloaded else ""
        lvl = "ok" if st == "success" else "warn"
        return {"agent": "Runner", "msg": f"Ejecucion {st}{extra}", "level": lvl}

    # -- ML engineer --
    if ev == "ml_plan_generated":
        mp = payload.get("metric_policy") if isinstance(payload.get("metric_policy"), dict) else {}
        cv = payload.get("cv_policy") if isinstance(payload.get("cv_policy"), dict) else {}
        metric = _fmt(mp.get("primary_metric"))
        strategy = _fmt(cv.get("strategy"))
        parts = []
        if metric:
            parts.append(f"metrica: {metric}")
        if strategy:
            parts.append(f"CV: {strategy}")
        desc = " — " + ", ".join(parts) if parts else ""
        return {"agent": "ML Engineer", "msg": f"Plan de entrenamiento generado{desc}", "level": "ok"}

    if ev == "ml_engineer_start":
        attempt = payload.get("ml_engineer_attempt") or payload.get("attempt_id")
        iteration = payload.get("iteration")
        parts = []
        if iteration is not None:
            parts.append(f"iter {_fmt(iteration)}")
        if attempt is not None:
            parts.append(f"intento {_fmt(attempt)}")
        tag = " (" + ", ".join(parts) + ")" if parts else ""
        return {"agent": "ML Engineer", "msg": f"Generando modelo{tag}...", "level": "info"}

    if ev == "metric_improvement_round_skipped":
        reason = _fmt(payload.get("reason"))
        return {"agent": "ML Engineer", "msg": f"Ronda de mejora omitida — {reason}", "level": "info"}

    # -- reviewer / review board --
    if ev == "review_board_complete":
        verdict = _fmt(payload.get("final_review_verdict"))
        return {"agent": "Review Board", "msg": f"Veredicto: {verdict}" if verdict else "Evaluacion completada", "level": "ok"}

    # -- translator --
    if ev == "translator_start":
        return {"agent": "Translator", "msg": "Generando informe ejecutivo...", "level": "info"}

    if ev == "translator_complete":
        return {"agent": "Translator", "msg": "Informe ejecutivo completado.", "level": "ok"}

    # -- generic _start / _complete patterns --
    if ev.endswith("_start"):
        return {"agent": agent, "msg": f"{agent} iniciado...", "level": "info"}

    if ev.endswith("_complete"):
        return {"agent": agent, "msg": f"{agent} completado.", "level": "ok"}

    # -- fallback: show event name with any useful payload keys --
    parts: List[str] = []
    for key in ("status", "reason", "strategy", "step"):
        v = _fmt(payload.get(key))
        if v:
            parts.append(f"{key}={v}")
    desc = " — " + ", ".join(parts) if parts else ""
    title = re.sub(r"[_\-]+", " ", event_name).strip()
    return {"agent": agent, "msg": f"{title}{desc}", "level": level}


def _iso_to_local_hms(ts_iso: str) -> str:
    """Convert an ISO-8601 timestamp to local-time HH:MM:SS."""
    try:
        dt = datetime.fromisoformat(ts_iso)
        return dt.astimezone().strftime("%H:%M:%S")
    except Exception:
        return ts_iso[:8] if len(ts_iso) >= 8 else ts_iso


def list_run_event_log(run_id: str, after_line: int = 0) -> Dict[str, Any]:
    """Read events.jsonl and return entries in the same format as worker_log."""
    raw = read_event_entries(run_id, after_line=after_line)
    entries: List[Dict[str, str]] = []
    for entry in raw:
        event_name = str(entry.get("event") or entry.get("type") or "").strip()
        payload = entry.get("payload") if isinstance(entry.get("payload"), dict) else {}
        log_line = _event_to_log_line(event_name, payload)
        if log_line is None:
            continue
        ts_raw = entry.get("timestamp") or entry.get("ts") or ""
        entries.append({
            "ts": _iso_to_local_hms(str(ts_raw)),
            "agent": log_line["agent"],
            "msg": log_line["msg"],
            "level": log_line["level"],
        })
    return {
        "run_id": run_id,
        "after_line": after_line,
        "next_after_line": after_line + len(raw),
        "entries": entries,
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
