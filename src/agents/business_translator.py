import os
import re
import html
import csv
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

from string import Template
import json
from src.utils.prompting import render_prompt
from src.utils.senior_protocol import SENIOR_TRANSLATION_PROTOCOL, SENIOR_EVIDENCE_RULE
from src.utils.text_encoding import sanitize_text, sanitize_text_payload
from src.utils.csv_dialect import (
    load_output_dialect,
    sniff_csv_dialect,
    read_csv_sample,
    coerce_number,
)


def _detect_primary_language(text: str, preferred_language: Optional[str] = None) -> str:
    """
    Detect primary language using stopword heuristics.
    Returns 'es' for Spanish and 'en' for English (fallback).
    """
    if preferred_language in {"es", "en"}:
        return preferred_language

    fallback = os.getenv("TRANSLATOR_DEFAULT_LANGUAGE", "en").strip().lower()
    if fallback not in {"es", "en"}:
        fallback = "en"

    if not text:
        return fallback

    text_lower = text.lower()
    words = set(re.findall(r"\b\w+\b", text_lower))
    if len(words) < 10:
        return fallback

    # Keep only high-frequency function words (no domain-specific terms).
    spanish_markers = {
        "el", "la", "los", "las", "un", "una", "unos", "unas",
        "de", "del", "que", "qué", "en", "para", "con", "por",
        "como", "cómo", "pero", "más", "este", "esta", "esto",
        "ese", "esa", "eso", "aquel", "aquella", "hay", "ser",
        "está", "están", "son", "tiene", "tienen", "puede",
        "sobre", "entre", "cuando", "donde", "sin", "según",
    }
    english_markers = {
        "the", "a", "an", "of", "to", "and", "in", "for", "with",
        "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "must", "shall",
        "this", "that", "these", "those", "it", "its",
        "from", "by", "on", "at", "or", "but", "not", "as",
    }

    spanish_count = len(words & spanish_markers)
    english_count = len(words & english_markers)
    if spanish_count >= english_count + 2:
        return "es"
    if english_count >= spanish_count + 2:
        return "en"
    return fallback

def _safe_load_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _normalize_artifact_index(entries):
    normalized = []
    for item in entries or []:
        if isinstance(item, dict) and item.get("path"):
            normalized.append(item)
        elif isinstance(item, str):
            normalized.append({"path": item, "artifact_type": "artifact"})
    return normalized

def _first_artifact_path(entries, artifact_type: str):
    for item in entries or []:
        if not isinstance(item, dict):
            continue
        if item.get("artifact_type") == artifact_type and item.get("path"):
            return item.get("path")
    return None

def _facts_from_insights(insights: Dict[str, Any], max_items: int = 8):
    if not isinstance(insights, dict):
        return []
    metrics = insights.get("metrics_summary")
    facts = []
    if isinstance(metrics, list):
        for item in metrics:
            if not isinstance(item, dict):
                continue
            metric = item.get("metric")
            value = item.get("value")
            if metric is None or value is None:
                continue
            facts.append({"source": "insights.json", "metric": metric, "value": value, "labels": {}})
            if len(facts) >= max_items:
                break
    deployment = insights.get("deployment_recommendation")
    confidence = insights.get("confidence")
    if deployment:
        facts.append(
            {
                "source": "insights.json",
                "metric": "deployment_recommendation",
                "value": deployment,
                "labels": {"confidence": confidence or ""},
            }
        )
    return facts

def _safe_load_csv(path: str, max_rows: int = 200):
    try:
        dialect = load_output_dialect() or sniff_csv_dialect(path)
        sample = read_csv_sample(path, dialect, max_rows)
        if not sample:
            return None
        columns = sample.get("columns", [])
        if isinstance(columns, list) and len(columns) == 1 and ";" in columns[0]:
            sniffed = sniff_csv_dialect(path)
            if sniffed.get("sep") != dialect.get("sep"):
                sample = read_csv_sample(path, sniffed, max_rows)
        return sample
    except Exception:
        return None

def _summarize_numeric_columns(rows: List[Dict[str, Any]], columns: List[str], decimal: str, max_cols: int = 12):
    numeric_summary = {}
    for col in columns:
        values = []
        for row in rows:
            raw = row.get(col)
            if raw is None or raw == "":
                continue
            num = coerce_number(raw, decimal)
            if num is not None:
                values.append(num)
        if values:
            numeric_summary[col] = {
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "n": len(values),
            }
        if len(numeric_summary) >= max_cols:
            break
    return numeric_summary

def _truncate_cell(text: str, max_len: int) -> str:
    cleaned = str(text).replace("\n", " ").replace("\r", " ").strip()
    cleaned = cleaned.replace("|", "/")
    if len(cleaned) <= max_len:
        return cleaned
    suffix = " [cut]"
    if max_len <= len(suffix):
        return cleaned[:max_len]
    return cleaned[: max(0, max_len - len(suffix))] + suffix

def _compact_header(columns: List[str], head: int = 10, tail: int = 6) -> str:
    if not columns:
        return ""
    cols = [str(c) for c in columns if c]
    if len(cols) <= head + tail:
        return ", ".join(cols)
    middle_count = len(cols) - head - tail
    return ", ".join(cols[:head] + [f"[+{middle_count} more]"] + cols[-tail:])


def _looks_like_path(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    lower = text.lower()
    if lower.startswith(("data/", "static/", "reports/", "artifacts/", "report/")):
        return True
    if "/" in text or "\\" in text:
        return True
    _, ext = os.path.splitext(lower)
    return ext in {
        ".csv",
        ".json",
        ".md",
        ".txt",
        ".png",
        ".jpg",
        ".jpeg",
        ".pdf",
        ".parquet",
        ".xlsx",
        ".xls",
        ".joblib",
        ".pkl",
        ".pickle",
    }


def _normalize_path(path: Any) -> str:
    text = str(path or "").strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text


def _human_size(num_bytes: Optional[int]) -> str:
    if not isinstance(num_bytes, int) or num_bytes < 0:
        return "-"
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024.0:.1f} KB"
    return f"{num_bytes / (1024.0 * 1024.0):.2f} MB"


def _count_csv_rows(path: str) -> Optional[int]:
    dialect = load_output_dialect() or sniff_csv_dialect(path) or {}
    sep = str(dialect.get("sep") or ",")
    if len(sep) != 1:
        sep = ","
    quotechar = str(dialect.get("quotechar") or '"')
    if len(quotechar) != 1:
        quotechar = '"'
    escapechar = dialect.get("escapechar")
    if not isinstance(escapechar, str) or len(escapechar) != 1:
        escapechar = None

    encodings = [str(dialect.get("encoding") or "utf-8"), "utf-8-sig", "latin-1"]
    tried = set()
    for enc in encodings:
        if enc in tried:
            continue
        tried.add(enc)
        try:
            with open(path, "r", encoding=enc, errors="replace", newline="") as handle:
                reader = csv.reader(
                    handle,
                    delimiter=sep,
                    quotechar=quotechar,
                    escapechar=escapechar,
                )
                next(reader, None)  # header
                return sum(1 for _ in reader)
        except Exception:
            continue
    return None


def _profile_artifact(path: str) -> Dict[str, Any]:
    profile: Dict[str, Any] = {"row_count": None, "column_count": None}
    if not path or not os.path.exists(path):
        return profile
    lower = path.lower()
    try:
        if lower.endswith(".csv"):
            sample = _safe_load_csv(path, max_rows=50) or {}
            cols = sample.get("columns") if isinstance(sample.get("columns"), list) else []
            if cols:
                profile["column_count"] = len(cols)
            row_count = _count_csv_rows(path)
            if row_count is None and isinstance(sample.get("row_count_sampled"), int):
                row_count = int(sample.get("row_count_sampled"))
            profile["row_count"] = row_count
        elif lower.endswith(".json"):
            payload = _safe_load_json(path)
            if isinstance(payload, list):
                profile["row_count"] = len(payload)
                if payload and isinstance(payload[0], dict):
                    profile["column_count"] = len(payload[0].keys())
            elif isinstance(payload, dict):
                profile["row_count"] = len(payload.keys())
                profile["column_count"] = len(payload.keys())
    except Exception:
        pass
    return profile


def render_table_html(
    headers: List[str],
    rows: List[List[Any]],
    max_rows: int = 10,
    table_class: str = "exec-table",
    raw_html_columns: Optional[List[int]] = None,
) -> str:
    if not headers or not rows:
        return "<p><em>No data available.</em></p>"

    raw_cols = set(raw_html_columns or [])
    safe_rows = [row for row in rows[:max_rows] if isinstance(row, list)]
    if not safe_rows:
        return "<p><em>No data available.</em></p>"

    lines: List[str] = [f'<table class="{html.escape(table_class)}">', "<thead><tr>"]
    for header in headers:
        lines.append(f"<th>{html.escape(str(header))}</th>")
    lines.append("</tr></thead><tbody>")
    for row in safe_rows:
        lines.append("<tr>")
        for idx, cell in enumerate(row):
            if idx in raw_cols:
                lines.append(f"<td>{str(cell)}</td>")
            else:
                lines.append(f"<td>{html.escape(str(cell))}</td>")
        lines.append("</tr>")
    lines.append("</tbody></table>")
    if len(rows) > max_rows:
        lines.append(f"<p><em>Showing {max_rows} of {len(rows)} rows.</em></p>")
    return "".join(lines)


def _status_badge(status: str) -> str:
    normalized = str(status or "").strip().lower()
    label = normalized.upper() if normalized else "UNKNOWN"
    css_class = {
        "ok": "status-ok",
        "present_optional": "status-ok",
        "missing_required": "status-error",
        "missing_optional": "status-warn",
        "warning": "status-warn",
        "error": "status-error",
    }.get(normalized, "status-neutral")
    return f'<span class="status-badge {css_class}">{html.escape(label)}</span>'


def _build_report_artifact_manifest(
    artifact_index: List[Dict[str, Any]],
    required_outputs: List[str],
    output_contract_report: Dict[str, Any],
    review_verdict: Optional[str],
    gate_context: Dict[str, Any],
    run_summary: Dict[str, Any],
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    artifact_index = artifact_index if isinstance(artifact_index, list) else []
    output_contract_report = output_contract_report if isinstance(output_contract_report, dict) else {}

    required_set: set[str] = set()
    for path in required_outputs or []:
        if _looks_like_path(path):
            required_set.add(_normalize_path(path))

    for path in output_contract_report.get("missing", []) or []:
        if _looks_like_path(path):
            required_set.add(_normalize_path(path))

    missing_set: set[str] = set(_normalize_path(p) for p in (output_contract_report.get("missing", []) or []) if _looks_like_path(p))

    artifact_report = output_contract_report.get("artifact_requirements_report")
    if isinstance(artifact_report, dict):
        files_report = artifact_report.get("files_report")
        if isinstance(files_report, dict):
            for path in files_report.get("missing", []) or []:
                if _looks_like_path(path):
                    normalized = _normalize_path(path)
                    required_set.add(normalized)
                    missing_set.add(normalized)

    path_to_type: Dict[str, str] = {}
    candidate_paths: List[str] = []
    seen_paths: set[str] = set()

    def _add_candidate(path_value: Any, artifact_type: str = "") -> None:
        if not _looks_like_path(path_value):
            return
        normalized = _normalize_path(path_value)
        if not normalized or normalized in seen_paths:
            if normalized and artifact_type and not path_to_type.get(normalized):
                path_to_type[normalized] = str(artifact_type)
            return
        seen_paths.add(normalized)
        candidate_paths.append(normalized)
        if artifact_type:
            path_to_type[normalized] = str(artifact_type)

    for item in artifact_index:
        if not isinstance(item, dict):
            continue
        _add_candidate(item.get("path"), item.get("artifact_type") or "")
    for path in required_set:
        _add_candidate(path, path_to_type.get(path, "required_output"))
    for path in output_contract_report.get("present", []) or []:
        _add_candidate(path, "present_output")
    for path in output_contract_report.get("missing", []) or []:
        _add_candidate(path, "required_output")

    items: List[Dict[str, Any]] = []
    for path in sorted(candidate_paths):
        present = os.path.exists(path)
        required = path in required_set
        if path in missing_set:
            status = "missing_required"
        elif required and present:
            status = "ok"
        elif required and not present:
            status = "missing_required"
        elif present:
            status = "present_optional"
        else:
            status = "missing_optional"
        stat_size = None
        stat_mtime = None
        if present:
            try:
                stat = os.stat(path)
                stat_size = int(stat.st_size)
                stat_mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
            except Exception:
                pass
        profile = _profile_artifact(path)
        items.append(
            {
                "path": path,
                "artifact_type": path_to_type.get(path) or "artifact",
                "required": bool(required),
                "present": bool(present),
                "status": status,
                "size_bytes": stat_size,
                "updated_at_utc": stat_mtime,
                "row_count": profile.get("row_count"),
                "column_count": profile.get("column_count"),
            }
        )

    required_total = sum(1 for item in items if item.get("required"))
    required_missing = sum(1 for item in items if item.get("required") and not item.get("present"))
    summary = {
        "required_total": required_total,
        "required_present": required_total - required_missing,
        "required_missing": required_missing,
        "optional_present": sum(1 for item in items if not item.get("required") and item.get("present")),
        "optional_missing": sum(1 for item in items if not item.get("required") and not item.get("present")),
    }

    manifest = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "run_id": run_id,
        "summary": summary,
        "items": items,
        "governance_snapshot": {
            "review_verdict": review_verdict,
            "run_outcome": run_summary.get("run_outcome") if isinstance(run_summary, dict) else None,
            "failed_gates": (
                run_summary.get("failed_gates", [])
                if isinstance(run_summary, dict) and isinstance(run_summary.get("failed_gates"), list)
                else (gate_context.get("failed_gates", []) if isinstance(gate_context, dict) else [])
            ),
            "required_fixes": gate_context.get("required_fixes", []) if isinstance(gate_context, dict) else [],
            "output_contract_status": output_contract_report.get("overall_status"),
        },
    }
    return manifest


def _build_artifact_inventory_table_html(manifest: Dict[str, Any], max_rows: int = 14) -> str:
    items = manifest.get("items", []) if isinstance(manifest, dict) else []
    rows: List[List[Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        rows.append(
            [
                item.get("path") or "",
                item.get("artifact_type") or "artifact",
                "yes" if item.get("required") else "no",
                _status_badge(item.get("status") or ""),
                item.get("row_count") if item.get("row_count") is not None else "-",
                item.get("column_count") if item.get("column_count") is not None else "-",
                _human_size(item.get("size_bytes")),
            ]
        )
    return render_table_html(
        ["Artifact", "Type", "Required", "Status", "Rows", "Cols", "Size"],
        rows,
        max_rows=max_rows,
        table_class="exec-table artifact-inventory",
        raw_html_columns=[3],
    )


def _build_artifact_compliance_table_html(
    manifest: Dict[str, Any],
    output_contract_report: Dict[str, Any],
    review_verdict: Optional[str],
    gate_context: Dict[str, Any],
    run_summary: Optional[Dict[str, Any]] = None,
) -> str:
    summary = manifest.get("summary", {}) if isinstance(manifest, dict) else {}
    overall_status = str(output_contract_report.get("overall_status") or "unknown").lower()
    failed_gates: List[str] = []
    if isinstance(run_summary, dict):
        summary_failed = run_summary.get("failed_gates")
        if isinstance(summary_failed, list):
            failed_gates = [str(g) for g in summary_failed if g]
    if not failed_gates and isinstance(gate_context, dict):
        context_failed = gate_context.get("failed_gates", [])
        if isinstance(context_failed, list):
            failed_gates = [str(g) for g in context_failed if g]
    rows: List[List[Any]] = [
        ["Output Contract Status", _status_badge(overall_status)],
        ["Review Verdict", html.escape(str(review_verdict or "UNKNOWN"))],
        ["Required Artifacts", str(summary.get("required_total", 0))],
        ["Required Missing", str(summary.get("required_missing", 0))],
        ["Failed Gates", html.escape(", ".join([str(g) for g in failed_gates[:6]]) or "none")],
    ]
    missing = output_contract_report.get("missing", []) if isinstance(output_contract_report, dict) else []
    if isinstance(missing, list) and missing:
        rows.append(["Missing Paths", "<br/>".join(html.escape(_normalize_path(path)) for path in missing[:8])])
    return render_table_html(
        ["Check", "Value"],
        rows,
        max_rows=10,
        table_class="exec-table artifact-compliance",
        raw_html_columns=[1],
    )


def _build_kpi_snapshot_table_html(
    metrics_payload: Dict[str, Any],
    data_adequacy_report: Dict[str, Any],
    decisioning_columns: List[str],
    executive_decision_label: str,
    max_metric_rows: int = 8,
) -> str:
    rows: List[List[Any]] = []
    rows.append(["Executive Decision", executive_decision_label or "UNKNOWN"])
    if isinstance(data_adequacy_report, dict):
        rows.append(["Data Adequacy Status", str(data_adequacy_report.get("status") or "unknown")])
    if decisioning_columns:
        rows.append(["Decisioning Columns", ", ".join([str(col) for col in decisioning_columns[:6]])])
    flat = _flatten_metrics(metrics_payload if isinstance(metrics_payload, dict) else {})
    metric_count = 0
    for key, value in flat:
        if metric_count >= max_metric_rows:
            break
        if _is_number(value):
            rows.append([f"metric:{key}", f"{float(value):.6g}"])
            metric_count += 1
    return render_table_html(
        ["KPI / Signal", "Value"],
        rows,
        max_rows=14,
        table_class="exec-table kpi-snapshot",
    )


def _load_run_timeline_tail(run_id: Optional[str], max_events: int = 12) -> List[Dict[str, Any]]:
    candidates: List[str] = []
    if run_id:
        candidates.append(os.path.join("runs", str(run_id), "events.jsonl"))
    candidates.append("events.jsonl")

    for path in candidates:
        if not path or not os.path.exists(path):
            continue
        try:
            rows: List[Dict[str, Any]] = []
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    if isinstance(payload, dict):
                        rows.append(payload)
            if not rows:
                continue
            tail = rows[-max_events:]
            normalized: List[Dict[str, Any]] = []
            for item in tail:
                normalized.append(
                    {
                        "ts": item.get("timestamp") or item.get("ts"),
                        "event": item.get("event") or item.get("type") or item.get("name"),
                        "status": item.get("status"),
                    }
                )
            return normalized
        except Exception:
            continue
    return []

def render_table_text(headers: List[str], rows: List[List[str]], max_rows: int = 8, max_cell_len: int = 28) -> str:
    """
    Render a professional ASCII table suitable for PDF monospaced fonts.

    Uses a clean grid format inspired by tabulate's "simple" style:
    +------------+------------+------------+
    | Header 1   | Header 2   | Header 3   |
    +============+============+============+
    | Value 1    | Value 2    | Value 3    |
    +------------+------------+------------+

    This looks professional in executive reports while avoiding markdown tables
    that break PDF generators.
    """
    if not headers or not rows:
        return "(No data available)"

    safe_rows = []
    for row in rows[:max_rows]:
        if not isinstance(row, list):
            continue
        safe_rows.append([_truncate_cell(cell, max_cell_len) for cell in row])
    if not safe_rows:
        return "(No data available)"

    # Calculate column widths
    widths = []
    for idx, header in enumerate(headers):
        header_text = _truncate_cell(header, max_cell_len)
        col_width = len(header_text)
        for row in safe_rows:
            if idx < len(row):
                col_width = max(col_width, len(row[idx]))
        widths.append(min(col_width + 2, max_cell_len))  # Add padding

    # Build grid lines
    def _build_separator(char: str = "-", corner: str = "+") -> str:
        parts = [corner]
        for width in widths:
            parts.append(char * (width + 2))
            parts.append(corner)
        return "".join(parts)

    def _build_row(cells: List[str]) -> str:
        parts = ["|"]
        for idx, width in enumerate(widths):
            cell = cells[idx] if idx < len(cells) else ""
            cell_text = _truncate_cell(cell, max_cell_len)
            parts.append(f" {cell_text.ljust(width)} ")
            parts.append("|")
        return "".join(parts)

    # Assemble table
    lines = []
    lines.append(_build_separator("-", "+"))
    lines.append(_build_row([_truncate_cell(h, max_cell_len) for h in headers]))
    lines.append(_build_separator("=", "+"))  # Double line under header

    for row in safe_rows:
        lines.append(_build_row(row))

    lines.append(_build_separator("-", "+"))

    # Add row count footer if truncated
    if len(rows) > max_rows:
        lines.append(f"  ... ({len(rows) - max_rows} more rows)")

    return "\n".join(lines)

def _flatten_metrics(metrics: Dict[str, Any], prefix: str = "") -> List[tuple[str, Any]]:
    items: List[tuple[str, Any]] = []
    if not isinstance(metrics, dict):
        return items
    for key, value in metrics.items():
        metric_key = f"{prefix}{key}" if prefix else str(key)
        if isinstance(value, dict):
            items.extend(_flatten_metrics(value, f"{metric_key}."))
        else:
            items.append((metric_key, value))
    return items

def _select_informative_columns(sample: Dict[str, Any], max_cols: int = 8, min_cols: int = 5) -> List[str]:
    if not isinstance(sample, dict):
        return []
    columns = sample.get("columns", []) or []
    rows = sample.get("rows", []) or []
    decimal = (sample.get("dialect_used") or {}).get("decimal") or "."
    if not columns:
        return []
    if not rows:
        return columns[:max_cols]
    stats = []
    for col in columns:
        values = []
        for row in rows[:50]:
            raw = row.get(col)
            if raw not in (None, ""):
                values.append(raw)
        if not values:
            continue
        numeric_hits = sum(1 for val in values if coerce_number(val, decimal) is not None)
        unique_count = len({str(val) for val in values})
        ratio = numeric_hits / max(len(values), 1)
        stats.append(
            {
                "col": col,
                "numeric_ratio": ratio,
                "unique_count": unique_count,
                "non_null": len(values),
            }
        )
    numeric_cols = [item for item in stats if item["numeric_ratio"] >= 0.6]
    cat_cols = [item for item in stats if item["numeric_ratio"] < 0.6]
    numeric_cols.sort(key=lambda item: item["non_null"], reverse=True)
    cat_cols.sort(key=lambda item: item["unique_count"])
    selected = [item["col"] for item in numeric_cols[:4]]
    selected.extend([item["col"] for item in cat_cols[:4] if item["col"] not in selected])
    if len(selected) < min_cols:
        for col in columns:
            if col not in selected:
                selected.append(col)
            if len(selected) >= min_cols:
                break
    return selected[:max_cols]

def _select_scored_columns(sample: Dict[str, Any], max_cols: int = 6) -> List[str]:
    if not isinstance(sample, dict):
        return []
    columns = sample.get("columns", []) or []
    if not columns:
        return []
    tokens = ["pred", "prob", "score", "segment", "cluster", "group", "rank", "risk", "expected", "optimal", "recommend"]
    preferred = []
    for col in columns:
        norm = col.lower()
        if any(tok in norm for tok in tokens):
            preferred.append(col)
    for col in columns:
        if col.lower() in {"row_id", "case_id", "caseid", "id"} and col not in preferred:
            preferred.insert(0, col)
    if not preferred:
        return _select_informative_columns(sample, max_cols=max_cols, min_cols=min(5, max_cols))
    deduped = []
    for col in preferred:
        if col not in deduped:
            deduped.append(col)
    return deduped[:max_cols]

def _rows_from_sample(sample: Dict[str, Any], columns: List[str], max_rows: int = 5) -> List[List[str]]:
    rows = sample.get("rows", []) if isinstance(sample, dict) else []
    out = []
    for row in rows[:max_rows]:
        out.append([str(row.get(col, "")) for col in columns])
    return out

def _metrics_table(metrics_payload: Dict[str, Any], max_items: int = 10) -> str:
    if not isinstance(metrics_payload, dict):
        return "No data available."
    model_perf = metrics_payload.get("model_performance") if isinstance(metrics_payload.get("model_performance"), dict) else {}
    items: List[tuple[str, str]] = []
    if isinstance(model_perf, dict):
        for key, value in model_perf.items():
            if isinstance(value, dict) and {"mean", "ci_lower", "ci_upper"}.issubset(value.keys()):
                mean = value.get("mean")
                lower = value.get("ci_lower")
                upper = value.get("ci_upper")
                items.append((f"model_performance.{key}", f"mean={mean} ci=[{lower}, {upper}]"))
            elif _is_number(value):
                items.append((f"model_performance.{key}", str(value)))
            if len(items) >= max_items:
                break
    if len(items) < max_items:
        flat = _flatten_metrics(metrics_payload)
        for key, value in flat:
            if len(items) >= max_items:
                break
            if key.startswith("model_performance."):
                continue
            if _is_number(value):
                items.append((key, str(value)))
    if not items:
        return "No data available."
    rows = [[metric, val] for metric, val in items[:max_items]]
    return render_table_text(["metric", "value"], rows, max_rows=max_items)

def _recommendations_table(preview: Dict[str, Any], max_rows: int = 3) -> str:
    if not isinstance(preview, dict):
        return "No data available."
    items = preview.get("items")
    if not isinstance(items, list) or not items:
        return "No data available."
    keys = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        for key in item.keys():
            keys[key] = keys.get(key, 0) + 1
    sorted_keys = [k for k, _ in sorted(keys.items(), key=lambda kv: kv[1], reverse=True)]
    headers = sorted_keys[:4]
    if not headers:
        return "No data available."
    rows = []
    for item in items[:max_rows]:
        if not isinstance(item, dict):
            continue
        rows.append([str(item.get(key, "")) for key in headers])
    return render_table_text(headers, rows, max_rows=max_rows)

def _pick_top_examples(rows: List[Dict[str, Any]], columns: List[str], value_keys: List[str], label_keys: List[str], decimal: str, max_rows: int = 3):
    if not rows or not columns:
        return None
    value_key = None
    for key in value_keys:
        if key in columns:
            value_key = key
            break
    if not value_key:
        return None
    def _coerce_num(row):
        raw = row.get(value_key)
        if raw is None or raw == "":
            return None
        return coerce_number(raw, decimal)
    scored = []
    for row in rows:
        val = _coerce_num(row)
        if val is None:
            continue
        scored.append((val, row))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    examples = []
    for val, row in scored[:max_rows]:
        example = {"value_key": value_key, "value": val}
        for label in label_keys:
            if label in row and row.get(label) not in (None, ""):
                example[label] = row.get(label)
        examples.append(example)
    return examples

def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False


def _coerce_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _extract_lift_value(data_adequacy_report: Dict[str, Any], metrics_payload: Dict[str, Any]) -> Optional[float]:
    if isinstance(data_adequacy_report, dict):
        signals = data_adequacy_report.get("signals", {})
        for key in ("metric_lift", "classification_lift", "regression_lift", "f1_lift", "mae_lift"):
            lift = _coerce_float(signals.get(key)) if isinstance(signals, dict) else None
            if lift is not None:
                return lift
    if isinstance(metrics_payload, dict):
        model_perf = metrics_payload.get("model_performance") if isinstance(metrics_payload.get("model_performance"), dict) else {}
        for key, value in model_perf.items():
            if "lift" in str(key).lower():
                lift = _coerce_float(value if not isinstance(value, dict) else value.get("mean"))
                if lift is not None:
                    return lift
    return None


def _derive_exec_decision(
    review_verdict: Optional[str],
    data_adequacy_report: Dict[str, Any],
    metrics_payload: Dict[str, Any],
) -> str:
    verdict = str(review_verdict or "").upper()
    status = str(data_adequacy_report.get("status") if isinstance(data_adequacy_report, dict) else "").lower()
    lift = _extract_lift_value(data_adequacy_report, metrics_payload)
    min_positive_lift = _coerce_float(os.getenv("TRANSLATOR_MIN_POSITIVE_LIFT", "0.01"))
    if min_positive_lift is None:
        min_positive_lift = 0.01

    if verdict in {"NEEDS_IMPROVEMENT", "REJECTED"}:
        return "NO_GO"
    if status in {"unknown", "insufficient_signal"}:
        return "GO_WITH_LIMITATIONS"
    if status == "data_limited":
        if lift is not None and lift < 0:
            return "NO_GO"
        return "GO_WITH_LIMITATIONS"
    if status == "sufficient_signal":
        if lift is None:
            return "GO_WITH_LIMITATIONS"
        if lift < 0:
            return "NO_GO"
        if lift < min_positive_lift:
            return "GO_WITH_LIMITATIONS"
        return "GO"
    return "GO_WITH_LIMITATIONS"


def _sanitize_report_text(text: str) -> str:
    if not text:
        return text
    text = sanitize_text(text)
    # Only collapse isolated ellipsis tokens to avoid corrupting paths or ranges.
    text = re.sub(r"(?<=\s)\.\.\.(?=\s|$)", ".", text)
    text = re.sub(r"(?m)^\s*\.\.\.\s*$", "", text)
    return text

def _sanitize_evidence_value(value: str) -> str:
    text = str(value).replace("\n", " ").replace("\r", " ").strip()
    return text.replace('"', "'")

def _build_evidence_items(evidence_paths: List[str], min_items: int = 3, max_items: int = 6):
    items = []
    for path in evidence_paths or []:
        if not path:
            continue
        if len(items) >= max_items:
            break
        clean_path = _sanitize_evidence_value(path)
        items.append({"claim": f"Artifact available: {clean_path}", "source": clean_path})
    while len(items) < min_items:
        items.append({"claim": "No verificable con artifacts actuales", "source": "missing"})
    return items

def _is_valid_evidence_source(source: str) -> bool:
    if not source:
        return False
    normalized = str(source).strip()
    if not normalized:
        return False
    lower = normalized.lower()
    if lower == "missing":
        return True
    if lower.startswith("script:"):
        return True
    if "/" in normalized or "\\" in normalized:
        return True
    for ext in (".json", ".csv", ".md", ".txt", ".py"):
        if ext in lower:
            return True
    return False

def _normalize_evidence_sources(report: str) -> str:
    if not report:
        return report
    lines = report.splitlines()
    out = []
    for line in lines:
        if "{claim" not in line or "source" not in line:
            out.append(line)
            continue
        match = re.search(r'source\s*:\s*\"([^\"]*)\"', line)
        if not match:
            match = re.search(r"source\s*:\s*'([^']*)'", line)
        if not match:
            out.append(line)
            continue
        source = match.group(1)
        if _is_valid_evidence_source(source):
            out.append(line)
            continue
        updated = line[: match.start(1)] + "missing" + line[match.end(1):]
        out.append(updated)
    return "\n".join(out)


def _parse_evidence_items_from_report(report: str) -> List[Dict[str, str]]:
    if not report:
        return []
    header = re.search(r"(?im)^\s*##\s+evidencia usada\s*$", report)
    if not header:
        return []
    section = report[header.end():]
    claims = re.findall(
        r"\{claim:\s*\"([^\"]*)\"\s*,\s*source:\s*\"([^\"]*)\"\s*\}",
        section,
        flags=re.IGNORECASE,
    )
    items: List[Dict[str, str]] = []
    for claim, source in claims:
        clean_claim = _sanitize_evidence_value(claim)
        clean_source = _sanitize_evidence_value(source)
        if not clean_claim:
            continue
        if not _is_valid_evidence_source(clean_source):
            clean_source = "missing"
        items.append({"claim": clean_claim, "source": clean_source})
    return items


def _canonical_evidence_section(evidence_paths: List[str], llm_items: Optional[List[Dict[str, str]]] = None) -> str:
    validated_llm_items: List[Dict[str, str]] = []
    for item in (llm_items or []):
        if not isinstance(item, dict):
            continue
        claim = _sanitize_evidence_value(item.get("claim", ""))
        source = _sanitize_evidence_value(item.get("source", ""))
        if not claim:
            continue
        if not _is_valid_evidence_source(source):
            source = "missing"
        validated_llm_items.append({"claim": claim, "source": source})

    if validated_llm_items:
        items = validated_llm_items[:6]
        generic = _build_evidence_items(evidence_paths, min_items=0, max_items=6)
        for item in generic:
            if len(items) >= 6:
                break
            if any(existing.get("source") == item.get("source") for existing in items):
                continue
            items.append(item)
        while len(items) < 3:
            items.append({"claim": "No verificable con artifacts actuales", "source": "missing"})
    else:
        items = _build_evidence_items(evidence_paths)

    evidence_lines = ["evidence:"]
    for item in items:
        claim = _sanitize_evidence_value(item.get("claim", ""))
        source = _sanitize_evidence_value(item.get("source", "missing")) or "missing"
        evidence_lines.append(f'{{claim: "{claim}", source: "{source}"}}')
    dedup_paths: List[str] = []
    for path in (evidence_paths or []):
        clean = _sanitize_evidence_value(path)
        if clean and clean not in dedup_paths:
            dedup_paths.append(clean)
        if len(dedup_paths) >= 8:
            break
    path_lines = [f"- {path}" for path in dedup_paths] or ["- missing"]
    return "\n".join(evidence_lines + ["", "Artifacts:"] + path_lines)


def _ensure_evidence_section(report: str, evidence_paths: List[str]) -> str:
    if not report:
        return report
    report = sanitize_text(report)
    llm_items = _parse_evidence_items_from_report(report)
    evidence_block = _canonical_evidence_section(evidence_paths, llm_items=llm_items)

    header_match = re.search(r"(?im)^\s*##\s+evidencia usada\s*$", report)
    if header_match:
        prefix = report[:header_match.start()].rstrip()
    else:
        prefix = report.rstrip()

    rebuilt = f"{prefix}\n\n## Evidencia usada\n\n{evidence_block}\n"
    return _normalize_evidence_sources(rebuilt)

def _extract_numeric_metrics(metrics: Dict[str, Any], max_items: int = 8):
    if not isinstance(metrics, dict):
        return []
    items = []
    for key, value in metrics.items():
        if _is_number(value):
            items.append((str(key), float(value)))
        if len(items) >= max_items:
            break
    return items

def _build_fact_cards(case_summary_ctx, scored_rows_ctx, weights_ctx, data_adequacy_ctx, max_items: int = 8):
    facts = []
    if isinstance(case_summary_ctx, dict):
        examples = case_summary_ctx.get("examples") or []
        for example in examples:
            value_key = example.get("value_key")
            value = example.get("value")
            if value_key and _is_number(value):
                labels = {k: v for k, v in example.items() if k not in {"value_key", "value"}}
                facts.append({
                    "source": "case_summary",
                    "metric": value_key,
                    "value": float(value),
                    "labels": labels,
                })
    if isinstance(scored_rows_ctx, dict):
        examples = scored_rows_ctx.get("examples") or []
        for example in examples:
            value_key = example.get("value_key")
            value = example.get("value")
            if value_key and _is_number(value):
                labels = {k: v for k, v in example.items() if k not in {"value_key", "value"}}
                facts.append({
                    "source": "predictions",
                    "metric": value_key,
                    "value": float(value),
                    "labels": labels,
                })
    if isinstance(weights_ctx, dict):
        for key in ("metrics", "classification", "regression", "propensity_model", "price_model"):
            metrics = weights_ctx.get(key)
            for metric_key, metric_val in _extract_numeric_metrics(metrics):
                facts.append({
                    "source": "weights",
                    "metric": metric_key,
                    "value": metric_val,
                    "labels": {"model_block": key},
                })
    if isinstance(data_adequacy_ctx, dict):
        signals = data_adequacy_ctx.get("signals", {})
        for metric_key, metric_val in _extract_numeric_metrics(signals):
            facts.append({
                "source": "data_adequacy_report.json",
                "metric": metric_key,
                "value": metric_val,
                "labels": {},
            })
    return facts[:max_items]


def _estimate_prompt_tokens(text: str) -> int:
    return max(1, len(str(text or "")) // 4)


def _normalize_decision_token(value: str) -> Optional[str]:
    text = str(value or "").upper()
    if "GO_WITH_LIMITATIONS" in text:
        return "GO_WITH_LIMITATIONS"
    if re.search(r"\bNO_GO\b", text):
        return "NO_GO"
    if re.search(r"\bGO\b", text):
        return "GO"
    return None


def _extract_report_decision(content: str) -> Optional[str]:
    if not content:
        return None
    for line in content.splitlines()[:40]:
        token = _normalize_decision_token(line)
        if token:
            return token
    token = _normalize_decision_token(content)
    return token


def _collect_numeric_reference_values(
    facts_context: List[Dict[str, Any]],
    metrics_payload: Dict[str, Any],
) -> List[float]:
    refs: List[float] = []
    for item in facts_context or []:
        if not isinstance(item, dict):
            continue
        value = _coerce_float(item.get("value"))
        if value is not None:
            refs.append(value)
    for _, value in _flatten_metrics(metrics_payload if isinstance(metrics_payload, dict) else {}):
        num = _coerce_float(value if not isinstance(value, dict) else value.get("mean"))
        if num is not None:
            refs.append(num)
    deduped: List[float] = []
    for value in refs:
        if any(abs(value - existing) <= 1e-9 for existing in deduped):
            continue
        deduped.append(value)
    return deduped


def _extract_report_metric_claims(content: str) -> List[Tuple[str, float, str]]:
    if not content:
        return []
    patterns = [
        ("accuracy", r"(?i)accuracy\s*[:=]?\s*(\d+(?:\.\d+)?)\s*%?"),
        ("f1", r"(?i)\bf1\b\s*[:=]?\s*(\d+(?:\.\d+)?)\s*%?"),
        ("auc", r"(?i)\b(?:roc[-_\s]?auc|auc)\b\s*[:=]?\s*(\d+(?:\.\d+)?)\s*%?"),
        ("gini", r"(?i)\b(?:normalized[_\s-]?gini|gini)\b\s*[:=]?\s*(\d+(?:\.\d+)?)\s*%?"),
        ("rmse", r"(?i)\brmse\b\s*[:=]?\s*(\d+(?:\.\d+)?)"),
        ("mae", r"(?i)\bmae\b\s*[:=]?\s*(\d+(?:\.\d+)?)"),
        ("r2", r"(?i)\br\^?2\b\s*[:=]?\s*(\d+(?:\.\d+)?)"),
        ("lift", r"(?i)\blift\b\s*[:=]?\s*([+-]?\d+(?:\.\d+)?)\s*%?"),
    ]
    claims: List[Tuple[str, float, str]] = []
    for metric_name, pattern in patterns:
        for match in re.finditer(pattern, content):
            raw = match.group(1)
            value = _coerce_float(raw)
            if value is None:
                continue
            full_match = match.group(0)
            claims.append((metric_name, float(value), full_match))
    return claims


def _value_matches_references(value: float, refs: List[float], relative_tolerance: float = 0.05) -> bool:
    if not refs:
        return True
    candidates = [value]
    if value > 1:
        candidates.append(value / 100.0)
    for candidate in candidates:
        for ref in refs:
            abs_tol = max(1e-6, abs(ref) * relative_tolerance)
            if abs(candidate - ref) <= abs_tol:
                return True
    return False


def _validate_report_structure(content: str, expected_language: str) -> List[str]:
    issues: List[str] = []
    if not content:
        return ["empty_report"]
    length = len(content)
    if length < 500:
        issues.append("report_too_short")
    if length > 30000:
        issues.append("report_too_long")
    decision_header = re.search(r"(?im)^\s*##\s+(Executive Decision|Decisi[oó]n Ejecutiva)\s*$", content)
    evidence_header = re.search(r"(?im)^\s*##\s+Evidencia\s+[Uu]sada\s*$", content)
    risk_header = re.search(r"(?im)^\s*##\s+(Risks|Riesgos)(\s*&?\s*(Limitations|Limitaciones))?\s*$", content)
    if not decision_header:
        issues.append("missing_decision_section")
    if not evidence_header:
        issues.append("missing_evidence_section")
    if not risk_header:
        issues.append("missing_risks_section")
    if expected_language == "es":
        if re.search(r"(?i)\b(the|therefore|however)\b", content[:600]):
            issues.append("possible_language_mix")
    return issues


def _validate_report(
    content: str,
    expected_decision: str,
    facts_context: List[Dict[str, Any]],
    metrics_payload: Dict[str, Any],
    plots: List[str],
    expected_language: str,
) -> Dict[str, Any]:
    structure_issues = _validate_report_structure(content, expected_language=expected_language)
    decision_in_report = _extract_report_decision(content)
    decision_issue = []
    if decision_in_report and expected_decision and decision_in_report != expected_decision:
        decision_issue.append(f"decision_mismatch:{decision_in_report}!={expected_decision}")
    elif not decision_in_report:
        decision_issue.append("decision_missing")

    refs = _collect_numeric_reference_values(facts_context, metrics_payload)
    metric_claims = _extract_report_metric_claims(content)
    unverified_metrics: List[str] = []
    for metric_name, value, excerpt in metric_claims:
        if not _value_matches_references(value, refs):
            unverified_metrics.append(f"{metric_name}:{excerpt}")

    allowed_plots = {str(path).replace("\\", "/") for path in (plots or [])}
    invalid_plots: List[str] = []
    for path in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", content):
        normalized = str(path).strip().replace("\\", "/")
        if allowed_plots and normalized not in allowed_plots:
            invalid_plots.append(normalized)

    critical_issues: List[str] = []
    for issue in structure_issues:
        if issue in {"missing_decision_section", "missing_evidence_section", "report_too_short"}:
            critical_issues.append(issue)
    critical_issues.extend(decision_issue)
    if len(unverified_metrics) > 2:
        critical_issues.append("unverified_metrics_gt2")
    if invalid_plots:
        critical_issues.append("invalid_plot_reference")

    return {
        "structure_issues": structure_issues,
        "decision_issue": decision_issue,
        "unverified_metrics": unverified_metrics,
        "invalid_plots": invalid_plots,
        "critical_issues": critical_issues,
        "has_critical": bool(critical_issues),
    }


def _score_report_quality(validation: Dict[str, Any]) -> int:
    score = 100
    score -= 8 * len(validation.get("structure_issues", []))
    score -= 12 * len(validation.get("decision_issue", []))
    score -= 5 * min(4, len(validation.get("unverified_metrics", [])))
    score -= 6 * len(validation.get("invalid_plots", []))
    return max(0, min(100, score))


def _build_repair_prompt(
    report: str,
    validation: Dict[str, Any],
    expected_decision: str,
    evidence_paths: List[str],
    target_language_code: str,
) -> str:
    issues = validation.get("critical_issues", []) + validation.get("structure_issues", [])
    issues_text = "\n".join(f"- {issue}" for issue in issues) or "- unknown_issue"
    evidence_paths_text = "\n".join(f"- {path}" for path in evidence_paths[:8]) or "- missing"
    return render_prompt(
        """
        Repair the executive report below without discarding useful content.
        Target language: $lang.
        Required decision label: $decision.

        Issues to fix:
        $issues

        Hard constraints:
        - Keep the report evidence-based and avoid inventing metrics.
        - Ensure sections exist: Decisión Ejecutiva, Riesgos, Evidencia Usada.
        - Keep "## Evidencia usada" with evidence:{claim,source} and artifact bullets.
        - Use these artifact paths when citing evidence:
        $evidence_paths

        Original report:
        ---
        $report
        ---
        """,
        lang=target_language_code,
        decision=expected_decision,
        issues=issues_text,
        evidence_paths=evidence_paths_text,
        report=report or "(empty)",
    )


def _generate_deterministic_fallback_report(
    *,
    target_language_code: str,
    executive_decision_label: str,
    business_objective: str,
    strategy_title: str,
    error_message: str,
    facts_context: List[Dict[str, Any]],
    evidence_paths: List[str],
) -> str:
    is_es = target_language_code == "es"
    title = "# Reporte Ejecutivo (Fallback Determinístico)" if is_es else "# Executive Report (Deterministic Fallback)"
    objective_title = "## Objetivo y enfoque" if is_es else "## Objective and approach"
    decision_title = "## Decisión Ejecutiva" if is_es else "## Executive Decision"
    risks_title = "## Riesgos" if is_es else "## Risks"
    actions_title = "## Próximas acciones" if is_es else "## Next actions"
    warning_note = (
        "Este reporte se generó de forma determinística por un fallo de traducción del LLM."
        if is_es else
        "This report was generated deterministically due to an LLM translation failure."
    )
    lines: List[str] = [
        title,
        "",
        decision_title,
        f"- {executive_decision_label}",
        "",
        objective_title,
        f"- {business_objective or 'N/A'}",
        f"- Strategy: {strategy_title or 'N/A'}",
        "",
        "## Evidence & Metrics",
    ]
    if facts_context:
        for fact in facts_context[:6]:
            lines.append(
                f"- {fact.get('metric', 'metric')}: {fact.get('value', 'N/A')} (source: {fact.get('source', 'missing')})"
            )
    else:
        lines.append("- No metric facts available in artifacts.")
    lines.extend([
        "",
        risks_title,
        f"- {warning_note}",
        f"- Error: {error_message}",
        "",
        actions_title,
        "- Review artifacts directly and regenerate the executive report.",
        "- Validate metric integrity before making a production decision.",
    ])
    lines.append("")
    lines.append("## Evidencia Usada")
    lines.append("")
    lines.append(_canonical_evidence_section(evidence_paths))
    return "\n".join(lines) + "\n"


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    raw = str(text).strip()
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    n = len(raw)
    for start in range(n):
        if raw[start] != "{":
            continue
        depth = 0
        for end in range(start, n):
            ch = raw[end]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = raw[start:end + 1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            return parsed
                    except Exception:
                        break
    return None


def _validate_outline_payload(payload: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    if not isinstance(payload, dict):
        return ["outline_not_dict"]
    decision = payload.get("executive_decision")
    if not isinstance(decision, dict) or not decision.get("label"):
        issues.append("outline_missing_executive_decision")
    sections = payload.get("sections")
    if not isinstance(sections, list) or len(sections) < 3:
        issues.append("outline_sections_insufficient")
        return issues
    for idx, section in enumerate(sections[:10]):
        if not isinstance(section, dict):
            issues.append(f"outline_section_{idx}_not_dict")
            continue
        heading = section.get("heading") or section.get("title")
        if not heading:
            issues.append(f"outline_section_{idx}_missing_heading")
        bullets = section.get("bullets")
        if not isinstance(bullets, list) or not bullets:
            issues.append(f"outline_section_{idx}_missing_bullets")
    return issues


def _build_outline_prompt(
    *,
    target_language_code: str,
    executive_decision_label: str,
    facts_block: Dict[str, Any],
    reporting_policy_context: Dict[str, Any],
    evidence_paths: List[str],
    execution_results: str,
) -> str:
    evidence_paths_text = "\n".join(f"- {p}" for p in evidence_paths[:8]) or "- missing"
    return render_prompt(
        """
You are a senior executive reporting planner.
Generate ONLY valid JSON (no markdown, no commentary), in language code: $target_language_code.

Goal: produce an outline plan for the final executive report.
The final decision label must be: $executive_decision_label

Reasoning workflow:
1. Decide what an executive needs to know first.
2. Group the evidence into a small number of sections that explain decision, evidence, risks, and next actions.
3. Use only supported claims; if evidence is weak, surface that uncertainty in the outline.

FACTS_BLOCK:
$facts_block_json

reporting_policy:
$reporting_policy_json

evidence_paths:
$evidence_paths

execution_results:
$execution_results

Return JSON with this schema:
{
  "executive_decision": {"label": "...", "reason": "..."},
  "sections": [
    {
      "id": "short_section_identifier",
      "heading": "...",
      "bullets": ["...", "..."],
      "evidence_refs": ["artifact/path.json", "..."]
    }
  ],
  "evidence_summary": [
    {"claim": "...", "source": "artifact/path.json"}
  ]
}
""",
        target_language_code=target_language_code,
        executive_decision_label=executive_decision_label,
        facts_block_json=json.dumps(facts_block, ensure_ascii=False),
        reporting_policy_json=json.dumps(reporting_policy_context or {}, ensure_ascii=False),
        evidence_paths=evidence_paths_text,
        execution_results=str(execution_results or "")[:12000],
    )


def _extract_steward_signal_pack(
    steward_summary: Dict[str, Any],
    data_profile: Dict[str, Any],
    dataset_semantics: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(steward_summary, dict):
        steward_summary = {}
    if not isinstance(data_profile, dict):
        data_profile = {}
    if not isinstance(dataset_semantics, dict):
        dataset_semantics = {}

    selectors = dataset_semantics.get("selectors", {}) if isinstance(dataset_semantics.get("selectors"), dict) else {}
    target_analysis = dataset_semantics.get("target_analysis", {}) if isinstance(dataset_semantics.get("target_analysis"), dict) else {}
    profile_stats = data_profile.get("basic_stats", {}) if isinstance(data_profile.get("basic_stats"), dict) else {}
    missingness = data_profile.get("missingness_top30", []) if isinstance(data_profile.get("missingness_top30"), list) else []
    associations = data_profile.get("feature_target_associations", []) if isinstance(data_profile.get("feature_target_associations"), list) else []
    compute_hints = data_profile.get("compute_hints", {}) if isinstance(data_profile.get("compute_hints"), dict) else {}

    top_missing: List[Dict[str, Any]] = []
    for item in missingness[:5]:
        if not isinstance(item, dict):
            continue
        top_missing.append(
            {
                "column": item.get("column"),
                "missing_frac": item.get("missing_frac"),
            }
        )

    top_assoc: List[Dict[str, Any]] = []
    for item in associations[:6]:
        if not isinstance(item, dict):
            continue
        score = item.get("score")
        if score is None:
            score = item.get("association")
        top_assoc.append(
            {
                "feature": item.get("feature") or item.get("column"),
                "target": item.get("target"),
                "method": item.get("method"),
                "score": score,
                "direction": item.get("direction"),
            }
        )

    return {
        "rows": profile_stats.get("n_rows"),
        "cols": profile_stats.get("n_cols"),
        "primary_target": selectors.get("primary_target") or target_analysis.get("primary_target"),
        "training_rows_rule": selectors.get("training_rows_rule"),
        "scoring_rows_rule": selectors.get("scoring_rows_rule_primary") or selectors.get("scoring_rows_rule_secondary"),
        "split_candidates": selectors.get("split_candidates", []),
        "target_null_frac": target_analysis.get("target_null_frac_exact"),
        "top_missingness": top_missing,
        "top_feature_target_associations": top_assoc,
        "compute_hints": {
            "scale_category": compute_hints.get("scale_category"),
            "estimated_memory_mb": compute_hints.get("estimated_memory_mb"),
            "cross_validation_feasible": compute_hints.get("cross_validation_feasible"),
            "deep_learning_feasible": compute_hints.get("deep_learning_feasible"),
        },
        "summary_excerpt": str(steward_summary.get("summary") or "")[:1200],
    }

class BusinessTranslatorAgent:
    def __init__(self, api_key: str = None):
        """
        Initializes the Business Translator Agent with Gemini 3 Flash.
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API Key is required.")

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(
            model_name="gemini-3-flash-preview",
            generation_config={"temperature": 0.2},  # Low temp for evidence-based executive reports
        )
        self.last_prompt = None
        self.last_response = None

    def generate_report(
        self,
        state: Dict[str, Any],
        error_message: Optional[str] = None,
        has_partial_visuals: bool = False,
        plots: Optional[List[str]] = None,
        translator_view: Optional[Dict[str, Any]] = None,
    ) -> str:
        if not isinstance(state, dict):
            state = {"execution_output": str(state), "business_objective": str(error_message or "")}
            error_message = None
        state = sanitize_text_payload(state)

        # Sanitize Visuals Context
        has_partial_visuals = bool(has_partial_visuals)
        plots = [str(p).replace("\\", "/") for p in (plots or []) if p]
        plot_reference_mode = "inline"
        if any(not p.startswith("static/plots/") for p in plots):
            plot_reference_mode = "figure_only"
        artifact_index = _normalize_artifact_index(
            state.get("artifact_index") or _safe_load_json("data/produced_artifact_index.json") or []
        )
        translator_view = translator_view or state.get("translator_view") or {}
        view_policy = translator_view.get("reporting_policy") if isinstance(translator_view, dict) else None
        view_inventory = translator_view.get("evidence_inventory") if isinstance(translator_view, dict) else None
        view_constraints = translator_view.get("constraints") if isinstance(translator_view, dict) else None
        if not isinstance(view_inventory, list) or not view_inventory:
            view_inventory = artifact_index

        def _artifact_available(path: str) -> bool:
            if artifact_index:
                return any(item.get("path") == path for item in artifact_index if isinstance(item, dict))
            return os.path.exists(path)
        
        # Safe extraction of strategy info
        strategy = state.get('selected_strategy', {})
        strategy_title = strategy.get('title', 'General Analysis')
        hypothesis = strategy.get('hypothesis', 'N/A')
        analysis_type = str(strategy.get('analysis_type', 'N/A'))
        
        # Review content
        review_verdict = state.get("review_verdict")
        if review_verdict:
            compliance = review_verdict
        else:
            review = state.get('review_feedback', {})
            if isinstance(review, dict):
                compliance = review.get('status', 'PENDING')
            else:
                # If it's a string (e.g. just the feedback text from older legacy flows or simple strings)
                compliance = "REVIEWED" if review else "PENDING"
        
        # Construct JSON Context for Visuals safely using json library
        visuals_context_data = {
            "has_partial_visuals": has_partial_visuals,
            "plots_count": len(plots),
            "plots_list": plots,
            "plot_reference_mode": plot_reference_mode,
        }
        visuals_context_json = json.dumps(visuals_context_data, ensure_ascii=False)
        run_id = state.get("run_id")
        contract = _safe_load_json("data/execution_contract.json") or {}
        decisioning_context = translator_view.get("decisioning_requirements") or contract.get("decisioning_requirements") or {}
        if not isinstance(decisioning_context, dict):
            decisioning_context = {}
        decisioning_context_json = json.dumps(decisioning_context, ensure_ascii=False)
        decisioning_columns = [
            str(col.get("name"))
            for col in (decisioning_context.get("output", {}).get("required_columns") or [])
            if isinstance(col, dict) and col.get("name")
        ]
        decisioning_columns_text = ", ".join(decisioning_columns) if decisioning_columns else "None requested."
        
        # Load optional artifacts for context
        integrity_audit = _safe_load_json("data/integrity_audit_report.json") or {}
        output_contract_report = _safe_load_json("data/output_contract_report.json") or {}
        case_alignment_report = _safe_load_json("data/case_alignment_report.json") or {}
        data_adequacy_report = _safe_load_json("data/data_adequacy_report.json") or {}
        alignment_check_report = _safe_load_json("data/alignment_check.json") or {}
        plot_insights = _safe_load_json("data/plot_insights.json") or {}
        insights = _safe_load_json("data/insights.json") or {}
        steward_summary = _safe_load_json("data/steward_summary.json") or {}
        data_profile = (
            _safe_load_json("data/data_profile.json")
            or _safe_load_json(os.path.join("work", "artifacts", "data_profile.json"))
            or {}
        )
        dataset_semantics = _safe_load_json("data/dataset_semantics.json") or {}
        cleaning_manifest = _safe_load_json("data/cleaning_manifest.json") or {}
        run_summary = _safe_load_json("data/run_summary.json") or {}
        recommendations_preview = _safe_load_json("reports/recommendations_preview.json") or {}
        metrics_payload = _safe_load_json("data/metrics.json") or {}

        # ── Canonical metric correction ──────────────────────────────
        # After metric loop baseline restoration, insights.json and
        # metrics.json on disk may reflect the LAST attempt (a regression)
        # rather than the selected incumbent.  state["primary_metric_state"]
        # is the authoritative source — updated by _finalize_metric_improvement_round.
        _pms = state.get("primary_metric_state") if isinstance(state.get("primary_metric_state"), dict) else {}
        _canonical_metric_name = str(_pms.get("primary_metric_name") or "").strip()
        _canonical_metric_value = _pms.get("primary_metric_value")
        metric_loop_context: Dict[str, Any] = {}
        if _canonical_metric_name and _canonical_metric_value is not None:
            # Patch metrics_payload so all downstream consumers see the correct value
            if isinstance(metrics_payload, dict) and metrics_payload:
                _disk_value = metrics_payload.get(_canonical_metric_name)
                if _disk_value is not None:
                    try:
                        if abs(float(_disk_value) - float(_canonical_metric_value)) > 1e-9:
                            metrics_payload[_canonical_metric_name] = _canonical_metric_value
                            model_perf = metrics_payload.get("model_performance")
                            if isinstance(model_perf, dict) and _canonical_metric_name in model_perf:
                                model_perf[_canonical_metric_name] = _canonical_metric_value
                    except (TypeError, ValueError):
                        pass
            # Patch insights metrics_summary
            if isinstance(insights, dict) and isinstance(insights.get("metrics_summary"), list):
                for _ms_item in insights["metrics_summary"]:
                    if isinstance(_ms_item, dict) and str(_ms_item.get("metric") or "").strip() == _canonical_metric_name:
                        try:
                            if abs(float(_ms_item.get("value", 0)) - float(_canonical_metric_value)) > 1e-9:
                                _ms_item["value"] = _canonical_metric_value
                        except (TypeError, ValueError):
                            pass
                        break

        # ── Metric loop narrative context ────────────────────────────
        # Load metric_loop_state for round-by-round narrative so the
        # translator can explain what the optimization loop explored.
        _mls = _safe_load_json("data/metric_loop_state.json") or {}
        if isinstance(_mls, dict) and _mls:
            _incumbent = _mls.get("incumbent") if isinstance(_mls.get("incumbent"), dict) else {}
            _best_obs = _mls.get("best_observed") if isinstance(_mls.get("best_observed"), dict) else {}
            _round = _mls.get("round") if isinstance(_mls.get("round"), dict) else {}
            metric_loop_context = {
                "active": bool(_mls.get("active")),
                "incumbent_value": _incumbent.get("metric_value"),
                "incumbent_source": _incumbent.get("source"),
                "best_observed_value": _best_obs.get("metric_value"),
                "best_observed_label": _best_obs.get("label"),
                "round_id": _round.get("round_id"),
                "rounds_allowed": _round.get("rounds_allowed"),
                "no_improve_streak": _round.get("no_improve_streak"),
                "patience": _round.get("patience"),
            }
        round_history = state.get("ml_improvement_round_history")
        if isinstance(round_history, list) and round_history:
            metric_loop_context["round_history"] = [
                {
                    "round_id": r.get("round_id"),
                    "baseline_metric": r.get("baseline_metric"),
                    "candidate_metric": r.get("candidate_metric"),
                    "kept": r.get("kept"),
                    "hypothesis": r.get("hypothesis"),
                }
                for r in round_history
                if isinstance(r, dict)
            ]

        steward_signal_pack = _extract_steward_signal_pack(
            steward_summary=steward_summary,
            data_profile=data_profile,
            dataset_semantics=dataset_semantics,
        )
        weights_path = _first_artifact_path(artifact_index, "weights")
        predictions_path = _first_artifact_path(artifact_index, "predictions")
        weights_payload = _safe_load_json(weights_path) if weights_path else None
        weights_payload = weights_payload or {}
        scored_rows = _safe_load_csv(predictions_path) if predictions_path else None
        case_summary = None
        cleaned_path = _first_artifact_path(artifact_index, "dataset")
        if not cleaned_path:
            cleaned_path = "data/cleaned_dataset.csv" if _artifact_available("data/cleaned_dataset.csv") else "data/cleaned_data.csv"
        cleaned_rows = _safe_load_csv(cleaned_path, max_rows=100) if _artifact_available(cleaned_path) else None
        business_objective = state.get("business_objective") or contract.get("business_objective") or ""
        executive_decision_label = _derive_exec_decision(
            review_verdict or compliance,
            data_adequacy_report,
            metrics_payload,
        )

        required_outputs: List[str] = []
        if isinstance(contract.get("required_outputs"), list):
            required_outputs.extend([_normalize_path(path) for path in contract.get("required_outputs") if _looks_like_path(path)])
        artifact_requirements = contract.get("artifact_requirements") if isinstance(contract.get("artifact_requirements"), dict) else {}
        if isinstance(artifact_requirements, dict):
            required_files = artifact_requirements.get("required_files")
            if isinstance(required_files, list):
                for item in required_files:
                    if isinstance(item, dict):
                        path = item.get("path") or item.get("output") or item.get("artifact")
                    else:
                        path = item
                    if _looks_like_path(path):
                        required_outputs.append(_normalize_path(path))
        # Keep insertion order
        required_outputs = [p for idx, p in enumerate(required_outputs) if p and p not in required_outputs[:idx]]

        def _summarize_integrity():
            issues = integrity_audit.get("issues", []) if isinstance(integrity_audit, dict) else []
            severity_counts = {}
            for i in issues:
                sev = str(i.get("severity", "unknown"))
                severity_counts[sev] = severity_counts.get(sev, 0) + 1
            top = issues[:3]
            return f"Issues by severity: {severity_counts}; Top3: {top}"

        def _summarize_contract():
            """V4.1: Summarize contract using V4.1 keys only, no legacy keys."""
            if not contract:
                return "No execution contract."
            return {
                "strategy_title": contract.get("strategy_title"),
                "business_objective": contract.get("business_objective"),
                "required_outputs": contract.get("required_outputs", []),
                "canonical_columns_count": len(contract.get("canonical_columns", []) or []),
                "derived_columns_count": len(contract.get("derived_columns", []) or []),
                "validation_requirements": contract.get("validation_requirements", {}),
                "qa_gates": contract.get("qa_gates", []),
                "cleaning_gates": contract.get("cleaning_gates", []),
                "reviewer_gates": contract.get("reviewer_gates", []),
                "business_alignment": contract.get("business_alignment", {}),
                "iteration_policy": contract.get("iteration_policy", {}),
                "decisioning_requirements": contract.get("decisioning_requirements", {}),
                "reporting_policy": contract.get("reporting_policy", {}),
            }

        def _summarize_output_contract():
            if not output_contract_report:
                return "No output contract report."
            miss = output_contract_report.get("missing", [])
            present = output_contract_report.get("present", [])
            return f"Outputs present={len(present)} missing={len(miss)}"

        def _summarize_steward():
            if not steward_summary:
                return "No steward summary."
            summary = steward_summary.get("summary", "")
            encoding = steward_summary.get("encoding")
            sep = steward_summary.get("sep")
            decimal = steward_summary.get("decimal")
            return {
                "summary": summary[:1200],
                "encoding": encoding,
                "sep": sep,
                "decimal": decimal,
                "file_size_bytes": steward_summary.get("file_size_bytes"),
            }

        def _summarize_cleaning():
            if not cleaning_manifest:
                return "No cleaning manifest."
            row_counts = cleaning_manifest.get("row_counts", {})
            conversions = cleaning_manifest.get("conversions", {})
            dropped = cleaning_manifest.get("dropped_rows", {})
            conversion_keys = []
            if isinstance(conversions, dict):
                conversion_keys = list(conversions.keys())[:12]
            elif isinstance(conversions, list):
                conversion_keys = [c.get("column") for c in conversions if isinstance(c, dict) and c.get("column")]
                conversion_keys = conversion_keys[:12]
            return {
                "row_counts": row_counts,
                "dropped_rows": dropped,
                "conversion_keys": conversion_keys,
            }

        def _summarize_weights():
            if not weights_payload:
                return "No weights/metrics payload."
            if not isinstance(weights_payload, dict):
                return weights_payload
            summary = {}
            for key in ("metrics", "weights", "propensity_model", "price_model", "optimization", "regression", "classification"):
                if key in weights_payload:
                    summary[key] = weights_payload.get(key)
            if not summary:
                summary["keys"] = list(weights_payload.keys())[:12]
            return summary

        def _summarize_model_metrics():
            metrics_summary = {}
            if isinstance(weights_payload, dict):
                for key in ("metrics", "propensity_model", "price_model", "optimization", "regression", "classification"):
                    if key in weights_payload:
                        metrics_summary[key] = weights_payload.get(key)
            if isinstance(run_summary, dict):
                run_metrics = run_summary.get("metrics")
                if run_metrics:
                    metrics_summary["run_summary_metrics"] = run_metrics
            if isinstance(insights, dict):
                metrics_summary["insights_metrics"] = insights.get("metrics_summary", [])
            if not metrics_summary:
                return "No explicit model metrics found."
            return metrics_summary

        def _summarize_case_summary():
            if not case_summary:
                return "No case summary artifact."
            columns = case_summary.get("columns", [])
            rows = case_summary.get("rows", [])
            decimal = (case_summary.get("dialect_used") or {}).get("decimal") or "."
            numeric_summary = _summarize_numeric_columns(rows, columns, decimal)
            examples = _pick_top_examples(rows, columns, value_keys=columns, label_keys=columns, decimal=decimal)
            return {
                "row_count_sampled": case_summary.get("row_count_sampled", 0),
                "columns": columns,
                "numeric_summary": numeric_summary,
                "examples": examples,
            }

        def _summarize_scored_rows():
            if not scored_rows:
                return "No predictions artifact."
            columns = scored_rows.get("columns", [])
            rows = scored_rows.get("rows", [])
            decimal = (scored_rows.get("dialect_used") or {}).get("decimal") or "."
            numeric_summary = _summarize_numeric_columns(rows, columns, decimal)
            examples = _pick_top_examples(rows, columns, value_keys=columns, label_keys=columns, decimal=decimal)
            return {
                "row_count_sampled": scored_rows.get("row_count_sampled", 0),
                "columns": columns,
                "numeric_summary": numeric_summary,
                "examples": examples,
            }

        def _summarize_run():
            if not run_summary:
                return "No run_summary.json."
            payload = {
                "status": run_summary.get("status"),
                "run_outcome": run_summary.get("run_outcome"),
                "failed_gates": run_summary.get("failed_gates", []),
                "warnings": run_summary.get("warnings", []),
                "metrics": run_summary.get("metrics", {}),
                "metric_ceiling_detected": run_summary.get("metric_ceiling_detected"),
                "ceiling_reason": run_summary.get("ceiling_reason"),
                "baseline_vs_model": run_summary.get("baseline_vs_model", []),
            }
            return payload

        def _summarize_gate_context():
            gate_context = state.get("last_successful_gate_context") or state.get("last_gate_context") or {}
            if not gate_context:
                return "No gate context."
            if isinstance(gate_context, dict):
                return {
                    "source": gate_context.get("source"),
                    "status": gate_context.get("status"),
                    "failed_gates": gate_context.get("failed_gates", []),
                    "required_fixes": gate_context.get("required_fixes", []),
                }
            return str(gate_context)[:1200]

        def _summarize_review_feedback():
            feedback = state.get("review_feedback") or state.get("execution_feedback") or ""
            if isinstance(feedback, dict):
                return feedback
            if isinstance(feedback, str):
                return feedback[:2000]
            return str(feedback)[:2000]

        def _summarize_data_adequacy():
            if not data_adequacy_report:
                return "No data adequacy report."
            return {
                "status": data_adequacy_report.get("status"),
                "reasons": data_adequacy_report.get("reasons", []),
                "recommendations": data_adequacy_report.get("recommendations", []),
                "signals": data_adequacy_report.get("signals", {}),
                "quality_gates_alignment": data_adequacy_report.get("quality_gates_alignment", {}),
                "consecutive_data_limited": data_adequacy_report.get("consecutive_data_limited"),
                "data_limited_threshold": data_adequacy_report.get("data_limited_threshold"),
                "threshold_reached": data_adequacy_report.get("threshold_reached"),
            }

        def _summarize_alignment_check():
            if not alignment_check_report:
                return "No alignment check."
            return {
                "status": alignment_check_report.get("status"),
                "failure_mode": alignment_check_report.get("failure_mode"),
                "summary": alignment_check_report.get("summary"),
                "requirements": alignment_check_report.get("requirements", []),
            }

        def _summarize_case_alignment():
            if not case_alignment_report:
                return "No case alignment report."
            status = case_alignment_report.get("status")
            failures = case_alignment_report.get("failures", [])
            metrics = case_alignment_report.get("metrics", {})
            thresholds = case_alignment_report.get("thresholds", {})
            return f"Status={status}; Failures={failures}; KeyMetrics={metrics}"

        def _case_alignment_business_status():
            if not case_alignment_report:
                return {
                    "label": "NO_DATA",
                    "status": "UNKNOWN",
                    "message": "No se encontrÃ³ reporte de alineaciÃ³n de casos.",
                    "recommendation": "Revisar si el proceso generÃ³ data/case_alignment_report.json.",
                }
            status = case_alignment_report.get("status")
            failures = case_alignment_report.get("failures", [])
            metrics = case_alignment_report.get("metrics", {})
            thresholds = case_alignment_report.get("thresholds", {})
            if status == "SKIPPED":
                return {
                    "label": "PENDIENTE_DEFINICION_GATES",
                    "status": "SKIPPED",
                    "message": "No se definieron gates de alineaciÃ³n de casos en el contrato.",
                    "recommendation": "Definir mÃ©tricas y umbrales en el contrato para evaluar preparaciÃ³n de negocio.",
                }
            if status == "PASS":
                return {
                    "label": "APTO_CONDICIONAL",
                    "status": "PASS",
                    "message": "La alineaciÃ³n con la lÃ³gica de casos cumple los umbrales definidos.",
                    "key_metrics": metrics,
                }
            # FAIL
            details = []
            for failure in failures:
                metric_val = metrics.get(failure)
                thresh_val = thresholds.get(failure.replace("case_means", "min"), thresholds.get(failure))
                if metric_val is not None:
                    details.append(f"{failure}={metric_val} (umbral={thresh_val})")
                else:
                    details.append(f"{failure} (umbral={thresh_val})")
            return {
                "label": "NO_APTO_PARA_PRODUCCION",
                "status": "FAIL",
                "message": "La soluciÃ³n no cumple los criterios de alineaciÃ³n por casos.",
                "details": details,
                "recommendation": "Priorizar reducciÃ³n de violaciones entre casos antes de considerar producciÃ³n.",
            }

        contract_context = _summarize_contract()
        integrity_context = _summarize_integrity()
        output_contract_context = _summarize_output_contract()
        case_alignment_context = _summarize_case_alignment()
        case_alignment_business_status = _case_alignment_business_status()
        alignment_check_context = _summarize_alignment_check()
        gate_context = _summarize_gate_context()
        review_feedback_context = _summarize_review_feedback()
        steward_context = _summarize_steward()
        cleaning_context = _summarize_cleaning()
        weights_context = _summarize_weights()
        case_summary_context = _summarize_case_summary()
        scored_rows_context = _summarize_scored_rows()
        run_summary_context = _summarize_run()
        data_adequacy_context = _summarize_data_adequacy()
        model_metrics_context = _summarize_model_metrics()
        facts_context = _facts_from_insights(insights) or _build_fact_cards(case_summary_context, scored_rows_context, weights_context, data_adequacy_context)
        artifacts_context = view_inventory if view_inventory else []
        raw_evidence_paths = []
        for item in artifacts_context or []:
            if isinstance(item, dict) and item.get("path"):
                raw_evidence_paths.append(str(item.get("path")))
            elif isinstance(item, str):
                raw_evidence_paths.append(str(item))
        raw_evidence_paths = [p for idx, p in enumerate(raw_evidence_paths) if p and p not in raw_evidence_paths[:idx]]
        raw_evidence_paths = raw_evidence_paths[:8]
        reporting_policy_context = view_policy if isinstance(view_policy, dict) else {}
        if not reporting_policy_context:
            reporting_policy_context = contract.get("reporting_policy", {}) if isinstance(contract, dict) else {}
        translator_view_context = translator_view if isinstance(translator_view, dict) else {}
        slot_payloads = {}
        if isinstance(insights, dict):
            slot_payloads = insights.get("slot_payloads") or {}
            if not slot_payloads:
                if insights.get("metrics_summary"):
                    slot_payloads["model_metrics"] = insights.get("metrics_summary")
                if insights.get("predictions_summary"):
                    slot_payloads["predictions_overview"] = insights.get("predictions_summary")
                if insights.get("segment_pricing_summary"):
                    slot_payloads["segment_pricing"] = insights.get("segment_pricing_summary")
                if insights.get("leakage_audit"):
                    slot_payloads["alignment_risks"] = insights.get("leakage_audit")

        slot_defs = reporting_policy_context.get("slots", []) if isinstance(reporting_policy_context, dict) else []
        missing_required_slots = []
        if isinstance(slot_defs, list):
            for slot in slot_defs:
                if not isinstance(slot, dict):
                    continue
                if slot.get("mode") != "required":
                    continue
                slot_id = slot.get("id")
                insights_key = slot.get("insights_key")
                payload = slot_payloads.get(slot_id) if slot_id else None
                if payload is None and insights_key and isinstance(insights, dict):
                    payload = insights.get(insights_key)
                if payload:
                    continue
                missing_required_slots.append(
                    {"id": slot_id, "sources": slot.get("sources", []), "insights_key": insights_key}
                )
        slot_coverage_context = {
            "slot_payloads": slot_payloads,
            "missing_required_slots": missing_required_slots,
        }

        manifest = _build_report_artifact_manifest(
            artifact_index=artifact_index,
            required_outputs=required_outputs,
            output_contract_report=output_contract_report,
            review_verdict=review_verdict or compliance,
            gate_context=gate_context if isinstance(gate_context, dict) else {},
            run_summary=run_summary if isinstance(run_summary, dict) else {},
            run_id=str(run_id) if run_id else None,
        )
        evidence_paths: List[str] = []
        manifest_items = manifest.get("items", []) if isinstance(manifest, dict) else []
        if isinstance(manifest_items, list):
            for item in manifest_items:
                if not isinstance(item, dict):
                    continue
                path = item.get("path")
                if item.get("present") and _looks_like_path(path):
                    evidence_paths.append(str(path))
        if not evidence_paths:
            evidence_paths = [path for path in raw_evidence_paths if _artifact_available(path)]
        if not evidence_paths:
            evidence_paths = list(raw_evidence_paths)
        evidence_paths = [p for idx, p in enumerate(evidence_paths) if p and p not in evidence_paths[:idx]][:8]
        artifact_inventory_table_html = _build_artifact_inventory_table_html(manifest)
        artifact_compliance_table_html = _build_artifact_compliance_table_html(
            manifest,
            output_contract_report,
            review_verdict or compliance,
            gate_context if isinstance(gate_context, dict) else {},
            run_summary if isinstance(run_summary, dict) else {},
        )
        kpi_snapshot_table_html = _build_kpi_snapshot_table_html(
            metrics_payload if isinstance(metrics_payload, dict) else {},
            data_adequacy_report if isinstance(data_adequacy_report, dict) else {},
            decisioning_columns,
            executive_decision_label,
        )
        run_timeline_context = _load_run_timeline_tail(str(run_id) if run_id else None, max_events=12)
        try:
            os.makedirs("data", exist_ok=True)
            with open("data/report_artifact_manifest.json", "w", encoding="utf-8") as f_manifest:
                json.dump(manifest, f_manifest, indent=2, ensure_ascii=False)
            with open("data/report_visual_tables.json", "w", encoding="utf-8") as f_tables:
                json.dump(
                    {
                        "artifact_inventory_table_html": artifact_inventory_table_html,
                        "artifact_compliance_table_html": artifact_compliance_table_html,
                        "kpi_snapshot_table_html": kpi_snapshot_table_html,
                    },
                    f_tables,
                    indent=2,
                    ensure_ascii=False,
                )
        except Exception:
            pass

        cleaned_sample_table_text = "No data available."
        if isinstance(cleaned_rows, dict) and cleaned_rows.get("rows"):
            cleaned_cols = _select_informative_columns(cleaned_rows, max_cols=8, min_cols=5)
            cleaned_rows_list = _rows_from_sample(cleaned_rows, cleaned_cols, max_rows=5)
            cleaned_sample_table_text = render_table_text(cleaned_cols, cleaned_rows_list, max_rows=5)

        scored_sample_table_text = "No data available."
        if isinstance(scored_rows, dict) and scored_rows.get("rows"):
            scored_cols = _select_scored_columns(scored_rows, max_cols=6)
            scored_rows_list = _rows_from_sample(scored_rows, scored_cols, max_rows=5)
            scored_sample_table_text = render_table_text(scored_cols, scored_rows_list, max_rows=5)

        artifact_headers_table_text = "No data available."
        header_rows = []
        if isinstance(cleaned_rows, dict) and isinstance(cleaned_rows.get("columns"), list):
            cols = cleaned_rows.get("columns") or []
            header_rows.append([cleaned_path, str(len(cols)), _compact_header(cols)])
        if isinstance(scored_rows, dict) and isinstance(scored_rows.get("columns"), list):
            cols = scored_rows.get("columns") or []
            header_rows.append(["data/scored_rows.csv", str(len(cols)), _compact_header(cols)])
        if header_rows:
            artifact_headers_table_text = render_table_text(
                ["artifact", "columns", "header_preview"],
                header_rows,
                max_rows=4,
                max_cell_len=80,
            )

        metrics_table_text = _metrics_table(metrics_payload, max_items=10)
        recommendations_table_text = _recommendations_table(recommendations_preview, max_rows=3)
        evidence_paths_text = "\n".join(f"- {path}" for path in evidence_paths) if evidence_paths else "No data available."

        # Prompt assembly (FACTS_BLOCK + NARRATIVE_GUIDE + CONTEXT_APPENDIX)
        error_condition_str = f"CRITICAL ERROR ENCOUNTERED: {error_message}" if error_message else "No critical errors."
        preferred_language = None
        if isinstance(view_constraints, dict):
            preferred_language = view_constraints.get("language") or view_constraints.get("report_language")
        preferred_language = str(preferred_language).strip().lower() if preferred_language else None
        if preferred_language not in {"es", "en"}:
            preferred_language = None
        target_language_code = _detect_primary_language(business_objective, preferred_language=preferred_language)
        target_language_name = "Spanish" if target_language_code == "es" else "English"

        run_outcome_token = _normalize_decision_token((run_summary or {}).get("run_outcome"))
        decision_discrepancy = None
        if run_outcome_token and run_outcome_token != executive_decision_label:
            decision_discrepancy = {
                "derived_decision": executive_decision_label,
                "run_outcome": run_outcome_token,
                "note": "run_summary outcome differs from deterministic derivation",
            }

        facts_block = {
            "executive_decision_label": executive_decision_label,
            "decision_discrepancy": decision_discrepancy,
            "business_objective": business_objective,
            "strategy_title": strategy_title,
            "review_verdict": review_verdict or compliance,
            "steward_signal_pack": steward_signal_pack,
            "data_adequacy": {
                "status": (data_adequacy_report or {}).get("status"),
                "reasons": (data_adequacy_report or {}).get("reasons", [])[:3],
                "recommendations": (data_adequacy_report or {}).get("recommendations", [])[:3],
            },
            "artifacts_summary": manifest.get("summary", {}) if isinstance(manifest, dict) else {},
            "decisioning_columns": decisioning_columns,
            "metrics_preview": _flatten_metrics(metrics_payload)[:14] if isinstance(metrics_payload, dict) else [],
            "canonical_primary_metric": {
                "name": _canonical_metric_name,
                "value": _canonical_metric_value,
                "source": "primary_metric_state (authoritative)",
            } if _canonical_metric_name and _canonical_metric_value is not None else None,
        }

        context_appendix = {
            "reporting_policy": reporting_policy_context,
            "translator_view": translator_view_context,
            "slot_payloads": slot_payloads,
            "slot_coverage": slot_coverage_context,
            "steward_signal_pack": steward_signal_pack,
            "steward_context": steward_context,
            "cleaning_context": cleaning_context,
            "run_summary_context": run_summary_context,
            "artifact_manifest": manifest,
            "data_adequacy_report_json": data_adequacy_report,
            "alignment_check": alignment_check_context,
            "model_metrics_context": model_metrics_context,
            "case_summary_context": case_summary_context,
            "scored_rows_context": scored_rows_context,
            "plot_insights_json": plot_insights,
            "recommendations_preview": recommendations_preview,
            "run_timeline_context": run_timeline_context,
            "metric_loop_context": metric_loop_context if metric_loop_context else None,
        }

        # ── Run narrative (structured summary built by graph.py) ─────
        run_narrative = state.get("run_narrative")
        if isinstance(run_narrative, dict) and run_narrative:
            run_narrative_section = json.dumps(run_narrative, ensure_ascii=False, indent=2)
        else:
            run_narrative_section = "Not available — use FACTS and DETAILED CONTEXT below."

        SYSTEM_PROMPT_TEMPLATE = Template("""
$senior_translation_protocol

$senior_evidence_rule

=== MISSION ===
Write an executive report in $target_language_name ($target_language_code)
for a decision-maker who has NOT seen the raw data. The report must enable
them to understand what happened, whether the results are trustworthy, and
what to do next.

=== REASONING WORKFLOW (MANDATORY) ===
Before writing, reason through these steps internally. Your report should
reflect this analysis, not just list data.

1. ASSESS THE OUTCOME
   - The deterministic system verdict is: $executive_decision_label
   - What was the business objective? Did the system achieve it?
   - If a metric improvement loop ran, what was the best metric achieved
     vs the baseline? How many techniques were tried?
   - If the canonical_primary_metric in FACTS_BLOCK differs from metrics
     on disk, trust the canonical value — it reflects the selected incumbent.

2. IDENTIFY WHAT MATTERS
   - From all the metrics and artifacts below, select the 3-5 most
     decision-relevant findings. Not everything deserves a mention.
   - Prioritize: primary metric performance, data quality issues,
     compliance failures, and risks that affect production readiness.

3. EXPLAIN WHY
   - Connect results to causes. If the metric improved, what technique
     drove it? If it degraded, what went wrong?
   - If there are contradictions between reviewers and metrics, flag them.

4. RECOMMEND ACTIONS
   - Be specific: "retry with X", "investigate Y in artifact Z",
     "deploy with caveat W" — not generic advice.

=== FACTS (do not alter values) ===
$facts_block_json

=== RUN NARRATIVE (primary context — what happened during this run) ===
$run_narrative_section

=== REFERENCE CONTEXT ===
Business Objective: $business_objective
Strategy: $strategy_title
Hypothesis: $hypothesis
Compliance: $compliance
Error condition: $error_condition

=== EVIDENCE SOURCES ===
Available artifacts: $evidence_paths_text
Metrics: $metrics_table_text
KPI Snapshot (HTML): $kpi_snapshot_table_html

=== DETAILED CONTEXT ===
Outline Plan (pass-1): $outline_plan_json
Artifact Inventory (HTML): $artifact_inventory_table_html
Artifact Compliance (HTML): $artifact_compliance_table_html
Cleaned Data Sample: $cleaned_sample_table_text
Scored Rows Sample: $scored_sample_table_text
Artifact Headers: $artifact_headers_table_text
Recommendations: $recommendations_table_text
Visuals: $visuals_context_json
Decisioning: $decisioning_context_json
Decisioning Columns: $decisioning_columns_text
Reporting Policy: $reporting_policy_context
Slot Coverage: $slot_coverage_context

=== APPENDIX (lower priority — use only if needed for depth) ===
$context_appendix_json

=== OUTPUT FORMAT ===
Markdown. No markdown pipe tables — use provided HTML tables where available.
The report must contain at minimum:
  1) Decision and rationale (## Decisión Ejecutiva)
  2) What happened and key findings (## Hallazgos Clave)
  3) Risks and limitations (## Riesgos)
  4) Recommended next actions (## Próximas Acciones)
  5) Evidence trail (## Evidencia Usada)
If the Outline Plan is non-empty, use it as a starting skeleton but adapt
freely to improve clarity.
""")

        execution_results = state.get("execution_output", "No execution results available.")
        USER_MESSAGE_TEMPLATE = """
Analyze the context above. Reason about what happened, what matters most,
and what the decision-maker should do. Then write the executive report.

EXECUTION FINDINGS:
$execution_results

The final section must be "## Evidencia Usada" with entries:
  {claim: "...", source: "artifact_path -> key"}
"""

        prompt_values = {
            "senior_translation_protocol": SENIOR_TRANSLATION_PROTOCOL,
            "senior_evidence_rule": SENIOR_EVIDENCE_RULE,
            "target_language_name": target_language_name,
            "target_language_code": target_language_code,
            "facts_block_json": json.dumps(facts_block, ensure_ascii=False),
            "run_narrative_section": run_narrative_section,
            "business_objective": business_objective,
            "strategy_title": strategy_title,
            "hypothesis": hypothesis,
            "compliance": compliance,
            "executive_decision_label": executive_decision_label,
            "error_condition": error_condition_str,
            "visuals_context_json": visuals_context_json,
            "decisioning_context_json": decisioning_context_json,
            "decisioning_columns_text": decisioning_columns_text,
            "evidence_paths_text": evidence_paths_text,
            "outline_plan_json": "{}",
            "reporting_policy_context": json.dumps(reporting_policy_context, ensure_ascii=False),
            "slot_coverage_context": json.dumps(slot_coverage_context, ensure_ascii=False),
            "artifact_inventory_table_html": artifact_inventory_table_html,
            "artifact_compliance_table_html": artifact_compliance_table_html,
            "kpi_snapshot_table_html": kpi_snapshot_table_html,
            "metrics_table_text": metrics_table_text,
            "cleaned_sample_table_text": cleaned_sample_table_text,
            "scored_sample_table_text": scored_sample_table_text,
            "artifact_headers_table_text": artifact_headers_table_text,
            "recommendations_table_text": recommendations_table_text,
            "context_appendix_json": json.dumps(context_appendix, ensure_ascii=False),
        }

        two_pass_enabled = str(os.getenv("TRANSLATOR_TWO_PASS_ENABLED", "1")).strip().lower() not in {
            "0",
            "off",
            "false",
            "no",
        }
        two_pass_notes: List[str] = []
        outline_payload: Dict[str, Any] = {}
        outline_issues: List[str] = []
        if two_pass_enabled:
            outline_prompt = _build_outline_prompt(
                target_language_code=target_language_code,
                executive_decision_label=executive_decision_label,
                facts_block=facts_block,
                reporting_policy_context=reporting_policy_context if isinstance(reporting_policy_context, dict) else {},
                evidence_paths=evidence_paths,
                execution_results=execution_results,
            )
            try:
                outline_response = self.model.generate_content(outline_prompt)
                outline_text = (getattr(outline_response, "text", "") or "").strip()
                parsed_outline = _extract_first_json_object(outline_text)
                if parsed_outline is None:
                    two_pass_notes.append("outline_parse_failed")
                else:
                    outline_issues = _validate_outline_payload(parsed_outline)
                    if outline_issues:
                        two_pass_notes.append("outline_validation_failed")
                    else:
                        outline_payload = parsed_outline
                        two_pass_notes.append("outline_generated")
            except Exception as outline_exc:
                two_pass_notes.append(f"outline_error:{outline_exc}")

        if outline_payload:
            prompt_values["outline_plan_json"] = json.dumps(outline_payload, ensure_ascii=False)
            context_appendix["outline_plan"] = outline_payload
            prompt_values["context_appendix_json"] = json.dumps(context_appendix, ensure_ascii=False)

        def _compose_prompt(values: Dict[str, str]) -> str:
            system_prompt = SYSTEM_PROMPT_TEMPLATE.substitute(values)
            user_message = render_prompt(USER_MESSAGE_TEMPLATE, execution_results=execution_results)
            return system_prompt + "\n\n" + user_message

        full_prompt = _compose_prompt(prompt_values)
        max_prompt_tokens = int(os.getenv("TRANSLATOR_MAX_PROMPT_TOKENS", "28000"))
        est_tokens = _estimate_prompt_tokens(full_prompt)
        prompt_budget_notes: List[str] = []

        if est_tokens > max_prompt_tokens:
            drop_order = [
                "context_appendix_json",
                "recommendations_table_text",
                "artifact_headers_table_text",
                "cleaned_sample_table_text",
                "scored_sample_table_text",
                "slot_coverage_context",
            ]
            for key in drop_order:
                value = str(prompt_values.get(key) or "")
                if not value or value.startswith("[omitted due to prompt budget"):
                    continue
                prompt_values[key] = f"[omitted due to prompt budget: {key}]"
                prompt_budget_notes.append(key)
                full_prompt = _compose_prompt(prompt_values)
                est_tokens = _estimate_prompt_tokens(full_prompt)
                if est_tokens <= max_prompt_tokens:
                    break

        if prompt_budget_notes:
            full_prompt += "\n\n[Prompt budget adjustments applied]\n- " + "\n- ".join(prompt_budget_notes)

        self.last_prompt = full_prompt

        def _persist_quality_audit(payload: Dict[str, Any]) -> None:
            try:
                os.makedirs("data", exist_ok=True)
                with open("data/translator_quality_check.json", "w", encoding="utf-8") as f_q:
                    json.dump(payload, f_q, ensure_ascii=False, indent=2)
            except Exception:
                pass

        try:
            response = self.model.generate_content(full_prompt)
            content = (getattr(response, "text", "") or "").strip()
            is_echo_response = content.strip() == full_prompt.strip()
            content = _sanitize_report_text(content)
            content = _ensure_evidence_section(content, evidence_paths)
            content = sanitize_text(content)

            validation = {
                "has_critical": False,
                "critical_issues": [],
                "unverified_metrics": [],
                "structure_issues": [],
                "invalid_plots": [],
                "decision_issue": [],
            }

            if not is_echo_response:
                validation = _validate_report(
                    content=content,
                    expected_decision=executive_decision_label,
                    facts_context=facts_context if isinstance(facts_context, list) else [],
                    metrics_payload=metrics_payload if isinstance(metrics_payload, dict) else {},
                    plots=plots,
                    expected_language=target_language_code,
                )
                if validation.get("has_critical"):
                    repair_prompt = _build_repair_prompt(
                        report=content,
                        validation=validation,
                        expected_decision=executive_decision_label,
                        evidence_paths=evidence_paths,
                        target_language_code=target_language_code,
                    )
                    repair_response = self.model.generate_content(repair_prompt)
                    repaired = (getattr(repair_response, "text", "") or "").strip()
                    repaired = _sanitize_report_text(repaired)
                    repaired = _ensure_evidence_section(repaired, evidence_paths)
                    repaired = sanitize_text(repaired)
                    repair_validation = _validate_report(
                        content=repaired,
                        expected_decision=executive_decision_label,
                        facts_context=facts_context if isinstance(facts_context, list) else [],
                        metrics_payload=metrics_payload if isinstance(metrics_payload, dict) else {},
                        plots=plots,
                        expected_language=target_language_code,
                    )
                    if repair_validation.get("has_critical"):
                        content = _generate_deterministic_fallback_report(
                            target_language_code=target_language_code,
                            executive_decision_label=executive_decision_label,
                            business_objective=business_objective,
                            strategy_title=strategy_title,
                            error_message=", ".join(repair_validation.get("critical_issues", [])),
                            facts_context=facts_context if isinstance(facts_context, list) else [],
                            evidence_paths=evidence_paths,
                        )
                        validation = repair_validation
                    else:
                        content = repaired
                        validation = repair_validation

            quality_score = 100 if is_echo_response else _score_report_quality(validation)
            quality_threshold = int(os.getenv("TRANSLATOR_MIN_QUALITY_SCORE", "60"))
            quality_retry_applied = False
            quality_retry_error = None
            quality_fallback_triggered = False

            if not is_echo_response and quality_score < quality_threshold:
                quality_retry_applied = True
                retry_validation = dict(validation)
                critical = retry_validation.get("critical_issues", [])
                if "low_quality_score" not in critical:
                    critical = list(critical) + ["low_quality_score"]
                    retry_validation["critical_issues"] = critical
                retry_validation["has_critical"] = True
                try:
                    quality_repair_prompt = _build_repair_prompt(
                        report=content,
                        validation=retry_validation,
                        expected_decision=executive_decision_label,
                        evidence_paths=evidence_paths,
                        target_language_code=target_language_code,
                    )
                    quality_repair_response = self.model.generate_content(quality_repair_prompt)
                    quality_candidate = (getattr(quality_repair_response, "text", "") or "").strip()
                    quality_candidate = _sanitize_report_text(quality_candidate)
                    quality_candidate = _ensure_evidence_section(quality_candidate, evidence_paths)
                    quality_candidate = sanitize_text(quality_candidate)
                    quality_candidate_validation = _validate_report(
                        content=quality_candidate,
                        expected_decision=executive_decision_label,
                        facts_context=facts_context if isinstance(facts_context, list) else [],
                        metrics_payload=metrics_payload if isinstance(metrics_payload, dict) else {},
                        plots=plots,
                        expected_language=target_language_code,
                    )
                    quality_candidate_score = _score_report_quality(quality_candidate_validation)
                    if quality_candidate_score >= quality_score:
                        content = quality_candidate
                        validation = quality_candidate_validation
                        quality_score = quality_candidate_score
                except Exception as quality_exc:
                    quality_retry_error = str(quality_exc)

            low_score_fallback_enabled = str(os.getenv("TRANSLATOR_LOW_SCORE_FALLBACK", "1")).strip().lower() not in {
                "0",
                "off",
                "false",
                "no",
            }
            if (
                not is_echo_response
                and quality_score < quality_threshold
                and low_score_fallback_enabled
            ):
                quality_fallback_triggered = True
                content = _generate_deterministic_fallback_report(
                    target_language_code=target_language_code,
                    executive_decision_label=executive_decision_label,
                    business_objective=business_objective,
                    strategy_title=strategy_title,
                    error_message=f"report_quality_below_threshold:{quality_score}<{quality_threshold}",
                    facts_context=facts_context if isinstance(facts_context, list) else [],
                    evidence_paths=evidence_paths,
                )
                validation = {
                    "has_critical": True,
                    "critical_issues": ["low_quality_score"],
                    "unverified_metrics": validation.get("unverified_metrics", []),
                    "structure_issues": validation.get("structure_issues", []),
                    "invalid_plots": validation.get("invalid_plots", []),
                    "decision_issue": validation.get("decision_issue", []),
                }
                quality_score = 0

            if not is_echo_response and validation.get("unverified_metrics") and not validation.get("has_critical"):
                warning_lines = "\n".join(f"- {item}" for item in validation.get("unverified_metrics", [])[:6])
                content = (
                    content.rstrip()
                    + "\n\n## Validation Notes\n"
                    + "Some metric claims could not be matched deterministically:\n"
                    + warning_lines
                    + "\n"
                )
            if decision_discrepancy:
                content = (
                    content.rstrip()
                    + "\n\n## Decision Reconciliation Note\n"
                    + f"Derived decision: {decision_discrepancy.get('derived_decision')} | "
                    + f"run_summary outcome: {decision_discrepancy.get('run_outcome')}\n"
                )
            if prompt_budget_notes:
                content = (
                    content.rstrip()
                    + "\n\n## Prompt Budget Note\n"
                    + "Some low-priority context blocks were compacted to stay within model limits.\n"
                )

            _persist_quality_audit(
                {
                    "prompt_estimated_tokens": est_tokens,
                    "max_prompt_tokens": max_prompt_tokens,
                    "prompt_budget_notes": prompt_budget_notes,
                    "two_pass": {
                        "enabled": two_pass_enabled,
                        "notes": two_pass_notes,
                        "outline_issues": outline_issues,
                        "outline_generated": bool(outline_payload),
                    },
                    "quality_threshold": quality_threshold,
                    "quality_retry_applied": quality_retry_applied,
                    "quality_retry_error": quality_retry_error,
                    "quality_fallback_triggered": quality_fallback_triggered,
                    "quality_score": quality_score,
                    "validation": validation,
                    "decision_discrepancy": decision_discrepancy,
                }
            )
            self.last_response = content
            return content
        except Exception as e:
            fallback = _generate_deterministic_fallback_report(
                target_language_code=target_language_code,
                executive_decision_label=executive_decision_label,
                business_objective=business_objective,
                strategy_title=strategy_title,
                error_message=str(e),
                facts_context=facts_context if isinstance(facts_context, list) else [],
                evidence_paths=evidence_paths,
            )
            _persist_quality_audit(
                {
                    "prompt_estimated_tokens": est_tokens,
                    "max_prompt_tokens": max_prompt_tokens,
                    "prompt_budget_notes": prompt_budget_notes,
                    "two_pass": {
                        "enabled": two_pass_enabled,
                        "notes": two_pass_notes,
                        "outline_issues": outline_issues,
                        "outline_generated": bool(outline_payload),
                    },
                    "quality_score": 0,
                    "validation": {"has_critical": True, "critical_issues": ["llm_exception"]},
                    "exception": str(e),
                }
            )
            self.last_response = fallback
            return fallback


