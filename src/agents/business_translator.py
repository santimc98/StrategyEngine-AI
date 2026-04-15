import os
import re
import html
import csv
import glob
import copy
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple
from dotenv import load_dotenv
from openai import OpenAI

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
from src.utils.dataset_semantics import build_target_lineage_summary
from src.utils.openrouter_reasoning import create_chat_completion_with_reasoning


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


def _safe_load_json_candidates(*paths: Optional[str]):
    for path in paths:
        if not path:
            continue
        payload = _safe_load_json(str(path))
        if payload not in (None, {}, []):
            return payload
    return None


def _load_authoritative_metrics_payload() -> Dict[str, Any]:
    for candidate in (
        "artifacts/ml/cv_metrics.json",
        "data/metrics.json",
    ):
        payload = _safe_load_json(candidate)
        if isinstance(payload, dict) and payload:
            return payload
    return {}

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


def _pearson_corr(pairs: List[Tuple[float, float]]) -> Optional[float]:
    if len(pairs) < 3:
        return None
    left = [item[0] for item in pairs]
    right = [item[1] for item in pairs]
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    cov = sum((a - mean_left) * (b - mean_right) for a, b in pairs)
    var_left = sum((a - mean_left) ** 2 for a in left)
    var_right = sum((b - mean_right) ** 2 for b in right)
    if var_left <= 0 or var_right <= 0:
        return None
    return cov / ((var_left ** 0.5) * (var_right ** 0.5))


def _summarize_scored_row_semantics(
    rows: List[Dict[str, Any]],
    columns: List[str],
    decimal: str,
) -> Dict[str, Any]:
    if not rows or not columns:
        return {}
    lower_to_col = {str(col).lower(): col for col in columns}
    score_priority = [
        "churn_probability",
        "probability",
        "prediction_probability",
        "predicted_probability",
        "score",
        "risk_score",
    ]
    score_col = None
    for candidate in score_priority:
        if candidate in lower_to_col:
            score_col = lower_to_col[candidate]
            break
    if score_col is None:
        for col in columns:
            norm = str(col).lower()
            if any(token in norm for token in ["prob", "score", "pred"]) and not any(
                token in norm for token in ["percentile", "rank", "tier", "bucket"]
            ):
                score_col = col
                break
    rank_col = None
    for col in columns:
        norm = str(col).lower()
        if "risk_percentile" == norm or ("risk" in norm and "percentile" in norm):
            rank_col = col
            break
    if rank_col is None:
        for col in columns:
            norm = str(col).lower()
            if any(token in norm for token in ["percentile", "rank"]) and col != score_col:
                rank_col = col
                break
    if not score_col or not rank_col:
        return {}

    pairs: List[Tuple[float, float]] = []
    for row in rows:
        score_val = coerce_number(row.get(score_col), decimal)
        rank_val = coerce_number(row.get(rank_col), decimal)
        if score_val is None or rank_val is None:
            continue
        pairs.append((float(score_val), float(rank_val)))
    corr = _pearson_corr(pairs)
    if corr is None:
        return {
            "score_column": score_col,
            "rank_column": rank_col,
            "direction": "unknown",
            "sample_pairs": len(pairs),
            "guidance": "Do not infer risk ranking direction from the column name alone.",
        }

    if corr <= -0.1:
        direction = "lower_rank_value_means_higher_risk"
        recommended_sort = [{"column": score_col, "order": "descending"}, {"column": rank_col, "order": "ascending"}]
        guidance = (
            f"Use {score_col} descending or {rank_col} ascending for high-risk prioritization; "
            f"do not describe higher {rank_col} as higher risk."
        )
    elif corr >= 0.1:
        direction = "higher_rank_value_means_higher_risk"
        recommended_sort = [{"column": rank_col, "order": "descending"}, {"column": score_col, "order": "descending"}]
        guidance = f"Use {rank_col} descending or {score_col} descending for high-risk prioritization."
    else:
        direction = "rank_column_not_monotonic_with_score"
        recommended_sort = [{"column": score_col, "order": "descending"}]
        guidance = f"Use {score_col} descending; treat {rank_col} direction as ambiguous."

    tier_col = None
    for col in columns:
        if "tier" in str(col).lower() or "priority" in str(col).lower():
            tier_col = col
            break
    tier_score_means: Dict[str, float] = {}
    if tier_col:
        grouped: Dict[str, List[float]] = {}
        for row in rows:
            tier = str(row.get(tier_col) or "").strip()
            score_val = coerce_number(row.get(score_col), decimal)
            if tier and score_val is not None:
                grouped.setdefault(tier, []).append(float(score_val))
        tier_score_means = {
            tier: sum(values) / len(values)
            for tier, values in grouped.items()
            if values
        }

    return {
        "score_column": score_col,
        "rank_column": rank_col,
        "direction": direction,
        "score_rank_correlation": round(float(corr), 6),
        "sample_pairs": len(pairs),
        "recommended_sort": recommended_sort,
        "guidance": guidance,
        "tier_column": tier_col,
        "tier_score_means": tier_score_means,
    }

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


def _extract_declared_artifact_path(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("path", "output_path", "output", "file", "filename", "expected_filename"):
            candidate = value.get(key)
            if _looks_like_path(candidate):
                return _normalize_path(candidate)
        return ""
    if isinstance(value, (list, tuple, set)):
        return ""
    if _looks_like_path(value):
        normalized = _normalize_path(value)
        if normalized.startswith(("{", "[")):
            return ""
        return normalized
    return ""


def _collect_declared_artifact_paths(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    seen: set[str] = set()
    paths: List[str] = []
    for item in values:
        normalized = _extract_declared_artifact_path(item)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        paths.append(normalized)
    return paths


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
    required_outputs: List[Any],
    output_contract_report: Dict[str, Any],
    review_verdict: Optional[str],
    gate_context: Dict[str, Any],
    run_summary: Dict[str, Any],
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    artifact_index = artifact_index if isinstance(artifact_index, list) else []
    output_contract_report = output_contract_report if isinstance(output_contract_report, dict) else {}

    required_set: set[str] = set()
    for path in _collect_declared_artifact_paths(required_outputs):
        required_set.add(path)

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
        matched_paths: List[str] = []
        if not present and glob.has_magic(path):
            try:
                matched_paths = [
                    _normalize_path(match_path)
                    for match_path in glob.glob(path)
                    if _normalize_path(match_path)
                ]
            except Exception:
                matched_paths = []
            present = bool(matched_paths)
        required = path in required_set
        if path in missing_set:
            status = "ok" if present else "missing_required"
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
                "matched_paths": matched_paths[:8],
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
    final_metric_records: Any,
    canonical_metric_name: str,
    canonical_metric_value: Optional[float],
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
    if canonical_metric_name and canonical_metric_value is not None:
        rows.append([f"metric:{canonical_metric_name}", f"{float(canonical_metric_value):.6g}"])
        return render_table_html(
            ["KPI / Signal", "Value"],
            rows,
            max_rows=14,
            table_class="exec-table kpi-snapshot",
        )
    if isinstance(final_metric_records, dict):
        final_metric_records = _build_metric_records(final_metric_records)
    metric_count = 0
    for item in final_metric_records if isinstance(final_metric_records, list) else []:
        if metric_count >= max_metric_rows:
            break
        if not isinstance(item, dict):
            continue
        key = str(item.get("metric") or "").strip()
        value = _coerce_float(item.get("value"))
        if value is None or not key:
            continue
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
    """Render a markdown pipe table that converts cleanly to HTML/PDF.

    Output example:
    | Header 1 | Header 2 | Header 3 |
    |----------|----------|----------|
    | Value 1  | Value 2  | Value 3  |
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

    def _md_row(cells: List[str]) -> str:
        escaped = [c.replace("|", "\\|") for c in cells]
        return "| " + " | ".join(escaped) + " |"

    lines = []
    lines.append(_md_row([_truncate_cell(h, max_cell_len) for h in headers]))
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for row in safe_rows:
        # Pad row to match header length
        padded = row + [""] * max(0, len(headers) - len(row))
        lines.append(_md_row(padded[:len(headers)]))

    if len(rows) > max_rows:
        lines.append(f"*... ({len(rows) - max_rows} more rows)*")

    return "\n".join(lines)

def _flatten_metrics(metrics: Dict[str, Any], prefix: str = "") -> List[tuple[str, Any]]:
    items: List[tuple[str, Any]] = []
    if not isinstance(metrics, dict):
        return items
    for key, value in metrics.items():
        metric_key = f"{prefix}{key}" if prefix else str(key)
        parts = [part for part in metric_key.split(".") if part]
        if any(left == right for left, right in zip(parts, parts[1:])):
            continue
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


def _metrics_table_from_records(records: Any, max_items: int = 10) -> str:
    if isinstance(records, dict):
        records = _build_metric_records(records)
    if not isinstance(records, list):
        return "No data available."
    rows: List[List[str]] = []
    for item in records[:max_items]:
        if not isinstance(item, dict):
            continue
        metric = str(item.get("metric") or "").strip()
        value = _coerce_float(item.get("value"))
        if not metric or value is None:
            continue
        rows.append([metric, f"{float(value):.6g}"])
    if not rows:
        return "No data available."
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


def _metric_key_score(metric_key: str, metric_name: str) -> int:
    key = str(metric_key or "").strip().lower()
    metric = str(metric_name or "").strip().lower()
    if not key or not metric:
        return -1
    if "std_" in key or key.endswith(".std") or ".std_" in key:
        return -1
    if key == f"mean_{metric}" or key.endswith(f".mean_{metric}"):
        return 100
    if key == metric or key.endswith(f".{metric}"):
        return 90
    if key.endswith(metric):
        return 60
    return -1


def _build_metric_records(metrics_payload: Dict[str, Any], max_items: int = 24) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not isinstance(metrics_payload, dict):
        return records
    for key, value in _flatten_metrics(metrics_payload):
        if not _is_number(value):
            continue
        records.append({"metric": key, "value": float(value)})
        if len(records) >= max_items:
            break
    return records


def _build_final_incumbent_metric_records(
    metrics_payload: Dict[str, Any],
    canonical_metric_name: str,
    canonical_metric_value: Optional[float],
    max_items: int = 24,
) -> List[Dict[str, Any]]:
    if canonical_metric_name and canonical_metric_value is not None:
        return [
            {
                "metric": str(canonical_metric_name).strip(),
                "value": float(canonical_metric_value),
                "source": "canonical_primary_metric",
            }
        ]
    return _build_metric_records(metrics_payload, max_items=max_items)


def _extract_primary_metric_value_from_records(records: Any, metric_name: str) -> Optional[float]:
    if isinstance(records, dict):
        records = _build_metric_records(records)
    if not isinstance(records, list):
        return None
    best_score = -1
    best_value: Optional[float] = None
    for item in records:
        if not isinstance(item, dict):
            continue
        metric_key = str(item.get("metric") or "").strip()
        metric_value = _coerce_float(item.get("value"))
        if metric_value is None:
            continue
        score = _metric_key_score(metric_key, metric_name)
        if score > best_score:
            best_score = score
            best_value = metric_value
    return best_value


def _metric_values_differ(left: Optional[float], right: Optional[float], tol: float = 1e-6) -> bool:
    if left is None or right is None:
        return False
    return abs(float(left) - float(right)) > tol


def _extract_primary_metric_value_from_plot_summary(
    summary_item: Dict[str, Any],
    metric_name: str,
) -> Optional[float]:
    metric = str(metric_name or "").strip().lower()
    if not metric or not isinstance(summary_item, dict):
        return None
    facts = summary_item.get("key_facts")
    if not isinstance(facts, list):
        facts = summary_item.get("facts")
    if not isinstance(facts, list):
        facts = [facts] if facts else []
    number_pattern = re.compile(r"[-+]?\d+(?:\.\d+)?")
    for raw_fact in facts:
        fact = str(raw_fact or "")
        fact_lower = fact.lower()
        if metric not in fact_lower:
            continue
        if "std" in fact_lower and "mean" not in fact_lower:
            continue
        numbers = number_pattern.findall(fact)
        if not numbers:
            continue
        try:
            return float(numbers[-1])
        except Exception:
            continue
    return None


def _match_metric_loop_round(metric_loop_context: Dict[str, Any], metric_value: Optional[float]) -> Optional[Dict[str, Any]]:
    if metric_value is None or not isinstance(metric_loop_context, dict):
        return None
    round_history = metric_loop_context.get("round_history")
    if not isinstance(round_history, list):
        return None
    for item in round_history:
        if not isinstance(item, dict):
            continue
        candidate_value = _coerce_float(item.get("candidate_metric"))
        if candidate_value is None:
            continue
        if not _metric_values_differ(candidate_value, metric_value):
            return item
    return None


def _build_metric_progress_summary(
    metric_loop_context: Dict[str, Any],
    canonical_metric_name: str,
    canonical_metric_value: Optional[float],
) -> Optional[Dict[str, Any]]:
    if not isinstance(metric_loop_context, dict):
        return None
    rounds = metric_loop_context.get("round_history")
    if not isinstance(rounds, list) or not rounds:
        return None
    baseline_start = _coerce_float(rounds[0].get("baseline_metric")) if isinstance(rounds[0], dict) else None
    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    rejected_after_metric_improvement: List[Dict[str, Any]] = []
    for item in rounds:
        if not isinstance(item, dict):
            continue
        metric_improved = (
            item.get("metric_improved")
            if isinstance(item.get("metric_improved"), bool)
            else item.get("improved_by_metric")
        )
        governance_approved = (
            item.get("governance_approved")
            if isinstance(item.get("governance_approved"), bool)
            else item.get("approved")
        )
        entry = {
            "round_id": item.get("round_id"),
            "baseline_metric": item.get("baseline_metric"),
            "candidate_metric": item.get("candidate_metric"),
            "kept": item.get("kept"),
            "hypothesis": item.get("hypothesis"),
            "metric_improved": metric_improved,
            "governance_approved": governance_approved,
        }
        if str(item.get("kept") or "").strip().lower() == "improved":
            accepted.append(entry)
        else:
            rejected.append(entry)
            if metric_improved:
                rejected_after_metric_improvement.append(entry)
    summary: Dict[str, Any] = {
        "metric_name": canonical_metric_name or metric_loop_context.get("metric_name"),
        "baseline_start": baseline_start,
        "improvement_history_scope": "historical_progress_only",
        "selected_incumbent_metric": canonical_metric_value,
        "accepted_rounds": accepted,
        "rejected_rounds": rejected,
        "rejected_after_metric_improvement": rejected_after_metric_improvement,
        "rounds_attempted": len(rounds),
    }
    if baseline_start is not None and canonical_metric_value is not None:
        summary["net_change_vs_start"] = float(canonical_metric_value) - float(baseline_start)
    return summary


def _format_board_metric_value(value: Any) -> str:
    coerced = _coerce_float(value)
    if coerced is None:
        return "unknown"
    return f"{float(coerced):.12g}"


def _build_final_incumbent_board_summary(
    review_board_verdict: Dict[str, Any],
    canonical_metric_name: str,
    canonical_metric_value: Optional[float],
) -> str:
    verdict = review_board_verdict if isinstance(review_board_verdict, dict) else {}
    finalization = (
        verdict.get("metric_round_finalization")
        if isinstance(verdict.get("metric_round_finalization"), dict)
        else {}
    )
    metric_name = str(
        finalization.get("metric_name")
        or canonical_metric_name
        or ""
    ).strip()
    if not finalization:
        return str(verdict.get("summary") or "").strip()

    kept = str(finalization.get("kept") or "").strip().lower() or "unknown"
    baseline_metric = _coerce_float(finalization.get("baseline_metric"))
    candidate_metric = _coerce_float(finalization.get("candidate_metric"))
    final_metric = _coerce_float(finalization.get("final_metric"))
    if final_metric is None and canonical_metric_value is not None:
        final_metric = float(canonical_metric_value)
    metric_improved = bool(
        finalization.get("metric_improved")
        if isinstance(finalization.get("metric_improved"), bool)
        else finalization.get("improved_by_metric")
    )
    governance_approved = bool(
        finalization.get("governance_approved")
        if isinstance(finalization.get("governance_approved"), bool)
        else finalization.get("approved")
    )
    candidate_status = str(verdict.get("candidate_assessment_status") or verdict.get("status") or "UNKNOWN").strip()
    metric_label = metric_name or "primary_metric"
    baseline_text = _format_board_metric_value(baseline_metric)
    candidate_text = _format_board_metric_value(candidate_metric)
    final_text = _format_board_metric_value(final_metric)

    if kept == "improved":
        return (
            f"The challenger became the final incumbent for {metric_label}. "
            f"Metric moved from {baseline_text} to {candidate_text}; final incumbent metric={final_text}. "
            f"Candidate governance status: {candidate_status}."
        )
    if kept == "baseline" and metric_improved:
        reason = "governance rejected the challenger" if not governance_approved else "it was not selected as the incumbent"
        return (
            f"The challenger improved {metric_label} from {baseline_text} to {candidate_text}, but {reason}. "
            f"The final incumbent remained the approved baseline at {final_text}. "
            f"Candidate governance status: {candidate_status}."
        )
    if kept == "baseline":
        governance_clause = "passed governance review" if governance_approved else "did not pass governance review"
        return (
            f"The challenger {governance_clause} but did not improve {metric_label} versus the incumbent "
            f"({baseline_text} vs candidate {candidate_text}). The final incumbent remained the approved baseline "
            f"at {final_text}. Candidate governance status: {candidate_status}."
        )
    return (
        f"Metric round completed for {metric_label}. Final incumbent metric={final_text}, "
        f"baseline={baseline_text}, candidate={candidate_text}, kept={kept}. "
        f"Candidate governance status: {candidate_status}."
    )


def _sanitize_review_board_verdict_for_translator(
    review_board_verdict: Dict[str, Any],
    canonical_metric_name: str,
    canonical_metric_value: Optional[float],
) -> Dict[str, Any]:
    if not isinstance(review_board_verdict, dict) or not review_board_verdict:
        return {}
    sanitized = copy.deepcopy(review_board_verdict)
    final_summary = str(
        sanitized.get("final_incumbent_summary")
        or _build_final_incumbent_board_summary(
            sanitized,
            canonical_metric_name,
            canonical_metric_value,
        )
        or ""
    ).strip()
    if final_summary:
        sanitized["final_incumbent_summary"] = final_summary
        sanitized["summary"] = final_summary

    finalization = (
        sanitized.get("metric_round_finalization")
        if isinstance(sanitized.get("metric_round_finalization"), dict)
        else {}
    )
    if finalization:
        deterministic_facts = (
            copy.deepcopy(sanitized.get("deterministic_facts"))
            if isinstance(sanitized.get("deterministic_facts"), dict)
            else {}
        )
        metrics_facts = (
            copy.deepcopy(deterministic_facts.get("metrics"))
            if isinstance(deterministic_facts.get("metrics"), dict)
            else {}
        )
        primary = (
            copy.deepcopy(metrics_facts.get("primary"))
            if isinstance(metrics_facts.get("primary"), dict)
            else {}
        )
        final_metric = _coerce_float(finalization.get("final_metric"))
        if final_metric is None and canonical_metric_value is not None:
            final_metric = float(canonical_metric_value)
        metric_name = str(finalization.get("metric_name") or canonical_metric_name or primary.get("name") or "").strip()
        if metric_name:
            primary["name"] = metric_name
            primary.setdefault("canonical_name", metric_name)
        if final_metric is not None:
            primary["value"] = float(final_metric)
            primary["source"] = "metric_round_finalization.final_incumbent"
        primary["baseline_value"] = finalization.get("baseline_metric")
        primary["candidate_value"] = finalization.get("candidate_metric")
        primary["final_value"] = final_metric
        primary["kept"] = finalization.get("kept")
        metrics_facts["primary"] = primary
        deterministic_facts["metrics"] = metrics_facts
        sanitized["deterministic_facts"] = deterministic_facts

    return sanitized


def _build_business_objective_summary(
    business_objective: Any,
    *,
    strategy_title: str = "",
    hypothesis: str = "",
    max_chars: int = 260,
) -> str:
    text = re.sub(r"\s+", " ", str(business_objective or "")).strip()
    if text:
        parts = re.split(r"(?<=[\.\!\?])\s+", text)
        summary_parts: List[str] = []
        total = 0
        for part in parts:
            part = part.strip()
            if not part:
                continue
            proposed = total + len(part) + (1 if summary_parts else 0)
            if summary_parts and proposed > max_chars:
                break
            summary_parts.append(part)
            total = proposed
            if total >= max_chars:
                break
        summary = " ".join(summary_parts).strip()
        if summary:
            if len(summary) > max_chars:
                return summary[: max_chars - 3].rstrip() + "..."
            return summary
    fallback_parts = [str(strategy_title or "").strip(), str(hypothesis or "").strip()]
    fallback = " - ".join([part for part in fallback_parts if part]).strip()
    if not fallback:
        return "Business objective available in detailed context."
    return fallback[: max_chars - 3].rstrip() + "..." if len(fallback) > max_chars else fallback


def _resolve_authoritative_data_adequacy_report(
    run_summary: Dict[str, Any],
    raw_report: Dict[str, Any],
) -> Dict[str, Any]:
    authoritative = (
        copy.deepcopy(run_summary.get("data_adequacy"))
        if isinstance(run_summary, dict) and isinstance(run_summary.get("data_adequacy"), dict)
        else {}
    )
    raw = copy.deepcopy(raw_report) if isinstance(raw_report, dict) else {}
    if not authoritative:
        return raw
    if not raw:
        return authoritative
    merged = raw
    for key, value in authoritative.items():
        if value is not None and value != "":
            merged[key] = value
    return merged


def _compact_metric_round_history_for_translator(round_history: Any) -> List[Dict[str, Any]]:
    if not isinstance(round_history, list):
        return []
    compacted: List[Dict[str, Any]] = []
    for record in round_history:
        if not isinstance(record, dict):
            continue
        compacted.append(
            {
                "round_id": record.get("round_id"),
                "baseline_metric": record.get("baseline_metric"),
                "candidate_metric": record.get("candidate_metric"),
                "kept": record.get("kept"),
                "hypothesis": record.get("hypothesis"),
                "metric_improved": record.get("metric_improved"),
                "improved_by_metric": record.get("improved_by_metric"),
                "governance_approved": record.get("governance_approved"),
                "approved": record.get("approved"),
                "review_signal_approved": record.get("review_signal_approved"),
            }
        )
    return compacted


def _build_cleaning_progress_summary(
    cleaning_manifest: Dict[str, Any],
    *,
    strategy_title: str = "",
    hypothesis: str = "",
) -> Optional[Dict[str, Any]]:
    if not isinstance(cleaning_manifest, dict) or not cleaning_manifest:
        return None
    row_counts = cleaning_manifest.get("row_counts") if isinstance(cleaning_manifest.get("row_counts"), dict) else {}
    rows_before = (
        row_counts.get("before")
        or row_counts.get("original")
        or row_counts.get("input")
        or row_counts.get("rows_before")
    )
    rows_after = (
        row_counts.get("after")
        or row_counts.get("final")
        or row_counts.get("output")
        or row_counts.get("rows_after")
    )
    conversions_raw = cleaning_manifest.get("conversions")
    conversion_items: List[str] = []
    if isinstance(conversions_raw, list):
        for item in conversions_raw:
            text = str(item or "").strip()
            if text:
                conversion_items.append(text)
    elif isinstance(conversions_raw, dict):
        for key, value in conversions_raw.items():
            label = str(key or "").strip()
            if not label:
                continue
            if value in (None, "", {}):
                conversion_items.append(label)
            else:
                conversion_items.append(f"{label}: {value}")
    gates_status = cleaning_manifest.get("cleaning_gates_status")
    if not isinstance(gates_status, dict):
        gates_status = {}
    passed_gates = [str(key) for key, value in gates_status.items() if str(value or "").strip().upper() == "PASSED"][:8]
    progress_summary: Dict[str, Any] = {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "row_delta": (rows_after - rows_before) if isinstance(rows_before, int) and isinstance(rows_after, int) else None,
        "key_operations": conversion_items[:8],
        "passed_gates": passed_gates,
        "output_dialect": cleaning_manifest.get("output_dialect", {}),
        "contract_conflicts_resolved": cleaning_manifest.get("contract_conflicts_resolved", []),
    }
    if strategy_title:
        progress_summary["strategy_title"] = strategy_title
    if hypothesis:
        progress_summary["strategy_hypothesis"] = hypothesis
    return progress_summary


def _normalize_missingness_entries(missingness_payload: Any, limit: int = 5) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    if isinstance(missingness_payload, list):
        for item in missingness_payload:
            if not isinstance(item, dict):
                continue
            column = str(item.get("column") or item.get("feature") or item.get("name") or "").strip()
            missing_frac = _coerce_float(
                item.get("missing_frac")
                if item.get("missing_frac") is not None
                else item.get("missingness")
            )
            if not column or missing_frac is None:
                continue
            entries.append({"column": column, "missing_frac": float(missing_frac)})
    elif isinstance(missingness_payload, dict):
        for column, value in missingness_payload.items():
            missing_frac = _coerce_float(value)
            column_name = str(column or "").strip()
            if not column_name or missing_frac is None:
                continue
            entries.append({"column": column_name, "missing_frac": float(missing_frac)})
    entries.sort(key=lambda item: item.get("missing_frac") or 0.0, reverse=True)
    return entries[:limit]


def _normalize_high_cardinality_entries(high_cardinality_payload: Any, limit: int = 5) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    if not isinstance(high_cardinality_payload, list):
        return entries
    for item in high_cardinality_payload:
        if not isinstance(item, dict):
            continue
        column = str(item.get("column") or item.get("feature") or item.get("name") or "").strip()
        if not column:
            continue
        unique_ratio = _coerce_float(item.get("unique_ratio"))
        n_unique = item.get("n_unique") if item.get("n_unique") is not None else item.get("unique")
        entries.append(
            {
                "column": column,
                "n_unique": n_unique,
                "unique_ratio": float(unique_ratio) if unique_ratio is not None else None,
            }
        )
    return entries[:limit]


def _count_dtype_family(dtypes: Dict[str, Any], family: str) -> int:
    count = 0
    for raw_dtype in dtypes.values():
        dtype = str(raw_dtype or "").strip().lower()
        if not dtype:
            continue
        if family == "numeric":
            if dtype.startswith(("int", "float", "uint", "bool", "decimal")):
                count += 1
        elif family == "datetime":
            if "datetime" in dtype or dtype == "date":
                count += 1
        elif family == "categorical":
            if dtype in {"object", "string", "category"}:
                count += 1
    return count


def _extract_conversion_items(cleaning_manifest: Dict[str, Any], limit: int = 8) -> List[str]:
    conversions_raw = cleaning_manifest.get("conversions") if isinstance(cleaning_manifest, dict) else None
    items: List[str] = []
    if isinstance(conversions_raw, list):
        for item in conversions_raw:
            text = str(item or "").strip()
            if text:
                items.append(text)
    elif isinstance(conversions_raw, dict):
        for key, value in conversions_raw.items():
            label = str(key or "").strip()
            if not label:
                continue
            if value in (None, "", {}):
                items.append(label)
            else:
                items.append(f"{label}: {value}")
    return items[:limit]


def _build_eda_fact_pack(
    *,
    cleaning_manifest: Dict[str, Any],
    data_profile: Dict[str, Any],
    dataset_semantics: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not isinstance(cleaning_manifest, dict):
        cleaning_manifest = {}
    if not isinstance(data_profile, dict):
        data_profile = {}
    if not isinstance(dataset_semantics, dict):
        dataset_semantics = {}
    if not cleaning_manifest and not data_profile and not dataset_semantics:
        return None

    row_counts = cleaning_manifest.get("row_counts") if isinstance(cleaning_manifest.get("row_counts"), dict) else {}
    rows_before = (
        row_counts.get("before")
        or row_counts.get("original")
        or row_counts.get("input")
        or row_counts.get("rows_before")
    )
    rows_after = (
        row_counts.get("after")
        or row_counts.get("final")
        or row_counts.get("output")
        or row_counts.get("rows_after")
    )
    retained_fraction: Optional[float] = None
    if isinstance(rows_before, (int, float)) and rows_before:
        retained_fraction = float(rows_after) / float(rows_before) if isinstance(rows_after, (int, float)) else None

    missingness_top = _normalize_missingness_entries(
        data_profile.get("missingness_top30")
        if data_profile.get("missingness_top30") not in (None, {})
        else data_profile.get("missingness"),
        limit=5,
    )
    missing_hotspots = [item for item in missingness_top if (item.get("missing_frac") or 0.0) > 0]

    high_cardinality = _normalize_high_cardinality_entries(
        data_profile.get("high_cardinality_columns")
        if isinstance(data_profile.get("high_cardinality_columns"), list)
        else data_profile.get("high_cardinality_columns_sample"),
        limit=5,
    )

    dtypes = data_profile.get("dtypes") if isinstance(data_profile.get("dtypes"), dict) else {}
    constant_columns = data_profile.get("constant_columns") if isinstance(data_profile.get("constant_columns"), list) else []
    leakage_flags_raw = data_profile.get("leakage_flags") if isinstance(data_profile.get("leakage_flags"), list) else []
    leakage_flags: List[str] = []
    for item in leakage_flags_raw[:5]:
        if not isinstance(item, dict):
            continue
        column = str(item.get("column") or "").strip()
        reason = str(item.get("reason") or "").strip()
        severity = str(item.get("severity") or "").strip()
        if not column:
            continue
        details = reason
        if severity:
            details = f"{details} [{severity}]" if details else f"[{severity}]"
        leakage_flags.append(f"{column}: {details}".strip(": "))

    cleaning_gates_status = cleaning_manifest.get("cleaning_gates_status") if isinstance(cleaning_manifest.get("cleaning_gates_status"), dict) else {}
    quality_flags: List[str] = []
    for gate_name, gate_status in cleaning_gates_status.items():
        normalized_status = str(gate_status or "").strip().upper()
        if not normalized_status or normalized_status == "PASSED":
            continue
        quality_flags.append(f"{gate_name}={gate_status}")
    quality_flags.extend(f"leakage_flag:{flag}" for flag in leakage_flags[:3])
    if constant_columns:
        quality_flags.append(f"constant_columns={','.join(str(col) for col in constant_columns[:3])}")

    target_analysis = dataset_semantics.get("target_analysis") if isinstance(dataset_semantics.get("target_analysis"), dict) else {}
    pack: Dict[str, Any] = {
        "row_retention": {
            "rows_before": rows_before,
            "rows_after": rows_after,
            "retained_fraction": retained_fraction,
        },
        "primary_target": dataset_semantics.get("primary_target") or target_analysis.get("primary_target"),
        "split_candidates": dataset_semantics.get("split_candidates", []) if isinstance(dataset_semantics.get("split_candidates"), list) else [],
        "top_missing_columns": missing_hotspots,
        "high_cardinality_columns": high_cardinality,
        "constant_columns": constant_columns[:5],
        "critical_type_conversions": _extract_conversion_items(cleaning_manifest, limit=8),
        "quality_flags": quality_flags[:8],
        "numeric_profile": {
            "numeric_columns": _count_dtype_family(dtypes, "numeric"),
            "datetime_columns": _count_dtype_family(dtypes, "datetime"),
            "categorical_columns": _count_dtype_family(dtypes, "categorical"),
        },
        "semantic_notes": dataset_semantics.get("notes", [])[:4] if isinstance(dataset_semantics.get("notes"), list) else [],
    }
    target_notes = target_analysis.get("notes", []) if isinstance(target_analysis.get("notes"), list) else []
    if target_notes:
        pack["target_notes"] = target_notes[:4]
    return pack


def _eda_plot_hints_from_fact_pack(eda_fact_pack: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    if not isinstance(eda_fact_pack, dict) or not eda_fact_pack:
        return {}

    hints: Dict[str, Dict[str, Any]] = {}
    missing_columns = eda_fact_pack.get("top_missing_columns") if isinstance(eda_fact_pack.get("top_missing_columns"), list) else []
    if missing_columns:
        facts = [f"columns_with_missing={len(missing_columns)}"]
        for item in missing_columns[:3]:
            if not isinstance(item, dict):
                continue
            column = str(item.get("column") or "").strip()
            missing_frac = _coerce_float(item.get("missing_frac"))
            if not column or missing_frac is None:
                continue
            facts.append(f"{column} missing={missing_frac * 100:.1f}%")
        hints["missing_values"] = {
            "title": "Missing values overview",
            "facts": facts[:4],
            "guidance": (
                "Use when explaining where data-quality risk was concentrated and why categorical handling or "
                "missing-value policy mattered for the final pipeline."
            ),
        }

    numeric_profile = eda_fact_pack.get("numeric_profile") if isinstance(eda_fact_pack.get("numeric_profile"), dict) else {}
    high_cardinality = eda_fact_pack.get("high_cardinality_columns") if isinstance(eda_fact_pack.get("high_cardinality_columns"), list) else []
    constant_columns = eda_fact_pack.get("constant_columns") if isinstance(eda_fact_pack.get("constant_columns"), list) else []
    numeric_facts: List[str] = []
    numeric_count = numeric_profile.get("numeric_columns")
    datetime_count = numeric_profile.get("datetime_columns")
    if isinstance(numeric_count, int):
        numeric_facts.append(f"numeric_columns={numeric_count}")
    if isinstance(datetime_count, int):
        numeric_facts.append(f"datetime_columns={datetime_count}")
    if constant_columns:
        numeric_facts.append(f"constant_columns={len(constant_columns)} ({', '.join(str(col) for col in constant_columns[:2])})")
    if high_cardinality:
        top_item = high_cardinality[0]
        if isinstance(top_item, dict):
            column = str(top_item.get("column") or "").strip()
            unique_ratio = _coerce_float(top_item.get("unique_ratio"))
            if column and unique_ratio is not None:
                numeric_facts.append(f"{column} unique_ratio={unique_ratio * 100:.1f}%")
    if numeric_facts:
        hints["numeric_distributions"] = {
            "title": "Numeric distributions overview",
            "facts": numeric_facts[:4],
            "guidance": (
                "Use when discussing scale dispersion, constant columns, or why numeric normalization and feature "
                "screening mattered before modeling."
            ),
        }
    return hints


def _enrich_plot_summaries_with_eda_fact_pack(
    plot_summaries: Optional[List[Dict[str, Any]]],
    plots: Optional[List[str]],
    eda_fact_pack: Optional[Dict[str, Any]],
) -> Optional[List[Dict[str, Any]]]:
    hints = _eda_plot_hints_from_fact_pack(eda_fact_pack if isinstance(eda_fact_pack, dict) else {})
    if not hints:
        return copy.deepcopy(plot_summaries) if isinstance(plot_summaries, list) else plot_summaries

    summaries: List[Dict[str, Any]] = copy.deepcopy(plot_summaries) if isinstance(plot_summaries, list) else []
    summary_by_file: Dict[str, Dict[str, Any]] = {}
    for item in summaries:
        if not isinstance(item, dict):
            continue
        filename = os.path.basename(str(item.get("filename") or item.get("path") or "")).strip()
        if filename:
            summary_by_file[filename.lower()] = item

    for raw_path in plots or []:
        filename = os.path.basename(str(raw_path or "")).strip()
        lower_name = filename.lower()
        if not lower_name:
            continue
        hint: Optional[Dict[str, Any]] = None
        if "missing" in lower_name:
            hint = hints.get("missing_values")
        elif "numeric_distributions" in lower_name or "eda_numeric" in lower_name:
            hint = hints.get("numeric_distributions")
        if not isinstance(hint, dict):
            continue

        target_item = summary_by_file.get(lower_name)
        if not isinstance(target_item, dict):
            target_item = {"filename": filename}
            summaries.append(target_item)
            summary_by_file[lower_name] = target_item

        existing_facts = target_item.get("key_facts")
        if not isinstance(existing_facts, list):
            existing_facts = target_item.get("facts")
        if not isinstance(existing_facts, list):
            existing_facts = [str(existing_facts)] if existing_facts else []
        existing_facts = [str(item).strip() for item in existing_facts if str(item or "").strip()]
        if not existing_facts or all("chart available" in fact.lower() for fact in existing_facts):
            target_item["facts"] = hint.get("facts", [])
        target_item.setdefault("title", hint.get("title"))
        if not str(target_item.get("guidance") or "").strip():
            target_item["guidance"] = hint.get("guidance")

    return summaries


def _summarize_data_engineer_change_summary(
    *,
    cleaning_manifest: Dict[str, Any],
    eda_fact_pack: Optional[Dict[str, Any]],
    cleaning_progress_summary: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not isinstance(cleaning_manifest, dict):
        cleaning_manifest = {}
    if not isinstance(eda_fact_pack, dict):
        eda_fact_pack = {}
    if not isinstance(cleaning_progress_summary, dict):
        cleaning_progress_summary = {}
    if not cleaning_manifest and not eda_fact_pack and not cleaning_progress_summary:
        return None

    initial_quality_signals: List[str] = []
    for item in (eda_fact_pack.get("top_missing_columns") or [])[:3]:
        if not isinstance(item, dict):
            continue
        column = str(item.get("column") or "").strip()
        missing_frac = _coerce_float(item.get("missing_frac"))
        if column and missing_frac is not None and missing_frac > 0:
            initial_quality_signals.append(f"{column} had {missing_frac * 100:.1f}% missingness")
    for item in (eda_fact_pack.get("high_cardinality_columns") or [])[:2]:
        if not isinstance(item, dict):
            continue
        column = str(item.get("column") or "").strip()
        unique_ratio = _coerce_float(item.get("unique_ratio"))
        if column and unique_ratio is not None:
            initial_quality_signals.append(f"{column} had {unique_ratio * 100:.1f}% uniqueness")
    constant_columns = eda_fact_pack.get("constant_columns") if isinstance(eda_fact_pack.get("constant_columns"), list) else []
    if constant_columns:
        initial_quality_signals.append(
            f"Constant columns detected: {', '.join(str(col) for col in constant_columns[:3])}"
        )

    accepted_interventions = cleaning_progress_summary.get("key_operations") if isinstance(cleaning_progress_summary.get("key_operations"), list) else []
    contractual_resolutions = cleaning_manifest.get("contract_conflicts_resolved") if isinstance(cleaning_manifest.get("contract_conflicts_resolved"), list) else []
    passed_gates = cleaning_progress_summary.get("passed_gates") if isinstance(cleaning_progress_summary.get("passed_gates"), list) else []
    quality_flags = eda_fact_pack.get("quality_flags") if isinstance(eda_fact_pack.get("quality_flags"), list) else []
    notes = cleaning_manifest.get("notes") if isinstance(cleaning_manifest.get("notes"), list) else []

    row_retention = eda_fact_pack.get("row_retention") if isinstance(eda_fact_pack.get("row_retention"), dict) else {}
    summary: Dict[str, Any] = {
        "initial_quality_signals": initial_quality_signals[:6],
        "accepted_interventions": [str(item) for item in accepted_interventions[:8]],
        "row_retention": row_retention,
        "gates_cleared": [str(item) for item in passed_gates[:8]],
        "contractual_resolutions": [str(item) for item in contractual_resolutions[:4]],
        "residual_risks": [str(item) for item in (quality_flags[:6] + [str(note) for note in notes[:2]])],
        "output_dialect": cleaning_progress_summary.get("output_dialect", {}),
    }
    return summary


def _metric_round_hypothesis_label(item: Dict[str, Any]) -> str:
    hypothesis = item.get("hypothesis")
    if isinstance(hypothesis, dict):
        return str(
            hypothesis.get("label")
            or hypothesis.get("title")
            or hypothesis.get("hypothesis")
            or ""
        ).strip()
    return str(hypothesis or "").strip()


def _summarize_ml_engineer_change_summary(
    *,
    metric_loop_context: Dict[str, Any],
    canonical_metric_name: str,
    canonical_metric_value: Optional[float],
    review_board_verdict: Optional[Dict[str, Any]] = None,
    final_incumbent_state: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    if not isinstance(metric_loop_context, dict):
        metric_loop_context = {}
    round_history = metric_loop_context.get("round_history")
    if not isinstance(round_history, list) or not round_history:
        incumbent_state = final_incumbent_state if isinstance(final_incumbent_state, dict) else {}
        review_before_board = ""
        if isinstance(review_board_verdict, dict):
            deterministic_pipeline = (
                review_board_verdict.get("deterministic_facts", {}).get("pipeline", {})
                if isinstance(review_board_verdict.get("deterministic_facts"), dict)
                else {}
            )
            review_before_board = str(
                deterministic_pipeline.get("review_verdict_before_board")
                or review_board_verdict.get("review_verdict_before_board")
                or review_board_verdict.get("final_review_verdict")
                or review_board_verdict.get("status")
                or ""
            ).strip()
        predictions_output = (
            incumbent_state.get("predictions_output")
            if isinstance(incumbent_state.get("predictions_output"), dict)
            else {}
        )
        baseline_summary: Dict[str, Any] = {
            "metric_name": canonical_metric_name or incumbent_state.get("primary_metric", {}).get("name"),
            "baseline_metric_start": canonical_metric_value,
            "selected_incumbent_metric": canonical_metric_value,
            "accepted_improvements": [],
            "rejected_experiments": [],
            "rejected_after_metric_improvement": [],
            "incumbent_promotions": [],
            "rounds_attempted": 0,
            "current_incumbent_basis": "approved_baseline",
            "baseline_incumbent": {
                "status": review_before_board or "approved_baseline",
                "metric": canonical_metric_value,
                "output_contract_status": incumbent_state.get("output_contract_status"),
                "runtime_status": incumbent_state.get("runtime_status"),
                "predictions_rows": predictions_output.get("row_count"),
            },
        }
        if isinstance(review_board_verdict, dict) and review_board_verdict:
            board_summary = str(
                review_board_verdict.get("final_incumbent_summary")
                or review_board_verdict.get("summary")
                or ""
            ).strip()
            required_actions = review_board_verdict.get("required_actions")
            if board_summary:
                baseline_summary["review_board_summary"] = board_summary[:600]
            if isinstance(required_actions, list) and required_actions:
                baseline_summary["review_board_required_actions"] = [str(item) for item in required_actions[:4]]
        return baseline_summary

    accepted_improvements: List[Dict[str, Any]] = []
    rejected_experiments: List[Dict[str, Any]] = []
    rejected_after_improvement: List[Dict[str, Any]] = []
    incumbent_promotions: List[Dict[str, Any]] = []
    for item in round_history:
        if not isinstance(item, dict):
            continue
        entry = {
            "round_id": item.get("round_id"),
            "hypothesis_label": _metric_round_hypothesis_label(item),
            "baseline_metric": item.get("baseline_metric"),
            "candidate_metric": item.get("candidate_metric"),
            "metric_improved": item.get("metric_improved"),
            "governance_approved": item.get("governance_approved")
            if isinstance(item.get("governance_approved"), bool)
            else item.get("approved"),
            "kept": item.get("kept"),
        }
        kept = str(item.get("kept") or "").strip().lower()
        if kept == "improved":
            accepted_improvements.append(entry)
            incumbent_promotions.append(
                {
                    "round_id": entry["round_id"],
                    "hypothesis_label": entry["hypothesis_label"],
                    "new_incumbent_metric": entry["candidate_metric"],
                }
            )
        else:
            rejected_experiments.append(entry)
            if entry.get("metric_improved") is True:
                rejected_after_improvement.append(entry)

    baseline_start = None
    if isinstance(round_history[0], dict):
        baseline_start = _coerce_float(round_history[0].get("baseline_metric"))
    summary: Dict[str, Any] = {
        "metric_name": canonical_metric_name or metric_loop_context.get("metric_name"),
        "baseline_metric_start": baseline_start,
        "selected_incumbent_metric": canonical_metric_value,
        "accepted_improvements": accepted_improvements,
        "rejected_experiments": rejected_experiments,
        "rejected_after_metric_improvement": rejected_after_improvement,
        "incumbent_promotions": incumbent_promotions,
        "rounds_attempted": len(round_history),
    }
    if accepted_improvements:
        summary["current_incumbent_basis"] = "last_accepted_improvement"
    else:
        summary["current_incumbent_basis"] = "original_approved_baseline"
    if isinstance(review_board_verdict, dict) and review_board_verdict:
        board_summary = str(
            review_board_verdict.get("final_incumbent_summary")
            or review_board_verdict.get("summary")
            or ""
        ).strip()
        required_actions = review_board_verdict.get("required_actions")
        if board_summary:
            summary["review_board_summary"] = board_summary[:600]
        if isinstance(required_actions, list) and required_actions:
            summary["review_board_required_actions"] = [str(item) for item in required_actions[:4]]
    return summary


def _summarize_run_causal_impact(
    *,
    data_engineer_change_summary: Optional[Dict[str, Any]],
    ml_engineer_change_summary: Optional[Dict[str, Any]],
    executive_decision_label: str,
) -> Optional[Dict[str, Any]]:
    data_summary = data_engineer_change_summary if isinstance(data_engineer_change_summary, dict) else {}
    ml_summary = ml_engineer_change_summary if isinstance(ml_engineer_change_summary, dict) else {}
    if not data_summary and not ml_summary:
        return None

    enablers: List[str] = []
    residual_constraints: List[str] = []
    if isinstance(data_summary.get("row_retention"), dict):
        row_retention = data_summary.get("row_retention") or {}
        before = row_retention.get("rows_before")
        after = row_retention.get("rows_after")
        if before is not None and after is not None:
            enablers.append(f"Data engineering preserved {after} of {before} rows while standardizing the modeling table.")
    accepted_interventions = data_summary.get("accepted_interventions") if isinstance(data_summary.get("accepted_interventions"), list) else []
    if accepted_interventions:
        enablers.append(f"Data engineering applied {len(accepted_interventions)} accepted cleaning interventions that materially improved dataset readiness.")
    accepted_improvements = ml_summary.get("accepted_improvements") if isinstance(ml_summary.get("accepted_improvements"), list) else []
    if accepted_improvements:
        first = accepted_improvements[0]
        enablers.append(
            f"ML engineering promoted {len(accepted_improvements)} incumbent improvement(s), starting with round {first.get('round_id')}."
        )
    baseline_incumbent = ml_summary.get("baseline_incumbent") if isinstance(ml_summary.get("baseline_incumbent"), dict) else {}
    if baseline_incumbent and not accepted_improvements:
        metric_name = str(ml_summary.get("metric_name") or "primary_metric").strip() or "primary_metric"
        metric_value = baseline_incumbent.get("metric")
        if metric_value is not None:
            enablers.append(
                f"ML engineering established an approved baseline incumbent with {metric_name}={metric_value}."
            )
    rejected_after_improvement = ml_summary.get("rejected_after_metric_improvement") if isinstance(ml_summary.get("rejected_after_metric_improvement"), list) else []
    if rejected_after_improvement:
        residual_constraints.append(
            f"{len(rejected_after_improvement)} numerically improved challenger(s) were rejected by governance and did not replace the incumbent."
        )
    for item in (data_summary.get("residual_risks") or [])[:4]:
        residual_constraints.append(str(item))
    if isinstance(ml_summary.get("review_board_required_actions"), list):
        residual_constraints.extend(str(item) for item in ml_summary.get("review_board_required_actions", [])[:3])

    focus_points: List[str] = []
    if accepted_interventions:
        focus_points.append("Explain which cleaning interventions changed dataset readiness or reduced modeling risk.")
    if accepted_improvements:
        focus_points.append("Show which accepted ML hypothesis became the new incumbent and why later challengers did not replace it.")
    if rejected_after_improvement:
        focus_points.append("Make clear that some challengers improved the metric numerically but were rejected for business or governance reasons.")

    return {
        "executive_decision_label": executive_decision_label,
        "engineering_enablers": enablers[:5],
        "residual_constraints": residual_constraints[:6],
        "narrative_focus_points": focus_points[:4],
    }


def _normalize_text_for_search(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _text_matches_any(value: Any, patterns: List[str]) -> bool:
    text = _normalize_text_for_search(value)
    if not text:
        return False
    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return True
    return False


def _coerce_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except Exception:
        return None


def _build_final_incumbent_state(
    *,
    executive_decision_label: str,
    run_outcome_token: str,
    review_board_verdict: Optional[Dict[str, Any]],
    output_contract_report: Optional[Dict[str, Any]],
    slot_payloads: Optional[Dict[str, Any]],
    canonical_metric_name: str,
    canonical_metric_value: Optional[float],
    run_summary: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    verdict = review_board_verdict if isinstance(review_board_verdict, dict) else {}
    deterministic_facts = verdict.get("deterministic_facts") if isinstance(verdict.get("deterministic_facts"), dict) else {}
    runtime = deterministic_facts.get("runtime") if isinstance(deterministic_facts.get("runtime"), dict) else {}
    output_contract = deterministic_facts.get("output_contract") if isinstance(deterministic_facts.get("output_contract"), dict) else {}
    pipeline = deterministic_facts.get("pipeline") if isinstance(deterministic_facts.get("pipeline"), dict) else {}
    outputs = slot_payloads if isinstance(slot_payloads, dict) else {}
    predictions = outputs.get("predictions_overview") if isinstance(outputs.get("predictions_overview"), dict) else {}
    metric_source = (
        deterministic_facts.get("metrics", {}).get("primary", {}).get("source")
        if isinstance(deterministic_facts.get("metrics"), dict)
        else None
    )
    output_status = str(
        output_contract.get("overall_status")
        or (output_contract_report or {}).get("overall_status")
        or ""
    ).strip()
    missing_required = output_contract.get("missing_required_artifacts")
    if not isinstance(missing_required, list):
        missing_required = (output_contract_report or {}).get("missing", []) or []
    return {
        "authoritative_decision": executive_decision_label,
        "run_outcome": run_outcome_token or ((run_summary or {}).get("run_outcome") if isinstance(run_summary, dict) else None),
        "runtime_status": str(runtime.get("status") or "").strip(),
        "output_contract_status": output_status,
        "required_missing_artifacts": [str(item) for item in missing_required if str(item or "").strip()],
        "review_verdict_before_board": str(
            pipeline.get("review_verdict_before_board")
            or verdict.get("review_verdict_before_board")
            or ""
        ).strip(),
        "primary_metric": {
            "name": canonical_metric_name,
            "value": canonical_metric_value,
            "source": metric_source or "primary_metric_state",
        } if canonical_metric_name and canonical_metric_value is not None else {},
        "predictions_output": {
            "row_count": _coerce_int(
                predictions.get("row_count_total")
                or predictions.get("row_count")
                or predictions.get("rows")
            ),
            "columns": predictions.get("columns") if isinstance(predictions.get("columns"), list) else [],
            "scoring_semantics": (
                predictions.get("scoring_semantics")
                if isinstance(predictions.get("scoring_semantics"), dict)
                else {}
            ),
        },
        "data_adequacy_status": str(
            ((run_summary or {}).get("data_adequacy") or {}).get("status")
            if isinstance((run_summary or {}).get("data_adequacy"), dict)
            else ""
        ).strip(),
    }


def _collect_governance_history_sources(
    *,
    review_board_verdict: Optional[Dict[str, Any]],
    ml_review_stack: Optional[Dict[str, Any]],
    data_adequacy_report: Optional[Dict[str, Any]],
) -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []

    def _add(source: str, value: Any) -> None:
        text = str(value or "").strip()
        if text:
            entries.append({"source": source, "text": text})

    verdict = review_board_verdict if isinstance(review_board_verdict, dict) else {}
    _add("review_board_verdict.summary", verdict.get("summary"))
    for idx, item in enumerate(verdict.get("required_actions") or []):
        _add(f"review_board_verdict.required_actions[{idx}]", item)
    for idx, item in enumerate(verdict.get("evidence") or []):
        if isinstance(item, dict):
            _add(f"review_board_verdict.evidence[{idx}]", item.get("claim"))

    stack = ml_review_stack if isinstance(ml_review_stack, dict) else {}
    result_evaluator = stack.get("result_evaluator") if isinstance(stack.get("result_evaluator"), dict) else {}
    final_pre_board = stack.get("final_pre_board") if isinstance(stack.get("final_pre_board"), dict) else {}
    iteration_handoff = stack.get("iteration_handoff") if isinstance(stack.get("iteration_handoff"), dict) else {}
    _add("ml_review_stack.result_evaluator.feedback", result_evaluator.get("feedback"))
    _add("ml_review_stack.final_pre_board.feedback", final_pre_board.get("feedback"))
    _add("ml_review_stack.iteration_handoff.runtime_error_tail", (iteration_handoff.get("feedback") or {}).get("runtime_error_tail"))
    for idx, item in enumerate((((data_adequacy_report or {}) if isinstance(data_adequacy_report, dict) else {}).get("reasons") or [])):
        _add(f"data_adequacy_report.reasons[{idx}]", item)
    return entries


def _find_matching_history_sources(
    history_sources: List[Dict[str, str]],
    patterns: List[str],
) -> List[Dict[str, str]]:
    matches: List[Dict[str, str]] = []
    for item in history_sources:
        text = str(item.get("text") or "").strip()
        if not text or not _text_matches_any(text, patterns):
            continue
        matches.append(
            {
                "source": str(item.get("source") or ""),
                "excerpt": text[:260],
            }
        )
    return matches


def _build_governance_contradiction_packet(
    *,
    review_board_verdict: Optional[Dict[str, Any]],
    ml_review_stack: Optional[Dict[str, Any]],
    data_adequacy_report: Optional[Dict[str, Any]],
    final_incumbent_state: Optional[Dict[str, Any]],
    decision_discrepancy: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    incumbent = final_incumbent_state if isinstance(final_incumbent_state, dict) else {}
    output_status = str(incumbent.get("output_contract_status") or "").strip().lower()
    runtime_status = str(incumbent.get("runtime_status") or "").strip().upper()
    predictions_output = incumbent.get("predictions_output") if isinstance(incumbent.get("predictions_output"), dict) else {}
    scoring_rows = _coerce_int(predictions_output.get("row_count"))
    metric_value = _coerce_float((incumbent.get("primary_metric") or {}).get("value")) if isinstance(incumbent.get("primary_metric"), dict) else None
    required_missing = incumbent.get("required_missing_artifacts") if isinstance(incumbent.get("required_missing_artifacts"), list) else []

    history_sources = _collect_governance_history_sources(
        review_board_verdict=review_board_verdict,
        ml_review_stack=ml_review_stack,
        data_adequacy_report=data_adequacy_report,
    )

    contradictions: List[Dict[str, Any]] = []

    if output_status == "ok" and not required_missing:
        patterns = [
            r"missing operational artifacts",
            r"required outputs?.*missing",
            r"faltan?\s+artefactos?",
            r"artefactos? de salida.*falt",
        ]
        matches = _find_matching_history_sources(history_sources, patterns)
        if matches:
            contradictions.append(
                {
                    "id": "missing_required_outputs_stale",
                    "why_contradicted": "Current deterministic facts show output_contract=ok with no missing required artifacts.",
                    "current_fact": "output_contract.ok_and_complete",
                    "matched_history_sources": matches[:4],
                }
            )

    if output_status == "ok" and (scoring_rows or 0) > 0:
        patterns = [
            r"scoring csv",
            r"churn_risk_scores\.csv",
            r"archivo de scores?",
            r"scores? operativos?.*missing",
            r"failing to produce.*scoring",
        ]
        matches = _find_matching_history_sources(history_sources, patterns)
        if matches:
            contradictions.append(
                {
                    "id": "missing_scoring_output_stale",
                    "why_contradicted": "Current deterministic facts and slot coverage show a populated scoring output for the final incumbent.",
                    "current_fact": "scoring_output.present",
                    "matched_history_sources": matches[:4],
                }
            )

    if runtime_status == "OK" and metric_value is not None:
        patterns = [
            r"pipeline_aborted_before_metrics",
            r"pipeline (was )?aborted",
            r"pipeline fue interrumpido",
            r"interrumpido antes de completar.*m[eé]tric",
        ]
        matches = _find_matching_history_sources(history_sources, patterns)
        if matches:
            contradictions.append(
                {
                    "id": "pipeline_aborted_before_metrics_stale",
                    "why_contradicted": "Current deterministic facts include a valid primary metric and successful runtime status for the final incumbent.",
                    "current_fact": "final_incumbent.metric_available",
                    "matched_history_sources": matches[:4],
                }
            )

    packet = {
        "has_contradictions": bool(contradictions or decision_discrepancy),
        "current_facts": {
            "runtime_status": runtime_status,
            "output_contract_status": output_status,
            "required_missing_count": len(required_missing),
            "scoring_row_count": scoring_rows,
            "primary_metric_value": metric_value,
        },
        "contradictions": contradictions,
        "decision_discrepancy": decision_discrepancy if isinstance(decision_discrepancy, dict) else None,
    }
    return packet


def _build_stale_or_rejected_history(
    *,
    review_board_verdict: Optional[Dict[str, Any]],
    ml_review_stack: Optional[Dict[str, Any]],
    governance_contradiction_packet: Optional[Dict[str, Any]],
    decision_discrepancy: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    verdict = review_board_verdict if isinstance(review_board_verdict, dict) else {}
    stack = ml_review_stack if isinstance(ml_review_stack, dict) else {}
    history: Dict[str, Any] = {
        "decision_discrepancy": decision_discrepancy if isinstance(decision_discrepancy, dict) else None,
        "board_status_history": {
            "final_review_verdict": verdict.get("final_review_verdict") or verdict.get("status"),
            "candidate_assessment_status": verdict.get("candidate_assessment_status"),
            "review_verdict_before_board": verdict.get("review_verdict_before_board"),
        },
        "stale_feedback_excerpts": [],
        "contradiction_ids": [],
    }
    stale_feedback_excerpts: List[Dict[str, str]] = []
    for source, text in [
        ("review_board_verdict.summary", verdict.get("summary")),
        ("ml_review_stack.result_evaluator.feedback", ((stack.get("result_evaluator") or {}).get("feedback") if isinstance(stack.get("result_evaluator"), dict) else None)),
        ("ml_review_stack.final_pre_board.feedback", ((stack.get("final_pre_board") or {}).get("feedback") if isinstance(stack.get("final_pre_board"), dict) else None)),
    ]:
        value = str(text or "").strip()
        if value:
            stale_feedback_excerpts.append({"source": source, "excerpt": value[:320]})
    history["stale_feedback_excerpts"] = stale_feedback_excerpts[:4]
    if isinstance(governance_contradiction_packet, dict):
        history["contradiction_ids"] = [
            str(item.get("id"))
            for item in (governance_contradiction_packet.get("contradictions") or [])
            if isinstance(item, dict) and item.get("id")
        ][:8]
    return history


def _prepare_translator_metric_views(
    *,
    metrics_payload: Dict[str, Any],
    slot_payloads: Dict[str, Any],
    model_metrics_context: Any,
    plot_summaries: Optional[List[Dict[str, Any]]],
    metric_loop_context: Dict[str, Any],
    canonical_metric_name: str,
    canonical_metric_value: Optional[float],
) -> Tuple[Dict[str, Any], Any, Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]]]:
    slot_payloads_out = copy.deepcopy(slot_payloads) if isinstance(slot_payloads, dict) else {}
    model_metrics_context_out = copy.deepcopy(model_metrics_context)
    plot_summaries_out = copy.deepcopy(plot_summaries) if isinstance(plot_summaries, list) else plot_summaries
    progress_summary = _build_metric_progress_summary(
        metric_loop_context,
        canonical_metric_name,
        canonical_metric_value,
    )
    final_metric_records = _build_final_incumbent_metric_records(
        metrics_payload,
        canonical_metric_name,
        canonical_metric_value,
    )

    observed_slot_value = _extract_primary_metric_value_from_records(
        slot_payloads_out.get("model_metrics"),
        canonical_metric_name,
    )
    if final_metric_records:
        if _metric_values_differ(observed_slot_value, canonical_metric_value):
            matched_round = _match_metric_loop_round(metric_loop_context, observed_slot_value)
            rejected_payload = slot_payloads_out.get("model_metrics")
            if rejected_payload:
                slot_payloads_out["rejected_challenger_metrics"] = {
                    "scope": "rejected_challenger",
                    "round_id": matched_round.get("round_id") if isinstance(matched_round, dict) else None,
                    "primary_metric_name": canonical_metric_name,
                    "primary_metric_value": observed_slot_value,
                    "records": rejected_payload,
                }
        slot_payloads_out["model_metrics"] = final_metric_records
        slot_payloads_out["model_metrics_scope"] = {
            "scope": "final_incumbent",
            "primary_metric_name": canonical_metric_name,
            "primary_metric_value": canonical_metric_value,
        }

    if isinstance(model_metrics_context_out, dict):
        if final_metric_records:
            model_metrics_context_out["final_incumbent_metrics"] = final_metric_records
        if "rejected_challenger_metrics" in slot_payloads_out:
            model_metrics_context_out["rejected_challenger_metrics"] = slot_payloads_out["rejected_challenger_metrics"]
        insights_metrics = model_metrics_context_out.get("insights_metrics")
        observed_context_value = _extract_primary_metric_value_from_records(insights_metrics, canonical_metric_name)
        if final_metric_records and _metric_values_differ(observed_context_value, canonical_metric_value):
            model_metrics_context_out["insights_metrics_scope"] = "rejected_challenger"

    if isinstance(plot_summaries_out, list):
        for item in plot_summaries_out:
            if not isinstance(item, dict):
                continue
            observed_plot_value = _extract_primary_metric_value_from_plot_summary(item, canonical_metric_name)
            if not _metric_values_differ(observed_plot_value, canonical_metric_value):
                item.setdefault("narrative_scope", "final_incumbent")
                continue
            matched_round = _match_metric_loop_round(metric_loop_context, observed_plot_value)
            item["narrative_scope"] = "rejected_challenger"
            round_id = matched_round.get("round_id") if isinstance(matched_round, dict) else None
            existing_title = str(item.get("title") or "").strip() or "Challenger diagnostics"
            if round_id is not None:
                item["title"] = f"Rejected challenger (round {round_id}): {existing_title}"
            else:
                item["title"] = f"Rejected challenger: {existing_title}"
            facts = item.get("key_facts")
            if not isinstance(facts, list):
                facts = item.get("facts")
            if not isinstance(facts, list):
                facts = [str(facts)] if facts else []
            prefix = "Use only for improvement history; not final incumbent KPI evidence."
            if round_id is not None:
                prefix = f"Rejected challenger from round {round_id}; use only for improvement history."
            suffix = None
            if canonical_metric_name and canonical_metric_value is not None:
                suffix = f"Final incumbent kept: {canonical_metric_name}={float(canonical_metric_value):.6f}"
            updated_facts = [prefix] + [str(fact) for fact in facts if fact]
            if suffix:
                updated_facts.append(suffix)
            item["facts"] = updated_facts[:6]
            item["guidance"] = (
                "Use only when narrating rejected challenger experiments or why the incumbent was kept. "
                "Do not use this artifact as final KPI/stability evidence."
            )

    return slot_payloads_out, model_metrics_context_out, plot_summaries_out, progress_summary


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


def _report_language_pack(target_language_code: Optional[str]) -> Dict[str, str]:
    language = str(target_language_code or "").strip().lower()
    if language == "es":
        return {
            "executive_decision": "Decisión Ejecutiva",
            "risks": "Riesgos",
            "evidence": "Evidencia usada",
            "uncertainty_placeholder": "No verificable con artifacts actuales",
            "confirmed_artifact": "Artefacto confirmado presente",
        }
    return {
        "executive_decision": "Executive Decision",
        "risks": "Risks",
        "evidence": "Evidence Used",
        "uncertainty_placeholder": "Not verifiable from current artifacts",
        "confirmed_artifact": "Confirmed artifact present",
    }


def _evidence_section_heading(target_language_code: Optional[str]) -> str:
    return f"## {_report_language_pack(target_language_code).get('evidence', 'Evidence Used')}"


def _contains_evidence_heading(text: str) -> bool:
    return bool(re.search(r"(?im)^\s*##\s+(evidencia usada|evidence used)\s*$", str(text or "")))


def _build_evidence_items(
    evidence_paths: List[str],
    min_items: int = 3,
    max_items: int = 6,
    *,
    target_language_code: str = "es",
):
    items = []
    placeholder = _report_language_pack(target_language_code).get(
        "uncertainty_placeholder",
        "No verificable con artifacts actuales",
    )
    confirmed_artifact_label = _report_language_pack(target_language_code).get(
        "confirmed_artifact",
        "Confirmed artifact present",
    )
    for path in evidence_paths or []:
        if not path:
            continue
        if len(items) >= max_items:
            break
        clean_path = _sanitize_evidence_value(path)
        items.append({"claim": f"{confirmed_artifact_label}: {clean_path}", "source": clean_path})
    while len(items) < min_items:
        items.append({"claim": placeholder, "source": "missing"})
    return items


def _select_confirmed_evidence_paths(
    manifest: Dict[str, Any],
    raw_evidence_paths: List[str],
    *,
    artifact_available_fn=None,
    max_items: int = 8,
) -> List[str]:
    confirmed: List[str] = []
    manifest_items = manifest.get("items", []) if isinstance(manifest, dict) else []
    if isinstance(manifest_items, list):
        for item in manifest_items:
            if not isinstance(item, dict):
                continue
            path = item.get("path")
            if item.get("present") and _looks_like_path(path):
                confirmed.append(str(path))

    if not confirmed and callable(artifact_available_fn):
        for path in raw_evidence_paths or []:
            if not _looks_like_path(path):
                continue
            if artifact_available_fn(path):
                confirmed.append(str(path))

    deduped: List[str] = []
    for path in confirmed:
        normalized = _normalize_path(path)
        if not normalized or normalized in deduped:
            continue
        deduped.append(normalized)
        if len(deduped) >= max_items:
            break
    return deduped

def _is_valid_evidence_source(source: str) -> bool:
    if not source:
        return False
    normalized = str(source).strip()
    if not normalized:
        return False
    lower = normalized.lower()
    if lower == "missing":
        return False
    if lower.startswith("script:"):
        return True
    if "/" in normalized or "\\" in normalized:
        return True
    for ext in (".json", ".csv", ".md", ".txt", ".py"):
        if ext in lower:
            return True
    return False


def _is_placeholder_evidence_claim(claim: str) -> bool:
    normalized = str(claim or "").strip().lower()
    return normalized.startswith("no verificable con artifacts actuales") or normalized.startswith("not verifiable from current artifacts")

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
    header = re.search(r"(?im)^\s*##\s+(evidencia usada|evidence used)\s*$", report)
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


def _canonical_evidence_section(
    evidence_paths: List[str],
    llm_items: Optional[List[Dict[str, str]]] = None,
    *,
    target_language_code: str = "es",
) -> str:
    validated_llm_items: List[Dict[str, str]] = []
    placeholder = _report_language_pack(target_language_code).get(
        "uncertainty_placeholder",
        "No verificable con artifacts actuales",
    )
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
        generic = _build_evidence_items(
            evidence_paths,
            min_items=0,
            max_items=6,
            target_language_code=target_language_code,
        )
        for item in generic:
            if len(items) >= 6:
                break
            if any(existing.get("source") == item.get("source") for existing in items):
                continue
            items.append(item)
        while len(items) < 3:
            items.append({"claim": placeholder, "source": "missing"})
    else:
        items = _build_evidence_items(evidence_paths, target_language_code=target_language_code)

    evidence_lines: List[str] = ["evidence:"]
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
    return "\n".join(evidence_lines + ["", "**Artifacts:**", ""] + path_lines)


def _ensure_evidence_section(report: str, evidence_paths: List[str], *, target_language_code: str = "es") -> str:
    if not report:
        return report
    report = sanitize_text(report)
    llm_items = _parse_evidence_items_from_report(report)
    evidence_block = _canonical_evidence_section(
        evidence_paths,
        llm_items=llm_items,
        target_language_code=target_language_code,
    )

    header_match = re.search(r"(?im)^\s*##\s+(evidencia usada|evidence used)\s*$", report)
    if header_match:
        prefix = report[:header_match.start()].rstrip()
    else:
        prefix = report.rstrip()

    rebuilt = f"{prefix}\n\n{_evidence_section_heading(target_language_code)}\n\n{evidence_block}\n"
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


def _split_report_sentences(content: str) -> List[str]:
    text = re.sub(r"\s+", " ", str(content or "")).strip()
    if not text:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def _detect_overconfident_operational_claims(content: str) -> List[str]:
    warnings: List[str] = []
    for sentence in _split_report_sentences(content):
        normalized = " ".join(sentence.split())
        lower = normalized.lower()
        if len(normalized) < 24:
            continue
        has_precise_policy = bool(
            re.search(
                r"(?i)\b\d+\s*(?:day|days|week|weeks|month|months|quarter|quarters)\b",
                normalized,
            )
        ) or any(
            token in lower
            for token in (
                "roadmap",
                "review gate",
                "go-live gate",
                "rollout gate",
                "remediation window",
                "deployment window",
            )
        )
        if not has_precise_policy:
            continue
        has_assertive_action = bool(
            re.search(
                r"(?i)\b(is approved|is required|is defined|is mandated|will be|must be|must remain|remains approved|set a hard review gate|deploy for|deployment is approved)\b",
                normalized,
            )
        )
        if not has_assertive_action:
            continue
        has_hedge = bool(
            re.search(
                r"(?i)\b(recommend|recommended|should|could|may|might|consider|propose|proposed|suggest|suggested|option|optional|if adopted|for a pilot)\b",
                normalized,
            )
        )
        if has_hedge:
            continue
        warnings.append(normalized[:220])
    return warnings


def _validate_report_structure(content: str, expected_language: str) -> List[str]:
    issues: List[str] = []
    if not content:
        return ["empty_report"]
    length = len(content)
    if length < 500:
        issues.append("report_too_short")
    if length > 30000:
        issues.append("report_too_long")
    # Flexible header matching — accept natural variations the LLM may produce
    decision_header = re.search(
        r"(?im)^\s*##\s+.{0,10}(Decisi[oó]n|Decision|Veredicto|Verdict).{0,30}$",
        content,
    )
    evidence_header = re.search(
        r"(?im)^\s*##\s+.{0,10}(Evidencia|Evidence).{0,30}$",
        content,
    )
    risk_header = re.search(
        r"(?im)^\s*##\s+.{0,10}(Riesgo|Risk|Limitaci|Limitation).{0,30}$",
        content,
    )
    decision_signal = decision_header or _extract_report_decision(content[:1200] or content)
    risk_signal = risk_header or re.search(
        r"(?is)\b(riesgo(?:s)?|risk(?:s)?|limitaci(?:o|ó)n(?:es)?|limitation(?:s)?)\b",
        content,
    )
    if not decision_signal:
        issues.append("missing_decision_section")
    if not evidence_header:
        issues.append("missing_evidence_section")
    if not risk_signal:
        issues.append("missing_risks_section")
    content_head = content[:1200]
    if expected_language == "es":
        if re.search(r"(?i)(\bthe\b|\btherefore\b|\bhowever\b|evidence used|executive decision|\brisks?\b)", content_head):
            issues.append("possible_language_mix")
    elif expected_language == "en":
        if re.search(r"(?im)(^\s*##\s+decisi|^\s*##\s+riesgos|^\s*##\s+evidencia usada|\bel\b|\bla\b|\blos\b|\blas\b|hallazgos)", content_head):
            issues.append("possible_language_mix")
    return issues


def _strip_evidence_tail_for_validation(content: str) -> str:
    if not content:
        return ""
    split_match = re.split(r"(?im)^\s*##\s*(evidence|evidencia|artifacts)\b", content, maxsplit=1)
    return split_match[0] if split_match else content


def _detect_contradicted_current_state_claims(
    content: str,
    governance_contradiction_packet: Optional[Dict[str, Any]] = None,
) -> List[str]:
    packet = governance_contradiction_packet if isinstance(governance_contradiction_packet, dict) else {}
    contradictions = packet.get("contradictions") if isinstance(packet.get("contradictions"), list) else []
    if not contradictions:
        return []
    text = _normalize_text_for_search(_strip_evidence_tail_for_validation(content))
    if not text:
        return []
    issues: List[str] = []
    pattern_map = {
        "pipeline_aborted_before_metrics_stale": [
            r"pipeline (was )?aborted",
            r"pipeline fue interrumpido",
            r"interrumpido antes de completar.*m[eé]tric",
        ],
        "missing_scoring_output_stale": [
            r"scoring csv",
            r"archivo de scores?",
            r"scores? operativos?.*(missing|falt|ausent)",
            r"faltan?.*scores?",
            r"failing to produce.*scoring",
        ],
        "missing_required_outputs_stale": [
            r"missing operational artifacts",
            r"required outputs?.*missing",
            r"faltan?\s+artefactos?",
            r"artefactos? de salida.*falt",
            r"impidi[oó] generar algunos artefactos",
        ],
    }
    for item in contradictions:
        contradiction_id = str(item.get("id") or "").strip()
        if not contradiction_id:
            continue
        for pattern in pattern_map.get(contradiction_id, []):
            if re.search(pattern, text, flags=re.IGNORECASE):
                issues.append(f"contradicted_current_state_claim:{contradiction_id}")
                break
    return list(dict.fromkeys(issues))


def _validate_report(
    content: str,
    expected_decision: str,
    facts_context: List[Dict[str, Any]],
    metrics_payload: Dict[str, Any],
    plots: List[str],
    expected_language: str,
    decision_discrepancy: Optional[Dict[str, Any]] = None,
    governance_contradiction_packet: Optional[Dict[str, Any]] = None,
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

    evidence_items = _parse_evidence_items_from_report(content)
    unsupported_evidence_claims: List[str] = []
    for item in evidence_items:
        if not isinstance(item, dict):
            continue
        claim = str(item.get("claim") or "").strip()
        source = str(item.get("source") or "").strip().lower()
        if not claim:
            continue
        if source == "missing" and not _is_placeholder_evidence_claim(claim):
            unsupported_evidence_claims.append(claim)

    allowed_plots = {str(path).replace("\\", "/") for path in (plots or [])}
    invalid_plots: List[str] = []
    for path in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", content):
        normalized = str(path).strip().replace("\\", "/")
        if allowed_plots and normalized not in allowed_plots:
            invalid_plots.append(normalized)
    reasoning_warnings = _detect_overconfident_operational_claims(content)
    contradicted_current_state_claims = _detect_contradicted_current_state_claims(
        content,
        governance_contradiction_packet=governance_contradiction_packet,
    )

    critical_issues: List[str] = []
    for issue in structure_issues:
        if issue in {"missing_decision_section", "missing_evidence_section", "report_too_short"}:
            critical_issues.append(issue)
    critical_issues.extend(decision_issue)
    if len(unverified_metrics) > 2:
        critical_issues.append("unverified_metrics_gt2")
    if unsupported_evidence_claims:
        critical_issues.append("unsupported_evidence_claims")
    if invalid_plots:
        critical_issues.append("invalid_plot_reference")
    if contradicted_current_state_claims:
        critical_issues.append("contradicted_current_state_claims")

    context_warnings: List[str] = []
    if isinstance(decision_discrepancy, dict) and decision_discrepancy:
        context_warnings.append("decision_discrepancy_authoritative_vs_derived")
    if reasoning_warnings:
        context_warnings.append("overconfident_operational_claims")

    return {
        "structure_issues": structure_issues,
        "decision_issue": decision_issue,
        "unverified_metrics": unverified_metrics,
        "unsupported_evidence_claims": unsupported_evidence_claims,
        "invalid_plots": invalid_plots,
        "reasoning_warnings": reasoning_warnings,
        "contradicted_current_state_claims": contradicted_current_state_claims,
        "context_warnings": context_warnings,
        "decision_discrepancy": decision_discrepancy if isinstance(decision_discrepancy, dict) else None,
        "critical_issues": critical_issues,
        "has_critical": bool(critical_issues),
    }


def _score_report_quality(validation: Dict[str, Any]) -> int:
    score = 100
    score -= 8 * len(validation.get("structure_issues", []))
    score -= 12 * len(validation.get("decision_issue", []))
    score -= 5 * min(4, len(validation.get("unverified_metrics", [])))
    score -= 10 * min(3, len(validation.get("unsupported_evidence_claims", [])))
    score -= 6 * len(validation.get("invalid_plots", []))
    score -= 7 * min(3, len(validation.get("reasoning_warnings", [])))
    score -= 12 * min(2, len(validation.get("contradicted_current_state_claims", [])))
    score -= 15 * min(1, len(validation.get("context_warnings", [])))
    return max(0, min(100, score))


def _build_repair_prompt(
    report: str,
    validation: Dict[str, Any],
    expected_decision: str,
    evidence_paths: List[str],
    target_language_code: str,
) -> str:
    issues = (
        validation.get("critical_issues", [])
        + validation.get("structure_issues", [])
        + [f"reasoning_warning: {item}" for item in validation.get("reasoning_warnings", [])[:4]]
    )
    issues_text = "\n".join(f"- {issue}" for issue in issues) or "- unknown_issue"
    evidence_paths_text = "\n".join(f"- {path}" for path in evidence_paths[:8]) or "- missing"
    language_pack = _report_language_pack(target_language_code)
    placeholder = language_pack.get("uncertainty_placeholder", "No verificable con artifacts actuales")
    return render_prompt(
        """
        Repair the executive report below without discarding useful content.
        Target language: $lang.
        Required decision label: $decision.

        Issues to fix:
        $issues

        Hard constraints:
        - Keep the report evidence-based and avoid inventing metrics.
        - The executive decision must exactly match the required decision label.
        - Distinguish supported facts, cautious inference, and recommended actions.
        - Use assertive language only for supported facts.
        - If you mention timelines, thresholds, gates, rollout policies, or remediation windows that are not explicit in the artifacts, frame them as recommendations or scenarios, not as established facts.
        - Do not present a substantive claim with source "missing".
        - Only use source "missing" for explicit uncertainty placeholders such as "$placeholder".
        - Ensure sections exist: $executive_decision_heading, $risks_heading, $evidence_heading.
        - Keep "$evidence_section_heading" with evidence:{claim,source} and artifact bullets.
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
        placeholder=placeholder,
        executive_decision_heading=language_pack.get("executive_decision"),
        risks_heading=language_pack.get("risks"),
        evidence_heading=language_pack.get("evidence"),
        evidence_section_heading=_evidence_section_heading(target_language_code),
    )


def _build_structured_repair_prompt(
    *,
    report: str,
    validation: Dict[str, Any],
    expected_decision: str,
    evidence_paths: List[str],
    target_language_code: str,
    artifact_registry_prompt_json: str,
) -> str:
    issues = (
        validation.get("critical_issues", [])
        + validation.get("structure_issues", [])
        + [f"reasoning_warning: {item}" for item in validation.get("reasoning_warnings", [])[:4]]
    )
    issues_text = "\n".join(f"- {issue}" for issue in issues) or "- unknown_issue"
    evidence_paths_text = "\n".join(f"- {path}" for path in evidence_paths[:8]) or "- missing"
    placeholder = _report_language_pack(target_language_code).get(
        "uncertainty_placeholder",
        "No verificable con artifacts actuales",
    )
    return render_prompt(
        """
        Repair the executive report response below.
        Convert it into a valid structured payload.

        Target language: $lang
        Required decision label: $decision

        Issues to fix:
        $issues

        Hard constraints:
        - Return ONLY valid JSON. No markdown, no commentary, no code fences.
        - Keep the report evidence-based and avoid inventing metrics.
        - The executive decision must exactly match the required decision label.
        - Distinguish supported facts, cautious inference, and recommended actions.
        - Use assertive language only for supported facts.
        - If you mention timelines, thresholds, gates, rollout policies, or remediation windows that are not explicit in the artifacts, frame them as recommendations or scenarios, not as established facts.
        - Use only artifact_key values that exist in the artifact registry below.
        - Do not emit a separate evidence heading block. Use the "evidence" array only.
        - If a claim is uncertain, express that uncertainty in the narrative and use the placeholder "$placeholder" only inside the evidence array when needed.
        - Charts and tables are optional. Use them only if they materially improve clarity or trust.

        Artifact registry:
        $artifact_registry

        Available evidence paths:
        $evidence_paths

        Required schema:
        {
          "title": "...",
          "blocks": [
            {"type": "heading", "level": 1, "text": "..."},
            {"type": "paragraph", "text": "..."},
            {"type": "bullet_list", "items": ["...", "..."]},
            {"type": "numbered_list", "items": ["...", "..."]},
            {
              "type": "artifact",
              "artifact_key": "chart_1",
              "lead_in": "One short contextual sentence before the artifact.",
              "analysis": ["Sentence 1 interpreting the artifact.", "Sentence 2 explaining business impact."]
            }
          ],
          "evidence": [
            {"claim": "...", "source": "artifact_path -> key"}
          ]
        }

        Previous invalid response:
        ---
        $report
        ---
        """,
        lang=target_language_code,
        decision=expected_decision,
        issues=issues_text,
        artifact_registry=artifact_registry_prompt_json or "[]",
        evidence_paths=evidence_paths_text,
        placeholder=placeholder,
        report=report or "(empty)",
    )


def _materialize_structured_report(
    *,
    content: str,
    artifact_registry: Dict[str, Dict[str, Any]],
    evidence_paths: List[str],
    target_language_code: str,
) -> Tuple[Optional[str], Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]], List[str]]:
    structured_payload = _extract_first_json_object(content)
    if structured_payload is None:
        return None, None, None, ["payload_parse_failed"]

    structured_payload_issues = _validate_structured_report_payload(structured_payload, artifact_registry)
    if structured_payload_issues:
        return None, None, structured_payload, structured_payload_issues

    hydrated_blocks = _hydrate_report_blocks(structured_payload, artifact_registry)
    if not hydrated_blocks:
        return None, None, structured_payload, ["hydrated_blocks_empty"]

    llm_evidence_items = (
        structured_payload.get("evidence")
        if isinstance(structured_payload.get("evidence"), list)
        else None
    )
    content_markdown = _render_report_blocks_to_markdown(
        hydrated_blocks,
        title=sanitize_text(str(structured_payload.get("title") or "")).strip(),
        evidence_paths=evidence_paths,
        llm_evidence_items=llm_evidence_items,
        target_language_code=target_language_code,
    )
    evidence_markdown = (
        f"{_evidence_section_heading(target_language_code)}\n\n"
        + _canonical_evidence_section(
            evidence_paths,
            llm_items=llm_evidence_items,
            target_language_code=target_language_code,
        )
    )
    blocks_out = list(hydrated_blocks)
    if not any(
        isinstance(block, dict)
        and block.get("type") == "markdown"
        and _contains_evidence_heading(str(block.get("content") or ""))
        for block in blocks_out
    ):
        blocks_out.append({"type": "markdown", "content": evidence_markdown})
    return content_markdown, blocks_out, structured_payload, []


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first valid JSON object from *text*.

    Uses **string-aware** brace matching (respects quoted strings and
    escape sequences) so braces inside string values are ignored.

    If no complete JSON object is found (e.g. the LLM truncated mid-
    output), attempts a *truncation repair*: finds the longest ``{…``
    prefix, closes any open arrays / objects, and re-parses.
    """
    if not text:
        return None
    raw = str(text).strip()

    # ── Fast path: entire text is valid JSON ──
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    # ── Strip markdown fences (```json … ```) wrapping ──
    cleaned = re.sub(r"```json\s*", "", raw, flags=re.IGNORECASE)
    cleaned = re.sub(r"```", "", cleaned).strip()
    if cleaned != raw:
        try:
            parsed = json.loads(cleaned)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            pass

    # ── String-aware brace scan (handles braces inside quoted values) ──
    n = len(raw)
    for scan_start in range(n):
        if raw[scan_start] != "{":
            continue
        depth = 0
        in_str = False
        escape = False
        for pos in range(scan_start, n):
            ch = raw[pos]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = raw[scan_start:pos + 1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            return parsed
                    except Exception:
                        break  # this opening brace failed; try next
        # If we exhausted the string without closing → try truncation repair
        if depth > 0:
            result = _repair_truncated_json(raw[scan_start:])
            if result is not None:
                return result
            break  # only try repair on the first plausible object
    return None


def _repair_truncated_json(fragment: str) -> Optional[Dict[str, Any]]:
    """Try to close a truncated JSON object and parse it.

    Strategy: walk the fragment tracking open structures (object / array /
    string).  At the point of truncation, close everything that is still
    open.  Then try ``json.loads``.
    """
    if not fragment or fragment[0] != "{":
        return None

    # Strip trailing garbage after the last meaningful JSON token.
    # LLM often appends ``\n```json\n{`` of a second attempt after truncation.
    # We split on the *first* obvious re-start of a new JSON block.
    re_start = re.search(r"```\s*json\s*\n\s*\{", fragment[1:], flags=re.IGNORECASE)
    if re_start:
        fragment = fragment[: re_start.start() + 1]

    # Walk and track open structures
    stack: list = []  # 'o' for object, 'a' for array
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
        elif ch == "]":
            if stack and stack[-1] == "a":
                stack.pop()
        if ch.strip():
            last_good = i

    # Trim to last meaningful character
    truncated = fragment[: last_good + 1]

    # If we were inside a string, close it
    if in_str:
        truncated = truncated.rstrip("\\") + '"'

    # Trim trailing comma (invalid before a closing bracket)
    truncated = truncated.rstrip().rstrip(",")

    # Close remaining open structures
    for item in reversed(stack):
        truncated += "]" if item == "a" else "}"

    try:
        parsed = json.loads(truncated)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return None


def _rescue_markdown_from_raw_json(raw_text: str) -> Optional[str]:
    """Last-resort rescue: extract readable markdown from raw/truncated JSON report.

    When the LLM returns a structured JSON report but it is truncated or
    malformed and cannot be fully parsed, this function extracts the text
    content from the ``blocks`` array as best-effort markdown so the user
    sees human-readable text instead of raw JSON.
    """
    if not raw_text:
        return None

    # Try to parse via _extract_first_json_object first (it does truncation repair)
    payload = _extract_first_json_object(raw_text)
    if payload and isinstance(payload.get("blocks"), list) and payload["blocks"]:
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
                items = block.get("items") or []
                for item in items:
                    lines.append(f"- {item}")
                lines.append("")
            elif btype == "numbered_list":
                items = block.get("items") or []
                for idx, item in enumerate(items, 1):
                    lines.append(f"{idx}. {item}")
                lines.append("")
            elif btype == "artifact":
                lead_in = block.get("lead_in") or ""
                if lead_in:
                    lines.append(f"\n{lead_in}\n")
                analysis = block.get("analysis") or []
                for item in analysis:
                    lines.append(f"- {item}")
                lines.append("")
            elif btype == "markdown":
                lines.append(f"\n{block.get('content', '')}\n")
        if lines:
            return "\n".join(lines)
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


def _build_embeddable_artifacts_catalog(
    *,
    plots: List[str],
    plot_summaries: Optional[List[Dict[str, Any]]],
    cleaned_sample_table_text: str,
    scored_sample_table_text: str,
    kpi_snapshot_table_html: str,
    artifact_inventory_table_html: str,
    artifact_compliance_table_html: str,
) -> str:
    """Build a unified catalog of all artifacts the translator can embed inline.

    Each entry tells the LLM: what the artifact is, what it shows, and exactly
    how to embed it.  The translator then reasons about WHERE in the narrative
    each artifact best supports a claim.
    """
    entries: List[str] = []

    # ── Plot summaries keyed by filename for O(1) lookup ──
    summary_by_file: Dict[str, Dict[str, Any]] = {}
    if isinstance(plot_summaries, list):
        for item in plot_summaries:
            if isinstance(item, dict):
                fname = os.path.basename(str(item.get("filename") or item.get("path") or ""))
                if fname:
                    summary_by_file[fname] = item

    # ── Charts ──
    if plots:
        for p in plots:
            fname = os.path.basename(p)
            name_stem = fname.rsplit(".", 1)[0].replace("_", " ").title()
            summary = summary_by_file.get(fname, {})
            title = summary.get("title") or name_stem
            facts = summary.get("key_facts") or summary.get("facts") or ""
            if isinstance(facts, list):
                facts = "; ".join(str(f) for f in facts[:4])
            desc = f"  description: {title}"
            if facts:
                desc += f"\n  key_facts: {facts}"
            else:
                desc += "\n  key_facts: Use this chart only if it supports a concrete cleaning, stability, or business-priority claim."
            entries.append(
                f"- type: chart\n"
                f"  embed: ![{title}]({p})\n"
                f"{desc}\n"
                f"  guidance: Place near the finding this chart illustrates. "
                f"Follow with 2-3 sentences interpreting what it reveals."
            )

    # ── CSV data previews ──
    if cleaned_sample_table_text and cleaned_sample_table_text != "No data available.":
        entries.append(
            f"- type: data_preview\n"
            f"  name: Cleaned dataset sample (first 5 rows)\n"
            f"  embed: paste the table below directly into the report markdown\n"
            f"  content:\n{cleaned_sample_table_text}\n"
            f"  guidance: Use when discussing data quality, cleaning results, "
            f"or explaining the structure of the input data."
        )

    if scored_sample_table_text and scored_sample_table_text != "No data available.":
        entries.append(
            f"- type: data_preview\n"
            f"  name: Model predictions sample (first 5 rows)\n"
            f"  embed: paste the table below directly into the report markdown\n"
            f"  content:\n{scored_sample_table_text}\n"
            f"  guidance: Use when discussing prediction quality, model outputs, "
            f"or demonstrating what the submission looks like."
        )

    # ── Pre-rendered HTML tables ──
    if kpi_snapshot_table_html and len(kpi_snapshot_table_html) > 30:
        entries.append(
            f"- type: html_table\n"
            f"  name: KPI Snapshot\n"
            f"  embed: paste the HTML directly (it renders in PDF)\n"
            f"  guidance: Use in the executive decision or key findings section "
            f"to give a quick metric overview."
        )

    if artifact_inventory_table_html and len(artifact_inventory_table_html) > 30:
        entries.append(
            f"- type: html_table\n"
            f"  name: Artifact Inventory\n"
            f"  embed: paste the HTML directly\n"
            f"  guidance: Use in the evidence trail or as an appendix to show "
            f"what was produced and its status."
        )

    if artifact_compliance_table_html and len(artifact_compliance_table_html) > 30:
        entries.append(
            f"- type: html_table\n"
            f"  name: Output Compliance\n"
            f"  embed: paste the HTML directly\n"
            f"  guidance: Use when discussing whether the pipeline met its "
            f"contractual obligations."
        )

    if not entries:
        return "No embeddable artifacts available for this run."

    return "\n\n".join(entries)


def _strip_html_tags(text: str) -> str:
    cleaned = re.sub(r"<[^>]+>", " ", str(text or ""))
    return re.sub(r"\s+", " ", cleaned).strip()


def _summarize_prompt_text(text: str, max_chars: int = 420) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max(0, max_chars - 6)].rstrip() + " [cut]"


def _build_embeddable_artifact_registry(
    *,
    plots: List[str],
    plot_summaries: Optional[List[Dict[str, Any]]],
    cleaned_sample_table_text: str,
    scored_sample_table_text: str,
    kpi_snapshot_table_html: str,
    artifact_inventory_table_html: str,
    artifact_compliance_table_html: str,
) -> Dict[str, Dict[str, Any]]:
    registry: Dict[str, Dict[str, Any]] = {}
    summary_by_file: Dict[str, Dict[str, Any]] = {}
    if isinstance(plot_summaries, list):
        for item in plot_summaries:
            if not isinstance(item, dict):
                continue
            fname = os.path.basename(str(item.get("filename") or item.get("path") or ""))
            if fname:
                summary_by_file[fname] = item

    for idx, plot_path in enumerate(plots or [], start=1):
        normalized_path = str(plot_path or "").replace("\\", "/").strip()
        if not normalized_path:
            continue
        fname = os.path.basename(normalized_path)
        name_stem = fname.rsplit(".", 1)[0].replace("_", " ").title()
        summary = summary_by_file.get(fname, {})
        title = str(summary.get("title") or name_stem or f"Chart {idx}").strip()
        facts = summary.get("key_facts") or summary.get("facts") or ""
        if isinstance(facts, list):
            facts = "; ".join(str(f) for f in facts[:4])
        summary_text = str(facts).strip() if str(facts or "").strip() else f"{title}. Chart artifact available for inline interpretation near the finding it supports."
        guidance = str(
            summary.get("guidance")
            or "Use inline near the specific finding it supports, then interpret the business implication."
        ).strip()
        registry[f"chart_{idx}"] = {
            "artifact_key": f"chart_{idx}",
            "artifact_type": "chart",
            "title": title,
            "path": normalized_path,
            "summary": _summarize_prompt_text(summary_text),
            "guidance": guidance,
        }

    if cleaned_sample_table_text and cleaned_sample_table_text != "No data available.":
        registry["cleaned_sample_preview"] = {
            "artifact_key": "cleaned_sample_preview",
            "artifact_type": "data_preview",
            "title": "Cleaned dataset sample",
            "content_markdown": cleaned_sample_table_text,
            "summary": "Preview of the cleaned dataset structure and representative rows.",
            "guidance": "Use when explaining data structure, cleaning outcome, or whether the modeling table is usable.",
        }

    if scored_sample_table_text and scored_sample_table_text != "No data available.":
        registry["scored_sample_preview"] = {
            "artifact_key": "scored_sample_preview",
            "artifact_type": "data_preview",
            "title": "Predictions sample",
            "content_markdown": scored_sample_table_text,
            "summary": "Preview of scored rows and model output structure.",
            "guidance": "Use when explaining prediction outputs or what downstream consumers would receive.",
        }

    if kpi_snapshot_table_html and len(kpi_snapshot_table_html) > 30:
        registry["kpi_snapshot"] = {
            "artifact_key": "kpi_snapshot",
            "artifact_type": "html_table",
            "title": "KPI Snapshot",
            "content_html": kpi_snapshot_table_html,
            "summary": "Table with executive decision, data adequacy status, decisioning columns, and top numeric metrics.",
            "guidance": "Use when a KPI table materially strengthens the executive decision or clarifies the final state.",
        }

    if artifact_inventory_table_html and len(artifact_inventory_table_html) > 30:
        registry["artifact_inventory"] = {
            "artifact_key": "artifact_inventory",
            "artifact_type": "html_table",
            "title": "Artifact Inventory",
            "content_html": artifact_inventory_table_html,
            "summary": "Inventory of generated artifacts with presence, row counts, column counts, and sizes.",
            "guidance": "Use when artifact traceability matters for the argument or evidence trail.",
        }

    if artifact_compliance_table_html and len(artifact_compliance_table_html) > 30:
        registry["artifact_compliance"] = {
            "artifact_key": "artifact_compliance",
            "artifact_type": "html_table",
            "title": "Output Compliance",
            "content_html": artifact_compliance_table_html,
            "summary": "Table summarizing output-contract status, review verdict, required artifacts, and failed gates.",
            "guidance": "Use when discussing delivery readiness, compliance, or governance blockers.",
        }

    return registry


def _artifact_registry_prompt_json(registry: Dict[str, Dict[str, Any]]) -> str:
    prompt_items: List[Dict[str, Any]] = []
    for artifact_key, item in registry.items():
        if not isinstance(item, dict):
            continue
        prompt_item: Dict[str, Any] = {
            "artifact_key": artifact_key,
            "artifact_type": item.get("artifact_type"),
            "title": item.get("title"),
            "summary": item.get("summary"),
            "guidance": item.get("guidance"),
        }
        if item.get("artifact_type") == "chart":
            prompt_item["path"] = item.get("path")
        elif item.get("artifact_type") == "data_preview":
            prompt_item["preview"] = _summarize_prompt_text(item.get("content_markdown", ""), max_chars=900)
        elif item.get("artifact_type") == "html_table":
            prompt_item["preview"] = _summarize_prompt_text(_strip_html_tags(item.get("content_html", "")), max_chars=300)
        prompt_items.append(prompt_item)
    return json.dumps(prompt_items, ensure_ascii=False, indent=2)


def _coerce_block_items(value: Any, max_items: int = 8) -> List[str]:
    if not isinstance(value, list):
        return []
    items: List[str] = []
    for item in value:
        text = sanitize_text(str(item or "")).strip()
        if not text:
            continue
        items.append(text)
        if len(items) >= max_items:
            break
    return items


def _validate_structured_report_payload(
    payload: Optional[Dict[str, Any]],
    registry: Dict[str, Dict[str, Any]],
) -> List[str]:
    issues: List[str] = []
    if not isinstance(payload, dict):
        return ["payload_not_dict"]
    blocks = payload.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        return ["missing_blocks"]
    supported = {"heading", "paragraph", "bullet_list", "numbered_list", "artifact"}
    if len(blocks) < 4:
        issues.append("blocks_insufficient")
    for idx, block in enumerate(blocks[:24]):
        if not isinstance(block, dict):
            issues.append(f"block_{idx}_not_dict")
            continue
        block_type = str(block.get("type") or "").strip().lower()
        if block_type not in supported:
            issues.append(f"block_{idx}_unsupported_type")
            continue
        if block_type == "heading":
            if not str(block.get("text") or "").strip():
                issues.append(f"block_{idx}_missing_heading_text")
        elif block_type == "paragraph":
            if not str(block.get("text") or "").strip():
                issues.append(f"block_{idx}_missing_paragraph_text")
        elif block_type in {"bullet_list", "numbered_list"}:
            if not _coerce_block_items(block.get("items")):
                issues.append(f"block_{idx}_missing_items")
        elif block_type == "artifact":
            artifact_key = str(block.get("artifact_key") or "").strip()
            if not artifact_key:
                issues.append(f"block_{idx}_missing_artifact_key")
            elif artifact_key not in registry:
                issues.append(f"block_{idx}_unknown_artifact_key")
    return issues


def _hydrate_report_blocks(
    payload: Dict[str, Any],
    registry: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    hydrated: List[Dict[str, Any]] = []
    for block in payload.get("blocks", []) or []:
        if not isinstance(block, dict):
            continue
        block_type = str(block.get("type") or "").strip().lower()
        if block_type == "heading":
            text = sanitize_text(str(block.get("text") or "")).strip()
            if not text:
                continue
            level = block.get("level", 2)
            try:
                level = int(level)
            except Exception:
                level = 2
            level = max(1, min(3, level))
            hydrated.append({"type": "heading", "level": level, "text": text})
            continue
        if block_type == "paragraph":
            text = sanitize_text(str(block.get("text") or "")).strip()
            if text:
                hydrated.append({"type": "paragraph", "text": text})
            continue
        if block_type in {"bullet_list", "numbered_list"}:
            items = _coerce_block_items(block.get("items"))
            if items:
                hydrated.append({"type": block_type, "items": items})
            continue
        if block_type == "artifact":
            artifact_key = str(block.get("artifact_key") or "").strip()
            artifact = registry.get(artifact_key)
            if not artifact:
                continue
            analysis_items = _coerce_block_items(block.get("analysis"), max_items=4)
            if not analysis_items:
                analysis_text = sanitize_text(str(block.get("analysis_text") or block.get("analysis_note") or "")).strip()
                if analysis_text:
                    analysis_items = [analysis_text]
            lead_in = sanitize_text(str(block.get("lead_in") or block.get("intro") or "")).strip()
            hydrated_block = {
                "type": "artifact",
                "artifact_key": artifact_key,
                "artifact_type": artifact.get("artifact_type"),
                "title": artifact.get("title"),
                "lead_in": lead_in,
                "analysis": analysis_items,
            }
            if artifact.get("path"):
                hydrated_block["path"] = artifact.get("path")
            if artifact.get("content_html"):
                hydrated_block["content_html"] = artifact.get("content_html")
            if artifact.get("content_markdown"):
                hydrated_block["content_markdown"] = artifact.get("content_markdown")
            hydrated.append(hydrated_block)
    return hydrated


def _render_report_blocks_to_markdown(
    blocks: List[Dict[str, Any]],
    *,
    title: str,
    evidence_paths: List[str],
    llm_evidence_items: Optional[List[Dict[str, str]]] = None,
    target_language_code: str = "es",
) -> str:
    parts: List[str] = []
    has_h1 = any(
        isinstance(block, dict) and block.get("type") == "heading" and int(block.get("level", 2)) == 1
        for block in (blocks or [])
        if isinstance(block, dict)
    )
    clean_title = sanitize_text(str(title or "")).strip()
    if clean_title and not has_h1:
        parts.append(f"# {clean_title}")

    for block in blocks or []:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "markdown":
            parts.append(str(block.get("content") or "").strip())
        elif block_type == "heading":
            level = max(1, min(3, int(block.get("level", 2))))
            parts.append(f'{"#" * level} {block.get("text", "").strip()}')
        elif block_type == "paragraph":
            parts.append(str(block.get("text") or "").strip())
        elif block_type == "bullet_list":
            items = [f"- {item}" for item in block.get("items", []) if str(item or "").strip()]
            if items:
                parts.append("\n".join(items))
        elif block_type == "numbered_list":
            numbered = [
                f"{idx}. {item}"
                for idx, item in enumerate(
                    [str(item).strip() for item in block.get("items", []) if str(item or "").strip()],
                    start=1,
                )
            ]
            if numbered:
                parts.append("\n".join(numbered))
        elif block_type == "artifact":
            lead_in = str(block.get("lead_in") or "").strip()
            if lead_in:
                parts.append(lead_in)
            artifact_type = str(block.get("artifact_type") or "").strip().lower()
            if artifact_type == "chart" and block.get("path"):
                title_text = str(block.get("title") or "Chart").strip() or "Chart"
                parts.append(f'![{title_text}]({str(block.get("path")).replace("\\", "/")})')
            elif artifact_type == "html_table" and block.get("content_html"):
                parts.append(str(block.get("content_html")).strip())
            elif artifact_type == "data_preview" and block.get("content_markdown"):
                parts.append(str(block.get("content_markdown")).strip())
            analysis_items = [str(item).strip() for item in block.get("analysis", []) if str(item or "").strip()]
            if analysis_items:
                parts.append("\n\n".join(analysis_items))

    if not any(
        isinstance(block, dict)
        and block.get("type") == "markdown"
        and _contains_evidence_heading(str(block.get("content") or ""))
        for block in (blocks or [])
    ):
        parts.append(_evidence_section_heading(target_language_code))
        parts.append(
            _canonical_evidence_section(
                evidence_paths,
                llm_items=llm_evidence_items,
                target_language_code=target_language_code,
            )
        )
    return _normalize_evidence_sources("\n\n".join(part for part in parts if str(part or "").strip()).strip() + "\n")


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
2. Design the narrative order that best serves the decision-maker for this run.
   The executive decision and its rationale should become clear early, but the
   exact sequence of evidence, risks, and actions is yours to decide.
3. For each section, decide which artifacts (charts, data previews, tables) should
   be embedded inline to support the claims. Charts and data previews should NOT be
   grouped in a separate "Visual Analysis" section — they belong next to the finding
   they illustrate.
4. Use only supported claims; if evidence is weak, surface that uncertainty.

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
      "evidence_refs": ["artifact/path.json", "..."],
      "inline_artifacts": ["static/plots/feature_importance.png", "..."]
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
    target_lineage_summary: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if not isinstance(steward_summary, dict):
        steward_summary = {}
    if not isinstance(data_profile, dict):
        data_profile = {}
    if not isinstance(dataset_semantics, dict):
        dataset_semantics = {}
    if not isinstance(target_lineage_summary, dict):
        target_lineage_summary = {}

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

    summary_excerpt = str(steward_summary.get("summary") or "")[:1200]
    summary_excerpt_scope = "preliminary_steward_assessment"
    summary_excerpt_warning = ""
    if bool(target_lineage_summary.get("preliminary_summary_conflicts_with_validated_semantics")):
        summary_excerpt_scope = "preliminary_steward_assessment"
        summary_excerpt_warning = (
            "The steward summary excerpt reflects the preliminary assessment and conflicts with the "
            "validated steward semantics. Preserve that disagreement explicitly if it matters."
        )

    return {
        "rows": profile_stats.get("n_rows"),
        "cols": profile_stats.get("n_cols"),
        "primary_target": selectors.get("primary_target") or target_analysis.get("primary_target"),
        "target_status": target_analysis.get("target_status"),
        "recommended_primary_target": target_analysis.get("recommended_primary_target"),
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
        "summary_excerpt": summary_excerpt,
        "summary_excerpt_scope": summary_excerpt_scope,
        "summary_excerpt_warning": summary_excerpt_warning,
        "target_lineage": target_lineage_summary,
    }

class BusinessTranslatorAgent:
    def __init__(self, api_key: str = None):
        """
        Initializes the Business Translator Agent via OpenRouter.
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is required.")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        self.model_name = os.getenv("TRANSLATOR_MODEL", "google/gemini-3-flash-preview").strip()
        if not self.model_name:
            self.model_name = "google/gemini-3-flash-preview"
        self.repair_model_name = os.getenv("TRANSLATOR_REPAIR_MODEL", self.model_name).strip()
        if not self.repair_model_name:
            self.repair_model_name = self.model_name
        self.last_prompt = None
        self.last_response = None
        self.last_report_blocks = None
        self.last_report_payload = None
        try:
            self._max_tokens = int(os.getenv("TRANSLATOR_MAX_TOKENS", "16384"))
        except Exception:
            self._max_tokens = 16384

    def _call_llm(self, prompt: str, *, model_name: Optional[str] = None) -> str:
        model = getattr(self, "model", None)
        if model is not None:
            if hasattr(model, "generate_content"):
                response = model.generate_content(prompt)
                text = getattr(response, "text", response)
                return str(text or "").strip()
            if callable(model):
                return str(model(prompt) or "").strip()
        selected_model = str(model_name or self.model_name or "").strip() or self.model_name
        response = create_chat_completion_with_reasoning(
            self.client,
            agent_name="translator",
            model_name=selected_model,
            call_kwargs={
                "model": selected_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": int(self._max_tokens),
            },
        )
        finish_reason = getattr(response.choices[0], "finish_reason", None)
        if finish_reason in ("length", "max_tokens"):
            print(
                f"WARNING [Translator]: LLM response truncated "
                f"(finish_reason={finish_reason}, max_tokens={self._max_tokens}). "
                f"Consider raising TRANSLATOR_MAX_TOKENS."
            )
        return (response.choices[0].message.content or "").strip()

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
            normalized = _normalize_path(path)
            if artifact_index:
                for item in artifact_index:
                    if not isinstance(item, dict):
                        continue
                    item_path = _normalize_path(item.get("path"))
                    if item_path != normalized:
                        continue
                    if "present" in item:
                        return bool(item.get("present"))
                    status = str(item.get("status") or "").strip().lower()
                    if status in {"ok", "present_optional"}:
                        return True
                    if os.path.exists(normalized):
                        return True
                return False
            return os.path.exists(normalized)
        
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
        work_dir = str(state.get("work_dir") or ".").strip() if isinstance(state, dict) else "."
        
        # Load optional artifacts for context
        integrity_audit = _safe_load_json("data/integrity_audit_report.json") or {}
        output_contract_report = _safe_load_json("data/output_contract_report.json") or {}
        case_alignment_report = _safe_load_json("data/case_alignment_report.json") or {}
        data_adequacy_report = _safe_load_json("data/data_adequacy_report.json") or {}
        ml_review_stack = _safe_load_json_candidates(
            "data/ml_review_stack.json",
            os.path.join(work_dir, "data", "ml_review_stack.json"),
            os.path.join("work", "data", "ml_review_stack.json"),
        ) or {}
        review_board_verdict = _safe_load_json_candidates(
            "data/review_board_verdict.json",
            os.path.join(work_dir, "data", "review_board_verdict.json"),
            os.path.join("work", "data", "review_board_verdict.json"),
        ) or {}
        # QA reviewer output. The QA agent frequently approves with warnings —
        # warnings in this agent are *materially significant* (they represent
        # the senior QA verdict on unresolved risks) and must be available to
        # the translator so the executive report can represent them honestly.
        qa_last_result = state.get("qa_last_result")
        if not isinstance(qa_last_result, dict):
            qa_last_result = {}
        alignment_check_report = _safe_load_json("data/alignment_check.json") or {}
        plot_insights = (
            _safe_load_json_candidates(
                "data/plot_insights.json",
                os.path.join(work_dir, "data", "plot_insights.json"),
            )
            or {}
        )
        insights = _safe_load_json("data/insights.json") or {}
        steward_summary = _safe_load_json("data/steward_summary.json") or {}
        data_profile = (
            _safe_load_json("data/data_profile.json")
            or _safe_load_json(os.path.join(work_dir, "data", "data_profile.json"))
            or _safe_load_json(os.path.join("work", "data", "data_profile.json"))
            or _safe_load_json(os.path.join("work", "artifacts", "data_profile.json"))
            or {}
        )
        dataset_semantics = (
            _safe_load_json("data/dataset_semantics.json")
            or _safe_load_json(os.path.join(work_dir, "data", "dataset_semantics.json"))
            or _safe_load_json(os.path.join("work", "data", "dataset_semantics.json"))
            or {}
        )
        target_lineage_summary = (
            _safe_load_json("data/target_lineage_summary.json")
            or _safe_load_json(os.path.join(work_dir, "data", "target_lineage_summary.json"))
            or _safe_load_json(os.path.join("work", "data", "target_lineage_summary.json"))
            or {}
        )
        if not isinstance(target_lineage_summary, dict) or not target_lineage_summary:
            target_lineage_summary = build_target_lineage_summary(
                steward_summary if isinstance(steward_summary, dict) else {},
                dataset_semantics if isinstance(dataset_semantics, dict) else {},
                contract if isinstance(contract, dict) else {},
            )
        cleaning_manifest = (
            state.get("cleaning_manifest")
            if isinstance(state.get("cleaning_manifest"), dict)
            else _safe_load_json_candidates(
                "data/cleaning_manifest.json",
                "artifacts/clean/cleaning_manifest.json",
                os.path.join(work_dir, "data", "cleaning_manifest.json"),
                os.path.join(work_dir, "artifacts", "clean", "cleaning_manifest.json"),
                os.path.join("work", "artifacts", "clean", "cleaning_manifest.json"),
            )
        ) or {}
        run_summary = _safe_load_json("data/run_summary.json") or {}
        recommendations_preview = _safe_load_json("reports/recommendations_preview.json") or {}
        metrics_payload = _load_authoritative_metrics_payload()
        data_adequacy_report = _resolve_authoritative_data_adequacy_report(
            run_summary if isinstance(run_summary, dict) else {},
            data_adequacy_report if isinstance(data_adequacy_report, dict) else {},
        )

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
            persisted_round_history = _mls.get("round_history")
            if isinstance(persisted_round_history, list) and persisted_round_history:
                metric_loop_context["round_history"] = _compact_metric_round_history_for_translator(
                    persisted_round_history
                )
        round_history = state.get("ml_improvement_round_history")
        if isinstance(round_history, list) and round_history:
            metric_loop_context["round_history"] = _compact_metric_round_history_for_translator(round_history)

        steward_signal_pack = _extract_steward_signal_pack(
            steward_summary=steward_summary,
            data_profile=data_profile,
            dataset_semantics=dataset_semantics,
            target_lineage_summary=target_lineage_summary,
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
        derived_decision_label = _derive_exec_decision(
            review_verdict or compliance,
            data_adequacy_report,
            metrics_payload,
        )
        run_outcome_token = _normalize_decision_token((run_summary or {}).get("run_outcome"))
        executive_decision_label = run_outcome_token or derived_decision_label

        required_outputs: List[str] = []
        required_outputs.extend(_collect_declared_artifact_paths(contract.get("required_outputs")))
        artifact_requirements = contract.get("artifact_requirements") if isinstance(contract.get("artifact_requirements"), dict) else {}
        if isinstance(artifact_requirements, dict):
            required_files = artifact_requirements.get("required_files")
            if isinstance(required_files, list):
                required_outputs.extend(_collect_declared_artifact_paths(required_files))
        # Keep insertion order
        required_outputs = [p for idx, p in enumerate(required_outputs) if p and p not in required_outputs[:idx]]

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
                if all(isinstance(c, dict) for c in conversions):
                    conversion_keys = [c.get("column") for c in conversions if isinstance(c, dict) and c.get("column")]
                else:
                    conversion_keys = [str(c) for c in conversions if str(c or "").strip()]
                conversion_keys = conversion_keys[:12]
            return {
                "row_counts": row_counts,
                "dropped_rows": dropped,
                "conversion_keys": conversion_keys,
                "cleaning_gates_status": cleaning_manifest.get("cleaning_gates_status", {}),
                "output_dialect": cleaning_manifest.get("output_dialect", {}),
                "contract_conflicts_resolved": cleaning_manifest.get("contract_conflicts_resolved", []),
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
            if isinstance(metrics_payload, dict) and metrics_payload:
                metrics_summary["final_metrics_payload"] = metrics_payload
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
            scoring_semantics = _summarize_scored_row_semantics(rows, columns, decimal)
            return {
                "row_count_sampled": scored_rows.get("row_count_sampled", 0),
                "columns": columns,
                "numeric_summary": numeric_summary,
                "examples": examples,
                "scoring_semantics": scoring_semantics,
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

        alignment_check_context = _summarize_alignment_check()
        gate_context = _summarize_gate_context()
        steward_context = _summarize_steward()
        cleaning_context = _summarize_cleaning()
        weights_context = _summarize_weights()
        case_summary_context = _summarize_case_summary()
        scored_rows_context = _summarize_scored_rows()
        run_summary_context = _summarize_run()
        data_adequacy_context = _summarize_data_adequacy()
        model_metrics_context = _summarize_model_metrics()
        cleaning_progress_summary = _build_cleaning_progress_summary(
            cleaning_manifest,
            strategy_title=strategy_title,
            hypothesis=hypothesis,
        )
        eda_fact_pack = _build_eda_fact_pack(
            cleaning_manifest=cleaning_manifest if isinstance(cleaning_manifest, dict) else {},
            data_profile=data_profile if isinstance(data_profile, dict) else {},
            dataset_semantics=dataset_semantics if isinstance(dataset_semantics, dict) else {},
        )
        data_engineer_change_summary = _summarize_data_engineer_change_summary(
            cleaning_manifest=cleaning_manifest if isinstance(cleaning_manifest, dict) else {},
            eda_fact_pack=eda_fact_pack,
            cleaning_progress_summary=cleaning_progress_summary,
        )
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
        plot_summaries = state.get("plot_summaries") if isinstance(state.get("plot_summaries"), list) else None
        if not isinstance(plot_summaries, list):
            plot_summaries = _safe_load_json_candidates(
                os.path.join(work_dir, "static", "plots", "plot_summaries.json"),
                os.path.join("work", "static", "plots", "plot_summaries.json"),
                "static/plots/plot_summaries.json",
            )
            if not isinstance(plot_summaries, list):
                plot_summaries = None
        plot_summaries = _enrich_plot_summaries_with_eda_fact_pack(
            plot_summaries,
            plots,
            eda_fact_pack,
        )
        slot_payloads, model_metrics_context, plot_summaries, metric_progress_summary = _prepare_translator_metric_views(
            metrics_payload=metrics_payload if isinstance(metrics_payload, dict) else {},
            slot_payloads=slot_payloads if isinstance(slot_payloads, dict) else {},
            model_metrics_context=model_metrics_context,
            plot_summaries=plot_summaries,
            metric_loop_context=metric_loop_context if isinstance(metric_loop_context, dict) else {},
            canonical_metric_name=_canonical_metric_name,
            canonical_metric_value=_canonical_metric_value,
        )
        sanitized_review_board_verdict = _sanitize_review_board_verdict_for_translator(
            review_board_verdict if isinstance(review_board_verdict, dict) else {},
            _canonical_metric_name,
            _canonical_metric_value,
        )
        final_incumbent_state = _build_final_incumbent_state(
            executive_decision_label=executive_decision_label,
            run_outcome_token=run_outcome_token,
            review_board_verdict=review_board_verdict if isinstance(review_board_verdict, dict) else {},
            output_contract_report=output_contract_report if isinstance(output_contract_report, dict) else {},
            slot_payloads=slot_payloads if isinstance(slot_payloads, dict) else {},
            canonical_metric_name=_canonical_metric_name,
            canonical_metric_value=_canonical_metric_value,
            run_summary=run_summary if isinstance(run_summary, dict) else {},
        )
        ml_engineer_change_summary = _summarize_ml_engineer_change_summary(
            metric_loop_context=metric_loop_context if isinstance(metric_loop_context, dict) else {},
            canonical_metric_name=_canonical_metric_name,
            canonical_metric_value=_canonical_metric_value,
            review_board_verdict=sanitized_review_board_verdict,
            final_incumbent_state=final_incumbent_state,
        )
        final_incumbent_metric_records = []
        if isinstance(slot_payloads, dict):
            final_incumbent_metric_records = slot_payloads.get("model_metrics") or []
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
        evidence_paths = _select_confirmed_evidence_paths(
            manifest,
            raw_evidence_paths,
            artifact_available_fn=_artifact_available,
            max_items=8,
        )
        artifact_inventory_table_html = _build_artifact_inventory_table_html(manifest)
        artifact_compliance_table_html = _build_artifact_compliance_table_html(
            manifest,
            output_contract_report,
            review_verdict or compliance,
            gate_context if isinstance(gate_context, dict) else {},
            run_summary if isinstance(run_summary, dict) else {},
        )
        kpi_snapshot_table_html = _build_kpi_snapshot_table_html(
            final_incumbent_metric_records,
            _canonical_metric_name,
            _canonical_metric_value,
            data_adequacy_report if isinstance(data_adequacy_report, dict) else {},
            decisioning_columns,
            executive_decision_label,
        )
        run_causal_impact_summary = _summarize_run_causal_impact(
            data_engineer_change_summary=data_engineer_change_summary,
            ml_engineer_change_summary=ml_engineer_change_summary,
            executive_decision_label=executive_decision_label,
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
            with open("data/eda_fact_pack.json", "w", encoding="utf-8") as f_eda:
                json.dump(eda_fact_pack or {}, f_eda, indent=2, ensure_ascii=False)
            with open("data/data_engineer_change_summary.json", "w", encoding="utf-8") as f_de:
                json.dump(data_engineer_change_summary or {}, f_de, indent=2, ensure_ascii=False)
            with open("data/ml_engineer_change_summary.json", "w", encoding="utf-8") as f_ml:
                json.dump(ml_engineer_change_summary or {}, f_ml, indent=2, ensure_ascii=False)
            with open("data/run_causal_impact_summary.json", "w", encoding="utf-8") as f_causal:
                json.dump(run_causal_impact_summary or {}, f_causal, indent=2, ensure_ascii=False)
            with open("data/final_incumbent_state.json", "w", encoding="utf-8") as f_incumbent:
                json.dump(final_incumbent_state or {}, f_incumbent, indent=2, ensure_ascii=False)
            with open("data/governance_contradiction_packet.json", "w", encoding="utf-8") as f_contradictions:
                json.dump(governance_contradiction_packet or {}, f_contradictions, indent=2, ensure_ascii=False)
            with open("data/stale_or_rejected_history.json", "w", encoding="utf-8") as f_history:
                json.dump(stale_or_rejected_history or {}, f_history, indent=2, ensure_ascii=False)
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

        metrics_table_text = _metrics_table_from_records(final_incumbent_metric_records, max_items=10)
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
        business_objective_summary = _build_business_objective_summary(
            business_objective,
            strategy_title=strategy_title,
            hypothesis=hypothesis,
        )

        decision_discrepancy = None
        if run_outcome_token and derived_decision_label and run_outcome_token != derived_decision_label:
            decision_discrepancy = {
                "authoritative_decision": executive_decision_label,
                "derived_decision": derived_decision_label,
                "run_outcome": run_outcome_token,
                "note": "derived reviewer/adequacy decision differs from authoritative run outcome",
            }

        governance_contradiction_packet = _build_governance_contradiction_packet(
            review_board_verdict=review_board_verdict if isinstance(review_board_verdict, dict) else {},
            ml_review_stack=ml_review_stack if isinstance(ml_review_stack, dict) else {},
            data_adequacy_report=data_adequacy_report if isinstance(data_adequacy_report, dict) else {},
            final_incumbent_state=final_incumbent_state,
            decision_discrepancy=decision_discrepancy,
        )
        stale_or_rejected_history = _build_stale_or_rejected_history(
            review_board_verdict=review_board_verdict if isinstance(review_board_verdict, dict) else {},
            ml_review_stack=ml_review_stack if isinstance(ml_review_stack, dict) else {},
            governance_contradiction_packet=governance_contradiction_packet,
            decision_discrepancy=decision_discrepancy,
        )

        # Build a compact, normalized QA signal block. QA warnings/soft
        # failures represent material caveats the executive report must not
        # silently drop. We keep this as a separate fact entry so the model
        # can reason about QA verdicts without having to parse the whole
        # qa_last_result dict.
        qa_status_raw = str(qa_last_result.get("status") or "").strip() or None
        qa_feedback_raw = str(qa_last_result.get("feedback") or "").strip()
        qa_review_signals: Dict[str, Any] = {
            "status": qa_status_raw,
            "status_is_warning": bool(
                qa_status_raw
                and (
                    "WARNING" in qa_status_raw.upper()
                    or "WITH_WARN" in qa_status_raw.upper()
                )
            ),
            "feedback_excerpt": (qa_feedback_raw[:1600] if qa_feedback_raw else ""),
            "failed_gates": [
                str(x) for x in (qa_last_result.get("failed_gates") or []) if str(x or "").strip()
            ][:12],
            "hard_failures": [
                str(x) for x in (qa_last_result.get("hard_failures") or []) if str(x or "").strip()
            ][:12],
            "soft_failures": [
                str(x) for x in (qa_last_result.get("soft_failures") or []) if str(x or "").strip()
            ][:12],
            "explicit_warnings": [
                str(x) for x in (qa_last_result.get("warnings") or []) if str(x or "").strip()
            ][:12],
            "required_fixes": [
                str(x) for x in (qa_last_result.get("required_fixes") or []) if str(x or "").strip()
            ][:8],
            "qa_gates_evaluated": [
                str(x) for x in (qa_last_result.get("qa_gates_evaluated") or []) if str(x or "").strip()
            ][:20],
        }
        # Keep a small evidence preview only when the QA verdict carries
        # non-empty soft/warning signals — it helps the translator cite the
        # caveat accurately without overloading the prompt.
        qa_evidence = qa_last_result.get("evidence")
        if isinstance(qa_evidence, list) and (
            qa_review_signals["status_is_warning"]
            or qa_review_signals["failed_gates"]
            or qa_review_signals["soft_failures"]
            or qa_review_signals["explicit_warnings"]
        ):
            qa_review_signals["evidence_preview"] = [
                {
                    "claim": str((item or {}).get("claim") or "")[:240],
                    "source": str((item or {}).get("source") or "")[:200],
                }
                for item in qa_evidence
                if isinstance(item, dict)
            ][:6]

        facts_block = {
            "executive_decision_label": executive_decision_label,
            "authoritative_run_outcome": run_outcome_token,
            "derived_decision_label": derived_decision_label,
            "decision_discrepancy": decision_discrepancy,
            "final_incumbent_state": final_incumbent_state,
            "governance_contradiction_packet": governance_contradiction_packet,
            "stale_or_rejected_history": stale_or_rejected_history,
            "qa_review_signals": qa_review_signals,
            "business_objective": business_objective,
            "strategy_title": strategy_title,
            "review_verdict": review_verdict or compliance,
            "steward_signal_pack": steward_signal_pack,
            "target_lineage": target_lineage_summary,
            "data_adequacy": {
                "status": (data_adequacy_report or {}).get("status"),
                "reasons": (data_adequacy_report or {}).get("reasons", [])[:3],
                "recommendations": (data_adequacy_report or {}).get("recommendations", [])[:3],
            },
            "cleaning_progress_summary": cleaning_progress_summary,
            "eda_fact_pack": eda_fact_pack,
            "data_engineer_change_summary": data_engineer_change_summary,
            "artifacts_summary": manifest.get("summary", {}) if isinstance(manifest, dict) else {},
            "decisioning_columns": decisioning_columns,
            "metrics_preview": final_incumbent_metric_records[:14] if isinstance(final_incumbent_metric_records, list) else [],
            "canonical_primary_metric": {
                "name": _canonical_metric_name,
                "value": _canonical_metric_value,
                "source": "primary_metric_state (authoritative)",
            } if _canonical_metric_name and _canonical_metric_value is not None else None,
            "scoring_output_semantics": (
                (final_incumbent_state.get("predictions_output") or {}).get("scoring_semantics")
                if isinstance(final_incumbent_state, dict)
                and isinstance(final_incumbent_state.get("predictions_output"), dict)
                else {}
            ),
            "ml_progress_summary": metric_progress_summary,
            "ml_engineer_change_summary": ml_engineer_change_summary,
            "run_causal_impact_summary": run_causal_impact_summary,
        }

        context_appendix = {
            "reporting_policy": reporting_policy_context,
            "translator_view": translator_view_context,
            "slot_payloads": slot_payloads,
            "slot_coverage": slot_coverage_context,
            "steward_signal_pack": steward_signal_pack,
            "target_lineage": target_lineage_summary,
            "steward_context": steward_context,
            "cleaning_context": cleaning_context,
            "cleaning_progress_summary": cleaning_progress_summary,
            "eda_fact_pack": eda_fact_pack,
            "data_engineer_change_summary": data_engineer_change_summary,
            "run_summary_context": run_summary_context,
            "artifact_manifest": manifest,
            "data_adequacy_report_json": data_adequacy_report,
            "review_board_verdict_json": sanitized_review_board_verdict,
            "ml_review_stack_json": ml_review_stack,
            "alignment_check": alignment_check_context,
            "model_metrics_context": model_metrics_context,
            "ml_engineer_change_summary": ml_engineer_change_summary,
            "case_summary_context": case_summary_context,
            "scored_rows_context": scored_rows_context,
            "plot_insights_json": plot_insights,
            "plot_summaries": plot_summaries,
            "recommendations_preview": recommendations_preview,
            "run_timeline_context": run_timeline_context,
            "metric_loop_context": metric_loop_context if metric_loop_context else None,
            "metric_progress_summary": metric_progress_summary,
            "run_causal_impact_summary": run_causal_impact_summary,
            "final_incumbent_state": final_incumbent_state,
            "governance_contradiction_packet": governance_contradiction_packet,
            "stale_or_rejected_history": stale_or_rejected_history,
        }

        # ── Pipeline scope awareness ─────────────────────────────────
        _contract_for_scope = state.get("execution_contract")
        _pipeline_scope = ""
        if isinstance(_contract_for_scope, dict):
            _pipeline_scope = str(_contract_for_scope.get("scope", "")).strip().lower()
        if _pipeline_scope == "cleaning_only":
            pipeline_scope_section = (
                "PIPELINE SCOPE: CLEANING_ONLY\n"
                "This run performed DATA CLEANING ONLY — no ML model was trained.\n"
                "Your report must focus on:\n"
                "  - Data quality improvements achieved (before vs after)\n"
                "  - Cleaning operations performed and their rationale\n"
                "  - Data validation results and gate compliance\n"
                "  - Recommendations for data usage or further processing\n"
                "Do NOT reference model performance, predictions, or ML metrics — they do not exist.\n"
                "The executive decision should assess whether the cleaned data meets quality standards."
            )
        elif _pipeline_scope == "ml_only":
            pipeline_scope_section = (
                "PIPELINE SCOPE: ML_ONLY\n"
                "This run used PRE-CLEANED data and focused on ML modeling only.\n"
                "Your report must focus on:\n"
                "  - Model performance and evaluation metrics\n"
                "  - Feature importance and model interpretability\n"
                "  - Predictions quality and business applicability\n"
                "Minimize discussion of data cleaning — it was not performed in this run."
            )
        else:
            pipeline_scope_section = (
                "PIPELINE SCOPE: FULL_PIPELINE\n"
                "This run performed the complete pipeline: data cleaning + ML modeling.\n"
                "Cover both data quality and model performance in your report."
            )

        # ── Run narrative (structured summary built by graph.py) ─────
        run_narrative = state.get("run_narrative")
        if isinstance(run_narrative, dict) and run_narrative:
            run_narrative_section = json.dumps(run_narrative, ensure_ascii=False, indent=2)
        else:
            run_narrative_section = "Not available — use FACTS and DETAILED CONTEXT below."

        SYSTEM_PROMPT_TEMPLATE = Template("""
$senior_translation_protocol

$senior_evidence_rule

=== PIPELINE SCOPE ===
$pipeline_scope_section

=== MISSION ===
Write an executive report in $target_language_name ($target_language_code)
for a decision-maker who has NOT seen the raw data. The report must enable
them to understand what happened, whether the results are trustworthy, and
what to do next.

=== SOURCE OF TRUTH AND PRECEDENCE ===
- FACTS_BLOCK is authoritative for the executive decision label. If it includes
  an authoritative run outcome, it overrides softer signals from reviewer verdicts,
  data adequacy, heuristics, or narrative text.
- FINAL_INCUMBENT_STATE is authoritative for the current selected system state:
  current deliverability, incumbent KPI, and whether required outputs exist now.
- If FACTS_BLOCK includes scoring_output_semantics, use it as the authority
  for rank/percentile direction and recommended sorting. Do not infer that a
  higher percentile/rank column means higher risk unless the deterministic
  scoring semantics say so.
- STALE_OR_REJECTED_HISTORY contains reviewer/governance history that may explain
  disagreements or rejected paths, but it is not the final current state.
- If GOVERNANCE_CONTRADICTION_PACKET lists contradictions, treat the contradicted
  blockers as stale or historical unless current deterministic facts independently confirm them.
- RUN NARRATIVE and DETAILED CONTEXT explain what happened and why; they do not
  override the authoritative executive outcome.
- Evidence must come from listed artifacts or explicit deterministic facts.
  If support is missing, state uncertainty explicitly instead of presenting the claim as established.
- Separate FINAL INCUMBENT STATE from IMPROVEMENT HISTORY.
  KPI snapshots, stability claims, and deployment recommendations must be grounded
  in the final incumbent only. Rejected challengers are exploration history, not final state.
- Distinguish clearly between supported facts, cautious inference, and recommended action.
  Use assertive language only for facts supported by artifacts or deterministic context.
  Timelines, thresholds, rollout gates, or policies not explicit in the artifacts
  must be framed as recommendations, not established run facts.

=== REPORT REASONING ===
Before writing, reason internally about these areas. Your report should
reflect this analysis, not just list data. The order and depth you give
each area is yours to decide based on what matters most for THIS run.

OUTCOME ASSESSMENT
- The authoritative executive outcome for this report is: $executive_decision_label
- What was the business objective? Did the system achieve it?
- If reviewer or adequacy signals differ from the authoritative outcome,
  explain the discrepancy but do NOT override the authoritative outcome.
- If target_lineage is present, preserve the difference between the
  preliminary steward assessment, the validated steward semantics, and the
  final contract target. Do not rewrite those stages as if they were always aligned.
- If a metric improvement loop ran, what was the best metric achieved
  vs the baseline? How many techniques were tried?
- If the canonical_primary_metric in FACTS_BLOCK differs from metrics
  on disk, trust the canonical value — it reflects the selected incumbent.
- If improvement history exists, make the ML engineer's progress visible,
  but keep final KPI/stability language tied to the selected incumbent only.

QA REVIEWER SIGNALS ARE MATERIAL CAVEATS
- facts_block.qa_review_signals carries the senior QA reviewer's final
  assessment of the pipeline's unresolved risks. Its status, feedback,
  explicit warnings, soft failures, and failed gates are not optional
  context — they are the reviewer's judgement on what the downstream
  consumer needs to know before acting on the result.
- If qa_review_signals.status is APPROVE_WITH_WARNINGS (or any variant
  flagged by status_is_warning), the report's limitations section must
  faithfully represent those warnings. Do not silently drop them just
  because the authoritative outcome is positive — the two coexist: the
  pipeline succeeded *and* the reviewer surfaced unresolved caveats.
- Read qa_review_signals.feedback_excerpt for the reviewer's own
  reasoning. When warnings inside that feedback describe a risk that
  affects metric interpretation (for example the validation method did
  not fully enforce its claimed invariants, a calibration caveat, or a
  specification deviation), that risk belongs in the limitations or
  "how confident can we be" section of the report, not omitted.
- soft_failures and explicit_warnings are named slots for caveats; list
  them plainly. failed_gates and hard_failures are harder blockers and
  must be surfaced with the corresponding evidence from the reviewer.
- You choose the phrasing and depth, but materially significant QA
  caveats may not be omitted. A silent omission of a senior reviewer's
  stated reservation is a reporting error, not a stylistic choice.

WHAT MATTERS
- From all the metrics and artifacts below, decide which findings are
  truly decision-relevant for this run. Include as much supporting detail
  as needed to make the report clear, rigorous, and useful.
- Prioritize: primary metric performance, data quality issues,
  compliance failures, and risks that affect production readiness.
- If cleaning work materially enabled the final result, surface the
  most important data-engineering operations and explain why they mattered.
- If an EDA Fact Pack exists, use it to identify the few missingness,
  cardinality, constant-column, or type-conversion signals that genuinely
  explain model readiness or residual risk.
- If engineering change summaries exist, distinguish clearly between
  accepted interventions, rejected experiments, and the concrete effect each
  had on data readiness, model quality, or deployment trust.

ENGINEERING IMPACT
- Which engineering interventions actually changed the system state?
- Describe accepted data-engineering and model-engineering moves that
  changed readiness, incumbent quality, or deployment trust at the level of
  detail that best serves the report.
- If experiments were rejected, mention them only when they explain why the
  final incumbent was kept or why deployment remains limited.
- Do not mention agents as workflow theater. Give credit to data engineering
  or model engineering work only when it materially changed the outcome.

CAUSALITY
- Connect results to causes. If the metric improved, what technique
  drove it? If it degraded, what went wrong?
- If there are contradictions between reviewers, governance outputs, and metrics, flag them.
- Explain how the problem was solved or partially solved through those
  engineering decisions, not as a flat chronological list of steps.

RECOMMENDED ACTIONS
- Be specific: "retry with X", "investigate Y in artifact Z",
  "deploy with caveat W" — not generic advice.
- If you recommend a timeline, threshold, governance gate, or rollout policy
  that is not explicit in the artifacts, make it clear that it is your
  recommendation rather than an established outcome of the run.

=== FACTS (do not alter values) ===
$facts_block_json

=== RUN NARRATIVE (primary context — what happened during this run) ===
$run_narrative_section

=== REFERENCE CONTEXT ===
Business Objective Summary: $business_objective_summary
Business Objective: $business_objective
Strategy: $strategy_title
Hypothesis: $hypothesis
Compliance: $compliance
Error condition: $error_condition

=== EVIDENCE SOURCES ===
Available artifacts: $evidence_paths_text
Metrics: $metrics_table_text

=== DETAILED CONTEXT ===
Outline Plan (pass-1): $outline_plan_json
Cleaned Data Sample: $cleaned_sample_table_text
Scored Rows Sample: $scored_sample_table_text
Artifact Headers: $artifact_headers_table_text
Recommendations: $recommendations_table_text
Cleaning Progress Summary: $cleaning_progress_summary_json
EDA Fact Pack: $eda_fact_pack_json
Data Engineer Change Summary: $data_engineer_change_summary_json
ML Engineer Change Summary: $ml_engineer_change_summary_json
Run Causal Impact Summary: $run_causal_impact_summary_json
Final Incumbent State: $final_incumbent_state_json
Governance Contradiction Packet: $governance_contradiction_packet_json
Stale or Rejected History: $stale_or_rejected_history_json
Visuals: $visuals_context_json
Decisioning: $decisioning_context_json
Decisioning Columns: $decisioning_columns_text
Reporting Policy: $reporting_policy_context
Slot Coverage: $slot_coverage_context
Metric Progress Summary: $metric_progress_summary_json

=== APPENDIX (lower priority — use only if needed for depth) ===
$context_appendix_json

=== RENDERABLE ARTIFACT REGISTRY ===
These artifacts are available for optional inline embedding. Decide yourself
Place each artifact where it best supports the narrative — do NOT group all
the narrative. No artifact is mandatory.

$embeddable_artifacts_registry_json

=== OUTPUT FORMAT ===
Return ONLY valid JSON. No markdown, no commentary, no code fences.

You are not writing raw HTML or raw markdown directly. Instead, design the
report as a sequence of narrative blocks that the renderer will materialize.
This gives you freedom over section order, whether a table needs a heading,
and where each chart or table should appear.

Schema:
{
  "title": "...",
  "blocks": [
    {"type": "heading", "level": 1, "text": "..."},
    {"type": "paragraph", "text": "..."},
    {"type": "bullet_list", "items": ["...", "..."]},
    {"type": "numbered_list", "items": ["...", "..."]},
    {
      "type": "artifact",
      "artifact_key": "kpi_snapshot",
      "lead_in": "Contextual lead-in before the artifact.",
      "analysis": ["Sentence 1 interpreting the artifact.", "Sentence 2 explaining business impact."]
    }
  ],
  "evidence": [
    {"claim": "...", "source": "artifact_path -> key"}
  ]
}

Rules:
- Make the authoritative executive decision and its rationale clear early.
- Structure and section order are yours to determine based on what matters most.
- Use artifact blocks only when they materially improve clarity or trust.
- Distinguish supported facts, cautious inference, and recommended actions in the wording.
- Do not present inferred rollout policies, exact remediation windows, or governance gates as established facts unless they appear explicitly in the run evidence.
- Do NOT create a separate "Visual Analysis" or "Análisis Visual" section.
- Charts, data previews, and summary tables belong inline next to the finding they support.
- Do NOT emit raw HTML tables or markdown image syntax inside text blocks.
- The Evidence trail will be rendered from the "evidence" array; do not add a separate evidence heading block.
- If the Outline Plan is non-empty, use it as a starting skeleton but adapt freely.
- Use the report to explain how the problem was solved, partially solved,
  or blocked by engineering decisions grounded in the run context.
- Not every artifact must be used. Select and place only those that
  strengthen the narrative. Skip artifacts that add no decision value.
""")

        execution_results = state.get("execution_output", "No execution results available.")
        USER_MESSAGE_TEMPLATE = """
Analyze the context above. Reason about what happened, what matters most,
and what the decision-maker should do. Then return the structured report payload.

EXECUTION FINDINGS:
$execution_results

"""

        artifact_registry = _build_embeddable_artifact_registry(
            plots=plots,
            plot_summaries=plot_summaries,
            cleaned_sample_table_text=cleaned_sample_table_text,
            scored_sample_table_text=scored_sample_table_text,
            kpi_snapshot_table_html=kpi_snapshot_table_html,
            artifact_inventory_table_html=artifact_inventory_table_html,
            artifact_compliance_table_html=artifact_compliance_table_html,
        )
        artifact_registry_prompt_json = _artifact_registry_prompt_json(artifact_registry)

        prompt_values = {
            "senior_translation_protocol": SENIOR_TRANSLATION_PROTOCOL,
            "senior_evidence_rule": SENIOR_EVIDENCE_RULE,
            "target_language_name": target_language_name,
            "target_language_code": target_language_code,
            "facts_block_json": json.dumps(facts_block, ensure_ascii=False),
            "run_narrative_section": run_narrative_section,
            "business_objective_summary": business_objective_summary,
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
            "metric_progress_summary_json": json.dumps(metric_progress_summary, ensure_ascii=False),
            "metrics_table_text": metrics_table_text,
            "cleaned_sample_table_text": cleaned_sample_table_text,
            "scored_sample_table_text": scored_sample_table_text,
            "artifact_headers_table_text": artifact_headers_table_text,
            "recommendations_table_text": recommendations_table_text,
            "cleaning_progress_summary_json": json.dumps(cleaning_progress_summary, ensure_ascii=False),
            "eda_fact_pack_json": json.dumps(eda_fact_pack, ensure_ascii=False),
            "data_engineer_change_summary_json": json.dumps(data_engineer_change_summary, ensure_ascii=False),
            "ml_engineer_change_summary_json": json.dumps(ml_engineer_change_summary, ensure_ascii=False),
            "run_causal_impact_summary_json": json.dumps(run_causal_impact_summary, ensure_ascii=False),
            "final_incumbent_state_json": json.dumps(final_incumbent_state, ensure_ascii=False),
            "governance_contradiction_packet_json": json.dumps(governance_contradiction_packet, ensure_ascii=False),
            "stale_or_rejected_history_json": json.dumps(stale_or_rejected_history, ensure_ascii=False),
            "context_appendix_json": json.dumps(context_appendix, ensure_ascii=False),
            "pipeline_scope_section": pipeline_scope_section,
            "embeddable_artifacts_registry_json": artifact_registry_prompt_json,
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
                outline_text = self._call_llm(outline_prompt)
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
            content = self._call_llm(full_prompt)
            is_echo_response = content.strip() == full_prompt.strip()
            self.last_report_blocks = None
            self.last_report_payload = None
            structured_payload = None
            structured_payload_issues: List[str] = []
            structured_layout_enabled = str(os.getenv("TRANSLATOR_STRUCTURED_LAYOUT_ENABLED", "1")).strip().lower() not in {
                "0",
                "off",
                "false",
                "no",
            }

            if structured_layout_enabled and not is_echo_response:
                materialized_content, materialized_blocks, structured_payload, structured_payload_issues = _materialize_structured_report(
                    content=content,
                    artifact_registry=artifact_registry,
                    evidence_paths=evidence_paths,
                    target_language_code=target_language_code,
                )
                if materialized_content is not None and materialized_blocks is not None:
                    content = materialized_content
                    self.last_report_blocks = materialized_blocks
                    self.last_report_payload = structured_payload
                else:
                    # ── Rescue: LLM returned truncated/malformed JSON ──
                    # Try to extract readable markdown from the raw JSON text
                    # so executive_summary.md is human-readable, not raw JSON.
                    rescued = _rescue_markdown_from_raw_json(content)
                    if rescued:
                        content = rescued

            content = _sanitize_report_text(content)
            if self.last_report_blocks is None:
                content = _ensure_evidence_section(
                    content,
                    evidence_paths,
                    target_language_code=target_language_code,
                )
            content = sanitize_text(content)

            validation = {
                "has_critical": False,
                "critical_issues": [],
                "unverified_metrics": [],
                "structure_issues": [],
                "invalid_plots": [],
                "decision_issue": [],
            }

            # ── Repair loop: up to MAX_REPAIR_ATTEMPTS, keep best attempt ──
            max_repair_attempts = int(os.getenv("TRANSLATOR_MAX_REPAIR_ATTEMPTS", "3"))
            repair_history: List[Dict[str, Any]] = []
            best_content = content
            best_score = 0
            best_validation = validation

            if not is_echo_response:
                validation = _validate_report(
                    content=content,
                    expected_decision=executive_decision_label,
                    facts_context=facts_context if isinstance(facts_context, list) else [],
                    metrics_payload=metrics_payload if isinstance(metrics_payload, dict) else {},
                    plots=plots,
                    expected_language=target_language_code,
                    decision_discrepancy=decision_discrepancy,
                    governance_contradiction_packet=governance_contradiction_packet,
                )
                best_score = _score_report_quality(validation)
                best_content = content
                best_validation = validation
                repair_history.append({
                    "attempt": 0,
                    "score": best_score,
                    "issues": validation.get("critical_issues", []) + validation.get("structure_issues", []) + validation.get("reasoning_warnings", []),
                })

                for repair_attempt in range(max_repair_attempts):
                    if not validation.get("has_critical") and not validation.get("reasoning_warnings") and best_score >= int(os.getenv("TRANSLATOR_MIN_QUALITY_SCORE", "60")):
                        break  # Report is good enough
                    try:
                        if self.last_report_blocks is not None:
                            self.last_report_blocks = None
                            self.last_report_payload = None
                        all_issues = validation.get("critical_issues", []) + validation.get("structure_issues", []) + [f"reasoning_warning: {item}" for item in validation.get("reasoning_warnings", [])[:4]]
                        if not validation.get("has_critical") and not validation.get("reasoning_warnings"):
                            # Quality is low but no critical — add hint
                            all_issues = list(all_issues) + ["low_quality_score"]
                        if structured_layout_enabled:
                            repair_prompt = _build_structured_repair_prompt(
                                report=content,
                                validation={**validation, "critical_issues": all_issues, "has_critical": True},
                                expected_decision=executive_decision_label,
                                evidence_paths=evidence_paths,
                                target_language_code=target_language_code,
                                artifact_registry_prompt_json=artifact_registry_prompt_json,
                            )
                        else:
                            repair_prompt = _build_repair_prompt(
                                report=content,
                                validation={**validation, "critical_issues": all_issues, "has_critical": True},
                                expected_decision=executive_decision_label,
                                evidence_paths=evidence_paths,
                                target_language_code=target_language_code,
                            )
                        repaired = self._call_llm(
                            repair_prompt,
                            model_name=getattr(self, "repair_model_name", None),
                        )
                        if structured_layout_enabled:
                            repaired_content, repaired_blocks, repaired_payload, repaired_structured_issues = _materialize_structured_report(
                                content=repaired,
                                artifact_registry=artifact_registry,
                                evidence_paths=evidence_paths,
                                target_language_code=target_language_code,
                            )
                            if repaired_content is not None and repaired_blocks is not None:
                                repaired = repaired_content
                                self.last_report_blocks = repaired_blocks
                                self.last_report_payload = repaired_payload
                                structured_payload_issues = repaired_structured_issues
                            else:
                                structured_payload_issues = repaired_structured_issues
                                # ── Rescue truncated JSON in repair attempt ──
                                rescued = _rescue_markdown_from_raw_json(repaired)
                                if rescued:
                                    repaired = rescued
                                repaired = _sanitize_report_text(repaired)
                                repaired = _ensure_evidence_section(
                                    repaired,
                                    evidence_paths,
                                    target_language_code=target_language_code,
                                )
                        else:
                            repaired = _sanitize_report_text(repaired)
                            repaired = _ensure_evidence_section(
                                repaired,
                                evidence_paths,
                                target_language_code=target_language_code,
                            )
                        repaired = sanitize_text(repaired)
                        repair_validation = _validate_report(
                            content=repaired,
                            expected_decision=executive_decision_label,
                            facts_context=facts_context if isinstance(facts_context, list) else [],
                            metrics_payload=metrics_payload if isinstance(metrics_payload, dict) else {},
                            plots=plots,
                            expected_language=target_language_code,
                            decision_discrepancy=decision_discrepancy,
                            governance_contradiction_packet=governance_contradiction_packet,
                        )
                        repair_score = _score_report_quality(repair_validation)
                        repair_history.append({
                            "attempt": repair_attempt + 1,
                            "score": repair_score,
                            "issues": repair_validation.get("critical_issues", []) + repair_validation.get("structure_issues", []),
                        })
                        # Always keep the best version seen so far
                        if repair_score >= best_score:
                            best_content = repaired
                            best_score = repair_score
                            best_validation = repair_validation
                        # Feed the best version back into the next repair iteration
                        content = best_content
                        validation = best_validation
                    except Exception as repair_exc:
                        repair_history.append({
                            "attempt": repair_attempt + 1,
                            "score": best_score,
                            "error": str(repair_exc),
                        })
                        break

            # Use the best version produced across all attempts
            content = best_content
            validation = best_validation
            quality_score = best_score if not is_echo_response else 100
            quality_threshold = int(os.getenv("TRANSLATOR_MIN_QUALITY_SCORE", "60"))

            if not is_echo_response and validation.get("unverified_metrics") and not validation.get("has_critical"):
                warning_lines = "\n".join(f"- {item}" for item in validation.get("unverified_metrics", [])[:6])
                content = (
                    content.rstrip()
                    + "\n\n## Validation Notes\n"
                    + "Some metric claims could not be matched deterministically:\n"
                    + warning_lines
                    + "\n"
                )
                if isinstance(self.last_report_blocks, list):
                    self.last_report_blocks.extend(
                        [
                            {"type": "heading", "level": 2, "text": "Validation Notes"},
                            {
                                "type": "bullet_list",
                                "items": [str(item) for item in validation.get("unverified_metrics", [])[:6] if item],
                            },
                        ]
                    )
            if decision_discrepancy:
                content = (
                    content.rstrip()
                    + "\n\n## Decision Reconciliation Note\n"
                    + f"Derived decision: {decision_discrepancy.get('derived_decision')} | "
                    + f"run_summary outcome: {decision_discrepancy.get('run_outcome')}\n"
                )
                if isinstance(self.last_report_blocks, list):
                    self.last_report_blocks.extend(
                        [
                            {"type": "heading", "level": 2, "text": "Decision Reconciliation Note"},
                            {
                                "type": "paragraph",
                                "text": (
                                    f"Derived decision: {decision_discrepancy.get('derived_decision')} | "
                                    f"run_summary outcome: {decision_discrepancy.get('run_outcome')}"
                                ),
                            },
                        ]
                    )
            if prompt_budget_notes:
                content = (
                    content.rstrip()
                    + "\n\n## Prompt Budget Note\n"
                    + "Some low-priority context blocks were compacted to stay within model limits.\n"
                )
                if isinstance(self.last_report_blocks, list):
                    self.last_report_blocks.extend(
                        [
                            {"type": "heading", "level": 2, "text": "Prompt Budget Note"},
                            {
                                "type": "paragraph",
                                "text": "Some low-priority context blocks were compacted to stay within model limits.",
                            },
                        ]
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
                    "structured_layout": {
                        "enabled": structured_layout_enabled,
                        "payload_parsed": bool(structured_payload),
                        "payload_issues": structured_payload_issues,
                        "block_count": len(self.last_report_blocks or []),
                    },
                    "quality_threshold": quality_threshold,
                    "repair_loop": {
                        "max_attempts": max_repair_attempts,
                        "history": repair_history,
                        "total_attempts": len(repair_history),
                    },
                    "quality_score": quality_score,
                    "validation": validation,
                    "decision_discrepancy": decision_discrepancy,
                }
            )
            self.last_response = content
            return content
        except Exception as e:
            # Even on exception, return best effort — never a deterministic stub
            error_report = (
                f"# Executive Report\n\n"
                f"## Decisión Ejecutiva\n\n"
                f"**{executive_decision_label}**\n\n"
                f"Report generation encountered an error: {e}\n\n"
                f"## Riesgos\n\n"
                f"- Report could not be fully generated due to: {e}\n"
                f"- Review artifacts directly for complete analysis.\n\n"
                f"## Evidencia Usada\n\n"
            )
            error_report += _canonical_evidence_section(evidence_paths)
            language_pack = _report_language_pack(target_language_code)
            error_report = (
                f"# Executive Report\n\n"
                f"## {language_pack.get('executive_decision')}\n\n"
                f"**{executive_decision_label}**\n\n"
                f"Report generation encountered an error: {e}\n\n"
                f"## {language_pack.get('risks')}\n\n"
                f"- Report could not be fully generated due to: {e}\n"
                f"- Review artifacts directly for complete analysis.\n\n"
                f"{_evidence_section_heading(target_language_code)}\n\n"
            )
            error_report += _canonical_evidence_section(
                evidence_paths,
                target_language_code=target_language_code,
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
                    "structured_layout": {
                        "enabled": False,
                        "payload_parsed": False,
                        "payload_issues": ["llm_exception"],
                        "block_count": 0,
                    },
                    "quality_score": 0,
                    "validation": {"has_critical": True, "critical_issues": ["llm_exception"]},
                    "exception": str(e),
                }
            )
            self.last_report_blocks = None
            self.last_report_payload = None
            self.last_response = error_report
            return error_report


