import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.utils.run_logger import register_run_log
from src.utils.context_pack import compress_long_lists
from src.utils.contract_accessors import get_required_outputs
from src.utils.review_status import normalize_status as normalize_review_status
from src.utils.text_encoding import sanitize_text, sanitize_text_payload

RUNS_DIR = "runs"

# ---------------------------------------------------------------------------
# P0 FIX: Anti-bloat constants
# ---------------------------------------------------------------------------
# Prefixes to NEVER copy into run bundles (prevents self-referencing loops
# and __pycache__ noise).
_BUNDLE_EXCLUDE_PREFIXES: List[str] = [
    "runs",
    "__pycache__",
    ".git",
    ".venv",
    "node_modules",
]

# CSVs larger than this are replaced by a lightweight .meta.json stub
# containing path, size, and sha256 — avoids multi-GB duplication.
_LARGE_CSV_MAX_BYTES: int = 50 * 1024 * 1024  # 50 MB

# Extensions subject to the size cap.
_LARGE_FILE_EXTENSIONS: set = {".csv", ".parquet", ".xlsx", ".xls", ".pkl", ".pickle", ".h5", ".hdf5"}

_RUN_DIRS: Dict[str, str] = {}
_RUN_ATTEMPTS: Dict[str, List[Dict[str, Any]]] = {}
_TEE_STREAM = None


class _TeeStream:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> None:
        for stream in self._streams:
            try:
                stream.write(data)
            except Exception:
                pass

    def flush(self) -> None:
        for stream in self._streams:
            try:
                stream.flush()
            except Exception:
                pass


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_text(path: str, text: str) -> None:
    _ensure_dir(os.path.dirname(path))
    clean_text = sanitize_text(str(text or ""))
    with open(path, "w", encoding="utf-8") as f:
        f.write(clean_text)


def _write_json(path: str, payload: Any) -> None:
    _ensure_dir(os.path.dirname(path))
    clean_payload = sanitize_text_payload(payload)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean_payload, f, indent=2, ensure_ascii=False)


def _safe_trace_fragment(value: Any, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "")).strip("._")
    if not text:
        text = fallback
    return text[:80]


def _safe_load_json(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _count_jsonl_rows(path: str) -> int:
    if not path or not os.path.exists(path):
        return 0
    count = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
    except Exception:
        return 0
    return count


def _hash_file(path: str) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        return out or None
    except Exception:
        return None


def _normalize_required_outputs(contract: Dict[str, Any]) -> List[str]:
    """V4.1: Use get_required_outputs accessor instead of spec_extraction."""
    if not isinstance(contract, dict):
        return []
    return get_required_outputs(contract)


def _compact_metric_round_entry(entry: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(entry, dict):
        return None
    try:
        round_id = int(entry.get("round_id"))
    except Exception:
        round_id = 0
    if round_id <= 0:
        return None
    hypothesis = entry.get("hypothesis") if isinstance(entry.get("hypothesis"), dict) else {}
    return {
        "round_id": round_id,
        "delta": entry.get("delta"),
        "kept": entry.get("kept"),
        "reason": entry.get("reason") or entry.get("forced_finalize_reason") or "",
        "hypothesis": {
            "action": hypothesis.get("action") if isinstance(hypothesis, dict) else "",
            "technique": hypothesis.get("technique") if isinstance(hypothesis, dict) else "",
            "signature": hypothesis.get("signature") if isinstance(hypothesis, dict) else "",
        },
    }


def _compact_metric_rounds(state: Dict[str, Any], trace_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    rounds: List[Dict[str, Any]] = []
    round_history = state.get("ml_improvement_round_history")
    if isinstance(round_history, list):
        for item in round_history:
            compact = _compact_metric_round_entry(item)
            if compact:
                rounds.append(compact)
    if rounds:
        return rounds[-24:]
    summary_rounds = trace_summary.get("metric_rounds") if isinstance(trace_summary, dict) else []
    if isinstance(summary_rounds, list):
        for item in summary_rounds:
            if isinstance(item, dict):
                rounds.append(
                    {
                        "round_id": item.get("round_id"),
                        "delta": item.get("delta"),
                        "kept": item.get("kept"),
                        "reason": item.get("reason"),
                        "hypothesis": {
                            "action": item.get("action"),
                            "technique": item.get("technique"),
                            "signature": item.get("signature"),
                        },
                    }
                )
    return rounds[-24:]


def _scan_run_outputs(run_dir: str) -> List[str]:
    produced: List[str] = []
    if not run_dir:
        return produced
    artifacts_dir = os.path.join(run_dir, "artifacts")
    report_dir = os.path.join(run_dir, "report")
    if os.path.isdir(artifacts_dir):
        for root, _, files in os.walk(artifacts_dir):
            for name in files:
                path = os.path.join(root, name)
                rel = os.path.relpath(path, artifacts_dir)
                produced.append(rel.replace("\\", "/"))
    if os.path.isdir(report_dir):
        for root, _, files in os.walk(report_dir):
            for name in files:
                path = os.path.join(root, name)
                rel = os.path.relpath(path, run_dir)
                produced.append(rel.replace("\\", "/"))
    return produced


def _normalize_rel_path(path: str) -> str:
    return os.path.normpath(path).replace("\\", "/").lstrip("./")


def _normalize_exclude_prefixes(exclude_prefixes: Optional[List[str]]) -> List[str]:
    prefixes = []
    for prefix in exclude_prefixes or []:
        if not prefix:
            continue
        normalized = _normalize_rel_path(prefix)
        if normalized:
            prefixes.append(normalized.rstrip("/"))
    return prefixes


def _should_copy_file(
    path: str,
    rel_path: str,
    since_epoch: Optional[float],
    exclude_prefixes: Optional[List[str]],
) -> bool:
    rel_norm = _normalize_rel_path(rel_path)
    # Merge caller-provided + global exclude prefixes.
    all_prefixes = _normalize_exclude_prefixes(
        (exclude_prefixes or []) + _BUNDLE_EXCLUDE_PREFIXES
    )
    for prefix in all_prefixes:
        if rel_norm == prefix or rel_norm.startswith(prefix + "/"):
            return False
    if since_epoch is not None:
        try:
            if os.path.getmtime(path) < float(since_epoch):
                return False
        except Exception:
            return False
    return True


def _is_oversized_data_file(path: str) -> bool:
    """Return True if path is a data file exceeding _LARGE_CSV_MAX_BYTES."""
    ext = os.path.splitext(path)[1].lower()
    if ext not in _LARGE_FILE_EXTENSIONS:
        return False
    try:
        return os.path.getsize(path) > _LARGE_CSV_MAX_BYTES
    except Exception:
        return False


def _write_stub_meta(src_path: str, dest_path: str) -> None:
    """Write a lightweight .meta.json instead of copying the large file."""
    meta = {
        "_stub": True,
        "original_path": os.path.abspath(src_path),
        "size_bytes": os.path.getsize(src_path),
        "size_mb": round(os.path.getsize(src_path) / (1024 * 1024), 2),
        "sha256": _hash_file(src_path),
        "note": "File too large for run_bundle. See original_path for the actual file.",
    }
    meta_dest = dest_path + ".meta.json"
    _ensure_dir(os.path.dirname(meta_dest))
    with open(meta_dest, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def init_run_bundle(
    run_id: str,
    state: Optional[Dict[str, Any]] = None,
    base_dir: str = RUNS_DIR,
    enable_tee: bool = True,
    run_dir: Optional[str] = None,
) -> str:
    run_dir = run_dir or os.path.join(base_dir, run_id)
    _ensure_dir(run_dir)
    for sub in ["contracts", "agents", "sandbox", "artifacts", "report"]:
        _ensure_dir(os.path.join(run_dir, sub))
    register_run_log(run_id, os.path.join(run_dir, "events.jsonl"))
    _RUN_DIRS[run_id] = run_dir
    _RUN_ATTEMPTS.setdefault(run_id, [])

    if enable_tee:
        app_log_path = os.path.join(run_dir, "app_log.txt")
        _ensure_dir(os.path.dirname(app_log_path))
        try:
            log_handle = open(app_log_path, "a", encoding="utf-8")
            global _TEE_STREAM
            if _TEE_STREAM is None:
                _TEE_STREAM = _TeeStream(sys.stdout, log_handle)
                sys.stdout = _TEE_STREAM
                sys.stderr = _TEE_STREAM
        except Exception:
            pass
    if state is not None:
        state["run_bundle_dir"] = run_dir
    return run_dir


def get_run_dir(run_id: str) -> Optional[str]:
    return _RUN_DIRS.get(run_id)


def log_agent_snapshot(
    run_id: str,
    agent: str,
    prompt: Optional[str] = None,
    response: Optional[Any] = None,
    context: Optional[Dict[str, Any]] = None,
    script: Optional[str] = None,
    verdicts: Optional[Any] = None,
    attempt: Optional[int] = None,
    iteration: Optional[int] = None,
    prompt_trace: Optional[List[Dict[str, Any]]] = None,
) -> None:
    run_dir = get_run_dir(run_id)
    if not run_dir:
        return
    if prompt:
        try:
            size_path = os.path.join(run_dir, "work", "data", "prompt_sizes.json")
            _ensure_dir(os.path.dirname(size_path))
            existing: Dict[str, Any] = {}
            if os.path.exists(size_path):
                with open(size_path, "r", encoding="utf-8") as handle:
                    existing = json.load(handle) if handle.readable() else {}
            if not isinstance(existing, dict):
                existing = {}
            agents = existing.get("agents") if isinstance(existing.get("agents"), dict) else {}
            entries = existing.get("entries") if isinstance(existing.get("entries"), list) else []
            prompt_len = len(str(prompt))
            agents.setdefault(agent, []).append(prompt_len)
            entries.append({"agent": agent, "chars": prompt_len})
            existing["agents"] = agents
            existing["entries"] = entries[-200:]
            with open(size_path, "w", encoding="utf-8") as handle:
                json.dump(existing, handle, indent=2, ensure_ascii=False)
        except Exception:
            pass
    base = os.path.join(run_dir, "agents", agent)
    if iteration is not None and attempt is not None:
        base = os.path.join(base, f"iteration_{iteration}", f"attempt_{attempt}")
    elif iteration is not None:
        base = os.path.join(base, f"iteration_{iteration}")
    elif attempt is not None:
        # Backwards-compatible path layout for callers that only pass attempt.
        base = os.path.join(base, f"iteration_{attempt}")
    _ensure_dir(base)
    if prompt:
        _write_text(os.path.join(base, "prompt.txt"), str(prompt))
    if response is not None:
        if isinstance(response, (dict, list)):
            _write_json(os.path.join(base, "response.json"), response)
        else:
            _write_text(os.path.join(base, "response.txt"), str(response))
    if context is not None:
        try:
            safe_context = compress_long_lists(context)[0]
        except Exception:
            safe_context = context
        _write_json(os.path.join(base, "context.json"), safe_context)
    if script:
        _write_text(os.path.join(base, "script.py"), script)
    if verdicts is not None:
        _write_json(os.path.join(base, "verdicts.json"), verdicts)
    if isinstance(prompt_trace, list) and prompt_trace:
        subcalls_dir = os.path.join(base, "subcalls")
        _ensure_dir(subcalls_dir)
        subcall_index: List[Dict[str, Any]] = []
        for idx, item in enumerate(prompt_trace, start=1):
            if not isinstance(item, dict):
                continue
            clean_item = sanitize_text_payload(dict(item))
            stage = _safe_trace_fragment(clean_item.get("stage"), f"subcall_{idx:02d}")
            stem = f"{idx:02d}_{stage}"
            prompt_body = clean_item.pop("prompt", None)
            response_body = clean_item.pop("response", None)
            if prompt_body is not None:
                prompt_rel = os.path.join("subcalls", f"{stem}_prompt.txt").replace("\\", "/")
                _write_text(os.path.join(base, prompt_rel), str(prompt_body))
                clean_item["prompt_path"] = prompt_rel
                clean_item["prompt_chars"] = len(str(prompt_body))
            if response_body is not None:
                response_rel = os.path.join("subcalls", f"{stem}_response.txt").replace("\\", "/")
                _write_text(os.path.join(base, response_rel), str(response_body))
                clean_item["response_path"] = response_rel
                clean_item["response_chars"] = len(str(response_body))
            clean_item.setdefault("index", idx)
            clean_item.setdefault("stage", str(item.get("stage") or stage))
            subcall_index.append(clean_item)
        if subcall_index:
            _write_json(os.path.join(base, "subcalls.json"), subcall_index)


def log_sandbox_attempt(
    run_id: str,
    step: str,
    attempt: int,
    code: str,
    stdout: str,
    stderr: str,
    outputs_listing: Any,
    downloaded_paths: Optional[List[str]] = None,
    exit_code: Optional[int] = None,
    error_tail: Optional[str] = None,
    success: Optional[bool] = None,
    stage: Optional[str] = None,
    exception_type: Optional[str] = None,
    exception_msg: Optional[str] = None,
) -> None:
    run_dir = get_run_dir(run_id)
    if not run_dir:
        return
    safe_step = step or "unknown"
    attempt_dir = os.path.join(run_dir, "sandbox", safe_step, f"attempt_{attempt}")
    _ensure_dir(attempt_dir)
    _write_text(os.path.join(attempt_dir, "code_sent.py"), code or "")
    _write_text(os.path.join(attempt_dir, "stdout.txt"), stdout or "")
    _write_text(os.path.join(attempt_dir, "stderr.txt"), stderr or "")
    if outputs_listing is not None:
        _write_json(os.path.join(attempt_dir, "outputs_listing.json"), outputs_listing)
    if downloaded_paths:
        dest_root = os.path.join(attempt_dir, "downloaded_artifacts")
        for src in downloaded_paths:
            if not src or not os.path.exists(src):
                continue
            rel = src if not os.path.isabs(src) else os.path.basename(src)
            dest = os.path.join(dest_root, rel)
            _ensure_dir(os.path.dirname(dest))
            try:
                shutil.copy2(src, dest)
            except Exception:
                pass
    record = {
        "step": safe_step,
        "attempt": attempt,
        "exit_code": exit_code,
        "error_tail": error_tail,
    }
    if success is not None:
        record["success"] = bool(success)
    if stage:
        record["stage"] = stage
    if exception_type:
        record["exception_type"] = exception_type
    if exception_msg:
        record["exception_msg"] = exception_msg
    _RUN_ATTEMPTS.setdefault(run_id, []).append(record)


def update_sandbox_attempt(run_id: str, step: str, attempt: int, **updates: Any) -> None:
    if not run_id or run_id not in _RUN_ATTEMPTS:
        return
    safe_step = step or "unknown"
    for record in reversed(_RUN_ATTEMPTS.get(run_id, [])):
        if record.get("step") == safe_step and record.get("attempt") == attempt:
            record.update({k: v for k, v in updates.items() if v is not None})
            break


def copy_run_artifacts(
    run_id: str,
    sources: List[str],
    since_epoch: Optional[float] = None,
    exclude_prefixes: Optional[List[str]] = None,
) -> None:
    run_dir = get_run_dir(run_id)
    if not run_dir:
        return
    dest_root = os.path.join(run_dir, "artifacts")
    _ensure_dir(dest_root)
    seen_hashes: Dict[str, str] = {}  # sha256 -> first dest path (dedup)
    for src in sources:
        if not src or not os.path.exists(src):
            continue
        try:
            if os.path.isdir(src):
                base_name = os.path.basename(src.rstrip("/\\"))
                for root, dirs, files in os.walk(src):
                    # Prune excluded dirs in-place to avoid walking into them
                    dirs[:] = [
                        d for d in dirs
                        if d not in {"runs", "__pycache__", ".git", ".venv", "node_modules"}
                    ]
                    for name in files:
                        path = os.path.join(root, name)
                        rel = os.path.relpath(path, src)
                        rel_path = os.path.join(base_name, rel)
                        if not _should_copy_file(path, rel_path, since_epoch, exclude_prefixes):
                            continue
                        dest = os.path.join(dest_root, os.path.normpath(rel_path))
                        # Large data files → write stub instead of copying
                        if _is_oversized_data_file(path):
                            _write_stub_meta(path, dest)
                            continue
                        _ensure_dir(os.path.dirname(dest))
                        shutil.copy2(path, dest)
            else:
                rel = src if not os.path.isabs(src) else os.path.basename(src)
                if not _should_copy_file(src, rel, since_epoch, exclude_prefixes):
                    continue
                dest = os.path.join(dest_root, os.path.normpath(rel))
                if _is_oversized_data_file(src):
                    _write_stub_meta(src, dest)
                    continue
                _ensure_dir(os.path.dirname(dest))
                shutil.copy2(src, dest)
        except Exception:
            pass


def copy_run_contracts(run_id: str, sources: List[str]) -> None:
    run_dir = get_run_dir(run_id)
    if not run_dir:
        return
    dest_root = os.path.join(run_dir, "contracts")
    _ensure_dir(dest_root)
    for src in sources:
        if not src or not os.path.exists(src):
            continue
        dest = os.path.join(dest_root, os.path.basename(src))
        try:
            shutil.copy2(src, dest)
        except Exception:
            pass


def copy_run_reports(
    run_id: str,
    sources: List[str],
    since_epoch: Optional[float] = None,
    exclude_prefixes: Optional[List[str]] = None,
) -> None:
    run_dir = get_run_dir(run_id)
    if not run_dir:
        return
    dest_root = os.path.join(run_dir, "report")
    _ensure_dir(dest_root)
    for src in sources:
        if not src or not os.path.exists(src):
            continue
        try:
            if os.path.isdir(src):
                base_name = os.path.basename(src.rstrip("/\\"))
                for root, dirs, files in os.walk(src):
                    dirs[:] = [
                        d for d in dirs
                        if d not in {"runs", "__pycache__", ".git", ".venv", "node_modules"}
                    ]
                    for name in files:
                        path = os.path.join(root, name)
                        rel = os.path.relpath(path, src)
                        rel_path = os.path.join(base_name, rel)
                        if not _should_copy_file(path, rel_path, since_epoch, exclude_prefixes):
                            continue
                        dest = os.path.join(dest_root, os.path.normpath(rel_path))
                        if _is_oversized_data_file(path):
                            _write_stub_meta(path, dest)
                            continue
                        _ensure_dir(os.path.dirname(dest))
                        shutil.copy2(path, dest)
            else:
                rel = src if not os.path.isabs(src) else os.path.basename(src)
                if not _should_copy_file(src, rel, since_epoch, exclude_prefixes):
                    continue
                dest = os.path.join(dest_root, os.path.normpath(rel))
                if _is_oversized_data_file(src):
                    _write_stub_meta(src, dest)
                    continue
                _ensure_dir(os.path.dirname(dest))
                shutil.copy2(src, dest)
        except Exception:
            pass


def write_run_manifest(
    run_id: str,
    state: Dict[str, Any],
    status_final: Optional[str] = None,
    started_at: Optional[str] = None,
    ended_at: Optional[str] = None,
) -> Optional[str]:
    run_dir = get_run_dir(run_id) or os.path.join(RUNS_DIR, run_id)
    csv_path = state.get("csv_path") or ""
    contracts_dir = os.path.join(run_dir, "contracts")
    work_dir = state.get("work_dir_abs") or state.get("work_dir") or ""
    work_dir_abs = os.path.abspath(work_dir) if work_dir else ""
    work_contract_path = os.path.join(work_dir_abs, "data", "execution_contract.json") if work_dir_abs else ""
    work_eval_path = os.path.join(work_dir_abs, "data", "evaluation_spec.json") if work_dir_abs else ""
    work_contract = _safe_load_json(work_contract_path) if work_contract_path else None
    work_eval = _safe_load_json(work_eval_path) if work_eval_path else None
    contract = work_contract or _safe_load_json(os.path.join(contracts_dir, "execution_contract.json")) or state.get("execution_contract") or {}
    evaluation_spec = work_eval or _safe_load_json(os.path.join(contracts_dir, "evaluation_spec.json")) or state.get("evaluation_spec") or {}
    artifact_index = (
        _safe_load_json(os.path.join(run_dir, "artifacts", "data", "produced_artifact_index.json"))
        or state.get("produced_artifact_index")
        or state.get("artifact_index")
        or []
    )
    output_contract = _safe_load_json(os.path.join(run_dir, "report", "output_contract_report.json"))
    if not output_contract:
        output_contract = _safe_load_json(os.path.join(run_dir, "artifacts", "data", "output_contract_report.json")) or {}
    run_summary = _safe_load_json(os.path.join(run_dir, "report", "run_summary.json"))
    if not run_summary:
        run_summary = _safe_load_json(os.path.join(run_dir, "artifacts", "data", "run_summary.json")) or {}
    required_outputs = _normalize_required_outputs(contract)
    produced_outputs = sorted(set(_scan_run_outputs(run_dir)))
    trace_summary_path = os.path.join(run_dir, "report", "governance", "ml_iteration_trace_summary.json")
    trace_summary = _safe_load_json(trace_summary_path)
    if not isinstance(trace_summary, dict):
        trace_summary = {}
    trace_journal_path = os.path.join(run_dir, "report", "governance", "ml_iteration_journal.jsonl")
    entries_count = trace_summary.get("entries_count")
    try:
        entries_count = int(entries_count)
    except Exception:
        entries_count = _count_jsonl_rows(trace_journal_path)
    metric_rounds = _compact_metric_rounds(state, trace_summary)
    iteration_trace = {
        "journal_exists": os.path.exists(trace_journal_path),
        "journal_relative_path": "report/governance/ml_iteration_journal.jsonl",
        "entries_count": int(entries_count),
        "summary_exists": bool(trace_summary),
        "summary_relative_path": "report/governance/ml_iteration_trace_summary.json" if trace_summary else None,
        "stages_count": trace_summary.get("stages_count", {}) if isinstance(trace_summary.get("stages_count"), dict) else {},
        "last_entry": trace_summary.get("last_entry", {}) if isinstance(trace_summary.get("last_entry"), dict) else {},
        "metric_improvement_round_count": int(state.get("ml_improvement_round_count", 0) or 0),
        "metric_improvement_attempted": bool(state.get("ml_improvement_attempted")),
        "metric_improvement_kept": state.get("ml_improvement_kept"),
        "metric_rounds_count": len(metric_rounds),
        "metric_rounds": metric_rounds,
        "metric_round_last": metric_rounds[-1] if metric_rounds else {},
    }

    manifest_path = os.path.join(run_dir, "run_manifest.json")
    existing = _safe_load_json(manifest_path)
    existing_dict = existing if isinstance(existing, dict) else {}

    raw_status = state.get("review_verdict_normalized") or run_summary.get("status") or state.get("review_verdict")
    normalized_status = normalize_review_status(raw_status)
    normalized_reason = state.get("review_feedback_normalized") or (state.get("last_gate_context") or {}).get("feedback")
    gates_summary = {
        "status": normalized_status,
        "failed_gates": run_summary.get("failed_gates", []) if isinstance(run_summary, dict) else [],
        "reason": normalized_reason,
    }

    manifest = dict(existing_dict)
    existing_input = existing_dict.get("input")
    if not csv_path and isinstance(existing_input, dict):
        csv_path = existing_input.get("path") or ""
    manifest.update(
        {
            "run_id": run_id,
            "started_at": started_at or existing_dict.get("started_at") or state.get("run_start_ts"),
            "ended_at": ended_at or datetime.utcnow().isoformat(),
            "git_commit": _git_commit(),
            "input": {
                "path": csv_path,
                "sha256": _hash_file(csv_path) or (existing_input or {}).get("sha256"),
                "dialect": {
                    "encoding": state.get("csv_encoding") or (existing_input or {}).get("dialect", {}).get("encoding"),
                    "sep": state.get("csv_sep") or (existing_input or {}).get("dialect", {}).get("sep"),
                    "decimal": state.get("csv_decimal") or (existing_input or {}).get("dialect", {}).get("decimal"),
                },
            },
            "models_by_agent": state.get("agent_models", {}) or existing_dict.get("models_by_agent", {}),
            "required_outputs": required_outputs,
            "produced_outputs": produced_outputs,
            "sandbox_attempts": _RUN_ATTEMPTS.get(run_id, []),
            "required_outputs_missing": output_contract.get("missing", []),
            "status_final": status_final or existing_dict.get("status_final") or gates_summary.get("status"),
            "gates_summary": gates_summary,
            "iteration_trace": iteration_trace,
            "contracts": {
                "execution_contract": bool(work_contract) or os.path.exists(os.path.join(contracts_dir, "execution_contract.json")),
                "evaluation_spec": bool(work_eval) or os.path.exists(os.path.join(contracts_dir, "evaluation_spec.json")),
                "artifact_index": os.path.exists(os.path.join(contracts_dir, "artifact_index.json")),
                "contract_min": False,
            },
        }
    )
    _write_json(manifest_path, manifest)
    return manifest_path
