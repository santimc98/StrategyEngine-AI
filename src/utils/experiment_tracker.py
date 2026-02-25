from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _tracker_path(run_id: str, base_dir: str = "runs") -> Path:
    token = str(run_id or "").strip()
    if not token:
        raise ValueError("run_id is required")
    base_path = Path(base_dir)
    if not base_path.is_absolute():
        base_path = _project_root() / base_path
    return base_path / token / "work" / "memory" / "experiment_tracker.jsonl"


def build_hypothesis_signature(
    *,
    technique: Any,
    target_columns: Any,
    feature_scope: Any,
    params: Any,
) -> str:
    tech = str(technique or "").strip().lower() or "unknown_technique"
    scope = str(feature_scope or "").strip().lower() or "model_features"
    if isinstance(target_columns, list):
        cols = sorted(
            [
                str(item).strip()
                for item in target_columns
                if str(item or "").strip()
            ]
        )
    else:
        col = str(target_columns or "").strip()
        cols = [col] if col else []
    params_payload = params if isinstance(params, dict) else {}
    params_json = json.dumps(params_payload, ensure_ascii=True, sort_keys=True)
    signature_base = "technique={};scope={};cols={};params={}".format(
        tech,
        scope,
        ",".join(cols),
        params_json,
    )
    digest = hashlib.sha1(signature_base.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return "hyp_" + digest


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "on"}:
            return True
        if token in {"0", "false", "no", "off"}:
            return False
    return bool(default)


def normalize_experiment_entry(entry_dict: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(entry_dict or {})
    payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())

    technique = str(payload.get("technique") or "").strip()
    signature = str(payload.get("signature") or "").strip()
    if not signature and technique:
        signature = build_hypothesis_signature(
            technique=technique,
            target_columns=payload.get("target_columns") or [],
            feature_scope=payload.get("feature_scope") or "model_features",
            params=payload.get("params") if isinstance(payload.get("params"), dict) else {},
        )
    if signature:
        payload["signature"] = signature

    delta = _as_float(payload.get("delta"))
    if delta is not None:
        payload["delta"] = float(delta)

    metric_value = _as_float(payload.get("metric_value"))
    if metric_value is not None:
        payload["metric_value"] = float(metric_value)

    stability_ok = payload.get("stability_ok")
    if stability_ok is not None:
        payload["stability_ok"] = _as_bool(stability_ok, default=True)

    cv_std = _as_float(payload.get("cv_std"))
    if cv_std is not None:
        payload["cv_std"] = abs(float(cv_std))

    generalization_gap = _as_float(payload.get("generalization_gap_abs"))
    if generalization_gap is None:
        generalization_gap = _as_float(payload.get("generalization_gap"))
    if generalization_gap is not None:
        payload["generalization_gap_abs"] = abs(float(generalization_gap))

    cost = _as_float(payload.get("cost"))
    if cost is None:
        token_cost = _as_float(payload.get("token_cost"))
        runtime_cost = _as_float(payload.get("runtime_cost"))
        pieces = [value for value in (token_cost, runtime_cost) if value is not None]
        if pieces:
            cost = sum(pieces)
    if cost is not None:
        payload["cost"] = max(0.0, float(cost))

    return payload


def append_hypothesis_memory(
    run_id: str,
    *,
    round_id: int,
    technique: str,
    target_columns: List[str] | None = None,
    feature_scope: str = "model_features",
    params: Dict[str, Any] | None = None,
    metric_name: str | None = None,
    metric_value: float | None = None,
    delta: float | None = None,
    stability_ok: bool | None = None,
    cost: float | None = None,
    status: str | None = None,
    phase: str | None = None,
    extra: Dict[str, Any] | None = None,
    base_dir: str = "runs",
) -> str | None:
    payload: Dict[str, Any] = {
        "event": "hypothesis_memory",
        "round_id": int(round_id or 0),
        "technique": str(technique or "").strip(),
        "target_columns": target_columns if isinstance(target_columns, list) else [],
        "feature_scope": str(feature_scope or "model_features"),
        "params": params if isinstance(params, dict) else {},
        "metric_name": str(metric_name or "").strip(),
        "metric_value": metric_value,
        "delta": delta,
        "stability_ok": stability_ok,
        "cost": cost,
        "status": str(status or "").strip(),
        "phase": str(phase or "").strip(),
    }
    if isinstance(extra, dict):
        payload.update(extra)
    return append_experiment_entry(run_id, payload, base_dir=base_dir)


def append_experiment_entry(run_id: str, entry_dict: Dict[str, Any], base_dir: str = "runs") -> str | None:
    if not run_id or not isinstance(entry_dict, dict):
        return None
    try:
        path = _tracker_path(run_id, base_dir=base_dir)
    except Exception:
        return None
    payload = normalize_experiment_entry(entry_dict)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return str(path)
    except Exception:
        return None


def load_recent_experiment_entries(run_id: str, k: int = 20, base_dir: str = "runs") -> List[Dict[str, Any]]:
    if not run_id:
        return []
    try:
        path = _tracker_path(run_id, base_dir=base_dir)
    except Exception:
        return []
    if not path.exists():
        return []
    entries: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    payload = json.loads(raw)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    entries.append(normalize_experiment_entry(payload))
    except Exception:
        return []
    limit = int(k or 0)
    if limit <= 0:
        return []
    return entries[-limit:]


def extract_metric_trend(
    entries: List[Dict[str, Any]],
    *,
    metric_name: str = "",
) -> List[Dict[str, Any]]:
    """
    Extract ordered metric values from experiment tracker entries.
    Returns list of {round, metric_value, delta, technique} for trazability.
    """
    trend: List[Dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        event = str(entry.get("event") or "").strip().lower()
        if event not in {"candidate_evaluated", "hypothesis_memory"}:
            continue
        metric_value = _as_float(entry.get("metric_value"))
        delta = _as_float(entry.get("delta"))
        if metric_value is None:
            continue
        trend.append({
            "round": entry.get("round_id") or entry.get("round"),
            "metric_value": metric_value,
            "delta": delta,
            "technique": str(entry.get("technique") or "").strip(),
        })
    return trend
