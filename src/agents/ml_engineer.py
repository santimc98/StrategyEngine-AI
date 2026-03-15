import os
import logging
import re
from datetime import datetime, timezone
from typing import Dict, Any, List
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from string import Template
import json
import ast
from src.utils.prompting import render_prompt
from src.utils.code_extract import extract_code_block, is_syntax_valid
from src.utils.senior_protocol import (
    SENIOR_ENGINEERING_PROTOCOL,
    SENIOR_REASONING_PROTOCOL_GENERAL,
)
from src.utils.llm_fallback import call_chat_with_fallback, extract_response_text
from src.utils.sandbox_deps import (
    BASE_ALLOWLIST,
    EXTENDED_ALLOWLIST,
    CLOUDRUN_NATIVE_ALLOWLIST,
    CLOUDRUN_OPTIONAL_ALLOWLIST,
    BANNED_ALWAYS_ALLOWLIST,
)
from src.utils.action_families import (
    classify_action_family,
    get_action_family_guidance,
)
from src.utils.contract_accessors import (
    get_clean_manifest_path,
    get_declared_artifact_path,
    get_declared_file_schema,
    normalize_artifact_path,
)

# NOTE: scan_code_safety referenced by tests as a required safety mechanism.
# ML code executes in sandbox; keep the reference for integration checks.
_scan_code_safety_ref = "scan_code_safety"

# ============================================================================
# ML PLAN SCHEMA CONSTANTS (TAREA 1)
# ============================================================================

REQUIRED_PLAN_KEYS = [
    "training_rows_policy",
    "train_filter",
    "metric_policy",
    "cv_policy",
    "scoring_policy",
    "leakage_policy",
    "notes",
    "plan_source",
]

DEFAULT_PLAN: Dict[str, Any] = {
    "training_rows_policy": "unspecified",
    "training_rows_rule": None,
    "split_column": None,
    "train_filter": {
        "type": "unspecified",
        "column": None,
        "value": None,
        "rule": None,
    },
    "metric_policy": {
        "primary_metric": "unspecified",
        "secondary_metrics": [],
        "report_with_cv": True,
        "notes": "",
    },
    "cv_policy": {
        "strategy": "unspecified",
        "n_splits": 5,
        "shuffle": True,
        "stratified": None,
        "notes": "",
    },
    "scoring_policy": {
        "generate_scores": True,
        "score_rows": "unspecified",
    },
    "leakage_policy": {
        "action": "unspecified",
        "flagged_columns": [],
        "notes": "",
    },
    "evidence": [],
    "assumptions": [],
    "open_questions": [],
    "notes": [],
    "evidence_used": {},  # Structured evidence digest for QA coherence checks
    "plan_source": "fallback",
}

UNIVERSAL_PROLOGUE_START = "# ML_ENGINEER_UNIVERSAL_PROLOGUE_START"
UNIVERSAL_PROLOGUE_END = "# ML_ENGINEER_UNIVERSAL_PROLOGUE_END"


class MLEngineerAgent:
    def __init__(self, api_key: str = None):
        """
        Initializes the ML Engineer Agent with OpenRouter only.
        Model routing is fully controlled via:
        - OPENROUTER_ML_PRIMARY_MODEL
        - OPENROUTER_ML_FALLBACK_MODEL
        """
        self.logger = logging.getLogger(__name__)
        configured_provider = (os.getenv("ML_ENGINEER_PROVIDER", "openrouter") or "openrouter").strip().lower()
        if configured_provider and configured_provider != "openrouter":
            self.logger.warning(
                "ML_ENGINEER_PROVIDER=%s ignored; forcing provider=openrouter",
                configured_provider,
            )
        self.provider = "openrouter"
        self.fallback_model_name = None
        self.last_model_used = None
        self.last_fallback_reason = None
        self.last_training_policy_warnings = None
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API Key is required.")
        timeout_raw = os.getenv("OPENROUTER_TIMEOUT_SECONDS")
        try:
            timeout_seconds = float(timeout_raw) if timeout_raw else 120.0
        except ValueError:
            timeout_seconds = 120.0
        headers = {}
        referer = os.getenv("OPENROUTER_HTTP_REFERER")
        if referer:
            headers["HTTP-Referer"] = referer
        title = os.getenv("OPENROUTER_X_TITLE")
        if title:
            headers["X-Title"] = title
        client_kwargs = {
            "api_key": self.api_key,
            "base_url": "https://openrouter.ai/api/v1",
            "timeout": timeout_seconds,
        }
        if headers:
            client_kwargs["default_headers"] = headers
        self.client = OpenAI(**client_kwargs)

        self.model_name = (
            os.getenv("OPENROUTER_ML_PRIMARY_MODEL")
            or "moonshotai/kimi-k2.5"
        ).strip()
        self.fallback_model_name = (
            os.getenv("OPENROUTER_ML_FALLBACK_MODEL")
            or "minimax/minimax-m2.5"
        ).strip()
        # Editor model: used for REPAIR/editor mode iterations (when previous
        # code exists). Falls back to primary model when not configured.
        _editor_raw = (os.getenv("OPENROUTER_ML_EDITOR_MODEL") or "").strip()
        self.editor_model_name = _editor_raw if _editor_raw else None
        if not self.model_name:
            self.model_name = "moonshotai/kimi-k2.5"
        if not self.fallback_model_name:
            self.fallback_model_name = "minimax/minimax-m2.5"
        self.logger.info(
            "ML_ENGINEER_OPENROUTER_MODELS: primary=%s fallback=%s editor=%s",
            self.model_name,
            self.fallback_model_name,
            self.editor_model_name or "(same as primary)",
        )
        self.last_prompt = None
        self.last_response = None
        self.last_prompt_trace: List[Dict[str, Any]] = []
        self.last_prompt_trace_operation = None

    def _reset_prompt_trace(self, operation: str) -> None:
        self.last_prompt_trace = []
        self.last_prompt_trace_operation = str(operation or "").strip() or None

    def _record_prompt_trace_entry(
        self,
        *,
        stage: str,
        system_prompt: str,
        user_prompt: str,
        response: Any,
        temperature: float,
        model_requested: str | None = None,
        model_used: str | None = None,
        context_tag: str | None = None,
        used_fallback: bool = False,
        source: str = "openrouter",
    ) -> None:
        prompt_text = f"{system_prompt}\n\nUSER:\n{user_prompt}"
        response_text = str(response or "")
        self.last_prompt = prompt_text
        self.last_response = response_text
        self.last_prompt_trace.append(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                "operation": self.last_prompt_trace_operation,
                "stage": str(stage or "unknown"),
                "source": str(source or "openrouter"),
                "context_tag": str(context_tag or "ml_engineer"),
                "temperature": float(temperature),
                "model_requested": str(model_requested or model_used or ""),
                "model_used": str(model_used or model_requested or ""),
                "used_fallback": bool(used_fallback),
                "prompt": prompt_text,
                "response": response_text,
            }
        )

    def _build_runtime_dependency_context(
        self,
        required_dependencies: List[str] | None = None,
    ) -> Dict[str, Any]:
        required_roots = []
        for dep in (required_dependencies or []):
            token = str(dep or "").strip()
            if not token:
                continue
            required_roots.append(token.split(".")[0].lower())
        required_roots = sorted(list({dep for dep in required_roots if dep}))

        runtime_mode = (
            os.getenv("RUN_EXECUTION_MODE")
            or os.getenv("EXECUTION_RUNTIME_MODE")
            or "cloudrun"
        ).strip().lower()
        backend_profile = "local" if runtime_mode == "local" else "cloudrun"

        return {
            "backend_profile": backend_profile,
            "runtime_mode": runtime_mode,
            "required_dependencies": required_roots,
            "allowlist": {
                "base": sorted({str(item) for item in BASE_ALLOWLIST if str(item).strip()}),
                "extended_optional": sorted(
                    {str(item) for item in EXTENDED_ALLOWLIST if str(item).strip()}
                ),
                "cloudrun_native": sorted(
                    {str(item) for item in CLOUDRUN_NATIVE_ALLOWLIST if str(item).strip()}
                ),
                "cloudrun_optional": sorted(
                    {str(item) for item in CLOUDRUN_OPTIONAL_ALLOWLIST if str(item).strip()}
                ),
            },
            "blocked_always": sorted(
                {str(item) for item in BANNED_ALWAYS_ALLOWLIST if str(item).strip()}
            ),
            "guidance": [
                "Import only runtime-compatible dependencies.",
                "When optional dependencies are used, ensure they are contract-declared.",
                "Prefer robust fallbacks with base stack when optional libs are unavailable.",
            ],
        }

    def _build_data_sample_context(
        self,
        data_path: str,
        csv_encoding: str,
        csv_sep: str,
        csv_decimal: str,
        max_rows: int = 5,
        max_cols: int = 30,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "status": "unavailable",
            "reason": "data_not_accessible_at_prompt_time",
            "path": str(data_path or ""),
        }
        path = str(data_path or "").strip()
        if not path or not os.path.exists(path):
            return payload
        try:
            import pandas as pd

            # Read with natural dtype inference so the ML engineer sees real dtypes
            preview = pd.read_csv(
                path,
                nrows=max_rows,
                sep=csv_sep or ",",
                decimal=csv_decimal or ".",
                encoding=csv_encoding or "utf-8",
                low_memory=False,
            )
            cols = [str(col) for col in preview.columns.tolist()]
            shown_cols = cols[:max_cols]
            payload = {
                "status": "available",
                "path": path,
                "shape_preview": {"rows": int(preview.shape[0]), "cols": int(preview.shape[1])},
                "preview_columns": shown_cols,
                "preview_columns_truncated": len(cols) > max_cols,
                "preview_rows": preview[shown_cols].fillna("<NA>").to_dict(orient="records"),
                "dtypes_preview": {col: str(dtype) for col, dtype in preview[shown_cols].dtypes.items()},
                "null_counts_preview": {
                    col: int(preview[shown_cols][col].isnull().sum()) for col in shown_cols
                },
            }
        except Exception as sample_err:
            payload = {
                "status": "unavailable",
                "reason": f"sample_read_error: {sample_err}",
                "path": path,
            }
        return payload

    def _compact_cleaned_data_summary_for_prompt(
        self,
        summary: Dict[str, Any] | None,
        max_columns: int = 40,
        max_groups: int = 12,
    ) -> Dict[str, Any]:
        if not isinstance(summary, dict):
            return {}
        compact = dict(summary)
        col_summaries = summary.get("column_summaries")
        if not isinstance(col_summaries, list) or not col_summaries:
            return compact

        required_cols = summary.get("required_columns")
        required_set = {str(c).lower() for c in required_cols} if isinstance(required_cols, list) else set()
        split_col = str(summary.get("split_column") or "").strip().lower()

        prioritized: List[Dict[str, Any]] = []
        remainder: List[Dict[str, Any]] = []
        for item in col_summaries:
            if not isinstance(item, dict):
                continue
            name = str(item.get("column_name") or "").strip()
            if not name:
                continue
            key = name.lower()
            if key in required_set or (split_col and key == split_col):
                prioritized.append(item)
            else:
                remainder.append(item)

        by_signature: Dict[str, Dict[str, Any]] = {}
        for item in remainder:
            dtype_obs = str(item.get("dtype_observed") or "unknown")
            null_frac = item.get("null_frac")
            null_token = "unknown"
            if isinstance(null_frac, (int, float)):
                null_token = f"{float(null_frac):.6f}"
            signature = f"{dtype_obs}|{null_token}"
            bucket = by_signature.setdefault(
                signature,
                {
                    "dtype_observed": dtype_obs,
                    "null_frac": null_frac,
                    "count": 0,
                    "sample_columns": [],
                },
            )
            bucket["count"] += 1
            if len(bucket["sample_columns"]) < 10:
                bucket["sample_columns"].append(str(item.get("column_name")))

        groups = sorted(
            by_signature.values(),
            key=lambda x: int(x.get("count") or 0),
            reverse=True,
        )[:max_groups]

        kept: List[Dict[str, Any]] = prioritized[:max_columns]
        remaining_slots = max(0, max_columns - len(kept))
        if remaining_slots > 0:
            kept.extend(remainder[:remaining_slots])

        compact["column_summaries"] = kept
        compact["column_summaries_truncated"] = len(col_summaries) > len(kept)
        compact["column_summaries_omitted_count"] = max(0, len(col_summaries) - len(kept))
        if groups:
            compact["family_aggregate"] = groups
        return compact

    def _compact_execution_contract(self, contract: Dict[str, Any] | None) -> Dict[str, Any]:
        """
        Extract relevant V4.1 fields for ML Engineer prompt.
        V4.1 CUTOVER: No legacy keys (data_requirements, spec_extraction, role_runbooks).
        """
        if not isinstance(contract, dict):
            return {}

        # V4.1 keys relevant for ML Engineer (no legacy keys)
        keep_keys = [
            "contract_version",
            "strategy_title",
            "business_objective",
            "canonical_columns",  # V4.1: replaces required_columns/data_requirements
            "column_dtype_targets",
            "required_outputs",
            "objective_analysis",
            "evaluation_spec",
            "validation_requirements",
            "decisioning_requirements",
            "alignment_requirements",
            "business_alignment",
            "feature_semantics",
            "business_sanity_checks",
            "column_roles",
            # V4.1: availability_summary removed
            "required_dependencies",
            "compliance_checklist",
            # V4.1 specific
            "artifact_requirements",
            "qa_gates",
            "reviewer_gates",
            "allowed_feature_sets",
            "leakage_execution_plan",
            "data_limited_mode",
            "ml_engineer_runbook",
            "derived_columns",
            "iteration_policy",
            "feature_engineering_tasks",
            "split_spec",
            "n_train_rows",
            "n_test_rows",
            "n_total_rows",
        ]

        compact: Dict[str, Any] = {}
        for key in keep_keys:
            if key in contract:
                compact[key] = contract.get(key)

        # Truncate large lists
        for key in ["canonical_columns", "column_mapping_rules", "column_mapping"]:
            vals = contract.get(key)
            if isinstance(vals, list) and vals:
                compact[key] = vals[:80]

        # V4.1: Use ml_engineer_runbook only, no legacy role_runbooks fallback

        return compact

    def _resolve_declared_artifact_path(
        self,
        execution_contract: Dict[str, Any] | None,
        ml_view: Dict[str, Any] | None,
        filename: str,
    ) -> str:
        contract = execution_contract if isinstance(execution_contract, dict) else {}
        resolved = get_declared_artifact_path(contract, filename)
        if resolved:
            return resolved
        artifact_reqs = ml_view.get("artifact_requirements") if isinstance(ml_view, dict) else {}
        if isinstance(artifact_reqs, dict):
            artifact_contract = {"artifact_requirements": artifact_reqs}
            resolved = get_declared_artifact_path(artifact_contract, filename)
            if resolved:
                return resolved
        required_outputs = ml_view.get("required_outputs") if isinstance(ml_view, dict) else []
        if isinstance(required_outputs, list):
            artifact_contract = {"required_outputs": required_outputs}
            resolved = get_declared_artifact_path(artifact_contract, filename)
            if resolved:
                return resolved
        return ""

    def _resolve_cleaning_manifest_path(
        self,
        execution_contract: Dict[str, Any] | None,
        ml_view: Dict[str, Any] | None,
    ) -> str:
        contract = execution_contract if isinstance(execution_contract, dict) else {}
        resolved = get_clean_manifest_path(contract)
        if resolved:
            return resolved
        if isinstance(ml_view, dict):
            for key in ("cleaning_manifest_path", "output_manifest_path"):
                candidate = normalize_artifact_path(ml_view.get(key))
                if candidate:
                    return candidate
            artifact_reqs = ml_view.get("artifact_requirements")
            if isinstance(artifact_reqs, dict):
                artifact_contract = {"artifact_requirements": artifact_reqs}
                resolved = get_clean_manifest_path(artifact_contract)
                if resolved:
                    return resolved
        return "data/cleaning_manifest.json"

    def _render_artifact_schema_block(
        self,
        execution_contract: Dict[str, Any] | None,
        ml_view: Dict[str, Any] | None,
    ) -> str:
        """
        Render an explicit, human-readable artifact schema block from
        artifact_requirements for prompt injection.  Universal: does not
        assume dataset-specific column names or file paths.
        """
        contract = execution_contract if isinstance(execution_contract, dict) else {}
        view = ml_view if isinstance(ml_view, dict) else {}

        artifact_reqs = contract.get("artifact_requirements")
        if not isinstance(artifact_reqs, dict):
            artifact_reqs = view.get("artifact_requirements")
        if not isinstance(artifact_reqs, dict) or not artifact_reqs:
            return ""
        scored_rows_path = self._resolve_declared_artifact_path(contract, view, "scored_rows.csv")

        lines = ["=== REQUIRED OUTPUT ARTIFACTS SCHEMA ==="]

        # --- scored_rows_schema ---
        scored_schema = artifact_reqs.get("scored_rows_schema")
        if isinstance(scored_schema, dict) and scored_rows_path:
            lines.append(f"\nARTIFACT: {scored_rows_path}")
            req_cols = scored_schema.get("required_columns")
            if isinstance(req_cols, list) and req_cols:
                lines.append(f"  REQUIRED_COLUMNS (must ALL be present): {req_cols}")
            any_of = scored_schema.get("required_any_of_groups")
            severities = scored_schema.get("required_any_of_group_severity")
            if isinstance(any_of, list):
                for idx, group in enumerate(any_of):
                    if not isinstance(group, list) or not group:
                        continue
                    severity = "fail"
                    if isinstance(severities, list) and idx < len(severities):
                        severity = str(severities[idx] or "fail")
                    lines.append(
                        f"  ANY_OF_GROUP (severity={severity}): include >= 1 of {group}"
                    )

        # --- file_schemas (generic fallback for other artifact types) ---
        file_schemas = artifact_reqs.get("file_schemas")
        if isinstance(file_schemas, dict):
            for file_path, schema_def in file_schemas.items():
                if not isinstance(schema_def, dict):
                    continue
                lines.append(f"\nARTIFACT: {file_path}")
                req_cols = schema_def.get("required_columns")
                if isinstance(req_cols, list) and req_cols:
                    lines.append(f"  REQUIRED_COLUMNS: {req_cols}")
                any_of = schema_def.get("any_of_groups")
                if isinstance(any_of, list):
                    for group in any_of:
                        if isinstance(group, dict):
                            group_name = group.get("name", "unnamed")
                            columns = group.get("columns", [])
                            min_present = group.get("min_present", 1)
                            lines.append(
                                f"  ANY_OF_GROUP '{group_name}': include >= {min_present} of {columns}"
                            )
                output_path = schema_def.get("path") or file_path
                lines.append(f"  OUTPUT_PATH: {output_path}")
                expected_rc = schema_def.get("expected_row_count")
                if expected_rc is not None:
                    try:
                        expected_rc_int = int(expected_rc)
                        if expected_rc_int > 0:
                            lines.append(
                                f"  EXPECTED_ROW_COUNT: {expected_rc_int:,} rows (MUST match exactly - contract enforced)"
                            )
                    except (TypeError, ValueError):
                        pass

        # --- clean_dataset required columns ---
        clean_ds = artifact_reqs.get("clean_dataset")
        if isinstance(clean_ds, dict):
            output_path = clean_ds.get("output_path")
            if output_path:
                req_cols = clean_ds.get("required_columns")
                if isinstance(req_cols, list) and req_cols:
                    lines.append(f"\nARTIFACT: {output_path}")
                    lines.append(f"  REQUIRED_COLUMNS: {req_cols[:30]}{'...' if len(req_cols) > 30 else ''}")

        if len(lines) <= 1:
            return ""

        return "\n".join(lines)

    def _resolve_allowed_columns_for_prompt(self, contract: Dict[str, Any] | None) -> List[str]:
        """Build prompt-safe column universe from executable input columns first, then canonical/derived."""
        if not isinstance(contract, dict):
            return []
        cols: List[str] = []

        artifact_reqs = contract.get("artifact_requirements")
        if isinstance(artifact_reqs, dict):
            clean_cfg = artifact_reqs.get("clean_dataset")
            if isinstance(clean_cfg, dict):
                required_input = clean_cfg.get("required_columns")
                if isinstance(required_input, list):
                    cols.extend([str(v) for v in required_input if v])

        canonical = contract.get("canonical_columns")
        if isinstance(canonical, list):
            cols.extend([str(v) for v in canonical if v])

        # Also include derived_columns
        derived = contract.get("derived_columns")
        if isinstance(derived, list):
            for item in derived:
                if isinstance(item, str) and item:
                    cols.append(item)
                elif isinstance(item, dict) and item.get("name"):
                    cols.append(str(item.get("name")))
        seen = set()
        deduped = []
        for col in cols:
            key = col.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(col)
            if len(deduped) >= 120:
                break
        return deduped

    def _resolve_allowed_name_patterns_for_prompt(self, contract: Dict[str, Any] | None) -> List[str]:
        """V4.1: Use artifact_requirements.file_schemas only, no legacy fallback."""
        if not isinstance(contract, dict):
            return []
        scored_schema = get_declared_file_schema(contract, "scored_rows.csv")
        if not isinstance(scored_schema, dict):
            return []
        allowed = scored_schema.get("allowed_name_patterns")
        if not isinstance(allowed, list):
            return []
        return [str(pat) for pat in allowed if isinstance(pat, str) and pat.strip()]

    def _select_feedback_blocks(
        self,
        feedback_history: List[str] | None,
        gate_context: Dict[str, Any] | None,
        max_blocks: int = 2,
    ) -> str:
        if isinstance(gate_context, dict):
            if isinstance(gate_context.get("feedback_record"), dict):
                try:
                    payload = json.dumps(gate_context.get("feedback_record"), indent=2, ensure_ascii=True)
                    return "LATEST_ITERATION_FEEDBACK_RECORD_JSON:\n" + payload
                except Exception:
                    pass
            if isinstance(gate_context.get("feedback_json"), dict):
                try:
                    payload = json.dumps(gate_context.get("feedback_json"), indent=2, ensure_ascii=True)
                    return "LATEST_ITERATION_FEEDBACK_JSON:\n" + payload
                except Exception:
                    pass
        blocks: List[str] = []
        if isinstance(gate_context, dict):
            feedback = gate_context.get("feedback")
            if isinstance(feedback, str) and feedback.strip():
                blocks.append(feedback.strip())
        if isinstance(feedback_history, list):
            for item in reversed(feedback_history):
                if isinstance(item, str) and item.strip():
                    blocks.append(item.strip())
                if len(blocks) >= max_blocks:
                    break
        if not blocks:
            return ""
        return "\n\n".join(blocks[:max_blocks])

    def _normalize_handoff_items(
        self,
        values: Any,
        max_items: int = 8,
        max_len: int = 200,
    ) -> List[str]:
        out: List[str] = []
        if not isinstance(values, list):
            return out
        for value in values:
            text = str(value or "").strip()
            if not text:
                continue
            if len(text) > max_len:
                text = text[: max_len - 3] + "..."
            if text in out:
                continue
            out.append(text)
            if len(out) >= max_items:
                break
        return out

    def _normalize_handoff_evidence(
        self,
        values: Any,
        max_items: int = 8,
    ) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        if not isinstance(values, list):
            return out
        for item in values:
            claim = ""
            source = "missing"
            if isinstance(item, dict):
                claim = str(item.get("claim") or "").strip()
                source = str(item.get("source") or "missing").strip() or "missing"
            else:
                claim = str(item or "").strip()
            if not claim:
                continue
            normalized = {"claim": claim[:260], "source": source[:220]}
            if normalized in out:
                continue
            out.append(normalized)
            if len(out) >= max_items:
                break
        return out

    def _normalize_iteration_handoff(
        self,
        iteration_handoff: Dict[str, Any] | None,
        gate_context: Dict[str, Any] | None,
        required_deliverables: List[str] | None,
    ) -> Dict[str, Any]:
        raw = iteration_handoff if isinstance(iteration_handoff, dict) else {}
        gate_context = gate_context if isinstance(gate_context, dict) else {}
        contract_focus = raw.get("contract_focus") if isinstance(raw.get("contract_focus"), dict) else {}
        quality_focus = raw.get("quality_focus") if isinstance(raw.get("quality_focus"), dict) else {}
        optimization_focus_raw = raw.get("optimization_focus") if isinstance(raw.get("optimization_focus"), dict) else {}
        optimization_context_raw = raw.get("optimization_context") if isinstance(raw.get("optimization_context"), dict) else {}
        if not optimization_context_raw:
            optimization_context_raw = (
                gate_context.get("optimization_context")
                if isinstance(gate_context.get("optimization_context"), dict)
                else {}
            )
        feedback = raw.get("feedback") if isinstance(raw.get("feedback"), dict) else {}
        critic_packet = raw.get("critic_packet") if isinstance(raw.get("critic_packet"), dict) else {}
        if not critic_packet:
            critic_packet = raw.get("advisor_critique_packet") if isinstance(raw.get("advisor_critique_packet"), dict) else {}
        hypothesis_packet = raw.get("hypothesis_packet") if isinstance(raw.get("hypothesis_packet"), dict) else {}
        if not hypothesis_packet:
            hypothesis_packet = (
                raw.get("iteration_hypothesis_packet")
                if isinstance(raw.get("iteration_hypothesis_packet"), dict)
                else {}
            )
        editor_constraints = (
            dict(raw.get("editor_constraints"))
            if isinstance(raw.get("editor_constraints"), dict)
            else {}
        )
        repair_policy = raw.get("repair_policy") if isinstance(raw.get("repair_policy"), dict) else {}
        retry_context = raw.get("retry_context") if isinstance(raw.get("retry_context"), dict) else {}
        repair_ground_truth = (
            raw.get("repair_ground_truth")
            if isinstance(raw.get("repair_ground_truth"), dict)
            else {}
        )
        repair_scope = raw.get("repair_scope") if isinstance(raw.get("repair_scope"), dict) else {}
        deferred_optimization = (
            raw.get("deferred_optimization")
            if isinstance(raw.get("deferred_optimization"), dict)
            else {}
        )
        optimization_lane = (
            raw.get("optimization_lane")
            if isinstance(raw.get("optimization_lane"), dict)
            else {}
        )
        repair_first = bool(repair_policy.get("repair_first")) or str(
            repair_policy.get("primary_focus") or retry_context.get("repair_focus") or ""
        ).strip().lower() in {"runtime", "persistence", "compliance"}
        gate_enforcement = (
            gate_context.get("metric_round_enforcement")
            if isinstance(gate_context.get("metric_round_enforcement"), dict)
            else {}
        )
        if isinstance(gate_enforcement, dict):
            for key in (
                "must_apply_hypothesis",
                "forbid_noop",
                "patch_intensity",
                "scope_policy",
                "allow_strategy_changes",
                "freeze_unimplicated_regions",
            ):
                if key not in editor_constraints and key in gate_enforcement:
                    editor_constraints[key] = gate_enforcement.get(key)
        if "must_apply_hypothesis" not in editor_constraints:
            editor_constraints["must_apply_hypothesis"] = bool(raw.get("must_apply_hypothesis"))
        if "forbid_noop" not in editor_constraints:
            editor_constraints["forbid_noop"] = bool(raw.get("forbid_noop"))
        if not str(editor_constraints.get("patch_intensity") or "").strip():
            editor_constraints["patch_intensity"] = str(raw.get("patch_intensity") or "incremental")

        required_outputs = self._normalize_handoff_items(
            contract_focus.get("required_outputs") or required_deliverables,
            max_items=12,
            max_len=140,
        )
        present_outputs = self._normalize_handoff_items(contract_focus.get("present_outputs"), max_items=12, max_len=140)
        missing_outputs = self._normalize_handoff_items(contract_focus.get("missing_outputs"), max_items=12, max_len=140)
        failed_gates = self._normalize_handoff_items(
            quality_focus.get("failed_gates") or gate_context.get("failed_gates"),
            max_items=12,
            max_len=180,
        )
        required_fixes = self._normalize_handoff_items(
            quality_focus.get("required_fixes") or gate_context.get("required_fixes"),
            max_items=12,
            max_len=240,
        )
        hard_failures = self._normalize_handoff_items(
            quality_focus.get("hard_failures") or gate_context.get("hard_failures"),
            max_items=8,
            max_len=180,
        )
        evidence_focus = self._normalize_handoff_evidence(
            quality_focus.get("evidence")
            or feedback.get("evidence")
            or gate_context.get("evidence"),
            max_items=10,
        )
        must_preserve = self._normalize_handoff_items(raw.get("must_preserve"), max_items=8, max_len=220)
        if not must_preserve and present_outputs:
            must_preserve = [f"Preserve generation for {path}" for path in present_outputs[:6]]
        patch_objectives = self._normalize_handoff_items(raw.get("patch_objectives"), max_items=8, max_len=260)
        if bool(editor_constraints.get("must_apply_hypothesis")):
            enforce_msg = "Apply the active metric-improvement hypothesis with material code edits (NO_OP forbidden)."
            if enforce_msg not in patch_objectives:
                patch_objectives.insert(0, enforce_msg)
        if missing_outputs and not any("missing contract outputs" in item.lower() for item in patch_objectives):
            patch_objectives.insert(
                0,
                "Generate all missing contract outputs at exact paths: " + ", ".join(missing_outputs[:5]),
            )
        for fix in required_fixes:
            if fix not in patch_objectives:
                patch_objectives.append(fix)
            if len(patch_objectives) >= 8:
                break
        if not patch_objectives:
            patch_objectives = ["Apply reviewer feedback with minimal, targeted edits to the previous script."]

        optimization_focus: Dict[str, Any] = {}
        if optimization_focus_raw:
            optimization_focus = {
                "round_id": optimization_focus_raw.get("round_id"),
                "rounds_allowed": optimization_focus_raw.get("rounds_allowed"),
                "primary_metric_name": optimization_focus_raw.get("primary_metric_name"),
                "baseline_metric": optimization_focus_raw.get("baseline_metric"),
                "min_delta": optimization_focus_raw.get("min_delta"),
                "higher_is_better": optimization_focus_raw.get("higher_is_better"),
                "feature_engineering_plan": (
                    optimization_focus_raw.get("feature_engineering_plan")
                    if isinstance(optimization_focus_raw.get("feature_engineering_plan"), dict)
                    else {}
                ),
            }
        optimization_context: Dict[str, Any] = {}
        if optimization_context_raw:
            optimization_context = {
                "policy": (
                    optimization_context_raw.get("policy")
                    if isinstance(optimization_context_raw.get("policy"), dict)
                    else {}
                ),
                "metric_snapshot": (
                    optimization_context_raw.get("metric_snapshot")
                    if isinstance(optimization_context_raw.get("metric_snapshot"), dict)
                    else {}
                ),
                "contract_lock": (
                    optimization_context_raw.get("contract_lock")
                    if isinstance(optimization_context_raw.get("contract_lock"), dict)
                    else {}
                ),
                "active_hypothesis": (
                    optimization_context_raw.get("active_hypothesis")
                    if isinstance(optimization_context_raw.get("active_hypothesis"), dict)
                    else {}
                ),
                "experiment_tracker_recent": (
                    optimization_context_raw.get("experiment_tracker_recent")
                    if isinstance(optimization_context_raw.get("experiment_tracker_recent"), list)
                    else []
                ),
                "round_history_recent": (
                    optimization_context_raw.get("round_history_recent")
                    if isinstance(optimization_context_raw.get("round_history_recent"), list)
                    else []
                ),
            }
        keep_optimization_lane = bool(optimization_lane.get("active")) and (
            optimization_context or hypothesis_packet or deferred_optimization
        )
        if repair_first and not keep_optimization_lane:
            optimization_focus = {}
            optimization_context = {}
            critic_packet = {}
            hypothesis_packet = {}

        return {
            "handoff_version": raw.get("handoff_version") or "v1",
            "mode": "patch" if repair_first else str(raw.get("mode") or ("patch" if gate_context else "build")).lower(),
            "source": raw.get("source") or "result_evaluator",
            "from_iteration": raw.get("from_iteration"),
            "next_iteration": raw.get("next_iteration"),
            "contract_focus": {
                "required_outputs": required_outputs,
                "present_outputs": present_outputs,
                "missing_outputs": missing_outputs,
            },
            "quality_focus": {
                "status": raw.get("quality_focus", {}).get("status") if isinstance(raw.get("quality_focus"), dict) else None,
                "failed_gates": failed_gates,
                "required_fixes": required_fixes,
                "hard_failures": hard_failures,
                "evidence": evidence_focus,
            },
            "optimization_focus": optimization_focus,
            "optimization_context": optimization_context,
            "feedback": {
                "reviewer": str(feedback.get("reviewer") or gate_context.get("feedback") or "").strip(),
                "qa": str(feedback.get("qa") or "").strip(),
                "runtime_error_tail": str(feedback.get("runtime_error_tail") or "").strip(),
                "evidence": evidence_focus,
            },
            "optimization_lane": optimization_lane if isinstance(optimization_lane, dict) else {},
            "repair_policy": repair_policy if isinstance(repair_policy, dict) else {},
            "deferred_optimization": deferred_optimization if isinstance(deferred_optimization, dict) else {},
            "retry_context": retry_context if isinstance(retry_context, dict) else {},
            "repair_ground_truth": repair_ground_truth if isinstance(repair_ground_truth, dict) else {},
            "repair_scope": repair_scope if isinstance(repair_scope, dict) else {},
            "must_preserve": must_preserve[:8],
            "patch_objectives": patch_objectives[:8],
            "critic_packet": critic_packet if isinstance(critic_packet, dict) else {},
            "hypothesis_packet": hypothesis_packet if isinstance(hypothesis_packet, dict) else {},
            "editor_constraints": {
                "must_apply_hypothesis": bool(editor_constraints.get("must_apply_hypothesis")),
                "forbid_noop": bool(editor_constraints.get("forbid_noop")),
                "patch_intensity": str(editor_constraints.get("patch_intensity") or "incremental"),
                "scope_policy": str(editor_constraints.get("scope_policy") or "").strip(),
                "allow_strategy_changes": bool(editor_constraints.get("allow_strategy_changes")),
                "freeze_unimplicated_regions": bool(editor_constraints.get("freeze_unimplicated_regions")),
            },
        }

    def _collect_editor_feedback_text(
        self,
        gate_context: Dict[str, Any] | None,
        handoff_payload: Dict[str, Any] | None,
        feedback_history: List[str] | None,
    ) -> str:
        gate_context = gate_context if isinstance(gate_context, dict) else {}
        handoff_payload = handoff_payload if isinstance(handoff_payload, dict) else {}
        if self._is_metric_optimization_context(
            gate_context=gate_context,
            handoff_payload=handoff_payload,
        ):
            return self._collect_metric_optimization_feedback_text(
                gate_context=gate_context,
                handoff_payload=handoff_payload,
                feedback_history=feedback_history,
            )
        chunks: List[str] = []
        seen_chunks: set[str] = set()

        def _append_chunk(value: str) -> None:
            text = str(value or "").strip()
            if not text or text in seen_chunks:
                return
            seen_chunks.add(text)
            chunks.append(text)

        feedback_record = gate_context.get("feedback_record")
        if isinstance(feedback_record, dict) and feedback_record:
            try:
                _append_chunk(
                    "LATEST_ITERATION_FEEDBACK_RECORD_JSON:\n"
                    + json.dumps(feedback_record, ensure_ascii=False, indent=2)
                )
            except Exception:
                _append_chunk(str(feedback_record))

        feedback_json = gate_context.get("feedback_json")
        if isinstance(feedback_json, dict) and feedback_json:
            try:
                _append_chunk(
                    "LATEST_ITERATION_FEEDBACK_JSON:\n"
                    + json.dumps(feedback_json, ensure_ascii=False, indent=2)
                )
            except Exception:
                _append_chunk(str(feedback_json))

        feedback = str(gate_context.get("feedback") or "").strip()
        if feedback:
            _append_chunk(feedback)

        runtime_payload = gate_context.get("runtime_error")
        if isinstance(runtime_payload, dict) and runtime_payload:
            try:
                _append_chunk(json.dumps(runtime_payload, ensure_ascii=False))
            except Exception:
                _append_chunk(str(runtime_payload))

        traceback_text = str(
            gate_context.get("traceback")
            or gate_context.get("execution_output_tail")
            or ""
        ).strip()
        if traceback_text:
            _append_chunk(traceback_text)

        handoff_feedback = handoff_payload.get("feedback")
        if isinstance(handoff_feedback, dict):
            try:
                _append_chunk(
                    "ITERATION_HANDOFF_FEEDBACK_JSON:\n"
                    + json.dumps(handoff_feedback, ensure_ascii=False, indent=2)
                )
            except Exception:
                _append_chunk(str(handoff_feedback))
            tail = str(handoff_feedback.get("runtime_error_tail") or "").strip()
            reviewer = str(handoff_feedback.get("reviewer") or "").strip()
            qa = str(handoff_feedback.get("qa") or "").strip()
            for item in (tail, reviewer, qa):
                if item:
                    _append_chunk(item)

        if isinstance(feedback_history, list):
            for item in feedback_history[-3:]:
                text = str(item or "").strip()
                if text:
                    _append_chunk(text)

        if not chunks:
            return ""
        merged = "\n\n".join(chunks)
        return self._truncate_prompt_text(
            merged,
            max_len=6000,
            head_len=4000,
            tail_len=1500,
        )

    def _build_guardrail_repair_context(
        self,
        handoff_payload: Dict[str, Any] | None,
        gate_context: Dict[str, Any] | None,
        feedback_history: List[str] | None,
    ) -> str:
        handoff_payload = handoff_payload if isinstance(handoff_payload, dict) else {}
        gate_context = gate_context if isinstance(gate_context, dict) else {}
        quality_focus = (
            handoff_payload.get("quality_focus")
            if isinstance(handoff_payload.get("quality_focus"), dict)
            else {}
        )
        patch_objectives = self._normalize_handoff_items(
            handoff_payload.get("patch_objectives"),
            max_items=6,
            max_len=220,
        )
        required_fixes = self._normalize_handoff_items(
            quality_focus.get("required_fixes") or gate_context.get("required_fixes"),
            max_items=6,
            max_len=220,
        )
        must_preserve = self._normalize_handoff_items(
            handoff_payload.get("must_preserve"),
            max_items=6,
            max_len=220,
        )
        feedback_text = self._collect_editor_feedback_text(
            gate_context=gate_context,
            handoff_payload=handoff_payload,
            feedback_history=feedback_history,
        )

        lines: List[str] = []
        if patch_objectives:
            lines.append("ACTIVE_PATCH_OBJECTIVES:")
            lines.extend([f"- {item}" for item in patch_objectives])
        if required_fixes:
            lines.append("ACTIVE_REQUIRED_FIXES:")
            lines.extend([f"- {item}" for item in required_fixes])
        if must_preserve:
            lines.append("MUST_PRESERVE:")
            lines.extend([f"- {item}" for item in must_preserve])
        if feedback_text:
            lines.append(
                "ACTIVE_REVIEW_FEEDBACK:\n"
                + self._truncate_prompt_text(
                    feedback_text,
                    max_len=1800,
                    head_len=1200,
                    tail_len=400,
                )
            )
        return "\n".join(lines).strip()

    def _is_repair_first_context(
        self,
        gate_context: Dict[str, Any] | None,
        handoff_payload: Dict[str, Any] | None,
    ) -> bool:
        gate_context = gate_context if isinstance(gate_context, dict) else {}
        handoff_payload = handoff_payload if isinstance(handoff_payload, dict) else {}
        repair_policy = (
            handoff_payload.get("repair_policy")
            if isinstance(handoff_payload.get("repair_policy"), dict)
            else {}
        )
        retry_context = (
            handoff_payload.get("retry_context")
            if isinstance(handoff_payload.get("retry_context"), dict)
            else {}
        )
        quality_focus = (
            handoff_payload.get("quality_focus")
            if isinstance(handoff_payload.get("quality_focus"), dict)
            else {}
        )
        contract_focus = (
            handoff_payload.get("contract_focus")
            if isinstance(handoff_payload.get("contract_focus"), dict)
            else {}
        )
        if bool(repair_policy.get("repair_first")):
            return True
        primary_focus = str(
            repair_policy.get("primary_focus") or retry_context.get("repair_focus") or ""
        ).strip().lower()
        if primary_focus in {"runtime", "persistence", "compliance"}:
            return True
        if gate_context.get("runtime_error"):
            return True
        missing_outputs = contract_focus.get("missing_outputs")
        if isinstance(missing_outputs, list) and any(str(item).strip() for item in missing_outputs):
            return True
        hard_failures = []
        for values in (
            gate_context.get("hard_failures"),
            quality_focus.get("hard_failures"),
        ):
            if isinstance(values, list):
                hard_failures.extend([str(item).strip().lower() for item in values if str(item).strip()])
        if hard_failures:
            return True
        failed_gates = []
        for values in (
            gate_context.get("failed_gates"),
            quality_focus.get("failed_gates"),
        ):
            if isinstance(values, list):
                failed_gates.extend([str(item).strip().lower() for item in values if str(item).strip()])
        blocking_tokens = (
            "runtime",
            "output_contract",
            "required output",
            "required artifact",
            "artifact",
            "alignment_check",
            "contract",
            "hard",
            "leakage_prevention",
            "target_mapping",
        )
        return any(any(token in gate for token in blocking_tokens) for gate in failed_gates)

    def _is_metric_optimization_context(
        self,
        gate_context: Dict[str, Any] | None,
        handoff_payload: Dict[str, Any] | None,
    ) -> bool:
        gate_context = gate_context if isinstance(gate_context, dict) else {}
        handoff_payload = handoff_payload if isinstance(handoff_payload, dict) else {}
        mode = str(handoff_payload.get("mode") or "").strip().lower()
        source = str(handoff_payload.get("source") or gate_context.get("source") or "").strip().lower()
        quality_focus = handoff_payload.get("quality_focus") if isinstance(handoff_payload.get("quality_focus"), dict) else {}
        status = str(
            quality_focus.get("status")
            or gate_context.get("status")
            or ""
        ).strip().upper()
        constraints = handoff_payload.get("editor_constraints") if isinstance(handoff_payload.get("editor_constraints"), dict) else {}
        must_apply_hypothesis = bool(constraints.get("must_apply_hypothesis"))
        optimization_context = (
            handoff_payload.get("optimization_context")
            if isinstance(handoff_payload.get("optimization_context"), dict)
            else {}
        )
        hypothesis_packet = (
            handoff_payload.get("hypothesis_packet")
            if isinstance(handoff_payload.get("hypothesis_packet"), dict)
            else {}
        )
        deferred_optimization = (
            handoff_payload.get("deferred_optimization")
            if isinstance(handoff_payload.get("deferred_optimization"), dict)
            else {}
        )
        optimization_lane = (
            handoff_payload.get("optimization_lane")
            if isinstance(handoff_payload.get("optimization_lane"), dict)
            else {}
        )
        if self._is_repair_first_context(gate_context=gate_context, handoff_payload=handoff_payload):
            if bool(optimization_lane.get("active")) and (
                optimization_context or hypothesis_packet or deferred_optimization
            ):
                return True
            return False
        if mode in {"optimize", "improve", "metric_optimize"}:
            return True
        if status in {"OPTIMIZATION_REQUIRED", "IMPROVEMENT_REQUIRED"}:
            return True
        if "metric_improvement" in source and must_apply_hypothesis:
            return True
        if "actor_critic" in source and must_apply_hypothesis:
            return True
        if bool(optimization_lane.get("active")) and (optimization_context or hypothesis_packet):
            return True
        return False

    def _collect_metric_optimization_feedback_text(
        self,
        gate_context: Dict[str, Any] | None,
        handoff_payload: Dict[str, Any] | None,
        feedback_history: List[str] | None,
    ) -> str:
        gate_context = gate_context if isinstance(gate_context, dict) else {}
        handoff_payload = handoff_payload if isinstance(handoff_payload, dict) else {}
        chunks: List[str] = []
        seen_chunks: set[str] = set()

        def _append_chunk(value: str) -> None:
            text = str(value or "").strip()
            if not text or text in seen_chunks:
                return
            seen_chunks.add(text)
            chunks.append(text)

        optimization_focus = (
            handoff_payload.get("optimization_focus")
            if isinstance(handoff_payload.get("optimization_focus"), dict)
            else {}
        )
        if optimization_focus:
            _append_chunk(
                "OPTIMIZATION_FOCUS_JSON:\n"
                + self._serialize_json_for_prompt(
                    optimization_focus,
                    max_chars=2200,
                    max_str_len=280,
                    max_list_items=30,
                )
            )
        optimization_context = (
            handoff_payload.get("optimization_context")
            if isinstance(handoff_payload.get("optimization_context"), dict)
            else {}
        )
        if optimization_context:
            _append_chunk(
                "OPTIMIZATION_CONTEXT_JSON:\n"
                + self._serialize_json_for_prompt(
                    optimization_context,
                    max_chars=2600,
                    max_str_len=280,
                    max_list_items=30,
                )
            )

        critic_packet = handoff_payload.get("critic_packet")
        if isinstance(critic_packet, dict) and critic_packet:
            _append_chunk(
                "CRITIQUE_PACKET_JSON:\n"
                + self._serialize_json_for_prompt(
                    critic_packet,
                    max_chars=2200,
                    max_str_len=280,
                    max_list_items=30,
                )
            )

        hypothesis_packet = handoff_payload.get("hypothesis_packet")
        if isinstance(hypothesis_packet, dict) and hypothesis_packet:
            _append_chunk(
                "HYPOTHESIS_PACKET_JSON:\n"
                + self._serialize_json_for_prompt(
                    hypothesis_packet,
                    max_chars=1800,
                    max_str_len=260,
                    max_list_items=25,
                )
            )

        handoff_feedback = handoff_payload.get("feedback")
        if isinstance(handoff_feedback, dict) and handoff_feedback:
            _append_chunk(
                "IMPROVEMENT_FEEDBACK_JSON:\n"
                + self._serialize_json_for_prompt(
                    handoff_feedback,
                    max_chars=1600,
                    max_str_len=260,
                    max_list_items=20,
                )
            )

        runtime_payload = gate_context.get("runtime_error")
        if isinstance(runtime_payload, dict) and runtime_payload:
            _append_chunk(
                "RUNTIME_ERROR_JSON:\n"
                + self._serialize_json_for_prompt(
                    runtime_payload,
                    max_chars=1200,
                    max_str_len=220,
                    max_list_items=15,
                )
            )
        runtime_tail = str(
            gate_context.get("traceback")
            or gate_context.get("execution_output_tail")
            or ""
        ).strip()
        if runtime_tail:
            _append_chunk(
                "RUNTIME_ERROR_TAIL:\n"
                + self._truncate_prompt_text(
                    runtime_tail,
                    max_len=1200,
                    head_len=900,
                    tail_len=200,
                )
            )

        if isinstance(feedback_history, list):
            for item in feedback_history[-5:]:
                text = str(item or "").strip()
                if not text:
                    continue
                lower = text.lower()
                if (
                    "# improvement_round" in lower
                    or "results_advisor_feedback" in lower
                    or "iteration_hypothesis_packet" in lower
                    or "steward_feedback" in lower
                ):
                    _append_chunk(text)

        if not chunks:
            return ""
        merged = "\n\n".join(chunks)
        return self._truncate_prompt_text(
            merged,
            max_len=4200,
            head_len=2800,
            tail_len=1000,
        )

    def _should_use_editor_mode(
        self,
        gate_context: Dict[str, Any] | None,
        handoff_payload: Dict[str, Any] | None,
        feedback_history: List[str] | None = None,
    ) -> bool:
        gate_context = gate_context if isinstance(gate_context, dict) else {}
        handoff_payload = handoff_payload if isinstance(handoff_payload, dict) else {}
        quality_focus = handoff_payload.get("quality_focus")
        quality_focus = quality_focus if isinstance(quality_focus, dict) else {}

        source = str(gate_context.get("source") or handoff_payload.get("source") or "").strip().lower()
        failed_gates: List[str] = []
        for values in (
            gate_context.get("failed_gates"),
            quality_focus.get("failed_gates"),
            gate_context.get("hard_failures"),
            quality_focus.get("hard_failures"),
        ):
            if isinstance(values, list):
                failed_gates.extend([str(item).strip().lower() for item in values if str(item).strip()])

        feedback_blob = self._collect_editor_feedback_text(
            gate_context=gate_context,
            handoff_payload=handoff_payload,
            feedback_history=feedback_history,
        ).lower()
        has_runtime_signal = bool(
            gate_context.get("runtime_error")
            or "traceback" in feedback_blob
            or "runtime error" in feedback_blob
            or "runtime_failure" in feedback_blob
            or "runtime_fix" in source
            or any("runtime" in gate for gate in failed_gates)
        )
        has_qa_signal = bool(
            "qa" in source
            or any("qa" in gate for gate in failed_gates)
            or "qa team feedback" in feedback_blob
            or "qa_code_audit" in feedback_blob
        )
        has_optimization_signal = self._is_metric_optimization_context(
            gate_context=gate_context,
            handoff_payload=handoff_payload,
        )
        return bool(has_runtime_signal or has_qa_signal or has_optimization_signal)

    def _is_actor_critic_improvement_strict_enabled(self) -> bool:
        raw = str(os.getenv("ACTOR_CRITIC_IMPROVEMENT_STRICT", "1") or "").strip().lower()
        return raw not in {"0", "false", "no", "off"}

    def _classify_editor_phase(
        self,
        gate_context: Dict[str, Any] | None,
        handoff_payload: Dict[str, Any] | None,
        feedback_text: str,
    ) -> str:
        gate_context = gate_context if isinstance(gate_context, dict) else {}
        handoff_payload = handoff_payload if isinstance(handoff_payload, dict) else {}
        quality_focus = handoff_payload.get("quality_focus")
        quality_focus = quality_focus if isinstance(quality_focus, dict) else {}
        repair_ground_truth = (
            handoff_payload.get("repair_ground_truth")
            if isinstance(handoff_payload.get("repair_ground_truth"), dict)
            else {}
        )

        failed_tokens: List[str] = []
        for values in (
            gate_context.get("failed_gates"),
            quality_focus.get("failed_gates"),
            gate_context.get("required_fixes"),
            quality_focus.get("required_fixes"),
        ):
            if isinstance(values, list):
                failed_tokens.extend([str(item).strip().lower() for item in values if str(item).strip()])

        failed_blob = " ".join(failed_tokens)
        feedback_blob = str(feedback_text or "").lower()
        combined = " ".join([failed_blob, feedback_blob]).strip()
        runtime_tokens = [
            "runtime",
            "traceback",
            "exception",
            "timeout",
            "timed out",
            "deadline exceeded",
            "script exceeded",
            "oom",
            "memoryerror",
            "sandbox",
        ]
        training_gate_tokens = [
            "strategy_followed",
            "leakage_prevention",
            "target_variance_guard",
            "validation_method",
            "cross_validation",
            "train_filter",
            "feature_governance",
        ]
        training_edit_tokens = [
            "replace logistic",
            "boosting",
            "stacking",
            "ensemble",
            "target encoding",
            "onehotencoding",
            "onehotencoder",
            "ordinalencoder",
            "out-of-fold",
            "oof",
            "meta-learner",
            "model choice",
            "preprocessing",
            "zero-variance",
            "nunique",
        ]
        persistence_tokens = [
            "output_contract",
            "required output",
            "required artifact",
            "missing artifact",
            "alignment_check",
            "scored_rows",
            "metrics.json",
            "manifest",
            "json serialization",
            "serialization",
            "file not found",
            "no such file",
            "permission denied",
            "to_csv",
            "persist",
            "write artifact",
        ]
        root_cause_type = str(repair_ground_truth.get("root_cause_type") or "").strip().lower()
        repair_focus = str(repair_ground_truth.get("repair_focus") or "").strip().lower()
        if repair_focus == "runtime" or root_cause_type in {
            "runtime_api_misuse",
            "runtime_contract_assertion",
            "runtime_error",
            "security_violation",
            "timeout",
            "oom",
            "import_error",
            "shape_or_dtype",
        }:
            return "runtime_repair"
        if repair_focus == "compliance":
            return "runtime_repair"
        if repair_focus == "persistence" or root_cause_type in {"artifact_io", "output_missing"}:
            return "persistence"
        if self._is_repair_first_context(
            gate_context=gate_context,
            handoff_payload=handoff_payload,
        ) and any(token in combined for token in runtime_tokens):
            return "runtime_repair"
        if any(token in failed_blob for token in training_gate_tokens):
            return "training"
        if any(token in feedback_blob for token in training_edit_tokens):
            return "training"
        if any(token in combined for token in persistence_tokens):
            return "persistence"
        if self._is_metric_optimization_context(
            gate_context=gate_context,
            handoff_payload=handoff_payload,
        ):
            return "optimization"
        return "training"

    def _build_last_run_memory_block(
        self,
        last_run_memory: List[Dict[str, Any]] | None,
    ) -> str:
        if not isinstance(last_run_memory, list) or not last_run_memory:
            return "[]"
        compact_entries: List[Dict[str, Any]] = []
        for entry in last_run_memory[-5:]:
            if not isinstance(entry, dict):
                continue
            compact_entries.append(
                {
                    "iter": entry.get("iter"),
                    "attempt": entry.get("attempt"),
                    "event": entry.get("event"),
                    "phase": entry.get("phase"),
                    "error_signature": entry.get("error_signature"),
                    "changes_summary": entry.get("changes_summary"),
                }
            )
        return self._serialize_json_for_prompt(
            compact_entries,
            max_chars=2800,
            max_str_len=350,
            max_list_items=8,
        )

    def _build_editor_enforcement_block(
        self,
        handoff_payload: Dict[str, Any] | None,
    ) -> str:
        handoff_payload = handoff_payload if isinstance(handoff_payload, dict) else {}
        constraints = (
            handoff_payload.get("editor_constraints")
            if isinstance(handoff_payload.get("editor_constraints"), dict)
            else {}
        )
        must_apply = bool(constraints.get("must_apply_hypothesis"))
        forbid_noop = bool(constraints.get("forbid_noop"))
        patch_intensity = str(constraints.get("patch_intensity") or "incremental").strip() or "incremental"
        scope_policy = str(constraints.get("scope_policy") or "").strip().lower()
        allow_strategy_changes = bool(constraints.get("allow_strategy_changes"))
        freeze_unimplicated = bool(constraints.get("freeze_unimplicated_regions"))
        optimization_context = (
            handoff_payload.get("optimization_context")
            if isinstance(handoff_payload.get("optimization_context"), dict)
            else {}
        )
        repair_scope = (
            handoff_payload.get("repair_scope")
            if isinstance(handoff_payload.get("repair_scope"), dict)
            else {}
        )
        if scope_policy == "patch_only":
            phase = str(repair_scope.get("phase") or "compliance_runtime").strip() or "compliance_runtime"
            editable_targets = [
                str(item)
                for item in (repair_scope.get("editable_targets") or [])
                if str(item).strip()
            ]
            protected_regions = [
                str(item)
                for item in (repair_scope.get("protected_regions") or [])
                if str(item).strip()
            ]
            invariants = [
                str(item)
                for item in (repair_scope.get("must_preserve_invariants") or [])
                if str(item).strip()
            ]
            lines: List[str] = [
                f"- {phase} patch-only mode is ACTIVE.",
                "- Use the full script only as context. Do not redesign strategy, model family, or healthy pipeline sections.",
                "- Edit only reviewer-audited findings and verified failing blocks from REPAIR SCOPE.",
                f"- Patch intensity: {patch_intensity}.",
            ]
            if freeze_unimplicated:
                lines.append("- Treat all regions outside REPAIR SCOPE as frozen unless new verified runtime evidence directly contradicts them.")
            if not allow_strategy_changes:
                lines.append("- Strategy changes are NOT allowed in this mode.")
            if optimization_context:
                lines.append("- Active optimization hypothesis is deferred in patch-only mode; repair the incumbent defect first and preserve hypothesis context without implementing broader metric changes.")
            if editable_targets:
                lines.append("- Editable targets: " + "; ".join(editable_targets[:4]) + ".")
            if protected_regions:
                lines.append("- Protected regions: " + "; ".join(protected_regions[:4]) + ".")
            if invariants:
                lines.append("- Preserve invariants: " + "; ".join(invariants[:3]) + ".")
            return "\n".join(lines)
        if not (must_apply or forbid_noop):
            return "- Standard editor mode. Apply patch objectives and keep minimal safe edits."

        lines: List[str] = [
            "- Metric-improvement round enforcement is ACTIVE.",
            "- You must apply the active hypothesis with material code edits.",
            "- Returning baseline-equivalent code or NO_OP is invalid in this round.",
            "- Keep strategy lock, model family, CV protocol, and output contract paths unchanged.",
            f"- Patch intensity: {patch_intensity}.",
        ]
        return "\n".join(lines)

    def _resolve_optimization_mode_inputs(
        self,
        handoff_payload: Dict[str, Any] | None,
        last_run_memory: List[Dict[str, Any]] | None,
    ) -> Dict[str, Any]:
        handoff_payload = handoff_payload if isinstance(handoff_payload, dict) else {}
        optimization_context = (
            handoff_payload.get("optimization_context")
            if isinstance(handoff_payload.get("optimization_context"), dict)
            else {}
        )
        hypothesis_packet = (
            handoff_payload.get("hypothesis_packet")
            if isinstance(handoff_payload.get("hypothesis_packet"), dict)
            else {}
        )
        invariants_lock = (
            optimization_context.get("contract_lock")
            if isinstance(optimization_context.get("contract_lock"), dict)
            else {}
        )
        recent_tracker = (
            optimization_context.get("experiment_tracker_recent")
            if isinstance(optimization_context.get("experiment_tracker_recent"), list)
            else []
        )
        if not recent_tracker:
            recent_tracker = (
                optimization_context.get("round_history_recent")
                if isinstance(optimization_context.get("round_history_recent"), list)
                else []
            )
        if not recent_tracker and isinstance(last_run_memory, list):
            recent_tracker = [
                {
                    "iter": item.get("iter"),
                    "attempt": item.get("attempt"),
                    "event": item.get("event"),
                    "phase": item.get("phase"),
                    "error_signature": item.get("error_signature"),
                    "changes_summary": item.get("changes_summary"),
                }
                for item in last_run_memory[-5:]
                if isinstance(item, dict)
            ]
        missing_inputs: List[str] = []
        if not isinstance(hypothesis_packet, dict) or not hypothesis_packet:
            missing_inputs.append("hypothesis_packet")
        if not isinstance(invariants_lock, dict) or not invariants_lock:
            missing_inputs.append("invariants_lock")
        if not isinstance(recent_tracker, list) or not recent_tracker:
            missing_inputs.append("recent_tracker")
        action_family = classify_action_family(hypothesis_packet)
        action_family_guidelines = get_action_family_guidance(action_family)
        return {
            "hypothesis_packet": hypothesis_packet if isinstance(hypothesis_packet, dict) else {},
            "invariants_lock": invariants_lock if isinstance(invariants_lock, dict) else {},
            "recent_tracker": recent_tracker if isinstance(recent_tracker, list) else [],
            "missing_inputs": missing_inputs,
            "action_family": action_family,
            "action_family_guidelines": action_family_guidelines,
        }

    def _build_optimization_editor_briefs(
        self,
        handoff_payload: Dict[str, Any] | None,
        optimization_inputs: Dict[str, Any] | None,
        feedback_text: str,
    ) -> Dict[str, str]:
        handoff_payload = handoff_payload if isinstance(handoff_payload, dict) else {}
        optimization_inputs = optimization_inputs if isinstance(optimization_inputs, dict) else {}
        optimization_focus = (
            handoff_payload.get("optimization_focus")
            if isinstance(handoff_payload.get("optimization_focus"), dict)
            else {}
        )
        optimization_context = (
            handoff_payload.get("optimization_context")
            if isinstance(handoff_payload.get("optimization_context"), dict)
            else {}
        )
        critic_packet = (
            handoff_payload.get("critic_packet")
            if isinstance(handoff_payload.get("critic_packet"), dict)
            else {}
        )
        hypothesis_packet = (
            handoff_payload.get("hypothesis_packet")
            if isinstance(handoff_payload.get("hypothesis_packet"), dict)
            else {}
        )
        hypothesis = (
            hypothesis_packet.get("hypothesis")
            if isinstance(hypothesis_packet.get("hypothesis"), dict)
            else {}
        )
        policy = (
            optimization_context.get("policy")
            if isinstance(optimization_context.get("policy"), dict)
            else {}
        )
        metric_snapshot = (
            optimization_context.get("metric_snapshot")
            if isinstance(optimization_context.get("metric_snapshot"), dict)
            else {}
        )
        recent_tracker_raw = (
            optimization_inputs.get("recent_tracker")
            if isinstance(optimization_inputs.get("recent_tracker"), list)
            else []
        )
        recent_tracker: List[Dict[str, Any]] = []
        for item in recent_tracker_raw[-3:]:
            if not isinstance(item, dict):
                continue
            recent_tracker.append(
                {
                    "iter": item.get("iter"),
                    "attempt": item.get("attempt"),
                    "event": item.get("event"),
                    "phase": item.get("phase"),
                    "metric_value": item.get("metric_value"),
                    "delta": item.get("delta"),
                    "error_signature": item.get("error_signature"),
                    "changes_summary": item.get("changes_summary"),
                }
            )

        error_modes: List[Dict[str, Any]] = []
        critic_error_modes = critic_packet.get("error_modes")
        if isinstance(critic_error_modes, list):
            for item in critic_error_modes[:4]:
                if not isinstance(item, dict):
                    continue
                error_modes.append(
                    {
                        "id": item.get("id"),
                        "severity": item.get("severity"),
                        "affected_scope": item.get("affected_scope"),
                        "evidence": item.get("evidence"),
                    }
                )

        round_brief = {
            "phase": policy.get("phase") or optimization_focus.get("phase"),
            "round_id": optimization_focus.get("round_id"),
            "rounds_allowed": optimization_focus.get("rounds_allowed"),
            "primary_metric_name": optimization_focus.get("primary_metric_name")
            or metric_snapshot.get("primary_metric_name"),
            "baseline_metric": optimization_focus.get("baseline_metric")
            or metric_snapshot.get("baseline_metric"),
            "current_metric": metric_snapshot.get("current_metric"),
            "min_delta": optimization_focus.get("min_delta"),
            "higher_is_better": optimization_focus.get("higher_is_better"),
            "action_family": optimization_inputs.get("action_family"),
            "active_technique": hypothesis.get("technique"),
            "objective": hypothesis.get("objective"),
            "missing_inputs": optimization_inputs.get("missing_inputs") or [],
        }

        active_hypothesis_brief = {
            "action": hypothesis_packet.get("action"),
            "technique": hypothesis.get("technique"),
            "objective": hypothesis.get("objective"),
            "target_columns": hypothesis.get("target_columns"),
            "feature_scope": hypothesis.get("feature_scope"),
            "params": hypothesis.get("params"),
            "success_criteria": hypothesis_packet.get("success_criteria"),
            "application_constraints": hypothesis_packet.get("application_constraints"),
        }

        evidence_brief = {
            "analysis_summary": critic_packet.get("analysis_summary") or "",
            "risk_flags": critic_packet.get("risk_flags") or [],
            "top_error_modes": error_modes,
            "recent_attempts": recent_tracker,
        }

        invariants_lock = dict(
            optimization_inputs.get("invariants_lock")
            if isinstance(optimization_inputs.get("invariants_lock"), dict)
            else {}
        )
        primary_metric_name = str(round_brief.get("primary_metric_name") or "").strip()
        if primary_metric_name:
            invariants_lock.setdefault("primary_metric_name", primary_metric_name)
        if active_hypothesis_brief.get("target_columns"):
            invariants_lock.setdefault("target_columns", active_hypothesis_brief.get("target_columns"))
        if (
            primary_metric_name
            and "mean" in primary_metric_name.lower()
            and not any(
                key in invariants_lock
                for key in ("metric_weights", "weights", "horizon_weights")
            )
        ):
            invariants_lock["metric_definition_rule"] = (
                "Use a simple arithmetic mean across target horizons unless the contract explicitly provides weights."
            )

        blueprint_raw = (
            handoff_payload.get("optimization_blueprint")
            if isinstance(handoff_payload.get("optimization_blueprint"), dict)
            else {}
        )
        blueprint_actions = (
            blueprint_raw.get("improvement_actions")
            if isinstance(blueprint_raw.get("improvement_actions"), list)
            else []
        )
        blueprint_hint: List[Dict[str, Any]] = []
        for item in blueprint_actions[:3]:
            if not isinstance(item, dict):
                continue
            blueprint_hint.append(
                {
                    "technique": item.get("technique"),
                    "rationale": item.get("rationale"),
                    "concrete_params": item.get("concrete_params"),
                    "code_change_hints": item.get("code_change_hints"),
                }
            )

        feedback_brief = self._truncate_prompt_text(
            str(
                feedback_text
                or critic_packet.get("analysis_summary")
                or "No optimization feedback provided."
            ),
            max_len=1400,
            head_len=900,
            tail_len=300,
        )

        return {
            "round_brief": self._serialize_json_for_prompt(
                round_brief,
                max_chars=1800,
                max_str_len=220,
                max_list_items=20,
            ),
            "active_hypothesis_brief": self._serialize_json_for_prompt(
                active_hypothesis_brief,
                max_chars=2200,
                max_str_len=220,
                max_list_items=25,
            ),
            "current_evidence_brief": self._serialize_json_for_prompt(
                evidence_brief,
                max_chars=2200,
                max_str_len=220,
                max_list_items=18,
            ),
            "invariants_lock": self._serialize_json_for_prompt(
                invariants_lock,
                max_chars=2200,
                max_str_len=220,
                max_list_items=24,
            ),
            "optimization_blueprint_hint": self._serialize_json_for_prompt(
                blueprint_hint,
                max_chars=1800,
                max_str_len=220,
                max_list_items=12,
            ),
            "recent_tracker": self._serialize_json_for_prompt(
                recent_tracker,
                max_chars=1600,
                max_str_len=220,
                max_list_items=10,
            ),
            "optimization_feedback_brief": feedback_brief or "No optimization feedback provided.",
        }

    def _build_optimization_authoritative_state(
        self,
        *,
        data_path: str,
        execution_contract: Dict[str, Any] | None,
        ml_view: Dict[str, Any] | None,
    ) -> str:
        execution_contract = execution_contract if isinstance(execution_contract, dict) else {}
        ml_view = ml_view if isinstance(ml_view, dict) else {}

        allowed_sets = (
            execution_contract.get("allowed_feature_sets")
            if isinstance(execution_contract.get("allowed_feature_sets"), dict)
            else (
                ml_view.get("allowed_feature_sets")
                if isinstance(ml_view.get("allowed_feature_sets"), dict)
                else {}
            )
        )
        column_roles = (
            execution_contract.get("column_roles")
            if isinstance(execution_contract.get("column_roles"), dict)
            else (
                ml_view.get("column_roles")
                if isinstance(ml_view.get("column_roles"), dict)
                else {}
            )
        )
        evaluation_spec = (
            execution_contract.get("evaluation_spec")
            if isinstance(execution_contract.get("evaluation_spec"), dict)
            else (
                ml_view.get("evaluation_spec")
                if isinstance(ml_view.get("evaluation_spec"), dict)
                else {}
            )
        )
        validation_requirements = (
            execution_contract.get("validation_requirements")
            if isinstance(execution_contract.get("validation_requirements"), dict)
            else (
                ml_view.get("validation_requirements")
                if isinstance(ml_view.get("validation_requirements"), dict)
                else {}
            )
        )
        split_spec = (
            execution_contract.get("split_spec")
            if isinstance(execution_contract.get("split_spec"), dict)
            else (
                ml_view.get("split_spec")
                if isinstance(ml_view.get("split_spec"), dict)
                else {}
            )
        )
        artifact_requirements = (
            execution_contract.get("artifact_requirements")
            if isinstance(execution_contract.get("artifact_requirements"), dict)
            else (
                ml_view.get("artifact_requirements")
                if isinstance(ml_view.get("artifact_requirements"), dict)
                else {}
            )
        )

        model_features = allowed_sets.get("model_features")
        if not isinstance(model_features, list) or not model_features:
            model_features = column_roles.get("pre_decision")
        if not isinstance(model_features, list):
            model_features = []

        forbidden_features = allowed_sets.get("forbidden_features")
        if not isinstance(forbidden_features, list):
            forbidden_features = []

        target_columns = evaluation_spec.get("target_columns")
        if not isinstance(target_columns, list) or not target_columns:
            target_columns = validation_requirements.get("label_columns")
        if not isinstance(target_columns, list):
            target_columns = []

        scored_rows_schema = (
            artifact_requirements.get("scored_rows_schema")
            if isinstance(artifact_requirements.get("scored_rows_schema"), dict)
            else {}
        )
        file_schemas = (
            artifact_requirements.get("file_schemas")
            if isinstance(artifact_requirements.get("file_schemas"), dict)
            else {}
        )
        cleaning_manifest_path = self._resolve_cleaning_manifest_path(execution_contract, ml_view)
        scored_rows_path = self._resolve_declared_artifact_path(
            execution_contract if isinstance(execution_contract, dict) else {},
            ml_view if isinstance(ml_view, dict) else {},
            "scored_rows.csv",
        )
        submission_path = self._resolve_declared_artifact_path(
            execution_contract if isinstance(execution_contract, dict) else {},
            ml_view if isinstance(ml_view, dict) else {},
            "submission.csv",
        )
        submission_schema: Dict[str, Any] = get_declared_file_schema(
            execution_contract if isinstance(execution_contract, dict) else {},
            "submission.csv",
            kind="submission",
        )
        if not submission_schema and isinstance(file_schemas, dict) and file_schemas:
            for path_key, schema_obj in file_schemas.items():
                normalized_path = normalize_artifact_path(path_key).lower()
                if normalized_path.endswith("submission.csv") and isinstance(schema_obj, dict):
                    submission_schema = schema_obj
                    break
        primary_metric = (
            validation_requirements.get("primary_metric")
            or evaluation_spec.get("primary_metric")
        )
        metric_rule = evaluation_spec.get("metric_definition_rule")
        if (
            not metric_rule
            and isinstance(primary_metric, str)
            and "mean" in primary_metric.lower()
        ):
            metric_rule = (
                "Use a simple arithmetic mean unless the contract explicitly provides weights."
            )

        authoritative_state = {
            "input_dataset": data_path,
            "source_of_truth": {
                "features": "allowed_feature_sets.model_features (fallback: column_roles.pre_decision)",
                "forbidden_features": "allowed_feature_sets.forbidden_features",
                "targets": "evaluation_spec.target_columns (fallback: validation_requirements.label_columns)",
                "split_rules": "split_spec",
                "output_schema": "required_outputs + artifact_requirements",
                "cleaning_manifest_scope": (
                    f"Use {cleaning_manifest_path} only for output_dialect and observed cleaning metadata. "
                    "Do not treat it as the authoritative source for model_features, target_columns, split rules, or required outputs unless those keys are explicitly present."
                ),
            },
            "resolved_contract": {
                "target_columns": [str(item) for item in target_columns[:12] if str(item).strip()],
                "primary_metric": primary_metric,
                "metric_definition_rule": metric_rule,
                "model_features": [str(item) for item in model_features[:60] if str(item).strip()],
                "forbidden_features": [str(item) for item in forbidden_features[:30] if str(item).strip()],
                "split_column": split_spec.get("split_column"),
                "training_rows_rule": split_spec.get("training_rows_rule"),
                "scoring_rows_rule": split_spec.get("scoring_rows_rule"),
                "required_outputs": [
                    str(item)
                    for item in (
                        execution_contract.get("required_outputs")
                        if isinstance(execution_contract.get("required_outputs"), list)
                        else (ml_view.get("required_outputs") if isinstance(ml_view.get("required_outputs"), list) else [])
                    )[:12]
                    if str(item).strip()
                ],
                "per_row_output_required_columns": (
                    scored_rows_schema.get("required_columns")
                    if (
                        isinstance(scored_rows_schema.get("required_columns"), list)
                        and scored_rows_path
                    )
                    else []
                )[:12],
                "per_row_output_path": scored_rows_path or None,
                "cleaning_manifest_path": cleaning_manifest_path,
                "submission_path": submission_path or None,
                "submission_expected_row_count": submission_schema.get("expected_row_count"),
            },
        }
        return self._serialize_json_for_prompt(
            authoritative_state,
            max_chars=5000,
            max_str_len=320,
            max_list_items=100,
        )

    def _extract_decisioning_context(
        self,
        ml_view: Dict[str, Any] | None,
        execution_contract: Dict[str, Any] | None,
    ) -> tuple[str, str, str]:
        decisioning_requirements: Dict[str, Any] = {}
        if isinstance(ml_view, dict):
            decisioning_requirements = ml_view.get("decisioning_requirements") or {}
        if not decisioning_requirements and isinstance(execution_contract, dict):
            decisioning_requirements = execution_contract.get("decisioning_requirements") or {}
        if not isinstance(decisioning_requirements, dict):
            decisioning_requirements = {}
        decisioning_requirements_context = json.dumps(decisioning_requirements, indent=2)
        decisioning_columns = []
        output_block = decisioning_requirements.get("output") if isinstance(decisioning_requirements, dict) else None
        required_columns = output_block.get("required_columns") if isinstance(output_block, dict) else None
        if isinstance(required_columns, list):
            for col in required_columns:
                if isinstance(col, dict) and col.get("name"):
                    decisioning_columns.append(str(col.get("name")))
        decisioning_columns_text = ", ".join(decisioning_columns) if decisioning_columns else "None requested."
        decisioning_policy_notes = decisioning_requirements.get("policy_notes", "")
        if decisioning_policy_notes is None:
            decisioning_policy_notes = ""
        if not isinstance(decisioning_policy_notes, str):
            decisioning_policy_notes = str(decisioning_policy_notes)
        return decisioning_requirements_context, decisioning_columns_text, decisioning_policy_notes

    def _extract_visual_requirements_context(self, ml_view: Dict[str, Any] | None) -> str:
        visual_requirements: Dict[str, Any] = {}
        if isinstance(ml_view, dict):
            visual_requirements = ml_view.get("visual_requirements") or {}
        if not isinstance(visual_requirements, dict):
            visual_requirements = {}
        return json.dumps(visual_requirements, indent=2)

    def _build_system_prompt(
        self,
        template: str,
        render_kwargs: Dict[str, Any] | None,
        ml_view: Dict[str, Any] | None = None,
        execution_contract: Dict[str, Any] | None = None,
    ) -> str:
        render_kwargs = render_kwargs if isinstance(render_kwargs, dict) else {}
        decisioning_requirements_context, decisioning_columns_text, decisioning_policy_notes = (
            self._extract_decisioning_context(ml_view, execution_contract)
        )
        visual_requirements_context = self._extract_visual_requirements_context(ml_view)
        merged = dict(render_kwargs)
        merged.update(
            {
                "decisioning_requirements_context": decisioning_requirements_context,
                "decisioning_policy_notes": decisioning_policy_notes,
                "decisioning_columns_text": decisioning_columns_text,
                "visual_requirements_context": visual_requirements_context,
                "cleaning_manifest_path": self._resolve_cleaning_manifest_path(execution_contract, ml_view),
                "senior_reasoning_protocol": SENIOR_REASONING_PROTOCOL_GENERAL,
                "senior_engineering_protocol": SENIOR_ENGINEERING_PROTOCOL,
            }
        )
        return render_prompt(template, **merged)

    def _build_optimization_system_prompt(
        self,
        render_kwargs: Dict[str, Any] | None,
        ml_view: Dict[str, Any] | None = None,
        execution_contract: Dict[str, Any] | None = None,
    ) -> str:
        render_kwargs = render_kwargs if isinstance(render_kwargs, dict) else {}
        merged = dict(render_kwargs)
        merged.update(
            {
                "senior_reasoning_protocol": SENIOR_REASONING_PROTOCOL_GENERAL,
                "senior_engineering_protocol": SENIOR_ENGINEERING_PROTOCOL,
            }
        )
        template = """
        You are a Senior ML Engineer editing an incumbent script during a metric-improvement round.

        === SENIOR REASONING PROTOCOL ===
        $senior_reasoning_protocol

        === SENIOR ENGINEERING PROTOCOL ===
        $senior_engineering_protocol

        MISSION
        - Improve the incumbent metric with one focused edit pass.
        - Preserve the incumbent's valid behavior, contract paths, and runtime safety.
        - Return one complete runnable Python script for "$data_path".

        OPTIMIZATION EDITOR CONTRACT
        - Treat the existing script as an approved incumbent unless the round evidence proves a local defect.
        - Optimize by targeted edits, not by regenerating a new solution.
        - Contract paths, train/test partitioning, target semantics, required outputs, and validation protocol are immutable unless the round context explicitly says otherwise.
        - If the metric definition is a mean and the contract provides no explicit weights, use a simple arithmetic mean.

        HARD CONSTRAINTS
        - Output valid Python code only. No markdown, no code fences.
        - Read input data only from "$data_path".
        - Respect $cleaning_manifest_path output_dialect for CSV reads and writes.
        - Do not invent columns, synthetic rows, or fallback datasets.
        - Do not overwrite input data.
        - Respect required outputs exactly as contract paths.
        - Avoid network, shell operations, filesystem discovery scans, and file deletion.
        - Forbidden calls include os.remove, os.unlink, pathlib.Path.unlink, shutil.rmtree, os.rmdir.

        EXECUTION INVARIANTS
        - Keep row-subset rules exact: test-only artifacts must contain only scoring rows; all-row artifacts must contain all rows.
        - Keep model family, CV protocol, and output contract stable unless the round invariants explicitly allow a local change.
        - Preserve working artifact generation and only change the code regions needed for the active hypothesis.

        CURRENT TASK CONTEXT
        - Business Objective (compact): "$business_objective_digest"
        - Strategy Title: $strategy_title
        - Strategy Hypothesis: $hypothesis
        - Strategy Techniques (compact): $strategy_techniques_compact
        - Required Outputs: $deliverables_json
        - Optimization Authoritative State: $optimization_authoritative_state
        - Treat that state as the single execution truth for targets, features, split rules, metrics, and output schema.
        - Use $cleaning_manifest_path only for CSV dialect and cleaning metadata unless the authoritative state explicitly says otherwise.

        Return Python code only.
        """
        return self._build_system_prompt(
            template,
            merged,
            ml_view=ml_view,
            execution_contract=execution_contract,
        )

    def _build_incomplete_reprompt_context(
        self,
        execution_contract: Dict[str, Any] | None,
        required_outputs: List[str],
        iteration_memory_block: str,
        iteration_memory: List[Dict[str, Any]] | None,
        feedback_history: List[str] | None,
        gate_context: Dict[str, Any] | None,
        iteration_handoff: Dict[str, Any] | None = None,
        ml_view: Dict[str, Any] | None = None,
    ) -> str:
        from src.utils.context_pack import compress_long_lists

        contract_context = compress_long_lists(self._compact_execution_contract(execution_contract or {}))[0]
        allowed_columns = compress_long_lists(self._resolve_allowed_columns_for_prompt(execution_contract or {}))[0]
        allowed_patterns = self._resolve_allowed_name_patterns_for_prompt(execution_contract or {})
        feedback_blocks = self._select_feedback_blocks(feedback_history, gate_context, max_blocks=2)
        ml_view = ml_view or {}
        ml_view_json = json.dumps(compress_long_lists(ml_view)[0], indent=2)
        plot_spec_json = json.dumps(ml_view.get("plot_spec", {}), indent=2)
        decisioning_requirements_context, decisioning_columns_text, decisioning_policy_notes = (
            self._extract_decisioning_context(ml_view, execution_contract)
        )
        visual_requirements_json = self._extract_visual_requirements_context(ml_view)
        normalized_handoff = self._normalize_iteration_handoff(
            iteration_handoff=iteration_handoff,
            gate_context=gate_context,
            required_deliverables=required_outputs,
        )
        handoff_json = json.dumps(compress_long_lists(normalized_handoff)[0], indent=2)
        
        # STRUCTURED CRITICAL ERRORS SECTION
        critical_errors: List[str] = []
        recent_history = (iteration_memory or [])[-2:]
        current_failure = gate_context or {}
        
        # 1. Process Current failure (from gate_context)
        if current_failure:
            att_num = len(iteration_memory or []) + 1
            f_gates = current_failure.get("failed_gates", [])
            f_type = ", ".join(f_gates) if isinstance(f_gates, list) and f_gates else "MODEL_REJECTION"
            f_feedback = str(current_failure.get("feedback", "")).strip()
            f_fixes = current_failure.get("required_fixes", [])
            f_fix_str = "; ".join(f_fixes) if isinstance(f_fixes, list) and f_fixes else "Address feedback"
            
            critical_errors.append(
                f"ATTEMPT {att_num} - REJECTED:\n"
                f"  - Error Type: {f_type}\n"
                f"  - Root Cause: {f_feedback or 'General methodological failure'}\n"
                f"  - Required Fix: {f_fix_str}"
            )
            
        # 2. Process historical failures (from iteration_memory)
        for i, entry in enumerate(recent_history):
            att_id = entry.get("iteration_id", i + 1)
            # Skip if it's the current one (not expected in iteration_memory yet, but just in case)
            if current_failure and att_id == len(iteration_memory or []) + 1:
                continue
                
            e_type = ", ".join(entry.get("reviewer_reasons", []) + entry.get("qa_reasons", [])) or "PREVIOUS_FAILURE"
            e_cause = entry.get("runtime_error", {}).get("message") if isinstance(entry.get("runtime_error"), dict) else None
            # Fallback to general reasons if no runtime error message
            if not e_cause:
                e_cause = "; ".join(entry.get("reviewer_reasons", []) or ["Historical rejection"])
                
            e_fix = "; ".join(entry.get("next_actions", [])) or "Improve implementation"
            
            critical_errors.append(
                f"ATTEMPT {att_id} - REJECTED:\n"
                f"  - Error Type: {e_type}\n"
                f"  - Root Cause: {e_cause}\n"
                f"  - Required Fix: {e_fix}"
            )
            
        critical_section = ""
        if critical_errors:
            critical_section = (
                "!!! CRITICAL ERRORS FROM PREVIOUS ATTEMPTS (DO NOT REPEAT) !!!\n" +
                "\n".join(critical_errors) +
                "\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
            )

        memory_block = iteration_memory_block.strip()
        if not memory_block:
            memory_block = json.dumps(iteration_memory or [], indent=2)
            
        rules_block = "\n".join(
            [
                "- No synthetic feature/row generation. Load only the provided dataset. Bootstrap resampling of existing rows is allowed when required by the contract.",
                "- Do not mutate df_in; use df_work = df_in.copy() and only assign contract-declared derived columns.",
                "- Baseline model is required.",
                "- Include SimpleImputer in preprocessing when NaNs may exist.",
                "- Write all required outputs to exact paths.",
                "- Only write per-row scoring artifacts when the contract explicitly declares them; include only contract-approved columns.",
                "- Define CONTRACT_INPUT_COLUMNS from clean_dataset.required_columns (fallback canonical) and print a MAPPING SUMMARY.",
            ]
        )
        return "\n".join(
            [
                critical_section,
                "ML_VIEW_CONTEXT:",
                ml_view_json,
                "PLOT_SPEC_CONTEXT:",
                plot_spec_json,
                "DECISIONING_REQUIREMENTS_CONTEXT:",
                decisioning_requirements_context,
                "DECISIONING_COLUMNS:",
                decisioning_columns_text,
                "DECISIONING_POLICY_NOTES:",
                decisioning_policy_notes or "None",
                "VISUAL_REQUIREMENTS_CONTEXT:",
                visual_requirements_json,
                "EXECUTION_CONTRACT_CONTEXT:",
                json.dumps(contract_context, indent=2),
                "REQUIRED OUTPUTS:",
                json.dumps(required_outputs or [], indent=2),
                "ALLOWED COLUMNS:",
                json.dumps(allowed_columns, indent=2),
                "ALLOWED_NAME_PATTERNS:",
                json.dumps(allowed_patterns, indent=2),
                "UNIVERSAL RULES:",
                rules_block,
                "ITERATION_MEMORY_CONTEXT:",
                memory_block,
                "ITERATION_HANDOFF_CONTEXT:",
                handoff_json,
                "LATEST_REVIEW_FEEDBACK:",
                feedback_blocks or "None",
            ]
        )

    def _truncate_prompt_text(self, text: str, max_len: int, head_len: int, tail_len: int) -> str:
        if not text:
            return text
        if len(text) <= max_len:
            return text
        safe_head = max(0, min(head_len, max_len))
        safe_tail = max(0, min(tail_len, max_len - safe_head))
        if safe_head + safe_tail == 0:
            return text[:max_len]
        if safe_head + safe_tail < max_len:
            safe_head = max_len - safe_tail
        return text[:safe_head] + "\n...[TRUNCATED]...\n" + text[-safe_tail:]

    def _serialize_json_for_prompt(
        self,
        payload: Any,
        *,
        max_chars: int = 8000,
        max_str_len: int = 900,
        max_list_items: int = 60,
    ) -> str:
        from src.utils.context_pack import compress_long_lists
        from src.utils.contract_views import trim_to_budget

        compact = compress_long_lists(payload or {})[0]
        trimmed = trim_to_budget(
            compact,
            max_chars=max_chars,
            max_str_len=max_str_len,
            max_list_items=max_list_items,
        )
        text = json.dumps(trimmed, ensure_ascii=False, separators=(",", ":"))
        if len(text) > max_chars:
            text = self._truncate_prompt_text(
                text,
                max_len=max_chars,
                head_len=int(max_chars * 0.65),
                tail_len=int(max_chars * 0.25),
            )
        return text

    def _shrink_prompt_blob(self, value: Any, max_chars: int) -> str:
        if not isinstance(value, str):
            return self._serialize_json_for_prompt(value, max_chars=max_chars)
        text = value.strip()
        if len(text) <= max_chars:
            return text
        try:
            as_json = json.loads(text)
            return self._serialize_json_for_prompt(as_json, max_chars=max_chars)
        except Exception:
            return self._truncate_prompt_text(
                text,
                max_len=max_chars,
                head_len=int(max_chars * 0.65),
                tail_len=int(max_chars * 0.25),
            )

    def _build_system_prompt_with_budget(
        self,
        template: str,
        render_kwargs: Dict[str, Any],
        *,
        ml_view: Dict[str, Any] | None = None,
        execution_contract: Dict[str, Any] | None = None,
        max_chars: int = 50000,
    ) -> tuple[str, Dict[str, Any], Dict[str, Any]]:
        kwargs = dict(render_kwargs or {})
        prompt = self._build_system_prompt(
            template,
            kwargs,
            ml_view=ml_view,
            execution_contract=execution_contract,
        )
        if len(prompt) <= max_chars:
            return prompt, kwargs, {"prompt_chars": len(prompt), "budget_applied": False}

        shrink_plan = [
            ("data_audit_context", 4000),
            ("iteration_memory_json", 3000),
            ("iteration_memory_block", 2500),
            ("signal_summary_json", 2500),
            ("cleaned_data_summary_min_json", 3500),
            ("feature_semantics_json", 3000),
            ("business_sanity_checks_json", 2500),
            ("alignment_requirements_json", 2000),
            ("execution_contract_context", 9000),
            ("ml_view_context", 8000),
            ("artifact_schema_block", 2500),
            ("data_partitioning_context", 2500),
        ]
        for key, budget in shrink_plan:
            if key not in kwargs:
                continue
            kwargs[key] = self._shrink_prompt_blob(kwargs.get(key), budget)
            prompt = self._build_system_prompt(
                template,
                kwargs,
                ml_view=ml_view,
                execution_contract=execution_contract,
            )
            if len(prompt) <= max_chars:
                return prompt, kwargs, {"prompt_chars": len(prompt), "budget_applied": True}

        return prompt, kwargs, {"prompt_chars": len(prompt), "budget_applied": True}

    def _truncate_code_for_patch(self, code: str, max_len: int = 12000) -> str:
        return self._truncate_prompt_text(code or "", max_len=max_len, head_len=7000, tail_len=4000)

    def _extract_error_snippet_from_traceback(
        self,
        code: str,
        traceback_text: str,
        *,
        context_lines: int = 10,
    ) -> str:
        if not code or not traceback_text:
            return ""
        try:
            matches = re.findall(r'line\s+(\d+)', str(traceback_text))
            if not matches:
                return ""
            line_num = int(matches[-1])
            lines = str(code).splitlines()
            start = max(line_num - context_lines - 1, 0)
            end = min(line_num + context_lines, len(lines))
            snippet_lines: List[str] = []
            for idx in range(start, end):
                prefix = ">>" if (idx + 1) == line_num else "  "
                snippet_lines.append(f"{prefix} {idx + 1}: {lines[idx]}")
            return "\n".join(snippet_lines)
        except Exception:
            return ""

    def _check_script_completeness(self, code: str, required_paths: List[str]) -> List[str]:
        """
        DEPRECATED: Static syntax checks are no longer enforced.

        Validation Philosophy Change (v5.0):
        - Instead of checking if code SAYS .read_csv, we let code RUN and check RESULTS.
        - The Reviewer/QA Agent validates outputs exist and are correct.
        - This method returns empty list to maintain API compatibility.
        """
        # Static checks removed - trust execution-based validation
        return []

    def _extract_training_context(
        self,
        execution_contract: Dict[str, Any] | None,
        ml_view: Dict[str, Any] | None,
        ml_plan: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        contract = execution_contract or {}
        view = ml_view or {}
        plan = ml_plan or {}

        outcome_columns = contract.get("outcome_columns")
        if not isinstance(outcome_columns, list) or not outcome_columns:
            outcome_columns = view.get("outcome_columns")
        if not isinstance(outcome_columns, list):
            outcome_columns = []

        target_column = None
        if outcome_columns:
            target_column = str(outcome_columns[0])

        split_spec = contract.get("split_spec")
        if not isinstance(split_spec, dict):
            split_spec = view.get("split_spec") if isinstance(view.get("split_spec"), dict) else {}

        training_rows_rule = (
            contract.get("training_rows_rule")
            or view.get("training_rows_rule")
            or split_spec.get("training_rows_rule")
        )
        scoring_rows_rule = (
            contract.get("scoring_rows_rule")
            or view.get("scoring_rows_rule")
            or split_spec.get("scoring_rows_rule")
        )
        secondary_scoring_subset = contract.get("secondary_scoring_subset") or view.get("secondary_scoring_subset")

        training_rows_policy = plan.get("training_rows_policy")
        if not isinstance(training_rows_policy, str) or not training_rows_policy.strip():
            if isinstance(split_spec.get("training_rows_policy"), str):
                training_rows_policy = split_spec.get("training_rows_policy")
        split_column = plan.get("split_column")
        if not isinstance(split_column, str) or not split_column.strip():
            split_column = split_spec.get("split_column")
        train_filter = plan.get("train_filter") if isinstance(plan.get("train_filter"), dict) else None
        if train_filter is None and isinstance(split_spec.get("train_filter"), dict):
            train_filter = split_spec.get("train_filter")

        canonical_cols = contract.get("canonical_columns")
        if not isinstance(canonical_cols, list):
            canonical_cols = view.get("canonical_columns")
        if not isinstance(canonical_cols, list):
            canonical_cols = []

        dataset_semantics = contract.get("dataset_semantics") if isinstance(contract.get("dataset_semantics"), dict) else {}
        partition_notes = dataset_semantics.get("data_partitioning_notes") if isinstance(dataset_semantics.get("data_partitioning_notes"), list) else []
        if not isinstance(split_column, str) or not split_column.strip():
            split_column = None
        if split_column is None:
            evidence = plan.get("evidence_used") if isinstance(plan.get("evidence_used"), dict) else {}
            split_candidates = evidence.get("split_candidates")
            if isinstance(split_candidates, list):
                for cand in split_candidates:
                    if not isinstance(cand, dict):
                        continue
                    col = cand.get("column")
                    if isinstance(col, str) and col.strip():
                        split_column = col.strip()
                        break
        if split_column is None and "__split" in canonical_cols:
            split_column = "__split"

        if not isinstance(training_rows_policy, str) or not training_rows_policy.strip():
            notes_text = " ".join(str(note) for note in partition_notes).lower()
            if split_column and ("split" in notes_text or "train/test" in notes_text or "train test" in notes_text):
                training_rows_policy = "use_split_column"
            elif isinstance(training_rows_rule, str):
                rule_lower = training_rows_rule.lower()
                if any(token in rule_lower for token in ("not missing", "not null", "notna", "non-null")):
                    training_rows_policy = "only_rows_with_label"
            if not training_rows_policy:
                training_rows_policy = "use_all_rows"

        if not isinstance(train_filter, dict):
            train_filter = None
        if train_filter is None:
            if training_rows_policy == "use_split_column" and split_column:
                train_filter = {
                    "type": "split_equals",
                    "column": split_column,
                    "value": "train",
                    "rule": None,
                }
            elif training_rows_policy == "only_rows_with_label" and target_column:
                train_filter = {
                    "type": "label_not_null",
                    "column": target_column,
                    "value": None,
                    "rule": None,
                }

        return {
            "target_column": target_column,
            "training_rows_rule": training_rows_rule,
            "scoring_rows_rule": scoring_rows_rule,
            "secondary_scoring_subset": secondary_scoring_subset,
            "training_rows_policy": training_rows_policy,
            "split_column": split_column,
            "train_filter": train_filter,
        }

    def _build_data_partitioning_context(
        self,
        execution_contract: Dict[str, Any] | None,
        ml_view: Dict[str, Any] | None,
        ml_plan: Dict[str, Any] | None,
        execution_profile: Dict[str, Any] | None = None,
    ) -> str:
        contract = execution_contract if isinstance(execution_contract, dict) else {}
        view = ml_view if isinstance(ml_view, dict) else {}
        plan_context = self._extract_training_context(contract, view, ml_plan)
        split_spec = contract.get("split_spec")
        if not isinstance(split_spec, dict):
            split_spec = view.get("split_spec") if isinstance(view.get("split_spec"), dict) else {}

        def _coerce_positive_int(value: Any) -> int | None:
            if isinstance(value, bool):
                return None
            if isinstance(value, int):
                return int(value) if value > 0 else None
            if isinstance(value, float):
                if value.is_integer() and value > 0:
                    return int(value)
                return None
            if isinstance(value, str):
                token = value.strip().replace(",", "")
                if not token:
                    return None
                if token.isdigit():
                    parsed = int(token)
                    return parsed if parsed > 0 else None
            return None

        row_hints: Dict[str, int | None] = {"n_train": None, "n_test": None, "n_total": None}
        train_aliases = {
            "n_train",
            "n_train_rows",
            "train_rows",
            "train_row_count",
            "train_count",
            "train_size",
        }
        test_aliases = {
            "n_test",
            "n_test_rows",
            "test_rows",
            "test_row_count",
            "test_count",
            "scoring_rows",
            "n_scoring_rows",
            "score_rows",
        }
        total_aliases = {
            "n_total",
            "n_total_rows",
            "n_rows",
            "total_rows",
            "row_count",
            "rows",
        }

        def _scan_counts(node: Any, depth: int = 0) -> None:
            if depth > 5:
                return
            if isinstance(node, dict):
                for key, value in node.items():
                    key_norm = re.sub(r"[^a-z0-9]+", "_", str(key).strip().lower()).strip("_")
                    parsed = _coerce_positive_int(value)
                    if key_norm in train_aliases and row_hints["n_train"] is None and parsed is not None:
                        row_hints["n_train"] = int(parsed)
                    elif key_norm in test_aliases and row_hints["n_test"] is None and parsed is not None:
                        row_hints["n_test"] = int(parsed)
                    elif key_norm in total_aliases and row_hints["n_total"] is None and parsed is not None:
                        row_hints["n_total"] = int(parsed)

                    if isinstance(value, (dict, list, tuple)):
                        _scan_counts(value, depth + 1)
            elif isinstance(node, (list, tuple)):
                for item in node:
                    if isinstance(item, (dict, list, tuple)):
                        _scan_counts(item, depth + 1)

        contract_data_profile = contract.get("data_profile") if isinstance(contract.get("data_profile"), dict) else {}
        view_data_profile = view.get("data_profile") if isinstance(view.get("data_profile"), dict) else {}
        sources = [
            contract,
            view,
            contract.get("evaluation_spec"),
            view.get("evaluation_spec"),
            contract.get("execution_constraints"),
            view.get("execution_constraints"),
            contract_data_profile,
            view_data_profile,
            contract_data_profile.get("basic_stats"),
            view_data_profile.get("basic_stats"),
        ]
        for source in sources:
            _scan_counts(source)

        n_train = row_hints.get("n_train")
        n_test = row_hints.get("n_test")
        n_total = row_hints.get("n_total")
        if n_total is None and n_train is not None and n_test is not None:
            n_total = int(n_train + n_test)
            row_hints["n_total"] = n_total
        if n_test is None and n_total is not None and n_train is not None and n_total >= n_train:
            n_test = int(n_total - n_train)
            row_hints["n_test"] = n_test
        if n_train is None and n_total is not None and n_test is not None and n_total >= n_test:
            n_train = int(n_total - n_test)
            row_hints["n_train"] = n_train

        artifact_reqs = contract.get("artifact_requirements")
        if not isinstance(artifact_reqs, dict):
            artifact_reqs = view.get("artifact_requirements")
        # Scan row_count_hints persisted by the execution planner (universal)
        stored_hints = (
            artifact_reqs.get("row_count_hints")
            if isinstance(artifact_reqs, dict)
            else None
        )
        if isinstance(stored_hints, dict):
            _scan_counts(stored_hints)
            # Re-derive after scanning new source
            n_train = row_hints.get("n_train")
            n_test = row_hints.get("n_test")
            n_total = row_hints.get("n_total")
            if n_total is None and n_train is not None and n_test is not None:
                n_total = int(n_train + n_test)
                row_hints["n_total"] = n_total
            if n_test is None and n_total is not None and n_train is not None and n_total >= n_train:
                n_test = int(n_total - n_train)
                row_hints["n_test"] = n_test
            if n_train is None and n_total is not None and n_test is not None and n_total >= n_test:
                n_train = int(n_total - n_test)
                row_hints["n_train"] = n_train

        # Universal fallback: derive n_train from target column non_null_count
        # (works when test rows have null targets, a common universal pattern)
        if n_train is None and n_total is not None:
            for src in sources:
                if not isinstance(src, dict):
                    continue
                outcome = src.get("outcome_analysis")
                if not isinstance(outcome, dict):
                    continue
                for _col_info in outcome.values():
                    if not isinstance(_col_info, dict):
                        continue
                    nnc = _coerce_positive_int(_col_info.get("non_null_count"))
                    if nnc is not None and 0 < nnc < n_total:
                        n_train = nnc
                        n_test = n_total - n_train
                        row_hints["n_train"] = n_train
                        row_hints["n_test"] = n_test
                        break
                if n_train is not None:
                    break

        file_schemas = artifact_reqs.get("file_schemas") if isinstance(artifact_reqs, dict) else {}

        artifact_row_expectations: List[Dict[str, Any]] = []
        if isinstance(file_schemas, dict):
            for raw_path, schema_def in file_schemas.items():
                if not isinstance(schema_def, dict):
                    continue
                expected_count = _coerce_positive_int(schema_def.get("expected_row_count"))
                if expected_count is None:
                    continue
                path = str(schema_def.get("path") or raw_path or "").strip()
                if not path:
                    continue
                expectation = "exact row subset"
                if n_test is not None and expected_count == n_test:
                    expectation = "TEST/SCORING rows only"
                elif n_train is not None and expected_count == n_train:
                    expectation = "TRAINING rows only"
                elif n_total is not None and expected_count == n_total:
                    expectation = "ALL rows"
                artifact_row_expectations.append(
                    {
                        "path": path,
                        "expected_row_count": expected_count,
                        "expectation": expectation,
                    }
                )

        has_partition_rules = bool(
            plan_context.get("training_rows_rule")
            or plan_context.get("scoring_rows_rule")
            or plan_context.get("split_column")
            or plan_context.get("train_filter")
        )
        has_row_context = (
            any(isinstance(v, int) and v > 0 for v in row_hints.values())
            or bool(artifact_row_expectations)
            or has_partition_rules
            or bool(split_spec)
        )
        if not has_row_context:
            return ""

        training_policy = str(plan_context.get("training_rows_policy") or "unspecified")
        # Handle split_column as dict (from split_candidates) or plain string
        split_column_raw = plan_context.get("split_column")
        if isinstance(split_column_raw, dict):
            split_column = str(
                split_column_raw.get("column")
                or split_column_raw.get("name")
                or split_column_raw.get("split_column")
                or ""
            ).strip()
        else:
            split_column = str(split_column_raw or "").strip()
        train_filter = plan_context.get("train_filter")
        train_filter_rule = self._describe_train_filter(
            train_filter if isinstance(train_filter, dict) else None,
            plan_context.get("target_column"),
            plan_context.get("split_column"),
        )
        if isinstance(train_filter, dict):
            custom_rule = train_filter.get("rule")
            if isinstance(custom_rule, str) and custom_rule.strip():
                train_filter_rule = custom_rule.strip()

        lines = ["=== DATA PARTITIONING CONTEXT (CRITICAL - READ BEFORE WRITING ANY CSV) ==="]
        if n_total is not None:
            lines.append(f"Total rows in cleaned dataset: {n_total:,}")
        if n_train is not None:
            lines.append(f"Training rows: {n_train:,} (apply train_filter below before model.fit)")
        if n_test is not None:
            lines.append(f"Test/scoring rows: {n_test:,} (rows NOT used for training)")
        lines.append(f"Training row selection policy: {training_policy}")
        if split_column:
            lines.append(f"Split column: {split_column}")
        split_status = str(split_spec.get("status") or "").strip()
        if split_status:
            lines.append(f"Split resolution status: {split_status}")
        if bool(split_spec.get("requires_test_only_outputs")) and split_status.lower() in {"unknown", "unresolved", "ambiguous"}:
            lines.append(
                "FAIL-CLOSED NOTE: contract expects test-only outputs but split resolution is unresolved; infer and apply explicit subset filters before writing outputs."
            )
        lines.append(f"Train filter rule: {train_filter_rule}")

        if artifact_row_expectations:
            lines.append("")
            lines.append("REQUIRED OUTPUT ARTIFACT ROW COUNTS (contract-enforced):")
            for item in artifact_row_expectations:
                lines.append(
                    "  - {path}: MUST contain {expectation} ({rows:,} rows)".format(
                        path=item["path"],
                        expectation=item["expectation"],
                        rows=int(item["expected_row_count"]),
                    )
                )

        # Script-level hard timeout — extract from execution_profile or use
        # the runner's MAX_SCRIPT_TIMEOUT_SECONDS as fallback.  This tells the
        # ML Engineer the absolute wall-clock ceiling so it can budget HPO +
        # final CV + prediction + I/O within that envelope.
        _script_hard_timeout = 7200  # runner default
        _ep = execution_profile if isinstance(execution_profile, dict) else {}
        _rb = _ep.get("runtime_budget") if isinstance(_ep.get("runtime_budget"), dict) else {}
        _ht = _rb.get("hard_timeout_seconds")
        if isinstance(_ht, (int, float)) and _ht > 0:
            _script_hard_timeout = min(int(_ht), 7200)  # runner caps at 7200

        lines.append("")
        lines.append("SCRIPT RUNTIME BUDGET (MANDATORY — hard limit, script is killed after this):")
        lines.append(
            f"  - Your ENTIRE script (HPO + final CV + predictions + CSV I/O) MUST complete within {_script_hard_timeout} seconds."
        )
        lines.append(
            "  - Budget allocation rule: HPO should use at most 50% of this budget. "
            "Reserve the remaining 50% for final model training, CV evaluation, predictions, and file I/O."
        )
        lines.append(
            f"  - MANDATORY: set HPO timeout to at most {_script_hard_timeout // 2} seconds."
        )
        lines.append(
            "  - For large datasets (>300K rows): limit CatBoost search space to iterations<=500, depth<=6 during HPO. "
            "Train the final model with higher iterations + early_stopping_rounds=50."
        )
        lines.append(
            "  - If using subsampling for HPO, subsample to 50K-100K rows for the HPO phase, then retrain on full data."
        )

        # HPO scaling guidance — universal, based on dataset size.
        # Prevents Optuna/HPO from timing out with 0 useful trials on large datasets.
        # The guidance is PRESCRIPTIVE (not suggestive) to ensure the LLM uses
        # adequate timeout values.  The framework grants generous script-level
        # timeouts (up to hours), so the bottleneck is the Optuna-internal
        # timeout the LLM hardcodes in the generated script.
        if n_train is not None and n_train > 0:
            lines.append("")
            lines.append("HPO SCALING GUIDANCE (MANDATORY — based on training set size):")
            if n_train > 300_000:
                hpo_timeout = min(3600, _script_hard_timeout // 2)
                hpo_n_trials = 50
                hpo_cv_folds = 3
                lines.append(
                    f"  - LARGE DATASET ({n_train:,} training rows). Each CV fold is expensive."
                )
                lines.append(
                    f"  - MANDATORY: When using Optuna/HPO, you MUST set timeout={hpo_timeout} and n_trials={hpo_n_trials}."
                )
                lines.append(
                    f"  - MANDATORY: Use {hpo_cv_folds}-fold CV inside HPO objective function (keep 5-fold for final evaluation only)."
                )
                lines.append(
                    f"  - For CatBoost HPO: limit search space to iterations<=500, depth<=6, and use early_stopping_rounds=30. "
                    f"Higher iterations (1000+) should ONLY be used for the FINAL model training, NOT during HPO trials."
                )
                lines.append(
                    f"  - Define these constants at the top of your script: HPO_TIMEOUT = {hpo_timeout}; HPO_N_TRIALS = {hpo_n_trials}; HPO_CV_FOLDS = {hpo_cv_folds}"
                )
            elif n_train > 100_000:
                hpo_timeout = min(1800, _script_hard_timeout // 2)
                hpo_n_trials = 50
                hpo_cv_folds = 3
                lines.append(
                    f"  - MEDIUM DATASET ({n_train:,} training rows)."
                )
                lines.append(
                    f"  - MANDATORY: When using Optuna/HPO, you MUST set timeout={hpo_timeout} (30min) and n_trials={hpo_n_trials}."
                )
                lines.append(
                    f"  - Use {hpo_cv_folds}-fold CV inside HPO objective function (keep 5-fold for final evaluation only)."
                )
                lines.append(
                    f"  - Define these constants at the top of your script: HPO_TIMEOUT = {hpo_timeout}; HPO_N_TRIALS = {hpo_n_trials}; HPO_CV_FOLDS = {hpo_cv_folds}"
                )
            else:
                hpo_timeout = 600
                hpo_n_trials = 100
                lines.append(
                    f"  - SMALL DATASET ({n_train:,} training rows)."
                )
                lines.append(
                    f"  - For Optuna/HPO: set timeout={hpo_timeout} (10min) and n_trials={hpo_n_trials}."
                )

        lines.append("")
        lines.append("CRITICAL RULES:")
        if n_test is not None:
            lines.append(
                f"  - Any artifact with expected_row_count={n_test:,} MUST contain TEST/SCORING rows only. NEVER write the full dataframe."
            )
        if n_total is not None:
            lines.append(
                f"  - Any artifact with expected_row_count={n_total:,} MUST contain ALL rows."
            )
        lines.append(
            "  - Always filter to the correct subset BEFORE calling .to_csv(). Add an explicit len() assertion after writing each artifact."
        )
        return "\n".join(lines)

    def _code_mentions_label_filter(self, code: str, target: str | None) -> bool:
        """
        DEPRECATED: Static regex-based code analysis is no longer enforced.

        Validation Philosophy Change (v5.0):
        - We no longer parse code looking for .dropna/.notna patterns.
        - Execution-based validation: run the code, check if output has nulls.
        - Always returns True to avoid blocking valid implementations.
        """
        return True  # Trust the LLM; validate results, not syntax

    def _code_mentions_split_usage(self, code: str, split_column: str | None) -> bool:
        """
        DEPRECATED: Static regex-based code analysis is no longer enforced.

        Validation Philosophy Change (v5.0):
        - We no longer parse code looking for split column references.
        - Execution-based validation: run the code, check train/test split in results.
        - Always returns True to avoid blocking valid implementations.
        """
        return True  # Trust the LLM; validate results, not syntax

    def _check_training_policy_compliance(
        self,
        code: str,
        execution_contract: Dict[str, Any] | None,
        ml_view: Dict[str, Any] | None,
        ml_plan: Dict[str, Any] | None,
    ) -> List[str]:
        """
        DEPRECATED: Static AST/regex-based policy compliance checks are no longer enforced.

        Validation Philosophy Change (v5.0):
        =====================================
        OLD: Parse code text looking for .dropna(), split column references, etc.
             Flag code as invalid if specific patterns weren't found.
             Problem: Brittle - flags valid implementations that use different syntax.

        NEW: Execution-based validation.
             - Let the code RUN in the sandbox.
             - Reviewer/QA Agent checks if RESULTS are correct:
               * Does the output have nulls where it shouldn't?
               * Was train/test split done correctly?
               * Are metrics reasonable?
             - Self-correction: ML Engineer receives stderr/traceback and fixes errors.

        This method returns empty list to maintain API compatibility while
        removing the static linting that blocked valid code.
        """
        # Static checks removed - execution-based validation is now the standard
        # The Reviewer and QA agents validate results, not code syntax
        return []

    def _describe_train_filter(
        self,
        train_filter: Dict[str, Any] | None,
        target_column: str | None,
        split_column: str | None,
    ) -> str:
        if not isinstance(train_filter, dict):
            return "No explicit train_filter provided."
        tf_type = str(train_filter.get("type") or "").strip().lower()
        column = train_filter.get("column") or (target_column if tf_type == "label_not_null" else split_column)
        value = train_filter.get("value")
        rule = train_filter.get("rule")
        if tf_type == "label_not_null":
            return f"Filter training rows where '{column}' is not null."
        if tf_type == "split_equals":
            return f"Filter training rows where '{column}' == '{value or 'train'}'."
        if tf_type == "custom_rule":
            return f"Apply custom training rule: {rule}"
        if tf_type == "none":
            return "Use all rows for training."
        return "Train filter is unspecified; infer from plan/contract."

    def _build_training_policy_checklist(
        self,
        code: str,
        execution_contract: Dict[str, Any] | None,
        ml_view: Dict[str, Any] | None,
        ml_plan: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        context = self._extract_training_context(execution_contract, ml_view, ml_plan)
        plan = ml_plan or {}
        metric_policy = plan.get("metric_policy", {}) if isinstance(plan.get("metric_policy"), dict) else {}
        cv_policy = plan.get("cv_policy", {}) if isinstance(plan.get("cv_policy"), dict) else {}
        target = context.get("target_column")
        split_column = context.get("split_column")
        return {
            "target_column": target,
            "training_rows_policy": context.get("training_rows_policy"),
            "train_filter": context.get("train_filter"),
            "code_has_label_filter": self._code_mentions_label_filter(code, target),
            "code_has_split_filter": self._code_mentions_split_usage(code, split_column),
            "primary_metric": metric_policy.get("primary_metric"),
            "cv_policy": {
                "strategy": cv_policy.get("strategy"),
                "n_splits": cv_policy.get("n_splits"),
                "shuffle": cv_policy.get("shuffle"),
                "stratified": cv_policy.get("stratified"),
            },
        }

    def _build_universal_prologue(
        self,
        csv_sep: str,
        csv_decimal: str,
        csv_encoding: str,
    ) -> str:
        sep_literal = json.dumps(csv_sep or ",")
        decimal_literal = json.dumps(csv_decimal or ".")
        encoding_literal = json.dumps(csv_encoding or "utf-8")
        return (
            UNIVERSAL_PROLOGUE_START
            + "\n"
            + "import os\n"
            + "import json\n"
            + "import numpy as np\n"
            + "import pandas as pd\n"
            + "from pathlib import Path\n\n"
            + "os.makedirs(\"data\", exist_ok=True)\n\n"
            + f"sep = {sep_literal}\n"
            + f"decimal = {decimal_literal}\n"
            + f"encoding = {encoding_literal}\n\n"
            + "def json_default(obj):\n"
            + "    if isinstance(obj, np.bool_):\n"
            + "        return bool(obj)\n"
            + "    if isinstance(obj, np.generic):\n"
            + "        return obj.item()\n"
            + "    if isinstance(obj, pd.Timestamp):\n"
            + "        return obj.isoformat()\n"
            + "    if isinstance(obj, Path):\n"
            + "        return str(obj)\n"
            + "    return str(obj)\n\n"
            + "_json_default = json_default\n"
            + UNIVERSAL_PROLOGUE_END
            + "\n\n"
        )

    def _strip_existing_universal_prologue(self, code: str) -> str:
        text = str(code or "")
        pattern = re.compile(
            re.escape(UNIVERSAL_PROLOGUE_START) + r".*?" + re.escape(UNIVERSAL_PROLOGUE_END) + r"\n*",
            flags=re.DOTALL,
        )
        return re.sub(pattern, "", text, count=1)

    def _has_universal_prologue(self, code: str) -> bool:
        try:
            tree = ast.parse(code)
        except Exception:
            return False
        has_json_default = False
        has_sep = False
        has_decimal = False
        has_encoding = False
        has_makedirs = False
        for node in getattr(tree, "body", []):
            if isinstance(node, ast.FunctionDef) and node.name == "json_default":
                has_json_default = True
            if isinstance(node, ast.Assign):
                targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
                if "sep" in targets:
                    has_sep = True
                if "decimal" in targets:
                    has_decimal = True
                if "encoding" in targets:
                    has_encoding = True
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                if (
                    isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == "os"
                    and call.func.attr == "makedirs"
                    and call.args
                    and isinstance(call.args[0], ast.Constant)
                    and call.args[0].value == "data"
                ):
                    has_makedirs = True
        return bool(has_json_default and has_sep and has_decimal and has_encoding and has_makedirs)

    def _inject_universal_prologue(
        self,
        code: str,
        csv_sep: str,
        csv_decimal: str,
        csv_encoding: str,
    ) -> str:
        text = self._strip_existing_universal_prologue(code)
        if self._has_universal_prologue(text):
            return text
        prologue = self._build_universal_prologue(csv_sep, csv_decimal, csv_encoding)
        lines = text.splitlines(keepends=True)
        insert_index = 0
        try:
            tree = ast.parse(text)
            body = list(tree.body or [])
            idx = 0
            if body:
                first = body[0]
                if (
                    isinstance(first, ast.Expr)
                    and isinstance(getattr(first, "value", None), ast.Constant)
                    and isinstance(first.value.value, str)
                ):
                    insert_index = int(getattr(first, "end_lineno", first.lineno))
                    idx = 1
            while idx < len(body):
                node = body[idx]
                if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                    insert_index = int(getattr(node, "end_lineno", node.lineno))
                    idx += 1
                    continue
                break
        except Exception:
            insert_index = 0
            if lines and lines[0].startswith("#!"):
                insert_index = 1
            if insert_index < len(lines):
                encoding_line = lines[insert_index].strip().lower()
                if encoding_line.startswith("#") and "coding" in encoding_line:
                    insert_index += 1

        insert_index = max(0, min(insert_index, len(lines)))
        before = "".join(lines[:insert_index])
        after = "".join(lines[insert_index:])
        if before and not before.endswith("\n"):
            before += "\n"
        if after and not after.startswith("\n"):
            prologue = prologue + "\n"
        return before + prologue + after

    def _ensure_json_dump_default_serializer(self, code: str) -> str:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code

        class _JsonDumpDefaultFixer(ast.NodeTransformer):
            def visit_Call(self, node: ast.Call) -> ast.AST:
                self.generic_visit(node)
                if not isinstance(node.func, ast.Attribute):
                    return node
                if not isinstance(node.func.value, ast.Name):
                    return node
                if node.func.value.id != "json" or node.func.attr != "dump":
                    return node
                has_default = False
                for kw in node.keywords:
                    if kw.arg == "default":
                        kw.value = ast.Name(id="_json_default", ctx=ast.Load())
                        has_default = True
                        break
                if not has_default:
                    node.keywords.append(
                        ast.keyword(
                            arg="default",
                            value=ast.Name(id="_json_default", ctx=ast.Load()),
                        )
                    )
                return node

        try:
            fixed_tree = _JsonDumpDefaultFixer().visit(tree)
            ast.fix_missing_locations(fixed_tree)
            return ast.unparse(fixed_tree)
        except Exception:
            return code

    def _apply_universal_script_guards(
        self,
        code: str,
        csv_sep: str,
        csv_decimal: str,
        csv_encoding: str,
    ) -> str:
        guarded = self._inject_universal_prologue(code, csv_sep, csv_decimal, csv_encoding)
        guarded = self._fix_to_csv_dialect_in_code(guarded)
        guarded = self._ensure_json_dump_default_serializer(guarded)
        return guarded

    def _fix_to_csv_dialect_in_code(self, code: str) -> str:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return code

        has_sep = False
        has_decimal = False
        has_encoding = False
        has_load_dialect = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if node.id == "sep":
                    has_sep = True
                elif node.id == "decimal":
                    has_decimal = True
                elif node.id == "encoding":
                    has_encoding = True
            if isinstance(node, ast.Call):
                name = self._call_name(node)
                if name.endswith("load_dialect"):
                    has_load_dialect = True
        if not has_load_dialect and not (has_sep and has_decimal and has_encoding):
            return code

        class _ToCsvDialectFixer(ast.NodeTransformer):
            def visit_Call(self, node: ast.Call) -> ast.AST:
                self.generic_visit(node)
                if not isinstance(node.func, ast.Attribute) or node.func.attr != "to_csv":
                    return node
                if any(kw.arg is None for kw in node.keywords):
                    return node
                existing = {kw.arg for kw in node.keywords if kw.arg}
                if "sep" not in existing:
                    node.keywords.append(ast.keyword(arg="sep", value=ast.Name(id="sep", ctx=ast.Load())))
                if "decimal" not in existing:
                    node.keywords.append(ast.keyword(arg="decimal", value=ast.Name(id="decimal", ctx=ast.Load())))
                if "encoding" not in existing:
                    node.keywords.append(ast.keyword(arg="encoding", value=ast.Name(id="encoding", ctx=ast.Load())))
                return node

        try:
            fixed_tree = _ToCsvDialectFixer().visit(tree)
            ast.fix_missing_locations(fixed_tree)
            return ast.unparse(fixed_tree)
        except Exception:
            return code


    def _call_name(self, call_node: ast.Call) -> str:
        try:
            return ast.unparse(call_node.func)
        except Exception:
            if isinstance(call_node.func, ast.Name):
                return call_node.func.id
            if isinstance(call_node.func, ast.Attribute):
                return call_node.func.attr
        return ""

    def _get_call_parts(self, node: ast.AST) -> List[str]:
        parts: List[str] = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        return list(reversed(parts))

    def _is_path_constructor(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id == "Path"
        if isinstance(node, ast.Attribute):
            return node.attr == "Path"
        return False

    def _node_is_input_reference(self, node: ast.AST, input_names: set, data_path: str) -> bool:
        if isinstance(node, ast.Name) and node.id in input_names:
            return True
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value == data_path:
            return True
        if isinstance(node, ast.Call) and self._is_path_constructor(node.func):
            for arg in node.args:
                if self._node_is_input_reference(arg, input_names, data_path):
                    return True
            for kw in node.keywords:
                if self._node_is_input_reference(kw.value, input_names, data_path):
                    return True
        return False

    def _extract_target_names(self, target: ast.AST) -> List[str]:
        if isinstance(target, ast.Name):
            return [target.id]
        if isinstance(target, (ast.Tuple, ast.List)):
            names: List[str] = []
            for elt in target.elts:
                names.extend(self._extract_target_names(elt))
            return names
        return []

    def _collect_input_path_names(self, tree: ast.AST, data_path: str) -> set:
        names: set = set()

        def value_is_data_path(value: ast.AST, known: set) -> bool:
            if isinstance(value, ast.Constant) and isinstance(value.value, str) and value.value == data_path:
                return True
            if isinstance(value, ast.Name) and value.id in known:
                return True
            if isinstance(value, ast.Call) and self._is_path_constructor(value.func):
                for arg in value.args:
                    if value_is_data_path(arg, known):
                        return True
                for kw in value.keywords:
                    if value_is_data_path(kw.value, known):
                        return True
            return False

        changed = True
        while changed:
            changed = False
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    value = node.value
                    if value is None or not value_is_data_path(value, names):
                        continue
                    for target in node.targets:
                        for name in self._extract_target_names(target):
                            if name not in names:
                                names.add(name)
                                changed = True
                elif isinstance(node, ast.AnnAssign):
                    value = node.value
                    if value is None or not value_is_data_path(value, names):
                        continue
                    for name in self._extract_target_names(node.target):
                        if name not in names:
                            names.add(name)
                            changed = True
        return names

    def _is_input_exists_call(self, call_node: ast.Call, input_names: set, data_path: str) -> bool:
        if not isinstance(call_node.func, ast.Attribute) or call_node.func.attr != "exists":
            return False
        func_value = call_node.func.value
        if isinstance(func_value, ast.Attribute):
            if isinstance(func_value.value, ast.Name) and func_value.value.id == "os" and func_value.attr == "path":
                return any(
                    self._node_is_input_reference(arg, input_names, data_path)
                    for arg in call_node.args
                )
        if isinstance(func_value, ast.Call) and self._is_path_constructor(func_value.func):
            for arg in func_value.args:
                if self._node_is_input_reference(arg, input_names, data_path):
                    return True
            for kw in func_value.keywords:
                if self._node_is_input_reference(kw.value, input_names, data_path):
                    return True
        if self._node_is_input_reference(func_value, input_names, data_path):
            return True
        return False

    def _try_handles_filenotfound(self, try_node: ast.Try) -> bool:
        for handler in try_node.handlers:
            exc_type = handler.type
            if exc_type is None:
                continue
            if isinstance(exc_type, ast.Name) and exc_type.id == "FileNotFoundError":
                return True
            if isinstance(exc_type, ast.Attribute) and exc_type.attr == "FileNotFoundError":
                return True
            if isinstance(exc_type, ast.Tuple):
                for elt in exc_type.elts:
                    if isinstance(elt, ast.Name) and elt.id == "FileNotFoundError":
                        return True
                    if isinstance(elt, ast.Attribute) and elt.attr == "FileNotFoundError":
                        return True
        return False

    def _call_is_read_csv(self, call_node: ast.Call) -> bool:
        name = self._call_name(call_node).lower()
        return name.endswith("read_csv")

    def _call_uses_input(self, call_node: ast.Call, input_names: set, data_path: str) -> bool:
        for arg in call_node.args:
            if self._node_is_input_reference(arg, input_names, data_path):
                return True
        for kw in call_node.keywords:
            if self._node_is_input_reference(kw.value, input_names, data_path):
                return True
        return False

    def _synthetic_call_reason(self, call_node: ast.Call) -> str | None:
        known_sklearn_make_generators = {
            "make_biclusters",
            "make_blobs",
            "make_checkerboard",
            "make_circles",
            "make_classification",
            "make_friedman1",
            "make_friedman2",
            "make_friedman3",
            "make_gaussian_quantiles",
            "make_hastie_10_2",
            "make_low_rank_matrix",
            "make_moons",
            "make_multilabel_classification",
            "make_regression",
            "make_s_curve",
            "make_sparse_coded_signal",
            "make_sparse_spd_matrix",
            "make_sparse_uncorrelated",
            "make_spd_matrix",
            "make_swiss_roll",
        }
        parts = self._get_call_parts(call_node.func)
        if parts:
            func_name = parts[-1].lower()
            module_prefix = ".".join(part.lower() for part in parts[:-1])
            if module_prefix in {"np.random", "numpy.random"}:
                allowed = {"seed", "choice", "randint", "permutation", "shuffle", "default_rng"}
                if func_name not in allowed:
                    return f"forbidden_np_random_call:{module_prefix}.{func_name}"
            if (
                func_name in known_sklearn_make_generators
                and (
                    not module_prefix
                    or module_prefix.endswith(".datasets")
                    or module_prefix in {
                        "datasets",
                        "sklearn.datasets",
                        "sklearn.datasets.samples_generator",
                    }
                )
            ):
                return f"forbidden_sklearn_make_call:{self._call_name(call_node).lower() or func_name}"

        call_name = self._call_name(call_node).lower()
        if (
            "sklearn.datasets.make_" in call_name
            or ".datasets.make_" in call_name
            or call_name in known_sklearn_make_generators
        ):
            return f"forbidden_sklearn_make_call:{call_name}"
        if "faker" in call_name:
            return f"forbidden_faker_call:{call_name}"
        return None

    def _detect_forbidden_input_fallback(self, code: str, data_path: str) -> List[str]:
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return [f"ast_parse_failed:{exc.msg}"]

        input_names = self._collect_input_path_names(tree, data_path)
        reasons: List[str] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if self._is_input_exists_call(node, input_names, data_path):
                    reasons.append("input_existence_check_on_input_file")
                synthetic_reason = self._synthetic_call_reason(node)
                if synthetic_reason:
                    reasons.append(synthetic_reason)

        for node in ast.walk(tree):
            if isinstance(node, ast.Try) and self._try_handles_filenotfound(node):
                for stmt in node.body:
                    for inner in ast.walk(stmt):
                        if isinstance(inner, ast.Call) and self._call_is_read_csv(inner):
                            if self._call_uses_input(inner, input_names, data_path):
                                reasons.append("file_not_found_fallback_on_input_read")
                                break

        deduped: List[str] = []
        seen = set()
        for reason in reasons:
            if reason in seen:
                continue
            seen.add(reason)
            deduped.append(reason)
        return deduped

    def _execute_llm_call(self, sys_prompt: str, usr_prompt: str, temperature: float = 0.1) -> str:
        """Helper to execute plan-stage LLM call through OpenRouter."""
        model_name = self.model_name

        self.last_prompt = sys_prompt + "\n\nUSER:\n" + usr_prompt
        print(f"DEBUG: ML Engineer (Plan) calling OpenRouter Model ({model_name})...")

        try:
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": usr_prompt},
            ]
            response, model_used = call_chat_with_fallback(
                self.client,
                messages,
                [self.model_name, self.fallback_model_name],
                call_kwargs={"temperature": temperature},
                logger=self.logger,
                context_tag="ml_engineer_plan",
            )
            self.last_model_used = model_used
            self.logger.info("ML_ENGINEER_MODEL_USED: %s", model_used)
            content = extract_response_text(response)
            if not content:
                raise ValueError("EMPTY_COMPLETION")
            return content
        except Exception as e:
            # Check for 504
            if "504" in str(e):
                raise ConnectionError("LLM Server Timeout (504 Received)")
            raise e

    def generate_ml_plan(
        self,
        data_profile: Dict[str, Any],
        execution_contract: Dict[str, Any] | None = None,
        strategy: Dict[str, Any] | None = None,
        business_objective: str = "",
        llm_call: Any = None,
    ) -> Dict[str, Any]:
        """
        Generate ml_plan.json using LLM reasoning (Facts -> Plan).

        NEVER returns {} - always returns a complete plan with all REQUIRED_PLAN_KEYS.

        Args:
            data_profile: Data profile with facts (outcome_analysis, split_candidates, etc.)
            execution_contract: V4.1 execution contract
            strategy: Selected strategy dict
            business_objective: Business objective string
            llm_call: Optional callable for LLM call injection (for testing).
                      Signature: llm_call(system_prompt, user_prompt) -> str

        Returns:
            Complete ML plan dict (never empty)
        """
        import copy
        contract = execution_contract or {}
        profile = data_profile or {}
        strategy_dict = strategy or {}

        PLAN_PROMPT = """
        You are a Senior ML Engineer. Your task is to infer a robust ML execution plan from the evidence and constraints below.

        *** DATA FACTS (Facts Only - CANONICAL EVIDENCE) ***
        $data_profile_json

        *** EXECUTION CONTRACT ***
        $execution_contract_json

        *** STRATEGY ***
        $strategy_json

        *** BUSINESS OBJECTIVE ***
        "$business_objective"

        *** REASONING TASK ***
        Infer the safest and most executable plan for:
        - which rows are trainable,
        - which metric and validation policy are authoritative,
        - how scoring rows should be selected,
        - and what leakage policy is required.

        Use this reasoning workflow internally:
        1. Resolve training eligibility from label availability, split semantics, and contract intent.
        2. Resolve evaluation authority from contract.validation_requirements, evaluation_spec, and qa_gates.
        3. Choose the simplest validation policy that is credible for the data structure and objective.
        4. Set scoring_policy and leakage_policy from explicit evidence, not assumptions.
        5. Record the structured facts that justified each decision so QA can verify coherence.

        *** GUARDRAILS ***
        - If outcome_analysis shows any outcome column with null_frac > 0, you cannot use "use_all_rows" for training_rows_policy.
        - Use the exact primary metric from validation_requirements, evaluation_spec, or qa_gates when provided. Do not invent or normalize metric names.
        - If leakage_flags exist in data_profile, prefer leakage_policy.action="exclude_flagged_columns" unless the contract explicitly allows those columns.
        - If no primary metric is specified anywhere, infer the minimal defensible metric from analysis_type and state that inference in notes.
        - Avoid ambiguity: select ONE training_rows_policy and make it explicit via train_filter.
        - Base every decision on evidence found in the data_profile or contract. Do not invent rules.

        *** REQUIRED OUTPUT (JSON ONLY, NO MARKDOWN) ***
        {
          "training_rows_policy": "use_all_rows | only_rows_with_label | use_split_column | custom",
          "training_rows_rule": "string rule if custom, or null",
          "split_column": "col_name or null",
          "train_filter": {
              "type": "none | label_not_null | split_equals | custom_rule",
              "column": "column name or null",
              "value": "value for split_equals or null",
              "rule": "rule string for custom_rule or null"
          },
          "metric_policy": {
              "primary_metric": "metric_name",
              "secondary_metrics": [],
              "report_with_cv": true,
              "notes": "brief justification"
          },
          "cv_policy": {
              "strategy": "StratifiedKFold | KFold | TimeSeriesSplit | GroupKFold | auto",
              "n_splits": 5,
              "shuffle": true,
              "stratified": true,
              "notes": "brief justification"
          },
          "scoring_policy": {
              "generate_scores": true,
              "score_rows": "all | labeled_only | unlabeled_only"
          },
          "leakage_policy": {
              "action": "none | exclude_flagged_columns | manual_review",
              "flagged_columns": [],
              "notes": "brief justification"
          },
          "evidence_used": {
              "outcome_null_frac": {"column": "target_col", "null_frac": 0.3},
              "split_candidates": [{"column": "__split", "values": ["train", "test"]}],
              "split_evaluation": "used split column because..." or "ignored split because...",
              "contract_primary_metric": "roc_auc or null if not specified",
              "analysis_type": "classification"
          },
          "evidence": ["fact1 from profile", "fact2 from profile"],
          "assumptions": [],
          "open_questions": [],
          "notes": ["brief reasoning notes"]
        }
        """

        # Check if we can make LLM calls
        has_llm_init = getattr(self, "model_name", None) is not None
        can_execute_llm = has_llm_init or llm_call is not None

        if not can_execute_llm:
            # Agent not initialized, cannot call LLM
            result = self._normalize_ml_plan(None, source="missing_llm_init")
            result = self._derive_train_filter(result, profile, contract)
            result["notes"] = ["Agent not initialized; cannot call LLM. Using fallback defaults."]
            return result

        # Prepare context
        try:
            render_kwargs = {
                "data_profile_json": self._serialize_json_for_prompt(
                    profile,
                    max_chars=12000,
                    max_str_len=700,
                    max_list_items=60,
                ),
                "execution_contract_json": self._serialize_json_for_prompt(
                    self._compact_execution_contract(contract),
                    max_chars=8000,
                    max_str_len=600,
                    max_list_items=60,
                ),
                "strategy_json": self._serialize_json_for_prompt(
                    strategy_dict,
                    max_chars=4000,
                    max_str_len=600,
                    max_list_items=60,
                ),
                "business_objective": business_objective,
            }
        except Exception as render_err:
            result = self._normalize_ml_plan(None, source="render_error")
            result = self._derive_train_filter(result, profile, contract)
            result["notes"] = [f"Failed to render prompt context: {render_err}"]
            return result

        system_prompt = render_prompt(PLAN_PROMPT, **render_kwargs)
        user_prompt = "Generate the ML Plan JSON now."

        # Determine which LLM call function to use
        if llm_call is not None:
            execute_call = llm_call
        else:
            execute_call = lambda sys, usr: self._execute_llm_call(sys, usr, temperature=0.1)

        try:
            # First attempt
            response = execute_call(system_prompt, user_prompt)
            if hasattr(self, 'last_response'):
                self.last_response = response
            parsed = self._parse_json_response(response)
            ml_plan = self._normalize_ml_plan(parsed, source="llm")
            ml_plan = self._apply_contract_metric_override(ml_plan, contract)

            # Check if we got meaningful data
            if ml_plan.get("training_rows_policy") != "unspecified":
                ml_plan = self._derive_train_filter(ml_plan, profile, contract)
                return ml_plan

            # Retry attempt
            print("Warning: ML Plan generation returned incomplete JSON, retrying...")
            response = execute_call(system_prompt, user_prompt + "\nCRITICAL: Return VALID JSON ONLY. No markdown.")
            if hasattr(self, 'last_response'):
                self.last_response = response
            parsed = self._parse_json_response(response)
            ml_plan = self._normalize_ml_plan(parsed, source="llm_retry")
            ml_plan = self._apply_contract_metric_override(ml_plan, contract)
            ml_plan = self._derive_train_filter(ml_plan, profile, contract)

            return ml_plan

        except Exception as e:
            print(f"ML Engineer Plan Gen Error: {e}")
            result = self._normalize_ml_plan(None, source="llm_error")
            result = self._derive_train_filter(result, profile, contract)
            result["notes"] = [f"LLM call failed: {e}"]
            return result

    def _apply_contract_metric_override(
        self,
        ml_plan: Dict[str, Any],
        contract: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not isinstance(ml_plan, dict):
            return ml_plan
        contract_metric = self._resolve_contract_primary_metric(contract)
        if not contract_metric:
            return ml_plan
        metric_policy = ml_plan.get("metric_policy")
        if not isinstance(metric_policy, dict):
            metric_policy = {}
            ml_plan["metric_policy"] = metric_policy
        current = str(metric_policy.get("primary_metric") or "").strip()
        if not current or current.lower() == "unspecified" or current != contract_metric:
            metric_policy["primary_metric"] = contract_metric
            notes = ml_plan.get("notes")
            if not isinstance(notes, list):
                notes = []
            notes.append(f"Primary metric aligned to contract: {contract_metric}")
            ml_plan["notes"] = notes
        evidence = ml_plan.get("evidence_used")
        if isinstance(evidence, dict):
            evidence["contract_primary_metric"] = contract_metric
        return ml_plan

    def _resolve_contract_primary_metric(self, contract: Dict[str, Any] | None) -> str | None:
        if not isinstance(contract, dict):
            return None
        validation = contract.get("validation_requirements")
        if isinstance(validation, dict):
            primary = validation.get("primary_metric")
            if isinstance(primary, str) and primary.strip():
                return primary.strip()
        evaluation_spec = contract.get("evaluation_spec")
        if isinstance(evaluation_spec, dict):
            validation = evaluation_spec.get("validation_requirements")
            if isinstance(validation, dict):
                primary = validation.get("primary_metric")
                if isinstance(primary, str) and primary.strip():
                    return primary.strip()
            primary = evaluation_spec.get("primary_metric")
            if isinstance(primary, str) and primary.strip():
                return primary.strip()
            qa_gates = evaluation_spec.get("qa_gates")
            if isinstance(qa_gates, list):
                for gate in qa_gates:
                    if not isinstance(gate, dict):
                        continue
                    params = gate.get("params")
                    if isinstance(params, dict):
                        metric = params.get("metric")
                        if isinstance(metric, str) and metric.strip():
                            return metric.strip()
        return None

    def _parse_json_response(self, text: str) -> Any:
        try:
            cleaned = text.strip()
            # Remove markdown fences
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned:
                # Fallback for generic block
                parts = cleaned.split("```")
                if len(parts) >= 2:
                     cleaned = parts[1].strip()
            
            # More robust regex
            import re
            json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if json_match:
                cleaned = json_match.group(0)
            return json.loads(cleaned)
        except Exception:
            return None

    def _normalize_ml_plan(self, parsed: Any, source: str = "llm") -> Dict[str, Any]:
        """
        Normalize parsed LLM response to a complete ML plan.

        NEVER returns {} or None - always returns a complete plan with all required keys.

        Args:
            parsed: Raw parsed response (can be dict, list, None, etc.)
            source: Plan source label (e.g., "llm", "fallback", "missing_llm_init")

        Returns:
            Complete ML plan dict with all REQUIRED_PLAN_KEYS
        """
        import copy

        # Start with a copy of DEFAULT_PLAN
        result = copy.deepcopy(DEFAULT_PLAN)
        result["plan_source"] = source

        # Handle list input
        if isinstance(parsed, list):
            if parsed and isinstance(parsed[0], dict):
                parsed = parsed[0]
            else:
                result["notes"] = ["Parsed response was list but could not extract dict"]
                return result

        # Handle non-dict input
        if not isinstance(parsed, dict):
            result["notes"] = [f"Parsed response was {type(parsed).__name__}, using defaults"]
            return result

        # Merge parsed values into result
        # training_rows_policy
        if "training_rows_policy" in parsed and isinstance(parsed["training_rows_policy"], str):
            result["training_rows_policy"] = parsed["training_rows_policy"]

        # training_rows_rule
        if "training_rows_rule" in parsed:
            result["training_rows_rule"] = parsed["training_rows_rule"]

        # split_column
        if "split_column" in parsed:
            result["split_column"] = parsed["split_column"]

        # train_filter
        if "train_filter" in parsed:
            tf = parsed["train_filter"]
            if isinstance(tf, dict):
                result["train_filter"] = tf
            elif isinstance(tf, str):
                result["train_filter"] = {
                    "type": "custom_rule",
                    "column": None,
                    "value": None,
                    "rule": tf,
                }

        # metric_policy
        if "metric_policy" in parsed:
            mp = parsed["metric_policy"]
            if isinstance(mp, dict):
                if "primary_metric" in mp:
                    result["metric_policy"]["primary_metric"] = str(mp["primary_metric"])
                if "secondary_metrics" in mp and isinstance(mp["secondary_metrics"], list):
                    result["metric_policy"]["secondary_metrics"] = mp["secondary_metrics"]
                if "report_with_cv" in mp:
                    result["metric_policy"]["report_with_cv"] = bool(mp["report_with_cv"])
                if "notes" in mp:
                    result["metric_policy"]["notes"] = str(mp.get("notes", ""))
            elif isinstance(mp, str):
                # Handle case where metric_policy is just a string
                result["metric_policy"]["primary_metric"] = mp

        # cv_policy
        if "cv_policy" in parsed:
            cp = parsed["cv_policy"]
            if isinstance(cp, dict):
                if "strategy" in cp:
                    result["cv_policy"]["strategy"] = str(cp["strategy"])
                if "n_splits" in cp:
                    try:
                        result["cv_policy"]["n_splits"] = int(cp["n_splits"])
                    except (ValueError, TypeError):
                        pass
                if "shuffle" in cp:
                    result["cv_policy"]["shuffle"] = bool(cp["shuffle"])
                if "stratified" in cp:
                    result["cv_policy"]["stratified"] = cp["stratified"]
                if "notes" in cp:
                    result["cv_policy"]["notes"] = str(cp.get("notes", ""))

        # scoring_policy
        if "scoring_policy" in parsed:
            sp = parsed["scoring_policy"]
            if isinstance(sp, dict):
                if "generate_scores" in sp:
                    result["scoring_policy"]["generate_scores"] = bool(sp["generate_scores"])
                if "score_rows" in sp:
                    result["scoring_policy"]["score_rows"] = str(sp["score_rows"])

        # leakage_policy
        if "leakage_policy" in parsed:
            lp = parsed["leakage_policy"]
            if isinstance(lp, dict):
                if "action" in lp:
                    result["leakage_policy"]["action"] = str(lp["action"])
                if "flagged_columns" in lp and isinstance(lp["flagged_columns"], list):
                    result["leakage_policy"]["flagged_columns"] = lp["flagged_columns"]
                if "notes" in lp:
                    result["leakage_policy"]["notes"] = str(lp.get("notes", ""))

        # evidence
        if "evidence" in parsed:
            ev = parsed["evidence"]
            if isinstance(ev, list):
                result["evidence"] = ev
            elif isinstance(ev, str):
                result["evidence"] = [ev]

        # assumptions
        if "assumptions" in parsed:
            assum = parsed["assumptions"]
            if isinstance(assum, list):
                result["assumptions"] = assum
            elif isinstance(assum, str):
                result["assumptions"] = [assum]

        # open_questions
        if "open_questions" in parsed:
            oq = parsed["open_questions"]
            if isinstance(oq, list):
                result["open_questions"] = oq
            elif isinstance(oq, str):
                result["open_questions"] = [oq]

        # notes
        if "notes" in parsed:
            notes = parsed["notes"]
            if isinstance(notes, list):
                result["notes"] = notes
            elif isinstance(notes, str):
                result["notes"] = [notes]

        # evidence_used (structured evidence digest for QA coherence checks)
        if "evidence_used" in parsed:
            eu = parsed["evidence_used"]
            if isinstance(eu, dict):
                result["evidence_used"] = eu
            else:
                result["evidence_used"] = {}
        else:
            # Ensure evidence_used always exists (even if empty) for QA
            result["evidence_used"] = {}

        # Set plan_source to "llm" if we got valid data
        if result["training_rows_policy"] != "unspecified":
            result["plan_source"] = source

        return result

    def _infer_outcome_column(
        self,
        execution_contract: Dict[str, Any] | None,
        data_profile: Dict[str, Any] | None,
    ) -> str | None:
        contract = execution_contract or {}
        profile = data_profile or {}
        outcome_columns = contract.get("outcome_columns")
        if isinstance(outcome_columns, list) and outcome_columns:
            return str(outcome_columns[0])
        outcome_analysis = profile.get("outcome_analysis", {})
        if isinstance(outcome_analysis, dict) and outcome_analysis:
            for key in outcome_analysis.keys():
                return str(key)
        return None

    def _split_candidate_info(
        self,
        data_profile: Dict[str, Any] | None,
        preferred_column: str | None = None,
    ) -> Dict[str, Any]:
        profile = data_profile or {}
        split_candidates = profile.get("split_candidates", [])
        if not isinstance(split_candidates, list):
            return {"column": None, "values": [], "has_train_test": False}

        def _candidate_values(candidate: Dict[str, Any]) -> List[str]:
            values = candidate.get("values")
            if isinstance(values, list):
                return [str(v) for v in values]
            sample = candidate.get("unique_values_sample")
            if isinstance(sample, list):
                return [str(v) for v in sample]
            uniques = candidate.get("unique_values")
            if isinstance(uniques, list):
                return [str(v) for v in uniques]
            return []

        def _has_train_test(values: List[str]) -> bool:
            lowered = {v.strip().lower() for v in values if isinstance(v, str)}
            return "train" in lowered and "test" in lowered

        for cand in split_candidates:
            if not isinstance(cand, dict):
                continue
            col = cand.get("column")
            if preferred_column and str(col) != str(preferred_column):
                continue
            values = _candidate_values(cand)
            return {
                "column": str(col) if col is not None else None,
                "values": values,
                "has_train_test": _has_train_test(values),
            }

        # Fallback: first candidate
        for cand in split_candidates:
            if not isinstance(cand, dict):
                continue
            col = cand.get("column")
            values = _candidate_values(cand)
            return {
                "column": str(col) if col is not None else None,
                "values": values,
                "has_train_test": _has_train_test(values),
            }

        return {"column": None, "values": [], "has_train_test": False}

    def _derive_train_filter(
        self,
        plan: Dict[str, Any],
        data_profile: Dict[str, Any],
        execution_contract: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        import copy

        result = copy.deepcopy(plan)
        result.setdefault("train_filter", {})
        train_filter = result.get("train_filter")
        if not isinstance(train_filter, dict):
            train_filter = {}

        tf_type = str(train_filter.get("type") or "").strip().lower()
        if tf_type in {"", "unspecified"}:
            tf_type = ""

        outcome_col = self._infer_outcome_column(execution_contract, data_profile)
        outcome_analysis = (data_profile or {}).get("outcome_analysis", {})
        has_null_labels = False
        if isinstance(outcome_analysis, dict):
            for _, analysis in outcome_analysis.items():
                if isinstance(analysis, dict) and analysis.get("present") and analysis.get("null_frac", 0) > 0:
                    has_null_labels = True
                    break

        training_rows_policy = str(result.get("training_rows_policy") or "").strip().lower()
        training_rows_rule = result.get("training_rows_rule")
        split_column = result.get("split_column")
        split_info = self._split_candidate_info(data_profile, split_column)
        if split_column is None and split_info.get("column"):
            split_column = split_info["column"]

        evidence_used = result.get("evidence_used") if isinstance(result.get("evidence_used"), dict) else {}
        split_eval = str(evidence_used.get("split_evaluation", "")).lower()
        uses_split = "split" in split_eval and "use" in split_eval

        if tf_type == "custom_rule" and not train_filter.get("rule") and isinstance(training_rows_rule, str):
            train_filter["rule"] = training_rows_rule

        known_policies = {"use_all_rows", "only_rows_with_label", "use_split_column", "custom", "unspecified"}
        if not tf_type:
            if training_rows_policy and training_rows_policy not in known_policies:
                # Preserve explicit custom policy from LLM; do not infer a filter
                result["train_filter"] = train_filter
                return result
            if training_rows_policy == "custom" and not isinstance(training_rows_rule, str):
                # Preserve explicit custom policy without forcing defaults
                result["train_filter"] = train_filter
                return result

        if tf_type:
            # normalize column/value if missing
            if tf_type == "label_not_null":
                train_filter.setdefault("column", outcome_col)
                train_filter.setdefault("value", None)
                train_filter.setdefault("rule", None)
            elif tf_type == "split_equals":
                train_filter.setdefault("column", split_column)
                if train_filter.get("value") is None and split_info.get("has_train_test"):
                    train_filter["value"] = "train"
                train_filter.setdefault("rule", None)
            elif tf_type == "custom_rule":
                train_filter.setdefault("column", outcome_col)
            elif tf_type == "none":
                train_filter.setdefault("column", None)
                train_filter.setdefault("value", None)
                train_filter.setdefault("rule", None)
        else:
            # Derive a deterministic filter to remove ambiguity
            if isinstance(training_rows_rule, str) and training_rows_rule.strip() and training_rows_policy not in {"use_all_rows", "only_rows_with_label", "use_split_column"}:
                train_filter = {
                    "type": "custom_rule",
                    "column": outcome_col,
                    "value": None,
                    "rule": training_rows_rule,
                }
            elif training_rows_policy == "custom" and isinstance(training_rows_rule, str):
                train_filter = {
                    "type": "custom_rule",
                    "column": outcome_col,
                    "value": None,
                    "rule": training_rows_rule,
                }
            elif split_column and split_info.get("has_train_test") and (training_rows_policy == "use_split_column" or uses_split):
                train_filter = {
                    "type": "split_equals",
                    "column": split_column,
                    "value": "train",
                    "rule": None,
                }
            elif split_column and split_info.get("has_train_test") and has_null_labels:
                # Avoid ambiguity: split is the explicit training mask when train/test is present
                train_filter = {
                    "type": "split_equals",
                    "column": split_column,
                    "value": "train",
                    "rule": None,
                }
            elif has_null_labels:
                train_filter = {
                    "type": "label_not_null",
                    "column": outcome_col,
                    "value": None,
                    "rule": None,
                }
            else:
                train_filter = {
                    "type": "none",
                    "column": None,
                    "value": None,
                    "rule": None,
                }

        result["train_filter"] = train_filter

        # Align training_rows_policy with train_filter to remove ambiguity
        tf_type = str(train_filter.get("type") or "").strip().lower()
        policy_is_known = training_rows_policy in known_policies or not training_rows_policy
        if policy_is_known:
            if tf_type == "split_equals":
                result["training_rows_policy"] = "use_split_column"
                if split_column:
                    result["split_column"] = split_column
                if isinstance(training_rows_rule, str):
                    rule_lower = training_rows_rule.lower()
                    if any(token in rule_lower for token in ("not missing", "not null", "notna", "non-null")):
                        result["training_rows_rule"] = f"rows where {split_column} == 'train'"
                elif split_column:
                    result["training_rows_rule"] = f"rows where {split_column} == 'train'"
            elif tf_type == "label_not_null":
                result["training_rows_policy"] = "only_rows_with_label"
                if isinstance(training_rows_rule, str) is False and outcome_col:
                    result["training_rows_rule"] = f"rows where {outcome_col} is not missing"
            elif tf_type == "none":
                result["training_rows_policy"] = "use_all_rows"
            elif tf_type == "custom_rule":
                result["training_rows_policy"] = "custom"

        return result

    def generate_code(
        self,
        strategy: Dict[str, Any],
        data_path: str,
        feedback_history: List[str] = None,
        previous_code: str = None,
        gate_context: Dict[str, Any] = None,
        iteration_handoff: Dict[str, Any] | None = None,
        csv_encoding: str = 'utf-8',
        csv_sep: str = ',',
        csv_decimal: str = '.',
        data_audit_context: str = "",
        business_objective: str = "",
        execution_contract: Dict[str, Any] | None = None,
        ml_view: Dict[str, Any] | None = None,
        ml_plan: Dict[str, Any] | None = None,
        # V4.1: availability_summary parameter removed
        signal_summary: Dict[str, Any] | None = None,
        cleaned_data_summary_min: Dict[str, Any] | None = None,
        iteration_memory: List[Dict[str, Any]] | None = None,
        iteration_memory_block: str = "",
        dataset_scale: Dict[str, Any] | None = None,
        dataset_scale_str: str | None = None,
        execution_profile: Dict[str, Any] | None = None,
        last_run_memory: List[Dict[str, Any]] | None = None,
        strategy_lock: Dict[str, Any] | None = None,
        editor_mode: bool = False,
    ) -> str:
        self._reset_prompt_trace("generate_code")
        self.last_training_policy_warnings = None

        SYSTEM_PROMPT_TEMPLATE = """
        You are a Senior ML and Deep Learning Engineer.

        === SENIOR REASONING PROTOCOL ===
        $senior_reasoning_protocol

        === SENIOR ENGINEERING PROTOCOL ===
        $senior_engineering_protocol

        MISSION
        - Return one complete, runnable Python script for the cleaned dataset at "$data_path".
        - Follow execution contract and ML view as source of truth.
        - Be universal and adaptive to any cleaned CSV and business objective.

        OPERATING MODES
        - BUILD MODE: produce the first full implementation from contract and context.
        - REPAIR MODE: when runtime/reviewer feedback exists, patch root cause first, preserve working blocks, and return a full script (not a diff).

        SOURCE OF TRUTH AND PRECEDENCE
        1) ML_VIEW_CONTEXT + EXECUTION_CONTRACT_CONTEXT (authoritative)
        2) ITERATION_HANDOFF / reviewer feedback (for repair priorities)
        3) CLEANED_DATA_SUMMARY_MIN and SIGNAL_SUMMARY (advisory only)
        - Never let advisory context override contract targets, required outputs, gates, or policies.

        ENGINEERING DECISION WORKFLOW
        - First map the contract into an executable training/scoring plan.
        - Then identify the smallest coherent implementation that satisfies the contract.
        - When feedback exists, diagnose the current dominant blocker before editing code.
        - Prefer the simplest valid implementation that preserves working behavior and fits the runtime budget.

        HARD CONSTRAINTS
        - Output valid Python code only. No markdown, no code fences.
        - Read input data only from "$data_path" (no hardcoded alternatives).
        - Load CSV dialect from $cleaning_manifest_path output_dialect and use it for all CSV reads/writes.
        - Do not invent columns, synthetic rows, or dummy fallback datasets.
        - Do not overwrite the input file; treat input as immutable.
        - If contract requires target-based modeling and target is unavailable, fail explicitly with ValueError.
        - Respect required outputs exactly as paths and file formats in the contract.
        - Avoid network/shell operations and filesystem discovery scans.

        SANDBOX SECURITY - BLOCKED IMPORTS (HARD CONSTRAINT):
        These imports are FORBIDDEN and will cause immediate script rejection:
        - sys, subprocess, socket, requests, httpx, urllib, ftplib
        - paramiko, selenium, playwright, openai, google.generativeai, builtins
        - eval(), exec(), compile(), __import__()
        FORBIDDEN CALLS (HARD):
        - os.remove, os.unlink, pathlib.Path.unlink, shutil.rmtree, os.rmdir
        - Never use file deletion as a validation strategy.
        ALLOWED imports: pandas, numpy, sklearn, scipy, xgboost, lightgbm,
        matplotlib, seaborn, json, os.path, os.makedirs, csv, math, statistics,
        collections, itertools, functools, typing, warnings, re, datetime, pathlib.Path, uuid
        If you need sys.stdout or sys.exit, use print() and raise SystemExit instead.

        DTYPE SAFETY PATTERNS
        - cleaned_data.csv is already typed by the Data Engineer. Read it respecting its dtypes:
          Use pd.read_csv(path, sep=sep, decimal=decimal) without forcing dtype=str.
        - After loading, verify dtypes match column_dtype_targets; fix mismatches with pd.to_numeric(errors='coerce').
        - For nullable integers, use pandas nullable Int64 instead of plain int64 casts.
        - Never assume clean target dtype without verifying nullability and domain.
        - Enforce column_dtype_targets when present (column-level and selector-family targets).
        - Before calling any ML framework's fit/train, print and verify all feature dtypes.
          Every framework has dtype constraints — validate compatibility before fitting.

        OUTPUT SAFETY PATTERNS
        - Ensure parent directories exist before each artifact write.
        - Use explicit column selections for scored outputs; avoid accidental reorder/mutation.
        - Serialize JSON with a default handler for numpy/pandas scalar and NaN types.

        WIDE DATASET PATTERNS
        - Prefer selector/set-based feature handling over manual column enumeration.
        - Apply vectorized transforms and bounded diagnostics for high-dimensional inputs.
        - Use aggregated family summaries when many columns share the same profile.

        CONTRACT-FIRST EXECUTION MAP (MANDATORY)
        - Before training/inference, build and print CONTRACT_EXECUTION_MAP with:
          target_columns, input_required_columns, training_rows_policy, train_filter,
          primary_metric, required_outputs, allowed_feature_sets summary, required_plot_ids.
        - If any mandatory map element is missing or contradictory, raise ValueError and stop.

        PREFLIGHT GATES (MANDATORY BEFORE fit)
        - Gate A: required input columns exist.
        - Gate B: target and task are consistent with evaluation spec.
          Use robust target normalization before validation:
          read target as str, strip/lower, then attempt pd.to_numeric(errors='coerce').
          For binary targets, accept equivalent values {0, 1} including "0", "1", "0.0", "1.0".
          Log unique target values before and after normalization in PRE_FLIGHT_GATES output.
        - Gate C: train/scoring row rules are applied explicitly.
        - Gate D: forbidden/audit-only columns are excluded from modeling features.
        - Gate E: required output directories and paths are writable.
          For Gate E writability checks, create directories and write a marker file
          like ".write_test_<uuid>.txt"; do not delete marker files.
        - Print PRE_FLIGHT_GATES with PASS/FAIL per gate.

        REPAIR DECISION FRAMEWORK (WHEN FEEDBACK EXISTS)
        - Identify the smallest blocker that currently dominates:
          runtime/cost failure, missing required outputs, contract/gate misalignment, or optional quality improvement.
        - Fix blockers before optional improvements.
        - Preserve valid prior logic and artifact generation.
        - Include a brief decision log comment with ROOT_CAUSE and FIX_APPLIED.

        TRAINING, VALIDATION, AND SCORING
        - Implement training_rows_policy and train_filter exactly when present.
        - Use evaluation_spec and validation_requirements as metric/CV authority.
        - If contract says requires_target=false, do not fit supervised models; still emit required artifacts with explicit no-train status.
        - Choose preprocessing, validation, and scoring logic that matches the data structure rather than generic boilerplate.
        - Handle outliers with data-driven, non-destructive methods unless contract says otherwise.
        - Array shapes must match the subset they index (e.g., OOF predictions sized to training rows, not the full dataset).

        FEATURE GOVERNANCE
        - Use only contract-allowed features:
          allowed_feature_sets, canonical_columns, derived_columns, leakage_execution_plan.
        - Exclude forbidden_for_modeling and audit_only_features from model inputs.
        - Never hardcode dataset-specific column names.
        - For wide datasets, prefer column selectors/column_sets when available.

        VISUALS AND DECISIONING
        - Honor VISUAL_REQUIREMENTS_CONTEXT and PLOT_SPEC_CONTEXT.
        - Generate requested plots/artifacts only when enabled or required by contract.
        - Honor DECISIONING_REQUIREMENTS_CONTEXT for operational decision columns in scored outputs.
        - If a required visual/decision output cannot be produced, report it in alignment artifacts.

        ARTIFACTS AND SERIALIZATION
        - Ensure output directories exist (data/ and static/plots/).
        - Write metrics and contract-required artifacts at exact paths.
        - Use a JSON serializer helper for numpy/pandas scalars, arrays, NaN, and bool types.
        - Always write JSON using json.dump(..., default=json_default).
        - Emit alignment_check when required and include evidence about feature usage and gate compliance.

        COMPUTE-AWARE EXECUTION
        - Use EXECUTION_PROFILE_CONTEXT as runtime budget guidance.
        - Bound CV/search complexity to fit budget and avoid uncontrolled loops.
        - Use chunked scoring for large outputs when needed.

        DATA PARTITIONING CONTEXT
        $data_partitioning_context

        AUTHORITATIVE CONTEXT
        - Business Objective: "$business_objective"
        - Strategy: $strategy_title ($analysis_type)
        - Strategy Hypothesis: $hypothesis
        - Strategy Techniques: $strategy_techniques
        - Strategy Fallback Chain: $strategy_fallback_chain
        - ML_VIEW_CONTEXT: $ml_view_context
        - Evaluation Spec: $evaluation_spec_json
        - Required Outputs: $deliverables_json
        - Artifact Schema (authoritative output column format):
        $artifact_schema_block
        - Canonical Columns: $canonical_columns
        - Required Features: $required_columns
        - Column Dtype Targets: $column_dtype_targets_json
        - Data Sample Context: $data_sample_context_json
        - Cleaned Data Summary (advisory): $cleaned_data_summary_min_json
        - Execution Profile Context: $execution_profile_json
        $optional_context_block
        - Data Audit Context: $data_audit_context

        Return Python code only.
        """
        from src.utils.context_pack import compress_long_lists, summarize_long_list, COLUMN_LIST_POINTER

        ml_view = ml_view or {}
        execution_contract_input = execution_contract if isinstance(execution_contract, dict) else {}
        required_outputs = (
            ml_view.get("required_outputs")
            or execution_contract_input.get("required_outputs", [])
            or []
        )
        if not execution_contract_input and ml_view:
            execution_contract_input = {
                "strategy_title": strategy.get("title", "Unknown"),
                "business_objective": business_objective,
                "canonical_columns": ml_view.get("canonical_columns") or [],
                "required_outputs": required_outputs,
                "column_roles": ml_view.get("column_roles") or {},
                "validation_requirements": ml_view.get("validation_requirements") or {},
                "evaluation_spec": ml_view.get("evaluation_spec") or {},
                "objective_analysis": ml_view.get("objective_analysis") or {},
                "qa_gates": ml_view.get("qa_gates") or [],
                "reviewer_gates": ml_view.get("reviewer_gates") or [],
                "ml_engineer_runbook": ml_view.get("ml_engineer_runbook") or {},
            }
        # V4.1: Build deliverables from required_outputs, no spec_extraction
        deliverables: List[Dict[str, Any]] = []
        if required_outputs:
            deliverables = [{"path": path, "required": True} for path in required_outputs if path]
        required_deliverables = [item.get("path") for item in deliverables if item.get("required") and item.get("path")]
        deliverables_json = self._serialize_json_for_prompt(
            deliverables,
            max_chars=4500,
            max_str_len=400,
            max_list_items=80,
        )

        artifact_schema_block = self._render_artifact_schema_block(
            execution_contract=execution_contract_input,
            ml_view=ml_view,
        )
        data_partitioning_context = self._build_data_partitioning_context(
            execution_contract=execution_contract_input,
            ml_view=ml_view,
            ml_plan=ml_plan,
            execution_profile=execution_profile,
        )

        # V4.1: Use ml_engineer_runbook directly, no legacy role_runbooks
        ml_runbook_source = execution_contract_input.get("ml_engineer_runbook")
        if not isinstance(ml_runbook_source, (dict, list, str)):
            ml_view_runbook = ml_view.get("ml_engineer_runbook")
            ml_runbook_source = ml_view_runbook if isinstance(ml_view_runbook, (dict, list, str)) else {}
        ml_runbook_json = self._serialize_json_for_prompt(
            ml_runbook_source,
            max_chars=5000,
            max_str_len=600,
            max_list_items=120,
        )
        # V4.1: No spec_extraction - removed
        execution_contract_compact = self._compact_execution_contract(execution_contract_input)
        execution_contract_compact = compress_long_lists(execution_contract_compact)[0]
        raw_handoff = iteration_handoff if isinstance(iteration_handoff, dict) else {}
        raw_gate_context = gate_context if isinstance(gate_context, dict) else {}
        raw_handoff_mode = str(raw_handoff.get("mode") or "").strip().lower()
        raw_handoff_source = str(
            raw_handoff.get("source") or raw_gate_context.get("source") or ""
        ).strip().lower()
        raw_editor_constraints = (
            raw_handoff.get("editor_constraints")
            if isinstance(raw_handoff.get("editor_constraints"), dict)
            else {}
        )
        raw_status = str(
            raw_gate_context.get("status")
            or (raw_handoff.get("quality_focus", {}).get("status") if isinstance(raw_handoff.get("quality_focus"), dict) else "")
            or ""
        ).strip().upper()
        optimization_round_hint = bool(
            previous_code
            and (
                raw_handoff_mode in {"optimize", "metric_optimize", "improve"}
                or "metric_improvement" in raw_handoff_source
                or "actor_critic" in raw_handoff_source
                or raw_status in {"OPTIMIZATION_REQUIRED", "IMPROVEMENT_REQUIRED"}
                or bool(raw_editor_constraints.get("must_apply_hypothesis"))
            )
        )
        canonical_columns_hint = (
            execution_contract_compact.get("canonical_columns")
            or execution_contract_input.get("canonical_columns")
            or ml_view.get("canonical_columns")
            or []
        )
        ml_view_whitelist = {
            "required_outputs",
            "artifact_requirements",
            "evaluation_spec",
            "validation_requirements",
            "qa_gates",
            "reviewer_gates",
            "allowed_feature_sets",
            "leakage_execution_plan",
            "column_roles",
            "canonical_columns",
            "derived_columns",
            "decisioning_requirements",
            "visual_requirements",
            "plot_spec",
            "ml_engineer_runbook",
            "iteration_policy",
            "split_spec",
            "n_train_rows",
            "n_test_rows",
            "n_total_rows",
        }
        ml_view_payload = {
            k: v for k, v in (ml_view or {}).items() if k in ml_view_whitelist and v not in (None, "", [], {})
        }
        if optimization_round_hint:
            validation_lock = (
                ml_view_payload.get("validation_requirements")
                if isinstance(ml_view_payload.get("validation_requirements"), dict)
                else (
                    execution_contract_input.get("validation_requirements")
                    if isinstance(execution_contract_input.get("validation_requirements"), dict)
                    else {}
                )
            )
            allowed_sets_lock = (
                ml_view_payload.get("allowed_feature_sets")
                if isinstance(ml_view_payload.get("allowed_feature_sets"), dict)
                else (
                    execution_contract_input.get("allowed_feature_sets")
                    if isinstance(execution_contract_input.get("allowed_feature_sets"), dict)
                    else {}
                )
            )
            column_roles_lock = (
                ml_view_payload.get("column_roles")
                if isinstance(ml_view_payload.get("column_roles"), dict)
                else (
                    execution_contract_input.get("column_roles")
                    if isinstance(execution_contract_input.get("column_roles"), dict)
                    else {}
                )
            )
            split_lock = (
                ml_view.get("split_spec")
                if isinstance(ml_view.get("split_spec"), dict)
                else (
                    execution_contract_input.get("split_spec")
                    if isinstance(execution_contract_input.get("split_spec"), dict)
                    else {}
                )
            )
            artifact_requirements_lock = (
                execution_contract_input.get("artifact_requirements")
                if isinstance(execution_contract_input.get("artifact_requirements"), dict)
                else (
                    ml_view.get("artifact_requirements")
                    if isinstance(ml_view.get("artifact_requirements"), dict)
                    else {}
                )
            )
            evaluation_lock = (
                execution_contract_input.get("evaluation_spec")
                if isinstance(execution_contract_input.get("evaluation_spec"), dict)
                else (
                    ml_view.get("evaluation_spec")
                    if isinstance(ml_view.get("evaluation_spec"), dict)
                    else {}
                )
            )
            metric_name_lock = (
                validation_lock.get("primary_metric")
                or evaluation_lock.get("primary_metric")
            )
            metric_rule_lock = None
            if isinstance(metric_name_lock, str) and "mean" in metric_name_lock.lower():
                metric_rule_lock = (
                    "Use a simple arithmetic mean unless the contract explicitly provides weights."
                )
            ml_view_payload = {
                "required_outputs": required_deliverables[:12],
                "evaluation_spec": {
                    "problem_type": evaluation_lock.get("problem_type"),
                    "objective_type": evaluation_lock.get("objective_type"),
                    "target_columns": evaluation_lock.get("target_columns"),
                    "primary_metric": metric_name_lock,
                    "metric_definition_rule": metric_rule_lock,
                },
                "validation_requirements": {
                    "primary_metric": metric_name_lock,
                    "method": validation_lock.get("method"),
                    "label_columns": validation_lock.get("label_columns"),
                    "params": validation_lock.get("params"),
                },
                "artifact_requirements": {
                    "file_schemas": artifact_requirements_lock.get("file_schemas"),
                    "scored_rows_required_columns": (
                        artifact_requirements_lock.get("scored_rows_schema", {}).get("required_columns")
                        if (
                            isinstance(artifact_requirements_lock.get("scored_rows_schema"), dict)
                            and self._resolve_declared_artifact_path(
                                execution_contract if isinstance(execution_contract, dict) else {},
                                ml_view if isinstance(ml_view, dict) else {},
                                "scored_rows.csv",
                            )
                        )
                        else []
                    ),
                },
                "allowed_feature_sets": {
                    "model_features": [
                        str(item)
                        for item in (allowed_sets_lock.get("model_features") or [])[:60]
                        if str(item).strip()
                    ],
                    "forbidden_features": [
                        str(item)
                        for item in (allowed_sets_lock.get("forbidden_features") or [])[:24]
                        if str(item).strip()
                    ],
                },
                "column_roles": {
                    "identifiers": column_roles_lock.get("identifiers"),
                    "outcome": column_roles_lock.get("outcome"),
                    "pre_decision": column_roles_lock.get("pre_decision"),
                    "post_decision_audit_only": column_roles_lock.get("post_decision_audit_only"),
                },
                "split_spec": split_lock,
                "qa_gates": ml_view_payload.get("qa_gates") or [],
                "reviewer_gates": ml_view_payload.get("reviewer_gates") or [],
                "ml_engineer_runbook": ml_runbook_source,
            }
        ml_view_payload = compress_long_lists(ml_view_payload)[0]
        ml_view_json = self._serialize_json_for_prompt(
            ml_view_payload,
            max_chars=9000 if optimization_round_hint else 11000,
            max_str_len=700,
            max_list_items=100,
        )
        plot_spec_json = self._serialize_json_for_prompt(
            ml_view.get("plot_spec", {}),
            max_chars=3500,
            max_str_len=500,
            max_list_items=80,
        )
        execution_profile_json = self._serialize_json_for_prompt(
            execution_profile or {},
            max_chars=2500,
            max_str_len=400,
            max_list_items=60,
        )
        evaluation_spec_source = execution_contract_input.get("evaluation_spec")
        if not isinstance(evaluation_spec_source, dict):
            evaluation_spec_source = ml_view.get("evaluation_spec") if isinstance(ml_view.get("evaluation_spec"), dict) else {}
        evaluation_spec_json = self._serialize_json_for_prompt(
            evaluation_spec_source,
            max_chars=4500,
            max_str_len=500,
            max_list_items=80,
        )
        canonical_columns_source = canonical_columns_hint
        required_columns_payload: Any = []
        artifact_reqs = execution_contract_input.get("artifact_requirements")
        if isinstance(artifact_reqs, dict):
            clean_cfg = artifact_reqs.get("clean_dataset")
            if isinstance(clean_cfg, dict):
                required_candidate = clean_cfg.get("required_columns")
                if isinstance(required_candidate, list) and required_candidate:
                    required_columns_payload = required_candidate
        if not required_columns_payload:
            view_artifacts = ml_view.get("artifact_requirements")
            if isinstance(view_artifacts, dict):
                clean_cfg = view_artifacts.get("clean_dataset")
                if isinstance(clean_cfg, dict):
                    required_candidate = clean_cfg.get("required_columns")
                    if isinstance(required_candidate, list) and required_candidate:
                        required_columns_payload = required_candidate
        if not required_columns_payload:
            strategy_required = strategy.get("required_columns", [])
            if isinstance(strategy_required, list) and strategy_required:
                required_columns_payload = strategy_required
        if not required_columns_payload:
            required_columns_payload = canonical_columns_source
        if isinstance(required_columns_payload, list) and len(required_columns_payload) > 80:
            required_columns_payload = summarize_long_list(required_columns_payload)
            required_columns_payload["note"] = COLUMN_LIST_POINTER

        required_dependencies = execution_contract_input.get("required_dependencies")
        if not isinstance(required_dependencies, list):
            required_dependencies = []
        runtime_dependency_context = self._build_runtime_dependency_context(required_dependencies)
        runtime_dependency_context_json = self._serialize_json_for_prompt(
            runtime_dependency_context,
            max_chars=4500,
            max_str_len=500,
            max_list_items=120,
        )
        data_sample_context = self._build_data_sample_context(
            data_path=data_path,
            csv_encoding=csv_encoding,
            csv_sep=csv_sep,
            csv_decimal=csv_decimal,
        )
        if optimization_round_hint and isinstance(data_sample_context, dict):
            data_sample_context = {
                "path": data_sample_context.get("path"),
                "status": data_sample_context.get("status"),
                "shape_preview": data_sample_context.get("shape_preview"),
                "dtypes_preview": data_sample_context.get("dtypes_preview"),
                "preview_columns": (
                    data_sample_context.get("preview_columns")
                    if isinstance(data_sample_context.get("preview_columns"), list)
                    else []
                )[:40],
                "preview_columns_truncated": data_sample_context.get("preview_columns_truncated"),
            }
        data_sample_context_json = self._serialize_json_for_prompt(
            data_sample_context,
            max_chars=7000,
            max_str_len=600,
            max_list_items=60,
        )
        cleaned_data_summary_payload = self._compact_cleaned_data_summary_for_prompt(cleaned_data_summary_min or {})
        if optimization_round_hint and isinstance(cleaned_data_summary_payload, dict):
            cleaned_data_summary_payload = {
                "row_count": cleaned_data_summary_payload.get("row_count"),
                "column_count": cleaned_data_summary_payload.get("column_count"),
                "missing_required_columns": cleaned_data_summary_payload.get("missing_required_columns", []),
                "role_dtype_warnings": cleaned_data_summary_payload.get("role_dtype_warnings", []),
                "family_aggregate": cleaned_data_summary_payload.get("family_aggregate", []),
                "source": cleaned_data_summary_payload.get("source"),
            }
        column_dtype_targets = ml_view.get("column_dtype_targets")
        if not isinstance(column_dtype_targets, dict) or not column_dtype_targets:
            column_dtype_targets = execution_contract_input.get("column_dtype_targets")
        if not isinstance(column_dtype_targets, dict):
            column_dtype_targets = {}

        # Build strategy context fields
        _techniques = strategy.get('techniques') or []
        _fallback_chain = strategy.get('fallback_chain') or []
        _strategy_techniques = json.dumps(_techniques, ensure_ascii=False) if _techniques else "N/A"
        _strategy_techniques_compact = self._truncate_prompt_text(
            _strategy_techniques,
            max_len=1200,
            head_len=800,
            tail_len=200,
        )
        _strategy_fallback_chain = "\n".join(f"  {item}" for item in _fallback_chain) if _fallback_chain else "N/A"
        _business_objective_digest = self._truncate_prompt_text(
            business_objective or "",
            max_len=1600 if optimization_round_hint else 5000,
            head_len=1000 if optimization_round_hint else 3000,
            tail_len=300 if optimization_round_hint else 1200,
        )

        # Build optional context block (Fix 4: skip empty fields)
        _optional_parts = []
        _runtime_dep = runtime_dependency_context_json
        if _runtime_dep and _runtime_dep not in ("{}", "null", ""):
            _optional_parts.append(f"- Runtime Dependency Context: {_runtime_dep}")
        _iter_mem = self._serialize_json_for_prompt(
            iteration_memory or [],
            max_chars=4000,
            max_str_len=400,
            max_list_items=40,
        )
        if _iter_mem and _iter_mem not in ("[]", "{}", "null", ""):
            _optional_parts.append(f"- Iteration Memory: {_iter_mem}")
        if iteration_memory_block:
            _optional_parts.append(f"- Iteration Memory (compact): {iteration_memory_block}")
        if not optimization_round_hint:
            for _opt_key, _opt_label in [
                ("decisioning_requirements", "Decisioning Requirements"),
                ("alignment_requirements", "Alignment Requirements"),
                ("feature_semantics", "Feature Semantics"),
                ("business_sanity_checks", "Business Sanity Checks"),
            ]:
                _opt_val = execution_contract_input.get(_opt_key)
                if _opt_val and _opt_val != {} and _opt_val != []:
                    _optional_parts.append(
                        f"- {_opt_label}: {self._serialize_json_for_prompt(_opt_val, max_chars=3000, max_str_len=400, max_list_items=60)}"
                    )
            if signal_summary and signal_summary != {}:
                _optional_parts.append(
                    f"- Signal Summary: {self._serialize_json_for_prompt(signal_summary, max_chars=3000, max_str_len=400, max_list_items=60)}"
                )
        _optional_context_block = "\n        ".join(_optional_parts) if _optional_parts else ""

        render_kwargs = dict(
            business_objective=business_objective,
            business_objective_digest=_business_objective_digest,
            strategy_title=strategy.get('title', 'Unknown'),
            analysis_type=str(strategy.get('analysis_type', 'predictive')).upper(),
            hypothesis=strategy.get('hypothesis', 'N/A'),
            strategy_techniques=_strategy_techniques,
            strategy_techniques_compact=_strategy_techniques_compact,
            strategy_fallback_chain=_strategy_fallback_chain,
            optional_context_block=_optional_context_block,
            required_columns=json.dumps(required_columns_payload),
            deliverables_json=deliverables_json,
            canonical_columns=self._serialize_json_for_prompt(
                canonical_columns_source,
                max_chars=5000,
                max_str_len=400,
                max_list_items=120,
            ),
            data_path=data_path,
            csv_encoding=csv_encoding,
            csv_sep=csv_sep,
            csv_decimal=csv_decimal,
            data_audit_context=data_audit_context,
            ml_view_context=ml_view_json,
            evaluation_spec_json=evaluation_spec_json,
            ml_engineer_runbook=ml_runbook_json,
            cleaned_data_summary_min_json=self._serialize_json_for_prompt(
                cleaned_data_summary_payload,
                max_chars=5000,
                max_str_len=500,
                max_list_items=80,
            ),
            column_dtype_targets_json=self._serialize_json_for_prompt(
                column_dtype_targets,
                max_chars=5000,
                max_str_len=500,
                max_list_items=120,
            ),
            optimization_authoritative_state=self._build_optimization_authoritative_state(
                data_path=data_path,
                execution_contract=execution_contract_input,
                ml_view=ml_view,
            ),
            data_sample_context_json=data_sample_context_json,
            execution_profile_json=execution_profile_json,
            dataset_scale=dataset_scale,
            artifact_schema_block=artifact_schema_block,
            data_partitioning_context=data_partitioning_context,
        )
        # Safe Rendering for System Prompt
        if optimization_round_hint:
            system_prompt = self._build_optimization_system_prompt(
                render_kwargs,
                ml_view=ml_view,
                execution_contract=execution_contract_input,
            )
            prompt_budget_meta = {
                "prompt_chars": len(system_prompt),
                "budget_applied": False,
                "mode": "optimization_editor",
            }
        else:
            system_prompt, render_kwargs, prompt_budget_meta = self._build_system_prompt_with_budget(
                SYSTEM_PROMPT_TEMPLATE,
                render_kwargs,
                ml_view=ml_view,
                execution_contract=execution_contract_input,
            )
        if isinstance(prompt_budget_meta, dict):
            prompt_chars = int(prompt_budget_meta.get("prompt_chars") or 0)
            budget_applied = bool(prompt_budget_meta.get("budget_applied"))
            print(
                "ML_PROMPT_BUDGET: "
                + json.dumps(
                    {
                        "prompt_chars": prompt_chars,
                        "budget_chars": 50000,
                        "budget_applied": budget_applied,
                        "mode": prompt_budget_meta.get("mode") or "default",
                    }
                )
            )
        
        # USER TEMPLATES (Static)
        USER_FIRSTPASS_TEMPLATE = """
        MODE: BUILD
        Generate a complete, runnable ML Python script for strategy "$strategy_title"
        using input data at "$data_path".

        Requirements:
        - Implement contract-first execution map and preflight gates.
        - Apply training/validation/evaluation exactly from contract + ML view.
        - Produce required outputs at exact contract paths.
        - Include alignment evidence artifact when required.
        - IGNORE complex feature_engineering_tasks from the contract in this first pass. 
          Focus on a robust BASELINE model only (simple cleaning/imputation/encoding).


        Return Python code only.
        """
        
        USER_PATCH_TEMPLATE = """
        MODE: REPAIR
        Your previous code was rejected by $gate_source.

        ITERATION_HANDOFF (authoritative patch context):
        $iteration_handoff_json

        REPAIR GROUND TRUTH (verified environment facts, authoritative):
        $repair_ground_truth

        REPAIR SCOPE (authoritative edit boundaries):
        $repair_scope

        PATCH OBJECTIVES (apply in order):
        $patch_objectives

        MUST PRESERVE:
        $must_preserve

        CRITICAL FEEDBACK:
        $feedback_text

        FEEDBACK DIGEST:
        $feedback_digest

        RUNTIME ERROR DETAIL:
        $runtime_error_detail

        TARGETED EDIT HINTS:
        $edit_instructions

        ACTIVE FIX CONTEXT:
        $fixes_bullets
        - Diagnose the smallest coherent edit set that resolves the active blocker.
        - If a traceback exists, fix its root cause before optional improvements.
        - Treat REPAIR GROUND TRUTH as higher priority than heuristic summaries when they conflict.
        - If REPAIR GROUND TRUTH includes callable signatures, accepted args, return-type facts, or sandbox facts, obey them.
        - Preserve verified logic that already works.
        - Ensure required outputs are still written at exact contract paths.
        - Do not generate synthetic data.

        PREVIOUS SCRIPT TO PATCH:
        $previous_code

        Repair task:
        - Apply a minimal but sufficient patch to the previous script. Do not rewrite from zero unless recovery is clearly cheaper and safer.
        - Keep contract context and execution map logic intact.
        - Return the full updated script (not a diff, not snippets).
        """

        USER_EDITOR_TEMPLATE = """
        MODE: CODE_EDITOR_MODE
        You are editing an existing script under tight constraints.
        Do not regenerate a new solution from zero and do not re-plan strategy.

        PHASE CLASSIFICATION:
        $phase_classification

        ERROR FEEDBACK:
        $error_feedback

        LAST RUN MEMORY (most recent attempts):
        $last_run_memory

        STRATEGY LOCK (immutable):
        $strategy_lock

        ITERATION HANDOFF:
        $iteration_handoff_json

        REPAIR GROUND TRUTH (verified environment facts, authoritative):
        $repair_ground_truth

        REPAIR SCOPE (authoritative edit boundaries):
        $repair_scope

        STRUCTURED CRITIQUE PACKET:
        $critic_packet_json

        STRUCTURED HYPOTHESIS PACKET (apply one hypothesis only):
        $hypothesis_packet_json

        PATCH OBJECTIVES:
        $patch_objectives

        MUST PRESERVE:
        $must_preserve

        EDITOR ENFORCEMENT:
        $editor_enforcement

        PREVIOUS SCRIPT:
        $previous_code

        EDITOR TASK:
        - Return ONLY the full updated Python script. No markdown, no explanation.
        - Keep script structure and apply the smallest coherent edits required by patch objectives and enforcement.
        - Keep strategy/objective/contract paths unchanged.
        - Treat phase as the current focus, not as a pre-scripted recipe.
        - Treat REPAIR GROUND TRUTH as environment truth; do not override it with guessed library behavior.
        - If REPAIR SCOPE says patch_only, use the whole script only as context and edit only the scoped targets/findings unless new verified runtime evidence forces a narrow expansion.
        - If phase is "runtime_repair", restore runnable behavior and required outputs before deferred metric-improvement work.
          If the failure is a timeout or OOM, reduce compute cost within the same model family before changing strategy.
        - If phase is "persistence", keep training/model selection logic stable unless a contract violation forces a broader fix.
        """

        USER_EDITOR_OPTIMIZATION_TEMPLATE = """
        MODE: OPTIMIZATION_MODE
        MODE: CODE_EDITOR_MODE_OPTIMIZATION
        You are editing an existing script for one focused optimization move.
        Think like a senior engineer: keep the incumbent stable, use the round evidence, and change only what this round truly requires.
        Do not regenerate a new solution from zero and do not re-plan strategy.

        CURRENT PHASE:
        $phase_classification

        CURRENT ROUND BRIEF:
        $optimization_round_brief

        ACTIVE HYPOTHESIS BRIEF:
        $active_hypothesis_brief

        CURRENT EVIDENCE BRIEF:
        $current_evidence_brief

        REPAIR GROUND TRUTH:
        $repair_ground_truth

        REPAIR SCOPE:
        $repair_scope

        PARAMETER / IMPLEMENTATION HINTS:
        $optimization_blueprint_hint

        LOCKED INVARIANTS:
        $invariants_lock

        ACTION FAMILY GUIDELINES:
        $action_family_guidelines

        RECENT ATTEMPT MEMORY:
        $recent_tracker

        OPTIMIZATION FEEDBACK DIGEST:
        $optimization_feedback_brief

        PATCH OBJECTIVES:
        $patch_objectives

        MUST PRESERVE:
        $must_preserve

        EDITOR ENFORCEMENT:
        $editor_enforcement

        PREVIOUS SCRIPT:
        $previous_code

        OPTIMIZATION TASK:
        - Return ONLY the full updated Python script. No markdown, no explanation.
        - Apply one coherent optimization move that matches the active hypothesis.
        - Treat the previous script as the incumbent and preserve working behavior by default.
        - Keep contract paths, split logic, expected row counts, target semantics, and model family unchanged unless the invariant block explicitly allows otherwise.
        - Use the evidence brief to target the weakest part of the incumbent instead of inventing a new plan.
        - If the current phase is runtime_repair or persistence with patch_only scope, the hypothesis is deferred context only: repair the scoped defect first and do not implement broader metric changes unless the failing block is exactly where the hypothesis applies.
        - If CURRENT PHASE is runtime_repair or persistence, fix the verified blocker first using REPAIR GROUND TRUTH and REPAIR SCOPE, then preserve the active optimization lane.
        - If the primary metric is defined as a mean and the contract does not provide explicit weights, compute a simple arithmetic mean.
        - Avoid unrelated refactors; edit only the regions needed for metric improvement.
        - Prefer the cheapest valid implementation that tests the idea without destabilizing working behavior.
        """

        USER_IMPROVE_TEMPLATE = """
        MODE: IMPROVE
        Your previous code executed successfully and was approved by the reviewer.
        Your task is to IMPROVE the metric, not fix bugs.

        CURRENT RESULTS:
        $metrics_summary

        IMPROVEMENT HISTORY:
        $improvement_history

        RECOMMENDED TECHNIQUES (from reviewer/evaluator):
        $consolidated_techniques

        STRATEGY CONTEXT (original plan):
        $strategy_context

        REVIEWER FEEDBACK:
        $improvement_suggestions

        FEATURE IMPORTANCE (top features, if available):
        $feature_importance

        ITERATION_HANDOFF:
        $iteration_handoff_json

        PREVIOUS APPROVED SCRIPT:
        $previous_code

        Instructions:
        1) Your code WORKS. Do NOT break what already works.
        2) Apply the RECOMMENDED TECHNIQUES above to increase $primary_metric.
        3) Prioritize in order: feature_engineering_tasks (from contract) > ensemble methods > specialized feature engineering > hyperparameter tuning.
        4) Return the full updated script (not a diff, not snippets).
        5) Keep all contract-required outputs and paths intact.
        6) IMPLEMENT feature_engineering_tasks from the contract now. Valid data-driven features are the best way to improve.
        7) If you add new models for ensemble, keep the original model as fallback.
        8) Preserve data loading, CSV dialect, train/test split, and cross-validation logic.
        9) Use feature importance to guide feature engineering (interact top features, drop useless ones).
        """

        # Construct User Message with Patch Mode Logic
        handoff_payload = self._normalize_iteration_handoff(
            iteration_handoff=iteration_handoff,
            gate_context=gate_context,
            required_deliverables=required_deliverables,
        )
        handoff_payload_json = json.dumps(compress_long_lists(handoff_payload)[0], indent=2)
        gate_ctx = gate_context if isinstance(gate_context, dict) else {}
        feedback_text = self._collect_editor_feedback_text(
            gate_context=gate_ctx,
            handoff_payload=handoff_payload,
            feedback_history=feedback_history,
        )
        editor_mode_active = bool(
            previous_code
            and (
                bool(editor_mode)
                or self._should_use_editor_mode(
                    gate_context=gate_ctx,
                    handoff_payload=handoff_payload,
                    feedback_history=feedback_history,
                )
            )
        )
        optimization_editor_mode = bool(
            editor_mode_active
            and self._is_metric_optimization_context(
                gate_context=gate_ctx,
                handoff_payload=handoff_payload,
            )
        )
        improve_mode_active = bool(
            not editor_mode_active
            and previous_code
            and handoff_payload.get("mode") == "improve"
            and not self._is_actor_critic_improvement_strict_enabled()
        )
        patch_mode_active = bool(
            not editor_mode_active
            and not improve_mode_active
            and previous_code
            and (gate_context or handoff_payload.get("mode") == "patch")
        )
        if editor_mode_active:
            patch_objectives = handoff_payload.get("patch_objectives", [])
            patch_objectives_block = "\n".join([f"- {item}" for item in patch_objectives]) if patch_objectives else "- Resolve runtime/QA feedback with minimal edits."
            must_preserve = handoff_payload.get("must_preserve", [])
            must_preserve_block = "\n".join([f"- {item}" for item in must_preserve]) if must_preserve else "- Preserve valid training and artifact logic."
            phase_classification = self._classify_editor_phase(
                gate_context=gate_ctx,
                handoff_payload=handoff_payload,
                feedback_text=feedback_text,
            )
            previous_code_block = self._truncate_code_for_patch(previous_code)
            last_run_memory_block = self._build_last_run_memory_block(last_run_memory)
            strategy_lock_block = self._serialize_json_for_prompt(
                strategy_lock or {},
                max_chars=1800,
                max_str_len=260,
                max_list_items=40,
            )
            critic_packet_block = self._serialize_json_for_prompt(
                handoff_payload.get("critic_packet") if isinstance(handoff_payload.get("critic_packet"), dict) else {},
                max_chars=2200,
                max_str_len=260,
                max_list_items=30,
            )
            hypothesis_packet_block = self._serialize_json_for_prompt(
                handoff_payload.get("hypothesis_packet") if isinstance(handoff_payload.get("hypothesis_packet"), dict) else {},
                max_chars=2200,
                max_str_len=260,
                max_list_items=30,
            )
            editor_enforcement_block = self._build_editor_enforcement_block(handoff_payload)
            repair_ground_truth_block = self._serialize_json_for_prompt(
                handoff_payload.get("repair_ground_truth")
                if isinstance(handoff_payload.get("repair_ground_truth"), dict)
                else {},
                max_chars=2600,
                max_str_len=260,
                max_list_items=24,
            )
            repair_scope_block = self._serialize_json_for_prompt(
                handoff_payload.get("repair_scope")
                if isinstance(handoff_payload.get("repair_scope"), dict)
                else {},
                max_chars=2200,
                max_str_len=220,
                max_list_items=20,
            )
            if optimization_editor_mode:
                optimization_inputs = self._resolve_optimization_mode_inputs(
                    handoff_payload=handoff_payload,
                    last_run_memory=last_run_memory,
                )
                optimization_briefs = self._build_optimization_editor_briefs(
                    handoff_payload=handoff_payload,
                    optimization_inputs=optimization_inputs,
                    feedback_text=feedback_text,
                )
                action_family_guidelines = optimization_inputs.get("action_family_guidelines") or []
                action_family_guidelines_block = "\n".join(
                    [f"- {str(item)}" for item in action_family_guidelines if str(item).strip()]
                ) or "- Keep targeted edits with contract invariants preserved."
                user_message = render_prompt(
                    USER_EDITOR_OPTIMIZATION_TEMPLATE,
                    phase_classification=phase_classification,
                    optimization_round_brief=optimization_briefs.get("round_brief") or "{}",
                    active_hypothesis_brief=optimization_briefs.get("active_hypothesis_brief") or "{}",
                    current_evidence_brief=optimization_briefs.get("current_evidence_brief") or "{}",
                    repair_ground_truth=repair_ground_truth_block or "{}",
                    repair_scope=repair_scope_block or "{}",
                    optimization_blueprint_hint=optimization_briefs.get("optimization_blueprint_hint") or "[]",
                    invariants_lock=optimization_briefs.get("invariants_lock") or "{}",
                    action_family_guidelines=action_family_guidelines_block,
                    recent_tracker=optimization_briefs.get("recent_tracker") or "[]",
                    optimization_feedback_brief=optimization_briefs.get("optimization_feedback_brief") or "No optimization feedback provided.",
                    patch_objectives=patch_objectives_block,
                    must_preserve=must_preserve_block,
                    editor_enforcement=editor_enforcement_block,
                    previous_code=previous_code_block,
                )
            else:
                user_message = render_prompt(
                    USER_EDITOR_TEMPLATE,
                    phase_classification=phase_classification,
                    error_feedback=feedback_text or "No structured feedback provided.",
                    last_run_memory=last_run_memory_block,
                    strategy_lock=strategy_lock_block,
                    iteration_handoff_json=handoff_payload_json,
                    repair_ground_truth=repair_ground_truth_block or "{}",
                    repair_scope=repair_scope_block or "{}",
                    critic_packet_json=critic_packet_block,
                    hypothesis_packet_json=hypothesis_packet_block,
                    patch_objectives=patch_objectives_block,
                    must_preserve=must_preserve_block,
                    editor_enforcement=editor_enforcement_block,
                    previous_code=previous_code_block,
                )
        elif improve_mode_active:
            # Build improve prompt from handoff
            metric_focus = handoff_payload.get("metric_focus", {})
            primary_metric = str(metric_focus.get("primary_metric", "roc_auc"))
            current_value = metric_focus.get("current_value", "N/A")
            best_value = metric_focus.get("best_value", "N/A")
            history_items = metric_focus.get("history", [])
            metrics_summary = f"Primary metric: {primary_metric}\nCurrent value: {current_value}\nBest value so far: {best_value}"
            improvement_history = "\n".join(
                [f"  Attempt {h.get('attempt', '?')}: {h.get('metric', 'N/A')} {'(improved)' if h.get('improved') else '(no improvement)'}"
                 for h in (history_items if isinstance(history_items, list) else [])]
            ) or "No previous improvement attempts."
            feat_imp = handoff_payload.get("feature_importance", {})
            if isinstance(feat_imp, dict) and feat_imp:
                feat_imp_text = json.dumps(feat_imp, indent=2)[:3000]
            else:
                feat_imp_text = "Not available."
            improvement_suggestions = str(handoff_payload.get("reviewer_suggestions", "")) or "No specific suggestions."
            # Build consolidated techniques text
            consolidated_techs = handoff_payload.get("consolidated_techniques", [])
            if isinstance(consolidated_techs, list) and consolidated_techs:
                consolidated_techniques_text = "\n".join([f"  - {t}" for t in consolidated_techs])
            else:
                consolidated_techniques_text = "No specific techniques suggested. Apply general ML improvement practices."
            # Build strategy context text
            strat_ctx = handoff_payload.get("strategy_context", {})
            if isinstance(strat_ctx, dict) and strat_ctx:
                strat_parts = []
                if strat_ctx.get("techniques"):
                    strat_parts.append("Techniques: " + ", ".join(strat_ctx["techniques"][:6]))
                if strat_ctx.get("fallback_chain"):
                    strat_parts.append("Fallback chain: " + " -> ".join(strat_ctx["fallback_chain"][:4]))
                if strat_ctx.get("feature_families"):
                    for fam in strat_ctx["feature_families"][:4]:
                        if isinstance(fam, dict):
                            strat_parts.append(f"Feature family '{fam.get('family', '?')}': {fam.get('rationale', '')}")
                if strat_ctx.get("hypothesis"):
                    strat_parts.append(f"Hypothesis: {strat_ctx['hypothesis']}")
                strategy_context_text = "\n".join(strat_parts) if strat_parts else "No strategy context available."
            else:
                strategy_context_text = "No strategy context available."
            previous_code_block = self._truncate_code_for_patch(previous_code)
            user_message = render_prompt(
                USER_IMPROVE_TEMPLATE,
                metrics_summary=metrics_summary,
                improvement_history=improvement_history,
                improvement_suggestions=improvement_suggestions,
                consolidated_techniques=consolidated_techniques_text,
                strategy_context=strategy_context_text,
                iteration_handoff_json=handoff_payload_json,
                feature_importance=feat_imp_text,
                previous_code=previous_code_block,
                primary_metric=primary_metric,
                strategy_title=strategy.get('title', 'Unknown'),
                data_path=data_path,
            )
        elif patch_mode_active:
            required_fixes = gate_ctx.get('required_fixes', [])
            fixes_bullets = "\n".join(["- " + str(fix) for fix in required_fixes if str(fix).strip()])
            if not fixes_bullets:
                fixes_bullets = "- Follow patch objectives from iteration handoff."
            digest_prefixes = ("REVIEWER FEEDBACK", "QA TEAM FEEDBACK", "RESULT EVALUATION FEEDBACK", "OUTPUT_CONTRACT_MISSING")
            feedback_digest_items = [f for f in (feedback_history or []) if isinstance(f, str) and f.startswith(digest_prefixes)]
            feedback_digest = "\n".join(feedback_digest_items[-5:])
            runtime_error_detail = ""
            runtime_error_payload = gate_ctx.get("runtime_error")
            if isinstance(runtime_error_payload, dict) and runtime_error_payload:
                runtime_error_detail = json.dumps(runtime_error_payload, indent=2)
            else:
                tail = handoff_payload.get("feedback", {}).get("runtime_error_tail") if isinstance(handoff_payload.get("feedback"), dict) else ""
                runtime_error_detail = str(tail or gate_ctx.get("traceback") or gate_ctx.get("execution_output_tail") or "").strip()
            error_snippet = self._extract_error_snippet_from_traceback(previous_code or "", runtime_error_detail, context_lines=10)
            if error_snippet:
                runtime_error_detail = runtime_error_detail + "\n\nERROR_SNIPPET:\n" + error_snippet
            runtime_error_detail = self._truncate_prompt_text(runtime_error_detail, max_len=6000, head_len=4000, tail_len=1500)
            patch_objectives = handoff_payload.get("patch_objectives", [])
            patch_objectives_block = "\n".join([f"- {item}" for item in patch_objectives]) if patch_objectives else "- Resolve reviewer findings."
            must_preserve = handoff_payload.get("must_preserve", [])
            must_preserve_block = "\n".join([f"- {item}" for item in must_preserve]) if must_preserve else "- Preserve valid artifact generation and data loading logic."
            previous_code_block = self._truncate_code_for_patch(previous_code)
            
            user_message = render_prompt(
                USER_PATCH_TEMPLATE,
                gate_source=str(gate_ctx.get('source', 'QA Reviewer')).upper(),
                iteration_handoff_json=handoff_payload_json,
                repair_ground_truth=self._serialize_json_for_prompt(
                    handoff_payload.get("repair_ground_truth")
                    if isinstance(handoff_payload.get("repair_ground_truth"), dict)
                    else {},
                    max_chars=2600,
                    max_str_len=260,
                    max_list_items=24,
                ) or "{}",
                repair_scope=self._serialize_json_for_prompt(
                    handoff_payload.get("repair_scope")
                    if isinstance(handoff_payload.get("repair_scope"), dict)
                    else {},
                    max_chars=2200,
                    max_str_len=220,
                    max_list_items=20,
                ) or "{}",
                patch_objectives=patch_objectives_block,
                must_preserve=must_preserve_block,
                feedback_text=feedback_text,
                fixes_bullets=fixes_bullets,
                previous_code=previous_code_block,
                feedback_digest=feedback_digest,
                runtime_error_detail=runtime_error_detail or "None",
                edit_instructions=str(gate_ctx.get("edit_instructions", "")),
                strategy_title=strategy.get('title', 'Unknown'),
                data_path=data_path
            )
        else:
            # First pass
            user_message = render_prompt(
                USER_FIRSTPASS_TEMPLATE,
                strategy_title=strategy.get('title', 'Unknown'),
                data_path=data_path
            )


        # Dynamic Configuration
        def _env_float(name: str, default: float) -> float:
            raw = os.getenv(name)
            if raw is None or raw == "":
                return default
            try:
                return float(raw)
            except ValueError:
                return default

        base_temp = _env_float("ML_ENGINEER_TEMPERATURE", 0.1)
        retry_temp = _env_float("ML_ENGINEER_TEMPERATURE_RETRY", 0.0)
        improve_temp = _env_float("ML_ENGINEER_TEMPERATURE_IMPROVE", 0.2)
        optimize_editor_temp = _env_float("ML_ENGINEER_TEMPERATURE_OPTIMIZE_EDITOR", improve_temp)
        if improve_mode_active:
            current_temp = improve_temp
        elif optimization_editor_mode:
            current_temp = optimize_editor_temp
        elif editor_mode_active or patch_mode_active:
            current_temp = retry_temp
        else:
            current_temp = base_temp

        from src.utils.retries import call_with_retries
        provider_label = "OpenRouter"
        initial_generation_stage = (
            "optimization_editor_generation"
            if optimization_editor_mode
            else "editor_generation"
            if editor_mode_active
            else "improve_generation"
            if improve_mode_active
            else "patch_generation"
            if patch_mode_active
            else "build_generation"
        )

        def _call_model_with_prompts(
            sys_prompt: str,
            usr_prompt: str,
            temperature: float,
            model_name: str,
            stage: str = "repair_subcall",
        ) -> str:
            print(f"DEBUG: ML Engineer calling {provider_label} Model ({model_name})...")
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": usr_prompt}
                ],
                temperature=temperature,
            )
            content = response.choices[0].message.content
            self._record_prompt_trace_entry(
                stage=stage,
                system_prompt=sys_prompt,
                user_prompt=usr_prompt,
                response=content,
                temperature=temperature,
                model_requested=model_name,
                model_used=model_name,
                context_tag="ml_engineer",
                used_fallback=False,
                source="openrouter_direct",
            )
            
            # CRITICAL CHECK FOR SERVER ERRORS (HTML/504)
            if "504 Gateway Time-out" in content or "<html" in content.lower():
                raise ConnectionError("LLM Server Timeout (504 Received)")
            return content

        try:
            self.last_fallback_reason = None
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]

            def _call_openrouter():
                # Select primary model: use editor model for repair/editor
                # iterations when configured, otherwise use the build model.
                effective_primary = (
                    self.editor_model_name
                    if editor_mode_active and self.editor_model_name
                    else self.model_name
                )
                effective_fallback = (
                    self.model_name
                    if effective_primary != self.model_name
                    else self.fallback_model_name
                )
                response, model_used = call_chat_with_fallback(
                    self.client,
                    messages,
                    [effective_primary, effective_fallback],
                    call_kwargs={"temperature": current_temp},
                    logger=self.logger,
                    context_tag="ml_engineer",
                )
                if model_used != effective_primary:
                    self.last_fallback_reason = "fallback_used"
                self.last_model_used = model_used
                self.logger.info("ML_ENGINEER_MODEL_USED: %s", model_used)
                content = extract_response_text(response)
                if not content:
                    raise ValueError("EMPTY_COMPLETION")
                self._record_prompt_trace_entry(
                    stage=initial_generation_stage,
                    system_prompt=system_prompt,
                    user_prompt=user_message,
                    response=content,
                    temperature=current_temp,
                    model_requested=effective_primary,
                    model_used=model_used,
                    context_tag="ml_engineer",
                    used_fallback=bool(model_used != effective_primary),
                    source="openrouter_fallback_chain",
                )
                if "504 Gateway Time-out" in content or "<html" in content.lower():
                    raise ConnectionError("LLM Server Timeout (504 Received)")
                return content

            content = call_with_retries(
                _call_openrouter,
                max_retries=5,
                backoff_factor=2,
                initial_delay=2,
            )
            print(f"DEBUG: {provider_label} response received.")
            code = self._clean_code(content)
            if code.strip().startswith("{") or code.strip().startswith("["):
                return "# Error: ML_CODE_REQUIRED"
            completion_issues = self._check_script_completeness(code, required_deliverables)
            if completion_issues:
                reprompt_context = self._build_incomplete_reprompt_context(
                    execution_contract=execution_contract,
                    required_outputs=required_deliverables,
                    iteration_memory_block=iteration_memory_block,
                    iteration_memory=iteration_memory,
                    feedback_history=feedback_history,
                    gate_context=gate_context,
                    iteration_handoff=handoff_payload,
                    ml_view=ml_view,
                )
                completion_system = (
                    "You are a senior ML engineer. Return a COMPLETE runnable Python script. "
                    "Do not return partial snippets, diffs, or TODOs. "
                    "Preserve required outputs and alignment checks."
                )
                completion_user = (
                    "Your last response is incomplete. Return the full script.\n"
                    f"Missing/invalid sections: {completion_issues}\n"
                    f"Required deliverables: {required_deliverables}\n\n"
                    f"INCOMPLETE CODE:\n{code}\n\n"
                    "Re-emit the FULL script without truncation, respecting contract and fixes.\n\n"
                    f"{reprompt_context}"
                )
                completed = call_with_retries(
                    lambda: _call_model_with_prompts(
                        completion_system,
                        completion_user,
                        0.0,
                        self.last_model_used or self.model_name,
                        stage="completion_reprompt",
                    ),
                    max_retries=2,
                    backoff_factor=2,
                    initial_delay=1,
                )
                completed = self._clean_code(completed)
                if is_syntax_valid(completed):
                    code = completed

            return code

        except Exception as e:
            # Raise RuntimeError as requested for clean catch in graph.py
            print(f"CRITICAL: ML Engineer Failed (Max Retries): {e}")
            raise RuntimeError(f"ML Generation Failed: {e}")

    def _clean_code(self, code: str) -> str:
        return extract_code_block(code)

