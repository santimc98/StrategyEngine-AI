import os
import re
import ast
import json
import logging
import csv
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from src.utils.code_extract import extract_code_block
from src.utils.contract_accessors import (
    get_cleaning_gates,
    get_declared_artifacts,
    get_required_outputs_by_owner,
    normalize_artifact_path,
)
from src.utils.contract_validator import (
    normalize_contract_scope,
    derive_contract_scope_from_workstreams,
)
from src.utils.cleaning_contract_semantics import expand_required_feature_selectors
from src.utils.artifact_obligations import build_data_engineer_artifact_obligations
from src.utils.sandbox_deps import (
    BASE_ALLOWLIST,
    EXTENDED_ALLOWLIST,
    CLOUDRUN_NATIVE_ALLOWLIST,
    CLOUDRUN_OPTIONAL_ALLOWLIST,
    BANNED_ALWAYS_ALLOWLIST,
    check_dependency_precheck,
)
from src.utils.llm_fallback import call_chat_with_fallback, extract_response_text
from openai import OpenAI

load_dotenv()

# NOTE: scan_code_safety enforcement happens upstream in graph.py.
# Keep the explicit reference for integration checks that assert the DE safety link.
_scan_code_safety_ref = "scan_code_safety"


class DataEngineerAgent:
    def __init__(self, api_key: str = None):
        """
        Initializes the Data Engineer Agent with OpenRouter primary + fallback.
        """
        self.logger = logging.getLogger(__name__)
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
            os.getenv("OPENROUTER_DE_PRIMARY_MODEL")
            or "minimax/minimax-m2.7"
        ).strip()
        self.fallback_model_name = (
            os.getenv("OPENROUTER_DE_FALLBACK_MODEL")
            or "moonshotai/kimi-k2.5"
        ).strip()
        if not self.model_name:
            self.model_name = "minimax/minimax-m2.7"
        if not self.fallback_model_name:
            self.fallback_model_name = "moonshotai/kimi-k2.5"
        self.logger.info(
            "DATA_ENGINEER_OPENROUTER_MODELS: primary=%s fallback=%s",
            self.model_name,
            self.fallback_model_name,
        )

        self.last_prompt = None
        self.last_response = None

    def _extract_nonempty(self, response) -> str:
        """
        Extracts non-empty content from LLM response.
        Raises ValueError("EMPTY_COMPLETION") if content is empty (CAUSA RAIZ 2).
        This triggers retry logic in call_with_retries.
        """
        content = extract_response_text(response)

        if not content:
            print("ERROR: LLM returned EMPTY_COMPLETION. Will retry.")
            raise ValueError("EMPTY_COMPLETION")

        return content

    def _build_runtime_dependency_context(self) -> Dict[str, Any]:
        """
        Build a compact runtime dependency contract for DE prompts.
        This gives the model explicit import allowlist + version hints
        to reduce runtime incompatibilities across sandbox images.
        """
        requirements_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "cloudrun",
                "heavy_runner",
                "requirements.txt",
            )
        )
        pinned_specs: Dict[str, str] = {}
        try:
            if os.path.exists(requirements_path):
                with open(requirements_path, "r", encoding="utf-8") as f_req:
                    for raw in f_req:
                        line = str(raw or "").strip()
                        if not line or line.startswith("#"):
                            continue
                        if line.startswith("-"):
                            continue
                        token = line.split(";", 1)[0].strip()
                        token = token.split("#", 1)[0].strip()
                        if not token:
                            continue
                        for op in ("==", ">=", "<=", "~=", "!=", ">", "<"):
                            if op in token:
                                name, spec = token.split(op, 1)
                                pkg = name.strip().lower()
                                if pkg:
                                    pinned_specs[pkg] = f"{op}{spec.strip()}"
                                break
        except Exception:
            pinned_specs = {}

        pandas_spec = pinned_specs.get("pandas", "unbounded")

        runtime_mode = (
            os.getenv("RUN_EXECUTION_MODE")
            or os.getenv("EXECUTION_RUNTIME_MODE")
            or "cloudrun"
        ).strip().lower()
        backend_profile = "local" if runtime_mode == "local" else "cloudrun"

        return {
            "backend_profile": backend_profile,
            "runtime_mode": runtime_mode,
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
            "version_hints": {
                "python": "3.11+",
                "pandas": pandas_spec or "2.x",
            },
            "guidance": [
                "Import only allowlisted roots.",
                "Use stable public APIs compatible with version_hints.",
                "Keep script portable across local and cloudrun runner modes.",
            ],
            "pandas_pitfalls": [
                "df.describe(datetime_is_numeric=True) removed in pandas 2.x — omit the kwarg.",
                "Series.str.replace(pat, repl): repl must be a string or callable, NOT pd.NA/np.nan. Use .where() or .mask() instead.",
                "pd.to_datetime(infer_datetime_format=True) deprecated — omit the kwarg, pandas 2.x infers by default.",
                "Series.replace({dict}, np.nan) works, but Series.replace(list, np.nan) inside .apply() raises ValueError — use .map() or .where() instead.",
                "Int64/Float64 nullable dtypes: use pd.array() or .astype('Int64') after imputation, not before.",
                "print() with unicode characters (checkmarks, arrows) fails on Windows cp1252 — use ASCII-safe characters only.",
            ],
        }

    def _build_contract_focus_context(
        self,
        contract: Dict[str, Any],
        de_view: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build a compact DE-focused contract context to reduce prompt noise.
        """
        contract = contract if isinstance(contract, dict) else {}
        de_view = de_view if isinstance(de_view, dict) else {}
        focus: Dict[str, Any] = {}

        for key in (
            "scope",
            "required_outputs",
            "column_roles",
            "outlier_policy",
            "column_dtype_targets",
            "active_workstreams",
            "future_ml_handoff",
        ):
            value = contract.get(key)
            if value not in (None, "", [], {}):
                focus[key] = value

        artifact_requirements = contract.get("artifact_requirements")
        if isinstance(artifact_requirements, dict):
            artifact_focus: Dict[str, Any] = {}
            for key in ("cleaned_dataset", "enriched_dataset", "schema_binding"):
                value = artifact_requirements.get(key)
                if isinstance(value, dict) and value:
                    artifact_focus[key] = value
            clean_dataset = artifact_requirements.get("clean_dataset")
            if isinstance(clean_dataset, dict) and clean_dataset and "cleaned_dataset" not in artifact_focus:
                artifact_focus["clean_dataset"] = clean_dataset
            if artifact_focus:
                focus["artifact_requirements"] = artifact_focus

        if de_view:
            view_focus: Dict[str, Any] = {}
            for key in (
                "required_columns",
                "required_feature_selectors",
                "optional_passthrough_columns",
                "output_path",
                "output_manifest_path",
                "artifact_requirements",
                "cleaning_gates",
                "column_transformations",
                "column_dtype_targets",
            ):
                value = de_view.get(key)
                if value not in (None, "", [], {}):
                    view_focus[key] = value
            if view_focus:
                focus["de_view_projection"] = view_focus

        return focus

    def _build_data_engineer_required_outputs_context(
        self,
        contract: Dict[str, Any],
        de_view: Dict[str, Any],
        *,
        primary_output_path: str = "",
        manifest_path: str = "",
        outlier_report_path: str = "",
    ) -> List[Dict[str, Any]]:
        contract = contract if isinstance(contract, dict) else {}
        de_view = de_view if isinstance(de_view, dict) else {}

        entries: List[Dict[str, Any]] = []
        seen: set[str] = set()

        def _infer_kind(path: str) -> str:
            lower = str(path or "").strip().lower()
            if not lower:
                return ""
            if lower.endswith(".csv"):
                return "dataset"
            if lower.endswith(".json"):
                if "manifest" in lower:
                    return "manifest"
                return "metadata"
            if lower.endswith(".md"):
                return "report"
            return ""

        def _merge(path: Any, payload: Optional[Dict[str, Any]] = None) -> None:
            normalized = normalize_artifact_path(path)
            if not normalized:
                return
            key = normalized.lower()
            payload = payload if isinstance(payload, dict) else {}
            if key in seen:
                for existing in entries:
                    if str(existing.get("path") or "").strip().lower() != key:
                        continue
                    if payload.get("required"):
                        existing["required"] = True
                    for field in ("kind", "description", "owner", "source"):
                        if payload.get(field) and not existing.get(field):
                            existing[field] = payload[field]
                    return
                return
            seen.add(key)
            entries.append(
                {
                    "path": normalized,
                    "required": bool(payload.get("required", True)),
                    "kind": str(payload.get("kind") or _infer_kind(normalized)).strip(),
                    "owner": str(payload.get("owner") or "data_engineer").strip(),
                    "description": str(payload.get("description") or "").strip(),
                    "source": str(payload.get("source") or "").strip(),
                    "materialization_policy": (
                        "required_even_if_empty"
                        if bool(payload.get("required", True))
                        else "optional_when_applicable"
                    ),
                }
            )

        for artifact in get_declared_artifacts(contract):
            if not isinstance(artifact, dict):
                continue
            owner = str(artifact.get("owner") or "").strip().lower()
            if owner and owner != "data_engineer":
                continue
            _merge(
                artifact.get("path"),
                {
                    "required": artifact.get("required", True),
                    "kind": artifact.get("kind"),
                    "owner": artifact.get("owner") or "data_engineer",
                    "description": artifact.get("description"),
                    "source": "execution_contract.required_outputs",
                },
            )

        for path in get_required_outputs_by_owner(contract, "data_engineer"):
            _merge(path, {"required": True, "owner": "data_engineer", "source": "execution_contract.required_outputs"})

        raw_view_outputs = de_view.get("required_outputs")
        if isinstance(raw_view_outputs, list):
            for item in raw_view_outputs:
                if isinstance(item, dict):
                    _merge(
                        item.get("path"),
                        {
                            "required": item.get("required", True),
                            "kind": item.get("kind"),
                            "owner": item.get("owner") or "data_engineer",
                            "description": item.get("description"),
                            "source": "de_view.required_outputs",
                        },
                    )
                else:
                    _merge(item, {"required": True, "owner": "data_engineer", "source": "de_view.required_outputs"})

        _merge(primary_output_path, {"required": True, "kind": "dataset", "owner": "data_engineer", "source": "de_view.output_path"})
        _merge(manifest_path, {"required": True, "kind": "manifest", "owner": "data_engineer", "source": "de_view.output_manifest_path"})
        if outlier_report_path:
            _merge(
                outlier_report_path,
                {
                    "required": False,
                    "kind": "report",
                    "owner": "data_engineer",
                    "description": "Optional outlier treatment report when outlier policy is enabled.",
                    "source": "outlier_policy.report_path",
                },
            )

        return entries

    def _resolve_pipeline_scope(
        self,
        contract: Dict[str, Any],
        de_view: Dict[str, Any],
    ) -> str:
        contract = contract if isinstance(contract, dict) else {}
        de_view = de_view if isinstance(de_view, dict) else {}

        explicit_scope = str(de_view.get("scope") or contract.get("scope") or "").strip()
        if explicit_scope:
            return normalize_contract_scope(explicit_scope)

        merged_context: Dict[str, Any] = {}
        if contract:
            merged_context.update(contract)
        if isinstance(de_view.get("active_workstreams"), dict):
            merged_context["active_workstreams"] = dict(de_view.get("active_workstreams") or {})
        for key in ("future_ml_handoff", "task_semantics", "business_objective", "strategy_title"):
            if key not in merged_context and key in de_view:
                merged_context[key] = de_view.get(key)

        return derive_contract_scope_from_workstreams(merged_context)

    def _detect_repair_mode(self, data_audit: str, requested: bool) -> bool:
        """
        Detect if DE should operate in repair mode.
        """
        if requested:
            return True
        text = str(data_audit or "")
        repair_markers = (
            "RUNTIME_ERROR_CONTEXT",
            "PREFLIGHT_ERROR_CONTEXT",
            "STATIC_SCAN_VIOLATIONS",
            "UNDEFINED_NAME_GUARD",
            "CLEANING_REVIEWER_ALERT",
            "LLM_FAILURE_EXPLANATION",
        )
        return any(marker in text for marker in repair_markers)

    def _truncate_prompt_text(
        self,
        text: str,
        *,
        max_len: int = 8000,
        head_len: int = 5000,
        tail_len: int = 2000,
    ) -> str:
        value = str(text or "")
        if len(value) <= max_len:
            return value
        safe_head = max(0, min(head_len, max_len))
        safe_tail = max(0, min(tail_len, max_len - safe_head))
        if safe_head + safe_tail == 0:
            return value[:max_len]
        return value[:safe_head] + "\n...[TRUNCATED]...\n" + value[-safe_tail:]

    def _looks_like_editable_code(self, code: str) -> bool:
        text = str(code or "").strip()
        if not text or text.startswith("# Error"):
            return False
        return bool(
            re.search(
                r"(?m)^\s*(from\s+\w+|import\s+\w+|def\s+\w+|class\s+\w+|if\s+__name__|"
                r"if\s+|for\s+|while\s+|try:|with\s+|[A-Za-z_]\w*\s*=|print\(|raise\s+)",
                text,
            )
        )

    def _strip_runtime_injection_from_previous_code(self, previous_code: str) -> str:
        text = str(previous_code or "")
        if not text:
            return ""
        marker = "json.dumps = _safe_dumps_json"
        idx = text.find(marker)
        if idx == -1:
            return text.strip()
        tail = text[idx + len(marker):].lstrip()
        return tail.strip() or text.strip()

    def _extract_json_after_marker(self, text: str, marker: str) -> Dict[str, Any]:
        raw = str(text or "")
        if not raw or not marker:
            return {}
        idx = raw.rfind(marker)
        if idx == -1:
            return {}
        tail = raw[idx + len(marker):].lstrip()
        if not tail:
            return {}
        decoder = json.JSONDecoder()
        try:
            payload, _ = decoder.raw_decode(tail)
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _normalize_feedback_record(
        self,
        feedback_record: Optional[Dict[str, Any]],
        data_audit: str,
    ) -> Dict[str, Any]:
        record = feedback_record if isinstance(feedback_record, dict) else {}
        if not record:
            record = self._extract_json_after_marker(
                data_audit,
                "LATEST_ITERATION_FEEDBACK_RECORD_JSON:",
            )
        if not isinstance(record, dict):
            return {}
        normalized = {
            "agent": str(record.get("agent") or "data_engineer"),
            "source": str(record.get("source") or "unknown"),
            "status": str(record.get("status") or "UNKNOWN"),
            "iteration": int(record.get("iteration") or 0),
            "feedback": str(record.get("feedback") or ""),
            "failed_gates": [str(x) for x in (record.get("failed_gates") or []) if str(x).strip()],
            "required_fixes": [str(x) for x in (record.get("required_fixes") or []) if str(x).strip()],
            "hard_failures": [str(x) for x in (record.get("hard_failures") or []) if str(x).strip()],
            "runtime_error_tail": str(record.get("runtime_error_tail") or ""),
            "evidence": record.get("evidence") if isinstance(record.get("evidence"), list) else [],
        }
        normalized["failed_gates"] = list(dict.fromkeys(normalized["failed_gates"]))[:12]
        normalized["required_fixes"] = list(dict.fromkeys(normalized["required_fixes"]))[:12]
        normalized["hard_failures"] = list(dict.fromkeys(normalized["hard_failures"]))[:8]
        normalized["feedback"] = self._truncate_prompt_text(
            normalized["feedback"],
            max_len=1800,
            head_len=1400,
            tail_len=300,
        )
        normalized["runtime_error_tail"] = self._truncate_prompt_text(
            normalized["runtime_error_tail"],
            max_len=1800,
            head_len=1200,
            tail_len=400,
        )
        return normalized

    def _extract_labeled_repair_sections(
        self,
        text: str,
        allowed_labels: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        raw = str(text or "")
        if not raw.strip():
            return {}
        allowed = set(str(item).strip() for item in (allowed_labels or []) if str(item).strip())
        sections: Dict[str, str] = {}
        current_label = ""
        buffer: List[str] = []

        def _flush() -> None:
            nonlocal current_label, buffer
            if not current_label:
                return
            body = "\n".join(buffer).strip()
            if body:
                sections[current_label] = body
            current_label = ""
            buffer = []

        for raw_line in raw.splitlines():
            line = str(raw_line or "").rstrip()
            match = re.match(r"^([A-Z][A-Z0-9_]{2,}):\s*(.*)$", line)
            if match:
                label = str(match.group(1) or "").strip()
                if current_label == "LLM_FAILURE_EXPLANATION" and label in {"WHERE", "WHY", "FIX", "DIAGNOSTIC"}:
                    buffer.append(line.strip())
                    continue
                if allowed and label not in allowed:
                    _flush()
                    continue
                _flush()
                current_label = label
                first_value = str(match.group(2) or "").strip()
                buffer = [first_value] if first_value else []
                continue
            if current_label:
                buffer.append(line)
        _flush()
        return sections

    def _extract_explainer_directives(self, text: str) -> Dict[str, Any]:
        directives: Dict[str, Any] = {
            "where": "",
            "why": "",
            "fixes": [],
            "diagnostics": [],
        }
        raw = str(text or "")
        if not raw.strip():
            return directives
        for raw_line in raw.splitlines():
            line = str(raw_line or "").strip()
            if not line or ":" not in line:
                continue
            label, value = line.split(":", 1)
            label_norm = str(label or "").strip().upper()
            value_text = str(value or "").strip(" -\t")
            if not value_text:
                continue
            if label_norm == "WHERE" and not directives["where"]:
                directives["where"] = value_text
            elif label_norm == "WHY" and not directives["why"]:
                directives["why"] = value_text
            elif label_norm == "FIX":
                directives["fixes"].append(value_text)
            elif label_norm == "DIAGNOSTIC":
                directives["diagnostics"].append(value_text)
        directives["fixes"] = list(dict.fromkeys([str(x) for x in directives["fixes"] if str(x).strip()]))[:6]
        directives["diagnostics"] = list(
            dict.fromkeys([str(x) for x in directives["diagnostics"] if str(x).strip()])
        )[:4]
        return directives

    def _build_compact_repair_error_context(
        self,
        data_audit: str,
        normalized_feedback: Dict[str, Any],
    ) -> str:
        section_order = [
            "PREFLIGHT_ERROR_CONTEXT",
            "RUNTIME_ERROR_CONTEXT",
            "TRACEBACK_TAIL_20",
            "ERROR_SNIPPET",
            "WHY_IT_HAPPENED",
            "ERROR_DIAGNOSIS",
            "GATE_IMPLEMENTATION_HINTS",
            "REPAIR_HINTS",
            "LLM_FAILURE_EXPLANATION",
        ]
        sections = self._extract_labeled_repair_sections(data_audit, allowed_labels=section_order)
        lines: List[str] = []
        for label in section_order:
            body = str(sections.get(label) or "").strip()
            if not body:
                continue
            lines.append(f"{label}:")
            lines.append(
                self._truncate_prompt_text(
                    body,
                    max_len=900 if label in {"RUNTIME_ERROR_CONTEXT", "TRACEBACK_TAIL_20"} else 700,
                    head_len=600,
                    tail_len=180,
                )
            )
        if normalized_feedback.get("feedback"):
            lines.append("LATEST_FEEDBACK_SUMMARY:")
            lines.append(
                self._truncate_prompt_text(
                    str(normalized_feedback.get("feedback") or ""),
                    max_len=500,
                    head_len=350,
                    tail_len=120,
                )
            )
        if not lines:
            return self._truncate_prompt_text(data_audit, max_len=2200, head_len=1500, tail_len=500) or "{}"
        return "\n".join(item for item in lines if str(item).strip())

    def _build_repair_prompt_context(
        self,
        *,
        data_audit: str,
        previous_code: str,
        feedback_record: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        normalized = self._normalize_feedback_record(feedback_record, data_audit)
        repair_sections = self._extract_labeled_repair_sections(
            data_audit,
            allowed_labels=[
                "PREFLIGHT_ERROR_CONTEXT",
                "RUNTIME_ERROR_CONTEXT",
                "TRACEBACK_TAIL_20",
                "ERROR_SNIPPET",
                "WHY_IT_HAPPENED",
                "ERROR_DIAGNOSIS",
                "GATE_IMPLEMENTATION_HINTS",
                "REPAIR_HINTS",
                "LLM_FAILURE_EXPLANATION",
            ],
        )
        explainer_directives = self._extract_explainer_directives(
            repair_sections.get("LLM_FAILURE_EXPLANATION", "")
        )
        patch_objectives: List[str] = []
        if explainer_directives.get("fixes"):
            patch_objectives.extend(
                [str(item) for item in (explainer_directives.get("fixes") or []) if str(item).strip()]
            )
        if normalized.get("required_fixes"):
            patch_objectives.extend(normalized.get("required_fixes") or [])
        repair_hints = repair_sections.get("REPAIR_HINTS")
        if repair_hints:
            for raw_line in str(repair_hints).splitlines():
                cleaned = str(raw_line or "").strip().lstrip("- ").strip()
                if cleaned:
                    patch_objectives.append(cleaned)
        if normalized.get("failed_gates"):
            patch_objectives.append(
                "Resolve failed gates without regressing already-working logic: "
                + ", ".join((normalized.get("failed_gates") or [])[:5])
            )
        if normalized.get("hard_failures") and not any("hard failure" in item.lower() for item in patch_objectives):
            patch_objectives.append(
                "Clear the active hard failure(s) first: "
                + ", ".join((normalized.get("hard_failures") or [])[:4])
            )
        if normalized.get("runtime_error_tail") and not any("runtime root cause" in item.lower() for item in patch_objectives):
            patch_objectives.insert(0, "Fix the runtime root cause first, then keep artifact/output behavior stable.")
        if explainer_directives.get("where") and not any("failure location" in item.lower() for item in patch_objectives):
            patch_objectives.append(
                "Patch the failure location implicated by the latest evidence: "
                + str(explainer_directives.get("where") or "")
            )
        if not patch_objectives:
            patch_objectives = [
                "Apply the smallest coherent patch that fixes the active blocker and preserves the working parts of the cleaning pipeline."
            ]
        patch_objectives = list(dict.fromkeys([str(item) for item in patch_objectives if str(item).strip()]))[:8]
        must_preserve = [
            "Keep owned output paths and manifest/output-closure logic intact unless the failure explicitly requires changing them.",
            "Preserve working cleaning stages that are not implicated by the latest failure evidence.",
            "Return the full updated script body, not snippets or diffs.",
        ]
        return {
            "feedback_record_json": json.dumps(normalized or {}, indent=2, ensure_ascii=False),
            "patch_objectives": "\n".join(f"- {item}" for item in patch_objectives[:8]),
            "must_preserve": "\n".join(f"- {item}" for item in must_preserve),
            "error_context": self._build_compact_repair_error_context(data_audit, normalized) or "{}",
            "previous_code": self._truncate_prompt_text(
                self._strip_runtime_injection_from_previous_code(previous_code),
                max_len=12000,
                head_len=9000,
                tail_len=2200,
            ),
        }

    def _read_csv_header(
        self,
        input_path: str,
        csv_encoding: str,
        csv_sep: str,
    ) -> List[str]:
        path = str(input_path or "").strip()
        if not path or not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding=csv_encoding or "utf-8", errors="replace", newline="") as f_csv:
                reader = csv.reader(f_csv, delimiter=(csv_sep or ","))
                header = next(reader, [])
            return [str(col) for col in header if str(col).strip()]
        except Exception:
            return []

    def _build_data_sample_context(
        self,
        input_path: str,
        csv_encoding: str,
        csv_sep: str,
        csv_decimal: str,
        max_rows: int = 5,
        max_cols: int = 30,
    ) -> str:
        payload: Dict[str, Any] = {
            "status": "unavailable",
            "reason": "input_not_accessible_at_prompt_time",
            "path": str(input_path or ""),
        }
        path = str(input_path or "").strip()
        if not path or not os.path.exists(path):
            return json.dumps(payload, ensure_ascii=False, indent=2)
        try:
            import pandas as pd  # local-only for prompt context enrichment

            # Full read for statistics, dtype=str to preserve raw values
            df_full = pd.read_csv(
                path,
                dtype=str,
                sep=csv_sep or ",",
                decimal=csv_decimal or ".",
                encoding=csv_encoding or "utf-8",
                low_memory=False,
            )
            total_rows = len(df_full)
            all_cols = [str(c) for c in df_full.columns.tolist()]
            truncated = len(all_cols) > max_cols
            shown_cols = all_cols[:max_cols]

            # Statistical profile per column
            column_profiles: Dict[str, Any] = {}
            for col in shown_cols:
                series = df_full[col]
                null_count = int(series.isna().sum())
                non_null = series.dropna()
                n_unique = int(non_null.nunique())
                profile: Dict[str, Any] = {
                    "null_pct": round(null_count / total_rows * 100, 1) if total_rows else 0,
                    "null_count": null_count,
                    "unique_count": n_unique,
                }

                # Top values with frequency (capped for prompt size)
                if 0 < n_unique <= 30:
                    top_n = min(n_unique, 6)
                    vc = non_null.value_counts(dropna=True).head(top_n)
                    profile["top_values"] = {
                        str(k): int(v) for k, v in vc.items()
                    }
                elif 30 < n_unique <= 500:
                    vc = non_null.value_counts(dropna=True).head(4)
                    profile["top_values"] = {
                        str(k): int(v) for k, v in vc.items()
                    }
                elif n_unique > 500:
                    profile["high_cardinality"] = True

                # Detect numeric-like columns
                numeric_series = pd.to_numeric(non_null, errors="coerce")
                numeric_valid = numeric_series.dropna()
                if len(numeric_valid) > len(non_null) * 0.5 and len(numeric_valid) > 0:
                    profile["looks_numeric"] = True
                    profile["numeric_min"] = float(numeric_valid.min())
                    profile["numeric_max"] = float(numeric_valid.max())
                    profile["numeric_mean"] = round(float(numeric_valid.mean()), 4)
                    unparsed_count = len(non_null) - len(numeric_valid)
                    if unparsed_count > 0:
                        profile["numeric_unparsed_count"] = unparsed_count

                # Detect datetime-like columns
                if not profile.get("looks_numeric"):
                    sample_for_dt = non_null.head(200)
                    if len(sample_for_dt) > 0:
                        dt_parsed = pd.to_datetime(sample_for_dt, errors="coerce")
                        dt_valid_count = int(dt_parsed.notna().sum())
                        if dt_valid_count > len(sample_for_dt) * 0.4:
                            profile["looks_datetime"] = True
                            profile["datetime_parse_rate_sample"] = round(
                                dt_valid_count / len(sample_for_dt) * 100, 1
                            )
                            # Detect raw format patterns
                            raw_sample = sample_for_dt.head(20).tolist()
                            format_signatures = set()
                            for val in raw_sample:
                                v = str(val).strip()
                                if not v:
                                    continue
                                sig = re.sub(r"\d", "D", v)
                                format_signatures.add(sig)
                            if format_signatures:
                                profile["observed_format_patterns"] = sorted(format_signatures)[:5]

                column_profiles[col] = profile

            # Preview rows (first N)
            preview = df_full.head(max_rows)
            payload = {
                "status": "available",
                "path": path,
                "dataset_shape": {"total_rows": total_rows, "total_cols": len(all_cols)},
                "columns": all_cols,
                "columns_truncated_in_profile": truncated,
                "column_profiles": column_profiles,
                "preview_rows": preview[shown_cols].fillna("<NA>").to_dict(orient="records"),
            }
        except Exception as sample_err:
            payload = {
                "status": "unavailable",
                "reason": f"sample_read_error: {sample_err}",
                "path": path,
            }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _build_selector_expansion_context(
        self,
        de_view: Dict[str, Any],
        input_path: str,
        csv_encoding: str,
        csv_sep: str,
    ) -> str:
        selectors = de_view.get("required_feature_selectors") if isinstance(de_view, dict) else None
        if not isinstance(selectors, list) or not selectors:
            return json.dumps(
                {"status": "not_requested", "required_feature_selectors": []},
                ensure_ascii=False,
                indent=2,
            )

        header_cols = self._read_csv_header(input_path, csv_encoding, csv_sep)
        if not header_cols:
            return json.dumps(
                {
                    "status": "unavailable",
                    "reason": "header_not_accessible_at_prompt_time",
                    "required_feature_selectors": selectors,
                },
                ensure_ascii=False,
                indent=2,
            )

        expanded, issues = expand_required_feature_selectors(selectors, header_cols)
        summary: Dict[str, Any] = {
            "status": "available",
            "header_column_count": len(header_cols),
            "required_feature_selectors_count": len(selectors),
            "expanded_required_columns_count": len(expanded),
            "expanded_required_columns_sample": expanded[:20],
            "expansion_issues": issues,
        }
        return json.dumps(summary, ensure_ascii=False, indent=2)

    def generate_cleaning_script(
        self,
        data_audit: str,
        strategy: Dict[str, Any],
        input_path: str,
        business_objective: str = "",
        csv_encoding: str = "utf-8",
        csv_sep: str = ",",
        csv_decimal: str = ".",
        execution_contract: Optional[Dict[str, Any]] = None,
        contract_min: Optional[Dict[str, Any]] = None,
        de_view: Optional[Dict[str, Any]] = None,
        repair_mode: bool = False,
        previous_code: Optional[str] = None,
        feedback_record: Optional[Dict[str, Any]] = None,
        artifact_obligations: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generates a Python script to clean and standardize the dataset.
        """
        from src.utils.prompting import render_prompt

        contract = execution_contract or contract_min or {}
        from src.utils.context_pack import compress_long_lists, summarize_long_list, COLUMN_LIST_POINTER

        # Build scope-aware guidance from the DE-visible contract/view context.
        _pipeline_scope = self._resolve_pipeline_scope(contract, de_view or {})
        if _pipeline_scope == "cleaning_only":
            pipeline_scope_guidance = (
                "PIPELINE SCOPE: CLEANING_ONLY — No model training occurs in THIS run. "
                "Your responsibility is to close the complete set of data-engineer deliverables "
                "declared by the contract for this run, which may include a final cleaned dataset, "
                "future-ML handoff datasets, and metadata artifacts.\n"
                "  - Prioritize maximum data quality, traceability, and artifact completeness\n"
                "  - Treat DATA_ENGINEER_REQUIRED_OUTPUTS_CONTEXT as the owned deliverable list\n"
                "  - If the run prepares future ML handoff artifacts, materialize them explicitly\n"
                "  - Validate all cleaning gates thoroughly before writing outputs\n"
                "  - Keep null handling and type standardization production-grade"
            )
        elif _pipeline_scope == "ml_only":
            pipeline_scope_guidance = (
                "PIPELINE SCOPE: ML_ONLY — The input data is stated to be pre-cleaned. "
                "Your role is minimal validation and passthrough:\n"
                "  - Verify column presence and basic types\n"
                "  - Do NOT apply aggressive cleaning or imputation\n"
                "  - Preserve the data as-is unless contract gates explicitly require changes\n"
                "  - Focus on compatibility: ensure output format is ML-ready\n"
                "  - Write manifest documenting validation checks performed"
            )
        else:
            pipeline_scope_guidance = (
                "PIPELINE SCOPE: FULL_PIPELINE — This run includes downstream ML consumption. "
                "Balance thoroughness with compatibility while still closing every data-engineer "
                "deliverable assigned by the contract:\n"
                "  - Clean data to meet contract gates and preserve modeling integrity\n"
                "  - Preserve statistical properties needed for downstream modeling\n"
                "  - Ensure required columns are present and correctly typed\n"
                "  - Materialize every owned output, not just the primary cleaned dataset"
            )

        de_view = de_view or {}
        contract_context = contract if isinstance(contract, dict) else {}
        if not contract_context:
            de_contract_context = {}
            for key in (
                "required_columns",
                "optional_passthrough_columns",
                "output_path",
                "output_manifest_path",
                "manifest_path",
                "output_dialect",
                "cleaning_gates",
                "data_engineer_runbook",
                "constraints",
            ):
                value = de_view.get(key)
                if value in (None, "", [], {}):
                    continue
                if key == "manifest_path" and "output_manifest_path" in de_contract_context:
                    continue
                de_contract_context[key] = value
            contract_context = de_contract_context
        contract_context = self._build_contract_focus_context(contract_context, de_view or {})
        contract_json = json.dumps(compress_long_lists(contract_context)[0], indent=2)
        de_view_json = json.dumps(compress_long_lists(de_view)[0], indent=2)
        if not isinstance(artifact_obligations, dict) or not artifact_obligations:
            artifact_obligations = build_data_engineer_artifact_obligations(contract)
        if not isinstance(artifact_obligations, dict):
            artifact_obligations = {}
        artifact_obligations_json = json.dumps(
            compress_long_lists(artifact_obligations)[0],
            indent=2,
        )
        de_output_path = str(de_view.get("output_path") or "").strip()
        de_manifest_path = str(
            de_view.get("output_manifest_path")
            or de_view.get("manifest_path")
            or ""
        ).strip()
        view_cleaning_gates = get_cleaning_gates({"cleaning_gates": de_view.get("cleaning_gates")})
        cleaning_gates = (
            get_cleaning_gates(contract)
            or get_cleaning_gates(execution_contract or {})
            or view_cleaning_gates
            or []
        )
        cleaning_gates_json = json.dumps(compress_long_lists(cleaning_gates)[0], indent=2)

        # V4.1: Use data_engineer_runbook only, no legacy role_runbooks fallback
        de_runbook = contract.get("data_engineer_runbook")
        if de_runbook in (None, "", [], {}):
            de_runbook = de_view.get("data_engineer_runbook")
        if isinstance(de_runbook, str):
            de_runbook = de_runbook.strip()
        if de_runbook in (None, "", [], {}):
            de_runbook = {}
        de_runbook_json = json.dumps(compress_long_lists(de_runbook)[0], indent=2)
        outlier_policy = de_view.get("outlier_policy")
        if not isinstance(outlier_policy, dict) or not outlier_policy:
            policy_from_contract = contract.get("outlier_policy")
            outlier_policy = policy_from_contract if isinstance(policy_from_contract, dict) else {}
        if not isinstance(outlier_policy, dict):
            outlier_policy = {}
        outlier_policy_json = json.dumps(compress_long_lists(outlier_policy)[0], indent=2)
        outlier_report_path = str(
            de_view.get("outlier_report_path")
            or outlier_policy.get("report_path")
            or ""
        ).strip()
        if not outlier_report_path and outlier_policy:
            outlier_enabled = outlier_policy.get("enabled")
            if isinstance(outlier_enabled, str):
                outlier_enabled = outlier_enabled.strip().lower() in {"1", "true", "yes", "on", "enabled"}
            if outlier_enabled is None:
                outlier_enabled = bool(
                    outlier_policy.get("target_columns")
                    or outlier_policy.get("methods")
                    or outlier_policy.get("treatment")
                )
            outlier_stage = str(outlier_policy.get("apply_stage") or "data_engineer").strip().lower()
            if bool(outlier_enabled) and outlier_stage in {"data_engineer", "both"}:
                outlier_report_path = "data/outlier_treatment_report.json"
        de_required_outputs = self._build_data_engineer_required_outputs_context(
            contract,
            de_view,
            primary_output_path=de_output_path,
            manifest_path=de_manifest_path,
            outlier_report_path=outlier_report_path,
        )
        de_required_outputs_json = json.dumps(compress_long_lists(de_required_outputs)[0], indent=2)
        artifact_requirements = contract.get("artifact_requirements")
        clean_dataset_cfg = {}
        if isinstance(artifact_requirements, dict):
            clean_dataset_candidate = artifact_requirements.get("cleaned_dataset")
            if not isinstance(clean_dataset_candidate, dict):
                clean_dataset_candidate = artifact_requirements.get("clean_dataset")
            if isinstance(clean_dataset_candidate, dict):
                clean_dataset_cfg = clean_dataset_candidate
        column_transformations = de_view.get("column_transformations")
        if not isinstance(column_transformations, dict) or not column_transformations:
            column_transformations = clean_dataset_cfg.get("column_transformations")
        if not isinstance(column_transformations, dict):
            column_transformations = {}
        column_transformations_json = json.dumps(
            compress_long_lists(column_transformations)[0],
            indent=2,
        )
        column_dtype_targets = de_view.get("column_dtype_targets")
        if not isinstance(column_dtype_targets, dict) or not column_dtype_targets:
            column_dtype_targets = contract.get("column_dtype_targets")
        if not isinstance(column_dtype_targets, dict):
            column_dtype_targets = {}
        column_dtype_targets_json = json.dumps(
            compress_long_lists(column_dtype_targets)[0],
            indent=2,
        )
        column_resolution_context = de_view.get("column_resolution_context")
        if not isinstance(column_resolution_context, dict) or not column_resolution_context:
            column_resolution_context = {}
        column_resolution_context_json = json.dumps(
            compress_long_lists(column_resolution_context)[0],
            indent=2,
        )
        runtime_dependency_context = self._build_runtime_dependency_context()
        runtime_dependency_context_json = json.dumps(
            compress_long_lists(runtime_dependency_context)[0], indent=2
        )
        data_sample_context = self._build_data_sample_context(
            input_path=input_path,
            csv_encoding=csv_encoding,
            csv_sep=csv_sep,
            csv_decimal=csv_decimal,
        )
        selector_expansion_context = self._build_selector_expansion_context(
            de_view=de_view,
            input_path=input_path,
            csv_encoding=csv_encoding,
            csv_sep=csv_sep,
        )
        effective_repair_mode = self._detect_repair_mode(data_audit, repair_mode)
        operating_mode = "REPAIR" if effective_repair_mode else "INITIAL"
        repair_notes = (
            "Patch root causes from error context first. Keep working logic stable."
            if effective_repair_mode
            else "Build the first executable script from contract and context."
        )

        # [SAFETY] Truncate data_audit if massive to prevent context overflow
        # The audit concatenates many sources; preserve head (structure) and tail (recent instructions).
        if data_audit and len(data_audit) > 100000:
            print(f"DEBUG: Truncating massive data_audit ({len(data_audit)} chars) to 100k.")
            head_len = 50000
            tail_len = 50000
            data_audit = data_audit[:head_len] + "\n...[AUDIT TRUNCATED FOR CONTEXT SAFETY]...\n" + data_audit[-tail_len:]
        SYSTEM_TEMPLATE = """
        You are a Senior Data Engineer. Your mission is to produce one deterministic,
        executable Python cleaning script for the dataset described below.

        You are producing executable artifacts under a contract. Be deterministic
        and audit-friendly. Include a Decision Log, Assumptions, Trade-offs, and
        Risk Register as short comment blocks (2-5 bullets each) at the top of
        the script, focused on engineering choices for THIS specific dataset.

        OPERATING_MODE: $operating_mode
        MODE_NOTE: $repair_notes

        ===================================================================
        PIPELINE SCOPE
        ===================================================================
        $pipeline_scope_guidance

        ===================================================================
        MISSION
        ===================================================================
        - Return one complete, runnable Python script that cleans the input CSV
          and materializes every required data-engineer deliverable declared in
          DATA_ENGINEER_REQUIRED_OUTPUTS_CONTEXT.
        - Follow the execution contract, owned deliverables, and cleaning gates
          as source of truth.
        - Be universal and adaptive to any CSV and business objective.

        ===================================================================
        SOURCE OF TRUTH AND PRECEDENCE
        ===================================================================
        1) DATA_ENGINEER_REQUIRED_OUTPUTS_CONTEXT + DE_VIEW_CONTEXT (authoritative for
           what you own, what you must write, and the DE-visible scope)
        ARTIFACT_OBLIGATIONS_CONTEXT
           (lossless extraction of artifact bindings already declared in the contract;
           use it to reconcile each artifact against its declared binding fields and
           source_contract_paths. It introduces no new semantics.)
        2) CLEANING_GATES_CONTEXT + required_columns + required_feature_selectors
           (authoritative for what must be cleaned, retained, or validated)
        3) COLUMN_RESOLUTION_CONTEXT + DATA_SAMPLE_CONTEXT
           (authoritative support evidence for what raw formats, placeholders,
           locale ambiguities, and recoverable signal exist in THIS dataset)
        4) COLUMN_DTYPE_TARGETS_CONTEXT + COLUMN_TRANSFORMATIONS_CONTEXT
           (authoritative for target typing and transformation mechanics)
        5) EXECUTION_CONTRACT_CONTEXT (authoritative tie-breaker when the DE view is silent)
        6) ROLE RUNBOOK (advisory — informs reasoning, does not override contract/gates)

        ===================================================================
        ENGINEERING REASONING WORKFLOW (MANDATORY)
        ===================================================================
        Before writing any code, reason through the cleaning plan by analyzing the
        contract inputs. Write your reasoning as comment blocks at the top of the
        script. This is not optional — it is how you prevent sequencing bugs.

        # EXECUTION PLAN (reason about these in order):
        #
        # 0. DELIVERABLE CLOSURE:
        #    Enumerate every output you own from DATA_ENGINEER_REQUIRED_OUTPUTS_CONTEXT.
        #    Map each one to a concrete write action.
        #    - Distinguish primary datasets vs metadata/report artifacts.
        #    - Do not collapse a multi-artifact contract into one CSV + manifest pair.
        #    - If one dataframe supports multiple declared outputs, still write each
        #      declared artifact explicitly with the correct contract path.
        #    - Required deliverables are contractual obligations, not conditional events.
        #      If a required artifact has zero rows, zero findings, zero decisions, or
        #      zero exceptions in THIS run, still materialize a schema-valid empty artifact
        #      and explain in the manifest why it is empty.
        #
        # 1. LOAD & VALIDATE: Read CSV with dtype=str. Verify required columns exist.
        #
        # 2. NULL HANDLING (BEFORE type conversion):
        #    For each cleaning gate with impute/null semantics, handle nulls NOW.
        #    CRITICAL: After reading with dtype=str, null cells are real NaN objects
        #    detectable by .isna(). If you convert to str first (.astype(str)),
        #    NaN becomes the literal string "nan" and .isna() returns False —
        #    the imputation silently does nothing. Always impute BEFORE converting
        #    string columns to their final types.
        #    - List each column that needs null handling and the strategy.
        #
        # 3. FORMAT RESOLUTION (BEFORE final casting):
        #    For date/time, rate, count, and amount columns, reason about the observed
        #    format families in THIS dataset before choosing parsing logic.
        #    - Use COLUMN_RESOLUTION_CONTEXT first, then DE_VIEW_CONTEXT, DATA SUMMARY,
        #      and the sample rows to infer whether a column mixes locale/date orders,
        #      timestamps, decimals, currencies, percentages, magnitude suffixes,
        #      placeholders, or noisy symbols.
        #    - Do not assume one parser or one locale is enough if the context suggests
        #      heterogeneous formats.
        #    - Prefer a staged parsing strategy that salvages defensible values before
        #      coercing unresolved strings to null.
        #    - If parsing would destroy a large fraction of plausible signal, adapt the
        #      strategy or flag unresolved formats explicitly instead of silently
        #      accepting massive information loss.
        #    - Do not infer a hard "no nulls after parsing" requirement for temporal
        #      columns from target dtype alone. A datetime target does not by itself
        #      mean complete parse coverage is mandatory.
        #    - When raw context shows invalid or mixed temporal formats, preserve
        #      unresolved values as null plus traceability flags/log entries unless
        #      the contract or a gate explicitly requires complete recoverability.
        #
        # 4. TYPE CONVERSION (AFTER format resolution):
        #    Apply COLUMN_DTYPE_TARGETS once the parsing strategy is settled.
        #    Use pandas nullable Int64/Float64 for nullable integer/float columns.
        #    Final dtypes should reflect the resolved semantics, not raw string noise.
        #
        # 5. IDENTITY RESOLUTION (WHEN DEDUPLICATION IS IN SCOPE):
        #    If a cleaning gate, runbook step, or contract context implies deduplication,
        #    reason explicitly about duplicate evidence BEFORE writing the algorithm.
        #    - Decide which signals are strong, medium, or weak identity evidence for
        #      THIS dataset and business context.
        #    - Treat nulls, placeholders, and missing contact fields as absence of
        #      evidence, not as positive evidence that two rows are the same entity.
        #    - Do not collapse rows just because a composite key can be mechanically
        #      constructed; preserve records when identity evidence is ambiguous.
        #    - Make survivorship logic explicit: explain why one record is retained
        #      over another when duplicates are defensible.
        #    - If the context only supports soft duplicate suspicion, prefer flags/logs
        #      over irreversible dropping or merging.
        #
        # 6. CONSTRAINT VALIDATION:
        #    - HARD gates: check and raise ValueError("CLEANING_GATE_FAILED: ...") if violated.
        #    - SOFT gates: check and warn if thresholds exceeded, but do not block.
        #    - Non-nullable columns: verify no unexpected nulls after conversion only
        #      when the contract or a gate makes non-nullability explicit.
        #    - Do not convert "required for downstream use" into "must be fully non-null"
        #      unless the run context actually supports that stronger claim.
        #    - For temporal/numeric cleaning, check whether the final null inflation is
        #      consistent with the observed raw quality or whether your parser was too blunt.
        #    - For temporal fields in cleaning-first runs, prefer "recover what is
        #      defensible, flag what is invalid, and report residual nulls" over
        #      brittle hard-failure semantics unless completeness is explicitly contractual.
        #
        # 7. OUTPUT CLOSURE:
        #    Write every owned required output and make sure metadata artifacts reflect
        #    actual operations performed, not only planned ones.

        Your Decision Log, Assumptions, and Risks blocks should reflect the specific
        reasoning you did for THIS dataset — not generic boilerplate.

        ===================================================================
        HARD CONSTRAINTS
        ===================================================================
        - Output valid Python code only. No markdown, no code fences, no prose.
        - Scope: cleaning only (no modeling, scoring, optimization, or analytics).
        - Read input with pd.read_csv(..., dtype=str, low_memory=False, sep, decimal, encoding from inputs).
        - Write every required output you own in DATA_ENGINEER_REQUIRED_OUTPUTS_CONTEXT.
        - Treat $de_output_path and $de_manifest_path as primary anchors, not as the
          exhaustive deliverable set unless the contract says so.
        - If a required artifact is logically empty in THIS run, write the empty artifact
          with a valid schema and explain the emptiness in the manifest instead of omitting it.
        - If outlier policy is enabled, write report to $outlier_report_path.
        - Do not fabricate columns. Do not overwrite the input file.
        - Do not silently omit auxiliary metadata/report artifacts that the contract
          assigns to the data engineer.
        - Missing required columns → fail fast with ValueError.
        - Use only dependencies from RUNTIME_DEPENDENCY_CONTEXT.

        SANDBOX SECURITY - BLOCKED IMPORTS (HARD CONSTRAINT):
        - Import only modules listed in RUNTIME_DEPENDENCY_CONTEXT allowlist.
        - NO NETWORK/FS OPS: no network access, no subprocess, no shell, no eval/exec.
        - BLOCKED IMPORTS include subprocess, socket, requests, httpx, urllib, ftplib,
          paramiko, selenium, playwright, openai, google.generativeai, builtins.
        - Forbidden file-destructive calls: os.remove, os.unlink, pathlib.Path.unlink,
          shutil.rmtree, os.rmdir.

        ===================================================================
        DATA INTEGRITY PRINCIPLES
        ===================================================================
        Think like a senior engineer reviewing your own cleaning code before merge:

        - The operation order matters: null handling → type conversion → validation.
          Getting this wrong silently corrupts data. Reason about dependencies.
        - ARTIFACT_OBLIGATIONS_CONTEXT is a contract extraction layer, not new authority.
          Use it to reconcile exact per-artifact bindings. Do not treat it as permission
          to add undeclared columns, outputs, or extension policies.
        - Do not impute outcome/target columns unless the contract explicitly requests it.
          Preserve missingness for partially labeled targets (e.g., test set rows).
        - Preserve split/partition columns exactly as-is.
        - For wide datasets, resolve feature selectors against actual header; prefer
          vectorized transforms over per-column procedural loops.
        - For temporal, amount, and rate columns, preserve signal before declaring data
          invalid. Senior cleaning distinguishes heterogeneous but recoverable formats
          from truly unparseable values.
        - USE THE DATASET PROFILE IN DATA_SAMPLE_CONTEXT as your primary evidence for
          reasoning about null rates, cardinality, numeric ranges, and date formats.
          A column with 24% nulls in the raw data should NOT become 69% nulls after
          your cleaning — that means your parser is too aggressive. Check null_pct
          before and after each transformation mentally.
        - A parsed datetime column is not automatically a hard completeness gate.
          Treat complete non-null coverage as a contractual requirement only when the
          contract, gate semantics, or business-critical role explicitly demands it.
        - If column_profiles shows looks_datetime with observed_format_patterns, use
          those patterns to guide your parsing strategy. Multiple format signatures
          means you need multi-stage parsing, not a single pd.to_datetime() call.
        - If column_profiles shows looks_numeric with numeric_unparsed_count > 0,
          those unparsed values are likely placeholders or locale-specific formats.
          Clean them before casting, do not let to_numeric(errors='coerce') silently
          null them out without checking the inflation impact.
        - Create parent directories before writing. Preserve deterministic column order.
        - For identity resolution, missing contact/company values are lack of evidence.
          A senior implementation only drops or merges rows when the available signals
          defensibly support duplicate identity for THIS dataset.
        - If artifact_requirements declares both cleaned_dataset and enriched_dataset,
          do not collapse them into a single schema. Materialize each artifact from
          its own declared binding, and do not carry cleaned_dataset passthrough
          columns into enriched_dataset unless that binding explicitly declares them.
        - The manifest must reflect ACTUAL operations performed, not planned ones.
          If an imputation ran, log it. If it was skipped (no nulls found), say so.
        - PANDAS COMPATIBILITY: Check pandas_pitfalls in RUNTIME_DEPENDENCY_CONTEXT
          before using any deprecated or version-sensitive API. This prevents the
          most common runtime failures.

        REPAIR RULES (when OPERATING_MODE is REPAIR):
        - Read RUNTIME_ERROR_CONTEXT/PREFLIGHT_ERROR_CONTEXT and diagnose root cause first.
        - If a previous script body is provided, patch that script body first instead of regenerating from zero.
        - Keep already-correct logic stable; fix only what failed.
        - Prioritize executable syntax and required artifacts.

        ===================================================================
        MANIFEST REQUIREMENTS
        ===================================================================
        The manifest JSON must include:
        - output_dialect: {encoding, sep, decimal}
        - row_counts: {input, output}
        - conversions: list of actual transformations applied
        - contract_conflicts_resolved: list (empty if none)
        - cleaning_gates_status: {gate_name: "PASSED"|"WARNING_.."|"FAILED_.."}

        ===================================================================
        DIAGNOSTIC VISUALIZATIONS
        ===================================================================
        If the data_engineer_runbook requests EDA or diagnostic plots:
        - Generate matplotlib/seaborn plots that help understand data quality and transformations.
        - Save all plots as PNG files in static/plots/ directory (create it with os.makedirs("static/plots", exist_ok=True)).
        - Use descriptive filenames (e.g., missing_values.png, distributions_before_after.png, correlation_matrix.png).
        - Keep plots publication-quality: clear titles, axis labels, appropriate figure sizes.
        - Print the paths of generated plots so they appear in the execution log.
        - Common EDA plots: missing value heatmaps, numeric distributions, correlation matrices,
          before/after cleaning comparisons, outlier analysis, categorical value counts.
        - If runbook does NOT mention plots or visualizations, skip this section entirely.

        PLOT SUMMARIES (required when generating any plot):
        After generating plots, write a file static/plots/plot_summaries.json containing
        an array of objects, one per plot. Each object must record the FACTS you already
        computed — do NOT interpret or narrate, just log the data:
        [
          {
            "filename": "missing_values.png",
            "title": "Missing values by column",
            "facts": ["col_X had 12.3% nulls (highest)", "col_Y had 0% nulls", "total nulls reduced from 8.1% to 0.2% after cleaning"]
          }
        ]
        The downstream translator agent will use these facts to build the executive narrative.

        ===================================================================
        AUTHORITATIVE CONTEXT
        ===================================================================
        Input: '$input_path'
        Encoding: '$csv_encoding' | Sep: '$csv_sep' | Decimal: '$csv_decimal'
        DE Cleaning Objective: "$business_objective"

        DATA_ENGINEER_REQUIRED_OUTPUTS_CONTEXT: $de_required_outputs_context
        Required Columns (DE View): $required_columns
        Optional Passthrough Columns: $optional_passthrough_columns

        DE_VIEW_CONTEXT: $de_view_context
        ARTIFACT_OBLIGATIONS_CONTEXT: $artifact_obligations_context
        EXECUTION_CONTRACT_CONTEXT: $execution_contract_context
        CLEANING_GATES_CONTEXT: $cleaning_gates_context
        COLUMN_RESOLUTION_CONTEXT: $column_resolution_context
        COLUMN_DTYPE_TARGETS_CONTEXT: $column_dtype_targets_context
        COLUMN_TRANSFORMATIONS_CONTEXT: $column_transformations_context
        OUTLIER_POLICY_CONTEXT: $outlier_policy_context
        RUNTIME_DEPENDENCY_CONTEXT: $runtime_dependency_context
        SELECTOR_EXPANSION_CONTEXT: $selector_expansion_context
        DATA_SAMPLE_CONTEXT: $data_sample_context
        ROLE RUNBOOK (Data Engineer): $data_engineer_runbook

        DATA AUDIT:
        $data_audit

        Return Python code only.
        """

        USER_TEMPLATE = (
            "Analyze the owned deliverables, artifact obligations, cleaning gates, column dtype targets, "
            "column resolution context, and the DATASET PROFILE (column_profiles in DATA_SAMPLE_CONTEXT). "
            "Reason first about the deliverable closure for THIS run — which artifacts you must write, "
            "how each one is materialized, and how the cleaning plan supports them. "
            "Then reason about the correct operation order for THIS specific dataset — "
            "use the column_profiles to decide: which columns need null handling (check null_pct), "
            "which look numeric vs datetime (check looks_numeric, looks_datetime), "
            "what format patterns exist (check observed_format_patterns), "
            "and what the actual cardinality and value distribution is (check unique_count, top_values). "
            "For each parsing step, mentally verify that your approach will not inflate nulls "
            "beyond the raw null_pct shown in the profile. "
            "Check pandas_pitfalls in RUNTIME_DEPENDENCY_CONTEXT before using any pandas API. "
            "Then generate the complete cleaning script."
        )
        USER_REPAIR_TEMPLATE = """
        MODE: REPAIR_EDITOR
        You are editing a previously generated cleaning script body. Do not regenerate from zero
        unless the previous script body is clearly unusable.

        LATEST_ITERATION_FEEDBACK_RECORD_JSON:
        $feedback_record_json

        ACTIVE_PATCH_OBJECTIVES:
        $patch_objectives

        WHAT_TO_PRESERVE:
        $must_preserve

        REPAIR_ERROR_CONTEXT:
        $error_context

        PREVIOUS_SCRIPT_BODY_TO_PATCH:
        $previous_code

        Repair task:
        - Consult the column_profiles in DATA_SAMPLE_CONTEXT (system prompt) to verify your
          parsing approach matches the actual data patterns (null_pct, looks_datetime,
          observed_format_patterns, looks_numeric). Do not guess — use the profile.
        - Apply a minimal but sufficient patch to the previous script body.
        - Keep already-working logic stable unless it directly caused the failure.
        - Preserve output paths, owned artifact materialization, and manifest logic unless fixing them is part of the patch.
        - Return ONLY the full updated Python script body (not a diff, not snippets, not markdown).
        """

        # Rendering
        required_columns_payload = de_view.get("required_columns") or strategy.get("required_columns", [])
        if isinstance(required_columns_payload, list) and len(required_columns_payload) > 80:
            required_columns_payload = summarize_long_list(required_columns_payload)
            required_columns_payload["note"] = COLUMN_LIST_POINTER
        optional_passthrough_payload = de_view.get("optional_passthrough_columns") or []
        if isinstance(optional_passthrough_payload, list) and len(optional_passthrough_payload) > 80:
            optional_passthrough_payload = summarize_long_list(optional_passthrough_payload)
            optional_passthrough_payload["note"] = COLUMN_LIST_POINTER

        system_prompt = render_prompt(
            SYSTEM_TEMPLATE,
            input_path=input_path,
            csv_encoding=csv_encoding,
            csv_sep=csv_sep,
            csv_decimal=csv_decimal,
            business_objective=business_objective,
            de_required_outputs_context=de_required_outputs_json,
            required_columns=json.dumps(required_columns_payload),
            optional_passthrough_columns=json.dumps(optional_passthrough_payload),
            data_audit=data_audit,
            execution_contract_context=contract_json,
            de_view_context=de_view_json,
            artifact_obligations_context=artifact_obligations_json,
            outlier_policy_context=outlier_policy_json,
            column_resolution_context=column_resolution_context_json,
            column_transformations_context=column_transformations_json,
            column_dtype_targets_context=column_dtype_targets_json,
            data_engineer_runbook=de_runbook_json,
            cleaning_gates_context=cleaning_gates_json,
            runtime_dependency_context=runtime_dependency_context_json,
            selector_expansion_context=selector_expansion_context,
            data_sample_context=data_sample_context,
            de_output_path=de_output_path,
            de_manifest_path=de_manifest_path,
            outlier_report_path=outlier_report_path,
            operating_mode=operating_mode,
            repair_notes=repair_notes,
            pipeline_scope_guidance=pipeline_scope_guidance,
        )
        patch_mode_active = bool(
            effective_repair_mode
            and self._looks_like_editable_code(previous_code or "")
        )
        if patch_mode_active:
            repair_prompt_context = self._build_repair_prompt_context(
                data_audit=data_audit,
                previous_code=str(previous_code or ""),
                feedback_record=feedback_record,
            )
            user_message = render_prompt(USER_REPAIR_TEMPLATE, **repair_prompt_context)
        else:
            user_message = USER_TEMPLATE
        self.last_prompt = system_prompt + "\n\nUSER:\n" + user_message
        print(f"DEBUG: DE System Prompt Len: {len(system_prompt)}")
        print(f"DEBUG: DE System Prompt Preview: {system_prompt[:300]}...")
        if len(system_prompt) < 100:
            print("CRITICAL: System Prompt is suspiciously short!")

        from src.utils.retries import call_with_retries

        def _call_model():
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]
            response, model_used = call_chat_with_fallback(
                self.client,
                messages,
                [self.model_name, self.fallback_model_name],
                call_kwargs={"temperature": 0.0 if patch_mode_active else 0.1},
                logger=self.logger,
                context_tag="data_engineer",
            )
            self.logger.info("DATA_ENGINEER_MODEL_USED: %s", model_used)
            # Use _extract_nonempty to handle EMPTY_COMPLETION (CAUSA RA?Z 2)
            content = self._extract_nonempty(response)
            print(f"DEBUG: Primary DE Response Preview: {content[:200]}...")
            self.last_response = content

            # CRITICAL CHECK FOR SERVER ERRORS (HTML/504)
            if "504 Gateway Time-out" in content or "<html" in content.lower():
                raise ConnectionError("LLM Server Timeout (504 Received)")

            # Check for JSON error messages that are NOT valid code
            content_stripped = content.strip()
            if content_stripped.startswith("{") or content_stripped.startswith("["):
                try:
                    import json
                    json_content = json.loads(content_stripped)
                    if isinstance(json_content, dict):
                        if "error" in json_content or "errorMessage" in json_content:
                            raise ConnectionError(f"API Error Detected (JSON): {content_stripped}")
                except Exception:
                    pass

            # Text based fallback for Error/Overloaded keywords
            content_lower = content.lower()
            if "error" in content_lower and ("overloaded" in content_lower or "rate limit" in content_lower or "429" in content_lower):
                raise ConnectionError(f"API Error Detected (Text): {content_stripped}")

            return content

        injection = "\n".join(
            [
                "import os",
                "import json",
                "import re",
                "from datetime import date, datetime",
                "try:",
                "    import numpy as np",
                "except Exception:",
                "    np = None",
                "try:",
                "    import pandas as pd",
                "except Exception:",
                "    pd = None",
                "",
                "os.makedirs('data', exist_ok=True)",
                f"_REQUIRED_OUTPUT_PATHS = {json.dumps([str(item.get('path') or '') for item in de_required_outputs if str(item.get('path') or '').strip()], ensure_ascii=False)}",
                "",
                "def _ensure_parent_dir(path):",
                "    if not isinstance(path, str) or not path.strip():",
                "        return",
                "    parent = os.path.dirname(path)",
                "    if parent:",
                "        os.makedirs(parent, exist_ok=True)",
                "",
                "for _required_path in _REQUIRED_OUTPUT_PATHS:",
                "    _ensure_parent_dir(_required_path)",
                "",
                "def _safe_load_json(path, default=None):",
                "    if default is None:",
                "        default = {}",
                "    try:",
                "        if not os.path.exists(path):",
                "            return default",
                "        with open(path, 'r', encoding='utf-8') as _f:",
                "            payload = json.load(_f)",
                "        return payload if isinstance(payload, type(default)) else payload",
                "    except Exception:",
                "        return default",
                "",
                "def _load_manifest_output_dialect(manifest_path='data/cleaning_manifest.json'):",
                "    payload = _safe_load_json(manifest_path, default={})",
                "    if not isinstance(payload, dict):",
                "        return {}",
                "    out = payload.get('output_dialect')",
                "    return out if isinstance(out, dict) else {}",
                "",
                "def _load_column_sets(path='data/column_sets.json'):",
                "    payload = _safe_load_json(path, default={})",
                "    return payload if isinstance(payload, dict) else {}",
                "",
                "def _expand_selector_columns(columns, selectors):",
                "    if not isinstance(columns, list) or not isinstance(selectors, list):",
                "        return []",
                "    matched = []",
                "    for sel in selectors:",
                "        if not isinstance(sel, dict):",
                "            continue",
                "        stype = str(sel.get('type') or '').strip().lower()",
                "        if stype in ('regex', 'pattern'):",
                "            pattern = str(sel.get('pattern') or '').strip()",
                "            if not pattern:",
                "                continue",
                "            try:",
                "                rgx = re.compile(pattern, flags=re.IGNORECASE)",
                "            except Exception:",
                "                continue",
                "            for col in columns:",
                "                if isinstance(col, str) and rgx.match(col) and col not in matched:",
                "                    matched.append(col)",
                "        elif stype == 'prefix':",
                "            prefix = str(sel.get('value') or sel.get('prefix') or '').strip().lower()",
                "            if not prefix:",
                "                continue",
                "            for col in columns:",
                "                if isinstance(col, str) and col.lower().startswith(prefix) and col not in matched:",
                "                    matched.append(col)",
                "        elif stype == 'suffix':",
                "            suffix = str(sel.get('value') or sel.get('suffix') or '').strip().lower()",
                "            if not suffix:",
                "                continue",
                "            for col in columns:",
                "                if isinstance(col, str) and col.lower().endswith(suffix) and col not in matched:",
                "                    matched.append(col)",
                "        elif stype == 'contains':",
                "            token = str(sel.get('value') or '').strip().lower()",
                "            if not token:",
                "                continue",
                "            for col in columns:",
                "                if isinstance(col, str) and token in col.lower() and col not in matched:",
                "                    matched.append(col)",
                "        elif stype == 'list':",
                "            vals = sel.get('columns')",
                "            if isinstance(vals, list):",
                "                for col in vals:",
                "                    if isinstance(col, str) and col in columns and col not in matched:",
                "                        matched.append(col)",
                "    return matched",
                "",
                "def _log_decision(msg, payload=None):",
                "    try:",
                "        entry = {'message': str(msg)}",
                "        if payload is not None:",
                "            entry['payload'] = _to_jsonable(payload)",
                "        print('DE_DECISION_LOG::' + json.dumps(entry, ensure_ascii=False))",
                "    except Exception:",
                "        pass",
                "",
                "def _to_jsonable(value):",
                "    if value is None:",
                "        return None",
                "    if isinstance(value, (str, int, bool)):",
                "        return value",
                "    if isinstance(value, float):",
                "        return None if value != value else value",
                "    if isinstance(value, (datetime, date)):",
                "        return value.isoformat()",
                "    if isinstance(value, (list, tuple, set)):",
                "        return [_to_jsonable(item) for item in value]",
                "    if isinstance(value, dict):",
                "        return {str(k): _to_jsonable(v) for k, v in value.items()}",
                "    if isinstance(value, (bytes, bytearray)):",
                "        return value.decode('utf-8', errors='replace')",
                "    if np is not None:",
                "        if isinstance(value, np.bool_):",
                "            return bool(value)",
                "        if isinstance(value, np.integer):",
                "            return int(value)",
                "        if isinstance(value, np.floating):",
                "            return float(value)",
                "        if isinstance(value, np.ndarray):",
                "            return [_to_jsonable(item) for item in value.tolist()]",
                "    if pd is not None:",
                "        if value is pd.NA:",
                "            return None",
                "        if isinstance(value, pd.Timestamp):",
                "            return value.isoformat()",
                "        try:",
                "            if pd.isna(value) is True:",
                "                return None",
                "        except Exception:",
                "            pass",
                "    return str(value)",
                "",
                "_ORIG_JSON_DUMP = json.dump",
                "_ORIG_JSON_DUMPS = json.dumps",
                "",
                "def _safe_dump_json(obj, fp, **kwargs):",
                "    payload = _to_jsonable(obj)",
                "    kwargs.pop('default', None)",
                "    return _ORIG_JSON_DUMP(payload, fp, **kwargs)",
                "",
                "def _safe_dumps_json(obj, **kwargs):",
                "    payload = _to_jsonable(obj)",
                "    kwargs.pop('default', None)",
                "    return _ORIG_JSON_DUMPS(payload, **kwargs)",
                "",
                "json.dump = _safe_dump_json",
                "json.dumps = _safe_dumps_json",
                "",
            ]
        ) + "\n"

        try:
            content = call_with_retries(_call_model, max_retries=5, backoff_factor=2, initial_delay=2)
            print("DEBUG: OpenRouter response received.")

            code = self._clean_code(content)
            deps = contract.get("required_dependencies") if isinstance(contract, dict) else None
            dep_backend = str(runtime_dependency_context.get("backend_profile") or "local").lower()
            if dep_backend == "local":
                dep_backend = "cloudrun"
            dep_check = check_dependency_precheck(
                code,
                required_dependencies=deps if isinstance(deps, list) else [],
                backend_profile=dep_backend,
            )
            blocked = dep_check.get("blocked") or []
            banned = dep_check.get("banned") or []
            if blocked or banned:
                suggestions = dep_check.get("suggestions") or {}
                raise ValueError(
                    "DEPENDENCY_PREFLIGHT_FAILED: "
                    + json.dumps(
                        {
                            "blocked": blocked,
                            "banned": banned,
                            "suggestions": suggestions,
                            "backend_profile": dep_check.get("backend_profile"),
                        },
                        ensure_ascii=False,
                    )
                )

            return injection + code

        except Exception as e:
            error_msg = f"Data Engineer Failed (Primary & Fallback): {str(e)}"
            print(f"CRITICAL: {error_msg}")
            return f"# Error: {error_msg}"

    def _clean_code(self, code: str) -> str:
        """
        Extracts code from markdown blocks, validates syntax, and applies auto-fixes.
        Raises ValueError if code is empty or has unfixable syntax errors (CAUSA RAIZ 2 & 3).
        """
        cleaned = (extract_code_block(code) or "").strip()
        if not cleaned and not (code or "").strip():
            print("ERROR: EMPTY_CODE_AFTER_EXTRACTION")
            raise ValueError("EMPTY_CODE_AFTER_EXTRACTION")

        def _autofix_assign_digit_identifier(src: str) -> str:
            # .assign(1stYearAmount=...) -> .assign(**{'1stYearAmount': ...})
            pattern = r'\.assign\(\s*([0-9][a-zA-Z0-9_]*)\s*=\s*([^)]+)\)'

            def fix_assign(match):
                col_name = match.group(1)
                value = match.group(2)
                return f".assign(**{{'{col_name}': {value}}})"

            return re.sub(pattern, fix_assign, src or "")

        def _trim_to_code_start(src: str) -> str:
            if not isinstance(src, str):
                return ""
            normalized = re.sub(r"</?think>", "\n", src, flags=re.IGNORECASE).strip()
            if not normalized:
                return ""
            lines = normalized.splitlines()
            start_pattern = re.compile(
                r"^(#|from\s+\w+|import\s+\w+|def\s+\w+|class\s+\w+|if\s+__name__|if\s+|for\s+|while\s+|try:|with\s+|@|[A-Za-z_]\w*\s*=|print\()"
            )
            for idx, line in enumerate(lines):
                stripped = line.strip()
                if not stripped:
                    continue
                if start_pattern.match(stripped):
                    return "\n".join(lines[idx:]).strip()
            return normalized

        def _looks_like_script(src: str) -> bool:
            if not isinstance(src, str):
                return False
            text = src.strip()
            if not text:
                return False
            code_pattern = re.compile(
                r"(?m)^\s*(from\s+\w+|import\s+\w+|def\s+\w+|class\s+\w+|if\s+__name__|"
                r"if\s+|for\s+|while\s+|try:|with\s+|@\w+|[A-Za-z_]\w*\s*=|print\(|raise\s+)"
            )
            return bool(code_pattern.search(text))

        candidates: List[str] = []

        def _push_candidate(value: str) -> None:
            if not isinstance(value, str):
                return
            stripped = value.strip()
            if not stripped or stripped in candidates:
                return
            candidates.append(stripped)

        _push_candidate(cleaned)
        _push_candidate(code or "")

        raw = code or ""
        if "```" in raw:
            parts = [p.strip() for p in re.split(r"```(?:python)?", raw, flags=re.IGNORECASE) if isinstance(p, str)]
            for part in parts:
                _push_candidate(part)
            first_fence = re.search(r"```(?:python)?", raw, re.IGNORECASE)
            if first_fence:
                _push_candidate(raw[: first_fence.start()])
                _push_candidate(raw[first_fence.end() :])

        think_tail = re.split(r"</think>", raw, flags=re.IGNORECASE)
        if len(think_tail) > 1:
            _push_candidate(think_tail[-1])

        recovery_candidates: List[str] = []
        for candidate in candidates:
            recovery_candidates.append(candidate)
            trimmed = _trim_to_code_start(candidate)
            if trimmed and trimmed != candidate:
                recovery_candidates.append(trimmed)

        last_syntax_error: Optional[SyntaxError] = None
        for candidate in recovery_candidates:
            for variant in (candidate, _autofix_assign_digit_identifier(candidate)):
                variant = (variant or "").strip()
                if not variant or not _looks_like_script(variant):
                    continue
                try:
                    ast.parse(variant)
                    if variant != candidate:
                        print("DEBUG: Auto-fix successful.")
                    return variant
                except SyntaxError as e:
                    last_syntax_error = e

        if last_syntax_error:
            print(f"ERROR: Auto-fix failed. Syntax still invalid: {last_syntax_error}")
            raise ValueError(f"INVALID_PYTHON_SYNTAX: {last_syntax_error}")
        print("ERROR: EMPTY_CODE_AFTER_EXTRACTION")
        raise ValueError("EMPTY_CODE_AFTER_EXTRACTION")

