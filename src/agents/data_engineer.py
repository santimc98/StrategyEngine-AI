import os
import re
import ast
import json
import logging
import csv
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from src.utils.static_safety_scan import scan_code_safety
from src.utils.code_extract import extract_code_block
from src.utils.senior_protocol import SENIOR_ENGINEERING_PROTOCOL
from src.utils.contract_accessors import get_cleaning_gates
from src.utils.cleaning_contract_semantics import expand_required_feature_selectors
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
                "python": "3.11",
                "pandas": pandas_spec,
            },
            "guidance": [
                "Import only allowlisted roots.",
                "Use stable public APIs compatible with version_hints.",
                "Avoid deprecated kwargs/behaviors when equivalent safe idioms exist.",
                "Keep script portable across local and cloudrun runner modes.",
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

        for key in ("scope", "required_outputs", "column_roles", "outlier_policy", "column_dtype_targets"):
            value = contract.get(key)
            if value not in (None, "", [], {}):
                focus[key] = value

        artifact_requirements = contract.get("artifact_requirements")
        if isinstance(artifact_requirements, dict):
            clean_dataset = artifact_requirements.get("clean_dataset")
            if isinstance(clean_dataset, dict) and clean_dataset:
                focus["artifact_requirements"] = {"clean_dataset": clean_dataset}

        if de_view:
            view_focus: Dict[str, Any] = {}
            for key in (
                "required_columns",
                "required_feature_selectors",
                "optional_passthrough_columns",
                "output_path",
                "output_manifest_path",
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

            preview = pd.read_csv(
                path,
                nrows=max_rows,
                dtype=str,
                sep=csv_sep or ",",
                decimal=csv_decimal or ".",
                encoding=csv_encoding or "utf-8",
                low_memory=False,
            )
            preview_cols = [str(col) for col in preview.columns.tolist()]
            truncated = len(preview_cols) > max_cols
            shown_cols = preview_cols[:max_cols]
            payload = {
                "status": "available",
                "path": path,
                "shape_preview": {"rows": int(preview.shape[0]), "cols": int(preview.shape[1])},
                "preview_columns": shown_cols,
                "preview_columns_truncated": truncated,
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
    ) -> str:
        """
        Generates a Python script to clean and standardize the dataset.
        """
        from src.utils.prompting import render_prompt

        contract = execution_contract or contract_min or {}
        from src.utils.context_pack import compress_long_lists, summarize_long_list, COLUMN_LIST_POINTER

        # Build scope-aware guidance
        _pipeline_scope = ""
        if isinstance(contract, dict):
            _pipeline_scope = str(contract.get("scope", "")).strip().lower()
        if _pipeline_scope == "cleaning_only":
            pipeline_scope_guidance = (
                "PIPELINE SCOPE: CLEANING_ONLY — Your output is the FINAL deliverable. "
                "There is NO downstream ML pipeline. Prioritize:\n"
                "  - Maximum data quality and completeness\n"
                "  - Thorough validation of all cleaning gates\n"
                "  - Detailed manifest documenting every transformation\n"
                "  - Production-ready output suitable for direct business use\n"
                "  - Comprehensive null handling and type standardization"
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
                "PIPELINE SCOPE: FULL_PIPELINE — Your output feeds into an ML Engineer. "
                "Balance thoroughness with ML compatibility:\n"
                "  - Clean data to meet contract gates\n"
                "  - Preserve statistical properties needed for modeling\n"
                "  - Ensure required columns are present and correctly typed\n"
                "  - The ML Engineer will use your output for feature engineering and training"
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
        artifact_requirements = contract.get("artifact_requirements")
        clean_dataset_cfg = {}
        if isinstance(artifact_requirements, dict):
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

        === SENIOR ENGINEERING PROTOCOL ===
        $senior_engineering_protocol

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
          and writes the cleaned output + manifest.
        - Follow the execution contract and cleaning gates as source of truth.
        - Be universal and adaptive to any CSV and business objective.

        ===================================================================
        SOURCE OF TRUTH AND PRECEDENCE
        ===================================================================
        1) CLEANING_GATES_CONTEXT + required_columns + required_feature_selectors (authoritative)
        2) COLUMN_DTYPE_TARGETS_CONTEXT + COLUMN_TRANSFORMATIONS_CONTEXT (authoritative)
        3) ROLE RUNBOOK (advisory — informs reasoning, does not override gates)

        ===================================================================
        ENGINEERING REASONING WORKFLOW (MANDATORY)
        ===================================================================
        Before writing any code, reason through the cleaning plan by analyzing the
        contract inputs. Write your reasoning as comment blocks at the top of the
        script. This is not optional — it is how you prevent sequencing bugs.

        # EXECUTION PLAN (reason about these in order):
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
        # 3. TYPE CONVERSION (AFTER null handling):
        #    Apply COLUMN_DTYPE_TARGETS. Use pd.to_numeric(errors='coerce') for
        #    numeric columns. For string columns, convert only after nulls are resolved.
        #    Use pandas nullable Int64/Float64 for nullable integer/float columns.
        #
        # 4. CONSTRAINT VALIDATION:
        #    - HARD gates: check and raise ValueError("CLEANING_GATE_FAILED: ...") if violated.
        #    - SOFT gates: check and warn if thresholds exceeded, but do not block.
        #    - Non-nullable columns: verify no unexpected nulls after conversion.
        #
        # 5. OUTPUT: Write cleaned CSV and manifest with actual operations performed.

        Your Decision Log, Assumptions, and Risks blocks should reflect the specific
        reasoning you did for THIS dataset — not generic boilerplate.

        ===================================================================
        HARD CONSTRAINTS
        ===================================================================
        - Output valid Python code only. No markdown, no code fences, no prose.
        - Scope: cleaning only (no modeling, scoring, optimization, or analytics).
        - Read input with pd.read_csv(..., dtype=str, low_memory=False, sep, decimal, encoding from inputs).
        - Write cleaned CSV to $de_output_path.
        - Write manifest to $de_manifest_path.
        - If outlier policy is enabled, write report to $outlier_report_path.
        - Do not fabricate columns. Do not overwrite the input file.
        - Missing required columns → fail fast with ValueError.
        - Use only dependencies from RUNTIME_DEPENDENCY_CONTEXT.

        SANDBOX SECURITY — BLOCKED IMPORTS:
        sys, subprocess, socket, requests, httpx, urllib, ftplib, paramiko,
        selenium, playwright, openai, google.generativeai, builtins,
        eval(), exec(), compile(), __import__()
        ALLOWED: pandas, numpy, sklearn, scipy, xgboost, lightgbm, matplotlib,
        seaborn, json, os.path, os.makedirs, csv, math, statistics, collections,
        itertools, functools, typing, warnings, re, datetime, pathlib.Path

        ===================================================================
        DATA INTEGRITY PRINCIPLES
        ===================================================================
        Think like a senior engineer reviewing your own cleaning code before merge:

        - The operation order matters: null handling → type conversion → validation.
          Getting this wrong silently corrupts data. Reason about dependencies.
        - Do not impute outcome/target columns unless the contract explicitly requests it.
          Preserve missingness for partially labeled targets (e.g., test set rows).
        - Preserve split/partition columns exactly as-is.
        - For wide datasets, resolve feature selectors against actual header; prefer
          vectorized transforms over per-column procedural loops.
        - Create parent directories before writing. Preserve deterministic column order.
        - The manifest must reflect ACTUAL operations performed, not planned ones.
          If an imputation ran, log it. If it was skipped (no nulls found), say so.

        REPAIR RULES (when OPERATING_MODE is REPAIR):
        - Read RUNTIME_ERROR_CONTEXT/PREFLIGHT_ERROR_CONTEXT and diagnose root cause first.
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
        AUTHORITATIVE CONTEXT
        ===================================================================
        Input: '$input_path'
        Encoding: '$csv_encoding' | Sep: '$csv_sep' | Decimal: '$csv_decimal'
        DE Cleaning Objective: "$business_objective"

        Required Columns (DE View): $required_columns
        Optional Passthrough Columns: $optional_passthrough_columns

        DE_VIEW_CONTEXT: $de_view_context
        EXECUTION_CONTRACT_CONTEXT: $execution_contract_context
        CLEANING_GATES_CONTEXT: $cleaning_gates_context
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
            "Analyze the cleaning gates, column dtype targets, and data sample. "
            "Reason about the correct operation order for THIS specific dataset — "
            "which columns need null handling before type conversion, which gates "
            "impose constraints, what the runbook advises. "
            "Then generate the complete cleaning script."
        )

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
            required_columns=json.dumps(required_columns_payload),
            optional_passthrough_columns=json.dumps(optional_passthrough_payload),
            data_audit=data_audit,
            execution_contract_context=contract_json,
            de_view_context=de_view_json,
            outlier_policy_context=outlier_policy_json,
            column_transformations_context=column_transformations_json,
            column_dtype_targets_context=column_dtype_targets_json,
            data_engineer_runbook=de_runbook_json,
            cleaning_gates_context=cleaning_gates_json,
            runtime_dependency_context=runtime_dependency_context_json,
            selector_expansion_context=selector_expansion_context,
            data_sample_context=data_sample_context,
            senior_engineering_protocol=SENIOR_ENGINEERING_PROTOCOL,
            de_output_path=de_output_path,
            de_manifest_path=de_manifest_path,
            outlier_report_path=outlier_report_path,
            operating_mode=operating_mode,
            repair_notes=repair_notes,
            pipeline_scope_guidance=pipeline_scope_guidance,
        )
        self.last_prompt = system_prompt + "\n\nUSER:\n" + USER_TEMPLATE
        print(f"DEBUG: DE System Prompt Len: {len(system_prompt)}")
        print(f"DEBUG: DE System Prompt Preview: {system_prompt[:300]}...")
        if len(system_prompt) < 100:
            print("CRITICAL: System Prompt is suspiciously short!")

        from src.utils.retries import call_with_retries

        def _call_model():
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": USER_TEMPLATE},
            ]
            response, model_used = call_chat_with_fallback(
                self.client,
                messages,
                [self.model_name, self.fallback_model_name],
                call_kwargs={"temperature": 0.1},
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
                f"_REQUIRED_OUTPUT_PATHS = {json.dumps([p for p in [de_output_path, de_manifest_path, outlier_report_path] if p], ensure_ascii=False)}",
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

