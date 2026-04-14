import csv
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

from src.utils.code_extract import extract_code_block
from src.utils.cleaning_contract_semantics import resolve_required_columns_for_cleaning
from src.utils.reviewer_llm import init_reviewer_llm

load_dotenv()

_CLEANING_FALLBACK_WARNING = (
    "CONTRACT_BROKEN_FALLBACK: cleaning_gates missing in cleaning_view; "
    "contract generation must include cleaning_gates (V4.1)."
)
_CONTRACT_MISSING_CLEANING_GATES = "CONTRACT_MISSING_CLEANING_GATES"
_LLM_DISABLED_WARNING = "LLM_DISABLED_NO_API_KEY"
_LLM_PARSE_WARNING = "LLM_PARSE_FAILED"
_LLM_CALL_WARNING = "LLM_CALL_FAILED"
_UNEVALUATED_HARD_GATES_WARNING = "UNEVALUATED_HARD_GATES"
_LLM_FAIL_CLOSED_REASON = "LLM_UNAVAILABLE_WITH_UNEVALUATED_HARD_GATES"


# ── Hardcoded regex/gates removed (seniority refactoring) ───────────────
# ID and percent column detection now relies on column_roles from the contract.
# Fallback cleaning gates removed: if the contract is missing cleaning_gates,
# the reviewer rejects with CONTRACT_MISSING_CLEANING_GATES to force contract
# regeneration instead of inventing its own gates.
# ────────────────────────────────────────────────────────────────────────

# Generic ID regex: matches common identifier column patterns without domain-specific terms
_GENERIC_ID_REGEX = r"(?i)(^id$|(?:_id$)|(?:^id_)|(?:^|[_\W])(?:id|entity|code|key)(?:[_\W]|$))"


class CleaningReviewerAgent:
    """
    LLM-driven cleaning reviewer with deterministic evidence checks.
    Falls back to deterministic mode if OPENROUTER_API_KEY is unavailable.
    """

    def __init__(self, api_key: Any = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.provider, self.client, self.model_name, self.model_warning = init_reviewer_llm(api_key)
        if self.model_warning:
            print(f"WARNING: {self.model_warning}")
        self.last_prompt = None
        self.last_response = None

    def review_cleaning(self, *args, **kwargs) -> Dict[str, Any]:
        try:
            request = _parse_review_inputs(args, kwargs)
            if request.get("failure_context"):
                result, prompt, response = _review_cleaning_failure(
                    cleaning_view=request["cleaning_view"],
                    failure_context=request.get("failure_context"),
                    client=self.client,
                    model_name=self.model_name,
                    provider=self.provider,
                )
            else:
                result, prompt, response = _review_cleaning_impl(
                    cleaning_view=request["cleaning_view"],
                    cleaned_csv_path=request["cleaned_csv_path"],
                    cleaning_manifest_path=request["cleaning_manifest_path"],
                    raw_csv_path=request.get("raw_csv_path"),
                    artifact_obligations=request.get("artifact_obligations"),
                    client=self.client,
                    model_name=self.model_name,
                    provider=self.provider,
                )
        except Exception as exc:
            result = _exception_result(exc)
            prompt = "cleaning_reviewer_exception"
            response = str(exc)

        self.last_prompt = prompt
        self.last_response = response
        return result


def _parse_review_inputs(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    if len(args) == 1 and isinstance(args[0], dict) and not kwargs.get("cleaned_csv_path"):
        return _parse_legacy_context(args[0])

    cleaning_view = args[0] if args else kwargs.get("cleaning_view") or {}
    if not isinstance(cleaning_view, dict):
        cleaning_view = {}
    input_dialect = kwargs.get("input_dialect")
    output_dialect = kwargs.get("output_dialect")
    if isinstance(input_dialect, dict) and not cleaning_view.get("input_dialect"):
        cleaning_view["input_dialect"] = input_dialect
    if isinstance(output_dialect, dict) and not cleaning_view.get("output_dialect"):
        cleaning_view["output_dialect"] = output_dialect
    if isinstance(kwargs.get("dialect"), dict) and not cleaning_view.get("dialect"):
        cleaning_view["dialect"] = kwargs.get("dialect")
    cleaned_csv_path = (
        (args[1] if len(args) > 1 else None)
        or kwargs.get("cleaned_csv_path")
        or kwargs.get("cleaned_path")
        or "data/cleaned_data.csv"
    )
    cleaning_manifest_path = (
        (args[2] if len(args) > 2 else None)
        or kwargs.get("cleaning_manifest_path")
        or kwargs.get("manifest_path")
        or "data/cleaning_manifest.json"
    )
    raw_csv_path = (
        (args[3] if len(args) > 3 else None)
        or kwargs.get("raw_csv_path")
        or kwargs.get("raw_path")
    )
    failure_context = kwargs.get("failure_context")
    if failure_context is None and isinstance(cleaning_view, dict):
        maybe_failure = cleaning_view.get("failure_context")
        if isinstance(maybe_failure, dict):
            failure_context = maybe_failure
    artifact_obligations = kwargs.get("artifact_obligations")
    return {
        "cleaning_view": cleaning_view,
        "cleaned_csv_path": str(cleaned_csv_path),
        "cleaning_manifest_path": str(cleaning_manifest_path),
        "raw_csv_path": str(raw_csv_path) if raw_csv_path else None,
        "failure_context": failure_context if isinstance(failure_context, dict) else None,
        "artifact_obligations": artifact_obligations if isinstance(artifact_obligations, dict) else None,
    }


def _parse_legacy_context(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse context to extract cleaning_view and paths.

    V4.1: Removed legacy required_columns fallback. Required columns should come from
    cleaning_view (populated from artifact_requirements.clean_dataset.required_columns
    or canonical_columns), not from loose legacy context keys.
    """
    context = context if isinstance(context, dict) else {}
    cleaning_view = context.get("cleaning_view") if isinstance(context.get("cleaning_view"), dict) else {}

    # V4.1: Only allow cleaning_gates from cleaning_view or explicit context key
    if not cleaning_view.get("cleaning_gates") and isinstance(context.get("cleaning_gates"), list):
        cleaning_view["cleaning_gates"] = context.get("cleaning_gates")

    # V4.1: REMOVED legacy required_columns fallback
    # Required columns must come from cleaning_view (contract-driven), not loose context

    # Dialect handling (still allowed for compatibility)
    if not cleaning_view.get("input_dialect"):
        input_dialect = context.get("input_dialect")
        if isinstance(input_dialect, dict):
            cleaning_view["input_dialect"] = input_dialect
    if not cleaning_view.get("output_dialect"):
        output_dialect = context.get("output_dialect")
        if isinstance(output_dialect, dict):
            cleaning_view["output_dialect"] = output_dialect
    if not cleaning_view.get("dialect"):
        legacy_dialect = context.get("dialect")
        if isinstance(legacy_dialect, dict):
            cleaning_view["dialect"] = legacy_dialect
            if not cleaning_view.get("input_dialect"):
                cleaning_view["input_dialect"] = legacy_dialect
    if not cleaning_view.get("column_roles") and isinstance(context.get("column_roles"), dict):
        cleaning_view["column_roles"] = context.get("column_roles")
    if not cleaning_view.get("outlier_policy") and isinstance(context.get("outlier_policy"), dict):
        cleaning_view["outlier_policy"] = context.get("outlier_policy")
    if not cleaning_view.get("column_dtype_targets") and isinstance(context.get("column_dtype_targets"), dict):
        cleaning_view["column_dtype_targets"] = context.get("column_dtype_targets")
    if not cleaning_view.get("dataset_profile") and isinstance(context.get("dataset_profile"), dict):
        cleaning_view["dataset_profile"] = context.get("dataset_profile")
    if not cleaning_view.get("cleaning_code") and isinstance(context.get("cleaning_code"), str):
        cleaning_view["cleaning_code"] = context.get("cleaning_code")

    cleaned_csv_path = (
        context.get("cleaned_csv_path")
        or context.get("cleaned_path")
        or context.get("cleaned_csv")
        or "data/cleaned_data.csv"
    )
    cleaning_manifest_path = (
        context.get("cleaning_manifest_path")
        or context.get("manifest_path")
        or "data/cleaning_manifest.json"
    )
    raw_csv_path = context.get("raw_csv_path") or context.get("raw_path")
    failure_context = context.get("failure_context") if isinstance(context, dict) else None
    artifact_obligations = context.get("artifact_obligations") if isinstance(context.get("artifact_obligations"), dict) else None
    return {
        "cleaning_view": cleaning_view,
        "cleaned_csv_path": str(cleaned_csv_path),
        "cleaning_manifest_path": str(cleaning_manifest_path),
        "raw_csv_path": str(raw_csv_path) if raw_csv_path else None,
        "failure_context": failure_context if isinstance(failure_context, dict) else None,
        "artifact_obligations": artifact_obligations,
    }


def _review_cleaning_failure(
    cleaning_view: Dict[str, Any],
    failure_context: Optional[Dict[str, Any]],
    client: Any,
    model_name: str,
    provider: str,
) -> Tuple[Dict[str, Any], str, str]:
    view = cleaning_view if isinstance(cleaning_view, dict) else {}
    failure_context = failure_context if isinstance(failure_context, dict) else {}
    error_details = str(failure_context.get("error_details") or "")
    code = str(failure_context.get("code") or "")
    stdout = str(failure_context.get("stdout") or "")
    stderr = str(failure_context.get("stderr") or "")

    root_line = ""
    if error_details:
        lines = [line for line in error_details.strip().splitlines() if line.strip()]
        if lines:
            root_line = lines[-1][:300]

    required_fixes: List[str] = []
    failed_checks: List[str] = ["CLEANING_RUNTIME_ERROR"]
    warnings: List[str] = []
    hard_failures: List[str] = ["CLEANING_RUNTIME_ERROR"]

    lower_error = error_details.lower()
    if "numpy.ndarray" in error_details and ".str" in error_details:
        required_fixes.append(
            "Avoid assigning np.where results to a Series used with .str; keep a pandas Series "
            "(use Series.where/mask or wrap back into a Series with the original index)."
        )
    if "keyerror" in lower_error:
        required_fixes.append(
            "Check column name normalization and ensure referenced columns exist before access; "
            "use a normalized header map for matching."
        )
    if "filenotfounderror" in lower_error:
        required_fixes.append("Read the input from the provided input_path; avoid hardcoded paths.")
    if "typeerror" in lower_error and "not supported between" in lower_error:
        required_fixes.append(
            "Normalize dtypes before comparisons; convert mixed-type columns with pd.to_numeric(errors='coerce') "
            "or consistent string casting before conditional logic."
        )
    if "valueerror" in lower_error and "could not convert" in lower_error:
        required_fixes.append(
            "Apply robust numeric conversion (pd.to_numeric(errors='coerce')) and handle nulls before arithmetic."
        )
    if "indexerror" in lower_error:
        required_fixes.append(
            "Guard positional indexing with explicit length checks; avoid assuming non-empty arrays/lists."
        )
    if "attributeerror" in lower_error and "has no attribute" in lower_error:
        required_fixes.append(
            "Verify object type before method access; ensure Series/DataFrame operations match actual object type."
        )
    if "memoryerror" in lower_error or "killed" in lower_error or "oom" in lower_error:
        required_fixes.append(
            "Reduce memory pressure with chunked processing, narrower dtypes, and avoiding full-data materialization."
        )
    if "importerror" in lower_error or "modulenotfounderror" in lower_error:
        required_fixes.append(
            "Use dependencies available in runtime allowlist and avoid optional imports not guaranteed by the executor."
        )
    if not required_fixes:
        required_fixes.append("Fix the runtime error and rerun the cleaning script without changing I/O paths.")

    feedback = "Cleaning failed during execution."
    if root_line:
        feedback += f" Root cause: {root_line}"

    result = {
        "status": "REJECTED",
        "feedback": feedback,
        "failed_checks": failed_checks,
        "required_fixes": required_fixes,
        "warnings": warnings,
        "cleaning_gates_evaluated": [],
        "hard_failures": hard_failures,
        "soft_failures": [],
        "gate_results": [],
        "contract_source_used": "runtime_error",
        "evidence": {
            "error_details": error_details[-2000:],
            "stdout_tail": stdout[-1000:],
            "stderr_tail": stderr[-1000:],
            "code_tail": code[-1000:],
        },
    }

    prompt = "cleaning_reviewer_runtime_failure"
    response = json.dumps(result, ensure_ascii=False)
    return result, prompt, response


def _review_cleaning_impl(
    cleaning_view: Dict[str, Any],
    cleaned_csv_path: str,
    cleaning_manifest_path: str,
    raw_csv_path: Optional[str],
    artifact_obligations: Optional[Dict[str, Any]],
    client: Any,
    model_name: str,
    provider: str,
) -> Tuple[Dict[str, Any], str, str]:
    view = cleaning_view if isinstance(cleaning_view, dict) else {}
    context_pack = view.get("context_pack") if isinstance(view, dict) else None
    gates, contract_source_used, warnings = _merge_cleaning_gates(view)
    gate_names = [gate["name"] for gate in gates]

    manifest = _load_json(cleaning_manifest_path)
    required_columns = _resolve_required_columns_for_review(
        view,
        manifest=manifest,
        artifact_obligations=artifact_obligations,
    )
    column_roles = _coerce_roles(view.get("column_roles"))
    dialect_in_context = _resolve_dialect(view.get("dialect"))
    dialect_raw = _resolve_dialect(view.get("input_dialect") or view.get("dialect") or dialect_in_context)
    dialect_cleaned = _resolve_dialect(
        view.get("output_dialect")
        or _extract_output_dialect_from_manifest(manifest)
        or view.get("dialect")
    )
    dialect_warnings: List[str] = []

    cleaned_header = _read_csv_header(cleaned_csv_path, dialect_cleaned)
    if cleaned_header and len(cleaned_header) == 1:
        header_text = str(cleaned_header[0])
        if any(token in header_text for token in [",", ";", "\t", "|"]):
            inferred_sep = _infer_delimiter_from_file(cleaned_csv_path)
            if inferred_sep and inferred_sep != dialect_cleaned.get("sep"):
                encoding = dialect_cleaned.get("encoding") or _infer_encoding(cleaned_csv_path)
                sample_text = _read_text_sample(cleaned_csv_path, encoding, 50000)
                inferred_decimal = _infer_decimal_from_sample(sample_text, inferred_sep)
                dialect_cleaned = _resolve_dialect(
                    {"sep": inferred_sep, "decimal": inferred_decimal, "encoding": encoding}
                )
                dialect_warnings.append(
                    "DIALECT_AUTO_INFERRED_FOR_CLEANED: input dialect mismatched; "
                    f"inferred sep={inferred_sep}, decimal={inferred_decimal}"
                )
                cleaned_header = _read_csv_header(cleaned_csv_path, dialect_cleaned)

    sample_str = _read_csv_sample(cleaned_csv_path, dialect_cleaned, cleaned_header, dtype=str, nrows=400)
    sample_infer = _read_csv_sample(cleaned_csv_path, dialect_cleaned, cleaned_header, dtype=None, nrows=400)
    raw_sample = None
    if raw_csv_path:
        raw_sample = _read_csv_sample(raw_csv_path, dialect_raw, None, dtype=str, nrows=200)

    # Load dataset_profile for LLM contextual reasoning (no_semantic_rescale, etc.)
    dataset_profile = view.get("dataset_profile")
    if not isinstance(dataset_profile, dict):
        dataset_profile = _load_json("data/dataset_profile.json")

    # Extract cleaning code from view for LLM intent verification
    cleaning_code = view.get("cleaning_code")
    column_resolution_context = view.get("column_resolution_context")
    if not isinstance(column_resolution_context, dict):
        column_resolution_context = {}
    if not isinstance(artifact_obligations, dict):
        artifact_obligations = {}
    outlier_policy = view.get("outlier_policy") if isinstance(view.get("outlier_policy"), dict) else {}
    outlier_report_path = str(
        view.get("outlier_report_path")
        or outlier_policy.get("report_path")
        or "data/outlier_treatment_report.json"
    )
    outlier_report = _load_json(outlier_report_path) if outlier_report_path else {}

    facts = _build_facts(
        cleaned_header=cleaned_header,
        required_columns=required_columns,
        manifest=manifest,
        sample_str=sample_str,
        sample_infer=sample_infer,
        raw_sample=raw_sample,
        gates=gates,
        column_roles=column_roles,
        dataset_profile=dataset_profile,
        outlier_policy=outlier_policy,
        outlier_report=outlier_report,
        outlier_report_path=outlier_report_path,
        cleaned_csv_path=cleaned_csv_path,
    )

    deterministic = _evaluate_gates_deterministic(
        gates=gates,
        required_columns=required_columns,
        cleaned_header=cleaned_header,
        cleaned_csv_path=cleaned_csv_path,
        sample_str=sample_str,
        sample_infer=sample_infer,
        manifest=manifest,
        raw_sample=raw_sample,
        column_roles=column_roles,
        model_features=_list_str(view.get("model_features")),
        allowed_feature_sets=view.get("allowed_feature_sets") or {},
        dataset_profile=dataset_profile,
        cleaning_code=cleaning_code,
        outlier_policy=outlier_policy,
        outlier_report=outlier_report,
        outlier_report_path=outlier_report_path,
        column_dtype_targets=view.get("column_dtype_targets") if isinstance(view.get("column_dtype_targets"), dict) else {},
    )
    warnings.extend(dialect_warnings)
    deterministic_result = _assemble_result(
        deterministic,
        gate_names,
        warnings,
        contract_source_used,
    )

    if not client or provider == "none":
        deterministic_result["warnings"].append(_LLM_DISABLED_WARNING)
        deterministic_result = _enforce_fail_closed_when_llm_unavailable(
            deterministic_result,
            reason=_LLM_DISABLED_WARNING,
        )
        # V4.1: Enforce contract-strict rejection before returning
        final_result = _enforce_contract_strict_rejection(normalize_cleaning_reviewer_result(deterministic_result))
        return final_result, "LLM_DISABLED", "deterministic"

    cleaning_quality_summary = _build_cleaning_quality_summary(
        cleaned_csv_path=cleaned_csv_path,
        raw_csv_path=raw_csv_path,
        dialect_cleaned=dialect_cleaned,
        dialect_raw=dialect_raw,
    )
    prompt, payload = _build_llm_prompt(
        gates=gates,
        required_columns=required_columns,
        dialect=dialect_cleaned,
        column_roles=column_roles,
        facts=facts,
        deterministic_gate_results=deterministic["gate_results"],
        contract_source_used=contract_source_used,
        context_pack=context_pack,
        cleaning_code=cleaning_code,
        dataset_profile=dataset_profile,
        column_resolution_context=column_resolution_context,
        artifact_obligations=artifact_obligations,
        cleaning_quality_summary=cleaning_quality_summary,
    )
    print(f"DEBUG: Cleaning Reviewer calling OpenRouter ({model_name})...")
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": (
                    "Analyze this evidence payload. Treat cleaning_quality_summary (if present) as supplementary "
                    "evidence for null inflation or datetime parsing issues, but keep header, manifest, gate params, "
                    "and deterministic results as the primary anchors for your judgment. "
                    "Then evaluate each cleaning gate and return your JSON verdict.\n\n"
                    + json.dumps(payload, ensure_ascii=True)
                )},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        content = response.choices[0].message.content
    except Exception as exc:
        deterministic_result["warnings"].append(f"{_LLM_CALL_WARNING}: {exc}")
        deterministic_result = _enforce_fail_closed_when_llm_unavailable(
            deterministic_result,
            reason=_LLM_CALL_WARNING,
        )
        # V4.1: Enforce contract-strict rejection before returning
        final_result = _enforce_contract_strict_rejection(normalize_cleaning_reviewer_result(deterministic_result))
        return final_result, prompt, str(exc)

    parsed = _parse_llm_json(content)
    if not parsed:
        deterministic_result["warnings"].append(_LLM_PARSE_WARNING)
        deterministic_result = _enforce_fail_closed_when_llm_unavailable(
            deterministic_result,
            reason=_LLM_PARSE_WARNING,
        )
        # V4.1: Enforce contract-strict rejection before returning
        final_result = _enforce_contract_strict_rejection(normalize_cleaning_reviewer_result(deterministic_result))
        return final_result, prompt, content

    merged = _merge_llm_with_deterministic(
        llm_result=parsed,
        deterministic=deterministic,
        gate_names=gate_names,
        contract_source_used=contract_source_used,
        warnings=warnings,
    )
    # V4.1: Enforce contract-strict rejection before returning (LLM cannot override)
    final_result = _enforce_contract_strict_rejection(normalize_cleaning_reviewer_result(merged))
    return final_result, prompt, content


def _exception_result(exc: Exception) -> Dict[str, Any]:
    return {
        "status": "REJECTED",
        "feedback": f"Cleaning reviewer exception: {exc}",
        "failed_checks": ["CLEANING_REVIEWER_EXCEPTION"],
        "required_fixes": ["Investigate cleaning reviewer failure."],
        "warnings": [str(exc)],
        "cleaning_gates_evaluated": [],
        "hard_failures": ["CLEANING_REVIEWER_EXCEPTION"],
        "soft_failures": [],
        "gate_results": [],
        "contract_source_used": "error",
    }


def _enforce_contract_strict_rejection(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    V4.1 Contract-Strict Mode: If cleaning_gates were missing from cleaning_view
    (contract_source_used == "fallback"), force REJECTED regardless of gate outcomes.

    This ensures the pipeline fails fast and triggers contract regeneration.
    """
    contract_source = result.get("contract_source_used", "")
    if contract_source != "fallback":
        return result

    # Force REJECTED - contract is broken
    result["status"] = "REJECTED"

    # Add hard failure if not already present
    hard_failures = result.get("hard_failures", [])
    if not isinstance(hard_failures, list):
        hard_failures = []
    if _CONTRACT_MISSING_CLEANING_GATES not in hard_failures:
        hard_failures.append(_CONTRACT_MISSING_CLEANING_GATES)
    result["hard_failures"] = hard_failures

    # Add to failed_checks
    failed_checks = result.get("failed_checks", [])
    if not isinstance(failed_checks, list):
        failed_checks = []
    if _CONTRACT_MISSING_CLEANING_GATES not in failed_checks:
        failed_checks.append(_CONTRACT_MISSING_CLEANING_GATES)
    result["failed_checks"] = failed_checks

    # Add required fix
    required_fixes = result.get("required_fixes", [])
    if not isinstance(required_fixes, list):
        required_fixes = []
    contract_fix = "Regenerate Execution Contract to include cleaning_gates in cleaning_view (V4.1)."
    if contract_fix not in required_fixes:
        required_fixes.insert(0, contract_fix)
    result["required_fixes"] = required_fixes

    # Update feedback to explain the rejection
    existing_feedback = result.get("feedback", "")
    contract_feedback = (
        "CONTRACT INCOMPLETE: cleaning_gates missing from cleaning_view. "
        "Cannot validate cleaning without contract-defined gates. "
        "Regenerate execution contract with cleaning_gates."
    )
    if contract_feedback not in existing_feedback:
        result["feedback"] = contract_feedback + (" " + existing_feedback if existing_feedback else "")

    return result


def _merge_cleaning_gates(view: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str, List[str]]:
    contract_gates = _normalize_cleaning_gates(view.get("cleaning_gates"))
    warnings: List[str] = []
    if contract_gates:
        merged = _dedupe_gates(contract_gates)
        source = "cleaning_view"
    else:
        # No fallback gates: reject with CONTRACT_MISSING_CLEANING_GATES
        merged = []
        source = "fallback"
        warnings.append(_CLEANING_FALLBACK_WARNING)
        warnings.append(_CONTRACT_MISSING_CLEANING_GATES)

    outlier_policy = view.get("outlier_policy") if isinstance(view.get("outlier_policy"), dict) else {}
    if _outlier_policy_enabled(outlier_policy):
        existing = {_normalize_gate_name(g.get("name")) for g in merged if isinstance(g, dict)}
        if "outlier_policy_applied" not in existing:
            strict = outlier_policy.get("strict")
            if isinstance(strict, str):
                strict = strict.strip().lower() in {"1", "true", "yes", "on", "required"}
            severity = "HARD" if strict is None or bool(strict) else "SOFT"
            gate_params = {
                "report_path": str(
                    outlier_policy.get("report_path")
                    or view.get("outlier_report_path")
                    or "data/outlier_treatment_report.json"
                ),
                "strict": bool(True if strict is None else strict),
            }
            merged.append(
                {
                    "name": "outlier_policy_applied",
                    "severity": severity,
                    "params": gate_params,
                }
            )
    return merged, source, warnings


def _outlier_policy_enabled(policy: Dict[str, Any]) -> bool:
    if not isinstance(policy, dict) or not policy:
        return False
    enabled = policy.get("enabled")
    if isinstance(enabled, str):
        enabled = enabled.strip().lower() in {"1", "true", "yes", "on", "enabled"}
    if enabled is not None:
        return bool(enabled)
    return bool(policy.get("target_columns") or policy.get("methods") or policy.get("treatment"))


def _normalize_cleaning_gates(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    normalized: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for gate in raw:
        if isinstance(gate, dict):
            name = _normalize_gate_name(gate.get("name") or gate.get("id") or gate.get("gate"))
            if not name:
                continue
            severity = _normalize_severity(gate.get("severity"), gate.get("required"))
            params = gate.get("params")
            if not isinstance(params, dict):
                params = {}
            if name in seen:
                continue
            seen.add(name)
            normalized.append({"name": name, "severity": severity, "params": params})
        elif isinstance(gate, str):
            name = _normalize_gate_name(gate)
            if not name or name in seen:
                continue
            seen.add(name)
            normalized.append({"name": name, "severity": "HARD", "params": {}})
    return normalized


def _dedupe_gates(gates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    deduped: List[Dict[str, Any]] = []
    for gate in gates:
        key = _normalize_gate_name(gate.get("name", ""))
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(gate)
    return deduped


def _normalize_severity(severity: Any, required: Any = None) -> str:
    if severity is None and required is not None:
        severity = "HARD" if bool(required) else "SOFT"
    sev = str(severity).strip().upper() if severity else "HARD"
    return sev if sev in {"HARD", "SOFT"} else "HARD"


def normalize_gate_name(name: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", (name or "").strip().lower()).strip("_")
    aliases = {
        "boolean_columns_normalized": "boolean_normalization",
    }
    return aliases.get(key, key)


def _normalize_gate_name(name: Any) -> str:
    if name is None:
        return ""
    key = normalize_gate_name(str(name))
    aliases = {
        "boolean_columns_normalized": "boolean_normalization",
    }
    key = aliases.get(key, key)
    return key or ""


def _resolve_dialect(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {"sep": ",", "decimal": ".", "encoding": "utf-8"}
    return {
        "sep": raw.get("sep", ","),
        "decimal": raw.get("decimal", "."),
        "encoding": raw.get("encoding", "utf-8"),
    }


def _extract_output_dialect_from_manifest(manifest: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(manifest, dict):
        return None
    output = manifest.get("output_dialect")
    return output if isinstance(output, dict) else None


def _infer_encoding(path: str) -> str:
    encodings = ("utf-8", "latin-1", "cp1252")
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as handle:
                handle.read(2048)
            return enc
        except Exception:
            continue
    return "utf-8"


def _read_text_sample(path: str, encoding: str, max_bytes: int) -> str:
    if not path or not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding=encoding, errors="replace") as handle:
            return handle.read(max_bytes)
    except Exception:
        return ""


def _infer_delimiter_from_file(path: str) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None
    encoding = _infer_encoding(path)
    sample = _read_text_sample(path, encoding, 50000)
    if not sample:
        return None
    delimiters = [",", ";", "\t", "|"]
    try:
        sniffed = csv.Sniffer().sniff(sample, delimiters=delimiters)
        if getattr(sniffed, "delimiter", None):
            return sniffed.delimiter
    except Exception:
        pass
    counts = {delim: sample.count(delim) for delim in delimiters}
    best = max(counts, key=counts.get)
    return best if counts.get(best, 0) > 0 else None


def _infer_decimal_from_sample(sample: str, sep: str) -> str:
    if not sample:
        return "."
    comma_hits = len(re.findall(r"\d+,\d+", sample))
    dot_hits = len(re.findall(r"\d+\.\d+", sample))
    if comma_hits > dot_hits:
        return ","
    if dot_hits > comma_hits:
        return "."
    return "."


def _load_json(path: str) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _read_csv_header(path: str, dialect: Dict[str, Any]) -> List[str]:
    if not path or not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(
            path,
            nrows=0,
            sep=dialect.get("sep", ","),
            decimal=dialect.get("decimal", "."),
            encoding=dialect.get("encoding", "utf-8"),
        )
        return [str(col) for col in df.columns if col]
    except Exception:
        return []


def _read_csv_sample(
    path: str,
    dialect: Dict[str, Any],
    columns: Optional[List[str]],
    dtype: Optional[Any],
    nrows: int,
) -> Optional[pd.DataFrame]:
    if not path or not os.path.exists(path):
        return None
    kwargs = {
        "nrows": nrows,
        "sep": dialect.get("sep", ","),
        "decimal": dialect.get("decimal", "."),
        "encoding": dialect.get("encoding", "utf-8"),
        "low_memory": False,
    }
    if dtype is not None:
        kwargs["dtype"] = dtype
    if columns:
        kwargs["usecols"] = columns
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        return None


def _clean_numeric_strings(values: pd.Series) -> pd.Series:
    cleaned = values.dropna().astype(str).str.strip()
    cleaned = cleaned[cleaned != ""]
    cleaned = cleaned[~cleaned.str.lower().isin({"nan", "none", "null"})]
    cleaned = cleaned.str.replace(" ", "", regex=False)
    return cleaned


def _best_numeric_parse_ratio(values: pd.Series) -> float:
    cleaned = _clean_numeric_strings(values)
    if cleaned.empty:
        return 1.0
    candidates = [
        cleaned,
        cleaned.str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
        cleaned.str.replace(",", "", regex=False),
    ]
    best_ratio = 0.0
    total = float(len(cleaned))
    for candidate in candidates:
        parsed = pd.to_numeric(candidate, errors="coerce")
        ratio = float(parsed.notna().sum()) / total if total else 1.0
        if ratio > best_ratio:
            best_ratio = ratio
    return best_ratio


def _check_numeric_parsing_validation(
    sample_str: Optional[pd.DataFrame],
    sample_infer: Optional[pd.DataFrame],
    params: Dict[str, Any],
) -> Tuple[List[str], Dict[str, Any]]:
    columns = _list_str(params.get("columns"))
    threshold = float(params.get("min_parse_ratio", 0.9))
    check_mode = str(params.get("check") or "no_string_remainders").strip().lower()
    evidence: Dict[str, Any] = {"check": check_mode, "threshold": threshold, "ratios": {}}
    if not columns:
        evidence["note"] = "no_columns_configured"
        return [], evidence

    issues: List[str] = []
    for col in columns:
        if sample_infer is not None and col in sample_infer.columns:
            if pd.api.types.is_numeric_dtype(sample_infer[col]):
                evidence["ratios"][col] = 1.0
                continue
        if sample_str is None or col not in sample_str.columns:
            evidence["ratios"][col] = None
            continue
        ratio = _best_numeric_parse_ratio(sample_str[col])
        evidence["ratios"][col] = round(ratio, 4)
        if ratio < threshold:
            issues.append(f"{col} parseable_ratio={ratio:.2f} < {threshold:.2f}")
    return issues, evidence


def _pick_first_existing(candidates: List[str], cleaned_header: List[str]) -> str:
    header_set = set(cleaned_header or [])
    for col in candidates:
        if col in header_set:
            return col
    return ""


def _is_null_like_text(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    text = str(value).strip().lower()
    return text in {"", "nan", "none", "null", "na", "<na>", "nat"}


def _build_null_mask(
    sample_str: Optional[pd.DataFrame],
    sample_infer: Optional[pd.DataFrame],
    column: str,
) -> Optional[pd.Series]:
    if sample_infer is not None and column in sample_infer.columns:
        try:
            return sample_infer[column].isna()
        except Exception:
            pass
    if sample_str is not None and column in sample_str.columns:
        try:
            return sample_str[column].map(_is_null_like_text)
        except Exception:
            return None
    return None


def _resolve_numeric_cast_columns(
    cleaned_header: List[str],
    column_roles: Dict[str, List[str]],
    params: Dict[str, Any],
) -> List[str]:
    columns = _list_str(params.get("columns"))
    if columns:
        return [col for col in columns if col in (cleaned_header or [])]

    candidates: List[str] = []
    candidates.extend(_columns_with_role_tokens(column_roles, {"feature", "predictor", "input"}))
    # Common generic feature naming used in tabular datasets.
    candidates.extend([col for col in (cleaned_header or []) if re.match(r"(?i)^var[_-]?\d+$", str(col))])
    if not candidates:
        candidates.extend(
            [
                col
                for col in (cleaned_header or [])
                if col
                and col not in _columns_with_role_tokens(column_roles, {"id", "identifier", "split", "partition"})
            ]
        )

    deduped: List[str] = []
    seen: set[str] = set()
    for col in candidates:
        if col in seen or col not in (cleaned_header or []):
            continue
        seen.add(col)
        deduped.append(col)
    return deduped


def _check_numeric_type_casting(
    sample_str: Optional[pd.DataFrame],
    sample_infer: Optional[pd.DataFrame],
    cleaned_header: List[str],
    params: Dict[str, Any],
    column_roles: Dict[str, List[str]],
) -> Tuple[List[str], Dict[str, Any]]:
    columns = _resolve_numeric_cast_columns(cleaned_header, column_roles, params)
    parse_params = dict(params or {})
    parse_params["columns"] = columns
    parse_params.setdefault("min_parse_ratio", 0.95)
    issues, evidence = _check_numeric_parsing_validation(sample_str, sample_infer, parse_params)
    evidence["columns_checked"] = len(columns)
    evidence["columns_preview"] = columns[:20]
    return issues, evidence


def _check_target_null_alignment_with_split(
    sample_str: Optional[pd.DataFrame],
    sample_infer: Optional[pd.DataFrame],
    cleaned_header: List[str],
    params: Dict[str, Any],
    column_roles: Dict[str, List[str]],
) -> Tuple[List[str], Dict[str, Any]]:
    target_candidates = _list_str(params.get("target_column")) + _list_str(params.get("target_columns"))
    split_candidates = _list_str(params.get("split_column")) + _list_str(params.get("split_columns"))
    if not target_candidates:
        target_candidates.extend(_columns_with_role_tokens(column_roles, {"target", "label", "outcome"}))
        target_candidates.extend(["target", "label", "y"])
    if not split_candidates:
        split_candidates.extend(_columns_with_role_tokens(column_roles, {"split", "partition", "fold"}))
        split_candidates.extend(["__split", "split", "partition", "fold"])

    target_col = _pick_first_existing(target_candidates, cleaned_header)
    split_col = _pick_first_existing(split_candidates, cleaned_header)

    evidence: Dict[str, Any] = {
        "target_column": target_col or None,
        "split_column": split_col or None,
    }
    if not target_col or not split_col:
        evidence["note"] = "target_or_split_column_missing"
        return [], evidence

    frame = sample_infer if sample_infer is not None else sample_str
    if frame is None or target_col not in frame.columns or split_col not in frame.columns:
        evidence["note"] = "sample_missing_required_columns"
        return [], evidence

    null_mask = _build_null_mask(sample_str, sample_infer, target_col)
    if null_mask is None:
        evidence["note"] = "target_null_mask_unavailable"
        return [], evidence

    total_rows = int(len(frame))
    null_count = int(null_mask.sum())
    evidence["total_rows_sampled"] = total_rows
    evidence["target_null_count"] = null_count
    if total_rows <= 0 or null_count <= 0:
        evidence["note"] = "no_target_nulls_in_sample"
        return [], evidence

    split_series = frame[split_col].astype(str).fillna("").map(lambda x: x.strip())
    null_split = split_series[null_mask]
    nonnull_split = split_series[~null_mask]
    if null_split.empty:
        evidence["note"] = "no_null_rows_after_filter"
        return [], evidence

    null_distribution = null_split.value_counts(dropna=False).to_dict()
    nonnull_distribution = nonnull_split.value_counts(dropna=False).to_dict()
    evidence["null_split_distribution"] = {str(k): int(v) for k, v in null_distribution.items()}
    evidence["nonnull_split_distribution"] = {str(k): int(v) for k, v in nonnull_distribution.items()}

    dominant_split = str(null_split.mode().iloc[0]) if not null_split.mode().empty else ""
    evidence["dominant_null_split_value"] = dominant_split or None
    if not dominant_split:
        return [], evidence

    null_outside_ratio = float((null_split != dominant_split).mean()) if len(null_split) else 0.0
    labeled_in_null_split_ratio = float((nonnull_split == dominant_split).mean()) if len(nonnull_split) else 0.0

    max_null_outside = float(params.get("max_nulls_outside_dominant_split", 0.05))
    max_labeled_in_null_split = float(params.get("max_labeled_in_null_split_ratio", 0.10))
    evidence["null_outside_ratio"] = round(null_outside_ratio, 4)
    evidence["labeled_in_null_split_ratio"] = round(labeled_in_null_split_ratio, 4)
    evidence["thresholds"] = {
        "max_nulls_outside_dominant_split": max_null_outside,
        "max_labeled_in_null_split_ratio": max_labeled_in_null_split,
    }

    issues: List[str] = []
    if null_outside_ratio > max_null_outside:
        issues.append(
            f"target null rows not aligned with split '{dominant_split}' "
            f"(outside_ratio={null_outside_ratio:.4f} > {max_null_outside:.4f})"
        )
    if labeled_in_null_split_ratio > max_labeled_in_null_split:
        issues.append(
            f"labeled rows detected in null-dominant split '{dominant_split}' "
            f"(ratio={labeled_in_null_split_ratio:.4f} > {max_labeled_in_null_split:.4f})"
        )
    return issues, evidence


def _check_id_uniqueness_validation(
    sample_str: Optional[pd.DataFrame],
    sample_infer: Optional[pd.DataFrame],
    cleaned_header: List[str],
    params: Dict[str, Any],
    column_roles: Dict[str, List[str]],
) -> Tuple[List[str], Dict[str, Any]]:
    columns = _list_str(params.get("columns")) + _list_str(params.get("id_columns"))
    if not columns:
        columns.extend(_columns_with_role_tokens(column_roles, {"id", "identifier", "key"}))
    if not columns:
        regex = params.get("identifier_name_regex") or _GENERIC_ID_REGEX
        try:
            pattern = re.compile(regex)
        except re.error:
            pattern = re.compile(_GENERIC_ID_REGEX)
        columns.extend([col for col in cleaned_header if pattern.search(str(col))])

    deduped: List[str] = []
    seen: set[str] = set()
    for col in columns:
        if col in seen or col not in cleaned_header:
            continue
        seen.add(col)
        deduped.append(col)
    columns = deduped

    evidence: Dict[str, Any] = {"id_columns": columns, "duplicate_ratio": {}}
    if not columns:
        evidence["note"] = "no_id_columns_detected"
        return [], evidence

    max_duplicate_ratio = float(params.get("max_duplicate_ratio", 0.0))
    min_samples = int(params.get("min_samples", 20))
    issues: List[str] = []

    for col in columns:
        series = None
        if sample_str is not None and col in sample_str.columns:
            series = sample_str[col]
        elif sample_infer is not None and col in sample_infer.columns:
            series = sample_infer[col]
        if series is None:
            evidence["duplicate_ratio"][col] = None
            continue
        cleaned = series.dropna().astype(str).map(lambda x: x.strip())
        cleaned = cleaned[cleaned != ""]
        total = int(len(cleaned))
        unique = int(cleaned.nunique(dropna=True))
        duplicate_ratio = float((total - unique) / total) if total else 0.0
        evidence["duplicate_ratio"][col] = round(duplicate_ratio, 4)
        evidence.setdefault("sample_counts", {})[col] = {"total": total, "unique": unique}
        if total < min_samples:
            continue
        if duplicate_ratio > max_duplicate_ratio:
            issues.append(
                f"{col} duplicate_ratio={duplicate_ratio:.4f} > {max_duplicate_ratio:.4f}"
            )
    return issues, evidence


def _compute_null_fraction(
    sample_infer: Optional[pd.DataFrame],
    sample_str: Optional[pd.DataFrame],
    column: str,
) -> Optional[float]:
    series = None
    if sample_infer is not None and column in sample_infer.columns:
        series = sample_infer[column]
    elif sample_str is not None and column in sample_str.columns:
        series = sample_str[column]
    if series is None:
        return None
    total = len(series)
    if total == 0:
        return None
    return float(series.isna().sum() / total)


def _list_str(value: Any) -> List[str]:
    def _is_compaction_marker(token: str) -> bool:
        text = str(token or "").strip()
        if not text:
            return True
        if re.match(r"^\.\.\.\(\d+\s+total\)$", text):
            return True
        return text in {"...", "...total", "...(total)"}

    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if _is_compaction_marker(text):
                continue
            out.append(text)
        return out
    if isinstance(value, str) and value.strip():
        text = value.strip()
        if text and not _is_compaction_marker(text):
            return [text]
    return []


def _load_column_inventory_names(path: str = "data/column_inventory.json") -> List[str]:
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            return [str(col) for col in payload if col]
        if isinstance(payload, dict):
            cols = payload.get("columns")
            if isinstance(cols, list):
                return [str(col) for col in cols if col]
    except Exception:
        return []
    return []


def _resolve_required_columns_for_review(
    view: Dict[str, Any],
    manifest: Optional[Dict[str, Any]] = None,
    artifact_obligations: Optional[Dict[str, Any]] = None,
) -> List[str]:
    preferred_output_path = str(view.get("output_path") or "").strip()
    obligated_required = _resolve_required_columns_from_artifact_obligations(
        artifact_obligations,
        preferred_output_path=preferred_output_path or None,
    )
    if obligated_required:
        return obligated_required

    required = view.get("required_columns")
    required_selectors = view.get("required_feature_selectors")
    column_transformations = view.get("column_transformations")
    if not isinstance(column_transformations, dict):
        column_transformations = {}

    base_required: List[str] = []
    if isinstance(required, list):
        base_required = _list_str(required)
    # If required_columns got compacted (count/head/tail), load from file.
    if not base_required:
        path = view.get("required_columns_path") or "data/required_columns.json"
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, list) and payload:
                    base_required = _list_str(payload)
                elif isinstance(payload, dict):
                    candidates = payload.get("required_columns")
                    if not isinstance(candidates, list):
                        candidates = payload.get("columns")
                    if isinstance(candidates, list) and candidates:
                        base_required = _list_str(candidates)
                    if not isinstance(required_selectors, list):
                        selectors_candidate = payload.get("required_feature_selectors")
                        if isinstance(selectors_candidate, list):
                            required_selectors = selectors_candidate
            except Exception:
                pass

    inventory_cols = _load_column_inventory_names("data/column_inventory.json")
    resolved = resolve_required_columns_for_cleaning(
        required_columns=base_required,
        required_feature_selectors=required_selectors,
        candidate_columns=inventory_cols,
        column_transformations=column_transformations,
        manifest=manifest if isinstance(manifest, dict) else {},
    )
    merged = resolved.get("required_columns")
    if isinstance(merged, list):
        return [str(col) for col in merged if isinstance(col, str) and col.strip()]
    return base_required


def _resolve_required_columns_from_artifact_obligations(
    artifact_obligations: Optional[Dict[str, Any]],
    *,
    preferred_output_path: Optional[str] = None,
) -> List[str]:
    if not isinstance(artifact_obligations, dict):
        return []
    bindings = artifact_obligations.get("artifact_bindings")
    if not isinstance(bindings, list):
        return []

    preferred_output_norm = str(preferred_output_path or "").strip().lower()
    candidates: List[Tuple[int, int, List[str]]] = []
    fallback_candidates: List[List[str]] = []

    for idx, binding in enumerate(bindings):
        if not isinstance(binding, dict):
            continue
        declared = binding.get("declared_binding")
        if not isinstance(declared, dict):
            continue

        required = _list_str(declared.get("required_columns"))
        if not required:
            schema_binding = declared.get("schema_binding")
            if isinstance(schema_binding, dict):
                required = _list_str(schema_binding.get("required_columns"))
        if not required:
            continue

        binding_name = str(binding.get("binding_name") or binding.get("artifact_name") or "").strip().lower()
        source_path = str(binding.get("source_contract_path") or "").strip().lower()
        output_path = str(declared.get("output_path") or "").strip().lower()

        score = 0
        if "cleaned_dataset" in binding_name or "cleaned_dataset" in source_path:
            score += 4
        elif "clean_dataset" in binding_name or "clean_dataset" in source_path:
            score += 3
        if preferred_output_norm and output_path and output_path == preferred_output_norm:
            score += 2

        deduped_required = _dedupe_list(required)
        fallback_candidates.append(deduped_required)
        candidates.append((score, idx, deduped_required))

    if not candidates:
        return []
    best = sorted(candidates, key=lambda item: (-item[0], item[1]))[0]
    if best[0] > 0:
        return best[2]
    if len(fallback_candidates) == 1:
        return fallback_candidates[0]
    return []


def _coerce_roles(raw: Any) -> Dict[str, List[str]]:
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, List[str]] = {}
    for key, val in raw.items():
        if isinstance(val, list):
            out[str(key)] = [str(item) for item in val if item]
    return out


def _build_facts(
    cleaned_header: List[str],
    required_columns: List[str],
    manifest: Dict[str, Any],
    sample_str: Optional[pd.DataFrame],
    sample_infer: Optional[pd.DataFrame],
    raw_sample: Optional[pd.DataFrame],
    gates: Optional[List[Dict[str, Any]]] = None,
    column_roles: Optional[Dict[str, List[str]]] = None,
    dataset_profile: Optional[Dict[str, Any]] = None,
    outlier_policy: Optional[Dict[str, Any]] = None,
    outlier_report: Optional[Dict[str, Any]] = None,
    outlier_report_path: Optional[str] = None,
    cleaned_csv_path: Optional[str] = None,
) -> Dict[str, Any]:
    facts: Dict[str, Any] = {}
    facts["cleaned_header"] = cleaned_header[:200]
    facts["required_columns"] = required_columns
    facts["missing_required_columns"] = [c for c in required_columns if c not in cleaned_header]
    forbidden_columns = _collect_forbidden_columns_for_review(gates or [], column_roles or {})
    if forbidden_columns:
        cleaned_header_norm = {str(col).strip().lower(): str(col) for col in cleaned_header if str(col).strip()}
        manifest_removed = _collect_manifest_removed_columns(manifest)
        manifest_removed_norm = {str(col).strip().lower(): str(col) for col in manifest_removed if str(col).strip()}
        facts["forbidden_columns"] = forbidden_columns
        forbidden_in_cleaned = [
            col for col in forbidden_columns if str(col).strip().lower() in cleaned_header_norm
        ]
        facts["forbidden_columns_present_in_cleaned_header"] = forbidden_in_cleaned
        facts["forbidden_columns_absent_in_cleaned_header"] = [
            col for col in forbidden_columns if str(col).strip().lower() not in cleaned_header_norm
        ]
        facts["forbidden_columns_declared_removed_in_manifest"] = [
            col for col in forbidden_columns if str(col).strip().lower() in manifest_removed_norm
        ]
        facts["required_columns_scope_note"] = (
            "missing_required_columns tracks columns expected in the cleaned artifact scope; "
            "it does not imply that those columns are still present in the cleaned header."
        )
        # If forbidden columns appear in the cleaned CSV but an enriched CSV
        # exists that correctly excludes them, the DE has properly separated
        # the full-fidelity cleaned output from the ML-ready enriched output.
        # Record this so the LLM doesn't penalise a correct two-file strategy.
        if forbidden_in_cleaned and cleaned_csv_path:
            enriched_header = _read_enriched_csv_header(cleaned_csv_path)
            if enriched_header is not None:
                enriched_norm = {str(c).strip().lower() for c in enriched_header if str(c).strip()}
                forbidden_in_enriched = [
                    col for col in forbidden_columns if str(col).strip().lower() in enriched_norm
                ]
                facts["enriched_csv_exists"] = True
                facts["forbidden_columns_present_in_enriched_header"] = forbidden_in_enriched
                facts["forbidden_columns_excluded_from_enriched"] = not bool(forbidden_in_enriched)
                if not forbidden_in_enriched:
                    facts["leakage_exclusion_note"] = (
                        "Forbidden columns are present in the cleaned CSV (full-fidelity archive) "
                        "but correctly excluded from the enriched CSV (ML-ready deliverable). "
                        "This is a valid two-file strategy — do NOT fail the leakage gate."
                    )

    row_counts = manifest.get("row_counts") or {}
    facts["row_counts"] = {
        "rows_before": manifest.get("rows_before") or row_counts.get("initial"),
        "rows_after": manifest.get("rows_after") or row_counts.get("final"),
    }
    conversions = manifest.get("conversions") if isinstance(manifest, dict) else {}
    dropped = manifest.get("dropped_columns") if isinstance(manifest, dict) else {}
    warnings = manifest.get("warnings") if isinstance(manifest, dict) else []
    facts["manifest_summary"] = {
        "conversions_count": len(conversions) if isinstance(conversions, dict) else 0,
        "dropped_columns_count": len(dropped) if isinstance(dropped, (list, dict)) else 0,
        "warnings": [str(w) for w in warnings[:5]] if isinstance(warnings, list) else [],
    }
    facts["column_stats_sample"] = _build_column_stats(sample_str, sample_infer, max_cols=40)
    facts["cleaned_sample_rows"] = _sample_rows(sample_str, max_rows=5)
    facts["raw_sample_rows"] = _sample_rows(raw_sample, max_rows=5)

    # Include numeric ranges from data profile for LLM context (no_semantic_rescale reasoning)
    if isinstance(dataset_profile, dict):
        numeric_summary = dataset_profile.get("numeric_summary", {})
        if numeric_summary:
            # Extract min/max ranges for key columns
            numeric_ranges = {}
            for col, stats in list(numeric_summary.items())[:50]:  # Limit to 50 columns
                if isinstance(stats, dict):
                    numeric_ranges[col] = {
                        "min": stats.get("min"),
                        "max": stats.get("max"),
                        "mean": stats.get("mean"),
                    }
            if numeric_ranges:
                facts["source_data_numeric_ranges"] = numeric_ranges

    if isinstance(outlier_policy, dict) and outlier_policy:
        policy_enabled = _outlier_policy_enabled(outlier_policy)
        report_present = isinstance(outlier_report, dict) and bool(outlier_report)
        report_columns_touched = _extract_outlier_report_columns(outlier_report)
        policy_summary: Dict[str, Any] = {
            "enabled": policy_enabled,
            "apply_stage": outlier_policy.get("apply_stage"),
            "target_columns": outlier_policy.get("target_columns"),
            "report_path": outlier_report_path,
            "report_present": report_present,
        }
        if report_present:
            if report_columns_touched:
                policy_summary["columns_touched"] = report_columns_touched
            if outlier_report.get("rows_affected") is not None:
                policy_summary["rows_affected"] = outlier_report.get("rows_affected")
            if outlier_report.get("flags_created") is not None:
                policy_summary["flags_created"] = outlier_report.get("flags_created")
            if outlier_report.get("status") is not None:
                policy_summary["status"] = outlier_report.get("status")
        manifest_outlier = manifest.get("outlier_treatment") if isinstance(manifest.get("outlier_treatment"), dict) else None
        if manifest_outlier:
            policy_summary["manifest_outlier_treatment"] = manifest_outlier
        facts["outlier_policy"] = policy_summary

    return facts


def _read_enriched_csv_header(cleaned_csv_path: str) -> Optional[List[str]]:
    """Try to find and read the header of an enriched CSV next to the cleaned one.

    DE scripts often produce both a cleaned (full-fidelity) CSV and an enriched
    (ML-ready) CSV in the same directory.  Common naming patterns:
      - dataset_enriched.csv
      - enriched_dataset.csv
      - dataset_clean_enriched.csv
    """
    if not cleaned_csv_path:
        return None
    parent = os.path.dirname(cleaned_csv_path)
    candidates = [
        os.path.join(parent, "dataset_enriched.csv"),
        os.path.join(parent, "enriched_dataset.csv"),
        os.path.join(parent, "dataset_clean_enriched.csv"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            try:
                df = pd.read_csv(candidate, nrows=0)
                return list(df.columns)
            except Exception:
                continue
    return None


def _collect_forbidden_columns_for_review(
    gates: List[Dict[str, Any]],
    column_roles: Dict[str, List[str]],
) -> List[str]:
    forbidden: List[str] = []
    for gate in gates:
        if not isinstance(gate, dict):
            continue
        gate_norm = _normalize_gate_name(gate.get("name"))
        if "leakage" not in gate_norm and "exclude" not in gate_norm:
            continue
        params = gate.get("params") if isinstance(gate.get("params"), dict) else {}
        for key in ("forbidden_columns", "forbidden_at_inference", "excluded_columns", "columns"):
            forbidden.extend(_list_str(params.get(key)))
    if isinstance(column_roles, dict):
        for role_name, columns in column_roles.items():
            role_key = str(role_name or "").strip().lower()
            if not role_key:
                continue
            if any(token in role_key for token in ("forbidden", "leakage", "post_decision", "postdecision")):
                forbidden.extend(_list_str(columns))
    return _dedupe_list(forbidden)


def _collect_manifest_removed_columns(manifest: Dict[str, Any]) -> List[str]:
    if not isinstance(manifest, dict):
        return []
    dropped = manifest.get("dropped_columns")
    if isinstance(dropped, dict):
        return _dedupe_list(list(dropped.keys()))
    if isinstance(dropped, list):
        return _dedupe_list(dropped)
    return []


def _build_column_stats(
    sample_str: Optional[pd.DataFrame],
    sample_infer: Optional[pd.DataFrame],
    max_cols: int,
) -> List[Dict[str, Any]]:
    if sample_str is None or sample_str.empty:
        return []
    columns = list(sample_str.columns)[:max_cols]
    stats: List[Dict[str, Any]] = []
    for col in columns:
        series = sample_str[col]
        total = len(series)
        null_frac = float(series.isna().sum() / total) if total else 0.0
        unique_count = int(series.nunique(dropna=True)) if total else 0
        examples = [str(v) for v in series.dropna().head(3).tolist()]
        dtype = "unknown"
        if sample_infer is not None and col in sample_infer.columns:
            dtype = str(sample_infer[col].dtype)
        stats.append(
            {
                "column": col,
                "dtype": dtype,
                "null_frac": round(null_frac, 4),
                "unique_count": unique_count,
                "examples": examples,
            }
        )
    return stats


def _sample_rows(sample_df: Optional[pd.DataFrame], max_rows: int) -> List[Dict[str, Any]]:
    if sample_df is None or sample_df.empty:
        return []
    rows = sample_df.head(max_rows).to_dict(orient="records")
    return rows if isinstance(rows, list) else []


def _evaluate_gates_deterministic(
    gates: List[Dict[str, Any]],
    required_columns: List[str],
    cleaned_header: List[str],
    cleaned_csv_path: str,
    sample_str: Optional[pd.DataFrame],
    sample_infer: Optional[pd.DataFrame],
    manifest: Dict[str, Any],
    raw_sample: Optional[pd.DataFrame],
    column_roles: Dict[str, List[str]],
    model_features: Optional[List[str]] = None,
    allowed_feature_sets: Any = None,
    dataset_profile: Optional[Dict[str, Any]] = None,
    cleaning_code: Optional[str] = None,
    outlier_policy: Optional[Dict[str, Any]] = None,
    outlier_report: Optional[Dict[str, Any]] = None,
    outlier_report_path: Optional[str] = None,
    column_dtype_targets: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    hard_failures: List[str] = []
    soft_failures: List[str] = []
    failed_checks: List[str] = []
    required_fixes: List[str] = []
    warnings: List[str] = []
    failure_summaries: List[str] = []
    warning_summaries: List[str] = []
    gate_results: List[Dict[str, Any]] = []
    model_features = _list_str(model_features)
    if not model_features and isinstance(allowed_feature_sets, dict):
        model_features = _list_str(allowed_feature_sets.get("model_features"))
    outlier_policy = outlier_policy if isinstance(outlier_policy, dict) else {}
    outlier_report = outlier_report if isinstance(outlier_report, dict) else {}
    column_dtype_targets = column_dtype_targets if isinstance(column_dtype_targets, dict) else {}
    training_rows_context: Optional[Dict[str, Any]] = None

    for gate in gates:
        name = gate["name"]
        gate_key = _normalize_gate_name(name)
        severity = gate["severity"]
        params = gate["params"]
        issues: List[str] = []
        evidence: Dict[str, Any] = {}
        evaluated = True
        severity_used = severity

        if gate_key == "required_columns_present":
            issues = _check_required_columns(required_columns, cleaned_header, cleaned_csv_path)
            evidence["missing"] = [c for c in required_columns if c not in cleaned_header]
        elif gate_key == "id_integrity":
            issues, id_evidence = _check_id_integrity(
                cleaned_header,
                sample_str,
                sample_infer,
                params,
                column_roles,
            )
            if isinstance(id_evidence, dict):
                evidence.update(id_evidence)
                if not bool(id_evidence.get("applies_if", True)):
                    warnings.append(f"id_integrity skipped: {id_evidence.get('skip_reason', 'not_applicable')}")
        elif gate_key == "no_semantic_rescale":
            explicit_patterns = _detect_explicit_rescale_patterns(cleaning_code)
            if explicit_patterns:
                issues = [
                    "Explicit numeric rescaling operations detected in cleaning code "
                    f"({', '.join(explicit_patterns[:5])})."
                ]
                evidence["patterns_found"] = explicit_patterns
                evidence["deterministic_support"] = "explicit_rescale_patterns"
                evaluated = True
            else:
                # Keep LLM delegation for ambiguous semantic cases while covering explicit ops deterministically.
                issues = []
                evidence["llm_delegated"] = True
                evidence["reason"] = "No explicit rescale patterns found; delegate semantic interpretation to LLM"
                evaluated = False
        elif gate_key == "no_synthetic_data":
            issues = _check_no_synthetic_data(manifest, cleaning_code=cleaning_code)
        elif gate_key == "row_count_sanity":
            skip_row_count, skip_reason, skip_evidence = _should_skip_row_count_sanity(manifest, params, column_roles)
            if isinstance(skip_evidence, dict):
                evidence.update(skip_evidence)
            if skip_row_count:
                issues = []
                evidence["applies_if"] = False
                evidence["skip_reason"] = skip_reason
                warnings.append(f"row_count_sanity skipped: {skip_reason}")
            else:
                issues = _check_row_count_sanity(manifest, params)
            evidence["row_counts"] = (manifest.get("row_counts") or {})
        elif gate_key == "feature_coverage_sanity":
            issues, evidence = _check_feature_coverage_sanity(
                cleaned_header=cleaned_header,
                required_columns=required_columns,
                column_roles=column_roles,
                model_features=model_features,
                allowed_feature_sets=allowed_feature_sets,
                dataset_profile=dataset_profile,
                params=params,
            )
        elif gate_key == "numeric_parsing_validation":
            issues, evidence = _check_numeric_parsing_validation(sample_str, sample_infer, params)
        elif gate_key == "numeric_type_casting_check":
            issues, evidence = _check_numeric_type_casting(
                sample_str=sample_str,
                sample_infer=sample_infer,
                cleaned_header=cleaned_header,
                params=params,
                column_roles=column_roles,
            )
        elif gate_key.endswith("_numeric_parseable") or "numeric_parseable" in gate_key:
            issues, evidence = _check_numeric_parseable_gate(
                cleaned_csv_path=cleaned_csv_path,
                cleaned_header=cleaned_header,
                sample_str=sample_str,
                sample_infer=sample_infer,
                params=params,
                manifest=manifest,
                gate_key=gate_key,
            )
            if not bool(evidence.get("applies_if", True)):
                warnings.append(
                    f"{gate_key} skipped: {evidence.get('skip_reason', 'not_applicable')}"
                )
                evaluated = False
        elif gate_key.endswith("_parseable") and _is_datetime_parseable_gate(gate_key, params):
            issues, evidence = _check_datetime_parseable_gate(
                cleaned_csv_path=cleaned_csv_path,
                cleaned_header=cleaned_header,
                sample_str=sample_str,
                params=params,
                manifest=manifest,
                gate_key=gate_key,
            )
            if not bool(evidence.get("applies_if", True)):
                warnings.append(
                    f"{gate_key} skipped: {evidence.get('skip_reason', 'not_applicable')}"
                )
                evaluated = False
        elif gate_key == "target_null_alignment_with_split":
            issues, evidence = _check_target_null_alignment_with_split(
                sample_str=sample_str,
                sample_infer=sample_infer,
                cleaned_header=cleaned_header,
                params=params,
                column_roles=column_roles,
            )
        elif gate_key == "id_uniqueness_validation":
            issues, evidence = _check_id_uniqueness_validation(
                sample_str=sample_str,
                sample_infer=sample_infer,
                cleaned_header=cleaned_header,
                params=params,
                column_roles=column_roles,
            )
        elif gate_key == "null_handling_verification":
            columns = _list_str(params.get("columns"))
            allow_nulls = params.get("allow_nulls", True)
            tolerance = params.get("tolerance")
            if tolerance is None:
                tolerance = params.get("null_tolerance", 0.0)
            try:
                tolerance_val = float(tolerance)
            except Exception:
                tolerance_val = 0.0
            evidence = {
                "columns": columns,
                "allow_nulls": bool(allow_nulls),
                "tolerance": tolerance_val,
                "null_frac": {},
            }
            if not columns:
                evidence["note"] = "no_columns_configured"
            for col in columns:
                if cleaned_header and col not in cleaned_header:
                    issues.append(f"Missing column: {col}")
                    severity_used = "HARD"
                    continue
                null_frac = _compute_null_fraction(sample_infer, sample_str, col)
                evidence["null_frac"][col] = None if null_frac is None else round(null_frac, 4)
                if null_frac is None:
                    warnings.append(f"NULL_HANDLING_EVIDENCE_MISSING: {col}")
                    continue
                if allow_nulls is False and null_frac > tolerance_val:
                    issues.append(f"{col} null_frac={null_frac:.4f} > {tolerance_val:.4f}")
            warn_threshold = params.get("warn_null_frac_threshold", 0.5)
            try:
                warn_threshold_val = float(warn_threshold)
            except Exception:
                warn_threshold_val = 0.5
            for col in model_features:
                if col in columns:
                    continue
                if cleaned_header and col not in cleaned_header:
                    continue
                null_frac = _compute_null_fraction(sample_infer, sample_str, col)
                if null_frac is None or null_frac < warn_threshold_val:
                    continue
                warnings.append(f"NULLS_OUTSIDE_GATE: {col} null_frac={null_frac:.4f}")
        elif gate_key == "boolean_normalization":
            issues, evidence = _check_boolean_normalization(
                cleaned_csv_path=cleaned_csv_path,
                cleaned_header=cleaned_header,
                sample_str=sample_str,
                raw_sample=raw_sample,
                params=params,
                column_dtype_targets=column_dtype_targets,
                manifest=manifest,
            )
            if not bool(evidence.get("applies_if", True)):
                warnings.append(
                    f"boolean_normalization skipped: {evidence.get('skip_reason', 'not_applicable')}"
                )
        elif gate_key in {"training_cohort_filter_enforced", "scoring_cohort_filter_enforced"}:
            issues, evidence = _check_split_condition_enforced(
                cleaned_csv_path=cleaned_csv_path,
                cleaned_header=cleaned_header,
                params=params,
                column_roles=column_roles,
            )
            if not bool(evidence.get("applies_if", True)):
                warnings.append(
                    f"{gate_key} skipped: {evidence.get('skip_reason', 'not_applicable')}"
                )
        elif (
            gate_key == "target_not_null_in_training"
            or ("not_null" in gate_key and str(params.get("partition") or "").strip().lower() == "training")
            or ("training" in gate_key and "not_null" in gate_key)
        ):
            if _resolve_partition_filter(params) or gate_key != "target_not_null_in_training":
                issues, evidence = _check_target_not_null_under_filter(
                    cleaned_csv_path=cleaned_csv_path,
                    cleaned_header=cleaned_header,
                    params=params,
                )
            else:
                issues, evidence = _check_target_not_null_in_training(
                    cleaned_csv_path=cleaned_csv_path,
                    cleaned_header=cleaned_header,
                    params=params,
                    column_roles=column_roles,
                    training_rows_context=training_rows_context,
                )
            if not bool(evidence.get("applies_if", True)):
                warnings.append(
                    f"{gate_key} skipped: {evidence.get('skip_reason', 'not_applicable')}"
                )
                evaluated = False
        elif "duplicate" in gate_key and ("training" in gate_key or params.get("partition")):
            issues, evidence = _check_no_exact_duplicates_under_filter(
                cleaned_csv_path=cleaned_csv_path,
                cleaned_header=cleaned_header,
                params=params,
            )
            if not bool(evidence.get("applies_if", True)):
                warnings.append(
                    f"{gate_key} skipped: {evidence.get('skip_reason', 'not_applicable')}"
                )
                evaluated = False
        elif gate_key in {"identifier_columns_excluded_from_features", "leakage_columns_excluded_from_feature_matrix"}:
            issues, evidence = _check_identifier_columns_excluded_from_features(
                cleaned_csv_path=cleaned_csv_path,
                cleaned_header=cleaned_header,
                sample_str=sample_str,
                sample_infer=sample_infer,
                params=params,
                column_roles=column_roles,
            )
            if not bool(evidence.get("applies_if", True)):
                warnings.append(
                    "identifier_columns_excluded_from_features skipped: "
                    + str(evidence.get("skip_reason", "not_applicable"))
                )
        elif gate_key == "arr_current_numeric_conversion_verified":
            issues, evidence = _check_arr_current_numeric_conversion_verified(
                cleaned_header=cleaned_header,
                sample_str=sample_str,
                sample_infer=sample_infer,
                params=params,
                manifest=manifest,
            )
            if not bool(evidence.get("applies_if", True)):
                warnings.append(
                    "arr_current_numeric_conversion_verified skipped: "
                    + str(evidence.get("skip_reason", "not_applicable"))
                )
        elif gate_key == "nps_forward_fill_temporal_integrity":
            issues, evidence = _check_nps_forward_fill_temporal_integrity(
                cleaned_csv_path=cleaned_csv_path,
                cleaned_header=cleaned_header,
                sample_str=sample_str,
                sample_infer=sample_infer,
                params=params,
                manifest=manifest,
                cleaning_code=cleaning_code,
            )
            if not bool(evidence.get("applies_if", True)):
                warnings.append(
                    "nps_forward_fill_temporal_integrity skipped: "
                    + str(evidence.get("skip_reason", "not_applicable"))
                )
        elif gate_key == "enforce_temporal_training_mask" or "temporal_ceiling" in gate_key:
            issues, evidence, training_rows_context = _check_enforce_temporal_training_mask(
                cleaned_csv_path=cleaned_csv_path,
                cleaned_header=cleaned_header,
                params=params,
                column_roles=column_roles,
            )
            if not bool(evidence.get("applies_if", True)):
                warnings.append(
                    f"enforce_temporal_training_mask skipped: {evidence.get('skip_reason', 'not_applicable')}"
                )
        elif gate_key == "outlier_policy_applied":
            issues, evidence = _check_outlier_policy_applied(
                outlier_policy=outlier_policy,
                outlier_report=outlier_report,
                outlier_report_path=outlier_report_path or params.get("report_path"),
                manifest=manifest,
                params=params,
            )
        else:
            evaluated = False
            evidence["deterministic_support"] = "not_implemented"

        passed = None
        if evaluated:
            passed = not issues
        gate_results.append(
            {
                "name": name,
                "severity": severity_used,
                "passed": passed,
                "issues": issues,
                "evidence": evidence or {"source": "deterministic"},
            }
        )

        if issues:
            _record_gate_failure(
                _normalize_gate_name(name),
                severity_used,
                issues,
                hard_failures,
                soft_failures,
                failed_checks,
                required_fixes,
                failure_summaries,
                warning_summaries,
            )

    status = "APPROVED"
    if hard_failures:
        status = "REJECTED"
    elif soft_failures:
        status = "APPROVE_WITH_WARNINGS"

    if hard_failures:
        feedback = "Cleaning reviewer rejected: " + " | ".join(failure_summaries)
    elif soft_failures:
        feedback = "Cleaning reviewer approved with warnings: " + " | ".join(warning_summaries)
    else:
        feedback = "Cleaning reviewer approved: all gates passed."

    if warning_summaries:
        warnings.extend(warning_summaries)

    return {
        "status": status,
        "feedback": feedback,
        "failed_checks": failed_checks,
        "required_fixes": required_fixes,
        "warnings": warnings,
        "hard_failures": hard_failures,
        "soft_failures": soft_failures,
        "gate_results": gate_results,
    }


def _assemble_result(
    deterministic: Dict[str, Any],
    gate_names: List[str],
    warnings: List[str],
    contract_source_used: str,
) -> Dict[str, Any]:
    result = dict(deterministic)
    result["warnings"] = _dedupe_list(list(warnings) + result.get("warnings", []))
    result["cleaning_gates_evaluated"] = gate_names
    result["contract_source_used"] = contract_source_used
    result.setdefault("gate_results", [])
    result.setdefault("hard_failures", [])
    result.setdefault("soft_failures", [])
    return result


def _build_cleaning_quality_summary(
    cleaned_csv_path: str,
    raw_csv_path: Optional[str],
    dialect_cleaned: Dict[str, Any],
    dialect_raw: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Build a per-column null-rate comparison (raw vs cleaned) for LLM review.

    Returns a compact dict the LLM can use to detect null inflation,
    datetime parsing failures, and unexpected data loss.
    """
    try:
        sep_c = dialect_cleaned.get("sep", ",")
        enc_c = dialect_cleaned.get("encoding", "utf-8")
        cleaned = pd.read_csv(cleaned_csv_path, dtype=str, sep=sep_c, encoding=enc_c, nrows=2000)
    except Exception:
        return None

    raw = None
    if raw_csv_path:
        try:
            sep_r = (dialect_raw or {}).get("sep", ",")
            enc_r = (dialect_raw or {}).get("encoding", "utf-8")
            raw = pd.read_csv(raw_csv_path, dtype=str, sep=sep_r, encoding=enc_r, nrows=2000)
        except Exception:
            pass

    summary: Dict[str, Any] = {"cleaned_rows": len(cleaned), "columns": {}}
    if raw is not None:
        summary["raw_rows"] = len(raw)
        if len(raw) > 0:
            summary["row_drop_pct"] = round(
                (1 - len(cleaned) / len(raw)) * 100, 1
            )

    for col in cleaned.columns:
        is_null = cleaned[col].isna() | (cleaned[col].str.strip() == "")
        cleaned_null_pct = round(float(is_null.sum()) / max(len(cleaned), 1) * 100, 1)
        col_info: Dict[str, Any] = {"cleaned_null_pct": cleaned_null_pct}

        if raw is not None and col in raw.columns:
            raw_null = raw[col].isna() | (raw[col].str.strip() == "")
            raw_null_pct = round(float(raw_null.sum()) / max(len(raw), 1) * 100, 1)
            col_info["raw_null_pct"] = raw_null_pct
            col_info["null_inflation_pp"] = round(cleaned_null_pct - raw_null_pct, 1)

        # Detect datetime columns that became mostly NaT
        non_null = cleaned[col].dropna()
        if len(non_null) > 0:
            sample_vals = non_null.head(20).tolist()
            nat_like = sum(1 for v in sample_vals if str(v).strip().lower() in ("nat", "none", "nan", ""))
            if nat_like > len(sample_vals) * 0.5 and cleaned_null_pct > 50:
                col_info["warning"] = "possible_datetime_parsing_failure"

        summary["columns"][col] = col_info

    # Keep only columns with notable null inflation or high null rates
    notable = {}
    for col, info in summary["columns"].items():
        if info.get("null_inflation_pp", 0) > 5 or info.get("cleaned_null_pct", 0) > 30:
            notable[col] = info
    if notable:
        summary["notable_columns"] = notable

    # Cap total payload size: only include notable columns in the final output
    if len(summary["columns"]) > 30:
        summary["columns"] = {k: v for k, v in list(summary["columns"].items())[:30]}
        summary["columns_truncated"] = True

    return summary


def _build_llm_prompt(
    gates: List[Dict[str, Any]],
    required_columns: List[str],
    dialect: Dict[str, Any],
    column_roles: Dict[str, List[str]],
    facts: Dict[str, Any],
    deterministic_gate_results: List[Dict[str, Any]],
    contract_source_used: str,
    context_pack: Optional[str] = None,
    cleaning_code: Optional[str] = None,
    dataset_profile: Optional[Dict[str, Any]] = None,
    column_resolution_context: Optional[Dict[str, Any]] = None,
    artifact_obligations: Optional[Dict[str, Any]] = None,
    cleaning_quality_summary: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    system_prompt = (
        "You are a Senior Data Scientist reviewing data cleaning output.\n\n"
        "MISSION\n"
        "- Evaluate whether the delivered cleaning work satisfies the requested cleaning contract for THIS run.\n"
        "- Use contextual reasoning grounded in contract, data evidence, and executed code.\n"
        "- Do not apply rigid heuristics when the context does not justify them.\n\n"
        "SOURCE OF TRUTH AND PRECEDENCE\n"
        "1. cleaning_gates + required_columns + column_roles + dialect + contract_source_used in the payload are authoritative review scope.\n"
        "2. artifact_obligations + column_resolution_context + cleaning_code + facts + data_profile + cleaning_quality_summary are primary evidence of what the Data Engineer actually did and what the data looked like.\n"
        "3. deterministic_gate_results are authoritative when they directly evaluate a contract gate from artifact evidence. Use them as the default resolution unless you find explicit contradictory evidence in cleaning_code, facts, or artifact_obligations.\n"
        "4. artifact_obligations is a lossless extraction of artifact bindings already declared in the contract. It introduces no new semantics.\n"
        "5. If sources conflict, preserve contract intent and prefer direct evidence from artifact_obligations/column_resolution_context/cleaning_code/data_profile over shallow pattern matching.\n\n"
        "REVIEW REASONING\n"
        "Before deciding on each gate, reason through:\n"
        "- What does the contract require? What gates were requested, what columns matter, what would count as a real violation?\n"
        "- What does the evidence show? Read artifact_obligations, facts, column_resolution_context, data_profile, cleaning_quality_summary, cleaning_code, and deterministic_gate_results.\n"
        "- For each gate, reason about the gate's intent AND its params scope. If a gate specifies params.column, ground your evaluation in that column — do not fail a gate based on evidence from a different column.\n"
        "- Reject only when you have contract-relevant evidence of a real violation.\n"
        "- If evidence is ambiguous or incomplete, prefer PASSED or PASSED_WITH_WARNING over unsupported failure.\n"
        "- Keep feedback tied to this dataset and this code path, not generic cleaning advice.\n\n"
        "HIGH-RISK EXAMPLES (GUIDANCE, NOT A SUBSTITUTE FOR REASONING)\n"
        "- no_semantic_rescale: do not infer rescaling from low numeric ranges alone. Look for explicit code evidence such as division by constants, scaler objects, or explicit multiplicative rescaling. If data is already in a low range but no such code exists, that is not a violation.\n"
        "- no_synthetic_data: distinguish between generating synthetic rows/datasets and limited stochastic operations such as noise or imputation support. Reject only when the code clearly fabricates data beyond the contract.\n\n"
        "- leakage_exclusion: reason from direct evidence of the cleaned artifact. If forbidden columns are absent from cleaned_header and/or explicitly listed as dropped in the manifest, that supports PASS. Do not treat missing_required_columns as evidence that forbidden columns are still present.\n\n"
        "- boolean_normalization on CSV artifacts: do not fail merely because a reloaded sample shows int64 instead of int8. Width-specific integer dtypes are not stable through CSV round-trips; judge this gate by value semantics (0/1/null) unless the contract explicitly requires a binary file format preserving dtype width.\n\n"
        "NULL INFLATION DETECTION (CRITICAL)\n"
        "- The payload may include 'cleaning_quality_summary' with per-column null rates before (raw) and after (cleaned) cleaning.\n"
        "- 'null_inflation_pp' = cleaned_null_pct − raw_null_pct. It measures how much the null rate increased, but it does NOT distinguish between two very different causes:\n"
        "  (a) ROW EXCLUSION: the DE dropped rows where a column had non-null values. This legitimately increases null_pct for that column in the surviving rows. Check: if row_drop_pct > 0 and the gate contract explicitly requests row exclusion on that column (e.g., exclude_debug_records), the inflation is expected — not a failure.\n"
        "  (b) VALUE DESTRUCTION: a broken parser or incorrect transformation converted valid values to null/NaT within the same rows. This is a real failure.\n"
        "- To distinguish (a) from (b): compare row_drop_pct with null_inflation_pp for the column in question. If the dataset lost rows AND the inflated column is the exclusion criterion (or strongly correlated with it), the inflation is a natural consequence of correct row filtering — not data destruction. Also check the cleaning_code for evidence of df.drop/filtering vs value-level nulling.\n"
        "- For datetime columns: check 'notable_columns' for 'possible_datetime_parsing_failure' warnings. A broken parser that massively inflates nulls is a HARD failure — require multi-stage parsing. But first verify the gate's params.column scope: only evaluate the column specified in the gate definition, not all date columns.\n"
        "- When assessing null inflation magnitude, reason about proportionality: placeholder cleanup or impossible-date quarantine produces moderate inflation, while parser destruction produces extreme inflation. Use the raw vs cleaned null rates and the cleaning_code evidence to distinguish.\n\n"
        "DECISION RULES\n"
        "- If any HARD gate fails, status must be REJECTED.\n"
        "- If only SOFT gates fail, status must be APPROVE_WITH_WARNINGS.\n"
        "- If no gates fail, status must be APPROVED.\n"
        "- When evidence is insufficient, mark the gate as PASSED and add a warning.\n"
        "- Do not reject based on data ranges alone; you must find contract-relevant evidence.\n\n"
        "EVIDENCE REQUIREMENT\n"
        "- Any REJECT must cite specific evidence from cleaning_code, facts, data_profile, or cleaning_quality_summary.\n"
        "- Format: EVIDENCE: <source>#<detail> -> <snippet>\n"
        "- Example: EVIDENCE: cleaning_code#line42 -> df['pixel'] = df['pixel'] / 255\n"
        "- If you cannot find concrete evidence, do not reject; explain the uncertainty as a warning.\n\n"
        "OUTPUT\n"
        "- Return JSON only with the specified schema.\n\n"
        "Required JSON schema:\n"
        "{\n"
        '  "status": "APPROVED" | "APPROVE_WITH_WARNINGS" | "REJECTED",\n'
        '  "feedback": "string",\n'
        '  "failed_checks": ["gate_name", ...],\n'
        '  "required_fixes": ["actionable fix", ...],\n'
        '  "warnings": ["warning", ...],\n'
        '  "hard_failures": ["gate_name", ...],\n'
        '  "soft_failures": ["gate_name", ...],\n'
        '  "gate_results": [\n'
        "     {\n"
        '       "name": "gate_name",\n'
        '       "severity": "HARD|SOFT",\n'
        '       "passed": true|false,\n'
        '       "issues": ["issue", ...],\n'
        '       "evidence": "specific code line or data evidence"\n'
        "     }\n"
        "  ],\n"
        '  "contract_source_used": "cleaning_view|fallback|merged"\n'
        "}\n"
    )

    payload = {
        "cleaning_gates": gates,
        "required_columns": required_columns,
        "dialect": dialect,
        "column_roles": column_roles,
        "facts": facts,
        "deterministic_gate_results": deterministic_gate_results,
        "contract_source_used": contract_source_used,
    }
    if context_pack:
        payload["context_pack"] = context_pack
    if column_resolution_context and isinstance(column_resolution_context, dict):
        payload["column_resolution_context"] = column_resolution_context
    if artifact_obligations and isinstance(artifact_obligations, dict):
        payload["artifact_obligations"] = artifact_obligations

    # CONTEXT TRIPLET for LLM reasoning
    # 1. Cleaning Code - what did the engineer actually execute?
    if cleaning_code:
        # Preserve both beginning and ending context for long scripts.
        max_code_len = 6000
        if len(cleaning_code) > max_code_len:
            half = max_code_len // 2
            head = cleaning_code[:half]
            tail = cleaning_code[-half:]
            payload["cleaning_code"] = (
                head
                + "\n... [TRUNCATED_MIDDLE] ...\n"
                + tail
            )
        else:
            payload["cleaning_code"] = cleaning_code

    # 2. Data Profile - what are the actual data characteristics?
    if dataset_profile and isinstance(dataset_profile, dict):
        profile_summary = {}
        if "numeric_summary" in dataset_profile:
            profile_summary["numeric_ranges"] = dataset_profile["numeric_summary"]
        if "column_types" in dataset_profile:
            profile_summary["column_types"] = dataset_profile["column_types"]
        if "basic_stats" in dataset_profile:
            profile_summary["basic_stats"] = dataset_profile["basic_stats"]
        # Include null rates and datetime parse info for quality assessment
        if "null_rates" in dataset_profile:
            profile_summary["null_rates"] = dataset_profile["null_rates"]
        if "datetime_columns" in dataset_profile:
            profile_summary["datetime_columns"] = dataset_profile["datetime_columns"]
        if "column_profiles" in dataset_profile:
            profile_summary["column_profiles"] = dataset_profile["column_profiles"]
        if profile_summary:
            payload["data_profile"] = profile_summary

    # 3. Cleaning Quality Summary - raw vs cleaned null rates comparison
    if cleaning_quality_summary and isinstance(cleaning_quality_summary, dict):
        payload["cleaning_quality_summary"] = cleaning_quality_summary

    return system_prompt, payload


def _parse_llm_json(content: str) -> Optional[Dict[str, Any]]:
    if not isinstance(content, str):
        return None
    cleaned = extract_code_block(content)
    try:
        parsed = json.loads(cleaned)
    except Exception:
        return None
    expected_keys = {
        "status",
        "feedback",
        "failed_checks",
        "required_fixes",
        "warnings",
        "hard_failures",
        "soft_failures",
        "gate_results",
        "contract_source_used",
    }

    def _is_reviewer_payload(obj: Any) -> bool:
        return isinstance(obj, dict) and bool(expected_keys.intersection(set(obj.keys())))

    if _is_reviewer_payload(parsed):
        return parsed

    if isinstance(parsed, list):
        for item in parsed:
            if _is_reviewer_payload(item):
                return item
        return None

    if isinstance(parsed, dict):
        for key in ("result", "review", "data", "payload", "response", "output"):
            nested = parsed.get(key)
            if _is_reviewer_payload(nested):
                return nested
            if isinstance(nested, list):
                for item in nested:
                    if _is_reviewer_payload(item):
                        return item
    return None


def _collect_unresolved_hard_gates(result: Dict[str, Any]) -> List[str]:
    unresolved: List[str] = []
    seen: set[str] = set()
    gate_results = result.get("gate_results")
    if not isinstance(gate_results, list):
        return unresolved
    for gate in gate_results:
        if not isinstance(gate, dict):
            continue
        severity = str(gate.get("severity", "HARD")).strip().upper()
        passed = gate.get("passed")
        if severity != "HARD" or passed is not None:
            continue
        name = str(gate.get("name") or "").strip() or "unknown_gate"
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        unresolved.append(name)
    return unresolved


def _enforce_fail_closed_when_llm_unavailable(result: Dict[str, Any], reason: str) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return result
    unresolved_hard = _collect_unresolved_hard_gates(result)
    if not unresolved_hard:
        return result

    warnings = result.get("warnings")
    if not isinstance(warnings, list):
        warnings = []
    if reason and reason not in warnings:
        warnings.append(reason)
    unresolved_note = f"{_UNEVALUATED_HARD_GATES_WARNING}: {', '.join(unresolved_hard)}"
    if unresolved_note not in warnings:
        warnings.append(unresolved_note)
    if _LLM_FAIL_CLOSED_REASON not in warnings:
        warnings.append(_LLM_FAIL_CLOSED_REASON)
    result["warnings"] = warnings

    hard_failures = result.get("hard_failures")
    if not isinstance(hard_failures, list):
        hard_failures = []
    for gate_name in unresolved_hard:
        if gate_name not in hard_failures:
            hard_failures.append(gate_name)
    result["hard_failures"] = hard_failures

    failed_checks = result.get("failed_checks")
    if not isinstance(failed_checks, list):
        failed_checks = []
    for gate_name in unresolved_hard:
        if gate_name not in failed_checks:
            failed_checks.append(gate_name)
    result["failed_checks"] = failed_checks

    required_fixes = result.get("required_fixes")
    if not isinstance(required_fixes, list):
        required_fixes = []
    fix_msg = (
        "Cleaning reviewer could not evaluate all HARD gates because LLM output was unavailable/invalid. "
        "Retry reviewer evaluation and ensure valid JSON response."
    )
    if fix_msg not in required_fixes:
        required_fixes.insert(0, fix_msg)
    result["required_fixes"] = required_fixes

    result["status"] = "REJECTED"
    feedback = str(result.get("feedback") or "").strip()
    reject_prefix = (
        "Cleaning reviewer rejected: unresolved HARD gates due to unavailable/invalid LLM review "
        f"({', '.join(unresolved_hard)})."
    )
    if reject_prefix not in feedback:
        result["feedback"] = f"{reject_prefix} {feedback}".strip()
    return result


def _merge_llm_with_deterministic(
    llm_result: Dict[str, Any],
    deterministic: Dict[str, Any],
    gate_names: List[str],
    contract_source_used: str,
    warnings: List[str],
) -> Dict[str, Any]:
    llm = _normalize_llm_result(llm_result)
    det = deterministic

    gate_results = _merge_gate_results(
        llm.get("gate_results", []),
        det.get("gate_results", []),
        gate_names,
    )
    summary = _summarize_gate_results(gate_results)
    merged_warnings = _dedupe_list(
        warnings + det.get("warnings", []) + llm.get("warnings", []) + summary["warning_summaries"]
    )
    required_fixes = _dedupe_list(summary["required_fixes"] + det.get("required_fixes", []))

    return {
        "status": summary["status"],
        "feedback": summary["feedback"],
        "failed_checks": summary["failed_checks"],
        "required_fixes": required_fixes,
        "warnings": merged_warnings,
        "hard_failures": summary["hard_failures"],
        "soft_failures": summary["soft_failures"],
        "gate_results": gate_results,
        "cleaning_gates_evaluated": gate_names,
        "contract_source_used": contract_source_used,
    }


def _normalize_llm_result(result: Dict[str, Any]) -> Dict[str, Any]:
    out = result if isinstance(result, dict) else {}
    out.setdefault("failed_checks", [])
    out.setdefault("required_fixes", [])
    out.setdefault("warnings", [])
    out.setdefault("hard_failures", [])
    out.setdefault("soft_failures", [])
    out.setdefault("gate_results", [])
    return out


def _merge_gate_results(
    llm_results: List[Any],
    det_results: List[Any],
    gate_names: List[str],
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for entry in det_results:
        if not isinstance(entry, dict):
            continue
        name = _normalize_gate_name(entry.get("name", ""))
        if not name:
            continue
        normalized = dict(entry)
        normalized["name"] = name
        merged[name] = normalized
    for entry in llm_results:
        if not isinstance(entry, dict):
            continue
        name = _normalize_gate_name(entry.get("name", ""))
        if not name:
            continue
        existing = merged.get(name)
        if existing and existing.get("passed") is not None:
            if not existing.get("evidence") and entry.get("evidence"):
                existing["evidence"] = entry.get("evidence")
            continue
        if existing:
            updated = dict(existing)
            updated.update(entry)
        else:
            updated = dict(entry)
        updated["name"] = name
        merged[name] = updated
    ordered = []
    for gate in gate_names:
        key = _normalize_gate_name(gate)
        if key in merged:
            entry = merged[key]
            entry["name"] = gate
            ordered.append(entry)
        else:
            ordered.append(
                {
                    "name": gate,
                    "severity": "HARD",
                    "passed": None,
                    "issues": [],
                    "evidence": "no_result",
                }
            )
    return ordered


def _summarize_gate_results(gate_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    hard_failures: List[str] = []
    soft_failures: List[str] = []
    failed_checks: List[str] = []
    required_fixes: List[str] = []
    failure_summaries: List[str] = []
    warning_summaries: List[str] = []

    for gate in gate_results:
        name = str(gate.get("name", "")).strip()
        passed = gate.get("passed")
        if passed is not False:
            continue
        severity = str(gate.get("severity", "HARD")).strip().upper()
        if severity not in {"HARD", "SOFT"}:
            severity = "HARD"
        issues = gate.get("issues") or []
        issues_text = "; ".join(str(issue) for issue in issues if issue)
        summary = f"{name}: {issues_text}" if issues_text else f"{name}: failed"
        if name and name not in failed_checks:
            failed_checks.append(name)
        if severity == "HARD":
            if name and name not in hard_failures:
                hard_failures.append(name)
            required_fixes.append(summary)
            failure_summaries.append(summary)
        else:
            if name and name not in soft_failures:
                soft_failures.append(name)
            warning_summaries.append(summary)

    status = "APPROVED"
    if hard_failures:
        status = "REJECTED"
    elif soft_failures:
        status = "APPROVE_WITH_WARNINGS"

    if hard_failures:
        feedback = "Cleaning reviewer rejected: " + " | ".join(failure_summaries)
    elif soft_failures:
        feedback = "Cleaning reviewer approved with warnings: " + " | ".join(warning_summaries)
    else:
        feedback = "Cleaning reviewer approved: all gates passed."

    return {
        "status": status,
        "feedback": feedback,
        "failed_checks": failed_checks,
        "required_fixes": required_fixes,
        "hard_failures": hard_failures,
        "soft_failures": soft_failures,
        "warning_summaries": warning_summaries,
    }


def _dedupe_list(items: List[Any]) -> List[str]:
    seen: set[str] = set()
    deduped: List[str] = []
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(text)
    return deduped


def _record_gate_failure(
    gate_name: str,
    severity: str,
    issues: List[str],
    hard_failures: List[str],
    soft_failures: List[str],
    failed_checks: List[str],
    required_fixes: List[str],
    failure_summaries: List[str],
    warning_summaries: List[str],
) -> None:
    summary = f"{gate_name}: " + "; ".join(issues)
    if gate_name not in failed_checks:
        failed_checks.append(gate_name)
    if severity == "HARD":
        if gate_name not in hard_failures:
            hard_failures.append(gate_name)
        required_fixes.append(summary)
        failure_summaries.append(summary)
    else:
        if gate_name not in soft_failures:
            soft_failures.append(gate_name)
        warning_summaries.append(summary)


def _check_required_columns(
    required_columns: List[str],
    cleaned_header: List[str],
    cleaned_csv_path: str,
) -> List[str]:
    if not required_columns:
        return []
    if not cleaned_header:
        return [f"Unable to read cleaned CSV header: {cleaned_csv_path}"]
    missing = [col for col in required_columns if col not in cleaned_header]
    if missing:
        return [f"Missing required columns: {', '.join(missing)}"]
    return []


def _detect_explicit_rescale_patterns(cleaning_code: Optional[str]) -> List[str]:
    if not isinstance(cleaning_code, str) or not cleaning_code.strip():
        return []
    lowered = cleaning_code.lower()
    patterns: List[tuple[str, str]] = [
        (r"/\s*255(\.0+)?\b", "divide_by_255"),
        (r"/\s*100(\.0+)?\b", "divide_by_100"),
        (r"\*\s*0\.00392\b", "multiply_0_00392"),
        (r"\bminmaxscaler\s*\(", "minmax_scaler"),
        (r"\bstandardscaler\s*\(", "standard_scaler"),
        (r"\brobustscaler\s*\(", "robust_scaler"),
        (r"\bnormalize\s*\(", "normalize_call"),
    ]
    found: List[str] = []
    for pattern, label in patterns:
        try:
            if re.search(pattern, lowered):
                found.append(label)
        except Exception:
            continue
    return found


def _check_id_integrity(
    cleaned_header: List[str],
    sample_str: Optional[pd.DataFrame],
    sample_infer: Optional[pd.DataFrame],
    params: Dict[str, Any],
    column_roles: Dict[str, List[str]],
) -> Tuple[List[str], Dict[str, Any]]:
    candidates, evidence = _resolve_id_integrity_candidates(cleaned_header, params, column_roles)
    if not candidates:
        return [], evidence

    detect_sci = bool(params.get("detect_scientific_notation", True))
    sci_threshold = float(params.get("scientific_notation_ratio_threshold", 0.02))
    dot0_threshold = float(params.get("dot_zero_ratio_threshold", 0.1))
    min_samples = int(params.get("min_samples", 20))

    issues: List[str] = []
    column_evidence: Dict[str, Any] = {}
    for col in candidates:
        values = _string_values(sample_str, col)
        if len(values) < min_samples:
            column_evidence[col] = {"samples": len(values), "skipped": "insufficient_samples"}
            continue
        sci_count = 0
        dot0_count = 0
        for val in values:
            lowered = val.lower()
            if detect_sci and ("e+" in lowered or "e-" in lowered):
                sci_count += 1
            if re.search(r"\.0+$", val):
                dot0_count += 1
        total = len(values)
        col_evidence: Dict[str, Any] = {
            "samples": total,
            "scientific_notation_count": sci_count,
            "dot_zero_count": dot0_count,
            "detect_scientific_notation": detect_sci,
            "scientific_notation_ratio_threshold": sci_threshold,
            "dot_zero_ratio_threshold": dot0_threshold,
        }
        if detect_sci and sci_count / total >= sci_threshold:
            issues.append(f"{col} contains scientific notation ({sci_count}/{total})")
        if dot0_count / total >= dot0_threshold:
            issues.append(f"{col} coerced to float-like values ({dot0_count}/{total} end with .0)")
        if sample_infer is not None and col in sample_infer.columns:
            if pd.api.types.is_float_dtype(sample_infer[col]):
                issues.append(f"{col} inferred as float dtype in cleaned data")
                col_evidence["inferred_dtype"] = str(sample_infer[col].dtype)
        column_evidence[col] = col_evidence

    evidence["columns_checked"] = candidates
    evidence["column_evidence"] = column_evidence
    return issues, evidence


def _check_no_synthetic_data(manifest: Dict[str, Any], cleaning_code: Optional[str] = None) -> List[str]:
    """
    Check for synthetic data generation in the cleaning process.

    Args:
        manifest: Cleaning manifest with warnings
        cleaning_code: Optional DE Python code for direct inspection (avoids file read)
    """
    issues: List[str] = []
    warnings = manifest.get("warnings") if isinstance(manifest, dict) else []
    if isinstance(warnings, list):
        for warning in warnings:
            if "synthetic" in str(warning).lower():
                issues.append("Manifest reports synthetic data usage")
                break

    # Use provided cleaning_code if available, else fall back to file
    code_to_check = cleaning_code
    if not code_to_check:
        path = os.path.join("artifacts", "data_engineer_last.py")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    code_to_check = handle.read()
            except Exception:
                code_to_check = ""

    if code_to_check:
        stripped = extract_code_block(code_to_check)
        text = stripped if stripped.strip() else code_to_check
        if _detect_synthetic_patterns(text):
            issues.append("Cleaning script appears to generate synthetic data")
    return issues


def _detect_synthetic_patterns(code: str) -> bool:
    lowered = code.lower()
    if "faker" in lowered or "make_classification" in lowered or "make_regression" in lowered:
        return True
    if "sklearn.datasets.make_" in lowered:
        return True
    if re.search(r"pd\.dataframe\([^)]*np\.random", lowered, re.DOTALL):
        return True
    if re.search(r"df\[['\"][^'\"]+['\"]\]\s*=\s*.*np\.random", lowered):
        return True
    return False


def _check_row_count_sanity(manifest: Dict[str, Any], params: Dict[str, Any]) -> List[str]:
    if not isinstance(manifest, dict):
        return []
    rows_before = manifest.get("rows_before")
    rows_after = manifest.get("rows_after")
    row_counts = manifest.get("row_counts") or {}
    if rows_before is None:
        for key in ("initial", "original", "input", "rows_before", "total"):
            if rows_before is None:
                rows_before = row_counts.get(key)
    if rows_after is None:
        for key in ("final", "after_cleaning", "output", "rows_after"):
            if rows_after is None:
                rows_after = row_counts.get(key)
    if not isinstance(rows_before, (int, float)) or not isinstance(rows_after, (int, float)):
        return []
    if rows_before <= 0:
        return []
    max_drop_pct = float(params.get("max_drop_pct", 5.0))
    max_dup_increase_pct = float(params.get("max_dup_increase_pct", 1.0))
    issues: List[str] = []
    if rows_after <= rows_before:
        drop_pct = (rows_before - rows_after) / rows_before * 100.0
        if drop_pct > max_drop_pct:
            issues.append(f"Row drop {drop_pct:.2f}% exceeds {max_drop_pct:.2f}%")
    else:
        increase_pct = (rows_after - rows_before) / rows_before * 100.0
        if increase_pct > max_dup_increase_pct:
            issues.append(f"Row increase {increase_pct:.2f}% exceeds {max_dup_increase_pct:.2f}%")
    return issues


def _check_feature_coverage_sanity(
    cleaned_header: List[str],
    required_columns: List[str],
    column_roles: Dict[str, List[str]],
    model_features: List[str],
    allowed_feature_sets: Any,
    dataset_profile: Optional[Dict[str, Any]],
    params: Dict[str, Any],
) -> Tuple[List[str], Dict[str, Any]]:
    issues: List[str] = []
    evidence: Dict[str, Any] = {}

    min_feature_count = params.get("min_feature_count", 3)
    try:
        min_feature_count = int(min_feature_count)
    except Exception:
        min_feature_count = 3
    min_feature_count = max(0, min_feature_count)

    check_against = str(params.get("check_against") or "data_atlas").strip().lower()
    min_feature_ratio = params.get("min_feature_ratio", 0.2)
    try:
        min_feature_ratio = float(min_feature_ratio)
    except Exception:
        min_feature_ratio = 0.2
    min_feature_ratio = max(0.0, min(1.0, min_feature_ratio))

    structural_cols: List[str] = []
    structural_cols.extend(_columns_with_role_tokens(column_roles, {"id", "identifier", "key"}))
    structural_cols.extend(_columns_with_role_tokens(column_roles, {"split", "partition", "fold"}))
    structural_cols.extend(_columns_with_role_tokens(column_roles, {"target", "label", "outcome"}))
    structural_cols.extend(_columns_with_role_tokens(column_roles, {"time", "timestamp", "date"}))

    dataset_semantics = (
        dataset_profile.get("dataset_semantics")
        if isinstance(dataset_profile, dict) and isinstance(dataset_profile.get("dataset_semantics"), dict)
        else {}
    )
    if isinstance(dataset_semantics, dict):
        for key in ("split_candidates", "id_candidates", "identifier_columns"):
            structural_cols.extend(_list_str(dataset_semantics.get(key)))
        structural_cols.extend(_list_str(dataset_semantics.get("primary_target")))
        structural_cols.extend(_list_str(dataset_semantics.get("target_columns")))

    structural_cols.extend(_list_str(params.get("target_column")))
    structural_cols.extend(_list_str(params.get("target_columns")))
    structural_cols.extend(_list_str(params.get("id_columns")))
    structural_cols.extend(_list_str(params.get("split_columns")))

    structural_norm = {str(col).strip().lower() for col in structural_cols if str(col).strip()}
    cleaned_feature_cols = [
        col for col in (cleaned_header or [])
        if str(col).strip() and str(col).strip().lower() not in structural_norm
    ]

    model_features = _list_str(model_features)
    if not model_features and isinstance(allowed_feature_sets, dict):
        model_features = _list_str(allowed_feature_sets.get("model_features"))
    expected_model_features = [
        col for col in model_features
        if str(col).strip() and str(col).strip().lower() not in structural_norm
    ]

    source_columns: List[str] = []
    if check_against in {"data_atlas", "column_inventory"}:
        source_columns = _load_column_inventory_names("data/column_inventory.json")
        if not source_columns and isinstance(dataset_profile, dict):
            source_columns = _list_str(dataset_profile.get("column_inventory"))
            if not source_columns:
                source_columns = _list_str(dataset_profile.get("columns"))

    if not source_columns:
        source_columns = _list_str(required_columns)
        if not source_columns and model_features:
            source_columns = list(model_features)

    source_feature_cols = [
        col for col in source_columns
        if str(col).strip() and str(col).strip().lower() not in structural_norm
    ]

    source_feature_count = len({str(col).strip().lower() for col in source_feature_cols})
    cleaned_feature_count = len({str(col).strip().lower() for col in cleaned_feature_cols})
    expected_model_count = len({str(col).strip().lower() for col in expected_model_features})

    min_candidates: List[int] = []
    if min_feature_count > 0:
        min_candidates.append(min_feature_count)
    if source_feature_count > 0:
        min_candidates.append(source_feature_count)
    if expected_model_count > 0:
        min_candidates.append(expected_model_count)
    min_required = min(min_candidates) if min_candidates else 0

    evidence.update(
        {
            "check_against": check_against,
            "min_feature_count": min_feature_count,
            "min_required_effective": min_required,
            "cleaned_feature_count": cleaned_feature_count,
            "cleaned_feature_sample": cleaned_feature_cols[:25],
            "source_feature_count": source_feature_count,
            "source_feature_sample": source_feature_cols[:25],
            "expected_model_feature_count": expected_model_count,
            "expected_model_feature_sample": expected_model_features[:25],
            "structural_columns_sample": sorted(structural_norm)[:25],
        }
    )

    if min_required > 0 and cleaned_feature_count < min_required:
        issues.append(
            f"cleaned dataset has {cleaned_feature_count} non-structural features; expected at least {min_required}"
        )

    if source_feature_count > 0:
        coverage_ratio = float(cleaned_feature_count / max(1, source_feature_count))
        evidence["feature_coverage_ratio"] = round(coverage_ratio, 4)
        if (
            cleaned_feature_count > 0
            and cleaned_feature_count < source_feature_count
            and coverage_ratio < min_feature_ratio
        ):
            issues.append(
                "cleaned feature coverage ratio "
                f"{coverage_ratio:.2f} is below minimum {min_feature_ratio:.2f} against source feature inventory"
            )
    else:
        evidence["feature_coverage_ratio"] = None
        evidence["skip_reason"] = "source_feature_inventory_unavailable"

    return issues, evidence


def _check_outlier_policy_applied(
    outlier_policy: Dict[str, Any],
    outlier_report: Dict[str, Any],
    outlier_report_path: Optional[str],
    manifest: Dict[str, Any],
    params: Dict[str, Any],
) -> Tuple[List[str], Dict[str, Any]]:
    issues: List[str] = []
    evidence: Dict[str, Any] = {}

    policy = outlier_policy if isinstance(outlier_policy, dict) else {}
    enabled = _outlier_policy_enabled(policy)
    apply_stage = str(policy.get("apply_stage") or "data_engineer").strip().lower()
    evidence["policy_enabled"] = enabled
    evidence["apply_stage"] = apply_stage
    evidence["report_path"] = outlier_report_path

    if not enabled:
        evidence["applies_if"] = False
        evidence["skip_reason"] = "policy_disabled_or_missing"
        return [], evidence

    if apply_stage not in {"", "data_engineer", "both"}:
        evidence["applies_if"] = False
        evidence["skip_reason"] = "policy_not_assigned_to_data_engineer"
        return [], evidence

    strict = params.get("strict", policy.get("strict", True))
    if isinstance(strict, str):
        strict = strict.strip().lower() in {"1", "true", "yes", "on", "required"}
    strict = bool(strict)
    evidence["strict"] = strict

    report = outlier_report if isinstance(outlier_report, dict) else {}
    report_present = bool(report)
    evidence["report_present"] = report_present
    if outlier_report_path:
        evidence["report_file_exists"] = os.path.exists(outlier_report_path)

    manifest_outlier = (
        manifest.get("outlier_treatment")
        if isinstance(manifest, dict) and isinstance(manifest.get("outlier_treatment"), dict)
        else {}
    )
    if manifest_outlier:
        evidence["manifest_outlier_treatment"] = manifest_outlier

    if not report_present:
        if strict:
            issues.append("outlier_treatment_report_missing_or_empty")
        else:
            evidence["warning"] = "outlier_treatment_report_missing_or_empty"

    policy_targets = _list_str(policy.get("target_columns")) or _list_str(params.get("target_columns"))
    evidence["policy_target_columns"] = policy_targets

    report_columns = _extract_outlier_report_columns(report)
    evidence["report_columns_touched"] = report_columns

    if policy_targets:
        if not report_columns:
            if strict:
                issues.append("outlier_treatment_columns_missing_in_report")
        else:
            policy_norm = {str(col).strip().lower() for col in policy_targets if col}
            report_norm = {str(col).strip().lower() for col in report_columns if col}
            missing_targets = sorted([col for col in policy_targets if str(col).strip().lower() not in report_norm])
            evidence["missing_target_columns_in_report"] = missing_targets
            if missing_targets and strict:
                issues.append(
                    "outlier_treatment_report_missing_target_columns: "
                    + ", ".join(missing_targets[:10])
                )

    manifest_applied = None
    if manifest_outlier:
        for key in ("policy_applied", "enabled", "applied"):
            if key in manifest_outlier:
                raw = manifest_outlier.get(key)
                if isinstance(raw, str):
                    manifest_applied = raw.strip().lower() in {"1", "true", "yes", "on", "applied"}
                else:
                    manifest_applied = bool(raw)
                break
    evidence["manifest_policy_applied"] = manifest_applied
    if strict and manifest_outlier and manifest_applied is False:
        issues.append("manifest_outlier_treatment_policy_applied_false")

    return issues, evidence


def _extract_outlier_report_columns(report: Dict[str, Any]) -> List[str]:
    """Extract the list of columns treated in an outlier report.

    DE scripts produce outlier reports in several formats:
      - {"columns_touched": ["col1", ...]}              (canonical)
      - {"target_columns": ["col1", ...]}               (list variant)
      - {"target_columns": {"col1": {...}, ...}}         (dict-keyed variant)
      - {"targets": ["col1", ...] | {"col1": {...}}}    (alternative contract variant)
      - {"columns": ["col1", ...] | {"col1": {...}}}    (legacy)
      - {"applied": [{"column": "col1", ...}, ...]}     (per-column detail variant)
      - {"actions": [{"column": "col1", ...}, ...]}     (action log variant)
      - {"treatments": [{"column": "col1", ...}, ...]}  (treatment log variant)

    This function must handle all of them to avoid false-positive gate failures.
    """
    if not isinstance(report, dict):
        return []
    report_columns = _list_str(report.get("columns_touched"))
    if not report_columns:
        target_cols = report.get("target_columns")
        if isinstance(target_cols, dict):
            # dict-keyed variant: {"employees": {...}, "annual_revenue": {...}}
            report_columns = [str(key).strip() for key in target_cols.keys() if str(key).strip()]
        else:
            report_columns = _list_str(target_cols)
    if not report_columns:
        targets = report.get("targets")
        if isinstance(targets, dict):
            report_columns = [str(key).strip() for key in targets.keys() if str(key).strip()]
        else:
            report_columns = _list_str(targets)
    raw_columns = report.get("columns")
    if not report_columns and isinstance(raw_columns, list):
        report_columns = _list_str(raw_columns)
    if not report_columns and isinstance(raw_columns, dict):
        report_columns = [str(key).strip() for key in raw_columns.keys() if str(key).strip()]
    # per-column detail variant: {"applied": [{"column": "employees", ...}, ...]}
    # "actions" is a common DE-produced variant we must also recognize to avoid
    # false-positive rejections when the report lists treated columns as action
    # entries (e.g. [{"column": "arr_current", "method": "cap", ...}, ...]).
    if not report_columns:
        for list_key in ("applied", "actions", "treatments", "treatment_records", "operations"):
            entries = report.get(list_key)
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if isinstance(entry, dict):
                    col = str(
                        entry.get("column")
                        or entry.get("field")
                        or entry.get("feature")
                        or entry.get("name")
                        or ""
                    ).strip()
                    if col:
                        report_columns.append(col)
            if report_columns:
                break
    if not report_columns:
        for list_key in ("decisions", "columns_analyzed"):
            entries = report.get(list_key)
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if isinstance(entry, dict):
                    col = str(entry.get("column") or "").strip()
                    if col:
                        report_columns.append(col)
            if report_columns:
                break
    # flat per-column detail variant:
    # {"employees": {"capped_count": 12, ...}, "annual_revenue": {...}}
    if not report_columns:
        reserved = {
            "enabled",
            "report_version",
            "method",
            "percentile",
            "notes",
            "summary",
            "metadata",
            "columns_touched",
            "target_columns",
            "targets",
            "columns",
            "applied",
            "treatments",
            "treatment_records",
            "operations",
        }
        detail_markers = {
            "action",
            "clipped_count",
            "capped_count",
            "flagged_rows",
            "threshold",
            "threshold_99pct",
            "threshold_95pct",
            "lower_bound",
            "upper_bound",
            "winsorized_count",
            "iqr_multiplier",
            "method",
        }
        for key, value in report.items():
            token = str(key).strip()
            if not token or token.lower() in reserved or not isinstance(value, dict):
                continue
            value_keys = {str(k).strip().lower() for k in value.keys() if str(k).strip()}
            if value_keys & detail_markers:
                report_columns.append(token)
    deduped: List[str] = []
    seen: set[str] = set()
    for column in report_columns:
        token = str(column).strip()
        if not token:
            continue
        lowered = token.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(token)
    return deduped


_BOOLEAN_TRUE_TOKENS = {"1", "1.0", "true", "t", "yes", "y", "si", "sí", "on"}
_BOOLEAN_FALSE_TOKENS = {"0", "0.0", "false", "f", "no", "n", "off"}
_BOOLEAN_NULL_TOKENS = {"", "nan", "null", "none", "<na>", "na", "nat"}
_TRAIN_TRUE_TOKENS = {"1", "1.0", "true", "t", "yes", "y", "train", "training"}
_TRAIN_FALSE_TOKENS = {"0", "0.0", "false", "f", "no", "n", "test", "holdout", "score", "scoring", "val", "validation"}


def _normalize_boolean_value_token(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        if value is pd.NA:
            return None
    except Exception:
        pass
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in _BOOLEAN_NULL_TOKENS:
        return None
    if lowered in _BOOLEAN_TRUE_TOKENS:
        return "1"
    if lowered in _BOOLEAN_FALSE_TOKENS:
        return "0"
    return f"INVALID::{text}"


def _is_int_like_target_dtype(target_dtype: str) -> bool:
    token = str(target_dtype or "").strip().lower()
    return token.startswith("int") or token in {"integer", "nullable_int", "nullable_integer", "bool", "boolean"}


def _read_csv_unique_string_values(csv_path: str, columns: List[str]) -> Dict[str, List[str]]:
    requested = [str(col).strip() for col in columns if str(col).strip()]
    if not csv_path or not os.path.exists(csv_path) or not requested:
        return {col: [] for col in requested}

    encoding = _infer_encoding(csv_path)
    delimiter = _infer_delimiter_from_file(csv_path) or ","
    seen_by_column: Dict[str, List[str]] = {col: [] for col in requested}
    seen_tokens: Dict[str, set[str]] = {col: set() for col in requested}

    try:
        with open(csv_path, "r", encoding=encoding, errors="replace", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            if not reader.fieldnames:
                return seen_by_column
            available = {str(name).strip(): name for name in reader.fieldnames if isinstance(name, str)}
            effective = [col for col in requested if col in available]
            if not effective:
                return seen_by_column
            for row in reader:
                for col in effective:
                    raw = row.get(available[col])
                    if raw is None:
                        continue
                    text = str(raw).strip()
                    if not text or text.lower() in _BOOLEAN_NULL_TOKENS:
                        continue
                    lowered = text.lower()
                    if lowered in seen_tokens[col]:
                        continue
                    seen_tokens[col].add(lowered)
                    if len(seen_by_column[col]) < 20:
                        seen_by_column[col].append(text)
    except Exception:
        return {col: [] for col in requested}

    return seen_by_column


def _resolve_boolean_normalization_targets(params: Dict[str, Any]) -> set[str]:
    allowed: set[str] = set()
    raw_values = []
    if isinstance(params.get("target_values"), list):
        raw_values.extend(params.get("target_values") or [])
    if isinstance(params.get("allowed_values"), list):
        raw_values.extend(params.get("allowed_values") or [])
    for raw in raw_values:
        normalized = _normalize_boolean_value_token(raw)
        if normalized in {"0", "1"}:
            allowed.add(normalized)
    return allowed or {"0", "1"}


def _resolve_allowed_cleaned_boolean_tokens(params: Dict[str, Any]) -> set[str]:
    allowed: set[str] = set()
    raw_values: List[Any] = []
    if isinstance(params.get("target_values"), list):
        raw_values.extend(params.get("target_values") or [])
    if isinstance(params.get("allowed_values"), list):
        raw_values.extend(params.get("allowed_values") or [])
    for raw in raw_values:
        if raw is None:
            continue
        if isinstance(raw, bool):
            allowed.add("true" if raw else "false")
            continue
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            if float(raw) in {0.0, 1.0}:
                token = str(int(float(raw)))
                allowed.add(token)
                allowed.add(f"{token}.0")
            continue
        token = str(raw).strip().lower()
        if not token or token in _BOOLEAN_NULL_TOKENS:
            continue
        allowed.add(token)
    return allowed or {"0", "1", "0.0", "1.0"}


def _looks_boolean_like_observation(values: List[str]) -> bool:
    normalized_present = False
    for value in values:
        normalized = _normalize_boolean_value_token(value)
        if normalized in {"0", "1"}:
            normalized_present = True
            continue
        if normalized and normalized.startswith("INVALID::"):
            return False
    return normalized_present


def _resolve_boolean_normalization_candidates(
    cleaned_header: List[str],
    params: Dict[str, Any],
    column_dtype_targets: Dict[str, Any],
    raw_sample: Optional[pd.DataFrame],
    sample_str: Optional[pd.DataFrame],
) -> Tuple[List[str], Dict[str, Any]]:
    evidence: Dict[str, Any] = {"applies_if": True}
    header_lookup = {str(col).strip().lower(): str(col) for col in (cleaned_header or []) if str(col).strip()}

    explicit_candidates: List[str] = []
    for key in ("columns", "target_columns", "boolean_columns"):
        for column in _list_str(params.get(key)):
            canonical = header_lookup.get(str(column).strip().lower())
            if canonical and canonical not in explicit_candidates:
                explicit_candidates.append(canonical)

    raw_candidates: List[str] = []
    raw_observed_values: Dict[str, List[str]] = {}
    for sample in (raw_sample, sample_str):
        if sample is None:
            continue
        for column in sample.columns:
            values = _string_values(sample, str(column))
            if not values:
                continue
            if _looks_boolean_like_observation(values):
                canonical = header_lookup.get(str(column).strip().lower())
                if canonical:
                    if canonical not in raw_candidates:
                        raw_candidates.append(canonical)
                    raw_observed_values.setdefault(canonical, values[:12])

    dtype_candidates: List[str] = []
    for column, spec in column_dtype_targets.items():
        if not isinstance(spec, dict):
            continue
        canonical = header_lookup.get(str(column).strip().lower())
        if not canonical:
            continue
        target_dtype = str(spec.get("target_dtype") or "").strip().lower()
        if not _is_int_like_target_dtype(target_dtype):
            continue
        observed = raw_observed_values.get(canonical) or _string_values(sample_str, canonical)
        if observed:
            if _looks_boolean_like_observation(observed):
                dtype_candidates.append(canonical)
        else:
            role = str(spec.get("role") or "").strip().lower()
            if any(token in role for token in ("outcome", "decision", "pre_decision")):
                dtype_candidates.append(canonical)

    candidates: List[str] = []
    for source_columns in (explicit_candidates, raw_candidates, dtype_candidates):
        for column in source_columns:
            if column not in candidates:
                candidates.append(column)

    evidence["explicit_columns"] = explicit_candidates
    evidence["raw_boolean_like_columns"] = raw_candidates
    evidence["dtype_hint_columns"] = dtype_candidates
    evidence["raw_boolean_like_values"] = {key: values[:8] for key, values in raw_observed_values.items()}

    if not candidates:
        evidence["applies_if"] = False
        evidence["skip_reason"] = "no_boolean_like_columns_detected"

    return candidates, evidence


def _check_boolean_normalization(
    cleaned_csv_path: str,
    cleaned_header: List[str],
    sample_str: Optional[pd.DataFrame],
    raw_sample: Optional[pd.DataFrame],
    params: Dict[str, Any],
    column_dtype_targets: Dict[str, Any],
    manifest: Dict[str, Any],
) -> Tuple[List[str], Dict[str, Any]]:
    issues: List[str] = []
    candidates, evidence = _resolve_boolean_normalization_candidates(
        cleaned_header=cleaned_header,
        params=params,
        column_dtype_targets=column_dtype_targets,
        raw_sample=raw_sample,
        sample_str=sample_str,
    )
    evidence["allowed_normalized_values"] = sorted(_resolve_boolean_normalization_targets(params))
    evidence["columns_checked"] = candidates

    declared_excluded = {
        str(col).strip()
        for key in ("column_exclusions", "dropped_columns", "forbidden_columns_removed", "removed_columns")
        for col in _list_str(manifest.get(key) if isinstance(manifest, dict) else [])
        if str(col).strip()
    }
    explicit_missing = [
        column
        for column in (_list_str(params.get("columns")) + _list_str(params.get("target_columns")))
        if (
            str(column).strip()
            and str(column).strip() not in (cleaned_header or [])
            and str(column).strip() not in declared_excluded
        )
    ]
    if explicit_missing:
        issues.append("boolean_normalization_missing_columns: " + ", ".join(explicit_missing[:10]))
        evidence["missing_explicit_columns"] = explicit_missing
    if declared_excluded:
        evidence["declared_excluded_columns"] = sorted(declared_excluded)

    if not bool(evidence.get("applies_if", True)):
        return issues, evidence

    allowed_values = _resolve_boolean_normalization_targets(params)
    allowed_cleaned_tokens = _resolve_allowed_cleaned_boolean_tokens(params)
    expected_dtype = str(params.get("expected_dtype") or "").strip().lower()
    cleaned_values = _read_csv_unique_string_values(cleaned_csv_path, candidates)
    cleaned_samples = {column: _string_values(sample_str, column)[:12] for column in candidates if sample_str is not None}
    evidence["allowed_cleaned_tokens"] = sorted(allowed_cleaned_tokens)
    evidence["expected_dtype"] = expected_dtype or None
    if expected_dtype.startswith("int") or expected_dtype in {"bool", "boolean"}:
        evidence["csv_roundtrip_dtype_width_not_enforced"] = True
    evidence["cleaned_unique_values"] = {
        column: (cleaned_values.get(column) or cleaned_samples.get(column) or [])[:10]
        for column in candidates
    }

    invalid_by_column: Dict[str, List[str]] = {}
    for column in candidates:
        values = cleaned_values.get(column) or cleaned_samples.get(column) or []
        invalid_tokens: List[str] = []
        for value in values:
            literal = str(value).strip().lower()
            if literal in allowed_cleaned_tokens:
                continue
            normalized = _normalize_boolean_value_token(value)
            if normalized is None:
                continue
            if normalized in allowed_values:
                rendered = str(value)
                if rendered not in invalid_tokens:
                    invalid_tokens.append(rendered)
                continue
            rendered = normalized.split("::", 1)[1] if normalized.startswith("INVALID::") else str(value)
            if rendered not in invalid_tokens:
                invalid_tokens.append(rendered)
        if invalid_tokens:
            invalid_by_column[column] = invalid_tokens[:8]
            issues.append(
                f"{column} contains non-normalized boolean values: {', '.join(invalid_tokens[:5])}"
            )

    if invalid_by_column:
        evidence["invalid_values"] = invalid_by_column

    return issues, evidence


def _read_csv_selected_columns(path: str, columns: List[str]) -> Optional[pd.DataFrame]:
    requested = [str(col).strip() for col in columns if str(col).strip()]
    if not path or not os.path.exists(path) or not requested:
        return None
    encoding = _infer_encoding(path)
    delimiter = _infer_delimiter_from_file(path) or ","
    try:
        header = pd.read_csv(path, nrows=0, sep=delimiter, encoding=encoding, low_memory=False)
    except Exception:
        return None
    available = [col for col in requested if col in [str(name) for name in header.columns]]
    if not available:
        return None
    try:
        return pd.read_csv(
            path,
            usecols=available,
            dtype="string",
            sep=delimiter,
            encoding=encoding,
            low_memory=False,
        )
    except Exception:
        return None


def _manifest_gate_status(manifest: Dict[str, Any], *gate_keys: Any) -> str:
    statuses = manifest.get("cleaning_gates_status") if isinstance(manifest, dict) else {}
    if not isinstance(statuses, dict):
        return ""
    normalized_keys = {_normalize_gate_name(key) for key in gate_keys if str(key or "").strip()}
    for key, value in statuses.items():
        if _normalize_gate_name(key) in normalized_keys:
            return str(value or "")
    return ""


def _manifest_status_passed(status: Any) -> bool:
    token = str(status or "").strip().lower()
    return token.startswith("passed") or token in {"pass", "ok", "true", "verified"}


def _is_datetime_parseable_gate(gate_key: str, params: Dict[str, Any]) -> bool:
    declared = " ".join(
        str(value or "")
        for value in (
            gate_key,
            params.get("column"),
            params.get("dtype"),
            params.get("target_dtype"),
            params.get("parse_type"),
        )
    ).lower()
    return any(token in declared for token in ("date", "time", "timestamp", "month", "snapshot"))


def _check_datetime_parseable_gate(
    *,
    cleaned_csv_path: str,
    cleaned_header: List[str],
    sample_str: Optional[pd.DataFrame],
    params: Dict[str, Any],
    manifest: Dict[str, Any],
    gate_key: str,
) -> Tuple[List[str], Dict[str, Any]]:
    issues: List[str] = []
    evidence: Dict[str, Any] = {"applies_if": True}
    column_candidates = _list_str(params.get("column")) + _list_str(params.get("columns"))
    column = _pick_first_existing(column_candidates, cleaned_header)
    threshold_raw = params.get("required_parse_ratio", params.get("min_parse_ratio", 1.0))
    try:
        threshold = float(threshold_raw)
    except Exception:
        threshold = 1.0
    evidence["column"] = column or None
    evidence["required_parse_ratio"] = threshold
    if not column:
        evidence["applies_if"] = False
        evidence["skip_reason"] = "parseable_column_missing"
        return issues, evidence

    manifest_status = _manifest_gate_status(manifest, gate_key, f"{column}_parseable")
    if manifest_status:
        evidence["manifest_gate_status"] = manifest_status

    frame = _read_csv_selected_columns(cleaned_csv_path, [column])
    source = "full_cleaned_csv"
    if frame is None or column not in frame.columns:
        if isinstance(sample_str, pd.DataFrame) and column in sample_str.columns:
            frame = sample_str[[column]].copy()
            source = "cleaned_sample"
        else:
            if _manifest_status_passed(manifest_status):
                evidence["source"] = "manifest_gate_status"
                return issues, evidence
            evidence["applies_if"] = False
            evidence["skip_reason"] = "parseable_column_unavailable"
            return issues, evidence

    series = frame[column]
    non_null_mask = ~series.map(_is_null_like_text)
    non_null = series[non_null_mask]
    parsed = _parse_datetime_series_robust(non_null) if len(non_null) else pd.Series([], dtype="datetime64[ns]")
    ratio = float(parsed.notna().sum()) / float(len(non_null)) if len(non_null) else 1.0
    evidence.update(
        {
            "source": source,
            "rows_checked": int(len(frame)),
            "non_null_rows_checked": int(len(non_null)),
            "parseable_ratio": round(ratio, 6),
            "parse_failures": int(parsed.isna().sum()),
            "sample_values": _string_values(frame, column)[:8],
        }
    )
    if ratio + 1e-12 < threshold:
        issues.append(f"{column} datetime parseable_ratio={ratio:.4f} < {threshold:.4f}")
    return issues, evidence


def _check_numeric_parseable_gate(
    *,
    cleaned_csv_path: str,
    cleaned_header: List[str],
    sample_str: Optional[pd.DataFrame],
    sample_infer: Optional[pd.DataFrame],
    params: Dict[str, Any],
    manifest: Dict[str, Any],
    gate_key: str,
) -> Tuple[List[str], Dict[str, Any]]:
    issues: List[str] = []
    evidence: Dict[str, Any] = {"applies_if": True}
    column_candidates = _list_str(params.get("column")) + _list_str(params.get("columns"))
    column = _pick_first_existing(column_candidates, cleaned_header)
    threshold_raw = params.get("required_parse_ratio", params.get("min_parse_ratio", 1.0))
    try:
        threshold = float(threshold_raw)
    except Exception:
        threshold = 1.0
    evidence["column"] = column or None
    evidence["required_parse_ratio"] = threshold
    if not column:
        evidence["applies_if"] = False
        evidence["skip_reason"] = "numeric_column_missing"
        return issues, evidence

    manifest_status = _manifest_gate_status(manifest, gate_key, f"{column}_numeric_parseable", f"{column}_numeric_conversion_verified")
    if manifest_status:
        evidence["manifest_gate_status"] = manifest_status

    if isinstance(sample_infer, pd.DataFrame) and column in sample_infer.columns:
        inferred_dtype = str(sample_infer[column].dtype)
        evidence["sample_inferred_dtype"] = inferred_dtype

    frame = _read_csv_selected_columns(cleaned_csv_path, [column])
    source = "full_cleaned_csv"
    if frame is None or column not in frame.columns:
        if isinstance(sample_str, pd.DataFrame) and column in sample_str.columns:
            frame = sample_str[[column]].copy()
            source = "cleaned_sample"
        elif isinstance(sample_infer, pd.DataFrame) and column in sample_infer.columns and pd.api.types.is_numeric_dtype(sample_infer[column]):
            evidence["source"] = "sample_inferred_dtype"
            evidence["parseable_ratio"] = 1.0
            return issues, evidence
        else:
            if _manifest_status_passed(manifest_status):
                evidence["source"] = "manifest_gate_status"
                return issues, evidence
            evidence["applies_if"] = False
            evidence["skip_reason"] = "numeric_column_unavailable"
            return issues, evidence

    ratio = _best_numeric_parse_ratio(frame[column])
    evidence.update(
        {
            "source": source,
            "rows_checked": int(len(frame)),
            "parseable_ratio": round(float(ratio), 6),
            "sample_values": _string_values(frame, column)[:8],
        }
    )
    if float(ratio) + 1e-12 < threshold:
        issues.append(f"{column} numeric parseable_ratio={float(ratio):.4f} < {threshold:.4f}")
    return issues, evidence


def _parse_datetime_series_robust(values: pd.Series) -> pd.Series:
    base = values.astype("string").fillna("")
    parsed = pd.to_datetime(base, errors="coerce")
    try:
        mixed = pd.to_datetime(base, errors="coerce", format="mixed")
        parsed = mixed.where(mixed.notna(), parsed)
    except (TypeError, ValueError):
        pass
    needs_dayfirst = base.str.contains(r"/", regex=True, na=False) | base.str.contains(
        r"^\d{2}-\d{2}-\d{4}", regex=True, na=False
    )
    alt = pd.Series(pd.NaT, index=base.index, dtype="datetime64[ns]")
    if bool(needs_dayfirst.any()):
        alt.loc[needs_dayfirst] = pd.to_datetime(base.loc[needs_dayfirst], errors="coerce", dayfirst=True)
    choose_alt = alt.notna() & parsed.isna()
    return alt.where(choose_alt, parsed)


def _resolve_training_indicator_candidates(
    cleaned_header: List[str],
    column_roles: Dict[str, List[str]],
    params: Dict[str, Any],
) -> List[str]:
    candidates = (
        _list_str(params.get("training_indicator_columns"))
        + _list_str(params.get("split_column"))
        + _list_str(params.get("split_columns"))
    )
    candidates.extend(_columns_with_role_tokens(column_roles, {"split", "partition", "fold"}))
    candidates.extend(["__split", "split", "partition", "is_train", "train_flag", "training_flag"])
    deduped: List[str] = []
    for candidate in candidates:
        if candidate in cleaned_header and candidate not in deduped:
            deduped.append(candidate)
    return deduped


def _normalize_training_indicator_mask(series: pd.Series) -> Optional[pd.Series]:
    normalized: List[Optional[bool]] = []
    recognized = 0
    for raw in series.astype("string").fillna("").tolist():
        token = str(raw).strip().lower()
        if token in _BOOLEAN_NULL_TOKENS:
            normalized.append(None)
            continue
        if token in _TRAIN_TRUE_TOKENS:
            normalized.append(True)
            recognized += 1
            continue
        if token in _TRAIN_FALSE_TOKENS:
            normalized.append(False)
            recognized += 1
            continue
        return None
    if recognized <= 0:
        return None
    return pd.Series(normalized, index=series.index, dtype="boolean")


def _resolve_temporal_cutoff(params: Dict[str, Any]) -> Optional[pd.Timestamp]:
    raw = params.get("training_cutoff") or params.get("cutoff") or params.get("max_allowed_date")
    if not raw:
        rule = str(params.get("rule") or params.get("filter") or params.get("required_condition") or "")
        match = re.search(r"(\d{4}-\d{2}-\d{2})", rule)
        raw = match.group(1) if match else None
    if not raw:
        return None
    parsed = pd.to_datetime(str(raw), errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.normalize()


def _check_enforce_temporal_training_mask(
    cleaned_csv_path: str,
    cleaned_header: List[str],
    params: Dict[str, Any],
    column_roles: Dict[str, List[str]],
) -> Tuple[List[str], Dict[str, Any], Optional[Dict[str, Any]]]:
    issues: List[str] = []
    evidence: Dict[str, Any] = {"applies_if": True}

    date_candidates = _list_str(params.get("column")) + _list_str(params.get("columns"))
    if not date_candidates:
        date_candidates.extend(_columns_with_role_tokens(column_roles, {"time", "date", "timestamp"}))
        date_candidates.extend(["created_at", "timestamp", "date"])
    date_col = _pick_first_existing(date_candidates, cleaned_header)
    cutoff = _resolve_temporal_cutoff(params)
    evidence["date_column"] = date_col or None
    evidence["training_cutoff"] = cutoff.date().isoformat() if cutoff is not None else None
    if not date_col:
        evidence["applies_if"] = False
        evidence["skip_reason"] = "temporal_column_missing"
        return issues, evidence, None
    if cutoff is None:
        evidence["applies_if"] = False
        evidence["skip_reason"] = "training_cutoff_missing_or_invalid"
        return issues, evidence, None

    indicator_candidates = _resolve_training_indicator_candidates(cleaned_header, column_roles, params)
    columns_to_load = [date_col] + indicator_candidates
    frame = _read_csv_selected_columns(cleaned_csv_path, columns_to_load)
    if frame is None or date_col not in frame.columns:
        evidence["applies_if"] = False
        evidence["skip_reason"] = "cleaned_csv_temporal_columns_unavailable"
        return issues, evidence, None

    parsed_dates = _parse_datetime_series_robust(frame[date_col])
    evidence["rows_read"] = int(len(frame))
    evidence["date_parse_failures"] = int(parsed_dates.isna().sum())
    future_mask = parsed_dates.notna() & (parsed_dates.dt.normalize() > cutoff)
    training_mask = parsed_dates.notna() & (parsed_dates.dt.normalize() <= cutoff)
    evidence["future_rows_present"] = int(future_mask.sum())

    mask_source = "date_cutoff_rule"
    indicator_column = None
    for candidate in indicator_candidates:
        if candidate not in frame.columns:
            continue
        indicator_mask = _normalize_training_indicator_mask(frame[candidate])
        if indicator_mask is None:
            continue
        indicator_column = candidate
        mask_source = f"indicator:{candidate}"
        evidence["training_indicator_column"] = candidate
        evidence["training_indicator_distinct_values"] = _string_values(frame, candidate)[:8]
        explicit_training = indicator_mask.fillna(False).astype(bool)
        training_mask = explicit_training & parsed_dates.notna()
        violating = explicit_training & future_mask
        if violating.any():
            issues.append(
                f"training-designated rows violate cutoff {cutoff.date().isoformat()} in {date_col}"
            )
            evidence["training_rows_after_cutoff"] = int(violating.sum())
        break

    evidence["training_mask_source"] = mask_source
    evidence["training_rows_count"] = int(training_mask.sum())
    if indicator_column is None and future_mask.any():
        evidence["note"] = (
            "future rows are allowed outside the contract-defined training subset; "
            "gate enforced on rows satisfying the training cutoff rule"
        )

    context = {
        "frame": frame,
        "date_column": date_col,
        "cutoff": cutoff,
        "training_mask": training_mask,
        "training_mask_source": mask_source,
    }
    return issues, evidence, context


def _check_target_not_null_in_training(
    cleaned_csv_path: str,
    cleaned_header: List[str],
    params: Dict[str, Any],
    column_roles: Dict[str, List[str]],
    training_rows_context: Optional[Dict[str, Any]],
) -> Tuple[List[str], Dict[str, Any]]:
    issues: List[str] = []
    evidence: Dict[str, Any] = {"applies_if": True}
    target_candidates = _list_str(params.get("column")) + _list_str(params.get("target_columns"))
    if not target_candidates:
        target_candidates.extend(_columns_with_role_tokens(column_roles, {"target", "label", "outcome"}))
        target_candidates.extend(["target", "label", "y"])
    target_col = _pick_first_existing(target_candidates, cleaned_header)
    evidence["target_column"] = target_col or None
    evidence["applies_to"] = str(params.get("applies_to") or "training_rows")
    if not target_col:
        evidence["applies_if"] = False
        evidence["skip_reason"] = "target_column_missing"
        return issues, evidence
    if not training_rows_context or not isinstance(training_rows_context, dict):
        evidence["applies_if"] = False
        evidence["skip_reason"] = "training_rows_context_unavailable"
        return issues, evidence

    frame = training_rows_context.get("frame")
    if not isinstance(frame, pd.DataFrame):
        evidence["applies_if"] = False
        evidence["skip_reason"] = "training_rows_frame_unavailable"
        return issues, evidence
    if target_col not in frame.columns:
        date_col = str(training_rows_context.get("date_column") or "").strip()
        reload_cols = [target_col]
        if date_col:
            reload_cols.append(date_col)
        reloaded = _read_csv_selected_columns(cleaned_csv_path, reload_cols)
        if reloaded is None or target_col not in reloaded.columns:
            evidence["applies_if"] = False
            evidence["skip_reason"] = "target_column_unavailable_in_cleaned_csv"
            return issues, evidence
        if date_col and date_col in reloaded.columns:
            frame = reloaded
            parsed_dates = _parse_datetime_series_robust(frame[date_col])
            cutoff = training_rows_context.get("cutoff")
            if isinstance(cutoff, pd.Timestamp):
                training_mask = parsed_dates.notna() & (parsed_dates.dt.normalize() <= cutoff)
            else:
                training_mask = parsed_dates.notna()
        else:
            frame = reloaded
            training_mask = pd.Series([True] * len(frame), index=frame.index)
    else:
        training_mask = training_rows_context.get("training_mask")
        if not isinstance(training_mask, pd.Series) or len(training_mask) != len(frame):
            evidence["applies_if"] = False
            evidence["skip_reason"] = "training_mask_unavailable"
            return issues, evidence

    target_series = frame[target_col]
    null_mask = target_series.map(_is_null_like_text)
    violating = training_mask.fillna(False).astype(bool) & null_mask.fillna(True).astype(bool)
    evidence["training_rows_count"] = int(training_mask.fillna(False).astype(bool).sum())
    evidence["null_training_rows"] = int(violating.sum())
    evidence["training_mask_source"] = str(training_rows_context.get("training_mask_source") or "unknown")
    if violating.any():
        issues.append(f"{target_col} contains null values inside training rows")
    return issues, evidence


def _extract_condition_columns(required_condition: str) -> List[str]:
    text = str(required_condition or "").strip()
    if not text:
        return []
    columns: List[str] = []
    patterns = (
        r"\b([A-Za-z_][A-Za-z0-9_]*)\s+IS\s+NOT\s+NULL\b",
        r"\b([A-Za-z_][A-Za-z0-9_]*)\s+IS\s+NULL\b",
        r"\b([A-Za-z_][A-Za-z0-9_]*)\s*(?:<=|>=|<|>|=)\s*(?:'[^']*'|\"[^\"]*\"|[^\s]+)",
    )
    for pattern in patterns:
        try:
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
        except re.error:
            matches = []
        for match in matches:
            token = str(match or "").strip()
            if token and token not in columns:
                columns.append(token)
    return columns


def _coerce_condition_series(
    series: pd.Series,
    raw_value: str,
) -> Tuple[pd.Series, Any, str]:
    token = str(raw_value or "").strip().strip("'\"")
    if re.match(r"^\d{4}-\d{2}-\d{2}(?:[ T].*)?$", token) or "/" in token or re.match(r"^\d{2}-\d{2}-\d{4}$", token):
        parsed_series = _parse_datetime_series_robust(series)
        parsed_value = pd.to_datetime(token, errors="coerce")
        if not pd.isna(parsed_value):
            return parsed_series.dt.normalize(), pd.Timestamp(parsed_value).normalize(), "datetime"
    try:
        numeric_value = float(token)
        numeric_series = pd.to_numeric(series.astype("string"), errors="coerce")
        if numeric_series.notna().any():
            return numeric_series, numeric_value, "numeric"
    except Exception:
        pass
    return series.astype("string"), token, "string"


def _evaluate_required_condition_mask(
    frame: pd.DataFrame,
    required_condition: str,
) -> Tuple[Optional[pd.Series], Dict[str, Any]]:
    evidence: Dict[str, Any] = {
        "required_condition": str(required_condition or "").strip(),
        "clauses": [],
        "unsupported_clauses": [],
        "missing_columns": [],
    }
    text = str(required_condition or "").strip()
    if not text:
        evidence["skip_reason"] = "required_condition_missing"
        return None, evidence

    clauses = [part.strip() for part in re.split(r"\bAND\b", text, flags=re.IGNORECASE) if part.strip()]
    if not clauses:
        evidence["skip_reason"] = "required_condition_unparseable"
        return None, evidence

    mask = pd.Series([True] * len(frame), index=frame.index, dtype="boolean")
    for clause in clauses:
        clause_info: Dict[str, Any] = {"clause": clause}
        match_not_null = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s+IS\s+NOT\s+NULL$", clause, flags=re.IGNORECASE)
        match_is_null = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s+IS\s+NULL$", clause, flags=re.IGNORECASE)
        match_compare = re.match(
            r"^([A-Za-z_][A-Za-z0-9_]*)\s*(<=|>=|<|>|=)\s*(?:'([^']*)'|\"([^\"]*)\"|([^\s]+))$",
            clause,
            flags=re.IGNORECASE,
        )

        if match_not_null:
            col = str(match_not_null.group(1) or "").strip()
            if col not in frame.columns:
                evidence["missing_columns"].append(col)
                return None, evidence
            clause_mask = ~frame[col].map(_is_null_like_text)
            clause_info.update({"column": col, "operator": "IS NOT NULL"})
        elif match_is_null:
            col = str(match_is_null.group(1) or "").strip()
            if col not in frame.columns:
                evidence["missing_columns"].append(col)
                return None, evidence
            clause_mask = frame[col].map(_is_null_like_text)
            clause_info.update({"column": col, "operator": "IS NULL"})
        elif match_compare:
            col = str(match_compare.group(1) or "").strip()
            op = str(match_compare.group(2) or "").strip()
            rhs = next(
                (
                    str(group).strip()
                    for group in match_compare.groups()[2:]
                    if isinstance(group, str) and str(group).strip()
                ),
                "",
            )
            if col not in frame.columns:
                evidence["missing_columns"].append(col)
                return None, evidence
            coerced_series, expected_value, value_kind = _coerce_condition_series(frame[col], rhs)
            clause_info.update(
                {
                    "column": col,
                    "operator": op,
                    "rhs": rhs,
                    "value_kind": value_kind,
                }
            )
            if value_kind == "string":
                lhs = coerced_series.astype("string").fillna("")
                rhs_token = str(expected_value or "")
                if op == "=":
                    clause_mask = lhs == rhs_token
                else:
                    evidence["unsupported_clauses"].append(clause)
                    return None, evidence
            else:
                lhs = coerced_series
                if op == "<=":
                    clause_mask = lhs <= expected_value
                elif op == ">=":
                    clause_mask = lhs >= expected_value
                elif op == "<":
                    clause_mask = lhs < expected_value
                elif op == ">":
                    clause_mask = lhs > expected_value
                elif op == "=":
                    clause_mask = lhs == expected_value
                else:
                    evidence["unsupported_clauses"].append(clause)
                    return None, evidence
            clause_mask = clause_mask.fillna(False)
        else:
            evidence["unsupported_clauses"].append(clause)
            return None, evidence

        clause_info["rows_matching"] = int(clause_mask.fillna(False).astype(bool).sum())
        evidence["clauses"].append(clause_info)
        mask = mask.fillna(False).astype(bool) & clause_mask.fillna(False).astype(bool)

    return mask.astype("boolean"), evidence


def _resolve_partition_filter(params: Dict[str, Any]) -> str:
    return str(
        params.get("filter")
        or params.get("partition_filter")
        or params.get("row_filter")
        or ""
    ).strip()


def _check_target_not_null_under_filter(
    cleaned_csv_path: str,
    cleaned_header: List[str],
    params: Dict[str, Any],
) -> Tuple[List[str], Dict[str, Any]]:
    issues: List[str] = []
    evidence: Dict[str, Any] = {"applies_if": True}
    target_candidates = _list_str(params.get("column")) + _list_str(params.get("target_columns"))
    target_col = _pick_first_existing(target_candidates, cleaned_header)
    filter_text = _resolve_partition_filter(params)
    evidence["target_column"] = target_col or None
    evidence["partition_filter"] = filter_text or None
    if not filter_text and str(params.get("partition") or "").strip().lower() == "training":
        evidence["applies_if"] = False
        evidence["skip_reason"] = "training_partition_filter_missing"
        return issues, evidence
    if not target_col:
        evidence["applies_if"] = False
        evidence["skip_reason"] = "target_column_missing"
        return issues, evidence
    columns = _extract_condition_columns(filter_text)
    if target_col not in columns:
        columns.append(target_col)
    frame = _read_csv_selected_columns(cleaned_csv_path, columns)
    if frame is None or target_col not in frame.columns:
        evidence["applies_if"] = False
        evidence["skip_reason"] = "cleaned_csv_target_or_filter_columns_unavailable"
        return issues, evidence
    if filter_text:
        partition_mask, condition_evidence = _evaluate_required_condition_mask(frame, filter_text)
        evidence["condition_evidence"] = condition_evidence
        if partition_mask is None:
            evidence["applies_if"] = False
            evidence["skip_reason"] = "partition_filter_unparseable"
            return issues, evidence
    else:
        partition_mask = pd.Series([True] * len(frame), index=frame.index, dtype="boolean")
        evidence["condition_evidence"] = {"required_condition": "", "note": "no_partition_filter_all_rows"}

    target_null = frame[target_col].map(_is_null_like_text)
    partition_bool = partition_mask.fillna(False).astype(bool)
    violating = partition_bool & target_null.fillna(True).astype(bool)
    evidence["rows_checked"] = int(len(frame))
    evidence["partition_rows"] = int(partition_bool.sum())
    evidence["null_rows_in_partition"] = int(violating.sum())
    if violating.any():
        issues.append(f"{target_col} contains null values inside filtered partition rows")
    return issues, evidence


def _check_no_exact_duplicates_under_filter(
    cleaned_csv_path: str,
    cleaned_header: List[str],
    params: Dict[str, Any],
) -> Tuple[List[str], Dict[str, Any]]:
    issues: List[str] = []
    evidence: Dict[str, Any] = {"applies_if": True}
    filter_text = _resolve_partition_filter(params)
    evidence["partition_filter"] = filter_text or None
    if not filter_text and str(params.get("partition") or "").strip().lower() == "training":
        evidence["applies_if"] = False
        evidence["skip_reason"] = "training_partition_filter_missing"
        return issues, evidence
    if not cleaned_csv_path or not os.path.exists(cleaned_csv_path):
        evidence["applies_if"] = False
        evidence["skip_reason"] = "cleaned_csv_unavailable"
        return issues, evidence
    try:
        delimiter = _infer_delimiter_from_file(cleaned_csv_path) or ","
        encoding = _infer_encoding(cleaned_csv_path)
        frame = pd.read_csv(cleaned_csv_path, dtype="string", sep=delimiter, encoding=encoding, low_memory=False)
    except Exception as exc:
        evidence["applies_if"] = False
        evidence["skip_reason"] = "cleaned_csv_read_failed"
        evidence["exception"] = str(exc)[:220]
        return issues, evidence

    if filter_text:
        filter_columns = _extract_condition_columns(filter_text)
        missing = [col for col in filter_columns if col not in frame.columns]
        if missing:
            evidence["applies_if"] = False
            evidence["skip_reason"] = "partition_filter_columns_missing"
            evidence["missing_columns"] = missing
            return issues, evidence
        partition_mask, condition_evidence = _evaluate_required_condition_mask(frame, filter_text)
        evidence["condition_evidence"] = condition_evidence
        if partition_mask is None:
            evidence["applies_if"] = False
            evidence["skip_reason"] = "partition_filter_unparseable"
            return issues, evidence
        scoped = frame.loc[partition_mask.fillna(False).astype(bool)]
    else:
        scoped = frame
        evidence["condition_evidence"] = {"required_condition": "", "note": "no_partition_filter_all_rows"}

    duplicate_mask = scoped.duplicated(keep=False)
    duplicate_rows = int(duplicate_mask.sum())
    evidence["rows_checked"] = int(len(frame))
    evidence["partition_rows"] = int(len(scoped))
    evidence["duplicate_rows_in_partition"] = duplicate_rows
    if duplicate_rows:
        issues.append(f"Exact duplicate rows remain in filtered partition: {duplicate_rows}")
    return issues, evidence


def _check_split_condition_enforced(
    cleaned_csv_path: str,
    cleaned_header: List[str],
    params: Dict[str, Any],
    column_roles: Dict[str, List[str]],
) -> Tuple[List[str], Dict[str, Any]]:
    issues: List[str] = []
    evidence: Dict[str, Any] = {"applies_if": True}
    split_label = str(params.get("split_label") or "").strip()
    required_condition = str(params.get("required_condition") or params.get("condition") or "").strip()
    evidence["split_label"] = split_label or None
    evidence["required_condition"] = required_condition or None
    if not split_label:
        evidence["applies_if"] = False
        evidence["skip_reason"] = "split_label_missing"
        return issues, evidence
    if not required_condition:
        evidence["applies_if"] = False
        evidence["skip_reason"] = "required_condition_missing"
        return issues, evidence

    split_candidates = _resolve_training_indicator_candidates(cleaned_header, column_roles, params)
    condition_columns = _extract_condition_columns(required_condition)
    frame = _read_csv_selected_columns(cleaned_csv_path, split_candidates + condition_columns)
    if frame is None:
        evidence["applies_if"] = False
        evidence["skip_reason"] = "cleaned_csv_columns_unavailable"
        return issues, evidence

    split_column = ""
    split_mask = None
    for candidate in split_candidates:
        if candidate not in frame.columns:
            continue
        series = frame[candidate].astype("string").fillna("").str.strip().str.lower()
        label_mask = series == split_label.lower()
        if bool(label_mask.any()):
            split_column = candidate
            split_mask = label_mask
            break
    evidence["split_column"] = split_column or None
    if split_mask is None:
        evidence["applies_if"] = False
        evidence["skip_reason"] = "split_label_not_found"
        return issues, evidence

    condition_mask, condition_evidence = _evaluate_required_condition_mask(frame, required_condition)
    evidence["condition_evidence"] = condition_evidence
    if condition_mask is None:
        evidence["applies_if"] = False
        evidence["skip_reason"] = "required_condition_unparseable"
        return issues, evidence

    violating = split_mask.fillna(False).astype(bool) & ~condition_mask.fillna(False).astype(bool)
    evidence["rows_in_split"] = int(split_mask.fillna(False).astype(bool).sum())
    evidence["violating_rows"] = int(violating.sum())
    if violating.any():
        issues.append(
            f"Rows labeled '{split_label}' violate required_condition: {required_condition}"
        )
    return issues, evidence


def _check_identifier_columns_excluded_from_features(
    cleaned_csv_path: str,
    cleaned_header: List[str],
    sample_str: Optional[pd.DataFrame],
    sample_infer: Optional[pd.DataFrame],
    params: Dict[str, Any],
    column_roles: Dict[str, List[str]],
) -> Tuple[List[str], Dict[str, Any]]:
    issues: List[str] = []
    evidence: Dict[str, Any] = {"applies_if": True}
    forbidden = (
        _list_str(params.get("forbidden_as_features"))
        or _list_str(params.get("forbidden_columns"))
        or _list_str(params.get("columns"))
    )
    evidence["forbidden_as_features"] = forbidden
    if not forbidden:
        evidence["applies_if"] = False
        evidence["skip_reason"] = "forbidden_as_features_missing"
        return issues, evidence

    header_lookup = {str(col).strip().lower(): str(col) for col in (cleaned_header or []) if str(col).strip()}
    present_forbidden = [
        header_lookup[str(col).strip().lower()]
        for col in forbidden
        if str(col).strip().lower() in header_lookup
    ]
    evidence["forbidden_present_in_cleaned_header"] = present_forbidden

    role_ctx = _resolve_id_role_context(cleaned_header, column_roles)
    passthrough_allowed = set(role_ctx.get("id_like") or []) | set(role_ctx.get("split_like") or [])
    for col in present_forbidden:
        lowered = str(col).strip().lower()
        if lowered.endswith("_id") or lowered in {"account_id", "snapshot_month_end"}:
            passthrough_allowed.add(col)
    evidence["passthrough_allowed"] = sorted(passthrough_allowed)

    unexpected_present = [col for col in present_forbidden if col not in passthrough_allowed]
    if unexpected_present:
        issues.append(
            "Forbidden feature columns still present in cleaned dataset: "
            + ", ".join(unexpected_present[:10])
        )

    passthrough_to_validate = [col for col in present_forbidden if col in passthrough_allowed]
    passthrough_frame = _read_csv_selected_columns(cleaned_csv_path, passthrough_to_validate) if passthrough_to_validate else None
    null_fractions: Dict[str, float] = {}
    for col in passthrough_to_validate:
        null_frac = None
        if isinstance(passthrough_frame, pd.DataFrame) and col in passthrough_frame.columns:
            series = passthrough_frame[col]
            total = len(series)
            if total > 0:
                null_frac = float(series.map(_is_null_like_text).sum() / total)
        if null_frac is None:
            null_frac = _compute_null_fraction(sample_infer, sample_str, col)
        if null_frac is None:
            continue
        null_fractions[col] = round(float(null_frac), 4)
        if float(null_frac) >= 0.9999:
            issues.append(f"Critical identifier '{col}' was destroyed (100% null)")
    evidence["passthrough_null_frac"] = null_fractions
    return issues, evidence


def _check_arr_current_numeric_conversion_verified(
    cleaned_header: List[str],
    sample_str: Optional[pd.DataFrame],
    sample_infer: Optional[pd.DataFrame],
    params: Dict[str, Any],
    manifest: Dict[str, Any],
) -> Tuple[List[str], Dict[str, Any]]:
    issues: List[str] = []
    evidence: Dict[str, Any] = {"applies_if": True}
    target_candidates = _list_str(params.get("column")) + _list_str(params.get("columns"))
    target_col = target_candidates[0] if target_candidates else ""
    if not target_col:
        evidence["applies_if"] = False
        evidence["skip_reason"] = "target_column_missing"
        return issues, evidence
    evidence["target_column"] = target_col

    manifest_text = ""
    try:
        manifest_text = json.dumps(manifest or {}, ensure_ascii=False).lower()
    except Exception:
        manifest_text = str(manifest or "").lower()
    gate_status = ""
    if isinstance(manifest, dict):
        gate_status = str((manifest.get("cleaning_gates_status") or {}).get("arr_current_numeric_conversion_verified") or "")
    documented_drop = any(
        token in manifest_text
        for token in (
            "dropped_from_features",
            "dropped from features",
            "dropped from model features",
            "warning_dropped_from_features",
        )
    ) or "dropped_from_features" in gate_status.lower()
    evidence["documented_model_feature_exclusion"] = bool(documented_drop)
    evidence["gate_status"] = gate_status or None

    if target_col not in (cleaned_header or []):
        if documented_drop:
            return issues, evidence
        evidence["applies_if"] = False
        evidence["skip_reason"] = "target_column_absent_in_cleaned_header"
        return issues, evidence

    inferred_dtype = str(sample_infer[target_col].dtype) if isinstance(sample_infer, pd.DataFrame) and target_col in sample_infer.columns else ""
    examples = _string_values(sample_str, target_col)[:8]
    evidence["inferred_dtype"] = inferred_dtype or None
    evidence["sample_values"] = examples

    if inferred_dtype and pd.api.types.is_numeric_dtype(sample_infer[target_col]):
        return issues, evidence
    if documented_drop:
        return issues, evidence

    currency_like = any(re.search(r"(?:^\$|^€|^£|\bEUR\b|\bUSD\b|\bGBP\b)", str(value), flags=re.IGNORECASE) for value in examples)
    if currency_like or (inferred_dtype.lower() == "object" and examples):
        issues.append("Column remains object type with currency strings")
    return issues, evidence


def _check_nps_forward_fill_temporal_integrity(
    cleaned_csv_path: str,
    cleaned_header: List[str],
    sample_str: Optional[pd.DataFrame],
    sample_infer: Optional[pd.DataFrame],
    params: Dict[str, Any],
    manifest: Dict[str, Any],
    cleaning_code: Optional[str],
) -> Tuple[List[str], Dict[str, Any]]:
    issues: List[str] = []
    evidence: Dict[str, Any] = {"applies_if": True}
    target_col = _pick_first_existing(_list_str(params.get("column")) + _list_str(params.get("columns")), cleaned_header)
    group_key = _pick_first_existing(_list_str(params.get("group_key")) + _list_str(params.get("group_keys")), cleaned_header)
    sort_key = _pick_first_existing(_list_str(params.get("sort_key")) + _list_str(params.get("sort_keys")), cleaned_header)
    evidence["column"] = target_col or None
    evidence["group_key"] = group_key or None
    evidence["sort_key"] = sort_key or None
    if not target_col or not group_key or not sort_key:
        evidence["applies_if"] = False
        evidence["skip_reason"] = "target_or_group_or_sort_column_missing"
        return issues, evidence

    frame = _read_csv_selected_columns(cleaned_csv_path, [group_key, sort_key, target_col])
    group_null_frac = None
    if isinstance(frame, pd.DataFrame) and group_key in frame.columns:
        total = len(frame[group_key])
        if total > 0:
            group_null_frac = float(frame[group_key].map(_is_null_like_text).sum() / total)
    if group_null_frac is None:
        group_null_frac = _compute_null_fraction(sample_infer, sample_str, group_key)
    evidence["group_key_null_frac"] = None if group_null_frac is None else round(float(group_null_frac), 4)
    if group_null_frac is not None and float(group_null_frac) >= 0.9999:
        issues.append(
            f"Forward fill cannot be validated because the group_key ({group_key}) is 100% null"
        )
        return issues, evidence

    code_text = str(cleaning_code or "")
    code_lower = code_text.lower()
    group_regex = re.compile(
        rf"groupby\(\s*['\"]{re.escape(group_key)}['\"]\s*\)\s*\[\s*['\"]{re.escape(target_col)}['\"]\s*\]\.ffill\(",
        flags=re.IGNORECASE,
    )
    sort_regex = re.compile(
        rf"sort_values\(\s*\[\s*['\"]{re.escape(group_key)}['\"]\s*,\s*['\"]{re.escape(sort_key)}['\"]\s*\]",
        flags=re.IGNORECASE,
    )
    has_group_ffill = bool(group_regex.search(code_text))
    has_sort = bool(sort_regex.search(code_text))
    evidence["groupby_ffill_detected"] = has_group_ffill
    evidence["sorted_by_group_and_time_detected"] = has_sort

    manifest_text = ""
    try:
        manifest_text = json.dumps(manifest or {}, ensure_ascii=False).lower()
    except Exception:
        manifest_text = str(manifest or "").lower()
    manifest_mentions_ffill = target_col.lower() in manifest_text and "forward_fill" in manifest_text
    evidence["manifest_mentions_forward_fill"] = manifest_mentions_ffill

    if has_group_ffill and has_sort:
        return issues, evidence
    if manifest_mentions_ffill and has_group_ffill:
        return issues, evidence

    issues.append("No evidence of partitioned forward-fill in cleaning_code")
    return issues, evidence


def _string_values(sample: Optional[pd.DataFrame], col: str) -> List[str]:
    if sample is None or col not in sample.columns:
        return []
    values: List[str] = []
    for val in sample[col].tolist():
        if val is None:
            continue
        text = str(val).strip()
        if not text or text.lower() == "nan":
            continue
        values.append(text)
    return values


def _columns_with_role_tokens(column_roles: Dict[str, List[str]], tokens: set[str]) -> List[str]:
    cols: List[str] = []
    for role, names in column_roles.items():
        if any(token in role.lower() for token in tokens):
            cols.extend(names)
    return cols


def _resolve_id_role_context(
    cleaned_header: List[str],
    column_roles: Dict[str, List[str]],
) -> Dict[str, Any]:
    header_set = set(cleaned_header or [])
    id_like = {col for col in _columns_with_role_tokens(column_roles, {"id", "identifier", "key"}) if col in header_set}
    target_like = {
        col for col in _columns_with_role_tokens(column_roles, {"target", "label", "outcome"}) if col in header_set
    }
    split_like = {
        col for col in _columns_with_role_tokens(column_roles, {"split", "partition", "fold"}) if col in header_set
    }
    excluded = target_like | split_like
    id_like -= excluded
    return {
        "id_like": sorted(id_like),
        "target_like": sorted(target_like),
        "split_like": sorted(split_like),
        "excluded": sorted(excluded),
    }


def _resolve_id_integrity_candidates(
    cleaned_header: List[str],
    params: Dict[str, Any],
    column_roles: Dict[str, List[str]],
) -> Tuple[List[str], Dict[str, Any]]:
    evidence: Dict[str, Any] = {"applies_if": True}
    if not cleaned_header:
        evidence["applies_if"] = False
        evidence["skip_reason"] = "cleaned_header_missing"
        evidence["candidate_columns"] = []
        return [], evidence

    role_ctx = _resolve_id_role_context(cleaned_header, column_roles)
    excluded = set(role_ctx.get("excluded") or [])

    explicit = _list_str(params.get("columns")) + _list_str(params.get("id_columns"))
    explicit_candidates = [col for col in explicit if col in cleaned_header and col not in excluded]

    role_candidates = [col for col in (role_ctx.get("id_like") or []) if col in cleaned_header]

    regex = params.get("identifier_name_regex") or _GENERIC_ID_REGEX
    try:
        pattern = re.compile(regex)
    except re.error:
        pattern = re.compile(_GENERIC_ID_REGEX)
    regex_candidates = [col for col in cleaned_header if col not in excluded and pattern.search(str(col))]

    candidates: List[str] = []
    candidate_source = "none"
    if explicit_candidates:
        candidates = explicit_candidates
        candidate_source = "params"
    elif role_candidates:
        candidates = role_candidates
        candidate_source = "column_roles"
    else:
        candidates = regex_candidates
        candidate_source = "regex_fallback"

    # Deduplicate while preserving order
    deduped: List[str] = []
    seen: set[str] = set()
    for col in candidates:
        if col in seen:
            continue
        seen.add(col)
        deduped.append(col)
    candidates = deduped

    evidence["candidate_source"] = candidate_source
    evidence["candidate_columns"] = candidates
    evidence["excluded_columns"] = role_ctx.get("excluded") or []
    evidence["target_role_columns"] = role_ctx.get("target_like") or []
    evidence["split_role_columns"] = role_ctx.get("split_like") or []

    if not candidates:
        evidence["applies_if"] = False
        evidence["skip_reason"] = "no_identifier_columns_detected"
    return candidates, evidence


def _extract_manifest_row_counts(manifest: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Dict[str, Any]]:
    if not isinstance(manifest, dict):
        return None, None, {}
    rows_before = manifest.get("rows_before")
    rows_after = manifest.get("rows_after")
    row_counts = manifest.get("row_counts") or {}
    if rows_before is None:
        for key in ("initial", "original", "input", "rows_before", "total"):
            if rows_before is None:
                rows_before = row_counts.get(key)
    if rows_after is None:
        for key in ("final", "after_cleaning", "output", "rows_after"):
            if rows_after is None:
                rows_after = row_counts.get(key)
    if not isinstance(rows_before, (int, float)):
        rows_before = None
    if not isinstance(rows_after, (int, float)):
        rows_after = None
    return rows_before, rows_after, row_counts


def _should_skip_row_count_sanity(
    manifest: Dict[str, Any],
    params: Dict[str, Any],
    column_roles: Dict[str, List[str]],
) -> Tuple[bool, str, Dict[str, Any]]:
    evidence: Dict[str, Any] = {"adaptive_policy": "none"}
    if not bool(params.get("allow_label_null_listwise_skip", True)):
        return False, "", evidence
    if bool(params.get("enforce_on_listwise_label_drop", False)):
        return False, "", evidence
    if not isinstance(manifest, dict):
        return False, "", evidence

    rows_before, rows_after, row_counts = _extract_manifest_row_counts(manifest)
    if rows_before is None or rows_after is None or rows_before <= 0 or rows_after > rows_before:
        return False, "", evidence

    dropped = float(rows_before - rows_after)
    if dropped <= 0:
        return False, "", evidence

    drop_reason = str((row_counts or {}).get("dropped_reason") or "").strip().lower()
    gate_results = manifest.get("gate_results") if isinstance(manifest.get("gate_results"), dict) else {}
    null_gate = gate_results.get("null_label_removal") if isinstance(gate_results, dict) else None
    rows_removed = None
    if isinstance(null_gate, dict):
        rows_removed = null_gate.get("rows_removed")
    if not isinstance(rows_removed, (int, float)):
        rows_removed = None

    role_ctx = _resolve_id_role_context(list((manifest.get("null_stats", {}).get("before", {}) or {}).keys()), column_roles)
    target_cols = role_ctx.get("target_like") or []
    null_stats = manifest.get("null_stats") if isinstance(manifest.get("null_stats"), dict) else {}
    before = null_stats.get("before") if isinstance(null_stats.get("before"), dict) else {}
    after = null_stats.get("after") if isinstance(null_stats.get("after"), dict) else {}
    if not target_cols and isinstance(before, dict):
        # Fallback: infer likely supervised targets from null stats (positive nulls cleaned to zero),
        # excluding role-tagged identifier/split columns.
        excluded_cols = set(role_ctx.get("id_like") or []) | set(role_ctx.get("split_like") or [])
        inferred_targets: List[str] = []
        for col, before_val in before.items():
            after_val = after.get(col) if isinstance(after, dict) else None
            if col in excluded_cols:
                continue
            if not isinstance(before_val, (int, float)) or not isinstance(after_val, (int, float)):
                continue
            if float(before_val) > 0 and float(after_val) == 0.0:
                inferred_targets.append(col)
        target_cols = inferred_targets

    if not target_cols:
        return False, "", evidence

    aligned_targets = True
    for col in target_cols:
        before_val = before.get(col)
        after_val = after.get(col)
        if not isinstance(before_val, (int, float)) or not isinstance(after_val, (int, float)):
            aligned_targets = False
            break
        if abs(float(before_val) - dropped) > 1.0 or float(after_val) != 0.0:
            aligned_targets = False
            break

    rows_removed_aligned = rows_removed is not None and abs(float(rows_removed) - dropped) <= 1.0
    if aligned_targets and (rows_removed_aligned or "listwise" in drop_reason):
        evidence["adaptive_policy"] = "label_null_listwise_drop"
        evidence["target_columns"] = target_cols
        evidence["rows_before"] = int(rows_before)
        evidence["rows_after"] = int(rows_after)
        evidence["rows_dropped"] = int(dropped)
        evidence["rows_removed_gate"] = int(rows_removed) if isinstance(rows_removed, (int, float)) else None
        return True, "drop_explained_by_label_null_listwise_removal", evidence

    return False, "", evidence


def _map_status_value(status: Any) -> str | None:
    if status is None:
        return None
    raw = str(status).strip()
    if not raw:
        return None
    if raw in {"APPROVED", "APPROVE_WITH_WARNINGS", "REJECTED"}:
        return raw
    normalized = re.sub(r"[\s\-]+", "_", raw.strip().lower())
    normalized = re.sub(r"_+", "_", normalized)
    if normalized in {"approved", "approve"}:
        return "APPROVED"
    if normalized in {"rejected", "reject", "failed", "fail"}:
        return "REJECTED"
    if "warn" in normalized and "approve" in normalized:
        return "APPROVE_WITH_WARNINGS"
    if normalized in {"approved_with_warning", "approved_with_warnings", "approve_with_warning", "approve_with_warnings"}:
        return "APPROVE_WITH_WARNINGS"
    return None


def normalize_cleaning_reviewer_result(result: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return {
            "status": "REJECTED",
            "feedback": "Cleaning reviewer returned invalid status.",
            "failed_checks": [],
            "required_fixes": [],
        }

    status_raw = result.get("status")
    status_exact = str(status_raw).strip() if status_raw is not None else ""
    mapped = _map_status_value(status_raw)
    normalized_applied = False

    if status_exact in {"APPROVED", "APPROVE_WITH_WARNINGS", "REJECTED"}:
        result["status"] = status_exact
    elif mapped:
        result["status"] = mapped
        normalized_applied = True
    else:
        result["status"] = "REJECTED"

    for field in ["failed_checks", "required_fixes"]:
        val = result.get(field, [])
        if isinstance(val, str):
            result[field] = [val]
        elif not isinstance(val, list):
            result[field] = []
    if "feedback" not in result:
        result["feedback"] = ""

    if normalized_applied:
        if "STATUS_ENUM_NORMALIZED" not in result["failed_checks"]:
            result["failed_checks"].append("STATUS_ENUM_NORMALIZED")

    if not mapped and status_exact not in {"APPROVED", "APPROVE_WITH_WARNINGS", "REJECTED"}:
        essential_missing = not result.get("feedback") and not result["failed_checks"] and not result["required_fixes"]
        if essential_missing:
            result["feedback"] = "Cleaning reviewer returned invalid status."

    if result.get("required_fixes"):
        if result.get("status") in {"APPROVED", "APPROVE_WITH_WARNINGS"}:
            result["status"] = "REJECTED"
            if result["feedback"]:
                result["feedback"] = result["feedback"] + " Status corrected due to required fixes."
            else:
                result["feedback"] = "Status corrected due to required fixes."

    return result
