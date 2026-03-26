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
    required_columns = _resolve_required_columns_for_review(view, manifest=manifest)
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
        dataset_profile=dataset_profile,
        outlier_policy=outlier_policy,
        outlier_report=outlier_report,
        outlier_report_path=outlier_report_path,
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
                    "Analyze this evidence payload. Pay special attention to cleaning_quality_summary "
                    "(if present) for null inflation and possible datetime parsing failures. "
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
    return re.sub(r"[^a-z0-9]+", "_", (name or "").strip().lower()).strip("_")


def _normalize_gate_name(name: Any) -> str:
    if name is None:
        return ""
    key = normalize_gate_name(str(name))
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
    text = str(value).strip().lower()
    return text in {"", "nan", "none", "null", "na"}


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
) -> List[str]:
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
    dataset_profile: Optional[Dict[str, Any]] = None,
    outlier_policy: Optional[Dict[str, Any]] = None,
    outlier_report: Optional[Dict[str, Any]] = None,
    outlier_report_path: Optional[str] = None,
) -> Dict[str, Any]:
    facts: Dict[str, Any] = {}
    facts["cleaned_header"] = cleaned_header[:200]
    facts["required_columns"] = required_columns
    facts["missing_required_columns"] = [c for c in required_columns if c not in cleaned_header]

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
        policy_summary: Dict[str, Any] = {
            "enabled": policy_enabled,
            "apply_stage": outlier_policy.get("apply_stage"),
            "target_columns": outlier_policy.get("target_columns"),
            "report_path": outlier_report_path,
            "report_present": report_present,
        }
        if report_present:
            if outlier_report.get("columns_touched") is not None:
                policy_summary["columns_touched"] = outlier_report.get("columns_touched")
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
        "3. deterministic_gate_results are supporting evidence only. They may help focus attention, but they do NOT override the contract or your evidence-based judgment.\n"
        "4. artifact_obligations is a lossless extraction of artifact bindings already declared in the contract. It introduces no new semantics.\n"
        "5. If sources conflict, preserve contract intent and prefer direct evidence from artifact_obligations/column_resolution_context/cleaning_code/data_profile over shallow pattern matching.\n\n"
        "REVIEW DECISION WORKFLOW (MANDATORY)\n"
        "1. Understand the contract first: what gates were requested, what columns matter, and what would count as a real violation.\n"
        "2. Read the available evidence pack: artifact_obligations, facts, column_resolution_context, data_profile, cleaning_quality_summary, cleaning_code, and deterministic_gate_results.\n"
        "3. For each gate, reason about the gate's intent before deciding pass/fail.\n"
        "4. Reject only when you have contract-relevant evidence of a real violation.\n"
        "5. If evidence is ambiguous or incomplete, prefer PASSED or PASSED_WITH_WARNING reasoning over unsupported failure.\n"
        "6. Keep feedback tied to this dataset and this code path, not generic cleaning advice.\n\n"
        "HIGH-RISK EXAMPLES (GUIDANCE, NOT A SUBSTITUTE FOR REASONING)\n"
        "- no_semantic_rescale: do not infer rescaling from low numeric ranges alone. Look for explicit code evidence such as division by constants, scaler objects, or explicit multiplicative rescaling. If data is already in a low range but no such code exists, that is not a violation.\n"
        "- no_synthetic_data: distinguish between generating synthetic rows/datasets and limited stochastic operations such as noise or imputation support. Reject only when the code clearly fabricates data beyond the contract.\n\n"
        "NULL INFLATION DETECTION (CRITICAL)\n"
        "- The payload may include 'cleaning_quality_summary' with per-column null rates before (raw) and after (cleaned) cleaning.\n"
        "- If any column shows null_inflation_pp > 35, the Data Engineer's parsing likely destroyed valid data.\n"
        "  Common cause: single-pass datetime parsing (e.g., dayfirst=True destroying ISO dates, or dayfirst=False destroying DD-MM-YYYY dates).\n"
        "- Check 'notable_columns' for columns flagged with 'possible_datetime_parsing_failure'.\n"
        "- A broken parser that inflates nulls from ~24% to ~94% is a HARD failure — require multi-stage parsing.\n"
        "- Expected null inflation from placeholders/impossible dates is ~25-30pp, not 70pp.\n\n"
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

    report_columns = _list_str(report.get("columns_touched"))
    if not report_columns:
        report_columns = _list_str(report.get("target_columns"))
    if not report_columns and isinstance(report.get("columns"), list):
        report_columns = _list_str(report.get("columns"))
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
