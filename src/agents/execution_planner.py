import json
import os
import ast
import copy
import math
from typing import Dict, Any, List, Optional, Tuple, Callable
from string import Template
import re
import difflib

from dotenv import load_dotenv
from openai import OpenAI
from src.utils.contract_validation import (
    DEFAULT_DATA_ENGINEER_RUNBOOK,
    DEFAULT_ML_ENGINEER_RUNBOOK,
)
from src.utils.contract_accessors import (
    get_canonical_columns,
    get_clean_dataset_output_path,
    get_cleaning_gates,
    get_column_roles,
    get_declared_artifact_path,
    get_derived_column_names,
    normalize_artifact_path,
    get_qa_gates,
    get_required_outputs,
    get_reviewer_gates,
    CONTRACT_VERSION_V41,
    normalize_contract_version,
)
from src.utils.run_bundle import get_run_dir
from src.utils.feature_selectors import infer_feature_selectors, compact_column_representation
from src.utils.cleaning_contract_semantics import (
    extract_selector_drop_reasons,
    selector_reference_matches_any,
    expand_required_feature_selectors,
)
from src.utils.column_sets import build_column_manifest, summarize_column_sets
from src.utils.contract_validator import (
    validate_contract_minimal_readonly,
    normalize_contract_scope,
    is_probably_path,
    is_file_path,
    _normalize_selector_entry,
    get_default_optimization_policy,
    normalize_optimization_policy,
)
from src.utils.contract_schema_registry import (
    build_contract_schema_examples_text,
    get_contract_schema_repair_action,
    apply_contract_schema_registry_repairs,
)
from src.utils.contract_response_schema import EXECUTION_CONTRACT_V42_MIN_SCHEMA
from src.utils.problem_capabilities import (
    infer_problem_capabilities,
    is_problem_family,
    resolve_problem_capabilities_from_contract,
)

load_dotenv()


_API_KEY_SENTINEL = object()


_QA_SEVERITIES = {"HARD", "SOFT"}
_CLEANING_SEVERITIES = {"HARD", "SOFT"}

# ── Token sets removed (seniority refactoring) ──────────────────────────
# Capability detection (resampling, decisioning, explanation, visualization)
# is now performed semantically by the LLM inside the contract prompt.
# See CAPABILITY_DETECTION_PROMPT below.
# ─────────────────────────────────────────────────────────────────────────

CAPABILITY_DETECTION_PROMPT = """
CAPABILITY DETECTION (reason from objective and strategy, do NOT use keyword matching):
When generating the contract, determine semantically whether the objective requires:
- resampling / cross-validation: Set resampling fields in validation_requirements.
- decisioning / ranking / action output: Set decisioning_requirements with appropriate columns.
- explanations / interpretability: Set explanation columns in scored_rows_schema.
- visualizations: Set visual_requirements in artifact_requirements.
Base your decisions on the MEANING of the business objective and strategy, not on the
presence or absence of specific keywords. Set boolean flags and structured specs in the
contract based on your semantic understanding of what the downstream agents will need.
"""

CONTRACT_SCHEMA_EXAMPLES_TEXT = build_contract_schema_examples_text()

_EXECUTION_CONTRACT_TOOL_NAME = "emit_execution_contract"
_EXECUTION_CONTRACT_PATCH_TOOL_NAME = "emit_execution_contract_patch"
_JSON_PATCH_VALUE_SCHEMA: Dict[str, Any] = {
    "anyOf": [
        {"type": "object", "additionalProperties": True},
        {"type": "array", "items": {}},
        {"type": "string"},
        {"type": "number"},
        {"type": "integer"},
        {"type": "boolean"},
        {"type": "null"},
    ]
}
_EXECUTION_CONTRACT_PATCH_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "changes": {
            "type": "object",
            "additionalProperties": _JSON_PATCH_VALUE_SCHEMA,
            "description": "Minimal nested fields to deep-merge into the current contract.",
        },
        "patch": {
            "type": "array",
            "description": "Optional JSON Patch operations.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "op": {"type": "string", "enum": ["add", "remove", "replace"]},
                    "path": {"type": "string"},
                    "value": _JSON_PATCH_VALUE_SCHEMA,
                },
                "required": ["op", "path"],
            },
        },
    },
}

MINIMAL_CONTRACT_COMPILER_PROMPT = """
You are an Execution Contract Compiler for a multi-agent business intelligence system.

Goal:
- Produce ONE JSON execution contract that downstream agents can execute and review.
- Reason from business objective, strategy, column inventory, and dataset profile to produce an executable contract.
- Obey a minimal stable interface, but do not let the schema turn into slot-filling without semantic justification.

Phased contract compilation protocol (use internally as a guide, not as a rigid checklist):
- Phase 1 FACTS_EXTRACTOR:
  - Extract only grounded facts from provided inputs (strategy/objective/column inventory/profile).
  - Resolve ambiguities conservatively; do not invent columns or artifacts.
- Phase 2 CONTRACT_BUILDER:
  - Turn the grounded facts into executable fields that downstream agents can consume.
  - Keep interfaces consumable by data_engineer, ml_engineer, QA, reviewers, and translator views.
- Phase 3 GATE_COMPOSER:
  - Compose only the gates justified by the contract and risk surface, using stable semantics (name/severity/params).
  - Prefer universal gate primitives; avoid one-off ad hoc gates when an equivalent primitive exists.
- Phase 4 VALIDATOR_REPAIR:
  - Verify schema/semantics and downstream-consumer compatibility.
  - If issues exist, apply minimal edits; do not regenerate unrelated valid sections.

Output discipline:
- Return only the final contract JSON object.
- Do not output phase traces, reasoning notes, or chain-of-thought.

Return format:
- Return ONLY valid JSON (no markdown, no code fences, no comments).

Minimal contract interface:
- scope: one of ["cleaning_only", "ml_only", "full_pipeline"]
  - cleaning_only: Use ONLY for pure ETL/data quality tasks where no modeling, analysis, or prediction is requested.
  - full_pipeline: The STANDARD for predictive/prescriptive constraints. Use this whenever a model or analysis is the goal, even if data is dirty (the pipeline handles cleaning automatically).
  - ml_only: Use only if input data is explicitly stated to be pre-cleaned and trusted.
- strategy_title: string
- business_objective: string
- output_dialect: object (csv sep/decimal/encoding when known)
- canonical_columns: list[str]
- required_outputs: list[str] artifact file paths (REQUIRED items only)
- required_output_artifacts (optional): list[object] rich metadata with:
  - path, required, owner, kind, description, id(optional)
- column_roles: object mapping role -> list[str]
  Role definitions (ML execution context):
  - outcome: ONLY the target variable(s) the model predicts (usually one column).
    Do NOT include ordinary features here even if they are binary/categorical.
  - pre_decision: ALL model input features available before prediction/decision.
    Includes numeric, categorical, binary, and engineered predictor inputs.
    Include standard derived columns (e.g. date parts). Experimental features should go in feature_engineering_tasks, NOT here.
  - decision: Decision/action output columns emitted by policy/model (often empty in contract stage).
    - If strategy implies optimization (e.g. "optimize price"), include the decision variable here.
  - identifiers: Entity keys/ids used for joins/traceability (non-predictive).
  - post_decision_audit_only: Columns used only for post-hoc analysis/compliance.
  CRITICAL: In predictive tasks, outcome MUST contain only the target column(s).
  Non-target attributes belong to pre_decision, not outcomes.
- column_dtype_targets: object mapping column -> dtype spec.
  Each value MUST be an object with key "target_dtype" (NOT "type").
- artifact_requirements: object
- iteration_policy: object with practical retry/iteration limits (small, numeric, and actionable)
- optimization_policy (recommended for v4.2, backward-compatible): object with:
  - enabled, max_rounds, quick_eval_folds, full_eval_folds, min_delta, patience,
    allow_model_switch, allow_ensemble, allow_hpo, allow_feature_engineering, allow_calibration
- outlier_policy (optional): object for robust outlier handling when strategy/data justify it.
  - recommended fields: enabled(bool), apply_stage("data_engineer"|"ml_engineer"|"both"),
    target_columns(list[str]), methods/treatment(object|list), report_path(file path), strict(bool).
  - recommended fields: enabled(bool), apply_stage("data_engineer"|"ml_engineer"|"both"),
    target_columns(list[str]), methods/treatment(object|list), report_path(file path), strict(bool).
- feature_engineering_plan / feature_engineering_tasks (legacy optional):
  - preserve if explicitly provided by upstream context.
  - do not require or invent FE plans at planner stage.
- derived_columns: list[str] OR object mapping new_column_name -> definition/source.
  - REQUIRED only when the contract explicitly declares derived columns.
  - Use to declare columns that do not exist in the raw data but will be created.
  - Downstream normalization may convert this field to a list of derived column names.

Scope-dependent required fields:
- If scope includes cleaning ("cleaning_only" or "full_pipeline"):
  - cleaning_gates: list of gate objects
  - data_engineer_runbook: object/list/string with actionable steps
  - artifact_requirements.clean_dataset.required_columns: list[str]
  - artifact_requirements.clean_dataset.required_feature_selectors (optional): list[object]
    to represent wide feature families compactly (regex/prefix/range/list/all_columns_except)
  - artifact_requirements.clean_dataset.output_path: file path for cleaned CSV
  - artifact_requirements.clean_dataset.output_manifest_path (or manifest_path): file path for cleaning manifest JSON
  - column_dtype_targets must include explicit dtype targets for anchor columns
    (target/split/identifier/time/decision) and selector-level dtype targets for wide families when applicable.
- If any column drop/scaling is requested, declare it explicitly in
  artifact_requirements.clean_dataset.column_transformations with:
    - drop_columns: list[str]
    - scale_columns: list[str]
    - drop_policy (optional but required for selector-based criteria drops): {
        "allow_selector_drops_when": list[str]  # e.g. ["constant","all_null","duplicate"]
      }
    - when referring to a declared selector from scale_columns, use stable selector refs
      like selector:<name>; do NOT emit selector:regex:... or selector:prefix:...
    - if selector-drop policy is active, any required_columns / optional_passthrough_columns /
      HARD-gate columns must remain outside selector coverage as non-droppable anchors
    (do not encode these decisions only in free-text runbook)
- If scope includes ML ("ml_only" or "full_pipeline"):
  - qa_gates: list of gate objects
  - reviewer_gates: list of gate objects
  - validation_requirements: object
  - ml_engineer_runbook: object/list/string with actionable steps
  - evaluation_spec: non-empty object for ML execution/review context
  - objective_analysis.problem_type OR evaluation_spec.objective_type must be present
  - column_dtype_targets must be coherent with evaluation/training expectations (nullable targets where labels can be missing).

Gate object contract (for cleaning_gates / qa_gates / reviewer_gates):
- preferred shape: {"name": string, "severity": "HARD"|"SOFT", "params": object}
- optional semantic keys: "condition", "evidence_required", "action_if_fail"
- avoid anonymous dicts: each gate must have a consumable identifier.
- if you start from semantic wording like metric/check/rule, map it into "name" and keep the original key inside params.

Hard rules:
- Do not invent columns not present in column_inventory.
- required_outputs must be artifact paths, never conceptual labels.
- Keep de_view executable from contract alone: include both cleaned output CSV and cleaning manifest JSON paths.
- Contract precedence for DE must be coherent: HARD cleaning_gates + required_columns + explicit column_transformations
  are binding and must not contradict runbook narrative.
- For full_pipeline contracts, clean_dataset coverage must include ML-required columns
  (outcome/decision/model features), either explicitly in required_columns/optional_passthrough_columns
  or through required_feature_selectors.
- For high-dimensional feature spaces, do not enumerate every feature column in narrative fields.
  Declare feature families with required_feature_selectors and keep explicit columns as anchors
  (target/split/ids/critical business fields).
- Prefer named selectors when using required_feature_selectors so downstream transforms can
  refer to them canonically via selector:<name>.
- Keep contract coherent with strategy + business objective.
- Treat strategy techniques as advisory hypotheses, not immutable requirements.
- If strategy wording conflicts with direct data evidence (profile ranges/dtypes/semantics), prioritize data evidence
  and encode the safer choice in contract fields.
- If observed target values are already numeric-binary (e.g., 0/1), do NOT add a label-to-number
  target_mapping_check gate or runbook instructions that remap textual labels.
- Avoid declaring dtype constraints that conflict with observed ranges (e.g., signed int8 when observed values exceed
  [-128, 127]).
- Every requirement must be consumable by at least one downstream agent view.
- Follow evidence_policy from INPUTS: use direct evidence first, grounded inference second, and avoid unsupported assumptions.
- You may add extra fields if useful, but do not omit required minimum fields.

CAPABILITY DETECTION (reason from objective and strategy, do NOT use keyword matching):
When generating the contract, determine semantically whether the objective requires:
- resampling / cross-validation: Set resampling fields in validation_requirements.
- decisioning / ranking / action output: Set decisioning_requirements with appropriate columns.
- explanations / interpretability: Set explanation columns in scored_rows_schema.
- visualizations: Set visual_requirements in artifact_requirements.
Base your decisions on the MEANING of the business objective and strategy, not on the
presence or absence of specific keywords. Set boolean flags and structured specs in the
contract based on your semantic understanding of what the downstream agents will need.
"""

CONTRACT_SOURCE_OF_TRUTH_POLICY_V1 = {
    "contract_is_single_source_of_truth": True,
    "views_are_projection_only": True,
    "runtime_must_not_inject_out_of_contract_requirements": True,
    "reviewers_validate_against_contract_plus_execution_evidence": True,
}

DOWNSTREAM_CONSUMER_INTERFACE_V1 = {
    "data_engineer": {
        "consumes_view": "de_view",
        "must_be_executable_from_contract": True,
        "required_contract_inputs": [
            "scope",
            "artifact_requirements.clean_dataset.output_path",
            "artifact_requirements.clean_dataset.output_manifest_path|manifest_path",
            "artifact_requirements.clean_dataset.required_columns",
            "artifact_requirements.clean_dataset.required_feature_selectors (optional)",
            "artifact_requirements.clean_dataset.column_transformations (optional)",
            "artifact_requirements.clean_dataset.column_transformations.drop_policy (optional)",
            "column_dtype_targets",
            "cleaning_gates",
            "data_engineer_runbook",
            "outlier_policy (optional)",
        ],
    },
    "cleaning_reviewer": {
        "consumes_view": "cleaning_review_view",
        "required_contract_inputs": [
            "scope",
            "cleaning_gates",
            "artifact_requirements.clean_dataset.output_path",
            "artifact_requirements.clean_dataset.output_manifest_path|manifest_path",
            "artifact_requirements.clean_dataset.required_columns",
            "artifact_requirements.clean_dataset.required_feature_selectors (optional)",
            "artifact_requirements.clean_dataset.column_transformations (optional)",
            "artifact_requirements.clean_dataset.column_transformations.drop_policy (optional)",
            "outlier_policy (optional)",
        ],
    },
    "ml_engineer": {
        "consumes_view": "ml_view",
        "required_contract_inputs": [
            "scope",
            "ml_engineer_runbook",
            "validation_requirements",
            "qa_gates",
            "reviewer_gates",
            "required_outputs",
            "column_dtype_targets",
        ],
        "objective_context_required": [
            "objective_analysis.problem_type|evaluation_spec.objective_type",
            "business_objective",
            "strategy_title",
        ],
    },
    "reviewer_board": {
        "consumes_view": "review_board_view",
        "required_contract_inputs": [
            "qa_gates",
            "reviewer_gates",
            "validation_requirements",
            "required_outputs",
            "artifact_requirements",
        ],
    },
    "business_translator": {
        "consumes_view": "translator_view",
        "required_contract_inputs": [
            "strategy_title",
            "business_objective",
            "required_outputs",
            "artifact_requirements",
        ],
    },
}

CONTRACT_EVIDENCE_POLICY_V1 = {
    "priority_order": ["direct_evidence", "grounded_inference", "assumption_last_resort"],
    "direct_evidence": "Field value is explicitly present in provided inputs.",
    "grounded_inference": (
        "Field value is deduced from strategy + business objective + column inventory + data profile; "
        "no invented columns, paths, or unsupported claims. "
        "EXCEPTION: derived_columns are ALLOWED if supported by strategy.feature_engineering "
        "(or strategy.evaluation_plan.feature_engineering)."
    ),
    "assumption_last_resort": (
        "Use only when required field cannot be left empty. Keep assumption conservative and record it under "
        "planner_notes.assumptions (optional)."
    ),
    "if_evidence_missing_for_optional_field": "omit_field_or_leave_empty_object",
    "if_evidence_missing_for_required_field": "use_minimal_safe_default_and_mark_assumption",
    "forbidden": [
        "invented_column_names",
        "conceptual_items_in_required_outputs",
        "requirements_without_downstream_consumer",
    ],
}

PLANNER_SEMANTIC_RESOLUTION_POLICY_V1 = {
    "strategy_techniques_role": "advisory_hypotheses",
    "conflict_resolution_priority": [
        "data_profile_compact_json",
        "dataset_semantics",
        "strategy_text",
    ],
    "dtype_semantic_guardrail": (
        "Choose dtype constraints compatible with observed value ranges; avoid narrow integer dtypes when out of range."
    ),
}

PHASED_CONTRACT_COMPILATION_PROTOCOL_V1 = {
    "phase_1_facts_extractor": {
        "goal": "derive grounded facts from input evidence only",
        "rules": [
            "no invented columns",
            "no invented artifact paths",
            "prefer direct evidence over inference",
        ],
    },
    "phase_2_contract_builder": {
        "goal": "populate required schema keys with scope-aware constraints",
        "rules": [
            "preserve downstream consumer compatibility",
            "use stable canonical field shapes",
            "keep required_outputs as list[str] paths",
        ],
    },
    "phase_3_gate_composer": {
        "goal": "emit executable gates with stable semantics",
        "rules": [
            "gate object shape: name/severity/params",
            "prefer reusable universal gate primitives",
            "bind gate params to evidence-backed columns/thresholds",
        ],
    },
    "phase_4_validator_repair": {
        "goal": "self-validate and patch minimally before return",
        "rules": [
            "fail-closed for schema/semantic violations",
            "minimal-diff repair over full regeneration",
            "preserve already-valid sections",
        ],
    },
}


def _normalize_text(*values: Any) -> str:
    tokens: List[str] = []
    for raw in values:
        if raw is None:
            continue
        if isinstance(raw, str):
            text = raw
        else:
            text = json.dumps(raw, ensure_ascii=False)
        cleaned = re.sub(r"[^0-9a-zA-ZÁÉÍÓÚáéíóúüÜñÑ]+", " ", text.lower())
        tokens.extend(cleaned.split())
    return " ".join(token for token in tokens if token)


def _extract_required_paths(artifact_requirements: Dict[str, Any]) -> List[str]:
    if not isinstance(artifact_requirements, dict):
        return []
    required_files = artifact_requirements.get("required_files")
    if not isinstance(required_files, list):
        return []
    paths: List[str] = []
    for entry in required_files:
        if not entry:
            continue
        if isinstance(entry, dict):
            path = entry.get("path") or entry.get("output") or entry.get("artifact")
        else:
            path = entry
        if path and is_probably_path(str(path)):
            paths.append(str(path))
    return paths


def _normalize_artifact_path(value: Any) -> str:
    if value is None:
        return ""
    path = str(value).strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    path = path.lstrip("/")
    path = re.sub(r"/{2,}", "/", path)
    return path


def _extract_file_like_outputs(raw_outputs: Any) -> List[str]:
    if not isinstance(raw_outputs, list):
        return []
    paths: List[str] = []
    seen: set[str] = set()
    for item in raw_outputs:
        candidate = ""
        if isinstance(item, dict):
            for key in ("path", "file", "output", "artifact"):
                value = item.get(key)
                normalized = _normalize_artifact_path(value)
                if normalized and is_probably_path(normalized):
                    candidate = normalized
                    break
        else:
            normalized = _normalize_artifact_path(item)
            if normalized and is_probably_path(normalized):
                candidate = normalized
        if not candidate:
            continue
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        paths.append(candidate)
    return paths


def normalize_artifact_requirements(contract: Dict[str, Any]) -> Dict[str, Any]:
    """
    Align required_outputs with artifact_requirements.required_files.
    Only file-like outputs are mirrored (universal path heuristics).
    """
    if not isinstance(contract, dict):
        return contract

    artifact_requirements = contract.get("artifact_requirements")
    if not isinstance(artifact_requirements, dict):
        artifact_requirements = {}

    required_files_raw = artifact_requirements.get("required_files")
    required_files_list = required_files_raw if isinstance(required_files_raw, list) else []
    use_dict_entries = any(isinstance(entry, dict) for entry in required_files_list)

    normalized_required_files: List[Any] = []
    seen_paths: set[str] = set()
    for entry in required_files_list:
        if isinstance(entry, dict):
            raw_path = entry.get("path") or entry.get("output") or entry.get("artifact")
            normalized_path = _normalize_artifact_path(raw_path)
            if not normalized_path or not is_probably_path(normalized_path):
                continue
            key = normalized_path.lower()
            if key in seen_paths:
                continue
            seen_paths.add(key)
            normalized_entry = dict(entry)
            normalized_entry["path"] = normalized_path
            normalized_required_files.append(normalized_entry if use_dict_entries else normalized_path)
        else:
            normalized_path = _normalize_artifact_path(entry)
            if not normalized_path or not is_probably_path(normalized_path):
                continue
            key = normalized_path.lower()
            if key in seen_paths:
                continue
            seen_paths.add(key)
            if use_dict_entries:
                normalized_required_files.append(
                    {"path": normalized_path, "description": "Auto-normalized required file path"}
                )
            else:
                normalized_required_files.append(normalized_path)

    required_output_paths = _extract_file_like_outputs(contract.get("required_outputs"))
    evaluation_spec = contract.get("evaluation_spec") if isinstance(contract.get("evaluation_spec"), dict) else {}
    evaluation_required_paths = _extract_file_like_outputs(evaluation_spec.get("required_outputs"))

    for path in required_output_paths + evaluation_required_paths:
        key = path.lower()
        if key in seen_paths:
            continue
        seen_paths.add(key)
        if use_dict_entries:
            normalized_required_files.append(
                {
                    "path": path,
                    "description": "Auto-added to align with required_outputs",
                }
            )
        else:
            normalized_required_files.append(path)

    artifact_requirements["required_files"] = normalized_required_files
    contract["artifact_requirements"] = artifact_requirements
    return contract


def _apply_planner_structural_support(contract: Dict[str, Any] | None) -> Dict[str, Any]:
    """
    Apply only non-semantic structural support to planner output.

    This helper is intentionally narrow: it may normalize contract shape and
    artifact path plumbing, but it must not infer/repair scope, targets,
    deliverables, gates, or other semantic choices that belong to the LLM.
    """
    if not isinstance(contract, dict):
        return {}
    supported = ensure_v41_schema(copy.deepcopy(contract))
    supported = normalize_artifact_requirements(supported)
    return supported


def _normalize_strategy_feature_engineering_payload(strategy: Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(strategy, dict):
        return {"techniques": [], "notes": "", "risk_level": "low"}

    raw = strategy.get("feature_engineering_strategy")
    techniques: List[Any] = []
    notes = ""
    risk_level = "low"

    if isinstance(raw, dict):
        candidate = raw.get("techniques")
        if isinstance(candidate, list):
            techniques = list(candidate)
        raw_notes = raw.get("notes")
        if isinstance(raw_notes, str):
            notes = raw_notes.strip()
        raw_risk = str(raw.get("risk_level") or "").strip().lower()
        if raw_risk in {"low", "med", "high"}:
            risk_level = raw_risk
        elif raw_risk == "medium":
            risk_level = "med"
    elif isinstance(raw, list):
        techniques = list(raw)

    if not techniques:
        candidate = strategy.get("feature_engineering")
        if isinstance(candidate, list):
            techniques = list(candidate)

    if not techniques:
        eval_plan = strategy.get("evaluation_plan")
        if isinstance(eval_plan, dict):
            candidate = eval_plan.get("feature_engineering")
            if isinstance(candidate, list):
                techniques = list(candidate)

    return {
        "techniques": techniques if isinstance(techniques, list) else [],
        "notes": notes,
        "risk_level": risk_level,
    }


def _extract_derived_columns_from_fe_techniques(techniques: List[Any]) -> List[str]:
    derived: List[str] = []
    seen: set[str] = set()
    if not isinstance(techniques, list):
        return derived
    for item in techniques:
        if not isinstance(item, dict):
            continue
        candidate = (
            item.get("output_column_name")
            or item.get("output_column")
            or item.get("derived_column")
            or item.get("name")
        )
        if not isinstance(candidate, str):
            continue
        name = candidate.strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        derived.append(name)
    return derived


def _ensure_feature_engineering_plan_from_strategy(
    contract: Dict[str, Any],
    strategy: Dict[str, Any] | None,
) -> Dict[str, Any]:
    if not isinstance(contract, dict):
        return contract

    normalized_fe = _normalize_strategy_feature_engineering_payload(strategy)
    existing = contract.get("feature_engineering_plan")
    existing_present = "feature_engineering_plan" in contract
    if not isinstance(existing, dict):
        existing = {}

    techniques = normalized_fe.get("techniques") if isinstance(normalized_fe.get("techniques"), list) else []
    notes_from_strategy = str(normalized_fe.get("notes") or "").strip()
    raw_constraints = strategy.get("feature_engineering_constraints") if isinstance(strategy, dict) else None
    constraints_from_strategy = raw_constraints if isinstance(raw_constraints, dict) and raw_constraints else {}
    if (
        not existing_present
        and not techniques
        and not notes_from_strategy
        and not constraints_from_strategy
    ):
        # Legacy field is optional: keep absent unless strategy/contracts explicitly provide FE content.
        return contract

    existing_derived_raw = existing.get("derived_columns")
    existing_entries: List[Any] = list(existing_derived_raw) if isinstance(existing_derived_raw, list) else []
    existing_names = _extract_derived_column_names(existing_entries)
    inferred_derived = _extract_derived_columns_from_fe_techniques(techniques)
    seen_derived: set[str] = {str(name).strip().lower() for name in existing_names if str(name).strip()}
    merged_derived_entries: List[Any] = list(existing_entries)
    for name in inferred_derived:
        candidate = str(name or "").strip()
        if not candidate:
            continue
        key = candidate.lower()
        if key in seen_derived:
            continue
        seen_derived.add(key)
        merged_derived_entries.append(candidate)
    if not merged_derived_entries:
        merged_derived_entries = []

    notes = str(existing.get("notes") or notes_from_strategy or "").strip()
    constraints = existing.get("constraints")
    if not isinstance(constraints, dict):
        constraints = {}
    if constraints_from_strategy:
        constraints = dict(constraints_from_strategy)

    contract["feature_engineering_plan"] = {
        "techniques": techniques,
        "derived_columns": merged_derived_entries,
        "constraints": constraints,
        "notes": notes,
    }
    return contract


def _ensure_optimization_policy(contract: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(contract, dict):
        return contract
    contract["optimization_policy"] = normalize_optimization_policy(contract.get("optimization_policy"))
    return contract


def _merge_scored_rows_schema(
    base_schema: Dict[str, Any] | None,
    incoming_schema: Dict[str, Any] | None,
) -> Dict[str, Any]:
    base = base_schema if isinstance(base_schema, dict) else {}
    incoming = incoming_schema if isinstance(incoming_schema, dict) else {}
    if not base and not incoming:
        return {}
    if not base:
        return dict(incoming)
    if not incoming:
        return dict(base)

    def _merge_list(key: str) -> List[str]:
        merged: List[str] = []
        seen: set[str] = set()
        for item in (base.get(key, []) or []) + (incoming.get(key, []) or []):
            if item is None:
                continue
            val = str(item)
            if not val or val in seen:
                continue
            seen.add(val)
            merged.append(val)
        return merged

    def _merge_groups(key: str) -> List[List[str]]:
        merged: List[List[str]] = []
        seen: set[tuple[str, ...]] = set()
        for group in (base.get(key, []) or []) + (incoming.get(key, []) or []):
            if not isinstance(group, list) or not group:
                continue
            normalized = tuple(sorted({str(item).lower() for item in group if item}))
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            merged.append([str(item) for item in group if item])
        return merged

    merged = dict(base)
    merged["required_columns"] = _merge_list("required_columns")
    merged["recommended_columns"] = _merge_list("recommended_columns")
    merged["required_any_of_groups"] = _merge_groups("required_any_of_groups")
    if base.get("required_any_of_group_severity"):
        merged["required_any_of_group_severity"] = list(base.get("required_any_of_group_severity") or [])
    elif incoming.get("required_any_of_group_severity"):
        merged["required_any_of_group_severity"] = list(incoming.get("required_any_of_group_severity") or [])
    return merged


def _merge_unique_values(values: List[str], extras: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item in values + extras:
        if not item:
            continue
        text = str(item)
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _normalize_column_token(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "", str(name or "").lower())


def _extract_decisioning_required_column_names(decisioning: Dict[str, Any] | None) -> List[str]:
    if not isinstance(decisioning, dict):
        return []
    if decisioning.get("required") is not True:
        return []
    output = decisioning.get("output")
    if not isinstance(output, dict):
        return []
    required_columns = output.get("required_columns")
    if not isinstance(required_columns, list):
        return []
    names: List[str] = []
    for entry in required_columns:
        if not entry:
            continue
        if isinstance(entry, dict):
            name = entry.get("name") or entry.get("column")
        else:
            name = entry
        if name:
            names.append(str(name))
    return names


def _is_prediction_like_column(name: str) -> bool:
    token = _normalize_column_token(name)
    if not token:
        return False
    return any(
        key in token
        for key in (
            "pred",
            "prob",
            "score",
            "risk",
            "likelihood",
            "chance",
        )
    )


def _align_decisioning_requirements_with_schema(
    decisioning: Dict[str, Any] | None,
    scored_rows_schema: Dict[str, Any] | None,
) -> Dict[str, Any]:
    if not isinstance(decisioning, dict):
        return {}
    if not isinstance(scored_rows_schema, dict):
        return decisioning
    required_cols = scored_rows_schema.get("required_columns")
    if not isinstance(required_cols, list) or not required_cols:
        return decisioning

    explanation_name = None
    for col in required_cols:
        if _normalize_column_token(col) == "explanation":
            explanation_name = str(col)
            break
    if explanation_name is None:
        for col in required_cols:
            if "driver" in _normalize_column_token(col):
                explanation_name = str(col)
                break
    if explanation_name is None:
        return decisioning

    output = decisioning.get("output")
    if not isinstance(output, dict):
        return decisioning
    required = output.get("required_columns")
    if not isinstance(required, list) or not required:
        return decisioning

    updated: List[Any] = []
    touched = False
    for entry in required:
        if isinstance(entry, dict):
            name = entry.get("name")
            role = str(entry.get("role") or "").lower()
            if role == "explanation" or _normalize_column_token(name) in {"explanation", "topdrivers", "topdriver"}:
                updated_entry = dict(entry)
                updated_entry["name"] = explanation_name
                updated.append(updated_entry)
                touched = True
            else:
                updated.append(entry)
        else:
            name = str(entry)
            if _normalize_column_token(name) in {"explanation", "topdrivers", "topdriver"}:
                updated.append(explanation_name)
                touched = True
            else:
                updated.append(entry)
    if not touched:
        return decisioning
    aligned = dict(decisioning)
    new_output = dict(output)
    new_output["required_columns"] = updated
    aligned["output"] = new_output
    return aligned


def _sync_execution_contract_outputs(contract: Dict[str, Any], contract_min: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(contract, dict) or not isinstance(contract_min, dict):
        return contract

    def _extract_output_path(item: Any) -> str:
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            for key in ("path", "output", "artifact"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    return value
        return ""

    required_outputs = contract.get("required_outputs")
    required_outputs_list = (
        [path for path in (_extract_output_path(item) for item in required_outputs) if path]
        if isinstance(required_outputs, list)
        else []
    )
    has_conceptual = any(item and not is_probably_path(item) for item in required_outputs_list)

    min_required_outputs = contract_min.get("required_outputs")
    min_required_outputs_list = (
        [path for path in (_extract_output_path(item) for item in min_required_outputs) if path and is_probably_path(path)]
        if isinstance(min_required_outputs, list)
        else []
    )

    if has_conceptual and min_required_outputs_list:
        contract["required_outputs"] = min_required_outputs_list
    elif not required_outputs_list and min_required_outputs_list:
        contract["required_outputs"] = min_required_outputs_list

    contract_artifacts = contract.get("artifact_requirements")
    if not isinstance(contract_artifacts, dict):
        contract_artifacts = {}
    min_artifacts = contract_min.get("artifact_requirements")
    if not isinstance(min_artifacts, dict):
        min_artifacts = {}

    min_required_files = _extract_required_paths(min_artifacts)
    contract_required_files = _extract_required_paths(contract_artifacts)
    if min_required_files:
        merged_files: List[Dict[str, Any]] = []
        seen = {path.lower() for path in contract_required_files if path}
        for path in contract_required_files:
            merged_files.append({"path": path, "description": ""})
        for path in min_required_files:
            key = path.lower()
            if key in seen:
                continue
            seen.add(key)
            merged_files.append({"path": path, "description": ""})
        if merged_files:
            contract_artifacts["required_files"] = merged_files
            contract["artifact_requirements"] = contract_artifacts

    contract_scored_schema = contract_artifacts.get("scored_rows_schema")
    min_scored_schema = min_artifacts.get("scored_rows_schema")
    merged_scored_schema = _merge_scored_rows_schema(contract_scored_schema, min_scored_schema)
    if merged_scored_schema:
        contract_artifacts["scored_rows_schema"] = merged_scored_schema
        contract["artifact_requirements"] = contract_artifacts

    merged_outputs: List[str] = []
    seen_outputs: set[str] = set()
    for item in (contract.get("required_outputs") or []) + min_required_files:
        path = _extract_output_path(item)
        if not path or not is_probably_path(path):
            continue
        if path in seen_outputs:
            continue
        seen_outputs.add(path)
        merged_outputs.append(path)
    if merged_outputs:
        contract["required_outputs"] = merged_outputs

    return contract


def _compress_text_preserve_ends(
    text: str,
    max_chars: int = 1800,
    head: int = 900,
    tail: int = 900,
) -> str:
    if not isinstance(text, str):
        return ""
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    head_len = max(0, min(head, max_chars))
    tail_len = max(0, min(tail, max_chars - head_len))
    if head_len + tail_len == 0:
        return text[:max_chars]
    if head_len + tail_len < max_chars:
        head_len = max_chars - tail_len
    return text[:head_len] + "\n...\n" + text[-tail_len:]



# _matches_any_phrase and _contains_decisioning_token removed (seniority refactoring):
# capability detection is now LLM-driven via contract prompt.


def _normalize_qa_gate_spec(item: Any) -> Dict[str, Any] | None:
    if isinstance(item, dict):
        name = (
            item.get("name")
            or item.get("id")
            or item.get("gate")
            or item.get("metric")
            or item.get("check")
            or item.get("rule")
            or item.get("title")
            or item.get("label")
        )
        if not name:
            return None
        severity = item.get("severity")
        required = item.get("required")
        if severity is None and required is not None:
            severity = "HARD" if bool(required) else "SOFT"
        severity = str(severity).upper() if severity else "HARD"
        if severity not in _QA_SEVERITIES:
            severity = "HARD"
        params = item.get("params")
        if not isinstance(params, dict):
            params = {}
        for param_key in ("metric", "check", "rule", "threshold", "target", "min", "max", "operator", "direction", "condition"):
            if param_key in item and param_key not in params:
                params[param_key] = item.get(param_key)
        gate_spec: Dict[str, Any] = {"name": str(name), "severity": severity, "params": params}
        for extra_key in ("condition", "evidence_required", "action_if_fail"):
            if extra_key in item:
                gate_spec[extra_key] = item.get(extra_key)
        return gate_spec
    if isinstance(item, str):
        name = item.strip()
        if not name:
            return None
        return {"name": name, "severity": "HARD", "params": {}}
    return None


def _normalize_qa_gates(raw_gates: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_gates, list):
        return []
    normalized: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw_gates:
        spec = _normalize_qa_gate_spec(item)
        if not spec:
            continue
        key = spec["name"].strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        normalized.append(spec)
    return normalized


def _normalize_cleaning_gate_spec(item: Any) -> Dict[str, Any] | None:
    if isinstance(item, dict):
        name = (
            item.get("name")
            or item.get("id")
            or item.get("gate")
            or item.get("metric")
            or item.get("check")
            or item.get("rule")
            or item.get("title")
            or item.get("label")
        )
        if not name:
            return None
        severity = item.get("severity")
        required = item.get("required")
        if severity is None and required is not None:
            severity = "HARD" if bool(required) else "SOFT"
        severity = str(severity).upper() if severity else "HARD"
        if severity not in _CLEANING_SEVERITIES:
            severity = "HARD"
        params = item.get("params")
        if not isinstance(params, dict):
            params = {}
        for param_key in ("metric", "check", "rule", "threshold", "target", "min", "max", "operator", "direction", "condition"):
            if param_key in item and param_key not in params:
                params[param_key] = item.get(param_key)
        gate_spec: Dict[str, Any] = {"name": str(name), "severity": severity, "params": params}
        for extra_key in ("condition", "evidence_required", "action_if_fail"):
            if extra_key in item:
                gate_spec[extra_key] = item.get(extra_key)
        return gate_spec
    if isinstance(item, str):
        name = item.strip()
        if not name:
            return None
        return {"name": name, "severity": "HARD", "params": {}}
    return None


def _normalize_cleaning_gates(raw_gates: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_gates, list):
        return []
    normalized: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw_gates:
        spec = _normalize_cleaning_gate_spec(item)
        if not spec:
            continue
        key = spec["name"].strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        normalized.append(spec)
    return normalized


def _build_default_cleaning_gates() -> List[Dict[str, Any]]:
    return [
        {"name": "required_columns_present", "severity": "HARD", "params": {}},
        {
            "name": "id_integrity",
            "severity": "HARD",
            "params": {
                "identifier_name_regex": (
                    r"(?i)(^id$|"
                    r"(?:^|[_\W])(?:id|entity|cod|code|key|partida|invoice|account)(?:[_\W]|$)|"
                    r"(?:_id$)|(?:^id_))"
                ),
                "detect_scientific_notation": True,
            },
        },
        {
            "name": "no_semantic_rescale",
            "severity": "HARD",
            "params": {
                "allow_percent_like_only": True,
                "percent_like_name_regex": r"(?i)%|pct|percent|plazo",
            },
        },
        {"name": "no_synthetic_data", "severity": "HARD", "params": {}},
        {
            "name": "row_count_sanity",
            "severity": "SOFT",
            "params": {"max_drop_pct": 5.0, "max_dup_increase_pct": 1.0},
        },
        {
            "name": "feature_coverage_sanity",
            "severity": "SOFT",
            "params": {
                "min_feature_count": 3,
                "check_against": "data_atlas",
            },
        },
    ]


def _apply_cleaning_gate_policy(raw_gates: Any) -> List[Dict[str, Any]]:
    gates = _normalize_cleaning_gates(raw_gates)
    default_gates = _build_default_cleaning_gates()
    if not gates:
        return default_gates
    existing = {_normalize_gate_name(gate.get("name")) for gate in gates if isinstance(gate, dict)}
    merged = list(gates)
    for gate in default_gates:
        if not isinstance(gate, dict):
            continue
        name = _normalize_gate_name(gate.get("name"))
        if name and name not in existing:
            merged.append(gate)
            existing.add(name)
    return merged


_DEFAULT_MISSING_CATEGORY_VALUE = "__MISSING__"
_UNSAFE_MISSING_CATEGORY_VALUES = {
    "",
    "none",
    "null",
    "nan",
    "na",
    "n/a",
    "nil",
}


def _normalize_missing_category_value(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    if cleaned.strip().lower() in _UNSAFE_MISSING_CATEGORY_VALUES:
        return None
    return cleaned


def _normalize_gate_name(name: Any) -> str:
    if not name:
        return ""
    text = str(name).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text


_TARGET_BINARY_NUMERIC_TOKENS = {"0", "1"}
_TARGET_NULL_TOKENS = {"", "nan", "none", "null", "na", "n/a", "<na>"}
_TARGET_MAPPING_GATE_NAMES = {
    "target_mapping_check",
    "target_mapping",
    "target_label_mapping",
    "target_map_check",
}
_TARGET_MAPPING_LABEL_HINTS = (
    "presence",
    "absence",
    "positive",
    "negative",
    "yes",
    "no",
    "true",
    "false",
)


def _normalize_target_value_token(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().strip("\"'")
    if not text:
        return ""
    lowered = text.lower()
    if lowered in _TARGET_NULL_TOKENS:
        return ""
    try:
        number = float(text)
    except Exception:
        return lowered
    if not math.isfinite(number):
        return ""
    rounded = round(number)
    if abs(number - rounded) < 1e-9:
        return str(int(rounded))
    return str(number)


def _collect_contract_target_candidates(contract: Dict[str, Any], data_profile: Dict[str, Any] | None = None) -> List[str]:
    candidates: List[str] = []

    def _append(values: Any) -> None:
        if isinstance(values, list):
            for item in values:
                _append(item)
            return
        if isinstance(values, (str, int, float)):
            col = str(values).strip()
            if col and col not in candidates:
                candidates.append(col)

    _append(contract.get("target_column"))
    _append(contract.get("target_columns"))
    _append(contract.get("outcome_columns"))
    roles = contract.get("column_roles")
    if isinstance(roles, dict):
        _append(roles.get("outcome"))
    objective_analysis = contract.get("objective_analysis")
    if isinstance(objective_analysis, dict):
        _append(objective_analysis.get("primary_target"))
        _append(objective_analysis.get("primary_targets"))
        _append(objective_analysis.get("target_column"))
        _append(objective_analysis.get("target_columns"))
        _append(objective_analysis.get("label_column"))
        _append(objective_analysis.get("label_columns"))
    evaluation_spec = contract.get("evaluation_spec")
    if isinstance(evaluation_spec, dict):
        _append(evaluation_spec.get("primary_target"))
        _append(evaluation_spec.get("primary_targets"))
        _append(evaluation_spec.get("target_column"))
        _append(evaluation_spec.get("target_columns"))
        _append(evaluation_spec.get("label_column"))
        _append(evaluation_spec.get("label_columns"))
        params = evaluation_spec.get("params")
        if isinstance(params, dict):
            _append(params.get("primary_target"))
            _append(params.get("target_column"))
            _append(params.get("target_columns"))
    if isinstance(data_profile, dict):
        outcome_analysis = data_profile.get("outcome_analysis")
        if isinstance(outcome_analysis, dict):
            for key in outcome_analysis.keys():
                if isinstance(key, str):
                    _append(key)
    return candidates


def _collect_profile_discrete_target_values(data_profile: Dict[str, Any], target_col: str) -> List[str]:
    if not isinstance(data_profile, dict) or not target_col:
        return []
    tokens: List[str] = []

    def _add(value: Any) -> None:
        token = _normalize_target_value_token(value)
        if token and token not in tokens:
            tokens.append(token)

    cardinality = data_profile.get("cardinality")
    if isinstance(cardinality, dict):
        target_card = cardinality.get(target_col)
        if isinstance(target_card, dict):
            top_values = target_card.get("top_values")
            if isinstance(top_values, list):
                for item in top_values:
                    if isinstance(item, dict):
                        _add(item.get("value"))
                    else:
                        _add(item)
            unique_values = target_card.get("unique_values_sample")
            if isinstance(unique_values, list):
                for item in unique_values:
                    if isinstance(item, dict):
                        _add(item.get("value"))
                    else:
                        _add(item)

    distributions = data_profile.get("distributions")
    if isinstance(distributions, dict):
        target_dist = distributions.get(target_col)
        if isinstance(target_dist, dict):
            top_values = target_dist.get("value_counts_top")
            if isinstance(top_values, list):
                for item in top_values:
                    if isinstance(item, dict):
                        _add(item.get("value"))
                    else:
                        _add(item)
            unique_values = target_dist.get("unique_values_sample")
            if isinstance(unique_values, list):
                for item in unique_values:
                    if isinstance(item, dict):
                        _add(item.get("value"))
                    else:
                        _add(item)

    outcome_analysis = data_profile.get("outcome_analysis")
    if isinstance(outcome_analysis, dict):
        target_outcome = outcome_analysis.get(target_col)
        if isinstance(target_outcome, dict):
            unique_values = target_outcome.get("unique_values_sample")
            if isinstance(unique_values, list):
                for item in unique_values:
                    if isinstance(item, dict):
                        _add(item.get("value"))
                    else:
                        _add(item)
    return tokens


def _build_target_observed_values_map(
    contract: Dict[str, Any] | None,
    data_profile: Dict[str, Any] | None,
) -> Dict[str, List[str]]:
    if not isinstance(contract, dict) or not isinstance(data_profile, dict):
        return {}
    observed: Dict[str, List[str]] = {}
    targets = _collect_contract_target_candidates(contract, data_profile)
    for target_col in targets:
        values = _collect_profile_discrete_target_values(data_profile, target_col)
        if values:
            observed[target_col] = values
    return observed


def _profile_numeric_binary_hint(data_profile: Dict[str, Any], target_col: str) -> bool:
    if not isinstance(data_profile, dict) or not target_col:
        return False
    n_unique: Optional[int] = None
    outcome_analysis = data_profile.get("outcome_analysis")
    if isinstance(outcome_analysis, dict):
        target_outcome = outcome_analysis.get(target_col)
        if isinstance(target_outcome, dict):
            raw_unique = target_outcome.get("n_unique")
            try:
                n_unique = int(raw_unique) if raw_unique is not None else None
            except Exception:
                n_unique = None
    if n_unique is None:
        cardinality = data_profile.get("cardinality")
        if isinstance(cardinality, dict):
            target_card = cardinality.get(target_col)
            if isinstance(target_card, dict):
                raw_unique = target_card.get("unique")
                try:
                    n_unique = int(raw_unique) if raw_unique is not None else None
                except Exception:
                    n_unique = None
    if n_unique is None or n_unique > 2:
        return False

    numeric_summary = data_profile.get("numeric_summary")
    if not isinstance(numeric_summary, dict):
        return False
    target_numeric = numeric_summary.get(target_col)
    if not isinstance(target_numeric, dict):
        return False
    try:
        min_val = float(target_numeric.get("min"))
        max_val = float(target_numeric.get("max"))
    except Exception:
        return False
    if not (math.isfinite(min_val) and math.isfinite(max_val)):
        return False
    return min_val >= 0.0 and max_val <= 1.0


def _is_target_mapping_gate(gate: Dict[str, Any]) -> bool:
    if not isinstance(gate, dict):
        return False
    name = (
        gate.get("name")
        or gate.get("id")
        or gate.get("gate")
        or gate.get("rule")
        or gate.get("check")
    )
    normalized = _normalize_gate_name(name)
    if not normalized:
        return False
    if normalized in _TARGET_MAPPING_GATE_NAMES:
        return True
    return "target" in normalized and "mapping" in normalized


def _gate_uses_non_numeric_target_labels(gate: Dict[str, Any]) -> bool:
    if not isinstance(gate, dict):
        return False
    params = gate.get("params")
    mapping = None
    if isinstance(params, dict):
        mapping = params.get("mapping")
    if mapping is None:
        mapping = gate.get("mapping")
    if not isinstance(mapping, dict) or not mapping:
        return True
    labels = [_normalize_target_value_token(key) for key in mapping.keys()]
    labels = [label for label in labels if label]
    if not labels:
        return True
    return any(label not in _TARGET_BINARY_NUMERIC_TOKENS for label in labels)


def _sanitize_target_mapping_runbook_text(text: str) -> Tuple[str, bool]:
    if not isinstance(text, str) or not text.strip():
        return text, False
    chunks = re.split(r"(?<=[\.\n])\s+", text)
    kept_chunks: List[str] = []
    removed = False
    for chunk in chunks:
        lower = chunk.lower()
        has_target = "target" in lower
        has_mapping = "map" in lower or "mapping" in lower
        has_arrow = "->" in chunk or " to " in lower
        has_label_hint = any(token in lower for token in _TARGET_MAPPING_LABEL_HINTS)
        if has_target and has_mapping and (has_arrow or has_label_hint):
            removed = True
            continue
        kept_chunks.append(chunk)
    sanitized = " ".join(part for part in kept_chunks if part is not None).strip()
    if removed and not sanitized:
        sanitized = (
            "Validate target domain from observed values; do not remap when target is already numeric binary."
        )
    return sanitized, removed


def _sanitize_target_mapping_runbook_payload(payload: Any) -> Tuple[Any, bool]:
    if isinstance(payload, str):
        return _sanitize_target_mapping_runbook_text(payload)
    if isinstance(payload, list):
        changed = False
        out: List[Any] = []
        for item in payload:
            sanitized_item, item_changed = _sanitize_target_mapping_runbook_payload(item)
            out.append(sanitized_item)
            changed = changed or item_changed
        return out, changed
    if isinstance(payload, dict):
        changed = False
        out: Dict[str, Any] = {}
        for key, value in payload.items():
            sanitized_value, value_changed = _sanitize_target_mapping_runbook_payload(value)
            out[key] = sanitized_value
            changed = changed or value_changed
        return out, changed
    return payload, False


def _sanitize_target_mapping_conflicts(
    contract: Dict[str, Any] | None,
    data_profile: Dict[str, Any] | None,
) -> Dict[str, Any]:
    if not isinstance(contract, dict):
        return {}
    repaired = copy.deepcopy(contract)
    if not isinstance(data_profile, dict) or not data_profile:
        return repaired

    observed_values = _build_target_observed_values_map(repaired, data_profile)
    numeric_binary_targets: List[str] = []
    for target_col in _collect_contract_target_candidates(repaired, data_profile):
        values = observed_values.get(target_col, [])
        has_binary_values = bool(values) and all(
            value in _TARGET_BINARY_NUMERIC_TOKENS for value in values
        )
        if has_binary_values or _profile_numeric_binary_hint(data_profile, target_col):
            if target_col not in numeric_binary_targets:
                numeric_binary_targets.append(target_col)
    if not numeric_binary_targets:
        return repaired

    removed_gates = 0
    cleaning_gates = repaired.get("cleaning_gates")
    if isinstance(cleaning_gates, list):
        sanitized_gates: List[Any] = []
        for gate in cleaning_gates:
            if isinstance(gate, dict) and _is_target_mapping_gate(gate):
                if _gate_uses_non_numeric_target_labels(gate):
                    removed_gates += 1
                    continue
            sanitized_gates.append(gate)
        repaired["cleaning_gates"] = sanitized_gates

    runbook_changed = False
    if "data_engineer_runbook" in repaired:
        sanitized_runbook, runbook_changed = _sanitize_target_mapping_runbook_payload(
            repaired.get("data_engineer_runbook")
        )
        if runbook_changed:
            repaired["data_engineer_runbook"] = sanitized_runbook

    if removed_gates or runbook_changed:
        notes = repaired.get("notes_for_engineers")
        if not isinstance(notes, list):
            notes = []
        target_preview = ", ".join(numeric_binary_targets[:3])
        note = (
            "Target mapping directives were sanitized because observed target values are already "
            "numeric-binary"
            + (f" ({target_preview})." if target_preview else ".")
        )
        if note not in notes:
            notes.append(note)
        repaired["notes_for_engineers"] = notes
    return repaired


def _ensure_missing_category_values(contract: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure preprocessing_requirements.nan_strategies missing_category entries
    have a safe missing_category_value and propagate it into null_handling_gate
    params for reviewer visibility.
    """
    if not isinstance(contract, dict):
        return contract
    prep = contract.get("preprocessing_requirements")
    if not isinstance(prep, dict):
        return contract
    nan_strategies = prep.get("nan_strategies")
    if not isinstance(nan_strategies, list):
        return contract

    missing_category_map: Dict[str, str] = {}
    updated = False
    for strategy in nan_strategies:
        if not isinstance(strategy, dict):
            continue
        strat_name = str(strategy.get("strategy") or "").strip().lower()
        if strat_name != "missing_category":
            continue
        col = strategy.get("column") or strategy.get("name")
        if not col:
            continue
        col_name = str(col)
        value = _normalize_missing_category_value(strategy.get("missing_category_value"))
        if not value:
            value = _DEFAULT_MISSING_CATEGORY_VALUE
            strategy["missing_category_value"] = value
            updated = True
        missing_category_map[col_name] = value

    if missing_category_map:
        gates = contract.get("cleaning_gates")
        if isinstance(gates, list):
            for gate in gates:
                if not isinstance(gate, dict):
                    continue
                gate_name = gate.get("name") or gate.get("id") or gate.get("gate")
                if _normalize_gate_name(gate_name) != "null_handling_gate":
                    continue
                params = gate.get("params")
                if not isinstance(params, dict):
                    params = {}
                missing_values = params.get("missing_category_values")
                if not isinstance(missing_values, dict):
                    missing_values = {}
                for col, value in missing_category_map.items():
                    if col not in missing_values:
                        missing_values[col] = value
                        updated = True
                params["missing_category_values"] = missing_values
                gate["params"] = params

    if updated:
        notes = contract.get("notes_for_engineers")
        if not isinstance(notes, list):
            notes = []
        msg = "Injected missing_category_value for missing_category strategies to avoid NA sentinel collisions."
        if msg not in notes:
            notes.append(msg)
        contract["notes_for_engineers"] = notes
    return contract


def _apply_sparse_optional_columns(
    contract: Dict[str, Any],
    data_profile: Dict[str, Any] | None,
    threshold: float = 0.98,
) -> Dict[str, Any]:
    """
    Mark ultra-sparse columns as optional passthrough based on data_profile missingness.
    This prevents required-column guards from blocking on legitimately sparse features.
    """
    if not isinstance(contract, dict) or not isinstance(data_profile, dict):
        return contract
    missingness = data_profile.get("missingness_top30")
    if not isinstance(missingness, dict) or not missingness:
        return contract

    try:
        from src.utils.contract_accessors import get_outcome_columns, get_decision_columns, get_column_roles
    except Exception:
        return contract

    outcomes = {str(c) for c in get_outcome_columns(contract)}
    decisions = {str(c) for c in get_decision_columns(contract)}
    roles = get_column_roles(contract)
    identifiers = set()
    if isinstance(roles, dict):
        raw_ids = roles.get("identifiers") or roles.get("identifier") or []
        if isinstance(raw_ids, list):
            identifiers = {str(c) for c in raw_ids if c}
        elif isinstance(raw_ids, str):
            identifiers = {raw_ids}

    canonical = contract.get("canonical_columns")
    canonical_set = {str(c) for c in canonical} if isinstance(canonical, list) else set()
    available = contract.get("available_columns")
    available_set = {str(c) for c in available} if isinstance(available, list) else set()

    allowed_set = available_set or canonical_set

    def _norm_name(name: Any) -> str:
        return re.sub(r"[^0-9a-zA-Z]+", "", str(name).lower())

    optional_candidates: List[str] = []
    for col, frac in missingness.items():
        try:
            frac_val = float(frac)
        except Exception:
            continue
        if frac_val < threshold:
            continue
        if allowed_set and str(col) not in allowed_set:
            continue
        if str(col) in outcomes or str(col) in decisions or str(col) in identifiers:
            continue
        optional_candidates.append(str(col))

    if not optional_candidates:
        return contract

    artifact_reqs = contract.get("artifact_requirements")
    if not isinstance(artifact_reqs, dict):
        artifact_reqs = {}
    schema_binding = artifact_reqs.get("schema_binding")
    if not isinstance(schema_binding, dict):
        schema_binding = {}
    optional_cols = schema_binding.get("optional_passthrough_columns")
    if not isinstance(optional_cols, list):
        optional_cols = []

    existing = {_norm_name(c): c for c in optional_cols if c}
    for col in optional_candidates:
        norm = _norm_name(col)
        if norm in existing:
            continue
        optional_cols.append(col)
        existing[norm] = col

    schema_binding["optional_passthrough_columns"] = optional_cols
    artifact_reqs["schema_binding"] = schema_binding
    contract["artifact_requirements"] = artifact_reqs

    notes = contract.get("notes_for_engineers")
    if not isinstance(notes, list):
        notes = []
    notes.append({
        "item": "Sparse columns treated as optional passthrough based on data_profile missingness.",
        "columns": optional_candidates,
        "null_frac_threshold": threshold,
    })
    contract["notes_for_engineers"] = notes
    return contract


def _infer_requires_target(strategy: Dict[str, Any], contract: Dict[str, Any]) -> bool:
    if not isinstance(strategy, dict):
        strategy = {}
    if not isinstance(contract, dict):
        contract = {}
    for source in (strategy, contract):
        for key in ("target_column", "target_columns", "outcome_columns", "decision_variable", "decision_variables"):
            val = source.get(key)
            if isinstance(val, list) and val:
                return True
            if isinstance(val, str) and val.strip() and val.lower() != "unknown":
                return True
    capabilities = infer_problem_capabilities(
        objective_text=str(contract.get("business_objective") or ""),
        objective_type=contract.get("objective_type"),
        problem_type=(contract.get("objective_analysis") or {}).get("problem_type")
        if isinstance(contract.get("objective_analysis"), dict)
        else None,
        evaluation_spec=contract.get("evaluation_spec") if isinstance(contract.get("evaluation_spec"), dict) else {},
        validation_requirements=contract.get("validation_requirements")
        if isinstance(contract.get("validation_requirements"), dict)
        else {},
        required_outputs=contract.get("required_outputs") if isinstance(contract.get("required_outputs"), list) else [],
        strategy=strategy,
    )
    if is_problem_family(
        capabilities,
        "classification",
        "regression",
        "forecasting",
        "ranking",
        "optimization",
        "survival_analysis",
    ):
        return True
    return False


def _allow_resampling_random(requires_target: bool, contract: Dict[str, Any]) -> bool:
    if requires_target:
        return True
    if not isinstance(contract, dict):
        return False
    validation = contract.get("validation_requirements", {})
    if not isinstance(validation, dict):
        return False
    method = str(validation.get("method") or "").strip().lower()
    return method in {"cross_validation", "bootstrap"}


def _build_default_qa_gates(
    strategy: Dict[str, Any],
    business_objective: str,
    contract: Dict[str, Any],
) -> List[Dict[str, Any]]:
    requires_target = _infer_requires_target(strategy, contract)
    allow_resampling = _allow_resampling_random(requires_target, contract)
    gates: List[Dict[str, Any]] = [
        {"name": "security_sandbox", "severity": "HARD", "params": {}},
        {"name": "must_read_input_csv", "severity": "HARD", "params": {}},
        {"name": "must_reference_contract_columns", "severity": "HARD", "params": {}},
        {"name": "no_synthetic_data", "severity": "HARD", "params": {"allow_resampling_random": allow_resampling}},
        {"name": "output_row_count_consistency", "severity": "HARD", "params": {}},
        {"name": "dialect_mismatch_handling", "severity": "SOFT", "params": {}},
        {"name": "group_split_required", "severity": "SOFT", "params": {}},
    ]
    if requires_target:
        gates.extend(
            [
                {"name": "target_variance_guard", "severity": "HARD", "params": {}},
                {"name": "leakage_prevention", "severity": "HARD", "params": {}},
                {"name": "train_eval_split", "severity": "SOFT", "params": {}},
            ]
        )
    return gates


def _apply_qa_gate_policy(
    raw_gates: Any,
    strategy: Dict[str, Any],
    business_objective: str,
    contract: Dict[str, Any],
) -> List[Dict[str, Any]]:
    gates = _normalize_qa_gates(raw_gates)
    if not gates:
        gates = _build_default_qa_gates(strategy, business_objective, contract)
    requires_target = _infer_requires_target(strategy, contract)
    allow_resampling = _allow_resampling_random(requires_target, contract)
    for gate in gates:
        if gate.get("name") == "no_synthetic_data":
            params = gate.get("params")
            if not isinstance(params, dict):
                params = {}
            params.setdefault("allow_resampling_random", allow_resampling)
            gate["params"] = params
    if not any(
        _normalize_gate_name(gate.get("name")) == "output_row_count_consistency"
        for gate in gates
        if isinstance(gate, dict)
    ):
        gates.append({"name": "output_row_count_consistency", "severity": "HARD", "params": {}})
    return gates


_KPI_ALIASES = {
    "accuracy": ["accuracy", "acc", "balanced accuracy", "balanced_accuracy"],
    "auc": ["auc", "auroc", "roc auc", "roc_auc"],
    "normalized_gini": [
        "normalized gini",
        "normalized_gini",
        "normalized gini coefficient",
        "kaggle normalized gini",
    ],
    "gini": ["gini", "gini coefficient", "gini_score"],
    "f1": ["f1", "f1-score", "f1_score"],
    "precision": ["precision", "prec"],
    "recall": ["recall", "sensitivity", "tpr"],
    "rmse": ["rmse", "root mean squared error"],
    "mae": ["mae", "mean absolute error"],
    "mse": ["mse", "mean squared error"],
    "r2": ["r2", "r^2", "r-squared", "rsquared", "r_squared"],
    "logloss": ["logloss", "log loss", "log_loss", "cross entropy", "cross_entropy"],
    "mape": ["mape", "mean absolute percentage error"],
    "pr_auc": ["pr_auc", "pr auc", "average precision", "average_precision"],
}


def _normalize_kpi_metric(value: Any) -> str:
    if value is None:
        return ""
    raw = str(value).strip().lower()
    if not raw:
        return ""
    raw = re.sub(r"[^a-z0-9]+", " ", raw).strip()
    for canonical, aliases in _KPI_ALIASES.items():
        for alias in aliases:
            alias_norm = re.sub(r"[^a-z0-9]+", " ", alias).strip()
            if raw == alias_norm:
                return canonical
    return ""


def _extract_kpi_from_list(values: Any) -> str:
    if not isinstance(values, list):
        return ""
    for item in values:
        if isinstance(item, dict):
            metric = item.get("metric") or item.get("name") or item.get("id")
            normalized = _normalize_kpi_metric(metric)
        else:
            normalized = _normalize_kpi_metric(item)
        if normalized:
            return normalized
    return ""


def _extract_kpi_from_text(text: str) -> str:
    if not text:
        return ""
    lower = text.lower()
    for canonical, aliases in _KPI_ALIASES.items():
        for alias in aliases:
            pattern = r"\b" + re.escape(alias.lower()) + r"\b"
            if re.search(pattern, lower):
                return canonical
    return ""


def _extract_json_object(text: str) -> Optional[str]:
    if not text:
        return None
    start = None
    depth = 0
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        if start is None:
            if ch == "{":
                start = i
                depth = 1
                in_str = False
                escape = False
            continue
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "\"":
                in_str = False
            continue
        if ch == "\"":
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start:i + 1]
    return None


def _repair_common_json_damage(text: str) -> str:
    if not isinstance(text, str):
        return ""
    repaired = text.strip()
    if not repaired:
        return repaired

    # Remove markdown fences and keep only first JSON object.
    repaired = repaired.replace("```json", "").replace("```", "").strip()
    extracted = _extract_json_object(repaired)
    if extracted:
        repaired = extracted

    # Drop trailing commas before object/array close.
    repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)

    # Fix split-string corruption where a naked text line follows a quoted value.
    repaired_lines: List[str] = []
    for raw_line in repaired.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        looks_like_json_line = stripped.startswith(("{", "}", "[", "]", "\"")) or ":" in stripped
        if (
            not looks_like_json_line
            and repaired_lines
            and repaired_lines[-1].rstrip().endswith('"')
            and not repaired_lines[-1].rstrip().endswith('",')
        ):
            prev = repaired_lines.pop().rstrip()
            if prev.endswith('"'):
                prev = prev[:-1]
            repaired_lines.append(f'{prev} {stripped.rstrip(",")}"')
            continue
        repaired_lines.append(line)
    if repaired_lines:
        repaired = "\n".join(repaired_lines)
        repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)

    return repaired


def _ensure_benchmark_kpi_gate(
    contract: Dict[str, Any],
    strategy: Dict[str, Any],
    business_objective: str,
) -> Dict[str, Any]:
    if not isinstance(contract, dict):
        return contract
    strategy = strategy if isinstance(strategy, dict) else {}

    kpi = _normalize_kpi_metric(strategy.get("success_metric"))
    if not kpi:
        kpi = _extract_kpi_from_list(strategy.get("recommended_evaluation_metrics"))
    if not kpi:
        kpi = _extract_kpi_from_text(business_objective or "")
    if not kpi:
        return contract

    qa_gates = contract.get("qa_gates")
    if not isinstance(qa_gates, list):
        qa_gates = []

    gate_exists = False
    for gate in qa_gates:
        if not isinstance(gate, dict):
            continue
        if gate.get("name") == "benchmark_kpi_report":
            params = gate.get("params")
            if not isinstance(params, dict):
                params = {}
            params["metric"] = kpi
            params["validation"] = "cross_validation_or_holdout"
            gate["type"] = gate.get("type") or "metric_report"
            gate["params"] = params
            gate.setdefault("severity", "warning")
            gate_exists = True
            break
        if gate.get("type") == "metric_report":
            params = gate.get("params")
            if isinstance(params, dict) and params.get("metric") == kpi:
                gate_exists = True
                break

    if not gate_exists:
        qa_gates.append(
            {
                "name": "benchmark_kpi_report",
                "type": "metric_report",
                "params": {"metric": kpi, "validation": "cross_validation_or_holdout"},
                "severity": "warning",
            }
        )

    contract["qa_gates"] = qa_gates

    validation = contract.get("validation_requirements")
    if not isinstance(validation, dict):
        validation = {}
    validation["primary_metric"] = kpi
    metrics_list = validation.get("metrics_to_report")
    if not isinstance(metrics_list, list):
        metrics_list = []
    metrics_norm = [str(m).strip().lower() for m in metrics_list if m]
    if kpi.lower() not in metrics_norm:
        metrics_list = [kpi] + [m for m in metrics_list if m]
    else:
        # Keep kpi first for deterministic downstream behavior.
        metrics_list = [kpi] + [m for m in metrics_list if str(m).strip().lower() != kpi.lower()]
    validation["metrics_to_report"] = list(dict.fromkeys(metrics_list))
    contract["validation_requirements"] = validation
    return contract


def _create_v41_skeleton(
    strategy: Dict[str, Any],
    business_objective: str,
    column_inventory: List[str] | None = None,
    output_dialect: Dict[str, str] | None = None,
    reason: str = "LLM failure",
    data_summary: str = ""
) -> Dict[str, Any]:
    """
    Returns a complete, safe V4.1 schema skeleton with all required fields.
    Used as fallback when LLM fails or for validation.
    Now respects basic strategy inputs (target, problem_type) to support testing.
    """
    strategy_title = strategy.get("title", "Unknown") if isinstance(strategy, dict) else "Unknown"
    required_cols = strategy.get("required_columns", []) if isinstance(strategy, dict) else []
    available_cols = column_inventory or []

    # Canonical columns should represent full available_columns (no truncation).
    canonical_cols = [str(c) for c in available_cols if c]
    missing_cols = []
    fuzzy_matches = {}

    inventory_map = {re.sub(r"[^0-9a-zA-Z]+", "", str(c).lower()): c for c in available_cols}

    def _find_in_inventory(name: str) -> str | None:
        return inventory_map.get(re.sub(r"[^0-9a-zA-Z]+", "", str(name).lower()))

    if required_cols and available_cols:
        for req in required_cols:
            found = _find_in_inventory(req)
            if found:
                if found not in canonical_cols:
                    canonical_cols.append(found)
            else:
                missing_cols.append(str(req))
                # Use difflib to find close matches
                close_matches = difflib.get_close_matches(
                    str(req).lower(),
                    [str(c).lower() for c in available_cols],
                    n=3,
                    cutoff=0.6,
                )
                if close_matches:
                    matched_originals = [
                        c for c in available_cols
                        if str(c).lower() in close_matches
                    ]
                    fuzzy_matches[str(req)] = matched_originals[:3]

    if not canonical_cols and required_cols:
        canonical_cols = [str(col) for col in required_cols if col]

    # Smart Fallback: Infer roles from strategy if present
    target_col = strategy.get("target_column") if isinstance(strategy, dict) else None
    col_roles_outcome = []
    derived_cols_list = []
    
    if target_col:
        real_target = _find_in_inventory(target_col)
        if real_target:
            col_roles_outcome.append(real_target)
        else:
            # Assume derived target
            col_roles_outcome.append(target_col)
            derived_cols_list.append(target_col)

    # Everything else in unknown for now (inventory metadata only - don't force requirements on them)
    available_set = set(available_cols)
    outcome_set = set(col_roles_outcome)
    unknown_cols = [c for c in available_cols if c not in outcome_set]  # canonical might be in unknown
    unknown_summary = {
        "count": len(unknown_cols),
        "sample": unknown_cols[:25],
    }

    # Parse types from summary for tests/fallback utility
    type_distribution = {}
    if data_summary:
        for line in data_summary.splitlines():
            clean_line = line.strip().strip("- ")
            if ":" not in clean_line: continue
            lbl, cols_txt = clean_line.split(":", 1)
            lbl = lbl.lower()
            kind = "unknown"
            if "date" in lbl: kind = "datetime"
            elif "num" in lbl: kind = "numeric"
            elif "cat" in lbl or "bool" in lbl: kind = "categorical"
            elif "text" in lbl: kind = "text"
            
            if kind != "unknown":
                if kind not in type_distribution: type_distribution[kind] = []
                for c_raw in re.split(r"[;,]", cols_txt):
                    if not c_raw.strip(): continue
                    real = _find_in_inventory(c_raw.strip()) or c_raw.strip()
                    type_distribution[kind].append(real)

    # Alias for artifact requirements
    outcome_cols = col_roles_outcome
    
    # Infer identifiers for artifact requirements fallback
    identifiers = []
    for col in available_cols:
        # Simple heuristic: exact 'id', or ends in '_id'/'Id' etc
        if re.search(r"(?i)\b(id|uuid|key)\b", col):
            identifiers.append(col)

    n_rows = 0

    problem_type = strategy.get("problem_type", "unknown") if isinstance(strategy, dict) else "unknown"
    feature_selectors = []
    if len(available_cols) > 200:
        feature_selectors, _remaining = infer_feature_selectors(
            available_cols, max_list_size=200, min_group_size=50
        )

    return {
        "contract_version": CONTRACT_VERSION_V41,
        "strategy_title": strategy_title,
        "business_objective": business_objective or "",
        
        "missing_columns_handling": {
            "missing_from_inventory": missing_cols,
            "attempted_fuzzy_matches": fuzzy_matches,
            "resolution": "unknown" if not missing_cols else "require_verification",
            "impact": "none" if not missing_cols else f"{len(missing_cols)} required columns not found",
            "contract_updates": {
                "canonical_columns_update": "Fuzzy matches suggested" if fuzzy_matches else "",
                "artifact_schema_update": "",
                "derived_plan_update": "",
                "gates_update": ""
            }
        },
        
        "execution_constraints": {
            "inplace_column_creation_policy": "unknown_or_forbidden",
            "preferred_patterns": ["df = df.assign(...)", "derived_arrays_then_concat", "build_new_df_from_columns"],
            "rationale": "Fallback: prefer safe patterns when uncertain."
        },
        
        "objective_analysis": {
            "problem_type": problem_type,
            "decision_variable": None,
            "business_decision": "unknown",
            "success_criteria": "unknown",
            "complexity": "unknown"
        },
        
        "data_analysis": {
            "dataset_size": n_rows,
            "features_with_nulls": [],
            "type_distribution": type_distribution,
            "risk_features": [],
            "data_sufficiency": "unknown"
        },
        
        "column_roles": {
            "pre_decision": [],
            "decision": [],
            "outcome": col_roles_outcome,
            "post_decision_audit_only": [],
            "unknown": []
        },
        "column_roles_unknown_summary": unknown_summary,
        
        "preprocessing_requirements": {},
        
        "feature_engineering_plan": {
            "derived_columns": derived_cols_list
        },
        
        "validation_requirements": {
            "method": "cross_validation",
            "stratification": False,
            "min_samples_required": 100
        },
        
        "leakage_execution_plan": {
            "audit_features": [],
            "method": "correlation_with_target",
            "threshold": 0.9,
            "action_if_exceeds": "exclude_from_features"
        },
        
        "optimization_specification": None,
        "segmentation_constraints": None,
        
        "data_limited_mode": {
            "is_active": False,
            "activation_reasons": [],
            "fallback_methodology": "unknown",
            "minimum_outputs": [],
            "artifact_reductions_allowed": True
        },
        
        "allowed_feature_sets": {
            "segmentation_features": [],
            "model_features": [],
            "audit_only_features": [],
            "forbidden_for_modeling": [],
            "rationale": "Fallback: empty feature sets pending analysis."
        },
        
        "artifact_requirements": {
            "required_files": [],
            "file_schemas": {},
        },
        
        "qa_gates": _apply_qa_gate_policy([], strategy, business_objective or "", {}),
        "cleaning_gates": _apply_cleaning_gate_policy([]),
        
        "reviewer_gates": [
            {"id": "methodology_alignment", "required": True, "description": "Methodology aligns with objective"},
            {"id": "business_value", "required": True, "description": "Business value demonstrated"}
        ],
        
        "data_engineer_runbook": DEFAULT_DATA_ENGINEER_RUNBOOK,
        "ml_engineer_runbook": DEFAULT_ML_ENGINEER_RUNBOOK,
        
        "available_columns": available_cols,
          "canonical_columns": canonical_cols,
          "derived_columns": derived_cols_list,
          "feature_selectors": feature_selectors,
          "canonical_columns_compact": compact_column_representation(available_cols, max_display=40)
          if feature_selectors
          else {},
        "required_outputs": [],
        
        "iteration_policy": {
            "max_iterations": 3,
            "early_stop_on_success": True
        },
        "optimization_policy": get_default_optimization_policy(),
        
        "unknowns": [
            {
                "item": reason,
                "impact": "Using skeletal V4.1 fallback",
                "mitigation": "Manual review required",
                "requires_verification": True
            }
        ],
        
        "assumptions": [
            "Minimal safe defaults used due to planner unavailability"
        ],
        
        "notes_for_engineers": [
            "This is a fallback contract. Proceed conservatively.",
            "Verify column inventory and semantics manually if possible."
        ]
    }


def _tokenize_name(value: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", " ", str(value).lower()).strip()


def _coerce_list(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item]
    if isinstance(value, str):
        return [value]
    return []


_MULTI_TARGET_MARKERS = (
    "multi_output",
    "multioutput",
    "multi_target",
    "multitarget",
    "multi_label",
    "multilabel",
    "multi_horizon",
    "multihorizon",
    "one_vs_rest_targets",
    "one_vs_rest_labels",
)
_TARGET_NAME_MARKERS = ("target", "label", "outcome", "response", "y_")
_TRAILING_TARGET_BUCKET_RE = re.compile(
    r"(?:[_\-\s]?(?:t\d+|\d+(?:h|hr|hrs|hour|hours|d|day|days|w|week|weeks|m|min|mins|month|months)))+$",
    flags=re.IGNORECASE,
)


def _collect_targetish_columns(values: Any) -> List[str]:
    cols: List[str] = []
    for value in _coerce_list(values):
        token = str(value or "").strip()
        if token and token not in cols:
            cols.append(token)
    return cols


def _target_family_signature(value: Any) -> str:
    token = str(value or "").strip().lower()
    if not token:
        return ""
    token = _TRAILING_TARGET_BUCKET_RE.sub("", token)
    token = re.sub(r"[_\-\s]+$", "", token)
    token = re.sub(r"[_\-\s]+", "_", token)
    return token


def _looks_like_target_semantic_column(value: Any) -> bool:
    token = str(value or "").strip().lower()
    if not token:
        return False
    normalized = re.sub(r"[^a-z0-9_]+", "_", token).strip("_")
    if any(marker in normalized for marker in _TARGET_NAME_MARKERS):
        return True
    return bool(_target_family_signature(normalized)) and normalized.startswith(("y_", "label_", "target_", "outcome_"))


def _has_multi_target_signal(*values: Any) -> bool:
    for value in values:
        if isinstance(value, dict):
            if _has_multi_target_signal(*value.values()):
                return True
            continue
        if isinstance(value, list):
            if _has_multi_target_signal(*value):
                return True
            continue
        text = str(value or "").strip().lower()
        if not text:
            continue
        if any(marker in text for marker in _MULTI_TARGET_MARKERS):
            return True
        if "12h" in text and "24h" in text:
            return True
    return False


def _collect_strategy_feature_family_hints(strategy_dict: Dict[str, Any]) -> List[str]:
    """
    Collect compact feature-family hints from strategy output.

    Hints are intentionally compact (not full expanded columns) so wide datasets
    do not explode prompt size.
    """
    families_raw = strategy_dict.get("feature_families")
    if not isinstance(families_raw, list) or not families_raw:
        return []

    hints: List[str] = []

    def _add(value: Any) -> None:
        if not isinstance(value, str):
            return
        token = value.strip()
        if not token or token in hints:
            return
        hints.append(token)

    for entry in families_raw:
        if isinstance(entry, str):
            _add(entry)
            continue
        if not isinstance(entry, dict):
            continue
        for key in ("selector_hint", "pattern", "family", "name", "description"):
            _add(entry.get(key))

    return hints


def _expand_strategy_feature_families(
    strategy_dict: Dict[str, Any],
    inventory: List[str],
) -> List[str]:
    """
    Expand strategist feature-family hints into concrete columns from inventory.

    Universal behavior:
    - never invent columns
    - only match against observed column inventory
    - supports common selector hint shapes (prefix, regex, numeric ranges)
    """
    families_raw = strategy_dict.get("feature_families")
    if not isinstance(families_raw, list) or not families_raw:
        return []
    if not inventory:
        return []

    inventory_set = set(inventory)
    matched: List[str] = []

    def _add_many(cols: List[str]) -> None:
        for col in cols:
            if col in inventory_set and col not in matched:
                matched.append(col)

    def _match_prefix(prefix: str) -> List[str]:
        token = str(prefix or "").strip()
        if not token:
            return []
        return [col for col in inventory if col.lower().startswith(token.lower())]

    def _match_regex(pattern: str) -> List[str]:
        expr = str(pattern or "").strip()
        if not expr:
            return []
        try:
            regex = re.compile(expr, flags=re.IGNORECASE)
        except re.error:
            return []
        return [col for col in inventory if regex.match(col)]

    def _match_numeric_range(prefix: str, start: int, end: int) -> List[str]:
        token = str(prefix or "").strip()
        if not token:
            return []
        lo = min(int(start), int(end))
        hi = max(int(start), int(end))
        rx = re.compile(rf"^{re.escape(token)}(\d+)$", flags=re.IGNORECASE)
        out: List[str] = []
        for col in inventory:
            m = rx.match(col)
            if not m:
                continue
            try:
                idx = int(m.group(1))
            except Exception:
                continue
            if lo <= idx <= hi:
                out.append(col)
        return out

    generic_tokens = {
        "feature",
        "features",
        "family",
        "families",
        "column",
        "columns",
        "numeric",
        "categorical",
        "all",
        "input",
        "inputs",
    }

    def _extract_family_tokens(text: str) -> List[str]:
        tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", str(text or ""))
        out: List[str] = []
        for token in tokens:
            key = token.lower()
            if key in generic_tokens:
                continue
            if len(key) <= 1:
                continue
            out.append(token)
        return list(dict.fromkeys(out))

    def _parse_hint(hint: str) -> List[str]:
        text = str(hint or "").strip()
        if not text:
            return []
        cols: List[str] = []

        # prefix:pixel
        prefix_match = re.search(r"prefix\s*:\s*([A-Za-z_][A-Za-z0-9_]*)", text, flags=re.IGNORECASE)
        if prefix_match:
            cols.extend(_match_prefix(prefix_match.group(1)))

        # pixel[0-783]
        for m in re.finditer(r"([A-Za-z_][A-Za-z0-9_]*)\[(\d+)\s*-\s*(\d+)\]", text):
            cols.extend(_match_numeric_range(m.group(1), int(m.group(2)), int(m.group(3))))

        # pixel0-pixel783 / pixel0 to pixel783 / pixel0 to 783
        range_patterns = [
            r"([A-Za-z_][A-Za-z0-9_]*)(\d+)\s*(?:to|-)\s*([A-Za-z_][A-Za-z0-9_]*)(\d+)",
            r"([A-Za-z_][A-Za-z0-9_]*)(\d+)\s*(?:to|-)\s*(\d+)",
        ]
        for pattern in range_patterns:
            for m in re.finditer(pattern, text, flags=re.IGNORECASE):
                groups = m.groups()
                if len(groups) == 4:
                    p1, s1, p2, s2 = groups
                    if str(p1).lower() != str(p2).lower():
                        continue
                    cols.extend(_match_numeric_range(p1, int(s1), int(s2)))
                elif len(groups) == 3:
                    p1, s1, s2 = groups
                    cols.extend(_match_numeric_range(p1, int(s1), int(s2)))

        # Explicit regex-like pattern hints: pattern 'pixel[0-9]+'
        quoted_patterns = re.findall(r"['\"]([^'\"]+)['\"]", text)
        for candidate in quoted_patterns:
            if any(ch in candidate for ch in ("[", "]", "(", ")", "^", "$", "+", "*", "\\", "|")):
                regex_candidate = candidate
                # Normalize naive ranges like [0-783] into \d+ (common LLM shorthand)
                regex_candidate = re.sub(r"\[\d+\s*-\s*\d+\]", r"\\d+", regex_candidate)
                cols.extend(_match_regex(rf"^{regex_candidate}$"))

        # Fallback family token prefix match
        if not cols:
            for token in _extract_family_tokens(text):
                pref_matches = _match_prefix(token)
                if pref_matches:
                    cols.extend(pref_matches)
                    break

        return list(dict.fromkeys(cols))

    for entry in families_raw:
        hints: List[str] = []
        if isinstance(entry, str):
            hints.append(entry)
        elif isinstance(entry, dict):
            for key in ("selector_hint", "family", "name", "pattern", "rationale", "description"):
                value = entry.get(key)
                if isinstance(value, str) and value.strip():
                    hints.append(value.strip())
        for hint in hints:
            _add_many(_parse_hint(hint))

    return matched


def select_relevant_columns(
    strategy: Dict[str, Any] | None,
    business_objective: str,
    domain_expert_critique: str,
    column_inventory: List[str] | None,
    data_profile_summary: str | None = None,
    constant_anchor_avoidance: List[str] | None = None,
) -> Dict[str, Any]:
    """
    Deterministically select a compact set of relevant columns for planning.
    """
    inventory = [str(col) for col in (column_inventory or []) if col is not None]
    inventory_set = set(inventory)
    inventory_norm_map: Dict[str, str] = {}
    for col in inventory:
        norm = _normalize_column_identifier(col)
        if norm and norm not in inventory_norm_map:
            inventory_norm_map[norm] = col

    def _resolve_inventory(name: str) -> Optional[str]:
        if not name:
            return None
        if name in inventory_set:
            return name
        norm = _normalize_column_identifier(name)
        return inventory_norm_map.get(norm)

    anchor_avoidance_set: set[str] = set()
    for raw_col in (constant_anchor_avoidance or []):
        resolved = _resolve_inventory(str(raw_col))
        if resolved:
            anchor_avoidance_set.add(resolved)

    sources: Dict[str, List[str]] = {
        "strategy_required_columns": [],
        "strategy_feature_families": [],
        "strategy_decision_columns": [],
        "strategy_outcome_columns": [],
        "strategy_audit_only_columns": [],
        "text_mentions": [],
        "heuristic": [],
    }

    def _add_unique(target: List[str], col: str) -> None:
        if not col or col in target:
            return
        target.append(col)

    def _add_source(col: Optional[str], key: str, collector: List[str]) -> None:
        if not col:
            return
        _add_unique(collector, col)
        _add_unique(sources[key], col)

    strategy_dict = strategy if isinstance(strategy, dict) else {}
    strategy_feature_family_hints = _collect_strategy_feature_family_hints(strategy_dict)
    required_cols_raw = _coerce_list(strategy_dict.get("required_columns"))
    decision_cols_raw = _coerce_list(
        strategy_dict.get("decision_columns")
        or strategy_dict.get("decision_variables")
        or strategy_dict.get("decision_column")
    )
    outcome_cols_raw = _coerce_list(
        strategy_dict.get("outcome_columns")
        or strategy_dict.get("target_column")
        or strategy_dict.get("target_columns")
        or strategy_dict.get("outcome_column")
    )
    audit_only_raw = _coerce_list(strategy_dict.get("audit_only_columns"))

    required_cols: List[str] = []
    family_cols: List[str] = []
    decision_cols: List[str] = []
    outcome_cols: List[str] = []
    audit_only_cols: List[str] = []
    family_cols_expanded = _expand_strategy_feature_families(strategy_dict, inventory)

    for col in required_cols_raw:
        _add_source(_resolve_inventory(col), "strategy_required_columns", required_cols)
    for col in family_cols_expanded:
        _add_source(_resolve_inventory(col), "strategy_feature_families", family_cols)
    for col in decision_cols_raw:
        _add_source(_resolve_inventory(col), "strategy_decision_columns", decision_cols)
    for col in outcome_cols_raw:
        _add_source(_resolve_inventory(col), "strategy_outcome_columns", outcome_cols)
    for col in audit_only_raw:
        _add_source(_resolve_inventory(col), "strategy_audit_only_columns", audit_only_cols)

    text_matches: List[str] = []
    text_blob = "\n".join([business_objective or "", domain_expert_critique or ""])
    if text_blob:
        quote_chars = "\"'`" + "\u2018\u2019\u201c\u201d"
        pattern = f"[{re.escape(quote_chars)}](.+?)[{re.escape(quote_chars)}]"
        for match in re.findall(pattern, text_blob):
            candidate = match.strip()
            if candidate in inventory_set:
                _add_source(candidate, "text_mentions", text_matches)

    heuristic_cols: List[str] = []
    if len(set(required_cols + family_cols + decision_cols + outcome_cols + audit_only_cols + text_matches)) < 4:
        patterns = [
            (re.compile(r"\b(target|label|outcome|success|converted)\b"), "target_like"),
            (re.compile(r"\b(price|amount|offer|quote|cost)\b"), "decision_like"),
            (re.compile(r"\b(id|uuid|key)\b"), "id_like"),
            (re.compile(r"\b(date|time|timestamp)\b"), "time_like"),
        ]
        for col in inventory:
            tokenized = _tokenize_name(col)
            if not tokenized:
                continue
            for regex, _label in patterns:
                if regex.search(tokenized):
                    _add_source(col, "heuristic", heuristic_cols)
                    break

    if anchor_avoidance_set:
        family_cols = [col for col in family_cols if col not in anchor_avoidance_set]
        text_matches = [col for col in text_matches if col not in anchor_avoidance_set]
        heuristic_cols = [col for col in heuristic_cols if col not in anchor_avoidance_set]
        # Keep source trace transparent for planner diagnostics.
        sources["constant_anchor_avoidance"] = sorted(anchor_avoidance_set)[:80]

    ordered: List[str] = []
    for col in required_cols:
        _add_unique(ordered, col)
    for col in family_cols:
        _add_unique(ordered, col)
    for col in outcome_cols + decision_cols:
        _add_unique(ordered, col)
    for col in audit_only_cols + text_matches + heuristic_cols:
        _add_unique(ordered, col)

    ordered = list(dict.fromkeys(ordered))
    total_relevant_count = len(ordered)

    def _int_env(name: str, default: int, *, lo: int, hi: int) -> int:
        try:
            value = int(os.getenv(name, str(default)))
        except Exception:
            value = default
        return max(lo, min(hi, value))

    # Adaptive budget for planner prompts: keep anchors explicit while preventing
    # high-dimensional feature families from ballooning token usage.
    inventory_count = len(inventory)
    if inventory_count <= 120:
        default_budget = 60
    elif inventory_count <= 500:
        default_budget = 120
    else:
        default_budget = 180
    max_relevant_columns = _int_env(
        "EXECUTION_PLANNER_MAX_RELEVANT_COLUMNS",
        default_budget,
        lo=25,
        hi=400,
    )

    def _even_sample(values: List[str], limit: int) -> List[str]:
        if limit <= 0 or not values:
            return []
        if len(values) <= limit:
            return values
        if limit == 1:
            return [values[0]]
        idxs = [round(i * (len(values) - 1) / (limit - 1)) for i in range(limit)]
        sampled: List[str] = []
        seen_idx: set[int] = set()
        for idx in idxs:
            if idx in seen_idx:
                continue
            seen_idx.add(idx)
            sampled.append(values[idx])
        if len(sampled) < limit:
            for value in values:
                if value in sampled:
                    continue
                sampled.append(value)
                if len(sampled) >= limit:
                    break
        return sampled[:limit]

    pinned_columns: List[str] = []
    for col in required_cols + outcome_cols + decision_cols + audit_only_cols + text_matches + heuristic_cols:
        _add_unique(pinned_columns, col)

    compact_relevant = list(ordered)
    if len(compact_relevant) > max_relevant_columns:
        pinned = [col for col in pinned_columns if col in inventory_set]
        remaining_budget = max(0, max_relevant_columns - len(pinned))
        pinned_norm = set(pinned)
        tail_candidates = [col for col in compact_relevant if col not in pinned_norm]
        sampled_tail = _even_sample(tail_candidates, remaining_budget)
        compact_relevant = []
        for col in pinned + sampled_tail:
            _add_unique(compact_relevant, col)
        if len(compact_relevant) > max_relevant_columns:
            compact_relevant = compact_relevant[:max_relevant_columns]

    # For high-dimensional datasets, note that column_sets.json provides grouped access
    is_high_dimensional = len(inventory) > 100
    omitted_policy = (
        "High-dimensional dataset: use column_sets.json for grouped feature access."
        if is_high_dimensional
        else "Ignored by default unless promoted by strategy/explicit mention; available via column_inventory."
    )
    relevant_truncated = len(compact_relevant) < total_relevant_count
    omitted_count = max(0, total_relevant_count - len(compact_relevant))

    return {
        "relevant_columns": [col for col in compact_relevant if col in inventory_set],
        "relevant_columns_total_count": total_relevant_count,
        "relevant_columns_truncated": relevant_truncated,
        "relevant_columns_omitted_count": omitted_count,
        "relevant_columns_compact": compact_column_representation(compact_relevant, max_display=40),
        "strategy_feature_family_hints": strategy_feature_family_hints,
        "strategy_feature_family_expanded_count": len(family_cols_expanded),
        "relevant_sources": sources,
        "omitted_columns_policy": omitted_policy,
    }


def _normalize_dtype_target(
    raw_dtype: str,
    *,
    missing_frac: float | None = None,
    role_hint: str = "",
) -> str:
    dtype = str(raw_dtype or "").strip().lower()
    role = str(role_hint or "").strip().lower()
    has_missing = isinstance(missing_frac, (int, float)) and float(missing_frac) > 0.0

    if role == "time_columns":
        return "datetime64[ns]"
    if role == "identifiers":
        return "object"
    if "datetime" in dtype or "date" in dtype or "time" in dtype:
        return "datetime64[ns]"
    if "bool" in dtype:
        return "boolean"
    if "int" in dtype:
        return "Int64" if has_missing else "int64"
    if any(tok in dtype for tok in ("float", "double", "number", "numeric", "decimal")):
        return "float64"
    if role in {"outcome", "decision"} and has_missing and not dtype:
        return "float64"
    if dtype in {"category", "categorical", "object", "string", "str"}:
        return "object"
    return "preserve"


def _selector_key_for_dtype_targets(selector: Dict[str, Any]) -> str:
    if not isinstance(selector, dict):
        return "selector:unknown"
    stype = str(selector.get("type") or "").strip().lower()
    if stype == "regex":
        pattern = str(selector.get("pattern") or "").strip()
        return f"selector:regex:{pattern}" if pattern else "selector:regex"
    if stype in {"prefix", "suffix", "contains"}:
        token = str(selector.get(stype) or selector.get("value") or "").strip()
        return f"selector:{stype}:{token}" if token else f"selector:{stype}"
    if stype == "prefix_numeric_range":
        prefix = str(selector.get("prefix") or "").strip()
        start = selector.get("start")
        end = selector.get("end")
        if prefix and isinstance(start, int) and isinstance(end, int):
            return f"selector:prefix_numeric_range:{prefix}[{start}:{end}]"
        return "selector:prefix_numeric_range"
    if stype == "list":
        cols = selector.get("columns")
        if isinstance(cols, list) and cols:
            return f"selector:list:{len(cols)}"
        return "selector:list"
    if stype in {"all_columns_except", "all_numeric_except"}:
        cols = selector.get("except_columns") or selector.get("value")
        if isinstance(cols, list) and cols:
            return f"selector:{stype}:{len(cols)}"
        return f"selector:{stype}"
    family = selector.get("family_id") or selector.get("name") or selector.get("family")
    if family:
        return f"selector:family:{str(family).strip()}"
    return "selector:unknown"


def _infer_column_dtype_targets(
    *,
    canonical_columns: List[str],
    column_roles: Dict[str, Any],
    data_profile: Dict[str, Any] | None,
    clean_dataset_cfg: Dict[str, Any] | None,
    column_inventory: List[str] | None,
) -> Dict[str, Dict[str, Any]]:
    profile = data_profile if isinstance(data_profile, dict) else {}
    clean_cfg = clean_dataset_cfg if isinstance(clean_dataset_cfg, dict) else {}
    inventory = [str(c) for c in (column_inventory or []) if str(c).strip()]
    inventory_set = set(inventory)
    dtypes = profile.get("dtypes")
    if not isinstance(dtypes, dict):
        dtypes = {}
    missingness = profile.get("missingness")
    if not isinstance(missingness, dict):
        missingness = {}

    role_map: Dict[str, str] = {}
    if isinstance(column_roles, dict):
        for role_name, values in column_roles.items():
            if not isinstance(values, list):
                continue
            role_key = str(role_name or "").strip()
            for col in values:
                col_name = str(col or "").strip()
                if col_name:
                    role_map[col_name] = role_key

    required_cols = _coerce_list(clean_cfg.get("required_columns"))
    passthrough_cols = _coerce_list(clean_cfg.get("optional_passthrough_columns"))
    anchor_cols: List[str] = []
    for col in required_cols + passthrough_cols:
        if col and col not in anchor_cols:
            anchor_cols.append(str(col))
    for role_name in ("outcome", "decision", "identifiers", "time_columns"):
        for col in _coerce_list(column_roles.get(role_name) if isinstance(column_roles, dict) else []):
            if col and col not in anchor_cols:
                anchor_cols.append(str(col))

    canonical_list = [str(col) for col in (canonical_columns or []) if str(col).strip()]
    use_full_canonical = len(canonical_list) <= 160
    explicit_cols = canonical_list if use_full_canonical else anchor_cols
    if not explicit_cols:
        explicit_cols = canonical_list[:120]
    explicit_cols = list(dict.fromkeys([col for col in explicit_cols if col]))

    dtype_targets: Dict[str, Dict[str, Any]] = {}
    for col in explicit_cols:
        if inventory_set and col not in inventory_set:
            continue
        role_hint = role_map.get(col, "")
        miss_frac = missingness.get(col)
        try:
            miss_val = float(miss_frac) if miss_frac is not None else None
        except Exception:
            miss_val = None
        target_dtype = _normalize_dtype_target(
            str(dtypes.get(col) or ""),
            missing_frac=miss_val,
            role_hint=role_hint,
        )
        dtype_targets[col] = {
            "target_dtype": target_dtype,
            "nullable": bool(miss_val and miss_val > 0.0),
            "role": role_hint or "unknown",
            "source": "data_profile",
        }

    selectors_raw = clean_cfg.get("required_feature_selectors")
    selectors = [item for item in selectors_raw if isinstance(item, dict)] if isinstance(selectors_raw, list) else []
    if selectors:
        selector_cols, selector_issues = expand_required_feature_selectors(selectors, inventory or canonical_list)
        selector_issues = selector_issues or []
        observed_targets: List[str] = []
        for col in selector_cols[:600]:
            role_hint = role_map.get(col, "")
            miss_frac = missingness.get(col)
            try:
                miss_val = float(miss_frac) if miss_frac is not None else None
            except Exception:
                miss_val = None
            observed_targets.append(
                _normalize_dtype_target(
                    str(dtypes.get(col) or ""),
                    missing_frac=miss_val,
                    role_hint=role_hint,
                )
            )
        selector_dtype = "preserve"
        if observed_targets:
            counts: Dict[str, int] = {}
            for item in observed_targets:
                counts[item] = counts.get(item, 0) + 1
            selector_dtype = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[0][0]

        for selector in selectors:
            key = _selector_key_for_dtype_targets(selector)
            entry = {
                "target_dtype": selector_dtype,
                "nullable": "preserve",
                "source": "selector_family",
            }
            if selector_cols:
                entry["matched_count"] = int(len(selector_cols))
                entry["matched_sample"] = selector_cols[:12]
            if selector_issues:
                entry["selector_issues"] = selector_issues[:6]
            dtype_targets[key] = entry

    return dtype_targets


def _infer_selector_type_from_payload(selector: Dict[str, Any]) -> str:
    if not isinstance(selector, dict):
        return ""
    selector_type = str(selector.get("type") or "").strip().lower()
    if selector_type:
        return selector_type
    if selector.get("pattern") or selector.get("regex"):
        return "regex"
    if selector.get("prefix"):
        if isinstance(selector.get("start"), int) and isinstance(selector.get("end"), int):
            return "prefix_numeric_range"
        return "prefix"
    if selector.get("suffix"):
        return "suffix"
    if selector.get("contains"):
        return "contains"
    if isinstance(selector.get("columns"), list):
        return "list"
    if isinstance(selector.get("except_columns"), list):
        return "all_columns_except"
    if isinstance(selector.get("value"), list):
        return "all_numeric_except"
    return ""


def _extract_derived_column_names(value: Any) -> List[str]:
    """Normalize derived_columns payload into a stable list of column names."""
    names: List[str] = []

    if isinstance(value, str):
        token = value.strip()
        if token:
            names.append(token)
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                token = item.strip()
                if token:
                    names.append(token)
            elif isinstance(item, dict):
                candidate = (
                    item.get("name")
                    or item.get("canonical_name")
                    or item.get("column")
                    or item.get("output_column_name")
                )
                if isinstance(candidate, str) and candidate.strip():
                    names.append(candidate.strip())
    elif isinstance(value, dict):
        for key, payload in value.items():
            key_name = str(key).strip() if key is not None else ""
            if key_name:
                names.append(key_name)
                continue
            if isinstance(payload, dict):
                candidate = (
                    payload.get("name")
                    or payload.get("canonical_name")
                    or payload.get("column")
                    or payload.get("output_column_name")
                )
                if isinstance(candidate, str) and candidate.strip():
                    names.append(candidate.strip())

    deduped: List[str] = []
    seen: set[str] = set()
    for name in names:
        norm = _normalize_column_identifier(name)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        deduped.append(name)
    return deduped


def _deterministic_repair_column_dtype_targets(contract: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(contract, dict):
        return contract
    targets = contract.get("column_dtype_targets")
    artifact_requirements = contract.get("artifact_requirements")
    clean_dataset = artifact_requirements.get("clean_dataset") if isinstance(artifact_requirements, dict) else None
    if targets in (None, {}) and isinstance(clean_dataset, dict):
        nested_targets = clean_dataset.get("column_dtype_targets")
        if isinstance(nested_targets, dict):
            targets = nested_targets
    if not isinstance(targets, dict):
        return contract

    repaired: Dict[str, Dict[str, Any]] = {}
    for raw_col, raw_spec in targets.items():
        col = str(raw_col or "").strip()
        if not col:
            continue
        if isinstance(raw_spec, str):
            dtype = raw_spec.strip() or "preserve"
            repaired[col] = {"target_dtype": dtype}
            continue
        if not isinstance(raw_spec, dict):
            repaired[col] = {"target_dtype": "preserve"}
            continue

        spec = dict(raw_spec)
        if "target_dtype" not in spec or not str(spec.get("target_dtype") or "").strip():
            for alias in ("type", "dtype", "data_type", "targetType"):
                if alias in spec and str(spec.get(alias) or "").strip():
                    spec["target_dtype"] = str(spec.pop(alias)).strip()
                    break
        if "target_dtype" not in spec or not str(spec.get("target_dtype") or "").strip():
            spec["target_dtype"] = "preserve"
        repaired[col] = spec

    if repaired:
        contract["column_dtype_targets"] = repaired
        if isinstance(clean_dataset, dict):
            clean_dataset["column_dtype_targets"] = copy.deepcopy(repaired)
    return contract


def _deterministic_repair_required_feature_selectors(contract: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(contract, dict):
        return contract
    artifact_requirements = contract.get("artifact_requirements")
    clean_dataset = artifact_requirements.get("clean_dataset") if isinstance(artifact_requirements, dict) else None
    if not isinstance(clean_dataset, dict):
        return contract

    selectors_raw = clean_dataset.get("required_feature_selectors")
    if selectors_raw is None:
        return contract
    if isinstance(selectors_raw, dict):
        selectors_raw = [selectors_raw]
    elif isinstance(selectors_raw, str):
        selectors_raw = [{"selector": selectors_raw}]
    if not isinstance(selectors_raw, list):
        return contract

    repaired: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def _push(selector_obj: Dict[str, Any]) -> None:
        fingerprint = json.dumps(selector_obj, sort_keys=True, ensure_ascii=False)
        if fingerprint in seen:
            return
        seen.add(fingerprint)
        repaired.append(selector_obj)

    for raw_selector in selectors_raw:
        selector_obj: Dict[str, Any] | None = None
        if isinstance(raw_selector, str):
            token = raw_selector.strip()
            if not token:
                continue
            if ":" in token:
                selector_obj = {"selector": token}
            elif "*" in token or "?" in token:
                if token.endswith("*") and token.count("*") == 1 and "?" not in token:
                    selector_obj = {"type": "prefix", "value": token[:-1]}
                elif token.startswith("*") and token.count("*") == 1 and "?" not in token:
                    selector_obj = {"type": "suffix", "value": token[1:]}
                else:
                    pattern = "^" + re.escape(token).replace("\\*", ".*").replace("\\?", ".") + "$"
                    selector_obj = {"type": "regex", "pattern": pattern}
            else:
                selector_obj = {"type": "list", "columns": [token]}
        elif isinstance(raw_selector, dict):
            selector_obj = dict(raw_selector)

        if not isinstance(selector_obj, dict):
            continue

        selector_obj = _normalize_selector_entry(selector_obj)
        selector_type = _infer_selector_type_from_payload(selector_obj)
        if not selector_type:
            continue
        selector_obj["type"] = selector_type

        if selector_type == "regex":
            pattern = selector_obj.get("pattern") or selector_obj.get("value") or selector_obj.get("regex")
            pattern_str = str(pattern or "").strip()
            if not pattern_str:
                continue
            selector_obj["pattern"] = pattern_str
            selector_obj.pop("value", None)
            selector_obj.pop("regex", None)
        elif selector_type in {"prefix", "suffix", "contains"}:
            value = selector_obj.get("value") or selector_obj.get(selector_type)
            value_str = str(value or "").strip()
            if not value_str:
                continue
            selector_obj["value"] = value_str
        elif selector_type == "list":
            cols = selector_obj.get("columns")
            if isinstance(cols, str):
                cols = [cols]
            if not isinstance(cols, list):
                continue
            columns = [str(col).strip() for col in cols if str(col).strip()]
            if not columns:
                continue
            selector_obj["columns"] = list(dict.fromkeys(columns))
        elif selector_type in {"all_columns_except", "all_numeric_except"}:
            cols = selector_obj.get("except_columns")
            if cols is None:
                cols = selector_obj.get("value")
            if cols is None:
                cols = selector_obj.get("columns")
            if isinstance(cols, str):
                cols = [cols]
            if not isinstance(cols, list):
                continue
            excluded = [str(col).strip() for col in cols if str(col).strip()]
            if not excluded:
                continue
            selector_obj["except_columns"] = list(dict.fromkeys(excluded))
            selector_obj.pop("value", None)
        elif selector_type == "prefix_numeric_range":
            prefix = str(selector_obj.get("prefix") or "").strip()
            start = selector_obj.get("start")
            end = selector_obj.get("end")
            if not prefix or not isinstance(start, int) or not isinstance(end, int):
                continue
            selector_obj["prefix"] = prefix

        _push(selector_obj)

    clean_dataset["required_feature_selectors"] = repaired
    return contract


def _collect_cleaning_inventory_candidates(contract: Dict[str, Any]) -> List[str]:
    if not isinstance(contract, dict):
        return []

    candidates: List[str] = []

    def _append(values: Any) -> None:
        if isinstance(values, list):
            for value in values:
                text = str(value or "").strip()
                if text and text not in candidates:
                    candidates.append(text)

    _append(contract.get("available_columns"))
    _append(contract.get("canonical_columns"))

    column_roles = contract.get("column_roles")
    if isinstance(column_roles, dict):
        for value in column_roles.values():
            _append(value)

    allowed_feature_sets = contract.get("allowed_feature_sets")
    if isinstance(allowed_feature_sets, dict):
        for value in allowed_feature_sets.values():
            _append(value)

    for key in ("outcome_columns", "target_columns", "decision_columns"):
        _append(contract.get(key))

    for key in ("target_column", "primary_target"):
        value = str(contract.get(key) or "").strip()
        if value and value not in candidates:
            candidates.append(value)

    return candidates


def _collect_cleaning_hard_gate_columns(contract: Dict[str, Any]) -> List[str]:
    gate_columns: List[str] = []

    def _append(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                text = str(item or "").strip()
                if text and text not in gate_columns:
                    gate_columns.append(text)
        elif isinstance(value, str):
            text = value.strip()
            if text and text not in gate_columns:
                gate_columns.append(text)

    def _consume_gate_container(gates: Any) -> None:
        if not isinstance(gates, list):
            return
        for gate in gates:
            if not isinstance(gate, dict):
                continue
            severity = str(gate.get("severity") or "").strip().upper()
            if severity and severity != "HARD":
                continue
            params = gate.get("params")
            if not isinstance(params, dict):
                continue
            for key in ("columns", "required_columns", "column", "target_column", "target_columns"):
                _append(params.get(key))

    _consume_gate_container(contract.get("cleaning_gates"))
    artifact_requirements = contract.get("artifact_requirements")
    clean_dataset = artifact_requirements.get("clean_dataset") if isinstance(artifact_requirements, dict) else None
    if isinstance(clean_dataset, dict):
        _consume_gate_container(clean_dataset.get("cleaning_gates"))
    return gate_columns


def _selector_stable_name(selector: Dict[str, Any], idx: int) -> str:
    for key in ("name", "id", "family", "role", "selector_hint"):
        value = str(selector.get(key) or "").strip()
        if value:
            return value
    return f"required_selector_{idx + 1}"


def _find_matching_declared_selector(ref: str, selectors: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    token = str(ref or "").strip()
    if not token:
        return None
    for selector in selectors:
        if selector_reference_matches_any(token, [selector]):
            return selector
        if token.lower().startswith("selector:"):
            nested = token.split(":", 1)[1].strip()
            if nested and selector_reference_matches_any(nested, [selector]):
                return selector
    return None


def _expand_inline_selector_reference(ref: str, inventory: List[str]) -> List[str]:
    token = str(ref or "").strip()
    if not token:
        return []
    if token.lower().startswith("selector:"):
        token = token.split(":", 1)[1].strip()
    if not token:
        return []

    selector_obj: Dict[str, Any] | None = None
    lower = token.lower()
    if lower.startswith(("regex:", "pattern:")):
        selector_obj = {"type": "regex", "pattern": token.split(":", 1)[1].strip()}
    elif lower.startswith("prefix:"):
        selector_obj = {"type": "prefix", "value": token.split(":", 1)[1].strip()}
    elif lower.startswith("suffix:"):
        selector_obj = {"type": "suffix", "value": token.split(":", 1)[1].strip()}
    elif lower.startswith("contains:"):
        selector_obj = {"type": "contains", "value": token.split(":", 1)[1].strip()}
    if not isinstance(selector_obj, dict):
        return []

    columns, issues = expand_required_feature_selectors([selector_obj], inventory)
    if issues:
        return []
    return columns


def _deterministic_repair_clean_dataset_consistency(contract: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(contract, dict):
        return contract

    artifact_requirements = contract.get("artifact_requirements")
    clean_dataset = artifact_requirements.get("clean_dataset") if isinstance(artifact_requirements, dict) else None
    if not isinstance(clean_dataset, dict):
        return contract

    selectors = clean_dataset.get("required_feature_selectors")
    if not isinstance(selectors, list):
        return contract

    inventory = _collect_cleaning_inventory_candidates(contract)
    if not inventory:
        return contract

    transforms = clean_dataset.get("column_transformations")
    if not isinstance(transforms, dict):
        transforms = {}
        clean_dataset["column_transformations"] = transforms

    normalized_selectors: List[Dict[str, Any]] = []
    for idx, selector in enumerate(selectors):
        if not isinstance(selector, dict):
            continue
        selector_obj = dict(selector)
        selector_obj.setdefault("name", _selector_stable_name(selector_obj, idx))
        normalized_selectors.append(selector_obj)
    if normalized_selectors:
        clean_dataset["required_feature_selectors"] = normalized_selectors
        selectors = normalized_selectors

    selector_drop_reasons, _ = extract_selector_drop_reasons(transforms)
    if selector_drop_reasons:
        drop_policy = transforms.get("drop_policy")
        if not isinstance(drop_policy, dict):
            drop_policy = {}
        drop_policy["allow_selector_drops_when"] = list(selector_drop_reasons)
        transforms["drop_policy"] = drop_policy
        for alias in ("allow_selector_drops_when", "allowed_reasons", "drop_reasons"):
            if alias in transforms:
                transforms.pop(alias, None)

    scale_columns = transforms.get("scale_columns")
    if isinstance(scale_columns, list) and scale_columns:
        normalized_scale: List[str] = []
        seen_scale: set[str] = set()
        for raw_entry in scale_columns:
            entry = str(raw_entry or "").strip()
            if not entry:
                continue
            selector = _find_matching_declared_selector(entry, selectors)
            if selector is None and entry.lower().startswith("selector:"):
                nested = entry.split(":", 1)[1].strip()
                selector = _find_matching_declared_selector(nested, selectors)
                if selector is not None:
                    entry = nested
            if selector is not None:
                entry = f"selector:{_selector_stable_name(selector, selectors.index(selector))}"
                if entry not in seen_scale:
                    seen_scale.add(entry)
                    normalized_scale.append(entry)
                continue

            expanded_cols = _expand_inline_selector_reference(entry, inventory)
            if expanded_cols:
                for col in expanded_cols:
                    if col not in seen_scale:
                        seen_scale.add(col)
                        normalized_scale.append(col)
                continue

            if entry not in seen_scale:
                seen_scale.add(entry)
                normalized_scale.append(entry)
        if normalized_scale:
            transforms["scale_columns"] = normalized_scale

    if selector_drop_reasons:
        anchor_columns: List[str] = []
        for bucket in (
            clean_dataset.get("required_columns"),
            clean_dataset.get("optional_passthrough_columns"),
            _collect_cleaning_hard_gate_columns(contract),
        ):
            if isinstance(bucket, list):
                for col in bucket:
                    text = str(col or "").strip()
                    if text and text not in anchor_columns:
                        anchor_columns.append(text)
        anchor_norms = {_normalize_column_identifier(col) for col in anchor_columns if col}
        required_columns = [
            str(col).strip()
            for col in (clean_dataset.get("required_columns") or [])
            if str(col).strip()
        ]

        repaired_selectors: List[Dict[str, Any]] = []
        seen_repaired: set[str] = set()
        for idx, selector in enumerate(selectors):
            if not isinstance(selector, dict):
                continue
            matched_columns, selector_issues = expand_required_feature_selectors([selector], inventory)
            if selector_issues:
                fingerprint = json.dumps(selector, sort_keys=True, ensure_ascii=False)
                if fingerprint not in seen_repaired:
                    seen_repaired.add(fingerprint)
                    repaired_selectors.append(selector)
                continue

            overlapping = [
                col for col in matched_columns if _normalize_column_identifier(col) in anchor_norms
            ]
            if not overlapping:
                fingerprint = json.dumps(selector, sort_keys=True, ensure_ascii=False)
                if fingerprint not in seen_repaired:
                    seen_repaired.add(fingerprint)
                    repaired_selectors.append(selector)
                continue

            for col in overlapping:
                if col not in required_columns:
                    required_columns.append(col)

            remaining = [
                col for col in matched_columns if _normalize_column_identifier(col) not in anchor_norms
            ]
            if not remaining:
                continue

            materialized_selector = {
                "type": "list",
                "columns": remaining,
                "name": _selector_stable_name(selector, idx),
            }
            for key in ("id", "family", "role", "selector_hint"):
                value = selector.get(key)
                if value not in (None, "", []):
                    materialized_selector[key] = copy.deepcopy(value)
            fingerprint = json.dumps(materialized_selector, sort_keys=True, ensure_ascii=False)
            if fingerprint not in seen_repaired:
                seen_repaired.add(fingerprint)
                repaired_selectors.append(materialized_selector)

        clean_dataset["required_columns"] = required_columns
        clean_dataset["required_feature_selectors"] = repaired_selectors

    return contract


def _deterministic_repair_gate_lists(contract: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(contract, dict):
        return contract

    def _normalize_gate(gate: Any, gate_key: str, idx: int) -> Dict[str, Any] | None:
        if isinstance(gate, str):
            name = gate.strip()
            if not name:
                return None
            return {"name": name, "severity": "HARD", "params": {}}
        if not isinstance(gate, dict):
            return None

        name = ""
        for key in ("name", "id", "gate", "metric", "check", "rule", "title", "label"):
            value = gate.get(key)
            if isinstance(value, str) and value.strip():
                name = value.strip()
                break
        if not name:
            name = f"{gate_key}_{idx + 1}"

        severity = str(gate.get("severity") or gate.get("level") or "HARD").strip().upper()
        if severity not in _QA_SEVERITIES:
            severity = "HARD"
        params = gate.get("params")
        params_dict = dict(params) if isinstance(params, dict) else {}
        for key in ("metric", "check", "rule", "threshold", "condition"):
            if key in gate and key not in params_dict:
                params_dict[key] = gate.get(key)

        normalized: Dict[str, Any] = {
            "name": name,
            "severity": severity,
            "params": params_dict,
        }
        for key in ("condition", "evidence_required", "action_if_fail"):
            if key in gate and gate.get(key) not in (None, ""):
                normalized[key] = gate.get(key)
        return normalized

    for gate_key in ("cleaning_gates", "qa_gates", "reviewer_gates"):
        gates = contract.get(gate_key)
        if not isinstance(gates, list):
            continue
        repaired = []
        for idx, gate in enumerate(gates):
            normalized = _normalize_gate(gate, gate_key, idx)
            if normalized is not None:
                repaired.append(normalized)
        if repaired:
            contract[gate_key] = repaired
    return contract


def _deterministic_repair_scope(contract: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(contract, dict):
        return contract
    artifact_requirements = contract.get("artifact_requirements")
    clean_dataset = artifact_requirements.get("clean_dataset") if isinstance(artifact_requirements, dict) else None
    has_cleaning = bool(
        isinstance(clean_dataset, dict)
        or isinstance(contract.get("cleaning_gates"), list)
        or contract.get("data_engineer_runbook")
    )
    has_ml = bool(
        isinstance(contract.get("qa_gates"), list)
        or isinstance(contract.get("reviewer_gates"), list)
        or isinstance(contract.get("validation_requirements"), dict)
        or contract.get("ml_engineer_runbook")
    )

    # Always prefer structural evidence over the raw scope string:
    # if the contract encodes both cleaning and ML requirements, it is a full_pipeline contract,
    # regardless of what the LLM wrote into the scope field.
    if has_cleaning and has_ml:
        contract["scope"] = "full_pipeline"
        return contract

    # If only one side of the pipeline is specified, infer the narrowest coherent scope.
    if has_cleaning and not has_ml:
        contract["scope"] = "cleaning_only"
        return contract
    if has_ml and not has_cleaning:
        contract["scope"] = "ml_only"
        return contract

    # No strong structural signal – fall back to an existing, valid scope if present,
    # otherwise default to full_pipeline to keep downstream views maximally informed.
    scope_raw = contract.get("scope")
    scope = normalize_contract_scope(scope_raw) if scope_raw is not None else ""
    if scope in {"cleaning_only", "ml_only", "full_pipeline"}:
        contract["scope"] = scope
    else:
        contract["scope"] = "full_pipeline"
    return contract


def _deterministic_repair_derived_columns(contract: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(contract, dict):
        return contract

    normalized = _extract_derived_column_names(contract.get("derived_columns"))
    contract["derived_columns"] = normalized
    return contract


def _merge_contract_missing_fields(
    primary: Dict[str, Any] | None,
    fallback: Dict[str, Any] | None,
) -> Dict[str, Any]:
    result = copy.deepcopy(primary) if isinstance(primary, dict) else {}
    source = fallback if isinstance(fallback, dict) else {}
    for key, value in source.items():
        if key not in result or result.get(key) in (None, "", [], {}):
            result[key] = copy.deepcopy(value)
            continue
        current = result.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            result[key] = _merge_contract_missing_fields(current, value)
    return result


def _deep_merge_contract_override(
    base: Dict[str, Any] | None,
    patch: Dict[str, Any] | None,
) -> Dict[str, Any]:
    result = copy.deepcopy(base) if isinstance(base, dict) else {}
    updates = patch if isinstance(patch, dict) else {}
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge_contract_override(result.get(key), value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _synthesize_task_semantics(contract: Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(contract, dict):
        return {}

    outcome_columns = _collect_targetish_columns(contract.get("outcome_columns"))
    target_columns = list(outcome_columns)
    for col in _collect_contract_target_candidates(contract):
        if col not in target_columns:
            target_columns.append(col)
    primary_target = ""
    for source in (
        contract.get("primary_target"),
        contract.get("target_column"),
        (contract.get("task_semantics") or {}).get("primary_target") if isinstance(contract.get("task_semantics"), dict) else None,
        target_columns,
    ):
        values = _collect_targetish_columns(source)
        if values:
            primary_target = values[0]
            break

    column_roles = contract.get("column_roles")
    identifier_columns: List[str] = []
    if isinstance(column_roles, dict):
        identifier_columns = _collect_targetish_columns(
            column_roles.get("identifiers") or column_roles.get("id")
        )

    split_spec = contract.get("split_spec") if isinstance(contract.get("split_spec"), dict) else {}
    split_column = str(
        split_spec.get("split_column")
        or (contract.get("evaluation_spec") or {}).get("split_column")
        or ""
    ).strip()
    training_rows_rule = str(
        contract.get("training_rows_rule")
        or split_spec.get("training_rows_rule")
        or ""
    ).strip()
    scoring_rows_rule = str(
        contract.get("scoring_rows_rule")
        or split_spec.get("scoring_rows_rule")
        or ""
    ).strip()
    training_rows_policy = str(split_spec.get("training_rows_policy") or "").strip()
    if not training_rows_policy:
        training_rows_policy = "only_rows_with_label" if training_rows_rule else "use_all_rows"

    artifact_requirements = (
        contract.get("artifact_requirements")
        if isinstance(contract.get("artifact_requirements"), dict)
        else {}
    )
    required_outputs = []
    for item in contract.get("required_outputs") or []:
        if isinstance(item, str) and item.strip():
            required_outputs.append(item.strip())
        elif isinstance(item, dict):
            path = item.get("path") or item.get("output") or item.get("artifact")
            if isinstance(path, str) and path.strip():
                required_outputs.append(path.strip())
    prediction_artifact = ""
    artifact_candidates = list(required_outputs)
    required_files = artifact_requirements.get("required_files")
    if isinstance(required_files, list):
        for entry in required_files:
            if isinstance(entry, dict):
                path = entry.get("path") or entry.get("output") or entry.get("artifact")
            else:
                path = entry
            if isinstance(path, str) and path.strip():
                artifact_candidates.append(path.strip())
    preferred_tokens = ("submission", "scored_rows", "prediction", "predictions", "forecast")
    for token in preferred_tokens:
        for path in artifact_candidates:
            if token in path.lower():
                prediction_artifact = path
                break
        if prediction_artifact:
            break
    if not prediction_artifact and artifact_candidates:
        prediction_artifact = artifact_candidates[0]

    required_prediction_columns: List[str] = []
    scored_schema = artifact_requirements.get("scored_rows_schema")
    if isinstance(scored_schema, dict):
        required_prediction_columns = [
            str(col).strip()
            for col in (scored_schema.get("required_columns") or [])
            if str(col).strip()
        ]
    if not required_prediction_columns:
        file_schemas = artifact_requirements.get("file_schemas")
        if isinstance(file_schemas, dict) and prediction_artifact:
            schema = file_schemas.get(prediction_artifact)
            if isinstance(schema, dict):
                required_prediction_columns = [
                    str(col).strip()
                    for col in (schema.get("required_columns") or [])
                    if str(col).strip()
                ]

    capabilities = resolve_problem_capabilities_from_contract(contract)
    objective_type = str(
        (contract.get("evaluation_spec") or {}).get("objective_type")
        or (contract.get("objective_analysis") or {}).get("problem_type")
        or ""
    ).strip()

    return {
        "problem_family": str(capabilities.get("family") or "unknown"),
        "objective_type": objective_type or str(capabilities.get("family") or "unknown"),
        "target_semantics": str(capabilities.get("target_semantics") or "unknown"),
        "output_mode": str(capabilities.get("output_mode") or "generic"),
        "primary_target": primary_target or None,
        "target_columns": list(target_columns),
        "multi_target": len(target_columns) > 1,
        "prediction_unit": {
            "kind": "row",
            "identifier_columns": list(identifier_columns),
        },
        "partitioning": {
            "split_column": split_column or None,
            "training_rows_rule": training_rows_rule or None,
            "training_rows_policy": training_rows_policy or None,
            "scoring_rows_rule": scoring_rows_rule or None,
            "secondary_scoring_subset": contract.get("secondary_scoring_subset"),
        },
        "output_schema": {
            "prediction_artifact": prediction_artifact or None,
            "required_outputs": list(dict.fromkeys(required_outputs)),
            "identifier_columns": list(identifier_columns),
            "required_prediction_columns": [
                col for col in required_prediction_columns if col not in set(identifier_columns)
            ],
        },
    }


def _deterministic_repair_target_semantics(contract: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(contract, dict):
        return contract

    repaired = contract
    outcome_columns = _collect_targetish_columns(repaired.get("outcome_columns"))
    column_roles = repaired.get("column_roles")
    if isinstance(column_roles, dict):
        for key in ("outcome", "target", "label"):
            for col in _collect_targetish_columns(column_roles.get(key)):
                if col not in outcome_columns:
                    outcome_columns.append(col)

    explicit_targets = _collect_contract_target_candidates(repaired)
    for col in explicit_targets:
        if col not in outcome_columns and _looks_like_target_semantic_column(col):
            outcome_columns.append(col)

    multi_target = len(outcome_columns) > 1 or _has_multi_target_signal(
        repaired.get("business_objective"),
        repaired.get("strategy_title"),
        repaired.get("objective_analysis"),
        repaired.get("evaluation_spec"),
        repaired.get("validation_requirements"),
    )
    primary_target = ""
    for source in (
        repaired.get("target_column"),
        repaired.get("primary_target"),
        (column_roles or {}).get("outcome") if isinstance(column_roles, dict) else None,
        repaired.get("outcome_columns"),
    ):
        values = _collect_targetish_columns(source)
        if values:
            primary_target = values[0]
            break
    if not primary_target and explicit_targets:
        primary_target = explicit_targets[0]
    if not primary_target and outcome_columns:
        primary_target = outcome_columns[0]

    if multi_target and outcome_columns:
        repaired["target_columns"] = list(outcome_columns)
        if primary_target:
            repaired["primary_target"] = primary_target
        for container_key in ("objective_analysis", "evaluation_spec", "validation_requirements"):
            container = repaired.get(container_key)
            if not isinstance(container, dict):
                container = {}
            else:
                container = dict(container)
            container["target_columns"] = list(outcome_columns)
            container["primary_targets"] = list(outcome_columns)
            container["label_columns"] = list(outcome_columns)
            if primary_target:
                container["primary_target"] = primary_target
                container["target_column"] = primary_target
                container["label_column"] = primary_target
            repaired[container_key] = container
        if isinstance(column_roles, dict):
            role_copy = dict(column_roles)
            role_copy["outcome"] = list(outcome_columns)
            repaired["column_roles"] = role_copy
        else:
            repaired["column_roles"] = {"outcome": list(outcome_columns)}
        repaired["outcome_columns"] = list(outcome_columns)
        return repaired

    if primary_target:
        repaired["target_column"] = primary_target
        repaired["primary_target"] = primary_target
        if not outcome_columns:
            repaired["outcome_columns"] = [primary_target]
        for container_key in ("objective_analysis", "evaluation_spec", "validation_requirements"):
            container = repaired.get(container_key)
            if not isinstance(container, dict):
                container = {}
            else:
                container = dict(container)
            container.setdefault("primary_target", primary_target)
            container.setdefault("target_column", primary_target)
            container.setdefault("label_column", primary_target)
            repaired[container_key] = container
    return repaired


def _deterministic_repair_task_semantics(contract: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(contract, dict):
        return contract
    repaired = contract
    task_semantics = _synthesize_task_semantics(repaired)
    if task_semantics:
        repaired["task_semantics"] = task_semantics
    return repaired


def _apply_deterministic_repairs(contract: Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(contract, dict):
        return {}
    repaired = copy.deepcopy(contract)
    repaired = apply_contract_schema_registry_repairs(repaired)
    repaired = _deterministic_repair_scope(repaired)
    repaired = _deterministic_repair_target_semantics(repaired)
    repaired = _deterministic_repair_task_semantics(repaired)
    repaired = _deterministic_repair_gate_lists(repaired)
    repaired = _deterministic_repair_required_feature_selectors(repaired)
    repaired = _deterministic_repair_clean_dataset_consistency(repaired)
    repaired = _deterministic_repair_column_dtype_targets(repaired)
    repaired = _deterministic_repair_derived_columns(repaired)
    return repaired


def _validation_result_accepted(validation_result: Dict[str, Any] | None) -> bool:
    if not isinstance(validation_result, dict):
        return False
    if not bool(validation_result.get("accepted", False)):
        return False
    if str(validation_result.get("status") or "").lower() == "error":
        return False
    summary = validation_result.get("summary")
    if isinstance(summary, dict):
        try:
            if int(summary.get("error_count", 0) or 0) > 0:
                return False
        except Exception:
            return False
    return True


def _build_contract_repair_hints(
    validation_result: Dict[str, Any] | None,
    max_hints: int = 5,
) -> List[str]:
    if not isinstance(validation_result, dict):
        return []
    issues = validation_result.get("issues")
    if not isinstance(issues, list):
        return []

    severity_rank = {"error": 0, "fail": 0, "warning": 1, "info": 2}
    ranked: List[Tuple[int, str, str]] = []
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        rule = str(issue.get("rule") or "").strip()
        if not rule:
            continue
        severity = str(issue.get("severity") or "warning").lower()
        message = str(issue.get("message") or "").strip()
        ranked.append((severity_rank.get(severity, 3), rule, message))

    ranked.sort(key=lambda row: row[0])
    hints: List[str] = []
    seen_rules: set[str] = set()
    for _, rule, message in ranked:
        if rule in seen_rules:
            continue
        seen_rules.add(rule)
        action = get_contract_schema_repair_action(rule)
        hint_text = action or message or "Apply minimal structural fix for this rule."
        hints.append(f"{rule}: {hint_text}")
        if len(hints) >= max(3, min(max_hints, 5)):
            break
    return hints


def _decode_json_pointer_token(token: str) -> str:
    return str(token).replace("~1", "/").replace("~0", "~")


def _apply_json_patch_ops(document: Dict[str, Any], patch_ops: List[Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = copy.deepcopy(document) if isinstance(document, dict) else {}
    if not isinstance(patch_ops, list):
        return result

    for op in patch_ops:
        if not isinstance(op, dict):
            continue
        action = str(op.get("op") or "").strip().lower()
        path = str(op.get("path") or "").strip()
        if action not in {"add", "replace", "remove"} or not path.startswith("/"):
            continue
        tokens = [_decode_json_pointer_token(tok) for tok in path.lstrip("/").split("/") if tok != ""]
        if not tokens:
            continue
        parent: Any = result
        for token in tokens[:-1]:
            if isinstance(parent, dict):
                nxt = parent.get(token)
                if not isinstance(nxt, (dict, list)):
                    if action in {"add", "replace"}:
                        parent[token] = {}
                        nxt = parent[token]
                    else:
                        parent = None
                        break
                parent = nxt
            elif isinstance(parent, list):
                try:
                    index = int(token)
                except Exception:
                    parent = None
                    break
                if index < 0 or index >= len(parent):
                    parent = None
                    break
                parent = parent[index]
            else:
                parent = None
                break
        if parent is None:
            continue

        leaf = tokens[-1]
        if isinstance(parent, dict):
            if action in {"add", "replace"}:
                parent[leaf] = copy.deepcopy(op.get("value"))
            elif action == "remove":
                parent.pop(leaf, None)
            continue

        if isinstance(parent, list):
            if leaf == "-":
                index = len(parent)
            else:
                try:
                    index = int(leaf)
                except Exception:
                    continue
            if action == "add":
                if 0 <= index <= len(parent):
                    parent.insert(index, copy.deepcopy(op.get("value")))
            elif action == "replace":
                if 0 <= index < len(parent):
                    parent[index] = copy.deepcopy(op.get("value"))
            elif action == "remove":
                if 0 <= index < len(parent):
                    parent.pop(index)

    return result


def _apply_minimal_contract_patch(contract: Dict[str, Any], patch_payload: Any) -> Dict[str, Any]:
    base = copy.deepcopy(contract) if isinstance(contract, dict) else {}
    if isinstance(patch_payload, list):
        return _apply_json_patch_ops(base, patch_payload)
    if not isinstance(patch_payload, dict):
        return base
    if {"op", "path"}.issubset(set(patch_payload.keys())):
        return _apply_json_patch_ops(base, [patch_payload])
    patch_ops = patch_payload.get("patch")
    if isinstance(patch_ops, list):
        return _apply_json_patch_ops(base, patch_ops)
    changes = patch_payload.get("changes")
    if isinstance(changes, dict):
        return _deep_merge_contract_override(base, changes)
    return _deep_merge_contract_override(base, patch_payload)


def _validate_repair_revalidate_loop(
    contract: Dict[str, Any] | None,
    validator_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    repair_provider: Callable[[Dict[str, Any], Dict[str, Any], List[str], int], Any] | None = None,
    max_iterations: int = 2,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    working = copy.deepcopy(contract) if isinstance(contract, dict) else {}
    trace: List[Dict[str, Any]] = []
    last_validation: Dict[str, Any] = {
        "status": "error",
        "accepted": False,
        "issues": [{"rule": "contract.invalid", "severity": "error", "message": "Invalid contract payload"}],
        "summary": {"error_count": 1, "warning_count": 0},
    }

    preserved_deliverables = None
    spec = working.get("spec_extraction")
    if isinstance(spec, dict) and isinstance(spec.get("deliverables"), list):
        preserved_deliverables = copy.deepcopy(spec.get("deliverables"))
    preserved_rich_outputs = (
        copy.deepcopy(working.get("required_output_artifacts"))
        if isinstance(working.get("required_output_artifacts"), list)
        else None
    )

    attempts = max(0, int(max_iterations)) + 1
    for attempt in range(1, attempts + 1):
        working = _apply_planner_structural_support(working)
        if preserved_deliverables is not None:
            spec_obj = working.get("spec_extraction")
            if not isinstance(spec_obj, dict):
                spec_obj = {}
            spec_obj["deliverables"] = copy.deepcopy(preserved_deliverables)
            working["spec_extraction"] = spec_obj
        if preserved_rich_outputs is not None:
            working["required_output_artifacts"] = copy.deepcopy(preserved_rich_outputs)

        try:
            last_validation = validator_fn(copy.deepcopy(working))
        except Exception as err:
            last_validation = {
                "status": "error",
                "accepted": False,
                "issues": [
                    {
                        "rule": "contract.validation_exception",
                        "severity": "error",
                        "message": str(err),
                    }
                ],
                "summary": {"error_count": 1, "warning_count": 0},
            }

        accepted = _validation_result_accepted(last_validation)
        trace.append(
            {
                "attempt": attempt,
                "accepted": accepted,
                "status": str(last_validation.get("status") or "unknown"),
                "hints": [],
            }
        )
        if accepted:
            break
        if attempt >= attempts or repair_provider is None:
            break

        hints = _build_contract_repair_hints(last_validation, max_hints=5)
        trace[-1]["hints"] = hints
        patch_payload = repair_provider(copy.deepcopy(working), copy.deepcopy(last_validation), hints, attempt)
        if patch_payload in (None, "", [], {}):
            break
        working = _apply_minimal_contract_patch(working, patch_payload)

    if preserved_deliverables is not None:
        spec_obj = working.get("spec_extraction")
        if not isinstance(spec_obj, dict):
            spec_obj = {}
        spec_obj["deliverables"] = copy.deepcopy(preserved_deliverables)
        working["spec_extraction"] = spec_obj
    if preserved_rich_outputs is not None:
        working["required_output_artifacts"] = copy.deepcopy(preserved_rich_outputs)

    return working, last_validation, trace


def build_contract_min(
    full_contract_or_partial: Dict[str, Any] | None,
    strategy: Dict[str, Any] | None,
    column_inventory: List[str] | None,
    relevant_columns: List[str] | None,
    target_candidates: List[Dict[str, Any]] | None = None,
    data_profile: Dict[str, Any] | None = None,
    business_objective_hint: str = "",
) -> Dict[str, Any]:
    """
    Build a compact contract_min that aligns agents on relevant columns and gates.

    Args:
        data_profile: Optional data profile containing constant_columns to exclude from
                     clean_dataset.required_columns. Columns marked as constant are
                     excluded from required_columns since they provide no information.
    """
    contract = copy.deepcopy(full_contract_or_partial) if isinstance(full_contract_or_partial, dict) else {}
    contract = _deterministic_repair_target_semantics(contract)
    strategy_dict = strategy if isinstance(strategy, dict) else {}
    inventory = [str(col) for col in (column_inventory or []) if col is not None]
    inventory_norms = {_normalize_column_identifier(col): col for col in inventory}
    canonical_order = {col: idx for idx, col in enumerate(inventory)}

    canonical_columns: List[str] = []
    contract_canonical = contract.get("canonical_columns")
    if isinstance(contract_canonical, list) and contract_canonical:
        canonical_columns = [str(col) for col in contract_canonical if col]
        # Guard: if contract canonical_columns covers < 50% of inventory,
        # the LLM output was likely truncated. Enrich with inventory columns.
        if inventory and len(canonical_columns) < len(inventory) * 0.5:
            print(f"CANONICAL_COLUMNS_GUARD: Contract canonical ({len(canonical_columns)}) covers <50% of inventory ({len(inventory)}). Enriching.")
            for col in inventory:
                if col not in canonical_columns:
                    canonical_columns.append(col)
    else:
        for col in (relevant_columns or []):
            if not col:
                continue
            if col in inventory:
                canonical_columns.append(col)
                continue
            norm = _normalize_column_identifier(col)
            resolved = inventory_norms.get(norm)
            if resolved:
                canonical_columns.append(resolved)

        if not canonical_columns:
            for col in _coerce_list(strategy_dict.get("required_columns")):
                norm = _normalize_column_identifier(col)
                resolved = inventory_norms.get(norm)
                if resolved:
                    canonical_columns.append(resolved)
        canonical_columns = list(dict.fromkeys(canonical_columns))

    def _filter_to_canonical(cols: List[str]) -> List[str]:
        canon_set = set(canonical_columns)
        return [col for col in cols if col in canon_set]

    roles_raw = contract.get("column_roles", {}) if isinstance(contract.get("column_roles"), dict) else {}
    roles = roles_raw
    if isinstance(roles_raw, dict) and roles_raw:
        role_keys = {
            "pre_decision",
            "decision",
            "outcome",
            "post_decision_audit_only",
            "unknown",
            "identifiers",
            "time_columns",
        }
        if all(key in role_keys for key in roles_raw.keys()):
            if all(isinstance(val, str) for val in roles_raw.values()):
                roles = {}
    if roles and all(isinstance(val, dict) and "role" in val for val in roles.values()):
        role_lists = {
            "pre_decision": [],
            "decision": [],
            "outcome": [],
            "post_decision_audit_only": [],
            "unknown": [],
        }
        role_aliases = {
            "pre_decision": "pre_decision",
            "predecision": "pre_decision",
            "pre_decision_features": "pre_decision",
            "decision": "decision",
            "decisions": "decision",
            "outcome": "outcome",
            "target": "outcome",
            "label": "outcome",
            "post_decision_audit_only": "post_decision_audit_only",
            "post_decision": "post_decision_audit_only",
            "audit_only": "post_decision_audit_only",
            "audit": "post_decision_audit_only",
            "unknown": "unknown",
        }
        for col, meta in roles.items():
            role_raw = str(meta.get("role") or "").strip().lower()
            role_key = re.sub(r"[^a-z0-9]+", "_", role_raw).strip("_")
            role_bucket = role_aliases.get(role_key, "unknown")
            norm = _normalize_column_identifier(col)
            resolved = inventory_norms.get(norm) or col
            role_lists[role_bucket].append(resolved)
        roles = role_lists
    roles_present = any(
        _coerce_list(roles.get(key))
        for key in ("pre_decision", "decision", "outcome", "post_decision_audit_only")
    )
    role_pre = _filter_to_canonical(_coerce_list(roles.get("pre_decision"))) if roles else []
    role_decision = _filter_to_canonical(_coerce_list(roles.get("decision"))) if roles else []
    role_outcome = _filter_to_canonical(_coerce_list(roles.get("outcome"))) if roles else []
    role_audit = _filter_to_canonical(_coerce_list(roles.get("post_decision_audit_only"))) if roles else []

    outcome_cols: List[str] = []
    decision_cols: List[str] = []
    audit_only_cols: List[str] = []
    identifiers: List[str] = []
    time_columns: List[str] = []

    def _resolve_candidate_targets() -> List[str]:
        resolved_targets: List[str] = []
        if not target_candidates:
            return resolved_targets
        # Only accept candidates with name-based evidence (score >= 2.0).
        # Pure statistical signals like low_cardinality (score ~1.0) are not
        # sufficient to classify a column as an outcome/target — they indicate
        # categorical features, not prediction targets.
        _MIN_TARGET_SCORE = 2.0
        for item in target_candidates:
            if not isinstance(item, dict):
                continue
            candidate_score = item.get("score")
            if isinstance(candidate_score, (int, float)) and candidate_score < _MIN_TARGET_SCORE:
                continue
            raw = item.get("column") or item.get("name") or item.get("candidate")
            if not raw:
                continue
            if raw in canonical_columns:
                resolved_targets.append(raw)
                continue
            norm = _normalize_column_identifier(raw)
            resolved = inventory_norms.get(norm)
            if resolved:
                if resolved not in canonical_columns:
                    canonical_columns.append(resolved)
                resolved_targets.append(resolved)
        return list(dict.fromkeys(resolved_targets))

    def _collect_declared_target_refs() -> List[str]:
        refs: List[str] = []

        def _append(values: Any) -> None:
            for col in _collect_targetish_columns(values):
                if col not in refs:
                    refs.append(col)

        for source in (
            strategy_dict.get("outcome_columns"),
            strategy_dict.get("target_column"),
            strategy_dict.get("target_columns"),
            contract.get("outcome_columns"),
            contract.get("target_column"),
            contract.get("target_columns"),
        ):
            _append(source)
        if isinstance(contract.get("column_roles"), dict):
            _append((contract.get("column_roles") or {}).get("outcome"))
        for container in ("objective_analysis", "evaluation_spec", "validation_requirements"):
            payload = contract.get(container)
            if not isinstance(payload, dict):
                continue
            for key in ("primary_target", "primary_targets", "target_column", "target_columns", "label_column", "label_columns"):
                _append(payload.get(key))
        return refs

    declared_target_refs = _collect_declared_target_refs()

    def _infer_multi_target_columns(anchor_targets: List[str]) -> List[str]:
        candidates = list(dict.fromkeys([col for col in anchor_targets if col]))
        target_like_columns = [col for col in canonical_columns if _looks_like_target_semantic_column(col)]
        if len(candidates) > 1:
            return sorted(candidates, key=lambda col: canonical_order.get(col, len(canonical_order)))
        if not _has_multi_target_signal(
            strategy_dict,
            contract.get("business_objective"),
            business_objective_hint,
            contract.get("strategy_title"),
            contract.get("evaluation_spec"),
            contract.get("objective_analysis"),
        ):
            return candidates

        anchor_families = {
            _target_family_signature(col)
            for col in candidates
            if _target_family_signature(col)
        }
        for col in target_like_columns:
            family = _target_family_signature(col)
            if anchor_families and family not in anchor_families:
                continue
            if col not in candidates:
                candidates.append(col)
        if len(candidates) <= 1 and len(target_like_columns) >= 2:
            shared_families = {
                _target_family_signature(col)
                for col in target_like_columns
                if _target_family_signature(col)
            }
            if len(shared_families) == 1:
                candidates = list(dict.fromkeys(target_like_columns))
        return sorted(candidates, key=lambda col: canonical_order.get(col, len(canonical_order)))

    if roles_present:
        outcome_cols = list(role_outcome)
        decision_cols = list(role_decision)
        audit_only_cols = list(role_audit)
        if not outcome_cols:
            outcome_cols = _filter_to_canonical(_resolve_candidate_targets())
    else:
        outcome_candidates = []
        decision_candidates = []
        audit_candidates = []
        outcome_candidates.extend(_coerce_list(strategy_dict.get("outcome_columns")))
        outcome_candidates.extend(_coerce_list(strategy_dict.get("target_column")))
        outcome_candidates.extend(_coerce_list(strategy_dict.get("target_columns")))
        outcome_candidates.extend(_coerce_list(contract.get("outcome_columns")))
        decision_candidates.extend(_coerce_list(strategy_dict.get("decision_columns")))
        decision_candidates.extend(_coerce_list(strategy_dict.get("decision_variables")))
        decision_candidates.extend(_coerce_list(contract.get("decision_columns")))
        decision_candidates.extend(_coerce_list(contract.get("decision_variables")))
        audit_candidates.extend(_coerce_list(strategy_dict.get("audit_only_columns")))
        if not outcome_candidates:
            outcome_candidates.extend(_resolve_candidate_targets())
        outcome_cols = _filter_to_canonical([col for col in outcome_candidates if col])
        decision_cols = _filter_to_canonical([col for col in decision_candidates if col])
        audit_only_cols = _filter_to_canonical([col for col in audit_candidates if col])

    inferred_multi_targets = _filter_to_canonical(
        _infer_multi_target_columns(outcome_cols or declared_target_refs or _resolve_candidate_targets())
    )
    if len(inferred_multi_targets) > len(outcome_cols):
        outcome_cols = inferred_multi_targets

    # STEWARD-FIRST ROLE INFERENCE
    # Trust the Steward's dataset_semantics.json over regex heuristics.
    # Only fall back to regex if Steward didn't provide the information.
    steward_semantics = {}
    if isinstance(data_profile, dict):
        steward_semantics = data_profile.get("dataset_semantics", {})
        if not isinstance(steward_semantics, dict):
            steward_semantics = {}

    # Extract Steward-identified roles
    steward_identifiers = _coerce_list(steward_semantics.get("identifier_columns"))
    steward_time_cols = _coerce_list(steward_semantics.get("time_columns"))
    steward_categorical = _coerce_list(steward_semantics.get("categorical_columns"))
    steward_target_columns: List[str] = []
    for source in (
        steward_semantics.get("target_columns"),
        steward_semantics.get("primary_targets"),
        (steward_semantics.get("target_analysis") or {}).get("target_columns")
        if isinstance(steward_semantics.get("target_analysis"), dict)
        else None,
        (steward_semantics.get("target_analysis") or {}).get("primary_targets")
        if isinstance(steward_semantics.get("target_analysis"), dict)
        else None,
    ):
        for col in _collect_targetish_columns(source):
            if col not in steward_target_columns:
                steward_target_columns.append(col)

    steward_primary_target: str | None = None
    for source in (
        steward_semantics.get("primary_target"),
        (steward_semantics.get("target_analysis") or {}).get("primary_target")
        if isinstance(steward_semantics.get("target_analysis"), dict)
        else None,
        data_profile.get("primary_target") if isinstance(data_profile, dict) else None,
        data_profile.get("target_column") if isinstance(data_profile, dict) else None,
        data_profile.get("target") if isinstance(data_profile, dict) else None,
        steward_target_columns,
    ):
        values = _collect_targetish_columns(source)
        if values:
            steward_primary_target = values[0]
            break
    # split_candidates may contain dicts ({"column": "is_train", ...}) or strings.
    # Do NOT use _coerce_list here — it converts dicts to their str() repr, making
    # _resolve_split_column unable to extract the column name from dict entries.
    def _coerce_split_candidates(raw: Any) -> list:
        if not raw:
            return []
        if isinstance(raw, list):
            return [item for item in raw if item]
        if isinstance(raw, (str, dict)):
            return [raw]
        return []

    steward_split_cols = _coerce_split_candidates(steward_semantics.get("split_candidates"))
    if not steward_split_cols:
        steward_split_cols = _coerce_split_candidates(data_profile.get("split_candidates")) if data_profile else []

    # Also check data_profile top-level for backward compatibility
    if not steward_identifiers:
        steward_identifiers = _coerce_list(data_profile.get("identifier_columns")) if data_profile else []
    if not steward_time_cols:
        steward_time_cols = _coerce_list(data_profile.get("time_columns")) if data_profile else []

    # Use Steward's analysis if available
    if steward_identifiers:
        for col in steward_identifiers:
            if col in canonical_columns and col not in identifiers:
                identifiers.append(col)
        print(f"STEWARD_IDENTIFIERS: Using Steward-provided identifiers: {identifiers}")

    if steward_time_cols:
        for col in steward_time_cols:
            if col in canonical_columns and col not in time_columns:
                time_columns.append(col)
        print(f"STEWARD_TIME_COLUMNS: Using Steward-provided time columns: {time_columns}")

    # Inject Steward's primary_target into outcome_cols when the LLM/strategy
    # failed to populate it.  This is the single-point fix that prevents the
    # cascade: empty outcome_columns → missing target_col in heavy_runner →
    # "target column missing" abort → ML Engineer never executes.
    if not outcome_cols and steward_target_columns:
        resolved_steward_targets = _filter_to_canonical(steward_target_columns)
        if resolved_steward_targets:
            outcome_cols = resolved_steward_targets
            print(f"STEWARD_TARGET_COLUMNS: Injected {resolved_steward_targets} into outcome_cols (was empty)")
    if not outcome_cols and steward_primary_target:
        if steward_primary_target in canonical_columns:
            outcome_cols = [steward_primary_target]
            print(f"STEWARD_PRIMARY_TARGET: Injected '{steward_primary_target}' into outcome_cols (was empty)")

    # FALLBACK: Only use regex heuristics if Steward didn't provide role information
    if not steward_identifiers or not steward_time_cols:
        token_patterns = {
            "id_like": re.compile(r"\b(id|uuid|key)\b"),
            "time_like": re.compile(r"\b(date|time|timestamp)\b"),
        }
        for col in canonical_columns:
            tokenized = _tokenize_name(col)
            if not steward_identifiers and token_patterns["id_like"].search(tokenized):
                if col not in identifiers:
                    identifiers.append(col)
            if not steward_time_cols and token_patterns["time_like"].search(tokenized):
                if col not in time_columns:
                    time_columns.append(col)
        if not steward_identifiers and identifiers:
            print(f"REGEX_FALLBACK_IDENTIFIERS: {identifiers}")
        if not steward_time_cols and time_columns:
            print(f"REGEX_FALLBACK_TIME_COLUMNS: {time_columns}")

    outcome_cols = list(dict.fromkeys(outcome_cols))
    decision_cols = list(dict.fromkeys(decision_cols))
    audit_only_cols = list(dict.fromkeys(audit_only_cols))
    identifiers = list(dict.fromkeys(identifiers))
    time_columns = list(dict.fromkeys(time_columns))

    inferred_multi_targets = _filter_to_canonical(_infer_multi_target_columns(outcome_cols or declared_target_refs))
    if len(inferred_multi_targets) > len(outcome_cols):
        outcome_cols = inferred_multi_targets

    assigned = set(outcome_cols + decision_cols + audit_only_cols + identifiers + time_columns)
    if roles_present:
        pre_decision = list(role_pre)
    else:
        pre_decision = [col for col in canonical_columns if col not in assigned]

    column_roles: Dict[str, List[str]] = {
        "pre_decision": pre_decision,
        "decision": decision_cols,
        "outcome": outcome_cols,
        "post_decision_audit_only": audit_only_cols,
        "unknown": [],
    }
    if identifiers:
        column_roles["identifiers"] = identifiers
    if time_columns:
        column_roles["time_columns"] = time_columns

    segmentation_features = list(pre_decision)
    model_features = list(pre_decision)
    if decision_cols:
        for col in decision_cols:
            if col not in model_features:
                model_features.append(col)
    forbidden_features = list(dict.fromkeys(outcome_cols + audit_only_cols))

    # Exclude split/partition columns from model_features — they are structural
    # (constant in training set) and provide no predictive signal.
    # steward_split_cols may contain dicts or strings; extract column names.
    _split_col_names: set[str] = set()
    for _sc in steward_split_cols:
        if isinstance(_sc, str) and _sc.strip():
            _split_col_names.add(_sc.strip())
        elif isinstance(_sc, dict):
            _scn = str(_sc.get("column") or _sc.get("name") or "").strip()
            if _scn:
                _split_col_names.add(_scn)
    if _split_col_names:
        filtered_model = [col for col in model_features if col not in _split_col_names]
        removed_splits = [col for col in model_features if col in _split_col_names]
        if removed_splits:
            model_features = filtered_model
            print(f"SPLIT_COLUMN_GUARD: Excluded split columns from model_features: {removed_splits}")

    allowed_sets_full = contract.get("allowed_feature_sets")
    if not isinstance(allowed_sets_full, dict):
        allowed_sets_full = {}
    missing_sets: List[str] = []

    def _full_list(key: str) -> tuple[list[str] | None, bool]:
        val = allowed_sets_full.get(key)
        if isinstance(val, list):
            return [str(c) for c in val if c is not None], True
        return None, False

    full_model, has_model = _full_list("model_features")
    full_seg, has_seg = _full_list("segmentation_features")
    full_forbidden, has_forbidden = _full_list("forbidden_for_modeling")
    if not has_forbidden:
        full_forbidden, has_forbidden = _full_list("forbidden_features")
    full_audit, has_audit = _full_list("audit_only_features")

    if has_model:
        model_features = full_model
    else:
        missing_sets.append("model_features")
    if has_seg:
        segmentation_features = full_seg
    else:
        missing_sets.append("segmentation_features")
    if has_forbidden:
        forbidden_features = full_forbidden
    else:
        missing_sets.append("forbidden_for_modeling")
    if not has_audit:
        missing_sets.append("audit_only_features")

    audit_only_features = full_audit if has_audit else list(audit_only_cols)
    if missing_sets:
        print(f"FALLBACK_FEATURE_SETS: {', '.join(sorted(set(missing_sets)))}")

    identifier_candidates = _resolve_identifier_candidates(contract, canonical_columns)
    if identifier_candidates:
        filtered_model = [col for col in model_features if col not in identifier_candidates]
        removed_ids = [col for col in model_features if col not in filtered_model]
        if removed_ids:
            model_features = filtered_model
            for col in removed_ids:
                if col not in audit_only_features:
                    audit_only_features.append(col)

    full_artifact_requirements = contract.get("artifact_requirements")
    if not isinstance(full_artifact_requirements, dict):
        full_artifact_requirements = {}

    # Dynamic required_outputs: trust deliverables-driven outputs from contract.
    # Safety nets removed — invariants are enforced by the deliverable linter.

    def _extract_paths_from_outputs(raw_outputs: list) -> List[str]:
        """Extract path strings from a list that may contain str or dict items."""
        paths: List[str] = []
        for item in raw_outputs or []:
            if isinstance(item, dict):
                path = item.get("path") or item.get("output") or item.get("artifact")
            elif isinstance(item, str):
                path = item
            else:
                continue
            if path and is_probably_path(str(path)):
                paths.append(str(path))
        return paths

    llm_outputs = contract.get("required_outputs")
    if isinstance(llm_outputs, list) and llm_outputs:
        required_outputs = _extract_paths_from_outputs(llm_outputs)
    else:
        # Fallback: artifact_requirements
        full_required_files = _extract_required_paths(full_artifact_requirements)
        required_outputs = list(dict.fromkeys(full_required_files or []))

    # P1.5: Infer feature selectors for wide datasets
    feature_selectors = []
    if len(canonical_columns) > 200:
        feature_selectors, remaining_cols = infer_feature_selectors(
            canonical_columns, max_list_size=200, min_group_size=50
        )
        if feature_selectors:
            print(f"FEATURE_SELECTORS: Inferred {len(feature_selectors)} selectors for {len(canonical_columns)} columns")

    declared_feature_selectors: List[Dict[str, Any]] = []
    full_clean_dataset = (
        full_artifact_requirements.get("clean_dataset")
        if isinstance(full_artifact_requirements, dict)
        else None
    )
    if isinstance(full_clean_dataset, dict) and full_clean_dataset.get("required_feature_selectors") is not None:
        selector_probe = {
            "artifact_requirements": {
                "clean_dataset": {
                    "required_feature_selectors": copy.deepcopy(full_clean_dataset.get("required_feature_selectors"))
                }
            }
        }
        selector_probe = _deterministic_repair_required_feature_selectors(selector_probe)
        declared_feature_selectors = (
            selector_probe.get("artifact_requirements", {})
            .get("clean_dataset", {})
            .get("required_feature_selectors", [])
        )
        if not isinstance(declared_feature_selectors, list):
            declared_feature_selectors = []
    if declared_feature_selectors:
        merged_selectors: List[Dict[str, Any]] = []
        seen_selectors: set[str] = set()
        for selector in declared_feature_selectors + (feature_selectors if isinstance(feature_selectors, list) else []):
            if not isinstance(selector, dict):
                continue
            fingerprint = json.dumps(selector, sort_keys=True, ensure_ascii=False)
            if fingerprint in seen_selectors:
                continue
            seen_selectors.add(fingerprint)
            merged_selectors.append(selector)
        feature_selectors = merged_selectors

    # P1.1: Determine scored_rows required columns based on objective type
    objective_type = contract.get("objective_type") or strategy_dict.get("objective_type") or ""

    # P1.5: Infer identifier column from canonical_columns instead of hardcoding "id"
    # Pattern matches: id, ID, _id, Id, row_id, etc.
    id_pattern = re.compile(r"^id$|^ID$|_id$|Id$|^row_id$|^index$", re.IGNORECASE)
    id_column = None
    for col in canonical_columns:
        if id_pattern.search(col):
            id_column = col
            break

    # P1.6: Keep required_columns minimal (only id if detected)
    scored_rows_required_columns = [id_column] if id_column else []

    # P1.6: Build universal any-of groups (no dataset hardcodes)
    required_any_of_groups = []
    required_any_of_group_severity = []

    # Group 1 (identificador): incluir id detectado + sinónimos genéricos
    group1 = ["id", "row_id", "index", "record_id", "case_id"]
    if id_column and id_column not in group1:
        group1.insert(0, id_column)
    required_any_of_groups.append(group1)
    required_any_of_group_severity.append("warning")  # Identifier is optional (warning)

    # Group 2 (predicción/score): sinónimos universales + target column names.
    # The ML Engineer often writes the prediction column using the original
    # target name (e.g. "Churn" instead of "probability").  Including the
    # resolved outcome columns ensures this common pattern is accepted.
    prediction_synonyms = [
        "prediction", "pred", "probability", "prob", "score",
        "risk_score", "predicted_prob", "predicted_value", "y_pred",
    ]
    for oc in outcome_cols:
        oc_clean = str(oc).strip()
        if oc_clean and oc_clean.lower() not in {s.lower() for s in prediction_synonyms}:
            prediction_synonyms.append(oc_clean)
    required_any_of_groups.append(prediction_synonyms)
    required_any_of_group_severity.append("fail")  # Prediction/score is critical (fail)

    # Group 3 (ranking/prioridad) solo si la familia del problema lo requiere.
    capabilities = resolve_problem_capabilities_from_contract(
        contract,
        objective_text=str(contract.get("business_objective") or ""),
    )
    if is_problem_family(capabilities, "ranking"):
        required_any_of_groups.append(["priority", "rank", "ranking", "triage_priority"])
        required_any_of_group_severity.append("fail")  # Ranking is critical when required

    scored_rows_schema = {
        "required_columns": scored_rows_required_columns,
        "required_any_of_groups": required_any_of_groups,
        "required_any_of_group_severity": required_any_of_group_severity,
        "recommended_columns": [],
    }
    scored_rows_schema = _merge_scored_rows_schema(
        scored_rows_schema,
        full_artifact_requirements.get("scored_rows_schema"),
    )
    decisioning_requirements = _align_decisioning_requirements_with_schema(
        contract.get("decisioning_requirements", {}),
        scored_rows_schema,
    )
    decisioning_required = _extract_decisioning_required_column_names(decisioning_requirements)
    if decisioning_required:
        scored_rows_schema["required_columns"] = _merge_unique_values(
            scored_rows_schema.get("required_columns", []) or [],
            decisioning_required,
        )
    required_cols_for_anyof = scored_rows_schema.get("required_columns")
    anyof_groups = scored_rows_schema.get("required_any_of_groups")
    if isinstance(required_cols_for_anyof, list) and isinstance(anyof_groups, list):
        prediction_group = None
        for group in anyof_groups:
            if not isinstance(group, list):
                continue
            if any(_is_prediction_like_column(item) for item in group):
                prediction_group = group
                break
        if prediction_group is not None:
            seen = {_normalize_column_token(item) for item in prediction_group if item}
            for col in required_cols_for_anyof:
                if not col or not _is_prediction_like_column(col):
                    continue
                norm = _normalize_column_token(col)
                if norm in seen:
                    continue
                prediction_group.append(col)
                seen.add(norm)

    required_files: List[Dict[str, Any]] = []
    if isinstance(full_artifact_requirements.get("required_files"), list):
        for entry in full_artifact_requirements.get("required_files") or []:
            if not entry:
                continue
            if isinstance(entry, dict):
                path = entry.get("path") or entry.get("output") or entry.get("artifact")
                if path and is_probably_path(str(path)):
                    required_files.append(
                        {"path": str(path), "description": str(entry.get("description") or "")}
                    )
            else:
                path = str(entry)
                if path and is_probably_path(path):
                    required_files.append({"path": path, "description": ""})

    for path in required_outputs:
        if not path:
            continue
        if not is_probably_path(str(path)):
            continue
        if any(str(path).lower() == str(item.get("path", "")).lower() for item in required_files):
            continue
        required_files.append({"path": str(path), "description": ""})

    def _coerce_positive_count(value: Any) -> Optional[int]:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            parsed = int(value)
            return parsed if parsed > 0 else None
        if isinstance(value, str):
            token = value.strip()
            if not token:
                return None
            try:
                parsed = int(float(token))
            except Exception:
                return None
            return parsed if parsed > 0 else None
        return None

    def _extract_row_count_hints(
        contract_payload: Dict[str, Any],
        profile_payload: Dict[str, Any] | None,
    ) -> Dict[str, int]:
        hints: Dict[str, int] = {}
        train_keys = (
            "n_train_rows",
            "train_rows",
            "n_train",
            "rows_train",
            "train_count",
        )
        test_keys = (
            "n_test_rows",
            "test_rows",
            "n_test",
            "rows_test",
            "test_count",
        )
        total_keys = (
            "n_total_rows",
            "total_rows",
            "n_rows",
            "row_count",
            "rows",
        )
        candidate_targets = [str(col) for col in (outcome_cols or []) if str(col).strip()]

        def _scan(source: Any) -> None:
            if not isinstance(source, dict):
                return
            if "n_train" not in hints:
                for key in train_keys:
                    parsed = _coerce_positive_count(source.get(key))
                    if parsed is not None:
                        hints["n_train"] = parsed
                        break
            if "n_test" not in hints:
                for key in test_keys:
                    parsed = _coerce_positive_count(source.get(key))
                    if parsed is not None:
                        hints["n_test"] = parsed
                        break
            if "n_total" not in hints:
                for key in total_keys:
                    parsed = _coerce_positive_count(source.get(key))
                    if parsed is not None:
                        hints["n_total"] = parsed
                        break
            basic_stats = source.get("basic_stats")
            if isinstance(basic_stats, dict):
                if "n_total" not in hints:
                    parsed = _coerce_positive_count(
                        basic_stats.get("n_rows")
                        or basic_stats.get("rows")
                        or basic_stats.get("row_count")
                    )
                    if parsed is not None:
                        hints["n_total"] = parsed
                if "n_train" not in hints:
                    parsed = _coerce_positive_count(
                        basic_stats.get("n_train_rows")
                        or basic_stats.get("train_rows")
                        or basic_stats.get("n_train")
                    )
                    if parsed is not None:
                        hints["n_train"] = parsed
                if "n_test" not in hints:
                    parsed = _coerce_positive_count(
                        basic_stats.get("n_test_rows")
                        or basic_stats.get("test_rows")
                        or basic_stats.get("n_test")
                    )
                    if parsed is not None:
                        hints["n_test"] = parsed

        def _coerce_ratio(value: Any) -> float | None:
            if isinstance(value, bool):
                return None
            if isinstance(value, (int, float)):
                ratio = float(value)
            elif isinstance(value, str):
                token = value.strip()
                if not token:
                    return None
                try:
                    ratio = float(token)
                except Exception:
                    return None
            else:
                return None
            if ratio < 0.0 or ratio > 1.0:
                return None
            return ratio

        def _resolve_outcome_entry(source: Dict[str, Any]) -> Dict[str, Any] | None:
            outcome_analysis = source.get("outcome_analysis")
            if not isinstance(outcome_analysis, dict) or not outcome_analysis:
                return None
            for target in candidate_targets:
                entry = outcome_analysis.get(target)
                if isinstance(entry, dict):
                    return entry
            for entry in outcome_analysis.values():
                if isinstance(entry, dict):
                    return entry
            return None

        def _scan_outcome_counts(source: Any) -> None:
            if not isinstance(source, dict):
                return
            if "n_train" in hints and "n_total" in hints:
                return
            outcome_entry = _resolve_outcome_entry(source)
            if not isinstance(outcome_entry, dict):
                return

            total = _coerce_positive_count(
                outcome_entry.get("total_count")
                or outcome_entry.get("n_total_rows")
                or outcome_entry.get("n_rows")
                or outcome_entry.get("row_count")
                or outcome_entry.get("rows")
            )
            non_null = _coerce_positive_count(
                outcome_entry.get("non_null_count")
                or outcome_entry.get("n_non_null")
                or outcome_entry.get("non_null_rows")
                or outcome_entry.get("labeled_rows")
                or outcome_entry.get("train_rows")
            )
            if non_null is None and isinstance(total, int):
                null_frac = _coerce_ratio(outcome_entry.get("null_frac"))
                if null_frac is not None:
                    inferred_non_null = int(round(total * (1.0 - null_frac)))
                    if inferred_non_null > 0 and inferred_non_null <= total:
                        non_null = inferred_non_null
            if isinstance(total, int) and total > 0 and "n_total" not in hints:
                hints["n_total"] = total
            if isinstance(non_null, int) and non_null > 0 and "n_train" not in hints:
                hints["n_train"] = non_null
            if (
                isinstance(total, int)
                and total > 0
                and isinstance(non_null, int)
                and non_null >= 0
                and total >= non_null
            ):
                inferred_test = total - non_null
                if inferred_test > 0 and "n_test" not in hints:
                    hints["n_test"] = inferred_test

        profile_dict = profile_payload if isinstance(profile_payload, dict) else {}
        dataset_profile = contract_payload.get("dataset_profile") if isinstance(contract_payload.get("dataset_profile"), dict) else {}
        contract_data_profile = contract_payload.get("data_profile") if isinstance(contract_payload.get("data_profile"), dict) else {}
        evaluation_spec = contract_payload.get("evaluation_spec") if isinstance(contract_payload.get("evaluation_spec"), dict) else {}
        execution_constraints = (
            contract_payload.get("execution_constraints")
            if isinstance(contract_payload.get("execution_constraints"), dict)
            else {}
        )
        for source in (
            contract_payload,
            dataset_profile,
            contract_data_profile,
            evaluation_spec,
            execution_constraints,
            profile_dict,
        ):
            _scan(source)
            _scan_outcome_counts(source)
            if isinstance(source, dict):
                _scan_outcome_counts(source.get("dataset_profile"))
                _scan_outcome_counts(source.get("data_profile"))
                _scan_outcome_counts(source.get("evaluation_spec"))

        # Fallback: extract row counts from dataset_semantics.target_analysis.
        # The Steward stores authoritative train/test split sizes there even
        # when outcome_analysis is empty (which is common).
        if "n_train" not in hints or "n_test" not in hints:
            _ds = profile_dict.get("dataset_semantics")
            if isinstance(_ds, dict):
                _ta = _ds.get("target_analysis")
                if isinstance(_ta, dict):
                    _ta_total = _coerce_positive_count(
                        _ta.get("target_total_count_exact")
                        or _ta.get("total_count")
                        or _ta.get("n_rows")
                    )
                    _ta_missing = _coerce_positive_count(
                        _ta.get("target_missing_count_exact")
                        or _ta.get("missing_count")
                        or _ta.get("null_count")
                    )
                    if isinstance(_ta_total, int) and isinstance(_ta_missing, int) and _ta_total > _ta_missing:
                        _ta_labeled = _ta_total - _ta_missing
                        if "n_train" not in hints and _ta_labeled > 0:
                            hints["n_train"] = _ta_labeled
                        if "n_test" not in hints and _ta_missing > 0:
                            hints["n_test"] = _ta_missing
                        if "n_total" not in hints and _ta_total > 0:
                            hints["n_total"] = _ta_total

        n_train = hints.get("n_train")
        n_test = hints.get("n_test")
        n_total = hints.get("n_total")
        if n_total is None and n_train is not None and n_test is not None:
            hints["n_total"] = int(n_train + n_test)
        if n_test is None and n_total is not None and n_train is not None and n_total >= n_train:
            hints["n_test"] = int(n_total - n_train)
        if n_train is None and n_total is not None and n_test is not None and n_total >= n_test:
            hints["n_train"] = int(n_total - n_test)
        return hints

    def _normalize_artifact_path(path_like: Any) -> str:
        text = str(path_like or "").strip().replace("\\", "/")
        while text.startswith("./"):
            text = text[2:]
        return text

    def _collect_artifact_kind_map(contract_payload: Dict[str, Any]) -> Dict[str, str]:
        mapping: Dict[str, str] = {}

        def _infer_kind_from_path(normalized_path: str) -> str | None:
            basename = normalized_path.lower().rsplit("/", 1)[-1]
            if not basename:
                return None
            if "submission" in basename:
                return "submission"
            if "scored_rows" in basename or "scored-rows" in basename:
                return "scored_rows"
            if "prediction" in basename or "predictions" in basename:
                return "prediction"
            if "forecast" in basename:
                return "forecast"
            if "ranking" in basename:
                return "ranking"
            return None

        def _ingest(items: Any) -> None:
            if not isinstance(items, list):
                return
            for entry in items:
                path = ""
                kind = ""
                if isinstance(entry, dict):
                    path = _normalize_artifact_path(
                        entry.get("path") or entry.get("output") or entry.get("artifact")
                    )
                    kind = str(entry.get("kind") or "").strip()
                elif isinstance(entry, str):
                    path = _normalize_artifact_path(entry)
                else:
                    continue
                if not kind and path:
                    inferred_kind = _infer_kind_from_path(path)
                    if inferred_kind:
                        kind = inferred_kind
                if path and kind and path.lower() not in mapping:
                    mapping[path.lower()] = kind

        _ingest(contract_payload.get("required_output_artifacts"))
        _ingest(contract_payload.get("required_outputs"))
        spec = contract_payload.get("spec_extraction")
        if isinstance(spec, dict):
            _ingest(spec.get("deliverables"))
        artifact_reqs = contract_payload.get("artifact_requirements")
        if isinstance(artifact_reqs, dict):
            _ingest(artifact_reqs.get("required_files"))
            file_schemas = artifact_reqs.get("file_schemas")
            if isinstance(file_schemas, dict):
                for raw_path in file_schemas.keys():
                    path = _normalize_artifact_path(raw_path)
                    inferred_kind = _infer_kind_from_path(path) if path else None
                    if path and inferred_kind and path.lower() not in mapping:
                        mapping[path.lower()] = inferred_kind
        return mapping

    row_count_hints = _extract_row_count_hints(contract, data_profile)

    # Robust fallback: if n_train/n_test are still missing but the data_profile
    # has outcome_analysis with non_null_count, derive them directly.  This is
    # universal — it works for any supervised learning task where the target
    # column has null values for the test/scoring split.
    if "n_train" not in row_count_hints and "n_total" in row_count_hints:
        _profile_for_fallback = data_profile if isinstance(data_profile, dict) else {}
        _oa = _profile_for_fallback.get("outcome_analysis")
        if isinstance(_oa, dict):
            for _target_col in outcome_cols:
                _target_entry = _oa.get(_target_col)
                if not isinstance(_target_entry, dict):
                    continue
                _nnc = _coerce_positive_count(_target_entry.get("non_null_count"))
                _total = row_count_hints["n_total"]
                if _nnc is not None and 0 < _nnc < _total:
                    row_count_hints["n_train"] = _nnc
                    row_count_hints["n_test"] = _total - _nnc
                    break
                # Also try deriving from null_frac
                _null_frac = _target_entry.get("null_frac")
                if isinstance(_null_frac, (int, float)) and 0.0 < float(_null_frac) < 1.0:
                    _inferred_train = int(round(_total * (1.0 - float(_null_frac))))
                    if 0 < _inferred_train < _total:
                        row_count_hints["n_train"] = _inferred_train
                        row_count_hints["n_test"] = _total - _inferred_train
                        break

    artifact_kind_map = _collect_artifact_kind_map(contract)

    def _resolve_expected_row_count(value: Any) -> Optional[int]:
        parsed = _coerce_positive_count(value)
        if parsed is not None:
            return parsed
        if not isinstance(value, str):
            return None
        token = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
        if not token:
            return None
        alias_map = {
            "n_train": row_count_hints.get("n_train"),
            "n_train_rows": row_count_hints.get("n_train"),
            "train_rows": row_count_hints.get("n_train"),
            "n_test": row_count_hints.get("n_test"),
            "n_test_rows": row_count_hints.get("n_test"),
            "test_rows": row_count_hints.get("n_test"),
            "n_total": row_count_hints.get("n_total"),
            "n_total_rows": row_count_hints.get("n_total"),
            "total_rows": row_count_hints.get("n_total"),
            "n_rows": row_count_hints.get("n_total"),
            "row_count": row_count_hints.get("n_total"),
        }
        candidate = alias_map.get(token)
        return int(candidate) if isinstance(candidate, int) and candidate > 0 else None

    def _infer_expected_row_count(path: str, kind: str) -> Optional[int]:
        lowered_path = path.lower()
        if not lowered_path.endswith(".csv"):
            return None
        normalized_kind = re.sub(r"[^a-z0-9]+", "_", str(kind or "").lower()).strip("_")
        basename = lowered_path.rsplit("/", 1)[-1]
        if not normalized_kind:
            if "submission" in basename:
                normalized_kind = "submission"
            elif "scored_rows" in basename or "scored-rows" in basename:
                normalized_kind = "scored_rows"
            elif "prediction" in basename or "predictions" in basename:
                normalized_kind = "prediction"
            elif "forecast" in basename:
                normalized_kind = "forecast"
            elif "ranking" in basename:
                normalized_kind = "ranking"
        n_total = row_count_hints.get("n_total")
        n_test = row_count_hints.get("n_test")
        if "scored_rows" in lowered_path or normalized_kind in {"scored_rows", "scored"}:
            return n_total
        if normalized_kind == "submission":
            return n_test
        if normalized_kind in {"prediction", "predictions", "forecast", "ranking_scores", "ranking", "recommendations"}:
            if isinstance(n_test, int) and isinstance(n_total, int) and n_test < n_total:
                return n_test
        return None

    raw_file_schemas = full_artifact_requirements.get("file_schemas", {})
    file_schemas: Dict[str, Any] = (
        copy.deepcopy(raw_file_schemas) if isinstance(raw_file_schemas, dict) else {}
    )
    normalized_file_schemas: Dict[str, Any] = {}
    for raw_path, raw_schema in file_schemas.items():
        schema_path = _normalize_artifact_path(raw_path)
        if not schema_path:
            continue
        schema_obj = dict(raw_schema) if isinstance(raw_schema, dict) else {}
        resolved_expected = _resolve_expected_row_count(schema_obj.get("expected_row_count"))
        if resolved_expected is None:
            resolved_expected = _infer_expected_row_count(
                schema_path,
                artifact_kind_map.get(schema_path.lower(), ""),
            )
        if resolved_expected is not None:
            schema_obj["expected_row_count"] = int(resolved_expected)
        normalized_file_schemas[schema_path] = schema_obj

    for entry in required_files:
        if not isinstance(entry, dict):
            continue
        schema_path = _normalize_artifact_path(entry.get("path"))
        if not schema_path or not schema_path.lower().endswith(".csv"):
            continue
        schema_obj = normalized_file_schemas.get(schema_path)
        if not isinstance(schema_obj, dict):
            schema_obj = {}
        if _resolve_expected_row_count(schema_obj.get("expected_row_count")) is None:
            inferred_expected = _infer_expected_row_count(
                schema_path,
                artifact_kind_map.get(schema_path.lower(), ""),
            )
            if inferred_expected is not None:
                schema_obj["expected_row_count"] = int(inferred_expected)
        if schema_obj:
            normalized_file_schemas[schema_path] = schema_obj

    # Pass 3: Create file_schema entries from artifact_kind_map for CSV paths
    # not already present.  This ensures artifacts like submission.csv get
    # expected_row_count even when the LLM-generated file_schemas is empty.
    # Universal: derives kind from naming conventions (submission, scored_rows,
    # prediction, forecast, ranking), not from any specific competition or dataset.
    for kind_path_lower, kind_value in artifact_kind_map.items():
        norm_path = _normalize_artifact_path(kind_path_lower)
        if not norm_path or not norm_path.lower().endswith(".csv"):
            continue
        if norm_path in normalized_file_schemas:
            continue
        inferred = _infer_expected_row_count(norm_path, kind_value)
        if inferred is not None:
            normalized_file_schemas[norm_path] = {"expected_row_count": int(inferred)}

    # SYNC FIX: Filter out constant columns from clean_dataset.required_columns
    # Constant columns provide no information and should be excluded from the final schema
    constant_columns_set: set[str] = set()
    dropped_constant_columns: List[str] = []
    if isinstance(data_profile, dict):
        constant_cols_raw = data_profile.get("constant_columns")
        if isinstance(constant_cols_raw, list):
            constant_columns_set = {str(c) for c in constant_cols_raw if c}
            # Normalize to match canonical_columns
            constant_norms = {_normalize_column_identifier(c): c for c in constant_columns_set}
            for col in canonical_columns:
                col_norm = _normalize_column_identifier(col)
                if col_norm in constant_norms or col in constant_columns_set:
                    dropped_constant_columns.append(col)

    inherited_optional_passthrough: List[str] = []
    inherited_required_columns: List[str] = []
    inherited_column_transformations: Dict[str, Any] = {}
    inherited_clean_output_path = (
        get_clean_dataset_output_path(full_contract_or_partial)
        if isinstance(full_contract_or_partial, dict)
        else ""
    )
    inherited_clean_manifest_path = (
        get_declared_artifact_path(
            full_contract_or_partial,
            "cleaning_manifest.json",
            owner="data_engineer",
            kind="manifest",
        )
        if isinstance(full_contract_or_partial, dict)
        else ""
    )
    if isinstance(full_clean_dataset, dict):
        required_raw = full_clean_dataset.get("required_columns")
        if isinstance(required_raw, list):
            inherited_required_columns = [
                str(col).strip() for col in required_raw if str(col).strip()
            ]
        optional_raw = full_clean_dataset.get("optional_passthrough_columns")
        if isinstance(optional_raw, list):
            inherited_optional_passthrough = [
                str(col).strip() for col in optional_raw if str(col).strip()
            ]
        transforms_raw = full_clean_dataset.get("column_transformations")
        if isinstance(transforms_raw, dict):
            inherited_column_transformations = copy.deepcopy(transforms_raw)
        output_path = str(full_clean_dataset.get("output_path") or "").strip()
        if output_path:
            inherited_clean_output_path = output_path
        manifest_path = str(
            full_clean_dataset.get("output_manifest_path")
            or full_clean_dataset.get("manifest_path")
            or ""
        ).strip()
        if manifest_path:
            inherited_clean_manifest_path = manifest_path

    # Compute clean_dataset.required_columns excluding constant columns.
    # Prefer explicit anchors declared upstream; only fall back to full canonical
    # coverage when the contract did not define anchors/selectors semantics.
    dropped_constant_norms = {
        _normalize_column_identifier(col) for col in dropped_constant_columns if col
    }
    canonical_norms = {_normalize_column_identifier(col): col for col in canonical_columns}
    clean_dataset_required_columns: List[str] = []
    if inherited_required_columns:
        for col in inherited_required_columns:
            resolved = canonical_norms.get(_normalize_column_identifier(col)) or col
            if _normalize_column_identifier(resolved) in dropped_constant_norms:
                continue
            if resolved not in clean_dataset_required_columns:
                clean_dataset_required_columns.append(resolved)
        for col in identifiers + outcome_cols + decision_cols + _collect_cleaning_hard_gate_columns(contract):
            resolved = canonical_norms.get(_normalize_column_identifier(col)) or ""
            if not resolved or _normalize_column_identifier(resolved) in dropped_constant_norms:
                continue
            if resolved not in clean_dataset_required_columns:
                clean_dataset_required_columns.append(resolved)
    else:
        clean_dataset_required_columns = [
            col for col in canonical_columns
            if _normalize_column_identifier(col) not in dropped_constant_norms
        ]

    clean_dataset_cfg_for_dtypes = {
        "required_columns": clean_dataset_required_columns,
        "required_feature_selectors": feature_selectors if isinstance(feature_selectors, list) else [],
        "optional_passthrough_columns": inherited_optional_passthrough,
    }
    column_dtype_targets = _infer_column_dtype_targets(
        canonical_columns=canonical_columns,
        column_roles=column_roles,
        data_profile=data_profile if isinstance(data_profile, dict) else {},
        clean_dataset_cfg=clean_dataset_cfg_for_dtypes,
        column_inventory=inventory,
    )

    artifact_requirements = {
        "clean_dataset": {
            "required_columns": clean_dataset_required_columns,
            "output_path": inherited_clean_output_path,
            "output_manifest_path": inherited_clean_manifest_path,
            "excluded_constant_columns": dropped_constant_columns if dropped_constant_columns else [],
            "required_feature_selectors": feature_selectors if isinstance(feature_selectors, list) else [],
            "column_dtype_targets": column_dtype_targets,
            "optional_passthrough_columns": inherited_optional_passthrough,
        },
        # P1.1: Formal file vs column separation
        "required_files": required_files,
        "scored_rows_schema": scored_rows_schema,
        "file_schemas": normalized_file_schemas,
        "row_count_hints": {
            k: v for k, v in row_count_hints.items()
            if isinstance(v, int) and v > 0
        },
        "schema_binding": {
            "required_columns": clean_dataset_required_columns,
            "optional_passthrough_columns": inherited_optional_passthrough,
        },
    }
    if inherited_column_transformations:
        artifact_requirements["clean_dataset"]["column_transformations"] = inherited_column_transformations

    data_engineer_runbook = "\n".join(
        [
            f"Produce {inherited_clean_output_path} containing ONLY the columns listed in required_columns.",
            "Your output CSV must match EXACTLY the required_columns list - no more, no less.",
            "If a column exists in raw data but is NOT in required_columns, DISCARD it (do not include in output).",
            "Constant columns (single unique value) have been pre-excluded from required_columns.",
            "Preserve column names; do not invent or rename columns.",
            f"Load using output_dialect from {inherited_clean_manifest_path} when available.",
            "Parse numeric/date fields conservatively; document conversions.",
            "If a required column is missing from input, report and stop (no fabrication).",
            "Do not derive targets or train models.",
            "Avoid advanced validation metrics (MAE/correlation); report only dtype and null counts.",
            f"Write {inherited_clean_manifest_path} with input/output dialect details.",
        ]
    )
    ml_engineer_runbook = "\n".join(
        [
            "Use allowed_feature_sets for modeling/segmentation.",
            "Never use forbidden_features in training or optimization.",
            "Produce only the artifacts explicitly declared by the execution contract.",
            f"Respect output_dialect from {inherited_clean_manifest_path}.",
            "Document leakage checks and any data_limited_mode fallback.",
            "If the contract declares an alignment artifact, include feature_usage there (used_features, target_columns, excluded_features).",
        ]
    )

    business_objective = (
        contract.get("business_objective")
        or strategy_dict.get("business_objective")
        or str(business_objective_hint or "").strip()
        or "Objective not specified; produce reliable and actionable artifacts."
    )
    if business_objective and len(business_objective) > 2000:
        business_objective = business_objective[:2000]

    omitted_columns_policy = contract.get("omitted_columns_policy") or (
        "Ignored by default unless promoted by strategy/explicit mention; available via column_inventory."
    )

    def _extract_preprocessing_requirements_min(source: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(source, dict):
            return {}
        prep = source.get("preprocessing_requirements")
        if not isinstance(prep, dict):
            return {}
        out: Dict[str, Any] = {}
        nan_strategies = prep.get("nan_strategies")
        if isinstance(nan_strategies, list):
            trimmed = []
            for item in nan_strategies:
                if not isinstance(item, dict):
                    continue
                entry: Dict[str, Any] = {}
                for key in ("column", "strategy", "missing_category_value", "owner", "group_by"):
                    if key in item and item.get(key) not in (None, ""):
                        entry[key] = item.get(key)
                if entry:
                    trimmed.append(entry)
            if trimmed:
                out["nan_strategies"] = trimmed
        default_val = prep.get("missing_category_default")
        if isinstance(default_val, str) and default_val.strip():
            out["missing_category_default"] = default_val.strip()
        return out

    prep_reqs_min = _extract_preprocessing_requirements_min(contract)

    reporting_policy = contract.get("reporting_policy")
    if not isinstance(reporting_policy, dict) or not reporting_policy:
        execution_plan = contract.get("execution_plan")
        if isinstance(execution_plan, dict):
            reporting_policy = build_reporting_policy(execution_plan, strategy_dict)
    plot_spec = reporting_policy.get("plot_spec") if isinstance(reporting_policy, dict) else None
    if not isinstance(plot_spec, dict) or not plot_spec:
        plot_spec = build_plot_spec(contract)
    if isinstance(plot_spec, dict) and plot_spec:
        if not isinstance(reporting_policy, dict):
            reporting_policy = {}
        reporting_policy = dict(reporting_policy)
        reporting_policy["plot_spec"] = {
            "enabled": bool(plot_spec.get("enabled", True)),
            "max_plots": int(plot_spec.get("max_plots", len(plot_spec.get("plots") or []))),
        }

    qa_gates = _apply_qa_gate_policy(
        contract.get("qa_gates"),
        strategy_dict,
        business_objective or "",
        contract,
    )
    cleaning_gates = _apply_cleaning_gate_policy(contract.get("cleaning_gates"))
    training_rows_rule = contract.get("training_rows_rule")
    scoring_rows_rule = contract.get("scoring_rows_rule")
    secondary_scoring_subset = contract.get("secondary_scoring_subset")
    data_partitioning_notes = contract.get("data_partitioning_notes")
    if not isinstance(data_partitioning_notes, list):
        data_partitioning_notes = []

    def _coerce_ratio(value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            ratio = float(value)
        elif isinstance(value, str):
            token = value.strip()
            if not token:
                return None
            try:
                ratio = float(token)
            except Exception:
                return None
        else:
            return None
        if ratio < 0.0 or ratio > 1.0:
            return None
        return ratio

    def _resolve_split_column() -> str | None:
        candidates: List[str] = []
        for col in steward_split_cols:
            if isinstance(col, str) and col.strip():
                candidates.append(col.strip())
            elif isinstance(col, dict):
                # Handle split_candidates dicts: {"column": "is_train", ...}
                col_name = str(
                    col.get("column") or col.get("name") or col.get("split_column") or ""
                ).strip()
                if col_name:
                    candidates.append(col_name)
        split_fields = (
            contract.get("split_column"),
            contract.get("split_columns"),
        )
        for value in split_fields:
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip())
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item.strip():
                        candidates.append(item.strip())
        eval_spec_local = contract.get("evaluation_spec")
        if isinstance(eval_spec_local, dict):
            value = eval_spec_local.get("split_column")
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip())
            values = eval_spec_local.get("split_columns")
            if isinstance(values, list):
                for item in values:
                    if isinstance(item, str) and item.strip():
                        candidates.append(item.strip())

        for cand in candidates:
            if cand in canonical_columns:
                return cand
            norm = _normalize_column_identifier(cand)
            resolved = inventory_norms.get(norm)
            if resolved:
                return resolved
        return candidates[0] if candidates else None

    def _resolve_target_for_partitioning() -> str | None:
        for source in (
            outcome_cols,
            contract.get("outcome_columns"),
            contract.get("target_column"),
            contract.get("target_columns"),
        ):
            values = _coerce_list(source)
            for raw in values:
                if not raw:
                    continue
                token = str(raw).strip()
                if not token:
                    continue
                if token in canonical_columns:
                    return token
                norm = _normalize_column_identifier(token)
                resolved = inventory_norms.get(norm)
                if resolved:
                    return resolved
        return None

    def _target_has_partial_labels(target_col: str | None) -> bool:
        if not isinstance(target_col, str) or not target_col.strip():
            return False
        target = target_col.strip()
        profile_candidates: List[Dict[str, Any]] = []
        for source in (
            data_profile,
            contract.get("data_profile"),
            contract.get("dataset_profile"),
            contract,
        ):
            if isinstance(source, dict):
                profile_candidates.append(source)
                nested = source.get("dataset_profile")
                if isinstance(nested, dict):
                    profile_candidates.append(nested)
                nested = source.get("data_profile")
                if isinstance(nested, dict):
                    profile_candidates.append(nested)

        for profile in profile_candidates:
            outcome_analysis = profile.get("outcome_analysis")
            if not isinstance(outcome_analysis, dict):
                continue
            entry = outcome_analysis.get(target)
            if not isinstance(entry, dict):
                continue
            total = _coerce_positive_count(
                entry.get("total_count")
                or entry.get("n_total_rows")
                or entry.get("n_rows")
                or entry.get("row_count")
                or entry.get("rows")
            )
            non_null = _coerce_positive_count(
                entry.get("non_null_count")
                or entry.get("n_non_null")
                or entry.get("non_null_rows")
                or entry.get("labeled_rows")
            )
            if non_null is None and isinstance(total, int):
                null_frac = _coerce_ratio(entry.get("null_frac"))
                if null_frac is not None:
                    inferred_non_null = int(round(total * (1.0 - null_frac)))
                    if inferred_non_null > 0 and inferred_non_null <= total:
                        non_null = inferred_non_null
            if isinstance(total, int) and isinstance(non_null, int) and total > non_null:
                return True
            null_frac = _coerce_ratio(entry.get("null_frac"))
            if null_frac is not None and null_frac > 0.0:
                return True
        n_train = row_count_hints.get("n_train")
        n_total = row_count_hints.get("n_total")
        return bool(
            isinstance(n_train, int)
            and isinstance(n_total, int)
            and n_total > n_train
        )

    split_column = _resolve_split_column()
    target_for_partitioning = _resolve_target_for_partitioning()
    partial_labels_detected = _target_has_partial_labels(target_for_partitioning)

    if not isinstance(training_rows_rule, str) or not training_rows_rule.strip():
        if partial_labels_detected and target_for_partitioning:
            training_rows_rule = f"rows where {target_for_partitioning} is not missing"
            data_partitioning_notes.append(
                "Training rows inferred from target missingness (target not null)."
            )
        elif split_column:
            training_rows_rule = f"rows where {split_column} indicates training split"
            data_partitioning_notes.append(
                f"Training rows inferred from split column '{split_column}'."
            )

    if not isinstance(scoring_rows_rule, str) or not scoring_rows_rule.strip():
        if partial_labels_detected and target_for_partitioning:
            scoring_rows_rule = f"rows where {target_for_partitioning} is missing"
            data_partitioning_notes.append(
                "Scoring rows inferred as rows without target labels."
            )
        elif split_column:
            scoring_rows_rule = f"rows where {split_column} indicates scoring/test split"
            data_partitioning_notes.append(
                f"Scoring rows inferred from split column '{split_column}'."
            )

    partitioning_notes_dedup: List[str] = []
    for note in data_partitioning_notes:
        text = str(note or "").strip()
        if text and text not in partitioning_notes_dedup:
            partitioning_notes_dedup.append(text)
    data_partitioning_notes = partitioning_notes_dedup

    n_train_rows = row_count_hints.get("n_train")
    n_test_rows = row_count_hints.get("n_test")
    n_total_rows = row_count_hints.get("n_total")
    has_subset_row_target = bool(
        isinstance(n_test_rows, int)
        and isinstance(n_total_rows, int)
        and n_test_rows > 0
        and n_total_rows > n_test_rows
    )

    def _is_subset_output_kind(kind: str) -> bool:
        normalized_kind = re.sub(r"[^a-z0-9]+", "_", str(kind or "").lower()).strip("_")
        return normalized_kind in {
            "submission",
            "prediction",
            "predictions",
            "forecast",
            "ranking",
            "ranking_scores",
            "recommendations",
        }

    requires_subset_outputs = False
    if has_subset_row_target:
        for path, kind in artifact_kind_map.items():
            if _is_subset_output_kind(kind):
                requires_subset_outputs = True
                break
            if "submission" in path:
                requires_subset_outputs = True
                break
        if not requires_subset_outputs:
            for schema in normalized_file_schemas.values():
                if not isinstance(schema, dict):
                    continue
                expected_rows = _resolve_expected_row_count(schema.get("expected_row_count"))
                if isinstance(expected_rows, int) and isinstance(n_total_rows, int) and expected_rows < n_total_rows:
                    requires_subset_outputs = True
                    break

    train_filter = contract.get("train_filter")
    if not isinstance(train_filter, dict):
        if partial_labels_detected and target_for_partitioning:
            train_filter = {
                "type": "label_not_null",
                "column": target_for_partitioning,
                "value": None,
                "rule": training_rows_rule,
            }
        elif split_column:
            train_filter = {
                "type": "split_equals",
                "column": split_column,
                "value": "train",
                "rule": training_rows_rule,
            }
        else:
            train_filter = {}

    split_status = "unknown"
    if isinstance(train_filter, dict) and train_filter:
        split_status = "resolved"
    elif split_column or training_rows_rule or scoring_rows_rule:
        split_status = "partial"
    if requires_subset_outputs and split_status == "unknown":
        data_partitioning_notes.append(
            "FAIL_CLOSED: subset output artifacts are required but split resolution is unresolved."
        )

    split_spec: Dict[str, Any] = {
        "status": split_status,
        "target_column": target_for_partitioning,
        "split_column": split_column,
        "training_rows_rule": training_rows_rule,
        "scoring_rows_rule": scoring_rows_rule,
        "training_rows_policy": (
            "only_rows_with_label"
            if partial_labels_detected and target_for_partitioning
            else ("use_split_column" if split_column else "use_all_rows")
        ),
        "train_filter": train_filter,
        "requires_test_only_outputs": requires_subset_outputs,
    }
    if isinstance(n_train_rows, int) and n_train_rows > 0:
        split_spec["n_train_rows"] = n_train_rows
    if isinstance(n_test_rows, int) and n_test_rows > 0:
        split_spec["n_test_rows"] = n_test_rows
    if isinstance(n_total_rows, int) and n_total_rows > 0:
        split_spec["n_total_rows"] = n_total_rows

    validation_requirements = contract.get("validation_requirements")
    if not isinstance(validation_requirements, dict) or not validation_requirements:
        # Build from strategy instead of hardcoding holdout/accuracy
        _strat_metric = _normalize_kpi_metric(strategy_dict.get("success_metric"))
        if not _strat_metric:
            _strat_metric = _extract_kpi_from_list(strategy_dict.get("recommended_evaluation_metrics"))
        if not _strat_metric:
            _strat_metric = _extract_kpi_from_text(business_objective_hint or "")
        _strat_metric = _strat_metric or "accuracy"

        _strat_validation = str(strategy_dict.get("validation_strategy") or "").strip().lower()
        _strat_method = "holdout"
        if "k-fold" in _strat_validation or "kfold" in _strat_validation or "cross" in _strat_validation:
            _strat_method = "cross_validation"

        _metrics_to_report = [_strat_metric]
        for m in _coerce_list(strategy_dict.get("recommended_evaluation_metrics")):
            _m_norm = _normalize_kpi_metric(m)
            if _m_norm and _m_norm not in _metrics_to_report:
                _metrics_to_report.append(_m_norm)

        validation_requirements = {
            "method": _strat_method,
            "metrics_to_report": _metrics_to_report,
            "primary_metric": _strat_metric,
        }

    objective_analysis = contract.get("objective_analysis")
    if not isinstance(objective_analysis, dict) or not objective_analysis:
        objective_analysis = {}
    current_problem_type = str(objective_analysis.get("problem_type") or "").strip()
    current_caps = infer_problem_capabilities(problem_type=current_problem_type)
    if not current_problem_type or str(current_caps.get("family") or "unknown") == "unknown":
        objective_analysis = dict(objective_analysis)
        normalized_caps = infer_problem_capabilities(
            objective_text=str(contract.get("business_objective") or ""),
            objective_type=objective_type,
            problem_type=current_problem_type,
            evaluation_spec=contract.get("evaluation_spec") if isinstance(contract.get("evaluation_spec"), dict) else {},
            validation_requirements=validation_requirements if isinstance(validation_requirements, dict) else {},
            required_outputs=contract.get("required_outputs") if isinstance(contract.get("required_outputs"), list) else [],
            strategy=strategy if isinstance(strategy, dict) else {},
        )
        objective_analysis["problem_type"] = str(normalized_caps.get("family") or "unspecified").strip() or "unspecified"

    evaluation_spec = contract.get("evaluation_spec")
    if not isinstance(evaluation_spec, dict) or not evaluation_spec:
        evaluation_spec = {}
    if not str(evaluation_spec.get("objective_type") or "").strip():
        evaluation_spec = dict(evaluation_spec)
        evaluation_spec["objective_type"] = str(objective_type or "unspecified").strip() or "unspecified"

    resolved_target_columns = list(dict.fromkeys([col for col in outcome_cols if str(col).strip()]))
    resolved_primary_target = None
    for source in (
        contract.get("target_column"),
        contract.get("primary_target"),
        strategy_dict.get("target_column"),
        strategy_dict.get("target_columns"),
        contract.get("target_columns"),
        steward_primary_target,
        resolved_target_columns,
    ):
        values = _collect_targetish_columns(source)
        for value in values:
            if value in resolved_target_columns or not resolved_target_columns:
                resolved_primary_target = value
                break
        if resolved_primary_target:
            break
    if resolved_target_columns:
        validation_requirements = dict(validation_requirements)
        objective_analysis = dict(objective_analysis)
        evaluation_spec = dict(evaluation_spec)
        if len(resolved_target_columns) > 1:
            validation_requirements["target_columns"] = list(resolved_target_columns)
            objective_analysis["target_columns"] = list(resolved_target_columns)
            objective_analysis["primary_targets"] = list(resolved_target_columns)
            objective_analysis["label_columns"] = list(resolved_target_columns)
            evaluation_spec["target_columns"] = list(resolved_target_columns)
            evaluation_spec["primary_targets"] = list(resolved_target_columns)
            evaluation_spec["label_columns"] = list(resolved_target_columns)
        if resolved_primary_target:
            validation_requirements.setdefault("target_column", resolved_primary_target)
            objective_analysis["primary_target"] = resolved_primary_target
            objective_analysis["target_column"] = resolved_primary_target
            objective_analysis["label_column"] = resolved_primary_target
            evaluation_spec["primary_target"] = resolved_primary_target
            evaluation_spec["target_column"] = resolved_primary_target
            evaluation_spec["label_column"] = resolved_primary_target

    output_dialect = contract.get("output_dialect")
    if not isinstance(output_dialect, dict):
        output_dialect = {"sep": ",", "decimal": ".", "encoding": "utf-8"}
    else:
        output_dialect = {
            "sep": str(output_dialect.get("sep") or ","),
            "decimal": str(output_dialect.get("decimal") or "."),
            "encoding": str(output_dialect.get("encoding") or "utf-8"),
        }

    iteration_policy = contract.get("iteration_policy")
    if isinstance(iteration_policy, str):
        iteration_policy = {"summary": iteration_policy.strip()} if iteration_policy.strip() else {}
    if not isinstance(iteration_policy, dict) or not iteration_policy:
        iteration_policy = {
            "max_cleaning_retries": 2,
            "max_training_retries": 2,
            "max_total_attempts": 4,
        }
    optimization_policy = normalize_optimization_policy(contract.get("optimization_policy"))

    scope = normalize_contract_scope(contract.get("scope"))
    if scope not in {"cleaning_only", "ml_only", "full_pipeline"}:
        has_cleaning = bool(artifact_requirements.get("clean_dataset"))
        has_ml = bool(validation_requirements)
        if has_cleaning and has_ml:
            scope = "full_pipeline"
        elif has_cleaning:
            scope = "cleaning_only"
        elif has_ml:
            scope = "ml_only"
        else:
            scope = "full_pipeline"

    from src.utils.contract_accessors import CONTRACT_VERSION_V41, normalize_contract_version
    # ------------------------------------------------------------------
    # Gate-aware feature reconciliation
    # ------------------------------------------------------------------
    # When the LLM defines qa_gates that forbid specific columns at
    # inference (e.g. leakage_prevention_auxiliary.forbidden_at_inference),
    # the deterministic fallback for model_features must respect those
    # constraints.  Without this, a column the LLM correctly marked as
    # forbidden can still end up in model_features because the fallback
    # initialises model_features from all pre_decision columns.
    _gate_forbidden_at_inference: set[str] = set()
    for gate in (qa_gates if isinstance(qa_gates, list) else []):
        if not isinstance(gate, dict):
            continue
        params = gate.get("params")
        if not isinstance(params, dict):
            continue
        for key in ("forbidden_at_inference", "forbidden_columns", "excluded_columns"):
            forbidden_list = params.get(key)
            if isinstance(forbidden_list, list):
                _gate_forbidden_at_inference.update(str(c) for c in forbidden_list if c)
    if _gate_forbidden_at_inference:
        _removed_cols = [c for c in model_features if c in _gate_forbidden_at_inference]
        if _removed_cols:
            model_features = [c for c in model_features if c not in _gate_forbidden_at_inference]
            print(
                f"GATE_FEATURE_RECONCILIATION: removed {len(_removed_cols)} column(s) "
                f"from model_features that are forbidden by qa_gates: {sorted(_removed_cols)}"
            )

    contract_min = {
        "contract_version": normalize_contract_version(contract.get("contract_version")),
        "scope": scope,
        "strategy_title": contract.get("strategy_title") or strategy_dict.get("title", "") or "Execution Plan",
        "business_objective": business_objective,
        "output_dialect": output_dialect,
        "canonical_columns": canonical_columns,
        "outcome_columns": outcome_cols,
        "target_column": resolved_primary_target,
        "target_columns": resolved_target_columns,
        "decision_columns": decision_cols,
        "column_roles": column_roles,
        "allowed_feature_sets": {
            "segmentation_features": segmentation_features,
            "model_features": model_features,
            "forbidden_features": forbidden_features,
            "audit_only_features": audit_only_features,
        },
        "artifact_requirements": artifact_requirements,
        "required_outputs": required_outputs,
        "feature_selectors": feature_selectors,  # P1.5: For wide datasets
        "qa_gates": qa_gates,
        "cleaning_gates": cleaning_gates,
        "reviewer_gates": [
            "strategy_followed",
            "metrics_present",
            "interpretability_ok",
        ],
        "data_engineer_runbook": data_engineer_runbook,
        "ml_engineer_runbook": ml_engineer_runbook,
        "omitted_columns_policy": omitted_columns_policy,
        "reporting_policy": reporting_policy or {},
        "decisioning_requirements": decisioning_requirements,
        "column_dtype_targets": column_dtype_targets,
        "validation_requirements": validation_requirements,
        "objective_analysis": objective_analysis,
        "evaluation_spec": evaluation_spec,
        "iteration_policy": iteration_policy,
        "optimization_policy": optimization_policy,
        "split_spec": split_spec,
    }
    if isinstance(n_train_rows, int) and n_train_rows > 0:
        contract_min["n_train_rows"] = n_train_rows
    if isinstance(n_test_rows, int) and n_test_rows > 0:
        contract_min["n_test_rows"] = n_test_rows
    if isinstance(n_total_rows, int) and n_total_rows > 0:
        contract_min["n_total_rows"] = n_total_rows
    if prep_reqs_min:
        contract_min["preprocessing_requirements"] = prep_reqs_min
    if training_rows_rule:
        contract_min["training_rows_rule"] = training_rows_rule
    if scoring_rows_rule:
        contract_min["scoring_rows_rule"] = scoring_rows_rule
    if secondary_scoring_subset:
        contract_min["secondary_scoring_subset"] = secondary_scoring_subset
    if data_partitioning_notes:
        contract_min["data_partitioning_notes"] = data_partitioning_notes
    contract_min = _deterministic_repair_clean_dataset_consistency(contract_min)
    task_semantics = _synthesize_task_semantics(contract_min)
    if task_semantics:
        contract_min["task_semantics"] = task_semantics
    return contract_min


def ensure_v41_schema(contract: Dict[str, Any], strict: bool = False) -> Dict[str, Any]:
    """
    Validates and fills missing V4.1 schema keys.
    Adds to 'unknowns' array when filling defaults.

    Args:
        contract: Contract dict from LLM
        strict: If True, raise error on missing keys (for tests)

    Returns:
        Contract with all V4.1 keys present
    """
    from src.utils.contract_accessors import CONTRACT_VERSION_V41, normalize_contract_version

    if not isinstance(contract, dict):
        return contract

    required_keys = [
        "contract_version", "strategy_title", "business_objective",
        "missing_columns_handling", "execution_constraints",
        "objective_analysis", "data_analysis", "column_roles",
        "preprocessing_requirements",
        "validation_requirements", "leakage_execution_plan",
        "optimization_specification", "segmentation_constraints",
        "data_limited_mode", "allowed_feature_sets",
        "artifact_requirements", "qa_gates", "cleaning_gates", "reviewer_gates",
        "data_engineer_runbook", "ml_engineer_runbook",
        "available_columns", "canonical_columns", "derived_columns",
        "required_outputs", "iteration_policy", "optimization_policy", "column_dtype_targets", "unknowns",
        "assumptions", "notes_for_engineers"
    ]

    repairs = []

    for key in required_keys:
        if key not in contract:
            if strict:
                raise ValueError(f"Missing required V4.1 key: {key}")

            # Fill with safe default
            if key == "contract_version":
                contract[key] = CONTRACT_VERSION_V41
            elif key == "optimization_policy":
                contract[key] = get_default_optimization_policy()
            elif key in ("optimization_specification", "segmentation_constraints"):
                contract[key] = None
            elif key in ("unknowns", "assumptions", "notes_for_engineers", "available_columns",
                         "canonical_columns", "derived_columns", "required_outputs"):
                contract[key] = []
            elif key in ("qa_gates", "cleaning_gates", "reviewer_gates"):
                contract[key] = []
            elif key in ("strategy_title", "business_objective"):
                contract[key] = ""
            else:
                contract[key] = {}

            repairs.append(f"Added missing key: {key}")

    # Ensure unknowns is a list
    unknowns = contract.get("unknowns")
    if not isinstance(unknowns, list):
        unknowns = []
        contract["unknowns"] = unknowns

    # Normalize contract version to V4.1
    version = contract.get("contract_version")
    normalized_version = normalize_contract_version(version)
    if version != normalized_version:
        old_version = version
        contract["contract_version"] = normalized_version
        unknowns.append({
            "item": f"Normalized contract_version from {old_version} to {normalized_version}",
            "impact": "Schema validation enforced V4.1 version",
            "mitigation": "Review LLM output quality",
            "requires_verification": False
        })
    elif version is None:
        contract["contract_version"] = CONTRACT_VERSION_V41

    # Add repair notes to unknowns
    for repair in repairs:
        unknowns.append({
            "item": repair,
            "impact": "Schema validation filled missing field",
            "mitigation": "Review LLM output quality",
            "requires_verification": False
        })

    # Legacy optional field: if present, keep dict shape.
    if "feature_engineering_plan" in contract and not isinstance(contract.get("feature_engineering_plan"), dict):
        contract["feature_engineering_plan"] = {}

    # v4.2-compatible policy defaults.
    contract["optimization_policy"] = normalize_optimization_policy(contract.get("optimization_policy"))

    return contract


def validate_artifact_requirements(contract: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates artifact_requirements to ensure required_columns are subset of canonical/derived columns.
    Moves non-canonical columns to optional_passthrough_columns.

    Args:
        contract: Contract dict with V4.1 schema

    Returns:
        Contract with validated artifact_requirements
    """
    if not isinstance(contract, dict):
        return contract

    # Get canonical and derived columns
    canonical_columns = contract.get("canonical_columns", [])
    derived_columns = _extract_derived_column_names(contract.get("derived_columns"))
    available_columns = contract.get("available_columns", [])

    if not isinstance(canonical_columns, list):
        canonical_columns = []
    if not isinstance(available_columns, list):
        available_columns = []

    # Build allowed column set (canonical + derived)
    allowed_columns_set = set(canonical_columns) | set(derived_columns)

    # Normalize for comparison (case-insensitive)
    allowed_norms = {_normalize_column_identifier(col): col for col in allowed_columns_set}
    available_norms = {_normalize_column_identifier(col): col for col in available_columns}

    # Get artifact_requirements
    artifact_requirements = contract.get("artifact_requirements", {})
    if not isinstance(artifact_requirements, dict):
        return contract

    schema_binding = artifact_requirements.get("schema_binding")
    if not isinstance(schema_binding, dict):
        schema_binding = {}
        artifact_requirements["schema_binding"] = schema_binding

    clean_dataset = artifact_requirements.get("clean_dataset")
    if not isinstance(clean_dataset, dict):
        clean_dataset = None

    if "required_columns" in schema_binding:
        required_columns = schema_binding.get("required_columns")
        if not isinstance(required_columns, list):
            # Preserve invalid type: do not mutate contract
            return contract
    else:
        required_columns = []

    if not required_columns:
        if clean_dataset and isinstance(clean_dataset.get("required_columns"), list):
            schema_binding["required_columns"] = [
                str(col) for col in clean_dataset.get("required_columns") if col
            ]
            required_columns = schema_binding["required_columns"]
        else:
            # V4.1: Use ONLY canonical_columns as fallback, NO legacy required_columns
            if isinstance(canonical_columns, list) and canonical_columns:
                schema_binding["required_columns"] = [str(col) for col in canonical_columns if col]
                required_columns = schema_binding["required_columns"]
            else:
                # No canonical columns available - record as unknown
                schema_binding["required_columns"] = []
                required_columns = []
                unknowns = contract.setdefault("unknowns", [])
                if isinstance(unknowns, list):
                    unknowns.append({
                        "item": "artifact_requirements.schema_binding.required_columns is empty",
                        "impact": "No columns specified for clean dataset validation",
                        "mitigation": "Ensure canonical_columns is populated in contract",
                        "requires_verification": True
                    })

    # Validate required_columns
    valid_required = []
    moved_to_optional = []

    for col in required_columns:
        if not col:
            continue
        col_norm = _normalize_column_identifier(col)

        # Check if column is in canonical or derived columns
        if col_norm in allowed_norms:
            valid_required.append(col)
        # If it's in available_columns but not canonical, move to optional
        elif col_norm in available_norms:
            moved_to_optional.append(col)
        # If it's not even in available_columns, it's invalid - skip it
        else:
            moved_to_optional.append(col)

    # Update schema_binding
    if moved_to_optional:
        schema_binding["required_columns"] = valid_required
        if clean_dataset is not None:
            clean_dataset["required_columns"] = list(valid_required)

        # Add to optional_passthrough_columns
        optional_passthrough = schema_binding.get("optional_passthrough_columns", [])
        if not isinstance(optional_passthrough, list):
            optional_passthrough = []

        # Add moved columns to optional_passthrough
        for col in moved_to_optional:
            if col not in optional_passthrough:
                optional_passthrough.append(col)

        schema_binding["optional_passthrough_columns"] = optional_passthrough

        # Document in unknowns
        unknowns = contract.get("unknowns", [])
        if not isinstance(unknowns, list):
            unknowns = []
            contract["unknowns"] = unknowns

        unknowns.append({
            "item": f"artifact_requirements.required_columns contained non-canonical columns: {moved_to_optional}",
            "impact": "Moved to optional_passthrough_columns to preserve them without enforcement",
            "mitigation": "These columns will be included in outputs if present in data, but are not required",
            "requires_verification": True,
            "auto_corrected": True
        })

    return contract


def _normalize_column_identifier(value: Any) -> str:
    if not value:
        return ""
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "", str(value).lower())
    return cleaned


def _resolve_identifier_candidates(
    contract: Dict[str, Any],
    canonical_columns: List[str] | None,
) -> set[str]:
    candidates: set[str] = set()
    roles = contract.get("column_roles") if isinstance(contract, dict) else None
    if isinstance(roles, dict):
        role_ids = roles.get("identifiers")
        if isinstance(role_ids, list):
            candidates.update([str(col) for col in role_ids if col])
    return candidates


def _build_allowed_column_norms(column_sets: List[str] | None, *more_sets: List[str] | None) -> set[str]:
    norms: set[str] = set()
    for collection in (column_sets, *more_sets):
        if not isinstance(collection, list):
            continue
        for col in collection:
            normed = _normalize_column_identifier(col)
            if normed:
                norms.add(normed)
    return norms


def _filter_leakage_audit_features(
    spec: Dict[str, Any],
    canonical_columns: List[str] | None,
    column_inventory: List[str] | None,
) -> List[str]:
    policy = spec.get("leakage_policy")
    if not isinstance(policy, dict):
        return []
    features = policy.get("audit_features")
    if not isinstance(features, list):
        return []
    allowed_norms = _build_allowed_column_norms(canonical_columns, column_inventory)
    filtered_out: List[str] = []

    if not allowed_norms:
        filtered_out = [str(item) for item in features if item]
        policy["audit_features"] = []
    else:
        kept: List[str] = []
        for item in features:
            if not item:
                continue
            normed = _normalize_column_identifier(item)
            if normed not in allowed_norms:
                filtered_out.append(str(item))
                continue
            kept.append(item)
        policy["audit_features"] = kept

    if filtered_out:
        detail = spec.get("leakage_policy_detail")
        if not isinstance(detail, dict):
            detail = {}
            spec["leakage_policy_detail"] = detail
        detail.setdefault("filtered_audit_features", [])
        existing = detail["filtered_audit_features"]
        existing.extend(filtered_out)
    return filtered_out

def parse_derive_from_expression(expr: str) -> Dict[str, Any]:
    if not expr or not isinstance(expr, str):
        return {}
    text = expr.strip()
    if not text:
        return {}

    def _coerce_values(raw: str) -> List[str]:
        if not raw:
            return []
        cleaned = raw.strip()
        try:
            parsed = ast.literal_eval(cleaned)
            if isinstance(parsed, (list, tuple, set)):
                return [str(item) for item in parsed]
            if isinstance(parsed, str):
                return [parsed]
            return [str(parsed)]
        except Exception:
            pass
        if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", "\""}:
            cleaned = cleaned[1:-1].strip()
        if cleaned.startswith("(") and cleaned.endswith(")"):
            cleaned = cleaned[1:-1].strip()
        if "," in cleaned:
            parts = [part.strip(" \"'") for part in cleaned.split(",") if part.strip()]
            return parts
        return [cleaned] if cleaned else []

    match = re.match(r"^\s*([A-Za-z0-9_][A-Za-z0-9_ %\.\-]*)\s*==\s*(.+?)\s*$", text)
    if match:
        column = match.group(1).strip()
        values = _coerce_values(match.group(2))
        return {"column": column, "positive_values": values}
    match = re.match(r"^\s*([A-Za-z0-9_][A-Za-z0-9_ %\.\-]*)\s+in\s+(.+?)\s*$", text, flags=re.IGNORECASE)
    if match:
        column = match.group(1).strip()
        values = _coerce_values(match.group(2))
        return {"column": column, "positive_values": values}
    token_match = re.search(r"[A-Za-z0-9_][A-Za-z0-9_ %\.\-]*", text)
    if token_match:
        column = token_match.group(0).strip()
        return {"column": column, "positive_values": []}
    return {}

def enforce_percentage_ranges(contract: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(contract, dict):
        return {}
    reqs = contract.get("data_requirements", []) or []
    for req in reqs:
        if not isinstance(req, dict):
            continue
        role = (req.get("role") or "").lower()
        expected = req.get("expected_range")
        if role == "percentage" and not expected:
            req["expected_range"] = [0, 1]
    notes = contract.get("notes_for_engineers")
    if not isinstance(notes, list):
        notes = []
    note = "Percentages must be normalized to 0-1; if values look like 0-100 scale, divide by 100."
    if note not in notes:
        notes.append(note)
    contract["notes_for_engineers"] = notes
    contract["data_requirements"] = reqs
    return contract


def build_dataset_profile(data_summary: str, column_inventory: List[str] | None = None) -> Dict[str, Any]:
    profile: Dict[str, Any] = {"column_count": len(column_inventory or [])}
    summary = (data_summary or "").strip()
    if summary:
        profile["summary_excerpt"] = summary[:400]
    return profile


def build_execution_plan(objective_type: str, dataset_profile: Dict[str, Any]) -> Dict[str, Any]:
    capabilities = infer_problem_capabilities(
        objective_text=str(objective_type or ""),
        objective_type=objective_type,
    )
    objective = str(capabilities.get("family") or "unknown").lower()
    gates = [
        {"id": "data_ok", "description": "Data availability and basic quality checks pass.", "required": True},
        {"id": "target_ok", "description": "Target is valid with sufficient variation.", "required": True},
        {"id": "leakage_ok", "description": "No post-outcome leakage in features.", "required": True},
        {"id": "runtime_ok", "description": "Pipeline executes without runtime failures.", "required": True},
        {"id": "eval_ok", "description": "Evaluation meets objective-specific thresholds.", "required": True},
    ]

    base_outputs = [
        {"artifact_type": "clean_dataset", "required": True, "description": "Cleaned dataset for downstream use."},
        {"artifact_type": "artifact_index", "required": True, "description": "Typed inventory of produced artifacts."},
        {"artifact_type": "insights", "required": True, "description": "Unified insights for downstream reporting."},
        {"artifact_type": "executive_summary", "required": True, "description": "Business-facing summary."},
    ]

    objective_outputs: Dict[str, List[Dict[str, Any]]] = {
        "classification": [
            {"artifact_type": "metrics", "required": True, "description": "Classification metrics."},
            {"artifact_type": "predictions", "required": True, "description": "Predicted labels/probabilities."},
            {"artifact_type": "confusion_matrix", "required": False, "description": "Error breakdown by class."},
        ],
        "regression": [
            {"artifact_type": "metrics", "required": True, "description": "Regression metrics."},
            {"artifact_type": "predictions", "required": True, "description": "Predicted numeric outputs."},
            {"artifact_type": "residuals", "required": False, "description": "Residual diagnostics."},
        ],
        "forecasting": [
            {"artifact_type": "metrics", "required": True, "description": "Forecasting metrics."},
            {"artifact_type": "forecast", "required": True, "description": "Forecast outputs."},
            {"artifact_type": "backtest", "required": False, "description": "Historical forecast evaluation."},
        ],
        "survival_analysis": [
            {"artifact_type": "metrics", "required": True, "description": "Survival metrics."},
            {"artifact_type": "predictions", "required": True, "description": "Risk scores or survival probabilities."},
            {"artifact_type": "calibration", "required": False, "description": "Survival calibration diagnostics."},
        ],
        "ranking": [
            {"artifact_type": "metrics", "required": True, "description": "Ranking metrics."},
            {"artifact_type": "ranking_scores", "required": True, "description": "Ranked scores output."},
            {"artifact_type": "ranking_report", "required": False, "description": "Ranking diagnostics."},
        ],
        "clustering": [
            {"artifact_type": "metrics", "required": True, "description": "Clustering diagnostics."},
            {"artifact_type": "segments", "required": True, "description": "Cluster or segment assignments."},
            {"artifact_type": "cluster_report", "required": False, "description": "Cluster interpretation artifact."},
        ],
        "optimization": [
            {"artifact_type": "metrics", "required": True, "description": "Optimization objective metrics."},
            {"artifact_type": "recommendations", "required": True, "description": "Decision recommendations or actions."},
            {"artifact_type": "policy_report", "required": False, "description": "Optimization policy diagnostics."},
        ],
        "descriptive": [
            {"artifact_type": "report", "required": True, "description": "Descriptive report artifact."},
        ],
    }
    optional_common = [
        {"artifact_type": "feature_importances", "required": False, "description": "Explainability artifact."},
        {"artifact_type": "error_analysis", "required": False, "description": "Failure mode analysis."},
        {"artifact_type": "plots", "required": False, "description": "Diagnostic plots."},
    ]

    outputs = list(base_outputs)
    outputs.extend(objective_outputs.get(objective, [{"artifact_type": "metrics", "required": True, "description": "Evaluation metrics."}]))
    outputs.extend(optional_common)

    return {
        "schema_version": "1",
        "objective_type": objective,
        "problem_capabilities": capabilities,
        "dataset_profile": dataset_profile or {},
        "gates": gates,
        "outputs": outputs,
    }


def build_reporting_policy(
    execution_plan: Dict[str, Any] | None,
    strategy: Dict[str, Any] | None = None,
    run_context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    outputs = execution_plan.get("outputs", []) if isinstance(execution_plan, dict) else []
    required_outputs = get_required_outputs(run_context if isinstance(run_context, dict) else {})
    output_types = {
        str(item.get("artifact_type"))
        for item in outputs
        if isinstance(item, dict) and item.get("artifact_type")
    }

    slots: List[Dict[str, Any]] = []

    def _add_slot(slot_id: str, mode: str, insights_key: str, sources: List[str] | None = None) -> None:
        if not slot_id or any(slot.get("id") == slot_id for slot in slots):
            return
        slot = {
            "id": slot_id,
            "mode": mode,
            "insights_key": insights_key,
            "sources": sources or [],
        }
        slots.append(slot)

    def _find_declared_output(*basenames: str) -> str:
        normalized_names = {str(name).strip().lower() for name in basenames if str(name).strip()}
        for path in required_outputs:
            candidate = normalize_artifact_path(path)
            if not candidate:
                continue
            if os.path.basename(candidate).lower() in normalized_names:
                return candidate
        return ""

    metrics_path = _find_declared_output("metrics.json")
    predictions_path = _find_declared_output("scored_rows.csv", "predictions.csv")
    alignment_path = _find_declared_output("alignment_check.json", "case_alignment_report.json")

    if "metrics" in output_types:
        _add_slot("model_metrics", "required", "metrics_summary", [metrics_path] if metrics_path else [])
    if "predictions" in output_types:
        _add_slot("predictions_overview", "conditional", "predictions_summary", [predictions_path] if predictions_path else [])
    if "feature_importances" in output_types:
        _add_slot("explainability", "optional", "feature_importances_summary", [])
    if "error_analysis" in output_types:
        _add_slot("error_analysis", "optional", "error_summary", [])
    if "forecast" in output_types:
        _add_slot("forecast_summary", "required", "forecast_summary", [])
    if "ranking_scores" in output_types:
        _add_slot("ranking_top", "required", "ranking_summary", [])

    _add_slot("alignment_risks", "conditional", "leakage_audit", [alignment_path] if alignment_path else [])
    _add_slot("segment_pricing", "conditional", "segment_pricing_summary", [predictions_path] if predictions_path else [])

    policy = {
        "audience": "executive",
        "language": "auto",
        "sections": [
            "decision",
            "objective_approach",
            "evidence_metrics",
            "business_impact",
            "risks_limitations",
            "next_actions",
            "visual_insights",
        ],
        "slots": slots,
        "constraints": {"no_markdown_tables": True},
    }

    policy.setdefault("demonstrative_examples_enabled", True)
    policy.setdefault("demonstrative_examples_when_outcome_in", ["NO_GO", "GO_WITH_LIMITATIONS"])
    policy.setdefault("max_examples", 5)
    policy.setdefault("require_strong_disclaimer", True)

    return policy


def build_plot_spec(contract_full: Dict[str, Any] | None) -> Dict[str, Any]:
    contract = contract_full if isinstance(contract_full, dict) else {}
    required_outputs = get_required_outputs(contract)
    outputs_lower = [str(path).lower() for path in required_outputs if path]
    cleaned_data_path = get_clean_dataset_output_path(contract)
    scored_rows_path = (
        get_declared_artifact_path(contract, "scored_rows.csv")
        or get_declared_artifact_path(contract, kind="predictions")
    )
    metrics_path = (
        get_declared_artifact_path(contract, "metrics.json", kind="metrics")
        or get_declared_artifact_path(contract, kind="metrics")
    )
    weights_path = (
        get_declared_artifact_path(contract, "weights.json", kind="weights")
        or get_declared_artifact_path(contract, kind="weights")
    )
    alignment_path = (
        get_declared_artifact_path(contract, "alignment_check.json")
        or get_declared_artifact_path(contract, kind="alignment")
    )

    def _has_output(token: str) -> bool:
        return any(token in path for path in outputs_lower)

    def _dedupe(values: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for item in values:
            if not item:
                continue
            key = str(item)
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
        return out

    def _sample(values: List[str], limit: int) -> List[str]:
        return _dedupe([str(v) for v in values if v])[:limit]

    def _sources(*values: str) -> List[str]:
        return _dedupe([str(v) for v in values if v])

    def _safe_column_name(name: str) -> str:
        if not name:
            return ""
        return re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_").lower()

    def _has_token(name: str, tokens: List[str]) -> bool:
        if not name:
            return False
        normalized = re.sub(r"[^0-9a-zA-Z]+", " ", str(name).lower())
        return any(tok in normalized.split() for tok in tokens)

    def _infer_objective_type() -> str:
        eval_spec = contract.get("evaluation_spec") if isinstance(contract, dict) else None
        if isinstance(eval_spec, dict) and eval_spec.get("objective_type"):
            return str(eval_spec.get("objective_type"))
        plan = contract.get("execution_plan") if isinstance(contract, dict) else None
        if isinstance(plan, dict) and plan.get("objective_type"):
            return str(plan.get("objective_type"))
        obj_analysis = contract.get("objective_analysis") if isinstance(contract, dict) else None
        if isinstance(obj_analysis, dict) and obj_analysis.get("problem_type"):
            return str(obj_analysis.get("problem_type"))
        return "unknown"

    objective_type = _infer_objective_type().lower()
    capabilities = resolve_problem_capabilities_from_contract(
        contract,
        objective_text=str(contract.get("business_objective") or ""),
    )

    is_ranking = is_problem_family(capabilities, "ranking")
    is_forecast = is_problem_family(capabilities, "forecasting")
    is_segmentation = is_problem_family(capabilities, "clustering")
    is_classification = is_problem_family(capabilities, "classification")
    is_regression = is_problem_family(capabilities, "regression")
    is_survival = is_problem_family(capabilities, "survival_analysis")

    canonical_columns = get_canonical_columns(contract)
    canonical_set = set(canonical_columns)

    def _filter_to_canonical(values: List[str]) -> List[str]:
        if not canonical_set:
            return [str(v) for v in values if v]
        return [str(v) for v in values if v in canonical_set]

    roles = get_column_roles(contract)
    pre_decision = _filter_to_canonical(_coerce_list(roles.get("pre_decision")))
    decision_cols = _filter_to_canonical(_coerce_list(roles.get("decision")))
    outcome_cols = _filter_to_canonical(_coerce_list(roles.get("outcome")))
    audit_only = _filter_to_canonical(
        _coerce_list(roles.get("post_decision_audit_only") or roles.get("audit_only"))
    )

    allowed_sets = contract.get("allowed_feature_sets")
    if not isinstance(allowed_sets, dict):
        allowed_sets = {}
    model_features = _filter_to_canonical(_coerce_list(allowed_sets.get("model_features")))
    segmentation_features = _filter_to_canonical(_coerce_list(allowed_sets.get("segmentation_features")))

    data_analysis = contract.get("data_analysis") if isinstance(contract, dict) else None
    type_dist = data_analysis.get("type_distribution") if isinstance(data_analysis, dict) else None
    numeric_cols = _filter_to_canonical(_coerce_list(type_dist.get("numeric"))) if isinstance(type_dist, dict) else []
    datetime_cols = _filter_to_canonical(_coerce_list(type_dist.get("datetime"))) if isinstance(type_dist, dict) else []
    categorical_cols = _filter_to_canonical(_coerce_list(type_dist.get("categorical"))) if isinstance(type_dist, dict) else []

    if not datetime_cols:
        time_tokens = ["date", "time", "timestamp", "period"]
        datetime_cols = [col for col in canonical_columns if _has_token(col, time_tokens)]

    derived_columns = get_derived_column_names(contract)
    score_tokens = ["score", "pred", "prob", "prediction"]
    pred_name_candidates = [col for col in derived_columns if _has_token(col, score_tokens)]

    pred_candidates: List[str] = ["prediction"]
    for outcome in outcome_cols[:3]:
        safe = _safe_column_name(outcome)
        if safe:
            pred_candidates.extend([f"pred_{safe}", f"predicted_{safe}", f"pred_prob_{safe}"])
    pred_candidates.extend(pred_name_candidates)
    pred_candidates = _sample(pred_candidates, 12)

    segment_tokens = ["segment", "segmentation", "cluster", "cohort", "group", "segmento", "cluster_id"]
    segment_candidates = [col for col in derived_columns if _has_token(col, segment_tokens)]
    if not segment_candidates:
        segment_candidates = [col for col in canonical_columns if _has_token(col, segment_tokens)]
    segment_candidates = _sample(segment_candidates, 8)

    has_scored_rows = _has_output("scored_rows.csv")
    has_metrics = _has_output("metrics.json")
    has_alignment = _has_output("alignment_check.json") or _has_output("case_alignment")
    has_weights = _has_output("weights.json")

    plots: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()

    def _add_plot(
        plot_id: str,
        title: str,
        goal: str,
        plot_type: str,
        preferred_sources: List[str],
        required_any: List[str] | None = None,
        required_all: List[str] | None = None,
        optional_cols: List[str] | None = None,
        compute: Dict[str, Any] | None = None,
        caption_template: str | None = None,
    ) -> None:
        if not plot_id or plot_id in seen_ids:
            return
        seen_ids.add(plot_id)
        plot = {
            "plot_id": plot_id,
            "title": title,
            "goal": goal,
            "type": plot_type,
            "inputs": {
                "preferred_sources": preferred_sources,
                "required_columns_any_of": [required_any] if required_any else [],
                "required_columns_all_of": required_all or [],
                "optional_columns": optional_cols or [],
            },
            "compute": compute or {},
            "caption_template": caption_template or "",
        }
        plots.append(plot)

    if len(canonical_columns) >= 4 and cleaned_data_path:
        _add_plot(
            "missingness_overview",
            "Missingness by column",
            "Quantify missing data to focus cleaning and feature engineering.",
            "bar",
            _sources(cleaned_data_path),
            optional_cols=_sample(canonical_columns, 24),
            compute={"metric": "missing_fraction", "top_k": 20},
            caption_template="Top missingness rates across columns (top {top_k}).",
        )

    if numeric_cols and cleaned_data_path:
        _add_plot(
            "numeric_distributions",
            "Numeric feature distributions",
            "Show the distribution of key numeric features.",
            "histogram",
            _sources(cleaned_data_path),
            required_any=_sample(numeric_cols, 12),
            optional_cols=_sample(numeric_cols, 12),
            compute={"x": "AUTO_NUMERIC", "max_columns": min(6, len(numeric_cols))},
            caption_template="Distributions for selected numeric features.",
        )

    if datetime_cols and (is_forecast or has_scored_rows) and _sources(cleaned_data_path, scored_rows_path):
        _add_plot(
            "trend_over_time",
            "Trend over time",
            "Highlight temporal trends for the primary target or prediction.",
            "timeseries",
            _sources(cleaned_data_path, scored_rows_path),
            required_any=_sample(datetime_cols, 6),
            optional_cols=_sample(datetime_cols, 6),
            compute={"x": "AUTO_TIME", "y": "AUTO_TARGET_OR_NUMERIC"},
            caption_template="Trend over time using available temporal columns.",
        )

    if has_scored_rows and pred_candidates and scored_rows_path:
        _add_plot(
            "score_distribution",
            "Prediction/score distribution",
            "Summarize the distribution of model outputs.",
            "histogram",
            _sources(scored_rows_path, cleaned_data_path),
            required_any=pred_candidates,
            compute={"x": "PREDICTION", "bins": 30},
            caption_template="Distribution of predicted scores.",
        )

    if has_scored_rows and outcome_cols and (is_classification or is_ranking) and scored_rows_path:
        _add_plot(
            "topk_lift",
            "Top-k outcome lift",
            "Show outcome rate across score buckets.",
            "bar",
            _sources(scored_rows_path),
            required_any=pred_candidates,
            required_all=[outcome_cols[0]],
            compute={"x": "PREDICTION", "y": outcome_cols[0], "group_by": "decile", "metric": "mean"},
            caption_template="Outcome rate by score decile.",
        )

    if has_scored_rows and outcome_cols and is_regression and scored_rows_path:
        _add_plot(
            "residuals_scatter",
            "Prediction vs actual",
            "Assess residuals and bias in regression outputs.",
            "scatter",
            _sources(scored_rows_path),
            required_any=pred_candidates,
            required_all=[outcome_cols[0]],
            compute={"x": "PREDICTION", "y": outcome_cols[0]},
            caption_template="Predicted vs actual values.",
        )

    if has_scored_rows and outcome_cols and pred_candidates and is_survival and scored_rows_path:
        _add_plot(
            "survival_risk_vs_time",
            "Risk vs event time",
            "Contrast predicted risk against observed event time on labeled rows.",
            "scatter",
            _sources(scored_rows_path),
            required_any=pred_candidates,
            required_all=[outcome_cols[0]],
            compute={"x": "PREDICTION", "y": outcome_cols[0], "color": "AUTO_EVENT"},
            caption_template="Predicted survival risk versus observed event time.",
        )

    if has_weights or has_metrics:
        _add_plot(
            "feature_weights",
            "Feature weights/importance",
            "Highlight the strongest feature contributions or weights.",
            "bar",
            _sources(weights_path, metrics_path),
            compute={"metric": "weights", "top_k": 20},
            caption_template="Top contributing features (if weights available).",
        )

    if has_alignment and alignment_path:
        _add_plot(
            "alignment_check_summary",
            "Alignment check summary",
            "Visualize alignment requirements or validation outcomes.",
            "bar",
            _sources(alignment_path),
            compute={"metric": "alignment_requirements", "top_k": 12},
            caption_template="Alignment check results by requirement.",
        )

    if (is_segmentation or segmentation_features or segment_candidates) and _sources(scored_rows_path, cleaned_data_path):
        _add_plot(
            "segment_sizes",
            "Segment distribution",
            "Show sizes or performance by segment where available.",
            "bar",
            _sources(scored_rows_path, cleaned_data_path),
            required_any=segment_candidates,
            compute={"x": "SEGMENT_COLUMN", "metric": "count", "top_k": 20},
            caption_template="Segment sizes based on available segment identifiers.",
        )

    trimmed_plots = plots
    max_plots = len(trimmed_plots)

    return {
        "enabled": bool(trimmed_plots),
        "max_plots": max_plots,
        "plots": trimmed_plots,
    }



# _contains_visual_token removed (seniority refactoring): visual detection is now LLM-driven.


def _map_plot_type(plot_type: str | None) -> str:
    if not plot_type:
        return "other"
    normalized = str(plot_type).lower()
    mapping = {
        "histogram": "distribution",
        "bar": "comparison",
        "line": "timeseries",
        "timeseries": "timeseries",
        "scatter": "comparison",
        "heatmap": "comparison",
        "box": "distribution",
        "pie": "comparison",
        "area": "timeseries",
    }
    return mapping.get(normalized, "other")


def _extract_columns_from_inputs(plot: Dict[str, Any]) -> List[str]:
    inputs = plot.get("inputs") if isinstance(plot.get("inputs"), dict) else {}
    columns: List[str] = []
    for key in ("required_columns_any_of", "required_columns_all_of", "optional_columns"):
        value = inputs.get(key)
        if isinstance(value, list):
            for entry in value:
                if isinstance(entry, list):
                    columns.extend([str(item) for item in entry if item])
                elif entry:
                    columns.append(str(entry))
    return columns


def _build_visual_requirements(
    contract: Dict[str, Any],
    strategy: Dict[str, Any],
    business_objective: str,
) -> Dict[str, Any]:
    vision_text = _normalize_text(
        strategy.get("analysis_type"),
        strategy.get("techniques"),
        strategy.get("notes"),
        strategy.get("description"),
        strategy.get("objective_type"),
        business_objective,
        contract.get("business_objective"),
        contract.get("strategy_title"),
    )
    # LLM-driven: visual requirements are now set by the LLM in the contract.
    # Fallback heuristic: enable visuals if strategy/objective mention visualization concepts.
    enabled = bool(vision_text and any(tok in vision_text.split() for tok in ("visual", "plot", "chart", "graph", "diagram", "figure")))
    # Check if contract already has explicit visual config from LLM
    existing_visual = contract.get("artifact_requirements", {}).get("visual_requirements") if isinstance(contract.get("artifact_requirements"), dict) else None
    if isinstance(existing_visual, dict) and existing_visual.get("enabled"):
        enabled = True
    required = isinstance(existing_visual, dict) and bool(existing_visual.get("required"))

    dataset_profile = (
        contract.get("dataset_profile") if isinstance(contract.get("dataset_profile"), dict) else {}
    )
    row_count = 0
    if dataset_profile:
        for key in ("row_count", "rows", "estimated_rows"):
            val = dataset_profile.get(key)
            if isinstance(val, (int, float)) and val > 0:
                row_count = int(val)
                break
    sampling_strategy = "random" if row_count > 50000 else "none"
    max_rows_for_plot = 5000
    if row_count and row_count < max_rows_for_plot:
        max_rows_for_plot = max(row_count, 1000)

    outputs_dir = "static/plots"
    artifact_reqs = contract.get("artifact_requirements")
    if isinstance(artifact_reqs, dict):
        outputs_dir = artifact_reqs.get("visual_outputs_dir") or artifact_reqs.get("outputs_dir") or outputs_dir

    plot_spec = build_plot_spec(contract) if enabled else {"enabled": False, "plots": [], "max_plots": 0}
    plots = plot_spec.get("plots") if isinstance(plot_spec.get("plots"), list) else []
    column_roles = get_column_roles(contract)
    outcome_cols = [str(c) for c in (column_roles.get("outcome") or []) if c]
    items: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    for idx, plot in enumerate(plots):
        if not isinstance(plot, dict):
            continue
        plot_id = str(plot.get("plot_id") or plot.get("id") or f"visual_{idx}")
        safe_id = re.sub(r"[^0-9a-zA-Z]+", "_", plot_id).strip("_").lower() or f"visual_{idx}"
        if safe_id in seen_ids:
            safe_id = f"{safe_id}_{idx}"
        seen_ids.add(safe_id)
        goal = str(plot.get("goal") or plot.get("title") or "Visual insight")
        inputs = plot.get("inputs") if isinstance(plot.get("inputs"), dict) else {}
        preferred_sources = [str(src) for src in (inputs.get("preferred_sources") or []) if src]
        columns_from_plot = _extract_columns_from_inputs(plot)
        requires_target = any(col in outcome_cols for col in columns_from_plot)
        requires_predictions = any("scored_rows.csv" in src for src in preferred_sources)
        requires_segments = "segment" in safe_id or "segment" in goal.lower()
        items.append(
            {
                "id": safe_id,
                "purpose": goal,
                "type": _map_plot_type(plot.get("type") or plot.get("plot_type")),
                "inputs": {
                    "requires_target": requires_target,
                    "requires_predictions": requires_predictions,
                    "requires_segments": requires_segments,
                    "columns_hint": columns_from_plot[:6],
                },
                "constraints": {
                    "max_rows_for_plot": max_rows_for_plot,
                    "sampling_strategy": sampling_strategy,
                },
                "expected_filename": f"{safe_id}.png",
            }
        )
    notes = (
        "Visual requirements are contract-driven. If items are listed, produce each exactly and store status in data/visuals_status.json."
        if items
        else "Visual requirements are disabled for this strategy."
    )
    return {
        "enabled": enabled,
        "required": required,
        "outputs_dir": outputs_dir,
        "items": items,
        "notes": notes,
        "plot_spec": plot_spec,
    }


def _ensure_contract_visual_policy(
    contract: Dict[str, Any],
    strategy: Dict[str, Any] | None,
    business_objective: str | None,
) -> Dict[str, Any]:
    """
    Ensure ML-capable contracts expose visual requirements and plot spec through
    canonical contract fields consumed by downstream views.

    Rules:
    - Preserve explicit LLM-provided visual config when present.
    - Fill missing pieces from contract/strategy context (no dataset hardcodes).
    - Keep reporting_policy.plot_spec aligned with artifact_requirements.visual_requirements.plot_spec.
    """
    if not isinstance(contract, dict):
        return contract
    scope = normalize_contract_scope(contract.get("scope"))
    if scope not in {"ml_only", "full_pipeline"}:
        return contract

    strategy_payload = strategy if isinstance(strategy, dict) else {}
    objective_text = str(
        business_objective
        or contract.get("business_objective")
        or ""
    )

    generated_visuals = _build_visual_requirements(contract, strategy_payload, objective_text)
    auto_plot_spec = build_plot_spec(contract)
    artifact_reqs = contract.get("artifact_requirements")
    if not isinstance(artifact_reqs, dict):
        artifact_reqs = {}
    existing_visuals = artifact_reqs.get("visual_requirements")
    has_explicit_visuals = isinstance(existing_visuals, dict) and bool(existing_visuals)
    explicit_enabled = (
        isinstance(existing_visuals, dict)
        and isinstance(existing_visuals.get("enabled"), bool)
    )
    explicit_required = (
        isinstance(existing_visuals, dict)
        and isinstance(existing_visuals.get("required"), bool)
    )

    merged_visuals: Dict[str, Any] = dict(generated_visuals)
    if isinstance(existing_visuals, dict):
        for key in ("enabled", "required", "outputs_dir", "notes"):
            if key in existing_visuals:
                merged_visuals[key] = existing_visuals.get(key)
        existing_items = existing_visuals.get("items")
        if isinstance(existing_items, list) and existing_items:
            merged_visuals["items"] = existing_items
        existing_plot_spec = existing_visuals.get("plot_spec")
        if isinstance(existing_plot_spec, dict) and existing_plot_spec:
            merged_visuals["plot_spec"] = existing_plot_spec

    plot_spec = merged_visuals.get("plot_spec")
    if has_explicit_visuals:
        if not isinstance(plot_spec, dict):
            plot_spec = {}
    else:
        generated_plots = (
            plot_spec.get("plots") if isinstance(plot_spec, dict) and isinstance(plot_spec.get("plots"), list) else []
        )
        if not generated_plots:
            plot_spec = auto_plot_spec
        elif not isinstance(plot_spec, dict) or not plot_spec:
            plot_spec = auto_plot_spec
    plots = plot_spec.get("plots") if isinstance(plot_spec.get("plots"), list) else []
    normalized_plot_spec = dict(plot_spec)
    normalized_plot_spec["enabled"] = bool(normalized_plot_spec.get("enabled", bool(plots)))
    normalized_plot_spec["max_plots"] = int(normalized_plot_spec.get("max_plots", len(plots)))
    merged_visuals["plot_spec"] = normalized_plot_spec

    if not explicit_enabled:
        merged_visuals["enabled"] = bool(plots)
    if not explicit_required:
        merged_visuals["required"] = bool(generated_visuals.get("required", False))
    outputs_dir = merged_visuals.get("outputs_dir")
    if not isinstance(outputs_dir, str) or not outputs_dir.strip():
        merged_visuals["outputs_dir"] = "static/plots"
    if not isinstance(merged_visuals.get("notes"), str):
        merged_visuals["notes"] = str(generated_visuals.get("notes") or "")

    items = merged_visuals.get("items")
    if not isinstance(items, list) or not items:
        merged_visuals["items"] = generated_visuals.get("items", []) if isinstance(generated_visuals.get("items"), list) else []
        items = merged_visuals.get("items")

    normalized_items: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    for idx, item in enumerate(items if isinstance(items, list) else []):
        if not isinstance(item, dict):
            continue
        plot_id = str(item.get("id") or item.get("plot_id") or f"visual_{idx}")
        safe_id = re.sub(r"[^0-9a-zA-Z]+", "_", plot_id).strip("_").lower() or f"visual_{idx}"
        if safe_id in seen_ids:
            safe_id = f"{safe_id}_{idx}"
        seen_ids.add(safe_id)
        normalized = dict(item)
        normalized["id"] = safe_id
        expected_filename = normalized.get("expected_filename")
        if not isinstance(expected_filename, str) or not expected_filename.strip():
            normalized["expected_filename"] = f"{safe_id}.png"
        normalized_items.append(normalized)
    merged_visuals["items"] = normalized_items
    if not isinstance(existing_visuals, dict) or "notes" not in existing_visuals:
        if bool(merged_visuals.get("enabled")) and normalized_items:
            merged_visuals["notes"] = (
                "Visual requirements are contract-driven. If items are listed, produce each exactly and "
                "store status in data/visuals_status.json."
            )
        else:
            merged_visuals["notes"] = "Visual requirements are disabled for this strategy."

    artifact_reqs["visual_requirements"] = merged_visuals
    contract["artifact_requirements"] = artifact_reqs

    required_outputs = contract.get("required_outputs")
    if not isinstance(required_outputs, list):
        required_outputs = []
    if bool(merged_visuals.get("required")) and normalized_items:
        outputs_dir = str(merged_visuals.get("outputs_dir") or "static/plots")
        for item in normalized_items:
            expected = item.get("expected_filename")
            if not expected:
                continue
            if os.path.isabs(str(expected)):
                plot_path = str(expected)
            else:
                plot_path = os.path.normpath(os.path.join(outputs_dir, str(expected)))
            if is_probably_path(plot_path) and plot_path not in required_outputs:
                required_outputs.append(plot_path)
    contract["required_outputs"] = required_outputs

    policy = contract.get("reporting_policy")
    if not isinstance(policy, dict):
        execution_plan = contract.get("execution_plan")
        policy = build_reporting_policy(
            execution_plan if isinstance(execution_plan, dict) else {},
            strategy_payload,
            contract,
        )
    policy = dict(policy)
    policy["plot_spec"] = merged_visuals.get("plot_spec")
    contract["reporting_policy"] = policy
    return contract


class ExecutionPlannerAgent:
    """
    LLM-driven planner that emits an execution contract (JSON) to guide downstream agents.
    Falls back to heuristic contract if the model call fails.
    """

    def __init__(self, api_key: Any = _API_KEY_SENTINEL):
        # Explicit `api_key=None` means "disable external LLM client" (used by tests/fallback paths).
        # Omitting the argument keeps env-based resolution for runtime behavior.
        if api_key is _API_KEY_SENTINEL:
            resolved_api_key = (
                os.getenv("EXECUTION_PLANNER_API_KEY")
                or os.getenv("OPENROUTER_API_KEY")
            )
        else:
            resolved_api_key = api_key
        self.api_key = str(resolved_api_key).strip() if resolved_api_key not in (None, "") else None
        self.provider = "openrouter"
        max_output_tokens = 16384
        try:
            max_output_tokens = int(os.getenv("EXECUTION_PLANNER_MAX_OUTPUT_TOKENS", "16384"))
        except Exception:
            max_output_tokens = 16384
        self._default_max_output_tokens = max(4000, max_output_tokens)
        context_limit_tokens = 65536
        try:
            context_limit_tokens = int(os.getenv("EXECUTION_PLANNER_CONTEXT_WINDOW_TOKENS", "65536"))
        except Exception:
            context_limit_tokens = 65536
        self._context_window_tokens = max(8192, context_limit_tokens)
        self._generation_config = {
            "temperature": 0.0,
            "top_p": 0.9,
            "response_mime_type": "application/json",
            "max_output_tokens": self._default_max_output_tokens,
        }
        schema_flag = str(os.getenv("EXECUTION_PLANNER_USE_RESPONSE_SCHEMA", "0")).strip().lower()
        self._use_response_schema = schema_flag not in {"0", "false", "no", "off", ""}
        base_url = os.getenv("EXECUTION_PLANNER_BASE_URL", "").strip()
        if not base_url:
            base_url = "https://openrouter.ai/api/v1"
        self.base_url = base_url or None
        self.model_name = (
            os.getenv("EXECUTION_PLANNER_PRIMARY_MODEL")
            or os.getenv("EXECUTION_PLANNER_MODEL")
            or "openai/gpt-5.4"
        ).strip()
        if not self.model_name:
            self.model_name = "openai/gpt-5.4"
        if not self.api_key:
            self.client = None
        else:
            client_kwargs: Dict[str, Any] = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            headers: Dict[str, str] = {}
            referer = os.getenv("OPENROUTER_HTTP_REFERER")
            if referer:
                headers["HTTP-Referer"] = referer
            title = os.getenv("OPENROUTER_X_TITLE")
            if title:
                headers["X-Title"] = title
            if headers:
                client_kwargs["default_headers"] = headers
            self.client = OpenAI(**client_kwargs)
        chain_raw = os.getenv("EXECUTION_PLANNER_MODEL_CHAIN", "")
        chain: List[str] = [self.model_name]
        if isinstance(chain_raw, str) and chain_raw.strip():
            for token in chain_raw.split(","):
                model = token.strip()
                if model and model not in chain:
                    chain.append(model)
        self.model_chain = chain
        self.last_prompt = None
        self.last_response = None
        self.last_contract_diagnostics = None

    @staticmethod
    def _estimate_prompt_tokens(prompt: str) -> int:
        # Simple approximation: Gemini tokenization is typically close to 3-4 chars/token in mixed JSON/text prompts.
        if not isinstance(prompt, str) or not prompt:
            return 1
        return max(1, len(prompt) // 4)

    def _generation_config_for_prompt(self, prompt: str, output_token_floor: int = 1024) -> Dict[str, Any]:
        prompt_tokens = self._estimate_prompt_tokens(prompt)
        available = self._context_window_tokens - prompt_tokens - 500
        floor = max(1024, int(output_token_floor))
        if available <= 0:
            budgeted_max = floor
        else:
            budgeted_max = min(self._default_max_output_tokens, max(floor, int(available)))
        config = dict(self._generation_config)
        config["max_output_tokens"] = int(budgeted_max)
        if self._use_response_schema:
            config["response_schema"] = copy.deepcopy(EXECUTION_CONTRACT_V42_MIN_SCHEMA)
        return config

    @staticmethod
    def _is_response_schema_unsupported_error(err: Exception) -> bool:
        message = str(err or "").lower()
        if "response_schema" not in message and "json_schema" not in message and "tool" not in message:
            return False
        unsupported_tokens = (
            "unknown field",
            "unknown name",
            "not supported",
            "unsupported",
            "unrecognized",
            "no such field",
            "schema not supported",
        )
        return any(token in message for token in unsupported_tokens)

    @staticmethod
    def _build_function_tool(name: str, description: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": copy.deepcopy(schema),
            },
        }

    @staticmethod
    def _extract_openai_response_text(response: Any) -> str:
        if response is None:
            return ""
        try:
            choices = getattr(response, "choices", None) or []
            if choices:
                message = getattr(choices[0], "message", None)
                tool_calls = getattr(message, "tool_calls", None) if message is not None else None
                if tool_calls:
                    function_payload = getattr(tool_calls[0], "function", None)
                    arguments = getattr(function_payload, "arguments", None)
                    if isinstance(arguments, str) and arguments.strip():
                        return arguments.strip()
                content = getattr(message, "content", None) if message is not None else None
                if isinstance(content, str) and content.strip():
                    return content.strip()
        except Exception:
            pass
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()
        return ""

    @staticmethod
    def _extract_openai_finish_reason(response: Any) -> Any:
        try:
            choices = getattr(response, "choices", None) or []
            if choices:
                return getattr(choices[0], "finish_reason", None)
        except Exception:
            return None
        return None

    def _generate_content_with_budget(
        self,
        model_client: Any,
        prompt: str,
        output_token_floor: int = 1024,
        *,
        model_name: str | None = None,
        tool_mode: str = "contract",
    ):
        generation_config = self._generation_config_for_prompt(prompt, output_token_floor=output_token_floor)
        if hasattr(model_client, "generate_content"):
            try:
                response = model_client.generate_content(prompt, generation_config=generation_config)
            except TypeError:
                response = model_client.generate_content(prompt)
            except Exception as err:
                if (
                    "response_schema" in generation_config
                    and self._is_response_schema_unsupported_error(err)
                ):
                    retry_config = dict(generation_config)
                    retry_config.pop("response_schema", None)
                    try:
                        response = model_client.generate_content(prompt, generation_config=retry_config)
                    except TypeError:
                        response = model_client.generate_content(prompt)
                    generation_config = retry_config
                else:
                    raise
            return response, generation_config

        selected_model = str(model_name or self.model_name or "").strip() or self.model_name
        schema = (
            copy.deepcopy(_EXECUTION_CONTRACT_PATCH_SCHEMA)
            if tool_mode == "patch"
            else copy.deepcopy(EXECUTION_CONTRACT_V42_MIN_SCHEMA)
        )
        tool = self._build_function_tool(
            _EXECUTION_CONTRACT_PATCH_TOOL_NAME if tool_mode == "patch" else _EXECUTION_CONTRACT_TOOL_NAME,
            "Return the final execution planner JSON payload.",
            schema,
        )
        messages = [{"role": "user", "content": prompt}]
        call_kwargs: Dict[str, Any] = {
            "model": selected_model,
            "messages": messages,
            "temperature": float(generation_config.get("temperature", 0.0)),
            "max_tokens": int(generation_config.get("max_output_tokens", self._default_max_output_tokens)),
            "tools": [tool],
            "tool_choice": {
                "type": "function",
                "function": {
                    "name": _EXECUTION_CONTRACT_PATCH_TOOL_NAME if tool_mode == "patch" else _EXECUTION_CONTRACT_TOOL_NAME
                },
            },
        }
        top_p = generation_config.get("top_p")
        if top_p is not None:
            call_kwargs["top_p"] = top_p
        try:
            response = model_client.chat.completions.create(**call_kwargs)
        except TypeError:
            retry_kwargs = dict(call_kwargs)
            retry_kwargs.pop("temperature", None)
            retry_kwargs.pop("top_p", None)
            response = model_client.chat.completions.create(**retry_kwargs)
        except Exception as err:
            if self._is_response_schema_unsupported_error(err):
                fallback_kwargs = dict(call_kwargs)
                fallback_kwargs.pop("tool_choice", None)
                response = model_client.chat.completions.create(**fallback_kwargs)
            else:
                raise
        return response, generation_config

    def _build_model_client(self, model_name: str) -> Any:
        if not self.api_key:
            return None
        return self.client

    def generate_contract(
        self,
        strategy: Dict[str, Any],
        data_summary: str = "",
        business_objective: str = "",
        column_inventory: list[str] | None = None,
        column_sets: Dict[str, Any] | None = None,
        column_manifest: Dict[str, Any] | None = None,
        output_dialect: Dict[str, str] | None = None,
        env_constraints: Dict[str, Any] | None = None,
        domain_expert_critique: str = "",
        data_profile: Dict[str, Any] | None = None,
        run_id: str | None = None
    ) -> Dict[str, Any]:

        def _norm(name: str) -> str:
            return re.sub(r"[^0-9a-zA-Z]+", "", str(name).lower())

        def _canonicalize_name(name: str) -> str:
            return str(name)

        def _resolve_exact_header(name: str) -> str | None:
            if not name or not column_inventory:
                return None
            norm_name = _norm(name)
            if not norm_name:
                return None
            best_match = None
            best_score = 0.0
            for raw in column_inventory:
                if raw is None:
                    continue
                raw_str = str(raw)
                raw_norm = _norm(raw_str)
                if raw_norm == norm_name:
                    return raw_str
                score = difflib.SequenceMatcher(None, norm_name, raw_norm).ratio()
                if score > best_score:
                    best_score = score
                    best_match = raw_str
            if best_score >= 0.9:
                return best_match
            return None

        def _infer_objective_type() -> str:
            objective_text = (business_objective or "").lower()
            strategy_obj = strategy if isinstance(strategy, dict) else {}
            analysis_type = str(strategy_obj.get("analysis_type") or "").lower()
            techniques_text = " ".join([str(t) for t in (strategy_obj.get("techniques") or [])]).lower()
            signal_text = " ".join([objective_text, analysis_type, techniques_text])
            prescriptive_tokens = [
                "optimiz",
                "maximize",
                "minimize",
                "pricing",
                "precio",
                "optimal",
                "optimo",
                "revenue",
                "expected value",
                "recommend",
                "allocation",
                "decision",
                "prescriptive",
                "ranking",
                "scoring",
            ]
            predictive_tokens = [
                "predict",
                "classification",
                "regression",
                "forecast",
                "probability",
                "probabilidad",
                "clasific",
                "conversion",
                "convert",
                "churn",
                "contract",
                "propensity",
                "predictive",
            ]
            causal_tokens = [
                "causal",
                "uplift",
                "impact",
                "intervention",
                "treatment",
            ]
            if any(tok in signal_text for tok in prescriptive_tokens):
                return "prescriptive"
            if any(tok in signal_text for tok in predictive_tokens):
                return "predictive"
            if any(tok in signal_text for tok in causal_tokens):
                return "causal"
            return "descriptive"

        def _safe_column_name(name: str) -> str:
            if not name:
                return ""
            return re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_").lower()

        def _deliverable_id_from_path(path: str) -> str:
            base = os.path.basename(str(path)) or str(path)
            cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", base).strip("_").lower()
            return cleaned or "deliverable"

        def _infer_deliverable_kind(path: str) -> str:
            from src.utils.contract_accessors import _infer_kind_from_path
            return _infer_kind_from_path(path)

        def _default_deliverable_description(path: str, kind: str) -> str:
            basename = os.path.basename(str(path or "")).lower()
            known_by_basename = {
                "cleaned_data.csv": "Cleaned dataset used for downstream modeling.",
                "metrics.json": "Model metrics and validation summary.",
                "weights.json": "Feature weights or scoring coefficients.",
                "case_summary.csv": "Per-case scoring summary.",
                "case_alignment_report.json": "Case alignment QA metrics.",
                "scored_rows.csv": "Row-level scores and key features.",
                "alignment_check.json": "Alignment check results for contract requirements.",
                "recommendations_preview.json": "Illustrative recommendation examples for the executive report.",
            }
            if basename in known_by_basename:
                return known_by_basename[basename]
            if str(path or "").replace("\\", "/").startswith("static/plots/"):
                return "Required diagnostic plots."
            if kind == "plot":
                return "Diagnostic plots required by the contract."
            if kind == "metrics":
                return "Metrics artifact required by the contract."
            if kind == "weights":
                return "Weights or scoring artifact required by the contract."
            return "Requested deliverable."

        def _infer_deliverable_owner(path: str) -> str:
            """Infer which engineer owns a deliverable based on its path."""
            lower = (path or "").lower()
            if any(tok in lower for tok in ("cleaned_data", "cleaning_manifest")):
                return "data_engineer"
            return "ml_engineer"

        def _build_deliverable(
            path: str,
            required: bool = True,
            kind: str | None = None,
            description: str | None = None,
            deliverable_id: str | None = None,
            owner: str | None = None,
        ) -> Dict[str, Any]:
            if not path:
                return {}
            kind_val = kind or _infer_deliverable_kind(path)
            desc_val = description or _default_deliverable_description(path, kind_val)
            deliverable_id = deliverable_id or _deliverable_id_from_path(path)
            return {
                "id": deliverable_id,
                "path": path,
                "required": bool(required),
                "kind": kind_val,
                "description": desc_val,
                "owner": owner or _infer_deliverable_owner(path),
            }

        def _normalize_deliverables(
            raw: Any,
            default_required: bool = True,
            required_paths: set[str] | None = None,
        ) -> List[Dict[str, Any]]:
            if not raw or not isinstance(raw, list):
                return []
            required_paths = {str(p) for p in (required_paths or set()) if p}
            normalized: List[Dict[str, Any]] = []
            for item in raw:
                if isinstance(item, str):
                    path = item
                    required = path in required_paths if required_paths else default_required
                    deliverable = _build_deliverable(path, required=required)
                    if deliverable:
                        normalized.append(deliverable)
                    continue
                if not isinstance(item, dict):
                    continue
                path = item.get("path") or item.get("output") or item.get("artifact")
                if not path:
                    continue
                required = item.get("required")
                if required is None:
                    required = path in required_paths if required_paths else default_required
                deliverable = _build_deliverable(
                    path=path,
                    required=bool(required),
                    kind=item.get("kind"),
                    description=item.get("description"),
                    deliverable_id=item.get("id"),
                    owner=item.get("owner"),
                )
                if deliverable:
                    normalized.append(deliverable)
            return normalized

        def _merge_deliverables(
            base: List[Dict[str, Any]],
            overrides: List[Dict[str, Any]],
        ) -> List[Dict[str, Any]]:
            merged = list(base)
            by_path = {item.get("path"): idx for idx, item in enumerate(merged) if item.get("path")}
            for item in overrides:
                path = item.get("path")
                if not path:
                    continue
                if path in by_path:
                    existing = merged[by_path[path]]
                    for key in ("id", "kind", "description", "owner"):
                        if item.get(key):
                            existing[key] = item.get(key)
                    if "required" in item and item.get("required") is not None:
                        existing["required"] = bool(item.get("required"))
                    merged[by_path[path]] = existing
                else:
                    merged.append(item)
                    by_path[path] = len(merged) - 1
            return merged

        def _ensure_unique_deliverable_ids(deliverables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            seen: set[str] = set()
            for item in deliverables:
                base_id = item.get("id") or _deliverable_id_from_path(item.get("path") or "")
                candidate = base_id
                suffix = 2
                while candidate in seen:
                    candidate = f"{base_id}_{suffix}"
                    suffix += 1
                item["id"] = candidate
                seen.add(candidate)
            return deliverables

        def _apply_deliverables(contract: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(contract, dict):
                return {}
            spec = contract.get("spec_extraction")
            if not isinstance(spec, dict):
                spec = {}

            def _normalize_path(p: str) -> str:
                return normalize_artifact_path(str(p))

            def _extract_required_paths(raw_outputs: Any) -> set[str]:
                paths: set[str] = set()
                if not isinstance(raw_outputs, list):
                    return paths
                for item in raw_outputs:
                    path: Any = ""
                    if isinstance(item, str):
                        path = item
                    elif isinstance(item, dict):
                        path = item.get("path") or item.get("output") or item.get("artifact")
                    if not path:
                        continue
                    normalized = _normalize_path(str(path).strip())
                    if normalized:
                        paths.add(normalized)
                return paths

            legacy_required = contract.get("required_outputs", []) or []
            legacy_required_paths = _extract_required_paths(legacy_required)
            legacy = _normalize_deliverables(legacy_required, default_required=True)
            existing = _normalize_deliverables(
                spec.get("deliverables"),
                default_required=True,
                required_paths=legacy_required_paths,
            )
            # LLM-first policy: normalize only deliverables explicitly declared
            # by the planner contract. Do not synthesize business artifacts from
            # objective defaults or platform heuristics.
            deliverables = _merge_deliverables(legacy, existing)
            deliverables = _ensure_unique_deliverable_ids(deliverables)
            spec["deliverables"] = deliverables
            contract["spec_extraction"] = spec

            # Auto-sync dual interface:
            # - required_output_artifacts: rich deliverables (all, required/optional)
            # - required_outputs: List[str] with required artifact paths only (compat)
            required_output_artifacts: List[Dict[str, Any]] = []
            required_output_paths_seen: set[str] = set()
            required_outputs_paths: List[str] = []
            for deliverable in deliverables:
                if not isinstance(deliverable, dict):
                    continue
                normalized_path = _normalize_path(deliverable.get("path") or "")
                if not normalized_path:
                    continue
                path_key = normalized_path.lower()
                if path_key in required_output_paths_seen:
                    continue
                required_output_paths_seen.add(path_key)
                is_required = bool(deliverable.get("required"))
                normalized_deliverable = _build_deliverable(
                    path=normalized_path,
                    required=is_required,
                    kind=deliverable.get("kind"),
                    description=deliverable.get("description"),
                    deliverable_id=deliverable.get("id"),
                    owner=deliverable.get("owner"),
                )
                if normalized_deliverable:
                    required_output_artifacts.append(normalized_deliverable)
                    if is_required:
                        required_outputs_paths.append(normalized_path)
            contract["required_output_artifacts"] = required_output_artifacts
            contract["required_outputs"] = required_outputs_paths

            artifact_reqs = contract.get("artifact_requirements", {})
            if isinstance(artifact_reqs, dict):
                visual_reqs = artifact_reqs.get("visual_requirements")
                if isinstance(visual_reqs, dict):
                    outputs_dir = visual_reqs.get("outputs_dir") or "static/plots"
                    items = visual_reqs.get("items") if isinstance(visual_reqs.get("items"), list) else []
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        expected = item.get("expected_filename")
                        if not expected:
                            continue
                        plot_path = (
                            expected
                            if os.path.isabs(expected)
                            else os.path.normpath(os.path.join(outputs_dir, expected))
                        )
                        normalized_plot_path = _normalize_path(str(plot_path))
                        if not is_probably_path(normalized_plot_path):
                            continue
                        plot_key = normalized_plot_path.lower()
                        if plot_key in required_output_paths_seen:
                            continue
                        required_output_paths_seen.add(plot_key)
                        plot_entry = _build_deliverable(
                            path=normalized_plot_path,
                            required=True,
                            kind="plot",
                            description="Required visualization artifact.",
                            owner="ml_engineer",
                        )
                        if plot_entry:
                            contract["required_output_artifacts"].append(plot_entry)
                            contract["required_outputs"].append(normalized_plot_path)

            return contract

        def _lint_deliverable_invariants(contract: Dict[str, Any], objective_type: str) -> List[Dict[str, Any]]:
            """Validate deliverable invariants by kind/owner.

            Returns list of structured missing_invariants. Does NOT mutate contract.
            """
            spec = contract.get("spec_extraction") or {}
            deliverables = spec.get("deliverables") or []
            missing: List[Dict[str, Any]] = []

            for d in deliverables:
                if not isinstance(d, dict):
                    continue
                if not d.get("kind"):
                    missing.append({
                        "invariant": "explicit_kind_required",
                        "severity": "error",
                        "message": f"Deliverable '{d.get('path')}' is missing explicit 'kind' field.",
                        "deliverable_path": d.get("path"),
                    })

            # INV-1: DE stage requires at least one required dataset(owner=data_engineer)
            if not any(isinstance(d, dict) and d.get("required") and d.get("kind") == "dataset"
                       and d.get("owner") == "data_engineer" for d in deliverables):
                missing.append({
                    "invariant": "de_requires_dataset",
                    "severity": "error",
                    "message": "Data engineer stage requires at least one required deliverable with kind='dataset' and owner='data_engineer'.",
                    "expected_kind": "dataset", "expected_owner": "data_engineer",
                })

            # INV-2: Model training requires metrics(owner=ml_engineer)
            involves_training = objective_type in ("predictive", "prescriptive", "causal")
            if involves_training:
                if not any(isinstance(d, dict) and d.get("required") and d.get("kind") == "metrics"
                           and d.get("owner") == "ml_engineer" for d in deliverables):
                    missing.append({
                        "invariant": "ml_requires_metrics",
                        "severity": "error",
                        "message": "Model training objective requires at least one required deliverable with kind='metrics' and owner='ml_engineer'.",
                        "expected_kind": "metrics", "expected_owner": "ml_engineer",
                    })

                # INV-3: Model training requires predictions OR submission
                if not any(isinstance(d, dict) and d.get("required") and d.get("kind") in ("predictions", "submission")
                           and d.get("owner") == "ml_engineer" for d in deliverables):
                    missing.append({
                        "invariant": "ml_requires_predictions_or_submission",
                        "severity": "error",
                        "message": "Model training objective requires at least one required deliverable with kind='predictions' or kind='submission' and owner='ml_engineer'.",
                        "expected_kind": "predictions|submission", "expected_owner": "ml_engineer",
                    })

            # INV-4: Kaggle/competition requires submission
            if _detect_submission_requirement():
                if not any(isinstance(d, dict) and d.get("required") and d.get("kind") == "submission"
                           and d.get("owner") == "ml_engineer" for d in deliverables):
                    missing.append({
                        "invariant": "competition_requires_submission",
                        "severity": "error",
                        "message": "Competition/Kaggle objective requires at least one required deliverable with kind='submission' and owner='ml_engineer'.",
                        "expected_kind": "submission", "expected_owner": "ml_engineer",
                    })

            return missing

        def _pattern_name(name: str) -> str:
            return re.sub(r"[^0-9a-zA-Z]+", "_", str(name).lower()).strip("_")

        # Ensure data_summary is a string
        data_summary_str = ""
        if isinstance(data_summary, dict):
            data_summary_str = json.dumps(data_summary, indent=2)
        else:
            data_summary_str = str(data_summary)

        constant_anchor_avoidance: List[str] = []
        if isinstance(data_profile, dict):
            const_exact = data_profile.get("constant_columns")
            if isinstance(const_exact, list) and const_exact:
                constant_anchor_avoidance = [str(c) for c in const_exact if c]
            else:
                const_sample = data_profile.get("constant_columns_sample")
                if isinstance(const_sample, list) and const_sample:
                    constant_anchor_avoidance = [str(c) for c in const_sample if c]

        relevant_payload = select_relevant_columns(
            strategy=strategy,
            business_objective=business_objective,
            domain_expert_critique=domain_expert_critique,
            column_inventory=column_inventory or [],
            data_profile_summary=data_summary_str,
            constant_anchor_avoidance=constant_anchor_avoidance,
        )
        relevant_columns = relevant_payload.get("relevant_columns", [])
        relevant_sources = relevant_payload.get("relevant_sources", {})
        omitted_columns_policy = relevant_payload.get("omitted_columns_policy", "")
        relevant_columns_total_count = int(relevant_payload.get("relevant_columns_total_count") or len(relevant_columns))
        relevant_columns_truncated = bool(relevant_payload.get("relevant_columns_truncated"))
        relevant_columns_omitted_count = int(relevant_payload.get("relevant_columns_omitted_count") or 0)
        relevant_columns_compact = relevant_payload.get("relevant_columns_compact", {})
        strategy_feature_family_hints = relevant_payload.get("strategy_feature_family_hints", [])
        strategy_feature_family_expanded_count = int(
            relevant_payload.get("strategy_feature_family_expanded_count") or 0
        )

        planner_dir = None
        if run_id:
            run_dir = get_run_dir(run_id)
            if run_dir:
                planner_dir = os.path.join(run_dir, "agents", "execution_planner")
                os.makedirs(planner_dir, exist_ok=True)

        planner_diag: List[Dict[str, Any]] = []
        self.last_planner_diag = planner_diag
        self.last_contract_min = None
        planner_candidate_invalid: Dict[str, Any] | None = None
        planner_candidate_invalid_raw: str | None = None
        planner_candidate_invalid_meta: Dict[str, Any] | None = None

        def _write_text(path: str, content: str) -> None:
            try:
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(content or "")
            except Exception:
                pass

        def _write_json(path: str, payload: Any) -> None:
            try:
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, indent=2, ensure_ascii=True)
            except Exception:
                pass

        def _persist_attempt(prompt_name: str, response_name: str, prompt_text: str, response_text: str | None) -> None:
            if not planner_dir:
                return
            if prompt_text is not None:
                _write_text(os.path.join(planner_dir, prompt_name), prompt_text)
            if response_text is not None:
                _write_text(os.path.join(planner_dir, response_name), response_text)

        def _persist_contracts(
            full_contract: Dict[str, Any] | None,
            diagnostics_payload: Dict[str, Any] | None = None,
            invalid_contract: Dict[str, Any] | None = None,
            invalid_raw: str | None = None,
            invalid_meta: Dict[str, Any] | None = None,
        ) -> None:
            if not planner_dir:
                return
            if full_contract:
                _write_json(os.path.join(planner_dir, "contract_full.json"), full_contract)
            if planner_diag:
                _write_json(os.path.join(planner_dir, "planner_diag.json"), {"attempts": planner_diag})
            if diagnostics_payload:
                _write_json(
                    os.path.join(planner_dir, "contract_diagnostics.json"),
                    diagnostics_payload,
                )
            if isinstance(invalid_contract, dict) and invalid_contract:
                _write_json(
                    os.path.join(planner_dir, "contract_candidate_invalid.json"),
                    invalid_contract,
                )
            if isinstance(invalid_raw, str) and invalid_raw.strip():
                _write_text(
                    os.path.join(planner_dir, "contract_candidate_invalid_raw.txt"),
                    invalid_raw,
                )
            if isinstance(invalid_meta, dict) and invalid_meta:
                _write_json(
                    os.path.join(planner_dir, "contract_candidate_invalid_meta.json"),
                    invalid_meta,
                )

        def _parse_json_response(raw_text: str) -> Tuple[Optional[Any], Optional[Exception]]:
            if not raw_text:
                return None, ValueError("Empty response text")
            cleaned = raw_text.replace("```json", "").replace("```", "").strip()
            candidates: List[str] = []
            for candidate in (
                cleaned,
                _extract_json_object(cleaned),
                _repair_common_json_damage(cleaned),
            ):
                if isinstance(candidate, str) and candidate.strip():
                    text = candidate.strip()
                    if text not in candidates:
                        candidates.append(text)

            last_err: Optional[Exception] = None
            decoder = json.JSONDecoder()
            for candidate in candidates:
                try:
                    return json.loads(candidate), None
                except Exception as err:
                    last_err = err
                # Allow valid JSON object with trailing noise.
                try:
                    parsed, end_idx = decoder.raw_decode(candidate)
                    trailing = candidate[end_idx:].strip()
                    if isinstance(parsed, dict) and not trailing:
                        return parsed, None
                    if isinstance(parsed, dict) and trailing:
                        return parsed, None
                except Exception as err:
                    last_err = err
                # Last-resort: python-literal dicts (single quotes, True/False/None).
                try:
                    literal = ast.literal_eval(candidate)
                    if isinstance(literal, dict):
                        return literal, None
                except Exception as err:
                    last_err = err
            return None, last_err or ValueError("Unable to parse planner JSON response")

        def _normalize_usage_metadata(raw_usage: Any) -> Optional[Dict[str, Any]]:
            if raw_usage is None:
                return None
            if isinstance(raw_usage, dict):
                return raw_usage
            if hasattr(raw_usage, "to_dict"):
                try:
                    return raw_usage.to_dict()
                except Exception:
                    pass
            usage_payload = {}
            for key in (
                "prompt_token_count",
                "candidates_token_count",
                "completion_token_count",
                "total_token_count",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
            ):
                try:
                    value = getattr(raw_usage, key)
                except Exception:
                    value = None
                if value is not None:
                    usage_payload[key] = value
            if "prompt_tokens" in usage_payload and "prompt_token_count" not in usage_payload:
                usage_payload["prompt_token_count"] = usage_payload.get("prompt_tokens")
            if "completion_tokens" in usage_payload and "candidates_token_count" not in usage_payload:
                usage_payload["candidates_token_count"] = usage_payload.get("completion_tokens")
            if "completion_tokens" in usage_payload and "completion_token_count" not in usage_payload:
                usage_payload["completion_token_count"] = usage_payload.get("completion_tokens")
            if "total_tokens" in usage_payload and "total_token_count" not in usage_payload:
                usage_payload["total_token_count"] = usage_payload.get("total_tokens")
            return usage_payload or {"value": str(raw_usage)}

        def _build_parse_feedback(raw_text: str | None, parse_error: Exception | None) -> str:
            if not parse_error and not raw_text:
                return "No parse diagnostics available."
            err_type = type(parse_error).__name__ if parse_error else "UnknownParseError"
            err_msg = str(parse_error) if parse_error else "Unknown parsing error."
            line_no: Optional[int] = None
            col_no: Optional[int] = None
            if parse_error is not None:
                if hasattr(parse_error, "lineno"):
                    try:
                        line_no = int(getattr(parse_error, "lineno"))
                    except Exception:
                        line_no = None
                if hasattr(parse_error, "colno"):
                    try:
                        col_no = int(getattr(parse_error, "colno"))
                    except Exception:
                        col_no = None
                if line_no is None:
                    m = re.search(r"line\s+(\d+)", err_msg)
                    if m:
                        try:
                            line_no = int(m.group(1))
                        except Exception:
                            line_no = None
            snippet = ""
            if isinstance(raw_text, str) and raw_text.strip() and line_no and line_no > 0:
                lines = raw_text.splitlines()
                idx = line_no - 1
                if 0 <= idx < len(lines):
                    snippet = lines[idx]
            out_lines = [
                f"- parse_error_type: {err_type}",
                f"- parse_error_message: {err_msg}",
            ]
            if line_no is not None:
                out_lines.append(f"- parse_error_line: {line_no}")
            if col_no is not None:
                out_lines.append(f"- parse_error_col: {col_no}")
            if snippet:
                out_lines.append(f"- parse_error_snippet: {snippet[:300]}")
            return "\n".join(out_lines)

        def _contract_is_accepted(validation_result: Dict[str, Any] | None) -> bool:
            if not isinstance(validation_result, dict):
                return False
            if not bool(validation_result.get("accepted", False)):
                return False
            status = str(validation_result.get("status") or "").lower()
            if status == "error":
                return False
            summary = validation_result.get("summary")
            if isinstance(summary, dict):
                try:
                    if int(summary.get("error_count", 0) or 0) > 0:
                        return False
                except Exception:
                    pass
            issues = validation_result.get("issues")
            blocking_warning_rules = {
                # Contract is semantically incomplete for wide-schema alignment.
                "contract.clean_dataset_selector_hints_unresolved",
                "contract.clean_dataset_required_feature_selectors",
            }
            if isinstance(issues, list):
                for issue in issues:
                    if not isinstance(issue, dict):
                        continue
                    sev = str(issue.get("severity") or "").lower()
                    if sev in {"error", "fail"}:
                        return False
                    if sev == "warning":
                        rule = str(issue.get("rule") or "").strip()
                        if rule in blocking_warning_rules:
                            return False
            return True

        def _compact_validation_feedback(validation_result: Dict[str, Any] | None, max_issues: int = 10) -> str:
            if not isinstance(validation_result, dict):
                return "No validation feedback available."
            issues = validation_result.get("issues")
            if not isinstance(issues, list) or not issues:
                return "No issues reported."
            lines: List[str] = []
            for issue in issues[:max_issues]:
                if not isinstance(issue, dict):
                    continue
                severity = str(issue.get("severity") or "warning").upper()
                rule = str(issue.get("rule") or "unknown_rule")
                message = str(issue.get("message") or "").strip()
                if message:
                    lines.append(f"- [{severity}] {rule}: {message}")
            if len(issues) > max_issues:
                lines.append(f"- ... ({len(issues) - max_issues} more issues)")
            return "\n".join(lines) if lines else "No issues reported."

        def _validate_contract_quality(contract_payload: Dict[str, Any] | None) -> Dict[str, Any]:
            payload_for_validation = _apply_planner_structural_support(
                contract_payload if isinstance(contract_payload, dict) else {}
            )
            base_result: Dict[str, Any]
            steward_semantics_payload: Dict[str, Any] = {}
            if isinstance(data_profile, dict):
                semantics_candidate = data_profile.get("dataset_semantics")
                if isinstance(semantics_candidate, dict) and semantics_candidate:
                    steward_semantics_payload = semantics_candidate
                observed_target_values = _build_target_observed_values_map(
                    payload_for_validation,
                    data_profile,
                )
                if observed_target_values:
                    steward_semantics_payload = dict(steward_semantics_payload)
                    steward_semantics_payload["target_observed_values"] = observed_target_values
            try:
                base_result = validate_contract_minimal_readonly(
                    copy.deepcopy(payload_for_validation or {}),
                    column_inventory=column_inventory,
                    steward_semantics=steward_semantics_payload,
                )
            except Exception as val_err:
                base_result = {
                    "status": "error",
                    "accepted": False,
                    "issues": [
                        {
                            "severity": "error",
                            "rule": "contract_validation_exception",
                            "message": str(val_err),
                        }
                    ],
                    "summary": {"error_count": 1, "warning_count": 0},
                }

            if not isinstance(base_result, dict):
                return {
                    "status": "error",
                    "accepted": False,
                    "issues": [
                        {
                            "severity": "error",
                            "rule": "contract_validation_result_invalid",
                            "message": "Contract validator returned an invalid result payload.",
                        }
                    ],
                    "summary": {"error_count": 1, "warning_count": 0},
                }

            return base_result

        _ROLE_BUCKETS = (
            "pre_decision",
            "decision",
            "outcome",
            "post_decision_audit_only",
            "identifiers",
            "time_columns",
            "unknown",
        )

        _ROLE_ALIASES = {
            "pre_decision": "pre_decision",
            "predecision": "pre_decision",
            "pre_decision_features": "pre_decision",
            "feature": "pre_decision",
            "features": "pre_decision",
            "model_feature": "pre_decision",
            "model_features": "pre_decision",
            "predictor": "pre_decision",
            "predictors": "pre_decision",
            "input": "pre_decision",
            "inputs": "pre_decision",
            "decision": "decision",
            "decisions": "decision",
            "action": "decision",
            "actions": "decision",
            "outcome": "outcome",
            "target": "outcome",
            "targets": "outcome",
            "label": "outcome",
            "labels": "outcome",
            "post_decision_audit_only": "post_decision_audit_only",
            "post_decision": "post_decision_audit_only",
            "audit_only": "post_decision_audit_only",
            "audit": "post_decision_audit_only",
            "forbidden": "post_decision_audit_only",
            "forbidden_for_modeling": "post_decision_audit_only",
            "id": "identifiers",
            "identifier": "identifiers",
            "identifiers": "identifiers",
            "key": "identifiers",
            "keys": "identifiers",
            "split": "identifiers",
            "split_identifier": "identifiers",
            "split_indicator": "identifiers",
            "partition": "identifiers",
            "fold": "identifiers",
            "time": "time_columns",
            "timestamp": "time_columns",
            "datetime": "time_columns",
            "date": "time_columns",
            "time_columns": "time_columns",
            "unknown": "unknown",
        }

        def _normalize_role_token(token: Any) -> str:
            key = re.sub(r"[^a-z0-9]+", "_", str(token or "").strip().lower()).strip("_")
            return _ROLE_ALIASES.get(key, "")

        def _looks_like_selector_token(value: str) -> bool:
            token = str(value or "").strip()
            if not token:
                return False
            low = token.lower()
            if low.startswith(("regex:", "pattern:", "prefix:", "suffix:", "contains:", "selector:")):
                return True
            if "*" in token or "?" in token:
                return True
            if token.startswith("^") or token.endswith("$"):
                return True
            if "\\d" in token or "\\w" in token or "\\s" in token:
                return True
            if any(ch in token for ch in ("[", "]", "(", ")", "{", "}", "|", "+")):
                return True
            if low.endswith(("_features", "_feature_set", "_family")):
                return True
            if low in {"features", "feature_set", "model_features", "all_features"}:
                return True
            return False

        def _selector_matches_column(selector: Dict[str, Any], column: str) -> bool:
            if not isinstance(selector, dict) or not isinstance(column, str) or not column.strip():
                return False
            selector_type = str(selector.get("type") or "").strip().lower()
            col = column.strip()
            if not selector_type:
                return False
            try:
                if selector_type in {"regex", "pattern"}:
                    pattern = str(selector.get("pattern") or "").strip()
                    return bool(pattern) and re.compile(pattern, flags=re.IGNORECASE).match(col) is not None
                if selector_type == "prefix":
                    prefix = str(selector.get("value") or selector.get("prefix") or "").strip()
                    return bool(prefix) and col.lower().startswith(prefix.lower())
                if selector_type == "suffix":
                    suffix = str(selector.get("value") or selector.get("suffix") or "").strip()
                    return bool(suffix) and col.lower().endswith(suffix.lower())
                if selector_type == "contains":
                    token = str(selector.get("value") or "").strip()
                    return bool(token) and token.lower() in col.lower()
                if selector_type == "list":
                    cols, _ = _to_clean_str_list(selector.get("columns"))
                    return col in set(cols)
                if selector_type in {"all_columns_except", "all_numeric_except"}:
                    excluded, _ = _to_clean_str_list(selector.get("except_columns") or selector.get("value"))
                    return col.lower() not in {c.lower() for c in excluded}
                if selector_type == "prefix_numeric_range":
                    prefix = str(selector.get("prefix") or "").strip()
                    start = selector.get("start")
                    end = selector.get("end")
                    if not prefix or not isinstance(start, int) or not isinstance(end, int):
                        return False
                    m = re.compile(rf"^{re.escape(prefix)}(\d+)$", flags=re.IGNORECASE).match(col)
                    if not m:
                        return False
                    idx = int(m.group(1))
                    lo = min(start, end)
                    hi = max(start, end)
                    return lo <= idx <= hi
            except Exception:
                return False
            return False

        def _column_matches_any_selector(column: str, selectors: Any) -> bool:
            if not isinstance(selectors, list):
                return False
            for selector in selectors:
                if _selector_matches_column(selector, column):
                    return True
            return False

        def _normalize_column_roles_payload(
            raw_roles: Any,
            canonical_columns: List[str] | None = None,
        ) -> Tuple[Dict[str, List[str]], List[str]]:
            issues: List[str] = []
            normalized: Dict[str, List[str]] = {bucket: [] for bucket in _ROLE_BUCKETS}

            canon_norm_map: Dict[str, str] = {}
            for col in (canonical_columns or []):
                norm = _normalize_column_identifier(col)
                if norm and norm not in canon_norm_map:
                    canon_norm_map[norm] = str(col)

            def _resolve_col(col_name: str) -> str:
                col = str(col_name or "").strip()
                if not col:
                    return ""
                if _looks_like_selector_token(col):
                    return col
                norm = _normalize_column_identifier(col)
                return canon_norm_map.get(norm) or col

            def _add(bucket: str, col_name: str) -> None:
                col = _resolve_col(col_name)
                if not col:
                    return
                values = normalized.setdefault(bucket, [])
                if col not in values:
                    values.append(col)

            if not isinstance(raw_roles, dict):
                return {}, ["column_roles must be an object."]
            if not raw_roles:
                return {}, ["column_roles cannot be empty."]

            value_types = {type(v).__name__ for v in raw_roles.values()}
            role_bucket_mode = any(isinstance(v, list) for v in raw_roles.values())

            if role_bucket_mode:
                for raw_role, cols in raw_roles.items():
                    bucket = _normalize_role_token(raw_role)
                    if not bucket:
                        issues.append(
                            f"column_roles role '{raw_role}' is not canonical; use one of {list(_ROLE_BUCKETS)}"
                        )
                        bucket = "unknown"
                    clean_cols, invalid_cols = _to_clean_str_list(cols)
                    if invalid_cols:
                        issues.append(f"column_roles['{raw_role}'] must be list[str].")
                    for col in clean_cols:
                        _add(bucket, col)
            else:
                for raw_col, raw_role in raw_roles.items():
                    role_token = ""
                    if isinstance(raw_role, dict):
                        role_token = str(raw_role.get("role") or "").strip()
                    elif isinstance(raw_role, str):
                        role_token = raw_role.strip()
                    if not role_token:
                        issues.append(f"column_roles mapping for '{raw_col}' is missing role.")
                        bucket = "unknown"
                    else:
                        bucket = _normalize_role_token(role_token)
                        if not bucket:
                            issues.append(
                                f"column_roles role '{role_token}' is not canonical; use one of {list(_ROLE_BUCKETS)}"
                            )
                            bucket = "unknown"
                    _add(bucket, str(raw_col))

            normalized = {k: v for k, v in normalized.items() if isinstance(v, list) and v}

            # Ensure feature bucket can be projected even when split/target buckets are present.
            if not normalized.get("pre_decision"):
                canonical = [str(c) for c in (canonical_columns or []) if c]
                assigned = set()
                for bucket in ("outcome", "decision", "post_decision_audit_only", "identifiers", "time_columns"):
                    assigned.update(str(c) for c in normalized.get(bucket, []) if c)
                inferred = [col for col in canonical if col not in assigned]
                if inferred:
                    normalized["pre_decision"] = inferred

            if not normalized.get("outcome"):
                issues.append("column_roles must include outcome/target bucket with at least one column.")

            if not normalized.get("pre_decision"):
                issues.append("column_roles must include pre_decision/model feature columns or resolvable selectors.")

            if issues:
                issues.append(f"column_roles mode detected: {sorted(value_types)}")
            return normalized, issues

        def _build_targeted_repair_actions(previous_validation: Dict[str, Any] | None) -> str:
            if not isinstance(previous_validation, dict):
                return "- Fix all structural and semantic issues while preserving unchanged valid fields."
            issues = previous_validation.get("issues")
            if not isinstance(issues, list):
                return "- Fix all structural and semantic issues while preserving unchanged valid fields."
            actions: List[str] = []
            seen: set[str] = set()
            rule_to_action = {
                "contract.cleaning_transforms_drop_conflict": (
                    "Ensure drop_columns NEVER contains columns listed in clean_dataset.required_columns."
                ),
                "contract.clean_dataset_selector_drop_required_conflict": (
                    "When selector-drop policy is active, keep required_columns outside required_feature_selectors "
                    "(required columns must be non-droppable anchors)."
                ),
                "contract.clean_dataset_selector_drop_passthrough_conflict": (
                    "When selector-drop policy is active, keep optional_passthrough_columns outside required_feature_selectors."
                ),
                "contract.cleaning_gate_selector_drop_conflict": (
                    "HARD cleaning gates must not depend on selector-covered columns when selector-drop policy is enabled."
                ),
                "contract.cleaning_transforms_scale_conflict": (
                    "Ensure scale_columns are covered by required_columns/optional_passthrough_columns or required_feature_selectors "
                    "(use concrete columns or explicit selector refs such as regex:/prefix:/selector:<name>)."
                ),
                "contract.clean_dataset_selector_drop_policy_missing": (
                    "When using required_feature_selectors with criteria-based drop directives, declare "
                    "column_transformations.drop_policy.allow_selector_drops_when."
                ),
                "contract.clean_dataset_ml_columns_missing": (
                    "Ensure every ML-required column is covered by clean_dataset.required_columns, passthrough, or selectors."
                ),
                "contract.clean_dataset_drop_ml_columns_conflict": (
                    "Do not drop columns needed by ML stage."
                ),
                "contract.canonical_columns_coverage": (
                    "Increase effective canonical coverage of column_inventory: include missing anchor columns "
                    "and/or declare required_feature_selectors so coverage reflects real feature families."
                ),
                "contract.outcome_columns_sanity": (
                    "Set outcome columns to target column(s) only; move non-target fields into pre_decision."
                ),
                "contract.model_features_empty": (
                    "Ensure allowed_feature_sets.model_features contains useful modeling features, "
                    "not only split/id structural columns."
                ),
                "contract.role_ontology": (
                    "Use canonical column_roles buckets only: pre_decision, decision, outcome, post_decision_audit_only, identifiers, time_columns, unknown."
                ),
                "contract.scope_unknown": (
                    "Set scope to one of: cleaning_only, ml_only, full_pipeline."
                ),
                "contract.required_outputs_path": (
                    "required_outputs must contain artifact file paths only (no conceptual labels)."
                ),
                "contract.column_dtype_targets": (
                    "Each column_dtype_targets entry must be an object with key 'target_dtype' "
                    "(NOT 'type'). Example: {\"target_dtype\": \"float64\", \"nullable\": false}. "
                    "Replace all {\"type\": X} with {\"target_dtype\": X}."
                ),
            }
            for issue in issues:
                if not isinstance(issue, dict):
                    continue
                rule = str(issue.get("rule") or "").strip()
                if not rule or rule in seen:
                    continue
                seen.add(rule)
                action = rule_to_action.get(rule) or get_contract_schema_repair_action(rule)
                if action:
                    actions.append(f"- {action}")
                else:
                    msg = str(issue.get("message") or "").strip()
                    if msg:
                        actions.append(f"- Resolve `{rule}`: {msg}")
                if len(actions) >= 8:
                    break
            if not actions:
                actions.append("- Fix all structural and semantic issues while preserving unchanged valid fields.")
            return "\n".join(actions)

        def _build_quality_repair_prompt(
            *,
            previous_contract: Dict[str, Any] | None,
            previous_validation: Dict[str, Any] | None,
            previous_response_text: str | None,
            previous_parse_feedback: str | None,
            original_inputs_text: str | None,
        ) -> str:
            required_top_level = [
                "scope",
                "strategy_title",
                "business_objective",
                "output_dialect",
                "canonical_columns",
                "required_outputs",
                "column_roles",
                "artifact_requirements",
                "iteration_policy",
            ]
            validation_feedback = _compact_validation_feedback(previous_validation)
            parse_feedback = (previous_parse_feedback or "").strip() or "No parse diagnostics available."
            previous_payload = (
                json.dumps(previous_contract, indent=2, ensure_ascii=False)
                if isinstance(previous_contract, dict)
                else (previous_response_text or "")
            )
            targeted_actions = _build_targeted_repair_actions(previous_validation)
            original_inputs_compact = _compress_text_preserve_ends(
                original_inputs_text or "",
                max_chars=20000,
                head=15000,
                tail=5000,
            )
            previous_payload_compact = _compress_text_preserve_ends(
                previous_payload or "",
                max_chars=20000,
                head=15000,
                tail=5000,
            )
            return (
                "Repair the previous execution contract.\n"
                "Return ONLY one valid JSON object (no markdown, no comments, no code fences).\n"
                "Use a repair workflow: re-ground facts, patch only the affected contract fields, then self-check.\n"
                "Patch policy: keep unchanged valid fields stable; modify ONLY fields needed to resolve listed issues.\n"
                "Schema registry examples:\n"
                + CONTRACT_SCHEMA_EXAMPLES_TEXT
                + "\n"
                "scope MUST be one of: cleaning_only, ml_only, full_pipeline.\n"
                "Do not invent columns outside column_inventory.\n"
                "Use downstream_consumer_interface + evidence_policy from ORIGINAL INPUTS.\n"
                "column_dtype_targets values MUST use key target_dtype (never key type).\n"
                "Example: {\"age\": {\"target_dtype\": \"float64\", \"nullable\": false}}.\n"
                "Do not remove required business intent; fix only structural/semantic contract errors.\n"
                "Preserve business objective, dataset context, and selected strategy from ORIGINAL INPUTS.\n"
                "column_roles MUST use canonical role buckets only: "
                "pre_decision, decision, outcome, post_decision_audit_only, identifiers, time_columns, unknown.\n"
                "For cleaning scopes, define artifact_requirements.clean_dataset.output_path and output_manifest_path.\n"
                "required_feature_selectors entries must be list[object] and each object must have key type.\n"
                "Example: [{\"type\": \"prefix\", \"value\": \"feature_\"}, {\"type\": \"regex\", \"pattern\": \"^pixel_\\\\d+$\"}].\n"
                "If cleaning requires dropping/scaling columns, declare them in "
                "artifact_requirements.clean_dataset.column_transformations.{drop_columns,scale_columns,drop_policy}; "
                "do not leave these decisions only in runbook prose.\n"
                "If required_feature_selectors are used and any drop-by-criteria is requested, define "
                "column_transformations.drop_policy.allow_selector_drops_when with explicit reasons.\n"
                "When scaling a selector-defined family on wide schemas, scale_columns may use selector refs "
                "(regex:/prefix:/suffix:/contains:/selector:<name>) instead of enumerating every column.\n"
                "If selector-drop policy is active, required_columns and optional_passthrough_columns MUST be non-droppable anchors "
                "and therefore must not overlap required_feature_selectors.\n"
                "If selector-drop policy is active, HARD cleaning gates must not rely on selector-covered columns.\n"
                "For wide feature families, you may declare compact selectors in "
                "artifact_requirements.clean_dataset.required_feature_selectors (regex/prefix/range/list).\n"
                "Avoid enumerating massive feature families as explicit lists; keep explicit anchors only.\n"
                "Never place wildcard selector tokens (e.g., pixel*) inside required_columns.\n"
                "If strategy/data indicate robust outlier handling, include optional outlier_policy with "
                "enabled/apply_stage/target_columns/report_path/strict.\n"
                "For ML scopes, include non-empty evaluation_spec and ensure objective_analysis.problem_type "
                "(or evaluation_spec.objective_type) and non-empty column_roles.\n"
                "For ML scopes, include artifact_requirements.visual_requirements and reporting_policy.plot_spec "
                "aligned with strategy/evidence so views can request the right visuals.\n"
                "Gate lists must be executable by downstream views: use gate objects with "
                "{name, severity, params} (severity in HARD|SOFT). If using metric/check/rule language, "
                "map it to name and keep semantic details in params.\n"
                "Gate example: {\"name\": \"no_nulls_target\", \"severity\": \"HARD\", \"params\": {\"column\": \"target\"}}.\n"
                "Top-level minimum keys (required interface for executable views): "
                + json.dumps(required_top_level)
                + "\n\nTargeted fixes to apply first:\n"
                + targeted_actions
                + "\n\nORIGINAL INPUTS (source of truth):\n"
                + original_inputs_compact
                + "\n\nValidation issues to fix:\n"
                + validation_feedback
                + "\n\nParse diagnostics from previous attempt:\n"
                + parse_feedback
                + "\n\nPrevious candidate contract/response:\n"
                + previous_payload_compact
            )

        _SCOPE_VALUES = {"cleaning_only", "ml_only", "full_pipeline"}

        def _scope_requires_cleaning(scope: str) -> bool:
            return scope in {"cleaning_only", "full_pipeline"}

        def _scope_requires_ml(scope: str) -> bool:
            return scope in {"ml_only", "full_pipeline"}

        def _normalize_scope(scope_value: Any) -> str:
            token = str(scope_value or "").strip().lower()
            if token in _SCOPE_VALUES:
                return token
            return normalize_contract_scope(scope_value)

        def _runbook_non_empty(value: Any) -> bool:
            if isinstance(value, str):
                return bool(value.strip())
            if isinstance(value, list):
                return any(bool(str(item).strip()) for item in value if item is not None)
            if isinstance(value, dict):
                if "steps" in value and isinstance(value.get("steps"), list):
                    return any(bool(str(step).strip()) for step in value.get("steps") or [] if step is not None)
                return bool(value)
            return False

        _drop_column_pattern = re.compile(
            r"\b(drop|discard|remove|eliminar|descartar|quitar)\b.{0,32}\b(column|columns|columna|columnas)\b",
            re.IGNORECASE,
        )
        _scale_column_pattern = re.compile(
            r"\b(scale|rescale|standardize|standardise|minmax|zscore|z-score|escalar|estandarizar)\b"
            r"|(\b(normalize|normalise|normalizar)\b.{0,24}\b(column|columns|feature|features|variable|variables|columna|columnas)\b)",
            re.IGNORECASE,
        )
        _action_negation_pattern = re.compile(
            r"(do\s+not|don't|must\s+not|never|avoid|forbid|forbidden|prohibido|no\s+debe|no\b|sin\b)",
            re.IGNORECASE,
        )

        def _flatten_text_payload(value: Any) -> str:
            if isinstance(value, str):
                return value
            if isinstance(value, list):
                return "\n".join(_flatten_text_payload(item) for item in value if item is not None)
            if isinstance(value, dict):
                parts: List[str] = []
                for _, item in value.items():
                    if item is None:
                        continue
                    parts.append(_flatten_text_payload(item))
                return "\n".join(part for part in parts if part)
            if value is None:
                return ""
            return str(value)

        def _to_clean_str_list(value: Any) -> Tuple[List[str], bool]:
            if value is None:
                return [], False
            if isinstance(value, str):
                cleaned = value.strip()
                return ([cleaned] if cleaned else []), False
            if not isinstance(value, list):
                return [], True
            values: List[str] = []
            had_invalid = False
            for item in value:
                if isinstance(item, str):
                    item_clean = item.strip()
                    if item_clean:
                        values.append(item_clean)
                    continue
                had_invalid = True
            return list(dict.fromkeys(values)), had_invalid

        def _has_non_negated_action(text: str, pattern: re.Pattern[str]) -> bool:
            if not isinstance(text, str) or not text.strip():
                return False
            for match in pattern.finditer(text):
                prefix = text[max(0, match.start() - 36):match.start()]
                if _action_negation_pattern.search(prefix):
                    continue
                return True
            return False

        def _extract_clean_dataset_transformations(
            clean_dataset: Dict[str, Any],
        ) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
            transform_block = clean_dataset.get("column_transformations")
            if transform_block is None:
                transform_block = {}
            if not isinstance(transform_block, dict):
                return [], [], [], [], ["artifact_requirements.clean_dataset.column_transformations must be an object"]

            def _collect(alias_keys: Tuple[str, ...]) -> Tuple[List[str], bool]:
                values: List[str] = []
                invalid = False
                for source in (transform_block, clean_dataset):
                    for key in alias_keys:
                        if key not in source:
                            continue
                        clean, has_invalid = _to_clean_str_list(source.get(key))
                        values.extend(clean)
                        invalid = invalid or has_invalid
                return list(dict.fromkeys(values)), invalid

            drop_columns, invalid_drop = _collect(
                ("drop_columns", "remove_columns", "columns_to_drop", "excluded_columns")
            )
            scale_columns, invalid_scale = _collect(
                ("scale_columns", "normalize_columns", "standardize_columns", "rescale_columns")
            )
            errors: List[str] = []
            if invalid_drop:
                errors.append("artifact_requirements.clean_dataset.column_transformations.drop_columns must be list[str]")
            if invalid_scale:
                errors.append("artifact_requirements.clean_dataset.column_transformations.scale_columns must be list[str]")
            transform_payload = dict(transform_block)
            if "drop_policy" not in transform_payload and "drop_policy" in clean_dataset:
                transform_payload["drop_policy"] = clean_dataset.get("drop_policy")
            selector_drop_reasons, selector_drop_errors = extract_selector_drop_reasons(transform_payload)
            for issue in selector_drop_errors:
                errors.append(f"artifact_requirements.clean_dataset.column_transformations.drop_policy: {issue}")

            criteria_drop_directives: List[str] = []
            feature_engineering = transform_block.get("feature_engineering")
            if isinstance(feature_engineering, list):
                for idx, item in enumerate(feature_engineering):
                    if not isinstance(item, dict):
                        continue
                    action = str(item.get("action") or "").strip().lower()
                    if action not in {"drop", "remove", "exclude"}:
                        continue
                    if any(item.get(key) for key in ("criteria", "condition", "rule", "rationale")):
                        criteria_drop_directives.append(f"feature_engineering[{idx}]")

            return drop_columns, scale_columns, selector_drop_reasons, criteria_drop_directives, errors

        def _extract_clean_dataset_selectors(clean_dataset: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
            raw_selectors = clean_dataset.get("required_feature_selectors")
            if raw_selectors is None:
                return [], []
            if not isinstance(raw_selectors, list):
                return [], ["artifact_requirements.clean_dataset.required_feature_selectors must be list[object]"]
            selectors: List[Dict[str, Any]] = []
            errors: List[str] = []
            for idx, item in enumerate(raw_selectors):
                if not isinstance(item, dict):
                    errors.append(
                        f"artifact_requirements.clean_dataset.required_feature_selectors[{idx}] must be an object"
                    )
                    continue
                # Auto-normalise LLM format variants (nested dict, string shorthand, etc.)
                item = _normalize_selector_entry(item)
                selector_type = str(item.get("type") or "").strip().lower()
                if not selector_type:
                    selector_type = _infer_selector_type_from_payload(item)
                    if selector_type:
                        item["type"] = selector_type
                if not selector_type:
                    errors.append(
                        f"artifact_requirements.clean_dataset.required_feature_selectors[{idx}] is missing type"
                    )
                    continue
                selector = dict(item)
                selector["type"] = selector_type
                selectors.append(selector)
            return selectors, errors

        def _extract_optional_passthrough_columns(clean_dataset: Dict[str, Any]) -> Tuple[List[str], List[str]]:
            optional_cols, invalid_optional = _to_clean_str_list(clean_dataset.get("optional_passthrough_columns"))
            errors: List[str] = []
            if invalid_optional:
                errors.append("artifact_requirements.clean_dataset.optional_passthrough_columns must be list[str]")
            return optional_cols, errors

        def _gate_list_valid(value: Any, gate_key: str) -> bool:
            if not isinstance(value, list) or not value:
                return False
            if gate_key == "qa_gates":
                return bool(get_qa_gates({"qa_gates": value}))
            if gate_key == "reviewer_gates":
                return bool(get_reviewer_gates({"reviewer_gates": value}))
            if gate_key == "cleaning_gates":
                return bool(get_cleaning_gates({"cleaning_gates": value}))
            for gate in value:
                if isinstance(gate, str) and gate.strip():
                    continue
                if isinstance(gate, dict):
                    for key in ("name", "id", "gate", "metric", "check", "rule", "title", "label"):
                        field_value = gate.get(key)
                        if isinstance(field_value, str) and field_value.strip():
                            break
                    else:
                        return False
                    continue
                return False
            return True

        def _build_contract_diagnostics(contract: Dict[str, Any], where: str, llm_success: bool) -> Dict[str, Any]:
            diagnostics: Dict[str, Any] = {
                "schema_version": 1,
                "policy": "llm_contract_immutable_post_generation",
                "interface_profile": "minimal_scope_v1",
                "where": where,
                "llm_success": bool(llm_success),
                "legacy_keys": [],
                "unknown_top_level_keys": [],
                "validation": {},
            }
            if not isinstance(contract, dict):
                diagnostics["validation"] = {
                    "status": "error",
                    "issues": [{"severity": "error", "rule": "contract_not_object", "message": "Contract is not a dictionary"}],
                }
                return diagnostics

            try:
                validation_result = _validate_contract_quality(copy.deepcopy(contract))
            except Exception as err:
                validation_result = {
                    "status": "error",
                    "accepted": False,
                    "issues": [
                        {
                            "severity": "error",
                            "rule": "contract_validation_exception",
                            "message": str(err),
                        }
                    ],
                    "summary": {"error_count": 1, "warning_count": 0},
                }

            status = str(validation_result.get("status") or "unknown").lower()
            issues = validation_result.get("issues") if isinstance(validation_result, dict) else []
            accepted = _contract_is_accepted(validation_result if isinstance(validation_result, dict) else None)

            if status == "error":
                print(f"CONTRACT_VALIDATION_ERROR: {len(issues) if isinstance(issues, list) else 0} issues found")
                if isinstance(issues, list):
                    for issue in issues:
                        if isinstance(issue, dict):
                            print(f"  - [{issue.get('severity')}] {issue.get('rule')}: {issue.get('message')}")
            elif status == "warning":
                print(f"CONTRACT_VALIDATION_WARNING: {len(issues) if isinstance(issues, list) else 0} issues found")
                if isinstance(issues, list):
                    for issue in issues:
                        if isinstance(issue, dict):
                            print(f"  - [{issue.get('severity')}] {issue.get('rule')}: {issue.get('message')}")

            diagnostics["validation"] = validation_result if isinstance(validation_result, dict) else {}
            error_rules: List[str] = []
            warning_rules: List[str] = []
            if isinstance(issues, list):
                for issue in issues:
                    if not isinstance(issue, dict):
                        continue
                    rule = str(issue.get("rule") or "").strip()
                    if not rule:
                        continue
                    severity = str(issue.get("severity") or "").lower()
                    if severity in {"error", "fail"}:
                        if rule not in error_rules:
                            error_rules.append(rule)
                    elif severity == "warning":
                        if rule not in warning_rules:
                            warning_rules.append(rule)
            diagnostics["quality_profile"] = {
                "top_error_rules": error_rules[:8],
                "top_warning_rules": warning_rules[:8],
            }
            diagnostics["summary"] = {
                "status": status,
                "issue_count": len(issues) if isinstance(issues, list) else 0,
                "accepted": accepted,
            }
            return diagnostics

        def _finalize_and_persist(contract, where, llm_success: bool):
            if isinstance(contract, dict) and contract:
                contract = _apply_planner_structural_support(contract)

                def _llm_minimal_repair_provider(
                    current_contract: Dict[str, Any],
                    validation_result: Dict[str, Any],
                    hints: List[str],
                    attempt: int,
                ) -> Any:
                    if not self.client:
                        return None
                    try:
                        model_client = self._build_model_client(self.model_name)
                    except Exception:
                        return None
                    if model_client is None:
                        return None

                    hints_lines = "\n".join(f"- {hint}" for hint in (hints or [])[:5]) or "- apply minimal fixes"
                    validation_feedback = _compact_validation_feedback(validation_result, max_issues=6)
                    contract_compact = _compress_text_preserve_ends(
                        json.dumps(current_contract, ensure_ascii=False, indent=2),
                        max_chars=12000,
                        head=8000,
                        tail=4000,
                    )
                    repair_prompt = (
                        "Repair this execution contract with MINIMAL edits.\n"
                        "Return ONLY valid JSON.\n"
                        "Preferred shape: {\"changes\": {...}} (minimal nested fields only).\n"
                        "Alternative shape: {\"patch\": [{\"op\":\"replace\",\"path\":\"/a/b\",\"value\":...}]}\n"
                        "Do NOT regenerate the full contract.\n"
                        "Keep required_outputs as list[str].\n"
                        "Preserve required_output_artifacts and spec_extraction.deliverables if present.\n"
                        + "Attempt "
                        + str(attempt)
                        + "/2.\n\n"
                        + "Top repair hints:\n"
                        + hints_lines
                        + "\n\n"
                        + "Validation issues:\n"
                        + validation_feedback
                        + "\n\n"
                        + "Current contract:\n"
                        + contract_compact
                    )
                    try:
                        response, _ = self._generate_content_with_budget(
                            model_client,
                            repair_prompt,
                            output_token_floor=768,
                            model_name=self.model_name,
                            tool_mode="patch",
                        )
                    except Exception:
                        return None

                    response_text = self._extract_openai_response_text(response)
                    if not response_text.strip():
                        return None
                    parsed, _ = _parse_json_response(response_text)
                    if not isinstance(parsed, dict):
                        return None
                    if isinstance(parsed.get("patch"), list):
                        return {"patch": parsed.get("patch")}
                    if isinstance(parsed.get("changes"), dict):
                        return {"changes": parsed.get("changes")}
                    if isinstance(parsed.get("judgment_patch"), dict):
                        return {"changes": parsed.get("judgment_patch")}
                    return {"changes": parsed}

                contract, post_validation, post_repair_trace = _validate_repair_revalidate_loop(
                    contract=contract,
                    validator_fn=lambda payload: _validate_contract_quality(payload),
                    repair_provider=_llm_minimal_repair_provider,
                    max_iterations=2,
                )
                if isinstance(post_repair_trace, list) and post_repair_trace:
                    if len(post_repair_trace) > 1:
                        print(
                            "POST_FINAL_REPAIR_LOOP: "
                            + ", ".join(
                                [
                                    f"attempt={row.get('attempt')} accepted={row.get('accepted')}"
                                    for row in post_repair_trace
                                    if isinstance(row, dict)
                                ]
                            )
                        )
                    if isinstance(post_validation, dict) and _validation_result_accepted(post_validation):
                        llm_success = True if self.client else llm_success
            diagnostics = _build_contract_diagnostics(contract if isinstance(contract, dict) else {}, where, llm_success)
            self.last_contract_diagnostics = diagnostics
            _persist_contracts(
                contract if isinstance(contract, dict) else {},
                diagnostics,
                invalid_contract=planner_candidate_invalid,
                invalid_raw=planner_candidate_invalid_raw,
                invalid_meta=planner_candidate_invalid_meta,
            )
            return contract

        target_candidates: List[Dict[str, Any]] = []
        if not self.client:
            planner_candidate_invalid_meta = {
                "reason": "planner_client_unavailable",
                "llm_required": True,
            }
            self.last_contract_min = None
            return _finalize_and_persist({}, where="execution_planner:no_client", llm_success=False)

        target_candidates: List[Dict[str, Any]] = []
        resolved_target = None
        potential_constant_anchor_avoidance: List[str] = []
        potential_constant_anchor_avoidance_source = "none"
        potential_constant_anchor_avoidance_confidence = "unknown"
        if isinstance(data_profile, dict):
            try:
                from src.utils.data_profile_compact import compact_data_profile_for_llm
                compact = compact_data_profile_for_llm(data_profile, contract=None)
                target_candidates = compact.get("target_candidates") if isinstance(compact, dict) else []
            except Exception:
                target_candidates = []
            # Context guardrail for planner consistency:
            # Prefer non-constant anchors when selector-drop policy may prune feature families.
            const_exact = data_profile.get("constant_columns")
            if isinstance(const_exact, list) and const_exact:
                potential_constant_anchor_avoidance = [str(c) for c in const_exact if c][:200]
                potential_constant_anchor_avoidance_source = "constant_columns"
                potential_constant_anchor_avoidance_confidence = str(
                    data_profile.get("constant_columns_confidence") or "high_full_or_complete"
                )
            else:
                const_sample = data_profile.get("constant_columns_sample")
                if isinstance(const_sample, list) and const_sample:
                    potential_constant_anchor_avoidance = [str(c) for c in const_sample if c][:200]
                    potential_constant_anchor_avoidance_source = "constant_columns_sample"
                    potential_constant_anchor_avoidance_confidence = str(
                        data_profile.get("constant_columns_confidence") or "low_sampled"
                    )
        if isinstance(target_candidates, list) and target_candidates:
            for item in target_candidates:
                if not isinstance(item, dict):
                    continue
                col = item.get("column") or item.get("name") or item.get("candidate")
                if not col:
                    continue
                resolved = _resolve_exact_header(col) or col
                if resolved:
                    resolved_target = resolved
                    break
        strategy_json = json.dumps(strategy, indent=2)
        column_sets_payload = column_sets if isinstance(column_sets, dict) else {}
        column_sets_summary = summarize_column_sets(column_sets_payload) if column_sets_payload else ""
        column_inventory_count = len(column_inventory or [])
        manifest_payload = column_manifest if isinstance(column_manifest, dict) else {}
        if not manifest_payload and column_inventory_count > 200:
            try:
                manifest_payload = build_column_manifest(
                    column_inventory or [],
                    column_sets=column_sets_payload,
                    roles={},
                )
            except Exception:
                manifest_payload = {}
        manifest_mode = str(manifest_payload.get("schema_mode") or "")
        manifest_families = manifest_payload.get("families") if isinstance(manifest_payload.get("families"), list) else []
        manifest_anchors = manifest_payload.get("anchors") if isinstance(manifest_payload.get("anchors"), list) else []
        manifest_for_prompt: Dict[str, Any] = {}
        if isinstance(manifest_payload, dict) and manifest_payload:
            manifest_for_prompt = dict(manifest_payload)
            if len(manifest_anchors) > 60:
                manifest_for_prompt["anchors"] = manifest_anchors[:60]
                manifest_for_prompt["anchors_truncated"] = True
                manifest_for_prompt["anchors_total_count"] = len(manifest_anchors)
            if len(manifest_families) > 12:
                manifest_for_prompt["families"] = manifest_families[:12]
                manifest_for_prompt["families_truncated"] = True
                manifest_for_prompt["families_total_count"] = len(manifest_families)
        column_inventory_sample = (column_inventory or [])[:25]
        inventory_truncated = column_inventory_count > 50
        if manifest_mode == "wide" and manifest_for_prompt:
            column_inventory_payload = {
                "mode": "manifest_reference",
                "total_columns": manifest_for_prompt.get("total_columns", column_inventory_count),
                "anchors": manifest_for_prompt.get("anchors", []),
                "families": manifest_for_prompt.get("families", []),
                "instruction": (
                    "Use anchors as explicit required columns and required_feature_selectors for dense families. "
                    "Do not enumerate all family members."
                ),
            }
        else:
            column_inventory_payload = column_inventory_sample if inventory_truncated else (column_inventory or [])
        column_inventory_compact = compact_column_representation(column_inventory or [], max_display=40)
        strategy_feature_families = []
        if isinstance(strategy, dict):
            families_candidate = strategy.get("feature_families")
            if isinstance(families_candidate, list):
                strategy_feature_families = families_candidate
        data_summary_for_prompt = _compress_text_preserve_ends(
            data_summary_str,
            max_chars=12000,
            head=8000,
            tail=4000,
        )
        critique_for_prompt = _compress_text_preserve_ends(
            domain_expert_critique or "",
            max_chars=2000,
            head=1300,
            tail=700,
        )
        user_input = f"""
strategy:
{strategy_json}

business_objective:
{business_objective}

relevant_columns:
{json.dumps(relevant_columns, indent=2)}

relevant_columns_total_count:
{relevant_columns_total_count}

relevant_columns_truncated:
{json.dumps(relevant_columns_truncated)}

relevant_columns_omitted_count:
{relevant_columns_omitted_count}

relevant_columns_compact:
{json.dumps(relevant_columns_compact, indent=2)}

relevant_sources:
{json.dumps(relevant_sources, indent=2)}

omitted_columns_policy:
{omitted_columns_policy}

column_inventory_count:
{column_inventory_count}

column_inventory_sample:
{json.dumps(column_inventory_sample, indent=2)}

column_inventory_truncated:
{json.dumps(inventory_truncated)}

column_inventory:
{json.dumps(column_inventory_payload, indent=2)}

column_inventory_compact:
{json.dumps(column_inventory_compact, indent=2)}

column_manifest:
{json.dumps(manifest_for_prompt, indent=2)}

column_manifest_mode:
{json.dumps(manifest_mode or "none")}

column_manifest_family_count:
{len(manifest_families)}

column_sets:
{json.dumps(column_sets_payload, indent=2)}

column_sets_summary:
{column_sets_summary}

strategy_feature_families:
{json.dumps(strategy_feature_families, indent=2)}

strategy_feature_family_hints:
{json.dumps(strategy_feature_family_hints, indent=2)}

strategy_feature_family_expanded_count:
{strategy_feature_family_expanded_count}

data_profile_summary:
{data_summary_for_prompt}

target_candidates:
{json.dumps(target_candidates, indent=2)}

resolved_target:
{json.dumps(resolved_target)}

column_inventory_path:
{json.dumps("data/column_inventory.json")}

column_sets_path:
{json.dumps("data/column_sets.json")}

required_columns_path:
{json.dumps("data/required_columns.json")}

output_dialect:
{json.dumps(output_dialect or "unknown")}

env_constraints:
{json.dumps(env_constraints or {"forbid_inplace_column_creation": True})}

contract_source_of_truth_policy:
{json.dumps(CONTRACT_SOURCE_OF_TRUTH_POLICY_V1, indent=2)}

downstream_consumer_interface:
{json.dumps(DOWNSTREAM_CONSUMER_INTERFACE_V1, indent=2)}

evidence_policy:
{json.dumps(CONTRACT_EVIDENCE_POLICY_V1, indent=2)}

planner_semantic_resolution_policy:
{json.dumps(PLANNER_SEMANTIC_RESOLUTION_POLICY_V1, indent=2)}

phased_contract_compilation_protocol:
{json.dumps(PHASED_CONTRACT_COMPILATION_PROTOCOL_V1, indent=2)}

contract_consistency_guardrails:
{json.dumps({
    "required_columns_must_be_non_droppable": True,
    "if_selector_drop_policy_active_required_and_passthrough_must_not_overlap_selectors": True,
    "if_selector_drop_policy_active_hard_gates_must_not_depend_on_selector_covered_columns": True,
}, indent=2)}

potential_constant_anchor_avoidance:
{json.dumps(potential_constant_anchor_avoidance, indent=2)}

potential_constant_anchor_avoidance_source:
{json.dumps(potential_constant_anchor_avoidance_source)}

potential_constant_anchor_avoidance_confidence:
{json.dumps(potential_constant_anchor_avoidance_confidence)}

domain_expert_critique:
{critique_for_prompt or "None"}
"""

        full_prompt = (
            MINIMAL_CONTRACT_COMPILER_PROMPT
            + "\n\nSCHEMA REGISTRY EXAMPLES:\n"
            + CONTRACT_SCHEMA_EXAMPLES_TEXT
            + "\n\nINPUTS:\n"
            + user_input
        )
        model_chain = [m for m in (self.model_chain or [self.model_name]) if m]

        contract: Dict[str, Any] | None = None
        llm_success = False
        best_candidate: Dict[str, Any] | None = None
        best_validation: Dict[str, Any] | None = None
        best_response_text: str | None = None
        best_parse_feedback: str | None = None
        best_error_count: Optional[int] = None
        best_warning_count: Optional[int] = None
        latest_candidate_for_repair: Dict[str, Any] | None = None
        latest_validation_for_repair: Dict[str, Any] | None = None
        latest_response_text_for_repair: str | None = None
        latest_parse_feedback_for_repair: str | None = None

        max_quality_rounds = 3
        try:
            max_quality_rounds = max(1, int(os.getenv("EXECUTION_PLANNER_QUALITY_ROUNDS", "3")))
        except Exception:
            max_quality_rounds = 3

        current_prompt = full_prompt
        current_prompt_name = "prompt_attempt_1.txt"
        attempt_counter = 0

        if contract is None:
            for quality_round in range(1, max_quality_rounds + 1):
                round_has_candidate = False
                for model_idx, model_name in enumerate(model_chain, start=1):
                    attempt_counter += 1
                    self.last_prompt = current_prompt
                    response_text = ""
                    response = None
                    parse_error: Optional[Exception] = None
                    finish_reason = None
                    usage_metadata = None
                    generation_config_used = None
                    validation_result: Dict[str, Any] | None = None
                    quality_accepted = False

                    if len(model_chain) > 1:
                        response_name = f"response_attempt_{quality_round}_m{model_idx}.txt"
                    else:
                        response_name = f"response_attempt_{quality_round}.txt"

                    try:
                        model_client = self._build_model_client(model_name)
                        if model_client is None:
                            parse_error = ValueError(f"Planner client unavailable for model {model_name}")
                        else:
                            response, generation_config_used = self._generate_content_with_budget(
                                model_client,
                                current_prompt,
                                output_token_floor=3072 * quality_round,
                                model_name=model_name,
                                tool_mode="contract",
                            )
                            response_text = self._extract_openai_response_text(response)
                            self.last_response = response_text
                    except Exception as err:
                        parse_error = err

                    if response is not None:
                        finish_reason = self._extract_openai_finish_reason(response)
                        usage_metadata = _normalize_usage_metadata(
                            getattr(response, "usage", None) or getattr(response, "usage_metadata", None)
                        )

                    _persist_attempt(current_prompt_name, response_name, current_prompt, response_text)

                    parsed, parse_exc = _parse_json_response(response_text) if response_text else (None, parse_error)
                    if parse_exc:
                        parse_error = parse_exc

                    had_json_parse_error = parsed is None or not isinstance(parsed, dict)
                    if parsed is not None and not isinstance(parsed, dict):
                        parse_error = ValueError("Parsed JSON is not an object")

                    quality_error_message = None
                    if parsed is not None and isinstance(parsed, dict):
                        round_has_candidate = True
                        candidate_for_validation = _apply_planner_structural_support(parsed)
                        try:
                            validation_result = _validate_contract_quality(copy.deepcopy(candidate_for_validation))
                        except Exception as val_err:
                            validation_result = {
                                "status": "error",
                                "accepted": False,
                                "issues": [
                                    {
                                        "severity": "error",
                                        "rule": "contract_validation_exception",
                                        "message": str(val_err),
                                    }
                                ],
                                "summary": {"error_count": 1, "warning_count": 0},
                            }
                        latest_candidate_for_repair = candidate_for_validation
                        latest_validation_for_repair = validation_result if isinstance(validation_result, dict) else None
                        latest_response_text_for_repair = response_text
                        latest_parse_feedback_for_repair = None
                        quality_accepted = _contract_is_accepted(validation_result)
                        if not quality_accepted:
                            primary_rule = None
                            issues = validation_result.get("issues") if isinstance(validation_result, dict) else None
                            if isinstance(issues, list):
                                for issue in issues:
                                    if isinstance(issue, dict) and issue.get("rule"):
                                        primary_rule = str(issue.get("rule"))
                                        break
                            quality_error_message = (
                                f"contract_quality_failed:{primary_rule}"
                                if primary_rule
                                else "contract_quality_failed"
                            )
                            summary = validation_result.get("summary") if isinstance(validation_result, dict) else {}
                            error_count = (
                                int(summary.get("error_count", 0))
                                if isinstance(summary, dict)
                                else 0
                            )
                            warning_count = (
                                int(summary.get("warning_count", 0))
                                if isinstance(summary, dict)
                                else 0
                            )
                            if (
                                best_error_count is None
                                or error_count < best_error_count
                                or (
                                    error_count == best_error_count
                                    and (best_warning_count is None or warning_count < best_warning_count)
                                )
                            ):
                                best_candidate = candidate_for_validation
                                best_validation = validation_result
                                best_response_text = response_text
                                best_error_count = error_count
                                best_warning_count = warning_count
                                best_parse_feedback = None
                    else:
                        parse_feedback = _build_parse_feedback(response_text, parse_error)
                        latest_response_text_for_repair = response_text
                        latest_parse_feedback_for_repair = parse_feedback
                        if best_response_text is None:
                            best_response_text = response_text
                            best_parse_feedback = parse_feedback

                    planner_diag.append(
                        {
                            "model_name": model_name,
                            "attempt_index": attempt_counter,
                            "quality_round": quality_round,
                            "prompt_char_len": len(current_prompt or ""),
                            "response_char_len": len(response_text or ""),
                            "finish_reason": str(finish_reason) if finish_reason is not None else None,
                            "generation_config": generation_config_used,
                            "usage_metadata": usage_metadata,
                            "had_json_parse_error": bool(had_json_parse_error),
                            "parse_error_type": type(parse_error).__name__ if parse_error else None,
                            "parse_error_message": str(parse_error) if parse_error else None,
                            "quality_status": (
                                str(validation_result.get("status") or "").lower()
                                if isinstance(validation_result, dict)
                                else None
                            ),
                            "quality_issue_rules": (
                                [
                                    str(issue.get("rule"))
                                    for issue in (validation_result.get("issues") or [])
                                    if isinstance(issue, dict) and issue.get("rule")
                                ][:12]
                                if isinstance(validation_result, dict)
                                else []
                            ),
                            "quality_error_count": (
                                int((validation_result.get("summary") or {}).get("error_count", 0))
                                if isinstance(validation_result, dict)
                                else None
                            ),
                            "quality_warning_count": (
                                int((validation_result.get("summary") or {}).get("warning_count", 0))
                                if isinstance(validation_result, dict)
                                else None
                            ),
                            "quality_accepted": quality_accepted,
                            "quality_error": quality_error_message,
                        }
                    )

                    if parsed is None or not isinstance(parsed, dict):
                        print(f"WARNING: Planner parse failed on attempt {attempt_counter} (model={model_name}).")
                        continue
                    if not quality_accepted:
                        print(f"WARNING: Planner contract rejected by quality gate on attempt {attempt_counter} (model={model_name}).")
                        continue

                    contract = _apply_planner_structural_support(parsed)
                    llm_success = True
                    break

                if contract is not None:
                    break
                if quality_round >= max_quality_rounds:
                    break

                current_prompt_name = f"prompt_attempt_{quality_round + 1}_repair.txt"
                repair_contract = (
                    latest_candidate_for_repair
                    if isinstance(latest_candidate_for_repair, dict)
                    else best_candidate
                )
                repair_validation = (
                    latest_validation_for_repair
                    if isinstance(latest_validation_for_repair, dict)
                    else best_validation
                )
                repair_response_text = (
                    latest_response_text_for_repair
                    if isinstance(latest_response_text_for_repair, str)
                    else best_response_text
                )
                repair_parse_feedback = (
                    latest_parse_feedback_for_repair
                    if isinstance(latest_parse_feedback_for_repair, str)
                    else best_parse_feedback
                )
                current_prompt = _build_quality_repair_prompt(
                    previous_contract=repair_contract,
                    previous_validation=repair_validation,
                    previous_response_text=repair_response_text,
                    previous_parse_feedback=repair_parse_feedback,
                    original_inputs_text=user_input,
                )
                if not round_has_candidate:
                    current_prompt = (
                        current_prompt
                        + "\n\nPrevious attempts failed JSON parsing. Repair by returning syntactically valid JSON only."
                    )

        if contract is None:
            invalid_contract = (
                latest_candidate_for_repair
                if isinstance(latest_candidate_for_repair, dict) and latest_candidate_for_repair
                else (best_candidate if isinstance(best_candidate, dict) and best_candidate else None)
            )
            invalid_raw = (
                latest_response_text_for_repair
                if isinstance(latest_response_text_for_repair, str) and latest_response_text_for_repair.strip()
                else (best_response_text if isinstance(best_response_text, str) and best_response_text.strip() else None)
            )
            planner_candidate_invalid = invalid_contract
            planner_candidate_invalid_raw = invalid_raw
            planner_candidate_invalid_meta = {
                "best_validation": best_validation if isinstance(best_validation, dict) else None,
                "latest_validation": (
                    latest_validation_for_repair if isinstance(latest_validation_for_repair, dict) else None
                ),
                "best_parse_feedback": best_parse_feedback,
                "latest_parse_feedback": latest_parse_feedback_for_repair,
                "attempt_count": len(planner_diag),
                "fallback_mode": "llm_candidates_only",
            }
            contract = _apply_planner_structural_support(
                invalid_contract if isinstance(invalid_contract, dict) else {}
            )
            llm_success = False

        if llm_success and resolved_target:
            print(
                f"INFO: Planner resolved target candidate '{resolved_target}' from profile context; "
                "contract remains immutable (diagnostic only)."
            )
        self.last_contract_min = None

        if llm_success and planner_diag:
            last_attempt = planner_diag[-1]
            if isinstance(last_attempt, dict) and last_attempt.get("attempt_index") is not None:
                print(f"SUCCESS: Execution Planner succeeded on attempt {last_attempt.get('attempt_index')}")
            else:
                print("SUCCESS: Execution Planner succeeded.")
        where = "execution_planner:final_contract" if llm_success else "execution_planner:quality_gate_failed"
        return _finalize_and_persist(contract, where=where, llm_success=llm_success)

    def generate_evaluation_spec(
        self,
        strategy: Dict[str, Any],
        contract: Dict[str, Any],
        data_summary: str = "",
        business_objective: str = "",
        column_inventory: list[str] | None = None,
    ) -> Dict[str, Any]:
        """
        Generate evaluation spec by extracting ONLY from V4.1 contract fields.
        NO legacy fields (data_requirements, spec_extraction) allowed.
        """
        if not isinstance(contract, dict):
            return {
                "confidence": 0.1,
                "qa_gates": [],
                "reviewer_gates": [],
                "artifact_requirements": {},
                "notes": ["Invalid contract structure"],
                "source": "error_fallback",
                "unknowns": ["Contract is not a valid dictionary"]
            }
        
        # Extract ONLY from V4.1 fields
        qa_gates = contract.get("qa_gates", [])
        cleaning_gates = contract.get("cleaning_gates", [])
        reviewer_gates = contract.get("reviewer_gates", [])
        artifact_requirements = contract.get("artifact_requirements", {})
        validation_requirements = contract.get("validation_requirements", {})
        leakage_execution_plan = contract.get("leakage_execution_plan", {})
        allowed_feature_sets = contract.get("allowed_feature_sets", {})
        canonical_columns = contract.get("canonical_columns", [])
        derived_columns = _extract_derived_column_names(contract.get("derived_columns"))
        required_outputs = contract.get("required_outputs", [])
        data_limited_mode = contract.get("data_limited_mode", {})
        
        # Build evaluation spec from V4.1 contract
        spec = {
            "qa_gates": qa_gates if isinstance(qa_gates, list) else [],
            "cleaning_gates": cleaning_gates if isinstance(cleaning_gates, list) else [],
            "reviewer_gates": reviewer_gates if isinstance(reviewer_gates, list) else [],
            "artifact_requirements": artifact_requirements if isinstance(artifact_requirements, dict) else {},
            "validation_requirements": validation_requirements if isinstance(validation_requirements, dict) else {},
            "leakage_execution_plan": leakage_execution_plan if isinstance(leakage_execution_plan, dict) else {},
            "allowed_feature_sets": allowed_feature_sets if isinstance(allowed_feature_sets, dict) else {},
            "canonical_columns": canonical_columns if isinstance(canonical_columns, list) else [],
            "derived_columns": derived_columns,
            "required_outputs": required_outputs if isinstance(required_outputs, list) else [],
            "data_limited_mode": data_limited_mode if isinstance(data_limited_mode, dict) else {},
            "confidence": 0.9,
            "source": "contract_driven_v41",
            "notes": ["Extracted directly from V4.1 contract fields"]
        }
        
        # Add unknowns if critical fields are missing
        unknowns = []
        if not qa_gates:
            unknowns.append("qa_gates missing from contract")
        if not cleaning_gates:
            unknowns.append("cleaning_gates missing from contract")
        if not reviewer_gates:
            unknowns.append("reviewer_gates missing from contract")
        if not required_outputs:
            unknowns.append("required_outputs missing from contract")
        
        if unknowns:
            spec["unknowns"] = unknowns
            spec["confidence"] = 0.6  # Lower confidence if fields missing
        
        return spec
