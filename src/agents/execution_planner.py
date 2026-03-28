import json
import os
import ast
import copy
import hashlib
import math
from typing import Dict, Any, List, Optional, Tuple, Callable
import re
import difflib

from dotenv import load_dotenv
from openai import OpenAI
from src.utils.contract_validation import (
    DEFAULT_DATA_ENGINEER_RUNBOOK,
    DEFAULT_ML_ENGINEER_RUNBOOK,
)
from src.utils.contract_accessors import (
    get_clean_dataset_output_path,
    get_dataset_artifact_binding,
    get_declared_artifact_path_by_intent,
    get_enriched_dataset_output_path,
    get_clean_manifest_path,
    get_cleaning_gates,
    get_column_roles,
    get_declared_artifact_path,
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
    resolve_contract_active_workstreams,
    derive_contract_scope_from_workstreams,
    is_probably_path,
    _normalize_selector_entry,
    get_default_optimization_policy,
    normalize_optimization_policy,
    normalize_optimization_direction,
    normalize_optimization_tie_breakers,
)
from src.utils.contract_schema_registry import (
    build_contract_schema_examples_text,
    get_contract_schema_repair_action,
    apply_contract_schema_registry_repairs,
)
from src.utils.contract_response_schema import (
    EXECUTION_CONTRACT_CANONICAL_REQUIRED_KEYS,
    EXECUTION_CONTRACT_V5_CANONICAL_REQUIRED_KEYS,
    EXECUTION_SEMANTIC_CORE_REQUIRED_KEYS,
    V5_AGENT_SECTION_KEYS,
)
from src.utils.problem_capabilities import (
    infer_problem_capabilities,
    is_problem_family,
    resolve_problem_capabilities_from_contract,
)

load_dotenv()


_API_KEY_SENTINEL = object()


_QA_SEVERITIES = {"HARD", "SOFT"}
_CLEANING_SEVERITIES = {"HARD", "SOFT"}
_DEFAULT_EXECUTION_PLANNER_COMPILER_MODEL = "google/gemini-3-flash-preview"

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
- visualizations: Reason about which diagnostic plots best represent this strategy's results. Assign EDA plots to data_engineer_runbook and model result plots to ml_engineer_runbook. Add "static/plots/*.png" to required_outputs.
Base your decisions on the MEANING of the business objective and strategy, not on the
presence or absence of specific keywords. Set boolean flags and structured specs in the
contract based on your semantic understanding of what the downstream agents will need.
"""

CONTRACT_SCHEMA_EXAMPLES_TEXT = build_contract_schema_examples_text()
COMPILER_OPERATIONAL_SCHEMA_EXAMPLES_TEXT = """
These are reference examples showing expected shapes and conventions for contract fields.
Use them as structural guidance, not as templates to fill blindly. Derive actual values from context.

CONTRACT VERSION 5.0 — HIERARCHICAL STRUCTURE

The contract is organized by agent hierarchy: shared context + per-agent sections.
Each downstream agent receives: shared + its own section (merged automatically).
This means: put fields needed by 2+ agents in "shared", and agent-exclusive fields in the agent's section.

FULL EXAMPLE (full_pipeline with model_training=true):

{
  "contract_version": "5.0",
  "shared": {
    "scope": "full_pipeline",
    "strategy_title": "Binary classification with gradient boosting",
    "business_objective": "Predict customer churn with high recall",
    "output_dialect": {"sep": ",", "decimal": ".", "encoding": "utf-8"},
    "canonical_columns": ["id", "feature_a", "feature_b", "target", "created_at"],
    "column_roles": {
      "pre_decision": ["feature_a", "feature_b"],
      "decision": [],
      "outcome": ["target"],
      "post_decision_audit_only": ["created_at"],
      "unknown": [],
      "identifiers": ["id"],
      "time_columns": []
    },
    "allowed_feature_sets": {
      "segmentation_features": [],
      "model_features": ["feature_a", "feature_b"],
      "forbidden_features": [],
      "audit_only_features": ["created_at"]
    },
    "task_semantics": {
      "problem_family": "binary_classification",
      "objective_type": "binary_classification",
      "primary_target": "target",
      "target_columns": ["target"],
      "prediction_unit": "customer"
    },
    "active_workstreams": {"cleaning": true, "feature_engineering": true, "model_training": true},
    "model_features": ["feature_a", "feature_b"],
    "column_dtype_targets": {
      "id": {"target_dtype": "int64", "nullable": false, "role": "identifiers"},
      "feature_a": {"target_dtype": "float64", "nullable": true, "role": "pre_decision"},
      "target": {"target_dtype": "int64", "nullable": false, "role": "outcome"}
    },
    "iteration_policy": {"max_iterations": 6, "metric_improvement_max": 4, "runtime_fix_max": 3, "compliance_bootstrap_max": 2}
  },
  "data_engineer": {
    "required_outputs": [
      {"intent": "cleaned_dataset", "path": "artifacts/clean/dataset_cleaned.csv", "required": true, "kind": "dataset"},
      {"intent": "cleaning_manifest", "path": "artifacts/clean/cleaning_manifest.json", "required": true, "kind": "manifest"},
      {"intent": "eda_plots", "path": "static/plots/*.png", "required": false, "kind": "visualization"}
    ],
    "cleaning_gates": [
      {"name": "verify_no_leakage", "severity": "HARD", "params": {"forbidden_columns": ["created_at"]}},
      {"name": "preferred_null_rate", "severity": "SOFT", "params": {"max_null_pct": 0.05}}
    ],
    "runbook": {
      "objectives": ["Clean and prepare dataset for ML training"],
      "constraints": ["Preserve all model_features and identifiers", "Do not drop outcome column"]
    },
    "artifact_requirements": {
      "cleaned_dataset": {
        "output_path": "artifacts/clean/dataset_cleaned.csv",
        "output_manifest_path": "artifacts/clean/cleaning_manifest.json",
        "required_columns": ["id", "feature_a", "feature_b", "target"]
      }
    }
  },
  "ml_engineer": {
    "required_outputs": [
      {"intent": "cv_metrics", "path": "artifacts/ml/cv_metrics.json", "required": true, "kind": "metrics"},
      {"intent": "predictions", "path": "artifacts/ml/predictions.csv", "required": true, "kind": "predictions"},
      {"intent": "model_plots", "path": "static/plots/*.png", "required": false, "kind": "visualization"}
    ],
    "qa_gates": [
      {"name": "metric_above_baseline", "severity": "HARD", "params": {"metric": "roc_auc", "min_value": 0.5}},
      {"name": "no_target_leakage", "severity": "HARD", "params": {}}
    ],
    "reviewer_gates": [
      {"name": "no_test_leakage", "severity": "HARD", "params": {}},
      {"name": "reproducible_seed", "severity": "SOFT", "params": {}}
    ],
    "runbook": {
      "objectives": ["Train competitive gradient boosting model optimizing for the primary metric"],
      "constraints": ["Use cross-validation for evaluation", "No data leakage"]
    },
    "evaluation_spec": {
      "objective_type": "binary_classification",
      "primary_target": "target",
      "primary_metric": "log_loss",
      "metric_definition_rule": "Binary cross-entropy averaged across all samples using predicted probabilities.",
      "label_columns": ["target"]
    },
    "validation_requirements": {
      "method": "cross_validation",
      "primary_metric": "log_loss",
      "metrics_to_report": ["log_loss", "roc_auc", "accuracy"],
      "params": {"n_splits": 5, "stratify": true}
    },
  "optimization_policy": {
      "enabled": true, "max_rounds": 4, "quick_eval_folds": 3, "full_eval_folds": 5,
      "min_delta": 0.001, "patience": 2,
      "optimization_direction": "minimize",
      "tie_breakers": [{"field": "cv_std", "direction": "minimize", "reason": "Prefer the more stable model when primary metric gains are similar."}],
      "allow_model_switch": false, "allow_ensemble": false, "allow_hpo": true,
      "allow_feature_engineering": true, "allow_calibration": false
    },
    "artifact_requirements": {
      "model_path": "artifacts/ml/model.pkl"
    }
  },
  "cleaning_reviewer": {
    "focus_areas": ["column preservation", "dtype consistency"]
  },
  "qa_reviewer": {
    "review_subject": "ml_engineer",
    "artifacts_to_verify": ["artifacts/ml/cv_metrics.json", "artifacts/ml/predictions.csv"]
  },
  "business_translator": {
    "reporting_policy": {"audience": "executive", "format": "pdf"},
    "visual_requirements": {"include_feature_importance": true, "include_metric_summary": true}
  }
}

--- FIELD REFERENCE ---

SHARED section (fields needed by 2+ agents):
  scope: "cleaning_only" | "ml_only" | "full_pipeline"
  strategy_title, business_objective: from semantic_core
  output_dialect: {"sep", "decimal", "encoding"}
  canonical_columns: all column names in the dataset
  column_roles: pre_decision, decision, outcome, post_decision_audit_only, unknown, identifiers, time_columns
    - outcome: ONLY target variable(s). Not ordinary features.
    - pre_decision: ALL model input features available before prediction.
    - decision: model/policy outputs (often empty at contract time).
  allowed_feature_sets: segmentation_features, model_features, forbidden_features, audit_only_features
  model_features: explicit column names the ML engineer should use as model inputs
  task_semantics: problem_family, objective_type, primary_target, target_columns, prediction_unit
  active_workstreams: cleaning, feature_engineering, model_training (booleans)
  column_dtype_targets: {col_name: {"target_dtype": str, "nullable": bool, "role": str}}
    IMPORTANT: Each entry MUST use key "target_dtype" (not "type" or "dtype").
    Include ALL canonical_columns when dataset has <=80 columns.
  iteration_policy: {max_iterations, metric_improvement_max, runtime_fix_max, compliance_bootstrap_max}
  future_ml_handoff: (when model_training=false but future target defined)

DATA_ENGINEER section:
  required_outputs: array of {intent, path, required, kind} — DE-specific deliverables
  cleaning_gates: array of gate objects for data quality
  runbook: objectives + constraints for cleaning task
  artifact_requirements: cleaned_dataset paths, required_columns, transforms
    CRITICAL: required_columns plus optional_passthrough_columns must preserve every operational dependency the contract needs. A column may be disallowed as a model feature and still need to survive for filtering, splitting, label scoping, lineage, or audit.
    Include output_manifest_path for downstream provenance chain.
  outlier_policy: (optional) {"enabled", "target_columns", "methods"}
  constraints: (optional) additional execution constraints

ML_ENGINEER section:
  required_outputs: array of {intent, path, required, kind} — ML-specific deliverables
    When model_training=true, a metrics JSON artifact MUST appear.
  qa_gates: quality gate objects for ML outputs
  reviewer_gates: code review gate objects
  runbook: objectives + constraints for ML task
  evaluation_spec: {objective_type, primary_target, primary_metric, metric_definition_rule, label_columns}
  validation_requirements: {method, primary_metric, metrics_to_report, params}
    primary_metric MUST equal evaluation_spec.primary_metric.
  optimization_policy: {enabled, max_rounds, quick_eval_folds, full_eval_folds, min_delta, patience, optimization_direction, tie_breakers, allow_*}
    - optimization_direction MUST be reasoned from the business objective and primary metric semantics, not guessed from defaults.
    - tie_breakers should be an ordered list of secondary comparison preferences only when they are justified by the run context.
  artifact_requirements: model output paths
  plot_spec, visual_requirements, decisioning_requirements: (optional)

CLEANING_REVIEWER section:
  Inherits cleaning_gates and full context from data_engineer section automatically.
  Only add overrides or focus directives specific to the review process.

QA_REVIEWER section:
  Inherits qa_gates and full context from ml_engineer section automatically.
  review_subject: which agent's work to review (typically "ml_engineer")
  artifacts_to_verify: list of artifact paths to check

BUSINESS_TRANSLATOR section:
  reporting_policy, plot_spec, visual_requirements, decisioning_requirements, evidence_inventory

--- GATE CONVENTIONS ---

Gate preferred shape: {"name": str, "severity": "HARD"|"SOFT", "params": object}
Optional extensions: action_type, column_phase, final_state, condition, evidence_required, action_if_fail.
HARD: failure means corrupt, unsafe, or silently wrong output.
SOFT: quality degraded but output remains usable.

--- VISUALIZATION PLANNING ---

Data Engineer plots (in data_engineer.runbook): EDA, missing values, distributions, correlations.
ML Engineer plots (in ml_engineer.runbook): feature importance, CV performance, learning curves, confusion matrices.
Each agent saves plots as PNG files in static/plots/ with descriptive names.
Add "static/plots/*.png" to the agent's required_outputs with required=false.
""".strip()

_EXECUTION_CONTRACT_TOOL_NAME = "emit_execution_contract"
_EXECUTION_SEMANTIC_CORE_TOOL_NAME = "emit_execution_semantic_core"
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
    "anyOf": [{"required": ["changes"]}, {"required": ["patch"]}],
    "properties": {
        "changes": {
            "type": "object",
            "minProperties": 1,
            "additionalProperties": _JSON_PATCH_VALUE_SCHEMA,
            "description": "Minimal nested fields to deep-merge into the current contract.",
        },
        "patch": {
            "type": "array",
            "minItems": 1,
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


def _read_openai_raw_response_body(raw_response: Any) -> str:
    if raw_response is None:
        return ""
    text_attr = getattr(raw_response, "text", None)
    try:
        if callable(text_attr):
            raw = text_attr()
        else:
            raw = text_attr
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        if isinstance(raw, str) and raw.strip():
            return raw
    except Exception:
        pass
    http_response = getattr(raw_response, "http_response", None)
    if http_response is not None:
        try:
            raw = getattr(http_response, "text", None)
            if callable(raw):
                raw = raw()
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            if isinstance(raw, str) and raw.strip():
                return raw
        except Exception:
            pass
    return ""


class _OpenRouterAdapter:
    """Adapter that exposes a generate_content-style call surface using OpenRouter (OpenAI API)."""

    def __init__(self, api_key: str, model_name: str):
        self.api_key = str(api_key or "").strip()
        self.model_name = str(model_name or "").strip()
        retries_raw = str(os.getenv("EXECUTION_PLANNER_TRANSPORT_MAX_RETRIES", "0")).strip()
        try:
            self.transport_max_retries = max(0, int(retries_raw))
        except Exception:
            self.transport_max_retries = 0
        self._client = OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
            max_retries=self.transport_max_retries,
        )

    def generate_content(self, prompt: str, generation_config: Dict[str, Any] | None = None):
        """Call OpenRouter and return a response object compatible with _extract_openai_response_text."""
        config = generation_config or {}
        call_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
        }
        if "temperature" in config:
            call_kwargs["temperature"] = config["temperature"]
        if "max_output_tokens" in config:
            call_kwargs["max_tokens"] = config["max_output_tokens"]
        if "top_p" in config:
            call_kwargs["top_p"] = config["top_p"]
        # Convert Gemini-style tools to OpenAI-style tools
        if "tools" in config and config["tools"]:
            openai_tools = []
            for tool in config["tools"]:
                if isinstance(tool, dict) and "function" in tool:
                    openai_tools.append(tool)
                elif isinstance(tool, dict) and "functionDeclarations" in tool:
                    for decl in tool["functionDeclarations"]:
                        openai_tools.append({
                            "type": "function",
                            "function": {
                                "name": decl.get("name", ""),
                                "description": decl.get("description", ""),
                                "parameters": decl.get("parametersJsonSchema", decl.get("parameters", {})),
                            },
                        })
            if openai_tools:
                call_kwargs["tools"] = openai_tools
        if "tool_config" in config and config["tool_config"]:
            tc = config["tool_config"]
            if isinstance(tc, dict):
                fcc = tc.get("functionCallingConfig", {})
                allowed = fcc.get("allowedFunctionNames", [])
                if allowed:
                    call_kwargs["tool_choice"] = {"type": "function", "function": {"name": allowed[0]}}
        use_raw_capture = str(os.getenv("EXECUTION_PLANNER_CAPTURE_RAW_RESPONSE", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if use_raw_capture:
            raw_api = getattr(self._client.chat.completions, "with_raw_response", None)
            if raw_api is None or not hasattr(raw_api, "create"):
                raise RuntimeError("with_raw_response transport unavailable while raw capture is enabled")
            raw_response = raw_api.create(**call_kwargs)
            response = raw_response.parse() if hasattr(raw_response, "parse") else raw_response
            raw_body = _read_openai_raw_response_body(raw_response)
            raw_http_response = getattr(raw_response, "http_response", None)
            request_id = getattr(response, "_request_id", None)
            if not request_id and raw_http_response is not None:
                headers = getattr(raw_http_response, "headers", None)
                if headers is not None:
                    try:
                        request_id = headers.get("x-request-id") or headers.get("request-id")
                    except Exception:
                        request_id = None
            if raw_body:
                try:
                    setattr(response, "_codex_raw_body", raw_body)
                except Exception:
                    pass
            try:
                setattr(response, "_codex_transport_mode", "with_raw_response")
                setattr(response, "_codex_transport_max_retries", self.transport_max_retries)
                if request_id:
                    setattr(response, "_codex_request_id", str(request_id))
            except Exception:
                pass
            return response

        response = self._client.chat.completions.create(**call_kwargs)
        try:
            setattr(response, "_codex_transport_mode", "standard")
            setattr(response, "_codex_transport_max_retries", self.transport_max_retries)
            request_id = getattr(response, "_request_id", None)
            if request_id:
                setattr(response, "_codex_request_id", str(request_id))
        except Exception:
            pass
        return response

SEMANTIC_EXECUTION_PLANNER_PROMPT = """
You are a Semantic Execution Planner for a multi-agent business intelligence system.

MISSION
Decide what this run should achieve and produce ONE semantic_core JSON object that captures the authoritative meaning of the run. A separate compiler step will later turn this into an executable contract. Your job ends at semantic intent.

FIVE CORE PRINCIPLES
1. Evidence-grounded: Every field must be supported by evidence from the business objective, strategy, column inventory, or dataset profile.
2. Semantic closure: If you declare a concept (future ML handoff, feature sets, column exclusions), close every dependency it implies. model_features must exclude columns you classified as leakage/admin/PII. But excluding a column from model_features does NOT mean the column can disappear from the run if task semantics, routing, filtering, temporal validation, or auditability still depend on it. Allowed feature sets must be concretely closed through model_features.
3. Downstream-executable: The compiler and downstream agents must be able to build an executable plan from your semantic_core alone plus data context.
4. No invention: Do not reference columns, artifacts, or capabilities that are not supported by the inputs.
5. Minimal: The smallest semantic_core that satisfies principles 1-4.

SOURCE OF TRUTH
1. business_objective + strategy define what this run must achieve.
2. column_inventory + dataset profile define what exists and what is defensible.
3. When strategy hints conflict with structural evidence from the data, prefer the safer structural interpretation.

REASONING WORKFLOW
Before emitting JSON, reason through:
1. What is the business asking this run to achieve now versus in a future run?
2. Classify columns into semantic roles grounded in business meaning, availability at prediction time, and operational use. A column that encodes the outcome or is only available after the prediction moment cannot be a model feature. But some non-feature columns are still operationally required to define row filters, scoring scope, split logic, temporal ordering, label availability, or audit trace.
3. Close every dependency: model_features must name explicit columns (not just conceptual families) whenever this run prepares a future modeling subset. Do not leave model_features empty while claiming future-ML readiness. If task_semantics, evaluation, or gates depend on a column, keep that dependency explicit even when the column is excluded from model_features.
4. Verify consistency: every column in model_features appears in column_roles.pre_decision. Every column in column_roles.outcome is excluded from model_features. Columns referenced by task_semantics, gating, or validation remain represented as operational dependencies even when they are not model features.

CANONICAL_COLUMNS
canonical_columns must list ALL columns from column_inventory that this run acknowledges. This is the dataset's full column manifest — not just anchor columns or role-specific subsets. If column_inventory has 42 columns and none are excluded, canonical_columns has 42 entries. The compiler and downstream agents use canonical_columns as the ground truth for what columns exist.

SCOPE REASONING
- scope is a routing signal, not the semantic brain of the contract.
- A future target does NOT imply model_training=true. If this run is cleaning/feature preparation, set model_training=false even when a target column exists.
- Include future_ml_handoff when model_training=false but a future target/modeling handoff is clearly defined.

GATE PRINCIPLES
Gates must be grounded in actual data risk, not template completeness.
- HARD: failure makes the output corrupt, unsafe, or silently wrong (leakage surviving, target missing, schema violation).
- SOFT: quality degraded but output remains usable (null rate above threshold, optional format).
- Do not create gates for impossible conditions. Each gate must address a plausible risk visible in the data.

RUNBOOK AND TECHNIQUE HANDLING — CRITICAL
The downstream agents (data_engineer, ml_engineer) are senior engineers who reason about method selection given data context. Your runbooks must give them freedom to reason.

Strategy techniques are ADVISORY HYPOTHESES. You MUST abstract them into objectives:
- If strategy says "LightGBM with learning_rate=0.05, num_leaves=31": write "Train a competitive gradient boosting model optimized for the primary metric."
- If strategy says "RepeatedStratifiedKFold(n_splits=5, n_repeats=2)": write "Use repeated stratified CV with sufficient folds for stable estimates."
- If strategy says "median imputation fitted on training folds": write "Handle missing values appropriately, fitting only on training data to prevent leakage."
- If strategy says "IsotonicRegression on OOF predictions": write "Calibrate predicted probabilities using out-of-fold estimates."

What to KEEP verbatim: hard constraints (leakage columns to exclude, submission row count, output format, monotonicity requirements).
What to ABSTRACT: model families, hyperparameters, CV configurations, imputation methods, calibration techniques, specific library names.

A runbook that reads like a recipe (step 1 do X, step 2 do Y with parameter Z) is WRONG. A runbook that states objectives and constraints, trusting the agent to reason about implementation, is CORRECT.

OUTPUT
- Return ONLY the semantic_core JSON object.
- Do not emit compilation-only sections (artifact_requirements, column_dtype_targets, iteration_policy, evaluation_spec, validation_requirements). Those belong to the compiler step.
"""

MINIMAL_CONTRACT_COMPILER_PROMPT = """
You are a Senior Execution Contract Compiler for a multi-agent business intelligence system.

MISSION
Compile SEMANTIC_CORE_AUTHORITY_JSON into ONE executable JSON contract using the V5.0 HIERARCHICAL FORMAT. Your job is compilation, not reinterpretation. Build the smallest executable contract that preserves the semantic core and gives each downstream agent exactly what it needs — organized by agent hierarchy.

SOURCE OF TRUTH
1. SEMANTIC_CORE_AUTHORITY_JSON is authoritative. Do not override it with SUPPORT_CONTEXT.
2. SUPPORT_CONTEXT improves compilation quality but never overrides semantic decisions.
3. Do not widen scope, renegotiate intent, or re-decide whether model_training is active.

V5.0 HIERARCHICAL STRUCTURE
The contract has a "shared" section plus one section per agent: data_engineer, ml_engineer, cleaning_reviewer, qa_reviewer, business_translator.

HIERARCHY PRINCIPLE: If a field is needed by 2+ agents, place it in "shared". If it is exclusive to one agent's execution, place it in that agent's section. Each agent's view is built automatically by merging shared + its section.

The cleaning_reviewer inherits the data_engineer context (cleaning_gates, artifact_requirements, etc.) automatically. The qa_reviewer inherits the ml_engineer context (qa_gates, evaluation_spec, etc.) automatically. You do NOT need to duplicate those fields — only add review-specific directives in the reviewer sections.

FIVE CORE PRINCIPLES
1. Evidence-grounded: Every field must be supported by evidence from semantic_core or support_context. Do not invent columns, paths, or artifacts.
2. Semantic closure: If you declare a concept anywhere, close its dependencies everywhere. model_features must appear in cleaned_dataset.required_columns. Evaluation metric must match in evaluation_spec and validation_requirements. HARD gate columns and task_semantics dependencies must be covered by required_columns or optional_passthrough_columns even when they are excluded from model_features.
3. Downstream-executable: Each downstream agent must be able to execute its task from its contract section alone (plus shared). The contract is the single source of truth for execution.
4. No invention: Do not reference columns not in column_inventory, do not create training sections when model_training=false, do not invent artifacts beyond what the semantic core implies.
5. Minimal: The shortest valid contract that satisfies principles 1-4. No filler sections, no padding with defaults when context provides specifics.

WHAT GOES IN "shared" (preserved verbatim from SEMANTIC_CORE where available)
scope, active_workstreams, future_ml_handoff, task_semantics, column_roles, allowed_feature_sets, model_features, strategy_title, business_objective, output_dialect, canonical_columns, column_dtype_targets, iteration_policy.
These are the shared context fields that multiple agents need.

WHAT YOU COMPILE PER AGENT (the operational layer)
For each agent section, materialize the execution fields that semantic_core does not directly provide:

data_engineer: required_outputs (DE deliverables), cleaning_gates, runbook, artifact_requirements (cleaned_dataset with output_manifest_path), constraints, outlier_policy.
ml_engineer: required_outputs (ML deliverables), qa_gates, reviewer_gates, runbook, evaluation_spec, validation_requirements, optimization_policy, artifact_requirements, plot_spec, visual_requirements, decisioning_requirements.
cleaning_reviewer: focus_areas or severity_overrides only if needed (inherits DE context automatically).
qa_reviewer: review_subject, artifacts_to_verify (inherits ML context automatically).
business_translator: reporting_policy, visual_requirements, decisioning_requirements.

ARTIFACT REQUIREMENTS — MANDATORY FIELDS
When scope includes cleaning (cleaning_only or full_pipeline):
- data_engineer.artifact_requirements.cleaned_dataset MUST include output_manifest_path (e.g., "artifacts/clean/cleaning_manifest.json"). The manifest is read by downstream reviewers and QA to verify cleaning provenance.
- data_engineer.required_outputs MUST include a cleaning_manifest artifact with intent "cleaning_manifest" pointing to the same path.
Omitting the manifest path causes downstream validation failures because the system cannot resolve the cleaning provenance chain.

VISUALIZATION PLANNING
Reason about what diagnostic visualizations would best represent the results of THIS specific strategy and assign them to the right agent:

Data Engineer visualizations (EDA / data quality — in data_engineer.runbook):
- Only when scope includes cleaning or feature engineering.
- Examples: missing value heatmaps, distribution plots, correlation matrices, outlier analysis.

ML Engineer visualizations (model results — in ml_engineer.runbook):
- Only when model_training=true.
- Examples: feature importance, CV fold performance, learning curves, confusion matrices, ROC/PR curves.

Adapt to the problem type (regression, classification, time series, anomaly detection, cleaning-only).
Each agent saves plots as PNG in static/plots/. Add "static/plots/*.png" to the agent's required_outputs with required=false.

COMPILATION REASONING
Before emitting JSON, reason through:
1. What did semantic_core already decide? Preserve it in "shared".
2. What does each agent need exclusively? Place it in that agent's section.
3. Are all dependencies closed? Every model_feature in cleaned_dataset.required_columns? Every HARD gate and task_semantics dependency covered? If a column is needed operationally but not as a model feature, preserve it through required_columns or optional_passthrough_columns. Every metric consistent across sections?
4. Is anything invented? Remove it. Is anything redundant? Remove it.

RUNBOOK PRINCIPLES
Runbooks state OBJECTIVES and CONSTRAINTS, not implementation recipes.
- BAD: "Apply median imputation to all features." GOOD: "Handle missing values appropriately per column distribution."
- BAD: "Use LightGBM with learning_rate=0.05" (copied from strategy). GOOD: "Train a competitive gradient boosting model, optimizing hyperparameters for the primary metric."
CRITICAL: Strategy techniques in semantic_core are advisory hypotheses. Do NOT transcribe them as prescriptive steps. Abstract into objectives and let the downstream agent reason about implementation.

GATE PRINCIPLES
- HARD: failure means corrupt, unsafe, or silently wrong output.
- SOFT: quality degraded but output remains usable.
- Each gate must address a distinct, plausible risk grounded in the data. 5 well-reasoned gates beat 15 templated ones.
- Place cleaning_gates in data_engineer section, qa_gates and reviewer_gates in ml_engineer section.

SCOPE-CONDITIONAL LOGIC
- When model_training=false: do not invent evaluation_spec, validation_requirements, or training sections in ml_engineer. Compile a handoff-ready cleaning/feature-prep contract.
- When model_training=true: ml_engineer MUST include evaluation_spec, validation_requirements, runbook, and optimization_policy. A metrics JSON artifact must appear in ml_engineer.required_outputs.

OUTPUT
- Return ONLY valid JSON. No markdown, no code fences, no reasoning traces.
- The JSON must have these top-level keys: contract_version ("5.0"), shared, data_engineer, ml_engineer, cleaning_reviewer, qa_reviewer, business_translator.
- Populate every required section with grounded content. Never return empty placeholders.
- Refer to OPERATIONAL SCHEMA EXAMPLES below for field shapes and conventions.
"""

CONTRACT_VALIDATION_ADJUDICATOR_PROMPT = """
You are a Contract Validation Adjudicator for a multi-agent execution planner.

Goal:
- Review only AMBIGUOUS validation issues after deterministic validation.
- Decide whether each ambiguous issue is a real blocking contradiction, a warning, or already semantically resolved.
- Reason from context, not from rigid lexical rules.

Authority and precedence:
- SEMANTIC_CORE_AUTHORITY_JSON is the semantic source of truth.
- COMPILED_CONTRACT_JSON is the candidate executable contract.
- AMBIGUOUS_ISSUES_JSON contains only issues already flagged as adjudicable by deterministic validation.
- SUPPORT_CONTEXT is reinforcement only; never override semantic_core with it.

Decision policy:
- clear: the contract is semantically valid for this issue and the issue should be removed.
- downgrade_warning: the contract is acceptable, but the issue should remain as a warning.
- keep_error: the issue is a real blocking contradiction and should remain an error.

Examples of acceptable semantic compilation:
- semantic required outputs like "cleaned_dataset" become concrete artifact outputs with matching intent.
- a cleaning gate whose action is to drop leakage/admin columns aligns with drop_columns rather than contradicting it.

Hard rules:
- Do not invent missing contract fields.
- Do not rewrite the contract.
- Only adjudicate the listed ambiguous issues.
- Return ONLY valid JSON.

Return format:
{
  "issue_verdicts": [
    {
      "issue_index": 0,
      "decision": "clear" | "downgrade_warning" | "keep_error",
      "reason": "short explanation"
    }
  ]
}


def _coerce_provider_text(value: Any, depth: int = 0) -> str:
    if depth > 8 or value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        for key in (
            "text",
            "content",
            "output_text",
            "reasoning",
            "reasoning_content",
            "value",
            "arguments",
        ):
            if key not in value:
                continue
            text = _coerce_provider_text(value.get(key), depth + 1)
            if text:
                return text
        for key in ("tool_calls", "parts", "items", "output"):
            nested = value.get(key)
            if isinstance(nested, list):
                parts = [_coerce_provider_text(item, depth + 1) for item in nested]
                joined = "\n".join(part for part in parts if part).strip()
                if joined:
                    return joined
        for key in ("message", "delta", "function", "function_call", "functionCall", "parsed", "response"):
            if key not in value:
                continue
            text = _coerce_provider_text(value.get(key), depth + 1)
            if text:
                return text
        for nested in value.values():
            text = _coerce_provider_text(nested, depth + 1)
            if text:
                return text
        return ""
    if isinstance(value, (list, tuple)):
        parts = [_coerce_provider_text(item, depth + 1) for item in value]
        return "\n".join(part for part in parts if part).strip()

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            text = _coerce_provider_text(model_dump(exclude_none=True), depth + 1)
            if text:
                return text
        except Exception:
            pass

    for attr in (
        "text",
        "content",
        "output_text",
        "reasoning",
        "reasoning_content",
        "value",
        "arguments",
        "parsed",
        "message",
        "delta",
        "function",
        "function_call",
        "functionCall",
        "parts",
    ):
        if not hasattr(value, attr):
            continue
        text = _coerce_provider_text(getattr(value, attr), depth + 1)
        if text:
            return text
    return ""


def _build_explicit_transport_failure(
    rule: str,
    message: str,
    *,
    phase: str,
    item: Any = None,
) -> Dict[str, Any]:
    return {
        "status": "error",
        "accepted": False,
        "issues": [
            {
                "severity": "error",
                "rule": rule,
                "message": message,
                "item": item,
            }
        ],
        "summary": {"error_count": 1, "warning_count": 0, "phase": phase},
    }
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
            "artifact_requirements.cleaned_dataset.output_path",
            "artifact_requirements.cleaned_dataset.output_manifest_path|manifest_path",
            "artifact_requirements.cleaned_dataset.required_columns",
            "artifact_requirements.cleaned_dataset.required_feature_selectors (optional)",
            "artifact_requirements.cleaned_dataset.column_transformations (optional)",
            "artifact_requirements.cleaned_dataset.column_transformations.drop_policy (optional)",
            "artifact_requirements.enriched_dataset.required_columns when future_ml_handoff or enriched output is declared",
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
            "artifact_requirements.cleaned_dataset.output_path",
            "artifact_requirements.cleaned_dataset.output_manifest_path|manifest_path",
            "artifact_requirements.cleaned_dataset.required_columns",
            "artifact_requirements.cleaned_dataset.required_feature_selectors (optional)",
            "artifact_requirements.cleaned_dataset.column_transformations (optional)",
            "artifact_requirements.cleaned_dataset.column_transformations.drop_policy (optional)",
            "artifact_requirements.enriched_dataset when a separate enriched handoff artifact is declared",
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
            "keep required_outputs as artifact outputs, using object form with intent when semantic deliverables are materialized",
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


def _safe_json_serializable(obj: Any) -> Any:
    """Sanitize an object for JSON serialization, replacing non-serializable values."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            try:
                json.dumps(v)
                result[k] = v
            except (TypeError, ValueError):
                result[k] = str(type(v).__name__)
        return result
    if isinstance(obj, (list, tuple)):
        return [_safe_json_serializable(item) for item in obj]
    return str(obj)


def _apply_schema_coercion(contract: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mechanical type/shape coercion only — no semantic invention.

    Retains only:
    - Schema registry repairs (dtype key aliases, selector type inference, gate shape)
    - derived_columns normalization (dict/mixed -> list[str])
    - Scope alias resolution
    - Optimization policy defaults merge
    - Contract version normalization
    """
    if not isinstance(contract, dict):
        return {}

    # Deserialize double-encoded JSON strings: some LLMs return nested objects
    # as stringified JSON (e.g. "clean_dataset": "{\"output_path\": ...}").
    # This is a mechanical transport artifact, not a semantic issue.
    for key, value in list(contract.items()):
        if isinstance(value, str) and value.strip().startswith("{"):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    contract[key] = parsed
            except (json.JSONDecodeError, ValueError):
                pass
        elif isinstance(value, dict):
            for sub_key, sub_value in list(value.items()):
                if isinstance(sub_value, str) and sub_value.strip().startswith(("{", "[")):
                    try:
                        parsed = json.loads(sub_value)
                        if isinstance(parsed, (dict, list)):
                            value[sub_key] = parsed
                    except (json.JSONDecodeError, ValueError):
                        pass

    # Coerce fields with well-known container types: if the LLM returned a
    # non-container value (e.g. -1, "none", null) where a list or dict is
    # expected, replace with the empty container.  This is pure transport
    # normalization — it preserves "empty" semantics without inventing content.
    _LIST_KEYS = (
        "canonical_columns", "required_outputs", "model_features",
        "cleaning_gates", "qa_gates", "reviewer_gates",
    )
    _DICT_KEYS = (
        "output_dialect", "column_roles", "allowed_feature_sets",
        "task_semantics", "active_workstreams", "future_ml_handoff",
        "artifact_requirements", "evaluation_spec", "validation_requirements",
        "iteration_policy", "column_dtype_targets",
    )
    for k in _LIST_KEYS:
        if k in contract and not isinstance(contract[k], list):
            contract[k] = []
    for k in _DICT_KEYS:
        if k in contract and not isinstance(contract[k], dict):
            contract[k] = {}
    # Same for nested fields inside known dicts
    for nested_dict_key in _DICT_KEYS:
        obj = contract.get(nested_dict_key)
        if isinstance(obj, dict):
            for sub_key, sub_val in list(obj.items()):
                if sub_key.endswith(("_to_report", "_columns", "_features")) and not isinstance(sub_val, list):
                    obj[sub_key] = [] if sub_val is not None and sub_val != "" else []
                elif sub_key == "params" and not isinstance(sub_val, dict):
                    obj[sub_key] = {}

    # Schema registry: dtype key aliases, selector type, gate shape
    contract = apply_contract_schema_registry_repairs(contract)

    # derived_columns: normalize dict/mixed -> list[str]
    contract["derived_columns"] = _extract_derived_column_names(contract.get("derived_columns"))

    # Scope compatibility projection: preserve explicit active workstreams as the
    # semantic authority and derive a routing-compatible scope from them.
    if isinstance(contract.get("active_workstreams"), dict):
        contract["scope"] = derive_contract_scope_from_workstreams(contract)
    elif "scope" in contract:
        contract["scope"] = normalize_contract_scope(contract.get("scope"))

    # Optimization policy defaults merge
    contract["optimization_policy"] = normalize_optimization_policy(contract.get("optimization_policy"))

    # Contract version normalization
    version = contract.get("contract_version")
    normalized_version = normalize_contract_version(version)
    if normalized_version != version:
        contract["contract_version"] = normalized_version
    elif version is None:
        contract["contract_version"] = CONTRACT_VERSION_V41

    return contract


def _apply_minimal_path_resolution(contract: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve dataset artifact bindings from explicit contract content if missing."""
    if not isinstance(contract, dict):
        return contract

    art_req = contract.get("artifact_requirements")
    if not isinstance(art_req, dict):
        art_req = {}
        contract["artifact_requirements"] = art_req

    clean_ds = art_req.get("cleaned_dataset")
    if not isinstance(clean_ds, dict):
        legacy_clean = art_req.get("clean_dataset")
        clean_ds = legacy_clean if isinstance(legacy_clean, dict) else {}
        art_req["cleaned_dataset"] = clean_ds
    if "clean_dataset" not in art_req and isinstance(clean_ds, dict):
        art_req["clean_dataset"] = clean_ds

    if not str(clean_ds.get("output_path") or clean_ds.get("output") or clean_ds.get("path") or "").strip():
        output_path = get_clean_dataset_output_path(contract)
        if output_path:
            clean_ds["output_path"] = output_path

    if not str(clean_ds.get("output_manifest_path") or clean_ds.get("manifest_path") or "").strip():
        manifest_path = get_clean_manifest_path(contract)
        if manifest_path:
            clean_ds["output_manifest_path"] = manifest_path

    enriched_ds = art_req.get("enriched_dataset")
    enriched_declared = isinstance(enriched_ds, dict)
    if not enriched_declared:
        enriched_ds = {}

    enriched_output_path = str(
        enriched_ds.get("output_path") or enriched_ds.get("output") or enriched_ds.get("path") or ""
    ).strip()
    if not enriched_output_path:
        enriched_output_path = get_enriched_dataset_output_path(contract)
        if enriched_output_path:
            enriched_ds["output_path"] = enriched_output_path

    if not isinstance(enriched_ds.get("required_columns"), list):
        allowed_sets = contract.get("allowed_feature_sets")
        if not isinstance(allowed_sets, dict):
            allowed_sets = {}

        model_features = []
        model_features_raw = allowed_sets.get("model_features")
        if isinstance(model_features_raw, list):
            model_features = [str(col).strip() for col in model_features_raw if str(col).strip()]

        target_columns: List[str] = []
        future_ml_handoff = contract.get("future_ml_handoff")
        if isinstance(future_ml_handoff, dict):
            target_columns_raw = future_ml_handoff.get("target_columns")
            if isinstance(target_columns_raw, list):
                target_columns = [str(col).strip() for col in target_columns_raw if str(col).strip()]
        if not target_columns:
            target_columns_raw = contract.get("target_columns")
            if isinstance(target_columns_raw, list):
                target_columns = [str(col).strip() for col in target_columns_raw if str(col).strip()]
        if not target_columns:
            roles = get_column_roles(contract)
            outcome_cols = roles.get("outcome")
            if isinstance(outcome_cols, list):
                target_columns = [str(col).strip() for col in outcome_cols if str(col).strip()]

        inferred_enriched_required = list(dict.fromkeys(model_features + target_columns))
        if inferred_enriched_required:
            enriched_ds["required_columns"] = inferred_enriched_required

    if enriched_declared or enriched_output_path or enriched_ds.get("required_columns"):
        art_req["enriched_dataset"] = enriched_ds

    return contract


def _apply_planner_structural_support(contract: Dict[str, Any] | None) -> Dict[str, Any]:
    """
    Build an optional deterministic compatibility projection of planner output.

    The returned object is never authoritative. It exists only for auxiliary
    projections, diagnostics, or backwards-compatible consumers. The raw
    planner contract remains the source of truth and must not be mutated by
    this helper.
    """
    if not isinstance(contract, dict):
        return {}
    supported = copy.deepcopy(contract)
    supported = _apply_minimal_path_resolution(supported)
    supported = normalize_artifact_requirements(supported)
    supported = _apply_schema_coercion(supported)
    return supported


def _infer_primary_metric_from_canonical(
    contract: Dict[str, Any],
    evaluation_spec: Dict[str, Any],
    task_semantics: Dict[str, Any],
) -> str:
    for source in (
        evaluation_spec.get("primary_metric"),
        contract.get("success_metric"),
        _extract_kpi_from_text(str(contract.get("business_objective") or "")),
        _extract_kpi_from_text(str(contract.get("strategy_title") or "")),
    ):
        normalized = _normalize_kpi_metric(source)
        if normalized:
            return normalized

    target_columns = _collect_targetish_columns(
        evaluation_spec.get("target_columns")
        or evaluation_spec.get("primary_targets")
        or task_semantics.get("target_columns")
        or contract.get("target_columns")
        or contract.get("outcome_columns")
    )
    multi_target = bool(len(target_columns) > 1 or task_semantics.get("multi_target"))

    text_hints = _collect_metric_hint_texts(
        contract.get("qa_gates"),
        contract.get("reviewer_gates"),
        contract.get("ml_engineer_runbook"),
        contract.get("data_engineer_runbook"),
        contract.get("notes_for_engineers"),
        contract.get("objective_analysis"),
        task_semantics,
    )
    for hint in text_hints:
        normalized = _extract_kpi_from_text(hint)
        if normalized:
            return normalized

    required_prediction_columns = [
        str(col).strip().lower()
        for col in (evaluation_spec.get("required_prediction_columns") or [])
        if str(col).strip()
    ]

    objective_type = str(
        evaluation_spec.get("objective_type")
        or task_semantics.get("objective_type")
        or task_semantics.get("problem_family")
        or ""
    ).strip().lower()
    if multi_target and any(
        token in objective_type for token in ("class", "probabilistic", "predictive")
    ):
        if any(col.startswith("prob_") for col in required_prediction_columns):
            if _targets_look_horizon_like(target_columns) or "horizon" in objective_type:
                return "mean_multi_horizon_log_loss"
            return "mean_multi_target_log_loss"
    if any(col.startswith("prob_") for col in required_prediction_columns):
        return "log_loss"
    if any(token in objective_type for token in ("regression", "forecast")):
        return "rmse"
    if "ranking" in objective_type:
        return "ndcg"
    if "class" in objective_type:
        return "roc_auc"
    return ""






def _unwrap_execution_contract_transport(payload: Any) -> Dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    contract = payload.get("contract")
    if isinstance(contract, dict):
        return contract
    return payload


def _is_meaningful_contract_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    return True


def _build_transport_validation(payload: Any) -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []
    if not isinstance(payload, dict):
        issues.append(
            {
                "severity": "error",
                "rule": "contract.transport_payload_not_object",
                "message": "Planner payload must be a JSON object.",
                "item": type(payload).__name__ if payload is not None else None,
            }
        )
        return {
            "status": "error",
            "accepted": False,
            "issues": issues,
            "summary": {"error_count": 1, "warning_count": 0, "phase": "transport"},
        }

    if not payload:
        issues.append(
            {
                "severity": "error",
                "rule": "contract.transport_payload_empty",
                "message": "Planner returned an empty JSON object; no canonical contract was transported.",
                "item": {},
            }
        )
        return {
            "status": "error",
            "accepted": False,
            "issues": issues,
            "summary": {"error_count": 1, "warning_count": 0, "phase": "transport"},
        }

    # Detect v5 hierarchical contract vs v4 flat contract
    _is_v5 = str(payload.get("contract_version", "")).startswith("5") or "shared" in payload
    if _is_v5:
        core_keys = tuple(
            key for key in (EXECUTION_CONTRACT_V5_CANONICAL_REQUIRED_KEYS + list(V5_AGENT_SECTION_KEYS))
            if key != "contract_version"
        )
    else:
        core_keys = tuple(
            key for key in EXECUTION_CONTRACT_CANONICAL_REQUIRED_KEYS if key != "contract_version"
        )
    present_keys = [key for key in core_keys if key in payload]
    meaningful_keys = [key for key in present_keys if _is_meaningful_contract_value(payload.get(key))]

    if not present_keys:
        issues.append(
            {
                "severity": "error",
                "rule": "contract.transport_payload_missing_core_keys",
                "message": "Planner tool payload omitted all canonical top-level contract sections.",
                "item": sorted(list(payload.keys()))[:20],
            }
        )
    elif not meaningful_keys:
        issues.append(
            {
                "severity": "error",
                "rule": "contract.transport_payload_trivial",
                "message": "Planner tool payload contains core keys but all are empty or placeholder values.",
                "item": present_keys,
            }
        )

    error_count = len([issue for issue in issues if str(issue.get("severity") or "").lower() in {"error", "fail"}])
    warning_count = len([issue for issue in issues if str(issue.get("severity") or "").lower() == "warning"])
    return {
        "status": "ok" if error_count == 0 else "error",
        "accepted": error_count == 0,
        "issues": issues,
        "summary": {"error_count": error_count, "warning_count": warning_count, "phase": "transport"},
    }




def _build_semantic_core_transport_validation(payload: Any) -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []
    if not isinstance(payload, dict):
        issues.append(
            {
                "severity": "error",
                "rule": "semantic_core.transport_payload_not_object",
                "message": "Semantic planner tool payload must be a JSON object.",
                "item": type(payload).__name__ if payload is not None else None,
            }
        )
        return {
            "status": "error",
            "accepted": False,
            "issues": issues,
            "summary": {"error_count": 1, "warning_count": 0, "phase": "semantic_transport"},
        }

    if not payload:
        issues.append(
            {
                "severity": "error",
                "rule": "semantic_core.transport_payload_empty",
                "message": "Semantic planner returned an empty JSON object.",
                "item": {},
            }
        )
        return {
            "status": "error",
            "accepted": False,
            "issues": issues,
            "summary": {"error_count": 1, "warning_count": 0, "phase": "semantic_transport"},
        }

    missing = [key for key in EXECUTION_SEMANTIC_CORE_REQUIRED_KEYS if key not in payload]
    if missing:
        issues.append(
            {
                "severity": "error",
                "rule": "semantic_core.missing_required_keys",
                "message": "Semantic planner omitted required semantic_core sections.",
                "item": missing[:20],
            }
        )

    empty_required = [
        key
        for key in EXECUTION_SEMANTIC_CORE_REQUIRED_KEYS
        if key in payload and not _is_meaningful_contract_value(payload.get(key))
    ]
    if empty_required:
        issues.append(
            {
                "severity": "error",
                "rule": "semantic_core.required_keys_empty",
                "message": "Semantic planner emitted required semantic_core sections with empty values.",
                "item": empty_required[:20],
            }
        )

    workstreams = payload.get("active_workstreams")
    if isinstance(workstreams, dict):
        if workstreams.get("model_training") is False:
            future_handoff = payload.get("future_ml_handoff")
            target_columns = (
                ((payload.get("task_semantics") or {}).get("target_columns"))
                if isinstance(payload.get("task_semantics"), dict)
                else None
            )
            if isinstance(target_columns, list) and target_columns and not isinstance(future_handoff, dict):
                issues.append(
                    {
                        "severity": "error",
                        "rule": "semantic_core.future_ml_handoff_missing",
                        "message": "future_ml_handoff is required when semantic_core defers model training but keeps a future target.",
                        "item": {"target_columns": target_columns[:8]},
                    }
                )

    error_count = len([issue for issue in issues if str(issue.get("severity") or "").lower() in {"error", "fail"}])
    warning_count = len([issue for issue in issues if str(issue.get("severity") or "").lower() == "warning"])
    return {
        "status": "ok" if error_count == 0 else "error",
        "accepted": error_count == 0,
        "issues": issues,
        "summary": {"error_count": error_count, "warning_count": warning_count, "phase": "semantic_transport"},
    }


def _canonicalize_string_list(values: Any) -> List[str]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []
    normalized: List[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(text)
    return normalized


def _normalize_semantic_token(value: Any) -> str:
    token = re.sub(r"[^0-9a-zA-Z]+", "_", str(value or "").strip().lower()).strip("_")
    return token


def _looks_like_materialized_output_path(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    text = value.strip()
    lower = text.lower()
    if "*" in text:
        return True
    if re.match(r"^[a-zA-Z]:[\\/]", text):
        return True
    if lower.startswith(("data/", "static/", "reports/", "artifacts/")):
        return True
    if "/" in text or "\\" in text:
        return True
    return False


def _extract_required_output_descriptor(item: Any) -> Dict[str, Any]:
    path = ""
    intent = ""
    semantic_aliases: List[str] = []

    def _add_alias(value: Any) -> None:
        if not isinstance(value, str):
            return
        text = value.strip()
        if not text:
            return
        normalized = _normalize_semantic_token(text)
        if normalized and normalized not in semantic_aliases:
            semantic_aliases.append(normalized)
        basename = os.path.basename(text.replace("\\", "/")).strip()
        if basename and basename != text:
            basename_norm = _normalize_semantic_token(basename)
            if basename_norm and basename_norm not in semantic_aliases:
                semantic_aliases.append(basename_norm)
        else:
            basename = text
        stem, ext = os.path.splitext(basename)
        if stem and ext:
            stem_norm = _normalize_semantic_token(stem)
            if stem_norm and stem_norm not in semantic_aliases:
                semantic_aliases.append(stem_norm)

    if isinstance(item, str):
        text = item.strip()
        if text:
            if _looks_like_materialized_output_path(text):
                path = text
            else:
                intent = text
                _add_alias(text)
    elif isinstance(item, dict):
        for key in ("path", "output", "file", "filename", "output_path"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                candidate = value.strip()
                if _looks_like_materialized_output_path(candidate):
                    path = candidate
                    break
        for key in ("intent", "artifact", "id", "name", "deliverable", "label", "semantic_output"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                cleaned = value.strip()
                _add_alias(cleaned)
                if not intent:
                    intent = cleaned
        if not intent:
            description = item.get("description")
            if isinstance(description, str) and description.strip() and len(description.strip()) <= 128:
                intent = description.strip()
        _add_alias(item.get("description"))
    elif item not in (None, ""):
        _add_alias(str(item))
        if not intent:
            intent = str(item)

    if path:
        _add_alias(path)
    if intent:
        _add_alias(intent)
    return {
        "path": path,
        "path_norm": str(path).replace("\\", "/").strip().lower() if _looks_like_materialized_output_path(path) else "",
        "intent": intent,
        "intent_norm": _normalize_semantic_token(intent),
        "semantic_norms": semantic_aliases,
        "raw": str(item),
    }


def _semantic_descriptor_token_parts(values: Any) -> set[str]:
    if isinstance(values, (str, bytes)):
        values = [values]
    elif not isinstance(values, (list, tuple, set)):
        values = [values]
    ignored = {
        "output",
        "outputs",
        "path",
        "paths",
        "file",
        "files",
        "artifact",
        "artifacts",
        "required",
        "requireds",
        "for",
        "and",
    }
    tokens: set[str] = set()
    for value in values:
        normalized = _normalize_semantic_token(value)
        if not normalized:
            continue
        for part in normalized.split("_"):
            token = str(part or "").strip().lower()
            if not token or token in ignored or token.isdigit():
                continue
            tokens.add(token)
    return tokens


def _expanded_semantic_capability_tokens(tokens: set[str]) -> set[str]:
    expanded = set(tokens)
    synonym_map = {
        "manifest": {"report", "validation", "audit", "provenance", "lineage"},
        "report": {"manifest", "summary", "note", "audit", "validation"},
        "summary": {"report", "table"},
        "table": {"summary"},
        "dataset": {"data", "rows"},
        "rows": {"dataset", "row"},
        "row": {"rows"},
        "scoring": {"score", "ranked", "prediction", "predictions"},
        "score": {"scoring", "ranked", "prediction", "predictions"},
        "ranked": {"scoring", "score"},
        "prediction": {"predictions", "scoring", "score"},
        "predictions": {"prediction", "scoring", "score"},
        "weight": {"weights", "spec", "configuration"},
        "weights": {"weight", "spec", "configuration"},
        "spec": {"weight", "weights", "configuration"},
        "configuration": {"spec", "weight", "weights"},
        "validation": {"report", "manifest", "audit"},
        "audit": {"report", "manifest", "validation"},
    }
    for token in list(tokens):
        expanded.update(synonym_map.get(token, set()))
    return expanded


def _collect_compiled_output_capability_tokens(
    compiled_contract: Dict[str, Any] | None,
    compiled_output_descriptors: List[Dict[str, Any]],
) -> set[str]:
    capability_tokens: set[str] = set()

    def _ingest(value: Any) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                capability_tokens.update(_semantic_descriptor_token_parts(key))
                _ingest(nested)
            return
        if isinstance(value, list):
            for nested in value:
                _ingest(nested)
            return
        if isinstance(value, str):
            capability_tokens.update(_semantic_descriptor_token_parts(value))

    for descriptor in compiled_output_descriptors:
        capability_tokens.update(_semantic_descriptor_token_parts(descriptor.get("semantic_norms") or []))
    if isinstance(compiled_contract, dict):
        for key in (
            "required_output_artifacts",
            "required_outputs",
            "artifact_requirements",
            "evaluation_spec",
            "validation_requirements",
            "decisioning_requirements",
            "visual_requirements",
        ):
            _ingest(compiled_contract.get(key))
        for section in ("data_engineer", "ml_engineer", "business_translator"):
            payload = compiled_contract.get(section)
            if isinstance(payload, dict):
                for key in (
                    "required_output_artifacts",
                    "required_outputs",
                    "artifact_requirements",
                    "evaluation_spec",
                    "validation_requirements",
                    "decisioning_requirements",
                    "visual_requirements",
                ):
                    _ingest(payload.get(key))
    return _expanded_semantic_capability_tokens(capability_tokens)


def _semantic_output_descriptor_matches_compiled_capabilities(
    semantic_descriptor: Dict[str, Any],
    compiled_output_descriptors: List[Dict[str, Any]],
    compiled_capability_tokens: set[str],
) -> bool:
    semantic_aliases = {
        str(token)
        for token in (semantic_descriptor.get("semantic_norms") or [])
        if str(token or "").strip()
    }
    semantic_tokens = _expanded_semantic_capability_tokens(_semantic_descriptor_token_parts(semantic_aliases))
    if not semantic_tokens:
        return False

    if semantic_tokens and semantic_tokens.issubset(compiled_capability_tokens):
        return True
    if len(semantic_tokens & compiled_capability_tokens) >= 2:
        return True

    for compiled_descriptor in compiled_output_descriptors:
        compiled_aliases = {
            str(token)
            for token in (compiled_descriptor.get("semantic_norms") or [])
            if str(token or "").strip()
        }
        if semantic_aliases and not semantic_aliases.isdisjoint(compiled_aliases):
            return True
        compiled_tokens = _expanded_semantic_capability_tokens(_semantic_descriptor_token_parts(compiled_aliases))
        overlap = semantic_tokens & compiled_tokens
        if not overlap:
            continue
        if overlap == semantic_tokens:
            return True
        if len(overlap) >= 2:
            return True
        if len(semantic_tokens) <= 2 and overlap == semantic_tokens:
            return True
        if len(overlap) / max(1, len(semantic_tokens)) >= 0.67:
            return True
    return False


def _build_semantic_guard_validation(
    semantic_core: Dict[str, Any] | None,
    compiled_contract: Dict[str, Any] | None,
) -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []
    if not isinstance(semantic_core, dict) or not semantic_core:
        return {
            "status": "ok",
            "accepted": True,
            "issues": [],
            "summary": {"error_count": 0, "warning_count": 0, "phase": "semantic_guard"},
        }
    # V5 hierarchical: flatten so the guard logic (which reads top-level keys) works unchanged.
    if isinstance(compiled_contract, dict) and str(compiled_contract.get("contract_version", "")).startswith("5"):
        from src.utils.contract_accessors import flatten_v5_contract
        compiled_contract = flatten_v5_contract(compiled_contract)
    if not isinstance(compiled_contract, dict) or not compiled_contract:
        issues.append(
            {
                "severity": "error",
                "rule": "semantic_guard.contract_missing",
                "message": "Compiled contract missing while semantic_core exists.",
                "item": {},
            }
        )
        return {
            "status": "error",
            "accepted": False,
            "issues": issues,
            "summary": {"error_count": 1, "warning_count": 0, "phase": "semantic_guard"},
        }

    def _record_conflict(
        rule: str,
        message: str,
        item: Any = None,
        *,
        severity: str = "error",
        extras: Dict[str, Any] | None = None,
    ) -> None:
        issue: Dict[str, Any] = {
            "severity": severity,
            "rule": rule,
            "message": message,
            "item": item,
        }
        if isinstance(extras, dict):
            issue.update(extras)
        issues.append(issue)

    def _resolved_compiled_role_bucket(bucket: str, compiled_roles_payload: Dict[str, Any]) -> set[str]:
        bucket = str(bucket or "").strip().lower()
        direct = set(x.lower() for x in _canonicalize_string_list(compiled_roles_payload.get(bucket)))
        if bucket != "pre_decision":
            return direct
        refined_union: set[str] = set()
        for refined_bucket in (
            "pre_decision",
            "identifiers",
            "time_columns",
            "post_decision_audit_only",
            "unknown",
        ):
            refined_union.update(
                x.lower() for x in _canonicalize_string_list(compiled_roles_payload.get(refined_bucket))
            )
        return refined_union

    def _semantic_pre_decision_anchor_columns() -> set[str]:
        anchors: set[str] = set()

        for source in (
            semantic_core.get("model_features"),
            (semantic_core.get("allowed_feature_sets") or {}).get("model_features")
            if isinstance(semantic_core.get("allowed_feature_sets"), dict)
            else None,
            (semantic_core.get("column_roles") or {}).get("operational_dependencies")
            if isinstance(semantic_core.get("column_roles"), dict)
            else None,
            (semantic_core.get("task_semantics") or {}).get("target_columns")
            if isinstance(semantic_core.get("task_semantics"), dict)
            else None,
            (semantic_core.get("task_semantics") or {}).get("primary_target")
            if isinstance(semantic_core.get("task_semantics"), dict)
            else None,
            (semantic_core.get("future_ml_handoff") or {}).get("target_columns")
            if isinstance(semantic_core.get("future_ml_handoff"), dict)
            else None,
            (semantic_core.get("future_ml_handoff") or {}).get("primary_target")
            if isinstance(semantic_core.get("future_ml_handoff"), dict)
            else None,
            (semantic_core.get("evaluation_spec") or {}).get("label_columns")
            if isinstance(semantic_core.get("evaluation_spec"), dict)
            else None,
            (semantic_core.get("evaluation_spec") or {}).get("primary_target")
            if isinstance(semantic_core.get("evaluation_spec"), dict)
            else None,
            (semantic_core.get("validation_requirements") or {}).get("target_columns")
            if isinstance(semantic_core.get("validation_requirements"), dict)
            else None,
            (semantic_core.get("validation_requirements") or {}).get("target_column")
            if isinstance(semantic_core.get("validation_requirements"), dict)
            else None,
            (semantic_core.get("validation_requirements") or {}).get("split_column")
            if isinstance(semantic_core.get("validation_requirements"), dict)
            else None,
            ((semantic_core.get("validation_requirements") or {}).get("params") or {}).get("split_column")
            if isinstance((semantic_core.get("validation_requirements") or {}).get("params"), dict)
            else None,
        ):
            anchors.update(x.lower() for x in _canonicalize_string_list(source))

        return anchors

    semantic_resolved_workstreams = resolve_contract_active_workstreams(semantic_core)
    compiled_resolved_workstreams = resolve_contract_active_workstreams(compiled_contract)
    semantic_expected_scope = derive_contract_scope_from_workstreams(
        {
            "scope": semantic_core.get("scope"),
            "active_workstreams": semantic_resolved_workstreams,
        }
    )
    compiled_effective_scope = derive_contract_scope_from_workstreams(
        {
            "scope": compiled_contract.get("scope"),
            "active_workstreams": compiled_resolved_workstreams,
        }
    )

    if semantic_expected_scope and compiled_effective_scope:
        if semantic_expected_scope != compiled_effective_scope:
            _record_conflict(
                "semantic_guard.scope_changed",
                "Compiled contract changed the effective authoritative scope from semantic_core.",
                {
                    "semantic_scope": semantic_core.get("scope"),
                    "semantic_effective_scope": semantic_expected_scope,
                    "compiled_scope": compiled_contract.get("scope"),
                    "compiled_effective_scope": compiled_effective_scope,
                },
            )

    semantic_workstreams = semantic_core.get("active_workstreams")
    compiled_workstreams = compiled_contract.get("active_workstreams")
    if semantic_resolved_workstreams:
        if not compiled_resolved_workstreams:
            _record_conflict(
                "semantic_guard.active_workstreams_missing",
                "Compiled contract omitted active_workstreams from semantic_core.",
                {},
            )
        else:
            for key in ("cleaning", "feature_engineering", "model_training"):
                if compiled_resolved_workstreams.get(key) is not semantic_resolved_workstreams.get(key):
                    _record_conflict(
                        "semantic_guard.active_workstreams_changed",
                        "Compiled contract changed active_workstreams from semantic_core.",
                        {
                            "field": key,
                            "semantic": semantic_resolved_workstreams.get(key),
                            "compiled": compiled_resolved_workstreams.get(key),
                            "semantic_declared": semantic_workstreams if isinstance(semantic_workstreams, dict) else None,
                            "compiled_declared": compiled_workstreams if isinstance(compiled_workstreams, dict) else None,
                        },
                    )

    semantic_task = semantic_core.get("task_semantics")
    compiled_task = compiled_contract.get("task_semantics")
    if isinstance(semantic_task, dict):
        if not isinstance(compiled_task, dict):
            _record_conflict(
                "semantic_guard.task_semantics_missing",
                "Compiled contract omitted task_semantics from semantic_core.",
                {},
            )
        else:
            for key in ("problem_family", "objective_type", "primary_target", "prediction_unit"):
                semantic_value = semantic_task.get(key)
                compiled_value = compiled_task.get(key)
                if semantic_value not in (None, "", []) and compiled_value not in (None, "", []):
                    if semantic_value != compiled_value:
                        _record_conflict(
                            "semantic_guard.task_semantics_changed",
                            "Compiled contract changed task_semantics from semantic_core.",
                            {"field": key, "semantic": semantic_value, "compiled": compiled_value},
                        )
            semantic_targets = _canonicalize_string_list(semantic_task.get("target_columns"))
            compiled_targets = _canonicalize_string_list(compiled_task.get("target_columns"))
            if semantic_targets and compiled_targets and semantic_targets != compiled_targets:
                _record_conflict(
                    "semantic_guard.target_columns_changed",
                    "Compiled contract changed target_columns from semantic_core.",
                    {"semantic": semantic_targets, "compiled": compiled_targets},
                )

    semantic_features = _canonicalize_string_list(semantic_core.get("model_features"))
    compiled_features = _canonicalize_string_list(compiled_contract.get("model_features"))
    if semantic_features and compiled_features != semantic_features:
        _record_conflict(
            "semantic_guard.model_features_changed",
            "Compiled contract changed top-level model_features from semantic_core.",
            {"semantic": semantic_features, "compiled": compiled_features},
        )

    semantic_outputs_raw = (
        semantic_core.get("required_outputs")
        if isinstance(semantic_core.get("required_outputs"), list)
        else []
    )
    compiled_outputs_raw = (
        compiled_contract.get("required_outputs")
        if isinstance(compiled_contract.get("required_outputs"), list)
        else []
    )
    semantic_output_descriptors = [
        _extract_required_output_descriptor(item) for item in semantic_outputs_raw if item not in (None, "")
    ]
    compiled_output_descriptors = [
        _extract_required_output_descriptor(item) for item in compiled_outputs_raw if item not in (None, "")
    ]
    compiled_output_paths = {
        desc.get("path_norm")
        for desc in compiled_output_descriptors
        if desc.get("path_norm")
    }
    compiled_output_intents: set[str] = set()
    for desc in compiled_output_descriptors:
        for token in desc.get("semantic_norms") or []:
            if token:
                compiled_output_intents.add(str(token))
    compiled_output_capability_tokens = _collect_compiled_output_capability_tokens(
        compiled_contract,
        compiled_output_descriptors,
    )
    missing_output_paths: List[str] = []
    missing_output_intents: List[str] = []
    for descriptor in semantic_output_descriptors:
        semantic_path = descriptor.get("path_norm") or ""
        semantic_intent = descriptor.get("intent_norm") or ""
        semantic_aliases = {
            str(token)
            for token in (descriptor.get("semantic_norms") or [])
            if str(token or "").strip()
        }
        if semantic_path:
            if semantic_path not in compiled_output_paths:
                missing_output_paths.append(descriptor.get("path") or descriptor.get("raw") or "")
            continue
        if semantic_aliases:
            if semantic_aliases.isdisjoint(compiled_output_intents):
                if not _semantic_output_descriptor_matches_compiled_capabilities(
                    descriptor,
                    compiled_output_descriptors,
                    compiled_output_capability_tokens,
                ):
                    missing_output_intents.append(descriptor.get("intent") or descriptor.get("raw") or "")
            continue
        if semantic_intent and semantic_intent not in compiled_output_intents:
            if _semantic_output_descriptor_matches_compiled_capabilities(
                descriptor,
                compiled_output_descriptors,
                compiled_output_capability_tokens,
            ):
                continue
            missing_output_intents.append(descriptor.get("intent") or descriptor.get("raw") or "")
    if missing_output_paths:
        _record_conflict(
            "semantic_guard.required_outputs_dropped",
            "Compiled contract dropped file-like required_outputs declared by semantic_core.",
            missing_output_paths,
        )
    if missing_output_intents:
        _record_conflict(
            "semantic_guard.required_outputs_dropped",
            "Compiled contract could not be deterministically proven to preserve every semantic required_output intent from semantic_core.",
            {
                "missing_semantic_outputs": missing_output_intents,
                "compiled_required_outputs": copy.deepcopy(compiled_outputs_raw),
                "compiled_capability_tokens_sample": sorted(list(compiled_output_capability_tokens))[:40],
            },
            severity="warning",
            extras={
                "adjudicable": True,
                "ambiguity_type": "required_output_materialization",
            },
        )

    semantic_roles = semantic_core.get("column_roles")
    compiled_roles = compiled_contract.get("column_roles")
    semantic_pre_decision_anchors = _semantic_pre_decision_anchor_columns()
    if isinstance(semantic_roles, dict):
        if not isinstance(compiled_roles, dict):
            _record_conflict(
                "semantic_guard.column_roles_missing",
                "Compiled contract omitted column_roles from semantic_core.",
                {},
            )
        else:
            for bucket in (
                "pre_decision",
                "decision",
                "outcome",
                "post_decision_audit_only",
                "identifiers",
                "time_columns",
            ):
                semantic_bucket = set(x.lower() for x in _canonicalize_string_list(semantic_roles.get(bucket)))
                if bucket == "pre_decision" and semantic_pre_decision_anchors:
                    semantic_bucket = semantic_bucket.intersection(semantic_pre_decision_anchors)
                compiled_bucket = _resolved_compiled_role_bucket(bucket, compiled_roles)
                if semantic_bucket and not semantic_bucket.issubset(compiled_bucket):
                    _record_conflict(
                        "semantic_guard.column_roles_changed",
                        "Compiled contract dropped semantic_core column role assignments.",
                        {
                            "bucket": bucket,
                            "missing": sorted(list(semantic_bucket - compiled_bucket)),
                        },
                    )

    semantic_future = semantic_core.get("future_ml_handoff")
    compiled_future = compiled_contract.get("future_ml_handoff")
    if isinstance(semantic_future, dict):
        if not isinstance(compiled_future, dict):
            _record_conflict(
                "semantic_guard.future_ml_handoff_missing",
                "Compiled contract omitted future_ml_handoff from semantic_core.",
                {},
            )
        else:
            for key in ("enabled", "primary_target", "readiness_goal"):
                semantic_value = semantic_future.get(key)
                compiled_value = compiled_future.get(key)
                if semantic_value not in (None, "", []) and compiled_value not in (None, "", []):
                    if semantic_value != compiled_value:
                        _record_conflict(
                            "semantic_guard.future_ml_handoff_changed",
                            "Compiled contract changed future_ml_handoff from semantic_core.",
                            {"field": key, "semantic": semantic_value, "compiled": compiled_value},
                        )

    error_count = len([issue for issue in issues if str(issue.get("severity") or "").lower() in {"error", "fail"}])
    warning_count = len([issue for issue in issues if str(issue.get("severity") or "").lower() == "warning"])
    return {
        "status": "ok" if error_count == 0 else "error",
        "accepted": error_count == 0,
        "issues": issues,
        "summary": {"error_count": error_count, "warning_count": warning_count, "phase": "semantic_guard"},
    }


def _merge_validation_results(base_result: Dict[str, Any] | None, extra_result: Dict[str, Any] | None) -> Dict[str, Any]:
    base = copy.deepcopy(base_result) if isinstance(base_result, dict) else {
        "status": "ok",
        "accepted": True,
        "issues": [],
        "summary": {"error_count": 0, "warning_count": 0},
    }
    extra = extra_result if isinstance(extra_result, dict) else None
    if not extra:
        return base
    merged_issues = list(base.get("issues") or [])
    merged_issues.extend(list(extra.get("issues") or []))
    error_count = 0
    warning_count = 0
    for issue in merged_issues:
        if not isinstance(issue, dict):
            continue
        severity = str(issue.get("severity") or "").lower()
        if severity in {"error", "fail"}:
            error_count += 1
        elif severity == "warning":
            warning_count += 1
    base["issues"] = merged_issues
    base["accepted"] = error_count == 0
    base["status"] = "ok" if error_count == 0 else "error"
    summary = dict(base.get("summary") or {})
    summary["error_count"] = error_count
    summary["warning_count"] = warning_count
    if extra.get("summary") and isinstance(extra.get("summary"), dict):
        summary["semantic_guard_phase"] = extra["summary"].get("phase")
    base["summary"] = summary
    return base


def _collect_adjudicable_validation_issues(validation_result: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    if not isinstance(validation_result, dict):
        return []
    issues = validation_result.get("issues")
    if not isinstance(issues, list):
        return []
    adjudicable: List[Dict[str, Any]] = []
    for idx, issue in enumerate(issues):
        if not isinstance(issue, dict):
            continue
        severity = str(issue.get("severity") or "").lower()
        if severity not in {"error", "fail", "warning"}:
            continue
        if issue.get("adjudicable") is True:
            payload = copy.deepcopy(issue)
            payload["issue_index"] = idx
            adjudicable.append(payload)
    return adjudicable


def _apply_validation_adjudication(
    validation_result: Dict[str, Any] | None,
    adjudication_result: Dict[str, Any] | None,
) -> Dict[str, Any]:
    if not isinstance(validation_result, dict):
        return {
            "status": "error",
            "accepted": False,
            "issues": [
                {
                    "severity": "error",
                    "rule": "contract.validation_result_missing",
                    "message": "Validation result missing before adjudication.",
                }
            ],
            "summary": {"error_count": 1, "warning_count": 0},
        }
    if not isinstance(adjudication_result, dict):
        return copy.deepcopy(validation_result)
    verdicts = adjudication_result.get("issue_verdicts")
    if not isinstance(verdicts, list) or not verdicts:
        return copy.deepcopy(validation_result)
    result = copy.deepcopy(validation_result)
    issues = list(result.get("issues") or [])
    clear_indices: set[int] = set()
    downgrade_indices: Dict[int, str] = {}
    keep_indices: Dict[int, str] = {}
    for verdict in verdicts:
        if not isinstance(verdict, dict):
            continue
        try:
            issue_index = int(verdict.get("issue_index"))
        except Exception:
            continue
        decision = str(verdict.get("decision") or "").strip().lower()
        reason = str(verdict.get("reason") or "").strip()
        if issue_index < 0 or issue_index >= len(issues):
            continue
        if decision == "clear":
            clear_indices.add(issue_index)
        elif decision == "downgrade_warning":
            downgrade_indices[issue_index] = reason
        elif decision == "keep_error":
            keep_indices[issue_index] = reason
    merged_issues: List[Dict[str, Any]] = []
    for idx, issue in enumerate(issues):
        if not isinstance(issue, dict):
            continue
        if idx in clear_indices:
            continue
        updated = dict(issue)
        if idx in downgrade_indices:
            updated["severity"] = "warning"
            updated["adjudicated"] = True
            updated["adjudication_decision"] = "downgrade_warning"
            if downgrade_indices[idx]:
                updated["adjudication_reason"] = downgrade_indices[idx]
        elif idx in keep_indices:
            updated["adjudicated"] = True
            updated["adjudication_decision"] = "keep_error"
            if keep_indices[idx]:
                updated["adjudication_reason"] = keep_indices[idx]
        merged_issues.append(updated)
    error_count = 0
    warning_count = 0
    for issue in merged_issues:
        severity = str(issue.get("severity") or "").lower()
        if severity in {"error", "fail"}:
            error_count += 1
        elif severity == "warning":
            warning_count += 1
    result["issues"] = merged_issues
    result["accepted"] = error_count == 0
    result["status"] = "ok" if error_count == 0 else "error"
    summary = dict(result.get("summary") or {})
    summary["error_count"] = error_count
    summary["warning_count"] = warning_count
    summary["adjudicated"] = True
    result["summary"] = summary
    result["adjudication"] = copy.deepcopy(adjudication_result)
    return result


def _build_patch_transport_validation(payload: Any) -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []
    if not isinstance(payload, dict):
        issues.append(
            {
                "severity": "error",
                "rule": "contract.patch_payload_not_object",
                "message": "Planner repair payload must be a JSON object.",
                "item": type(payload).__name__ if payload is not None else None,
            }
        )
    elif not payload:
        issues.append(
            {
                "severity": "error",
                "rule": "contract.patch_payload_empty",
                "message": "Planner repair payload was empty; no incremental repair was transported.",
                "item": {},
            }
        )
    else:
        patch_ops = payload.get("patch")
        changes = payload.get("changes")
        has_patch_ops = isinstance(patch_ops, list) and len(patch_ops) > 0
        has_changes = isinstance(changes, dict) and any(
            _is_meaningful_contract_value(value) for value in changes.values()
        )
        is_single_op = {"op", "path"}.issubset(set(payload.keys()))
        if not (has_patch_ops or has_changes or is_single_op):
            issues.append(
                {
                    "severity": "error",
                    "rule": "contract.patch_payload_trivial",
                    "message": "Planner repair payload must include non-empty changes or patch operations.",
                    "item": sorted(list(payload.keys()))[:20],
                }
            )

    error_count = len([issue for issue in issues if str(issue.get("severity") or "").lower() in {"error", "fail"}])
    warning_count = len([issue for issue in issues if str(issue.get("severity") or "").lower() == "warning"])
    return {
        "status": "ok" if error_count == 0 else "error",
        "accepted": error_count == 0,
        "issues": issues,
        "summary": {"error_count": error_count, "warning_count": warning_count, "phase": "transport"},
    }


def _transport_validation_accepted(result: Dict[str, Any] | None) -> bool:
    if not isinstance(result, dict):
        return False
    if not bool(result.get("accepted", False)):
        return False
    summary = result.get("summary")
    if isinstance(summary, dict):
        try:
            if int(summary.get("error_count", 0) or 0) > 0:
                return False
        except Exception:
            return False
    return True


def _split_validation_issues_by_phase(validation_result: Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(validation_result, dict):
        return {}
    issues = validation_result.get("issues")
    if not isinstance(issues, list):
        issues = []
    projection_prefixes = (
        "contract.de_view_",
        "contract.ml_view_",
        "contract.reviewer_view_",
        "contract.qa_view_",
        "contract.view_projection_",
    )
    projection_rules = {"contract.view_projection_exception"}
    canonical_issues: List[Dict[str, Any]] = []
    projection_issues: List[Dict[str, Any]] = []
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        rule = str(issue.get("rule") or "")
        if rule in projection_rules or any(rule.startswith(prefix) for prefix in projection_prefixes):
            projection_issues.append(issue)
        else:
            canonical_issues.append(issue)

    def _pack(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        error_count = len(
            [issue for issue in items if str(issue.get("severity") or "").lower() in {"error", "fail"}]
        )
        warning_count = len(
            [issue for issue in items if str(issue.get("severity") or "").lower() == "warning"]
        )
        return {
            "status": "ok" if error_count == 0 else "error",
            "accepted": error_count == 0,
            "issues": items,
            "summary": {"error_count": error_count, "warning_count": warning_count},
        }

    return {
        "canonical_validation": _pack(canonical_issues),
        "projection_validation": _pack(projection_issues),
    }





def _ensure_optimization_policy(contract: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(contract, dict):
        return contract
    optimization_policy = normalize_optimization_policy(contract.get("optimization_policy"))
    validation_requirements = (
        contract.get("validation_requirements")
        if isinstance(contract.get("validation_requirements"), dict)
        else {}
    )
    evaluation_spec = (
        contract.get("evaluation_spec")
        if isinstance(contract.get("evaluation_spec"), dict)
        else {}
    )
    direction = _coerce_optimization_direction_from_contract(
        optimization_policy,
        validation_requirements,
        evaluation_spec,
    )
    if direction != "unspecified":
        optimization_policy["optimization_direction"] = direction
    tie_breakers = _coerce_optimization_tie_breakers_from_contract(
        optimization_policy,
        validation_requirements,
        evaluation_spec,
    )
    if tie_breakers:
        optimization_policy["tie_breakers"] = tie_breakers
    contract["optimization_policy"] = optimization_policy
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


def _repair_deliverable_invariants(contract: Dict[str, Any], errors: List[Dict[str, Any]] | None) -> Dict[str, Any]:
    if not isinstance(contract, dict):
        return {}

    artifacts = contract.get("required_output_artifacts")
    if not isinstance(artifacts, list):
        artifacts = []
        contract["required_output_artifacts"] = artifacts

    required_outputs = contract.get("required_outputs")
    if not isinstance(required_outputs, list):
        required_outputs = []
        contract["required_outputs"] = required_outputs

    default_paths = {
        "dataset": "data/cleaned_data.csv",
        "metrics": "data/metrics.json",
        "predictions": "data/predictions.csv",
        "submission": "data/submission.csv",
        "report": "reports/report.json",
    }

    for err in errors or []:
        if not isinstance(err, dict):
            continue
        expected_kind_raw = str(err.get("expected_kind") or "").strip()
        expected_owner = str(err.get("expected_owner") or "").strip()
        if not expected_kind_raw or not expected_owner:
            continue

        candidate_kinds = [token.strip() for token in expected_kind_raw.split("|") if token.strip()]
        if not candidate_kinds:
            continue

        promoted = False
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                continue
            if artifact.get("kind") not in candidate_kinds or str(artifact.get("owner") or "").strip() != expected_owner:
                continue
            if artifact.get("required"):
                continue
            artifact["required"] = True
            artifact_path = str(artifact.get("path") or "").strip()
            if artifact_path and artifact_path not in required_outputs:
                required_outputs.append(artifact_path)
            promoted = True
            break

        if promoted:
            continue

        chosen_kind = candidate_kinds[0]
        default_path = default_paths.get(chosen_kind, f"data/{chosen_kind}.csv")
        artifacts.append(
            {
                "id": f"auto_{chosen_kind}",
                "path": default_path,
                "required": True,
                "kind": chosen_kind,
                "description": f"Auto-generated to satisfy {err.get('invariant', 'unknown')} invariant.",
                "owner": expected_owner,
            }
        )
        if default_path not in required_outputs:
            required_outputs.append(default_path)

    spec = contract.get("spec_extraction")
    if isinstance(spec, dict):
        spec["deliverables"] = artifacts

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


def _collect_authoritative_target_columns(contract: Dict[str, Any] | None) -> List[str]:
    authoritative: List[str] = []

    if not isinstance(contract, dict):
        return authoritative

    def _append(values: Any) -> None:
        if isinstance(values, list):
            for item in values:
                _append(item)
            return
        if isinstance(values, (str, int, float)):
            token = str(values).strip()
            if token and token not in authoritative:
                authoritative.append(token)

    for scope_key in ("evaluation_spec", "validation_requirements", "objective_analysis"):
        scope = contract.get(scope_key)
        if not isinstance(scope, dict):
            continue
        for key in (
            "target_columns",
            "primary_targets",
            "label_columns",
            "primary_target",
            "target_column",
            "label_column",
        ):
            _append(scope.get(key))
        params = scope.get("params")
        if isinstance(params, dict):
            for key in (
                "target_columns",
                "primary_targets",
                "label_columns",
                "primary_target",
                "target_column",
                "label_column",
            ):
                _append(params.get(key))

    return authoritative


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
    "mean_multi_horizon_log_loss": [
        "mean_multi_horizon_log_loss",
        "mean multi horizon log loss",
        "average multi horizon log loss",
        "aggregated multi horizon log loss",
        "mean multi-horizon log loss",
        "average multi-horizon log loss",
        "aggregated multi-horizon log loss",
    ],
    "mean_multi_target_log_loss": [
        "mean_multi_target_log_loss",
        "mean multi target log loss",
        "average multi target log loss",
        "aggregated multi target log loss",
        "mean multi-target log loss",
        "average multi-target log loss",
        "aggregated multi-target log loss",
        "mean multi output log loss",
        "average multi output log loss",
        "mean multi-output log loss",
    ],
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
    tokens = [token for token in raw.split() if token]
    metric_like_tokens = {
        "metric",
        "metrics",
        "score",
        "scores",
        "loss",
        "logloss",
        "auc",
        "gini",
        "error",
        "rmse",
        "mae",
        "mape",
        "r2",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "concordance",
        "ndcg",
        "map",
        "mrr",
    }
    if 1 <= len(tokens) <= 6 and any(token in metric_like_tokens for token in tokens):
        return "_".join(tokens)
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


def _collect_metric_hint_texts(*sources: Any) -> List[str]:
    hints: List[str] = []

    def _append(node: Any) -> None:
        if isinstance(node, str):
            text = node.strip()
            if text:
                hints.append(text)
            return
        if isinstance(node, dict):
            for value in node.values():
                _append(value)
            return
        if isinstance(node, list):
            for item in node:
                _append(item)

    for source in sources:
        _append(source)
    return hints


def _targets_look_horizon_like(target_columns: List[str]) -> bool:
    for column in target_columns:
        text = str(column or "").strip().lower()
        if not text:
            continue
        if re.search(r"\b\d+\s*h\b", text) or re.search(r"_(\d+h|\d+d|\d+w)\b", text):
            return True
        if any(token in text for token in ("horizon", "12h", "24h", "48h", "72h")):
            return True
    return False


def _derive_metric_definition_rule(primary_metric: Any) -> str:
    metric = str(primary_metric or "").strip().lower()
    if metric.startswith("mean_") and (
        "log_loss" in metric or "logloss" in metric or metric.endswith("_loss")
    ):
        return "Use a simple arithmetic mean unless the contract explicitly provides weights."
    return ""


def _coerce_optimization_direction_from_contract(
    optimization_policy: Any,
    validation_requirements: Any,
    evaluation_spec: Any,
) -> str:
    candidates: List[Any] = []
    if isinstance(optimization_policy, dict):
        candidates.extend(
            [
                optimization_policy.get("optimization_direction"),
                optimization_policy.get("direction"),
                optimization_policy.get("metric_direction"),
            ]
        )
    if isinstance(validation_requirements, dict):
        candidates.extend(
            [
                validation_requirements.get("optimization_direction"),
                validation_requirements.get("direction"),
                validation_requirements.get("metric_direction"),
            ]
        )
    if isinstance(evaluation_spec, dict):
        candidates.extend(
            [
                evaluation_spec.get("optimization_direction"),
                evaluation_spec.get("direction"),
                evaluation_spec.get("metric_direction"),
            ]
        )
    for candidate in candidates:
        normalized = normalize_optimization_direction(candidate)
        if normalized != "unspecified":
            return normalized
    return "unspecified"


def _coerce_optimization_tie_breakers_from_contract(
    optimization_policy: Any,
    validation_requirements: Any,
    evaluation_spec: Any,
) -> List[Dict[str, Any]]:
    for source in (optimization_policy, validation_requirements, evaluation_spec):
        if not isinstance(source, dict):
            continue
        for key in ("tie_breakers", "tie_breaker_policy", "secondary_ordering"):
            normalized = normalize_optimization_tie_breakers(source.get(key))
            if normalized:
                return normalized
    return []


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


def _close_truncated_json_tail(text: str) -> str:
    if not isinstance(text, str):
        return ""
    repaired = text.rstrip()
    if not repaired:
        return repaired

    stack: List[str] = []
    in_str = False
    escape = False
    for ch in repaired:
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
            continue
        if ch in "{[":
            stack.append(ch)
            continue
        if ch == "}" and stack and stack[-1] == "{":
            stack.pop()
            continue
        if ch == "]" and stack and stack[-1] == "[":
            stack.pop()
            continue

    if in_str:
        repaired += '"'

    closing: List[str] = []
    while stack:
        opener = stack.pop()
        closing.append("}" if opener == "{" else "]")
    if closing:
        repaired += "".join(closing)

    repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
    return repaired


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

    repaired = _close_truncated_json_tail(repaired)

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
    data_profile: Dict[str, Any] | None = None,
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

    # Build numeric column set from data_profile for all_numeric_except support
    _numeric_type_tokens = {"float64", "float32", "int64", "int32", "int16", "int8", "numeric", "float", "int"}
    _profile_dtypes: Dict[str, str] = {}
    if isinstance(data_profile, dict):
        _raw_dtypes = data_profile.get("dtypes")
        if isinstance(_raw_dtypes, dict):
            _profile_dtypes = {str(k): str(v).lower() for k, v in _raw_dtypes.items()}
    _numeric_inventory_cols: set[str] = set()
    for _col in inventory:
        _dtype = _profile_dtypes.get(_col, "")
        if any(tok in _dtype for tok in _numeric_type_tokens):
            _numeric_inventory_cols.add(_col)

    def _parse_hint(hint: str) -> List[str]:
        text = str(hint or "").strip()
        if not text:
            return []
        cols: List[str] = []

        # all_numeric_except / all_columns_except: "all_numeric_except excluding col1, col2, ..."
        _all_except_match = re.match(
            r"all_(numeric|columns?)_?except\b[:\s]*(.*)",
            text,
            flags=re.IGNORECASE,
        )
        if _all_except_match:
            _mode = _all_except_match.group(1).lower()
            _except_text = _all_except_match.group(2)
            # Strip leading "excluding" or "except"
            _except_text = re.sub(r"^\s*(excluding|except)\s*", "", _except_text, flags=re.IGNORECASE)
            _except_names = {s.strip() for s in _except_text.split(",") if s.strip()}
            _base = _numeric_inventory_cols if "numeric" in _mode else inventory_set
            cols.extend(col for col in inventory if col in _base and col not in _except_names)
            return list(dict.fromkeys(cols))

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
        # Check both quoted patterns and the raw text for regex metacharacters.
        _regex_meta_chars = {"[", "]", "(", ")", "^", "$", "+", "*", "\\", "|"}
        quoted_patterns = re.findall(r"['\"]([^'\"]+)['\"]", text)
        for candidate in quoted_patterns:
            if any(ch in candidate for ch in _regex_meta_chars):
                regex_candidate = candidate
                regex_candidate = re.sub(r"\[\d+\s*-\s*\d+\]", r"\\d+", regex_candidate)
                cols.extend(_match_regex(rf"^{regex_candidate}$"))

        # Unquoted regex: if the hint itself looks like a regex (starts with ^
        # or contains |, grouping, etc.) and no matches yet, try it directly.
        if not cols and any(ch in text for ch in _regex_meta_chars):
            # Avoid false positives: only if it looks like a standalone pattern
            # (no commas suggesting CSV, no long prose)
            _stripped = text.strip()
            if "," not in _stripped and len(_stripped) < 100:
                cols.extend(_match_regex(_stripped))

        # CSV-style comma-separated column names: "col_a, col_b, col_c"
        if not cols and "," in text:
            _csv_parts = [p.strip() for p in text.split(",") if p.strip()]
            # Accept as CSV if most parts are exact inventory matches
            _csv_matched = [p for p in _csv_parts if p in inventory_set]
            if len(_csv_matched) >= len(_csv_parts) * 0.5:
                cols.extend(_csv_matched)

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
    data_profile: Dict[str, Any] | None = None,
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
    family_cols_expanded = _expand_strategy_feature_families(strategy_dict, inventory, data_profile=data_profile)

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



def _synthesize_selectors_from_allowed_feature_sets(contract: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Synthesize required_feature_selectors from allowed_feature_sets selector
    references and column_roles.  When allowed_feature_sets.model_features
    contains tokens like ``selector:NUMERIC_PREDICTORS``, we resolve them
    against column_roles.pre_decision (the canonical source for predictor
    columns in the contract).
    """
    if not isinstance(contract, dict):
        return []
    allowed = contract.get("allowed_feature_sets")
    if not isinstance(allowed, dict):
        return []

    # Collect selector:NAME references from model_features / segmentation_features
    selector_refs: Dict[str, str] = {}
    for key in ("model_features", "segmentation_features"):
        entries = allowed.get(key)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            token = str(entry or "").strip()
            if token.lower().startswith("selector:"):
                name = token.split(":", 1)[1].strip()
                if name:
                    selector_refs[name] = token

    if not selector_refs:
        return []

    # Resolve each selector name against column_roles
    column_roles = contract.get("column_roles")
    if not isinstance(column_roles, dict):
        return []

    # Build a map from selector family names to column lists.
    # Convention: NUMERIC_PREDICTORS → pre_decision columns.
    # For other names, attempt heuristic mapping via role buckets.
    _FAMILY_TO_ROLES: Dict[str, List[str]] = {
        "numeric_predictors": ["pre_decision"],
        "predictors": ["pre_decision"],
        "features": ["pre_decision"],
        "targets": ["outcome"],
        "multi_horizon_targets": ["outcome"],
        "auxiliary_signals": ["post_decision_audit_only"],
    }

    synthesized: List[Dict[str, Any]] = []
    for name in selector_refs:
        name_lower = name.lower().replace("-", "_").replace(" ", "_")
        role_keys = _FAMILY_TO_ROLES.get(name_lower)
        if not role_keys:
            # Try partial match
            for family_key, roles in _FAMILY_TO_ROLES.items():
                if family_key in name_lower or name_lower in family_key:
                    role_keys = roles
                    break
        if not role_keys:
            continue

        columns: List[str] = []
        for role_key in role_keys:
            role_cols = column_roles.get(role_key)
            if isinstance(role_cols, list):
                for col in role_cols:
                    col_str = str(col or "").strip()
                    if col_str and col_str not in columns:
                        columns.append(col_str)

        if columns:
            synthesized.append({
                "type": "list",
                "columns": columns,
                "name": name,
            })

    return synthesized



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
    clean_dataset = None
    if isinstance(artifact_requirements, dict):
        clean_dataset = artifact_requirements.get("cleaned_dataset")
        if not isinstance(clean_dataset, dict):
            clean_dataset = artifact_requirements.get("clean_dataset")
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

    authoritative_target_columns = _collect_authoritative_target_columns(contract)
    if authoritative_target_columns:
        target_columns = list(authoritative_target_columns)
    else:
        outcome_columns = _collect_targetish_columns(contract.get("outcome_columns"))
        target_columns = list(outcome_columns)
        for col in _collect_contract_target_candidates(contract):
            if col not in target_columns:
                target_columns.append(col)
    primary_target = ""
    for source in (
        authoritative_target_columns,
        (contract.get("task_semantics") or {}).get("primary_target") if isinstance(contract.get("task_semantics"), dict) else None,
        contract.get("primary_target"),
        contract.get("target_column"),
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

    contract_for_artifact_resolution = (
        full_contract_or_partial
        if isinstance(full_contract_or_partial, dict)
        else contract
    )

    def _infer_owned_dataset_output_path(intent: str, basename_tokens: List[str]) -> str:
        if isinstance(contract_for_artifact_resolution, dict):
            declared = get_declared_artifact_path_by_intent(
                contract_for_artifact_resolution,
                intent,
                owner="data_engineer",
                required_only=True,
            )
            if declared:
                return declared
        lowered_tokens = [str(token).strip().lower() for token in basename_tokens if str(token).strip()]
        for raw_path in required_outputs:
            normalized = _normalize_artifact_path(raw_path)
            lower = normalized.lower()
            if not lower.endswith(".csv"):
                continue
            if any(token in lower for token in lowered_tokens):
                return normalized
        return ""

    # P1.5: Infer feature selectors for wide datasets
    feature_selectors = []
    if len(canonical_columns) > 200:
        feature_selectors, remaining_cols = infer_feature_selectors(
            canonical_columns, max_list_size=200, min_group_size=50
        )
        if feature_selectors:
            print(f"FEATURE_SELECTORS: Inferred {len(feature_selectors)} selectors for {len(canonical_columns)} columns")

    declared_feature_selectors: List[Dict[str, Any]] = []
    full_cleaned_dataset = (
        get_dataset_artifact_binding(contract_for_artifact_resolution, "cleaned_dataset")
        if isinstance(contract_for_artifact_resolution, dict)
        else {}
    )
    full_enriched_dataset = (
        get_dataset_artifact_binding(contract_for_artifact_resolution, "enriched_dataset")
        if isinstance(contract_for_artifact_resolution, dict)
        else {}
    )
    if isinstance(full_cleaned_dataset, dict) and full_cleaned_dataset.get("required_feature_selectors") is not None:
        selector_probe = {
            "artifact_requirements": {
                "cleaned_dataset": {
                    "required_feature_selectors": copy.deepcopy(full_cleaned_dataset.get("required_feature_selectors"))
                }
            }
        }
        selector_probe = apply_contract_schema_registry_repairs(selector_probe)
        declared_feature_selectors = (
            selector_probe.get("artifact_requirements", {})
            .get("cleaned_dataset", {})
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
        get_clean_dataset_output_path(contract_for_artifact_resolution)
        if isinstance(contract_for_artifact_resolution, dict)
        else ""
    )
    if not inherited_clean_output_path:
        inherited_clean_output_path = _infer_owned_dataset_output_path(
            "cleaned_dataset",
            ["cleaned", "clean_dataset", "dataset_cleaned", "dataset_limpio"],
        )
    inherited_clean_manifest_path = (
        get_declared_artifact_path(
            contract_for_artifact_resolution,
            "cleaning_manifest.json",
            owner="data_engineer",
            kind="manifest",
        )
        if isinstance(contract_for_artifact_resolution, dict)
        else ""
    )
    inherited_enriched_output_path = (
        get_enriched_dataset_output_path(contract_for_artifact_resolution)
        if isinstance(contract_for_artifact_resolution, dict)
        else ""
    )
    if not inherited_enriched_output_path:
        inherited_enriched_output_path = _infer_owned_dataset_output_path(
            "enriched_dataset",
            ["enriched", "dataset_enriched", "dataset_enriquecido", "model_ready", "handoff"],
        )
    inherited_enriched_required_columns: List[str] = []
    if isinstance(full_cleaned_dataset, dict):
        required_raw = full_cleaned_dataset.get("required_columns")
        if isinstance(required_raw, list):
            inherited_required_columns = [
                str(col).strip() for col in required_raw if str(col).strip()
            ]
        optional_raw = full_cleaned_dataset.get("optional_passthrough_columns")
        if isinstance(optional_raw, list):
            inherited_optional_passthrough = [
                str(col).strip() for col in optional_raw if str(col).strip()
            ]
        transforms_raw = full_cleaned_dataset.get("column_transformations")
        if isinstance(transforms_raw, dict):
            inherited_column_transformations = copy.deepcopy(transforms_raw)
        output_path = str(full_cleaned_dataset.get("output_path") or "").strip()
        if output_path:
            inherited_clean_output_path = output_path
        manifest_path = str(
            full_cleaned_dataset.get("output_manifest_path")
            or full_cleaned_dataset.get("manifest_path")
            or ""
        ).strip()
        if manifest_path:
            inherited_clean_manifest_path = manifest_path
    if isinstance(full_enriched_dataset, dict):
        enriched_required_raw = full_enriched_dataset.get("required_columns")
        if isinstance(enriched_required_raw, list):
            inherited_enriched_required_columns = [
                str(col).strip() for col in enriched_required_raw if str(col).strip()
            ]
        output_path = str(full_enriched_dataset.get("output_path") or "").strip()
        if output_path:
            inherited_enriched_output_path = output_path

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

    enriched_required_columns: List[str] = []
    if inherited_enriched_required_columns:
        for col in inherited_enriched_required_columns:
            resolved = canonical_norms.get(_normalize_column_identifier(col)) or col
            if _normalize_column_identifier(resolved) in dropped_constant_norms and resolved not in outcome_cols:
                continue
            if resolved not in enriched_required_columns:
                enriched_required_columns.append(resolved)
    else:
        future_target_columns = []
        future_ml_handoff = contract.get("future_ml_handoff")
        if isinstance(future_ml_handoff, dict):
            target_cols = future_ml_handoff.get("target_columns")
            if isinstance(target_cols, list):
                future_target_columns = [str(col).strip() for col in target_cols if str(col).strip()]
        if not future_target_columns:
            future_target_columns = [str(col).strip() for col in outcome_cols if str(col).strip()]
        for col in model_features + future_target_columns:
            resolved = canonical_norms.get(_normalize_column_identifier(col)) or col
            if not resolved:
                continue
            if resolved not in enriched_required_columns:
                enriched_required_columns.append(resolved)

    artifact_requirements = {
        "cleaned_dataset": {
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
    if inherited_enriched_output_path or enriched_required_columns:
        artifact_requirements["enriched_dataset"] = {
            "required_columns": enriched_required_columns,
            "output_path": inherited_enriched_output_path,
        }
    if inherited_column_transformations:
        artifact_requirements["cleaned_dataset"]["column_transformations"] = inherited_column_transformations
    if isinstance(full_artifact_requirements.get("clean_dataset"), dict):
        artifact_requirements["clean_dataset"] = copy.deepcopy(full_artifact_requirements.get("clean_dataset"))
    elif "cleaned_dataset" in artifact_requirements:
        artifact_requirements["clean_dataset"] = copy.deepcopy(artifact_requirements["cleaned_dataset"])

    data_engineer_runbook_lines = [
        f"Produce {inherited_clean_output_path} containing ONLY artifact_requirements.cleaned_dataset.required_columns.",
        "Your cleaned_dataset CSV must match EXACTLY the cleaned_dataset.required_columns list - no more, no less.",
        "If a column exists in raw data but is NOT in cleaned_dataset.required_columns, DISCARD it from cleaned_dataset unless it is explicitly declared as optional_passthrough_columns for that artifact.",
        "Constant columns (single unique value) have been pre-excluded from cleaned_dataset.required_columns.",
        "Preserve column names; do not invent or rename columns.",
        f"Load using output_dialect from {inherited_clean_manifest_path} when available.",
        "Parse numeric/date fields conservatively; document conversions.",
        "If a required column is missing from input, report and stop (no fabrication).",
        "Do not derive targets or train models.",
        "Avoid advanced validation metrics (MAE/correlation); report only dtype and null counts.",
        f"Write {inherited_clean_manifest_path} with input/output dialect details.",
    ]
    if inherited_enriched_output_path:
        data_engineer_runbook_lines.extend(
            [
                f"Produce {inherited_enriched_output_path} as a DISTINCT enriched/model-ready artifact when declared.",
                "The enriched_dataset schema must match artifact_requirements.enriched_dataset.required_columns exactly.",
                "Do not carry cleaned_dataset optional_passthrough_columns into enriched_dataset unless artifact_requirements.enriched_dataset explicitly declares them.",
            ]
        )
    data_engineer_runbook = "\n".join(data_engineer_runbook_lines)
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

    authoritative_target_columns = _collect_authoritative_target_columns(contract)
    resolved_target_columns = list(
        dict.fromkeys([col for col in authoritative_target_columns if str(col).strip()])
    )
    if not resolved_target_columns:
        resolved_target_columns = list(dict.fromkeys([col for col in outcome_cols if str(col).strip()]))
    resolved_primary_target = None
    for source in (
        authoritative_target_columns,
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
    explicit_optimization_direction = _coerce_optimization_direction_from_contract(
        optimization_policy,
        validation_requirements,
        evaluation_spec,
    )
    if explicit_optimization_direction != "unspecified":
        optimization_policy["optimization_direction"] = explicit_optimization_direction
    explicit_tie_breakers = _coerce_optimization_tie_breakers_from_contract(
        optimization_policy,
        validation_requirements,
        evaluation_spec,
    )
    if explicit_tie_breakers:
        optimization_policy["tie_breakers"] = explicit_tie_breakers

    active_workstreams = resolve_contract_active_workstreams(contract)
    scope = derive_contract_scope_from_workstreams({**contract, "active_workstreams": active_workstreams})

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
        "active_workstreams": active_workstreams,
        "future_ml_handoff": (
            copy.deepcopy(contract.get("future_ml_handoff"))
            if isinstance(contract.get("future_ml_handoff"), dict)
            else {}
        ),
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
    task_semantics = _synthesize_task_semantics(contract_min)
    if task_semantics:
        contract_min["task_semantics"] = task_semantics
    return contract_min


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

    clean_dataset = artifact_requirements.get("cleaned_dataset")
    if not isinstance(clean_dataset, dict):
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
    """Minimal fallback — returns an empty enabled spec.

    Plot selection is now LLM-driven: the execution planner reasons about which
    visualizations each agent should produce based on the strategy context.
    This function only signals that visuals are structurally allowed.
    """
    return {"enabled": True, "max_plots": 0, "plots": []}


def _build_visual_requirements(
    contract: Dict[str, Any],
    strategy: Dict[str, Any],
    business_objective: str,
) -> Dict[str, Any]:
    """Minimal visual requirements — LLM-driven visualization selection.

    The execution planner LLM now reasons about which plots each agent should
    produce. This function only provides structural defaults and preserves
    any explicit LLM-provided visual config from the contract.
    """
    existing_visual = (
        contract.get("artifact_requirements", {}).get("visual_requirements")
        if isinstance(contract.get("artifact_requirements"), dict)
        else None
    )
    if isinstance(existing_visual, dict) and existing_visual:
        return existing_visual

    outputs_dir = "static/plots"
    artifact_reqs = contract.get("artifact_requirements")
    if isinstance(artifact_reqs, dict):
        outputs_dir = artifact_reqs.get("visual_outputs_dir") or artifact_reqs.get("outputs_dir") or outputs_dir

    return {
        "enabled": True,
        "required": False,
        "outputs_dir": outputs_dir,
        "items": [],
        "notes": "Visualization selection is LLM-driven. The execution planner specifies which plots each agent should generate.",
        "plot_spec": build_plot_spec(contract),
    }


def _ensure_contract_visual_policy(
    contract: Dict[str, Any],
    strategy: Dict[str, Any] | None,
    business_objective: str | None,
) -> Dict[str, Any]:
    """Ensure contracts expose visual requirements through canonical fields.

    Preserves explicit LLM-provided visual config. Only fills structural
    defaults when the LLM didn't specify visualization requirements.
    """
    if not isinstance(contract, dict):
        return contract
    active_workstreams = resolve_contract_active_workstreams(contract)
    if not bool(active_workstreams.get("model_training")):
        return contract

    strategy_payload = strategy if isinstance(strategy, dict) else {}
    objective_text = str(
        business_objective or contract.get("business_objective") or ""
    )

    visual_reqs = _build_visual_requirements(contract, strategy_payload, objective_text)

    artifact_reqs = contract.get("artifact_requirements")
    if not isinstance(artifact_reqs, dict):
        artifact_reqs = {}
    artifact_reqs["visual_requirements"] = visual_reqs
    contract["artifact_requirements"] = artifact_reqs

    policy = contract.get("reporting_policy")
    if not isinstance(policy, dict):
        execution_plan = contract.get("execution_plan")
        policy = build_reporting_policy(
            execution_plan if isinstance(execution_plan, dict) else {},
            strategy_payload,
            contract,
        )
    policy = dict(policy)
    policy["plot_spec"] = visual_reqs.get("plot_spec", {"enabled": True, "max_plots": 0, "plots": []})
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
            resolved_api_key = os.getenv("OPENROUTER_API_KEY")
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
        self.model_name = (
            os.getenv("EXECUTION_PLANNER_PRIMARY_MODEL")
            or os.getenv("EXECUTION_PLANNER_MODEL")
            or "google/gemini-3.1-pro-preview"
        ).strip()
        if not self.model_name:
            self.model_name = "google/gemini-3.1-pro-preview"
        self.base_url = None
        if not self.api_key:
            self.client = None
        else:
            self.client = _OpenRouterAdapter(self.api_key, self.model_name)
        chain_raw = os.getenv("EXECUTION_PLANNER_MODEL_CHAIN", "")
        chain: List[str] = [self.model_name]
        if isinstance(chain_raw, str) and chain_raw.strip():
            for token in chain_raw.split(","):
                model = token.strip()
                if model and model not in chain:
                    chain.append(model)
        self.model_chain = chain
        self._default_model_chain = list(chain)
        # Task B (contract compilation) defaults to Flash to keep the
        # semantic pass on Pro while preserving a lower-cost compiler path.
        # The UI/runtime can still override this explicitly.
        _compiler_model_raw = (
            os.getenv("EXECUTION_PLANNER_COMPILER_MODEL", "").strip()
        )
        self.compiler_model_name: str = _compiler_model_raw or _DEFAULT_EXECUTION_PLANNER_COMPILER_MODEL
        self.last_prompt = None
        self.last_response = None
        self.last_contract_diagnostics = None
        self.last_semantic_core = None
        self.last_llm_call_trace = None

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
        return config

    @staticmethod
    def _build_gemini_function_tool(name: str, description: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "functionDeclarations": [
                {
                    "name": name,
                    "description": description,
                    "parametersJsonSchema": copy.deepcopy(schema),
                }
            ]
        }

    @staticmethod
    def _build_gemini_tool_config(name: str) -> Dict[str, Any]:
        return {
            "functionCallingConfig": {
                "mode": "ANY",
                "allowedFunctionNames": [name],
            }
        }

    @staticmethod
    def _coerce_provider_text(value: Any, depth: int = 0) -> str:
        if depth > 8 or value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (int, float, bool)):
            return str(value)
        if isinstance(value, dict):
            for key in (
                "text",
                "content",
                "output_text",
                "reasoning",
                "reasoning_content",
                "value",
                "arguments",
            ):
                if key not in value:
                    continue
                text = ExecutionPlannerAgent._coerce_provider_text(value.get(key), depth + 1)
                if text:
                    return text
            for key in ("tool_calls", "parts", "items", "output"):
                nested = value.get(key)
                if isinstance(nested, list):
                    parts = [ExecutionPlannerAgent._coerce_provider_text(item, depth + 1) for item in nested]
                    joined = "\n".join(part for part in parts if part).strip()
                    if joined:
                        return joined
            for key in ("message", "delta", "function", "function_call", "functionCall", "parsed", "response"):
                if key not in value:
                    continue
                text = ExecutionPlannerAgent._coerce_provider_text(value.get(key), depth + 1)
                if text:
                    return text
            for nested in value.values():
                text = ExecutionPlannerAgent._coerce_provider_text(nested, depth + 1)
                if text:
                    return text
            return ""
        if isinstance(value, (list, tuple)):
            parts = [ExecutionPlannerAgent._coerce_provider_text(item, depth + 1) for item in value]
            return "\n".join(part for part in parts if part).strip()
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            try:
                text = ExecutionPlannerAgent._coerce_provider_text(model_dump(exclude_none=True), depth + 1)
                if text:
                    return text
            except Exception:
                pass
        for attr in (
            "text",
            "content",
            "output_text",
            "reasoning",
            "reasoning_content",
            "value",
            "arguments",
            "parsed",
            "message",
            "delta",
            "function",
            "function_call",
            "functionCall",
            "parts",
        ):
            try:
                raw = getattr(value, attr)
            except Exception:
                continue
            text = ExecutionPlannerAgent._coerce_provider_text(raw, depth + 1)
            if text:
                return text
        return ""

    @staticmethod
    def _build_explicit_transport_failure(
        rule: str,
        message: str,
        *,
        phase: str,
        item: Any = None,
    ) -> Dict[str, Any]:
        return {
            "status": "error",
            "accepted": False,
            "issues": [
                {
                    "severity": "error",
                    "rule": rule,
                    "message": message,
                    "item": item,
                }
            ],
            "summary": {"error_count": 1, "warning_count": 0, "phase": phase},
        }

    @staticmethod
    def _extract_openai_response_text(response: Any) -> str:
        if response is None:
            return ""
        try:
            function_calls = getattr(response, "function_calls", None)
            if isinstance(function_calls, list) and function_calls:
                first_call = function_calls[0]
                args = getattr(first_call, "args", None)
                if isinstance(args, dict):
                    return json.dumps(args, ensure_ascii=False)
                if isinstance(args, str) and args.strip():
                    return args.strip()
        except Exception:
            pass
        try:
            candidates = getattr(response, "candidates", None) or []
            if candidates:
                first_candidate = candidates[0]
                content = getattr(first_candidate, "content", None)
                parts = getattr(content, "parts", None) if content is not None else None
                if isinstance(parts, list):
                    for part in parts:
                        function_call = getattr(part, "function_call", None) or getattr(part, "functionCall", None)
                        if function_call is not None:
                            args = getattr(function_call, "args", None)
                            if isinstance(args, dict):
                                return json.dumps(args, ensure_ascii=False)
                            if isinstance(args, str) and args.strip():
                                return args.strip()
        except Exception:
            pass
        try:
            choices = getattr(response, "choices", None)
            if isinstance(choices, list) and choices:
                message = getattr(choices[0], "message", None)
                tool_calls = getattr(message, "tool_calls", None) if message is not None else None
                if tool_calls:
                    function_payload = getattr(tool_calls[0], "function", None)
                    arguments = getattr(function_payload, "arguments", None)
                    if isinstance(arguments, str) and arguments.strip():
                        return arguments.strip()
                content = getattr(message, "content", None) if message is not None else None
                text = ExecutionPlannerAgent._coerce_provider_text(content)
                if text:
                    return text
                for attr in ("reasoning", "reasoning_content", "parsed"):
                    text = ExecutionPlannerAgent._coerce_provider_text(getattr(message, attr, None))
                    if text:
                        return text
        except Exception:
            pass
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()
        raw_body = getattr(response, "_codex_raw_body", None)
        if isinstance(raw_body, str) and raw_body.strip():
            try:
                parsed_raw = json.loads(raw_body)
            except Exception:
                parsed_raw = raw_body
            text = ExecutionPlannerAgent._coerce_provider_text(parsed_raw)
            if text:
                return text
        model_dump = getattr(response, "model_dump", None)
        if callable(model_dump):
            try:
                dumped = model_dump(exclude_none=True)
            except Exception:
                dumped = None
            text = ExecutionPlannerAgent._coerce_provider_text(dumped)
            if text:
                return text
        return ""

    @staticmethod
    def _extract_completion_tokens(response: Any, usage_metadata: Any = None) -> int:
        if isinstance(usage_metadata, dict):
            try:
                return int(usage_metadata.get("completion_tokens") or 0)
            except Exception:
                pass
        usage = getattr(response, "usage", None) or getattr(response, "usage_metadata", None)
        if usage is not None:
            try:
                return int(getattr(usage, "completion_tokens", None) or 0)
            except Exception:
                pass
            if isinstance(usage, dict):
                try:
                    return int(usage.get("completion_tokens") or 0)
                except Exception:
                    pass
        return 0

    @staticmethod
    def _extract_openai_finish_reason(response: Any) -> Any:
        try:
            candidates = getattr(response, "candidates", None) or []
            if candidates:
                first_candidate = candidates[0]
                return getattr(first_candidate, "finish_reason", None) or getattr(first_candidate, "finishReason", None)
        except Exception:
            return None
        try:
            choices = getattr(response, "choices", None)
            if isinstance(choices, list) and choices:
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
            used_config = dict(generation_config or {})
            used_config.pop("response_schema", None)
            used_config.pop("tools", None)
            used_config.pop("tool_config", None)
            try:
                response = model_client.generate_content(prompt, generation_config=used_config)
            except TypeError:
                response = model_client.generate_content(prompt)
            except Exception:
                raise
            return response, used_config

        raise TypeError("Execution planner model client must expose generate_content.")

    def _build_model_client(self, model_name: str) -> Any:
        if not self.api_key:
            return None
        if not isinstance(self.client, _OpenRouterAdapter):
            return self.client
        if str(model_name or "").strip() == str(self.client.model_name or "").strip():
            return self.client
        return _OpenRouterAdapter(self.api_key, model_name)

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
            data_profile=data_profile,
        )
        relevant_columns = relevant_payload.get("relevant_columns", [])
        relevant_sources = relevant_payload.get("relevant_sources", {})
        omitted_columns_policy = relevant_payload.get("omitted_columns_policy", "")
        # NOTE: relevant_columns_compact, strategy_feature_family_hints,
        # relevant_columns_total_count, etc. were removed — no longer injected
        # into user_input or compiler_support_context after context cleanup.

        planner_dir = None
        if run_id:
            run_dir = get_run_dir(run_id)
            if run_dir:
                planner_dir = os.path.join(run_dir, "agents", "execution_planner")
                os.makedirs(planner_dir, exist_ok=True)

        planner_diag: List[Dict[str, Any]] = []
        planner_llm_call_trace: List[Dict[str, Any]] = []
        adjudication_cache: Dict[str, Dict[str, Any]] = {}
        self.last_planner_diag = planner_diag
        self.last_llm_call_trace = planner_llm_call_trace
        self.last_contract_min = None
        self.last_contract_canonical = None
        planner_candidate_invalid: Dict[str, Any] | None = None
        planner_candidate_invalid_raw: str | None = None
        planner_candidate_invalid_meta: Dict[str, Any] | None = None
        planner_semantic_core: Dict[str, Any] | None = None
        planner_contract_canonical: Dict[str, Any] | None = None
        last_semantic_transport_validation: Dict[str, Any] | None = None

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

        def _prompt_fingerprint(prompt_text: str) -> str:
            text = prompt_text if isinstance(prompt_text, str) else ""
            return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]

        def _record_llm_call(
            stage: str,
            *,
            model_name: str,
            prompt_text: str,
            response_text: Optional[str] = None,
            usage_metadata: Any = None,
            cached: bool = False,
            extra: Optional[Dict[str, Any]] = None,
            response_obj: Any = None,
        ) -> None:
            entry: Dict[str, Any] = {
                "stage": str(stage or "").strip() or "unknown",
                "model_name": str(model_name or "").strip(),
                "prompt_char_len": len(prompt_text or ""),
                "prompt_fingerprint": _prompt_fingerprint(prompt_text or ""),
                "response_char_len": len(response_text or ""),
                "cached": bool(cached),
            }
            normalized_usage = _normalize_usage_metadata(usage_metadata)
            if isinstance(normalized_usage, dict):
                entry["usage_metadata"] = normalized_usage
            transport_mode = getattr(response_obj, "_codex_transport_mode", None)
            if transport_mode:
                entry["transport_mode"] = str(transport_mode)
            transport_max_retries = getattr(response_obj, "_codex_transport_max_retries", None)
            if isinstance(transport_max_retries, int):
                entry["transport_max_retries"] = int(transport_max_retries)
            request_id = (
                getattr(response_obj, "_codex_request_id", None)
                or getattr(response_obj, "_request_id", None)
            )
            if request_id:
                entry["request_id"] = str(request_id)
            if isinstance(extra, dict) and extra:
                entry.update(_safe_json_serializable(extra))
            planner_llm_call_trace.append(entry)

        def _persist_attempt(prompt_name: str, response_name: str, prompt_text: str, response_text: str | None) -> None:
            if not planner_dir:
                return
            if prompt_text is not None:
                _write_text(os.path.join(planner_dir, prompt_name), prompt_text)
            if response_text is not None:
                _write_text(os.path.join(planner_dir, response_name), response_text)

        def _persist_contracts(
            full_contract: Dict[str, Any] | None,
            semantic_core: Dict[str, Any] | None = None,
            canonical_contract: Dict[str, Any] | None = None,
            projection_contract: Dict[str, Any] | None = None,
            diagnostics_payload: Dict[str, Any] | None = None,
            invalid_contract: Dict[str, Any] | None = None,
            invalid_raw: str | None = None,
            invalid_meta: Dict[str, Any] | None = None,
        ) -> None:
            if not planner_dir:
                return
            if semantic_core:
                _write_json(os.path.join(planner_dir, "semantic_core.json"), semantic_core)
            if canonical_contract:
                _write_json(os.path.join(planner_dir, "contract_canonical.json"), canonical_contract)
            if full_contract:
                _write_json(os.path.join(planner_dir, "contract_full.json"), full_contract)
                _write_json(os.path.join(planner_dir, "contract_raw.json"), full_contract)
            if projection_contract:
                _write_json(os.path.join(planner_dir, "contract_projection.json"), projection_contract)
            if planner_diag:
                _write_json(
                    os.path.join(planner_dir, "planner_diag.json"),
                    {"attempts": planner_diag, "llm_call_trace": planner_llm_call_trace},
                )
            if diagnostics_payload:
                _write_json(
                    os.path.join(planner_dir, "contract_diagnostics.json"),
                    diagnostics_payload,
                )
                _write_json(
                    os.path.join(planner_dir, "contract_validation_report.json"),
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
                "promptTokenCount",
                "candidatesTokenCount",
                "totalTokenCount",
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
            if "promptTokenCount" in usage_payload and "prompt_token_count" not in usage_payload:
                usage_payload["prompt_token_count"] = usage_payload.get("promptTokenCount")
            if "candidatesTokenCount" in usage_payload and "candidates_token_count" not in usage_payload:
                usage_payload["candidates_token_count"] = usage_payload.get("candidatesTokenCount")
            if "totalTokenCount" in usage_payload and "total_token_count" not in usage_payload:
                usage_payload["total_token_count"] = usage_payload.get("totalTokenCount")
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
            payload_for_validation = copy.deepcopy(
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

        def _adjudicate_ambiguous_validation_issues(
            semantic_core_payload: Dict[str, Any] | None,
            compiled_contract: Dict[str, Any] | None,
            validation_result: Dict[str, Any] | None,
            *,
            model_client: Any,
            model_name: str,
            trace_stage: str = "validation_adjudication",
        ) -> Dict[str, Any]:
            adjudicable_issues = _collect_adjudicable_validation_issues(validation_result)
            if not adjudicable_issues or model_client is None:
                return copy.deepcopy(validation_result) if isinstance(validation_result, dict) else {}
            semantic_json = _compress_text_preserve_ends(
                json.dumps(semantic_core_payload or {}, ensure_ascii=False, indent=2),
                max_chars=7000,
                head=4500,
                tail=2500,
            )
            contract_json = _compress_text_preserve_ends(
                json.dumps(compiled_contract or {}, ensure_ascii=False, indent=2),
                max_chars=12000,
                head=8000,
                tail=4000,
            )
            issues_json = json.dumps(adjudicable_issues, ensure_ascii=False, indent=2)
            support_context_payload = {
                "business_objective": business_objective or strategy.get("business_objective") or "",
                "strategy_title": strategy.get("strategy_title") or strategy.get("title") or "",
                "column_inventory_size": len(column_inventory or []),
                "column_inventory_sample": [str(col) for col in (column_inventory or [])[:25] if col],
            }
            prompt = (
                CONTRACT_VALIDATION_ADJUDICATOR_PROMPT
                + "\n\nSEMANTIC_CORE_AUTHORITY_JSON:\n"
                + semantic_json
                + "\n\nCOMPILED_CONTRACT_JSON:\n"
                + contract_json
                + "\n\nAMBIGUOUS_ISSUES_JSON:\n"
                + issues_json
                + "\n\nSUPPORT_CONTEXT:\n"
                + json.dumps(support_context_payload, ensure_ascii=False, indent=2)
            )
            cache_key = hashlib.sha256(
                (
                    str(model_name or "")
                    + "\n"
                    + str(trace_stage or "")
                    + "\n"
                    + prompt
                ).encode("utf-8", errors="replace")
            ).hexdigest()
            cached_payload = adjudication_cache.get(cache_key)
            if isinstance(cached_payload, dict):
                _record_llm_call(
                    trace_stage,
                    model_name=model_name,
                    prompt_text=prompt,
                    response_text=None,
                    usage_metadata=cached_payload.get("_usage_metadata"),
                    cached=True,
                    extra={"adjudicable_issue_count": len(adjudicable_issues)},
                )
                return copy.deepcopy(cached_payload.get("result") or {})
            try:
                response, _ = self._generate_content_with_budget(
                    model_client,
                    prompt,
                    output_token_floor=768,
                    model_name=model_name,
                    tool_mode="contract",
                )
                response_text = self._extract_openai_response_text(response)
                parsed_payload, _ = _parse_json_response(response_text)
                _record_llm_call(
                    trace_stage,
                    model_name=model_name,
                    prompt_text=prompt,
                    response_text=response_text,
                    usage_metadata=getattr(response, "usage", None) or getattr(response, "usage_metadata", None),
                    cached=False,
                    extra={"adjudicable_issue_count": len(adjudicable_issues)},
                    response_obj=response,
                )
                if not isinstance(parsed_payload, dict):
                    return copy.deepcopy(validation_result) if isinstance(validation_result, dict) else {}
                adjudicated = _apply_validation_adjudication(validation_result, parsed_payload)
                if isinstance(adjudicated, dict):
                    summary = dict(adjudicated.get("summary") or {})
                    summary["adjudicator_model"] = model_name
                    adjudicated["summary"] = summary
                    adjudication_cache[cache_key] = {
                        "result": copy.deepcopy(adjudicated),
                        "_usage_metadata": _normalize_usage_metadata(
                            getattr(response, "usage", None) or getattr(response, "usage_metadata", None)
                        ),
                    }
                return adjudicated
            except Exception:
                return copy.deepcopy(validation_result) if isinstance(validation_result, dict) else {}

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
                    "required_outputs must materialize to artifact paths; when preserving semantic deliverables, use object entries with path + intent."
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
            required_top_level = list(EXECUTION_CONTRACT_V5_CANONICAL_REQUIRED_KEYS) + list(V5_AGENT_SECTION_KEYS)
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
            _required_keys_csv = ", ".join(required_top_level)
            return (
                "Repair the previous execution contract by returning the COMPLETE corrected contract.\n"
                "Return ONLY valid JSON (no markdown, no comments, no code fences).\n"
                'Use V5.0 hierarchical structure: contract_version "5.0", shared, data_engineer, ml_engineer, cleaning_reviewer, qa_reviewer, business_translator.\n'
                "Include ALL sections from the previous contract that were valid, plus fix the issues listed below.\n"
                "Every key listed here MUST be present: " + _required_keys_csv + ".\n"
                "Do NOT return a patch — return the full contract with fixes applied.\n"
                "Schema registry examples:\n"
                + CONTRACT_SCHEMA_EXAMPLES_TEXT
                + "\n"
                "scope MUST be inside shared and be one of: cleaning_only, ml_only, full_pipeline.\n"
                "If the run is data cleaning / feature preparation for a future ML run, set shared.active_workstreams.model_training=false and prefer scope=cleaning_only.\n"
                "Do not invent columns outside column_inventory.\n"
                "Use downstream_consumer_interface + evidence_policy from ORIGINAL INPUTS.\n"
                "shared.column_dtype_targets values MUST use key target_dtype (never key type).\n"
                "Example: {\"age\": {\"target_dtype\": \"float64\", \"nullable\": false}}.\n"
                "Do not remove required business intent; fix only structural/semantic contract errors.\n"
                "Preserve business objective, dataset context, and selected strategy from ORIGINAL INPUTS.\n"
                "column_roles MUST use canonical role buckets only: "
                "pre_decision, decision, outcome, post_decision_audit_only, identifiers, time_columns, unknown.\n"
                "For cleaning scopes, define artifact_requirements.cleaned_dataset.output_path and output_manifest_path.\n"
                "If a separate model-ready/enriched dataset is required, declare artifact_requirements.enriched_dataset separately.\n"
                "required_feature_selectors entries must be list[object] and each object must have key type.\n"
                "Example: [{\"type\": \"prefix\", \"value\": \"feature_\"}, {\"type\": \"regex\", \"pattern\": \"^pixel_\\\\d+$\"}].\n"
                "If cleaning requires dropping/scaling columns, declare them in "
                "artifact_requirements.cleaned_dataset.column_transformations.{drop_columns,scale_columns,drop_policy}; "
                "do not leave these decisions only in runbook prose.\n"
                "If required_feature_selectors are used and any drop-by-criteria is requested, define "
                "column_transformations.drop_policy.allow_selector_drops_when with explicit reasons.\n"
                "When scaling a selector-defined family on wide schemas, scale_columns may use selector refs "
                "(regex:/prefix:/suffix:/contains:/selector:<name>) instead of enumerating every column.\n"
                "If selector-drop policy is active, required_columns and optional_passthrough_columns MUST be non-droppable anchors "
                "and therefore must not overlap required_feature_selectors.\n"
                "If selector-drop policy is active, HARD cleaning gates must not rely on selector-covered columns.\n"
                "For wide feature families, you may declare compact selectors in "
                "artifact_requirements.cleaned_dataset.required_feature_selectors (regex/prefix/range/list).\n"
                "Avoid enumerating massive feature families as explicit lists; keep explicit anchors only.\n"
                "Never place wildcard selector tokens (e.g., pixel*) inside required_columns.\n"
                "If strategy/data indicate robust outlier handling, include optional outlier_policy with "
                "enabled/apply_stage/target_columns/report_path/strict.\n"
                "When active_workstreams.model_training=true, include non-empty evaluation_spec and validation_requirements and ensure objective_analysis.problem_type "
                "(or evaluation_spec.objective_type) and non-empty column_roles.\n"
                "When active_workstreams.model_training=true, include artifact_requirements.visual_requirements and reporting_policy.plot_spec "
                "aligned with strategy/evidence so views can request the right visuals.\n"
                "When active_workstreams.model_training=false but future_ml_handoff is in scope, do not invent training/CV sections; instead make future_ml_handoff and enriched_dataset coverage explicit.\n"
                "Gate lists must be executable by downstream views: use gate objects with "
                "{name, severity, params} (severity in HARD|SOFT). If using metric/check/rule language, "
                "map it to name and keep semantic details in params.\n"
                "Gate example: {\"name\": \"no_nulls_target\", \"severity\": \"HARD\", \"params\": {\"column\": \"target\"}}.\n"
                "Canonical top-level minimum keys to preserve: "
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
                return [], [], [], [], ["artifact_requirements.cleaned_dataset.column_transformations must be an object"]

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
                errors.append("artifact_requirements.cleaned_dataset.column_transformations.drop_columns must be list[str]")
            if invalid_scale:
                errors.append("artifact_requirements.cleaned_dataset.column_transformations.scale_columns must be list[str]")
            transform_payload = dict(transform_block)
            if "drop_policy" not in transform_payload and "drop_policy" in clean_dataset:
                transform_payload["drop_policy"] = clean_dataset.get("drop_policy")
            selector_drop_reasons, selector_drop_errors = extract_selector_drop_reasons(transform_payload)
            for issue in selector_drop_errors:
                errors.append(f"artifact_requirements.cleaned_dataset.column_transformations.drop_policy: {issue}")

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
                return [], ["artifact_requirements.cleaned_dataset.required_feature_selectors must be list[object]"]
            selectors: List[Dict[str, Any]] = []
            errors: List[str] = []
            for idx, item in enumerate(raw_selectors):
                if not isinstance(item, dict):
                    errors.append(
                        f"artifact_requirements.cleaned_dataset.required_feature_selectors[{idx}] must be an object"
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
                        f"artifact_requirements.cleaned_dataset.required_feature_selectors[{idx}] is missing type"
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
                errors.append("artifact_requirements.cleaned_dataset.optional_passthrough_columns must be list[str]")
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
            transport_validation = None
            if isinstance(planner_candidate_invalid_meta, dict):
                candidate_transport = planner_candidate_invalid_meta.get("transport_validation")
                if isinstance(candidate_transport, dict):
                    transport_validation = candidate_transport
            if isinstance(transport_validation, dict):
                diagnostics["transport_validation"] = transport_validation
            if not isinstance(contract, dict):
                diagnostics["validation"] = {
                    "status": "error",
                    "issues": [{"severity": "error", "rule": "contract_not_object", "message": "Contract is not a dictionary"}],
                }
                return diagnostics

            if (
                isinstance(transport_validation, dict)
                and not _transport_validation_accepted(transport_validation)
                and (not contract or not any(_is_meaningful_contract_value(value) for value in contract.values()))
            ):
                diagnostics["validation"] = transport_validation
                split_phases = _split_validation_issues_by_phase(transport_validation)
                diagnostics.update(split_phases)
                issues = transport_validation.get("issues") if isinstance(transport_validation, dict) else []
                diagnostics["quality_profile"] = {
                    "top_error_rules": [
                        str(issue.get("rule"))
                        for issue in (issues or [])
                        if isinstance(issue, dict) and str(issue.get("severity") or "").lower() in {"error", "fail"}
                    ][:8],
                    "top_warning_rules": [],
                }
                diagnostics["summary"] = {
                    "status": "error",
                    "issue_count": len(issues) if isinstance(issues, list) else 0,
                    "accepted": False,
                }
                return diagnostics

            try:
                validation_result = _merge_validation_results(
                    _validate_contract_quality(copy.deepcopy(contract)),
                    _build_semantic_guard_validation(planner_semantic_core, contract),
                )
                diagnostics_model_client = self._build_model_client(self.model_name) if self.client else None
                validation_result = _adjudicate_ambiguous_validation_issues(
                    planner_semantic_core,
                    contract,
                    validation_result,
                    model_client=diagnostics_model_client,
                    model_name=self.model_name,
                    trace_stage="contract_diagnostics_adjudication",
                )
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
            diagnostics.update(_split_validation_issues_by_phase(validation_result))
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
            projection_contract = None
            if (
                isinstance(contract, dict)
                and contract
                and any(_is_meaningful_contract_value(value) for value in contract.values())
            ):
                projection_contract = _apply_planner_structural_support(contract)

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
                        "Keep required_outputs as artifact outputs; if rich object entries already exist, preserve them.\n"
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

                post_validation = _merge_validation_results(
                    _validate_contract_quality(contract),
                    _build_semantic_guard_validation(planner_semantic_core, contract),
                )
                post_validation = _adjudicate_ambiguous_validation_issues(
                    planner_semantic_core,
                    contract,
                    post_validation,
                    model_client=self._build_model_client(self.model_name) if self.client else None,
                    model_name=self.model_name,
                    trace_stage="finalize_post_validation_adjudication",
                )
            diagnostics = _build_contract_diagnostics(contract if isinstance(contract, dict) else {}, where, llm_success)
            if isinstance(diagnostics, dict):
                diagnostics["llm_call_trace"] = copy.deepcopy(planner_llm_call_trace)
            self.last_contract_diagnostics = diagnostics
            self.last_llm_call_trace = copy.deepcopy(planner_llm_call_trace)
            _persist_contracts(
                contract if isinstance(contract, dict) else {},
                semantic_core=planner_semantic_core,
                canonical_contract=planner_contract_canonical,
                projection_contract=projection_contract,
                diagnostics_payload=diagnostics,
                invalid_contract=planner_candidate_invalid,
                invalid_raw=planner_candidate_invalid_raw,
                invalid_meta=planner_candidate_invalid_meta,
            )
            self.last_semantic_core = (
                copy.deepcopy(planner_semantic_core)
                if isinstance(planner_semantic_core, dict)
                else None
            )
            self.last_contract_canonical = (
                copy.deepcopy(planner_contract_canonical)
                if isinstance(planner_contract_canonical, dict)
                else None
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
        # column_inventory_compact removed — no longer in compiler_support_context
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
        compiler_data_summary = _compress_text_preserve_ends(
            data_summary_str,
            max_chars=6000,
            head=4000,
            tail=2000,
        )
        compiler_strategy_context: Dict[str, Any] = {}
        if isinstance(strategy, dict):
            for key in (
                "title",
                "objective_type",
                "objective_reasoning",
                "success_metric",
                "validation_strategy",
                "validation_rationale",
                "analysis_type",
                "hypothesis",
                "required_columns",
                "techniques",
                "fallback_chain",
                "reasoning",
            ):
                value = strategy.get(key)
                if value not in (None, "", [], {}):
                    compiler_strategy_context[key] = value
            if strategy_feature_families:
                compiler_strategy_context["feature_families"] = strategy_feature_families
        compiler_dtypes = {}
        if isinstance(data_profile, dict):
            dtypes_candidate = data_profile.get("dtypes")
            if isinstance(dtypes_candidate, dict):
                compiler_dtypes = dtypes_candidate
        # ── Task B support context: lean reinforcement for compiler ──
        # Only what the compiler needs beyond the semantic_core.
        # Meta-policies (source_of_truth, precedence, operational_targets,
        # agent_interface_policy) are already in the prompt or enforced by
        # the closing checklist — no need to repeat in the context payload.
        compiler_support_context_payload = {
            "business_objective": business_objective,
            "strategy_compilation_reinforcement": compiler_strategy_context,
            "column_manifest": {
                "mode": manifest_mode or "none",
                "anchors": manifest_for_prompt.get("anchors", []) if isinstance(manifest_for_prompt, dict) else [],
                "families": manifest_for_prompt.get("families", []) if isinstance(manifest_for_prompt, dict) else [],
            },
            "data_profile_summary": compiler_data_summary,
            "observed_dtype_hints": compiler_dtypes,
            "output_dialect": output_dialect or "unknown",
            "domain_expert_critique": critique_for_prompt or "None",
        }
        compiler_support_context = json.dumps(
            compiler_support_context_payload,
            indent=2,
            ensure_ascii=False,
        )
        # ── Task A user_input: lean context for semantic reasoning ──
        # Single authoritative source per concept. No redundant column views.
        user_input = f"""
strategy:
{strategy_json}

business_objective:
{business_objective}

column_inventory ({column_inventory_count} columns):
{json.dumps(column_inventory_payload, indent=2)}

column_manifest:
{json.dumps(manifest_for_prompt, indent=2)}

column_sets:
{json.dumps(column_sets_payload, indent=2)}

output_dialect:
{json.dumps(output_dialect or "unknown")}

resolved_target:
{json.dumps(resolved_target)}

data_profile_summary:
{data_summary_for_prompt}

domain_expert_critique:
{critique_for_prompt or "None"}
"""

        # Build a scope-aware closing checklist from the canonical schema so the
        # LLM self-verifies completeness before returning.  The checklist is
        # derived from the authoritative required-keys list — never hardcoded
        # for a specific dataset or competition.
        semantic_required_keys_csv = ", ".join(EXECUTION_SEMANTIC_CORE_REQUIRED_KEYS)
        semantic_checklist = (
            "\n\nBEFORE YOU RETURN â€” self-check your semantic_core JSON.\n"
            f"Every key listed here MUST be a top-level key in your output: {semantic_required_keys_csv}.\n"
            "If the current run prepares data/features for future ML, active_workstreams.model_training must stay false.\n"
            "If a future target exists but model_training=false, include future_ml_handoff.\n"
            "Do not emit compilation-only sections like artifact_requirements, iteration_policy, column_dtype_targets, evaluation_spec, or validation_requirements.\n"
        )
        semantic_prompt = (
            SEMANTIC_EXECUTION_PLANNER_PROMPT
            + "\n\nINPUTS:\n"
            + user_input
            + semantic_checklist
        )

        _v5_top_level_keys = ", ".join(EXECUTION_CONTRACT_V5_CANONICAL_REQUIRED_KEYS + list(V5_AGENT_SECTION_KEYS))
        _closing_checklist = (
            "\n\nBEFORE YOU RETURN — self-check your JSON against this V5.0 contract completeness checklist.\n"
            f"These MUST be top-level keys in your output: {_v5_top_level_keys}.\n"
            'contract_version MUST be "5.0".\n'
            "If any top-level section is missing, add it now. Omitting a required section will fail validation.\n"
            "V5.0 hierarchy self-check:\n"
            "- shared: contains scope, strategy_title, business_objective, output_dialect, canonical_columns, "
            "column_roles, allowed_feature_sets, task_semantics, active_workstreams, model_features, "
            "column_dtype_targets, iteration_policy. If any is missing, add it.\n"
            "- data_engineer: contains required_outputs (DE deliverables), cleaning_gates, runbook, artifact_requirements.\n"
            "- ml_engineer: contains required_outputs (ML deliverables), qa_gates, reviewer_gates, runbook. "
            "When model_training=true: also evaluation_spec, validation_requirements, optimization_policy.\n"
            "- cleaning_reviewer, qa_reviewer, business_translator: present even if minimal.\n"
            "Semantic closure self-check:\n"
            "- If targets are declared, ml_engineer.evaluation_spec must name the primary target and label columns.\n"
            "- If a metric appears anywhere, it must match in ml_engineer.evaluation_spec.primary_metric and ml_engineer.validation_requirements.primary_metric.\n"
            "- When model_training=true, ml_engineer.optimization_policy should state optimization_direction explicitly and only include tie_breakers that are justified by the run context.\n"
            "- If model_features imply ML input columns, data_engineer.artifact_requirements.cleaned_dataset.required_columns must cover them.\n"
            "- If anchor columns exist (outcome, identifiers, split, time), shared.column_dtype_targets must include entries for those anchors.\n"
            "- Do not create agent_interfaces — the v5 hierarchy replaces it.\n"
            "- Do not create any failure_explainer section."
        )

        full_prompt = (
            MINIMAL_CONTRACT_COMPILER_PROMPT
            + "\n\nOPERATIONAL SCHEMA EXAMPLES:\n"
            + COMPILER_OPERATIONAL_SCHEMA_EXAMPLES_TEXT
            + "\n\nSUPPORT_CONTEXT:\n"
            + compiler_support_context
            + _closing_checklist
        )
        model_chain = [m for m in (self.model_chain or [self.model_name]) if m]
        # Planner is intentionally one-shot: one semantic pass + one compile pass.
        # Do not fan out across fallback models here; if the first attempt fails,
        # persist the failure and abort so we can improve reasoning quality instead
        # of hiding it behind retries.
        active_model_chain = model_chain[:1]
        explicit_chain_override = list(model_chain) != list(getattr(self, "_default_model_chain", [self.model_name]))
        # Task B (compilation) can use a separate model optimized for
        # instruction-following and JSON output via EXECUTION_PLANNER_COMPILER_MODEL.
        compiler_model_chain = (
            [self.compiler_model_name]
            if self.compiler_model_name and not explicit_chain_override
            else active_model_chain
        )

        contract: Dict[str, Any] | None = None
        llm_success = False
        semantic_success = False
        best_candidate: Dict[str, Any] | None = None
        best_canonical_candidate: Dict[str, Any] | None = None
        best_validation: Dict[str, Any] | None = None
        best_response_text: str | None = None
        best_parse_feedback: str | None = None
        best_error_count: Optional[int] = None
        best_warning_count: Optional[int] = None
        latest_candidate_for_repair: Dict[str, Any] | None = None
        latest_canonical_for_repair: Dict[str, Any] | None = None
        latest_validation_for_repair: Dict[str, Any] | None = None
        latest_response_text_for_repair: str | None = None
        latest_parse_feedback_for_repair: str | None = None

        max_quality_rounds = 1

        current_prompt = semantic_prompt
        current_prompt_name = "semantic_prompt_attempt_1.txt"
        current_tool_mode = "semantic"
        attempt_counter = 0

        semantic_response_text: str | None = None
        if contract is None:
            for model_idx, model_name in enumerate(active_model_chain, start=1):
                attempt_counter += 1
                response_text = ""
                response = None
                parse_error = None
                finish_reason = None
                usage_metadata = None
                generation_config_used: Dict[str, Any] | None = None
                self.last_prompt = current_prompt
                self.last_response = None
                response_name = (
                    f"semantic_response_attempt_1_m{model_idx}.txt"
                    if len(active_model_chain) > 1
                    else "semantic_response_attempt_1.txt"
                )
                try:
                    model_client = self._build_model_client(model_name)
                    if model_client is None:
                        parse_error = ValueError(f"Planner client unavailable for model {model_name}")
                    else:
                        response, generation_config_used = self._generate_content_with_budget(
                            model_client,
                            current_prompt,
                            output_token_floor=2048,
                            model_name=model_name,
                            tool_mode="semantic",
                        )
                        response_text = self._extract_openai_response_text(response)
                        semantic_response_text = response_text
                        self.last_response = response_text
                except Exception as err:
                    parse_error = err

                if response is not None:
                    finish_reason = self._extract_openai_finish_reason(response)
                    usage_metadata = _normalize_usage_metadata(
                        getattr(response, "usage", None) or getattr(response, "usage_metadata", None)
                    )
                    if not str(response_text or "").strip():
                        completion_tokens = self._extract_completion_tokens(response, usage_metadata)
                        if completion_tokens > 0 and parse_error is None:
                            parse_error = ValueError(
                                f"EMPTY_COMPLETION_WITH_TOKENS: finish_reason={finish_reason} completion_tokens={completion_tokens}"
                            )

                _persist_attempt(current_prompt_name, response_name, current_prompt, response_text)
                _record_llm_call(
                    "semantic_core",
                    model_name=model_name,
                    prompt_text=current_prompt,
                    response_text=response_text,
                    usage_metadata=usage_metadata,
                    cached=False,
                    extra={
                        "attempt_index": attempt_counter,
                        "quality_round": 0,
                        "finish_reason": str(finish_reason) if finish_reason is not None else None,
                    },
                    response_obj=response,
                )

                parsed_payload, parse_exc = _parse_json_response(response_text) if response_text else (None, parse_error)
                if parse_exc:
                    parse_error = parse_exc
                parsed_semantic = parsed_payload if isinstance(parsed_payload, dict) else None
                if (
                    not str(response_text or "").strip()
                    and isinstance(parse_error, ValueError)
                    and "EMPTY_COMPLETION_WITH_TOKENS" in str(parse_error)
                ):
                    semantic_validation = ExecutionPlannerAgent._build_explicit_transport_failure(
                        "semantic_core.transport_empty_completion",
                        "Semantic planner response text was empty despite non-zero completion tokens; provider payload could not be extracted.",
                        phase="semantic_transport",
                        item={
                            "finish_reason": str(finish_reason) if finish_reason is not None else None,
                            "completion_tokens": self._extract_completion_tokens(response, usage_metadata),
                        },
                    )
                else:
                    semantic_validation = _build_semantic_core_transport_validation(parsed_semantic)
                last_semantic_transport_validation = semantic_validation
                if parsed_payload is not None and not isinstance(parsed_payload, dict):
                    parse_error = ValueError("Parsed semantic_core JSON is not an object")
                elif not _transport_validation_accepted(semantic_validation):
                    transport_issue = None
                    issues = semantic_validation.get("issues")
                    if isinstance(issues, list):
                        for issue in issues:
                            if isinstance(issue, dict) and issue.get("rule"):
                                transport_issue = str(issue.get("rule"))
                                break
                    parse_error = ValueError(
                        f"Semantic core invalid: {transport_issue or 'missing_or_trivial_semantic_core'}"
                    )
                    parsed_semantic = None

                planner_diag.append(
                    {
                        "stage": "semantic_core",
                        "model_name": model_name,
                        "attempt_index": attempt_counter,
                        "quality_round": 0,
                        "prompt_char_len": len(current_prompt or ""),
                        "response_char_len": len(response_text or ""),
                        "finish_reason": str(finish_reason) if finish_reason is not None else None,
                        "generation_config": _safe_json_serializable(generation_config_used),
                        "usage_metadata": usage_metadata,
                        "had_json_parse_error": not isinstance(parsed_payload, dict),
                        "transport_status": str(semantic_validation.get("status") or "").lower(),
                        "transport_issue_rules": [
                            str(issue.get("rule"))
                            for issue in (semantic_validation.get("issues") or [])
                            if isinstance(issue, dict) and issue.get("rule")
                        ][:12],
                        "parse_error_type": type(parse_error).__name__ if parse_error else None,
                        "parse_error_message": str(parse_error) if parse_error else None,
                        "quality_status": None,
                        "quality_issue_rules": [],
                        "quality_error_count": None,
                        "quality_warning_count": None,
                        "quality_accepted": bool(parsed_semantic),
                        "quality_error": None if parsed_semantic else "semantic_core_invalid",
                    }
                )

                if isinstance(parsed_semantic, dict):
                    planner_semantic_core = copy.deepcopy(parsed_semantic)
                    semantic_success = True
                    break

        if not semantic_success and not isinstance(planner_candidate_invalid_meta, dict):
            planner_candidate_invalid_meta = {
                "reason": "semantic_core_invalid",
                "attempt_count": len(planner_diag),
                "fallback_mode": "semantic_stage_failed",
                "transport_validation": last_semantic_transport_validation,
            }
            planner_candidate_invalid_raw = semantic_response_text

        full_prompt = ""
        if isinstance(planner_semantic_core, dict):
            semantic_authority_json = json.dumps(planner_semantic_core, indent=2, ensure_ascii=False)
            full_prompt = (
                MINIMAL_CONTRACT_COMPILER_PROMPT
                + "\n\nOPERATIONAL SCHEMA EXAMPLES:\n"
                + COMPILER_OPERATIONAL_SCHEMA_EXAMPLES_TEXT
                + "\n\nSEMANTIC_CORE_AUTHORITY_JSON:\n"
                + semantic_authority_json
                + "\n\nSUPPORT_CONTEXT:\n"
                + compiler_support_context
                + _closing_checklist
            )
            current_prompt = full_prompt
            current_prompt_name = "prompt_attempt_1.txt"
            current_tool_mode = "contract"

        if contract is None and semantic_success:
            for quality_round in range(1, max_quality_rounds + 1):
                round_has_candidate = False
                for model_idx, model_name in enumerate(compiler_model_chain, start=1):
                    attempt_counter += 1
                    self.last_prompt = current_prompt
                    response_text = ""
                    response = None
                    parse_error: Optional[Exception] = None
                    finish_reason = None
                    usage_metadata = None
                    generation_config_used = None
                    transport_validation_result: Dict[str, Any] | None = None
                    validation_result: Dict[str, Any] | None = None
                    quality_accepted = False

                    if len(compiler_model_chain) > 1:
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
                                tool_mode=current_tool_mode,
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
                        if not str(response_text or "").strip():
                            completion_tokens = self._extract_completion_tokens(response, usage_metadata)
                            if completion_tokens > 0 and parse_error is None:
                                parse_error = ValueError(
                                    f"EMPTY_COMPLETION_WITH_TOKENS: finish_reason={finish_reason} completion_tokens={completion_tokens}"
                                )

                    _persist_attempt(current_prompt_name, response_name, current_prompt, response_text)
                    _record_llm_call(
                        "contract_patch_repair" if current_tool_mode == "patch" else "contract_compile",
                        model_name=model_name,
                        prompt_text=current_prompt,
                        response_text=response_text,
                        usage_metadata=usage_metadata,
                        cached=False,
                        extra={
                            "attempt_index": attempt_counter,
                            "quality_round": quality_round,
                            "tool_mode": current_tool_mode,
                            "finish_reason": str(finish_reason) if finish_reason is not None else None,
                        },
                        response_obj=response,
                    )

                    parsed_payload, parse_exc = _parse_json_response(response_text) if response_text else (None, parse_error)
                    if parse_exc:
                        parse_error = parse_exc

                    parsed: Dict[str, Any] | None = None
                    had_json_parse_error = parsed_payload is None or not isinstance(parsed_payload, dict)
                    if current_tool_mode == "patch":
                        transport_validation_result = _build_patch_transport_validation(parsed_payload)
                        repair_base = (
                            latest_canonical_for_repair
                            if isinstance(latest_canonical_for_repair, dict)
                            else best_canonical_candidate
                        )
                        if parsed_payload is not None and not isinstance(parsed_payload, dict):
                            parse_error = ValueError("Parsed JSON patch is not an object")
                        elif not isinstance(repair_base, dict) or not repair_base:
                            parse_error = ValueError("Patch repair requested without a canonical base contract")
                        elif not _transport_validation_accepted(transport_validation_result):
                            patch_issue = None
                            issues = transport_validation_result.get("issues")
                            if isinstance(issues, list):
                                for issue in issues:
                                    if isinstance(issue, dict) and issue.get("rule"):
                                        patch_issue = str(issue.get("rule"))
                                        break
                            parse_error = ValueError(
                                f"Patch payload invalid: {patch_issue or 'empty_or_trivial_patch'}"
                            )
                        else:
                            parsed = _apply_minimal_contract_patch(repair_base, parsed_payload)
                            had_json_parse_error = False
                    else:
                        parsed = _unwrap_execution_contract_transport(parsed_payload)
                        if (
                            not str(response_text or "").strip()
                            and isinstance(parse_error, ValueError)
                            and "EMPTY_COMPLETION_WITH_TOKENS" in str(parse_error)
                        ):
                            transport_validation_result = ExecutionPlannerAgent._build_explicit_transport_failure(
                                "contract.transport_empty_completion",
                                "Contract compiler response text was empty despite non-zero completion tokens; provider payload could not be extracted.",
                                phase="transport",
                                item={
                                    "finish_reason": str(finish_reason) if finish_reason is not None else None,
                                    "completion_tokens": self._extract_completion_tokens(response, usage_metadata),
                                },
                            )
                            parsed = None
                        else:
                            transport_validation_result = _build_transport_validation(parsed)
                        if parsed_payload is not None and not isinstance(parsed_payload, dict):
                            parse_error = ValueError("Parsed JSON is not an object")
                        elif parsed is None and parse_error is None:
                            parse_error = ValueError("Parsed planner payload is missing a contract object")
                        elif not _transport_validation_accepted(transport_validation_result):
                            transport_issue = None
                            issues = transport_validation_result.get("issues")
                            if isinstance(issues, list):
                                for issue in issues:
                                    if isinstance(issue, dict) and issue.get("rule"):
                                        transport_issue = str(issue.get("rule"))
                                        break
                            parse_error = ValueError(
                                f"Transport payload invalid: {transport_issue or 'empty_or_trivial_contract'}"
                            )
                            parsed = None

                    quality_error_message = None
                    if parsed is not None and isinstance(parsed, dict):
                        round_has_candidate = True
                        planner_contract_canonical = copy.deepcopy(parsed)
                        candidate_for_validation = copy.deepcopy(parsed)
                        try:
                            validation_result = _merge_validation_results(
                                _validate_contract_quality(copy.deepcopy(candidate_for_validation)),
                                _build_semantic_guard_validation(planner_semantic_core, candidate_for_validation),
                            )
                            validation_result = _adjudicate_ambiguous_validation_issues(
                                planner_semantic_core,
                                candidate_for_validation,
                                validation_result,
                                model_client=model_client,
                                model_name=model_name,
                                trace_stage="compile_validation_adjudication",
                            )
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
                        latest_canonical_for_repair = copy.deepcopy(parsed)
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
                                best_canonical_candidate = copy.deepcopy(parsed)
                                best_validation = validation_result
                                best_response_text = response_text
                                best_error_count = error_count
                                best_warning_count = warning_count
                                best_parse_feedback = None
                    else:
                        parse_feedback = _build_parse_feedback(response_text, parse_error)
                        latest_response_text_for_repair = response_text
                        latest_parse_feedback_for_repair = parse_feedback
                        previous_invalid_meta = (
                            copy.deepcopy(planner_candidate_invalid_meta)
                            if isinstance(planner_candidate_invalid_meta, dict)
                            else {}
                        )
                        planner_candidate_invalid_meta = {
                            "transport_validation": transport_validation_result,
                            "attempt_count": len(planner_diag) + 1,
                            "fallback_mode": "transport_failure",
                        }
                        if (
                            current_tool_mode == "patch"
                            and isinstance(previous_invalid_meta.get("transport_validation"), dict)
                        ):
                            planner_candidate_invalid_meta["transport_validation"] = previous_invalid_meta.get(
                                "transport_validation"
                            )
                        if best_response_text is None:
                            best_response_text = response_text
                            best_parse_feedback = parse_feedback

                    planner_diag.append(
                        {
                            "stage": "contract_compile",
                            "model_name": model_name,
                            "attempt_index": attempt_counter,
                            "quality_round": quality_round,
                            "prompt_char_len": len(current_prompt or ""),
                            "response_char_len": len(response_text or ""),
                            "finish_reason": str(finish_reason) if finish_reason is not None else None,
                            "generation_config": _safe_json_serializable(generation_config_used),
                            "usage_metadata": usage_metadata,
                            "had_json_parse_error": bool(had_json_parse_error),
                            "transport_status": (
                                str(transport_validation_result.get("status") or "").lower()
                                if isinstance(transport_validation_result, dict)
                                else None
                            ),
                            "transport_issue_rules": (
                                [
                                    str(issue.get("rule"))
                                    for issue in (transport_validation_result.get("issues") or [])
                                    if isinstance(issue, dict) and issue.get("rule")
                                ][:12]
                                if isinstance(transport_validation_result, dict)
                                else []
                            ),
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

                    planner_contract_canonical = copy.deepcopy(parsed)
                    contract = copy.deepcopy(parsed)
                    llm_success = True
                    break

                if contract is not None:
                    break
                if quality_round >= max_quality_rounds:
                    break

                current_prompt_name = f"prompt_attempt_{quality_round + 1}_repair.txt"
                current_tool_mode = "contract"
                repair_contract = (
                    latest_canonical_for_repair
                    if isinstance(latest_canonical_for_repair, dict)
                    else best_canonical_candidate
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
                # Repair uses the full original prompt (with all context including
                # column_inventory) plus validation feedback appended.  This ensures
                # the LLM has the same information as attempt 1 and can reason about
                # all columns, not just those that survived text compression.
                validation_feedback = _compact_validation_feedback(repair_validation)
                targeted_actions = _build_targeted_repair_actions(repair_validation)
                repair_suffix = (
                    "\n\n--- REPAIR FEEDBACK FROM PREVIOUS ATTEMPT ---\n"
                    "Your previous contract was rejected. Fix the issues below and return the COMPLETE corrected contract.\n"
                    f"Every required top-level key MUST be present: {', '.join(EXECUTION_CONTRACT_V5_CANONICAL_REQUIRED_KEYS + list(V5_AGENT_SECTION_KEYS))}.\n"
                    'contract_version MUST be "5.0". Use V5.0 hierarchical structure (shared + per-agent sections).\n'
                )
                if targeted_actions.strip():
                    repair_suffix += f"\nTargeted fixes:\n{targeted_actions}\n"
                if validation_feedback.strip() and validation_feedback != "No validation feedback available.":
                    repair_suffix += f"\nValidation issues:\n{validation_feedback}\n"
                parse_fb = (repair_parse_feedback or "").strip()
                if parse_fb:
                    repair_suffix += f"\nParse diagnostics:\n{parse_fb}\n"
                if not round_has_candidate:
                    repair_suffix += "\nPrevious attempts failed JSON parsing. Return syntactically valid JSON only.\n"
                current_prompt = full_prompt + repair_suffix

        if contract is None:
            invalid_contract = (
                best_canonical_candidate
                if isinstance(best_canonical_candidate, dict) and best_canonical_candidate
                else (
                    latest_canonical_for_repair
                    if isinstance(latest_canonical_for_repair, dict) and latest_canonical_for_repair
                    else None
                )
            )
            invalid_raw = (
                best_response_text
                if isinstance(best_response_text, str) and best_response_text.strip()
                else (
                    latest_response_text_for_repair
                    if isinstance(latest_response_text_for_repair, str) and latest_response_text_for_repair.strip()
                    else None
                )
            )
            planner_candidate_invalid = invalid_contract
            planner_candidate_invalid_raw = invalid_raw
            previous_invalid_meta = planner_candidate_invalid_meta if isinstance(planner_candidate_invalid_meta, dict) else {}
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
            if isinstance(previous_invalid_meta, dict) and previous_invalid_meta:
                planner_candidate_invalid_meta = {**previous_invalid_meta, **planner_candidate_invalid_meta}
            if isinstance(invalid_contract, dict) and any(_is_meaningful_contract_value(value) for value in invalid_contract.values()):
                contract = copy.deepcopy(invalid_contract)
            else:
                contract = {}
            if isinstance(invalid_contract, dict):
                planner_contract_canonical = copy.deepcopy(invalid_contract)
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
