"""Execution Planner contract schemas."""

from typing import Any, Dict, List
import copy


_REQUIRED_OUTPUT_ITEM_SCHEMA: Dict[str, Any] = {
    "anyOf": [
        {"type": "string", "minLength": 1},
        {
            "type": "object",
            "required": ["path"],
            "properties": {
                "path": {"type": "string", "minLength": 1},
                "intent": {"type": "string", "minLength": 1},
                "required": {"type": "boolean"},
                "owner": {"type": "string"},
                "kind": {"type": "string"},
                "description": {"type": "string"},
                "id": {"type": "string"},
                "source": {"type": "string"},
            },
            "additionalProperties": True,
        },
    ]
}


_REQUIRED_OUTPUTS_SCHEMA: Dict[str, Any] = {
    "type": "array",
    "items": copy.deepcopy(_REQUIRED_OUTPUT_ITEM_SCHEMA),
    "minItems": 1,
}


OPTIMIZATION_POLICY_MIN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": [
        "enabled",
        "max_rounds",
        "quick_eval_folds",
        "full_eval_folds",
        "min_delta",
        "patience",
        "allow_model_switch",
        "allow_ensemble",
        "allow_hpo",
        "allow_feature_engineering",
        "allow_calibration",
    ],
    "properties": {
        "enabled": {"type": "boolean"},
        "max_rounds": {"type": "integer", "minimum": 0},
        "quick_eval_folds": {"type": "integer", "minimum": 0},
        "full_eval_folds": {"type": "integer", "minimum": 0},
        "min_delta": {"type": "number", "minimum": 0},
        "patience": {"type": "integer", "minimum": 0},
        "allow_model_switch": {"type": "boolean"},
        "allow_ensemble": {"type": "boolean"},
        "allow_hpo": {"type": "boolean"},
        "allow_feature_engineering": {"type": "boolean"},
        "allow_calibration": {"type": "boolean"},
    },
    "additionalProperties": True,
}


# V4 canonical required keys — kept for backward compat in execution_planner
# validation scoring (non-v5 path).
EXECUTION_CONTRACT_CANONICAL_REQUIRED_KEYS: List[str] = [
    "contract_version",
    "scope",
    "strategy_title",
    "business_objective",
    "output_dialect",
    "canonical_columns",
    "required_outputs",
    "column_roles",
    "allowed_feature_sets",
    "task_semantics",
    "active_workstreams",
    "artifact_requirements",
    "cleaning_gates",
    "qa_gates",
    "reviewer_gates",
    "iteration_policy",
    "column_dtype_targets",
    "model_features",
    "data_engineer_runbook",
]


EXECUTION_SEMANTIC_CORE_REQUIRED_KEYS: List[str] = [
    "scope",
    "strategy_title",
    "business_objective",
    "output_dialect",
    "canonical_columns",
    "required_outputs",
    "column_roles",
    "allowed_feature_sets",
    "task_semantics",
    "active_workstreams",
    "model_features",
    "cleaning_gates",
    "qa_gates",
    "reviewer_gates",
    "data_engineer_runbook",
    "optimization_policy",
]


_RUNBOOK_SCHEMA: Dict[str, Any] = {
    "anyOf": [
        {"type": "object", "additionalProperties": True},
        {"type": "array", "items": {}},
        {"type": "string", "minLength": 1},
    ]
}


_GATE_OBJECT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "severity": {"type": "string", "enum": ["HARD", "SOFT"]},
        "params": {"type": "object", "additionalProperties": True},
        "action_type": {
            "type": "string",
            "enum": ["drop", "parse", "coerce", "impute", "standardize", "derive", "check"],
        },
        "column_phase": {
            "type": "string",
            "enum": ["input", "transform", "output"],
        },
        "final_state": {
            "type": "string",
            "enum": ["removed", "retained", "derived", "validated"],
        },
        "condition": {"type": "string"},
        "evidence_required": {"type": "string"},
        "action_if_fail": {"type": "string"},
    },
    "additionalProperties": True,
}


_GATE_LIST_SCHEMA: Dict[str, Any] = {
    "type": "array",
    "items": {
        "anyOf": [
            {"type": "string", "minLength": 1},
            copy.deepcopy(_GATE_OBJECT_SCHEMA),
        ]
    },
}


_ACTIVE_WORKSTREAMS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["cleaning", "feature_engineering", "model_training"],
    "properties": {
        "cleaning": {"type": "boolean"},
        "feature_engineering": {"type": "boolean"},
        "model_training": {"type": "boolean"},
        "review": {"type": "boolean"},
        "translation": {"type": "boolean"},
        "notes": {"type": "string"},
    },
    "additionalProperties": True,
}


_FUTURE_ML_HANDOFF_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "enabled": {"type": "boolean"},
        "primary_target": {"type": "string"},
        "target_columns": {"type": "array", "items": {"type": "string"}},
        "readiness_goal": {"type": "string"},
        "notes": {"type": "string"},
    },
    "additionalProperties": True,
}


EXECUTION_SEMANTIC_CORE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": list(EXECUTION_SEMANTIC_CORE_REQUIRED_KEYS),
    "properties": {
        "scope": {"type": "string", "enum": ["cleaning_only", "ml_only", "full_pipeline"]},
        "strategy_title": {"type": "string", "minLength": 1},
        "business_objective": {"type": "string", "minLength": 1},
        "output_dialect": {
            "type": "object",
            "required": ["sep", "decimal", "encoding"],
            "properties": {
                "sep": {"type": "string"},
                "decimal": {"type": "string"},
                "encoding": {"type": "string"},
            },
            "additionalProperties": True,
        },
        "canonical_columns": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "required_outputs": copy.deepcopy(_REQUIRED_OUTPUTS_SCHEMA),
        "column_roles": {
            "type": "object",
            "required": [
                "pre_decision",
                "decision",
                "outcome",
                "post_decision_audit_only",
                "unknown",
                "identifiers",
                "time_columns",
            ],
            "properties": {
                "pre_decision": {"type": "array", "items": {"type": "string"}},
                "decision": {"type": "array", "items": {"type": "string"}},
                "outcome": {"type": "array", "items": {"type": "string"}},
                "post_decision_audit_only": {"type": "array", "items": {"type": "string"}},
                "unknown": {"type": "array", "items": {"type": "string"}},
                "identifiers": {"type": "array", "items": {"type": "string"}},
                "time_columns": {"type": "array", "items": {"type": "string"}},
            },
            "additionalProperties": True,
        },
        "allowed_feature_sets": {
            "type": "object",
            "required": [
                "segmentation_features",
                "model_features",
                "forbidden_features",
                "audit_only_features",
            ],
            "properties": {
                "segmentation_features": {"type": "array", "items": {"type": "string"}},
                "model_features": {"type": "array", "items": {"type": "string"}},
                "forbidden_features": {"type": "array", "items": {"type": "string"}},
                "audit_only_features": {"type": "array", "items": {"type": "string"}},
            },
            "additionalProperties": True,
        },
        "task_semantics": {
            "type": "object",
            "required": ["problem_family", "objective_type"],
            "properties": {
                "problem_family": {"type": "string"},
                "objective_type": {"type": "string"},
                "primary_target": {"type": "string"},
                "target_columns": {"type": "array", "items": {"type": "string"}},
                "prediction_unit": {"type": "string"},
                "output_schema": {"type": "object", "additionalProperties": True},
            },
            "additionalProperties": True,
        },
        "active_workstreams": _ACTIVE_WORKSTREAMS_SCHEMA,
        "future_ml_handoff": _FUTURE_ML_HANDOFF_SCHEMA,
        "model_features": {"type": "array", "items": {"type": "string"}},
        "cleaning_gates": _GATE_LIST_SCHEMA,
        "qa_gates": _GATE_LIST_SCHEMA,
        "reviewer_gates": _GATE_LIST_SCHEMA,
        "data_engineer_runbook": _RUNBOOK_SCHEMA,
        "optimization_policy": copy.deepcopy(OPTIMIZATION_POLICY_MIN_SCHEMA),
    },
    "additionalProperties": True,
}


EXECUTION_SEMANTIC_CORE_TRANSPORT_SCHEMA: Dict[str, Any] = copy.deepcopy(EXECUTION_SEMANTIC_CORE_SCHEMA)
EXECUTION_SEMANTIC_CORE_TRANSPORT_SCHEMA["required"] = list(EXECUTION_SEMANTIC_CORE_REQUIRED_KEYS)


# ── V5 Hierarchical Contract Schema ──────────────────────────────────
#
# V5 organises the contract by agent hierarchy:
#   { "shared": {...}, "data_engineer": {...}, "ml_engineer": {...}, ... }
#
# View generation becomes a trivial merge:
#   de_view  = shared + data_engineer
#   ml_view  = shared + ml_engineer
#   cleaning_view = shared + data_engineer + cleaning_reviewer
#   qa_view  = shared + ml_engineer + qa_reviewer
#   reviewer_view = shared + ml_engineer
#   translator_view = shared + business_translator
#   results_advisor_view = shared
# ─────────────────────────────────────────────────────────────────────

_V5_SHARED_REQUIRED_KEYS: List[str] = [
    "scope",
    "strategy_title",
    "business_objective",
    "output_dialect",
    "canonical_columns",
    "column_roles",
    "allowed_feature_sets",
    "task_semantics",
    "active_workstreams",
    "model_features",
    "column_dtype_targets",
    "iteration_policy",
]

EXECUTION_CONTRACT_V5_CANONICAL_REQUIRED_KEYS: List[str] = [
    "contract_version",
    "shared",
    "data_engineer",
    "ml_engineer",
]


# Agent section keys recognized by v5 dispatch logic.
V5_AGENT_SECTION_KEYS: List[str] = [
    "data_engineer",
    "ml_engineer",
    "cleaning_reviewer",
    "qa_reviewer",
    "business_translator",
]
