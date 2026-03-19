"""Execution Planner canonical contract transport schemas."""

from typing import Any, Dict, List
import copy


EXECUTION_CONTRACT_V41_MIN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": [
        "contract_version",
        "scope",
        "strategy_title",
        "business_objective",
        "output_dialect",
        "canonical_columns",
        "column_roles",
        "allowed_feature_sets",
        "artifact_requirements",
        "required_outputs",
        "iteration_policy",
    ],
    "properties": {
        "contract_version": {"type": "string"},
        "scope": {
            "type": "string",
            "enum": ["cleaning_only", "ml_only", "full_pipeline"],
        },
        "strategy_title": {"type": "string"},
        "business_objective": {"type": "string"},
        "output_dialect": {
            "type": "object",
            "required": ["sep", "decimal", "encoding"],
            "properties": {
                "sep": {"type": "string"},
                "decimal": {"type": "string"},
                "encoding": {"type": "string"},
            },
            "additionalProperties": False,
        },
        "canonical_columns": {
            "type": "array",
            "items": {"type": "string"},
        },
        "required_outputs": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
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
            "additionalProperties": False,
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
            "additionalProperties": False,
        },
        "artifact_requirements": {
            "type": "object",
            "additionalProperties": True,
        },
        "iteration_policy": {
            "type": "object",
            "additionalProperties": True,
        },
    },
    "additionalProperties": True,
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
        "max_rounds": {"type": "integer", "minimum": 1},
        "quick_eval_folds": {"type": "integer", "minimum": 1},
        "full_eval_folds": {"type": "integer", "minimum": 1},
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


EXECUTION_CONTRACT_V42_MIN_SCHEMA: Dict[str, Any] = copy.deepcopy(EXECUTION_CONTRACT_V41_MIN_SCHEMA)
EXECUTION_CONTRACT_V42_MIN_SCHEMA["properties"] = dict(EXECUTION_CONTRACT_V42_MIN_SCHEMA.get("properties") or {})
EXECUTION_CONTRACT_V42_MIN_SCHEMA["properties"]["contract_version"] = {
    "type": "string",
    "enum": ["4.1", "4.2"],
}
EXECUTION_CONTRACT_V42_MIN_SCHEMA["properties"]["optimization_policy"] = copy.deepcopy(
    OPTIMIZATION_POLICY_MIN_SCHEMA
)


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
    "artifact_requirements",
    "cleaning_gates",
    "qa_gates",
    "reviewer_gates",
    "evaluation_spec",
    "validation_requirements",
    "iteration_policy",
    "column_dtype_targets",
    "data_engineer_runbook",
    "ml_engineer_runbook",
]


_RUNBOOK_SCHEMA: Dict[str, Any] = {
    "anyOf": [
        {"type": "object", "additionalProperties": True},
        {"type": "array", "items": {}},
        {"type": "string", "minLength": 1},
    ]
}


_GATE_LIST_SCHEMA: Dict[str, Any] = {
    "type": "array",
    "items": {
        "anyOf": [
            {"type": "string", "minLength": 1},
            {"type": "object", "additionalProperties": True},
        ]
    },
}


_COLUMN_DTYPE_TARGETS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": {
        "type": "object",
        "properties": {
            "target_dtype": {"type": "string"},
            "nullable": {"type": "boolean"},
            "role": {"type": "string"},
            "source": {"type": "string"},
        },
        "required": ["target_dtype"],
        "additionalProperties": True,
    },
}


EXECUTION_CONTRACT_CANONICAL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": list(EXECUTION_CONTRACT_CANONICAL_REQUIRED_KEYS),
    "properties": {
        "contract_version": {"type": "string", "enum": ["4.1", "4.2"]},
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
        "required_outputs": {"type": "array", "items": {"type": "string"}, "minItems": 1},
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
        "artifact_requirements": {"type": "object", "additionalProperties": True},
        "column_dtype_targets": _COLUMN_DTYPE_TARGETS_SCHEMA,
        "cleaning_gates": _GATE_LIST_SCHEMA,
        "qa_gates": _GATE_LIST_SCHEMA,
        "reviewer_gates": _GATE_LIST_SCHEMA,
        "validation_requirements": {"type": "object", "additionalProperties": True},
        "data_engineer_runbook": _RUNBOOK_SCHEMA,
        "ml_engineer_runbook": _RUNBOOK_SCHEMA,
        "evaluation_spec": {"type": "object", "additionalProperties": True},
        "iteration_policy": {"type": "object", "additionalProperties": True},
        "optimization_policy": copy.deepcopy(OPTIMIZATION_POLICY_MIN_SCHEMA),
    },
    "additionalProperties": True,
}


EXECUTION_CONTRACT_TRANSPORT_SCHEMA: Dict[str, Any] = copy.deepcopy(EXECUTION_CONTRACT_CANONICAL_SCHEMA)
# Transport schema is used by the LLM tool call — keep it flexible so the
# repair loop can fix omissions.  Only the canonical validation enforces all keys.
_TRANSPORT_ONLY_REQUIRED = [
    "contract_version", "scope", "strategy_title", "business_objective",
    "output_dialect", "canonical_columns", "required_outputs", "column_roles",
    "allowed_feature_sets", "task_semantics", "artifact_requirements",
]
EXECUTION_CONTRACT_TRANSPORT_SCHEMA["required"] = _TRANSPORT_ONLY_REQUIRED
