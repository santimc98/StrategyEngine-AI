"""Execution Planner canonical contract transport schemas."""

from typing import Any, Dict, List
import copy


_AGENT_INTERFACE_BLOCK_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "minProperties": 1,
    "additionalProperties": True,
}


_AGENT_INTERFACES_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "data_engineer": copy.deepcopy(_AGENT_INTERFACE_BLOCK_SCHEMA),
        "ml_engineer": copy.deepcopy(_AGENT_INTERFACE_BLOCK_SCHEMA),
        "cleaning_reviewer": copy.deepcopy(_AGENT_INTERFACE_BLOCK_SCHEMA),
        "qa_reviewer": copy.deepcopy(_AGENT_INTERFACE_BLOCK_SCHEMA),
        "reviewer": copy.deepcopy(_AGENT_INTERFACE_BLOCK_SCHEMA),
        "translator": copy.deepcopy(_AGENT_INTERFACE_BLOCK_SCHEMA),
        "results_advisor": copy.deepcopy(_AGENT_INTERFACE_BLOCK_SCHEMA),
    },
    "additionalProperties": True,
}


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
        "agent_interfaces": copy.deepcopy(_AGENT_INTERFACES_SCHEMA),
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


_COLUMN_DTYPE_TARGETS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "minProperties": 1,
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


_ITERATION_POLICY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "minProperties": 1,
    "required": [
        "max_iterations",
        "metric_improvement_max",
        "runtime_fix_max",
        "compliance_bootstrap_max",
    ],
    "properties": {
        "max_iterations": {"type": "integer", "minimum": 1},
        "metric_improvement_max": {"type": "integer", "minimum": 0},
        "runtime_fix_max": {"type": "integer", "minimum": 0},
        "compliance_bootstrap_max": {"type": "integer", "minimum": 0},
    },
    "additionalProperties": True,
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
        "artifact_requirements": {"type": "object", "minProperties": 1, "additionalProperties": True},
        "model_features": {"type": "array", "items": {"type": "string"}},
        "column_dtype_targets": _COLUMN_DTYPE_TARGETS_SCHEMA,
        "cleaning_gates": _GATE_LIST_SCHEMA,
        "qa_gates": _GATE_LIST_SCHEMA,
        "reviewer_gates": _GATE_LIST_SCHEMA,
        "validation_requirements": {"type": "object", "additionalProperties": True},
        "data_engineer_runbook": _RUNBOOK_SCHEMA,
        "ml_engineer_runbook": _RUNBOOK_SCHEMA,
        "evaluation_spec": {"type": "object", "additionalProperties": True},
        "iteration_policy": copy.deepcopy(_ITERATION_POLICY_SCHEMA),
        "optimization_policy": copy.deepcopy(OPTIMIZATION_POLICY_MIN_SCHEMA),
        "agent_interfaces": copy.deepcopy(_AGENT_INTERFACES_SCHEMA),
    },
    "additionalProperties": True,
}


EXECUTION_CONTRACT_TRANSPORT_SCHEMA: Dict[str, Any] = copy.deepcopy(EXECUTION_CONTRACT_CANONICAL_SCHEMA)
# Transport schema must require the same keys as the canonical schema.
# When the schema marks keys as optional, LLMs deprioritize them regardless
# of prompt instructions — the tool schema has higher weight than text.
EXECUTION_CONTRACT_TRANSPORT_SCHEMA["required"] = list(EXECUTION_CONTRACT_CANONICAL_REQUIRED_KEYS)


EXECUTION_SEMANTIC_CORE_TRANSPORT_SCHEMA: Dict[str, Any] = copy.deepcopy(EXECUTION_SEMANTIC_CORE_SCHEMA)
EXECUTION_SEMANTIC_CORE_TRANSPORT_SCHEMA["required"] = list(EXECUTION_SEMANTIC_CORE_REQUIRED_KEYS)
