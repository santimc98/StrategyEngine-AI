from __future__ import annotations

from typing import Any, Dict, List, Tuple

try:
    from jsonschema import Draft7Validator  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Draft7Validator = None  # type: ignore


TARGET_COLUMN_MACROS = {
    "ALL_NUMERIC",
    "ALL_CATEGORICAL",
    "ALL_TEXT",
    "ALL_DATETIME",
    "ALL_BOOLEAN",
}


ADVISOR_CRITIQUE_PACKET_V1_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "$id": "https://multiagent-bi/schemas/advisor_critique_packet-1.0.schema.json",
    "title": "Advisor Critique Packet",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "packet_type",
        "packet_version",
        "run_id",
        "iteration",
        "timestamp_utc",
        "primary_metric_name",
        "higher_is_better",
        "metric_comparison",
        "validation_signals",
        "error_modes",
        "risk_flags",
        "active_gates_context",
        "analysis_summary",
        "strictly_no_code_advice",
    ],
    "properties": {
        "packet_type": {"const": "advisor_critique_packet"},
        "packet_version": {"const": "1.0"},
        "run_id": {"type": "string", "minLength": 1},
        "iteration": {"type": "integer", "minimum": 0},
        "timestamp_utc": {"type": "string", "format": "date-time"},
        "primary_metric_name": {"type": "string", "minLength": 1},
        "higher_is_better": {"type": "boolean"},
        "metric_comparison": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "baseline_value",
                "candidate_value",
                "delta_abs",
                "delta_rel",
                "min_delta_required",
                "meets_min_delta",
            ],
            "properties": {
                "baseline_value": {"type": "number"},
                "candidate_value": {"type": "number"},
                "delta_abs": {"type": "number"},
                "delta_rel": {"type": "number"},
                "min_delta_required": {"type": "number", "minimum": 0},
                "meets_min_delta": {"type": "boolean"},
            },
        },
        "validation_signals": {
            "type": "object",
            "additionalProperties": False,
            "required": ["validation_mode"],
            "properties": {
                "validation_mode": {
                    "type": "string",
                    "enum": ["cv", "holdout", "cv_and_holdout", "unknown"],
                },
                "cv": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["cv_mean", "cv_std", "fold_count", "variance_level"],
                    "properties": {
                        "cv_mean": {"type": "number"},
                        "cv_std": {"type": "number", "minimum": 0},
                        "fold_count": {"type": "integer", "minimum": 2},
                        "variance_level": {
                            "type": "string",
                            "enum": ["low", "medium", "high", "unknown"],
                        },
                    },
                },
                "holdout": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "metric_value",
                        "split_name",
                        "sample_count",
                        "class_distribution_shift",
                    ],
                    "properties": {
                        "metric_value": {"type": "number"},
                        "split_name": {"type": "string", "minLength": 1},
                        "sample_count": {"type": "integer", "minimum": 1},
                        "class_distribution_shift": {
                            "type": "string",
                            "enum": ["low", "medium", "high", "unknown"],
                        },
                        "positive_class_rate": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                },
                "generalization_gap": {"type": "number"},
            },
            "allOf": [
                {
                    "if": {"properties": {"validation_mode": {"const": "cv"}}},
                    "then": {"required": ["cv"]},
                },
                {
                    "if": {"properties": {"validation_mode": {"const": "holdout"}}},
                    "then": {"required": ["holdout"]},
                },
                {
                    "if": {"properties": {"validation_mode": {"const": "cv_and_holdout"}}},
                    "then": {"required": ["cv", "holdout"]},
                },
            ],
        },
        "error_modes": {
            "type": "array",
            "minItems": 0,
            "maxItems": 5,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "id",
                    "severity",
                    "confidence",
                    "evidence",
                    "affected_scope",
                    "metric_impact_direction",
                ],
                "properties": {
                    "id": {"type": "string", "minLength": 1},
                    "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "evidence": {"type": "string", "minLength": 1, "maxLength": 500},
                    "affected_scope": {"type": "string", "minLength": 1},
                    "metric_impact_direction": {
                        "type": "string",
                        "enum": ["negative", "neutral", "positive"],
                    },
                },
            },
        },
        "risk_flags": {
            "type": "array",
            "minItems": 0,
            "maxItems": 10,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
        "active_gates_context": {
            "type": "array",
            "minItems": 0,
            "maxItems": 30,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
        "analysis_summary": {"type": "string", "minLength": 1, "maxLength": 280},
        "strictly_no_code_advice": {"const": True},
    },
}


ITERATION_HYPOTHESIS_PACKET_V1_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "$id": "https://multiagent-bi/schemas/iteration_hypothesis_packet-1.0.schema.json",
    "title": "Iteration Hypothesis Packet",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "packet_type",
        "packet_version",
        "run_id",
        "iteration",
        "hypothesis_id",
        "action",
        "hypothesis",
        "application_constraints",
        "success_criteria",
        "tracker_context",
        "explanation",
        "fallback_if_not_applicable",
    ],
    "definitions": {
        "targetColumnMacro": {
            "type": "string",
            "enum": sorted(TARGET_COLUMN_MACROS),
        }
    },
    "properties": {
        "packet_type": {"const": "iteration_hypothesis_packet"},
        "packet_version": {"const": "1.0"},
        "run_id": {"type": "string", "minLength": 1},
        "iteration": {"type": "integer", "minimum": 1},
        "hypothesis_id": {"type": "string", "pattern": "^h_[a-zA-Z0-9_-]{6,64}$"},
        "action": {"type": "string", "enum": ["APPLY", "NO_OP"]},
        "hypothesis": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "technique",
                "objective",
                "target_columns",
                "feature_scope",
                "params",
                "expected_effect",
            ],
            "properties": {
                "technique": {"type": "string", "minLength": 1},
                "objective": {"type": "string", "minLength": 1, "maxLength": 220},
                "target_columns": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 200,
                    "uniqueItems": True,
                    "items": {
                        "anyOf": [
                            {"$ref": "#/definitions/targetColumnMacro"},
                            {"type": "string", "minLength": 1, "maxLength": 128},
                        ]
                    },
                },
                "feature_scope": {
                    "type": "string",
                    "enum": [
                        "model_features",
                        "segmentation_features",
                        "audit_only_features",
                        "all_features",
                    ],
                },
                "params": {"type": "object", "additionalProperties": True},
                "expected_effect": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["target_error_modes", "direction"],
                    "properties": {
                        "target_error_modes": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 5,
                            "uniqueItems": True,
                            "items": {"type": "string", "minLength": 1},
                        },
                        "direction": {"type": "string", "enum": ["positive", "neutral", "negative"]},
                    },
                },
            },
        },
        "application_constraints": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "edit_mode",
                "max_code_regions_to_change",
                "forbid_replanning",
                "forbid_model_family_switch",
                "must_keep",
            ],
            "properties": {
                "edit_mode": {"const": "incremental"},
                "max_code_regions_to_change": {"type": "integer", "minimum": 1, "maximum": 8},
                "forbid_replanning": {"type": "boolean"},
                "forbid_model_family_switch": {"type": "boolean"},
                "must_keep": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 10,
                    "uniqueItems": True,
                    "items": {"type": "string", "minLength": 1},
                },
            },
        },
        "success_criteria": {
            "type": "object",
            "additionalProperties": False,
            "required": ["primary_metric_name", "min_delta", "must_pass_active_gates"],
            "properties": {
                "primary_metric_name": {"type": "string", "minLength": 1},
                "min_delta": {"type": "number", "minimum": 0},
                "must_pass_active_gates": {"type": "boolean"},
            },
        },
        "tracker_context": {
            "type": "object",
            "additionalProperties": False,
            "required": ["signature", "is_duplicate", "duplicate_of"],
            "properties": {
                "signature": {"type": "string", "minLength": 1, "maxLength": 512},
                "is_duplicate": {"type": "boolean"},
                "duplicate_of": {"type": ["string", "null"]},
            },
        },
        "explanation": {"type": "string", "minLength": 1, "maxLength": 280},
        "fallback_if_not_applicable": {"const": "NO_OP"},
        "timestamp_utc": {"type": "string", "format": "date-time"},
    },
    "allOf": [
        {
            "if": {"properties": {"action": {"const": "APPLY"}}},
            "then": {"properties": {"hypothesis": {"properties": {"technique": {"not": {"const": "NO_OP"}}}}}},
        },
        {
            "if": {"properties": {"action": {"const": "NO_OP"}}},
            "then": {"properties": {"hypothesis": {"properties": {"technique": {"const": "NO_OP"}}}}},
        },
        {
            "if": {
                "properties": {
                    "tracker_context": {
                        "properties": {"is_duplicate": {"const": True}},
                        "required": ["is_duplicate"],
                    }
                }
            },
            "then": {
                "properties": {
                    "action": {"const": "NO_OP"},
                    "tracker_context": {"properties": {"duplicate_of": {"type": "string", "minLength": 1}}},
                }
            },
        },
        {
            "if": {
                "properties": {
                    "tracker_context": {
                        "properties": {"is_duplicate": {"const": False}},
                        "required": ["is_duplicate"],
                    }
                }
            },
            "then": {"properties": {"tracker_context": {"properties": {"duplicate_of": {"type": "null"}}}}},
        },
    ],
}


EXPERIMENT_HYPOTHESIS_PACKET_V2_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "$id": "https://multiagent-bi/schemas/experiment_hypothesis_packet-2.0.schema.json",
    "title": "Experiment Hypothesis Packet V2",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "packet_type",
        "packet_version",
        "run_id",
        "round",
        "hypothesis_id",
        "action",
        "technique",
        "objective",
        "target_columns",
        "params",
        "success_criteria",
    ],
    "definitions": {
        "targetColumnMacro": {
            "type": "string",
            "enum": sorted(TARGET_COLUMN_MACROS),
        }
    },
    "properties": {
        "packet_type": {"const": "experiment_hypothesis_packet"},
        "packet_version": {"const": "2.0"},
        "run_id": {"type": "string", "minLength": 1},
        "round": {"type": "integer", "minimum": 1},
        "hypothesis_id": {"type": "string", "pattern": "^h_[a-zA-Z0-9_-]{6,64}$"},
        "action": {"type": "string", "enum": ["APPLY", "NO_OP"]},
        "technique": {"type": "string", "minLength": 1},
        "objective": {"type": "string", "minLength": 1, "maxLength": 280},
        "target_columns": {
            "type": "array",
            "minItems": 1,
            "maxItems": 200,
            "uniqueItems": True,
            "items": {
                "anyOf": [
                    {"$ref": "#/definitions/targetColumnMacro"},
                    {"type": "string", "minLength": 1, "maxLength": 128},
                ]
            },
        },
        "params": {"type": "object", "additionalProperties": True},
        "success_criteria": {
            "type": "object",
            "additionalProperties": False,
            "required": ["primary_metric_name", "min_delta", "must_pass_active_gates"],
            "properties": {
                "primary_metric_name": {"type": "string", "minLength": 1},
                "min_delta": {"type": "number", "minimum": 0},
                "must_pass_active_gates": {"type": "boolean"},
            },
        },
        "constraints": {"type": "object", "additionalProperties": True},
        "timestamp_utc": {"type": "string", "format": "date-time"},
    },
}


EXPERIMENT_RESULT_PACKET_V2_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "$id": "https://multiagent-bi/schemas/experiment_result_packet-2.0.schema.json",
    "title": "Experiment Result Packet V2",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "packet_type",
        "packet_version",
        "run_id",
        "round",
        "hypothesis_id",
        "status",
        "primary_metric_name",
        "baseline_metric",
        "candidate_metric",
        "delta_abs",
        "meets_min_delta",
        "gates_passed",
    ],
    "properties": {
        "packet_type": {"const": "experiment_result_packet"},
        "packet_version": {"const": "2.0"},
        "run_id": {"type": "string", "minLength": 1},
        "round": {"type": "integer", "minimum": 1},
        "hypothesis_id": {"type": "string", "pattern": "^h_[a-zA-Z0-9_-]{6,64}$"},
        "status": {"type": "string", "enum": ["SUCCESS", "FAILED", "REJECTED"]},
        "primary_metric_name": {"type": "string", "minLength": 1},
        "baseline_metric": {"type": "number"},
        "candidate_metric": {"type": "number"},
        "delta_abs": {"type": "number"},
        "meets_min_delta": {"type": "boolean"},
        "gates_passed": {"type": "boolean"},
        "failed_gates": {
            "type": "array",
            "minItems": 0,
            "maxItems": 30,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
        "hard_failures": {
            "type": "array",
            "minItems": 0,
            "maxItems": 30,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
        "artifacts_present": {
            "type": "array",
            "minItems": 0,
            "maxItems": 200,
            "items": {"type": "string", "minLength": 1},
        },
        "analysis_summary": {"type": "string", "minLength": 1, "maxLength": 500},
        "timestamp_utc": {"type": "string", "format": "date-time"},
    },
}


def _iter_jsonschema_errors(packet: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    if Draft7Validator is None:
        return []
    try:
        validator = Draft7Validator(schema)
        errors = sorted(validator.iter_errors(packet), key=lambda err: list(err.path))
        out: List[str] = []
        for err in errors[:20]:
            path = ".".join([str(part) for part in err.path]) or "$"
            out.append(path + ": " + err.message)
        return out
    except Exception:
        return []


def validate_advisor_critique_packet(packet: Any) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    if not isinstance(packet, dict):
        return False, ["packet must be an object"]
    if packet.get("packet_type") != "advisor_critique_packet":
        errors.append("packet_type must be advisor_critique_packet")
    if packet.get("packet_version") != "1.0":
        errors.append("packet_version must be 1.0")
    if packet.get("strictly_no_code_advice") is not True:
        errors.append("strictly_no_code_advice must be true")
    summary = str(packet.get("analysis_summary") or "")
    if not summary.strip():
        errors.append("analysis_summary is required")
    if len(summary) > 280:
        errors.append("analysis_summary length must be <= 280")
    error_modes = packet.get("error_modes")
    if not isinstance(error_modes, list):
        errors.append("error_modes must be an array")
    elif len(error_modes) > 5:
        errors.append("error_modes maxItems is 5")
    validation = packet.get("validation_signals")
    if not isinstance(validation, dict):
        errors.append("validation_signals must be an object")
    else:
        mode = str(validation.get("validation_mode") or "")
        if mode not in {"cv", "holdout", "cv_and_holdout", "unknown"}:
            errors.append("validation_signals.validation_mode invalid")
        if mode in {"cv", "cv_and_holdout"} and not isinstance(validation.get("cv"), dict):
            errors.append("validation_signals.cv required for cv/cv_and_holdout")
        if mode in {"holdout", "cv_and_holdout"} and not isinstance(validation.get("holdout"), dict):
            errors.append("validation_signals.holdout required for holdout/cv_and_holdout")
    errors.extend(_iter_jsonschema_errors(packet, ADVISOR_CRITIQUE_PACKET_V1_SCHEMA))
    return len(errors) == 0, errors


def normalize_target_columns(values: Any) -> List[str]:
    if isinstance(values, str):
        raw_values = [values]
    elif isinstance(values, list):
        raw_values = values
    else:
        raw_values = []
    out: List[str] = []
    seen = set()
    for item in raw_values:
        token = str(item or "").strip()
        if not token:
            continue
        upper = token.upper()
        normalized = upper if upper in TARGET_COLUMN_MACROS else token
        if normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    if not out:
        return ["ALL_NUMERIC"]
    return out


def build_noop_iteration_hypothesis_packet(
    *,
    run_id: str,
    iteration: int,
    signature: str,
    duplicate_of: str | None,
    primary_metric_name: str,
    min_delta: float,
    explanation: str,
) -> Dict[str, Any]:
    return {
        "packet_type": "iteration_hypothesis_packet",
        "packet_version": "1.0",
        "run_id": str(run_id or "unknown_run"),
        "iteration": int(iteration or 1),
        "hypothesis_id": "h_noop000",
        "action": "NO_OP",
        "hypothesis": {
            "technique": "NO_OP",
            "objective": "No-op hypothesis due to duplicate/insufficient new signal.",
            "target_columns": ["ALL_NUMERIC"],
            "feature_scope": "model_features",
            "params": {},
            "expected_effect": {
                "target_error_modes": ["no_new_actionable_signal"],
                "direction": "neutral",
            },
        },
        "application_constraints": {
            "edit_mode": "incremental",
            "max_code_regions_to_change": 3,
            "forbid_replanning": True,
            "forbid_model_family_switch": True,
            "must_keep": ["data_split_logic", "cv_protocol", "output_paths_contract"],
        },
        "success_criteria": {
            "primary_metric_name": str(primary_metric_name or "primary_metric"),
            "min_delta": float(min_delta or 0.0),
            "must_pass_active_gates": True,
        },
        "tracker_context": {
            "signature": str(signature or "noop"),
            "is_duplicate": True if duplicate_of else False,
            "duplicate_of": str(duplicate_of) if duplicate_of else None,
        },
        "explanation": str(explanation or "No-op hypothesis."),
        "fallback_if_not_applicable": "NO_OP",
    }


def validate_iteration_hypothesis_packet(packet: Any) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    if not isinstance(packet, dict):
        return False, ["packet must be an object"]
    if packet.get("packet_type") != "iteration_hypothesis_packet":
        errors.append("packet_type must be iteration_hypothesis_packet")
    if packet.get("packet_version") != "1.0":
        errors.append("packet_version must be 1.0")
    action = str(packet.get("action") or "")
    if action not in {"APPLY", "NO_OP"}:
        errors.append("action must be APPLY or NO_OP")
    hypothesis = packet.get("hypothesis")
    if not isinstance(hypothesis, dict):
        errors.append("hypothesis must be an object")
    else:
        target_columns = hypothesis.get("target_columns")
        if not isinstance(target_columns, list) or not target_columns:
            errors.append("hypothesis.target_columns must be a non-empty array")
        else:
            normalized = normalize_target_columns(target_columns)
            if len(normalized) != len(target_columns):
                errors.append("hypothesis.target_columns contain invalid/empty values")
        if action == "NO_OP" and str(hypothesis.get("technique") or "") != "NO_OP":
            errors.append("NO_OP action requires hypothesis.technique=NO_OP")
        if action == "APPLY" and str(hypothesis.get("technique") or "") == "NO_OP":
            errors.append("APPLY action cannot use hypothesis.technique=NO_OP")
    tracker = packet.get("tracker_context")
    if not isinstance(tracker, dict):
        errors.append("tracker_context must be an object")
    else:
        is_duplicate = bool(tracker.get("is_duplicate"))
        duplicate_of = tracker.get("duplicate_of")
        if is_duplicate and not isinstance(duplicate_of, str):
            errors.append("duplicate hypotheses require tracker_context.duplicate_of string")
        if not is_duplicate and duplicate_of is not None:
            errors.append("non-duplicate hypotheses require tracker_context.duplicate_of=null")
        if is_duplicate and action != "NO_OP":
            errors.append("duplicate hypotheses must set action=NO_OP")
    errors.extend(_iter_jsonschema_errors(packet, ITERATION_HYPOTHESIS_PACKET_V1_SCHEMA))
    return len(errors) == 0, errors


def validate_experiment_hypothesis_packet_v2(packet: Any) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    if not isinstance(packet, dict):
        return False, ["packet must be an object"]
    if packet.get("packet_type") != "experiment_hypothesis_packet":
        errors.append("packet_type must be experiment_hypothesis_packet")
    if packet.get("packet_version") != "2.0":
        errors.append("packet_version must be 2.0")
    action = str(packet.get("action") or "")
    if action not in {"APPLY", "NO_OP"}:
        errors.append("action must be APPLY or NO_OP")
    technique = str(packet.get("technique") or "").strip()
    if action == "NO_OP" and technique != "NO_OP":
        errors.append("NO_OP action requires technique=NO_OP")
    target_columns = packet.get("target_columns")
    if not isinstance(target_columns, list) or not target_columns:
        errors.append("target_columns must be a non-empty array")
    else:
        normalized = normalize_target_columns(target_columns)
        if len(normalized) != len(target_columns):
            errors.append("target_columns contain invalid/empty values")
    errors.extend(_iter_jsonschema_errors(packet, EXPERIMENT_HYPOTHESIS_PACKET_V2_SCHEMA))
    return len(errors) == 0, errors


def validate_experiment_result_packet_v2(packet: Any) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    if not isinstance(packet, dict):
        return False, ["packet must be an object"]
    if packet.get("packet_type") != "experiment_result_packet":
        errors.append("packet_type must be experiment_result_packet")
    if packet.get("packet_version") != "2.0":
        errors.append("packet_version must be 2.0")
    status = str(packet.get("status") or "")
    if status not in {"SUCCESS", "FAILED", "REJECTED"}:
        errors.append("status must be SUCCESS|FAILED|REJECTED")
    if packet.get("gates_passed") is True and status == "REJECTED":
        errors.append("REJECTED status cannot have gates_passed=true")
    if packet.get("meets_min_delta") is True:
        try:
            delta = float(packet.get("delta_abs"))
            if delta < 0:
                errors.append("delta_abs must be >= 0 when meets_min_delta=true")
        except Exception:
            errors.append("delta_abs must be numeric")
    errors.extend(_iter_jsonschema_errors(packet, EXPERIMENT_RESULT_PACKET_V2_SCHEMA))
    return len(errors) == 0, errors
