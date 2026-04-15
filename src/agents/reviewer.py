import os
import json
import ast
import copy
from typing import Dict, Any, List
from dotenv import load_dotenv
from src.utils.reviewer_llm import init_reviewer_llm
from src.utils.senior_protocol import SENIOR_EVIDENCE_RULE
from src.utils.sandbox_paths import CANONICAL_CLEANED_REL, CANONICAL_MANIFEST_REL
from src.utils.reviewer_response_schema import (
    build_reviewer_eval_response_schema,
    build_reviewer_response_schema,
)
from src.utils.review_context_packets import build_review_context_packet
from src.utils.llm_json_repair import JsonObjectParseError, parse_json_object_with_repair
from src.utils.contract_first_gates import apply_contract_first_gate_policy
from src.utils.openrouter_reasoning import create_chat_completion_with_reasoning

load_dotenv()


def _coerce_llm_response_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text
    candidates = getattr(response, "candidates", None)
    if isinstance(candidates, list):
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None)
            if not isinstance(parts, list):
                continue
            chunks: List[str] = []
            for part in parts:
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str) and part_text.strip():
                    chunks.append(part_text.strip())
            if chunks:
                return "\n".join(chunks)
    return str(response or "")


def _normalize_reviewer_gate_name(item: Any) -> str:
    if isinstance(item, dict):
        for key in ("name", "id", "gate", "check", "metric", "rule", "title", "label"):
            value = item.get(key)
            if value:
                return str(value).strip()
        return ""
    if item is None:
        return ""
    return str(item).strip()


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        return token in {"1", "true", "yes", "y", "on"}
    return False


def _extract_metric_round_context(
    evaluation_spec: Dict[str, Any] | None,
    reviewer_view: Dict[str, Any] | None,
) -> bool:
    for block in (evaluation_spec, reviewer_view):
        if not isinstance(block, dict):
            continue
        for key in ("metric_improvement_round_active", "ml_improvement_round_active"):
            if _coerce_bool(block.get(key)):
                return True
        handoff = block.get("iteration_handoff")
        if isinstance(handoff, dict):
            source = str(handoff.get("source") or "").strip().lower()
            mode = str(handoff.get("mode") or "").strip().lower()
            if "metric_improvement" in source or mode == "optimize":
                return True
    return False


_REVIEWER_INVARIANT_GATES: List[str] = [
    "reviewer_syntax_validity",
    "runtime_failure",
    "llm_review_unavailable",
    "contract_required_artifacts_missing",
    "insufficient_deterministic_evidence",
]


def apply_reviewer_gate_filter(result: Dict[str, Any], reviewer_gates: List[Any]) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return {}
    active_gate_names: List[str] = [
        name
        for name in (_normalize_reviewer_gate_name(g) for g in (reviewer_gates or []))
        if name
    ]
    hard_gate_names: set[str] = {
        _normalize_reviewer_gate_name(g).lower()
        for g in (reviewer_gates or [])
        if isinstance(g, dict) and str(g.get("severity") or "HARD").upper() == "HARD"
        and _normalize_reviewer_gate_name(g)
    }
    for invariant_gate in _REVIEWER_INVARIANT_GATES:
        if invariant_gate not in active_gate_names:
            active_gate_names.append(invariant_gate)
        hard_gate_names.add(invariant_gate.lower())
    return apply_contract_first_gate_policy(
        dict(result),
        active_gate_names,
        actor="reviewer",
        hard_gate_names=hard_gate_names,
    )


def _collect_string_literals(tree: ast.AST | None) -> set[str]:
    literals: set[str] = set()
    if tree is None:
        return literals
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            literals.add(node.value)
    return literals


def _resolve_target_columns(
    evaluation_spec: Dict[str, Any] | None,
    reviewer_view: Dict[str, Any] | None,
) -> List[str]:
    targets: List[str] = []
    for block in (reviewer_view, evaluation_spec):
        if not isinstance(block, dict):
            continue
        for key in ("target_column", "target", "primary_target"):
            value = block.get(key)
            if isinstance(value, str) and value.strip():
                targets.append(value.strip())
        for key in ("target_columns", "outcome_columns"):
            values = block.get(key)
            if isinstance(values, list):
                targets.extend(str(v).strip() for v in values if str(v).strip())
        roles = block.get("column_roles")
        if isinstance(roles, dict):
            for key in ("outcome", "target"):
                values = roles.get(key)
                if isinstance(values, list):
                    targets.extend(str(v).strip() for v in values if str(v).strip())
    dedup: List[str] = []
    seen: set[str] = set()
    for t in targets:
        if not t:
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(t)
    return dedup


def _normalize_output_path(path: Any) -> str:
    text = str(path or "").strip().replace("\\", "/")
    if not text:
        return ""
    return text.lstrip("/")


def _extract_output_entry(item: Any) -> tuple[str, str]:
    if isinstance(item, str):
        return _normalize_output_path(item), ""
    if not isinstance(item, dict):
        return "", ""
    candidate = (
        item.get("path")
        or item.get("output_path")
        or item.get("output")
        or item.get("artifact")
        or item.get("file")
    )
    owner = str(item.get("owner") or "").strip()
    return _normalize_output_path(candidate), owner


def _is_data_engineer_output_path(path: str, owner: str | None = None) -> bool:
    owner_key = str(owner or "").strip().lower()
    if owner_key == "data_engineer":
        return True
    normalized = _normalize_output_path(path).lower()
    if not normalized:
        return False
    canonical = {
        _normalize_output_path(CANONICAL_CLEANED_REL).lower(),
        _normalize_output_path(CANONICAL_MANIFEST_REL).lower(),
        "data/cleaned_full.csv",
    }
    if normalized in canonical:
        return True
    basename = os.path.basename(normalized)
    return basename in {"cleaned_data.csv", "cleaning_manifest.json", "cleaned_full.csv"}


def _resolve_ml_required_outputs_for_reviewer(
    reviewer_view: Dict[str, Any] | None,
    evaluation_spec: Dict[str, Any] | None,
) -> List[str]:
    values: List[tuple[str, str]] = []
    for block in (reviewer_view, evaluation_spec):
        if not isinstance(block, dict):
            continue
        required_outputs = block.get("required_outputs")
        if isinstance(required_outputs, list):
            for item in required_outputs:
                values.append(_extract_output_entry(item))
        verification = block.get("verification")
        if isinstance(verification, dict):
            verification_outputs = verification.get("required_outputs")
            if isinstance(verification_outputs, list):
                for item in verification_outputs:
                    values.append(_extract_output_entry(item))

    filtered: List[str] = []
    seen: set[str] = set()
    for path, owner in values:
        if not path:
            continue
        if _is_data_engineer_output_path(path, owner):
            continue
        key = path.lower()
        if key in seen:
            continue
        seen.add(key)
        filtered.append(path)
    return filtered


def _deterministic_reviewer_prechecks(
    code: str,
    evaluation_spec: Dict[str, Any] | None,
    reviewer_view: Dict[str, Any] | None,
) -> Dict[str, Any]:
    output = {
        "hard_failures": [],
        "failed_gates": [],
        "required_fixes": [],
        "warnings": [],
    }
    try:
        tree = ast.parse(code or "")
    except SyntaxError as err:
        msg = f"Syntax error detected before reviewer LLM step: {err}"
        output["hard_failures"].append("reviewer_syntax_validity")
        output["failed_gates"].append("reviewer_syntax_validity")
        output["required_fixes"].append(
            "Fix Python syntax errors before requesting review."
        )
        output["warnings"].append(msg)
        return output

    literals = _collect_string_literals(tree)
    lowered = (code or "").lower()

    expected_data_paths: List[str] = []
    for block in (reviewer_view, evaluation_spec):
        if not isinstance(block, dict):
            continue
        for key in ("ml_data_path", "data_path", "input_path"):
            value = block.get(key)
            if isinstance(value, str) and value.strip():
                expected_data_paths.append(value.strip())
    expected_data_paths = list(dict.fromkeys(expected_data_paths))

    required_outputs = _resolve_ml_required_outputs_for_reviewer(reviewer_view, evaluation_spec)
    targets = _resolve_target_columns(evaluation_spec, reviewer_view)
    strict_data_loading = bool(expected_data_paths or required_outputs or targets)

    has_read_csv = ("pd.read_csv(" in lowered) or (".read_csv(" in lowered)
    if not has_read_csv:
        if strict_data_loading:
            output["hard_failures"].append("reviewer_data_loading_missing")
            output["failed_gates"].append("reviewer_data_loading_missing")
            output["required_fixes"].append(
                "Load the cleaned input dataset explicitly with pandas.read_csv before feature/label preparation."
            )
        output["warnings"].append(
            "Deterministic precheck: pandas.read_csv not detected."
        )

    if expected_data_paths:
        expected_mentions = 0
        for path in expected_data_paths:
            base = os.path.basename(path)
            if path in literals or path.lower() in lowered or (base and (base in literals or base.lower() in lowered)):
                expected_mentions += 1
                break
        if expected_mentions == 0:
            output["hard_failures"].append("reviewer_input_path_binding")
            output["failed_gates"].append("reviewer_input_path_binding")
            output["required_fixes"].append(
                "Bind dataset loading to the contract-provided input path (or its basename) to ensure path traceability."
            )

    if required_outputs:
        mentioned = 0
        for path in required_outputs:
            if path in literals or path.lower() in lowered:
                mentioned += 1
        if mentioned == 0:
            output["hard_failures"].append("reviewer_required_outputs_traceability")
            output["failed_gates"].append("reviewer_required_outputs_traceability")
            output["required_fixes"].append(
                "Write required artifacts at the exact contract output paths."
            )
            output["warnings"].append(
                "Deterministic precheck: none of the required output paths appear in code literals."
            )

    if targets:
        mentions_target = any((target in literals) or (target.lower() in lowered) for target in targets)
        if not mentions_target:
            output["hard_failures"].append("reviewer_target_binding_missing")
            output["failed_gates"].append("reviewer_target_binding_missing")
            output["required_fixes"].append(
                "Reference and bind contract target/outcome columns explicitly in label preparation and evaluation logic."
            )
            output["warnings"].append(
                "Deterministic precheck: target/outcome columns are not explicitly referenced."
            )
    # De-duplicate lists to avoid repeated guidance.
    output["hard_failures"] = list(dict.fromkeys([str(x) for x in output.get("hard_failures", []) if str(x).strip()]))
    output["failed_gates"] = list(dict.fromkeys([str(x) for x in output.get("failed_gates", []) if str(x).strip()]))
    output["required_fixes"] = list(dict.fromkeys([str(x) for x in output.get("required_fixes", []) if str(x).strip()]))
    return output


def _normalize_evidence_items(values: Any, max_items: int = 8) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    if not isinstance(values, list):
        return out
    for item in values:
        claim = ""
        source = "missing"
        if isinstance(item, dict):
            claim = str(item.get("claim") or "").strip()
            source = str(item.get("source") or "missing").strip() or "missing"
        else:
            claim = str(item or "").strip()
        if not claim:
            continue
        normalized = {"claim": claim, "source": source}
        if normalized in out:
            continue
        out.append(normalized)
        if len(out) >= max_items:
            break
    return out


def _extract_execution_diagnostics(
    reviewer_view: Dict[str, Any] | None,
    evaluation_spec: Dict[str, Any] | None,
) -> Dict[str, Any]:
    diagnostics = (
        reviewer_view.get("execution_diagnostics")
        if isinstance(reviewer_view, dict)
        else None
    )
    if not isinstance(diagnostics, dict):
        diagnostics = (
            evaluation_spec.get("execution_diagnostics")
            if isinstance(evaluation_spec, dict)
            else {}
        )
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    return diagnostics


def _deterministic_diagnostics_blockers(execution_diagnostics: Dict[str, Any] | None) -> Dict[str, Any]:
    execution_diagnostics = execution_diagnostics if isinstance(execution_diagnostics, dict) else {}
    blockers = [str(x) for x in (execution_diagnostics.get("hard_blockers") or []) if x]
    output_missing = [
        str(x) for x in (execution_diagnostics.get("output_contract_missing") or []) if str(x).strip()
    ]
    runtime_status = str(execution_diagnostics.get("runtime_status") or "").strip().upper()
    overall_status = str(execution_diagnostics.get("output_contract_overall_status") or "").strip().lower()

    hard_failures: List[str] = []
    failed_gates: List[str] = []
    required_fixes: List[str] = []
    evidence: List[Dict[str, str]] = []

    def _add_failure(name: str) -> None:
        if name not in hard_failures:
            hard_failures.append(name)
        if name not in failed_gates:
            failed_gates.append(name)

    for blocker in blockers:
        _add_failure(blocker)

    if runtime_status == "FAILED_RUNTIME":
        _add_failure("runtime_failure")
        required_fixes.append("Fix runtime failure before requesting reviewer approval.")
        evidence.append({"claim": "Runtime status is FAILED_RUNTIME.", "source": "execution_diagnostics#runtime_status"})

    if overall_status == "error":
        _add_failure("output_contract_error")
        required_fixes.append("Resolve output contract validation errors and regenerate artifacts.")
        evidence.append(
            {
                "claim": "Output contract status is error.",
                "source": "execution_diagnostics#output_contract_overall_status",
            }
        )

    if output_missing:
        _add_failure("contract_required_artifacts_missing")
        for path in output_missing[:8]:
            fix = f"Generate required artifact at path: {path}"
            if fix not in required_fixes:
                required_fixes.append(fix)
            evidence.append(
                {
                    "claim": f"Required output missing: {path}",
                    "source": "execution_diagnostics#output_contract_missing",
                }
            )

    return {
        "hard_failures": hard_failures,
        "failed_gates": failed_gates,
        "required_fixes": required_fixes,
        "evidence": evidence[:8],
    }


def _truncate_exec_output(text: str, max_len: int = 8000) -> str:
    value = str(text or "")
    if len(value) <= max_len:
        return value
    head_len = max_len // 2
    tail_len = max_len - head_len
    return value[:head_len] + "\n...[TRUNCATED_MIDDLE]...\n" + value[-tail_len:]

class ReviewerAgent:
    def __init__(self, api_key: str = None):
        """
        Initializes the Reviewer Agent with MIMO v2 Flash.
        """
        self.provider, self.client, self.model_name, self.model_warning = init_reviewer_llm(api_key)
        if self.model_warning:
            print(f"WARNING: {self.model_warning}")
        self.last_prompt = None
        self.last_response = None
        self.last_json_parse_trace = None
        self._generation_config = {
            "temperature": float(os.getenv("REVIEWER_GEMINI_TEMPERATURE", "0.2")),
            "top_p": 0.9,
            "top_k": 40,
            "max_output_tokens": int(os.getenv("REVIEWER_GEMINI_MAX_TOKENS", "32768")),
            "response_mime_type": "application/json",
        }
        schema_flag = str(os.getenv("REVIEWER_USE_RESPONSE_SCHEMA", "0")).strip().lower()
        self._use_response_schema = schema_flag not in {"0", "false", "no", "off", ""}

    def _generation_config_for_review(self, active_gate_names: List[str] | None = None) -> Dict[str, Any]:
        config = dict(self._generation_config)
        if self._use_response_schema:
            config["response_schema"] = copy.deepcopy(
                build_reviewer_response_schema(active_gate_names or [])
            )
        return config

    def _generation_config_for_evaluation(self, active_gate_names: List[str] | None = None) -> Dict[str, Any]:
        config = dict(self._generation_config)
        if self._use_response_schema:
            config["response_schema"] = copy.deepcopy(
                build_reviewer_eval_response_schema(active_gate_names or [])
            )
        return config

    def _generate_gemini_json(
        self,
        prompt: str,
        *,
        generation_config: Dict[str, Any] | None = None,
    ) -> tuple[str, Dict[str, Any]]:
        if not self.client:
            raise RuntimeError("Gemini client not configured")
        config = dict(generation_config or self._generation_config)
        try:
            response = self.client.generate_content(prompt, generation_config=config)
            return _coerce_llm_response_text(response), config
        except Exception as err:
            if not self._is_response_schema_unsupported_error(err):
                raise
            fallback_config = dict(config)
            fallback_config.pop("response_schema", None)
            response = self.client.generate_content(prompt, generation_config=fallback_config)
            return _coerce_llm_response_text(response), fallback_config

    @staticmethod
    def _is_response_schema_unsupported_error(err: Exception) -> bool:
        message = str(err or "").lower()
        if "response_schema" not in message:
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

    def _attempt_llm_json_repair(
        self,
        raw_text: str,
        *,
        schema: Dict[str, Any] | None,
        repair_label: str,
    ) -> tuple[Dict[str, Any] | None, Dict[str, Any]]:
        trace: Dict[str, Any] = {
            "repair_label": str(repair_label or "reviewer_json"),
            "repair_attempted": False,
            "repair_succeeded": False,
            "provider": self.provider,
            "model": self.model_name,
        }
        if not isinstance(schema, dict) or not schema:
            return None, trace
        if not self.client or self.provider == "none":
            return None, trace
        raw = str(raw_text or "").strip()
        if not raw:
            return None, trace

        trace["repair_attempted"] = True
        schema_json = json.dumps(schema, ensure_ascii=True)
        raw_preview = raw[:12000]
        repair_prompt = (
            "You are a strict JSON repair tool. Return ONLY one JSON object, no markdown.\n"
            "TASK: Repair/normalize RAW_JSON so it conforms to TARGET_SCHEMA.\n"
            "RULES:\n"
            "- Keep original semantic intent when possible.\n"
            "- If malformed/truncated, complete minimally with safe defaults.\n"
            "- Do not invent gate names not present in data.\n"
            "TARGET_SCHEMA:\n"
            + schema_json
            + "\nRAW_JSON:\n"
            + raw_preview
        )

        try:
            if self.provider == "gemini":
                repaired_text, used_config = self._generate_gemini_json(
                    repair_prompt,
                    generation_config={
                        "temperature": 0.0,
                        "response_mime_type": "application/json",
                        "response_schema": copy.deepcopy(schema),
                    },
                )
                trace["used_response_schema"] = bool(
                    isinstance(used_config, dict) and "response_schema" in used_config
                )
            else:
                response = create_chat_completion_with_reasoning(
                    self.client,
                    agent_name="reviewer",
                    model_name=self.model_name,
                    call_kwargs={
                        "model": self.model_name,
                        "messages": [
                            {"role": "system", "content": "Return only valid JSON."},
                            {"role": "user", "content": repair_prompt},
                        ],
                        "response_format": {"type": "json_object"},
                        "temperature": 0.0,
                    },
                )
                repaired_text = response.choices[0].message.content
                trace["used_response_schema"] = False
            parsed, parsed_trace = parse_json_object_with_repair(
                str(repaired_text or ""),
                actor="reviewer_json_repair",
            )
            trace["repair_succeeded"] = isinstance(parsed, dict)
            trace["repair_parse_trace"] = parsed_trace
            return parsed if isinstance(parsed, dict) else None, trace
        except Exception as exc:
            trace["repair_error"] = f"{type(exc).__name__}: {exc}"[:240]
            return None, trace

    def _parse_json_payload_with_llm_repair(
        self,
        text: str,
        *,
        schema: Dict[str, Any] | None,
        repair_label: str,
    ) -> Dict[str, Any]:
        try:
            parsed = self._parse_json_payload(text)
            trace = dict(self.last_json_parse_trace or {})
            trace["repair_via_llm"] = False
            self.last_json_parse_trace = trace
            return parsed
        except Exception:
            base_trace = dict(self.last_json_parse_trace or {})
            repaired, repair_trace = self._attempt_llm_json_repair(
                text,
                schema=schema,
                repair_label=repair_label,
            )
            if isinstance(repaired, dict):
                merged_trace = dict(base_trace)
                merged_trace["repair_via_llm"] = True
                merged_trace["llm_repair"] = repair_trace
                self.last_json_parse_trace = merged_trace
                print(
                    "REVIEWER_JSON_REPAIR_PASS: "
                    + f"label={repair_label} success=True provider={self.provider} model={self.model_name}"
                )
                return repaired
            merged_trace = dict(base_trace)
            merged_trace["repair_via_llm"] = False
            merged_trace["llm_repair"] = repair_trace
            self.last_json_parse_trace = merged_trace
            print(
                "REVIEWER_JSON_REPAIR_PASS: "
                + f"label={repair_label} success=False provider={self.provider} model={self.model_name}"
            )
            raise

    def review_code(
        self,
        code: str,
        analysis_type: str = "predictive",
        business_objective: str = "",
        strategy_context: str = "",
        evaluation_spec: Dict[str, Any] | None = None,
        reviewer_view: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        
        output_format_instructions = """
        Return a raw JSON object:
        {
            "status": "APPROVED" | "APPROVE_WITH_WARNINGS" | "REJECTED",
            "feedback": "Detailed explanation of what to fix if rejected, or 'Looks good' if approved.",
            "failed_gates": ["List", "of", "failed", "principles"],
            "required_fixes": ["List", "of", "specific", "instructions", "for", "the", "engineer"],
            "evidence": [
                {"claim": "Short claim", "source": "artifact_path#key_or_script_path:line or missing"}
            ],
            "improvement_suggestions": {
                "techniques": ["List of specific ML techniques to try next, e.g. 'ensemble averaging of multiple boosting models', 'add polynomial interaction features between top features', 'tune learning_rate and max_depth via grid search'"],
                "no_further_improvement": false
            }
        }

        IMPORTANT for improvement_suggestions:
        - ALWAYS populate this field, even when status is APPROVED.
        - "techniques": 2-5 concrete, actionable suggestions for improving the primary metric. Be specific (name models, features, hyperparameters).
        - "no_further_improvement": set to true ONLY if the code already uses advanced techniques (ensemble, feature engineering, tuning) AND the metric is near theoretical maximum. Otherwise false.
        - Focus on: ensemble methods, feature engineering, hyperparameter tuning, validation improvements.
        """

        from src.utils.prompting import render_prompt

        reviewer_view = reviewer_view or {}
        execution_diagnostics = _extract_execution_diagnostics(reviewer_view, evaluation_spec)
        deterministic_prechecks = _deterministic_reviewer_prechecks(code, evaluation_spec, reviewer_view)
        hard_prechecks = [str(x) for x in (deterministic_prechecks.get("hard_failures") or []) if x]
        precheck_warnings = [str(x) for x in (deterministic_prechecks.get("warnings") or []) if x]
        diagnostics_blockers = _deterministic_diagnostics_blockers(execution_diagnostics)
        # Diagnostics blockers are advisory context for the LLM reviewer,
        # not blocking gates.  The LLM reviewer prompt already receives
        # execution_diagnostics_json and deterministic_prechecks_json and is
        # instructed to trust execution results over static code patterns.

        # Only SyntaxError is a true blocker (code cannot even parse).
        if "reviewer_syntax_validity" in hard_prechecks:
            return {
                "status": "REJECTED",
                "feedback": "Reviewer deterministic precheck failed before LLM review.",
                "failed_gates": [str(x) for x in (deterministic_prechecks.get("failed_gates") or []) if x],
                "required_fixes": [str(x) for x in (deterministic_prechecks.get("required_fixes") or []) if x],
                "hard_failures": ["reviewer_syntax_validity"],
                "warnings": precheck_warnings,
                "evidence": [],
            }

        eval_spec_json = json.dumps(evaluation_spec or {}, indent=2)
        reviewer_gates = []
        if isinstance(evaluation_spec, dict):
            reviewer_gates = evaluation_spec.get("reviewer_gates") or evaluation_spec.get("gates") or []
        view_gates = reviewer_view.get("reviewer_gates")
        if isinstance(view_gates, list) and view_gates:
            reviewer_gates = view_gates
        active_reviewer_gates = [
            name
            for name in (_normalize_reviewer_gate_name(item) for item in reviewer_gates)
            if name
        ]
        allowed_columns = []
        if isinstance(evaluation_spec, dict):
            for key in ("allowed_columns", "canonical_columns", "required_columns", "contract_columns"):
                cols = evaluation_spec.get(key)
                if isinstance(cols, list) and cols:
                    allowed_columns = [str(c) for c in cols if c]
                    break
        if isinstance(reviewer_view.get("required_outputs"), list) and reviewer_view.get("required_outputs"):
            allowed_columns = allowed_columns or []
        strategy_summary = reviewer_view.get("strategy_summary") or strategy_context
        objective_type = reviewer_view.get("objective_type") or analysis_type
        expected_metrics = reviewer_view.get("expected_metrics") or []
        metric_round_active = _extract_metric_round_context(evaluation_spec, reviewer_view)
        deterministic_prechecks_json = json.dumps(deterministic_prechecks, indent=2)
        review_context_packet = build_review_context_packet(
            code,
            reviewer_gates,
            code_path_hint=str(
                reviewer_view.get("subject_code_path_hint")
                or reviewer_view.get("code_path_hint")
                or "artifacts/ml_engineer_last.py"
            ),
            context_blocks=[
                reviewer_view,
                evaluation_spec,
                execution_diagnostics,
            ],
        )
        hard_blocker_packet_json = json.dumps(review_context_packet, indent=2)

        SYSTEM_PROMPT_TEMPLATE = """
        You are a Senior Technical Lead and Security Auditor.

        === EVIDENCE RULE ===
        $senior_evidence_rule
        
        CONTEXT: 
        - Objective Type: "$analysis_type"
        - Strategy Summary: "$strategy_context"
        - Evaluation Spec (JSON): $evaluation_spec_json
        - Reviewer View (JSON): $reviewer_view_json
        - Reviewer Gates (only these can fail): $reviewer_gates
        - ACTIVE_REVIEWER_GATES (names only): $active_reviewer_gates_json
        - Allowed Columns (if provided): $allowed_columns_json
        - Expected Metrics (if provided): $expected_metrics_json
        - Metric Improvement Round Active: $metric_round_active
        - Execution Diagnostics (JSON): $execution_diagnostics_json
        - Deterministic Prechecks (JSON): $deterministic_prechecks_json
        - HARD_BLOCKER_PACKET (JSON): $hard_blocker_packet_json
        
        ### CRITERIA FOR APPROVAL (QUALITY FIRST PRINCIPLES)

        1. **SECURITY & SAFETY (Non-Negotiable):**
           - No malicious code, no external network calls (except sanctioned APIs), no file system deletions outside `data/`.
           
        2. **METHOD FIT & EXECUTION RELIABILITY (Evidence-First, Context-Relative):**
           - Judge the chosen approach relative to the active Reviewer Gates, Evaluation Spec, Strategy Context,
             and concrete execution evidence. Do NOT inject a generic preferred methodology from memory.
           - Baseline simplicity, cross-validation style, model family preferences, and similar design choices are
             only relevant when they materially affect correctness, trustworthiness, or a gate that is actually active.
           - If the code produces valid metrics.json and alignment_check.json, trust the execution evidence over
             static pattern matching unless there is concrete evidence of leakage, invalid evaluation, or broken outputs.
           - Missing assumption handling is a warning unless it demonstrably invalidates the results.
           
        3. **BUSINESS OBJECTIVE FIT (The "So What?"):**
           - Does this analysis *actually* answer: "$business_objective"?
           - Evaluate explainability, performance, calibration, latency, and other trade-offs only as they matter
             for this objective and the supplied strategy/evaluation context.
           - Reject only when the chosen method materially contradicts an explicit objective need or active gate,
             and cite the evidence for that contradiction.

        4. **ENGINEERING STANDARDS:**
           - **Robustness:** Will this crash on empty inputs? (e.g., `df.empty` checks).
           - **Modernity:** No deprecated library calls (e.g., `use_label_encoder` in XGBoost).
           - **Cleanliness:** Code must be syntactically correct and runnable.

        5. **COLUMN MAPPING INTEGRITY:**
           - If Allowed Columns are provided, do NOT use hardcoded column names outside that list.
           - If Allowed Columns are missing, flag hardcoded columns as WARNING only (do not reject).
        
        ### VERDICT LOGIC
        - **REJECT**: Critical Security Violations, Data Leakage, Wrong Method (Regression for Classification), Syntax Errors, Missing Imports.
        - **APPROVE_WITH_WARNINGS**: Minor issues (e.g. suboptimal parameter, messy comments, slight style deviations) that do NOT affect correctness or safety.
        - **APPROVED**: Code is clean, safe, and correct.

        ### SPEC-DRIVEN EVALUATION (MANDATORY)
        - Only fail gates that appear in Reviewer Gates.
        - If a rule is NOT present in Reviewer Gates, you may mention it as a warning but MUST NOT reject for it.
        - failed_gates/hard_failures MUST be an exact subset of ACTIVE_REVIEWER_GATES.
        - Never invent reviewer gates; non-active findings are warning-only feedback.
        - If Reviewer Gates is empty, fall back to the general criteria but prefer APPROVE_WITH_WARNINGS when uncertain.
        - If Metric Improvement Round Active=true, do NOT emit warnings about canonical baseline simplicity, single-model baseline establishment, or preferred baseline model family unless those constraints are present in ACTIVE_REVIEWER_GATES.

        ### PRIORITY REVIEW FOCUS
        - Treat HARD_BLOCKER_PACKET as prioritized evidence, not as an automatic rejection list.
        - Re-check active_hard_gates_summary against code_lines_of_interest before approving.
        - If known_restored_candidate_risks is non-empty, assume the candidate may have reintroduced an older blocker and verify that risk explicitly.
        - If you return APPROVED or APPROVE_WITH_WARNINGS despite an item in HARD_BLOCKER_PACKET, explain why that item is resolved or unsupported by evidence.

        ### EVIDENCE REQUIREMENT
        - Any REJECT or warning must cite evidence from the provided artifacts or code.
        - Include evidence in feedback using: EVIDENCE: <artifact_path>#<key> -> <short snippet>
        - If you cannot find evidence, downgrade to APPROVE_WITH_WARNINGS and state NO_EVIDENCE_FOUND.
        - SELF-CHECK BEFORE REJECT: without at least one concrete evidence item, you must not reject.
        - Populate the "evidence" list with sufficient items to support your claims. If evidence is missing, use source="missing".
        - Evidence sources must be artifact paths or script paths; otherwise use source="missing".

        ### OUTPUT FORMAT
        $output_format_instructions
        """
        
        system_prompt = render_prompt(
            SYSTEM_PROMPT_TEMPLATE,
            analysis_type=str(objective_type).upper(),
            business_objective=business_objective,
            strategy_context=strategy_summary,
            evaluation_spec_json=eval_spec_json,
            reviewer_view_json=json.dumps(reviewer_view, indent=2),
            reviewer_gates=reviewer_gates,
            active_reviewer_gates_json=json.dumps(active_reviewer_gates, indent=2),
            allowed_columns_json=json.dumps(allowed_columns, indent=2),
            expected_metrics_json=json.dumps(expected_metrics, indent=2),
            metric_round_active=str(bool(metric_round_active)).lower(),
            execution_diagnostics_json=json.dumps(execution_diagnostics, indent=2),
            deterministic_prechecks_json=deterministic_prechecks_json,
            hard_blocker_packet_json=hard_blocker_packet_json,
            output_format_instructions=output_format_instructions,
            senior_evidence_rule=SENIOR_EVIDENCE_RULE,
        )
        
        USER_PROMPT_TEMPLATE = "REVIEW THIS CODE:\n\n$code"
        user_prompt = render_prompt(USER_PROMPT_TEMPLATE, code=code)
        self.last_prompt = system_prompt + "\n\n" + user_prompt

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        if not self.client or self.provider == "none":
            return {
                "status": "REJECTED",
                "feedback": "Reviewer LLM unavailable; fail-closed policy prevents approval without review.",
                "failed_gates": ["LLM_REVIEW_UNAVAILABLE"],
                "required_fixes": ["Retry when reviewer LLM is available."],
                "hard_failures": ["LLM_REVIEW_UNAVAILABLE"],
                "warnings": precheck_warnings,
            }

        try:
            print(f"DEBUG: Reviewer calling OpenRouter ({self.model_name})...")
            response = create_chat_completion_with_reasoning(
                self.client,
                agent_name="reviewer",
                model_name=self.model_name,
                call_kwargs={
                    "model": self.model_name,
                    "messages": messages,
                    "response_format": {'type': 'json_object'},
                    "temperature": 0.0,
                },
            )
            content = response.choices[0].message.content
            self.last_response = content
            result = self._parse_json_payload_with_llm_repair(
                content,
                schema=build_reviewer_response_schema(active_reviewer_gates),
                repair_label="review_code",
            )
            parse_trace = self.last_json_parse_trace if isinstance(self.last_json_parse_trace, dict) else {}
            
            # Normalize lists
            for field in ['failed_gates', 'required_fixes']:
                val = result.get(field, [])
                if isinstance(val, str):
                    result[field] = [val]
                elif not isinstance(val, list):
                    result[field] = []
                else:
                    result[field] = val
            result["evidence"] = _normalize_evidence_items(result.get("evidence"), max_items=8)
            if parse_trace:
                result["json_parse_trace"] = parse_trace

            result = apply_reviewer_gate_filter(result, reviewer_gates)
            if precheck_warnings:
                warning_block = "\n".join(f"- {item}" for item in precheck_warnings)
                existing_feedback = str(result.get("feedback") or "").strip()
                appended = (
                    f"{existing_feedback}\nDeterministic precheck warnings:\n{warning_block}"
                    if existing_feedback
                    else f"Deterministic precheck warnings:\n{warning_block}"
                )
                result["feedback"] = appended
                if str(result.get("status") or "").upper() == "APPROVED":
                    result["status"] = "APPROVE_WITH_WARNINGS"
            return result

        except Exception as e:
            print(f"Reviewer API Error: {e}")
            return {
                "status": "REJECTED",
                "feedback": (
                    f"Reviewer unavailable (API error: {e}). "
                    "Fail-closed policy requires retry with reviewer LLM available."
                ),
                "failed_gates": ["LLM_REVIEW_UNAVAILABLE"],
                "required_fixes": ["Retry when reviewer LLM is available."],
                "hard_failures": ["LLM_REVIEW_UNAVAILABLE"],
                "warnings": precheck_warnings,
                "evidence": [{"claim": "Reviewer API call failed.", "source": "missing"}],
            }

    def _parse_json_payload(self, text: str) -> Dict[str, Any]:
        try:
            parsed, trace = parse_json_object_with_repair(text or "", actor="reviewer")
        except JsonObjectParseError as err:
            self.last_json_parse_trace = err.trace if isinstance(err.trace, dict) else {}
            raise
        self.last_json_parse_trace = trace
        return parsed

    def evaluate_results(
        self,
        execution_output: str,
        business_objective: str,
        strategy_context: str,
        evaluation_spec: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Evaluates execution results.
        Phase 1: Deterministic Runtime Error Triage (No LLM).
        Phase 2: Semantic LLM Evaluation (Only if no errors).
        """
        
        # --- PHASE 1: DETERMINISTIC PRE-TRIAGE ---
        # Check for Tracebacks or Execution Errors
        if "Traceback (most recent call last)" in execution_output or "EXECUTION ERROR" in execution_output:
            print("Reviewer: Runtime Error Detected. Skipping LLM eval.")
            
            # Specific Fixes for Common Errors
            failed_gates = ["Runtime Correctness"]
            required_fixes = []
            
            # 1. String to Float Conversion Error (Common in Correlation/ROC with dirty data)
            if "could not convert string to float" in execution_output:
                required_fixes.append("Convert target/features to numeric using `pd.to_numeric(..., errors='coerce')` or `df.factorize()`.")
                required_fixes.append("Map binary strings (yes/no) to 0/1 ensuring `map({'yes':1, 'no':0})` handles case.")
                required_fixes.append("Drop non-numeric columns before Correlation Matrix.")
            
            # 2. General Traceback Fallback
            if not required_fixes:
                required_fixes.append("Fix the Python Runtime Error shown in the logs.")
                required_fixes.append("Wrap dangerous blocks (like plotting or modeling) in try/except.")

            return {
                "status": "NEEDS_IMPROVEMENT",
                "feedback": f"Runtime Error detected in execution. Fix the code to run successfully.\nError Snippet: {execution_output[-500:]}",
                "failed_gates": failed_gates,
                "required_fixes": required_fixes,
                "retry_worth_it": True
            }

        # --- PHASE 2: LLM SEMANTIC EVALUATION ---
        output_format_instructions = """
        Return a raw JSON object:
        {
            "status": "APPROVED" | "NEEDS_IMPROVEMENT",
            "feedback": "Specific instructions for the ML Engineer.",
            "failed_gates": [],
            "required_fixes": [],
            "retry_worth_it": true | false,
            "evidence": [
                {"claim": "Short claim", "source": "artifact_path#key_or_script_path:line or missing"}
            ],
            "improvement_suggestions": {
                "techniques": ["List of 2-5 specific ML techniques to improve the metric"],
                "no_further_improvement": false
            }
        }

        IMPORTANT for improvement_suggestions:
        - ALWAYS populate this field, even when status is APPROVED.
        - "techniques": 2-5 concrete, actionable suggestions for improving the primary metric.
          Be specific: name models, feature engineering approaches, hyperparameters to tune.
          Examples: "Add ensemble averaging of multiple boosting model predictions",
          "Engineer interaction features between age and max_hr", "Tune regularization (l2_leaf_reg) via cross-validated grid search".
        - "no_further_improvement": set to true ONLY if the code already uses advanced ensemble + feature engineering + tuning AND the metric appears near theoretical maximum. Otherwise false.
        """

        # Truncate with head+tail for better context preservation.
        truncated_output = _truncate_exec_output(execution_output, max_len=8000)

        from src.utils.prompting import render_prompt
        
        eval_spec_json = json.dumps(evaluation_spec or {}, indent=2)

        SYSTEM_PROMPT_TEMPLATE = """
        You are a Senior Data Science Lead.
        Your goal is to evaluate the RESULTS of an analysis against the Business Objective.

        === EVIDENCE RULE ===
        $senior_evidence_rule
        
        *** BUSINESS OBJECTIVE ***
        "$business_objective"
        
        *** STRATEGY CONTEXT ***
        $strategy_context

        *** EVALUATION SPEC (JSON) ***
        $evaluation_spec_json
        
        *** EXECUTION OUTPUT (Truncated) ***
        $truncated_output
        
        *** EVALUATION CRITERIA (BUSINESS-FIRST) ***
        Use the criteria below as a reasoning framework, not as a rigid checklist. Weight them according to the active evaluation spec and available evidence.
        1. **Answer Quality:** Does the output provide a clear answer/insight relevant to the BUSINESS OBJECTIVE?
        2. **Visuals:** Are plots generated when required? (If required by spec but missing, flag as warning).
        3. **Metrics - BUSINESS-RELATIVE EVALUATION (CRITICAL):**
           - DO NOT use arbitrary fixed thresholds (e.g., "Accuracy > 0.5").
           - INSTEAD, evaluate metrics RELATIVE TO:
             a) The BASELINE model (if provided): Is the final model better than the baseline?
             b) The PROBLEM DIFFICULTY: Imbalanced classes, noisy data, or limited features justify lower scores.
             c) The BUSINESS VALUE: A 60% accuracy model that identifies 3x more leads than random is valuable.
           - LOW ABSOLUTE SCORES CAN BE VALUABLE if they represent meaningful improvement over baseline.
           - Example: AUC=0.65 on a 1% positive class is excellent; AUC=0.65 on balanced data is mediocre.
        4. **Validation:** If predictive, was validation performed? (Cross-validation preferred but holdout acceptable).
        5. **Safety Outputs:** Does alignment_check.json exist? (Execution-aware check, not code pattern matching).
        - Only enforce criteria that are required by the Evaluation Spec.

        *** SENIOR TECH LEAD MINDSET ***
        You care that the system WORKS, is SAFE, and SOLVES THE PROBLEM.
        You do NOT care if the code looks exactly like a textbook example.

        *** DECISION POLICY ***
        - If results are "weak" in absolute terms but methodology is sound and improves over baseline => APPROVE with note.
        - REJECT only if there are specific TECHNICAL fixes that would materially improve results.
        - If "Traceback" or "Error" persists in output despite your checks, REJECT.

        *** EVIDENCE REQUIREMENT ***
        - Any NEEDS_IMPROVEMENT or warning must cite evidence from artifacts or execution output.
        - Include evidence in feedback using: EVIDENCE: <artifact_path>#<key> -> <short snippet>
        - If you cannot find evidence, downgrade to APPROVE_WITH_WARNINGS and state NO_EVIDENCE_FOUND.
        - SELF-CHECK BEFORE REJECT: without at least one concrete evidence item, you must not reject.
        - Populate the "evidence" list with sufficient items to support your claims. If evidence is missing, use source="missing".
        - Evidence sources must be artifact paths or script paths; otherwise use source="missing".

        OUTPUT FORMAT (JSON):
        $output_format_instructions
        """
        
        system_prompt = render_prompt(
            SYSTEM_PROMPT_TEMPLATE,
            business_objective=business_objective,
            strategy_context=strategy_context,
            truncated_output=truncated_output,
            evaluation_spec_json=eval_spec_json,
            output_format_instructions=output_format_instructions,
            senior_evidence_rule=SENIOR_EVIDENCE_RULE,
        )
        self.last_prompt = system_prompt + "\n\nEvaluate results."
        eval_reviewer_gates: List[str] = []
        if isinstance(evaluation_spec, dict):
            raw_gates = evaluation_spec.get("reviewer_gates") or evaluation_spec.get("gates") or []
            if isinstance(raw_gates, list):
                eval_reviewer_gates = [
                    name
                    for name in (_normalize_reviewer_gate_name(item) for item in raw_gates)
                    if name
                ]

        if not self.client or self.provider == "none":
            return self._deterministic_eval_fallback(
                execution_output=execution_output,
                reason="LLM unavailable",
            )

        try:
            print(f"DEBUG: Reviewer evaluation calling OpenRouter ({self.model_name})...")
            response = create_chat_completion_with_reasoning(
                self.client,
                agent_name="reviewer",
                model_name=self.model_name,
                call_kwargs={
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": "Evaluate results."}
                    ],
                    "response_format": {'type': 'json_object'},
                    "temperature": 0.1,
                },
            )
            content = response.choices[0].message.content
            self.last_response = content
            result = self._parse_json_payload_with_llm_repair(
                content,
                schema=build_reviewer_eval_response_schema(eval_reviewer_gates),
                repair_label="evaluate_results",
            )
            parse_trace = self.last_json_parse_trace if isinstance(self.last_json_parse_trace, dict) else {}
            
            # Defaults for backward compatibility
            if "failed_gates" not in result: result["failed_gates"] = []
            if "required_fixes" not in result: result["required_fixes"] = []
            if "retry_worth_it" not in result: result["retry_worth_it"] = True
            result["evidence"] = _normalize_evidence_items(result.get("evidence"), max_items=8)
            if parse_trace:
                result["json_parse_trace"] = parse_trace
            result = apply_reviewer_gate_filter(result, eval_reviewer_gates)
            
            return result
            
        except Exception as e:
            print(f"Reviewer Evaluation Error: {e}")
            return self._deterministic_eval_fallback(
                execution_output=execution_output,
                reason=f"LLM error: {e}",
            )

    def _deterministic_eval_fallback(self, execution_output: str, reason: str) -> Dict[str, Any]:
        output = str(execution_output or "")
        lower = output.lower()
        if "traceback (most recent call last)" in lower or "execution error" in lower:
            return {
                "status": "NEEDS_IMPROVEMENT",
                "feedback": (
                    "Deterministic fallback: runtime failure detected while reviewer LLM was unavailable. "
                    f"Reason: {reason}"
                ),
                "failed_gates": ["runtime_failure", "LLM_EVAL_UNAVAILABLE"],
                "required_fixes": [
                    "Fix runtime failure and rerun execution.",
                    "Retry reviewer evaluation when LLM is available.",
                ],
                "retry_worth_it": True,
                "hard_failures": ["runtime_failure"],
                "evidence": [{"claim": "Runtime failure marker detected.", "source": "execution_output_tail"}],
            }

        missing_required = None
        has_metrics = False
        for raw_line in output.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("output_contract_missing_required="):
                try:
                    missing_required = int(line.split("=", 1)[1].strip())
                except Exception:
                    missing_required = None
            elif line.startswith("metrics="):
                metric_blob = line.split("=", 1)[1].strip()
                has_metrics = bool(metric_blob)

        if isinstance(missing_required, int) and missing_required > 0:
            return {
                "status": "NEEDS_IMPROVEMENT",
                "feedback": (
                    "Deterministic fallback: missing required contract artifacts while reviewer LLM is unavailable. "
                    f"Reason: {reason}"
                ),
                "failed_gates": ["contract_required_artifacts_missing", "LLM_EVAL_UNAVAILABLE"],
                "required_fixes": [
                    "Generate all contract-required artifacts before requesting approval.",
                    "Retry reviewer evaluation when LLM is available.",
                ],
                "retry_worth_it": True,
                "hard_failures": ["contract_required_artifacts_missing"],
                "evidence": [
                    {
                        "claim": f"output_contract_missing_required={missing_required}",
                        "source": "deterministic_evidence#output_contract_missing_required",
                    }
                ],
            }

        if has_metrics and (missing_required == 0 or missing_required is None):
            return {
                "status": "APPROVE_WITH_WARNINGS",
                "feedback": (
                    "Deterministic fallback: reviewer LLM unavailable but deterministic evidence indicates "
                    "metrics and no missing required outputs."
                ),
                "failed_gates": ["LLM_EVAL_UNAVAILABLE"],
                "required_fixes": ["Retry reviewer evaluation with LLM available for full semantic audit."],
                "retry_worth_it": False,
                "evidence": [{"claim": "metrics summary present in deterministic evidence.", "source": "deterministic_evidence#metrics"}],
            }

        return {
            "status": "NEEDS_IMPROVEMENT",
            "feedback": (
                "Deterministic fallback: insufficient structured evidence to approve while reviewer LLM is unavailable. "
                f"Reason: {reason}"
            ),
            "failed_gates": ["LLM_EVAL_UNAVAILABLE", "insufficient_deterministic_evidence"],
            "required_fixes": [
                "Ensure metrics and required outputs are generated and persisted.",
                "Retry reviewer evaluation when LLM is available.",
            ],
            "retry_worth_it": True,
            "hard_failures": ["LLM_EVAL_UNAVAILABLE"],
            "evidence": [{"claim": "No conclusive deterministic evidence for approval.", "source": "missing"}],
        }
