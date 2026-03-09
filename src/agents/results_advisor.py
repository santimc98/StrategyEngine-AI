import os
import json
import csv
import re
import copy
from statistics import median
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from src.utils.retries import call_with_retries
from src.utils.senior_protocol import SENIOR_EVIDENCE_RULE
from src.utils.reviewer_llm import init_reviewer_llm
from src.utils.actor_critic_schemas import validate_advisor_critique_packet
from src.utils.llm_json_repair import JsonObjectParseError, parse_json_object_with_repair
from src.utils.results_advisor_response_schema import (
    build_results_advisor_critique_response_schema,
)
from src.utils.problem_capabilities import infer_problem_capabilities, is_problem_family, normalize_problem_family

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


class ResultsAdvisorAgent:
    """
    Generate insights and improvement advice from evaluation artifacts.
    """

    def __init__(self, api_key: Any = None):
        explicit_api_key = api_key
        self.api_key = explicit_api_key if explicit_api_key is not None else os.getenv("MIMO_API_KEY")
        if not self.api_key:
            self.client = None
        else:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.xiaomimimo.com/v1",
                timeout=None,
            )
        self.model_name = "mimo-v2-flash"
        self.last_prompt = None
        self.last_response = None
        self.fe_advice_mode = self._normalize_fe_advice_mode(
            os.getenv("RESULTS_ADVISOR_FE_MODE", "hybrid")
        )
        self.fe_provider = "none"
        self.fe_client = None
        self.fe_model_name = None
        self.fe_model_warning = None
        self.critique_mode = self._normalize_critique_mode(
            os.getenv("RESULTS_ADVISOR_CRITIQUE_MODE", "hybrid")
        )
        self._generation_config = {
            "temperature": float(
                os.getenv(
                    "RESULTS_ADVISOR_GEMINI_TEMPERATURE",
                    os.getenv("REVIEWER_GEMINI_TEMPERATURE", "0.2"),
                )
            ),
            "top_p": 0.9,
            "top_k": 40,
            "max_output_tokens": int(
                os.getenv(
                    "RESULTS_ADVISOR_GEMINI_MAX_TOKENS",
                    os.getenv("REVIEWER_GEMINI_MAX_TOKENS", "8192"),
                )
            ),
            "response_mime_type": "application/json",
        }
        schema_flag = str(
            os.getenv(
                "RESULTS_ADVISOR_USE_RESPONSE_SCHEMA",
                os.getenv("REVIEWER_USE_RESPONSE_SCHEMA", "0"),
            )
        ).strip().lower()
        self._use_response_schema = schema_flag not in {"0", "false", "no", "off", ""}
        self.last_fe_advice_meta: Dict[str, Any] = {
            "mode": self.fe_advice_mode,
            "source": "deterministic",
            "provider": "none",
            "model": None,
        }
        self.last_critique_meta: Dict[str, Any] = {
            "mode": self.critique_mode,
            "source": "deterministic",
            "provider": "none",
            "model": None,
        }
        self.last_critique_packet: Dict[str, Any] = {}
        self.last_critique_error: Dict[str, Any] = {}
        # Tests frequently instantiate with api_key="" to force non-network mode.
        if explicit_api_key is not None and str(explicit_api_key).strip() == "":
            return
        if (
            self.fe_advice_mode in {"llm", "hybrid"}
            or self.critique_mode in {"llm", "hybrid"}
        ):
            (
                self.fe_provider,
                self.fe_client,
                self.fe_model_name,
                self.fe_model_warning,
            ) = init_reviewer_llm(None)

    def _generation_config_for_critique(self) -> Dict[str, Any]:
        config = dict(self._generation_config)
        if self._use_response_schema:
            config["response_schema"] = copy.deepcopy(
                build_results_advisor_critique_response_schema()
            )
        return config

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

    def _generate_gemini_json(
        self,
        prompt: str,
        *,
        generation_config: Dict[str, Any],
    ) -> tuple[str, Dict[str, Any]]:
        used_config = dict(generation_config or {})
        try:
            response = self.fe_client.generate_content(prompt, generation_config=used_config)
        except TypeError:
            response = self.fe_client.generate_content(prompt)
        except Exception as err:
            if "response_schema" in used_config and self._is_response_schema_unsupported_error(err):
                retry_config = dict(used_config)
                retry_config.pop("response_schema", None)
                try:
                    response = self.fe_client.generate_content(prompt, generation_config=retry_config)
                except TypeError:
                    response = self.fe_client.generate_content(prompt)
                used_config = retry_config
            else:
                raise
        return _coerce_llm_response_text(response), used_config

    def generate_ml_advice(self, context: Dict[str, Any]) -> str:
        if not context:
            return ""
        insights = self.generate_insights(context)
        summary_lines = insights.get("summary_lines", []) if isinstance(insights, dict) else []
        if summary_lines:
            return "\n".join(summary_lines)
        if not self.client:
            return self._fallback(context)

        context_snippet = self._truncate(json.dumps(context, ensure_ascii=True), 4000)
        system_prompt = (
            "You are a senior ML reviewer focused on business alignment. "
            "=== EVIDENCE RULE ===\n"
            + SENIOR_EVIDENCE_RULE
            + "\n"
            "Given evaluation artifacts and business alignment, produce improvement guidance. "
            "Prioritize structural changes when alignment metrics fail (objective/constraints/penalties) "
            "before tuning hyperparameters. Compare current vs previous iteration if provided. "
            "Return 3-6 short lines. Format each line as: "
            "ISSUE: <what failed>; WHY: <root cause>; FIX: <specific change>. "
            "Do NOT include code. Do NOT restate the full metrics dump. "
            "Then add a mini-section:\n"
            "evidence:\n"
            "- claim: <short claim>; source: <artifact_path#key_or_script_path:line or missing>\n"
            "Provide 3-8 evidence items; if evidence is missing, use source=missing. "
            "Evidence sources must be artifact paths or script paths; otherwise use source=missing."
        )
        user_prompt = "CONTEXT:\n" + context_snippet + "\n"
        self.last_prompt = system_prompt + "\n\n" + user_prompt

        def _call_model():
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
            )
            return response.choices[0].message.content

        try:
            content = call_with_retries(_call_model, max_retries=2)
        except Exception:
            return self._fallback(context)

        self.last_response = content
        return (content or "").strip()

    def generate_feature_engineering_advice(self, context: Dict[str, Any]) -> str:
        """
        FE coach for editor mode (deterministic/llm/hybrid).
        Returns short actionable bullets (no STOP/CONTINUE decisions).
        """
        deterministic = self._generate_feature_engineering_advice_deterministic(context)
        mode = self.fe_advice_mode
        self.last_fe_advice_meta = {
            "mode": mode,
            "source": "deterministic",
            "provider": self.fe_provider,
            "model": self.fe_model_name,
        }
        if mode == "deterministic":
            return deterministic

        llm_text = self._generate_feature_engineering_advice_llm(context)
        if llm_text:
            self.last_fe_advice_meta = {
                "mode": mode,
                "source": "llm",
                "provider": self.fe_provider,
                "model": self.fe_model_name,
            }
            return llm_text

        self.last_fe_advice_meta = {
            "mode": mode,
            "source": "deterministic_fallback",
            "provider": self.fe_provider,
            "model": self.fe_model_name,
        }
        return deterministic

    def _normalize_fe_advice_mode(self, raw: Any) -> str:
        value = str(raw or "").strip().lower()
        if value in {"llm", "hybrid", "deterministic"}:
            return value
        return "hybrid"

    def _normalize_critique_mode(self, raw: Any) -> str:
        value = str(raw or "").strip().lower()
        if value in {"llm", "hybrid", "deterministic"}:
            return value
        return "hybrid"

    def generate_critique_packet(self, context: Dict[str, Any]) -> Dict[str, Any]:
        deterministic = self._generate_critique_packet_deterministic(context)
        mode = self.critique_mode
        self.last_critique_error = {}
        self.last_critique_meta = {
            "mode": mode,
            "source": "deterministic",
            "provider": self.fe_provider,
            "model": self.fe_model_name,
        }
        if mode == "deterministic":
            self.last_critique_packet = deterministic
            return deterministic

        llm_packet = self._generate_critique_packet_llm(context)
        valid_packet, validation_errors = validate_advisor_critique_packet(llm_packet)
        if valid_packet:
            self.last_critique_meta = {
                "mode": mode,
                "source": "llm",
                "provider": self.fe_provider,
                "model": self.fe_model_name,
            }
            self.last_critique_packet = llm_packet
            return llm_packet

        repaired_packet, repair_meta = self._repair_critique_packet(
            llm_packet=llm_packet,
            deterministic_packet=deterministic,
            context=context,
            validation_errors=validation_errors,
        )
        if isinstance(repaired_packet, dict):
            repaired_ok, repaired_errors = validate_advisor_critique_packet(repaired_packet)
            if repaired_ok:
                self.last_critique_meta = {
                    "mode": mode,
                    "source": str(repair_meta.get("source") or "llm_repaired"),
                    "provider": self.fe_provider,
                    "model": self.fe_model_name,
                    "repair_meta": repair_meta,
                }
                self.last_critique_packet = repaired_packet
                return repaired_packet
            validation_errors = repaired_errors

        if llm_packet:
            print(
                "RESULTS_ADVISOR_CRITIQUE_SCHEMA_INVALID: "
                + f"provider={self.fe_provider} model={self.fe_model_name} "
                + f"errors={validation_errors[:4]}"
            )
        self.last_critique_meta = {
            "mode": mode,
            "source": "deterministic_fallback",
            "provider": self.fe_provider,
            "model": self.fe_model_name,
            "validation_errors": validation_errors[:6],
        }
        if repair_meta:
            self.last_critique_meta["repair_meta"] = repair_meta
        if self.last_critique_error:
            self.last_critique_meta["llm_error"] = dict(self.last_critique_error)
        self.last_critique_packet = deterministic
        return deterministic

    def _repair_critique_packet(
        self,
        *,
        llm_packet: Dict[str, Any],
        deterministic_packet: Dict[str, Any],
        context: Dict[str, Any],
        validation_errors: List[str],
    ) -> tuple[Dict[str, Any] | None, Dict[str, Any]]:
        trace: Dict[str, Any] = {
            "attempted": False,
            "source": None,
            "provider": self.fe_provider,
            "model": self.fe_model_name,
            "input_errors": list(validation_errors or [])[:8],
        }
        if not isinstance(llm_packet, dict) or not llm_packet:
            trace["skipped"] = "empty_llm_packet"
            return None, trace
        normalized = self._normalize_critique_packet_candidate(llm_packet, deterministic_packet, context)
        normalized_ok, normalized_errors = validate_advisor_critique_packet(normalized)
        if normalized_ok:
            trace["attempted"] = True
            trace["source"] = "llm_repair_normalized"
            trace["normalized_only"] = True
            print(
                "RESULTS_ADVISOR_CRITIQUE_REPAIR: "
                + f"source=normalize success=True provider={self.fe_provider} model={self.fe_model_name}"
            )
            return normalized, trace

        trace["attempted"] = True
        trace["normalized_only"] = False
        trace["normalize_errors"] = normalized_errors[:6]
        if not self.fe_client or self.fe_provider == "none":
            print(
                "RESULTS_ADVISOR_CRITIQUE_REPAIR: "
                + "source=normalize success=False reason=llm_unavailable"
            )
            return None, trace

        schema = build_results_advisor_critique_response_schema()
        schema_json = self._truncate(json.dumps(schema, ensure_ascii=True), 10000)
        llm_packet_json = self._truncate(json.dumps(llm_packet or {}, ensure_ascii=True), 10000)
        raw_preview = self._truncate(str(self.last_response or ""), 8000)
        deterministic_json = self._truncate(json.dumps(deterministic_packet, ensure_ascii=True), 8000)
        errors_json = self._truncate(json.dumps(validation_errors[:8], ensure_ascii=True), 2000)
        repair_prompt = (
            "You are a strict JSON repair tool. Return ONLY one JSON object, no markdown.\n"
            "Repair the candidate packet so it strictly matches TARGET_SCHEMA.\n"
            "Keep semantic intent. When fields are missing/invalid, use deterministic fallback packet.\n"
            "TARGET_SCHEMA:\n"
            + schema_json
            + "\nVALIDATION_ERRORS:\n"
            + errors_json
            + "\nCANDIDATE_PACKET_JSON:\n"
            + llm_packet_json
            + "\nRAW_RESPONSE_PREVIEW:\n"
            + raw_preview
            + "\nDETERMINISTIC_FALLBACK_PACKET:\n"
            + deterministic_json
        )

        try:
            if self.fe_provider == "gemini":
                generation_config = self._generation_config_for_critique()
                generation_config["response_schema"] = copy.deepcopy(schema)
                repaired_text, used_config = self._generate_gemini_json(
                    repair_prompt,
                    generation_config=generation_config,
                )
                trace["used_response_schema"] = "response_schema" in used_config
            else:
                response = self.fe_client.chat.completions.create(
                    model=self.fe_model_name,
                    messages=[
                        {"role": "system", "content": "Return only valid JSON."},
                        {"role": "user", "content": repair_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                repaired_text = response.choices[0].message.content
                trace["used_response_schema"] = False
            repaired_packet = self._parse_json_object(repaired_text)
            normalized_repaired = self._normalize_critique_packet_candidate(
                repaired_packet,
                deterministic_packet,
                context,
            )
            repaired_ok, repaired_errors = validate_advisor_critique_packet(normalized_repaired)
            trace["repair_errors"] = repaired_errors[:6]
            trace["repair_llm_ok"] = repaired_ok
            if repaired_ok:
                trace["source"] = "llm_repair_pass"
                print(
                    "RESULTS_ADVISOR_CRITIQUE_REPAIR: "
                    + f"source=llm_repair_pass success=True provider={self.fe_provider} model={self.fe_model_name}"
                )
                return normalized_repaired, trace
        except Exception as exc:
            trace["repair_exception"] = f"{type(exc).__name__}: {exc}"[:240]

        print(
            "RESULTS_ADVISOR_CRITIQUE_REPAIR: "
            + f"source=llm_repair_pass success=False provider={self.fe_provider} model={self.fe_model_name}"
        )
        return None, trace

    def _normalize_critique_packet_candidate(
        self,
        packet: Dict[str, Any] | None,
        deterministic_packet: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        base = copy.deepcopy(deterministic_packet if isinstance(deterministic_packet, dict) else {})
        incoming = packet if isinstance(packet, dict) else {}

        def _as_number(value: Any, default: float) -> float:
            try:
                if isinstance(value, bool):
                    raise ValueError("bool is not a numeric payload")
                return float(value)
            except Exception:
                return float(default)

        def _as_int(value: Any, default: int) -> int:
            try:
                if isinstance(value, bool):
                    raise ValueError("bool is not integer payload")
                return int(value)
            except Exception:
                return int(default)

        def _as_bool(value: Any, default: bool) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
            if isinstance(value, str):
                token = value.strip().lower()
                if token in {"1", "true", "yes", "on"}:
                    return True
                if token in {"0", "false", "no", "off"}:
                    return False
            return bool(default)

        def _dedup_strings(values: Any, max_items: int) -> List[str]:
            out: List[str] = []
            seen: set[str] = set()
            if not isinstance(values, list):
                return out
            for item in values:
                text = str(item or "").strip()
                if not text:
                    continue
                key = text.lower()
                if key in seen:
                    continue
                seen.add(key)
                out.append(text)
                if len(out) >= max_items:
                    break
            return out

        severity_map = {
            "critical": "high",
            "severe": "high",
            "blocker": "high",
            "warn": "medium",
            "warning": "medium",
            "med": "medium",
            "minor": "low",
        }
        impact_map = {
            "down": "negative",
            "decrease": "negative",
            "negative_bias": "negative",
            "positive_bias": "positive",
            "up": "positive",
            "increase": "positive",
            "none": "neutral",
            "flat": "neutral",
        }

        normalized: Dict[str, Any] = {
            "packet_type": "advisor_critique_packet",
            "packet_version": "1.0",
            "run_id": str(incoming.get("run_id") or base.get("run_id") or context.get("run_id") or "unknown_run"),
            "iteration": _as_int(incoming.get("iteration"), _as_int(base.get("iteration"), 0)),
            "timestamp_utc": str(
                incoming.get("timestamp_utc")
                or base.get("timestamp_utc")
                or datetime.now(timezone.utc).isoformat()
            ),
            "primary_metric_name": str(
                incoming.get("primary_metric_name")
                or base.get("primary_metric_name")
                or context.get("primary_metric_name")
                or "primary_metric"
            ),
            "higher_is_better": _as_bool(
                incoming.get("higher_is_better"),
                _as_bool(base.get("higher_is_better"), bool(context.get("higher_is_better", True))),
            ),
            "metric_comparison": {},
            "validation_signals": {},
            "error_modes": [],
            "risk_flags": _dedup_strings(
                incoming.get("risk_flags") if "risk_flags" in incoming else base.get("risk_flags"),
                max_items=10,
            ),
            "active_gates_context": _dedup_strings(
                incoming.get("active_gates_context")
                if "active_gates_context" in incoming
                else base.get("active_gates_context"),
                max_items=30,
            ),
            "analysis_summary": self._truncate(
                str(incoming.get("analysis_summary") or base.get("analysis_summary") or "Model critique packet."),
                280,
            ),
            "strictly_no_code_advice": True,
        }

        base_metric = base.get("metric_comparison") if isinstance(base.get("metric_comparison"), dict) else {}
        incoming_metric = incoming.get("metric_comparison") if isinstance(incoming.get("metric_comparison"), dict) else {}
        metric_comparison = {
            "baseline_value": _as_number(
                incoming_metric.get("baseline_value"),
                _as_number(base_metric.get("baseline_value"), 0.0),
            ),
            "candidate_value": _as_number(
                incoming_metric.get("candidate_value"),
                _as_number(base_metric.get("candidate_value"), 0.0),
            ),
            "delta_abs": _as_number(
                incoming_metric.get("delta_abs"),
                _as_number(base_metric.get("delta_abs"), 0.0),
            ),
            "delta_rel": _as_number(
                incoming_metric.get("delta_rel"),
                _as_number(base_metric.get("delta_rel"), 0.0),
            ),
            "min_delta_required": _as_number(
                incoming_metric.get("min_delta_required"),
                _as_number(base_metric.get("min_delta_required"), float(context.get("min_delta", 0.0005) or 0.0005)),
            ),
            "meets_min_delta": _as_bool(
                incoming_metric.get("meets_min_delta"),
                _as_bool(base_metric.get("meets_min_delta"), False),
            ),
        }
        normalized["metric_comparison"] = metric_comparison

        base_validation = base.get("validation_signals") if isinstance(base.get("validation_signals"), dict) else {}
        incoming_validation = (
            incoming.get("validation_signals")
            if isinstance(incoming.get("validation_signals"), dict)
            else {}
        )
        validation_mode = str(
            incoming_validation.get("validation_mode")
            or base_validation.get("validation_mode")
            or "unknown"
        ).strip().lower()
        if validation_mode not in {"cv", "holdout", "cv_and_holdout", "unknown"}:
            validation_mode = "unknown"
        validation_signals: Dict[str, Any] = {"validation_mode": validation_mode}

        def _normalize_cv_payload(raw_cv: Any, fallback_cv: Any) -> Dict[str, Any] | None:
            source_cv = raw_cv if isinstance(raw_cv, dict) else fallback_cv if isinstance(fallback_cv, dict) else None
            if not isinstance(source_cv, dict):
                return None
            variance_level = str(source_cv.get("variance_level") or "unknown").strip().lower()
            if variance_level not in {"low", "medium", "high", "unknown"}:
                variance_level = "unknown"
            return {
                "cv_mean": _as_number(source_cv.get("cv_mean"), 0.0),
                "cv_std": max(0.0, _as_number(source_cv.get("cv_std"), 0.0)),
                "fold_count": max(2, _as_int(source_cv.get("fold_count"), 5)),
                "variance_level": variance_level,
            }

        def _normalize_holdout_payload(raw_holdout: Any, fallback_holdout: Any) -> Dict[str, Any] | None:
            source_holdout = (
                raw_holdout
                if isinstance(raw_holdout, dict)
                else fallback_holdout
                if isinstance(fallback_holdout, dict)
                else None
            )
            if not isinstance(source_holdout, dict):
                return None
            shift = str(source_holdout.get("class_distribution_shift") or "unknown").strip().lower()
            if shift not in {"low", "medium", "high", "unknown"}:
                shift = "unknown"
            payload = {
                "metric_value": _as_number(source_holdout.get("metric_value"), 0.0),
                "split_name": str(source_holdout.get("split_name") or "holdout"),
                "sample_count": max(1, _as_int(source_holdout.get("sample_count"), 1)),
                "class_distribution_shift": shift,
            }
            if isinstance(source_holdout.get("positive_class_rate"), (int, float)):
                rate = float(source_holdout.get("positive_class_rate"))
                payload["positive_class_rate"] = min(1.0, max(0.0, rate))
            return payload

        base_cv = base_validation.get("cv")
        base_holdout = base_validation.get("holdout")
        cv_payload = _normalize_cv_payload(incoming_validation.get("cv"), base_cv)
        holdout_payload = _normalize_holdout_payload(incoming_validation.get("holdout"), base_holdout)
        if validation_mode in {"cv", "cv_and_holdout"} and isinstance(cv_payload, dict):
            validation_signals["cv"] = cv_payload
        if validation_mode in {"holdout", "cv_and_holdout"} and isinstance(holdout_payload, dict):
            validation_signals["holdout"] = holdout_payload
        if "generalization_gap" in incoming_validation or "generalization_gap" in base_validation:
            validation_signals["generalization_gap"] = _as_number(
                incoming_validation.get("generalization_gap"),
                _as_number(base_validation.get("generalization_gap"), 0.0),
            )
        normalized["validation_signals"] = validation_signals

        raw_error_modes = incoming.get("error_modes") if isinstance(incoming.get("error_modes"), list) else []
        if not raw_error_modes and isinstance(base.get("error_modes"), list):
            raw_error_modes = base.get("error_modes")
        normalized_error_modes: List[Dict[str, Any]] = []
        for idx, item in enumerate(raw_error_modes):
            if not isinstance(item, dict):
                continue
            severity_token = str(item.get("severity") or "medium").strip().lower()
            severity_token = severity_map.get(severity_token, severity_token)
            if severity_token not in {"low", "medium", "high"}:
                severity_token = "medium"
            impact_token = str(item.get("metric_impact_direction") or "neutral").strip().lower()
            impact_token = impact_map.get(impact_token, impact_token)
            if impact_token not in {"negative", "neutral", "positive"}:
                impact_token = "neutral"
            confidence = _as_number(item.get("confidence"), 0.5)
            normalized_error_modes.append(
                {
                    "id": str(item.get("id") or f"mode_{idx+1}"),
                    "severity": severity_token,
                    "confidence": min(1.0, max(0.0, confidence)),
                    "evidence": self._truncate(str(item.get("evidence") or "No evidence provided."), 500),
                    "affected_scope": str(item.get("affected_scope") or "model"),
                    "metric_impact_direction": impact_token,
                }
            )
            if len(normalized_error_modes) >= 5:
                break
        normalized["error_modes"] = normalized_error_modes
        return normalized

    def _record_critique_error(
        self,
        *,
        stage: str,
        detail: str,
        raw_preview: Optional[str] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "stage": str(stage or "unknown"),
            "detail": str(detail or "")[:260],
            "provider": self.fe_provider,
            "model": self.fe_model_name,
        }
        if raw_preview:
            payload["raw_preview"] = self._truncate(str(raw_preview), 260)
        self.last_critique_error = payload
        log_line = (
            "RESULTS_ADVISOR_CRITIQUE_LLM_ERROR: "
            + f"stage={payload['stage']} provider={payload.get('provider')} "
            + f"model={payload.get('model')} detail={payload.get('detail')}"
        )
        if raw_preview:
            log_line += " raw_preview=" + str(payload.get("raw_preview"))
        print(log_line)

    def _generate_critique_packet_llm(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(context, dict):
            self._record_critique_error(stage="invalid_context", detail="Context is not a dictionary.")
            return {}
        if not self.fe_client or self.fe_provider == "none":
            self._record_critique_error(
                stage="client_unavailable",
                detail="Reviewer LLM client is unavailable or provider is none.",
            )
            return {}

        phase = str(context.get("phase") or "baseline_review").strip() or "baseline_review"
        payload = {
            "run_id": str(context.get("run_id") or "unknown_run"),
            "iteration": int(context.get("iteration") or 0),
            "phase": phase,
            "primary_metric_name": str(context.get("primary_metric_name") or "primary_metric"),
            "higher_is_better": bool(context.get("higher_is_better", True)),
            "min_delta": float(context.get("min_delta", 0.0005) or 0.0005),
            "baseline_metrics": context.get("baseline_metrics") if isinstance(context.get("baseline_metrics"), dict) else {},
            "candidate_metrics": context.get("candidate_metrics") if isinstance(context.get("candidate_metrics"), dict) else {},
            "active_gates_context": context.get("active_gates_context") if isinstance(context.get("active_gates_context"), list) else [],
            "dataset_profile": context.get("dataset_profile") if isinstance(context.get("dataset_profile"), dict) else {},
            "column_roles": context.get("column_roles") if isinstance(context.get("column_roles"), dict) else {},
        }
        compact_payload = self._truncate(json.dumps(payload, ensure_ascii=True), 14000)
        system_prompt = (
            "You are a senior ML critic. Return ONLY a JSON object with no markdown. "
            "Do not provide code advice. Focus on metric deltas, validation reliability, and error modes. "
            "Phase tells whether this is baseline assessment or candidate review."
        )
        user_prompt = (
            "Populate the critique packet schema below.\n"
            "Reason internally in three passes before writing JSON:\n"
            "1. Compare baseline vs candidate and decide whether any metric gain is material.\n"
            "2. Assess how trustworthy the validation evidence is.\n"
            "3. Surface the smallest set of error modes and risk flags that actually explain the result.\n"
            "Keep the packet concise and evidence-driven.\n\n"
            "JSON schema:\n"
            "{packet_type:'advisor_critique_packet',packet_version:'1.0',run_id,iteration,timestamp_utc,"
            "primary_metric_name,higher_is_better,metric_comparison:{baseline_value,candidate_value,delta_abs,delta_rel,min_delta_required,meets_min_delta},"
            "validation_signals:{validation_mode,cv?,holdout?,generalization_gap?},"
            "error_modes:[{id,severity,confidence,evidence,affected_scope,metric_impact_direction}],"
            "risk_flags:[...],active_gates_context:[...],analysis_summary,strictly_no_code_advice:true}\n"
            "Max 5 error_modes, analysis_summary max 280 chars.\n"
            "CONTEXT_JSON:\n"
            + compact_payload
        )

        try:
            if self.fe_provider == "gemini":
                generation_config = self._generation_config_for_critique()
                raw_text, used_config = self._generate_gemini_json(
                    system_prompt + "\n\n" + user_prompt,
                    generation_config=generation_config,
                )
                if "response_schema" in generation_config and "response_schema" not in used_config:
                    print(
                        "RESULTS_ADVISOR_CRITIQUE_SCHEMA_FALLBACK: "
                        + f"provider={self.fe_provider} model={self.fe_model_name} reason=response_schema_unsupported"
                    )
            else:
                response = self.fe_client.chat.completions.create(
                    model=self.fe_model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                )
                raw_text = response.choices[0].message.content
        except Exception as exc:
            self._record_critique_error(
                stage="api_call_exception",
                detail=f"{type(exc).__name__}: {exc}",
            )
            return {}

        self.last_response = str(raw_text or "")
        packet = self._parse_json_object(raw_text)
        if not packet:
            self._record_critique_error(
                stage="json_parse_failed",
                detail="LLM returned non-JSON or malformed JSON for critique packet.",
                raw_preview=str(raw_text or "")[:800],
            )
            return {}
        return packet

    def _parse_json_object(self, raw_text: Any) -> Dict[str, Any]:
        text = str(raw_text or "").strip()
        if not text:
            return {}
        try:
            parsed, _trace = parse_json_object_with_repair(text, actor="results_advisor")
            if isinstance(parsed, dict):
                return parsed
        except JsonObjectParseError:
            pass
        cleaned = text.replace("```json", "").replace("```", "").strip()
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(cleaned[start : end + 1])
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                return {}
        return {}

    def _metric_higher_is_better_name(self, metric_name: str) -> bool:
        token = self._normalize_metric_token(metric_name)
        lower_is_better_tokens = {
            "rmse",
            "mae",
            "mse",
            "mape",
            "smape",
            "loss",
            "logloss",
            "error",
        }
        return token not in lower_is_better_tokens

    def _extract_cv_fold_count(self, metrics_payload: Dict[str, Any]) -> int:
        for key in ("cv_folds", "n_folds", "fold_count", "cv_fold_count"):
            value = metrics_payload.get(key)
            try:
                parsed = int(value)
            except Exception:
                continue
            if parsed >= 2:
                return parsed
        return 5

    def _extract_cv_and_holdout_signals(
        self,
        metrics_payload: Dict[str, Any],
        metric_name: str,
        dataset_profile: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        flat = self._flatten_numeric_metrics(metrics_payload or {})
        metric_token = self._normalize_metric_token(metric_name)

        cv_metric_value = None
        holdout_metric_value = None
        cv_std_value = None
        for key, value in flat.items():
            key_token = self._normalize_metric_token(key)
            if not key_token:
                continue
            if metric_token and metric_token in key_token:
                if cv_metric_value is None and ("cv" in key_token or "cross" in key_token):
                    cv_metric_value = float(value)
                if holdout_metric_value is None and any(
                    token in key_token for token in ("holdout", "validation", "val", "test")
                ):
                    holdout_metric_value = float(value)
            if cv_std_value is None and "cv" in key_token and "std" in key_token:
                cv_std_value = float(value)

        if cv_std_value is None:
            fallback_std = metrics_payload.get("cv_std")
            if isinstance(fallback_std, (int, float)):
                cv_std_value = float(fallback_std)
        if cv_metric_value is None:
            cv_metric_value = self._extract_metric_value_for_name(metrics_payload, metric_name)
        if holdout_metric_value is None:
            for holdout_key in ("holdout_score", "validation_score", "val_score", "test_score"):
                value = metrics_payload.get(holdout_key)
                if isinstance(value, (int, float)):
                    holdout_metric_value = float(value)
                    break

        validation_mode = "unknown"
        validation_signals: Dict[str, Any] = {}
        if cv_metric_value is not None and holdout_metric_value is not None:
            validation_mode = "cv_and_holdout"
        elif cv_metric_value is not None:
            validation_mode = "cv"
        elif holdout_metric_value is not None:
            validation_mode = "holdout"
        validation_signals["validation_mode"] = validation_mode

        if cv_metric_value is not None:
            cv_std = abs(float(cv_std_value)) if isinstance(cv_std_value, (int, float)) else 0.0
            variance_level = "low"
            if cv_std >= 0.03:
                variance_level = "high"
            elif cv_std >= 0.015:
                variance_level = "medium"
            validation_signals["cv"] = {
                "cv_mean": float(cv_metric_value),
                "cv_std": float(cv_std),
                "fold_count": int(self._extract_cv_fold_count(metrics_payload)),
                "variance_level": variance_level,
            }

        if holdout_metric_value is not None:
            sample_count = None
            for key in ("holdout_sample_count", "sample_count", "validation_rows"):
                value = context.get(key)
                try:
                    parsed = int(value)
                except Exception:
                    continue
                if parsed > 0:
                    sample_count = parsed
                    break
            if sample_count is None:
                rows = dataset_profile.get("n_rows") if isinstance(dataset_profile, dict) else None
                try:
                    parsed_rows = int(rows)
                except Exception:
                    parsed_rows = 0
                sample_count = max(1, parsed_rows) if parsed_rows > 0 else 1
            validation_signals["holdout"] = {
                "metric_value": float(holdout_metric_value),
                "split_name": str(context.get("holdout_split_name") or "holdout"),
                "sample_count": int(sample_count),
                "class_distribution_shift": str(
                    context.get("class_distribution_shift") or "unknown"
                ),
            }

        if cv_metric_value is not None and holdout_metric_value is not None:
            validation_signals["generalization_gap"] = float(holdout_metric_value - cv_metric_value)
        return validation_signals

    def _build_critique_error_modes(
        self,
        *,
        metrics_payload: Dict[str, Any],
        validation_signals: Dict[str, Any],
        meets_min_delta: bool,
    ) -> List[Dict[str, Any]]:
        error_modes: List[Dict[str, Any]] = []
        flat = self._flatten_numeric_metrics(metrics_payload or {})

        minority_recall = None
        for key, value in flat.items():
            token = self._normalize_metric_token(key)
            if "minority" in token and "recall" in token:
                minority_recall = float(value)
                break
        if minority_recall is not None and minority_recall < 0.35:
            error_modes.append(
                {
                    "id": "minority_class_recall_low",
                    "severity": "high",
                    "confidence": 0.85,
                    "evidence": "Minority-class recall below 0.35 in validation outputs.",
                    "affected_scope": "minority_class",
                    "metric_impact_direction": "negative",
                }
            )

        cv_block = validation_signals.get("cv") if isinstance(validation_signals.get("cv"), dict) else {}
        cv_std = cv_block.get("cv_std")
        if isinstance(cv_std, (int, float)) and float(cv_std) >= 0.03:
            error_modes.append(
                {
                    "id": "fold_instability",
                    "severity": "medium",
                    "confidence": 0.78,
                    "evidence": "Cross-validation std indicates high fold variance.",
                    "affected_scope": "cross_validation",
                    "metric_impact_direction": "negative",
                }
            )

        gap = validation_signals.get("generalization_gap")
        if isinstance(gap, (int, float)) and abs(float(gap)) >= 0.02:
            error_modes.append(
                {
                    "id": "generalization_gap_high",
                    "severity": "medium",
                    "confidence": 0.72,
                    "evidence": "Large CV-holdout gap suggests unstable generalization.",
                    "affected_scope": "holdout",
                    "metric_impact_direction": "negative",
                }
            )

        if not bool(meets_min_delta):
            error_modes.append(
                {
                    "id": "delta_below_threshold",
                    "severity": "low",
                    "confidence": 0.8,
                    "evidence": "Metric delta does not meet configured minimum threshold.",
                    "affected_scope": "primary_metric",
                    "metric_impact_direction": "negative",
                }
            )
        return error_modes[:5]

    def _generate_critique_packet_deterministic(self, context: Dict[str, Any]) -> Dict[str, Any]:
        context = context if isinstance(context, dict) else {}
        run_id = str(context.get("run_id") or "unknown_run")
        phase = str(context.get("phase") or "baseline_review").strip() or "baseline_review"
        try:
            iteration = int(context.get("iteration") or 0)
        except Exception:
            iteration = 0
        primary_metric_name = str(context.get("primary_metric_name") or "primary_metric").strip() or "primary_metric"
        higher_is_better = bool(
            context.get("higher_is_better")
            if context.get("higher_is_better") is not None
            else self._metric_higher_is_better_name(primary_metric_name)
        )
        min_delta = self._coerce_number(context.get("min_delta"), ".")
        if min_delta is None:
            min_delta = 0.0005
        min_delta = max(0.0, float(min_delta))

        baseline_metrics = context.get("baseline_metrics")
        if not isinstance(baseline_metrics, dict):
            baseline_metrics = {}
        candidate_metrics = context.get("candidate_metrics")
        if not isinstance(candidate_metrics, dict):
            candidate_metrics = baseline_metrics

        baseline_value = self._extract_metric_value_for_name(baseline_metrics, primary_metric_name)
        candidate_value = self._extract_metric_value_for_name(candidate_metrics, primary_metric_name)
        if baseline_value is None:
            baseline_value = 0.0
        if candidate_value is None:
            candidate_value = baseline_value
        delta_abs = float(candidate_value - baseline_value)
        baseline_abs = abs(float(baseline_value)) if abs(float(baseline_value)) > 1e-9 else 1.0
        delta_rel = float(delta_abs / baseline_abs)
        if higher_is_better:
            meets_min_delta = delta_abs >= min_delta
        else:
            meets_min_delta = (-delta_abs) >= min_delta

        dataset_profile = context.get("dataset_profile")
        if not isinstance(dataset_profile, dict):
            dataset_profile = {}
        validation_signals = self._extract_cv_and_holdout_signals(
            candidate_metrics,
            primary_metric_name,
            dataset_profile,
            context,
        )
        error_modes = self._build_critique_error_modes(
            metrics_payload=candidate_metrics,
            validation_signals=validation_signals,
            meets_min_delta=meets_min_delta,
        )
        risk_flags: List[str] = []
        for mode in error_modes:
            mode_id = str(mode.get("id") or "")
            if mode_id == "minority_class_recall_low":
                risk_flags.append("class_imbalance_sensitivity")
            elif mode_id == "fold_instability":
                risk_flags.append("potential_overfitting")
            elif mode_id == "generalization_gap_high":
                risk_flags.append("generalization_instability")
        if not risk_flags and not meets_min_delta:
            risk_flags.append("no_material_metric_gain")

        active_gates = context.get("active_gates_context")
        if not isinstance(active_gates, list):
            active_gates = []
        active_gates = [str(item) for item in active_gates if str(item).strip()][:30]

        summary_parts = []
        if phase == "candidate_review":
            summary_parts.append("Candidate round assessment.")
        else:
            summary_parts.append("Baseline assessment context.")
        summary_parts.append("No material gain vs baseline." if not meets_min_delta else "Candidate meets min delta.")
        if any(mode.get("id") == "minority_class_recall_low" for mode in error_modes):
            summary_parts.append("Minority-class signal remains weak.")
        if any(mode.get("id") == "fold_instability" for mode in error_modes):
            summary_parts.append("Cross-validation variance is elevated.")
        analysis_summary = " ".join(summary_parts).strip() or "Deterministic critique packet generated."
        analysis_summary = analysis_summary[:280]

        packet: Dict[str, Any] = {
            "packet_type": "advisor_critique_packet",
            "packet_version": "1.0",
            "run_id": run_id,
            "iteration": iteration,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "primary_metric_name": primary_metric_name,
            "higher_is_better": higher_is_better,
            "metric_comparison": {
                "baseline_value": float(baseline_value),
                "candidate_value": float(candidate_value),
                "delta_abs": float(delta_abs),
                "delta_rel": float(delta_rel),
                "min_delta_required": float(min_delta),
                "meets_min_delta": bool(meets_min_delta),
            },
            "validation_signals": validation_signals,
            "error_modes": error_modes,
            "risk_flags": list(dict.fromkeys(risk_flags)),
            "active_gates_context": active_gates,
            "analysis_summary": analysis_summary,
            "strictly_no_code_advice": True,
        }
        valid_packet, errors = validate_advisor_critique_packet(packet)
        if valid_packet:
            return packet
        packet["error_modes"] = []
        packet["risk_flags"] = ["critique_schema_repair_applied"]
        packet["analysis_summary"] = "Deterministic critique packet generated with limited diagnostics."
        packet["validation_signals"] = {"validation_mode": "unknown"}
        packet["strictly_no_code_advice"] = True
        _, _ = validate_advisor_critique_packet(packet)
        if errors:
            packet["_validation_warnings"] = errors[:4]
        return packet

    def _generate_feature_engineering_advice_llm(self, context: Dict[str, Any]) -> str:
        if not isinstance(context, dict):
            return ""
        if not self.fe_client or self.fe_provider == "none":
            return ""

        primary_metric = str(context.get("primary_metric_name") or "primary_metric").strip() or "primary_metric"
        payload = {
            "primary_metric_name": primary_metric,
            "baseline_metrics": context.get("baseline_metrics") if isinstance(context.get("baseline_metrics"), dict) else {},
            "feature_engineering_plan": context.get("feature_engineering_plan") if isinstance(context.get("feature_engineering_plan"), dict) else {},
            "dataset_profile": context.get("dataset_profile") if isinstance(context.get("dataset_profile"), dict) else {},
            "column_roles": context.get("column_roles") if isinstance(context.get("column_roles"), dict) else {},
            "baseline_ml_script_snippet": self._truncate(str(context.get("baseline_ml_script_snippet") or ""), 5000),
        }
        compact_payload = self._truncate(json.dumps(payload, ensure_ascii=True), 14000)
        system_prompt = (
            "You are a senior ML feature-engineering coach. "
            "Return ONLY concise bullet points in plain text (max 12 lines). "
            "Do not decide STOP/CONTINUE. "
            "Give incremental script-edit guidance for editor mode. "
            "Reason from the current script, plan, and dataset signals before suggesting edits. "
            "Focus on: where to edit, leakage guards, CV-safe fitting, and contract alignment."
        )
        user_prompt = (
            "CONTEXT_JSON:\n"
            + compact_payload
            + "\n\nINSTRUCTIONS:\n"
            "- Keep existing script structure and change the minimum necessary.\n"
            "- If plan techniques exist, prioritize the ones with the clearest ROI for the current signals.\n"
            "- If plan is empty, provide only low-cost, evidence-backed suggestions.\n"
            "- Mention where to edit (e.g., build_features, preprocessing block).\n"
        )

        try:
            if self.fe_provider == "gemini":
                response = self.fe_client.generate_content(system_prompt + "\n\n" + user_prompt)
                content = response.text
            else:
                response = self.fe_client.chat.completions.create(
                    model=self.fe_model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.1,
                )
                content = response.choices[0].message.content
        except Exception:
            return ""

        return self._normalize_llm_advice_bullets(content)

    def _normalize_llm_advice_bullets(self, raw_text: Any) -> str:
        text = str(raw_text or "").strip()
        if not text:
            return ""
        text = text.replace("```text", "").replace("```", "").strip()
        lines: List[str] = []
        for raw_line in text.splitlines():
            line = str(raw_line or "").strip()
            if not line:
                continue
            lowered = line.lower()
            if lowered in {"advice:", "suggestions:", "output:"}:
                continue
            if line.startswith(("- ", "* ")):
                cleaned = "- " + line[2:].strip()
            elif re.match(r"^\d+\.\s+", line):
                cleaned = "- " + re.sub(r"^\d+\.\s+", "", line).strip()
            else:
                cleaned = "- " + line
            if cleaned not in lines:
                lines.append(cleaned)
            if len(lines) >= 12:
                break
        return "\n".join(lines[:12]).strip()

    def _generate_feature_engineering_advice_deterministic(self, context: Dict[str, Any]) -> str:
        """
        Deterministic FE coach for editor mode.
        Returns short actionable bullets (no STOP/CONTINUE decisions).
        """
        if not isinstance(context, dict):
            return "- No-op suggestions: keep baseline script and only ensure required outputs/metrics persist."

        baseline_metrics = context.get("baseline_metrics")
        if not isinstance(baseline_metrics, dict):
            baseline_metrics = {}
        primary_metric = str(context.get("primary_metric_name") or "primary_metric").strip() or "primary_metric"
        baseline_snippet = str(context.get("baseline_ml_script_snippet") or "").strip()
        fe_plan = context.get("feature_engineering_plan")
        if not isinstance(fe_plan, dict):
            fe_plan = {}

        techniques_raw = fe_plan.get("techniques")
        techniques: List[Any] = techniques_raw if isinstance(techniques_raw, list) else []
        plan_notes = str(fe_plan.get("notes") or "").strip()
        current_metric = self._extract_metric_value_for_name(baseline_metrics, primary_metric)

        lines: List[str] = []
        metric_value_text = "n/a" if current_metric is None else f"{current_metric:.6f}".rstrip("0").rstrip(".")
        lines.append(f"- Objetivo: subir {primary_metric} desde baseline={metric_value_text} con cambios minimos al script actual.")
        lines.append("- Punto de insercion: crea o extiende `build_features(df)` y llamala antes del split/CV.")
        lines.append("- Guardrail: cualquier transformacion aprendida (encoders/imputers/grouping) debe ajustarse dentro de cada fold de CV.")

        if techniques:
            for technique_line in self._render_plan_technique_lines(techniques):
                lines.append(technique_line)
        else:
            fallback_lines = self._build_universal_fallback_fe_lines(context)
            lines.extend(fallback_lines)

        if "build_features(" in baseline_snippet:
            lines.append("- Reusa la funcion `build_features` existente y modifica solo bloques internos, sin reestructurar training/persistencia.")
        else:
            lines.append("- Si no existe `build_features`, agrega una funcion compacta y deja intacto el resto del pipeline.")

        if plan_notes:
            lines.append(f"- Nota del plan: {plan_notes[:180]}")

        # Deterministic, short payload: max ~12 lines.
        compact = [line for line in lines if isinstance(line, str) and line.strip()]
        return "\n".join(compact[:12])

    def generate_insights(self, context: Dict[str, Any]) -> Dict[str, Any]:
        produced_index = (
            context.get("produced_artifact_index")
            or context.get("artifact_index")
            or self._safe_load_json("data/produced_artifact_index.json")
            or []
        )
        artifact_index = self._normalize_artifact_index(produced_index)
        if not artifact_index:
            output_report = context.get("output_contract_report") or self._safe_load_json("data/output_contract_report.json")
            present = output_report.get("present", []) if isinstance(output_report, dict) else []
            artifact_index = [
                {"path": path, "artifact_type": self._infer_artifact_type(path)}
                for path in present
                if path
            ]
        objective_type = context.get("objective_type") or context.get("strategy_spec", {}).get("objective_type") or "unknown"

        reporting_policy = context.get("reporting_policy") if isinstance(context, dict) else {}
        slot_defs = reporting_policy.get("slots", []) if isinstance(reporting_policy, dict) else []
        allowed_slots = {slot.get("id") for slot in slot_defs if isinstance(slot, dict) and slot.get("id")}

        metrics_artifacts = self._find_artifacts_by_type(artifact_index, "metrics")
        predictions_artifacts = self._find_artifacts_by_type(artifact_index, "predictions")
        error_artifacts = self._find_artifacts_by_type(artifact_index, "error_analysis")
        importances_artifacts = self._find_artifacts_by_type(artifact_index, "feature_importances")

        metrics_payload = self._safe_load_json(metrics_artifacts[0]) if metrics_artifacts else {}
        predictions_summary = self._summarize_csv(predictions_artifacts[0]) if predictions_artifacts else {}
        error_payload = self._safe_load_json(error_artifacts[0]) if error_artifacts else {}
        importances_payload = self._safe_load_json(importances_artifacts[0]) if importances_artifacts else {}
        alignment_payload = self._safe_load_json("data/alignment_check.json") or {}
        case_alignment_payload = self._safe_load_json("data/case_alignment_report.json") or {}

        metrics_summary = self._extract_metrics_summary(metrics_payload, objective_type)
        if not metrics_summary and isinstance(metrics_payload, dict):
            nested = metrics_payload.get("metrics")
            metrics_summary = self._extract_metrics_summary(nested, objective_type)
        primary_metric_name = self._resolve_primary_metric_name(
            context,
            metrics_payload,
            metrics_summary,
            objective_type,
        )
        deployment_info = self._compute_deployment_recommendation(metrics_payload, predictions_summary)
        if primary_metric_name:
            deployment_info = dict(deployment_info)
            deployment_info["primary_metric"] = primary_metric_name
        risks = []
        recommendations = []
        summary_lines: List[str] = []

        metrics_present = bool(metrics_artifacts) and isinstance(metrics_payload, dict)
        if not metrics_summary and not metrics_present:
            risks.append("Metrics artifact missing or empty; evaluation confidence is limited.")
            recommendations.append("Generate a metrics artifact aligned to the objective type.")
        elif not metrics_summary and metrics_present:
            recommendations.append("Metrics artifact present but no numeric metrics detected; populate key metrics.")
        if predictions_summary:
            summary_lines.append(
                f"Predictions preview includes {predictions_summary.get('row_count', 0)} rows."
            )
        if error_payload:
            summary_lines.append("Error analysis artifact available; review failure patterns.")
        if importances_payload:
            summary_lines.append("Feature importance artifact available; use for explainability.")

        output_report = context.get("output_contract_report", {}) or {}
        missing_required = output_report.get("missing", []) if isinstance(output_report, dict) else []
        if missing_required:
            risks.append(f"Missing required outputs: {missing_required}")
            recommendations.append("Ensure required artifacts are written before pipeline completion.")

        review_feedback = str(context.get("review_feedback") or "")
        leakage_risk_flagged = self._feedback_indicates_leakage_risk(review_feedback) or self._alignment_indicates_leakage_risk(alignment_payload)
        if leakage_risk_flagged:
            risks.append("Leakage risk signal detected in reviewer/alignment evidence.")
            recommendations.append("Audit feature availability timing and exclude post-outcome fields.")

        capabilities = infer_problem_capabilities(
            objective_text=str(objective_type or ""),
            objective_type=objective_type,
        )
        if is_problem_family(capabilities, "classification"):
            recommendations.append("Check class balance and calibrate thresholds if needed.")
        elif is_problem_family(capabilities, "regression"):
            recommendations.append("Inspect residuals and consider robust loss if heavy tails exist.")
        elif is_problem_family(capabilities, "forecasting"):
            recommendations.append("Validate forecast horizon and compare against naive baselines.")
        elif is_problem_family(capabilities, "ranking"):
            recommendations.append("Validate ordering metrics and consider pairwise loss if rankings are unstable.")
        elif is_problem_family(capabilities, "survival_analysis"):
            recommendations.append("Validate censoring handling and horizon calibration for survival risk outputs.")
        elif is_problem_family(capabilities, "clustering"):
            recommendations.append("Stress-test cluster stability and interpretability before using segments operationally.")
        elif is_problem_family(capabilities, "optimization"):
            recommendations.append("Verify constraint satisfaction and compare objective lift against simple baseline policies.")

        # PR4: ResultsAdvisor is a pure critic; loop control belongs to graph policy.
        iteration_recommendation: Dict[str, Any] = {}

        artifacts_used = []
        for path in (metrics_artifacts + predictions_artifacts + error_artifacts + importances_artifacts):
            artifacts_used.append(path)

        segment_pricing_summary = self._build_segment_pricing_summary(predictions_artifacts[0]) if predictions_artifacts else []
        leakage_audit = self._extract_leakage_audit(alignment_payload)
        validation_summary = self._extract_validation_summary(alignment_payload)
        case_or_bucket_summary = self._extract_case_alignment_summary(case_alignment_payload)

        slot_payloads = {}
        if metrics_summary:
            slot_payloads["model_metrics"] = metrics_summary
        if predictions_summary:
            slot_payloads["predictions_overview"] = predictions_summary
        if segment_pricing_summary:
            slot_payloads["segment_pricing"] = segment_pricing_summary
        if leakage_audit:
            slot_payloads["alignment_risks"] = leakage_audit
        if validation_summary:
            slot_payloads["validation_summary"] = validation_summary
        if case_or_bucket_summary:
            slot_payloads["case_or_bucket_summary"] = case_or_bucket_summary
        if allowed_slots:
            slot_payloads = {k: v for k, v in slot_payloads.items() if k in allowed_slots}

        insights = {
            "schema_version": "1",
            "objective_type": objective_type,
            "artifacts_used": artifacts_used,
            "metrics_summary": metrics_summary,
            "predictions_summary": predictions_summary,
            "overall_scored_rows_row_count": predictions_summary.get("row_count") if isinstance(predictions_summary, dict) else None,
            "segment_pricing_summary": segment_pricing_summary,
            "leakage_audit": leakage_audit,
            "slot_payloads": slot_payloads,
            "risks": risks,
            "recommendations": recommendations,
            "summary_lines": summary_lines,
            "deployment_recommendation": deployment_info.get("deployment_recommendation"),
            "confidence": deployment_info.get("confidence"),
            "primary_metric": deployment_info.get("primary_metric") or primary_metric_name,
            "iteration_recommendation": iteration_recommendation,
        }
        self.last_response = insights
        if not summary_lines and metrics_summary:
            summary_lines.append("Metrics artifact available; review key performance indicators.")
        if not summary_lines:
            summary_lines.append("Limited artifacts available; generate metrics and predictions for insights.")
        return insights

    def _truncate(self, text: str, max_len: int) -> str:
        if not text:
            return ""
        if len(text) <= max_len:
            return text
        return text[:max_len]

    def _normalize_metric_token(self, value: str) -> str:
        return re.sub(r"[^0-9a-zA-Z]+", "", str(value or "").lower())

    def _extract_metric_value_for_name(self, metrics: Dict[str, Any], metric_name: str) -> Optional[float]:
        if not isinstance(metrics, dict):
            return None
        target = self._normalize_metric_token(metric_name)
        if not target:
            return None
        flat = self._flatten_numeric_metrics(metrics)
        ranked_keys: List[str] = []
        for key in flat.keys():
            norm = self._normalize_metric_token(key)
            if not norm:
                continue
            if norm == target:
                ranked_keys.insert(0, key)
            elif target in norm:
                ranked_keys.append(key)
        for key in ranked_keys:
            val = flat.get(key)
            if isinstance(val, (int, float)):
                return float(val)
        # Generic fallback
        for key in ("primary_metric_value", "metric_value", "score", "value"):
            val = metrics.get(key)
            num = self._coerce_number(val, ".")
            if num is not None:
                return float(num)
        return None

    def _render_plan_technique_lines(self, techniques: List[Any]) -> List[str]:
        lines: List[str] = []
        seen: set[str] = set()
        for item in techniques:
            if len(lines) >= 5:
                break
            if isinstance(item, dict):
                technique = str(item.get("technique") or item.get("name") or "").strip()
                columns = item.get("columns")
                if isinstance(columns, list) and columns:
                    col_text = ", ".join(str(col) for col in columns[:4])
                else:
                    col_text = ""
            else:
                technique = str(item or "").strip()
                col_text = ""
            if not technique:
                continue
            key = self._normalize_metric_token(technique)
            if key in seen:
                continue
            seen.add(key)
            if col_text:
                lines.append(f"- Tecnica: {technique}. Aplicala en columnas [{col_text}] dentro de `build_features(df)`.")
            else:
                lines.append(f"- Tecnica: {technique}. Integrala en `build_features(df)` con validacion dentro de CV.")
        return lines

    def _build_universal_fallback_fe_lines(self, context: Dict[str, Any]) -> List[str]:
        dataset_profile = context.get("dataset_profile")
        if not isinstance(dataset_profile, dict):
            dataset_profile = {}
        missingness = self._has_missingness_signal(dataset_profile)
        high_cardinality = self._has_high_cardinality_signal(dataset_profile)
        lines: List[str] = []
        if missingness:
            lines.append("- Fallback #1: agrega missing indicators (`col_is_missing`) para columnas con nulos relevantes.")
        if high_cardinality:
            lines.append("- Fallback #2: agrupa categorias raras en `__OTHER__` antes del encoding para reducir ruido.")
        if not lines:
            lines.append("- No-op suggestions: no hay senales fuertes para FE adicional; conserva baseline y robustece validacion.")
        return lines[:2]

    def _has_missingness_signal(self, profile: Dict[str, Any]) -> bool:
        if not isinstance(profile, dict):
            return False
        if profile.get("features_with_nulls"):
            return True
        missing = profile.get("missingness")
        if isinstance(missing, dict) and missing:
            return True
        ratio = self._coerce_number(profile.get("missing_ratio"), ".")
        if ratio is not None and ratio > 0:
            return True
        return False

    def _has_high_cardinality_signal(self, profile: Dict[str, Any]) -> bool:
        if not isinstance(profile, dict):
            return False
        candidates = [
            profile.get("high_cardinality_columns"),
            profile.get("high_cardinality"),
        ]
        for entry in candidates:
            if isinstance(entry, list) and entry:
                return True
            if isinstance(entry, dict) and entry:
                return True
        return False

    def _safe_load_json(self, path: str) -> Any:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except Exception:
            return {}

    def _load_output_dialect(self, manifest_path: str = "data/cleaning_manifest.json") -> Dict[str, Any]:
        manifest = self._safe_load_json(manifest_path)
        if not isinstance(manifest, dict):
            return {}
        dialect = manifest.get("output_dialect") or manifest.get("dialect") or {}
        if not isinstance(dialect, dict):
            return {}
        sep = dialect.get("sep") or dialect.get("delimiter")
        decimal = dialect.get("decimal")
        encoding = dialect.get("encoding")
        cleaned = {}
        if sep:
            cleaned["sep"] = str(sep)
        if decimal:
            cleaned["decimal"] = str(decimal)
        if encoding:
            cleaned["encoding"] = str(encoding)
        return cleaned

    def _sniff_csv_dialect(self, path: str) -> Dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = []
                for _ in range(5):
                    line = f.readline()
                    if not line:
                        break
                    lines.append(line)
        except Exception:
            return {"sep": ",", "decimal": ".", "encoding": "utf-8"}

        sample_text = "".join(lines)
        header = lines[0] if lines else ""
        sep = ";" if header.count(";") > header.count(",") else ","
        comma_decimals = len(re.findall(r"\d+,\d+", sample_text))
        dot_decimals = len(re.findall(r"\d+\.\d+", sample_text))
        decimal = "," if comma_decimals > dot_decimals else "."
        return {"sep": sep, "decimal": decimal, "encoding": "utf-8"}

    def _coerce_number(self, raw: Any, decimal: str) -> Optional[float]:
        if raw is None:
            return None
        text = str(raw).strip()
        if not text:
            return None
        text = text.replace(" ", "")
        if decimal == ",":
            if text.count(",") == 1 and text.count(".") >= 1:
                text = text.replace(".", "")
            text = text.replace(",", ".")
        try:
            return float(text)
        except Exception:
            return None

    def _read_csv_summary(self, path: str, dialect: Dict[str, Any], max_rows: int) -> Dict[str, Any]:
        sep = dialect.get("sep") or ","
        decimal = dialect.get("decimal") or "."
        encoding = dialect.get("encoding") or "utf-8"
        rows: List[Dict[str, Any]] = []
        row_count_total = 0
        try:
            with open(path, "r", encoding=str(encoding), errors="replace") as f:
                reader = csv.DictReader(f, delimiter=str(sep))
                columns = reader.fieldnames or []
                for row in reader:
                    row_count_total += 1
                    if len(rows) < max_rows:
                        rows.append(row)
            numeric_cols = self._numeric_columns(rows, columns, decimal)
            return {
                "row_count": row_count_total,
                "row_count_total": row_count_total,
                "row_count_sampled": len(rows),
                "columns": columns,
                "numeric_columns": numeric_cols,
                "examples": rows[: min(5, len(rows))],
                "dialect_used": {"sep": sep, "decimal": decimal, "encoding": encoding},
            }
        except Exception:
            return {}

    def _summarize_csv(self, path: str, max_rows: int = 200) -> Dict[str, Any]:
        if not path:
            return {}
        manifest_dialect = self._load_output_dialect()
        dialect = manifest_dialect or self._sniff_csv_dialect(path)
        summary = self._read_csv_summary(path, dialect, max_rows)
        columns = summary.get("columns", [])
        if (
            isinstance(columns, list)
            and len(columns) == 1
            and isinstance(columns[0], str)
            and ";" in columns[0]
        ):
            sniffed = self._sniff_csv_dialect(path)
            if sniffed.get("sep") != dialect.get("sep"):
                summary = self._read_csv_summary(path, sniffed, max_rows)
        return summary

    def _numeric_columns(self, rows: List[Dict[str, Any]], columns: List[str], decimal: str) -> List[str]:
        numeric_cols: List[str] = []
        for col in columns:
            values = []
            for row in rows:
                raw = row.get(col)
                if raw in (None, ""):
                    continue
                values.append(self._coerce_number(raw, decimal))
            non_null = [v for v in values if v is not None]
            if non_null and len(non_null) >= max(1, len(values) // 3):
                numeric_cols.append(col)
        return numeric_cols

    def _extract_metrics_summary(
        self,
        metrics: Dict[str, Any],
        objective_type: str = "unknown",
        max_items: int = 8,
    ) -> List[Dict[str, Any]]:
        if not isinstance(metrics, dict):
            return []
        items: List[Dict[str, Any]] = []
        objective = str(objective_type or "unknown").lower()
        model_perf = metrics.get("model_performance") if isinstance(metrics.get("model_performance"), dict) else {}
        seg_stats = metrics.get("segmentation_stats") if isinstance(metrics.get("segmentation_stats"), dict) else {}
        priority_keys = [
            "primary_metric_value",
            "accuracy",
            "roc_auc",
            "auc",
            "pr_auc",
            "average_precision",
            "normalized_gini",
            "gini",
            "precision",
            "f1",
            "recall",
            "rmsle",
            "logloss",
            "rmse",
            "mae",
            "mape",
            "r2",
            "silhouette_score",
            "training_samples",
        ]
        for key in priority_keys:
            if key in model_perf:
                num = self._coerce_number(model_perf.get(key), ".")
                if num is not None:
                    items.append({"metric": f"model_performance.{key}", "value": num})
        seg_keys = ["n_segments", "min_segment_size", "median_segment_size"]
        for key in seg_keys:
            if key in seg_stats:
                num = self._coerce_number(seg_stats.get(key), ".")
                if num is not None:
                    items.append({"metric": f"segmentation_stats.{key}", "value": num})
        if len(items) >= max_items:
            return items[:max_items]
        flat_metrics = self._flatten_numeric_metrics(metrics)
        priority_tokens = self._objective_metric_priority(objective)
        used_keys = {item["metric"] for item in items if isinstance(item, dict) and item.get("metric")}
        for token in priority_tokens:
            for key, value in flat_metrics.items():
                norm = key.lower()
                if token in norm and key not in used_keys:
                    items.append({"metric": key, "value": value})
                    used_keys.add(key)
                    if len(items) >= max_items:
                        return items
        if len(items) >= max_items:
            return items[:max_items]
        for key, value in flat_metrics.items():
            if key in used_keys:
                continue
            items.append({"metric": key, "value": value})
            if len(items) >= max_items:
                break
        return items

    def _normalize_metric_key(self, name: str) -> str:
        return re.sub(r"[^0-9a-zA-Z]+", "", str(name).lower())

    def _pick_primary_metric_with_ci(self, model_perf: Dict[str, Any]) -> tuple[str | None, Dict[str, Any] | None]:
        if not isinstance(model_perf, dict):
            return None, None
        for key, value in model_perf.items():
            if self._normalize_metric_key(key) == "revenuelift" and isinstance(value, dict):
                return str(key), value
        for key, value in model_perf.items():
            if isinstance(value, dict) and all(k in value for k in ("mean", "ci_lower", "ci_upper")):
                return str(key), value
        return None, None

    def _resolve_primary_metric_name(
        self,
        context: Dict[str, Any],
        metrics_payload: Dict[str, Any],
        metrics_summary: List[Dict[str, Any]],
        objective_type: str,
    ) -> str | None:
        if not isinstance(context, dict):
            return None
        snapshot = context.get("primary_metric_snapshot") or {}
        if isinstance(snapshot, dict):
            name = snapshot.get("primary_metric_name")
            if isinstance(name, str) and name.strip():
                return name.strip()
        if isinstance(metrics_payload, dict):
            payload_primary = metrics_payload.get("primary_metric")
            if isinstance(payload_primary, str) and payload_primary.strip():
                return payload_primary.strip()
            model_perf = (
                metrics_payload.get("model_performance")
                if isinstance(metrics_payload.get("model_performance"), dict)
                else {}
            )
            perf_primary = model_perf.get("primary_metric")
            if isinstance(perf_primary, str) and perf_primary.strip():
                return perf_primary.strip()
            perf_primary_name = model_perf.get("primary_metric_name")
            if isinstance(perf_primary_name, str) and perf_primary_name.strip():
                return perf_primary_name.strip()
        execution_contract = context.get("execution_contract")
        if isinstance(execution_contract, dict):
            validation = execution_contract.get("validation_requirements")
            if isinstance(validation, dict):
                primary = validation.get("primary_metric")
                if isinstance(primary, str) and primary.strip():
                    return primary.strip()
            evaluation = execution_contract.get("evaluation_spec")
            if isinstance(evaluation, dict):
                primary = evaluation.get("primary_metric")
                if isinstance(primary, str) and primary.strip():
                    return primary.strip()
        evaluation_spec = context.get("evaluation_spec")
        if isinstance(evaluation_spec, dict):
            validation = evaluation_spec.get("validation_requirements")
            if isinstance(validation, dict):
                primary = validation.get("primary_metric")
                if isinstance(primary, str) and primary.strip():
                    return primary.strip()
            primary = evaluation_spec.get("primary_metric")
            if isinstance(primary, str) and primary.strip():
                return primary.strip()
            qa_gates = evaluation_spec.get("qa_gates")
            if isinstance(qa_gates, list):
                for gate in qa_gates:
                    if not isinstance(gate, dict):
                        continue
                    params = gate.get("params")
                    if isinstance(params, dict):
                        metric = params.get("metric")
                        if isinstance(metric, str) and metric.strip():
                            return metric.strip()
        if isinstance(metrics_summary, list) and metrics_summary:
            first = metrics_summary[0]
            if isinstance(first, dict):
                metric = first.get("metric")
                if isinstance(metric, str) and metric.strip():
                    return metric.strip()
        model_perf = metrics_payload.get("model_performance") if isinstance(metrics_payload.get("model_performance"), dict) else {}
        priority_tokens = self._objective_metric_priority(objective_type)
        for token in priority_tokens:
            for key in model_perf.keys():
                if token in str(key).lower():
                    return str(key)
        return None

    def _extract_row_count(self, metrics: Dict[str, Any], predictions_summary: Dict[str, Any]) -> Optional[int]:
        row_count = None
        if isinstance(predictions_summary, dict):
            rc = predictions_summary.get("row_count")
            num = self._coerce_number(rc, ".")
            if num is not None:
                row_count = int(num)
        model_perf = metrics.get("model_performance") if isinstance(metrics.get("model_performance"), dict) else {}
        for key in ["training_samples", "n_samples", "n_rows", "rows", "row_count"]:
            if row_count is not None:
                break
            num = self._coerce_number(model_perf.get(key), ".")
            if num is not None:
                row_count = int(num)
        return row_count

    def _compute_deployment_recommendation(
        self,
        metrics: Dict[str, Any],
        predictions_summary: Dict[str, Any],
        min_rows: int = 200,
    ) -> Dict[str, Any]:
        recommendation = "PILOT"
        confidence = "LOW"
        model_perf = metrics.get("model_performance") if isinstance(metrics.get("model_performance"), dict) else {}
        metric_name, metric_payload = self._pick_primary_metric_with_ci(model_perf)
        if not metric_name or not isinstance(metric_payload, dict):
            return {
                "deployment_recommendation": recommendation,
                "confidence": confidence,
            }
        mean = self._coerce_number(metric_payload.get("mean"), ".")
        lower = self._coerce_number(metric_payload.get("ci_lower"), ".")
        upper = self._coerce_number(metric_payload.get("ci_upper"), ".")
        if mean is None or lower is None or upper is None:
            return {
                "deployment_recommendation": recommendation,
                "confidence": confidence,
            }
        width = max(0.0, upper - lower)
        normalized_width = width / max(abs(mean), 1.0)
        non_negative = all(val >= 0 for val in [mean, lower, upper])
        ratio_like = non_negative and all(abs(val - 1.0) <= 0.2 for val in [mean, lower, upper])
        baseline = 1.0 if ratio_like else 0.0
        includes_baseline = lower <= baseline <= upper
        row_count = self._extract_row_count(metrics, predictions_summary)

        if includes_baseline:
            recommendation = "PILOT"
            confidence = "MEDIUM" if normalized_width <= 0.1 else "LOW"
        else:
            if row_count is not None and row_count >= min_rows:
                recommendation = "GO"
                confidence = "HIGH" if normalized_width <= 0.1 else "MEDIUM"
            else:
                recommendation = "PILOT"
                confidence = "MEDIUM" if normalized_width <= 0.1 else "LOW"

        return {
            "deployment_recommendation": recommendation,
            "confidence": confidence,
            "primary_metric": metric_name,
        }

    def _feedback_indicates_leakage_risk(self, text: str) -> bool:
        lowered = str(text or "").lower()
        if "leak" not in lowered:
            return False
        sentences = re.split(r"[\n\r.!?]+", lowered)
        if not sentences:
            sentences = [lowered]
        negative_tokens = (
            "no leakage",
            "without leakage",
            "leakage-free",
            "leakage free",
            "prevents leakage",
            "prevent leakage",
            "prevents target leakage",
            "mitigate leakage",
            "mitigated leakage",
            "no deterministic target leakage",
            "leakage check passed",
            "no leakage detected",
        )
        positive_tokens = (
            "leakage risk",
            "risk of leakage",
            "leakage detected",
            "target leakage detected",
            "possible leakage",
            "potential leakage",
            "leakage found",
            "leaky feature",
            "post-outcome",
            "post outcome",
        )
        for sentence in sentences:
            fragment = sentence.strip()
            if "leak" not in fragment:
                continue
            if any(token in fragment for token in negative_tokens):
                continue
            if any(token in fragment for token in positive_tokens):
                return True
            if "risk" in fragment or "detected" in fragment or "found" in fragment:
                return True
        return False

    def _alignment_indicates_leakage_risk(self, alignment_payload: Any) -> bool:
        if not isinstance(alignment_payload, dict) or not alignment_payload:
            return False
        failure_mode = str(alignment_payload.get("failure_mode") or "").lower()
        if "leak" in failure_mode:
            return True

        requirements = alignment_payload.get("requirements")
        if isinstance(requirements, list):
            for req in requirements:
                if not isinstance(req, dict):
                    continue
                name = str(req.get("name") or req.get("id") or "").lower()
                if "leak" not in name:
                    continue
                status = str(req.get("status") or req.get("state") or "").upper()
                if status in {"FAIL", "FAILED", "ERROR", "REJECTED", "WARN", "WARNING"}:
                    return True
                if bool(req.get("detected")) or bool(req.get("is_leakage")):
                    return True

        leak_payload = self._extract_leakage_audit(alignment_payload)
        if isinstance(leak_payload, dict):
            action = str(leak_payload.get("action_taken") or "").lower()
            corr = self._coerce_number(leak_payload.get("correlation"), ".")
            threshold = self._coerce_number(leak_payload.get("threshold"), ".")
            if "remove" in action or "exclude" in action or "drop" in action:
                return True
            if corr is not None and threshold is not None and abs(corr) >= abs(threshold):
                return True
        return False

    def _flatten_numeric_metrics(self, metrics: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
        if not isinstance(metrics, dict):
            return {}
        flat: Dict[str, float] = {}
        for key, value in metrics.items():
            metric_key = f"{prefix}{key}" if prefix else str(key)
            if isinstance(value, dict):
                flat.update(self._flatten_numeric_metrics(value, f"{metric_key}."))
                continue
            if isinstance(value, list) and value:
                nums = []
                for v in value:
                    n = self._coerce_number(v, ".")
                    if n is not None:
                        nums.append(float(n))
                if nums and len(nums) == len(value):
                    flat[f"{metric_key}_mean"] = sum(nums) / len(nums)
                continue
            num = self._coerce_number(value, ".")
            if num is not None:
                flat[str(metric_key)] = float(num)
        return flat

    def _objective_metric_priority(self, objective_type: str) -> List[str]:
        family = normalize_problem_family(objective_type)
        if family == "classification":
            return ["roc_auc", "auc", "f1", "precision", "recall", "accuracy", "balanced_accuracy", "pr_auc"]
        if family == "regression":
            return ["rmse", "mae", "mse", "r2", "mape", "smape"]
        if family == "forecasting":
            return ["mape", "smape", "rmse", "mae", "coverage", "pinball"]
        if family == "survival_analysis":
            return ["concordance_index", "concordance", "integrated_brier_score", "ibs", "mae_uncensored"]
        if family == "ranking":
            return ["spearman", "kendall", "ndcg", "map", "mrr", "gini"]
        if family == "clustering":
            return ["silhouette", "davies_bouldin", "calinski_harabasz", "ari", "nmi"]
        if family == "optimization":
            return ["objective_value", "expected_value", "revenue", "cost", "constraint_violation_rate"]
        return ["roc_auc", "f1", "rmse", "mae", "r2", "spearman"]

    def _pick_column(self, columns: List[str], candidates: List[str]) -> Optional[str]:
        lower_map = {col.lower(): col for col in columns}
        for cand in candidates:
            if cand in lower_map:
                return lower_map[cand]
        for col in columns:
            col_lower = col.lower()
            for cand in candidates:
                if cand in col_lower:
                    return col
        return None

    def _build_segment_pricing_summary(self, path: str) -> List[Dict[str, Any]]:
        if not path:
            return []
        base_dialect = self._load_output_dialect()
        dialect = base_dialect or self._sniff_csv_dialect(path)
        for _ in range(2):
            sep = dialect.get("sep") or ","
            decimal = dialect.get("decimal") or "."
            encoding = dialect.get("encoding") or "utf-8"
            try:
                with open(path, "r", encoding=str(encoding), errors="replace") as f:
                    reader = csv.DictReader(f, delimiter=str(sep))
                    columns = reader.fieldnames or []
                    if not columns:
                        return []
                    if len(columns) == 1 and ";" in columns[0] and dialect.get("sep") != ";":
                        dialect = self._sniff_csv_dialect(path)
                        continue

                    segment_col = self._pick_column(columns, ["client_segment", "segment", "cluster", "group"])
                    price_col = self._pick_column(
                        columns,
                        [
                            "recommended_price",
                            "optimal_price",
                            "optimal_price_recommendation",
                            "price_recommendation",
                            "recommended_price_value",
                        ],
                    )
                    prob_col = self._pick_column(
                        columns,
                        [
                            "predicted_success_prob",
                            "predicted_probability",
                            "success_prob",
                            "probability",
                        ],
                    )
                    expected_rev_col = self._pick_column(
                        columns,
                        [
                            "expected_revenue",
                            "expected_value",
                            "expected_rev",
                            "expected_profit",
                        ],
                    )

                    if not segment_col:
                        return []

                    buckets: Dict[str, Dict[str, List[float] | int]] = {}
                    for row in reader:
                        seg_raw = row.get(segment_col)
                        if seg_raw is None or seg_raw == "":
                            continue
                        seg_key = str(seg_raw).strip()
                        if seg_key not in buckets:
                            buckets[seg_key] = {
                                "count": 0,
                                "prices": [],
                                "probs": [],
                                "revenues": [],
                            }
                        bucket = buckets[seg_key]
                        bucket["count"] = int(bucket["count"]) + 1
                        if price_col:
                            price_val = self._coerce_number(row.get(price_col), decimal)
                            if price_val is not None:
                                bucket["prices"].append(price_val)
                        if prob_col:
                            prob_val = self._coerce_number(row.get(prob_col), decimal)
                            if prob_val is not None:
                                bucket["probs"].append(prob_val)
                        if expected_rev_col:
                            rev_val = self._coerce_number(row.get(expected_rev_col), decimal)
                            if rev_val is not None:
                                bucket["revenues"].append(rev_val)

                    summary = []
                    for segment, bucket in buckets.items():
                        prices = bucket["prices"]
                        probs = bucket["probs"]
                        revenues = bucket["revenues"]
                        optimal_price = float(median(prices)) if prices else None
                        mean_prob = float(sum(probs) / len(probs)) if probs else None
                        mean_rev = float(sum(revenues) / len(revenues)) if revenues else None
                        summary.append(
                            {
                                "segment": segment,
                                "n": int(bucket["count"]),
                                "optimal_price": optimal_price,
                                "mean_prob": mean_prob,
                                "mean_expected_revenue": mean_rev,
                            }
                        )
                    return summary
            except Exception:
                return []
        return []

    def _extract_leakage_audit(self, alignment_payload: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(alignment_payload, dict):
            return None

        def _normalize(candidate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            if not isinstance(candidate, dict):
                return None
            feature = candidate.get("feature") or candidate.get("column") or candidate.get("field")
            target = candidate.get("target") or candidate.get("label") or candidate.get("outcome")
            corr = (
                candidate.get("correlation_coefficient")
                or candidate.get("correlation")
                or candidate.get("corr")
                or candidate.get("spearman")
                or candidate.get("pearson")
            )
            threshold = candidate.get("threshold") or candidate.get("corr_threshold")
            action = candidate.get("action_taken") or candidate.get("action") or candidate.get("action_if_exceeds")
            if corr is None and feature is None and target is None:
                return None
            return {
                "feature": feature,
                "target": target,
                "correlation": corr,
                "threshold": threshold,
                "action_taken": action,
            }

        for key in ("leakage_audit", "leakage", "leakage_check"):
            candidate = alignment_payload.get(key)
            if isinstance(candidate, dict):
                normalized = _normalize(candidate)
                if normalized:
                    return normalized

        requirements = alignment_payload.get("requirements")
        if isinstance(requirements, list):
            for req in requirements:
                if not isinstance(req, dict):
                    continue
                name = str(req.get("name") or req.get("id") or "")
                if "leak" not in name.lower():
                    continue
                evidence = req.get("evidence")
                if isinstance(evidence, dict):
                    normalized = _normalize(evidence)
                    if normalized:
                        return normalized
                normalized = _normalize(req)
                if normalized:
                    return normalized

        return None

    def _extract_validation_summary(self, alignment_payload: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(alignment_payload, dict) or not alignment_payload:
            return None
        status = alignment_payload.get("status") or alignment_payload.get("overall_status")
        failure_mode = alignment_payload.get("failure_mode")
        summary = alignment_payload.get("summary") or alignment_payload.get("notes")
        if status is None and failure_mode is None and summary is None:
            return None
        return {"status": status, "failure_mode": failure_mode, "summary": summary}

    def _extract_case_alignment_summary(self, payload: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(payload, dict) or not payload:
            return None
        status = payload.get("status")
        mode = payload.get("mode")
        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        summary = {"status": status, "mode": mode}
        if metrics:
            summary["metrics"] = metrics
        return summary if status or metrics else None

    def _normalize_artifact_index(self, entries: List[Any]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for item in entries or []:
            if isinstance(item, dict) and item.get("path"):
                normalized.append(item)
            elif isinstance(item, str):
                normalized.append({"path": item, "artifact_type": "artifact"})
        return normalized

    def _infer_artifact_type(self, path: str) -> str:
        lower = str(path).lower()
        if "metrics" in lower:
            return "metrics"
        if "alignment" in lower:
            return "report"
        if "error" in lower:
            return "error_analysis"
        if "importance" in lower:
            return "feature_importances"
        if "scored_rows" in lower or "predictions" in lower:
            return "predictions"
        return "artifact"

    def _find_artifacts_by_type(self, entries: List[Dict[str, Any]], artifact_type: str) -> List[str]:
        matches = []
        for item in entries or []:
            if not isinstance(item, dict):
                continue
            if item.get("artifact_type") == artifact_type and item.get("path"):
                matches.append(item["path"])
        return matches

    def _fallback(self, context: Dict[str, Any]) -> str:
        lines: List[str] = []
        output_report = context.get("output_contract_report", {}) or {}
        review_feedback = str(context.get("review_feedback") or "")
        if output_report.get("missing"):
            lines.append(
                "ISSUE: required outputs missing; WHY: outputs not saved; FIX: ensure all required artifacts are written."
            )
        if self._feedback_indicates_leakage_risk(review_feedback):
            lines.append(
                "ISSUE: leakage risk flagged; WHY: post-outcome fields may be in features; FIX: audit feature timing and remove leaks."
            )
        metrics = context.get("metrics") or {}
        if not metrics:
            lines.append(
                "ISSUE: metrics missing; WHY: evaluation artifacts absent; FIX: generate metrics aligned to the objective."
            )
        if not lines:
            lines.append(
                "ISSUE: limited insights; WHY: insufficient artifacts; FIX: produce metrics, predictions, and error analysis."
            )
        evidence_lines = ["evidence:"]
        for line in lines[:3]:
            evidence_lines.append(f"- claim: {line}; source: missing")
        return "\n".join(lines + evidence_lines)
