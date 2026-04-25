import os
import json
import re
import hashlib
import logging
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv
from openai import OpenAI
from src.utils.senior_protocol import SENIOR_STRATEGY_PROTOCOL
from src.utils.llm_fallback import call_chat_with_fallback, extract_response_text
from src.utils.actor_critic_schemas import (
    build_noop_iteration_hypothesis_packet,
    normalize_target_columns,
    validate_iteration_hypothesis_packet,
)
from src.utils.experiment_tracker import build_hypothesis_signature
from src.utils.optimization_policy_guard import (
    compact_optimization_policy_constraints,
    filter_optimization_actions,
    hypothesis_action_from_packet,
    optimization_policy_violations,
    resolve_optimization_policy,
)

load_dotenv()


class _TextResponse:
    def __init__(self, text: str):
        self.text = text


class _OpenRouterModelShim:
    """
    Backward-compatible generate_content interface.
    Tests and legacy callers can monkeypatch `agent.model.generate_content`.
    """

    def __init__(self, agent: "StrategistAgent"):
        self._agent = agent

    def generate_content(self, prompt: str) -> _TextResponse:
        call_kwargs: dict = {"temperature": self._agent._pending_temperature}
        max_tokens = self._agent._max_tokens
        if max_tokens and max_tokens > 0:
            call_kwargs["max_tokens"] = max_tokens
        response, model_used = call_chat_with_fallback(
            llm_client=self._agent.client,
            messages=[{"role": "user", "content": prompt}],
            model_chain=self._agent.model_chain,
            call_kwargs=call_kwargs,
            logger=self._agent.logger,
            context_tag=self._agent._pending_context_tag,
        )
        content = extract_response_text(response)
        if not content:
            raise ValueError("EMPTY_COMPLETION")
        self._agent.last_model_used = model_used
        # Capture finish_reason for truncation detection
        self._agent._last_finish_reason = None
        try:
            choices = getattr(response, "choices", None)
            if choices:
                self._agent._last_finish_reason = getattr(choices[0], "finish_reason", None)
        except Exception:
            pass
        return _TextResponse(str(content))


class StrategistAgent:
    def __init__(self, api_key: str = None):
        """
        Initializes the Strategist Agent with OpenRouter primary + fallback models.
        """
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API Key is required.")

        timeout_raw = os.getenv("OPENROUTER_TIMEOUT_SECONDS")
        try:
            timeout_seconds = float(timeout_raw) if timeout_raw else 120.0
        except ValueError:
            timeout_seconds = 120.0
        headers = {}
        referer = os.getenv("OPENROUTER_HTTP_REFERER")
        if referer:
            headers["HTTP-Referer"] = referer
        title = os.getenv("OPENROUTER_X_TITLE")
        if title:
            headers["X-Title"] = title
        client_kwargs = {
            "api_key": self.api_key,
            "base_url": "https://openrouter.ai/api/v1",
            "timeout": timeout_seconds,
        }
        if headers:
            client_kwargs["default_headers"] = headers
        self.client = OpenAI(**client_kwargs)

        self.model_name = (
            os.getenv("STRATEGIST_MODEL")
            or os.getenv("OPENROUTER_STRATEGIST_PRIMARY_MODEL")
            or "z-ai/glm-5"
        )
        self.fallback_model_name = (
            os.getenv("STRATEGIST_FALLBACK_MODEL")
            or os.getenv("OPENROUTER_STRATEGIST_FALLBACK_MODEL")
            or "z-ai/glm-4.7"
        )
        self.model_chain = [m for m in [self.model_name, self.fallback_model_name] if m]

        self.last_prompt = None
        self.last_response = None
        self.last_repair_prompt = None
        self.last_repair_response = None
        self.last_json_repair_meta = None
        self.last_model_used = None
        self._pending_temperature = 0.1
        self._pending_context_tag = "strategist_generate"
        self._last_finish_reason = None
        _max_tokens_raw = os.getenv("STRATEGIST_MAX_TOKENS", "32768")
        try:
            self._max_tokens = max(1024, int(_max_tokens_raw))
        except (ValueError, TypeError):
            self._max_tokens = 32768
        self.iteration_mode = self._normalize_iteration_mode(
            os.getenv("STRATEGIST_ITERATION_MODE", "llm")
        )
        self.last_iteration_hypothesis: Dict[str, Any] = {}
        self.last_iteration_meta: Dict[str, Any] = {
            "mode": self.iteration_mode,
            "source": "llm" if self.iteration_mode != "deterministic" else "deterministic",
            "model": None,
        }
        self.model = _OpenRouterModelShim(self)

    def _call_model(self, prompt: str, *, temperature: float, context_tag: str) -> str:
        self._pending_temperature = temperature
        self._pending_context_tag = context_tag
        response = self.model.generate_content(prompt)
        content = (getattr(response, "text", "") or "").strip()
        if not content:
            content = extract_response_text(response)
        if not content:
            raise ValueError("EMPTY_COMPLETION")
        if not self.last_model_used:
            self.last_model_used = self.model_name
        return content

    def _normalize_iteration_mode(self, raw: Any) -> str:
        value = str(raw or "").strip().lower()
        if value in {"llm", "hybrid", "deterministic"}:
            return value
        return "llm"

    def generate_iteration_hypothesis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        mode = self.iteration_mode
        self.last_iteration_meta = {
            "mode": mode,
            "source": "llm" if mode != "deterministic" else "deterministic",
            "model": self.last_model_used or self.model_name if mode != "deterministic" else None,
        }
        if mode == "deterministic":
            deterministic_packet = self._generate_iteration_hypothesis_deterministic(context)
            self.last_iteration_hypothesis = deterministic_packet
            return deterministic_packet

        llm_packet = self._generate_iteration_hypothesis_llm(context)
        finalized_llm_packet, errors = self._finalize_reasoned_iteration_hypothesis(llm_packet, context)
        if finalized_llm_packet:
            self.last_iteration_meta = {
                "mode": mode,
                "source": "llm",
                "model": self.last_model_used or self.model_name,
            }
            self.last_iteration_hypothesis = finalized_llm_packet
            return finalized_llm_packet

        if mode == "hybrid":
            deterministic_packet = self._generate_iteration_hypothesis_deterministic(context)
            self.last_iteration_meta = {
                "mode": mode,
                "source": "deterministic_fallback",
                "model": self.last_model_used or self.model_name,
                "validation_errors": errors[:6],
            }
            self.last_iteration_hypothesis = deterministic_packet
            return deterministic_packet

        self.last_iteration_meta = {
            "mode": mode,
            "source": "llm_noop_fallback",
            "model": self.last_model_used or self.model_name,
            "validation_errors": errors[:6],
        }
        llm_noop_packet = self._build_iteration_llm_noop_packet(context, errors)
        self.last_iteration_hypothesis = llm_noop_packet
        return llm_noop_packet

    def _parse_json_object(self, raw_text: Any) -> Dict[str, Any]:
        text = str(raw_text or "").strip()
        if not text:
            return {}
        cleaned = self._clean_json(text)
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

    def _collect_tracker_signatures(self, tracker_entries: Any) -> set[str]:
        """Collect hypothesis signatures from tracker entries.

        Entries where the technique suffered a runtime failure (timeout, crash,
        sandbox error) are excluded so that the technique is not considered
        "already tried" and can be retried with different parameters.
        """
        signatures: set[str] = set()
        if not isinstance(tracker_entries, list):
            return signatures
        for entry in tracker_entries:
            if not isinstance(entry, dict):
                continue
            # Skip runtime-failed entries — the technique was never truly
            # evaluated so it should remain eligible for retry.
            if entry.get("runtime_failed"):
                continue
            extra = entry.get("extra") if isinstance(entry.get("extra"), dict) else {}
            if extra.get("runtime_failed"):
                continue
            direct = str(entry.get("signature") or entry.get("hypothesis_signature") or "").strip()
            if direct:
                signatures.add(direct)
            tracker_ctx = entry.get("tracker_context")
            if isinstance(tracker_ctx, dict):
                nested = str(tracker_ctx.get("signature") or "").strip()
                if nested:
                    signatures.add(nested)
        return signatures

    def _summarize_iteration_round_history(self, round_history: Any) -> List[Dict[str, Any]]:
        if not isinstance(round_history, list):
            return []
        summarized: List[Dict[str, Any]] = []
        for entry in round_history[-6:]:
            if not isinstance(entry, dict):
                continue
            summarized.append(
                {
                    "round_id": entry.get("round_id"),
                    "hypothesis": str(entry.get("hypothesis") or entry.get("technique") or "").strip() or None,
                    "metric_improved": entry.get("metric_improved"),
                    "governance_approved": entry.get("governance_approved"),
                    "kept": entry.get("kept"),
                    "baseline_metric": entry.get("baseline_metric"),
                    "candidate_metric": entry.get("candidate_metric"),
                    "reason": str(entry.get("reason") or "").strip() or None,
                }
            )
        return summarized

    def _resolve_iteration_packet_basics(
        self,
        context: Dict[str, Any],
    ) -> Tuple[str, int, str, float, set[str]]:
        context = context if isinstance(context, dict) else {}
        run_id = str(context.get("run_id") or "unknown_run").strip() or "unknown_run"
        try:
            iteration = int(context.get("iteration") or 1)
        except Exception:
            iteration = 1
        if iteration <= 0:
            iteration = 1
        primary_metric_name = str(context.get("primary_metric_name") or "primary_metric").strip() or "primary_metric"
        try:
            min_delta = float(context.get("min_delta", 0.0005) or 0.0005)
        except Exception:
            min_delta = 0.0005
        if min_delta < 0:
            min_delta = 0.0
        tracker_entries = context.get("experiment_tracker")
        if not isinstance(tracker_entries, list):
            tracker_entries = []
        known_signatures = self._collect_tracker_signatures(tracker_entries)
        return run_id, iteration, primary_metric_name, min_delta, known_signatures

    def _build_iteration_llm_noop_packet(
        self,
        context: Dict[str, Any],
        errors: List[str] | None = None,
    ) -> Dict[str, Any]:
        run_id, iteration, primary_metric_name, min_delta, _ = self._resolve_iteration_packet_basics(context)
        explanation = "Strategist reasoning did not yield a valid actionable hypothesis."
        if errors:
            explanation += " Validation: " + "; ".join(str(item) for item in errors[:2] if str(item).strip())
        signature = f"llm_noop_round_{iteration}"
        return build_noop_iteration_hypothesis_packet(
            run_id=run_id,
            iteration=iteration,
            signature=signature,
            duplicate_of=None,
            primary_metric_name=primary_metric_name,
            min_delta=min_delta,
            explanation=explanation[:280],
        )

    def _finalize_reasoned_iteration_hypothesis(
        self,
        packet: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[str]]:
        if not isinstance(packet, dict) or not packet:
            return {}, ["llm_packet_missing"]

        run_id, iteration, primary_metric_name, min_delta, known_signatures = (
            self._resolve_iteration_packet_basics(context)
        )
        hypothesis = packet.get("hypothesis") if isinstance(packet.get("hypothesis"), dict) else {}
        fallback_signature = build_hypothesis_signature(
            technique=hypothesis.get("technique"),
            target_columns=normalize_target_columns(hypothesis.get("target_columns")),
            feature_scope=hypothesis.get("feature_scope"),
            params=hypothesis.get("params"),
        )
        if not fallback_signature:
            fallback_signature = f"reasoned_round_{iteration}"

        sanitized = self._sanitize_iteration_hypothesis_packet(
            packet,
            primary_metric_name=primary_metric_name,
            min_delta=min_delta,
            fallback_signature=fallback_signature,
            prefer_noop_on_missing_technique=True,
        )
        tracker_context = (
            sanitized.get("tracker_context")
            if isinstance(sanitized.get("tracker_context"), dict)
            else {}
        )
        signature = str(tracker_context.get("signature") or fallback_signature).strip() or fallback_signature
        if signature in known_signatures:
            return (
                build_noop_iteration_hypothesis_packet(
                    run_id=run_id,
                    iteration=iteration,
                    signature=signature,
                    duplicate_of=signature,
                    primary_metric_name=primary_metric_name,
                    min_delta=min_delta,
                    explanation=(
                        "Strategist-selected hypothesis duplicates prior tracker evidence; emit NO_OP instead of replaying it."
                    )[:280],
                ),
                [],
            )

        policy = resolve_optimization_policy(context)
        policy_violations = optimization_policy_violations(
            hypothesis_action_from_packet(sanitized),
            policy,
        )
        if policy_violations:
            return (
                build_noop_iteration_hypothesis_packet(
                    run_id=run_id,
                    iteration=iteration,
                    signature=signature,
                    duplicate_of=signature,
                    primary_metric_name=primary_metric_name,
                    min_delta=min_delta,
                    explanation=(
                        "Selected hypothesis violates contract optimization_policy: "
                        + ", ".join(policy_violations[:4])
                        + ". Emit NO_OP rather than asking ML engineer to contradict the contract."
                    )[:280],
                ),
                [],
            )

        valid_packet, errors = validate_iteration_hypothesis_packet(sanitized)
        if valid_packet:
            return sanitized, []
        return {}, errors

    def _dedupe_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        deduped: List[Dict[str, Any]] = []
        seen_signatures: set[str] = set()
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            signature = build_hypothesis_signature(
                technique=candidate.get("technique"),
                target_columns=candidate.get("target_columns"),
                feature_scope=candidate.get("feature_scope"),
                params=candidate.get("params"),
            )
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            deduped.append(candidate)
        return deduped

    def _dataset_has_missingness_signal(self, dataset_profile: Dict[str, Any]) -> bool:
        if not isinstance(dataset_profile, dict):
            return False
        for key in ("missingness", "missingness_top30"):
            block = dataset_profile.get(key)
            if not isinstance(block, dict):
                continue
            for value in block.values():
                try:
                    if float(value) > 0.0:
                        return True
                except Exception:
                    continue
        return False

    def _dataset_has_categorical_signal(self, dataset_profile: Dict[str, Any]) -> bool:
        """Check if dataset has any categorical columns (universal)."""
        if not isinstance(dataset_profile, dict):
            return False
        col_types = dataset_profile.get("column_types")
        if isinstance(col_types, dict):
            for type_key in ("categorical", "low_cardinality", "object", "string"):
                cols = col_types.get(type_key)
                if isinstance(cols, list) and len(cols) > 0:
                    return True
        dtypes = dataset_profile.get("dtypes") or dataset_profile.get("dtype_distribution")
        if isinstance(dtypes, dict):
            for dtype_name, count in dtypes.items():
                if "object" in str(dtype_name).lower() or "categ" in str(dtype_name).lower():
                    try:
                        if int(count) > 0:
                            return True
                    except Exception:
                        pass
        cardinality = dataset_profile.get("cardinality")
        if isinstance(cardinality, dict) and len(cardinality) > 0:
            return True
        return False

    def _dataset_has_numeric_signal(self, dataset_profile: Dict[str, Any]) -> bool:
        """Check if dataset has numeric columns (universal)."""
        if not isinstance(dataset_profile, dict):
            return True  # assume numeric exists unless proven otherwise
        col_types = dataset_profile.get("column_types")
        if isinstance(col_types, dict):
            for type_key in ("numeric", "float", "int", "continuous"):
                cols = col_types.get(type_key)
                if isinstance(cols, list) and len(cols) > 0:
                    return True
        dtypes = dataset_profile.get("dtypes") or dataset_profile.get("dtype_distribution")
        if isinstance(dtypes, dict):
            for dtype_name, count in dtypes.items():
                if any(t in str(dtype_name).lower() for t in ("float", "int", "numeric")):
                    try:
                        if int(count) > 0:
                            return True
                    except Exception:
                        pass
        return True  # default True: assume numeric exists

    def _filter_candidates_by_feasibility(
        self,
        candidates: List[Dict[str, Any]],
        dataset_profile: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Filter out candidates whose technique is infeasible given the dataset
        profile. Universal: does not hardcode dataset-specific logic.
        """
        if not isinstance(dataset_profile, dict) or not dataset_profile:
            return candidates

        has_missing = self._dataset_has_missingness_signal(dataset_profile)
        has_categorical = self._dataset_has_categorical_signal(dataset_profile)
        has_high_card = self._dataset_has_high_cardinality_signal(dataset_profile)
        has_numeric = self._dataset_has_numeric_signal(dataset_profile)

        feasibility_map = {
            "missing_indicators": has_missing,
            "rare_category_grouping": has_high_card,
            "frequency_encoding": has_categorical,
            "quantile_binning": has_numeric,
        }

        filtered: List[Dict[str, Any]] = []
        for candidate in candidates:
            technique = str(candidate.get("technique") or "").strip().lower()
            is_feasible = feasibility_map.get(technique)
            if is_feasible is False:
                print(
                    f"STRATEGIST_FEASIBILITY_FILTER: "
                    f"technique={technique} filtered_out=True "
                    f"reason=infeasible_for_dataset"
                )
                continue
            filtered.append(candidate)

        return filtered

    def _dataset_has_high_cardinality_signal(self, dataset_profile: Dict[str, Any]) -> bool:
        if not isinstance(dataset_profile, dict):
            return False
        high_card_cols = dataset_profile.get("high_cardinality_columns")
        if isinstance(high_card_cols, list) and any(str(item or "").strip() for item in high_card_cols):
            return True
        cardinality = dataset_profile.get("cardinality")
        if not isinstance(cardinality, dict):
            return False
        row_count = 0
        try:
            row_count = int(dataset_profile.get("basic_stats", {}).get("n_rows") or dataset_profile.get("n_rows") or 0)
        except Exception:
            row_count = 0
        dynamic_threshold = max(50, int(row_count * 0.01)) if row_count > 0 else 50
        for payload in cardinality.values():
            if not isinstance(payload, dict):
                continue
            try:
                unique_count = int(payload.get("unique") or 0)
            except Exception:
                unique_count = 0
            if unique_count >= dynamic_threshold and unique_count > 20:
                return True
        return False

    def _extract_dataset_n_rows(self, dataset_profile: Dict[str, Any]) -> int:
        if not isinstance(dataset_profile, dict):
            return 0
        basic_stats = dataset_profile.get("basic_stats") if isinstance(dataset_profile.get("basic_stats"), dict) else {}
        candidates = [
            basic_stats.get("n_rows"),
            dataset_profile.get("n_rows"),
            dataset_profile.get("row_count"),
        ]
        for value in candidates:
            try:
                rows = int(value or 0)
            except Exception:
                rows = 0
            if rows > 0:
                return rows
        return 0

    def _normalize_technique_name(self, technique: Any) -> str:
        return re.sub(r"[^a-z0-9]+", "_", str(technique or "").strip().lower()).strip("_")

    def _candidate_family_key(self, technique: Any) -> str:
        token = self._normalize_technique_name(technique)
        if not token:
            return "generic"
        ordered_tags = [
            "target_encoding",
            "frequency_encoding",
            "rare_category_grouping",
            "missingness",
            "regularization",
            "quantile_binning",
            "interaction_features",
            "ensemble",
            "variance_reduction",
            "hpo",
            "categorical_features",
        ]
        tags = self._technique_tags(technique)
        for tag in ordered_tags:
            if tag in tags:
                return tag
        return token

    def _technique_tags(self, technique: Any) -> set[str]:
        token = self._normalize_technique_name(technique)
        tags: set[str] = set()
        if not token:
            return tags
        if any(part in token for part in ("missing", "imput")):
            tags.add("missingness")
        if any(part in token for part in ("categor", "rare", "frequency", "target_encoding", "ordinal", "one_hot")):
            tags.add("categorical_features")
        if "rare" in token or "group" in token:
            tags.add("rare_category_grouping")
        if "frequency" in token or "count_encoding" in token:
            tags.add("frequency_encoding")
        if "target_encoding" in token or "kfold_target_encoding" in token:
            tags.add("target_encoding")
        if any(part in token for part in ("regularization", "lr_reduction", "shrinkage", "dropout")):
            tags.add("regularization")
        if any(part in token for part in ("quantile", "binning", "bucket")):
            tags.add("quantile_binning")
        if any(part in token for part in ("interaction", "cross", "ratio", "polynomial")):
            tags.add("interaction_features")
        if any(part in token for part in ("stack", "ensemble", "blend")):
            tags.add("ensemble")
        if any(part in token for part in ("multi_seed", "seed_averaging", "bagging")):
            tags.add("variance_reduction")
        if any(part in token for part in ("hyperparameter", "optuna", "hpo", "tuning")):
            tags.add("hpo")
        return tags

    def _extract_tracker_technique(self, entry: Dict[str, Any]) -> str:
        if not isinstance(entry, dict):
            return ""
        direct = str(entry.get("technique") or "").strip()
        if direct:
            return direct
        hypothesis = entry.get("hypothesis")
        if isinstance(hypothesis, dict):
            nested = str(hypothesis.get("technique") or hypothesis.get("name") or "").strip()
            if nested:
                return nested
        extra = entry.get("extra")
        if isinstance(extra, dict):
            nested = str(extra.get("technique") or "").strip()
            if nested:
                return nested
        return ""

    def _collect_tracker_technique_stats(self, tracker_entries: List[Dict[str, Any]]) -> tuple[Dict[str, Dict[str, Any]], List[str]]:
        stats: Dict[str, Dict[str, Any]] = {}
        recent_families: List[str] = []
        recent_family_seen: set[str] = set()
        recent_entries = tracker_entries[-8:] if isinstance(tracker_entries, list) else []
        for entry in recent_entries:
            technique = self._extract_tracker_technique(entry)
            family = self._candidate_family_key(technique)
            if family and family not in recent_family_seen:
                recent_families.append(family)
                recent_family_seen.add(family)
        for entry in tracker_entries if isinstance(tracker_entries, list) else []:
            technique = self._extract_tracker_technique(entry)
            norm = self._normalize_technique_name(technique)
            if not norm:
                continue
            bucket = stats.setdefault(
                norm,
                {
                    "attempts": 0,
                    "runtime_failed": 0,
                    "non_improving": 0,
                    "negative_deltas": 0,
                    "positive_deltas": 0,
                    "last_delta": None,
                    "last_approved": None,
                },
            )
            bucket["attempts"] += 1
            runtime_failed = bool(entry.get("runtime_failed"))
            extra = entry.get("extra") if isinstance(entry.get("extra"), dict) else {}
            if extra.get("runtime_failed"):
                runtime_failed = True
            if runtime_failed:
                bucket["runtime_failed"] += 1
            delta = entry.get("delta")
            try:
                delta_value = float(delta)
            except Exception:
                delta_value = None
            if delta_value is not None:
                bucket["last_delta"] = float(delta_value)
                if delta_value > 0:
                    bucket["positive_deltas"] += 1
                elif delta_value < 0:
                    bucket["negative_deltas"] += 1
            improved = entry.get("improved_by_metric")
            approved = entry.get("approved")
            if improved is False or approved is False:
                bucket["non_improving"] += 1
            if isinstance(approved, bool):
                bucket["last_approved"] = approved
        return stats, recent_families

    def _candidate_cost_penalty(self, technique: Any, dataset_profile: Dict[str, Any]) -> float:
        rows = self._extract_dataset_n_rows(dataset_profile)
        if rows <= 0:
            return 0.0
        tags = self._technique_tags(technique)
        penalty = 0.0
        if rows >= 250000:
            if "ensemble" in tags:
                penalty += 3.0
            if "hpo" in tags:
                penalty += 2.5
            if "variance_reduction" in tags:
                penalty += 2.0
            if "target_encoding" in tags:
                penalty += 1.5
        elif rows >= 100000:
            if "ensemble" in tags:
                penalty += 1.5
            if "hpo" in tags:
                penalty += 1.0
            if "variance_reduction" in tags:
                penalty += 0.75
        return penalty

    def _score_iteration_candidate(
        self,
        candidate: Dict[str, Any],
        *,
        critique_packet: Dict[str, Any],
        dataset_profile: Dict[str, Any],
        tracker_stats: Dict[str, Dict[str, Any]],
        recent_families: List[str],
    ) -> tuple[float, List[str]]:
        technique = str(candidate.get("technique") or "").strip()
        technique_norm = self._normalize_technique_name(technique)
        tags = self._technique_tags(technique)
        family = self._candidate_family_key(technique)
        score = 0.0
        reasons: List[str] = []

        if str(candidate.get("_source") or "").strip() == "model_analyst_blueprint":
            score += 1.5
            reasons.append("blueprint_prior")
        priority_raw = candidate.get("_priority")
        try:
            priority = int(priority_raw)
        except Exception:
            priority = 5
        score += max(0, 6 - max(1, min(priority, 10))) * 0.35

        error_modes = critique_packet.get("error_modes") if isinstance(critique_packet.get("error_modes"), list) else []
        risk_flags = {
            str(item or "").strip().lower()
            for item in (critique_packet.get("risk_flags") or [])
            if str(item or "").strip()
        }
        mode_tag_map = {
            "fold_instability": {"missingness", "regularization", "variance_reduction", "quantile_binning"},
            "minority_class_recall_low": {"rare_category_grouping", "frequency_encoding", "target_encoding", "ensemble"},
            "generalization_gap_high": {"regularization", "quantile_binning", "variance_reduction"},
            "delta_below_threshold": {"target_encoding", "interaction_features", "ensemble", "hpo"},
            "metric_stagnation": {"target_encoding", "interaction_features", "ensemble", "hpo"},
        }
        severity_weight = {"high": 3.0, "medium": 2.0, "low": 1.0}
        for item in error_modes:
            if not isinstance(item, dict):
                continue
            mode_id = str(item.get("id") or "").strip().lower()
            if not mode_id:
                continue
            matching_tags = mode_tag_map.get(mode_id, set())
            if tags & matching_tags:
                weight = severity_weight.get(str(item.get("severity") or "").strip().lower(), 1.0)
                score += 1.5 * weight
                reasons.append(f"targets_{mode_id}")

        if "class_imbalance_sensitivity" in risk_flags and tags & {
            "rare_category_grouping",
            "frequency_encoding",
            "target_encoding",
        }:
            score += 1.75
            reasons.append("risk_class_imbalance")
        if {"potential_overfitting", "generalization_instability"} & risk_flags and tags & {
            "regularization",
            "quantile_binning",
            "variance_reduction",
        }:
            score += 1.75
            reasons.append("risk_generalization")
        if "no_material_metric_gain" in risk_flags and tags & {"target_encoding", "interaction_features", "ensemble", "hpo"}:
            score += 1.25
            reasons.append("risk_stagnation")

        if self._dataset_has_missingness_signal(dataset_profile) and "missingness" in tags:
            score += 1.75
            reasons.append("data_missingness")
        if self._dataset_has_high_cardinality_signal(dataset_profile) and tags & {
            "rare_category_grouping",
            "frequency_encoding",
            "target_encoding",
        }:
            score += 2.0
            reasons.append("data_high_cardinality")
        if self._dataset_has_categorical_signal(dataset_profile) and "categorical_features" in tags:
            score += 1.0
            reasons.append("data_categorical")
        if self._dataset_has_numeric_signal(dataset_profile) and tags & {"quantile_binning", "interaction_features"}:
            score += 0.75
            reasons.append("data_numeric")

        history = tracker_stats.get(technique_norm, {})
        if history:
            positive_deltas = int(history.get("positive_deltas", 0) or 0)
            non_improving = int(history.get("non_improving", 0) or 0)
            negative_deltas = int(history.get("negative_deltas", 0) or 0)
            runtime_failed = int(history.get("runtime_failed", 0) or 0)
            if positive_deltas > 0:
                score += min(2.0, 0.75 * positive_deltas)
                reasons.append("historical_positive_delta")
            if non_improving > 0:
                score -= min(6.0, 2.0 * non_improving)
                reasons.append("historical_non_improving")
            if negative_deltas > 0:
                score -= min(4.0, 1.25 * negative_deltas)
                reasons.append("historical_negative_delta")
            if runtime_failed > 0 and non_improving == 0:
                score -= 0.5
                reasons.append("runtime_retry_risk")

        if family and family not in recent_families:
            score += 0.75
            reasons.append("family_diversity")
        elif family and family in recent_families:
            score -= 0.25
            reasons.append("recent_family_repeat")

        cost_penalty = self._candidate_cost_penalty(technique, dataset_profile)
        if cost_penalty > 0:
            score -= cost_penalty
            reasons.append("cost_scaled_to_dataset")

        objective = str(candidate.get("objective") or "").strip()
        if objective:
            score += 0.15

        return score, list(dict.fromkeys(reasons))

    def _rank_iteration_candidates(
        self,
        candidates: List[Dict[str, Any]],
        *,
        critique_packet: Dict[str, Any],
        dataset_profile: Dict[str, Any],
        tracker_entries: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        tracker_stats, recent_families = self._collect_tracker_technique_stats(tracker_entries)
        ranked: List[Dict[str, Any]] = []
        for idx, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                continue
            signature = build_hypothesis_signature(
                technique=candidate.get("technique"),
                target_columns=candidate.get("target_columns"),
                feature_scope=candidate.get("feature_scope"),
                params=candidate.get("params"),
            )
            score, reasons = self._score_iteration_candidate(
                candidate,
                critique_packet=critique_packet,
                dataset_profile=dataset_profile,
                tracker_stats=tracker_stats,
                recent_families=recent_families,
            )
            try:
                ranking_priority = int(candidate.get("_priority", 5) or 5)
            except Exception:
                ranking_priority = 5
            enriched = dict(candidate)
            enriched["_signature"] = signature
            enriched["_ranking_score"] = round(float(score), 3)
            enriched["_ranking_reasons"] = reasons[:6]
            enriched["_ranking_index"] = idx
            enriched["_ranking_priority"] = ranking_priority
            ranked.append(enriched)

        ranked.sort(
            key=lambda item: (
                float(item.get("_ranking_score", 0.0)),
                -int(item.get("_ranking_priority", 5) or 5),
                1 if str(item.get("_source") or "").strip() == "model_analyst_blueprint" else 0,
                -int(item.get("_ranking_index", 0) or 0),
            ),
            reverse=True,
        )
        return ranked

    def _candidate_techniques_from_plan(
        self,
        feature_engineering_plan: Dict[str, Any],
        dataset_profile: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        techniques_raw = feature_engineering_plan.get("techniques")
        if not isinstance(techniques_raw, list):
            techniques_raw = []
        candidates: List[Dict[str, Any]] = []
        for item in techniques_raw:
            if isinstance(item, dict):
                technique = str(item.get("technique") or item.get("name") or "").strip()
                if not technique:
                    continue
                candidates.append(
                    {
                        "technique": technique,
                        "target_columns": normalize_target_columns(item.get("columns")),
                        "feature_scope": str(item.get("feature_scope") or "model_features"),
                        "params": item.get("params") if isinstance(item.get("params"), dict) else {},
                        "objective": str(item.get("rationale") or item.get("notes") or "").strip(),
                    }
                )
                continue
            technique = str(item or "").strip()
            if not technique:
                continue
            candidates.append(
                {
                    "technique": technique,
                    "target_columns": ["ALL_NUMERIC"],
                    "feature_scope": "model_features",
                    "params": {},
                    "objective": "",
                }
            )
        candidates = self._dedupe_candidates(candidates)
        if isinstance(dataset_profile, dict) and dataset_profile:
            candidates = self._filter_candidates_by_feasibility(candidates, dataset_profile)
        return candidates

    def _fallback_candidates_from_critique(
        self,
        critique_packet: Dict[str, Any],
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        error_modes = critique_packet.get("error_modes") if isinstance(critique_packet.get("error_modes"), list) else []
        ids = {str(item.get("id") or "").strip().lower() for item in error_modes if isinstance(item, dict)}
        dataset_profile = {}
        if isinstance(context, dict) and isinstance(context.get("dataset_profile"), dict):
            dataset_profile = context.get("dataset_profile") or {}
        has_missing_signal = self._dataset_has_missingness_signal(dataset_profile)
        has_high_card_signal = self._dataset_has_high_cardinality_signal(dataset_profile)

        out: List[Dict[str, Any]] = []
        if "fold_instability" in ids and has_missing_signal:
            out.append(
                {
                    "technique": "missing_indicators",
                    "target_columns": ["ALL_NUMERIC"],
                    "feature_scope": "model_features",
                    "params": {"indicator_suffix": "_is_missing"},
                    "objective": "Inject missingness indicators to stabilize fold behavior with low-risk edits.",
                }
            )
        elif "fold_instability" in ids:
            out.append(
                {
                    "technique": "regularization_tuning",
                    "target_columns": ["ALL"],
                    "feature_scope": "model_features",
                    "params": {},
                    "objective": "Stabilize fold behavior via regularization tuning.",
                }
            )
        if has_missing_signal and "fold_instability" not in ids:
            out.append(
                {
                    "technique": "missing_indicators",
                    "target_columns": ["ALL_NUMERIC"],
                    "feature_scope": "model_features",
                    "params": {"indicator_suffix": "_is_missing"},
                    "objective": "Inject missingness indicators for features with missing values.",
                }
            )
        if "minority_class_recall_low" in ids or has_high_card_signal:
            out.append(
                {
                    "technique": "rare_category_grouping",
                    "target_columns": ["ALL_CATEGORICAL"],
                    "feature_scope": "model_features",
                    "params": {"min_frequency": 0.01},
                    "objective": "Reduce noise in long-tail categories affecting minority-class recall.",
                }
            )
        if has_high_card_signal:
            out.append(
                {
                    "technique": "frequency_encoding",
                    "target_columns": ["ALL_CATEGORICAL"],
                    "feature_scope": "model_features",
                    "params": {"normalize": True},
                    "objective": "Add frequency signals for high-cardinality categories with compact features.",
                }
            )
        if "generalization_gap_high" in ids:
            out.append(
                {
                    "technique": "quantile_binning",
                    "target_columns": ["ALL_NUMERIC"],
                    "feature_scope": "model_features",
                    "params": {"q": 20, "drop_duplicates": True},
                    "objective": "Apply bounded quantile bins to reduce over-sensitive continuous splits.",
                }
            )
        # Apply feasibility filter against dataset profile
        out = self._filter_candidates_by_feasibility(out, dataset_profile)
        # If error-mode matching produced no candidates, add generic improvement candidates
        if not out:
            out.append(
                {
                    "technique": "target_encoding",
                    "target_columns": ["ALL_CATEGORICAL"],
                    "feature_scope": "model_features",
                    "params": {"cv": 5, "smoothing": 10.0},
                    "objective": "Add K-fold target encoding for categoricals to extract more signal than ordinal encoding.",
                }
            )
            out.append(
                {
                    "technique": "multi_seed_averaging",
                    "target_columns": ["ALL"],
                    "feature_scope": "model_features",
                    "params": {"seeds": [42, 123, 456, 789, 2024], "aggregation": "mean"},
                    "objective": "Train with multiple random seeds and average predictions for variance reduction.",
                }
            )
            out = self._filter_candidates_by_feasibility(out, dataset_profile)
        if not out:
            out.append(
                {
                    "technique": "hyperparameter_tuning",
                    "target_columns": ["ALL"],
                    "feature_scope": "model_features",
                    "params": {},
                    "objective": "Safe universal fallback: tune model hyperparameters.",
                }
            )
        return self._dedupe_candidates(out)

    def _generate_iteration_hypothesis_llm(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(context, dict):
            return {}
        critique_packet = context.get("critique_packet")
        if not isinstance(critique_packet, dict):
            critique_packet = {}
        feature_engineering_plan = context.get("feature_engineering_plan")
        if not isinstance(feature_engineering_plan, dict):
            feature_engineering_plan = {}
        tracker_entries = context.get("experiment_tracker")
        if not isinstance(tracker_entries, list):
            tracker_entries = []
        round_history = self._summarize_iteration_round_history(context.get("round_history"))

        optimization_blueprint = context.get("optimization_blueprint")
        if not isinstance(optimization_blueprint, dict):
            optimization_blueprint = {}
        blueprint_actions = optimization_blueprint.get("improvement_actions")
        if not isinstance(blueprint_actions, list):
            blueprint_actions = []
        optimization_policy = resolve_optimization_policy(context)
        blueprint_actions, blocked_blueprint_actions = filter_optimization_actions(
            blueprint_actions,
            optimization_policy,
        )

        llm_context = {
            "run_id": context.get("run_id"),
            "iteration": context.get("iteration"),
            "primary_metric_name": context.get("primary_metric_name"),
            "min_delta": context.get("min_delta"),
            "optimization_policy": compact_optimization_policy_constraints(optimization_policy),
            "policy_blocked_blueprint_actions": blocked_blueprint_actions[:6],
            "critique_packet": critique_packet,
            "feature_engineering_plan": feature_engineering_plan,
            "experiment_tracker": tracker_entries[-10:],
            "round_history": round_history,
            "dataset_profile_signals": {
                "has_missing_signal": self._dataset_has_missingness_signal(
                    context.get("dataset_profile") if isinstance(context.get("dataset_profile"), dict) else {}
                ),
                "has_high_cardinality_signal": self._dataset_has_high_cardinality_signal(
                    context.get("dataset_profile") if isinstance(context.get("dataset_profile"), dict) else {}
                ),
                "n_rows": self._extract_dataset_n_rows(
                    context.get("dataset_profile") if isinstance(context.get("dataset_profile"), dict) else {}
                ),
            },
        }
        if blueprint_actions:
            llm_context["optimization_blueprint_actions"] = blueprint_actions[:6]

        blueprint_instruction = ""
        if blueprint_actions:
            blueprint_instruction = (
                "An optimization blueprint is provided in 'optimization_blueprint_actions'. "
                "Use it as a source of evidence and candidate ideas, not as a fixed queue and not as an override. "
                "Prefer a blueprint action only when it still aligns with critique_packet, "
                "dataset_profile_signals, round_history, and recent experiment_tracker outcomes. "
                "Down-rank techniques that recently regressed, duplicate prior work, or look too expensive "
                "for the likely ROI. When you choose a blueprint action, reuse its concrete_params "
                "and code_change_hint when they still make sense.\n"
            )

        llm_prompt = (
            "You are selecting the next metric-improvement experiment for a senior ML workflow.\n"
            "Return ONLY JSON for one iteration_hypothesis_packet. No markdown. No extra text.\n\n"
            "Reasoning workflow:\n"
            "1. Diagnose what recent evidence suggests about the current incumbent: variance issue, underfitting, feature signal gap, "
            "search-space issue, calibration problem, or no credible next move.\n"
            "2. Use critique_packet, round_history, experiment_tracker, and dataset_profile_signals to decide whether to exploit the incumbent's strongest signal, pivot away from repeated regressions, or stop.\n"
            "3. Choose the single highest-ROI next hypothesis for THIS context, balancing expected lift, compute cost, and recent regressions.\n"
            "4. Prefer the cheapest valid experiment that meaningfully tests the idea.\n"
            "5. Do not repeat a duplicate hypothesis signature. If every credible option is duplicate or low-value, emit NO_OP.\n\n"
            "Rules:\n"
            "- Exactly one hypothesis; action APPLY or NO_OP.\n"
            "- If duplicate signature then action must be NO_OP.\n"
            "- optimization_policy is a hard contract. Do not choose hypotheses listed in policy_blocked_blueprint_actions, "
            "and do not emit any hypothesis that violates an allow_* flag.\n"
            "- Allowed target macros: ALL_NUMERIC, ALL_CATEGORICAL, ALL_TEXT, ALL_DATETIME, ALL_BOOLEAN.\n"
            "- You own the selection. feature_engineering_plan and optimization_blueprint_actions are advisory inputs, not a mandatory queue.\n"
            "- Use critique_packet, round_history, experiment_tracker, dataset_profile_signals, and feature_engineering_plan as primary evidence.\n"
            + blueprint_instruction
            + "\nContext:\n"
            + json.dumps(llm_context, ensure_ascii=False)
        )
        try:
            content = self._call_model(
                llm_prompt,
                temperature=0.0,
                context_tag="strategist_iteration_hypothesis",
            )
        except Exception:
            return {}
        return self._parse_json_object(content)

    def _generate_iteration_hypothesis_deterministic(self, context: Dict[str, Any]) -> Dict[str, Any]:
        context = context if isinstance(context, dict) else {}
        run_id = str(context.get("run_id") or "unknown_run")
        try:
            iteration = int(context.get("iteration") or 1)
        except Exception:
            iteration = 1
        if iteration <= 0:
            iteration = 1
        primary_metric_name = str(context.get("primary_metric_name") or "primary_metric").strip() or "primary_metric"
        try:
            min_delta = float(context.get("min_delta", 0.0005) or 0.0005)
        except Exception:
            min_delta = 0.0005
        if min_delta < 0:
            min_delta = 0.0

        critique_packet = context.get("critique_packet")
        if not isinstance(critique_packet, dict):
            critique_packet = {}
        feature_engineering_plan = context.get("feature_engineering_plan")
        if not isinstance(feature_engineering_plan, dict):
            feature_engineering_plan = {}
        tracker_entries = context.get("experiment_tracker")
        if not isinstance(tracker_entries, list):
            tracker_entries = []
        known_signatures = self._collect_tracker_signatures(tracker_entries)

        dataset_profile = context.get("dataset_profile") if isinstance(context.get("dataset_profile"), dict) else {}

        # --- Extract optimization blueprint candidates (from ModelAnalyst) ---
        optimization_blueprint = context.get("optimization_blueprint")
        if not isinstance(optimization_blueprint, dict):
            optimization_blueprint = {}
        blueprint_actions = optimization_blueprint.get("improvement_actions")
        if not isinstance(blueprint_actions, list):
            blueprint_actions = []
        optimization_policy = resolve_optimization_policy(context)
        blueprint_actions, _blocked_blueprint_actions = filter_optimization_actions(
            blueprint_actions,
            optimization_policy,
        )
        blueprint_candidates = []
        for bp_action in blueprint_actions:
            if not isinstance(bp_action, dict):
                continue
            technique = str(bp_action.get("technique") or "").strip()
            if not technique:
                continue
            blueprint_candidates.append({
                "technique": technique,
                "target_columns": bp_action.get("target_columns", ["ALL_NUMERIC"]),
                "feature_scope": "model_features",
                "params": bp_action.get("concrete_params") if isinstance(bp_action.get("concrete_params"), dict) else {},
                "objective": str(bp_action.get("code_change_hint") or f"Apply {technique} from model analysis blueprint"),
                "_source": "model_analyst_blueprint",
                "_priority": int(bp_action.get("priority", 5)) if isinstance(bp_action.get("priority"), int) else 5,
            })
        blueprint_candidates.sort(key=lambda x: x.get("_priority", 5))

        candidates = self._candidate_techniques_from_plan(feature_engineering_plan, dataset_profile=dataset_profile)
        if blueprint_candidates:
            candidates = self._dedupe_candidates(blueprint_candidates + candidates)
        if not candidates:
            candidates = self._fallback_candidates_from_critique(critique_packet, context=context)
        candidates, _blocked_candidates = filter_optimization_actions(candidates, optimization_policy)
        ranked_candidates = self._rank_iteration_candidates(
            candidates,
            critique_packet=critique_packet,
            dataset_profile=dataset_profile,
            tracker_entries=tracker_entries,
        )

        selected_candidate = None
        selected_signature = ""
        for candidate in ranked_candidates:
            signature = str(candidate.get("_signature") or "").strip()
            if not signature:
                signature = build_hypothesis_signature(
                    technique=candidate.get("technique"),
                    target_columns=candidate.get("target_columns"),
                    feature_scope=candidate.get("feature_scope"),
                    params=candidate.get("params"),
                )
            if signature not in known_signatures:
                selected_candidate = candidate
                selected_signature = signature
                break

        duplicate_of = None
        action = "APPLY"
        if selected_candidate is None:
            selected_candidate = ranked_candidates[0] if ranked_candidates else {
                "technique": "NO_OP",
                "target_columns": ["ALL_NUMERIC"],
                "feature_scope": "model_features",
                "params": {},
                "objective": "No actionable candidate found.",
            }
            selected_signature = build_hypothesis_signature(
                technique=selected_candidate.get("technique"),
                target_columns=selected_candidate.get("target_columns"),
                feature_scope=selected_candidate.get("feature_scope"),
                params=selected_candidate.get("params"),
            )
            if selected_signature in known_signatures:
                duplicate_of = selected_signature
            action = "NO_OP"

        target_columns = normalize_target_columns(selected_candidate.get("target_columns"))
        technique = str(selected_candidate.get("technique") or "NO_OP").strip() or "NO_OP"
        feature_scope = str(selected_candidate.get("feature_scope") or "model_features").strip() or "model_features"
        params = selected_candidate.get("params") if isinstance(selected_candidate.get("params"), dict) else {}
        objective = str(selected_candidate.get("objective") or "").strip()
        if not objective:
            objective = "Apply one focused feature-engineering change aligned to critique findings."

        error_modes = critique_packet.get("error_modes")
        if not isinstance(error_modes, list):
            error_modes = []
        target_error_modes = [
            str(item.get("id"))
            for item in error_modes
            if isinstance(item, dict) and str(item.get("id") or "").strip()
        ][:3]
        if not target_error_modes:
            target_error_modes = ["metric_stagnation"]
        ranking_reasons = (
            [str(item) for item in (selected_candidate.get("_ranking_reasons") or []) if str(item).strip()]
            if isinstance(selected_candidate, dict)
            else []
        )
        ranking_reason_text = ", ".join(ranking_reasons[:3])

        packet = {
            "packet_type": "iteration_hypothesis_packet",
            "packet_version": "1.0",
            "run_id": run_id,
            "iteration": iteration,
            "hypothesis_id": "h_" + hashlib.sha1(selected_signature.encode("utf-8")).hexdigest()[:8],
            "action": action,
            "hypothesis": {
                "technique": technique if action == "APPLY" else "NO_OP",
                "objective": objective,
                "target_columns": target_columns,
                "feature_scope": feature_scope,
                "params": params,
                "expected_effect": {
                    "target_error_modes": target_error_modes,
                    "direction": "positive" if action == "APPLY" else "neutral",
                },
            },
            "application_constraints": {
                "edit_mode": "incremental",
                "max_code_regions_to_change": 5,
                "forbid_replanning": True,
                "forbid_model_family_switch": False,
                "must_keep": ["data_split_logic", "output_paths_contract"],
            },
            "success_criteria": {
                "primary_metric_name": primary_metric_name,
                "min_delta": float(min_delta),
                "must_pass_active_gates": True,
            },
            "tracker_context": {
                "signature": selected_signature,
                "is_duplicate": bool(duplicate_of),
                "duplicate_of": duplicate_of,
            },
            "explanation": (
                (
                    "Evidence-ranked candidate selected from blueprint/plan: " + ranking_reason_text
                    if ranking_reason_text and action == "APPLY"
                    else "Single hypothesis selected from feature_engineering_plan and critique packet."
                )
                if action == "APPLY"
                else "No-op hypothesis because selected signature already exists in experiment tracker."
            )[:280],
            "fallback_if_not_applicable": "NO_OP",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        # Enrich with blueprint-specific context if candidate came from ModelAnalyst
        if selected_candidate and selected_candidate.get("_source") == "model_analyst_blueprint":
            packet["hypothesis"]["blueprint_params"] = selected_candidate.get("params", {})
            packet["hypothesis"]["code_change_hint"] = selected_candidate.get("objective", "")
            if ranking_reason_text:
                packet["explanation"] = (
                    f"Blueprint-ranked: {ranking_reason_text}. {selected_candidate.get('objective', '')}"
                )[:280]
            else:
                packet["explanation"] = f"Blueprint-driven: {selected_candidate.get('objective', '')}"[:280]

        valid_packet, errors = validate_iteration_hypothesis_packet(packet)
        if valid_packet:
            return packet
        repaired_packet = self._sanitize_iteration_hypothesis_packet(
            packet,
            primary_metric_name=primary_metric_name,
            min_delta=min_delta,
            fallback_signature=selected_signature,
        )
        repaired_valid, repaired_errors = validate_iteration_hypothesis_packet(repaired_packet)
        if repaired_valid:
            return repaired_packet
        return build_noop_iteration_hypothesis_packet(
            run_id=run_id,
            iteration=iteration,
            signature=selected_signature or "noop_signature",
            duplicate_of=duplicate_of or selected_signature or "noop_signature",
            primary_metric_name=primary_metric_name,
            min_delta=min_delta,
            explanation=(
                "Schema fallback to no-op hypothesis."
                + (
                    " Validation: "
                    + "; ".join((errors + repaired_errors)[:2])
                    if (errors or repaired_errors)
                    else ""
                )
            )[:280],
        )

    def _sanitize_iteration_hypothesis_packet(
        self,
        packet: Dict[str, Any],
        *,
        primary_metric_name: str,
        min_delta: float,
        fallback_signature: str,
        prefer_noop_on_missing_technique: bool = False,
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = dict(packet) if isinstance(packet, dict) else {}
        out["packet_type"] = "iteration_hypothesis_packet"
        out["packet_version"] = "1.0"

        run_id = str(out.get("run_id") or "unknown_run").strip() or "unknown_run"
        out["run_id"] = run_id

        try:
            iteration = int(out.get("iteration") or 1)
        except Exception:
            iteration = 1
        if iteration <= 0:
            iteration = 1
        out["iteration"] = iteration

        tracker_context = out.get("tracker_context") if isinstance(out.get("tracker_context"), dict) else {}
        signature = str(tracker_context.get("signature") or fallback_signature or "hypothesis_signature").strip()
        if not signature:
            signature = "hypothesis_signature"
        if len(signature) > 512:
            signature = signature[:512]
        duplicate_of_raw = tracker_context.get("duplicate_of")
        duplicate_of = str(duplicate_of_raw).strip() if duplicate_of_raw not in (None, "") else None
        is_duplicate = bool(tracker_context.get("is_duplicate"))
        if is_duplicate and not duplicate_of:
            duplicate_of = signature

        action = str(out.get("action") or "NO_OP").strip().upper()
        action = action if action in {"APPLY", "NO_OP"} else "NO_OP"
        if is_duplicate:
            action = "NO_OP"
        out["action"] = action

        hypothesis = out.get("hypothesis") if isinstance(out.get("hypothesis"), dict) else {}
        technique = str(hypothesis.get("technique") or "").strip()
        if action == "NO_OP":
            technique = "NO_OP"
        elif not technique or technique.upper() == "NO_OP":
            if prefer_noop_on_missing_technique:
                action = "NO_OP"
                out["action"] = action
                technique = "NO_OP"
                is_duplicate = False
                duplicate_of = None
            else:
                technique = "missing_indicators"

        objective = str(hypothesis.get("objective") or "").strip()
        if not objective:
            objective = (
                "No-op because strategist did not provide a concrete next experiment."
                if action == "NO_OP"
                else "Apply one focused feature-engineering change aligned to critique findings."
            )
        if len(objective) > 220:
            objective = objective[:217].rstrip() + "..."

        target_columns = normalize_target_columns(hypothesis.get("target_columns"))
        if len(target_columns) > 200:
            target_columns = target_columns[:200]

        feature_scope = str(hypothesis.get("feature_scope") or "model_features").strip() or "model_features"
        if feature_scope not in {
            "model_features",
            "segmentation_features",
            "audit_only_features",
            "all_features",
        }:
            feature_scope = "model_features"

        params = hypothesis.get("params") if isinstance(hypothesis.get("params"), dict) else {}
        expected_effect = (
            hypothesis.get("expected_effect")
            if isinstance(hypothesis.get("expected_effect"), dict)
            else {}
        )
        target_error_modes_raw = (
            expected_effect.get("target_error_modes")
            if isinstance(expected_effect.get("target_error_modes"), list)
            else []
        )
        target_error_modes: List[str] = []
        for item in target_error_modes_raw:
            token = str(item or "").strip()
            if not token or token in target_error_modes:
                continue
            target_error_modes.append(token)
            if len(target_error_modes) >= 5:
                break
        if not target_error_modes:
            target_error_modes = ["metric_stagnation"]
        direction = str(expected_effect.get("direction") or "").strip().lower()
        if direction not in {"positive", "neutral", "negative"}:
            direction = "neutral" if action == "NO_OP" else "positive"

        out["hypothesis"] = {
            "technique": technique,
            "objective": objective,
            "target_columns": target_columns,
            "feature_scope": feature_scope,
            "params": params,
            "expected_effect": {
                "target_error_modes": target_error_modes,
                "direction": direction,
            },
        }

        app_constraints = (
            out.get("application_constraints")
            if isinstance(out.get("application_constraints"), dict)
            else {}
        )
        try:
            max_regions = int(app_constraints.get("max_code_regions_to_change", 5) or 5)
        except Exception:
            max_regions = 5
        max_regions = min(12, max(1, max_regions))
        must_keep_raw = app_constraints.get("must_keep") if isinstance(app_constraints.get("must_keep"), list) else []
        must_keep: List[str] = []
        for item in must_keep_raw:
            token = str(item or "").strip()
            if not token or token in must_keep:
                continue
            must_keep.append(token)
            if len(must_keep) >= 10:
                break
        if not must_keep:
            must_keep = ["data_split_logic", "output_paths_contract"]
        out["application_constraints"] = {
            "edit_mode": "incremental",
            "max_code_regions_to_change": max_regions,
            "forbid_replanning": bool(app_constraints.get("forbid_replanning", True)),
            "forbid_model_family_switch": bool(app_constraints.get("forbid_model_family_switch", False)),
            "must_keep": must_keep,
        }

        success_criteria = (
            out.get("success_criteria")
            if isinstance(out.get("success_criteria"), dict)
            else {}
        )
        try:
            safe_min_delta = float(success_criteria.get("min_delta", min_delta) or min_delta)
        except Exception:
            safe_min_delta = float(min_delta)
        if safe_min_delta < 0:
            safe_min_delta = 0.0
        out["success_criteria"] = {
            "primary_metric_name": str(
                success_criteria.get("primary_metric_name") or primary_metric_name or "primary_metric"
            ).strip()
            or "primary_metric",
            "min_delta": safe_min_delta,
            "must_pass_active_gates": bool(success_criteria.get("must_pass_active_gates", True)),
        }

        if action != "NO_OP":
            is_duplicate = False
            duplicate_of = None
        out["tracker_context"] = {
            "signature": signature,
            "is_duplicate": bool(is_duplicate),
            "duplicate_of": duplicate_of if is_duplicate else None,
        }

        hypothesis_id = str(out.get("hypothesis_id") or "").strip()
        if not re.match(r"^h_[a-zA-Z0-9_-]{6,64}$", hypothesis_id):
            hypothesis_id = "h_" + hashlib.sha1(signature.encode("utf-8")).hexdigest()[:8]
        out["hypothesis_id"] = hypothesis_id

        explanation = str(out.get("explanation") or "").strip()
        if not explanation:
            explanation = (
                "Single hypothesis selected from feature_engineering_plan and critique packet."
                if action == "APPLY"
                else "No-op hypothesis because selected signature already exists in experiment tracker."
            )
        if len(explanation) > 280:
            explanation = explanation[:277].rstrip() + "..."
        out["explanation"] = explanation
        out["fallback_if_not_applicable"] = "NO_OP"
        timestamp = str(out.get("timestamp_utc") or "").strip()
        if not timestamp:
            timestamp = datetime.now(timezone.utc).isoformat()
        out["timestamp_utc"] = timestamp
        return out

    def _get_wide_schema_threshold(self) -> int:
        raw = os.getenv("STRATEGIST_WIDE_SCHEMA_THRESHOLD", "240")
        try:
            value = int(raw)
        except Exception:
            value = 240
        return max(50, min(value, 5000))

    def _get_wide_required_columns_max(self) -> int:
        raw = os.getenv("STRATEGIST_WIDE_REQUIRED_COLUMNS_MAX", "48")
        try:
            value = int(raw)
        except Exception:
            value = 48
        return max(8, min(value, 256))

    def _get_json_repair_attempts(self) -> int:
        raw = os.getenv("STRATEGIST_JSON_REPAIR_ATTEMPTS", "2")
        try:
            value = int(raw)
        except Exception:
            value = 2
        return max(0, min(value, 4))

    def _get_json_repair_max_chars(self) -> int:
        raw = os.getenv("STRATEGIST_JSON_REPAIR_MAX_CHARS", "180000")
        try:
            value = int(raw)
        except Exception:
            value = 180000
        return max(12000, min(value, 400000))

    def _get_strategy_count(self) -> int:
        raw = os.getenv("STRATEGIST_STRATEGY_COUNT", "1")
        try:
            value = int(raw)
        except Exception:
            value = 1
        return 1 if value <= 1 else 3

    def _get_diversity_threshold(self) -> float:
        raw = os.getenv("STRATEGIST_DIVERSITY_SIMILARITY_THRESHOLD", "0.8")
        try:
            value = float(raw)
        except Exception:
            value = 0.8
        return max(0.5, min(value, 0.98))

    def _get_diversity_repair_attempts(self) -> int:
        raw = os.getenv("STRATEGIST_DIVERSITY_REPAIR_ATTEMPTS", "1")
        try:
            value = int(raw)
        except Exception:
            value = 1
        return max(0, min(value, 3))

    def _column_families(self, allowed_columns: List[str]) -> List[Dict[str, Any]]:
        buckets: Dict[str, List[Tuple[int, str]]] = {}
        for col in allowed_columns:
            if not isinstance(col, str):
                continue
            m = re.match(r"^([A-Za-z_]+)(\d+)$", col.strip())
            if not m:
                continue
            prefix = m.group(1)
            idx = int(m.group(2))
            buckets.setdefault(prefix, []).append((idx, col))
        families: List[Dict[str, Any]] = []
        for prefix, pairs in buckets.items():
            if len(pairs) < 6:
                continue
            pairs_sorted = sorted(pairs, key=lambda x: x[0])
            families.append(
                {
                    "prefix": prefix,
                    "count": len(pairs_sorted),
                    "index_min": pairs_sorted[0][0],
                    "index_max": pairs_sorted[-1][0],
                    "examples": [
                        pairs_sorted[0][1],
                        pairs_sorted[len(pairs_sorted) // 2][1],
                        pairs_sorted[-1][1],
                    ],
                }
            )
        families.sort(key=lambda item: int(item.get("count", 0)), reverse=True)
        return families

    def _build_inventory_payload(
        self,
        allowed_columns: List[str],
        column_sets: Dict[str, Any],
        column_manifest: Dict[str, Any],
        *,
        wide_schema_mode: bool,
    ) -> str:
        if not wide_schema_mode:
            return json.dumps(allowed_columns, ensure_ascii=False)

        if isinstance(column_manifest, dict) and column_manifest:
            payload = {
                "mode": "wide_schema_column_manifest",
                "schema_mode": column_manifest.get("schema_mode") or "wide",
                "total_columns": int(column_manifest.get("total_columns") or len(allowed_columns)),
                "anchors": column_manifest.get("anchors") if isinstance(column_manifest.get("anchors"), list) else [],
                "families": column_manifest.get("families") if isinstance(column_manifest.get("families"), list) else [],
                "instruction": (
                    "Use anchors as explicit required_columns. "
                    "For dense families, use feature_families selector hints; do not enumerate all members."
                ),
            }
            return json.dumps(payload, ensure_ascii=False)

        families = self._column_families(allowed_columns)
        family_columns = set()
        for fam in families:
            prefix = str(fam.get("prefix") or "")
            if not prefix:
                continue
            pattern = re.compile(rf"^{re.escape(prefix)}\d+$")
            for col in allowed_columns:
                if pattern.match(col):
                    family_columns.add(col)

        special_columns = [col for col in allowed_columns if col not in family_columns]
        payload = {
            "mode": "wide_schema_compact_inventory",
            "total_columns": len(allowed_columns),
            "inventory_fingerprint_sha1": hashlib.sha1(
                "\n".join(allowed_columns).encode("utf-8", errors="ignore")
            ).hexdigest(),
            "feature_families_detected": families,
            "explicit_non_family_columns": special_columns,
            "column_sets_hints": column_sets if isinstance(column_sets, dict) else {},
            "instruction": (
                "Use explicit_non_family_columns for exact names and feature_families_detected for dense "
                "numeric families. Do not enumerate every member of large families in required_columns."
            ),
        }
        return json.dumps(payload, ensure_ascii=False)

    def _build_json_repair_prompt(
        self,
        *,
        raw_output: str,
        parse_error: str,
        data_summary: str,
        user_request: str,
    ) -> str:
        from src.utils.prompting import render_prompt

        REPAIR_TEMPLATE = """
        You are repairing a strategist JSON that failed to parse.
        IMPORTANT: Edit the provided JSON draft. Do NOT rewrite from scratch.

        Return ONLY valid raw JSON.
        No markdown, no commentary.

        Constraints:
        - Preserve existing strategy meaning, titles, metrics, and reasoning whenever present.
        - Keep the same top-level schema: {"strategies":[...]}.
        - If the tail is truncated (unclosed brackets/strings), complete ALL missing fields using context.
        - CRITICAL: Each strategy MUST have a non-empty "techniques" array with at least 2-3 ML techniques.
        - CRITICAL: Each strategy MUST have a non-empty "required_columns" array listing columns from the data.
        - Remove dangling commas and invalid tokens.

        *** USER REQUEST ***
        "$user_request"

        *** DATA SUMMARY (REFERENCE) ***
        $data_summary

        *** PARSE ERROR ***
        $parse_error

        *** BROKEN JSON DRAFT TO EDIT ***
        $broken_json
        """
        return render_prompt(
            REPAIR_TEMPLATE,
            user_request=user_request,
            data_summary=data_summary,
            parse_error=parse_error,
            broken_json=raw_output,
        )

    def _strategy_token_set(self, strategy: Dict[str, Any]) -> set[str]:
        tokens: set[str] = set()
        if not isinstance(strategy, dict):
            return tokens

        def _add_text(value: Any) -> None:
            text = str(value or "").strip().lower()
            if not text:
                return
            for tok in re.findall(r"[a-z0-9_]{3,}", text):
                tokens.add(tok)

        for key in ("objective_type", "analysis_type", "title", "validation_strategy", "success_metric"):
            _add_text(strategy.get(key))

        for item in strategy.get("techniques") or []:
            _add_text(item)
        for name in self._extract_required_column_names(strategy):
            _add_text(name)
        return tokens

    def _strategy_similarity(self, left: Dict[str, Any], right: Dict[str, Any]) -> float:
        left_tokens = self._strategy_token_set(left)
        right_tokens = self._strategy_token_set(right)
        if left_tokens and right_tokens:
            union = left_tokens | right_tokens
            overlap = left_tokens & right_tokens
            jaccard = float(len(overlap) / max(len(union), 1))
        else:
            jaccard = 0.0
        left_title = str(left.get("title") or "")
        right_title = str(right.get("title") or "")
        title_sim = SequenceMatcher(None, left_title.lower(), right_title.lower()).ratio()
        return max(jaccard, title_sim)

    def _find_redundant_strategy_pairs(
        self,
        strategies: List[Dict[str, Any]],
        threshold: float,
    ) -> List[Dict[str, Any]]:
        issues: List[Dict[str, Any]] = []
        for idx_a in range(len(strategies)):
            for idx_b in range(idx_a + 1, len(strategies)):
                left = strategies[idx_a] if isinstance(strategies[idx_a], dict) else {}
                right = strategies[idx_b] if isinstance(strategies[idx_b], dict) else {}
                sim = self._strategy_similarity(left, right)
                if sim >= threshold:
                    issues.append(
                        {
                            "left_index": idx_a,
                            "right_index": idx_b,
                            "similarity": round(float(sim), 4),
                            "left_title": str(left.get("title") or f"strategy_{idx_a}"),
                            "right_title": str(right.get("title") or f"strategy_{idx_b}"),
                        }
                    )
        return issues

    def _build_diversity_repair_prompt(
        self,
        *,
        payload: Dict[str, Any],
        redundancy_report: List[Dict[str, Any]],
        allowed_columns: List[str],
        max_required_columns: Optional[int],
        data_summary: str,
        user_request: str,
        strategy_count: int,
    ) -> str:
        from src.utils.prompting import render_prompt

        template = """
You are repairing strategist output to ensure strategy diversity.
Keep business objective alignment, but make strategies materially distinct.

Return ONLY valid raw JSON with top-level key "strategies".
Do not include markdown.

Constraints:
- Preserve exact column names from AUTHORIZED_COLUMN_INVENTORY.
- Keep each strategy executable and coherent.
- Each strategy must differ in at least one of:
  1) technique family
  2) validation strategy rationale
  3) fallback chain
- Keep strategy count = $strategy_count
- If required_columns budget exists, respect it.

*** USER REQUEST ***
$user_request

*** DATA SUMMARY ***
$data_summary

*** AUTHORIZED COLUMN INVENTORY ***
$authorized_column_inventory

*** REQUIRED COLUMNS BUDGET ***
$required_columns_budget

*** REDUNDANCY REPORT ***
$redundancy_report

*** CURRENT STRATEGIES JSON (EDIT THIS) ***
$payload_json
"""
        budget_text = (
            str(max_required_columns)
            if isinstance(max_required_columns, int) and max_required_columns > 0
            else "none"
        )
        return render_prompt(
            template,
            user_request=user_request,
            data_summary=data_summary,
            authorized_column_inventory=json.dumps(allowed_columns, ensure_ascii=False),
            required_columns_budget=budget_text,
            redundancy_report=json.dumps(redundancy_report, ensure_ascii=False, indent=2),
            payload_json=json.dumps(payload, ensure_ascii=False, indent=2),
            strategy_count=str(strategy_count),
        )

    def _ensure_strategy_diversity(
        self,
        *,
        payload: Dict[str, Any],
        data_summary: str,
        user_request: str,
        allowed_columns: List[str],
        max_required_columns: Optional[int],
        strategy_count: int,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if not isinstance(payload, dict):
            return {"strategies": []}, {"status": "invalid_payload"}
        strategies = payload.get("strategies")
        if not isinstance(strategies, list):
            return payload, {"status": "missing_strategies"}
        strategies = [s for s in strategies if isinstance(s, dict)]
        if len(strategies) <= 1:
            return payload, {"status": "not_applicable", "strategy_count": len(strategies)}
        if any(bool(s.get("_autofilled_variant")) for s in strategies):
            return payload, {
                "status": "autofill_skip",
                "strategy_count": len(strategies),
                "repair_applied": False,
            }

        threshold = self._get_diversity_threshold()
        redundancy = self._find_redundant_strategy_pairs(strategies, threshold)
        if not redundancy:
            return payload, {
                "status": "ok",
                "threshold": threshold,
                "redundant_pairs": [],
                "repair_applied": False,
            }

        attempts = self._get_diversity_repair_attempts()
        if attempts <= 0:
            return payload, {
                "status": "redundant_no_repair",
                "threshold": threshold,
                "redundant_pairs": redundancy,
                "repair_applied": False,
            }

        best_payload = payload
        best_redundancy = redundancy
        for attempt in range(1, attempts + 1):
            prompt = self._build_diversity_repair_prompt(
                payload=best_payload,
                redundancy_report=best_redundancy,
                allowed_columns=allowed_columns,
                max_required_columns=max_required_columns,
                data_summary=data_summary,
                user_request=user_request,
                strategy_count=strategy_count,
            )
            try:
                repaired = self._call_model(
                    prompt,
                    temperature=0.0,
                    context_tag="strategist_diversity_repair",
                )
                candidate_parsed = json.loads(self._clean_json(repaired))
                candidate_payload = self._normalize_strategist_output(candidate_parsed)
                candidate_payload = self._apply_authoritative_strategy_hardening(candidate_payload, allowed_columns)
                candidate_payload = self._enforce_strategy_count(candidate_payload, strategy_count)
                candidate_validation = self._validate_required_columns(
                    candidate_payload,
                    allowed_columns,
                    max_required_columns=max_required_columns,
                )
                if int(candidate_validation.get("invalid_count", 0)) > 0:
                    continue
                candidate_strategies = [
                    s for s in (candidate_payload.get("strategies") or []) if isinstance(s, dict)
                ]
                candidate_redundancy = self._find_redundant_strategy_pairs(candidate_strategies, threshold)
                if len(candidate_redundancy) < len(best_redundancy):
                    best_payload = candidate_payload
                    best_redundancy = candidate_redundancy
                if not candidate_redundancy:
                    return best_payload, {
                        "status": "repaired",
                        "threshold": threshold,
                        "repair_applied": True,
                        "attempts_used": attempt,
                        "redundant_pairs": [],
                    }
            except Exception:
                continue

        return best_payload, {
            "status": "partially_repaired" if len(best_redundancy) < len(redundancy) else "redundant_unresolved",
            "threshold": threshold,
            "repair_applied": len(best_redundancy) < len(redundancy),
            "attempts_used": attempts,
            "redundant_pairs": best_redundancy,
        }

    def _parse_with_json_repair(
        self,
        raw_json: str,
        *,
        data_summary: str,
        user_request: str,
    ) -> Tuple[Any, Dict[str, Any]]:
        try:
            parsed = json.loads(raw_json)
            return parsed, {
                "status": "ok",
                "repair_applied": False,
                "attempts_used": 0,
            }
        except json.JSONDecodeError as decode_err:
            attempts = self._get_json_repair_attempts()
            if attempts <= 0:
                raise

            print(f"Strategist JSON parse failed. Starting repair attempts. error={decode_err}")
            current_draft = raw_json
            max_chars = self._get_json_repair_max_chars()
            last_err: Exception = decode_err
            for attempt in range(1, attempts + 1):
                print(f"Strategist JSON repair attempt {attempt}/{attempts}")
                bounded_draft = current_draft if len(current_draft) <= max_chars else current_draft[-max_chars:]
                repair_prompt = self._build_json_repair_prompt(
                    raw_output=bounded_draft,
                    parse_error=str(last_err),
                    data_summary=data_summary,
                    user_request=user_request,
                )
                self.last_repair_prompt = repair_prompt
                repaired_content = self._call_model(
                    repair_prompt,
                    temperature=0.0,
                    context_tag="strategist_json_repair",
                )
                self.last_repair_response = repaired_content
                candidate_raw = self._clean_json(repaired_content)
                try:
                    parsed = json.loads(candidate_raw)
                    return parsed, {
                        "status": "repaired",
                        "repair_applied": True,
                        "attempts_used": attempt,
                        "initial_error": str(decode_err),
                    }
                except json.JSONDecodeError as attempt_err:
                    print(f"Strategist JSON repair attempt failed: {attempt_err}")
                    last_err = attempt_err
                    # Keep trying from the latest candidate, preserving prior reconstruction work.
                    current_draft = candidate_raw or current_draft

            raise decode_err

    def generate_strategies(
        self,
        data_summary: str,
        user_request: str,
        column_inventory: Optional[List[str]] = None,
        column_sets: Optional[Dict[str, Any]] = None,
        column_manifest: Optional[Dict[str, Any]] = None,
        compute_constraints: Optional[Dict[str, Any]] = None,
        column_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generates a single strategy based on the data summary and user request.
        """
        
        from src.utils.prompting import render_prompt
        allowed_columns = self._normalize_column_inventory(column_inventory)
        column_sets_payload = column_sets if isinstance(column_sets, dict) else {}
        column_manifest_payload = column_manifest if isinstance(column_manifest, dict) else {}
        wide_schema_threshold = self._get_wide_schema_threshold()
        wide_schema_mode = len(allowed_columns) > wide_schema_threshold
        required_columns_budget = self._get_wide_required_columns_max() if wide_schema_mode else None
        inventory_payload = self._build_inventory_payload(
            allowed_columns,
            column_sets_payload,
            column_manifest_payload,
            wide_schema_mode=wide_schema_mode,
        )
        wide_schema_guidance = (
            (
                f"WIDE-SCHEMA MODE ACTIVE: total columns={len(allowed_columns)} > threshold={wide_schema_threshold}. "
                "Avoid enumerating dense feature families in required_columns. Use compact role-critical anchors "
                "and optionally capture families under feature_families."
            )
            if wide_schema_mode
            else "NORMAL-SCHEMA MODE: list required_columns explicitly, only those relevant to the strategy."
        )
        required_columns_budget_guidance = (
            (
                f"required_columns budget in this run: <= {required_columns_budget} per strategy. "
                "If more features are relevant, keep anchors in required_columns and describe families in feature_families."
            )
            if required_columns_budget
            else "No explicit required_columns budget in this run."
        )
        strategy_count = self._get_strategy_count()
        if strategy_count == 1:
            strategy_count_guidance = (
                "Generate 1 primary strategy with complete detail. "
                "Include a concise fallback_chain inside the same strategy."
            )
        else:
            strategy_count_guidance = "Generate 3 distinct strategies."
        strategy_goal = (
            "craft ONE optimal strategy"
            if strategy_count == 1
            else f"craft {strategy_count} materially distinct executable strategies"
        )
        compute_constraints_payload = (
            compute_constraints
            if isinstance(compute_constraints, dict)
            else {}
        )
        column_metadata_payload = json.dumps(
            column_metadata if isinstance(column_metadata, dict) else {},
            ensure_ascii=False,
        )

        SYSTEM_PROMPT_TEMPLATE = """
        You are a Chief Data Strategist inside a multi-agent system. Your goal is to $strategy_goal
        that downstream AI engineers can execute successfully.

        === SENIOR STRATEGY PROTOCOL ===
        $senior_strategy_protocol

        *** MISSION ***
        - Reason from the business objective and dataset evidence first.
        - Produce the smallest strategy that is genuinely fit for THIS run.
        - Optimize for downstream executability, not for generic ML completeness.

        *** SOURCE OF TRUTH AND PRECEDENCE ***
        1) USER REQUEST + business objective language (authoritative for real intent)
        2) DATASET SUMMARY + STEWARD_FACTS + COLUMN METADATA (authoritative for feasibility)
        3) AUTHORIZED COLUMN INVENTORY / COLUMN SETS / COLUMN MANIFEST (authoritative for column names and families)
        4) COMPUTE CONSTRAINTS (authoritative for practicality)
        5) Generic data science patterns are advisory only and must never override the current run context
        6) If free-text narrative conflicts with STEWARD_FACTS, DATASET_SEMANTICS_SUMMARY, or COLUMN METADATA,
           the structured facts win. Do not blend contradictory rules into a compromise.

        *** DATASET SUMMARY ***
        $data_summary

        *** AUTHORIZED COLUMN INVENTORY (SOURCE OF TRUTH FOR COLUMN NAMES) ***
        $authorized_column_inventory

        *** COLUMN SETS (OPTIONAL, MAY BE EMPTY) ***
        $column_sets

        *** COLUMN MANIFEST (OPTIONAL, MAY BE EMPTY) ***
        $column_manifest

        *** COLUMN METADATA (STRUCTURED) ***
        $column_metadata

        This metadata provides:
        - "target": The explicit prediction target column (DO NOT guess — use this).
        - "split_column": The train/test split column (exclude from model features).
        - "id_columns": Identifier columns (exclude from model features).
        - "column_types": Columns grouped by type (numeric vs low_cardinality/categorical).
        - "column_stats": Per-column statistics (dtype, missing_rate, cardinality, numeric ranges, top values for low-cardinality).
        - "quality_flags": Data quality issues (high missingness, constant columns).
        Use this to make informed decisions about feature engineering, model choice, and validation strategy.

        *** SCHEMA MODE GUIDANCE ***
        $wide_schema_guidance

        *** REQUIRED COLUMNS BUDGET ***
        $required_columns_budget_guidance

        *** COMPUTE CONSTRAINTS ***
        $compute_constraints

        *** USER REQUEST ***
        "$user_request"

        *** STRATEGY REASONING ***
        Before writing the strategy, reason through these areas (order is yours to decide):
        - What is the real business intent of THIS run?
        - What operational scope does it need? (cleaning_only, ml_only, or full_pipeline)
        - What objective_type best captures the intent? (this is a compact label for your conclusion, not the starting point)
        - What validation, techniques, and artifacts are calibrated to that scope?
        - What is the minimal credible plan for THIS dataset — not a generically impressive one?

        *** SCOPE AWARENESS ***
        Not all business objectives require a full ML pipeline. Determine what the objective actually needs
        and set scope_recommendation accordingly:
        - **cleaning_only**: Data quality / ETL objectives — strategy focused on cleaning, quality gates, validation. No ML models.
        - **ml_only**: Pre-cleaned data — strategy focused on modeling, feature engineering, evaluation. Minimal cleaning.
        - **full_pipeline**: End-to-end prediction/optimization work — cleaning through evaluation.

        Your techniques, required_columns, and evaluation approach must align with the scope you identify.

        *** YOUR TASK ***
        Design a strategy using FIRST PRINCIPLES REASONING.
        Do not start from canned problem categories.
        Instead, reason through WHAT the business is trying to achieve, WHAT decision follows,
        WHAT this run should and should not do, and only then assign compact labels such as
        objective_type or scope_recommendation.

        CRITICAL: Your reasoning must be universal and adaptable to ANY objective, not hardcoded to specific problem types.

        *** BUSINESS UNDERSTANDING ***
        Before proposing techniques, reason through:
        - What is the business trying to learn, achieve, or decide?
        - What action or decision follows from the analysis?
        - What success metric captures business value (not generic ML metrics)?

        Then translate to a data science framing:
        - **objective_type**: Common types include descriptive, predictive, prescriptive, causal, comparative — but use whatever label best captures the intent. If none fits, propose your own with clear reasoning.
        - **objective_reasoning**: WHY this type fits the business goal (2-3 sentences)
        - **success_metric**: The metric that captures business success
        - **scope_recommendation**: One of [cleaning_only, ml_only, full_pipeline]
        - **scope_reasoning**: WHY this run needs that scope and what is out of scope

        *** STRATEGY ROADMAP THINKING (for iterative ML work only when scope requires it) ***
        If this looks like predictive or optimization work that may evolve across multiple
        rounds, reason about WHICH levers matter most for THIS dataset and THIS business goal.
        Use these levers as a toolbox, not as a mandatory sequence:
        - baseline robustness and validation credibility
        - feature signal extraction / encoding
        - regularization / calibration
        - hyperparameter search
        - variance reduction
        - ensembling / model diversification

        For each lever, ask:
        - What evidence says this lever has ROI now?
        - What is the cheapest valid version of this idea?
        - What could fail, and what is the fallback if it disappoints?

        It is valid to recommend only a small number of techniques when extra complexity
        is unlikely to pay off. Do NOT force every strategy to include every phase or
        every advanced technique.

        *** FEASIBILITY AND STRATEGY DESIGN ***
        Your decisions must be driven by THIS dataset's context, not arbitrary thresholds or canned capability lists.

        For every technique you propose, reason through:
        - **Statistical power**: Does the data have enough observations per feature to support this method?
        - **Signal quality**: Given the data profile (missing rates, cardinality, variance), is this method appropriate?
        - **Compute-value tradeoff**: Is the added complexity justified by expected lift over a simpler baseline?
        - **Failure modes**: What happens if this underperforms? Define a credible fallback.

        For each technique, state WHY it fits this data profile, WHAT could fail, and WHAT the fallback is.

        Key principles:
        - Never filter the target variable to a single class if the goal involves comparison or prediction.
        - Be broad with feature selection — include columns that might carry information.
        - Be clear about what exactly you are solving for and why.
           
        *** VALIDATION STRATEGY ***
        Choose validation strategy from the actual data structure, leakage risk, and compute budget.
        Reason about temporal ordering, grouped entities, label imbalance, sample size vs feature complexity,
        and whether the budget supports repeated CV or only a lighter estimate.
        Use numeric thresholds only when the current dataset context makes them defensible.

        *** METRICS AND ARTIFACTS ***
        Reason through what metrics and artifacts best measure success for THIS run's scope and objective.
        Think about what the business cares about, what the risks are, what validates the approach,
        and which outputs are genuinely needed — not a generic checklist.

        *** CRITICAL OUTPUT RULES ***
        - RETURN ONLY RAW JSON. NO MARKDOWN. NO COMMENTS.
        - The output must be a dictionary with a single key "strategies" containing a LIST of exactly $strategy_count objects.
        - Strategy count guidance: $strategy_count_guidance
        - The object must include these keys:
          {
            "title": "Strategy name",
            "objective_type": "Label that best captures business intent (common: descriptive, predictive, prescriptive, causal, comparative — or propose your own)",
            "objective_reasoning": "2-3 sentences explaining WHY this objective_type fits the business goal",
            "success_metric": "Primary business metric (not generic ML metric)",
            "scope_recommendation": "One of: cleaning_only, ml_only, full_pipeline",
            "scope_reasoning": "2-3 sentences explaining what this run must include and exclude",
            "recommended_evaluation_metrics": ["list", "of", "metrics", "to", "track"],
            "validation_strategy": "Strategy name with data-driven rationale (e.g., 'time_split: data has temporal ordering')",
            "validation_rationale": "2-3 sentences explaining WHY this validation fits the data structure",
            "analysis_type": "Brief label (e.g. 'Price Optimization', 'Churn Prediction')",
            "hypothesis": "What you expect to find or achieve",
            "required_columns": ["exact", "column", "names", "from", "summary"],
            "audit_only_columns": ["optional", "columns", "to", "preserve", "for", "audit", "reporting", "or", "stratification", "but", "not", "model", "inputs"],
            "feature_families": [{"family": "optional", "rationale": "optional", "selector_hint": "optional"}],
            "techniques": ["list", "of", "data science techniques"],
            "feasibility_analysis": {
              "statistical_power": "Assessment of n/p ratio and sample adequacy",
              "signal_quality": "Assessment of data quality for proposed method",
              "compute_value_tradeoff": "Is complexity justified by expected lift?"
            },
            "recommended_artifacts": [{"artifact_type": "string", "required": true, "rationale": "why"}],
            "fallback_chain": ["Primary approach", "Credible fallback", "Simpler safe option"],
            "expected_lift": "Estimated magnitude/direction of value relative to a simpler baseline",
            "estimated_difficulty": "Low | Medium | High (with data-driven justification)",
            "reasoning": "Why this strategy is optimal for the data and objective"
          }
        - "required_columns": Use EXACT column names from AUTHORIZED COLUMN INVENTORY only.
        - If WIDE-SCHEMA MODE is active, keep required_columns compact and role-critical.
        - In WIDE-SCHEMA MODE, do NOT enumerate all members of dense numeric families in required_columns.
          Use explicit anchor columns in required_columns and document families in feature_families.
        - In WIDE-SCHEMA MODE, required_columns count must stay inside the configured budget.
        - NEVER invent, rename, abbreviate, or infer columns not present in AUTHORIZED COLUMN INVENTORY.
        - If uncertain about a column, omit it (do not hallucinate).
        - If you conclude a column should be excluded from the initial model, quarantined, or used only for audit/reporting/stratification,
          list it in "audit_only_columns" instead of leaving that decision only in prose.
        - "objective_reasoning" must connect business goal → objective_type.
        - "scope_recommendation" and "scope_reasoning" must reflect THIS run, not generic defaults.
        - "feasibility_analysis" is required — no technique without data-driven justification.
        - "recommended_artifacts" must be scoped to THIS run. Cleaning-only runs should not require predictions.
        - "fallback_chain" must be concise and credible - every strategy needs a Plan B.
        - "reasoning" must include: why this fits the objective, what could fail, and recovery plan.
        """
        
        system_prompt = render_prompt(
            SYSTEM_PROMPT_TEMPLATE,
            strategy_goal=strategy_goal,
            data_summary=data_summary,
            authorized_column_inventory=inventory_payload,
            column_sets=json.dumps(column_sets_payload, ensure_ascii=False),
            column_manifest=json.dumps(column_manifest_payload, ensure_ascii=False),
            column_metadata=column_metadata_payload,
            wide_schema_guidance=wide_schema_guidance,
            required_columns_budget_guidance=required_columns_budget_guidance,
            compute_constraints=json.dumps(compute_constraints_payload, ensure_ascii=False),
            user_request=user_request,
            senior_strategy_protocol=SENIOR_STRATEGY_PROTOCOL,
            strategy_count=str(strategy_count),
            strategy_count_guidance=strategy_count_guidance,
        )
        self.last_prompt = system_prompt
        
        try:
            original_max_tokens = self._max_tokens
            # --- Truncation-aware generation with retry ---
            _max_generation_attempts = 2
            content = None
            for _gen_attempt in range(1, _max_generation_attempts + 1):
                content = self._call_model(
                    system_prompt,
                    temperature=0.1,
                    context_tag="strategist_generate",
                )
                _fr = str(self._last_finish_reason or "").lower()
                _is_truncated = _fr in ("length", "max_tokens")
                if _is_truncated and _gen_attempt < _max_generation_attempts:
                    print(
                        f"STRATEGIST_TRUNCATION_DETECTED: finish_reason={self._last_finish_reason}, "
                        f"response_len={len(content or '')}, attempt={_gen_attempt}/{_max_generation_attempts}. "
                        f"Retrying with higher max_tokens."
                    )
                    # Bump max_tokens for retry
                    self._max_tokens = min(self._max_tokens * 2, 65536)
                    continue
                break
            self._max_tokens = original_max_tokens
            self.last_response = content
            cleaned_content = self._clean_json(content)
            parsed, json_repair_meta = self._parse_with_json_repair(
                cleaned_content,
                data_summary=data_summary,
                user_request=user_request,
            )
            self.last_json_repair_meta = json_repair_meta

            # Normalization (Fix for crash 'list' object has no attribute 'get')
            payload = self._normalize_strategist_output(parsed)
            payload = self._apply_authoritative_strategy_hardening(payload, allowed_columns)
            payload = self._enforce_strategy_count(payload, strategy_count)
            payload, diversity_validation = self._ensure_strategy_diversity(
                payload=payload,
                data_summary=data_summary,
                user_request=user_request,
                allowed_columns=allowed_columns,
                max_required_columns=required_columns_budget,
                strategy_count=strategy_count,
            )
            column_validation = self._validate_required_columns(
                payload,
                allowed_columns,
                max_required_columns=required_columns_budget,
            )
            max_repairs = self._get_column_repair_attempts()
            if (
                allowed_columns
                and (
                    int(column_validation.get("invalid_count", 0)) > 0
                    or int(column_validation.get("over_budget_count", 0)) > 0
                )
                and max_repairs > 0
            ):
                payload, column_validation = self._repair_required_columns_with_llm(
                    payload=payload,
                    data_summary=data_summary,
                    user_request=user_request,
                    allowed_columns=allowed_columns,
                    column_sets=column_sets_payload,
                    validation_report=column_validation,
                    max_attempts=max_repairs,
                    max_required_columns=required_columns_budget,
                )
            column_validation["wide_schema_mode"] = wide_schema_mode
            column_validation["required_columns_budget"] = required_columns_budget
            payload["column_validation"] = column_validation
            payload["diversity_validation"] = diversity_validation
            payload["json_repair"] = json_repair_meta

            # Build strategy_spec from LLM reasoning (not hardcoded inference)
            strategy_spec = self._build_strategy_spec_from_llm(payload, data_summary, user_request)
            payload["strategy_spec"] = strategy_spec
            payload["strategy_count_requested"] = strategy_count
            return payload

        except Exception as e:
            self._max_tokens = original_max_tokens
            print(f"Strategist Error: {e}")
            # Fallback simple strategy
            fallback = {"strategies": [{
                "title": "Error Fallback Strategy",
                "objective_type": "descriptive",
                "objective_reasoning": "API error occurred. Defaulting to descriptive analysis as safest fallback.",
                "success_metric": "Data quality summary",
                "recommended_evaluation_metrics": ["completeness", "consistency"],
                "validation_strategy": "manual_review",
                "analysis_type": "statistical",
                "hypothesis": "Could not generate complex strategy. Analyzing basic correlations.",
                "required_columns": [],
                "techniques": ["correlation_analysis"],
                "estimated_difficulty": "Low",
                "reasoning": f"OpenRouter API Failed: {e}"
            }]}
            fallback["column_validation"] = self._validate_required_columns(
                fallback,
                allowed_columns,
                max_required_columns=required_columns_budget,
            )
            fallback = self._enforce_strategy_count(fallback, strategy_count)
            fallback["json_repair"] = {
                "status": "fallback",
                "repair_applied": False,
                "attempts_used": 0,
            }
            fallback["diversity_validation"] = {
                "status": "fallback",
                "repair_applied": False,
                "redundant_pairs": [],
            }
            strategy_spec = self._build_strategy_spec_from_llm(fallback, data_summary, user_request)
            fallback["strategy_spec"] = strategy_spec
            fallback["strategy_count_requested"] = strategy_count
            return fallback

    def _enforce_strategy_count(self, payload: Dict[str, Any], strategy_count: int) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            return {"strategies": []}
        strategies = payload.get("strategies")
        if not isinstance(strategies, list):
            strategies = []
        strategies = [
            self._normalize_feature_engineering_aliases(s)
            for s in strategies
            if isinstance(s, dict)
        ]
        if strategy_count <= 1:
            payload["strategies"] = strategies[:1]
            return payload
        if len(strategies) >= 3:
            payload["strategies"] = strategies[:3]
            return payload
        # Backfill with safest available entries if model under-produced.
        while len(strategies) < 3 and strategies:
            strategies.append(dict(strategies[-1]))
            strategies[-1]["title"] = f"{strategies[-1].get('title', 'Strategy')} (Variant {len(strategies)})"
            strategies[-1]["_autofilled_variant"] = True
        payload["strategies"] = strategies
        return payload

    def _normalize_feature_engineering_aliases(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize FE naming across legacy/new strategist schemas."""
        if not isinstance(strategy, dict):
            return {}

        normalized = dict(strategy)
        top_level_fe = normalized.get("feature_engineering")
        legacy_fe = normalized.get("feature_engineering_strategy")
        eval_plan = normalized.get("evaluation_plan")
        eval_plan_fe = eval_plan.get("feature_engineering") if isinstance(eval_plan, dict) else None

        resolved_techniques: List[Any] = []
        notes = ""
        risk_level = "low"

        # Preferred shape: object with techniques/notes/risk_level
        if isinstance(legacy_fe, dict):
            techniques = legacy_fe.get("techniques")
            if isinstance(techniques, list):
                resolved_techniques = list(techniques)
            notes_raw = legacy_fe.get("notes")
            if isinstance(notes_raw, str):
                notes = notes_raw.strip()
            risk_raw = str(legacy_fe.get("risk_level") or "").strip().lower()
            if risk_raw in {"low", "med", "high"}:
                risk_level = risk_raw
            elif risk_raw == "medium":
                risk_level = "med"

        # Legacy alias: list
        if not resolved_techniques:
            if isinstance(top_level_fe, list):
                resolved_techniques = list(top_level_fe)
            elif isinstance(legacy_fe, list):
                resolved_techniques = list(legacy_fe)
            elif isinstance(eval_plan_fe, list):
                resolved_techniques = list(eval_plan_fe)
            elif isinstance(eval_plan_fe, dict):
                techniques = eval_plan_fe.get("techniques")
                if isinstance(techniques, list):
                    resolved_techniques = list(techniques)

        fe_strategy_obj = {
            "techniques": resolved_techniques if isinstance(resolved_techniques, list) else [],
            "notes": notes,
            "risk_level": risk_level,
        }
        normalized["feature_engineering_strategy"] = fe_strategy_obj
        # Backward-compatible alias for current downstream consumers.
        normalized["feature_engineering"] = list(fe_strategy_obj.get("techniques") or [])

        if isinstance(eval_plan, dict):
            eval_plan_copy = dict(eval_plan)
            eval_plan_copy["feature_engineering"] = list(fe_strategy_obj.get("techniques") or [])
            normalized["evaluation_plan"] = eval_plan_copy

        return normalized

    def _normalize_strategist_output(self, parsed: Any) -> Dict[str, Any]:
        """
        Normalize the LLM output into a stable dictionary structure.
        """
        strategies = []
        if isinstance(parsed, dict):
            raw_strategies = parsed.get("strategies")
            if isinstance(raw_strategies, list):
                # Filter non-dict elements
                strategies = [
                    self._normalize_feature_engineering_aliases(s)
                    for s in raw_strategies
                    if isinstance(s, dict)
                ]
                parsed["strategies"] = strategies
                return parsed
            elif raw_strategies is None:
                # Interpret the dict itself as a single strategy if explicit "strategies" key is missing
                strategies = [self._normalize_feature_engineering_aliases(parsed)]
            elif isinstance(raw_strategies, dict):
                 strategies = [self._normalize_feature_engineering_aliases(raw_strategies)]
            else:
                 strategies = []
            
            # If parsed had 'strategies' but it wasn't a list, we just normalized it into 'strategies'.
            # However, if parsed is the strategy itself, we wrap it.
            if "strategies" not in parsed:
                 return {"strategies": strategies}
            # Update normalized strategies
            parsed["strategies"] = strategies
            return parsed
        
        elif isinstance(parsed, list):
            strategies = [
                self._normalize_feature_engineering_aliases(elem)
                for elem in parsed
                if isinstance(elem, dict)
            ]
            return {"strategies": strategies}
        
        else:
            return {"strategies": []}

    def _clean_json(self, text: str) -> str:
        text = re.sub(r'```json', '', text)
        text = re.sub(r'```', '', text)
        return text.strip()

    def _normalize_column_inventory(self, column_inventory: Any) -> List[str]:
        if not isinstance(column_inventory, list):
            return []
        normalized: List[str] = []
        seen = set()
        for col in column_inventory:
            if not isinstance(col, str):
                continue
            name = col.strip()
            if not name:
                continue
            if name in seen:
                continue
            seen.add(name)
            normalized.append(name)
        return normalized

    def _normalize_strategy_column_list(
        self,
        values: Any,
        allowed_columns: List[str],
    ) -> List[str]:
        allowed_lookup = {
            str(col).strip().lower(): str(col).strip()
            for col in allowed_columns
            if isinstance(col, str) and str(col).strip()
        }
        normalized: List[str] = []
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list):
            return normalized
        for item in values:
            candidate = None
            if isinstance(item, str):
                candidate = item.strip()
            elif isinstance(item, dict):
                maybe_name = item.get("name") or item.get("column")
                if isinstance(maybe_name, str):
                    candidate = maybe_name.strip()
            if not candidate:
                continue
            canonical = allowed_lookup.get(candidate.lower())
            if canonical and canonical not in normalized:
                normalized.append(canonical)
        return normalized

    def _collect_strategy_reasoning_fragments(self, strategy: Dict[str, Any]) -> List[str]:
        fragments: List[str] = []

        def _add(value: Any) -> None:
            if not isinstance(value, str):
                return
            text = value.strip()
            if text:
                fragments.append(text)

        for key in (
            "objective_reasoning",
            "scope_reasoning",
            "validation_rationale",
            "hypothesis",
            "reasoning",
            "expected_lift",
        ):
            _add(strategy.get(key))

        fallback_chain = strategy.get("fallback_chain")
        if isinstance(fallback_chain, list):
            for item in fallback_chain:
                _add(item)

        feasibility = strategy.get("feasibility_analysis")
        if isinstance(feasibility, dict):
            for item in feasibility.values():
                _add(item)

        feature_families = strategy.get("feature_families")
        if isinstance(feature_families, list):
            for entry in feature_families:
                if isinstance(entry, str):
                    _add(entry)
                    continue
                if not isinstance(entry, dict):
                    continue
                for key in ("family", "rationale", "selector_hint", "description", "pattern"):
                    _add(entry.get(key))

        recommended_artifacts = strategy.get("recommended_artifacts")
        if isinstance(recommended_artifacts, list):
            for entry in recommended_artifacts:
                if not isinstance(entry, dict):
                    continue
                _add(entry.get("rationale"))

        return fragments

    def _infer_audit_only_columns_from_reasoning(
        self,
        strategy: Dict[str, Any],
        allowed_columns: List[str],
    ) -> List[str]:
        if not isinstance(strategy, dict) or not allowed_columns:
            return []

        cue_patterns = (
            re.compile(r"\bexclude(?:d|s)?\b", flags=re.IGNORECASE),
            re.compile(r"\bquarantin(?:e|ed|ing)\b", flags=re.IGNORECASE),
            re.compile(r"\baudit[- ]only\b", flags=re.IGNORECASE),
            re.compile(r"\breport(?:ing)?[- ]only\b", flags=re.IGNORECASE),
            re.compile(r"\bstratification variable\b", flags=re.IGNORECASE),
            re.compile(r"\bevaluat(?:e|ed|ion) separately\b", flags=re.IGNORECASE),
            re.compile(r"\boutside the initial model\b", flags=re.IGNORECASE),
            re.compile(r"\bexclude from (?:the )?(?:initial )?model\b", flags=re.IGNORECASE),
            re.compile(r"\bshould be excluded from (?:the )?(?:initial )?model\b", flags=re.IGNORECASE),
            re.compile(r"\bdo not include in (?:the )?(?:initial )?model\b", flags=re.IGNORECASE),
            re.compile(r"\bmust not be model features?\b", flags=re.IGNORECASE),
            re.compile(r"\bshould not be model features?\b", flags=re.IGNORECASE),
            re.compile(r"\bnot as a model feature\b", flags=re.IGNORECASE),
            re.compile(r"\bused only for (?:audit|reporting|stratification)\b", flags=re.IGNORECASE),
        )

        inferred: List[str] = []
        fragments = self._collect_strategy_reasoning_fragments(strategy)
        if not fragments:
            return inferred

        allowed_patterns = []
        for canonical in allowed_columns:
            if not isinstance(canonical, str) or not canonical.strip():
                continue
            tokens = [tok for tok in re.split(r"[_\s\-]+", canonical.lower()) if tok]
            if not tokens:
                continue
            if len(tokens) == 1:
                pattern_text = re.escape(tokens[0])
            else:
                pattern_text = r"[_\s\-]+".join(re.escape(tok) for tok in tokens)
            allowed_patterns.append(
                (
                    canonical,
                    re.compile(
                        rf"(?<![A-Za-z0-9]){pattern_text}(?![A-Za-z0-9])",
                        flags=re.IGNORECASE,
                    ),
                )
            )

        for fragment in fragments:
            for sentence in re.split(r"[\n.;]+", fragment):
                snippet = sentence.strip()
                if not snippet:
                    continue
                if not any(pattern.search(snippet) for pattern in cue_patterns):
                    continue
                for canonical, pattern in allowed_patterns:
                    if pattern.search(snippet) and canonical not in inferred:
                        inferred.append(canonical)

        return inferred

    def _apply_authoritative_strategy_hardening(
        self,
        payload: Dict[str, Any],
        allowed_columns: List[str],
    ) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            return {"strategies": []}
        strategies = payload.get("strategies")
        if not isinstance(strategies, list) or not strategies:
            return payload

        hardened_payload = dict(payload)
        hardened_strategies: List[Dict[str, Any]] = []
        hardening_notes: List[Dict[str, Any]] = []

        for idx, strategy in enumerate(strategies):
            if not isinstance(strategy, dict):
                continue
            strategy_copy = dict(strategy)
            explicit_audit_only = self._normalize_strategy_column_list(
                strategy_copy.get("audit_only_columns"),
                allowed_columns,
            )
            inferred_audit_only = self._infer_audit_only_columns_from_reasoning(
                strategy_copy,
                allowed_columns,
            )
            merged_audit_only: List[str] = []
            for col in explicit_audit_only + inferred_audit_only:
                if col not in merged_audit_only:
                    merged_audit_only.append(col)
            if merged_audit_only:
                strategy_copy["audit_only_columns"] = merged_audit_only
            if inferred_audit_only:
                hardening_notes.append(
                    {
                        "strategy_index": idx,
                        "strategy_title": strategy_copy.get("title", f"strategy_{idx}"),
                        "derived_audit_only_columns": inferred_audit_only,
                    }
                )
            hardened_strategies.append(strategy_copy)

        hardened_payload["strategies"] = hardened_strategies
        if hardening_notes:
            hardened_payload["authoritative_hardening"] = {
                "status": "applied",
                "notes": hardening_notes,
            }
        return hardened_payload

    def _extract_required_column_names(self, strategy: Dict[str, Any]) -> List[str]:
        required_raw = strategy.get("required_columns") if isinstance(strategy, dict) else None
        if not isinstance(required_raw, list):
            return []
        names: List[str] = []
        for item in required_raw:
            name = None
            if isinstance(item, str):
                name = item.strip()
            elif isinstance(item, dict):
                maybe_name = item.get("name") or item.get("column")
                if isinstance(maybe_name, str):
                    name = maybe_name.strip()
            if name:
                names.append(name)
        return list(dict.fromkeys(names))

    def _validate_required_columns(
        self,
        payload: Dict[str, Any],
        allowed_columns: List[str],
        *,
        max_required_columns: Optional[int] = None,
    ) -> Dict[str, Any]:
        strategies = payload.get("strategies") if isinstance(payload, dict) else []
        if not isinstance(strategies, list):
            strategies = []
        if not allowed_columns:
            return {
                "status": "skipped_no_inventory",
                "allowed_column_count": 0,
                "checked_strategy_count": len(strategies),
                "invalid_count": 0,
                "invalid_details": [],
                "max_required_columns": max_required_columns,
                "over_budget_count": 0,
                "over_budget_details": [],
            }
        allowed = set(allowed_columns)
        invalid_details: List[Dict[str, Any]] = []
        invalid_count = 0
        over_budget_details: List[Dict[str, Any]] = []
        over_budget_count = 0
        for idx, strategy in enumerate(strategies):
            if not isinstance(strategy, dict):
                continue
            names = self._extract_required_column_names(strategy)
            invalid = [name for name in names if name not in allowed]
            if invalid:
                invalid_count += len(invalid)
                invalid_details.append(
                    {
                        "strategy_index": idx,
                        "strategy_title": strategy.get("title", f"strategy_{idx}"),
                        "required_count": len(names),
                        "invalid_columns": invalid,
                    }
                )
            if isinstance(max_required_columns, int) and max_required_columns > 0 and len(names) > max_required_columns:
                over_budget_count += 1
                over_budget_details.append(
                    {
                        "strategy_index": idx,
                        "strategy_title": strategy.get("title", f"strategy_{idx}"),
                        "required_count": len(names),
                        "max_required_columns": max_required_columns,
                    }
                )
        if invalid_count > 0:
            status = "invalid_required_columns"
        elif over_budget_count > 0:
            status = "required_columns_over_budget"
        else:
            status = "ok"
        return {
            "status": status,
            "allowed_column_count": len(allowed_columns),
            "checked_strategy_count": len(strategies),
            "invalid_count": invalid_count,
            "invalid_details": invalid_details,
            "max_required_columns": max_required_columns,
            "over_budget_count": over_budget_count,
            "over_budget_details": over_budget_details,
        }

    def _get_column_repair_attempts(self) -> int:
        raw = os.getenv("STRATEGIST_COLUMN_REPAIR_ATTEMPTS", "1")
        try:
            val = int(raw)
        except Exception:
            val = 1
        return max(0, min(val, 3))

    def _build_required_columns_repair_prompt(
        self,
        payload: Dict[str, Any],
        data_summary: str,
        user_request: str,
        allowed_columns: List[str],
        column_sets: Dict[str, Any],
        validation_report: Dict[str, Any],
        max_required_columns: Optional[int] = None,
    ) -> str:
        from src.utils.prompting import render_prompt

        REPAIR_PROMPT_TEMPLATE = """
        You are repairing strategist JSON after validation found required_columns issues.

        Return ONLY raw JSON. No markdown, no comments.
        Keep the same top-level structure: {"strategies":[...]}.
        Preserve strategy meaning, but fix required_columns so every entry is in AUTHORIZED COLUMN INVENTORY.
        Never invent or rename columns.
        If a strategy cannot justify a column from inventory, return [] for required_columns.
        If there is a max_required_columns budget, keep required_columns compact and move broad families to feature_families.

        *** USER REQUEST ***
        "$user_request"

        *** DATASET SUMMARY ***
        $data_summary

        *** AUTHORIZED COLUMN INVENTORY ***
        $authorized_column_inventory

        *** COLUMN SETS ***
        $column_sets

        *** CURRENT STRATEGY JSON ***
        $current_payload

        *** VALIDATION FAILURES ***
        $validation_report

        *** REQUIRED COLUMNS BUDGET ***
        $required_columns_budget_hint
        """
        budget_hint = (
            f"max_required_columns={max_required_columns}"
            if isinstance(max_required_columns, int) and max_required_columns > 0
            else "no_budget"
        )
        return render_prompt(
            REPAIR_PROMPT_TEMPLATE,
            user_request=user_request,
            data_summary=data_summary,
            authorized_column_inventory=json.dumps(allowed_columns, ensure_ascii=False),
            column_sets=json.dumps(column_sets or {}, ensure_ascii=False),
            current_payload=json.dumps(payload, ensure_ascii=False),
            validation_report=json.dumps(validation_report, ensure_ascii=False),
            required_columns_budget_hint=budget_hint,
        )

    def _repair_required_columns_with_llm(
        self,
        payload: Dict[str, Any],
        data_summary: str,
        user_request: str,
        allowed_columns: List[str],
        column_sets: Dict[str, Any],
        validation_report: Dict[str, Any],
        max_attempts: int,
        max_required_columns: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        best_payload = payload
        best_validation = validation_report
        best_invalid = int(validation_report.get("invalid_count", 0))
        best_over_budget = int(validation_report.get("over_budget_count", 0))
        attempts_used = 0
        repaired = False
        for _ in range(max_attempts):
            if best_invalid <= 0 and best_over_budget <= 0:
                break
            attempts_used += 1
            repair_prompt = self._build_required_columns_repair_prompt(
                payload=best_payload,
                data_summary=data_summary,
                user_request=user_request,
                allowed_columns=allowed_columns,
                column_sets=column_sets,
                validation_report=best_validation,
                max_required_columns=max_required_columns,
            )
            self.last_repair_prompt = repair_prompt
            try:
                repaired_content = self._call_model(
                    repair_prompt,
                    temperature=0.1,
                    context_tag="strategist_repair",
                )
                self.last_repair_response = repaired_content
                repaired_raw = self._clean_json(repaired_content)
                repaired_parsed = json.loads(repaired_raw)
                candidate = self._normalize_strategist_output(repaired_parsed)
                candidate = self._apply_authoritative_strategy_hardening(candidate, allowed_columns)
                candidate_validation = self._validate_required_columns(
                    candidate,
                    allowed_columns,
                    max_required_columns=max_required_columns,
                )
                candidate_invalid = int(candidate_validation.get("invalid_count", 0))
                candidate_over_budget = int(candidate_validation.get("over_budget_count", 0))
                current_score = best_invalid * 1000 + best_over_budget
                candidate_score = candidate_invalid * 1000 + candidate_over_budget
                if candidate_score < current_score:
                    best_payload = candidate
                    best_validation = candidate_validation
                    best_invalid = candidate_invalid
                    best_over_budget = candidate_over_budget
                    repaired = True
                if candidate_invalid == 0 and candidate_over_budget == 0:
                    break
            except Exception as repair_err:
                print(f"Strategist required_columns repair error: {repair_err}")
                continue
        best_validation = dict(best_validation or {})
        best_validation["repair_attempts_used"] = attempts_used
        best_validation["repair_applied"] = repaired
        return best_payload, best_validation

    def _infer_scope_recommendation(self, primary: Dict[str, Any], user_request: str) -> str:
        if not isinstance(primary, dict):
            primary = {}
        scope_raw = str(primary.get("scope_recommendation") or "").strip().lower()
        if scope_raw in {"cleaning_only", "ml_only", "full_pipeline"}:
            return scope_raw

        combined_parts = [
            str(user_request or ""),
            str(primary.get("objective_reasoning") or ""),
            str(primary.get("scope_reasoning") or ""),
            str(primary.get("reasoning") or ""),
            str(primary.get("analysis_type") or ""),
        ]
        techniques = primary.get("techniques")
        if isinstance(techniques, list):
            combined_parts.extend(str(item or "") for item in techniques)
        combined = " ".join(combined_parts).lower()

        has_cleaning_signal = any(
            tok in combined
            for tok in (
                "clean",
                "quality",
                "deduplic",
                "standardiz",
                "normaliz",
                "leakage",
                "audit",
                "etl",
                "missing values",
            )
        )
        has_training_signal = any(
            tok in combined
            for tok in (
                "train",
                "classifier",
                "regression",
                "forecast",
                "predictive model",
                "cross-validation",
                "roc_auc",
                "f1",
                "precision",
                "recall",
            )
        )
        if has_cleaning_signal and not has_training_signal:
            return "cleaning_only"
        if has_training_signal and not has_cleaning_signal:
            return "ml_only"
        if has_training_signal and has_cleaning_signal:
            return "full_pipeline"
        return "full_pipeline"

    def _default_metrics_for_scope(self, scope: str, objective_type: str) -> List[str]:
        scope_norm = str(scope or "").strip().lower()
        if scope_norm == "cleaning_only":
            return [
                "data_quality_pass_rate",
                "retained_rows_after_cleaning",
                "transformation_traceability_completeness",
            ]
        if scope_norm == "ml_only":
            return [
                "primary_business_metric",
                "generalization_metric",
                "decision_readiness",
            ]
        return [
            "data_quality_pass_rate",
            "primary_business_metric",
            "decision_readiness",
        ]

    def _default_validation_for_scope(self, scope: str, objective_type: str) -> Tuple[str, str]:
        scope_norm = str(scope or "").strip().lower()
        if scope_norm == "cleaning_only":
            return (
                "data_quality_validation_with_rule_based_checks",
                "This run is focused on cleaning and preparation, so validation should emphasize data quality, leakage prevention, and traceability rather than model performance.",
            )
        if scope_norm == "ml_only":
            return (
                "data_structure_driven_validation_requires_explicit_selection",
                "Validation should be chosen from the current objective, data structure, leakage risk, and compute budget. Treat this as a placeholder that still needs explicit downstream specification.",
            )
        return (
            "staged_validation_with_cleaning_checks_and_model_evaluation",
            "A full pipeline run must validate both data preparation integrity and downstream model credibility without mixing those judgments.",
        )

    def _default_artifacts_for_scope(self, scope: str) -> List[Dict[str, Any]]:
        scope_norm = str(scope or "").strip().lower()
        if scope_norm == "cleaning_only":
            return [
                {"artifact_type": "clean_dataset", "required": True, "rationale": "Primary cleaned dataset for direct use or future modeling."},
                {"artifact_type": "enriched_dataset", "required": True, "rationale": "Feature-enriched dataset when this run prepares future modeling handoff."},
                {"artifact_type": "data_dictionary", "required": True, "rationale": "Traceability for column semantics, types, and transformations."},
                {"artifact_type": "decision_log", "required": True, "rationale": "Audit trail for exclusions, imputation, and deduplication decisions."},
                {"artifact_type": "diagnostics", "required": False, "rationale": "Optional quality diagnostics or issue summaries."},
            ]
        if scope_norm == "ml_only":
            return [
                {"artifact_type": "metrics", "required": True, "rationale": "Evaluation results for the trained or validated model."},
                {"artifact_type": "predictions_or_scores", "required": True, "rationale": "Primary modeling output for ranking, scoring, or prediction."},
                {"artifact_type": "diagnostics", "required": False, "rationale": "Optional model diagnostics or error analysis."},
                {"artifact_type": "explainability", "required": False, "rationale": "Optional explanations when interpretability matters."},
            ]
        return [
            {"artifact_type": "clean_dataset", "required": True, "rationale": "Prepared dataset required before downstream modeling."},
            {"artifact_type": "metrics", "required": True, "rationale": "Evaluation evidence for the modeling stage."},
            {"artifact_type": "predictions_or_scores", "required": True, "rationale": "Primary output when the run includes predictive or scoring work."},
            {"artifact_type": "diagnostics", "required": False, "rationale": "Optional diagnostics spanning preparation and modeling."},
            {"artifact_type": "explainability", "required": False, "rationale": "Optional explanations when needed by the business or reviewers."},
        ]

    def _build_strategy_spec_from_llm(self, strategy_payload: Dict[str, Any], data_summary: str, user_request: str) -> Dict[str, Any]:
        """
        Build strategy_spec using LLM-generated reasoning instead of hardcoded inference.

        The LLM now provides:
        - objective_type (reasoned, not inferred)
        - objective_reasoning (explicit connection to business goal)
        - success_metric (business metric, not generic ML metric)
        - recommended_evaluation_metrics (reasoned, not mapped from dict)
        - validation_strategy (reasoned, not mapped from dict)
        """
        strategies = []
        if isinstance(strategy_payload, dict):
            strategies = strategy_payload.get("strategies", []) or []
        primary = strategies[0] if strategies else {}
        if not isinstance(primary, dict):
            primary = {}

        scope_recommendation = self._infer_scope_recommendation(primary, user_request)

        # Preserve the LLM judgment when present; otherwise keep the objective open
        # instead of silently forcing the strategy into a predictive framing.
        objective_type = str(primary.get("objective_type") or "").strip().lower()
        if not objective_type:
            # LLM failed to produce an objective_type — default to predictive as the
            objective_type = "unspecified"

        # Use LLM-generated metrics (with scope-aware defaults only if missing)
        metrics = primary.get("recommended_evaluation_metrics", [])
        if not metrics:
            metrics = self._default_metrics_for_scope(scope_recommendation, objective_type)

        validation_strategy = str(primary.get("validation_strategy") or "").strip()
        validation_rationale = str(primary.get("validation_rationale") or "").strip()
        if not validation_strategy or not validation_rationale:
            default_validation_strategy, default_validation_rationale = self._default_validation_for_scope(
                scope_recommendation,
                objective_type,
            )
            if not validation_strategy:
                validation_strategy = default_validation_strategy
            if not validation_rationale:
                validation_rationale = default_validation_rationale

        target_columns: List[str] = []

        def _add_targets(values: Any) -> None:
            if isinstance(values, str):
                candidate = values.strip()
                if candidate and candidate not in target_columns:
                    target_columns.append(candidate)
                return
            if isinstance(values, list):
                for item in values:
                    if not isinstance(item, str):
                        continue
                    candidate = item.strip()
                    if candidate and candidate not in target_columns:
                        target_columns.append(candidate)

        _add_targets(primary.get("target_columns"))
        _add_targets(primary.get("target_column"))
        _add_targets(primary.get("outcome_columns"))
        _add_targets(primary.get("outcome_column"))
        _add_targets(primary.get("label_columns"))
        _add_targets(primary.get("label_column"))

        if not target_columns:
            summary_text = str(data_summary or "")
            for pattern in (
                r'"primary_target"\s*:\s*"([^"]+)"',
                r"'primary_target'\s*:\s*'([^']+)'",
                r"\bprimary_target\s*[:=]\s*[\"']?([A-Za-z0-9_.-]+)",
                r"\btarget_column\s*[:=]\s*[\"']?([A-Za-z0-9_.-]+)",
            ):
                match = re.search(pattern, summary_text, flags=re.IGNORECASE)
                if not match:
                    continue
                candidate = str(match.group(1) or "").strip()
                if candidate and candidate not in target_columns:
                    target_columns.append(candidate)
                    break

        if not target_columns:
            required_names = self._extract_required_column_names(primary if isinstance(primary, dict) else {})
            for name in required_names:
                if re.search(r"(?i)(^target$|^label$|^y$|target|label|outcome)", str(name)):
                    target_columns.append(str(name))
                    break

        # Extract new context-aware fields from LLM output
        feasibility_analysis = primary.get("feasibility_analysis", {})
        if not feasibility_analysis:
            # Provide sensible defaults if LLM didn't generate
            feasibility_analysis = {
                "statistical_power": "Not assessed - using default assumptions",
                "signal_quality": "Not assessed - using default assumptions",
                "compute_value_tradeoff": "Not assessed - using default assumptions",
            }

        fallback_chain = primary.get("fallback_chain", [])
        if not fallback_chain:
            if scope_recommendation == "cleaning_only":
                fallback_chain = [
                    "Primary cleaning and traceability pipeline",
                    "Reduced cleaning pipeline with explicit unresolved issues",
                    "Descriptive audit only with blocked handoff",
                ]
            elif scope_recommendation == "ml_only":
                fallback_chain = [
                    "Primary modeling approach",
                    "Simpler baseline model",
                    "Minimal safe benchmark",
                ]
            else:
                fallback_chain = [
                    "Primary full pipeline",
                    "Reduced-risk baseline pipeline",
                    "Cleaning-first handoff with deferred modeling",
                ]

        expected_lift = primary.get("expected_lift", "Not quantified - baseline comparison recommended")

        # Extract context-aware Feature Engineering strategy (support naming aliases).
        feature_engineering_strategy = primary.get("feature_engineering_strategy", [])
        if isinstance(feature_engineering_strategy, dict):
            techniques = feature_engineering_strategy.get("techniques")
            feature_engineering_strategy = techniques if isinstance(techniques, list) else []
        if not isinstance(feature_engineering_strategy, list) or not feature_engineering_strategy:
            candidate = primary.get("feature_engineering")
            if isinstance(candidate, list):
                feature_engineering_strategy = candidate
        if not isinstance(feature_engineering_strategy, list) or not feature_engineering_strategy:
            eval_plan_primary = primary.get("evaluation_plan")
            if isinstance(eval_plan_primary, dict) and isinstance(eval_plan_primary.get("feature_engineering"), list):
                feature_engineering_strategy = eval_plan_primary.get("feature_engineering")
        if not isinstance(feature_engineering_strategy, list):
            feature_engineering_strategy = []

        evaluation_plan = {
            "objective_type": objective_type,
            "metrics": metrics,
            "target_columns": target_columns,
            "validation": {
                "strategy": validation_strategy,
                "rationale": validation_rationale,
            },
            "feature_engineering": feature_engineering_strategy,
            "feasibility": feasibility_analysis,
            "fallback_chain": fallback_chain,
            "expected_lift": expected_lift,
        }

        # Keep simple leakage heuristics (universal)
        leakage_risks: List[str] = []
        combined = " ".join([str(data_summary or "").lower(), str(user_request or "").lower()])
        if any(tok in combined for tok in ["post", "after", "outcome", "result"]):
            leakage_risks.append("Potential post-outcome fields may leak target information.")
        if "target" in combined:
            leakage_risks.append("Exclude target or target-derived fields from features.")

        recommended_artifacts = primary.get("recommended_artifacts", [])
        if not isinstance(recommended_artifacts, list) or not recommended_artifacts:
            recommended_artifacts = self._default_artifacts_for_scope(scope_recommendation)

        return {
            "objective_type": objective_type,
            "scope_recommendation": scope_recommendation,
            "target_columns": target_columns,
            "feature_engineering": feature_engineering_strategy,
            "evaluation_plan": evaluation_plan,
            "leakage_risks": leakage_risks,
            "recommended_artifacts": recommended_artifacts,
        }
