import json
import os
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from openai import OpenAI

from src.utils.senior_protocol import SENIOR_STRATEGY_PROTOCOL

# domain_knowledge import removed (seniority refactoring):
# Domain guidance is now inferred by the LLM from data context and business objective.

load_dotenv()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _to_list_of_str(values: Any, max_items: int = 12) -> List[str]:
    if not isinstance(values, list):
        return []
    out: List[str] = []
    for item in values:
        text = str(item or "").strip()
        if not text:
            continue
        out.append(text)
        if len(out) >= max_items:
            break
    return out


class DomainExpertAgent:
    def __init__(self, api_key: str = None):
        """
        Domain Expert with LLM + deterministic validation/scoring fallback.
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is required for Domain Expert.")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        self.model_name = os.getenv("DOMAIN_EXPERT_MODEL", "google/gemini-3-flash-preview")
        self.last_prompt = None
        self.last_response = None

    def evaluate_strategies(
        self,
        data_summary: str,
        business_objective: str,
        strategies: List[Dict[str, Any]],
        *,
        compute_constraints: Dict[str, Any] | None = None,
        dataset_memory_context: str = "",
    ) -> Dict[str, Any]:
        """
        Evaluate strategies with LLM, then enforce deterministic validation.
        If LLM fails or output is weak, deterministic scoring fills gaps.
        """
        normalized_strategies = self._normalize_strategies(strategies)
        llm_error = ""
        llm_reviews: List[Dict[str, Any]] = []

        prompt = self._build_prompt(
            data_summary=data_summary,
            business_objective=business_objective,
            strategies=normalized_strategies,
            compute_constraints=compute_constraints or {},
            dataset_memory_context=dataset_memory_context,
        )
        self.last_prompt = prompt

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            content = (response.choices[0].message.content or "").strip()
            self.last_response = content
            parsed = self._parse_json_response(content)
            raw_reviews = parsed.get("reviews") if isinstance(parsed, dict) else []
            if isinstance(raw_reviews, list):
                llm_reviews = raw_reviews
        except Exception as err:
            llm_error = str(err)
            self.last_response = f"DOMAIN_EXPERT_ERROR: {llm_error}"

        deterministic_reviews = self._score_deterministic(
            strategies=normalized_strategies,
            business_objective=business_objective,
            compute_constraints=compute_constraints or {},
        )
        validated_reviews, validation_meta = self._validate_reviews(
            llm_reviews=llm_reviews,
            deterministic_reviews=deterministic_reviews,
            strategy_count=len(normalized_strategies),
        )

        if llm_error:
            validation_meta["llm_error"] = llm_error

        return {
            "reviews": validated_reviews,
            "review_validation": validation_meta,
        }

    def _build_prompt(
        self,
        *,
        data_summary: str,
        business_objective: str,
        strategies: List[Dict[str, Any]],
        compute_constraints: Dict[str, Any],
        dataset_memory_context: str,
    ) -> str:
        from src.utils.prompting import render_prompt

        SYSTEM_PROMPT_TEMPLATE = """
You are a Senior Domain Expert and Business Strategy Reviewer.
Your goal is to select the most BUSINESS-VALUABLE and EXECUTABLE strategy.

=== SENIOR STRATEGY PROTOCOL ===
$senior_strategy_protocol

*** BUSINESS OBJECTIVE ***
"$business_objective"

*** DATA CONTEXT ***
$data_summary

*** COMPUTE CONSTRAINTS ***
$compute_constraints

*** DATASET MEMORY CONTEXT (PRIOR RUNS) ***
$dataset_memory_context

*** DOMAIN GUIDANCE ***
Infer domain-specific best practices and risks from the data context and business objective.
Do not rely on pre-defined domain templates. Reason about the specific domain, data characteristics,
and business goals to identify relevant risks, best practices, and evaluation criteria.

*** CANDIDATE STRATEGIES (INDEXED) ***
$strategies_text

TASK:
Score EACH strategy from 0 to 10. Use these criteria:
1) Business Alignment
2) Technical Feasibility (data + compute constraints)
3) Implementability by DE/ML agents
4) Risk Assessment (leakage, unrealistic assumptions, missing requirements)

Rules:
- Evaluate all strategies by strategy_index.
- If a strategy includes fallback_chain, feasibility_analysis, and valid required_columns, reward it.
- If strategy metadata indicates invalid required columns or compute infeasibility, penalize.
- If major risks exist, score must reflect that (avoid high score with critical risks).
- Return strict JSON only.

Output schema:
{
  "reviews": [
    {
      "strategy_index": 0,
      "title": "Strategy Title",
      "score": 8.1,
      "reasoning": "Why this score is justified",
      "risks": ["risk 1", "risk 2"],
      "recommendation": "Actionable recommendation"
    }
  ]
}
"""

        return render_prompt(
            SYSTEM_PROMPT_TEMPLATE,
            senior_strategy_protocol=SENIOR_STRATEGY_PROTOCOL,
            business_objective=business_objective,
            data_summary=data_summary or "",
            compute_constraints=json.dumps(compute_constraints or {}, ensure_ascii=True),
            dataset_memory_context=(dataset_memory_context or "")[:3000],
            strategies_text=json.dumps(strategies, ensure_ascii=True, indent=2),
        )

    def _normalize_strategies(self, strategies: Any) -> List[Dict[str, Any]]:
        if not isinstance(strategies, list):
            return []
        out: List[Dict[str, Any]] = []
        for idx, item in enumerate(strategies):
            if not isinstance(item, dict):
                continue
            cloned = dict(item)
            cloned["_strategy_index"] = idx
            cloned["_title"] = str(item.get("title") or f"strategy_{idx}")
            out.append(cloned)
        return out

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        cleaned = self._clean_json(text or "")
        if not cleaned:
            return {}
        try:
            parsed = json.loads(cleaned)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                snippet = cleaned[start : end + 1]
                try:
                    parsed = json.loads(snippet)
                    return parsed if isinstance(parsed, dict) else {}
                except Exception:
                    return {}
        return {}

    def _score_deterministic(
        self,
        *,
        strategies: List[Dict[str, Any]],
        business_objective: str,
        compute_constraints: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        reviews: List[Dict[str, Any]] = []
        objective_tokens = set(re.findall(r"[a-zA-Z]{4,}", str(business_objective or "").lower()))
        max_runtime = int(_safe_float((compute_constraints or {}).get("max_runtime_seconds"), 0.0))
        for strat in strategies:
            idx = int(strat.get("_strategy_index") or 0)
            title = str(strat.get("_title") or f"strategy_{idx}")
            score = 0.0
            risks: List[str] = []
            reasons: List[str] = []

            fallback_chain = strat.get("fallback_chain")
            has_fallback = isinstance(fallback_chain, list) and len(fallback_chain) >= 2
            if has_fallback:
                score += 2.0
                reasons.append("Has fallback chain for graceful degradation.")
            else:
                risks.append("No explicit fallback_chain.")

            feasibility = strat.get("feasibility_analysis")
            has_feasibility = isinstance(feasibility, dict) and any(str(v).strip() for v in feasibility.values())
            if has_feasibility:
                score += 2.0
                reasons.append("Includes feasibility analysis.")
            else:
                risks.append("Missing feasibility analysis.")

            meta = strat.get("_meta") if isinstance(strat.get("_meta"), dict) else {}
            col_meta = meta.get("column_validation") if isinstance(meta.get("column_validation"), dict) else {}
            invalid_count = int(col_meta.get("invalid_count") or 0)
            over_budget_count = int(col_meta.get("over_budget_count") or 0)
            if invalid_count == 0 and over_budget_count == 0:
                score += 3.0
                reasons.append("required_columns validation passed.")
            else:
                risks.append("required_columns has invalid or over-budget entries.")

            techniques = strat.get("techniques")
            if isinstance(techniques, list) and techniques:
                score += 1.0
                reasons.append("Technique stack is explicit.")
            else:
                risks.append("No explicit techniques.")

            alignment_blob = " ".join(
                [
                    str(strat.get("objective_reasoning") or ""),
                    str(strat.get("reasoning") or ""),
                    str(strat.get("hypothesis") or ""),
                    str(strat.get("analysis_type") or ""),
                    str(title),
                ]
            ).lower()
            overlap = len([tok for tok in objective_tokens if tok in alignment_blob])
            if overlap > 0:
                score += min(2.0, 0.4 * overlap)
                reasons.append("Reasoning aligns with business objective vocabulary.")
            else:
                risks.append("Weak explicit linkage to business objective.")

            difficulty = str(strat.get("estimated_difficulty") or "").lower()
            if max_runtime > 0 and difficulty == "high" and max_runtime < 1200:
                score -= 1.5
                risks.append("High difficulty may exceed runtime budget.")

            score = max(0.0, min(10.0, round(score, 2)))
            reasoning = "; ".join(reasons) if reasons else "Deterministic baseline score from structure and feasibility."
            recommendation = "Proceed" if score >= 6.0 else "Revise strategy before execution."
            reviews.append(
                {
                    "strategy_index": idx,
                    "title": title,
                    "score": score,
                    "reasoning": reasoning,
                    "risks": risks[:8],
                    "recommendation": recommendation,
                    "score_source": "deterministic_fallback",
                }
            )
        return reviews

    def _validate_reviews(
        self,
        *,
        llm_reviews: List[Dict[str, Any]],
        deterministic_reviews: List[Dict[str, Any]],
        strategy_count: int,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        by_index: Dict[int, Dict[str, Any]] = {}
        by_title: Dict[str, Dict[str, Any]] = {}
        for item in deterministic_reviews:
            idx = int(item.get("strategy_index") or 0)
            by_index[idx] = item
            by_title[str(item.get("title") or "").lower()] = item

        normalized_llm: List[Dict[str, Any]] = []
        for raw in llm_reviews if isinstance(llm_reviews, list) else []:
            if not isinstance(raw, dict):
                continue
            idx = raw.get("strategy_index")
            idx_int = None
            try:
                if idx is not None:
                    idx_int = int(idx)
            except Exception:
                idx_int = None
            title = str(raw.get("title") or "").strip()
            score = _safe_float(raw.get("score"), 0.0)
            score = max(0.0, min(10.0, round(score, 2)))
            reasoning = str(raw.get("reasoning") or "").strip()
            risks = _to_list_of_str(raw.get("risks"), max_items=10)
            recommendation = str(raw.get("recommendation") or "").strip()
            if idx_int is None and title:
                # Fuzzy title mapping fallback
                title_lower = title.lower()
                best_title = ""
                best_ratio = 0.0
                for known in by_title.keys():
                    ratio = SequenceMatcher(None, known, title_lower).ratio()
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_title = known
                if best_title and best_ratio >= 0.7:
                    idx_int = int(by_title[best_title].get("strategy_index") or 0)
            if idx_int is None:
                continue

            # Consistency guard: high score + critical risks is incoherent.
            critical_risk = any(
                tok in " ".join(risks).lower()
                for tok in ("critical", "leakage", "not feasible", "infeasible", "invalid required")
            )
            if critical_risk and score >= 8.0:
                score = 5.5
                reasoning = (
                    f"{reasoning} Score adjusted by consistency guard due to critical risks."
                    if reasoning
                    else "Score adjusted by consistency guard due to critical risks."
                )
            normalized_llm.append(
                {
                    "strategy_index": idx_int,
                    "title": title or str(by_index.get(idx_int, {}).get("title") or f"strategy_{idx_int}"),
                    "score": score,
                    "reasoning": reasoning,
                    "risks": risks,
                    "recommendation": recommendation or "Review constraints before execution.",
                    "score_source": "llm",
                }
            )

        # LLM-primary: use LLM reviews when available, deterministic only as safety net.
        llm_by_idx: Dict[int, Dict[str, Any]] = {}
        for item in normalized_llm:
            idx = int(item.get("strategy_index") or 0)
            if 0 <= idx < max(0, strategy_count):
                llm_by_idx[idx] = item

        det_by_idx: Dict[int, Dict[str, Any]] = {int(r["strategy_index"]): dict(r) for r in deterministic_reviews}

        reviews: List[Dict[str, Any]] = []
        llm_used = 0
        for idx in range(max(0, strategy_count)):
            if idx in llm_by_idx:
                base = llm_by_idx[idx]
                llm_used += 1
            elif idx in det_by_idx:
                base = det_by_idx[idx]
            else:
                continue
            # Selectability policy requested by audit.
            selectable = _safe_float(base.get("score"), 0.0) >= 3.0
            base["selectable"] = bool(selectable)
            reviews.append(base)

        return reviews, {
            "status": "ok" if len(reviews) == strategy_count else "partial",
            "strategy_count": strategy_count,
            "reviews_count": len(reviews),
            "llm_reviews_received": len(normalized_llm),
            "llm_reviews_used": llm_used,
            "fallback_reviews_used": max(0, len(reviews) - llm_used),
            "minimum_selectable_score": 3.0,
        }

    def evaluate_deterministic(
        self,
        data_summary: str,
        business_objective: str,
        strategies: List[Dict[str, Any]],
        *,
        compute_constraints: Dict[str, Any] | None = None,
        dataset_memory_context: str = "",
    ) -> Dict[str, Any]:
        """
        Deterministic-only strategy evaluation — no LLM call.

        Scores strategies using structural/feasibility heuristics
        (_score_deterministic), then applies the same consistency guard
        and selectability policy as the LLM path.

        Returns the same shape as evaluate_strategies() for full
        downstream compatibility.
        """
        normalized_strategies = self._normalize_strategies(strategies)
        deterministic_reviews = self._score_deterministic(
            strategies=normalized_strategies,
            business_objective=business_objective,
            compute_constraints=compute_constraints or {},
        )

        # Apply consistency guard (critical risk + high score → cap)
        for review in deterministic_reviews:
            risks = review.get("risks") or []
            score = _safe_float(review.get("score"), 0.0)
            critical_risk = any(
                tok in " ".join(risks).lower()
                for tok in ("critical", "leakage", "not feasible", "infeasible", "invalid required")
            )
            if critical_risk and score >= 8.0:
                review["score"] = 5.5
                review["reasoning"] = (
                    f"{review.get('reasoning', '')} Score adjusted by consistency guard due to critical risks."
                )

            # Selectability policy: score >= 3.0
            review["selectable"] = bool(_safe_float(review.get("score"), 0.0) >= 3.0)

        strategy_count = len(normalized_strategies)
        return {
            "reviews": deterministic_reviews,
            "review_validation": {
                "status": "ok" if len(deterministic_reviews) == strategy_count else "partial",
                "strategy_count": strategy_count,
                "reviews_count": len(deterministic_reviews),
                "llm_reviews_received": 0,
                "llm_reviews_used": 0,
                "fallback_reviews_used": len(deterministic_reviews),
                "minimum_selectable_score": 3.0,
                "mode": "deterministic",
            },
        }

    def _clean_json(self, text: str) -> str:
        text = re.sub(r"```json", "", text)
        text = re.sub(r"```", "", text)
        return text.strip()
