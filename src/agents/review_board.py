import json
from typing import Any, Dict, List

from src.utils.reviewer_llm import init_reviewer_llm
from src.utils.senior_protocol import SENIOR_EVIDENCE_RULE


class ReviewBoardAgent:
    """
    Final adjudicator for reviewer outputs.
    Consolidates Reviewer, QA Reviewer, and Results Advisor findings.
    """

    def __init__(self, api_key: str = None):
        self.provider, self.client, self.model_name, self.model_warning = init_reviewer_llm(api_key)
        if self.model_warning:
            print(f"WARNING: {self.model_warning}")
        self.last_prompt = None
        self.last_response = None

    def adjudicate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        context = context or {}
        if not self.client or self.provider == "none":
            return self._fallback(context)

        output_format = """
        Return a raw JSON object:
        {
          "status": "APPROVED" | "APPROVE_WITH_WARNINGS" | "NEEDS_IMPROVEMENT" | "REJECTED",
          "summary": "Single concise paragraph.",
          "failed_areas": ["reviewer_alignment", "qa_gates", "results_quality"],
          "required_actions": ["action 1", "action 2"],
          "confidence": "high" | "medium" | "low",
          "evidence": [
            {"claim": "Short claim", "source": "artifact_path#key_or_script_path:line or missing"}
          ]
        }
        """
        system_prompt = (
            "You are the Review Board for a multi-agent ML system.\n"
            "Your job is to issue the final verdict using the evidence from:\n"
            "- Reviewer (strategy/contract/code alignment)\n"
            "- QA Reviewer (universal + contract QA gates)\n"
            "- Results Advisor (quality and improvement potential)\n\n"
            "If REVIEW_STACK_CONTEXT contains deterministic_facts, treat those facts as canonical for\n"
            "numeric values, metric thresholds, and artifact completeness.\n"
            "If reviewer text conflicts with deterministic_facts, explicitly flag it in failed_areas and required_actions.\n\n"
            "=== EVIDENCE RULE ===\n"
            f"{SENIOR_EVIDENCE_RULE}\n\n"
            "Decision policy:\n"
            "1) If unresolved hard failures exist (runtime hard blockers, QA hard failures, deterministic blockers), return REJECTED.\n"
            "2) If hard blockers are absent but contract/spec fixes are required, return NEEDS_IMPROVEMENT.\n"
            "3) If only advisory/optimization items remain, return APPROVE_WITH_WARNINGS.\n"
            "4) If all critical areas pass, return APPROVED.\n"
            "5) Use progress_tracker when available: if performance regressed and blockers remain unresolved, avoid optimistic approval.\n"
            "6) Use iteration_history when available to detect progress trends, plateaus, or regressions across iterations.\n"
            "   If metrics have plateaued for 2+ iterations with no improvement, flag it in required_actions.\n"
            "Do not invent evidence.\n\n"
            "Evidence policy:\n"
            "- Prefer deterministic_facts entries when available.\n"
            "- Cite source keys from deterministic_facts or reviewer packets in every critical claim.\n\n"
            "Output format:\n"
            f"{output_format}"
        )
        user_prompt = "REVIEW_STACK_CONTEXT:\n" + json.dumps(context, ensure_ascii=True, indent=2)
        self.last_prompt = system_prompt + "\n\n" + user_prompt

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            content = response.choices[0].message.content
            self.last_response = content
            parsed = json.loads(self._clean_json(content))
            normalized = self._normalize(parsed, context)
            normalized = self._apply_conflict_reconciliation(normalized, context)
            return self._apply_progress_policy(normalized, context)
        except Exception:
            return self._fallback(context)

    def _collect_required_actions(self, payload: Dict[str, Any]) -> List[str]:
        actions: List[str] = []
        if not isinstance(payload, dict):
            return actions
        for key in ("required_actions", "required_fixes"):
            values = payload.get(key)
            if not isinstance(values, list):
                continue
            for item in values:
                text = str(item).strip()
                if text and text not in actions:
                    actions.append(text)
                if len(actions) >= 20:
                    return actions
        return actions

    def _fallback(self, context: Dict[str, Any]) -> Dict[str, Any]:
        qa = context.get("qa_reviewer") if isinstance(context.get("qa_reviewer"), dict) else {}
        reviewer = context.get("reviewer") if isinstance(context.get("reviewer"), dict) else {}
        evaluator = context.get("result_evaluator") if isinstance(context.get("result_evaluator"), dict) else {}
        runtime = context.get("runtime") if isinstance(context.get("runtime"), dict) else {}
        progress = context.get("progress_tracker") if isinstance(context.get("progress_tracker"), dict) else {}

        statuses = [
            str(qa.get("status", "")).upper(),
            str(reviewer.get("status", "")).upper(),
            str(evaluator.get("status", "")).upper(),
        ]
        hard_failures = []
        for payload in (qa, reviewer, evaluator):
            hf = payload.get("hard_failures")
            if isinstance(hf, list):
                hard_failures.extend(str(x) for x in hf if x)

        reviewer_status = str(reviewer.get("status", "")).upper()
        qa_status = str(qa.get("status", "")).upper()
        status_conflict = (
            (reviewer_status in {"APPROVED", "APPROVE_WITH_WARNINGS"} and qa_status in {"REJECTED", "NEEDS_IMPROVEMENT"})
            or (qa_status in {"APPROVED", "APPROVE_WITH_WARNINGS"} and reviewer_status in {"REJECTED", "NEEDS_IMPROVEMENT"})
        )

        required_actions: List[str] = []
        for payload in (reviewer, qa, evaluator):
            for action in self._collect_required_actions(payload):
                if action not in required_actions:
                    required_actions.append(action)
                if len(required_actions) >= 20:
                    break
            if len(required_actions) >= 20:
                break

        if hard_failures:
            status = "REJECTED" if runtime.get("runtime_fix_terminal") else "NEEDS_IMPROVEMENT"
        elif "NEEDS_IMPROVEMENT" in statuses or "REJECTED" in statuses:
            status = "NEEDS_IMPROVEMENT"
        elif "APPROVE_WITH_WARNINGS" in statuses:
            status = "APPROVE_WITH_WARNINGS"
        else:
            status = "APPROVED"

        failed_areas: List[str] = []
        if hard_failures:
            failed_areas.append("qa_gates")
        if status_conflict:
            failed_areas.append("cross_reviewer_conflict")
        if status in {"NEEDS_IMPROVEMENT", "REJECTED"} and "results_quality" not in failed_areas:
            failed_areas.append("results_quality")
        if isinstance(progress, dict) and progress.get("available") and progress.get("improved") is False:
            if "progress_regression" not in failed_areas:
                failed_areas.append("progress_regression")

        if status in {"NEEDS_IMPROVEMENT", "REJECTED"} and not required_actions:
            required_actions.append("Apply reviewer-required fixes and rerun.")
        if isinstance(progress, dict) and progress.get("available") and progress.get("improved") is False:
            action = (
                "Investigate regression in primary metric and keep previous successful artifacts/logic stable while patching blockers."
            )
            if action not in required_actions:
                required_actions.append(action)

        # Check iteration_history for plateau detection
        iteration_history = context.get("iteration_history") if isinstance(context.get("iteration_history"), list) else []
        if len(iteration_history) >= 2:
            recent = iteration_history[-2:]
            metrics = [h.get("primary_metric") for h in recent if isinstance(h, dict) and h.get("primary_metric") is not None]
            if len(metrics) == 2:
                try:
                    if abs(float(metrics[1]) - float(metrics[0])) < 1e-6:
                        if "metric_plateau" not in failed_areas:
                            failed_areas.append("metric_plateau")
                        plateau_action = "Metrics have plateaued across iterations. Consider changing approach or hyperparameters."
                        if plateau_action not in required_actions:
                            required_actions.append(plateau_action)
                except (ValueError, TypeError):
                    pass

        summary = "Fallback board verdict from reviewer packets."
        if status_conflict:
            summary += " Conflict detected between reviewer and qa_reviewer verdicts."
        if isinstance(progress, dict) and progress.get("available"):
            metric_name = str(progress.get("metric_name") or "metric")
            delta = progress.get("delta")
            try:
                delta_txt = f"{float(delta):.6g}"
            except Exception:
                delta_txt = str(delta)
            summary += f" Progress tracker: {metric_name} delta={delta_txt}."
        if iteration_history:
            summary += f" Iteration history available ({len(iteration_history)} prior iterations)."

        return {
            "status": status,
            "summary": summary,
            "failed_areas": failed_areas,
            "required_actions": required_actions,
            "confidence": "medium",
            "evidence": [],
        }

    def _normalize(self, payload: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        status = str(payload.get("status", "")).strip().upper()
        if status not in {"APPROVED", "APPROVE_WITH_WARNINGS", "NEEDS_IMPROVEMENT", "REJECTED"}:
            status = self._fallback(context)["status"]
        failed_areas = payload.get("failed_areas")
        required_actions = payload.get("required_actions")
        evidence = payload.get("evidence")
        return {
            "status": status,
            "summary": str(payload.get("summary", "")).strip(),
            "failed_areas": [str(x) for x in failed_areas] if isinstance(failed_areas, list) else [],
            "required_actions": [str(x) for x in required_actions] if isinstance(required_actions, list) else [],
            "confidence": str(payload.get("confidence", "medium")).lower() if payload.get("confidence") else "medium",
            "evidence": evidence if isinstance(evidence, list) else [],
        }

    def _apply_conflict_reconciliation(self, verdict: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        packet = dict(verdict or {})
        qa = context.get("qa_reviewer") if isinstance(context.get("qa_reviewer"), dict) else {}
        reviewer = context.get("reviewer") if isinstance(context.get("reviewer"), dict) else {}
        reviewer_status = str(reviewer.get("status", "")).upper()
        qa_status = str(qa.get("status", "")).upper()
        conflict = (
            (reviewer_status in {"APPROVED", "APPROVE_WITH_WARNINGS"} and qa_status in {"REJECTED", "NEEDS_IMPROVEMENT"})
            or (qa_status in {"APPROVED", "APPROVE_WITH_WARNINGS"} and reviewer_status in {"REJECTED", "NEEDS_IMPROVEMENT"})
        )
        if not conflict:
            return packet

        failed_areas = packet.get("failed_areas")
        if not isinstance(failed_areas, list):
            failed_areas = []
        if "cross_reviewer_conflict" not in failed_areas:
            failed_areas.append("cross_reviewer_conflict")
        packet["failed_areas"] = failed_areas

        required_actions = packet.get("required_actions")
        if not isinstance(required_actions, list):
            required_actions = []
        action = "Reconcile Reviewer vs QA findings and rerun after conflict resolution."
        if action not in required_actions:
            required_actions.append(action)
        packet["required_actions"] = required_actions

        status = str(packet.get("status", "")).upper()
        if status == "APPROVED":
            packet["status"] = "APPROVE_WITH_WARNINGS"
        summary = str(packet.get("summary", "")).strip()
        conflict_note = "Conflict detected between reviewer and qa_reviewer packets."
        packet["summary"] = f"{summary} {conflict_note}".strip() if summary else conflict_note
        return packet

    def _apply_progress_policy(self, verdict: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        packet = dict(verdict or {})
        progress = context.get("progress_tracker") if isinstance(context.get("progress_tracker"), dict) else {}
        if not isinstance(progress, dict) or not progress.get("available"):
            return packet
        metric_name = str(progress.get("metric_name") or "metric")
        delta = progress.get("delta")
        try:
            delta_txt = f"{float(delta):.6g}"
        except Exception:
            delta_txt = str(delta)
        improved = progress.get("improved")

        failed_areas = packet.get("failed_areas")
        if not isinstance(failed_areas, list):
            failed_areas = []
        required_actions = packet.get("required_actions")
        if not isinstance(required_actions, list):
            required_actions = []

        if improved is False and str(packet.get("status", "")).upper() == "APPROVED":
            packet["status"] = "APPROVE_WITH_WARNINGS"
            if "progress_regression" not in failed_areas:
                failed_areas.append("progress_regression")
            action = (
                f"Review regression in {metric_name} (delta={delta_txt}) before final approval."
            )
            if action not in required_actions:
                required_actions.append(action)

        summary = str(packet.get("summary") or "").strip()
        progress_note = f"Progress tracker: {metric_name} delta={delta_txt}, improved={bool(improved)}."
        packet["summary"] = f"{summary} {progress_note}".strip() if summary else progress_note
        packet["failed_areas"] = failed_areas
        packet["required_actions"] = required_actions
        return packet

    def _clean_json(self, text: str) -> str:
        cleaned = str(text or "")
        cleaned = cleaned.replace("```json", "")
        cleaned = cleaned.replace("```", "")
        return cleaned.strip()
