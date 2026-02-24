import json
import os

from src.graph import graph as graph_mod


class _ReviewerApprovedStub:
    def evaluate_results(self, *_args, **_kwargs):
        return {"status": "APPROVED", "feedback": "ok", "retry_worth_it": False}

    def review_code(self, *_args, **_kwargs):
        return {
            "status": "APPROVED",
            "feedback": "review ok",
            "failed_gates": [],
            "required_fixes": [],
            "hard_failures": [],
        }


class _QaWarningsStub:
    def review_code(self, *_args, **_kwargs):
        return {
            "status": "APPROVE_WITH_WARNINGS",
            "feedback": "qa warnings",
            "failed_gates": [],
            "required_fixes": [],
            "hard_failures": [],
            "warnings": ["minor warning"],
        }


def test_result_evaluator_persists_qa_and_reviewer_packets_with_warnings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"roc_auc": 0.81}, f)

    state = {
        "execution_output": "OK",
        "selected_strategy": {"title": "test"},
        "business_objective": "test objective",
        "generated_code": "print('ok')\n",
        "execution_contract": {"spec_extraction": {"case_taxonomy": []}},
        "evaluation_spec": {},
        "iteration_count": 0,
        "feedback_history": [],
    }

    monkeypatch.setattr(graph_mod, "reviewer", _ReviewerApprovedStub())
    monkeypatch.setattr(graph_mod, "qa_reviewer", _QaWarningsStub())

    result = graph_mod.run_result_evaluator(state)
    qa_packet = result.get("qa_last_result") or {}
    reviewer_packet = result.get("reviewer_last_result") or {}

    assert qa_packet.get("status") == "APPROVE_WITH_WARNINGS"
    assert reviewer_packet.get("status") == "APPROVED"

    merged_state = dict(state)
    merged_state.update(result)
    assert graph_mod._has_real_baseline_reviewer_approval(merged_state) is True


def test_has_real_baseline_reviewer_approval_falls_back_to_review_stack() -> None:
    state = {
        "ml_review_stack": {
            "reviewer": {"status": "APPROVED"},
            "qa_reviewer": {"status": "APPROVE_WITH_WARNINGS"},
        }
    }
    assert graph_mod._has_real_baseline_reviewer_approval(state) is True


def test_build_ml_iteration_journal_entry_includes_handoff_and_parse_repair_flags():
    state = {
        "iteration_count": 1,
        "generated_code": "print('ok')",
        "ml_improvement_round_count": 1,
        "ml_improvement_hypothesis_packet": {
            "action": "APPLY",
            "tracker_context": {"signature": "hyp_test"},
            "hypothesis": {"technique": "missing_indicators"},
        },
        "ml_improvement_round_history": [
            {"round_id": 1, "delta": 0.0012, "kept": "improved", "reason": "candidate_selected"}
        ],
        "iteration_handoff": {
            "source": "result_evaluator",
            "mode": "patch",
            "from_iteration": 1,
            "next_iteration": 2,
        },
    }
    entry = graph_mod._build_ml_iteration_journal_entry(
        state=state,
        preflight_issues=[],
        runtime_error=None,
        outputs_present=["data/metrics.json"],
        outputs_missing=[],
        reviewer_verdict="APPROVED",
        reviewer_reasons=[],
        qa_verdict="APPROVE_WITH_WARNINGS",
        qa_reasons=["warning"],
        next_actions=["keep going"],
        stage="review_complete",
        reviewer_packet={"status": "APPROVED", "json_parse_trace": {"used_repair": True}},
        qa_packet={"status": "APPROVE_WITH_WARNINGS", "json_parse_trace": {"used_repair": False}},
        iteration_handoff=state["iteration_handoff"],
    )

    assert entry.get("handoff_meta", {}).get("source") == "result_evaluator"
    assert entry.get("handoff_meta", {}).get("next_iteration") == 2
    assert entry.get("reviewer_json_repair_used") is True
    assert entry.get("qa_json_repair_used") is False
    assert entry.get("metric_round", {}).get("round_id") == 1
    assert entry.get("metric_round", {}).get("technique") == "missing_indicators"
    assert entry.get("metric_round", {}).get("kept") == "improved"
