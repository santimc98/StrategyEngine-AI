import json
import os

from src.graph import graph as graph_mod


class _StubBoardWarnings:
    def __init__(self):
        self.last_prompt = None
        self.last_response = None

    def adjudicate(self, _context):
        return {
            "status": "APPROVE_WITH_WARNINGS",
            "summary": "Runtime failed but report can proceed with caveats.",
            "failed_areas": ["runtime"],
            "required_actions": ["Document runtime failure in report."],
            "confidence": "high",
            "evidence": [],
        }


class _StubBoardNeedsImprovement:
    def __init__(self):
        self.last_prompt = None
        self.last_response = None

    def adjudicate(self, _context):
        return {
            "status": "NEEDS_IMPROVEMENT",
            "summary": "Must retry with contract-compliant artifacts.",
            "failed_areas": ["qa_gates"],
            "required_actions": ["Retry ML iteration with fixes."],
            "confidence": "high",
            "evidence": [],
        }

class _StubBoardMetricOnlyNeedsImprovement:
    def __init__(self):
        self.last_prompt = None
        self.last_response = None

    def adjudicate(self, _context):
        return {
            "status": "NEEDS_IMPROVEMENT",
            "summary": "Metric is below ideal benchmark.",
            "failed_areas": ["metric_gap"],
            "required_actions": ["Try additional model families to improve KPI."],
            "confidence": "medium",
            "evidence": [],
        }


class _StubBoardSpuriousAreas:
    def __init__(self):
        self.last_prompt = None
        self.last_response = None

    def adjudicate(self, _context):
        return {
            "status": "APPROVE_WITH_WARNINGS",
            "summary": "Minor caveats only.",
            "failed_areas": ["qa_gates", "reviewer_alignment"],
            "required_actions": [],
            "confidence": "high",
            "evidence": [],
        }


class _RuntimePathReviewer:
    def evaluate_results(self, *_args, **_kwargs):
        raise AssertionError("evaluate_results must be skipped when runtime markers are present.")


def test_check_evaluation_terminal_runtime_stops():
    state = {
        "runtime_fix_terminal": True,
        "review_verdict": "NEEDS_IMPROVEMENT",
    }
    assert graph_mod.check_evaluation(state) == "approved"


def test_run_review_board_persists_verdict(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    monkeypatch.setattr(graph_mod, "review_board", _StubBoardWarnings())
    state = {
        "review_verdict": "NEEDS_IMPROVEMENT",
        "review_feedback": "Runtime failure detected.",
        "feedback_history": [],
        "last_gate_context": {"failed_gates": ["runtime_failure"], "required_fixes": []},
        "ml_review_stack": {
            "runtime": {"status": "FAILED_RUNTIME", "runtime_fix_terminal": True},
            "result_evaluator": {"status": "NEEDS_IMPROVEMENT"},
            "reviewer": {"status": "SKIPPED"},
            "qa_reviewer": {"status": "SKIPPED"},
            "results_advisor": {"status": "APPROVE_WITH_WARNINGS"},
        },
    }

    result = graph_mod.run_review_board(state)
    assert result["review_verdict"] == "APPROVE_WITH_WARNINGS"
    assert os.path.exists("data/review_board_verdict.json")
    with open("data/review_board_verdict.json", "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["status"] == "APPROVE_WITH_WARNINGS"
    assert payload["final_review_verdict"] == "APPROVE_WITH_WARNINGS"


def test_run_result_evaluator_runtime_failure_builds_review_stack(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    monkeypatch.setattr(graph_mod, "reviewer", _RuntimePathReviewer())
    state = {
        "execution_output": "HEAVY_RUNNER_ERROR: timeout while streaming artifacts",
        "selected_strategy": {},
        "business_objective": "",
        "execution_contract": {"spec_extraction": {"case_taxonomy": []}},
        "evaluation_spec": {},
        "iteration_count": 0,
        "feedback_history": [],
    }

    result = graph_mod.run_result_evaluator(state)
    assert result["review_verdict"] == "NEEDS_IMPROVEMENT"
    assert "ml_review_stack" in result
    assert result["ml_review_stack"]["runtime"]["status"] == "FAILED_RUNTIME"
    assert result["ml_review_stack"]["result_evaluator"]["status"] == result["ml_review_stack"]["final_pre_board"]["status"]
    assert result["ml_review_stack"]["result_evaluator"]["raw_status"] == "NEEDS_IMPROVEMENT"
    assert isinstance(result["ml_review_stack"].get("deterministic_facts"), dict)
    assert os.path.exists("data/ml_review_stack.json")


def test_run_review_board_increments_iteration_on_escalation_to_needs_improvement(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    monkeypatch.setattr(graph_mod, "review_board", _StubBoardNeedsImprovement())
    state = {
        "iteration_count": 0,
        "review_verdict": "APPROVE_WITH_WARNINGS",
        "review_feedback": "Non-blocking warning.",
        "feedback_history": [],
        "last_gate_context": {"failed_gates": [], "required_fixes": []},
        "ml_review_stack": {
            "runtime": {"status": "OK", "runtime_fix_terminal": False},
            "result_evaluator": {"status": "APPROVE_WITH_WARNINGS"},
            "reviewer": {"status": "APPROVED"},
            "qa_reviewer": {"status": "REJECTED", "failed_gates": ["gate_x"]},
            "results_advisor": {"status": "APPROVE_WITH_WARNINGS"},
        },
    }

    result = graph_mod.run_review_board(state)
    assert result["review_verdict"] == "NEEDS_IMPROVEMENT"
    assert result["iteration_count"] == 1


def test_run_review_board_does_not_double_increment_when_already_needs_improvement(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    monkeypatch.setattr(graph_mod, "review_board", _StubBoardNeedsImprovement())
    state = {
        "iteration_count": 2,
        "review_verdict": "NEEDS_IMPROVEMENT",
        "review_feedback": "Already needs retry.",
        "feedback_history": [],
        "last_gate_context": {"failed_gates": ["qa_gates"], "required_fixes": ["Retry"]},
        "ml_review_stack": {
            "runtime": {"status": "OK", "runtime_fix_terminal": False},
            "result_evaluator": {"status": "NEEDS_IMPROVEMENT"},
            "reviewer": {"status": "APPROVED"},
            "qa_reviewer": {"status": "APPROVED"},
            "results_advisor": {"status": "APPROVE_WITH_WARNINGS"},
        },
    }

    result = graph_mod.run_review_board(state)
    assert result["review_verdict"] == "NEEDS_IMPROVEMENT"
    assert "iteration_count" not in result


def test_run_review_board_metric_only_needs_improvement_is_downgraded(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    monkeypatch.setattr(graph_mod, "review_board", _StubBoardMetricOnlyNeedsImprovement())
    state = {
        "iteration_count": 0,
        "review_verdict": "APPROVE_WITH_WARNINGS",
        "review_feedback": "Metric advisory only.",
        "feedback_history": [],
        "execution_output": "OK",
        "last_gate_context": {"failed_gates": [], "required_fixes": []},
        "ml_review_stack": {
            "runtime": {"status": "OK", "runtime_fix_terminal": False},
            "result_evaluator": {"status": "APPROVE_WITH_WARNINGS"},
            "reviewer": {"status": "APPROVED"},
            "qa_reviewer": {"status": "APPROVED"},
            "results_advisor": {"status": "APPROVE_WITH_WARNINGS"},
        },
    }

    result = graph_mod.run_review_board(state)
    assert result["review_verdict"] == "APPROVE_WITH_WARNINGS"
    assert "iteration_count" not in result
    assert any("REVIEW_BOARD_POLICY" in item for item in (result.get("feedback_history") or []))


def test_run_review_board_reconciles_provisional_attempt_cycle(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    monkeypatch.setattr(graph_mod, "review_board", _StubBoardNeedsImprovement())
    state = {
        "iteration_count": 0,
        "execution_attempt": 1,
        "review_verdict": "APPROVE_WITH_WARNINGS",
        "review_feedback": "Provisional approval before board.",
        "feedback_history": [],
        "last_gate_context": {"failed_gates": [], "required_fixes": []},
        "attempt_ledger": [
            {
                "phase": "compliance",
                "outcome": "approved",
                "root_cause": "",
                "source": "run_result_evaluator",
                "execution_attempt": 1,
                "iteration_count": 0,
                "cycle_key": "result_evaluator|1|1|compliance|approved",
            }
        ],
        "ml_review_stack": {
            "runtime": {"status": "OK", "runtime_fix_terminal": False},
            "result_evaluator": {"status": "APPROVE_WITH_WARNINGS"},
            "reviewer": {"status": "APPROVED"},
            "qa_reviewer": {"status": "REJECTED", "failed_gates": ["gate_x"]},
            "results_advisor": {"status": "APPROVE_WITH_WARNINGS"},
        },
    }

    result = graph_mod.run_review_board(state)
    assert result["review_verdict"] == "NEEDS_IMPROVEMENT"
    assert result["attempt_ledger"][-1]["outcome"] == "needs_improvement"
    assert result["attempt_ledger"][-1]["phase"] == "compliance"


def test_run_review_board_drops_spurious_failed_areas_when_packets_are_clean(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    monkeypatch.setattr(graph_mod, "review_board", _StubBoardSpuriousAreas())
    state = {
        "review_verdict": "APPROVED",
        "review_feedback": "All checks passed.",
        "feedback_history": [],
        "last_gate_context": {"failed_gates": [], "required_fixes": []},
        "ml_review_stack": {
            "runtime": {"status": "OK", "runtime_fix_terminal": False},
            "result_evaluator": {"status": "APPROVED"},
            "reviewer": {"status": "APPROVED", "failed_gates": [], "hard_failures": []},
            "qa_reviewer": {"status": "APPROVE_WITH_WARNINGS", "failed_gates": [], "hard_failures": []},
            "results_advisor": {"status": "APPROVED"},
        },
    }

    result = graph_mod.run_review_board(state)
    assert result["review_verdict"] == "APPROVE_WITH_WARNINGS"
    assert not result["last_gate_context"].get("failed_gates")
    assert any("REVIEW_BOARD_AREA_SANITIZE" in item for item in (result.get("feedback_history") or []))

    with open("data/review_board_verdict.json", "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert payload.get("failed_areas") == []


def test_run_review_board_preserves_metric_optimization_handoff(tmp_path, monkeypatch):
    class _StubBoardApproved:
        def __init__(self):
            self.last_prompt = None
            self.last_response = None

        def adjudicate(self, _context):
            return {
                "status": "APPROVED",
                "summary": "Round approved.",
                "failed_areas": [],
                "required_actions": [],
                "confidence": "high",
                "evidence": [],
            }

    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    monkeypatch.setattr(graph_mod, "review_board", _StubBoardApproved())
    state = {
        "review_verdict": "APPROVED",
        "review_feedback": "Round approved.",
        "feedback_history": [],
        "ml_improvement_round_active": True,
        "last_gate_context": {"failed_gates": [], "required_fixes": []},
        "ml_improvement_hypothesis_packet": {
            "action": "APPLY",
            "hypothesis": {"technique": "missing_indicators", "target_columns": ["ALL_NUMERIC"]},
        },
        "ml_improvement_critique_packet": {"analysis_summary": "baseline stable"},
        "iteration_handoff": {
            "mode": "optimize",
            "source": "actor_critic_metric_improvement",
            "optimization_context": {"policy": {"phase": "explore"}},
            "hypothesis_packet": {
                "action": "APPLY",
                "hypothesis": {"technique": "missing_indicators", "target_columns": ["ALL_NUMERIC"]},
            },
            "critic_packet": {"analysis_summary": "baseline stable"},
            "patch_objectives": ["Apply active hypothesis."],
        },
        "ml_review_stack": {
            "runtime": {"status": "OK", "runtime_fix_terminal": False},
            "result_evaluator": {"status": "APPROVED"},
            "reviewer": {"status": "APPROVED"},
            "qa_reviewer": {"status": "APPROVED"},
            "results_advisor": {"status": "APPROVED"},
        },
    }

    result = graph_mod.run_review_board(state)
    handoff = result.get("iteration_handoff", {})
    assert handoff.get("mode") == "optimize"
    assert handoff.get("source") == "actor_critic_metric_improvement"
    assert isinstance(handoff.get("hypothesis_packet"), dict) and handoff.get("hypothesis_packet")


def test_run_review_board_switches_to_repair_first_handoff_when_blocking_retry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    monkeypatch.setattr(graph_mod, "review_board", _StubBoardNeedsImprovement())
    state = {
        "review_verdict": "APPROVE_WITH_WARNINGS",
        "review_feedback": "Blocking QA conflict remains.",
        "feedback_history": [],
        "last_gate_context": {"failed_gates": [], "required_fixes": [], "hard_failures": []},
        "iteration_handoff": {
            "mode": "optimize",
            "source": "actor_critic_metric_improvement",
            "patch_objectives": ["Metric-improvement round: apply active hypothesis 'catboost' with material edits (NO_OP forbidden)."],
            "editor_constraints": {"must_apply_hypothesis": True, "forbid_noop": True, "patch_intensity": "aggressive"},
            "optimization_context": {"policy": {"phase": "explore"}},
            "hypothesis_packet": {"action": "APPLY", "hypothesis": {"technique": "catboost"}},
        },
        "ml_review_stack": {
            "runtime": {"status": "OK", "runtime_fix_terminal": False},
            "result_evaluator": {"status": "APPROVE_WITH_WARNINGS"},
            "reviewer": {"status": "APPROVED"},
            "qa_reviewer": {"status": "REJECTED", "hard_failures": ["target_mapping_check"]},
            "results_advisor": {"status": "APPROVE_WITH_WARNINGS"},
        },
    }

    result = graph_mod.run_review_board(state)

    handoff = result.get("iteration_handoff", {})
    assert handoff.get("mode") == "patch"
    assert handoff.get("source") == "review_board_repair_first"
    assert handoff.get("repair_policy", {}).get("primary_focus") == "compliance"
    assert handoff.get("editor_constraints", {}).get("must_apply_hypothesis") is False
    assert handoff.get("deferred_optimization", {}).get("resume_condition")


def test_contract_consistency_promotes_implied_hard_gate_failure() -> None:
    packet = graph_mod._normalize_review_packet_for_state(
        {
            "status": "APPROVE_WITH_WARNINGS",
            "feedback": (
                "The code uses Ordinal Encoding instead of the requested "
                "K-fold Target Encoding for categorical features."
            ),
            "failed_gates": [],
            "required_fixes": [],
            "hard_failures": [],
            "evidence": [
                {
                    "claim": "Ordinal Encoding is used instead of the requested K-fold Target Encoding.",
                    "source": "script:120",
                }
            ],
        },
        default_status="UNKNOWN",
    )
    contract = {
        "qa_gates": [
            {
                "name": "target_encoding_leakage_guard",
                "severity": "HARD",
                "params": {"method": "k-fold_out_of_fold"},
            }
        ]
    }

    enforced = graph_mod._enforce_review_packet_contract_consistency(
        packet,
        contract=contract,
        actor="qa_reviewer",
    )

    assert enforced.get("status") == "REJECTED"
    assert "target_encoding_leakage_guard" in (enforced.get("failed_gates") or [])
    assert "target_encoding_leakage_guard" in (enforced.get("hard_failures") or [])


def test_contract_consistency_keeps_advisory_gate_suggestion_non_blocking() -> None:
    packet = graph_mod._normalize_review_packet_for_state(
        {
            "status": "APPROVE_WITH_WARNINGS",
            "feedback": "Implement K-fold Target Encoding to improve performance.",
            "failed_gates": [],
            "required_fixes": [],
            "hard_failures": [],
        },
        default_status="UNKNOWN",
    )
    contract = {
        "qa_gates": [
            {
                "name": "target_encoding_leakage_guard",
                "severity": "HARD",
                "params": {"method": "k-fold_out_of_fold"},
            }
        ]
    }

    enforced = graph_mod._enforce_review_packet_contract_consistency(
        packet,
        contract=contract,
        actor="qa_reviewer",
    )

    assert enforced.get("status") == "APPROVE_WITH_WARNINGS"
    assert not enforced.get("failed_gates")
    assert not enforced.get("hard_failures")


def test_contract_consistency_ignores_positive_gate_mentions_when_failure_is_elsewhere() -> None:
    packet = graph_mod._normalize_review_packet_for_state(
        {
            "status": "APPROVE_WITH_WARNINGS",
            "feedback": (
                "The code successfully passes all critical QA gates, including strict split isolation and target mapping. "
                "However, the implementation uses HistGradientBoostingClassifier instead of CatBoostClassifier."
            ),
            "failed_gates": [],
            "required_fixes": [],
            "hard_failures": [],
        },
        default_status="UNKNOWN",
    )
    contract = {
        "qa_gates": [
            {"name": "leakage_prevention_split", "severity": "HARD"},
            {"name": "target_mapping_check", "severity": "HARD"},
        ]
    }

    enforced = graph_mod._enforce_review_packet_contract_consistency(
        packet,
        contract=contract,
        actor="qa_reviewer",
    )

    assert enforced.get("status") == "APPROVE_WITH_WARNINGS"
    assert not enforced.get("failed_gates")
    assert not enforced.get("hard_failures")


def test_blocking_signal_ignores_advisory_leakage_wording() -> None:
    assert graph_mod._looks_blocking_retry_signal(
        "Exclude post-outcome features and document leakage prevention."
    ) is False
    assert graph_mod._looks_blocking_retry_signal(
        "qa_reviewer_hard_failure:target_encoding_leakage_guard"
    ) is True


def test_normalize_reason_tags_avoids_positive_baseline_and_leakage_mentions() -> None:
    positive_tags = graph_mod._normalize_reason_tags(
        "Baseline DummyClassifier is implemented and data leakage is prevented by excluding forbidden columns."
    )
    negative_tags = graph_mod._normalize_reason_tags(
        "Baseline missing and leakage risk detected in the current pipeline."
    )

    assert "baseline_missing" not in positive_tags
    assert "leakage" not in positive_tags
    assert "baseline_missing" in negative_tags
    assert "leakage" in negative_tags
