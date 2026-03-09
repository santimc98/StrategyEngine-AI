import json
from pathlib import Path

from src.graph import graph as graph_mod


class _StubReviewerApproveWithRejectingCodeAudit:
    def evaluate_results(self, *_args, **_kwargs):
        return {"status": "APPROVED", "feedback": "Baseline evaluator ok."}

    def review_code(self, *_args, **_kwargs):
        return {
            "status": "REJECTED",
            "feedback": "Code audit rejected.",
            "failed_gates": ["synthetic_gate"],
            "required_fixes": ["Apply synthetic fix."],
            "hard_failures": ["synthetic_hard_failure"],
        }


class _StubQARejecting:
    def review_code(self, *_args, **_kwargs):
        return {
            "status": "REJECTED",
            "feedback": "QA rejected.",
            "failed_gates": ["qa_gate_fail"],
            "required_fixes": ["Apply QA fix."],
            "hard_failures": ["qa_hard_failure"],
        }


class _StubReviewerBaselineGateReject:
    def evaluate_results(self, *_args, **_kwargs):
        return {"status": "APPROVED", "feedback": "Baseline evaluator ok."}

    def review_code(self, *_args, **_kwargs):
        return {
            "status": "REJECTED",
            "feedback": "Baseline simplicity violated during optimization.",
            "failed_gates": ["baseline_simplicity_enforcement"],
            "required_fixes": ["Remove blending from optimization round."],
            "hard_failures": ["baseline_simplicity_enforcement"],
        }


class _StubQAApproved:
    def review_code(self, *_args, **_kwargs):
        return {
            "status": "APPROVED",
            "feedback": "QA passed.",
            "failed_gates": [],
            "required_fixes": [],
            "hard_failures": [],
        }


def test_metric_round_hybrid_guarded_keeps_reviewer_qa_nonblocking(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("data").mkdir(parents=True, exist_ok=True)
    Path("data/metrics.json").write_text(json.dumps({"roc_auc": 0.801}), encoding="utf-8")

    monkeypatch.setattr(graph_mod, "reviewer", _StubReviewerApproveWithRejectingCodeAudit())
    monkeypatch.setattr(graph_mod, "qa_reviewer", _StubQARejecting())

    state = {
        "execution_output": "OK",
        "selected_strategy": {},
        "business_objective": "",
        "generated_code": "print('ok')",
        "execution_contract": {
            "iteration_policy": {"metric_round_review_mode": "hybrid_guarded"},
            "spec_extraction": {"case_taxonomy": []},
        },
        "evaluation_spec": {},
        "iteration_count": 0,
        "feedback_history": [],
        "ml_improvement_round_active": True,
    }

    result = graph_mod.run_result_evaluator(state)

    assert result.get("review_verdict") in {"APPROVED", "APPROVE_WITH_WARNINGS"}
    stack = result.get("ml_review_stack") if isinstance(result.get("ml_review_stack"), dict) else {}
    assert (stack.get("reviewer") or {}).get("status") == "APPROVE_WITH_WARNINGS"
    assert (stack.get("qa_reviewer") or {}).get("status") == "APPROVE_WITH_WARNINGS"
    hard = (result.get("last_gate_context") or {}).get("hard_failures") or []
    assert "qa_rejected" not in hard
    assert any(
        "METRIC_IMPROVEMENT_REVIEW_MODE" in str(item)
        for item in (result.get("feedback_history") or [])
    )


def test_finalize_metric_round_uses_candidate_review_critique(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("data").mkdir(parents=True, exist_ok=True)
    metrics_path = Path("data/metrics.json")
    metrics_path.write_text(json.dumps({"roc_auc": 0.8}), encoding="utf-8")
    snapshot_dir = Path("work/ml_baseline_snapshot")
    output_paths = ["data/metrics.json"]
    graph_mod._snapshot_ml_outputs(output_paths, snapshot_dir)
    metrics_path.write_text(json.dumps({"roc_auc": 0.801}), encoding="utf-8")

    captured = {}

    def _stub_candidate_critique(context):
        captured.update(context if isinstance(context, dict) else {})
        return {
            "packet_type": "advisor_critique_packet",
            "packet_version": "1.0",
            "run_id": "run_test",
            "iteration": int(context.get("iteration", 0)) if isinstance(context, dict) else 0,
            "timestamp_utc": "2026-02-20T00:00:00+00:00",
            "primary_metric_name": "roc_auc",
            "higher_is_better": True,
            "metric_comparison": {
                "baseline_value": 0.8,
                "candidate_value": 0.801,
                "delta_abs": 0.001,
                "delta_rel": 0.00125,
                "min_delta_required": 0.0005,
                "meets_min_delta": False,
            },
            "validation_signals": {"validation_mode": "unknown"},
            "error_modes": [],
            "risk_flags": ["no_material_metric_gain"],
            "active_gates_context": [],
            "analysis_summary": "Candidate review says no material gain.",
            "strictly_no_code_advice": True,
        }

    monkeypatch.setattr(graph_mod.results_advisor, "generate_critique_packet", _stub_candidate_critique)
    monkeypatch.setattr(
        graph_mod.results_advisor,
        "last_critique_meta",
        {"mode": "deterministic", "source": "deterministic", "provider": "none", "model": None},
    )

    state = {
        "review_verdict": "NEEDS_IMPROVEMENT",
        "execution_contract": {},
        "ml_improvement_round_active": True,
        "ml_improvement_primary_metric_name": "roc_auc",
        "ml_improvement_baseline_metric": 0.8,
        "ml_improvement_min_delta": 0.0005,
        "ml_improvement_higher_is_better": True,
        "ml_improvement_output_paths": output_paths,
        "ml_improvement_snapshot_dir": str(snapshot_dir),
        "ml_improvement_baseline_review_verdict": "APPROVED",
        "ml_improvement_baseline_metrics": {"roc_auc": 0.8},
        "feedback_history": [],
        "last_gate_context": {"failed_gates": [], "hard_failures": []},
    }

    route = graph_mod.check_evaluation(state)
    restored = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert route == "approved"
    assert state.get("ml_improvement_kept") == "baseline"
    assert restored.get("roc_auc") == 0.8
    assert captured.get("phase") == "candidate_review"


def test_metric_round_reviewer_cannot_block_on_baseline_only_gate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("data").mkdir(parents=True, exist_ok=True)
    Path("data/metrics.json").write_text(json.dumps({"roc_auc": 0.801}), encoding="utf-8")

    monkeypatch.setattr(graph_mod, "reviewer", _StubReviewerBaselineGateReject())
    monkeypatch.setattr(graph_mod, "qa_reviewer", _StubQAApproved())

    state = {
        "execution_output": "OK",
        "selected_strategy": {},
        "business_objective": "",
        "generated_code": "print('ok')",
        "execution_contract": {
            "iteration_policy": {"metric_round_review_mode": "hybrid_guarded"},
            "spec_extraction": {"case_taxonomy": []},
            "reviewer_gates": [
                {
                    "name": "baseline_simplicity_enforcement",
                    "severity": "HARD",
                    "params": {"forbidden_techniques": ["stacking", "blending"]},
                },
                {"name": "submission_schema_compliance", "severity": "HARD"},
            ],
        },
        "evaluation_spec": {
            "reviewer_gates": [
                {
                    "name": "baseline_simplicity_enforcement",
                    "severity": "HARD",
                    "params": {"forbidden_techniques": ["stacking", "blending"]},
                },
                {"name": "submission_schema_compliance", "severity": "HARD"},
            ]
        },
        "iteration_count": 0,
        "feedback_history": [],
        "ml_improvement_round_active": True,
    }

    result = graph_mod.run_result_evaluator(state)

    stack = result.get("ml_review_stack") if isinstance(result.get("ml_review_stack"), dict) else {}
    reviewer_packet = stack.get("reviewer") if isinstance(stack.get("reviewer"), dict) else {}
    assert reviewer_packet.get("status") == "APPROVE_WITH_WARNINGS"
    assert reviewer_packet.get("failed_gates") == []
    assert reviewer_packet.get("hard_failures") == []
