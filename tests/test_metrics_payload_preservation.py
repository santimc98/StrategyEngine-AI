import json
from pathlib import Path

import pytest

from src.graph import graph as graph_mod


class _StubReviewerApproved:
    def evaluate_results(self, *_args, **_kwargs):
        return {"status": "APPROVED", "feedback": "ok"}

    def review_code(self, *_args, **_kwargs):
        return {
            "status": "APPROVED",
            "feedback": "review ok",
            "failed_gates": [],
            "required_fixes": [],
            "hard_failures": [],
        }


class _StubQAApproved:
    def review_code(self, *_args, **_kwargs):
        return {
            "status": "APPROVED",
            "feedback": "qa ok",
            "failed_gates": [],
            "required_fixes": [],
            "hard_failures": [],
        }


def test_run_result_evaluator_keeps_model_block_metrics_payload(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("data").mkdir(parents=True, exist_ok=True)

    raw_metrics = {
        "ensemble": {
            "cv_auc_mean": 0.9551356310013077,
            "cv_auc_std": 0.00045293468625677986,
            "cv_logloss_mean": 0.26861767457456504,
        },
        "meta": {"primary_metric": "auc"},
    }
    Path("data/metrics.json").write_text(json.dumps(raw_metrics), encoding="utf-8")
    Path("data/scored_rows.csv").write_text("id,prediction,target\n1,0.9,1\n2,0.2,0\n", encoding="utf-8")
    Path("data/submission.csv").write_text("id,prediction\n1,0.9\n2,0.2\n", encoding="utf-8")

    monkeypatch.setattr(graph_mod, "reviewer", _StubReviewerApproved())
    monkeypatch.setattr(graph_mod, "qa_reviewer", _StubQAApproved())
    monkeypatch.setattr(
        graph_mod.results_advisor,
        "generate_insights",
        lambda _ctx: {
            "summary_lines": ["ok"],
            "risks": [],
            "recommendations": [],
            "iteration_recommendation": {},
        },
    )

    state = {
        "execution_output": "OK",
        "selected_strategy": {"analysis_type": "classification"},
        "business_objective": "",
        "generated_code": "print('ok')",
        "execution_contract": {
            "required_outputs": ["data/metrics.json", "data/scored_rows.csv", "data/submission.csv"],
            "artifact_requirements": {
                "required_files": ["data/metrics.json", "data/scored_rows.csv", "data/submission.csv"]
            },
            "validation_requirements": {"primary_metric": "auc"},
            "evaluation_spec": {"objective_type": "predictive"},
            "spec_extraction": {"case_taxonomy": []},
        },
        "evaluation_spec": {"objective_type": "predictive"},
        "iteration_count": 0,
        "feedback_history": [],
    }

    result = graph_mod.run_result_evaluator(state)

    assert result.get("review_verdict") in {"APPROVED", "APPROVE_WITH_WARNINGS"}
    persisted_metrics = json.loads(Path("data/metrics.json").read_text(encoding="utf-8"))
    assert isinstance(persisted_metrics.get("ensemble"), dict)
    assert persisted_metrics.get("source") != "computed_fallback"
    assert not Path("data/metrics_fallback.json").exists()

    primary_snapshot = result.get("primary_metric_snapshot") if isinstance(result, dict) else {}
    assert isinstance(primary_snapshot, dict)
    assert primary_snapshot.get("primary_metric_value") == pytest.approx(0.9551356310013077, abs=1e-12)


def test_run_result_evaluator_prefers_explicit_primary_metric_value_in_cv_metrics_payload(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("data").mkdir(parents=True, exist_ok=True)

    raw_metrics = {
        "primary_metric_name": "mean_multi_horizon_log_loss",
        "primary_metric_value": 0.13175253791631902,
        "per_horizon_log_loss": {
            "label_12h": 0.24156589678725365,
            "label_24h": 0.12465146604581857,
            "label_48h": 0.09324487589139654,
            "label_72h": 0.06754791294080732,
        },
        "horizon_details": [
            {"target": "label_12h", "oof_log_loss_raw": 0.24156589678725365},
        ],
        "feature_engineering": {
            "interaction_gate": {
                "baseline_mean_log_loss": 0.3375940863668576,
                "candidate_mean_log_loss": 0.3211599042940719,
            }
        },
    }
    Path("data/metrics.json").write_text(json.dumps(raw_metrics), encoding="utf-8")
    Path("data/scored_rows.csv").write_text("id,prediction,target\n1,0.9,1\n2,0.2,0\n", encoding="utf-8")
    Path("data/submission.csv").write_text("id,prediction\n1,0.9\n2,0.2\n", encoding="utf-8")

    monkeypatch.setattr(graph_mod, "reviewer", _StubReviewerApproved())
    monkeypatch.setattr(graph_mod, "qa_reviewer", _StubQAApproved())
    monkeypatch.setattr(
        graph_mod.results_advisor,
        "generate_insights",
        lambda _ctx: {
            "summary_lines": ["ok"],
            "risks": [],
            "recommendations": [],
            "iteration_recommendation": {},
        },
    )

    state = {
        "execution_output": "OK",
        "selected_strategy": {"analysis_type": "predictive"},
        "business_objective": "",
        "generated_code": "print('ok')",
        "execution_contract": {
            "required_outputs": ["data/metrics.json", "data/scored_rows.csv", "data/submission.csv"],
            "artifact_requirements": {
                "required_files": ["data/metrics.json", "data/scored_rows.csv", "data/submission.csv"]
            },
            "validation_requirements": {"primary_metric": "mean_multi_horizon_log_loss"},
            "evaluation_spec": {
                "objective_type": "predictive",
                "primary_metric": "mean_multi_horizon_log_loss",
            },
            "spec_extraction": {"case_taxonomy": []},
        },
        "evaluation_spec": {
            "objective_type": "predictive",
            "primary_metric": "mean_multi_horizon_log_loss",
        },
        "iteration_count": 0,
        "feedback_history": [],
    }

    result = graph_mod.run_result_evaluator(state)

    primary_snapshot = result.get("primary_metric_snapshot") if isinstance(result, dict) else {}
    assert isinstance(primary_snapshot, dict)
    assert primary_snapshot.get("primary_metric_value") == pytest.approx(0.13175253791631902, abs=1e-12)
    assert primary_snapshot.get("primary_metric_path") == "primary_metric_value"
