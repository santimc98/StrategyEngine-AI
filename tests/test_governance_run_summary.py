import json
import os

from src.utils.governance import build_run_summary


def test_run_summary_outcome_with_limitations(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    with open("data/metrics.json", "w", encoding="utf-8") as f:
        json.dump({"auc": 0.51, "baseline_auc": 0.5}, f)
    with open("data/output_contract_report.json", "w", encoding="utf-8") as f:
        json.dump({"missing": []}, f)
    with open("data/execution_contract.json", "w", encoding="utf-8") as f:
        json.dump({"counterfactual_policy": "observational_only"}, f)
    summary = build_run_summary({"review_verdict": "APPROVED"})
    assert summary.get("run_outcome") == "GO_WITH_LIMITATIONS"
    assert summary.get("metric_ceiling_detected") is True


def test_run_summary_integrity_critical_forces_no_go(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    with open("data/metrics.json", "w", encoding="utf-8") as f:
        json.dump({"auc": 0.6}, f)
    with open("data/output_contract_report.json", "w", encoding="utf-8") as f:
        json.dump({"missing": []}, f)
    with open("data/integrity_audit_report.json", "w", encoding="utf-8") as f:
        json.dump(
            {"issues": [{"type": "MISSING_COLUMN", "severity": "critical"}]},
            f,
        )

    summary = build_run_summary({"review_verdict": "APPROVED"})
    assert summary.get("run_outcome") == "NO_GO"
    assert "integrity_critical" in summary.get("failed_gates", [])
    assert summary.get("integrity_critical_count") == 1


def test_run_summary_integrity_warning_does_not_force_no_go(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    with open("data/metrics.json", "w", encoding="utf-8") as f:
        json.dump({"auc": 0.6}, f)
    with open("data/output_contract_report.json", "w", encoding="utf-8") as f:
        json.dump({"missing": []}, f)
    with open("data/integrity_audit_report.json", "w", encoding="utf-8") as f:
        json.dump(
            {"issues": [{"type": "OPTIONAL_COLUMN_MISSING", "severity": "warning"}]},
            f,
        )

    summary = build_run_summary({"review_verdict": "APPROVED"})
    assert summary.get("integrity_critical_count") == 0
    assert "integrity_critical" not in summary.get("failed_gates", [])
    assert summary.get("run_outcome") in {"GO", "GO_WITH_LIMITATIONS"}


def test_run_summary_reports_all_contract_views(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    views_dir = os.path.join("data", "contracts", "views")
    os.makedirs(views_dir, exist_ok=True)
    for name in [
        "de_view",
        "ml_view",
        "cleaning_view",
        "qa_view",
        "reviewer_view",
        "translator_view",
        "results_advisor_view",
    ]:
        with open(os.path.join(views_dir, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump({"role": name}, f)

    summary = build_run_summary({"review_verdict": "APPROVED"})
    present = set((summary.get("contract_views") or {}).get("present") or [])
    assert "cleaning_view" in present
    assert "qa_view" in present


def test_run_summary_drops_broad_qa_gate_when_qa_packet_has_no_findings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    with open("data/output_contract_report.json", "w", encoding="utf-8") as f:
        json.dump({"missing": []}, f)

    state = {
        "review_verdict": "APPROVE_WITH_WARNINGS",
        "last_gate_context": {"status": "APPROVE_WITH_WARNINGS", "failed_gates": ["qa_gates"]},
        "qa_last_result": {"status": "APPROVE_WITH_WARNINGS", "failed_gates": [], "hard_failures": []},
    }
    summary = build_run_summary(state)
    assert "qa_gates" not in (summary.get("failed_gates") or [])


def test_run_summary_ignores_stale_pipeline_aborted_reason_when_metrics_exist(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    with open("data/output_contract_report.json", "w", encoding="utf-8") as f:
        json.dump({"missing": []}, f)
    with open("reports/evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_performance": {
                    "primary_metric": "RMSLE",
                    "primary_metric_value": 0.4249,
                    "cv_rmsle_mean": 0.4249,
                }
            },
            f,
        )
    with open("data/data_adequacy_report.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "status": "insufficient_signal",
                "reasons": ["pipeline_aborted_before_metrics"],
                "recommendations": [],
                "quality_gates_alignment": {"status": "partial", "mapped_gates": {}, "unmapped_gates": {}},
            },
            f,
        )

    summary = build_run_summary({"review_verdict": "APPROVED"})
    assert summary.get("metrics", {}).get("metric_pool_size", 0) > 0
    assert summary.get("metric_ceiling_detected") is False


def test_run_summary_prefers_state_metric_snapshot_over_stale_metrics_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    with open("data/metrics.json", "w", encoding="utf-8") as f:
        json.dump({"qa_gates": {"submission_schema_exact": {"rows": 95}}}, f)
    with open("data/output_contract_report.json", "w", encoding="utf-8") as f:
        json.dump({"missing": []}, f)
    with open("data/review_board_verdict.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "metric_round_finalization": {
                    "metric_name": "mean_multi_horizon_log_loss",
                    "kept": "baseline",
                    "baseline_metric": 0.330705410118,
                    "candidate_metric": 0.37590299155,
                    "final_metric": 0.330705410118,
                }
            },
            f,
        )

    state = {
        "review_verdict": "APPROVED",
        "metrics_report": {
            "primary_metric_name": "mean_multi_horizon_log_loss",
            "primary_metric_value": 0.330705410118,
        },
    }

    summary = build_run_summary(state)

    metric_improvement = summary.get("metric_improvement") or {}
    assert metric_improvement.get("metric_name") == "mean_multi_horizon_log_loss"
    assert metric_improvement.get("final_metric_artifact") == 0.330705410118


def test_run_summary_uses_metric_loop_state_to_clear_stale_pipeline_abort_and_build_metric_improvement(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    with open("data/output_contract_report.json", "w", encoding="utf-8") as f:
        json.dump({"missing": []}, f)
    with open("data/data_adequacy_report.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "status": "insufficient_signal",
                "reasons": ["pipeline_aborted_before_metrics"],
                "recommendations": ["legacy"],
                "quality_gates_alignment": {"status": "partial", "mapped_gates": {}, "unmapped_gates": {}},
            },
            f,
        )

    state = {
        "review_verdict": "APPROVED",
        "metrics_report": {
            "primary_metric_name": "mean_multi_horizon_log_loss",
            "primary_metric_value": 0.033403701215064016,
        },
        "metric_loop_state": {
            "schema_version": "v1",
            "target": {"name": "mean_multi_horizon_log_loss"},
            "round": {
                "round_id": 3,
                "baseline": {"metric_value": 0.033403701215064016},
            },
            "candidate": {"metric_value": 0.3950342981965467},
            "incumbent": {
                "label": "incumbent",
                "metric_name": "mean_multi_horizon_log_loss",
                "metric_value": 0.033403701215064016,
            },
            "selection": {"selected_label": "incumbent", "reason": "monotonic_metric_degradation"},
            "controller": {"active": False, "continue_round": False, "force_finalize_reason": "monotonic_metric_degradation"},
        },
    }

    summary = build_run_summary(state)

    adequacy = summary.get("data_adequacy") or {}
    metric_improvement = summary.get("metric_improvement") or {}
    assert adequacy.get("status") == "ok"
    assert adequacy.get("reasons") == []
    assert metric_improvement.get("metric_name") == "mean_multi_horizon_log_loss"
    assert metric_improvement.get("kept") == "incumbent"
    assert metric_improvement.get("final_metric_reported") == 0.033403701215064016


def test_run_summary_prefers_clean_state_output_contract_over_stale_persisted_report(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    with open("data/metrics.json", "w", encoding="utf-8") as f:
        json.dump({"auc": 0.61}, f)
    with open("data/output_contract_report.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "overall_status": "error",
                "missing": ["artifacts/ml/cv_metrics.json"],
            },
            f,
        )

    state = {
        "review_verdict": "APPROVED",
        "last_successful_output_contract_report": {
            "overall_status": "ok",
            "missing": [],
            "present": ["artifacts/ml/cv_metrics.json"],
        },
    }

    summary = build_run_summary(state)

    assert "output_contract_missing" not in (summary.get("failed_gates") or [])
    assert "output_contract_report.overall_status=error" not in (summary.get("governance_reasons") or [])


def test_run_summary_loads_cv_metrics_artifact_when_data_metrics_file_is_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs(os.path.join("artifacts", "ml"), exist_ok=True)
    os.makedirs("data", exist_ok=True)
    with open("artifacts/ml/cv_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "primary_metric_name": "mae",
                "primary_metric_value": 40.342921920665276,
                "metrics": {"mae": 40.342921920665276, "rmse": 49.680905103143104},
            },
            f,
        )
    with open("data/output_contract_report.json", "w", encoding="utf-8") as f:
        json.dump({"missing": []}, f)

    summary = build_run_summary({"review_verdict": "APPROVED"})

    assert summary.get("metrics", {}).get("metric_pool_size", 0) > 0


def test_run_summary_loads_legacy_cv_metrics_artifact_schema(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs(os.path.join("artifacts", "ml"), exist_ok=True)
    os.makedirs("data", exist_ok=True)
    with open("artifacts/ml/cv_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "primary_metric": "MAE",
                "aggregate_metrics": {
                    "MAE_mean": 2088.3858989698074,
                    "MAE_std": 592.4474116389869,
                },
                "training_rows": 85,
            },
            f,
        )
    with open("data/output_contract_report.json", "w", encoding="utf-8") as f:
        json.dump({"missing": []}, f)

    summary = build_run_summary({"review_verdict": "APPROVED"})

    baseline_pairs = summary.get("baseline_vs_model") or []
    metric_summary = summary.get("metrics") or {}
    assert metric_summary.get("metric_pool_size", 0) > 0
    assert isinstance(baseline_pairs, list)
