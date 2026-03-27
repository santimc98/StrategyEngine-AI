import json
import os
import time

from src.graph import graph as graph_mod


def test_build_review_board_facts_prefers_best_output_contract_and_insights_metrics(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)

    insights = {
        "metrics_summary": [
            {"metric": "model_performance.Normalized Gini", "value": 0.3125},
            {"metric": "model_performance.ROC-AUC", "value": 0.6562},
        ]
    }
    with open("data/insights.json", "w", encoding="utf-8") as handle:
        json.dump(insights, handle)

    state = {
        "execution_contract": {"evaluation_spec": {"objective_type": "classification"}},
        "output_contract_report": {
            "overall_status": "error",
            "present": ["data/cleaned_data.csv"],
            "missing": ["reports/evaluation_metrics.json"],
        },
        "best_attempt_output_contract_report": {
            "overall_status": "ok",
            "present": ["reports/evaluation_metrics.json"],
            "missing": [],
        },
    }

    facts = graph_mod._build_review_board_facts(state)

    assert facts["output_contract"]["missing_required_artifacts"] == []
    assert facts["metrics"]["primary"]["name"] in {"Normalized Gini", "ROC-AUC"}
    assert facts["metrics"]["primary"]["value"] is not None
    assert facts["metrics"]["primary"]["source"] in {"data/insights.json", "metrics.normalized"}


def test_finalize_heavy_execution_updates_best_attempt_snapshot(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    with open("data/cleaned_data.csv", "w", encoding="utf-8") as handle:
        handle.write("x,target\n1,0\n")
    with open("reports/evaluation_metrics.json", "w", encoding="utf-8") as handle:
        json.dump({"model_performance": {"accuracy": 0.9}}, handle)

    state = {
        "execution_attempt": 0,
        "ml_data_path": "data/cleaned_data.csv",
    }
    contract = {
        "required_outputs": ["reports/evaluation_metrics.json"],
    }

    result = graph_mod._finalize_heavy_execution(
        state=state,
        output="Execution completed successfully.",
        exec_start_ts=time.time() - 1.0,
        contract=contract,
        eval_spec={},
        csv_sep=",",
        csv_decimal=".",
        csv_encoding="utf-8",
        counters={},
        run_id=None,
        attempt_id=2,
        visuals_missing=False,
    )

    assert result["last_attempt_valid"] is True
    assert result["best_attempt_id"] == 2
    assert result["best_attempt_output_contract_report"]["missing"] == []
    assert os.path.isdir(result["best_attempt_dir"])


def test_score_attempt_prefers_better_cv_metrics_for_lower_is_better_metric(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    metrics_a = tmp_path / "attempt_a" / "artifacts" / "ml" / "cv_metrics.json"
    metrics_b = tmp_path / "attempt_b" / "artifacts" / "ml" / "cv_metrics.json"
    metrics_a.parent.mkdir(parents=True, exist_ok=True)
    metrics_b.parent.mkdir(parents=True, exist_ok=True)
    metrics_a.write_text(
        json.dumps(
            {
                "primary_metric_name": "mae",
                "primary_metric_value": 40.3429,
                "metrics": {"mae": 40.3429},
            }
        ),
        encoding="utf-8",
    )
    metrics_b.write_text(
        json.dumps(
            {
                "primary_metric_name": "mae",
                "primary_metric_value": 2081.7057,
                "metrics": {"mae": 2081.7057},
            }
        ),
        encoding="utf-8",
    )

    report = {"present": ["artifacts/ml/cv_metrics.json"], "missing": []}
    contract = {"validation_requirements": {"primary_metric": "mae"}}
    better_score = graph_mod._score_attempt(True, report, [], [str(metrics_a)], contract)
    worse_score = graph_mod._score_attempt(True, report, [], [str(metrics_b)], contract)

    assert better_score > worse_score


def test_promote_best_attempt_restores_metrics_report_and_primary_metric_state(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    best_dir = tmp_path / "artifacts" / "best_attempt"
    metrics_path = best_dir / "artifacts" / "ml" / "cv_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "primary_metric_name": "mae",
        "primary_metric_value": 40.3429,
        "metrics": {"mae": 40.3429},
    }
    metrics_path.write_text(json.dumps(metrics_payload), encoding="utf-8")
    metadata = {
        "artifact_index": [{"path": "artifacts/ml/cv_metrics.json"}],
        "output_contract_report": {"overall_status": "ok", "missing": [], "present": ["artifacts/ml/cv_metrics.json"]},
        "execution_output": "Recovered execution output",
        "plots_local": [],
        "metrics_payload": metrics_payload,
        "metrics_path": "artifacts/ml/cv_metrics.json",
        "primary_metric_state": {
            "primary_metric_name": "mae",
            "primary_metric_canonical_name": "mae",
            "primary_metric_value": 40.3429,
            "primary_metric_source": "artifacts/ml/cv_metrics.json",
            "primary_metric_path": "metrics.mae",
            "higher_is_better": False,
        },
    }
    (best_dir / "best_attempt.json").write_text(json.dumps(metadata), encoding="utf-8")

    updates = graph_mod._promote_best_attempt({"best_attempt_dir": str(best_dir)})

    assert updates["metrics_report"]["primary_metric_value"] == 40.3429
    assert updates["primary_metric_state"]["primary_metric_value"] == 40.3429
    persisted = json.loads((tmp_path / "data" / "metric_state.json").read_text(encoding="utf-8"))
    assert persisted["primary_metric_value"] == 40.3429
