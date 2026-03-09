import json
import os

from src.utils.data_adequacy import build_data_adequacy_report


def test_data_adequacy_uses_manifest_dialect(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    manifest = {"output_dialect": {"sep": ";", "decimal": ",", "encoding": "utf-8"}}
    with open(os.path.join("data", "cleaning_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)
    with open(os.path.join("data", "cleaned_data.csv"), "w", encoding="utf-8") as handle:
        handle.write("a;b\n1;2\n3;4\n")

    state = {
        "execution_contract": {
            "validation_requirements": {"metrics": {"roc_auc": 0.7}},
        }
    }
    report = build_data_adequacy_report(state)

    assert report.get("status") != "unknown"
    assert not any("cleaned_data_missing" in reason for reason in report.get("reasons", []))


def test_data_adequacy_infers_objective_family_from_metrics(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "cleaned_data.csv"), "w", encoding="utf-8") as handle:
        handle.write("x\n1\n2\n")
    with open(os.path.join("data", "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump({"model_performance": {"rmse": 1.2}}, handle)

    report = build_data_adequacy_report({})

    assert report.get("signals", {}).get("objective_type") in {"regression", "forecasting"}


def test_data_adequacy_reads_evaluation_metrics_artifact(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    with open(os.path.join("data", "cleaned_data.csv"), "w", encoding="utf-8") as handle:
        handle.write("feature,target\n1,2\n2,3\n")
    with open(os.path.join("reports", "evaluation_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_performance": {
                    "primary_metric": "RMSLE",
                    "primary_metric_value": 0.42,
                    "cv_rmsle_mean": 0.42,
                }
            },
            handle,
        )

    report = build_data_adequacy_report(
        {"execution_contract": {"evaluation_spec": {"objective_type": "regression"}}}
    )

    reasons = [str(item) for item in (report.get("reasons") or [])]
    assert "pipeline_aborted_before_metrics" not in reasons


def test_data_adequacy_supports_survival_metrics_without_regression_fallback(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("data", exist_ok=True)
    with open(os.path.join("data", "cleaned_data.csv"), "w", encoding="utf-8") as handle:
        handle.write("feature,time_to_event,event\n1,10,1\n2,12,0\n")
    with open(os.path.join("data", "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_performance": {
                    "primary_metric": "concordance_index",
                    "primary_metric_value": 0.71,
                    "best_model_concordance_index": 0.71,
                }
            },
            handle,
        )

    report = build_data_adequacy_report(
        {
            "execution_contract": {
                "business_objective": "Estimate time to event under censoring",
                "objective_analysis": {"problem_type": "survival_analysis"},
                "evaluation_spec": {
                    "survival_time_col": "time_to_event",
                    "survival_event_col": "event",
                },
                "validation_requirements": {"primary_metric": "concordance_index"},
            }
        }
    )

    reasons = [str(item) for item in (report.get("reasons") or [])]
    signals = report.get("signals") or {}

    assert signals.get("objective_type") == "survival_analysis"
    assert signals.get("primary_metric_name") == "concordance_index"
    assert "primary_metric_missing" not in reasons
    assert "regression_metric_missing" not in reasons
