import json

from src.utils.metric_eval import (
    canonicalize_metrics_report_file,
    normalize_metrics_report_payload,
)


def test_normalize_metrics_report_payload_lifts_legacy_cv_metrics_schema():
    payload = {
        "model_name": "ridge_regression",
        "training_rows": 85,
        "feature_columns_used": ["Size", "Sector"],
        "fold_metrics": [
            {"MAE": 2986.63, "RMSE": 3623.71, "fold": 1, "train_rows": 15, "valid_rows": 14},
            {"MAE": 1321.86, "RMSE": 1682.24, "fold": 2, "train_rows": 29, "valid_rows": 14},
        ],
        "aggregate_metrics": {
            "MAE_mean": 2088.3858989698074,
            "MAE_std": 592.4474116389869,
            "RMSE_mean": 2459.1621062065706,
        },
        "primary_metric": "MAE",
    }

    normalized = normalize_metrics_report_payload(payload)

    assert normalized["primary_metric_name"] == "MAE"
    assert normalized["primary_metric_canonical_name"] == "mae"
    assert normalized["primary_metric_value"] == 2088.3858989698074
    assert normalized["mean_mae"] == 2088.3858989698074
    assert normalized["std_mae"] == 592.4474116389869
    assert normalized["mean_rmse"] == 2459.1621062065706
    assert normalized["feature_columns"] == ["Size", "Sector"]
    assert normalized["n_train_rows"] == 85
    assert normalized["model_family"] == "ridge_regression"
    assert normalized["model_performance"]["primary_metric_value"] == 2088.3858989698074


def test_canonicalize_metrics_report_file_rewrites_legacy_payload(tmp_path):
    path = tmp_path / "cv_metrics.json"
    path.write_text(
        json.dumps(
            {
                "aggregate_metrics": {"MAE_mean": 10.5, "MAE_std": 1.25},
                "primary_metric": "MAE",
                "training_rows": 20,
            }
        ),
        encoding="utf-8",
    )

    normalized = canonicalize_metrics_report_file(str(path))
    persisted = json.loads(path.read_text(encoding="utf-8"))

    assert normalized["mean_mae"] == 10.5
    assert normalized["std_mae"] == 1.25
    assert normalized["primary_metric_value"] == 10.5
    assert persisted["mean_mae"] == 10.5
    assert persisted["std_mae"] == 1.25
    assert persisted["primary_metric_canonical_name"] == "mae"


def test_normalize_metrics_report_payload_resolves_ordering_violation_reduction() -> None:
    payload = {
        "primary_metric_name": "Reduction in case-level ordering violations",
        "full_data_ordering": {
            "baseline_violation_count": 19,
            "calibrated_violation_count": 16,
            "violation_reduction": 3,
        },
    }

    normalized = normalize_metrics_report_payload(payload)

    assert normalized["primary_metric_canonical_name"] == "violation_reduction"
    assert normalized["primary_metric_value"] == 3.0
    assert normalized["higher_is_better"] is True
    assert normalized["model_performance"]["primary_metric_value"] == 3.0
    assert normalized["model_performance"]["higher_is_better"] is True
