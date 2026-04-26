import json

from src.graph import graph as graph_mod
import pytest
from src.utils.metric_eval import resolve_metric_value


def test_extract_primary_metric_for_board_prefers_contract_metric_over_heuristic(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state = {
        "execution_contract": {
            "evaluation_spec": {
                "objective_type": "classification",
                "primary_metric": "accuracy",
            }
        }
    }
    metrics_report = {
        "source": "state.metrics_report",
        "model_performance": {
            "roc_auc": 0.93,
            "accuracy": 0.81,
        },
    }

    primary = graph_mod._extract_primary_metric_for_board(state, metrics_report)

    assert primary.get("name") == "accuracy"
    assert primary.get("value") == 0.81
    assert primary.get("source") == "state.metrics_report"


def test_extract_primary_metric_for_board_ignores_snapshot_when_contract_metric_differs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state = {
        "primary_metric_snapshot": {
            "primary_metric_name": "roc_auc",
            "primary_metric_value": 0.94,
        },
        "execution_contract": {
            "evaluation_spec": {
                "objective_type": "classification",
                "primary_metric": "accuracy",
            }
        },
    }
    metrics_report = {
        "source": "state.metrics_report",
        "model_performance": {
            "roc_auc": 0.94,
            "accuracy": 0.79,
        },
    }

    primary = graph_mod._extract_primary_metric_for_board(state, metrics_report)

    assert primary.get("name") == "accuracy"
    assert primary.get("value") == 0.79
    assert primary.get("source") != "primary_metric_snapshot"


def test_extract_primary_metric_for_board_reports_missing_contract_metric(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state = {
        "execution_contract": {
            "evaluation_spec": {
                "objective_type": "classification",
                "primary_metric": "normalized_gini",
            }
        }
    }
    metrics_report = {
        "source": "state.metrics_report",
        "model_performance": {
            "roc_auc": 0.9,
            "accuracy": 0.78,
        },
    }

    primary = graph_mod._extract_primary_metric_for_board(state, metrics_report)

    assert primary.get("name") == "normalized_gini"
    assert primary.get("value") is None
    assert primary.get("source") == "contract.primary_metric_missing"


def test_extract_primary_metric_for_board_reads_nested_primary_metric_value(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state = {
        "execution_contract": {
            "evaluation_spec": {
                "objective_type": "regression",
                "primary_metric": "RMSLE",
            }
        }
    }
    metrics_report = {
        "source": "reports/evaluation_metrics.json",
        "model_performance": {
            "primary_metric": {
                "name": "RMSLE",
                "value": 0.4265,
                "std": 0.01,
            },
            "secondary_metrics": {"MAE": 322.2},
        },
    }

    primary = graph_mod._extract_primary_metric_for_board(state, metrics_report)

    assert primary.get("name") in {"RMSLE", "primary_metric"}
    assert primary.get("value") == 0.4265
    assert primary.get("source") != "contract.primary_metric_missing"


def test_normalize_metrics_report_payload_supports_model_block_cv_metrics() -> None:
    payload = {
        "ensemble": {
            "cv_auc_mean": 0.9551,
            "cv_auc_std": 0.0004,
            "cv_logloss_mean": 0.2686,
        },
        "catboost": {
            "cv_auc_mean": 0.9550,
        },
    }

    normalized = graph_mod._normalize_metrics_report_payload(payload)

    assert graph_mod._metrics_report_has_values(normalized) is True
    model_perf = normalized.get("model_performance")
    assert isinstance(model_perf, dict)
    assert model_perf.get("ensemble.cv_auc_mean") == pytest.approx(0.9551, abs=1e-9)


def test_extract_primary_metric_for_board_reads_contract_metric_from_model_blocks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state = {
        "execution_contract": {
            "validation_requirements": {
                "primary_metric": "auc",
            }
        }
    }
    metrics_report = {
        "source": "data/metrics.json",
        "ensemble": {
            "cv_auc_mean": 0.9551356310013077,
            "cv_auc_std": 0.00045293468625677986,
            "cv_logloss_mean": 0.26861767457456504,
        },
    }

    primary = graph_mod._extract_primary_metric_for_board(state, metrics_report)

    assert primary.get("value") == pytest.approx(0.9551356310013077, abs=1e-12)
    assert primary.get("source") != "contract.primary_metric_missing"


def test_extract_primary_metric_for_board_supports_string_numeric_metrics(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state = {
        "execution_contract": {
            "validation_requirements": {
                "primary_metric": "auc",
            }
        }
    }
    metrics_report = {
        "source": "data/metrics.json",
        "model_results": {
            "best_auc": "0.9342",
        },
    }

    primary = graph_mod._extract_primary_metric_for_board(state, metrics_report)

    assert primary.get("value") == pytest.approx(0.9342, abs=1e-12)
    assert primary.get("source") != "contract.primary_metric_missing"


def test_resolve_metric_value_prefers_explicit_primary_metric_value_over_nested_logloss_details() -> None:
    metrics_report = {
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

    resolved = resolve_metric_value(metrics_report, "mean_multi_horizon_log_loss")

    assert resolved.get("value") == pytest.approx(0.13175253791631902, abs=1e-12)
    assert resolved.get("matched_key") == "primary_metric_value"


def test_extract_primary_metric_for_board_prefers_explicit_primary_metric_value_for_cv_metrics_shape(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    state = {
        "execution_contract": {
            "evaluation_spec": {
                "objective_type": "predictive",
                "primary_metric": "mean_multi_horizon_log_loss",
            }
        }
    }
    metrics_report = {
        "source": "artifacts/ml/cv_metrics.json",
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

    primary = graph_mod._extract_primary_metric_for_board(state, metrics_report)

    assert primary.get("name") == "mean_multi_horizon_log_loss"
    assert primary.get("value") == pytest.approx(0.13175253791631902, abs=1e-12)
    assert primary.get("matched_key") == "primary_metric_value"


def test_extract_primary_metric_for_board_resolves_contract_alias_to_nested_auc_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state = {
        "execution_contract": {
            "validation_requirements": {
                "primary_metric": "ROC-AUC",
            }
        }
    }
    metrics_report = {
        "source": "data/metrics.json",
        "model_performance": {
            "cv_results.overall_oof_auc": 0.9160376006743025,
            "cv_results.std_fold_auc": 0.0009317617858213929,
        },
    }

    primary = graph_mod._extract_primary_metric_for_board(state, metrics_report)

    assert primary.get("name") == "ROC-AUC"
    assert primary.get("value") == pytest.approx(0.9160376006743025, abs=1e-12)
    assert primary.get("source") != "contract.primary_metric_missing"


def test_build_primary_metric_state_prefers_explicit_primary_metric_when_contract_metric_is_unavailable() -> None:
    metrics_report = {
        "source": "artifact:artifacts/reports/evaluation_summary.json",
        "primary_metric_name": "mean_multi_horizon_log_loss",
        "primary_metric_value": 0.330705410118,
        "qa_gates": {
            "submission_schema_exact": {
                "rows": 95,
            }
        },
    }

    metric_state = graph_mod._build_primary_metric_state(
        state={},
        metrics_report=metrics_report,
        weights_report={},
        objective_type="predictive",
        evaluation_spec={},
        contract={},
    )

    assert metric_state.get("primary_metric_name") == "mean_multi_horizon_log_loss"
    assert metric_state.get("primary_metric_value") == pytest.approx(0.330705410118, abs=1e-12)
    assert metric_state.get("primary_metric_path") == "primary_metric_value"


def test_build_primary_metric_state_keeps_contract_mean_metric_when_payload_primary_name_is_wrong() -> None:
    metrics_report = {
        "source": "artifact:artifacts/ml/cv_metrics.json",
        "metric_name": "mean_multi_horizon_log_loss",
        "primary_metric_name": "per_horizon.label_12h.oof_log_loss_raw",
        "mean_multi_horizon_log_loss": 0.330705410118,
        "per_horizon": {
            "label_12h": {"oof_log_loss_raw": 0.20169210514337835},
            "label_24h": {"oof_log_loss_raw": 0.12675962448829395},
        },
    }

    metric_state = graph_mod._build_primary_metric_state(
        state={},
        metrics_report=metrics_report,
        weights_report={},
        objective_type="predictive",
        evaluation_spec={"primary_metric": "mean_multi_horizon_log_loss"},
        contract={"validation_requirements": {"primary_metric": "mean_multi_horizon_log_loss"}},
    )

    assert metric_state.get("target_metric_name") == "mean_multi_horizon_log_loss"
    assert metric_state.get("primary_metric_name") == "mean_multi_horizon_log_loss"
    assert metric_state.get("primary_metric_value") == pytest.approx(0.330705410118, abs=1e-12)


def test_declared_metrics_payload_prefers_contract_ml_evaluation_over_clean_drift(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "artifacts" / "clean").mkdir(parents=True)
    (tmp_path / "artifacts" / "ml").mkdir(parents=True)
    (tmp_path / "artifacts" / "clean" / "feature_drift_baseline.json").write_text(
        json.dumps(
            {
                "model_performance": {
                    "row_count": 596802,
                    "feature_count": 44,
                    "mean": 10512.8,
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "artifacts" / "ml" / "evaluation_report.json").write_text(
        json.dumps({"holdout": {"quadratic_weighted_kappa": 0.7770295727962795}}),
        encoding="utf-8",
    )
    contract = {
        "validation_requirements": {"primary_metric": "quadratic_weighted_kappa"},
        "required_outputs": [
            "artifacts/clean/feature_drift_baseline.json",
            "artifacts/ml/evaluation_report.json",
        ],
    }

    payload, path = graph_mod._load_declared_artifact_payload(
        contract,
        {},
        "metrics.json",
        kind="metrics",
    )

    assert path == "artifacts/ml/evaluation_report.json"
    assert payload["holdout"]["quadratic_weighted_kappa"] == pytest.approx(0.7770295727962795)


def test_primary_metric_rejects_physically_impossible_qwk_and_uses_valid_report(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state = {
        "primary_metric_state": {
            "primary_metric_name": "quadratic_weighted_kappa",
            "primary_metric_value": 34.76319999608677,
            "primary_metric_source": "artifact:artifacts/ml/inference_benchmark.json",
        },
        "execution_contract": {
            "validation_requirements": {
                "primary_metric": "quadratic_weighted_kappa",
            }
        },
    }
    metrics_report = {
        "source": "artifact:artifacts/ml/evaluation_report.json",
        "holdout": {
            "quadratic_weighted_kappa": 0.7770295727962795,
        },
    }

    primary = graph_mod._extract_primary_metric_for_board(state, metrics_report)

    assert primary.get("value") == pytest.approx(0.7770295727962795)
    assert primary.get("source") == "artifact:artifacts/ml/evaluation_report.json"
