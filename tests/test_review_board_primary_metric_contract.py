from src.graph import graph as graph_mod
import pytest


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
