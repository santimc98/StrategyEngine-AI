from src.graph import graph as graph_module


def test_infer_problem_type_for_heavy_detects_survival_analysis():
    contract = {
        "objective_analysis": {"problem_type": "survival_analysis"},
        "validation_requirements": {
            "primary_metric": "concordance_index",
            "metrics_to_report": ["integrated_brier_score", "mae_uncensored"],
        },
        "evaluation_spec": {"problem_type": "survival_analysis"},
    }

    problem_type = graph_module._infer_problem_type_for_heavy(
        contract,
        contract["evaluation_spec"],
        {},
        "target_json",
    )

    assert problem_type == "survival_analysis"


def test_validate_selected_strategy_executability_uses_runtime_supported_deps(monkeypatch):
    strategy = {
        "techniques": ["survival analysis with Cox proportional hazards"],
        "required_columns": ["features_json"],
        "estimated_difficulty": "medium",
    }

    monkeypatch.setattr(graph_module.importlib.util, "find_spec", lambda _name: None)

    result = graph_module._validate_selected_strategy_executability(
        strategy,
        review={"score": 8.0},
        compute_constraints={"runtime_mode": "local"},
    )

    assert result["ok"] is True
    assert result["blockers"] == []
    assert any(
        str(warning).startswith("runtime_dependency_install:")
        for warning in (result.get("warnings") or [])
    )
    assert result.get("dependency_backend") == "cloudrun"
