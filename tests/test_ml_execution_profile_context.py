from src.graph.graph import _build_ml_execution_profile_for_prompt


def test_ml_execution_profile_has_runtime_budget_shape():
    state = {
        "dataset_scale_hints": {
            "scale": "large",
            "est_rows": 1_250_000,
            "file_mb": 320.5,
            "n_cols": 45,
        },
        "dataset_scale": "large",
    }
    profile = _build_ml_execution_profile_for_prompt(state, code="", ml_plan={})

    assert isinstance(profile, dict)
    assert profile.get("backend") in {"cloudrun", "local"}
    runtime_budget = profile.get("runtime_budget")
    assert isinstance(runtime_budget, dict)
    assert isinstance(runtime_budget.get("hard_timeout_seconds"), int)
    assert "adaptation_order" in profile
    assert isinstance(profile.get("adaptation_order"), list)
    assert profile.get("objective")

