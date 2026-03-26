from src.agents.model_analyst import ModelAnalystAgent
from src.utils.action_families import ACTION_FAMILIES


def test_model_analyst_deterministic_blueprint_is_finalized(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_ANALYST_MODE", "deterministic")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    agent = ModelAnalystAgent(api_key=None)

    blueprint = agent.analyze_baseline(
        {
            "script_code": (
                "from catboost import CatBoostClassifier\n"
                "model = CatBoostClassifier(iterations=800, learning_rate=0.06, depth=8, random_state=42)\n"
            ),
            "metrics": {"n_train_rows": 25000, "n_features": 12},
            "dataset_profile": {
                "n_train_rows": 25000,
                "n_columns": 12,
                "categorical_columns": ["city", "segment"],
            },
            "primary_metric": "auc",
            "models_used": ["catboost"],
        }
    )

    assert blueprint.get("framework") == "catboost"
    assert blueprint.get("model_type") == "gradient_boosting"
    assert blueprint.get("blueprint_version") == "1.0"
    assert isinstance(blueprint.get("timestamp_utc"), str) and blueprint.get("timestamp_utc")
    actions = blueprint.get("improvement_actions")
    assert isinstance(actions, list) and actions
    assert len(actions) <= 6
    assert all((action.get("action_family") in ACTION_FAMILIES) for action in actions if isinstance(action, dict))


def test_model_analyst_validate_blueprint_normalizes_action_fields(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_ANALYST_MODE", "deterministic")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    agent = ModelAnalystAgent(api_key=None)
    payload = {
        "model_type": "gradient_boosting",
        "improvement_actions": [
            {
                "technique": "test_technique",
                "action_family": "not_a_real_family",
                "priority": "9",
            }
        ],
    }

    valid = agent._validate_blueprint(payload)

    assert valid is True
    action = payload["improvement_actions"][0]
    assert action.get("action_family") == "hyperparameter_search"
    assert action.get("priority") == 5


def test_model_analyst_hybrid_without_client_falls_back_to_deterministic(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_ANALYST_MODE", "hybrid")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    agent = ModelAnalystAgent(api_key=None)

    blueprint = agent.analyze_baseline(
        {
            "script_code": "from sklearn.ensemble import RandomForestClassifier\nmodel = RandomForestClassifier(random_state=42)\n",
            "metrics": {"n_train_rows": 1200},
            "dataset_profile": {"n_train_rows": 1200, "n_columns": 8},
            "primary_metric": "accuracy",
        }
    )

    assert blueprint.get("blueprint_version") == "1.0"
    assert isinstance(blueprint.get("improvement_actions"), list)
