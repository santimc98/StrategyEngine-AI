from src.agents.execution_planner import _ensure_feature_engineering_plan_from_strategy


def test_ensure_feature_engineering_plan_from_strategy_maps_object_payload() -> None:
    contract = {}
    strategy = {
        "feature_engineering_strategy": {
            "techniques": [
                {
                    "technique": "interaction",
                    "columns": ["a", "b"],
                    "output_column_name": "a_x_b",
                }
            ],
            "notes": "Priorizar variables estables.",
            "risk_level": "med",
        }
    }

    out = _ensure_feature_engineering_plan_from_strategy(contract, strategy)
    fep = out.get("feature_engineering_plan", {})
    assert isinstance(fep.get("techniques"), list)
    assert fep.get("derived_columns") == ["a_x_b"]
    assert fep.get("notes") == "Priorizar variables estables."


def test_ensure_feature_engineering_plan_from_strategy_backfills_empty_structure() -> None:
    out = _ensure_feature_engineering_plan_from_strategy({}, {})
    assert "feature_engineering_plan" not in out
