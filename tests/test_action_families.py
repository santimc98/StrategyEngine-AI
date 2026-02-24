from src.utils.action_families import (
    classify_action_family,
    get_action_family_guidance,
    normalize_action_family,
)


def test_classify_action_family_uses_explicit_family() -> None:
    packet = {
        "action_family": "calibration",
        "hypothesis": {
            "technique": "isotonic post-calibration",
        },
    }
    assert classify_action_family(packet) == "calibration"


def test_classify_action_family_from_technique_keywords() -> None:
    packet = {
        "hypothesis": {
            "technique": "run bounded hyperparameter tuning with optuna",
            "params": {"search": "bayesian"},
        },
    }
    assert classify_action_family(packet) == "hyperparameter_search"


def test_get_action_family_guidance_is_stable() -> None:
    family = normalize_action_family("ensemble_or_stacking")
    guidance = get_action_family_guidance(family)
    assert family == "ensemble_or_stacking"
    assert isinstance(guidance, list) and len(guidance) >= 2
    assert all(isinstance(item, str) and item.strip() for item in guidance)

