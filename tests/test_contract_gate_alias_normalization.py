from src.utils.contract_accessors import (
    filter_gate_list_for_phase,
    get_cleaning_gates,
    get_qa_gates,
    get_reviewer_gates,
)


def test_get_qa_gates_accepts_metric_alias_and_preserves_params():
    contract = {
        "qa_gates": [
            {"metric": "roc_auc", "severity": "hard", "threshold": 0.82},
            {"name": "stability_check", "severity": "SOFT"},
        ]
    }

    gates = get_qa_gates(contract)

    assert len(gates) == 2
    assert gates[0]["name"] == "roc_auc"
    assert gates[0]["severity"] == "HARD"
    assert gates[0]["params"].get("metric") == "roc_auc"
    assert gates[0]["params"].get("threshold") == 0.82


def test_get_reviewer_gates_accepts_check_alias():
    contract = {
        "reviewer_gates": [
            {"check": "artifact_completeness", "severity": "HARD"},
            {"rule": "contract_alignment", "severity": "SOFT"},
        ]
    }

    gates = get_reviewer_gates(contract)

    assert len(gates) == 2
    assert [g["name"] for g in gates] == ["artifact_completeness", "contract_alignment"]


def test_get_cleaning_gates_accepts_rule_alias():
    contract = {"cleaning_gates": [{"rule": "required_columns_present"}]}

    gates = get_cleaning_gates(contract)

    assert len(gates) == 1
    assert gates[0]["name"] == "required_columns_present"
    assert gates[0]["severity"] == "HARD"


def test_filter_gate_list_for_phase_drops_baseline_only_reviewer_gates_in_metric_round():
    reviewer_gates = [
        {"name": "baseline_simplicity_enforcement", "severity": "HARD", "params": {}},
        {"name": "model_selection_priority", "severity": "SOFT", "params": {"primary": "CatBoostClassifier"}},
        {"name": "submission_schema_compliance", "severity": "HARD", "params": {}},
    ]

    filtered = filter_gate_list_for_phase(reviewer_gates, "metric_round", actor="reviewer")

    assert [gate["name"] for gate in filtered] == ["submission_schema_compliance"]


def test_filter_gate_list_for_phase_respects_explicit_metric_round_metadata():
    reviewer_gates = [
        {
            "name": "optimization_budget_guard",
            "severity": "HARD",
            "params": {},
            "applies_to": ["metric_round"],
        },
        {
            "name": "baseline_simplicity_enforcement",
            "severity": "HARD",
            "params": {},
            "applies_to": ["baseline_only"],
        },
    ]

    filtered = filter_gate_list_for_phase(reviewer_gates, "metric_round", actor="reviewer")

    assert [gate["name"] for gate in filtered] == ["optimization_budget_guard"]
