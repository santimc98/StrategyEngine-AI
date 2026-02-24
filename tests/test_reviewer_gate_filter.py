from src.agents.reviewer import apply_reviewer_gate_filter


def test_reviewer_gate_filter_handles_dict_gate_specs() -> None:
    result = {
        "status": "REJECTED",
        "feedback": "gate failures",
        "failed_gates": ["decision_policy_feasibility", "unknown_gate"],
        "required_fixes": [],
    }
    reviewer_gates = [
        {"gate": "decision_policy_feasibility", "condition": "policy thresholds defined"},
        {"gate": "driver_transparency", "condition": "drivers documented"},
    ]

    filtered = apply_reviewer_gate_filter(result, reviewer_gates)

    assert filtered["status"] == "REJECTED"
    assert filtered["failed_gates"] == ["decision_policy_feasibility"]


def test_reviewer_gate_filter_downgrades_when_no_allowed_gate_fails() -> None:
    result = {
        "status": "REJECTED",
        "feedback": "only unknown gate",
        "failed_gates": ["not_in_contract"],
        "hard_failures": ["not_in_contract"],
        "required_fixes": [],
    }
    reviewer_gates = [
        {"gate": "decision_policy_feasibility"},
    ]

    filtered = apply_reviewer_gate_filter(result, reviewer_gates)

    assert filtered["status"] == "APPROVE_WITH_WARNINGS"
    assert filtered["failed_gates"] == []
    assert filtered["hard_failures"] == []
    assert "NON_ACTIVE_GATE_WARNINGS" in str(filtered["feedback"])
