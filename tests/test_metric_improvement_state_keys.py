from src.graph.graph import AgentState


def test_agent_state_declares_metric_improvement_persistence_keys() -> None:
    annotations = getattr(AgentState, "__annotations__", {})
    required_keys = [
        "primary_metric_state",
        "primary_metric_snapshot",
        "metric_loop_state",
        "metric_history",
        "ml_improvement_round_active",
        "ml_improvement_continue",
        "ml_improvement_loop_complete",
        "ml_improvement_attempted",
        "ml_improvement_round_count",
        "ml_improvement_rounds_allowed",
        "ml_improvement_current_round_id",
        "ml_improvement_hypothesis_packet",
        "ml_improvement_critique_packet",
        "ml_improvement_candidate_critique_packet",
    ]
    missing = [key for key in required_keys if key not in annotations]
    assert not missing, f"Missing AgentState keys: {missing}"
