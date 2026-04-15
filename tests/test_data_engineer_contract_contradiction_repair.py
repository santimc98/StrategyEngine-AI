from src.graph import graph as graph_mod


def test_data_engineer_repair_context_flags_impossible_cleaning_gate():
    state = {
        "execution_contract": {
            "data_engineer": {
                "cleaning_gates": [
                    {
                        "name": "snapshot_month_end_parsed",
                        "severity": "HARD",
                        "params": {
                            "column": "snapshot_month_end",
                            "expected_dtype": "datetime",
                            "expected_unique_range": [115, 125],
                            "expected_time_span_days_range": [650, 750],
                        },
                    }
                ]
            }
        },
        "dataset_profile": {
            "dtypes": {"snapshot_month_end": "object"},
            "cardinality": {
                "snapshot_month_end": {
                    "unique": 120,
                }
            },
        },
    }
    runtime_output = (
        "ValueError: CLEANING_GATE_FAILED: snapshot_month_end_parsed - "
        "dtype=datetime64[ns], unique=24, span=700"
    )

    repair_context = graph_mod._build_repair_ground_truth(
        state=state,
        retry_context={
            "error_type": "shape_or_dtype",
            "repair_focus": "runtime",
            "specific_error": runtime_output,
        },
        runtime_output=runtime_output,
        missing_outputs=["artifacts/clean/dataset_clean_with_features.csv"],
        present_outputs=[],
        failed_gates=["runtime_failure"],
        required_fixes=[],
        evidence_focus=[],
    )

    assert repair_context["root_cause_type"] == "contract_contradiction"
    assert repair_context["repair_focus"] == "contract_or_gate_semantics"
    facts = repair_context["verified_facts"]
    conflict = next(
        item["value"]
        for item in facts
        if item.get("fact") == "contract_gate_semantic_contradiction"
    )
    assert conflict["gate_name"] == "snapshot_month_end_parsed"
    assert conflict["actual_unique"] == 24
    assert conflict["raw_unique"] == 120
    assert conflict["expected_unique_range"] == [115.0, 125.0]
