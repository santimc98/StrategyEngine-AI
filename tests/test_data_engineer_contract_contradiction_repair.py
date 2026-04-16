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


def test_data_engineer_repair_context_flags_unsupported_datetime_span_gate():
    state = {
        "execution_contract": {
            "data_engineer": {
                "cleaning_gates": [
                    {
                        "name": "datetime_parse_success_account_created_at",
                        "severity": "HARD",
                        "evidence_source": "cleaned CSV dtype of account_created_at and span",
                        "params": {
                            "column": "account_created_at",
                            "required_dtype": "datetime64[ns]",
                            "expected_time_span_days_min": 1010,
                            "expected_time_span_days_max": 1050,
                        },
                    }
                ]
            }
        },
        "dataset_profile": {
            "temporal_normalization_facts": [
                {
                    "column": "account_created_at",
                    "time_span_days": 558.8,
                    "time_span_confidence": "medium",
                    "parse_policy": "format_family_aware_explicit_yearfirst",
                }
            ]
        },
    }
    runtime_output = (
        "ValueError: CLEANING_GATE_FAILED: datetime_parse_success_account_created_at: "
        "account_created_at time span 558 outside [1010, 1050]"
    )

    repair_context = graph_mod._build_repair_ground_truth(
        state=state,
        retry_context={
            "error_type": "runtime_error",
            "repair_focus": "runtime",
            "specific_error": runtime_output,
        },
        runtime_output=runtime_output,
        missing_outputs=["artifacts/clean/account_snapshots_ml_ready.csv"],
        present_outputs=[],
        failed_gates=["runtime_failure"],
        required_fixes=[],
        evidence_focus=[],
    )

    assert repair_context["root_cause_type"] == "contract_contradiction"
    assert repair_context["repair_focus"] == "contract_or_gate_semantics"
    conflict = next(
        item["value"]
        for item in repair_context["verified_facts"]
        if item.get("fact") == "contract_gate_semantic_contradiction"
    )
    assert conflict["gate_name"] == "datetime_parse_success_account_created_at"
    assert conflict["actual_span_days"] == 558.0
    assert conflict["expected_time_span_days_range"] == [1010.0, 1050.0]
    assert conflict["profile_time_span_days"] == 558.8
    assert conflict["time_span_confidence"] == "medium"
    assert any("Do not keep enforcing" in item for item in repair_context["repair_directives"])


def test_de_heavy_error_kind_treats_cleaning_gate_failure_as_code_error():
    kind = graph_mod._classify_de_heavy_runtime_error_kind(
        error_kind_hint="code_or_infra_unknown",
        heavy_script_error=None,
        runtime_error_text=(
            "HEAVY_RUNNER_ERROR: missing_artifacts=['artifacts/clean/out.csv']\n"
            "Traceback (most recent call last):\n"
            "ValueError: CLEANING_GATE_FAILED: datetime_parse_success_account_created_at"
        ),
        heavy_error="",
    )

    assert kind == "code"
