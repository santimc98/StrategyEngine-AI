from src.agents import execution_planner as planner


def _datetime_gate_contract(expected_unique_range=None):
    params = {
        "column": "snapshot_month_end",
        "expected_dtype": "datetime",
        "expected_time_span_days_range": [650, 750],
    }
    if expected_unique_range is not None:
        params["expected_unique_range"] = expected_unique_range
    return {
        "data_engineer": {
            "cleaning_gates": [
                {
                    "name": "snapshot_month_end_parsed",
                    "severity": "HARD",
                    "params": params,
                    "evidence_source": "cleaned CSV header and parsed date diagnostics",
                }
            ]
        }
    }


def _mixed_datetime_profile():
    return {
        "dtypes": {"snapshot_month_end": "object"},
        "cardinality": {
            "snapshot_month_end": {
                "unique": 120,
                "top_values": [
                    {"value": "2024/07/31"},
                    {"value": "2024-08-31T18:00:00"},
                    {"value": "12-31-2024"},
                ],
            }
        },
        "text_summary": {
            "snapshot_month_end": {
                "datetime_like_ratio": 0.1984,
            }
        },
    }


def test_gate_parameter_semantics_rejects_raw_datetime_cardinality_as_parsed_unique():
    semantics = planner._validate_gate_parameter_semantics(
        _datetime_gate_contract(expected_unique_range=[115, 125]),
        _mixed_datetime_profile(),
    )

    assert semantics["status"] == "violations"
    violation = semantics["violations"][0]
    assert violation["kind"] == "raw_datetime_cardinality_as_parsed_unique"
    assert violation["severity"] == "hard"
    assert violation["column"] == "snapshot_month_end"

    result = planner._gate_parameter_semantics_validation_result(semantics)
    assert result["accepted"] is False
    assert result["status"] == "error"
    assert result["issues"][0]["rule"] == (
        "contract.gate_parameter_semantics.raw_datetime_cardinality_as_parsed_unique"
    )


def test_gate_parameter_semantics_allows_datetime_gate_without_raw_unique_threshold():
    semantics = planner._validate_gate_parameter_semantics(
        _datetime_gate_contract(expected_unique_range=None),
        _mixed_datetime_profile(),
    )

    assert semantics == {"status": "ok", "violations": []}


def test_gate_parameter_semantics_uses_temporal_normalization_facts():
    profile = {
        "temporal_normalization_facts": [
            {
                "column": "snapshot_month_end",
                "raw_unique_count": 120,
                "has_mixed_raw_formats": True,
                "raw_format_families": ["ymd_dash", "ymd_slash", "ambiguous_mdy_dmy_dash"],
                "canonical_unique_counts": {"timestamp": 48, "date": 24, "month_period": 24},
                "normalization_collapse_risk": "high",
            }
        ]
    }

    semantics = planner._validate_gate_parameter_semantics(
        _datetime_gate_contract(expected_unique_range=[115, 125]),
        profile,
    )

    assert semantics["status"] == "violations"
    assert semantics["violations"][0]["raw_unique"] == 120
