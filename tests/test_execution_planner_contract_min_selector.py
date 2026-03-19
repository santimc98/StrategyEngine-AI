from src.agents.execution_planner import (
    _apply_schema_coercion,
    ExecutionPlannerAgent,
    build_contract_min,
    select_relevant_columns,
)
from src.utils.contract_validator import validate_contract_minimal_readonly


class DummyResponse:
    def __init__(self, text: str) -> None:
        self.text = text
        self.candidates = []
        self.usage_metadata = None


class DummyClient:
    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.calls = 0

    def generate_content(self, _prompt: str) -> DummyResponse:
        idx = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        return DummyResponse(self._responses[idx])


def test_select_relevant_columns_compact() -> None:
    inventory = [f"col_{i}" for i in range(200)]
    strategy = {
        "required_columns": ["col_1", "col_5", "col_10", "col_20", "col_30", "col_40"],
        "decision_variables": ["col_50"],
        "target_column": "col_60",
    }
    payload = select_relevant_columns(
        strategy=strategy,
        business_objective='Use "col_70" for analysis.',
        domain_expert_critique="",
        column_inventory=inventory,
        data_profile_summary="",
    )
    relevant = payload["relevant_columns"]
    assert set(strategy["required_columns"]).issubset(set(relevant))
    assert len(relevant) <= 30

    contract_min = build_contract_min({}, strategy, inventory, relevant)
    assert contract_min["canonical_columns"]
    assert contract_min["artifact_requirements"]
    assert contract_min["qa_gates"]
    assert contract_min["column_roles"].get("unknown") == []


def test_select_relevant_columns_wide_family_uses_compact_projection() -> None:
    inventory = ["label", "__split"] + [f"pixel_{i}" for i in range(1200)]
    strategy = {
        "required_columns": ["label", "__split"],
        "feature_families": [{"family": "pixel", "selector_hint": "pixel_*"}],
    }
    payload = select_relevant_columns(
        strategy=strategy,
        business_objective="Clasificar imágenes con píxeles.",
        domain_expert_critique="",
        column_inventory=inventory,
        data_profile_summary="",
    )
    relevant = payload["relevant_columns"]
    assert "label" in relevant
    assert "__split" in relevant
    assert payload.get("relevant_columns_truncated") is True
    assert int(payload.get("relevant_columns_total_count") or 0) > len(relevant)
    assert int(payload.get("relevant_columns_omitted_count") or 0) >= 1
    assert "pixel_*" in (payload.get("strategy_feature_family_hints") or [])


def test_execution_planner_invalid_json_failure_does_not_invent_deterministic_contract() -> None:
    agent = ExecutionPlannerAgent(api_key=None)
    agent.client = DummyClient(
        [
            '{"contract_version": 2, "rationale":',
            '{"contract_version": 2, "rationale":',
        ]
    )
    inventory = [f"col_{i}" for i in range(10)]
    strategy = {"required_columns": ["col_1", "col_2"]}

    contract = agent.generate_contract(
        strategy=strategy,
        data_summary="",
        business_objective="Test objective",
        column_inventory=inventory,
        output_dialect={"sep": ",", "decimal": ".", "encoding": "utf-8"},
        env_constraints={"forbid_inplace_column_creation": True},
        domain_expert_critique="",
    )

    assert isinstance(contract, dict)
    assert contract == {}
    diagnostics = agent.last_contract_diagnostics or {}
    summary = diagnostics.get("summary") if isinstance(diagnostics, dict) else {}
    assert summary.get("accepted") is False
    transport = diagnostics.get("transport_validation") if isinstance(diagnostics, dict) else {}
    assert isinstance(transport, dict)
    assert transport.get("accepted") is False
    assert agent.last_contract_min is None


def test_contract_min_inherits_roles_from_full() -> None:
    inventory = ["A", "B", "C", "D", "E"]
    strategy = {"required_columns": ["A", "B", "C", "D"]}
    full_contract = {
        "business_objective": "Test objective",
        "column_roles": {
            "pre_decision": ["A"],
            "decision": ["B"],
            "outcome": ["C"],
            "post_decision_audit_only": ["D"],
        },
    }
    relevant = ["A", "B", "C", "D"]
    contract_min = build_contract_min(full_contract, strategy, inventory, relevant)
    roles = contract_min.get("column_roles", {})
    assert roles.get("pre_decision") == ["A"]
    assert roles.get("decision") == ["B"]
    assert roles.get("outcome") == ["C"]
    assert roles.get("post_decision_audit_only") == ["D"]
    allowed = contract_min.get("allowed_feature_sets", {})
    assert allowed.get("segmentation_features") == ["A"]
    assert allowed.get("model_features") == ["A", "B"]
    assert set(allowed.get("forbidden_features") or []) == {"C", "D"}


def test_contract_min_normalizes_column_role_maps() -> None:
    inventory = ["ColA", "ColB", "ColC", "ColD"]
    strategy = {"required_columns": ["ColA", "ColB", "ColC", "ColD"]}
    full_contract = {
        "column_roles": {
            "ColA": {"role": "pre_decision"},
            "ColB": {"role": "decision"},
            "ColC": {"role": "outcome"},
            "ColD": {"role": "post_decision_audit_only"},
        }
    }
    relevant = ["ColA", "ColB", "ColC", "ColD"]
    contract_min = build_contract_min(full_contract, strategy, inventory, relevant)
    roles = contract_min.get("column_roles", {})
    assert roles.get("pre_decision") == ["ColA"]
    assert roles.get("decision") == ["ColB"]
    assert roles.get("outcome") == ["ColC"]
    assert roles.get("post_decision_audit_only") == ["ColD"]


def test_contract_min_inherits_scored_rows_schema_required_columns() -> None:
    inventory = ["PassengerId", "Age"]
    strategy = {"required_columns": ["PassengerId", "Age"]}
    full_contract = {
        "artifact_requirements": {
            "required_files": [{"path": "data/scored_rows.csv"}],
            "scored_rows_schema": {"required_columns": ["Individual_Triage_List_CSV"]},
        }
    }
    relevant = ["PassengerId", "Age"]
    contract_min = build_contract_min(full_contract, strategy, inventory, relevant)
    schema = contract_min.get("artifact_requirements", {}).get("scored_rows_schema", {})
    required_cols = schema.get("required_columns", [])
    assert "Individual_Triage_List_CSV" in required_cols
    assert any(path == "data/scored_rows.csv" for path in contract_min.get("required_outputs", []))


def test_contract_min_includes_decisioning_required_columns() -> None:
    inventory = ["id", "feature_a"]
    strategy = {"required_columns": ["id", "feature_a"]}
    full_contract = {
        "decisioning_requirements": {
            "enabled": True,
            "required": True,
            "output": {
                "file": "data/scored_rows.csv",
                "required_columns": [
                    {"name": "priority_score", "role": "score"},
                    {"name": "priority_rank", "role": "priority"},
                    {"name": "explanation", "role": "explanation"},
                ],
            },
        },
        "artifact_requirements": {
            "required_files": [{"path": "data/scored_rows.csv"}],
        },
    }
    contract_min = build_contract_min(full_contract, strategy, inventory, inventory)
    schema = contract_min.get("artifact_requirements", {}).get("scored_rows_schema", {})
    required_cols = schema.get("required_columns", [])
    assert "priority_score" in required_cols
    assert "priority_rank" in required_cols
    assert "explanation" in required_cols


def test_contract_min_adds_prediction_like_required_to_anyof_group() -> None:
    inventory = ["id", "feature_a"]
    strategy = {"required_columns": ["id", "feature_a"]}
    full_contract = {
        "artifact_requirements": {
            "required_files": [{"path": "data/scored_rows.csv"}],
            "scored_rows_schema": {"required_columns": ["survival_probability"]},
        }
    }
    contract_min = build_contract_min(full_contract, strategy, inventory, inventory)
    schema = contract_min.get("artifact_requirements", {}).get("scored_rows_schema", {})
    groups = schema.get("required_any_of_groups", [])
    assert any(
        isinstance(group, list) and "survival_probability" in group
        for group in groups
    )


def test_contract_min_aligns_decisioning_explanation_name() -> None:
    inventory = ["id", "feature_a"]
    strategy = {"required_columns": ["id", "feature_a"]}
    full_contract = {
        "decisioning_requirements": {
            "enabled": True,
            "required": True,
            "output": {
                "file": "data/scored_rows.csv",
                "required_columns": [
                    {"name": "top_drivers", "role": "explanation"},
                ],
            },
        },
        "artifact_requirements": {
            "required_files": [{"path": "data/scored_rows.csv"}],
            "scored_rows_schema": {"required_columns": ["explanation"]},
        },
    }
    contract_min = build_contract_min(full_contract, strategy, inventory, inventory)
    decisioning = contract_min.get("decisioning_requirements", {})
    required_cols = decisioning.get("output", {}).get("required_columns", [])
    assert any(
        isinstance(entry, dict) and entry.get("name") == "explanation"
        for entry in required_cols
    )


def test_contract_min_infers_expected_row_count_from_profile_hints() -> None:
    inventory = ["id", "is_train", "feature_a"]
    strategy = {"required_columns": ["id", "is_train", "feature_a"]}
    row_hints = {"n_train_rows": 630000, "n_test_rows": 270000}
    full_contract = {
        "required_output_artifacts": [
            {"path": "data/scored_rows.csv", "kind": "predictions", "required": True},
            {"path": "data/submission.csv", "kind": "submission", "required": True},
        ],
        "artifact_requirements": {
            "required_files": [
                {"path": "data/scored_rows.csv"},
                {"path": "data/submission.csv"},
            ],
        },
        "dataset_profile": row_hints,
    }

    contract_min = build_contract_min(
        full_contract,
        strategy,
        inventory,
        inventory,
        data_profile=row_hints,
    )
    file_schemas = contract_min.get("artifact_requirements", {}).get("file_schemas", {})
    assert file_schemas.get("data/submission.csv", {}).get("expected_row_count") == 270000
    assert file_schemas.get("data/scored_rows.csv", {}).get("expected_row_count") == 900000


def test_contract_min_resolves_expected_row_count_alias_tokens() -> None:
    inventory = ["id", "is_train", "feature_a"]
    strategy = {"required_columns": ["id", "is_train", "feature_a"]}
    row_hints = {"n_train_rows": 10, "n_test_rows": 4}
    full_contract = {
        "artifact_requirements": {
            "required_files": [{"path": "data/submission.csv"}],
            "file_schemas": {"data/submission.csv": {"expected_row_count": "n_test"}},
        },
        "dataset_profile": row_hints,
    }

    contract_min = build_contract_min(
        full_contract,
        strategy,
        inventory,
        inventory,
        data_profile=row_hints,
    )
    file_schemas = contract_min.get("artifact_requirements", {}).get("file_schemas", {})
    assert file_schemas.get("data/submission.csv", {}).get("expected_row_count") == 4


def test_contract_min_infers_expected_row_count_when_deliverable_kind_is_missing() -> None:
    inventory = ["id", "is_train", "feature_a"]
    strategy = {"required_columns": ["id", "is_train", "feature_a"]}
    row_hints = {"n_train_rows": 630000, "n_test_rows": 270000}
    full_contract = {
        "required_output_artifacts": [
            {"path": "data/submission.csv", "required": True},
            {"path": "data/scored_rows.csv", "required": True},
        ],
        "spec_extraction": {
            "deliverables": [
                {"path": "data/submission.csv", "required": True},
                {"path": "data/scored_rows.csv", "required": True},
            ]
        },
        "artifact_requirements": {
            "required_files": [
                {"path": "data/submission.csv"},
                {"path": "data/scored_rows.csv"},
            ]
        },
        "dataset_profile": row_hints,
    }

    contract_min = build_contract_min(
        full_contract,
        strategy,
        inventory,
        inventory,
        data_profile=row_hints,
    )
    file_schemas = contract_min.get("artifact_requirements", {}).get("file_schemas", {})
    assert file_schemas.get("data/submission.csv", {}).get("expected_row_count") == 270000
    assert file_schemas.get("data/scored_rows.csv", {}).get("expected_row_count") == 900000


def test_contract_min_infers_row_hints_from_outcome_analysis_counts() -> None:
    inventory = ["id", "feature_a", "target", "is_train"]
    strategy = {"required_columns": ["id", "feature_a", "target", "is_train"]}
    data_profile = {
        "basic_stats": {"n_rows": 900000},
        "outcome_analysis": {
            "target": {
                "non_null_count": 630000,
                "total_count": 900000,
                "null_frac": 0.3,
            }
        },
        "split_candidates": [{"column": "is_train", "unique_values_sample": ["1", "0"]}],
    }
    full_contract = {
        "outcome_columns": ["target"],
        "required_outputs": ["data/submission.csv", "data/scored_rows.csv"],
        "artifact_requirements": {"required_files": [{"path": "data/submission.csv"}, {"path": "data/scored_rows.csv"}]},
    }

    contract_min = build_contract_min(
        full_contract,
        strategy,
        inventory,
        inventory,
        data_profile=data_profile,
    )
    file_schemas = contract_min.get("artifact_requirements", {}).get("file_schemas", {})
    split_spec = contract_min.get("split_spec", {})

    assert file_schemas.get("data/submission.csv", {}).get("expected_row_count") == 270000
    assert file_schemas.get("data/scored_rows.csv", {}).get("expected_row_count") == 900000
    assert split_spec.get("n_train_rows") == 630000
    assert split_spec.get("n_test_rows") == 270000
    assert split_spec.get("training_rows_policy") == "only_rows_with_label"


def test_contract_min_infers_expected_rows_from_required_outputs_paths_only() -> None:
    inventory = ["id", "feature_a", "target"]
    strategy = {"required_columns": ["id", "feature_a", "target"]}
    full_contract = {
        "required_outputs": ["data/submission.csv", "data/scored_rows.csv"],
        "artifact_requirements": {"required_files": [{"path": "data/submission.csv"}, {"path": "data/scored_rows.csv"}]},
        "dataset_profile": {"n_train_rows": 630000, "n_test_rows": 270000},
    }

    contract_min = build_contract_min(
        full_contract,
        strategy,
        inventory,
        inventory,
        data_profile={"n_train_rows": 630000, "n_test_rows": 270000},
    )
    file_schemas = contract_min.get("artifact_requirements", {}).get("file_schemas", {})
    assert file_schemas.get("data/submission.csv", {}).get("expected_row_count") == 270000
    assert file_schemas.get("data/scored_rows.csv", {}).get("expected_row_count") == 900000


def test_contract_min_preserves_multi_output_targets_and_declared_selectors() -> None:
    inventory = [
        "event_id",
        "__split",
        "label_12h",
        "label_24h",
        "label_48h",
        "label_72h",
        "feature_a",
        "feature_b",
    ]
    strategy = {
        "required_columns": inventory,
        "target_columns": ["label_24h"],
        "title": "Probabilidades 12h 24h 48h 72h",
    }
    full_contract = {
        "business_objective": (
            "Predecir probabilidades de impacto a 12h, 24h, 48h y 72h "
            "en el formato oficial de submission."
        ),
        "evaluation_spec": {"objective_type": "multi_output_classification"},
        "artifact_requirements": {
            "clean_dataset": {
                "required_feature_selectors": [
                    {
                        "type": "all_numeric_except",
                        "value": [
                            "event_id",
                            "__split",
                            "label_12h",
                            "label_24h",
                            "label_48h",
                            "label_72h",
                        ],
                    }
                ]
            }
        },
    }
    data_profile = {
        "dataset_semantics": {
            "primary_target": "label_24h",
            "notes": [
                "The analytical task requires predicting all four label horizons."
            ],
        }
    }

    contract_min = build_contract_min(
        full_contract,
        strategy,
        inventory,
        inventory,
        data_profile=data_profile,
        business_objective_hint=full_contract["business_objective"],
    )

    expected_targets = ["label_12h", "label_24h", "label_48h", "label_72h"]
    assert contract_min.get("outcome_columns") == expected_targets
    assert contract_min.get("target_columns") == expected_targets
    assert contract_min.get("target_column") == "label_24h"
    task_semantics = contract_min.get("task_semantics") or {}
    assert task_semantics.get("problem_family") == "classification"
    assert task_semantics.get("multi_target") is True
    assert task_semantics.get("target_columns") == expected_targets
    assert (task_semantics.get("prediction_unit") or {}).get("kind") == "row"
    selectors = (
        contract_min.get("artifact_requirements", {})
        .get("clean_dataset", {})
        .get("required_feature_selectors", [])
    )
    # Post-migration: selectors pass through with minimal coercion (type inferred),
    # no heavy normalization (value→except_columns rename, name synthesis).
    assert len(selectors) == 1
    assert selectors[0]["type"] == "all_numeric_except"


def test_contract_min_canonicalizes_scale_selector_refs_and_selector_drop_anchors() -> None:
    inventory = ["event_id", "__split", "target", "feature_a", "feature_b", "feature_c"]
    strategy = {"required_columns": inventory, "target_column": "target"}
    full_contract = {
        "canonical_columns": inventory,
        "column_roles": {
            "pre_decision": ["feature_a", "feature_b", "feature_c"],
            "decision": [],
            "outcome": ["target"],
            "identifiers": ["event_id"],
            "unknown": [],
        },
        "allowed_feature_sets": {
            "segmentation_features": ["feature_a", "feature_b", "feature_c"],
            "model_features": ["feature_a", "feature_b", "feature_c"],
            "forbidden_features": ["target"],
            "audit_only_features": [],
        },
        "artifact_requirements": {
            "clean_dataset": {
                "required_columns": ["event_id", "__split", "target", "feature_a"],
                "required_feature_selectors": [
                    {"type": "regex", "pattern": "^feature_[abc]$"}
                ],
                "column_transformations": {
                    "scale_columns": ["selector:regex:^feature_[abc]$"],
                    "drop_policy": {
                        "allow_selector_drops_when": (
                            "Columns outside anchors may be dropped when constant or duplicate."
                        )
                    },
                },
            }
        },
        "cleaning_gates": [
            {
                "name": "feature_b_present",
                "severity": "HARD",
                "params": {"columns": ["feature_b"]},
            }
        ],
    }

    contract_min = build_contract_min(full_contract, strategy, inventory, inventory)
    clean_dataset = (contract_min.get("artifact_requirements") or {}).get("clean_dataset") or {}
    selectors = clean_dataset.get("required_feature_selectors") or []
    transforms = clean_dataset.get("column_transformations") or {}

    # Post-migration: selectors pass through with type inference only.
    # Heavy normalization (regex→list conversion, name synthesis, scale_columns
    # canonicalization) is no longer applied deterministically.
    assert len(selectors) >= 1
    assert selectors[0]["type"] == "regex"



def test_schema_coercion_normalizes_scope_and_version() -> None:
    """Test that _apply_schema_coercion handles scope aliases and version."""
    contract = {
        "scope": "cleaning",
        "optimization_policy": {"enabled": "true", "max_rounds": "3"},
        "derived_columns": {"col_a": "source_a", "col_b": "source_b"},
    }
    result = _apply_schema_coercion(contract)
    assert result["scope"] == "cleaning_only"
    assert result["derived_columns"] == ["col_a", "col_b"]
    assert result["optimization_policy"]["enabled"] is True
    assert result["optimization_policy"]["max_rounds"] == 3
    assert "contract_version" in result

