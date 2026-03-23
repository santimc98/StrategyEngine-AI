from src.utils.artifact_obligations import build_data_engineer_artifact_obligations


def test_build_data_engineer_artifact_obligations_extracts_declared_bindings_with_traceability():
    contract = {
        "artifact_requirements": {
            "cleaned_dataset": {
                "output_path": "artifacts/clean/cleaned_dataset.csv",
                "output_manifest_path": "artifacts/clean/cleaning_manifest.json",
                "required_columns": ["lead_id", "created_at"],
                "optional_passthrough_columns": ["raw_event_ts"],
                "column_transformations": {
                    "drop_columns": ["internal_debug_flag"],
                },
            },
            "enriched_dataset": {
                "output_path": "artifacts/clean/enriched_dataset.csv",
                "required_columns": ["created_at", "score_target"],
            },
            "schema_binding": {
                "output_path": "artifacts/schema/schema_contract_for_next_run.json",
                "required_columns": ["created_at", "score_target"],
            },
        }
    }

    obligations = build_data_engineer_artifact_obligations(contract)

    assert obligations["role"] == "data_engineer"
    bindings = {entry["binding_name"]: entry for entry in obligations["artifact_bindings"]}
    assert set(bindings) == {"cleaned_dataset", "enriched_dataset", "schema_binding"}
    cleaned = bindings["cleaned_dataset"]
    assert cleaned["binding_contract_key"] == "cleaned_dataset"
    assert cleaned["source_contract_path"] == "artifact_requirements.cleaned_dataset"
    assert cleaned["declared_binding"]["optional_passthrough_columns"] == ["raw_event_ts"]
    assert (
        cleaned["field_source_paths"]["output_path"]
        == "artifact_requirements.cleaned_dataset.output_path"
    )
    assert (
        cleaned["field_source_paths"]["column_transformations.drop_columns"]
        == "artifact_requirements.cleaned_dataset.column_transformations.drop_columns"
    )


def test_build_data_engineer_artifact_obligations_preserves_legacy_clean_dataset_alias_without_inventing_new_fields():
    contract = {
        "artifact_requirements": {
            "clean_dataset": {
                "output_path": "artifacts/clean/dataset_limpio.csv",
                "required_columns": ["lead_id", "feature_a"],
            }
        }
    }

    obligations = build_data_engineer_artifact_obligations(contract)

    bindings = obligations["artifact_bindings"]
    assert len(bindings) == 1
    assert bindings[0]["binding_name"] == "cleaned_dataset"
    assert bindings[0]["binding_contract_key"] == "clean_dataset"
    assert bindings[0]["source_contract_path"] == "artifact_requirements.clean_dataset"
    assert "must_not_contain" not in bindings[0]


def test_build_data_engineer_artifact_obligations_returns_empty_when_contract_has_no_de_bindings():
    assert build_data_engineer_artifact_obligations({"artifact_requirements": {"scored_rows_schema": {}}}) == {}
