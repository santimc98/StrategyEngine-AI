from src.utils.contract_accessors import get_outcome_columns


def test_get_outcome_columns_reads_validation_requirements_target_column():
    contract = {
        "validation_requirements": {
            "target_column": "target",
        }
    }
    assert get_outcome_columns(contract) == ["target"]


def test_get_outcome_columns_prefers_explicit_outcome_columns():
    contract = {
        "outcome_columns": ["claim_flag"],
        "validation_requirements": {
            "target_column": "target",
        },
    }
    assert get_outcome_columns(contract) == ["claim_flag"]


def test_get_outcome_columns_reads_task_semantics_targets():
    contract = {
        "task_semantics": {
            "primary_target": "label_24h",
            "target_columns": ["label_12h", "label_24h", "label_48h", "label_72h"],
        }
    }

    assert get_outcome_columns(contract) == ["label_12h", "label_24h", "label_48h", "label_72h"]
