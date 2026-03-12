"""
Tests for contract_validator.py (P1.1 and P1.2).
"""
import pytest
from src.utils.contract_validator import (
    is_file_path,
    is_column_name,
    detect_output_ambiguity,
    normalize_artifact_requirements,
    validate_contract,
)


class TestAmbiguityDetection:
    """Test P1.1: Detect ambiguity between files and columns."""

    def test_is_file_path_with_extension(self):
        assert is_file_path("data/metrics.json") is True
        assert is_file_path("cleaned_data.csv") is True
        assert is_file_path("report.pdf") is True
        assert is_file_path("model.pkl") is True

    def test_is_file_path_with_path_separator(self):
        assert is_file_path("data/file") is True
        assert is_file_path("reports/output") is True

    def test_is_column_name_simple(self):
        assert is_column_name("probability_of_claim") is True
        assert is_column_name("score") is True
        assert is_column_name("id") is True
        assert is_column_name("Age") is True

    def test_is_column_name_rejects_paths(self):
        assert is_column_name("data/file.csv") is False
        assert is_column_name("reports/output.json") is False

    def test_detect_ambiguity_moves_columns_to_required_columns(self):
        """Test that column-like entries in required_outputs are moved."""
        required_outputs = [
            "data/metrics.json",  # File
            "probability_of_claim",  # Column (no extension)
            "data/scored_rows.csv",  # File
            "priority_score",  # Column
        ]

        files, columns, warnings, conceptual_outputs = detect_output_ambiguity(required_outputs)

        assert len(files) == 2
        assert len(columns) == 2
        assert len(warnings) == 2
        assert len(conceptual_outputs) == 0

        file_paths = [f["path"] for f in files]
        assert "data/metrics.json" in file_paths
        assert "data/scored_rows.csv" in file_paths

        col_names = [c["name"] for c in columns]
        assert "probability_of_claim" in col_names
        assert "priority_score" in col_names

    def test_normalize_artifact_requirements_from_legacy(self):
        """Test normalization of legacy required_outputs format."""
        contract = {
            "required_outputs": [
                "data/cleaned_data.csv",
                "data/metrics.json",
                "predicted_value",  # Column, not file
            ]
        }

        artifact_req, warnings = normalize_artifact_requirements(contract)

        # Check that files are properly extracted
        required_files = artifact_req.get("required_files", [])
        file_paths = [f.get("path", "") for f in required_files]
        assert "data/cleaned_data.csv" in file_paths
        assert "data/metrics.json" in file_paths

        # Check that columns are moved to scored_rows_schema
        scored_schema = artifact_req.get("scored_rows_schema", {})
        required_columns = scored_schema.get("required_columns", [])
        assert "predicted_value" in required_columns

        # Should have warning about the move
        assert len(warnings) > 0
        assert any("predicted_value" in w.get("item", "") for w in warnings)

    def test_normalize_artifact_requirements_preserves_rich_output_metadata(self):
        contract = {
            "required_outputs": [
                {"path": "data/metrics.json", "required": True, "owner": "ml_engineer", "kind": "metrics"},
            ],
            "required_output_artifacts": [
                {"path": "data/metrics.json", "required": True, "owner": "ml_engineer", "kind": "metrics"},
            ],
            "spec_extraction": {
                "deliverables": [
                    {"path": "data/metrics.json", "required": True, "owner": "ml_engineer", "kind": "metrics"},
                ]
            },
        }

        normalize_artifact_requirements(contract)

        assert isinstance(contract.get("required_outputs"), list)
        assert isinstance(contract["required_outputs"][0], dict)
        assert isinstance(contract.get("required_output_artifacts"), list)
        assert contract["required_output_artifacts"][0].get("owner") == "ml_engineer"
        spec = contract.get("spec_extraction", {})
        assert isinstance(spec.get("deliverables"), list)
        assert spec["deliverables"][0].get("path") == "data/metrics.json"

    def test_normalize_artifact_requirements_backfills_required_outputs_as_paths(self):
        contract = {
            "artifact_requirements": {
                "required_files": [{"path": "data/custom_output.csv"}],
            }
        }

        normalize_artifact_requirements(contract)

        required_outputs = contract.get("required_outputs", [])
        assert isinstance(required_outputs, list)
        assert all(isinstance(path, str) for path in required_outputs)
        assert "data/custom_output.csv" in required_outputs

    def test_normalize_artifact_requirements_does_not_inject_default_files(self):
        contract = {"artifact_requirements": {}}

        artifact_req, warnings = normalize_artifact_requirements(contract)

        assert warnings == []
        assert artifact_req.get("required_files") == []
        assert "scored_rows_schema" not in artifact_req

    def test_normalize_artifact_requirements_does_not_promote_file_schemas_to_required(self):
        contract = {
            "artifact_requirements": {
                "file_schemas": {"outputs/submission.csv": {"expected_row_count": 95}},
            }
        }

        artifact_req, warnings = normalize_artifact_requirements(contract)

        assert warnings == []
        assert artifact_req.get("required_files") == []
        assert artifact_req.get("file_schemas", {}).get("outputs/submission.csv", {}).get("expected_row_count") == 95


class TestContractValidation:
    """Test P1.2: Contract Self-Consistency Gate."""

    def test_validate_contract_ok(self):
        """Test valid contract passes validation."""
        contract = {
            "canonical_columns": ["col_a", "col_b", "target"],
            "outcome_columns": ["target"],
            "decision_columns": [],  # No action space defined
            "allowed_feature_sets": {
                "core": ["col_a", "col_b"],
            },
            "artifact_requirements": {
                "required_files": [
                    {"path": "data/cleaned_data.csv"},
                    {"path": "data/metrics.json"},
                ],
                "scored_rows_schema": {
                    "required_columns": ["id", "score"],
                },
            },
        }

        result = validate_contract(contract)

        assert result["status"] in ["ok", "warning"]  # No errors

    def test_validate_contract_feature_not_in_canonical(self):
        """Test that features not in canonical_columns trigger warning."""
        contract = {
            "canonical_columns": ["col_a", "col_b"],
            "allowed_feature_sets": {
                "core": ["col_a", "col_c"],  # col_c not in canonical
            },
        }

        result = validate_contract(contract)

        assert result["status"] == "warning"
        issues = result["issues"]
        assert any(
            "col_c" in str(issue.get("item", ""))
            for issue in issues
        )

    def test_validate_contract_accepts_set_alias_when_selectors_exist(self):
        """SET_n aliases are valid when feature selectors are present."""
        contract = {
            "canonical_columns": ["label", "pixel0", "pixel1"],
            "feature_selectors": [{"type": "regex", "pattern": "^pixel\\d+$"}],
            "allowed_feature_sets": {
                "model_features": ["SET_1"],
            },
        }

        result = validate_contract(contract)

        assert not any(
            issue.get("rule") == "feature_set_consistency" and issue.get("item") == "SET_1"
            for issue in result.get("issues", [])
        )

    def test_validate_contract_set_alias_without_selectors_warns(self):
        """SET_n aliases without selector context should still warn."""
        contract = {
            "canonical_columns": ["label", "pixel0", "pixel1"],
            "allowed_feature_sets": {
                "model_features": ["SET_1"],
            },
        }

        result = validate_contract(contract)

        assert any(
            issue.get("rule") == "feature_set_consistency" and issue.get("item") == "SET_1"
            for issue in result.get("issues", [])
        )

    def test_validate_contract_decision_columns_without_levers(self):
        """Test that decision_columns without action_space triggers warning."""
        contract = {
            "canonical_columns": ["col_a", "price"],
            "decision_columns": ["price"],  # No action_space/levers defined
            "business_objective": "Predict sales",  # No lever keywords
        }

        result = validate_contract(contract)

        assert result["status"] == "warning"
        issues = result["issues"]
        assert any(
            "decision_columns" in issue.get("rule", "")
            for issue in issues
        )

    def test_validate_contract_empty_required_files(self):
        """Test that empty required_files triggers warning."""
        contract = {
            "artifact_requirements": {
                "required_files": [],
            },
        }

        result = validate_contract(contract)

        # LLM-first policy: no default business artifacts are injected.
        norm_req = result.get("normalized_artifact_requirements", {})
        assert norm_req.get("required_files", []) == []
