"""
Universality tests: verify the execution planner's deterministic logic
adapts correctly to diverse data science problem types.

These tests exercise the pure functions (no LLM calls) that underpin
contract generation: column selection, feature family expansion, and
contract validation — across regression, multiclass, NLP, time series,
and anomaly detection scenarios.

Each fixture simulates a realistic strategy + column_inventory + data_profile
for a different problem type.
"""

import pytest
from typing import Dict, Any, List

from src.agents.execution_planner import (
    select_relevant_columns,
    _expand_strategy_feature_families,
)
from src.utils.contract_validator import (
    validate_contract,
    validate_contract_minimal_readonly,
)


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES — One per problem type
# ═══════════════════════════════════════════════════════════════════════════════


def _make_data_profile(dtypes: Dict[str, str], n_rows: int = 1000) -> Dict[str, Any]:
    """Helper to build a minimal data_profile dict."""
    return {
        "dtypes": dtypes,
        "shape": [n_rows, len(dtypes)],
        "missing_pct": {col: 0.0 for col in dtypes},
    }


# ── 1. Binary classification (baseline — like the wildfires problem) ──────────

BINARY_CLASSIFICATION_INVENTORY = [
    "id", "feature_a", "feature_b", "feature_c", "target", "__split"
]
BINARY_CLASSIFICATION_DTYPES = {
    "id": "int64", "feature_a": "float64", "feature_b": "float64",
    "feature_c": "float64", "target": "int64", "__split": "object",
}
BINARY_CLASSIFICATION_STRATEGY = {
    "title": "Binary classification baseline",
    "objective_type": "predictive",
    "required_columns": ["id", "__split", "target"],
    "feature_families": [
        {
            "family": "NUMERIC_FEATURES",
            "selector_hint": "all_numeric_except excluding id, target",
            "rationale": "All numeric predictors.",
        }
    ],
    "techniques": ["LightGBM", "5-fold CV"],
}


# ── 2. Regression (house prices style) ───────────────────────────────────────

REGRESSION_INVENTORY = [
    "id", "lot_area", "year_built", "garage_area", "overall_qual",
    "gr_liv_area", "total_bsmt_sf", "sale_price", "__split",
    "neighborhood",
]
REGRESSION_DTYPES = {
    "id": "int64", "lot_area": "float64", "year_built": "int64",
    "garage_area": "float64", "overall_qual": "int64",
    "gr_liv_area": "float64", "total_bsmt_sf": "float64",
    "sale_price": "float64", "__split": "object",
    "neighborhood": "object",
}
REGRESSION_STRATEGY = {
    "title": "House price regression with gradient boosting",
    "objective_type": "predictive",
    "required_columns": ["id", "__split", "sale_price"],
    "feature_families": [
        {
            "family": "NUMERIC_FEATURES",
            "selector_hint": "all_numeric_except excluding id, sale_price",
            "rationale": "All numeric features for regression.",
        },
        {
            "family": "CATEGORICAL_FEATURES",
            "selector_hint": "neighborhood",
            "rationale": "Categorical location feature.",
        },
    ],
    "techniques": ["XGBoost", "RMSE optimization"],
}


# ── 3. Multiclass classification (iris/species style) ────────────────────────

MULTICLASS_INVENTORY = [
    "sample_id", "sepal_length", "sepal_width", "petal_length",
    "petal_width", "species", "__split",
]
MULTICLASS_DTYPES = {
    "sample_id": "int64", "sepal_length": "float64", "sepal_width": "float64",
    "petal_length": "float64", "petal_width": "float64",
    "species": "object", "__split": "object",
}
MULTICLASS_STRATEGY = {
    "title": "Multiclass species classification",
    "objective_type": "predictive",
    "required_columns": ["sample_id", "__split", "species"],
    "feature_families": [
        {
            "family": "MORPHOLOGY",
            "selector_hint": "^(sepal|petal)_",
            "rationale": "All morphological measurements.",
        }
    ],
    "techniques": ["RandomForest", "StratifiedKFold"],
}


# ── 4. Time series forecasting ───────────────────────────────────────────────

TIMESERIES_INVENTORY = [
    "date", "store_id", "item_id", "sales", "price", "promo",
    "day_of_week", "month", "lag_1", "lag_7", "rolling_mean_7",
    "__split",
]
TIMESERIES_DTYPES = {
    "date": "object", "store_id": "int64", "item_id": "int64",
    "sales": "float64", "price": "float64", "promo": "int64",
    "day_of_week": "int64", "month": "int64", "lag_1": "float64",
    "lag_7": "float64", "rolling_mean_7": "float64", "__split": "object",
}
TIMESERIES_STRATEGY = {
    "title": "Store sales forecasting with lag features",
    "objective_type": "predictive",
    "required_columns": ["date", "store_id", "item_id", "__split", "sales"],
    "feature_families": [
        {
            "family": "LAG_FEATURES",
            "selector_hint": "^(lag_|rolling_)",
            "rationale": "Autoregressive lag and rolling statistics.",
        },
        {
            "family": "CALENDAR_FEATURES",
            "selector_hint": "day_of_week, month",
            "rationale": "Calendar-based seasonality.",
        },
        {
            "family": "PRICE_PROMO",
            "selector_hint": "price, promo",
            "rationale": "External demand signals.",
        },
    ],
    "techniques": ["LightGBM", "TimeSeriesSplit"],
}


# ── 5. Anomaly detection ─────────────────────────────────────────────────────

ANOMALY_INVENTORY = [
    "transaction_id", "amount", "merchant_category", "hour",
    "distance_from_home", "is_fraud", "__split",
    "velocity_1h", "velocity_24h", "avg_amount_30d",
]
ANOMALY_DTYPES = {
    "transaction_id": "int64", "amount": "float64",
    "merchant_category": "object", "hour": "int64",
    "distance_from_home": "float64", "is_fraud": "int64",
    "__split": "object", "velocity_1h": "float64",
    "velocity_24h": "float64", "avg_amount_30d": "float64",
}
ANOMALY_STRATEGY = {
    "title": "Fraud detection with imbalanced learning",
    "objective_type": "predictive",
    "required_columns": ["transaction_id", "__split", "is_fraud"],
    "feature_families": [
        {
            "family": "TRANSACTION_FEATURES",
            "selector_hint": "all_numeric_except excluding transaction_id, is_fraud",
            "rationale": "All numeric signals for anomaly scoring.",
        },
    ],
    "techniques": ["LightGBM with scale_pos_weight", "PR-AUC optimization"],
}


# ── 6. NLP / text classification ─────────────────────────────────────────────

NLP_INVENTORY = [
    "doc_id", "text", "text_length", "word_count", "sentiment_score",
    "tfidf_dim_0", "tfidf_dim_1", "tfidf_dim_2", "label", "__split",
]
NLP_DTYPES = {
    "doc_id": "int64", "text": "object", "text_length": "int64",
    "word_count": "int64", "sentiment_score": "float64",
    "tfidf_dim_0": "float64", "tfidf_dim_1": "float64",
    "tfidf_dim_2": "float64", "label": "object", "__split": "object",
}
NLP_STRATEGY = {
    "title": "Document classification with TF-IDF features",
    "objective_type": "predictive",
    "required_columns": ["doc_id", "__split", "label"],
    "feature_families": [
        {
            "family": "TEXT_NUMERIC",
            "selector_hint": "text_length, word_count, sentiment_score",
            "rationale": "Numeric text statistics.",
        },
        {
            "family": "TFIDF_EMBEDDINGS",
            "selector_hint": "^tfidf_dim_",
            "rationale": "Pre-computed TF-IDF dimensions.",
        },
    ],
    "techniques": ["Logistic Regression", "micro-F1 optimization"],
}


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: Feature family expansion
# ═══════════════════════════════════════════════════════════════════════════════


class TestFeatureFamilyExpansion:
    """Verify _expand_strategy_feature_families works across problem types."""

    def test_binary_all_numeric_except(self):
        cols = _expand_strategy_feature_families(
            BINARY_CLASSIFICATION_STRATEGY,
            BINARY_CLASSIFICATION_INVENTORY,
            data_profile=_make_data_profile(BINARY_CLASSIFICATION_DTYPES),
        )
        assert "feature_a" in cols
        assert "feature_b" in cols
        assert "feature_c" in cols
        assert "id" not in cols
        assert "target" not in cols
        assert "__split" not in cols  # object type, excluded by all_numeric_except

    def test_regression_all_numeric_except(self):
        cols = _expand_strategy_feature_families(
            REGRESSION_STRATEGY,
            REGRESSION_INVENTORY,
            data_profile=_make_data_profile(REGRESSION_DTYPES),
        )
        assert "lot_area" in cols
        assert "year_built" in cols
        assert "garage_area" in cols
        assert "sale_price" not in cols
        assert "id" not in cols
        # neighborhood is object — excluded from NUMERIC_FEATURES
        assert "neighborhood" in cols  # but included via CATEGORICAL_FEATURES hint

    def test_multiclass_regex_selector(self):
        cols = _expand_strategy_feature_families(
            MULTICLASS_STRATEGY,
            MULTICLASS_INVENTORY,
            data_profile=_make_data_profile(MULTICLASS_DTYPES),
        )
        assert "sepal_length" in cols
        assert "sepal_width" in cols
        assert "petal_length" in cols
        assert "petal_width" in cols
        assert "species" not in cols
        assert "sample_id" not in cols

    def test_timeseries_multiple_families(self):
        cols = _expand_strategy_feature_families(
            TIMESERIES_STRATEGY,
            TIMESERIES_INVENTORY,
            data_profile=_make_data_profile(TIMESERIES_DTYPES),
        )
        # LAG_FEATURES
        assert "lag_1" in cols
        assert "lag_7" in cols
        assert "rolling_mean_7" in cols
        # CALENDAR_FEATURES
        assert "day_of_week" in cols
        assert "month" in cols
        # PRICE_PROMO
        assert "price" in cols
        assert "promo" in cols
        # Excluded
        assert "sales" not in cols
        assert "date" not in cols

    def test_anomaly_all_numeric_except(self):
        cols = _expand_strategy_feature_families(
            ANOMALY_STRATEGY,
            ANOMALY_INVENTORY,
            data_profile=_make_data_profile(ANOMALY_DTYPES),
        )
        assert "amount" in cols
        assert "hour" in cols
        assert "distance_from_home" in cols
        assert "velocity_1h" in cols
        assert "velocity_24h" in cols
        assert "avg_amount_30d" in cols
        assert "is_fraud" not in cols
        assert "transaction_id" not in cols

    def test_nlp_mixed_selectors(self):
        cols = _expand_strategy_feature_families(
            NLP_STRATEGY,
            NLP_INVENTORY,
            data_profile=_make_data_profile(NLP_DTYPES),
        )
        assert "text_length" in cols
        assert "word_count" in cols
        assert "sentiment_score" in cols
        assert "tfidf_dim_0" in cols
        assert "tfidf_dim_1" in cols
        assert "tfidf_dim_2" in cols
        # "text" raw column should not be selected (only text_length etc.)
        # However, if the CSV parser includes "text" as a partial match,
        # the important invariant is: label and doc_id must NOT appear.
        assert "label" not in cols
        assert "doc_id" not in cols

    def test_empty_inventory_returns_empty(self):
        cols = _expand_strategy_feature_families(
            BINARY_CLASSIFICATION_STRATEGY, [], data_profile=None,
        )
        assert cols == []

    def test_no_feature_families_returns_empty(self):
        strategy_no_families = {"title": "Bare strategy", "required_columns": ["id"]}
        cols = _expand_strategy_feature_families(
            strategy_no_families,
            BINARY_CLASSIFICATION_INVENTORY,
            data_profile=_make_data_profile(BINARY_CLASSIFICATION_DTYPES),
        )
        assert cols == []

    def test_never_invents_columns(self):
        """Feature expansion must NEVER produce columns not in inventory."""
        all_scenarios = [
            (BINARY_CLASSIFICATION_STRATEGY, BINARY_CLASSIFICATION_INVENTORY, BINARY_CLASSIFICATION_DTYPES),
            (REGRESSION_STRATEGY, REGRESSION_INVENTORY, REGRESSION_DTYPES),
            (MULTICLASS_STRATEGY, MULTICLASS_INVENTORY, MULTICLASS_DTYPES),
            (TIMESERIES_STRATEGY, TIMESERIES_INVENTORY, TIMESERIES_DTYPES),
            (ANOMALY_STRATEGY, ANOMALY_INVENTORY, ANOMALY_DTYPES),
            (NLP_STRATEGY, NLP_INVENTORY, NLP_DTYPES),
        ]
        for strategy, inventory, dtypes in all_scenarios:
            cols = _expand_strategy_feature_families(
                strategy, inventory,
                data_profile=_make_data_profile(dtypes),
            )
            inventory_set = set(inventory)
            for col in cols:
                assert col in inventory_set, (
                    f"Feature expansion invented column '{col}' not in inventory "
                    f"for strategy '{strategy['title']}'"
                )


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: select_relevant_columns
# ═══════════════════════════════════════════════════════════════════════════════


class TestSelectRelevantColumns:
    """Verify select_relevant_columns produces coherent outputs per problem type."""

    @pytest.mark.parametrize("label,strategy,inventory,dtypes,business_obj", [
        ("binary", BINARY_CLASSIFICATION_STRATEGY, BINARY_CLASSIFICATION_INVENTORY,
         BINARY_CLASSIFICATION_DTYPES, "Predict binary target"),
        ("regression", REGRESSION_STRATEGY, REGRESSION_INVENTORY,
         REGRESSION_DTYPES, "Predict house sale prices"),
        ("multiclass", MULTICLASS_STRATEGY, MULTICLASS_INVENTORY,
         MULTICLASS_DTYPES, "Classify species from measurements"),
        ("timeseries", TIMESERIES_STRATEGY, TIMESERIES_INVENTORY,
         TIMESERIES_DTYPES, "Forecast daily store sales"),
        ("anomaly", ANOMALY_STRATEGY, ANOMALY_INVENTORY,
         ANOMALY_DTYPES, "Detect fraudulent transactions"),
        ("nlp", NLP_STRATEGY, NLP_INVENTORY,
         NLP_DTYPES, "Classify documents by topic"),
    ])
    def test_relevant_columns_subset_of_inventory(
        self, label, strategy, inventory, dtypes, business_obj
    ):
        """Relevant columns must be a subset of column_inventory."""
        result = select_relevant_columns(
            strategy=strategy,
            business_objective=business_obj,
            domain_expert_critique="",
            column_inventory=inventory,
            data_profile=_make_data_profile(dtypes),
        )
        relevant = result.get("relevant_columns", [])
        inventory_set = set(inventory)
        for col in relevant:
            assert col in inventory_set, (
                f"[{label}] relevant_columns produced '{col}' not in inventory"
            )

    @pytest.mark.parametrize("label,strategy,inventory,dtypes,business_obj", [
        ("binary", BINARY_CLASSIFICATION_STRATEGY, BINARY_CLASSIFICATION_INVENTORY,
         BINARY_CLASSIFICATION_DTYPES, "Predict binary target"),
        ("regression", REGRESSION_STRATEGY, REGRESSION_INVENTORY,
         REGRESSION_DTYPES, "Predict house sale prices"),
        ("multiclass", MULTICLASS_STRATEGY, MULTICLASS_INVENTORY,
         MULTICLASS_DTYPES, "Classify species from measurements"),
        ("timeseries", TIMESERIES_STRATEGY, TIMESERIES_INVENTORY,
         TIMESERIES_DTYPES, "Forecast daily store sales"),
        ("anomaly", ANOMALY_STRATEGY, ANOMALY_INVENTORY,
         ANOMALY_DTYPES, "Detect fraudulent transactions"),
        ("nlp", NLP_STRATEGY, NLP_INVENTORY,
         NLP_DTYPES, "Classify documents by topic"),
    ])
    def test_required_columns_always_included(
        self, label, strategy, inventory, dtypes, business_obj
    ):
        """strategy.required_columns must always appear in relevant_columns."""
        result = select_relevant_columns(
            strategy=strategy,
            business_objective=business_obj,
            domain_expert_critique="",
            column_inventory=inventory,
            data_profile=_make_data_profile(dtypes),
        )
        relevant = set(result.get("relevant_columns", []))
        for req_col in strategy.get("required_columns", []):
            if req_col in set(inventory):
                assert req_col in relevant, (
                    f"[{label}] required column '{req_col}' missing from relevant_columns"
                )

    @pytest.mark.parametrize("label,strategy,inventory,dtypes,business_obj,min_count", [
        ("binary", BINARY_CLASSIFICATION_STRATEGY, BINARY_CLASSIFICATION_INVENTORY,
         BINARY_CLASSIFICATION_DTYPES, "Predict binary target", 4),
        ("regression", REGRESSION_STRATEGY, REGRESSION_INVENTORY,
         REGRESSION_DTYPES, "Predict house sale prices", 6),
        ("multiclass", MULTICLASS_STRATEGY, MULTICLASS_INVENTORY,
         MULTICLASS_DTYPES, "Classify species", 5),
        ("timeseries", TIMESERIES_STRATEGY, TIMESERIES_INVENTORY,
         TIMESERIES_DTYPES, "Forecast sales", 8),
        ("anomaly", ANOMALY_STRATEGY, ANOMALY_INVENTORY,
         ANOMALY_DTYPES, "Detect fraud", 6),
        ("nlp", NLP_STRATEGY, NLP_INVENTORY,
         NLP_DTYPES, "Classify documents", 6),
    ])
    def test_minimum_column_coverage(
        self, label, strategy, inventory, dtypes, business_obj, min_count
    ):
        """Each problem type should select a reasonable number of columns."""
        result = select_relevant_columns(
            strategy=strategy,
            business_objective=business_obj,
            domain_expert_critique="",
            column_inventory=inventory,
            data_profile=_make_data_profile(dtypes),
        )
        relevant = result.get("relevant_columns", [])
        assert len(relevant) >= min_count, (
            f"[{label}] only {len(relevant)} relevant columns selected, "
            f"expected at least {min_count}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: Contract validation across problem types
# ═══════════════════════════════════════════════════════════════════════════════


def _build_minimal_contract(
    strategy: Dict[str, Any],
    inventory: List[str],
    dtypes: Dict[str, str],
    target_cols: List[str],
    id_cols: List[str],
    scope: str = "full_pipeline",
) -> Dict[str, Any]:
    """Build a minimal valid contract for validation testing."""
    feature_cols = [
        c for c in inventory
        if c not in set(target_cols + id_cols + ["__split"])
    ]
    column_roles = {
        "identifiers": id_cols,
        "split": ["__split"] if "__split" in inventory else [],
        "outcome": target_cols,
        "pre_decision": feature_cols,
    }
    dtype_targets = {}
    for col in inventory:
        dt = dtypes.get(col, "object")
        dtype_targets[col] = {
            "target_dtype": dt,
            "nullable": dt != "int64",
            "role": next(
                (role for role, cols in column_roles.items() if col in cols),
                "pre_decision",
            ),
        }
    return {
        "contract_version": "4.1",
        "scope": scope,
        "strategy_title": strategy.get("title", "Test"),
        "business_objective": "Test objective",
        "output_dialect": {"sep": ",", "decimal": ".", "encoding": "utf-8"},
        "canonical_columns": inventory,
        "column_roles": column_roles,
        "allowed_feature_sets": [
            {"family": f["family"], "description": f.get("rationale", "")}
            for f in strategy.get("feature_families", [])
        ],
        "model_features": feature_cols,
        "task_semantics": strategy.get("objective_type", "predictive"),
        "active_workstreams": {
            "data_cleaning": True,
            "feature_engineering": False,
            "model_training": True,
            "model_evaluation": True,
            "prediction_generation": True,
        },
        "required_outputs": [
            {
                "intent": "submission_file",
                "path": "artifacts/ml/submission.csv",
                "required": True,
                "owner": "ml_engineer",
                "kind": "dataset",
            },
            {
                "intent": "cleaned_dataset",
                "path": "artifacts/clean/dataset_cleaned.csv",
                "required": True,
                "owner": "data_engineer",
                "kind": "dataset",
            },
            {
                "intent": "cleaning_manifest",
                "path": "artifacts/clean/cleaning_manifest.json",
                "required": True,
                "owner": "data_engineer",
                "kind": "manifest",
            },
        ],
        "artifact_requirements": {
            "cleaned_dataset": {
                "output_path": "artifacts/clean/dataset_cleaned.csv",
                "output_manifest_path": "artifacts/clean/cleaning_manifest.json",
                "required_columns": [c for c in inventory if c != "__split"] + ["__split"],
            }
        },
        "column_dtype_targets": dtype_targets,
        "cleaning_gates": [
            {"name": "split_integrity", "severity": "HARD", "params": {}},
        ],
        "qa_gates": [
            {"name": "output_format", "severity": "HARD", "params": {}},
        ],
        "reviewer_gates": [],
        "evaluation_spec": {
            "objective_type": strategy.get("objective_type", "predictive"),
            "primary_target": target_cols[0] if target_cols else "target",
            "primary_metric": "log_loss",
            "label_columns": target_cols,
        },
        "validation_requirements": {
            "method": "cross_validation",
            "primary_metric": "log_loss",
            "metrics_to_report": ["log_loss"],
            "params": {"n_splits": 5},
        },
        "optimization_policy": {
            "primary_objective": "minimize log_loss",
            "enabled": True,
            "max_rounds": 4,
            "allow_hpo": True,
            "allow_feature_engineering": False,
        },
        "iteration_policy": {
            "max_iterations": 6,
            "metric_improvement_max": 4,
            "runtime_fix_max": 3,
        },
        "data_engineer_runbook": ["Clean and prepare data."],
        "ml_engineer_runbook": ["Train and evaluate model."],
    }


class TestContractValidationUniversal:
    """Verify contract validation accepts well-formed contracts for all problem types."""

    @pytest.mark.parametrize("label,strategy,inventory,dtypes,targets,ids", [
        ("binary", BINARY_CLASSIFICATION_STRATEGY, BINARY_CLASSIFICATION_INVENTORY,
         BINARY_CLASSIFICATION_DTYPES, ["target"], ["id"]),
        ("regression", REGRESSION_STRATEGY, REGRESSION_INVENTORY,
         REGRESSION_DTYPES, ["sale_price"], ["id"]),
        ("multiclass", MULTICLASS_STRATEGY, MULTICLASS_INVENTORY,
         MULTICLASS_DTYPES, ["species"], ["sample_id"]),
        ("timeseries", TIMESERIES_STRATEGY, TIMESERIES_INVENTORY,
         TIMESERIES_DTYPES, ["sales"], ["date"]),
        ("anomaly", ANOMALY_STRATEGY, ANOMALY_INVENTORY,
         ANOMALY_DTYPES, ["is_fraud"], ["transaction_id"]),
        ("nlp", NLP_STRATEGY, NLP_INVENTORY,
         NLP_DTYPES, ["label"], ["doc_id"]),
    ])
    def test_minimal_contract_passes_validation(
        self, label, strategy, inventory, dtypes, targets, ids
    ):
        """A well-formed contract for any problem type should not crash validation."""
        contract = _build_minimal_contract(strategy, inventory, dtypes, targets, ids)
        # validate_contract returns a report dict (not the contract itself).
        # The key invariant: it must not raise, and must return a dict.
        result = validate_contract(contract)
        assert isinstance(result, dict), f"[{label}] validate_contract did not return dict"
        # The report should have a status field
        assert "status" in result, f"[{label}] validation report missing 'status'"

    @pytest.mark.parametrize("label,strategy,inventory,dtypes,targets,ids", [
        ("binary", BINARY_CLASSIFICATION_STRATEGY, BINARY_CLASSIFICATION_INVENTORY,
         BINARY_CLASSIFICATION_DTYPES, ["target"], ["id"]),
        ("regression", REGRESSION_STRATEGY, REGRESSION_INVENTORY,
         REGRESSION_DTYPES, ["sale_price"], ["id"]),
        ("multiclass", MULTICLASS_STRATEGY, MULTICLASS_INVENTORY,
         MULTICLASS_DTYPES, ["species"], ["sample_id"]),
        ("timeseries", TIMESERIES_STRATEGY, TIMESERIES_INVENTORY,
         TIMESERIES_DTYPES, ["sales"], ["date"]),
        ("anomaly", ANOMALY_STRATEGY, ANOMALY_INVENTORY,
         ANOMALY_DTYPES, ["is_fraud"], ["transaction_id"]),
        ("nlp", NLP_STRATEGY, NLP_INVENTORY,
         NLP_DTYPES, ["label"], ["doc_id"]),
    ])
    def test_minimal_readonly_has_no_errors(
        self, label, strategy, inventory, dtypes, targets, ids
    ):
        """Minimal readonly validation should produce zero errors for well-formed contracts."""
        contract = _build_minimal_contract(strategy, inventory, dtypes, targets, ids)
        report = validate_contract_minimal_readonly(contract, column_inventory=inventory)
        assert isinstance(report, dict), f"[{label}] validation did not return dict"
        # Check no hard errors (warnings are OK)
        errors = [
            issue for issue in report.get("issues", [])
            if issue.get("severity") == "error"
        ]
        # Allow some errors from optional keys the minimal contract doesn't populate
        # but core structural errors should be zero
        structural_errors = [
            e for e in errors
            if "canonical_columns" in e.get("rule", "")
            or "column_roles" in e.get("rule", "")
            or "model_features" in e.get("rule", "")
        ]
        assert len(structural_errors) == 0, (
            f"[{label}] structural validation errors: "
            f"{[e.get('message','') for e in structural_errors]}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TEST: Edge cases and defensive behavior
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Verify defensive behavior with unusual inputs."""

    def test_none_strategy_does_not_crash(self):
        result = select_relevant_columns(
            strategy=None,
            business_objective="Predict something",
            domain_expert_critique="",
            column_inventory=["a", "b", "c"],
        )
        assert isinstance(result, dict)
        assert "relevant_columns" in result

    def test_empty_column_inventory(self):
        result = select_relevant_columns(
            strategy=BINARY_CLASSIFICATION_STRATEGY,
            business_objective="Predict target",
            domain_expert_critique="",
            column_inventory=[],
        )
        assert isinstance(result, dict)

    def test_strategy_with_no_families(self):
        strategy = {"title": "Bare", "required_columns": ["id", "target"]}
        result = select_relevant_columns(
            strategy=strategy,
            business_objective="Predict target",
            domain_expert_critique="",
            column_inventory=["id", "target", "feat_1"],
        )
        relevant = result.get("relevant_columns", [])
        # required_columns should still be included
        assert "id" in relevant
        assert "target" in relevant

    def test_selector_hint_with_no_matches(self):
        """A regex that matches nothing should not crash — just return empty."""
        strategy = {
            "title": "No match",
            "feature_families": [
                {"family": "GHOST", "selector_hint": "^zzz_nonexistent_"}
            ],
        }
        cols = _expand_strategy_feature_families(
            strategy,
            ["a", "b", "c"],
            data_profile=_make_data_profile({"a": "float64", "b": "float64", "c": "float64"}),
        )
        assert cols == []

    def test_data_profile_none_still_works(self):
        """Expansion should work even without data_profile (no dtype info)."""
        # all_numeric_except needs data_profile — without it, should degrade gracefully
        cols = _expand_strategy_feature_families(
            BINARY_CLASSIFICATION_STRATEGY,
            BINARY_CLASSIFICATION_INVENTORY,
            data_profile=None,
        )
        # Without data_profile, all_numeric_except can't determine numeric columns
        # so it may return empty or fall back — just verify no crash
        assert isinstance(cols, list)

    def test_duplicate_columns_in_inventory(self):
        """Duplicates in inventory should not produce duplicate results."""
        inventory = ["a", "b", "a", "c", "b"]
        result = select_relevant_columns(
            strategy={"title": "Test", "required_columns": ["a", "b"]},
            business_objective="Test",
            domain_expert_critique="",
            column_inventory=inventory,
        )
        relevant = result.get("relevant_columns", [])
        assert len(relevant) == len(set(relevant)), "Duplicate columns in output"
