"""
Contract Validation Utilities - V4.1 Compatible.

This module provides default runbook configurations for data engineer and ML engineer agents.
These are used as defaults when contracts don't specify custom runbooks.
"""

import copy
from typing import Any, Dict, List


DEFAULT_DATA_ENGINEER_RUNBOOK: Dict[str, Any] = {
    "version": "2",
    "goals": [
        "Deliver a cleaned dataset and manifest that respect the execution contract dialect, roles, and expected kinds.",
        "Preserve required and potentially useful columns; avoid destructive conversions.",
        "Produce deterministic, audit-friendly output — every transformation must be traceable in the manifest.",
    ],
    "requirements": [
        "Load data using the declared dialect (sep/decimal/encoding) from contract/manifest — do not hardcode format literals.",
        "Convert columns to their expected_kind (numeric, datetime, categorical) using appropriate parsing for THIS dataset's observed formats.",
        "Canonical columns must contain cleaned values in the output; helper/shadow columns are intermediate aids, not final deliverables.",
        "Derive all required columns from preprocessing_requirements; ensure they exist in the output.",
        "Perform a post-cleaning self-audit: for each required column, report dtype and null_frac.",
        "Preserve exact canonical_name strings (including spaces/symbols); do not normalize away punctuation.",
        "Do not drop required/derived columns solely for being constant; record as constant in manifest.",
        "If a required source column for derivations is missing, fail fast with a clear error — do not silently default.",
        "Only enforce existence for source='input' columns; source='derived' columns must be created after mapping.",
        "If a derived column has derived_owner='ml_engineer', leave it absent for downstream derivation.",
        "When numeric parsing produces mostly NaN while raw data had high non-null rate, treat it as a parsing failure — investigate and adjust rather than silently proceeding.",
    ],
    "boundaries": [
        "Scope is cleaning only — do not create downstream ML artifacts, models, or analytics.",
        "Do not compute advanced validation metrics (MAE/correlation/score); leave analytics to ML/QA.",
        "Do not fabricate constant placeholders for derived columns without a formula or depends_on.",
        "Numeric parsing must reason about locale: if dialect says decimal=',' and raw data shows dots, investigate whether dots are thousands separators before stripping them.",
    ],
    "reasoning_guidance": [
        "Use canonical_name (if provided) for consistent references across mapping, validation, and derivations.",
        "Verify required columns after normalization/mapping — do not treat pre-mapped absence as missing.",
        "If a numeric-looking column is typed as object/string, treat conversion as a risk and verify with sample data.",
        "If raw samples contain currency symbols, text prefixes, or mixed formats, sanitize before numeric conversion.",
        "If normalization causes name collisions, choose deterministically and log a warning.",
        "If conversion yields excessive NaN, revert and log instead of dropping required columns.",
        "If derived columns are required, confirm source inputs exist and document NA handling assumptions.",
        "When referencing contract/config content in code, ensure it is valid Python (JSON null/true/false must be handled).",
    ],
    "validation_guidance": [
        "Print CLEANING_VALIDATION with dtype and null_frac; validation diagnostics must not raise errors.",
        "Ensure manifest JSON serialization handles numpy/pandas scalar types safely.",
    ],
    "manifest_requirements": {
        "include": ["input_dialect", "output_dialect", "column_mapping", "dropped_columns", "conversions", "conversions_meta", "type_checks"],
    },
}


DEFAULT_ML_ENGINEER_RUNBOOK: Dict[str, Any] = {
    "version": "2",
    "goals": [
        "Train/evaluate models aligned to the strategy and execution contract.",
        "Produce interpretable outputs and required artifacts without leakage.",
        "Choose modeling approaches that match the data structure, business objective, and runtime budget.",
    ],
    "requirements": [
        "Load data using dialect/output_dialect from the cleaning manifest.",
        "Honor the import allowlist; banned packages cause immediate script rejection.",
        "Treat column_roles and artifact_requirements from the contract as source-of-truth for outputs.",
        "Include a variance guard on the target — constant targets cannot be modeled.",
        "Build feature sets only from contract-allowed columns; print a mapping summary.",
        "Ensure output directories exist before writing artifacts.",
        "If derived columns already exist in cleaned data, use them as-is; only derive if missing.",
        "Compare against baseline_metric when one exists and report the comparison.",
        "Write per-row scoring artifacts only when the contract explicitly declares them, at the declared path.",
        "Handle numpy/pandas scalar types safely when serializing JSON artifacts.",
        "If contract includes alignment_requirements, produce alignment_check.json with per-requirement evidence, feature usage, and PASS/WARN/FAIL status.",
        "If contract includes decision_variables, treat them as decision inputs (not leakage by default) and document selection-bias risks.",
        "If contract includes missing_sentinels, handle sentinel values as missing during modeling.",
    ],
    "boundaries": [
        "Do not import sys or any blocked sandbox packages.",
        "Do not add noise/jitter to target values.",
        "Do not treat decision_variables as forbidden leakage unless the contract explicitly marks them post-outcome with no decision context.",
        "Do not invent outputs or artifacts not requested by the contract.",
    ],
    "reasoning_guidance": [
        "Use canonical_name (if provided) for consistent column references across agents.",
        "If target_type is ordinal/ranking, consider whether predictive regression is actually the right objective.",
        "Validate weight constraints and explain regularization choices.",
        "If decision_variables exist, reason about how they interact with the modeling objective and whether they are observed for all rows.",
        "If missing_sentinels exist, reason about whether sentinel handling could bias training or scoring.",
        "Choose preprocessing, validation, and scoring logic that matches the data structure rather than applying generic boilerplate.",
    ],
    "validation_guidance": [
        "Run leakage and variance checks before training.",
        "Report concentration metrics (HHI/max weight/near-zero) for scoring weights when applicable.",
        "Print QA_SELF_CHECK with satisfied checklist items.",
        "Print ALIGNMENT_CHECK with status when alignment_check.json is required.",
    ],
    "outputs": {
        "required": ["data/cleaned_data.csv"],
        "optional": [],
    },
}
