from src.graph import graph as graph_mod
from unittest.mock import patch


def _profile(n_rows: int, n_cols: int, dtype: str = "float64"):
    return {
        "basic_stats": {
            "n_rows": n_rows,
            "n_cols": n_cols,
        },
        "dtypes": {f"c{i}": dtype for i in range(n_cols)},
    }


def test_backend_selection_prefers_sandbox_when_memory_is_safe():
    state = {
        "dataset_scale_hints": {
            "file_mb": 15.0,
            "est_rows": 120_000,
            "cols": 30,
        }
    }
    data_profile = _profile(120_000, 30, "float64")
    ml_plan = {"cv_policy": {"n_splits": 5}}

    use_heavy, reason = graph_mod._should_use_heavy_runner(state, data_profile, ml_plan)

    assert use_heavy is False
    assert reason == "sandbox_memory_safe"
    decision = state.get("backend_memory_estimate") or {}
    assert decision.get("estimate", {}).get("estimated_peak_gb") is not None


def test_backend_selection_uses_heavy_when_memory_exceeds_safe_budget():
    state = {
        "dataset_scale_hints": {
            "file_mb": 350.0,
            "est_rows": 2_500_000,
            "cols": 200,
        }
    }
    data_profile = _profile(2_500_000, 200, "float64")
    ml_plan = {"cv_policy": {"n_splits": 5}}

    use_heavy, reason = graph_mod._should_use_heavy_runner(state, data_profile, ml_plan)

    assert use_heavy is True
    assert reason == "est_memory_exceeds_sandbox_safe"
    decision = state.get("backend_memory_estimate") or {}
    estimate = decision.get("estimate") or {}
    assert (estimate.get("estimated_peak_gb") or 0) > (decision.get("safe_budget_gb") or 0)


def test_backend_selection_promotes_to_heavy_after_sandbox_memory_failure():
    state = {
        "e2b_memory_failure_detected": True,
    }
    data_profile = _profile(50_000, 20, "float64")
    ml_plan = {"cv_policy": {"n_splits": 3}}

    use_heavy, reason = graph_mod._should_use_heavy_runner(state, data_profile, ml_plan)

    assert use_heavy is True
    assert reason == "sandbox_memory_failure_fallback"


def test_backend_selection_uses_heavy_for_large_dataset_with_uncertain_estimate():
    state = {
        "dataset_scale_hints": {
            "file_mb": 120.0,
            "est_rows": 1_400_000,
        },
        "column_inventory": [f"c{i}" for i in range(60)],
    }
    data_profile = {
        "basic_stats": {
            "n_rows": 7_800,
            "n_cols": 0,
        },
        "dtypes": {},
        "sampling": {
            "was_sampled": True,
            "total_rows_in_file": 1_400_000,
        },
    }
    ml_plan = {"cv_policy": {"n_splits": 5}}

    use_heavy, reason = graph_mod._should_use_heavy_runner(state, data_profile, ml_plan)

    assert use_heavy is True
    assert reason in {"large_dataset_uncertain_memory_estimate", "est_memory_exceeds_sandbox_safe"}
    decision = state.get("backend_memory_estimate") or {}
    estimate = decision.get("estimate") or {}
    assert (estimate.get("n_rows") or 0) >= 1_400_000
    assert (estimate.get("n_cols") or 0) >= 60


def test_memory_pressure_detector_matches_common_oom_signatures():
    assert graph_mod._is_memory_pressure_error("MemoryError: Unable to allocate 3.2 GiB")
    assert graph_mod._is_memory_pressure_error("Process Killed (exit code 137)")
    assert not graph_mod._is_memory_pressure_error("ValueError: invalid literal for int()")


def test_extract_required_columns_falls_back_to_canonical_columns():
    contract = {
        "canonical_columns": ["id", "target", "feature_a"],
        "artifact_requirements": {},
    }
    required, source = graph_mod._extract_required_columns_from_contract(contract, {})
    assert required == ["id", "target", "feature_a"]
    assert source == "contract.canonical_columns"


def test_de_backend_selection_uses_total_header_columns_signal():
    state = {
        "dataset_scale_hints": {
            "file_mb": 278.0,
            "est_rows": 1_498_232,
        }
    }
    use_heavy, reason = graph_mod._should_use_heavy_runner_for_data_engineer(
        state,
        required_cols_count=10,
        total_cols_count=60,
    )
    assert use_heavy is True
    assert reason in {"de_est_memory_exceeds_sandbox_safe", "de_memory_uncertain_guard"}
    decision = state.get("de_backend_memory_estimate") or {}
    assert decision.get("n_cols") == 60


def test_de_heavy_failure_does_not_fallback_to_sandbox_in_same_cycle():
    assert graph_mod._should_run_sandbox_after_de_heavy_result(None) is True
    assert graph_mod._should_run_sandbox_after_de_heavy_result({"unavailable": True}) is True
    assert graph_mod._should_run_sandbox_after_de_heavy_result({"ok": True, "unavailable": False}) is False
    assert graph_mod._should_run_sandbox_after_de_heavy_result({"ok": False, "unavailable": False}) is False


def test_detect_de_heavy_runner_protocol_mismatch_from_ml_missing_outputs():
    error_payload = {
        "error": "Script completed but missing required outputs: ['data/metrics.json', 'data/scored_rows.csv', 'data/alignment_check.json']"
    }
    downloaded = {"data/cleaning_manifest.json": "data/cleaning_manifest.json"}
    heavy_log = "SUCCESS: Cleaned data written to data/cleaned_data.csv"
    assert (
        graph_mod._detect_de_heavy_runner_protocol_mismatch(
            "data_engineer_cleaning",
            error_payload,
            downloaded,
            heavy_log,
        )
        is True
    )


def test_detect_de_heavy_runner_protocol_mismatch_requires_de_mode():
    error_payload = {
        "error": "Script completed but missing required outputs: ['data/metrics.json', 'data/scored_rows.csv', 'data/alignment_check.json']"
    }
    downloaded = {"data/cleaning_manifest.json": "data/cleaning_manifest.json"}
    heavy_log = "SUCCESS: Cleaned data written to data/cleaned_data.csv"
    assert (
        graph_mod._detect_de_heavy_runner_protocol_mismatch(
            "execute_code",
            error_payload,
            downloaded,
            heavy_log,
        )
        is False
    )


def test_ml_backend_selection_forces_cloudrun_when_contract_requires_heavy_deps():
    state = {
        "execution_contract": {"required_dependencies": ["torch"]},
        "dataset_scale_hints": {"file_mb": 5.0, "est_rows": 1000, "cols": 10},
    }
    with patch.object(graph_mod, "_get_execution_runtime_mode", return_value="cloudrun"), \
         patch.object(graph_mod, "_get_heavy_runner_config", return_value={"job": "j"}), \
         patch.object(graph_mod, "_should_use_heavy_runner", return_value=(False, "sandbox_memory_safe")):
        decision = graph_mod._resolve_ml_backend_selection(state)

    assert decision.get("use_heavy") is True
    assert decision.get("backend_profile") == "cloudrun"
    assert decision.get("reason") == "cloudrun_only_mode"
    assert decision.get("heavy_deps_required") is True


def test_ml_backend_selection_heavy_deps_without_cloudrun_stays_sandbox():
    state = {
        "execution_contract": {"required_dependencies": ["torch"]},
    }
    with patch.object(graph_mod, "_get_execution_runtime_mode", return_value="cloudrun"), \
         patch.object(graph_mod, "_get_heavy_runner_config", return_value=None):
        decision = graph_mod._resolve_ml_backend_selection(state)

    assert decision.get("use_heavy") is False
    assert decision.get("backend_profile") == "cloudrun"
    assert decision.get("reason") == "cloudrun_required_by_dependencies_but_unavailable"
    assert decision.get("heavy_deps_required") is True


def test_ml_backend_selection_forces_cloudrun_when_script_imports_heavy_libs():
    state = {
        "execution_contract": {"required_dependencies": []},
        "generated_code": "import torch\nprint('train')\n",
        "dataset_scale_hints": {"file_mb": 2.0, "est_rows": 4000, "cols": 12},
    }
    with patch.object(graph_mod, "_get_execution_runtime_mode", return_value="cloudrun"), \
         patch.object(graph_mod, "_get_heavy_runner_config", return_value={"job": "j"}), \
         patch.object(graph_mod, "_should_use_heavy_runner", return_value=(False, "sandbox_memory_safe")):
        decision = graph_mod._resolve_ml_backend_selection(state)

    assert decision.get("use_heavy") is True
    assert decision.get("backend_profile") == "cloudrun"
    assert decision.get("reason") == "cloudrun_only_mode"
    assert decision.get("heavy_deps_from_code") is True
    assert "torch" in (decision.get("heavy_imports_in_code") or [])


def test_ml_backend_selection_marks_unavailable_when_script_imports_heavy_libs():
    state = {
        "execution_contract": {"required_dependencies": []},
        "generated_code": "from transformers import AutoModel\n",
    }
    with patch.object(graph_mod, "_get_execution_runtime_mode", return_value="cloudrun"), \
         patch.object(graph_mod, "_get_heavy_runner_config", return_value=None):
        decision = graph_mod._resolve_ml_backend_selection(state)

    assert decision.get("use_heavy") is False
    assert decision.get("backend_profile") == "cloudrun"
    assert decision.get("reason") == "cloudrun_required_by_script_imports_but_unavailable"
    assert decision.get("heavy_deps_from_code") is True
