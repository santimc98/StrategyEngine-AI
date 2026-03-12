from src.agents.execution_planner import _ensure_contract_visual_policy, build_plot_spec


def _base_ml_contract() -> dict:
    return {
        "scope": "full_pipeline",
        "strategy_title": "Risk scoring with calibrated probabilities",
        "business_objective": "Generate risk score, confidence, and operational review policy.",
        "canonical_columns": ["id", "feature_num", "target", "__split"],
        "column_roles": {
            "id": ["id"],
            "pre_decision": ["feature_num"],
            "outcome": ["target"],
            "stratification": ["__split"],
        },
        "required_outputs": [
            "data/cleaned_data.csv",
            "data/scored_rows.csv",
            "data/metrics.json",
            "data/alignment_check.json",
        ],
        "evaluation_spec": {
            "objective_type": "binary_classification",
            "target_type": "binary",
        },
        "objective_analysis": {"problem_type": "classification"},
        "artifact_requirements": {
            "clean_dataset": {
                "required_columns": ["id", "feature_num", "target", "__split"],
                "output_path": "data/cleaned_data.csv",
                "output_manifest_path": "data/cleaning_manifest.json",
            }
        },
    }


def test_ensure_contract_visual_policy_populates_ml_visual_requirements():
    contract = _base_ml_contract()
    strategy = {
        "analysis_type": "predictive",
        "techniques": ["gradient boosting", "calibration", "explainability"],
    }

    out = _ensure_contract_visual_policy(contract, strategy, contract["business_objective"])

    artifact_reqs = out.get("artifact_requirements") or {}
    visual = artifact_reqs.get("visual_requirements") or {}
    plot_spec = visual.get("plot_spec") or {}
    plots = plot_spec.get("plots") or []

    assert isinstance(visual, dict)
    assert isinstance(plot_spec, dict)
    assert plots
    assert visual.get("enabled") is True
    assert plot_spec.get("max_plots") == len(plots)

    policy = out.get("reporting_policy") or {}
    assert isinstance(policy.get("plot_spec"), dict)
    assert len((policy.get("plot_spec") or {}).get("plots") or []) == len(plots)


def test_ensure_contract_visual_policy_preserves_explicit_visual_config():
    contract = _base_ml_contract()
    contract["artifact_requirements"]["visual_requirements"] = {
        "enabled": False,
        "required": False,
        "outputs_dir": "static/custom_plots",
        "items": [{"id": "custom_plot", "expected_filename": "custom_plot.png"}],
        "plot_spec": {"enabled": False, "max_plots": 1, "plots": []},
    }
    contract["required_outputs"] = list(contract["required_outputs"])
    before_outputs = list(contract["required_outputs"])

    out = _ensure_contract_visual_policy(contract, {"analysis_type": "predictive"}, contract["business_objective"])
    visual = (out.get("artifact_requirements") or {}).get("visual_requirements") or {}

    assert visual.get("enabled") is False
    assert visual.get("required") is False
    assert visual.get("outputs_dir") == "static/custom_plots"
    assert (visual.get("items") or [])[0].get("id") == "custom_plot"
    assert (visual.get("plot_spec") or {}).get("enabled") is False
    assert out.get("required_outputs") == before_outputs


def test_build_plot_spec_uses_dynamic_max_plots():
    contract = _base_ml_contract()
    plot_spec = build_plot_spec(contract)
    plots = plot_spec.get("plots") or []

    assert plot_spec.get("max_plots") == len(plots)


def test_build_plot_spec_uses_declared_artifact_paths():
    contract = _base_ml_contract()
    contract["required_outputs"] = [
        "artifacts/features/custom_cleaned.csv",
        "artifacts/outputs/scored_rows.csv",
        "artifacts/reports/metrics.json",
        "artifacts/checks/alignment_check.json",
    ]
    contract["artifact_requirements"] = {
        "clean_dataset": {
            "required_columns": ["id", "feature_num", "target", "__split"],
            "output_path": "artifacts/features/custom_cleaned.csv",
            "output_manifest_path": "artifacts/manifests/custom_clean_manifest.json",
        },
        "file_schemas": {
            "artifacts/reports/metrics.json": {"expected_row_count": 1},
            "artifacts/checks/alignment_check.json": {"expected_row_count": 1},
        },
    }

    plot_spec = build_plot_spec(contract)
    plots = plot_spec.get("plots") or []
    preferred_sources = {
        source
        for plot in plots
        for source in ((plot.get("inputs") or {}).get("preferred_sources") or [])
    }

    assert "artifacts/features/custom_cleaned.csv" in preferred_sources
    assert "artifacts/outputs/scored_rows.csv" in preferred_sources
    assert "artifacts/reports/metrics.json" in preferred_sources
    assert "artifacts/checks/alignment_check.json" in preferred_sources
    assert "data/cleaned_data.csv" not in preferred_sources
    assert "data/scored_rows.csv" not in preferred_sources
