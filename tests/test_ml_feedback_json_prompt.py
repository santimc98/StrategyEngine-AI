from src.agents.ml_engineer import MLEngineerAgent
from types import SimpleNamespace


def _assert_contains_all(text: str, *needles: str) -> None:
    for needle in needles:
        assert needle in text


def _assert_contains_terms(text: str, *terms: str) -> None:
    lowered = text.lower()
    for term in terms:
        assert term.lower() in lowered


class _FakeOpenAI:
    def __init__(self, api_key=None, base_url=None, timeout=None, default_headers=None):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.default_headers = default_headers


class _TraceOpenAI(_FakeOpenAI):
    queued_outputs = []

    def __init__(self, api_key=None, base_url=None, timeout=None, default_headers=None):
        super().__init__(api_key=api_key, base_url=base_url, timeout=timeout, default_headers=default_headers)
        self.chat = SimpleNamespace(completions=self)

    def create(self, model=None, messages=None, temperature=None):
        content = self.queued_outputs.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


def test_runtime_dependency_context_includes_universal_pandas_sklearn_interop_pitfalls():
    agent = MLEngineerAgent()
    context = agent._build_runtime_dependency_context(["pandas", "scikit-learn"])

    assert context["version_hints"]["pandas"]
    assert context["version_hints"]["scikit_learn"]
    assert context["pandas_sklearn_interop_pitfalls"]
    assert any("pd.NA" in str(item) for item in context["pandas_sklearn_interop_pitfalls"])
    assert any("legacy sklearn kwargs" in str(item).lower() for item in context["pandas_sklearn_interop_pitfalls"])


def test_editor_mode_prompt_includes_structured_feedback_json(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-openrouter")
    monkeypatch.setattr("src.agents.ml_engineer.OpenAI", _FakeOpenAI)

    def _fake_call_chat_with_fallback(client, messages, models, call_kwargs=None, logger=None, context_tag=None):
        return {"dummy": True}, models[0]

    monkeypatch.setattr("src.agents.ml_engineer.call_chat_with_fallback", _fake_call_chat_with_fallback)
    monkeypatch.setattr(
        "src.agents.ml_engineer.extract_response_text",
        lambda response: "import json\nprint('ok')\n",
    )

    agent = MLEngineerAgent()
    _ = agent.generate_code(
        strategy={"title": "Test Strategy", "analysis_type": "predictive", "required_columns": []},
        data_path="data/cleaned_data.csv",
        feedback_history=["legacy reviewer text"],
        previous_code="print('previous')\n",
        gate_context={
            "source": "reviewer",
            "status": "REJECTED",
            "feedback": "legacy reviewer text",
            "feedback_json": {
                "version": "v1",
                "status": "REJECTED",
                "failed_gates": ["submission_format_validation"],
                "required_fixes": ["Write required outputs at exact paths."],
            },
            "failed_gates": ["submission_format_validation"],
            "required_fixes": ["Write required outputs at exact paths."],
        },
        iteration_handoff={"mode": "patch"},
        execution_contract={"required_outputs": ["data/metrics.json"], "canonical_columns": []},
        ml_view={"required_outputs": ["data/metrics.json"]},
        editor_mode=True,
    )

    prompt = str(agent.last_prompt or "")
    assert "LATEST_ITERATION_FEEDBACK_JSON" in prompt
    assert "submission_format_validation" in prompt


def test_build_prompt_uses_strategy_context_without_default_model_recipe(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-openrouter")
    monkeypatch.setattr("src.agents.ml_engineer.OpenAI", _FakeOpenAI)

    def _fake_call_chat_with_fallback(client, messages, models, call_kwargs=None, logger=None, context_tag=None):
        return {"dummy": True}, models[0]

    monkeypatch.setattr("src.agents.ml_engineer.call_chat_with_fallback", _fake_call_chat_with_fallback)
    monkeypatch.setattr(
        "src.agents.ml_engineer.extract_response_text",
        lambda response: "import json\nprint('ok')\n",
    )

    agent = MLEngineerAgent()
    _ = agent.generate_code(
        strategy={
            "title": "Pricing Strategy",
            "analysis_type": "predictive",
            "required_columns": [],
            "techniques": ["time-aware regression on allowed features"],
            "fallback_chain": ["Fallback: simpler linear model if the primary family is incompatible."],
        },
        data_path="data/cleaned_data.csv",
        execution_contract={
            "required_outputs": [
                {"path": "artifacts/ml/cv_metrics.json", "required": True, "intent": "cv_metrics"},
                {"path": "static/plots/*.png", "required": False, "intent": "model_plots"},
            ],
            "artifact_requirements": {
                "clean_dataset": {"required_columns": ["CurrentPhase", "FiscalId"]},
            },
        },
        ml_view={
            "required_outputs": [
                {"path": "artifacts/ml/cv_metrics.json", "required": True, "intent": "cv_metrics"},
                {"path": "static/plots/*.png", "required": False, "intent": "model_plots"},
            ],
            "allowed_feature_sets": {"model_features": ["Size", "Debtors"]},
        },
    )

    prompt = str(agent.last_prompt or "")
    _assert_contains_all(
        prompt,
        "Model Input Candidates:",
        '["Size","Debtors"]',
        "Artifact-required Clean Columns:",
        '["CurrentPhase","FiscalId"]',
        "Do not add speculative model families",
    )
    assert "Required Features:" not in prompt
    assert "robust BASELINE model only" not in prompt
    _assert_contains_terms(prompt, "authoritative context, not prompt defaults")


def test_build_prompt_preserves_optional_required_output_flags(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-openrouter")
    monkeypatch.setattr("src.agents.ml_engineer.OpenAI", _FakeOpenAI)

    def _fake_call_chat_with_fallback(client, messages, models, call_kwargs=None, logger=None, context_tag=None):
        return {"dummy": True}, models[0]

    monkeypatch.setattr("src.agents.ml_engineer.call_chat_with_fallback", _fake_call_chat_with_fallback)
    monkeypatch.setattr(
        "src.agents.ml_engineer.extract_response_text",
        lambda response: "import json\nprint('ok')\n",
    )

    agent = MLEngineerAgent()
    _ = agent.generate_code(
        strategy={"title": "Output Policy Strategy", "analysis_type": "predictive", "required_columns": []},
        data_path="data/cleaned_data.csv",
        execution_contract={
            "required_outputs": [
                {"path": "artifacts/ml/cv_metrics.json", "required": True, "intent": "cv_metrics", "kind": "metrics"},
                {"path": "static/plots/*.png", "required": False, "intent": "model_plots", "kind": "visualization"},
            ],
        },
        ml_view={
            "required_outputs": [
                {"path": "artifacts/ml/cv_metrics.json", "required": True, "intent": "cv_metrics", "kind": "metrics"},
                {"path": "static/plots/*.png", "required": False, "intent": "model_plots", "kind": "visualization"},
            ],
        },
    )

    prompt = str(agent.last_prompt or "")
    _assert_contains_all(prompt, "artifacts/ml/cv_metrics.json", "static/plots/*.png", '"required":false')
    _assert_contains_terms(prompt, '"intent":"model_plots"')


def test_build_prompt_does_not_invent_artifact_required_columns_from_canonical(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-openrouter")
    monkeypatch.setattr("src.agents.ml_engineer.OpenAI", _FakeOpenAI)

    def _fake_call_chat_with_fallback(client, messages, models, call_kwargs=None, logger=None, context_tag=None):
        return {"dummy": True}, models[0]

    monkeypatch.setattr("src.agents.ml_engineer.call_chat_with_fallback", _fake_call_chat_with_fallback)
    monkeypatch.setattr(
        "src.agents.ml_engineer.extract_response_text",
        lambda response: "import json\nprint('ok')\n",
    )

    agent = MLEngineerAgent()
    _ = agent.generate_code(
        strategy={"title": "Column Policy Strategy", "analysis_type": "predictive", "required_columns": []},
        data_path="data/cleaned_data.csv",
        execution_contract={
            "canonical_columns": ["target", "identifier", "outcome_flag"],
            "column_roles": {
                "pre_decision": ["feature_a", "feature_b"],
                "outcome": ["outcome_flag"],
                "identifiers": ["identifier"],
            },
            "required_outputs": ["artifacts/ml/cv_metrics.json"],
        },
        ml_view={
            "canonical_columns": ["target", "identifier", "outcome_flag"],
            "column_roles": {
                "pre_decision": ["feature_a", "feature_b"],
                "outcome": ["outcome_flag"],
                "identifiers": ["identifier"],
            },
            "required_outputs": ["artifacts/ml/cv_metrics.json"],
        },
    )

    prompt = str(agent.last_prompt or "")
    _assert_contains_all(prompt, 'Model Input Candidates: ["feature_a","feature_b"]', "Artifact-required Clean Columns: []")
    assert 'Artifact-required Clean Columns: ["target","identifier","outcome_flag"]' not in prompt


def test_generate_code_prompt_preserves_string_runbook(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-openrouter")
    monkeypatch.setattr("src.agents.ml_engineer.OpenAI", _FakeOpenAI)

    def _fake_call_chat_with_fallback(client, messages, models, call_kwargs=None, logger=None, context_tag=None):
        return {"dummy": True}, models[0]

    monkeypatch.setattr("src.agents.ml_engineer.call_chat_with_fallback", _fake_call_chat_with_fallback)
    monkeypatch.setattr(
        "src.agents.ml_engineer.extract_response_text",
        lambda response: "import json\nprint('ok')\n",
    )

    runbook_text = (
        "Parse target_json, extract time_to_hit_hours and event, "
        "then train a discrete-time hazard baseline."
    )

    agent = MLEngineerAgent()
    _ = agent.generate_code(
        strategy={"title": "Survival Strategy", "analysis_type": "survival_analysis", "required_columns": []},
        data_path="data/cleaned_data.csv",
        execution_contract={
            "required_outputs": ["data/metrics.json"],
            "canonical_columns": ["features_json", "target_json"],
            "ml_engineer_runbook": runbook_text,
        },
        ml_view={
            "required_outputs": ["data/metrics.json"],
            "canonical_columns": ["features_json", "target_json"],
            "ml_engineer_runbook": runbook_text,
        },
    )

    prompt = str(agent.last_prompt or "")
    _assert_contains_all(prompt, "time_to_hit_hours")
    _assert_contains_terms(prompt, "discrete-time hazard baseline")


def test_metric_optimization_editor_prompt_uses_optimization_template(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-openrouter")
    monkeypatch.setattr("src.agents.ml_engineer.OpenAI", _FakeOpenAI)

    def _fake_call_chat_with_fallback(client, messages, models, call_kwargs=None, logger=None, context_tag=None):
        return {"dummy": True}, models[0]

    monkeypatch.setattr("src.agents.ml_engineer.call_chat_with_fallback", _fake_call_chat_with_fallback)
    monkeypatch.setattr(
        "src.agents.ml_engineer.extract_response_text",
        lambda response: "import json\nprint('ok')\n",
    )

    agent = MLEngineerAgent()
    _ = agent.generate_code(
        strategy={"title": "Test Strategy", "analysis_type": "predictive", "required_columns": []},
        data_path="data/cleaned_data.csv",
        feedback_history=["# IMPROVEMENT_ROUND\nRESULTS_ADVISOR_FEEDBACK: apply one FE hypothesis."],
        previous_code="print('baseline')\n",
        gate_context={
            "source": "metric_improvement_optimizer",
            "status": "OPTIMIZATION_REQUIRED",
            "feedback": "Optimization round active.",
            "failed_gates": [],
            "required_fixes": ["Apply hypothesis with material edits."],
        },
        iteration_handoff={
            "mode": "optimize",
            "source": "actor_critic_metric_improvement",
            "optimization_focus": {
                "round_id": 1,
                "rounds_allowed": 1,
                "primary_metric_name": "mean_multi_horizon_log_loss",
                "baseline_metric": 0.80,
                "min_delta": 0.0005,
                "higher_is_better": False,
                "feature_engineering_plan": {"techniques": [{"technique": "missing_indicators"}]},
            },
            "optimization_context": {
                "policy": {"phase": "explore", "bundle_size": 1},
                "metric_snapshot": {"primary_metric_name": "mean_multi_horizon_log_loss", "baseline_metric": 0.80},
                "contract_lock": {"required_outputs": ["data/metrics.json"]},
            },
            "editor_constraints": {
                "must_apply_hypothesis": True,
                "forbid_noop": True,
                "patch_intensity": "aggressive",
            },
            "critic_packet": {"analysis_summary": "Baseline stable, no gain yet."},
            "hypothesis_packet": {
                "action": "APPLY",
                "hypothesis": {"technique": "missing_indicators", "target_columns": ["ALL_NUMERIC"]},
            },
            "repair_ground_truth": {
                "root_cause_type": "runtime_api_misuse",
                "repair_focus": "runtime",
                "failure_signature": "TypeError: OneHotEncoder got an unexpected keyword argument 'sparse'",
                "compatibility_notes": [
                    "sklearn.preprocessing.OneHotEncoder does not accept 'sparse'. Verified parameters: categories, drop, sparse_output, dtype, handle_unknown."
                ],
                "environment_facts": [
                    {
                        "fact": "callable_signature",
                        "resolved_symbol": "sklearn.preprocessing.OneHotEncoder",
                        "value": "(*, categories='auto', drop=None, sparse_output=True, dtype=<class 'numpy.float64'>, handle_unknown='error', min_frequency=None, max_categories=None, feature_name_combiner='concat')",
                    }
                ],
                "verified_facts": [
                    {"fact": "unexpected_keyword_argument", "value": "sparse", "source": "runtime_traceback"}
                ],
            },
        },
        execution_contract={
            "required_outputs": [
                {"path": "data/metrics.json", "required": True, "intent": "cv_metrics"},
                {"path": "static/plots/*.png", "required": False, "intent": "model_plots"},
            ],
            "canonical_columns": [],
        },
        ml_view={
            "required_outputs": [
                {"path": "data/metrics.json", "required": True, "intent": "cv_metrics"},
                {"path": "static/plots/*.png", "required": False, "intent": "model_plots"},
            ],
            "allowed_feature_sets": {
                "model_features": ["feature_a", "feature_b"],
                "forbidden_features": ["label_12h"],
            },
            "evaluation_spec": {
                "target_columns": ["label_12h", "label_24h"],
                "primary_metric": "mean_multi_horizon_log_loss",
            },
            "split_spec": {
                "split_column": "__split",
                "training_rows_rule": "rows where label_12h is not missing",
                "scoring_rows_rule": "rows where label_12h is missing",
            },
            "artifact_requirements": {
                "file_schemas": {"submission.csv": {"expected_row_count": 95}},
                "scored_rows_schema": {"required_columns": ["event_id", "prob_12h", "prob_24h"]},
            },
        },
        editor_mode=True,
    )

    prompt = str(agent.last_prompt or "")
    _assert_contains_all(
        prompt,
        "MODE: CODE_EDITOR_MODE_OPTIMIZATION",
        "OPTIMIZATION EDITOR CONTRACT",
        "CURRENT TASK CONTEXT",
        "CURRENT ROUND BRIEF:",
        "ACTIVE HYPOTHESIS (proposal to test, not a literal recipe):",
        "CURRENT EVIDENCE BRIEF:",
        "LOCKED INVARIANTS:",
        "Optimization Authoritative State:",
        "allowed_feature_sets.model_features",
        '"feature_a"',
        "target_columns",
        '"label_12h"',
    )
    _assert_contains_terms(prompt, "cleaning_manifest.json", "csv dialect", "cleaning metadata", "simple arithmetic mean")
    _assert_contains_terms(prompt, "closest compatible variant", "verified environment facts", "onehotencoder", "sparse_output")
    assert "ARTIFACT: data/scored_rows.csv" not in prompt
    assert "STRUCTURED CRITIQUE PACKET:" not in prompt
    assert "OPTIMIZATION CONTEXT (authoritative current round):" not in prompt
    assert "CONTRACT-FIRST EXECUTION MAP (MANDATORY)" not in prompt
    assert "FEATURE GOVERNANCE" not in prompt
    assert '"required":false' in prompt
    assert "{'intent':" not in prompt
    assert '"target_scope":"contract-defined target scope"' in prompt
    assert "HYPOTHESIS TO TEST:" not in prompt


def test_optimization_authoritative_state_accepts_submission_schema_alias_paths(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-openrouter")
    monkeypatch.setattr("src.agents.ml_engineer.OpenAI", _FakeOpenAI)

    def _fake_call_chat_with_fallback(client, messages, models, call_kwargs=None, logger=None, context_tag=None):
        return {"dummy": True}, models[0]

    monkeypatch.setattr("src.agents.ml_engineer.call_chat_with_fallback", _fake_call_chat_with_fallback)
    monkeypatch.setattr(
        "src.agents.ml_engineer.extract_response_text",
        lambda response: "import json\nprint('ok')\n",
    )

    agent = MLEngineerAgent()
    _ = agent.generate_code(
        strategy={"title": "Alias Schema Strategy", "analysis_type": "predictive", "required_columns": []},
        data_path="data/cleaned_data.csv",
        previous_code="print('baseline')\n",
        execution_contract={
            "required_outputs": ["outputs/submission.csv", "data/metrics.json"],
            "artifact_requirements": {
                "clean_dataset": {"output_manifest_path": "artifacts/manifests/custom_clean_manifest.json"},
                "file_schemas": {"outputs/submission.csv": {"expected_row_count": 95}},
                "scored_rows_schema": {"required_columns": ["event_id", "prob_12h"]},
            },
        },
        ml_view={
            "required_outputs": ["outputs/submission.csv", "data/metrics.json"],
            "cleaning_manifest_path": "artifacts/manifests/custom_clean_manifest.json",
            "evaluation_spec": {
                "target_columns": ["label_12h", "label_24h"],
                "primary_metric": "mean_multi_horizon_log_loss",
            },
            "allowed_feature_sets": {"model_features": ["feature_a"]},
            "artifact_requirements": {
                "clean_dataset": {"output_manifest_path": "artifacts/manifests/custom_clean_manifest.json"},
                "file_schemas": {"outputs/submission.csv": {"expected_row_count": 95}},
                "scored_rows_schema": {"required_columns": ["event_id", "prob_12h"]},
            },
        },
        gate_context={
            "source": "metric_improvement_optimizer",
            "status": "OPTIMIZATION_REQUIRED",
            "feedback": "Optimization round active.",
            "failed_gates": [],
            "required_fixes": ["Apply hypothesis with material edits."],
        },
        iteration_handoff={
            "mode": "optimize",
            "source": "actor_critic_metric_improvement",
            "optimization_context": {
                "policy": {"phase": "explore", "bundle_size": 1},
                "metric_snapshot": {"primary_metric_name": "mean_multi_horizon_log_loss", "baseline_metric": 0.80},
                "contract_lock": {"required_outputs": ["data/metrics.json"]},
            },
            "hypothesis_packet": {
                "action": "APPLY",
                "hypothesis": {"technique": "missing_indicators", "target_columns": ["ALL_NUMERIC"]},
            },
            "editor_constraints": {"must_apply_hypothesis": True, "forbid_noop": True},
        },
        editor_mode=True,
    )

    prompt = str(agent.last_prompt or "")
    _assert_contains_all(
        prompt,
        "Optimization Authoritative State:",
        "submission_expected_row_count",
        "outputs/submission.csv",
        "95",
        "artifacts/manifests/custom_clean_manifest.json",
    )


def test_editor_prompt_includes_authoritative_repair_ground_truth(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-openrouter")
    monkeypatch.setattr("src.agents.ml_engineer.OpenAI", _FakeOpenAI)

    def _fake_call_chat_with_fallback(client, messages, models, call_kwargs=None, logger=None, context_tag=None):
        return {"dummy": True}, models[0]

    monkeypatch.setattr("src.agents.ml_engineer.call_chat_with_fallback", _fake_call_chat_with_fallback)
    monkeypatch.setattr(
        "src.agents.ml_engineer.extract_response_text",
        lambda response: "import json\nprint('ok')\n",
    )

    agent = MLEngineerAgent()
    _ = agent.generate_code(
        strategy={"title": "Repair Strategy", "analysis_type": "predictive", "required_columns": []},
        data_path="data/cleaned_data.csv",
        previous_code="import pathlib\nprint('previous')\n",
        gate_context={
            "source": "Execution Runtime",
            "status": "REJECTED",
            "feedback": "Traceback (most recent call last): TypeError",
            "failed_gates": ["runtime_failure"],
            "required_fixes": ["Fix the failing API call."],
        },
        iteration_handoff={
            "mode": "patch",
            "repair_policy": {"repair_first": True, "primary_focus": "runtime"},
            "retry_context": {"error_type": "runtime_api_misuse", "repair_focus": "runtime"},
            "repair_ground_truth": {
                "root_cause_type": "runtime_api_misuse",
                "repair_focus": "runtime",
                "failure_signature": "TypeError: Path.read_text() got an unexpected keyword argument 'extra'",
                "environment_facts": [
                    {
                        "fact": "callable_signature",
                        "resolved_symbol": "pathlib.Path.read_text",
                        "value": "(self, encoding=None, errors=None, newline=None)",
                    }
                ],
                "verified_facts": [
                    {"fact": "unexpected_keyword_argument", "value": "extra", "source": "runtime_traceback"}
                ],
            },
        },
        execution_contract={"required_outputs": ["data/metrics.json"], "canonical_columns": []},
        ml_view={"required_outputs": ["data/metrics.json"]},
        editor_mode=True,
    )

    prompt = str(agent.last_prompt or "")
    _assert_contains_all(prompt, "pathlib.Path.read_text", "unexpected_keyword_argument")
    _assert_contains_terms(prompt, "repair ground truth", "verified environment facts", "authoritative")


def test_editor_prompt_enforces_patch_only_repair_scope(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-openrouter")
    monkeypatch.setattr("src.agents.ml_engineer.OpenAI", _FakeOpenAI)

    def _fake_call_chat_with_fallback(client, messages, models, call_kwargs=None, logger=None, context_tag=None):
        return {"dummy": True}, models[0]

    monkeypatch.setattr("src.agents.ml_engineer.call_chat_with_fallback", _fake_call_chat_with_fallback)
    monkeypatch.setattr(
        "src.agents.ml_engineer.extract_response_text",
        lambda response: "import json\nprint('ok')\n",
    )

    agent = MLEngineerAgent()
    _ = agent.generate_code(
        strategy={"title": "Repair Scope Strategy", "analysis_type": "predictive", "required_columns": []},
        data_path="data/cleaned_data.csv",
        previous_code="def check_writable(path):\n    os.remove(path)\n",
        gate_context={
            "source": "Execution Runtime",
            "status": "REJECTED",
            "feedback": "Security violation in check_writable.",
            "failed_gates": ["runtime_failure"],
            "required_fixes": ["Remove os.remove from check_writable."],
        },
        iteration_handoff={
            "mode": "patch",
            "repair_policy": {"repair_first": True, "primary_focus": "runtime"},
            "editor_constraints": {
                "patch_intensity": "surgical",
                "scope_policy": "patch_only",
                "allow_strategy_changes": False,
                "freeze_unimplicated_regions": True,
            },
            "repair_scope": {
                "phase": "compliance_runtime",
                "scope_policy": "patch_only",
                "editable_targets": ["script_line:2", "call_site:os.remove"],
                "protected_regions": ["training_strategy_and_model_family"],
                "must_preserve_invariants": [
                    "Use the full script as context, but treat unrelated regions as frozen by default.",
                    "Do not widen scope unless new verified runtime evidence directly implicates another block.",
                ],
            },
            "repair_ground_truth": {
                "root_cause_type": "runtime_error",
                "repair_focus": "runtime",
                "failure_signature": "CRITICAL: Security Violations:",
            },
        },
        execution_contract={"required_outputs": ["data/metrics.json"], "canonical_columns": []},
        ml_view={"required_outputs": ["data/metrics.json"]},
        editor_mode=True,
    )

    prompt = str(agent.last_prompt or "")
    _assert_contains_all(prompt, "patch_only", "script_line:2", "call_site:os.remove")
    _assert_contains_terms(
        prompt,
        "repair scope",
        "authoritative edit boundaries",
        "compliance_runtime",
        "patch-only mode",
        "active",
        "outside repair scope",
        "frozen",
        "editable targets",
    )


def test_metric_optimization_runtime_repair_keeps_optimization_editor_prompt(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-openrouter")
    monkeypatch.setattr("src.agents.ml_engineer.OpenAI", _FakeOpenAI)

    def _fake_call_chat_with_fallback(client, messages, models, call_kwargs=None, logger=None, context_tag=None):
        return {"dummy": True}, models[0]

    monkeypatch.setattr("src.agents.ml_engineer.call_chat_with_fallback", _fake_call_chat_with_fallback)
    monkeypatch.setattr(
        "src.agents.ml_engineer.extract_response_text",
        lambda response: "import json\nprint('ok')\n",
    )

    agent = MLEngineerAgent()
    _ = agent.generate_code(
        strategy={"title": "Optimization Repair Strategy", "analysis_type": "predictive", "required_columns": []},
        data_path="data/cleaned_data.csv",
        previous_code="print('baseline')\n",
        gate_context={
            "source": "Execution Runtime",
            "status": "REJECTED",
            "feedback": "ValueError: inconsistent sample lengths during calibration.",
            "failed_gates": ["runtime_failure"],
            "required_fixes": ["Fix the runtime root cause first."],
            "runtime_error": {"type": "shape_or_dtype", "summary": "ValueError: inconsistent sample lengths"},
        },
        iteration_handoff={
            "mode": "patch",
            "source": "result_evaluator_repair_first",
            "repair_policy": {"repair_first": True, "primary_focus": "runtime"},
            "optimization_lane": {
                "active": True,
                "resume_after_repair": True,
                "repair_first": True,
                "repair_focus": "runtime",
                "source": "result_evaluator_repair_first",
                "active_technique": "out_of_fold_probability_calibration_per_horizon",
            },
            "optimization_context": {
                "policy": {"phase": "explore", "bundle_size": 1},
                "metric_snapshot": {"primary_metric_name": "logloss", "baseline_metric": 0.80},
                "contract_lock": {"required_outputs": ["data/metrics.json"]},
            },
            "hypothesis_packet": {
                "action": "APPLY",
                "hypothesis": {
                    "technique": "out_of_fold_probability_calibration_per_horizon",
                    "target_columns": ["ALL_NUMERIC"],
                },
            },
            "editor_constraints": {
                "must_apply_hypothesis": False,
                "forbid_noop": False,
                "patch_intensity": "surgical",
                "scope_policy": "patch_only",
                "allow_strategy_changes": False,
                "freeze_unimplicated_regions": True,
            },
            "repair_scope": {
                "phase": "compliance_runtime",
                "scope_policy": "patch_only",
                "editable_targets": ["calibration_scoring_loop"],
                "protected_regions": ["model_selection", "output_paths_contract"],
                "must_preserve_invariants": ["Keep incumbent outputs and split logic stable."],
            },
            "repair_ground_truth": {
                "root_cause_type": "shape_or_dtype",
                "repair_focus": "runtime",
                "failure_signature": "ValueError: inconsistent sample lengths",
            },
        },
        execution_contract={"required_outputs": ["data/metrics.json"], "canonical_columns": []},
        ml_view={"required_outputs": ["data/metrics.json"]},
        editor_mode=True,
    )

    prompt = str(agent.last_prompt or "")
    _assert_contains_all(
        prompt,
        "MODE: CODE_EDITOR_MODE_OPTIMIZATION",
        "CURRENT ROUND BRIEF:",
        "CURRENT PHASE:",
        "runtime_repair",
        "calibration_scoring_loop",
    )
    _assert_contains_terms(prompt, "repair ground truth", "repair scope")


def test_generate_code_records_subcall_trace_with_completion_reprompt(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-openrouter")
    monkeypatch.setattr("src.agents.ml_engineer.OpenAI", _TraceOpenAI)

    def _fake_call_chat_with_fallback(client, messages, models, call_kwargs=None, logger=None, context_tag=None):
        return {"dummy": True}, models[0]

    monkeypatch.setattr("src.agents.ml_engineer.call_chat_with_fallback", _fake_call_chat_with_fallback)
    monkeypatch.setattr(
        "src.agents.ml_engineer.extract_response_text",
        lambda response: "print('partial')\n",
    )

    _TraceOpenAI.queued_outputs = ["import json\nprint('completed')\n"]
    agent = MLEngineerAgent()

    checks = {"count": 0}

    def _fake_completeness(code, required_outputs):
        checks["count"] += 1
        return ["missing outputs"] if checks["count"] == 1 else []

    monkeypatch.setattr(agent, "_check_script_completeness", _fake_completeness)

    _ = agent.generate_code(
        strategy={"title": "Trace Strategy", "analysis_type": "predictive", "required_columns": []},
        data_path="data/cleaned_data.csv",
        execution_contract={"required_outputs": ["data/metrics.json"], "canonical_columns": []},
        ml_view={"required_outputs": ["data/metrics.json"]},
    )

    trace = agent.last_prompt_trace
    assert len(trace) == 2
    assert trace[0]["stage"] == "build_generation"
    assert trace[1]["stage"] == "completion_reprompt"
    _assert_contains_terms(trace[1]["prompt"], "complete runnable python script")
    assert "print('completed')" in trace[1]["response"]


def test_normalize_iteration_handoff_falls_back_to_feedback_record():
    agent = MLEngineerAgent()
    normalized = agent._normalize_iteration_handoff(
        iteration_handoff={
            "mode": "patch",
            "source": "result_evaluator",
            "contract_focus": {"required_outputs": ["artifacts/ml/cv_metrics.json"]},
            "quality_focus": {"failed_gates": ["metric_above_random_baseline"]},
            "feedback": {"reviewer": "review ok", "qa": "qa mismatch"},
        },
        gate_context={
            "feedback_record": {
                "retry_context": {
                    "error_type": "review_gate_failure",
                    "repair_focus": "compliance",
                },
                "repair_ground_truth": {
                    "root_cause_type": "review_gate_failure",
                    "repair_focus": "compliance",
                    "failure_signature": "Failed gates: metric_above_random_baseline",
                },
                "repair_scope": {
                    "phase": "compliance_runtime",
                    "scope_policy": "patch_only",
                },
                "incumbent_brief": {
                    "primary_metric": "top_decile_lift",
                    "incumbent_score": 9.995,
                    "model_info": "LightGBM",
                },
            }
        },
        required_deliverables=["artifacts/ml/cv_metrics.json"],
    )

    assert normalized["retry_context"]["error_type"] == "review_gate_failure"
    assert normalized["repair_ground_truth"]["root_cause_type"] == "review_gate_failure"
    assert normalized["repair_scope"]["scope_policy"] == "patch_only"
    assert normalized["incumbent_brief"]["incumbent_score"] == 9.995
