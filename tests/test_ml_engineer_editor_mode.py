import os

os.environ.setdefault("DEEPSEEK_API_KEY", "dummy")
os.environ.setdefault("GOOGLE_API_KEY", "dummy")
os.environ.setdefault("SANDBOX_PROVIDER", "local")
os.environ.setdefault("OPENROUTER_API_KEY", "dummy-openrouter")

from src.graph.graph import run_engineer


def test_editor_mode_skips_plan_generation(monkeypatch):
    calls = {"plan_called": False}

    def _fake_plan(*args, **kwargs):
        calls["plan_called"] = True
        raise AssertionError("generate_ml_plan should not run in editor mode")

    def _fake_generate_code(*, editor_mode=False, last_run_memory=None, **kwargs):
        calls["editor_mode"] = editor_mode
        calls["last_run_memory"] = last_run_memory
        return "print('patched')"

    monkeypatch.setattr("src.graph.graph.ml_engineer.generate_ml_plan", _fake_plan, raising=True)
    monkeypatch.setattr("src.graph.graph.ml_engineer.generate_code", _fake_generate_code, raising=True)
    monkeypatch.setattr(
        "src.graph.graph.load_recent_memory",
        lambda run_id, k=5: [{"iter": 1, "attempt": 1, "event": "runtime_error"}],
        raising=True,
    )

    state = {
        "run_id": "unit_editor_mode",
        "selected_strategy": {"title": "Strategy", "analysis_type": "predictive", "required_columns": []},
        "feedback_history": [],
        "data_summary": "",
        "business_objective": "",
        "csv_encoding": "utf-8",
        "csv_sep": ",",
        "csv_decimal": ".",
        "generated_code": "print('prev')",
        "last_generated_code": "print('prev')",
        "last_gate_context": {
            "source": "Execution Runtime",
            "status": "REJECTED",
            "feedback": "Traceback (most recent call last): ValueError: boom",
            "failed_gates": ["Runtime Stability"],
            "required_fixes": ["Fix the exception."],
        },
        "iteration_count": 0,
        "execution_contract": {"required_outputs": ["data/metrics.json"], "canonical_columns": []},
    }

    result = run_engineer(state)

    assert calls["plan_called"] is False
    assert calls.get("editor_mode") is True
    assert isinstance(calls.get("last_run_memory"), list) and calls["last_run_memory"]
    assert result.get("generated_code") == "print('patched')"


def test_editor_mode_forced_after_first_iteration_with_reviewer_feedback(monkeypatch):
    calls = {"plan_called": False}

    def _fake_plan(*args, **kwargs):
        calls["plan_called"] = True
        raise AssertionError("generate_ml_plan should not run after baseline iteration")

    def _fake_generate_code(*, editor_mode=False, previous_code=None, **kwargs):
        calls["editor_mode"] = editor_mode
        calls["previous_code"] = previous_code
        return "print('patched-reviewer')"

    monkeypatch.setattr("src.graph.graph.ml_engineer.generate_ml_plan", _fake_plan, raising=True)
    monkeypatch.setattr("src.graph.graph.ml_engineer.generate_code", _fake_generate_code, raising=True)
    monkeypatch.setattr(
        "src.graph.graph.load_recent_memory",
        lambda run_id, k=5: [],
        raising=True,
    )

    state = {
        "run_id": "unit_editor_mode_review",
        "selected_strategy": {"title": "Strategy", "analysis_type": "predictive", "required_columns": []},
        "feedback_history": ["Reviewer asked for small targeted fixes."],
        "data_summary": "",
        "business_objective": "",
        "csv_encoding": "utf-8",
        "csv_sep": ",",
        "csv_decimal": ".",
        "generated_code": "print('baseline')",
        "last_generated_code": "print('baseline')",
        "last_gate_context": {
            "source": "review_board",
            "status": "NEEDS_IMPROVEMENT",
            "feedback": "Fix submission schema and keep training unchanged.",
            "failed_gates": ["submission_format_validation"],
            "required_fixes": ["Only patch persistence/output section."],
        },
        "iteration_count": 1,
        "execution_contract": {"required_outputs": ["data/metrics.json"], "canonical_columns": []},
    }

    result = run_engineer(state)

    assert calls["plan_called"] is False
    assert calls.get("editor_mode") is True
    assert calls.get("previous_code") == "print('baseline')"
    assert result.get("generated_code") == "print('patched-reviewer')"


def test_metric_round_apply_guard_reprompts_once_when_hypothesis_not_applied(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    calls = {"count": 0}
    previous_code = (
        "def build_features(df):\n"
        "    return df\n"
        "\n"
        "def train(df):\n"
        "    return build_features(df)\n"
    )
    changed_code = (
        "def build_features(df):\n"
        "    df = df.copy()\n"
        "    for col in df.select_dtypes(include=['number']).columns:\n"
        "        df[f\"{col}_is_missing\"] = df[col].isna().astype('Int64')\n"
        "    return df\n"
        "\n"
        "def train(df):\n"
        "    return build_features(df)\n"
    )

    def _fake_generate_code(*, editor_mode=False, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return previous_code
        return changed_code

    monkeypatch.setattr("src.graph.graph.ml_engineer.generate_code", _fake_generate_code, raising=True)
    monkeypatch.setattr(
        "src.graph.graph.load_recent_memory",
        lambda run_id, k=5: [],
        raising=True,
    )

    state = {
        "run_id": "unit_metric_round_guard",
        "selected_strategy": {"title": "Strategy", "analysis_type": "predictive", "required_columns": []},
        "feedback_history": [],
        "data_summary": "",
        "business_objective": "",
        "csv_encoding": "utf-8",
        "csv_sep": ",",
        "csv_decimal": ".",
        "generated_code": previous_code,
        "last_generated_code": previous_code,
        "last_gate_context": {
            "source": "review_board",
            "status": "NEEDS_IMPROVEMENT",
            "feedback": "Apply one hypothesis with material edits.",
            "failed_gates": ["metric_improvement_round"],
            "required_fixes": ["Apply active hypothesis."],
        },
        "iteration_count": 1,
        "execution_contract": {"required_outputs": ["data/metrics.json"], "canonical_columns": []},
        "ml_improvement_round_active": True,
        "ml_improvement_hypothesis_packet": {
            "action": "APPLY",
            "hypothesis": {"technique": "missing_indicators", "target_columns": ["ALL_NUMERIC"]},
        },
    }

    result = run_engineer(state)

    assert calls["count"] == 2
    assert result.get("generated_code") == changed_code
    apply_guard = result.get("ml_improvement_apply_guard_report") or {}
    assert apply_guard.get("repair_attempted") is True
    assert apply_guard.get("applied") is True


def test_metric_round_forces_editor_mode_even_on_first_iteration(monkeypatch):
    calls = {}

    def _fake_generate_code(*, editor_mode=False, previous_code=None, **kwargs):
        calls["editor_mode"] = editor_mode
        calls["previous_code"] = previous_code
        return "print('optimized')"

    monkeypatch.setattr("src.graph.graph.ml_engineer.generate_code", _fake_generate_code, raising=True)
    monkeypatch.setattr(
        "src.graph.graph.load_recent_memory",
        lambda run_id, k=5: [],
        raising=True,
    )

    state = {
        "run_id": "unit_metric_round_editor_force",
        "selected_strategy": {"title": "Strategy", "analysis_type": "predictive", "required_columns": []},
        "feedback_history": [],
        "data_summary": "",
        "business_objective": "",
        "csv_encoding": "utf-8",
        "csv_sep": ",",
        "csv_decimal": ".",
        "generated_code": "print('baseline')",
        "last_generated_code": "print('baseline')",
        "last_gate_context": {
            "source": "metric_improvement_optimizer",
            "status": "OPTIMIZATION_REQUIRED",
            "feedback": "Apply hypothesis with material edits.",
            "failed_gates": [],
            "required_fixes": ["Apply hypothesis."],
        },
        "iteration_handoff": {
            "mode": "optimize",
            "source": "actor_critic_metric_improvement",
            "editor_constraints": {"must_apply_hypothesis": True, "forbid_noop": True},
            "hypothesis_packet": {"action": "APPLY", "hypothesis": {"technique": "missing_indicators"}},
        },
        "ml_improvement_round_active": True,
        "iteration_count": 0,
        "execution_contract": {"required_outputs": ["data/metrics.json"], "canonical_columns": []},
    }

    result = run_engineer(state)

    assert calls.get("editor_mode") is True
    assert calls.get("previous_code") == "print('baseline')"
    assert result.get("generated_code") == "print('optimized')"
