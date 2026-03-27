import os

os.environ.setdefault("DEEPSEEK_API_KEY", "dummy")
os.environ.setdefault("GOOGLE_API_KEY", "dummy")
os.environ.setdefault("SANDBOX_PROVIDER", "local")
os.environ.setdefault("OPENROUTER_API_KEY", "dummy-openrouter")

from src.graph.graph import (
    cleaning_reviewer,
    data_engineer,
    execution_planner,
    failure_explainer,
    get_runtime_agent_models,
    ml_engineer,
    qa_reviewer,
    results_advisor,
    review_board,
    reviewer,
    set_runtime_agent_models,
    steward,
    strategist,
    translator,
)


def test_set_runtime_agent_models_derives_ml_editor_from_primary(monkeypatch):
    monkeypatch.delenv("OPENROUTER_ML_EDITOR_MODEL", raising=False)
    original_model = getattr(ml_engineer, "model_name", "")
    original_editor = getattr(ml_engineer, "editor_model_name", "")
    try:
        models = set_runtime_agent_models({"ml_engineer": "openai/gpt-5.4"})
        assert ml_engineer.model_name == "openai/gpt-5.4"
        assert ml_engineer.editor_model_name == "openai/gpt-5.4-mini"
        assert models["ml_engineer"] == "openai/gpt-5.4"
        assert models["ml_engineer_editor"] == "openai/gpt-5.4-mini"
    finally:
        ml_engineer.model_name = original_model
        ml_engineer.editor_model_name = original_editor


def test_set_runtime_agent_models_accepts_explicit_ml_editor_override(monkeypatch):
    monkeypatch.delenv("OPENROUTER_ML_EDITOR_MODEL", raising=False)
    original_model = getattr(ml_engineer, "model_name", "")
    original_editor = getattr(ml_engineer, "editor_model_name", "")
    try:
        models = set_runtime_agent_models(
            {
                "ml_engineer": "openai/gpt-5.4",
                "ml_engineer_editor": "openai/gpt-5.4-mini",
            }
        )
        assert ml_engineer.model_name == "openai/gpt-5.4"
        assert ml_engineer.editor_model_name == "openai/gpt-5.4-mini"
        assert models["ml_engineer_editor"] == "openai/gpt-5.4-mini"
        assert get_runtime_agent_models()["ml_engineer_editor"] == "openai/gpt-5.4-mini"
    finally:
        ml_engineer.model_name = original_model
        ml_engineer.editor_model_name = original_editor


def test_set_runtime_agent_models_updates_extended_runtime_slots(monkeypatch):
    monkeypatch.delenv("OPENROUTER_ML_EDITOR_MODEL", raising=False)
    originals = {
        "steward_model": getattr(steward, "model_name", ""),
        "strategist_model": getattr(strategist, "model_name", ""),
        "strategist_fallback": getattr(strategist, "fallback_model_name", ""),
        "strategist_chain": list(getattr(strategist, "model_chain", []) or []),
        "execution_planner_model": getattr(execution_planner, "model_name", ""),
        "execution_planner_compiler": getattr(execution_planner, "compiler_model_name", ""),
        "execution_planner_chain": list(getattr(execution_planner, "model_chain", []) or []),
        "execution_planner_default_chain": list(getattr(execution_planner, "_default_model_chain", []) or []),
        "execution_planner_client_model": getattr(getattr(execution_planner, "client", None), "model_name", ""),
        "data_engineer_model": getattr(data_engineer, "model_name", ""),
        "data_engineer_fallback": getattr(data_engineer, "fallback_model_name", ""),
        "ml_engineer_model": getattr(ml_engineer, "model_name", ""),
        "ml_engineer_editor": getattr(ml_engineer, "editor_model_name", ""),
        "ml_engineer_fallback": getattr(ml_engineer, "fallback_model_name", ""),
        "cleaning_reviewer_model": getattr(cleaning_reviewer, "model_name", ""),
        "reviewer_model": getattr(reviewer, "model_name", ""),
        "qa_reviewer_model": getattr(qa_reviewer, "model_name", ""),
        "review_board_model": getattr(review_board, "model_name", ""),
        "translator_model": getattr(translator, "model_name", ""),
        "results_advisor_model": getattr(results_advisor, "model_name", ""),
        "results_advisor_critique": getattr(results_advisor, "critique_model_name", ""),
        "results_advisor_llm": getattr(results_advisor, "fe_model_name", ""),
        "failure_explainer_model": getattr(failure_explainer, "_model_name", ""),
    }
    try:
        overrides = {
            "steward": "google/gemini-3-flash-preview",
            "strategist": "openai/gpt-5.4",
            "strategist_fallback": "moonshotai/kimi-k2.5",
            "execution_planner": "google/gemini-3.1-pro-preview",
            "execution_planner_compiler": "google/gemini-3-flash-preview",
            "data_engineer": "openai/gpt-5.4-mini",
            "data_engineer_fallback": "minimax/minimax-m2.5",
            "ml_engineer": "openai/gpt-5.4",
            "ml_engineer_editor": "openai/gpt-5.4-mini",
            "ml_engineer_fallback": "moonshotai/kimi-k2.5",
            "cleaning_reviewer": "google/gemini-3-flash-preview",
            "reviewer": "openai/gpt-5.4-mini",
            "qa_reviewer": "openai/gpt-5.4-mini",
            "review_board": "openai/gpt-5.4-mini",
            "translator": "google/gemini-3-flash-preview",
            "results_advisor": "google/gemini-3-flash-preview",
            "results_advisor_critique": "openai/gpt-5.4",
            "results_advisor_llm": "google/gemini-3-flash-preview",
            "failure_explainer": "google/gemini-3-flash-preview",
        }
        models = set_runtime_agent_models(overrides)

        assert steward.model_name == overrides["steward"]
        assert strategist.model_name == overrides["strategist"]
        assert strategist.fallback_model_name == overrides["strategist_fallback"]
        assert strategist.model_chain == [overrides["strategist"], overrides["strategist_fallback"]]
        assert execution_planner.model_name == overrides["execution_planner"]
        assert execution_planner.compiler_model_name == overrides["execution_planner_compiler"]
        assert execution_planner.model_chain == [overrides["execution_planner"]]
        assert execution_planner._default_model_chain == [overrides["execution_planner"]]
        assert getattr(execution_planner.client, "model_name", "") == overrides["execution_planner"]
        assert data_engineer.model_name == overrides["data_engineer"]
        assert data_engineer.fallback_model_name == overrides["data_engineer_fallback"]
        assert ml_engineer.model_name == overrides["ml_engineer"]
        assert ml_engineer.editor_model_name == overrides["ml_engineer_editor"]
        assert ml_engineer.fallback_model_name == overrides["ml_engineer_fallback"]
        assert cleaning_reviewer.model_name == overrides["cleaning_reviewer"]
        assert reviewer.model_name == overrides["reviewer"]
        assert qa_reviewer.model_name == overrides["qa_reviewer"]
        assert review_board.model_name == overrides["review_board"]
        assert translator.model_name == overrides["translator"]
        assert results_advisor.model_name == overrides["results_advisor"]
        assert results_advisor.critique_model_name == overrides["results_advisor_critique"]
        assert results_advisor.fe_model_name == overrides["results_advisor_llm"]
        assert failure_explainer._model_name == overrides["failure_explainer"]
        for key, value in overrides.items():
            assert models[key] == value
        runtime_snapshot = get_runtime_agent_models()
        for key, value in overrides.items():
            assert runtime_snapshot[key] == value
    finally:
        steward.model_name = originals["steward_model"]
        strategist.model_name = originals["strategist_model"]
        strategist.fallback_model_name = originals["strategist_fallback"]
        strategist.model_chain = originals["strategist_chain"]
        execution_planner.model_name = originals["execution_planner_model"]
        execution_planner.compiler_model_name = originals["execution_planner_compiler"]
        execution_planner.model_chain = originals["execution_planner_chain"]
        execution_planner._default_model_chain = originals["execution_planner_default_chain"]
        if getattr(execution_planner, "client", None) is not None:
            execution_planner.client.model_name = originals["execution_planner_client_model"]
        data_engineer.model_name = originals["data_engineer_model"]
        data_engineer.fallback_model_name = originals["data_engineer_fallback"]
        ml_engineer.model_name = originals["ml_engineer_model"]
        ml_engineer.editor_model_name = originals["ml_engineer_editor"]
        ml_engineer.fallback_model_name = originals["ml_engineer_fallback"]
        cleaning_reviewer.model_name = originals["cleaning_reviewer_model"]
        reviewer.model_name = originals["reviewer_model"]
        qa_reviewer.model_name = originals["qa_reviewer_model"]
        review_board.model_name = originals["review_board_model"]
        translator.model_name = originals["translator_model"]
        results_advisor.model_name = originals["results_advisor_model"]
        results_advisor.critique_model_name = originals["results_advisor_critique"]
        results_advisor.fe_model_name = originals["results_advisor_llm"]
        failure_explainer._model_name = originals["failure_explainer_model"]
