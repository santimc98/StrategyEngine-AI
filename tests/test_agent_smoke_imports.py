"""
Smoke tests: verify every agent can be imported and instantiated without errors.

These tests catch missing imports (like the `import copy` bug in ml_engineer.py),
broken module-level code, and constructor issues — all WITHOUT calling any LLM API.
"""

import pytest


# ── Module-level import tests ─────────────────────────────────────────────────
# Each test imports the module and the main class. If any stdlib import is
# missing (e.g. `import copy`), or module-level code crashes, these fail fast.


class TestAgentImports:
    """Verify that every agent module can be imported without errors."""

    def test_import_execution_planner(self):
        from src.agents.execution_planner import ExecutionPlannerAgent
        assert ExecutionPlannerAgent is not None

    def test_import_ml_engineer(self):
        from src.agents.ml_engineer import MLEngineerAgent
        assert MLEngineerAgent is not None

    def test_import_data_engineer(self):
        from src.agents.data_engineer import DataEngineerAgent
        assert DataEngineerAgent is not None

    def test_import_reviewer(self):
        from src.agents.reviewer import ReviewerAgent
        assert ReviewerAgent is not None

    def test_import_qa_reviewer(self):
        from src.agents.qa_reviewer import QAReviewerAgent
        assert QAReviewerAgent is not None

    def test_import_cleaning_reviewer(self):
        from src.agents.cleaning_reviewer import CleaningReviewerAgent
        assert CleaningReviewerAgent is not None

    def test_import_steward(self):
        from src.agents.steward import StewardAgent
        assert StewardAgent is not None

    def test_import_strategist(self):
        from src.agents.strategist import StrategistAgent
        assert StrategistAgent is not None

    def test_import_model_analyst(self):
        from src.agents.model_analyst import ModelAnalystAgent
        assert ModelAnalystAgent is not None

    def test_import_results_advisor(self):
        from src.agents.results_advisor import ResultsAdvisorAgent
        assert ResultsAdvisorAgent is not None

    def test_import_review_board(self):
        from src.agents.review_board import ReviewBoardAgent
        assert ReviewBoardAgent is not None

    def test_import_business_translator(self):
        from src.agents.business_translator import BusinessTranslatorAgent
        assert BusinessTranslatorAgent is not None

    def test_import_failure_explainer(self):
        from src.agents.failure_explainer import FailureExplainerAgent
        assert FailureExplainerAgent is not None


# ── Instantiation tests ───────────────────────────────────────────────────────
# Verify each agent can be constructed with api_key=None (no LLM calls).
# This catches issues in __init__ that go beyond import-time errors.


class TestAgentInstantiation:
    """Verify that every agent can be instantiated with api_key=None."""

    def test_instantiate_execution_planner(self):
        from src.agents.execution_planner import ExecutionPlannerAgent
        agent = ExecutionPlannerAgent(api_key=None)
        assert agent is not None

    def test_instantiate_ml_engineer(self):
        from src.agents.ml_engineer import MLEngineerAgent
        agent = MLEngineerAgent(api_key=None)
        assert agent is not None

    def test_instantiate_data_engineer(self):
        from src.agents.data_engineer import DataEngineerAgent
        agent = DataEngineerAgent(api_key=None)
        assert agent is not None

    def test_instantiate_reviewer(self):
        from src.agents.reviewer import ReviewerAgent
        agent = ReviewerAgent(api_key=None)
        assert agent is not None

    def test_instantiate_qa_reviewer(self):
        from src.agents.qa_reviewer import QAReviewerAgent
        agent = QAReviewerAgent(api_key=None)
        assert agent is not None

    def test_instantiate_cleaning_reviewer(self):
        from src.agents.cleaning_reviewer import CleaningReviewerAgent
        agent = CleaningReviewerAgent(api_key=None)
        assert agent is not None

    def test_instantiate_steward(self):
        from src.agents.steward import StewardAgent
        agent = StewardAgent(api_key=None)
        assert agent is not None

    def test_instantiate_strategist(self):
        from src.agents.strategist import StrategistAgent
        agent = StrategistAgent(api_key=None)
        assert agent is not None

    def test_instantiate_model_analyst(self):
        from src.agents.model_analyst import ModelAnalystAgent
        agent = ModelAnalystAgent(api_key=None)
        assert agent is not None

    def test_instantiate_results_advisor(self):
        from src.agents.results_advisor import ResultsAdvisorAgent
        agent = ResultsAdvisorAgent(api_key=None)
        assert agent is not None

    def test_instantiate_review_board(self):
        from src.agents.review_board import ReviewBoardAgent
        agent = ReviewBoardAgent(api_key=None)
        assert agent is not None

    def test_instantiate_business_translator(self):
        from src.agents.business_translator import BusinessTranslatorAgent
        agent = BusinessTranslatorAgent(api_key=None)
        assert agent is not None

    def test_instantiate_failure_explainer(self):
        from src.agents.failure_explainer import FailureExplainerAgent
        agent = FailureExplainerAgent(api_key=None)
        assert agent is not None


# ── Critical stdlib import verification ───────────────────────────────────────
# Verify that modules which use `copy.deepcopy` actually import `copy`.
# This is a targeted regression test for the ml_engineer.py bug.


class TestStdlibImportsPresent:
    """Verify critical stdlib modules are imported where used."""

    @pytest.mark.parametrize("module_path", [
        "src/agents/execution_planner.py",
        "src/agents/ml_engineer.py",
        "src/agents/reviewer.py",
        "src/agents/qa_reviewer.py",
        "src/agents/results_advisor.py",
    ])
    def test_copy_imported_where_used(self, module_path):
        """If a module uses copy.deepcopy/copy.copy, it must import copy."""
        from pathlib import Path
        content = Path(module_path).read_text(encoding="utf-8")
        uses_copy = "copy.deepcopy" in content or "copy.copy" in content
        if uses_copy:
            has_import = "import copy" in content
            assert has_import, (
                f"{module_path} uses copy.deepcopy/copy.copy but does not "
                f"import the copy module"
            )

    @pytest.mark.parametrize("module_path", [
        "src/agents/execution_planner.py",
        "src/agents/ml_engineer.py",
        "src/agents/data_engineer.py",
        "src/agents/reviewer.py",
        "src/agents/qa_reviewer.py",
        "src/agents/cleaning_reviewer.py",
        "src/agents/steward.py",
        "src/agents/strategist.py",
        "src/agents/model_analyst.py",
        "src/agents/results_advisor.py",
        "src/agents/review_board.py",
        "src/agents/business_translator.py",
        "src/agents/failure_explainer.py",
    ])
    def test_syntax_valid(self, module_path):
        """Every agent source file must parse without syntax errors."""
        import ast
        from pathlib import Path
        source = Path(module_path).read_text(encoding="utf-8")
        try:
            ast.parse(source)
        except SyntaxError as e:
            pytest.fail(f"{module_path} has syntax error: {e}")


# ── Key utility imports ───────────────────────────────────────────────────────
# The graph and core utils must also import cleanly.


class TestCoreUtilImports:
    """Verify critical utility modules import without errors."""

    def test_import_contract_views(self):
        from src.utils.contract_views import build_de_view, build_ml_view
        assert build_de_view is not None
        assert build_ml_view is not None

    def test_import_contract_validator(self):
        from src.utils.contract_validator import validate_contract
        assert validate_contract is not None

    def test_import_contract_schema_registry(self):
        from src.utils.contract_schema_registry import apply_contract_schema_registry_repairs
        assert apply_contract_schema_registry_repairs is not None

    def test_import_feature_selectors(self):
        from src.utils.feature_selectors import infer_feature_selectors
        assert infer_feature_selectors is not None

    def test_import_sandbox_provider(self):
        from src.utils.sandbox_provider import SandboxProvider
        assert SandboxProvider is not None

    def test_import_governance_reducer(self):
        from src.utils.governance_reducer import compute_governance_verdict
        assert compute_governance_verdict is not None

    def test_import_llm_fallback(self):
        from src.utils.llm_fallback import call_chat_with_fallback
        assert call_chat_with_fallback is not None

    def test_import_code_extract(self):
        from src.utils.code_extract import extract_code_block, is_syntax_valid
        assert extract_code_block is not None
        assert is_syntax_valid is not None

    def test_import_output_contract(self):
        from src.utils.output_contract import build_output_contract_report
        assert build_output_contract_report is not None

    def test_import_data_profile_compact(self):
        from src.utils.data_profile_compact import compact_data_profile_for_llm
        assert compact_data_profile_for_llm is not None
