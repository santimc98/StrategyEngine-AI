import json
import pytest
from typing import Any, Dict
from unittest.mock import MagicMock, patch
from src.agents.strategist import StrategistAgent
from src.graph.graph import run_strategist


def _mock_llm_response(payload: Dict[str, Any]) -> MagicMock:
    response = MagicMock()
    response.text = json.dumps(payload)
    return response


def _assert_contains_all(text: str, *needles: str) -> None:
    for needle in needles:
        assert needle in text


def _assert_contains_terms(text: str, *terms: str) -> None:
    lowered = text.lower()
    for term in terms:
        assert term.lower() in lowered


class TestStrategistNormalization:
    
    def setup_method(self):
        self.agent = StrategistAgent(api_key="dummy_key")

    def test_normalize_dict_with_strategies_list(self):
        """Case: parsed = {'strategies': [{'title': 'A'}, {'title': 'B'}]}"""
        parsed = {"strategies": [{"title": "A"}, {"title": "B"}]}
        normalized = self.agent._normalize_strategist_output(parsed)
        assert isinstance(normalized, dict)
        assert "strategies" in normalized
        assert len(normalized["strategies"]) == 2
        assert normalized["strategies"][0]["title"] == "A"

    def test_normalize_dict_with_strategies_dict(self):
        """Case: parsed = {'strategies': {'title': 'A'}} -> convert to list"""
        parsed = {"strategies": {"title": "A"}}
        normalized = self.agent._normalize_strategist_output(parsed)
        assert isinstance(normalized["strategies"], list)
        assert len(normalized["strategies"]) == 1
        assert normalized["strategies"][0]["title"] == "A"

    def test_normalize_list_of_dicts(self):
        """Case: parsed = [{'title': 'A'}, {'title': 'B'}] -> wrap in dict"""
        parsed = [{"title": "A"}, {"title": "B"}]
        normalized = self.agent._normalize_strategist_output(parsed)
        assert isinstance(normalized, dict)
        assert "strategies" in normalized
        assert len(normalized["strategies"]) == 2

    def test_normalize_single_strategy_dict_without_key(self):
        """Case: parsed = {'title': 'A', ...} -> wrap in strategies list"""
        parsed = {"title": "A", "objective_type": "descriptive"}
        normalized = self.agent._normalize_strategist_output(parsed)
        assert isinstance(normalized, dict)
        assert "strategies" in normalized
        assert len(normalized["strategies"]) == 1
        assert normalized["strategies"][0]["title"] == "A"

    def test_normalize_feature_engineering_aliases(self):
        parsed = {
            "strategies": [
                {
                    "title": "A",
                    "feature_engineering_strategy": [{"technique": "interaction", "columns": ["x", "y"]}],
                }
            ]
        }
        normalized = self.agent._normalize_strategist_output(parsed)
        strategy = normalized["strategies"][0]
        fe_strategy = strategy.get("feature_engineering_strategy")
        assert isinstance(strategy.get("feature_engineering"), list)
        assert isinstance(fe_strategy, dict)
        assert isinstance(fe_strategy.get("techniques"), list)
        assert fe_strategy.get("risk_level") in {"low", "med", "high"}
        assert strategy.get("feature_engineering") == fe_strategy.get("techniques")

    def test_normalize_garbage(self):
        """Case: parsed = 'garbage' or None -> empty strategies"""
        assert self.agent._normalize_strategist_output("garbage")["strategies"] == []
        assert self.agent._normalize_strategist_output(None)["strategies"] == []
        assert self.agent._normalize_strategist_output(123)["strategies"] == []

    def test_validate_required_columns_detects_invalid_names(self):
        payload = {
            "strategies": [
                {"title": "s1", "required_columns": ["valid_a", "invalid_x"]},
                {"title": "s2", "required_columns": [{"name": "valid_b"}, {"column": "invalid_y"}]},
            ]
        }
        validation = self.agent._validate_required_columns(payload, ["valid_a", "valid_b"])
        assert validation["status"] == "invalid_required_columns"
        assert validation["invalid_count"] == 2
        assert len(validation["invalid_details"]) == 2

    def test_build_strategy_spec_includes_target_columns(self):
        payload = {
            "strategies": [
                {
                    "title": "Churn Plan",
                    "objective_type": "predictive",
                    "target_columns": ["churn_flag"],
                    "required_columns": ["customer_id", "churn_flag"],
                }
            ]
        }

        spec = self.agent._build_strategy_spec_from_llm(
            payload,
            data_summary='{"primary_target":"churn_flag"}',
            user_request="Predict churn",
        )

        assert spec.get("target_columns") == ["churn_flag"]
        assert (spec.get("evaluation_plan") or {}).get("target_columns") == ["churn_flag"]

    def test_build_strategy_spec_reads_feature_engineering_alias(self):
        payload = {
            "strategies": [
                {
                    "title": "FE Plan",
                    "objective_type": "predictive",
                    "feature_engineering": [{"technique": "log_transform", "columns": ["income"]}],
                }
            ]
        }
        spec = self.agent._build_strategy_spec_from_llm(
            payload,
            data_summary="{}",
            user_request="Predict value",
        )
        assert isinstance(spec.get("feature_engineering"), list)
        assert (spec.get("evaluation_plan") or {}).get("feature_engineering") == spec.get("feature_engineering")

    def test_build_strategy_spec_preserves_descriptive_when_llm_reasoned_cleaning_scope(self):
        payload = {
            "strategies": [
                {
                    "title": "CRM Cleaning Prep",
                    "objective_type": "descriptive",
                    "scope_recommendation": "cleaning_only",
                    "objective_reasoning": "This run is for audit and preparation, not model training.",
                    "validation_strategy": "data_quality_validation_with_rule_based_checks",
                    "validation_rationale": "The run should validate cleaning quality and leakage controls.",
                    "recommended_evaluation_metrics": ["retained_rows_after_cleaning"],
                    "recommended_artifacts": [
                        {"artifact_type": "clean_dataset", "required": True, "rationale": "Needed for handoff."},
                        {"artifact_type": "data_dictionary", "required": True, "rationale": "Needed for traceability."},
                    ],
                }
            ]
        }

        spec = self.agent._build_strategy_spec_from_llm(
            payload,
            data_summary='{"primary_target":"converted_to_opportunity_90d"}',
            user_request="Auditar y limpiar el CRM antes de modelar",
        )

        assert spec.get("objective_type") == "descriptive"
        assert spec.get("scope_recommendation") == "cleaning_only"
        artifact_types = [a.get("artifact_type") for a in (spec.get("recommended_artifacts") or []) if isinstance(a, dict)]
        assert "predictions_or_scores" not in artifact_types
        assert "clean_dataset" in artifact_types

    def test_generate_prompt_includes_senior_contextual_sections_for_scope_and_artifacts(self):
        payload = {
            "strategies": [
                {
                    "title": "Baseline Churn",
                    "objective_type": "predictive",
                    "scope_recommendation": "ml_only",
                    "scope_reasoning": "Data is already prepared and this run should focus on modeling only.",
                    "objective_reasoning": "Predict churn probability for retention actions.",
                    "success_metric": "roc_auc",
                    "recommended_evaluation_metrics": ["roc_auc"],
                    "validation_strategy": "stratified_cv",
                    "validation_rationale": "Class balance should be preserved.",
                    "analysis_type": "Churn Prediction",
                    "hypothesis": "Core behavioral variables are predictive.",
                    "required_columns": ["target", "feature_a"],
                    "feature_families": [],
                    "techniques": ["gradient_boosting", "logistic_regression"],
                    "feasibility_analysis": {
                        "statistical_power": "adequate",
                        "signal_quality": "moderate",
                        "compute_value_tradeoff": "acceptable",
                    },
                    "recommended_artifacts": [
                        {"artifact_type": "metrics", "required": True, "rationale": "Evaluate the model."},
                        {"artifact_type": "predictions_or_scores", "required": True, "rationale": "Primary output."},
                    ],
                    "fallback_chain": ["gradient_boosting", "logistic_regression"],
                    "expected_lift": "3-5% over naive baseline",
                    "estimated_difficulty": "Medium",
                    "reasoning": "Balanced baseline strategy with robust fallback.",
                }
            ]
        }
        self.agent.model.generate_content = MagicMock(return_value=_mock_llm_response(payload))
        self.agent.generate_strategies(
            data_summary="summary",
            user_request="predict churn",
            column_inventory=["target", "feature_a"],
        )
        prompt = self.agent.last_prompt or ""
        _assert_contains_all(prompt, "*** MISSION ***", "*** SOURCE OF TRUTH AND PRECEDENCE ***")
        _assert_contains_terms(prompt, "strategy reasoning workflow", "scope_recommendation", "recommended_artifacts")

    @patch.dict("os.environ", {"STRATEGIST_COLUMN_REPAIR_ATTEMPTS": "1"})
    def test_generate_strategies_repairs_required_columns_with_inventory(self):
        initial = {
            "strategies": [
                {
                    "title": "risk",
                    "objective_type": "predictive",
                    "required_columns": ["ps_car_01", "target"],
                    "recommended_evaluation_metrics": ["auc"],
                }
            ]
        }
        repaired = {
            "strategies": [
                {
                    "title": "risk",
                    "objective_type": "predictive",
                    "required_columns": ["ps_car_01_cat", "target"],
                    "recommended_evaluation_metrics": ["auc"],
                }
            ]
        }
        self.agent.model.generate_content = MagicMock(
            side_effect=[_mock_llm_response(initial), _mock_llm_response(repaired)]
        )
        output = self.agent.generate_strategies(
            data_summary="summary",
            user_request="goal",
            column_inventory=["ps_car_01_cat", "target"],
            column_sets={"all_features": ["ps_*"]},
        )
        assert output["column_validation"]["status"] == "ok"
        assert output["column_validation"]["invalid_count"] == 0
        assert output["strategies"][0]["required_columns"] == ["ps_car_01_cat", "target"]

    def test_generate_prompt_excludes_feature_engineering_step(self):
        payload = {
            "strategies": [
                {
                    "title": "Baseline Churn",
                    "objective_type": "predictive",
                    "objective_reasoning": "Predict churn probability for retention actions.",
                    "success_metric": "roc_auc",
                    "recommended_evaluation_metrics": ["roc_auc"],
                    "validation_strategy": "stratified_cv",
                    "validation_rationale": "Class balance should be preserved.",
                    "analysis_type": "Churn Prediction",
                    "hypothesis": "Core behavioral variables are predictive.",
                    "required_columns": ["target", "feature_a"],
                    "feature_families": [],
                    "techniques": ["gradient_boosting", "logistic_regression"],
                    "feasibility_analysis": {
                        "statistical_power": "adequate",
                        "signal_quality": "moderate",
                        "compute_value_tradeoff": "acceptable",
                    },
                    "fallback_chain": ["gradient_boosting", "logistic_regression"],
                    "expected_lift": "3-5% over naive baseline",
                    "estimated_difficulty": "Medium",
                    "reasoning": "Balanced baseline strategy with robust fallback.",
                }
            ]
        }
        self.agent.model.generate_content = MagicMock(return_value=_mock_llm_response(payload))
        self.agent.generate_strategies(
            data_summary="summary",
            user_request="predict churn",
            column_inventory=["target", "feature_a"],
        )
        prompt = self.agent.last_prompt or ""
        assert "FEATURE ENGINEERING REASONING" not in prompt
        assert '"feature_engineering_strategy"' not in prompt

    @patch.dict("os.environ", {"STRATEGIST_STRATEGY_COUNT": "3"})
    def test_generate_prompt_aligns_strategy_goal_with_requested_count(self):
        payload = {
            "strategies": [
                {
                    "title": "Strategy A",
                    "objective_type": "predictive",
                    "objective_reasoning": "Reason A.",
                    "success_metric": "roc_auc",
                    "scope_recommendation": "ml_only",
                    "scope_reasoning": "Model-only run.",
                    "recommended_evaluation_metrics": ["roc_auc"],
                    "validation_strategy": "stratified_cv",
                    "validation_rationale": "Balanced classes.",
                    "analysis_type": "Churn Prediction",
                    "hypothesis": "A",
                    "required_columns": ["target", "feature_a"],
                    "feature_families": [],
                    "techniques": ["gradient_boosting"],
                    "feasibility_analysis": {
                        "statistical_power": "adequate",
                        "signal_quality": "moderate",
                        "compute_value_tradeoff": "acceptable",
                    },
                    "recommended_artifacts": [],
                    "fallback_chain": ["logistic_regression"],
                    "expected_lift": "small",
                    "estimated_difficulty": "Medium",
                    "reasoning": "Reason A",
                },
                {
                    "title": "Strategy B",
                    "objective_type": "predictive",
                    "objective_reasoning": "Reason B.",
                    "success_metric": "f1",
                    "scope_recommendation": "ml_only",
                    "scope_reasoning": "Model-only run.",
                    "recommended_evaluation_metrics": ["f1"],
                    "validation_strategy": "stratified_cv",
                    "validation_rationale": "Balanced classes.",
                    "analysis_type": "Churn Prediction",
                    "hypothesis": "B",
                    "required_columns": ["target", "feature_a"],
                    "feature_families": [],
                    "techniques": ["logistic_regression"],
                    "feasibility_analysis": {
                        "statistical_power": "adequate",
                        "signal_quality": "moderate",
                        "compute_value_tradeoff": "acceptable",
                    },
                    "recommended_artifacts": [],
                    "fallback_chain": ["naive_bayes"],
                    "expected_lift": "small",
                    "estimated_difficulty": "Low",
                    "reasoning": "Reason B",
                },
                {
                    "title": "Strategy C",
                    "objective_type": "predictive",
                    "objective_reasoning": "Reason C.",
                    "success_metric": "precision_at_k",
                    "scope_recommendation": "ml_only",
                    "scope_reasoning": "Model-only run.",
                    "recommended_evaluation_metrics": ["precision_at_k"],
                    "validation_strategy": "stratified_cv",
                    "validation_rationale": "Balanced classes.",
                    "analysis_type": "Churn Prediction",
                    "hypothesis": "C",
                    "required_columns": ["target", "feature_a"],
                    "feature_families": [],
                    "techniques": ["random_forest"],
                    "feasibility_analysis": {
                        "statistical_power": "adequate",
                        "signal_quality": "moderate",
                        "compute_value_tradeoff": "acceptable",
                    },
                    "recommended_artifacts": [],
                    "fallback_chain": ["extra_trees"],
                    "expected_lift": "small",
                    "estimated_difficulty": "Medium",
                    "reasoning": "Reason C",
                },
            ]
        }
        self.agent.model.generate_content = MagicMock(return_value=_mock_llm_response(payload))
        self.agent.generate_strategies(
            data_summary="summary",
            user_request="predict churn",
            column_inventory=["target", "feature_a"],
        )
        prompt = self.agent.last_prompt or ""
        _assert_contains_terms(prompt, "craft 3", "materially distinct", "executable strategies")
        assert "craft ONE optimal strategy" not in prompt

    def test_generate_strategies_restores_max_tokens_after_truncation_retry(self):
        original_max_tokens = self.agent._max_tokens
        payload = {
            "strategies": [
                {
                    "title": "Retry Strategy",
                    "objective_type": "predictive",
                    "objective_reasoning": "Retry test.",
                    "success_metric": "roc_auc",
                    "scope_recommendation": "ml_only",
                    "scope_reasoning": "Model-only run.",
                    "recommended_evaluation_metrics": ["roc_auc"],
                    "validation_strategy": "stratified_cv",
                    "validation_rationale": "Balanced classes.",
                    "analysis_type": "Churn Prediction",
                    "hypothesis": "Retry",
                    "required_columns": ["target", "feature_a"],
                    "feature_families": [],
                    "techniques": ["gradient_boosting"],
                    "feasibility_analysis": {
                        "statistical_power": "adequate",
                        "signal_quality": "moderate",
                        "compute_value_tradeoff": "acceptable",
                    },
                    "recommended_artifacts": [],
                    "fallback_chain": ["logistic_regression"],
                    "expected_lift": "small",
                    "estimated_difficulty": "Medium",
                    "reasoning": "Retry reasoning",
                }
            ]
        }

        observed_max_tokens = []

        def fake_call_model(_prompt, *, temperature, context_tag):
            observed_max_tokens.append(self.agent._max_tokens)
            self.agent._last_finish_reason = "length" if len(observed_max_tokens) == 1 else "stop"
            return json.dumps(payload)

        with patch.object(self.agent, "_call_model", side_effect=fake_call_model):
            output = self.agent.generate_strategies(
                data_summary="summary",
                user_request="predict churn",
                column_inventory=["target", "feature_a"],
            )

        assert output["strategies"][0]["title"] == "Retry Strategy"
        assert observed_max_tokens[0] == original_max_tokens
        assert observed_max_tokens[1] == min(original_max_tokens * 2, 32768)
        assert self.agent._max_tokens == original_max_tokens


class TestGraphStrategistIntegration:
    
    @patch("src.graph.graph.strategist")
    def test_run_strategist_handles_list_return_legacy(self, mock_strategist):
        """
        Simulate strategist.generate_strategies returning a LIST (legacy bug).
        run_strategist must catch it and form a valid state update.
        """
        # Simulate legacy behavior: returning a list directly
        mock_output = [{"title": "Legacy Strategy", "objective_type": "predictive"}]
        mock_strategist.generate_strategies.return_value = mock_output
        
        # Minimal state
        state = {
            "business_objective": "Test Goal",
            "data_summary": "Some data",
            "run_id": "test_run"
        }
        
        # Mock other dependencies if needed (none strictly needed for this logic branch)
        
        result = run_strategist(state)
        
        # Assertions
        assert "strategies" in result
        strategies_wrapper = result["strategies"]
        assert "strategies" in strategies_wrapper
        strategies_list = strategies_wrapper["strategies"]
        
        assert isinstance(strategies_list, list)
        assert len(strategies_list) == 1
        assert strategies_list[0]["title"] == "Legacy Strategy"
        # Strategy spec falls back to empty dict if not dict result
        assert result.get("strategy_spec") == {}

    @patch("src.graph.graph.strategist")
    def test_run_strategist_handles_dict_return(self, mock_strategist):
        """
        Simulate normal V2 behavior: returning a DICT.
        """
        mock_output = {
            "strategies": [{"title": "Modern Strategy"}],
            "strategy_spec": {"spec": "details"}
        }
        mock_strategist.generate_strategies.return_value = mock_output
        
        state = {"business_objective": "Test", "run_id": "test_run"}
        result = run_strategist(state)
        
        assert result["strategies"]["strategies"][0]["title"] == "Modern Strategy"
        assert result["strategy_spec"]["spec"] == "details"

    @patch("src.graph.graph.strategist")
    def test_run_strategist_passes_inventory_and_column_sets(self, mock_strategist):
        mock_strategist.generate_strategies.return_value = {"strategies": [{"title": "s"}]}
        state = {
            "business_objective": "Test",
            "run_id": "test_run",
            "column_inventory_columns": ["a", "b", "target"],
            "column_sets": {"pre_decision": ["a", "b"]},
        }
        run_strategist(state)
        _, kwargs = mock_strategist.generate_strategies.call_args
        assert kwargs["column_inventory"] == ["a", "b", "target"]
        assert kwargs["column_sets"] == {"pre_decision": ["a", "b"]}
