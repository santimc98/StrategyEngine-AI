"""ModelAnalyst agent — deep baseline code & metrics analysis for optimization blueprints."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from src.utils.action_families import ACTION_FAMILIES
from src.utils.llm_json_repair import parse_json_object_with_repair
from src.utils.llm_fallback import call_chat_with_fallback, extract_response_text

load_dotenv()


_ACTION_FAMILIES_STR = ", ".join(ACTION_FAMILIES)

_MAX_SCRIPT_CHARS = 12000

_LLM_PROMPT_TEMPLATE = """\
You are a senior ML optimization analyst. Your job is to analyze a baseline model's code
and metrics, reason about what would ACTUALLY improve the primary metric for THIS specific
situation, and produce a prioritized action plan.

=== BASELINE CODE ===
{script_code}

=== METRICS (from cross-validation) ===
{metrics_summary}

=== DATASET PROFILE ===
{dataset_profile_summary}

=== PRIMARY METRIC ===
{primary_metric}

=== MODELS USED ===
{models_used}

=== COMPUTE BUDGET ===
Hard timeout: 7200 seconds (2 hours). ALL proposed optimizations must complete within
this budget. Every technique you propose must be computationally feasible — estimate
the wall-clock time for your parameters before proposing them.

=== YOUR TASK ===
Think like a senior data scientist. Reason about what matters for THIS model,
THIS dataset, and THIS metric.

DIAGNOSIS
- What is the current metric value? How good is it already?
- Assess CV fold variance: is there enough instability to justify variance reduction,
  or is the model already stable enough that multi-seed gives diminishing returns?
- Is there a train-validation gap suggesting overfitting? If so, regularization may
  have more ROI than added complexity.
- What techniques are already applied? Don't suggest what's already there.

IMPROVEMENT PRIORITIES
- Given the dataset size, metric quality, and remaining compute budget, which improvements
  will give the best return per compute-second invested?
- Be realistic about expected gains at the current performance level.
- A technique that completes within budget with modest gains beats one that times out.

COMPUTE FEASIBILITY
- Estimate wall-clock time for your proposed parameters: consider dataset size, fold count,
  seed count, iterations, and framework speed characteristics.
- It is BETTER to succeed with conservative parameters than to timeout with ambitious ones.
  A timed-out technique gives zero improvement.

KNOWN TECHNIQUES (use as your toolbox, not as a checklist — only propose what makes sense):
- Multi-seed averaging: average predictions from N seeds. Reduces variance.
- Stacking: OOF predictions from diverse models → meta-learner. Captures complementarity.
- Target encoding: K-fold regularized encoding for high-cardinality categoricals.
- Pseudo-labeling: use high-confidence test predictions as training data.
- HPO (Optuna): Bayesian hyperparameter search. Diminishing returns at high baselines.
- Probability calibration: isotonic/Platt for probability-based metrics (log-loss, Brier).
- Feature interactions: pairwise products of top features.
- Learning rate reduction with more iterations.
- Diverse ensembling: combine different frameworks (CatBoost + LightGBM + XGBoost).

CONSTRAINTS:
- action_family MUST be one of: {action_families}
- priority 1 = highest expected impact, 5 = lowest
- concrete_params must contain REAL values the ML engineer can use directly
- expected_delta should be conservative and justified by your reasoning
- Include a "reasoning" field explaining WHY this technique helps for this specific case

Respond ONLY with valid JSON:
{{
  "model_type": "<gradient_boosting|neural_network|linear|tree|svm|other>",
  "framework": "<catboost|lightgbm|xgboost|sklearn|pytorch|tensorflow|keras|other>",
  "baseline_assessment": "<2-3 sentence diagnosis: metric quality, variance level, overfitting risk, key gaps>",
  "improvement_actions": [
    {{
      "technique": "<technique_name>",
      "action_family": "<one of the allowed families>",
      "concrete_params": {{}},
      "code_change_hint": "<what to change in the code>",
      "reasoning": "<why this technique helps for THIS specific case>",
      "estimated_compute_seconds": 0,
      "expected_delta": 0.001,
      "priority": 1
    }}
  ],
  "total_expected_delta": 0.003
}}
"""


class ModelAnalystAgent:
    """Analyzes baseline model code and metrics to produce an optimization blueprint."""

    def __init__(self, api_key: Any = None):
        self.mode = self._normalize_mode(
            os.getenv("MODEL_ANALYST_MODE", "hybrid")
        )
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.client: Any = None
        self.model_name = (
            os.getenv("OPENROUTER_MODEL_ANALYST_PRIMARY_MODEL")
            or "z-ai/glm-5"
        ).strip()
        self.fallback_model_name = (
            os.getenv("OPENROUTER_MODEL_ANALYST_FALLBACK_MODEL")
            or "z-ai/glm-4.7"
        ).strip()
        self.temperature = float(
            os.getenv("MODEL_ANALYST_TEMPERATURE", "0.1")
        )

        if api_key is not None and str(api_key).strip() == "":
            return

        if self.mode in {"llm", "hybrid"} and self.api_key:
            timeout_raw = os.getenv("OPENROUTER_TIMEOUT_SECONDS")
            try:
                timeout_seconds = float(timeout_raw) if timeout_raw else 120.0
            except ValueError:
                timeout_seconds = 120.0
            headers: Dict[str, str] = {}
            referer = os.getenv("OPENROUTER_HTTP_REFERER")
            if referer:
                headers["HTTP-Referer"] = referer
            title = os.getenv("OPENROUTER_X_TITLE")
            if title:
                headers["X-Title"] = title
            client_kwargs: Dict[str, Any] = {
                "api_key": self.api_key,
                "base_url": "https://openrouter.ai/api/v1",
                "timeout": timeout_seconds,
            }
            if headers:
                client_kwargs["default_headers"] = headers
            self.client = OpenAI(**client_kwargs)
            print(
                f"MODEL_ANALYST_OPENROUTER: primary={self.model_name} "
                f"fallback={self.fallback_model_name}"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_baseline(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Produce an optimization_blueprint from baseline code + metrics.

        Philosophy:
        - LLM/hybrid mode: trust the LLM's reasoning about what matters for THIS
          specific model/dataset/metric. Only apply compute-budget guardrails.
        - Deterministic mode: fallback when no LLM available — uses heuristic rules.
        """
        context = context if isinstance(context, dict) else {}

        if self.mode == "deterministic":
            deterministic = self._analyze_baseline_deterministic(context)
            return self._finalize_blueprint(deterministic, context)

        # LLM/hybrid: let the LLM reason freely, then validate feasibility
        llm_result = self._analyze_baseline_llm(context)
        if llm_result and self._validate_blueprint(llm_result):
            return self._finalize_blueprint(llm_result, context)

        # LLM failed — fall back to deterministic
        print("MODEL_ANALYST: LLM analysis failed, falling back to deterministic")
        deterministic = self._analyze_baseline_deterministic(context)
        return self._finalize_blueprint(deterministic, context)

    # ------------------------------------------------------------------
    # LLM analysis
    # ------------------------------------------------------------------

    def _analyze_baseline_llm(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if self.client is None:
            return None

        script_code = str(context.get("script_code") or "")
        if len(script_code) > _MAX_SCRIPT_CHARS:
            script_code = script_code[:_MAX_SCRIPT_CHARS] + "\n... [truncated]"

        metrics = context.get("metrics") if isinstance(context.get("metrics"), dict) else {}
        dataset_profile = context.get("dataset_profile") if isinstance(context.get("dataset_profile"), dict) else {}
        primary_metric = str(context.get("primary_metric") or "unknown")
        models_used = context.get("models_used") if isinstance(context.get("models_used"), list) else []

        # Enrich metrics with diagnostic signals the LLM needs for reasoning
        enriched_metrics = dict(metrics)
        # Compute per-fold time estimate so LLM can reason about budget
        _n_rows = 0
        for _rk in ("n_train_rows", "n_rows", "n_samples"):
            _rv = dataset_profile.get(_rk) or metrics.get(_rk)
            if _rv is not None:
                try:
                    _n_rows = int(_rv)
                except (TypeError, ValueError):
                    pass
                if _n_rows > 0:
                    break
        if _n_rows > 0:
            _code_iters = 0
            import re as _re
            for m in _re.finditer(r'(?:iterations|n_estimators)\s*[=:]\s*(\d+)', script_code):
                try:
                    _code_iters = max(_code_iters, int(m.group(1)))
                except (ValueError, TypeError):
                    pass
            _fw = self._detect_framework(script_code.lower())
            _pfs = self._estimate_per_fold_seconds(_fw, _n_rows, max_iterations=_code_iters)
            _cv = 3 if _n_rows > 100_000 else 5
            enriched_metrics["_estimated_per_fold_seconds"] = round(_pfs, 1)
            enriched_metrics["_estimated_single_retrain_seconds"] = round(_pfs * _cv, 0)
            enriched_metrics["_remaining_budget_seconds"] = 7200

        metrics_summary = json.dumps(enriched_metrics, indent=2, default=str)[:4000]
        dataset_profile_summary = self._summarize_dataset_profile(dataset_profile)

        prompt = _LLM_PROMPT_TEMPLATE.format(
            script_code=script_code,
            metrics_summary=metrics_summary,
            dataset_profile_summary=dataset_profile_summary,
            primary_metric=primary_metric,
            models_used=", ".join(str(m) for m in models_used) if models_used else "unknown",
            action_families=_ACTION_FAMILIES_STR,
        )

        try:
            content = self._call_llm(prompt)
            parsed, _ = parse_json_object_with_repair(content)
            if isinstance(parsed, dict):
                return parsed
        except Exception as exc:
            print(f"MODEL_ANALYST_LLM_ERROR: {exc}")
        return None

    def _call_llm(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You are an expert ML model analyst. Return only valid JSON."},
            {"role": "user", "content": prompt},
        ]
        model_chain = [self.model_name, self.fallback_model_name]
        response, used_model = call_chat_with_fallback(
            llm_client=self.client,
            messages=messages,
            model_chain=model_chain,
            call_kwargs={"temperature": self.temperature},
            logger=None,
            context_tag="model_analyst",
        )
        print(f"MODEL_ANALYST_LLM: used_model={used_model}")
        return extract_response_text(response)

    # ------------------------------------------------------------------
    # Deterministic analysis (regex heuristics)
    # ------------------------------------------------------------------

    def _analyze_baseline_deterministic(self, context: Dict[str, Any]) -> Dict[str, Any]:
        script = str(context.get("script_code") or "")
        metrics = context.get("metrics") if isinstance(context.get("metrics"), dict) else {}
        dataset_profile = context.get("dataset_profile") if isinstance(context.get("dataset_profile"), dict) else {}
        code_lower = script.lower()

        framework = self._detect_framework(code_lower)
        model_type = self._detect_model_type(framework, code_lower)
        actions: List[Dict[str, Any]] = []

        # ── Compute budget estimation for viability checks ──
        # Used to skip ensemble/stacking proposals that are physically
        # impossible within the framework timeout.
        _n_rows_est = 0
        for _rk in ("n_train_rows", "n_rows", "n_samples", "n_train"):
            _rv = dataset_profile.get(_rk) or metrics.get(_rk)
            if _rv is not None:
                try:
                    _n_rows_est = int(_rv)
                except (TypeError, ValueError):
                    pass
                if _n_rows_est > 0:
                    break
        # Extract complexity hints from baseline script for better estimation
        _code_max_iters = 0
        _code_max_depth = 0
        import re as _re_local
        for _iters_match in _re_local.finditer(r'(?:iterations|n_estimators)\s*[=:]\s*(\d+)', script):
            try:
                _code_max_iters = max(_code_max_iters, int(_iters_match.group(1)))
            except (ValueError, TypeError):
                pass
        for _depth_match in _re_local.finditer(r'(?:depth|max_depth)\s*[=:]\s*(\d+)', script):
            try:
                _code_max_depth = max(_code_max_depth, int(_depth_match.group(1)))
            except (ValueError, TypeError):
                pass
        _per_fold_sec = self._estimate_per_fold_seconds(
            framework, _n_rows_est,
            max_iterations=_code_max_iters, max_depth=_code_max_depth,
        ) if _n_rows_est > 0 else 0.0
        # Framework timeout ceiling (conservative: use 7200s as reference)
        _framework_budget = 7200.0
        _cv_folds_default = 3 if _n_rows_est > 100_000 else 5

        # 1. Early stopping
        has_early_stopping = any(
            kw in code_lower
            for kw in ["early_stopping_rounds", "early_stopping", "earlystopping", "patience"]
        )
        if not has_early_stopping and script.strip():
            params = self._early_stopping_params(framework)
            actions.append({
                "technique": "early_stopping",
                "action_family": "hyperparameter_search",
                "concrete_params": params,
                "code_change_hint": "Add early stopping to prevent overfitting and find optimal iteration count",
                "expected_delta": 0.0005,
                "priority": 1,
            })

        # 2. HPO (Optuna / grid search)
        has_hpo = any(
            kw in code_lower
            for kw in ["optuna", "hyperopt", "gridsearchcv", "randomizedsearchcv", "bayesian"]
        )
        if not has_hpo and script.strip():
            params = self._hpo_params(framework, metrics, dataset_profile)
            actions.append({
                "technique": "optuna_hpo",
                "action_family": "hyperparameter_search",
                "concrete_params": params,
                "code_change_hint": f"Add Optuna HPO with n_trials={params['n_trials']}, timeout={params['timeout_seconds']}s, cv_folds={params.get('cv_folds', 5)} inside objective. Use early stopping inside each trial to stay within time budget",
                "expected_delta": 0.002,
                "priority": 1,
            })

        # 3. Ensemble weight optimization
        model_count = self._count_models_in_code(script)
        has_weight_opt = any(
            kw in code_lower
            for kw in ["optimize_weight", "scipy.optimize", "weight_search", "optim"]
        )
        if model_count >= 2 and not has_weight_opt:
            actions.append({
                "technique": "weighted_ensemble",
                "action_family": "ensemble_or_stacking",
                "concrete_params": {
                    "method": "scipy_minimize",
                    "objective": "minimize_logloss_or_maximize_auc",
                    "bounds": [0.0, 1.0],
                },
                "code_change_hint": "Optimize ensemble weights using scipy.optimize instead of equal averaging",
                "expected_delta": 0.0003,
                "priority": 2,
            })

        # 4. Feature interactions
        has_interactions = any(
            kw in code_lower
            for kw in ["polynomialfeatures", "interaction", "feature_cross", "pairwise"]
        )
        n_features = 0
        if isinstance(dataset_profile.get("n_columns"), int):
            n_features = dataset_profile["n_columns"]
        elif isinstance(metrics.get("n_features"), int):
            n_features = metrics["n_features"]
        if not has_interactions and n_features > 2 and n_features <= 50:
            actions.append({
                "technique": "interaction_features",
                "action_family": "feature_engineering",
                "concrete_params": {
                    "method": "top_k_pairwise_products",
                    "k": min(10, n_features * (n_features - 1) // 2),
                    "selection": "mutual_information",
                },
                "code_change_hint": "Add top-k pairwise feature interactions selected by mutual information",
                "expected_delta": 0.0005,
                "priority": 3,
            })

        # 5. Multi-seed averaging — scale seeds & iterations to dataset size
        seed_matches = re.findall(r"(?:random_state|seed)\s*[=:]\s*(\d+)", script)
        unique_seeds = set(seed_matches)
        if len(unique_seeds) <= 1 and script.strip():
            # Scale seeds inversely with dataset size:
            #   <50K rows  → 5 seeds (high variance benefit)
            #   50-200K    → 3 seeds
            #   200K-500K  → 3 seeds with reduced iterations
            #   >500K      → 2 seeds with aggressive early stopping
            _all_seeds = [42, 123, 456, 789, 2024]
            if _n_rows_est > 500_000:
                _ms_seeds = _all_seeds[:2]
                _ms_iters = max(300, _code_max_iters // 3) if _code_max_iters > 0 else 400
                _ms_early_stop = 30
                _ms_hint = (
                    f"Multi-seed averaging with {len(_ms_seeds)} seeds (large dataset: {_n_rows_est:,} rows). "
                    f"Use {_ms_iters} iterations with early_stopping={_ms_early_stop} to fit within timeout. "
                    "Train full pipeline with each seed and average final predictions."
                )
            elif _n_rows_est > 200_000:
                _ms_seeds = _all_seeds[:3]
                _ms_iters = max(400, _code_max_iters // 2) if _code_max_iters > 0 else 500
                _ms_early_stop = 50
                _ms_hint = (
                    f"Multi-seed averaging with {len(_ms_seeds)} seeds (medium-large dataset). "
                    f"Use {_ms_iters} iterations with early_stopping={_ms_early_stop}."
                )
            elif _n_rows_est > 50_000:
                _ms_seeds = _all_seeds[:3]
                _ms_iters = _code_max_iters if _code_max_iters > 0 else 1000
                _ms_early_stop = 50
                _ms_hint = f"Multi-seed averaging with {len(_ms_seeds)} seeds."
            else:
                _ms_seeds = _all_seeds[:5]
                _ms_iters = _code_max_iters if _code_max_iters > 0 else 1000
                _ms_early_stop = 100
                _ms_hint = f"Multi-seed averaging with {len(_ms_seeds)} seeds."

            _ms_params: Dict[str, Any] = {
                "seeds": _ms_seeds,
                "aggregation": "mean",
            }
            # Only override iterations/early_stopping for large datasets
            if _n_rows_est > 200_000:
                _ms_params["max_iterations"] = _ms_iters
                _ms_params["early_stopping_rounds"] = _ms_early_stop
            actions.append({
                "technique": "multi_seed_averaging",
                "action_family": "ensemble_or_stacking",
                "concrete_params": _ms_params,
                "code_change_hint": _ms_hint,
                "expected_delta": 0.0002,
                "priority": 4,
            })

        # 6. Learning rate reduction with more iterations
        lr_match = re.search(r"learning_rate['\"]?\s*[:=]\s*([\d.]+)", script)
        if lr_match:
            current_lr = float(lr_match.group(1))
            if current_lr >= 0.03:
                actions.append({
                    "technique": "lr_reduction",
                    "action_family": "hyperparameter_search",
                    "concrete_params": {
                        "current_lr": current_lr,
                        "suggested_lr": round(current_lr / 3, 4),
                        "increase_iterations_factor": 3,
                    },
                    "code_change_hint": f"Reduce learning rate from {current_lr} to {round(current_lr / 3, 4)} and increase iterations 3x",
                    "expected_delta": 0.0005,
                    "priority": 2,
                })

        # 7. Calibration
        has_calibration = any(
            kw in code_lower
            for kw in ["calibratedclassifiercv", "calibration", "isotonic", "platt"]
        )
        primary_metric = str(context.get("primary_metric") or "").lower()
        if not has_calibration and primary_metric in {"logloss", "log_loss", "brier", "brier_score"}:
            actions.append({
                "technique": "probability_calibration",
                "action_family": "calibration",
                "concrete_params": {
                    "method": "isotonic",
                    "cv": 5,
                },
                "code_change_hint": "Add CalibratedClassifierCV with isotonic regression for better probability estimates",
                "expected_delta": 0.001,
                "priority": 3,
            })

        # 8. Stacking (if only simple averaging) — scale CV/subsample to dataset size
        has_stacking = any(
            kw in code_lower
            for kw in ["stackingclassifier", "stackingregressor", "stacking", "meta_model", "meta_learner"]
        )
        if model_count >= 2 and not has_stacking:
            _stack_cv = 3 if _n_rows_est > 100_000 else 5
            _stack_params: Dict[str, Any] = {
                "meta_learner": "LogisticRegression" if "classif" in code_lower or "auc" in primary_metric else "Ridge",
                "cv": _stack_cv,
                "passthrough": False,
            }
            _stack_hint = "Replace simple averaging with stacking using a meta-learner on out-of-fold predictions"
            if _n_rows_est > 300_000:
                _stack_params["subsample_for_oof"] = 0.5
                _stack_params["base_learner_iterations"] = max(300, _code_max_iters // 3) if _code_max_iters > 0 else 400
                _stack_params["early_stopping_rounds"] = 30
                _stack_hint += (
                    f". Large dataset ({_n_rows_est:,} rows): use {_stack_cv}-fold OOF with 50% subsample, "
                    f"reduced iterations, and early stopping to fit within timeout"
                )
            actions.append({
                "technique": "stacking_ensemble",
                "action_family": "ensemble_or_stacking",
                "concrete_params": _stack_params,
                "code_change_hint": _stack_hint,
                "expected_delta": 0.0005,
                "priority": 4,
            })

        # 9. Target encoding for categorical features
        has_target_encoding = any(
            kw in code_lower
            for kw in ["targetencoder", "target_encoding", "target_encode", "leave_one_out", "loo_encoding", "woeencoder"]
        )
        has_categoricals = any(
            kw in code_lower
            for kw in ["ordinalencoder", "labelencoder", "onehotencoder", "category_encoders", "cat_features", "categorical_feature"]
        )
        if has_categoricals and not has_target_encoding and script.strip():
            actions.append({
                "technique": "target_encoding",
                "action_family": "feature_engineering",
                "concrete_params": {
                    "method": "kfold_target_encoding",
                    "cv": 5,
                    "smoothing": 10.0,
                    "handle_unknown": "global_mean",
                },
                "code_change_hint": "Add K-fold target encoding for categorical features (fit on train folds, transform on held-out fold to prevent leakage)",
                "expected_delta": 0.0005,
                "priority": 3,
            })

        # 10. Pseudo-labeling (semi-supervised learning)
        has_pseudo_label = any(
            kw in code_lower
            for kw in ["pseudo_label", "pseudolabel", "semi_supervised", "self_training", "selftraining"]
        )
        _n_test_est = 0
        for _tk in ("n_test_rows", "n_test", "n_test_samples"):
            _tv = dataset_profile.get(_tk) or metrics.get(_tk)
            if _tv is not None:
                try:
                    _n_test_est = int(_tv)
                except (TypeError, ValueError):
                    pass
                if _n_test_est > 0:
                    break
        # Only suggest pseudo-labeling if test set is substantial (>10% of total)
        _total_est = _n_rows_est + _n_test_est
        if (
            not has_pseudo_label
            and _n_test_est > 0
            and _total_est > 0
            and (_n_test_est / _total_est) > 0.10
            and script.strip()
        ):
            actions.append({
                "technique": "pseudo_labeling",
                "action_family": "feature_engineering",
                "concrete_params": {
                    "confidence_threshold_high": 0.95,
                    "confidence_threshold_low": 0.05,
                    "max_pseudo_ratio": 0.3,
                    "retrain_epochs": 1,
                },
                "code_change_hint": (
                    "After initial training, predict on test set. Add high-confidence predictions "
                    "(prob > 0.95 or < 0.05) as pseudo-labeled training data and retrain the model"
                ),
                "expected_delta": 0.0003,
                "priority": 5,
            })

        assessment = f"{model_type} model using {framework}"
        if not has_hpo:
            assessment += ", no HPO"
        if not has_early_stopping:
            assessment += ", no early stopping"
        if model_count >= 2 and not has_weight_opt:
            assessment += ", equal-weight ensemble"

        # ── Adaptive viability filter ──
        # Instead of silently discarding expensive techniques, adapt them to fit
        # within the compute budget.  This ensures multi-seed, stacking, etc.
        # are always proposed — just with resource-aware parameters.
        if _per_fold_sec > 0 and _n_rows_est > 0:
            viable_actions: List[Dict[str, Any]] = []
            for action in actions:
                tech = action.get("technique", "")
                family = action.get("action_family", "")
                params = dict(action.get("concrete_params") or {})

                # Estimate total compute seconds for this action
                est_total: float = 0.0
                if family == "ensemble_or_stacking":
                    if tech == "multi_seed_averaging":
                        n_seeds = len(params.get("seeds", [42, 123, 456]))
                        est_total = _per_fold_sec * _cv_folds_default * n_seeds
                    elif tech == "stacking_ensemble":
                        n_base = max(model_count, 2)
                        est_total = _per_fold_sec * _cv_folds_default * (n_base + 1)
                    elif tech == "weighted_ensemble":
                        est_total = _per_fold_sec * _cv_folds_default * 3
                    elif tech == "diverse_ensemble_averaging":
                        n_models = len(params.get("models", []))
                        if n_models < 2:
                            n_models = 3
                        est_total = _per_fold_sec * _cv_folds_default * n_models
                    else:
                        est_total = _per_fold_sec * _cv_folds_default * 3
                elif family == "hyperparameter_search" and tech in ("optuna_hpo", "optuna_hyperparameter_optimization"):
                    hpo_trials = params.get("n_trials", 50)
                    hpo_folds = params.get("cv_folds", _cv_folds_default)
                    est_total = _per_fold_sec * hpo_folds * hpo_trials
                else:
                    est_total = _per_fold_sec * _cv_folds_default

                budget_limit = _framework_budget * 0.85
                if est_total <= budget_limit:
                    viable_actions.append(action)
                else:
                    # ── Adapt the technique to fit within budget instead of discarding ──
                    adapted = dict(action)
                    adapted["concrete_params"] = dict(params)
                    adapted_reason = ""

                    if tech == "multi_seed_averaging":
                        # Reduce seeds AND iterations to fit budget
                        all_seeds = params.get("seeds", [42, 123, 456, 789, 2024])
                        # First try reducing seeds
                        max_seeds = max(2, int(budget_limit / (_per_fold_sec * _cv_folds_default)))
                        max_seeds = min(max_seeds, len(all_seeds))
                        adapted["concrete_params"]["seeds"] = all_seeds[:max_seeds]
                        # If still too expensive, also reduce iterations
                        new_est = _per_fold_sec * _cv_folds_default * max_seeds
                        if new_est > budget_limit and _code_max_iters > 300:
                            reduction = budget_limit / new_est
                            new_iters = max(300, int(_code_max_iters * reduction))
                            adapted["concrete_params"]["max_iterations"] = new_iters
                            adapted["concrete_params"]["early_stopping_rounds"] = 30
                            adapted_reason = f"seeds {len(all_seeds)}->{max_seeds}, iters->{new_iters}, early_stop=30"
                        else:
                            adapted_reason = f"seeds {len(all_seeds)}->{max_seeds}"
                        adapted["code_change_hint"] = (
                            f"Multi-seed averaging with {max_seeds} seeds (budget-adapted). "
                            "Train the full pipeline with each seed and average final predictions."
                        )

                    elif tech == "stacking_ensemble":
                        # Reduce CV folds for stacking and/or use subsample for base learners
                        stacking_cv = 3
                        n_base = max(model_count, 2)
                        new_est = _per_fold_sec * stacking_cv * (n_base + 1)
                        if new_est > budget_limit and _n_rows_est > 100_000:
                            # Use subsampling for OOF generation
                            subsample_frac = min(0.5, budget_limit / new_est)
                            adapted["concrete_params"]["subsample_for_oof"] = round(subsample_frac, 2)
                            adapted_reason = f"cv=3, subsample={subsample_frac:.0%}"
                        else:
                            adapted_reason = "cv=3"
                        adapted["concrete_params"]["cv"] = stacking_cv
                        adapted["code_change_hint"] = (
                            f"Stacking with {stacking_cv}-fold OOF (budget-adapted). "
                            "Generate out-of-fold predictions from diverse base learners, "
                            "then train a meta-learner (LogisticRegression/Ridge) on them."
                        )

                    elif tech == "weighted_ensemble":
                        adapted["concrete_params"]["use_precomputed_oof"] = True
                        adapted["code_change_hint"] = (
                            "Optimize ensemble weights using precomputed OOF predictions "
                            "(no retraining needed — just scipy.optimize on existing predictions)"
                        )
                        adapted_reason = "use precomputed OOF"

                    elif tech in ("optuna_hpo", "optuna_hyperparameter_optimization"):
                        # Reduce trials to fit budget
                        max_trials = max(5, int(budget_limit / (_per_fold_sec * _cv_folds_default)))
                        adapted["concrete_params"]["n_trials"] = min(max_trials, params.get("n_trials", 50))
                        adapted["concrete_params"]["timeout_seconds"] = int(budget_limit * 0.8)
                        adapted_reason = f"trials->{adapted['concrete_params']['n_trials']}"

                    else:
                        # Generic fallback: still include but flag as budget-constrained
                        adapted["code_change_hint"] = (
                            str(action.get("code_change_hint", "")) +
                            f" [BUDGET NOTE: estimated {est_total:.0f}s vs budget {budget_limit:.0f}s — "
                            "use subsampling or fewer folds to fit within budget]"
                        )
                        adapted_reason = "budget-constrained"

                    print(
                        f"BLUEPRINT_VIABILITY: Adapted '{tech}' to fit budget "
                        f"({est_total:.0f}s -> budget {budget_limit:.0f}s): {adapted_reason}"
                    )
                    viable_actions.append(adapted)
            actions = viable_actions

        return {
            "model_type": model_type,
            "framework": framework,
            "baseline_assessment": assessment,
            "improvement_actions": actions,
            "total_expected_delta": round(sum(a.get("expected_delta", 0) for a in actions), 4),
        }

    # ------------------------------------------------------------------
    # Validation & merging
    # ------------------------------------------------------------------

    def _validate_blueprint(self, blueprint: Dict[str, Any]) -> bool:
        if not isinstance(blueprint, dict):
            return False
        actions = blueprint.get("improvement_actions")
        if not isinstance(actions, list) or not actions:
            return False
        if not blueprint.get("model_type"):
            return False
        for action in actions:
            if not isinstance(action, dict):
                return False
            if not action.get("technique"):
                return False
            family = str(action.get("action_family") or "")
            if family and family not in ACTION_FAMILIES:
                action["action_family"] = "hyperparameter_search"
            try:
                p = int(action.get("priority", 3))
                action["priority"] = max(1, min(5, p))
            except (ValueError, TypeError):
                action["priority"] = 3
        return True

    def _finalize_blueprint(self, blueprint: Dict[str, Any], context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Apply compute-budget guardrails and finalize the blueprint.

        This is the safety layer: it ADAPTS (not drops) actions that exceed the
        compute budget, ensuring every proposed technique has a chance to execute.
        """
        blueprint["blueprint_version"] = "1.0"
        blueprint["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        actions = blueprint.get("improvement_actions")
        if not isinstance(actions, list):
            return blueprint

        ctx = context if isinstance(context, dict) else {}
        dp = ctx.get("dataset_profile") if isinstance(ctx.get("dataset_profile"), dict) else {}
        mx = ctx.get("metrics") if isinstance(ctx.get("metrics"), dict) else {}
        _n_rows = 0
        for _rk in ("n_train_rows", "n_rows", "n_samples", "n_train"):
            _rv = dp.get(_rk) or mx.get(_rk)
            if _rv is not None:
                try:
                    _n_rows = int(_rv)
                except (TypeError, ValueError):
                    pass
                if _n_rows > 0:
                    break

        if _n_rows <= 0:
            blueprint["improvement_actions"] = actions[:6]
            return blueprint

        fw = str(blueprint.get("framework") or "catboost").lower()
        # Extract baseline iterations/depth from code for estimation
        import re as _re
        _code = str(ctx.get("script_code") or "")
        _code_iters = 0
        _code_depth = 0
        for m in _re.finditer(r'(?:iterations|n_estimators)\s*[=:]\s*(\d+)', _code):
            try:
                _code_iters = max(_code_iters, int(m.group(1)))
            except (ValueError, TypeError):
                pass
        for m in _re.finditer(r'(?:depth|max_depth)\s*[=:]\s*(\d+)', _code):
            try:
                _code_depth = max(_code_depth, int(m.group(1)))
            except (ValueError, TypeError):
                pass

        _pfs = self._estimate_per_fold_seconds(
            fw, _n_rows,
            max_iterations=_code_iters, max_depth=_code_depth,
        )
        _budget = 7200.0
        _budget_limit = _budget * 0.85
        _cv = 3 if _n_rows > 100_000 else 5

        guarded: List[Dict[str, Any]] = []
        for act in actions:
            if not isinstance(act, dict):
                continue
            family = str(act.get("action_family", ""))
            tech = str(act.get("technique", "")).lower()
            params = dict(act.get("concrete_params") or {})

            # Estimate compute time
            est = 0.0
            if family == "ensemble_or_stacking":
                if "seed" in tech:
                    n_seeds = len(params.get("seeds", [1, 2, 3]))
                    est = _pfs * _cv * n_seeds
                elif "stacking" in tech:
                    n_base = len(params.get("models", [])) or 3
                    stacking_cv = params.get("cv", _cv)
                    est = _pfs * stacking_cv * (n_base + 1)
                else:
                    n_m = len(params.get("models", [])) or 3
                    est = _pfs * _cv * n_m
            elif "hpo" in tech or "hyperparameter" in tech:
                hpo_t = params.get("n_trials", 50)
                hpo_cv = params.get("cv_folds", _cv)
                est = _pfs * hpo_cv * hpo_t
            else:
                est = _pfs * _cv

            if est <= _budget_limit:
                guarded.append(act)
            else:
                # ADAPT instead of drop — reduce parameters to fit budget
                adapted = dict(act)
                adapted["concrete_params"] = dict(params)
                reason = ""

                if "seed" in tech:
                    max_seeds = max(2, int(_budget_limit / (_pfs * _cv)))
                    all_seeds = params.get("seeds", [42, 123, 456])
                    adapted["concrete_params"]["seeds"] = all_seeds[:max_seeds]
                    new_est = _pfs * _cv * max_seeds
                    if new_est > _budget_limit and _code_iters > 300:
                        reduction = _budget_limit / new_est
                        new_iters = max(300, int(_code_iters * reduction))
                        adapted["concrete_params"]["max_iterations"] = new_iters
                        adapted["concrete_params"]["early_stopping_rounds"] = 30
                        reason = f"seeds->{max_seeds}, iters->{new_iters}"
                    else:
                        reason = f"seeds->{max_seeds}"

                elif "stacking" in tech:
                    adapted["concrete_params"]["cv"] = 3
                    if _n_rows > 200_000:
                        adapted["concrete_params"]["subsample_for_oof"] = 0.5
                        reason = "cv=3, subsample=50%"
                    else:
                        reason = "cv=3"

                elif "hpo" in tech or "hyperparameter" in tech:
                    max_trials = max(5, int(_budget_limit / (_pfs * _cv)))
                    adapted["concrete_params"]["n_trials"] = min(max_trials, params.get("n_trials", 50))
                    adapted["concrete_params"]["timeout_seconds"] = int(_budget_limit * 0.8)
                    reason = f"trials->{adapted['concrete_params']['n_trials']}"

                else:
                    reason = "budget-constrained"

                print(
                    f"BLUEPRINT_GUARDRAIL: Adapted '{tech}' "
                    f"({est:.0f}s -> budget {_budget_limit:.0f}s): {reason}"
                )
                guarded.append(adapted)

        blueprint["improvement_actions"] = guarded[:6]
        return blueprint

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------

    def _detect_framework(self, code_lower: str) -> str:
        if "catboost" in code_lower:
            return "catboost"
        if "lightgbm" in code_lower or "lgb" in code_lower:
            return "lightgbm"
        if "xgboost" in code_lower or "xgb" in code_lower:
            return "xgboost"
        if "torch" in code_lower or "pytorch" in code_lower:
            return "pytorch"
        if "tensorflow" in code_lower or "tf." in code_lower:
            return "tensorflow"
        if "keras" in code_lower:
            return "keras"
        if "sklearn" in code_lower or "scikit" in code_lower:
            return "sklearn"
        return "other"

    def _detect_model_type(self, framework: str, code_lower: str) -> str:
        if framework in {"catboost", "lightgbm", "xgboost"}:
            return "gradient_boosting"
        if framework in {"pytorch", "tensorflow", "keras"}:
            return "neural_network"
        if "randomforest" in code_lower or "random_forest" in code_lower:
            return "tree"
        if "logisticregression" in code_lower or "logistic" in code_lower:
            return "linear"
        if "svm" in code_lower or "svc" in code_lower or "svr" in code_lower:
            return "svm"
        return "other"

    def _count_models_in_code(self, script: str) -> int:
        model_patterns = [
            r"CatBoostClassifier|CatBoostRegressor|CatBoost\(",
            r"LGBMClassifier|LGBMRegressor|lgb\.train",
            r"XGBClassifier|XGBRegressor|xgb\.train",
            r"RandomForestClassifier|RandomForestRegressor",
            r"GradientBoostingClassifier|GradientBoostingRegressor",
            r"LogisticRegression",
            r"nn\.Module|torch\.nn",
            r"Sequential\(",
            r"ExtraTreesClassifier|ExtraTreesRegressor",
        ]
        count = 0
        for pattern in model_patterns:
            if re.search(pattern, script):
                count += 1
        return count

    def _early_stopping_params(self, framework: str) -> Dict[str, Any]:
        if framework == "catboost":
            return {"early_stopping_rounds": 50, "eval_metric": "auto"}
        if framework == "lightgbm":
            return {"early_stopping_rounds": 50, "verbose": -1}
        if framework == "xgboost":
            return {"early_stopping_rounds": 50, "eval_metric": "auto"}
        if framework in {"pytorch", "tensorflow", "keras"}:
            return {"patience": 10, "min_delta": 0.0001, "restore_best_weights": True}
        return {"early_stopping_rounds": 50}

    @staticmethod
    def _estimate_per_fold_seconds(
        framework: str,
        n_rows: int,
        *,
        max_iterations: int = 0,
        max_depth: int = 0,
    ) -> float:
        """Estimate wall-clock seconds for ONE model training on ONE fold.

        Conservative heuristic calibrated on real run data:
          - CatBoost 594K rows, depth=8, iters=1000 → ~800s/fold (observed)
          - LightGBM is ~3x faster than CatBoost
          - XGBoost is ~2x faster than CatBoost

        The base rate assumes moderate params (depth≤6, iterations≤500).
        When ``max_iterations`` or ``max_depth`` are provided, a complexity
        multiplier is applied to avoid underestimation on expensive configs.

        Returns a per-fold estimate in seconds.
        """
        # Base rate: seconds per 10K rows (moderate params: depth≤6, iters≤500)
        _rate: Dict[str, float] = {
            "catboost": 2.5,
            "lightgbm": 0.8,
            "xgboost": 1.4,
        }
        rate = _rate.get(framework.lower(), 2.5)
        base = max(5.0, (n_rows / 10_000) * rate)

        # Complexity multiplier for iterations above moderate baseline
        iter_mult = 1.0
        if max_iterations > 500:
            # Linear scaling: 1000 iters → 2x, 2000 iters → 4x
            iter_mult = max_iterations / 500.0

        # Complexity multiplier for tree depth above moderate baseline
        depth_mult = 1.0
        if max_depth > 6:
            # Exponential-ish: depth 8 → ~2.5x, depth 10 → ~4x
            depth_mult = 2.0 ** ((max_depth - 6) / 2.0)

        return base * iter_mult * depth_mult

    def _hpo_params(self, framework: str, metrics: Dict[str, Any], dataset_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        n_rows = 0
        if isinstance(dataset_profile, dict):
            for key in ("n_train_rows", "n_rows", "n_samples"):
                val = dataset_profile.get(key)
                if val is not None:
                    try:
                        n_rows = int(val)
                    except (ValueError, TypeError):
                        pass
                    if n_rows > 0:
                        break
        if n_rows <= 0 and isinstance(metrics, dict):
            for key in ("n_train_rows", "n_rows"):
                val = metrics.get(key)
                if val is not None:
                    try:
                        n_rows = int(val)
                    except (ValueError, TypeError):
                        pass
                    if n_rows > 0:
                        break

        # ── HPO scaling tiers (aligned with ML Engineer MANDATORY guidance) ──
        # These must stay in sync with the prescriptive constants in
        # ml_engineer.py's _build_hpo_scaling_guidance() so the blueprint
        # never contradicts the prompt the ML Engineer receives.
        if n_rows > 300_000:
            n_trials = 50
            timeout = 3600
            cv_folds = 3
        elif n_rows > 100_000:
            n_trials = 50
            timeout = 1800
            cv_folds = 3
        elif n_rows > 10_000:
            n_trials = 100
            timeout = 600
            cv_folds = 5
        else:
            n_trials = 100
            timeout = 600
            cv_folds = 5
        # ── Computational viability cap ──
        # Ensure n_trials × cv_folds × per_fold_time fits within the HPO
        # timeout with margin.  This prevents generating blueprints that
        # are physically impossible to execute.
        if n_rows > 0:
            # Use worst-case param_space bounds for estimation
            _max_iters = 0
            _max_depth = 0
            if framework == "catboost":
                _max_iters, _max_depth = 2000, 10
            elif framework == "xgboost":
                _max_iters, _max_depth = 2000, 10
            elif framework == "lightgbm":
                _max_iters, _max_depth = 2000, 12
            per_fold_sec = self._estimate_per_fold_seconds(
                framework, n_rows,
                max_iterations=_max_iters, max_depth=_max_depth,
            )
            cost_per_trial = per_fold_sec * cv_folds
            # Leave 20% margin for overhead (data loading, pruning, final eval)
            usable_budget = timeout * 0.80
            max_viable_trials = max(5, int(usable_budget / cost_per_trial))
            if n_trials > max_viable_trials:
                n_trials = max_viable_trials

        base: Dict[str, Any] = {"n_trials": n_trials, "timeout_seconds": timeout, "cv_folds": cv_folds}
        if framework == "catboost":
            base["param_space"] = {
                "learning_rate": [0.01, 0.1],
                "depth": [4, 10],
                "l2_leaf_reg": [1.0, 10.0],
                "min_data_in_leaf": [5, 100],
                "iterations": [500, 2000],
            }
        elif framework == "lightgbm":
            base["param_space"] = {
                "learning_rate": [0.01, 0.1],
                "num_leaves": [15, 127],
                "min_child_samples": [5, 100],
                "reg_alpha": [0.0, 10.0],
                "reg_lambda": [0.0, 10.0],
                "max_depth": [-1, 12],
            }
        elif framework == "xgboost":
            base["param_space"] = {
                "learning_rate": [0.01, 0.1],
                "max_depth": [3, 10],
                "min_child_weight": [1, 10],
                "reg_alpha": [0.0, 10.0],
                "reg_lambda": [0.0, 10.0],
                "subsample": [0.6, 1.0],
            }
        elif framework in {"pytorch", "tensorflow", "keras"}:
            base["param_space"] = {
                "learning_rate": [1e-5, 1e-2],
                "batch_size": [32, 256],
                "dropout": [0.1, 0.5],
                "weight_decay": [1e-6, 1e-3],
            }
        else:
            base["param_space"] = {
                "learning_rate": [0.01, 0.1],
                "max_depth": [3, 10],
            }
        return base

    def _summarize_dataset_profile(self, profile: Dict[str, Any]) -> str:
        if not profile:
            return "No dataset profile available."
        lines = []
        for key in ("n_rows", "n_train_rows", "n_test_rows", "n_columns", "n_features"):
            val = profile.get(key)
            if val is not None:
                lines.append(f"{key}: {val:,}" if isinstance(val, (int, float)) else f"{key}: {val}")
        # Compute size category for LLM guidance
        n_train = None
        for k in ("n_train_rows", "n_rows", "n_samples"):
            v = profile.get(k)
            if v is not None:
                try:
                    n_train = int(v)
                except (TypeError, ValueError):
                    pass
                if n_train and n_train > 0:
                    break
        if n_train and n_train > 0:
            if n_train > 500_000:
                lines.append(f"DATASET_SIZE: LARGE ({n_train:,} rows) — compute budget is the primary constraint; scale parameters accordingly")
            elif n_train > 200_000:
                lines.append(f"DATASET_SIZE: MEDIUM-LARGE ({n_train:,} rows) — balance thoroughness with compute budget")
            elif n_train > 50_000:
                lines.append(f"DATASET_SIZE: MEDIUM ({n_train:,} rows) — moderate compute pressure")
            else:
                lines.append(f"DATASET_SIZE: SMALL ({n_train:,} rows) — compute budget is unlikely to be a constraint")
        cat_cols = profile.get("categorical_columns") or profile.get("categorical_features")
        if isinstance(cat_cols, list):
            lines.append(f"categorical_features ({len(cat_cols)}): {cat_cols[:10]}")
        num_cols = profile.get("numeric_columns") or profile.get("numeric_features")
        if isinstance(num_cols, list):
            lines.append(f"numeric_features ({len(num_cols)}): {num_cols[:10]}")
        missing_pct = profile.get("missing_pct") or profile.get("missing_percentage")
        if missing_pct is not None:
            lines.append(f"missing_percentage: {missing_pct}")
        class_balance = profile.get("class_balance") or profile.get("target_distribution")
        if class_balance is not None:
            lines.append(f"class_balance: {class_balance}")
        return "\n".join(lines) if lines else "Minimal profile available."

    @staticmethod
    def _normalize_mode(raw: Any) -> str:
        val = str(raw or "hybrid").strip().lower()
        if val in {"deterministic", "llm", "hybrid"}:
            return val
        return "hybrid"
