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


def _coerce_llm_response_text(response: Any) -> str:
    """Extract text from Gemini or OpenAI-compatible response objects."""
    if isinstance(response, str):
        return response
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text
    candidates = getattr(response, "candidates", None)
    if isinstance(candidates, list):
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None)
            if not isinstance(parts, list):
                continue
            chunks: List[str] = []
            for part in parts:
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str) and part_text.strip():
                    chunks.append(part_text.strip())
            if chunks:
                return "\n".join(chunks)
    return str(response or "")


_ACTION_FAMILIES_STR = ", ".join(ACTION_FAMILIES)

_MAX_SCRIPT_CHARS = 12000

_LLM_PROMPT_TEMPLATE = """\
You are an expert ML model analyst. Analyze the baseline model code and metrics below.
Your goal: identify CONCRETE, ACTIONABLE improvements ordered by expected impact on the primary metric.

=== MODEL CODE ===
{script_code}

=== METRICS ===
{metrics_summary}

=== DATASET PROFILE ===
{dataset_profile_summary}

=== PRIMARY METRIC ===
{primary_metric}

=== MODELS USED ===
{models_used}

=== ANALYSIS INSTRUCTIONS ===
Analyze the code for these categories (skip categories that are not applicable):

1. HYPERPARAMETER GAPS: Are key hyperparameters at defaults or suboptimal values?
   Missing early_stopping? Missing learning rate schedule? Suboptimal regularization?
2. FEATURE ENGINEERING GAPS: Missing interaction features? No target encoding for
   high-cardinality categoricals? No binning for skewed numerics? Missing polynomial features?
3. ENSEMBLE GAPS: Single model when ensemble would help? Equal weights in ensemble?
   No stacking? Missing diversity in base models (e.g., only tree models)?
4. TRAINING PROTOCOL GAPS: No multi-seed averaging? Fixed single random state?
   No data augmentation when applicable?
5. CALIBRATION GAPS: No probability calibration? Predictions not well-calibrated?
6. LOSS/OBJECTIVE GAPS: Default loss function? Could benefit from custom objective
   or focal loss for imbalanced data?
7. VARIANCE REDUCTION GAPS: No multi-seed averaging? Only a single random state?
   Multi-seed averaging (train same pipeline with 5-10 seeds, average predictions)
   is one of the most reliable ways to gain +0.001-0.003 in any metric with zero
   risk of overfitting. Always suggest it if not already present.
8. ADVANCED ENCODING GAPS: Using simple ordinal/one-hot for categoricals?
   Target encoding (with proper K-fold leave-one-out regularization to prevent leakage)
   often extracts more signal than ordinal encoding for boosting models.
   Frequency encoding and count encoding are also useful complements.
9. SEMI-SUPERVISED GAPS: Large unlabeled test set available? Pseudo-labeling
   (using high-confidence predictions on unlabeled data as additional training data)
   is a standard technique that leverages the test set distribution.
   Only recommend if test_rows > 10% of total rows.
10. STACKING GAPS: Multiple models but only simple averaging or blending?
    True 2-level stacking (out-of-fold predictions from diverse base learners
    fed into a meta-learner like LogisticRegression/Ridge) captures model
    complementarity better than any weighted average.

LATE-STAGE OPTIMIZATION PRIORITY ORDER (when baseline is already strong):
When the baseline metric is already high (e.g., AUC > 0.90), the biggest gains
come from variance reduction and model diversity, NOT from hyperparameter micro-tuning.
Priority order for high-baseline scenarios:
  1. Multi-seed averaging (safest, most reliable gain)
  2. Stacking with diverse base learners (LightGBM + CatBoost + XGBoost + Linear)
  3. Target encoding / advanced feature encoding
  4. Pseudo-labeling (if large unlabeled set available)
  5. Probability calibration (for log-loss or probability-based metrics)
  6. HPO fine-tuning (diminishing returns at high baselines)

For each gap found, provide a concrete improvement action with ACTUAL parameter values
(not placeholders). Be specific about what code to change.

IMPORTANT:
- action_family MUST be one of: {action_families}
- priority 1 = highest impact, 5 = lowest
- concrete_params must contain real values the engineer can use directly
- expected_delta should be conservative (do not overestimate)
- For HPO: scale n_trials to dataset size. Large datasets (>500K rows) need fewer trials
  (10-15) with strict timeout. Small datasets (<10K) can afford 50+ trials.
  Always include timeout_seconds in concrete_params to prevent sandbox timeouts.
- The execution sandbox has a hard timeout of ~7200s. All optimization must complete within this budget.

Respond ONLY with valid JSON matching this schema:
{{
  "model_type": "<gradient_boosting|neural_network|linear|tree|svm|other>",
  "framework": "<catboost|lightgbm|xgboost|sklearn|pytorch|tensorflow|keras|other>",
  "baseline_assessment": "<brief 1-2 sentence assessment of the baseline model>",
  "improvement_actions": [
    {{
      "technique": "<specific_technique_name>",
      "action_family": "<one of the allowed families>",
      "concrete_params": {{}},
      "code_change_hint": "<1-2 line description of what to change>",
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
        """Produce an optimization_blueprint from baseline code + metrics."""
        context = context if isinstance(context, dict) else {}
        deterministic = self._analyze_baseline_deterministic(context)

        if self.mode == "deterministic":
            return self._finalize_blueprint(deterministic, context)

        llm_result = self._analyze_baseline_llm(context)
        if llm_result and self._validate_blueprint(llm_result):
            merged = self._merge_llm_and_deterministic(llm_result, deterministic)
            return self._finalize_blueprint(merged, context)

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

        metrics_summary = json.dumps(metrics, indent=2, default=str)[:3000]
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

        # 5. Multi-seed averaging
        seed_matches = re.findall(r"(?:random_state|seed)\s*[=:]\s*(\d+)", script)
        unique_seeds = set(seed_matches)
        if len(unique_seeds) <= 1 and script.strip():
            actions.append({
                "technique": "multi_seed_averaging",
                "action_family": "ensemble_or_stacking",
                "concrete_params": {
                    "seeds": [42, 123, 456, 789, 2024],
                    "aggregation": "mean",
                },
                "code_change_hint": "Train with multiple random seeds and average predictions for stability",
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

        # 8. Stacking (if only simple averaging)
        has_stacking = any(
            kw in code_lower
            for kw in ["stackingclassifier", "stackingregressor", "stacking", "meta_model", "meta_learner"]
        )
        if model_count >= 2 and not has_stacking:
            actions.append({
                "technique": "stacking_ensemble",
                "action_family": "ensemble_or_stacking",
                "concrete_params": {
                    "meta_learner": "LogisticRegression" if "classif" in code_lower or "auc" in primary_metric else "Ridge",
                    "cv": 5,
                    "passthrough": False,
                },
                "code_change_hint": "Replace simple averaging with stacking using a meta-learner on out-of-fold predictions",
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
                        # Reduce seeds to fit budget; minimum 3 seeds for meaningful averaging
                        max_seeds = max(3, int(budget_limit / (_per_fold_sec * _cv_folds_default)))
                        all_seeds = params.get("seeds", [42, 123, 456, 789, 2024])
                        adapted["concrete_params"]["seeds"] = all_seeds[:max_seeds]
                        adapted["code_change_hint"] = (
                            f"Multi-seed averaging with {max_seeds} seeds (budget-adapted). "
                            "Train the full pipeline with each seed and average final predictions."
                        )
                        adapted_reason = f"seeds {len(all_seeds)}->{max_seeds}"

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

    def _merge_llm_and_deterministic(
        self, llm: Dict[str, Any], det: Dict[str, Any]
    ) -> Dict[str, Any]:
        llm_actions = llm.get("improvement_actions") or []
        det_actions = det.get("improvement_actions") or []

        llm_families = {str(a.get("action_family")) for a in llm_actions if isinstance(a, dict)}
        llm_techniques = {str(a.get("technique")).lower() for a in llm_actions if isinstance(a, dict)}

        merged = list(llm_actions)
        for da in det_actions:
            if not isinstance(da, dict):
                continue
            da_family = str(da.get("action_family"))
            da_technique = str(da.get("technique")).lower()
            if da_family not in llm_families or da_technique not in llm_techniques:
                merged.append(da)

        merged.sort(key=lambda x: int(x.get("priority", 5)) if isinstance(x.get("priority"), int) else 5)

        result = dict(llm)
        result["improvement_actions"] = merged[:6]
        result["total_expected_delta"] = round(
            sum(a.get("expected_delta", 0) for a in result["improvement_actions"] if isinstance(a.get("expected_delta"), (int, float))),
            4,
        )
        return result

    def _finalize_blueprint(self, blueprint: Dict[str, Any], context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        blueprint["blueprint_version"] = "1.0"
        blueprint["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        actions = blueprint.get("improvement_actions")
        if isinstance(actions, list):
            # ── Final viability filter (applies to LLM + deterministic actions) ──
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
            if _n_rows > 0:
                fw = str(blueprint.get("framework") or "catboost").lower()
                # Extract worst-case iterations/depth from param_space
                _bp_iters = 0
                _bp_depth = 0
                _bp_ps = blueprint.get("hpo_params", {})
                if isinstance(_bp_ps, dict):
                    _bp_ps_space = _bp_ps.get("param_space", {})
                    if isinstance(_bp_ps_space, dict):
                        for _ik in ("iterations", "n_estimators"):
                            _iv = _bp_ps_space.get(_ik)
                            if isinstance(_iv, list) and _iv:
                                try:
                                    _bp_iters = max(_bp_iters, int(max(_iv)))
                                except (TypeError, ValueError):
                                    pass
                        for _dk in ("depth", "max_depth"):
                            _dv = _bp_ps_space.get(_dk)
                            if isinstance(_dv, list) and _dv:
                                try:
                                    _bp_depth = max(_bp_depth, int(max(_dv)))
                                except (TypeError, ValueError):
                                    pass
                _pfs = self._estimate_per_fold_seconds(
                    fw, _n_rows,
                    max_iterations=_bp_iters, max_depth=_bp_depth,
                )
                _budget = 7200.0
                _cv = 3 if _n_rows > 100_000 else 5
                viable = []
                for act in actions:
                    if not isinstance(act, dict):
                        continue
                    family = str(act.get("action_family", ""))
                    tech = str(act.get("technique", ""))
                    params = act.get("concrete_params") if isinstance(act.get("concrete_params"), dict) else {}
                    est = 0.0
                    if family == "ensemble_or_stacking":
                        if "seed" in tech:
                            n_seeds = len(params.get("seeds", [1, 2, 3]))
                            est = _pfs * _cv * n_seeds
                        elif "stacking" in tech:
                            est = _pfs * _cv * 4  # conservative
                        else:
                            # weighted_ensemble, diverse_ensemble, etc.
                            n_m = len(params.get("models", [])) or 3
                            est = _pfs * _cv * n_m
                    elif "hpo" in tech.lower() or "hyperparameter" in tech.lower():
                        hpo_t = params.get("n_trials", 50)
                        hpo_cv = params.get("cv_folds", _cv)
                        est = _pfs * hpo_cv * hpo_t
                    else:
                        est = _pfs * _cv  # single retrain

                    if est <= _budget * 0.85:
                        viable.append(act)
                    else:
                        print(
                            f"BLUEPRINT_VIABILITY_FINAL: Dropped '{tech}' — est={est:.0f}s "
                            f"> budget={_budget * 0.85:.0f}s (per_fold={_pfs:.1f}s, rows={_n_rows})"
                        )
                actions = viable
            blueprint["improvement_actions"] = actions[:6]
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
        for key in ("n_rows", "n_columns", "n_features"):
            val = profile.get(key)
            if val is not None:
                lines.append(f"{key}: {val}")
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
