import ast
import re
from typing import Dict, Any, List

from src.utils.contract_accessors import get_decision_columns
from src.utils.problem_capabilities import (
    metric_higher_is_better,
    resolve_problem_capabilities_from_contract,
)


def validate_decision_variable_isolation(
    code: str,
    execution_contract: Dict[str, Any]
) -> Dict[str, Any]:
    """
    V4.1: Validates appropriate use of decision variables based on problem type.

    CONTEXT-AWARE VALIDATION (reads from execution_contract):
    - For PRICE OPTIMIZATION (maximize revenue = price × P(success|price)):
      Decision variable MUST be in features to model price sensitivity
    - For RESOURCE ALLOCATION (assign X units to maximize outcome):
      Decision variable should NOT be in features (not causal)

    V4.1: Uses get_decision_columns() accessor to extract decision variables from:
      - decision_columns (top-level)
      - column_roles["decision"] (V4.1 role->list format)
    No fallback to legacy feature_availability.

    Args:
        code: Generated Python code to validate
        execution_contract: V4.1 contract with objective_analysis and column_roles

    Returns:
        {
            "passed": bool,
            "error_message": str,  # Empty if passed
            "violated_variables": List[str]  # Decision vars found in features (if violation)
        }
    """
    # 1. Identify problem type from contract
    obj_analysis = execution_contract.get("objective_analysis", {})
    capabilities = resolve_problem_capabilities_from_contract(execution_contract)
    problem_type = str(capabilities.get("family") or obj_analysis.get("problem_type") or "").lower()
    decision_var = obj_analysis.get("decision_variable")
    success_criteria = str(obj_analysis.get("success_criteria", "")).lower()

    # If not an optimization problem, skip this check
    if problem_type != "optimization":
        return {"passed": True, "error_message": "", "violated_variables": []}

    # 2. Determine if decision variable SHOULD be in features (context-aware)
    # For PRICE/DISCOUNT optimization where we model elasticity (price → probability),
    # decision variable MUST be in features
    is_price_optimization = any(kw in success_criteria for kw in [
        "price *", "revenue", "expected value", "elasticity", "conversion probability"
    ])

    if is_price_optimization and decision_var:
        # For price optimization, decision variable SHOULD be in model_features
        # We're modeling P(success | price, ...) so price must be a feature
        # This is NOT leakage - it's the causal mechanism we're modeling
        return {"passed": True, "error_message": "", "violated_variables": []}

    # 3. V4.1: Extract decision variables using accessor (no legacy fallback)
    # get_decision_columns handles both decision_columns and column_roles["decision"]
    decision_vars = get_decision_columns(execution_contract)

    if not decision_vars:
        return {"passed": True, "error_message": "", "violated_variables": []}

    # 3. Scan code for these variables used as features
    violated_variables = []
    try:
        tree = ast.parse(code)
        
        for node in ast.walk(tree):
            # A) Look for assignments to variable names naming features
            # e.g., features = ['F1', 'price']
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    target_id = ""
                    if isinstance(target, ast.Name):
                        target_id = target.id.lower()
                    elif isinstance(target, ast.Attribute):
                        target_id = target.attr.lower()
                    
                    # Pattern-based matching for feature variable names
                    if any(kw in target_id for kw in ["feature", "x_col", "predictor", "model_cols"]):
                        # Inspect the value being assigned
                        for child in ast.walk(node.value):
                            if isinstance(child, ast.Constant) and isinstance(child.value, str):
                                if child.value in decision_vars:
                                    violated_variables.append(child.value)
                                    
            # B) Look for direct indexing/selection like df[['a', 'price']] or df.loc[:, ['price']]
            if isinstance(node, ast.Subscript):
                # Check slice
                sl = node.slice
                # df[['a', 'b']] -> slice is List
                if isinstance(sl, ast.List):
                    for elt in sl.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            if elt.value in decision_vars:
                                violated_variables.append(elt.value)
                # df.loc[:, ['a', 'b']] -> slice is Tuple, look at index 1
                elif isinstance(sl, ast.Tuple) and len(sl.elts) >= 2:
                    col_part = sl.elts[1]
                    if isinstance(col_part, ast.List):
                        for elt in col_part.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                if elt.value in decision_vars:
                                    violated_variables.append(elt.value)
                            
    except Exception:
        # Fallback to robust regex if AST fails (e.g. partial code or syntax error not caught yet)
        for var in decision_vars:
            # Look for the variable name quoted inside a list-like structure in assignment
            # This is broad but avoids missing leakage in less standard code styles.
            # e.g., features = ["price"] or features = ["F1", "price"]
            pattern = rf"['\"]{re.escape(var)}['\"]"
            if re.search(pattern, code):
                # We double check if it's likely a feature list
                if re.search(rf"(feature|x_col|predictor|model_cols).*=.*{re.escape(var)}", code, re.IGNORECASE):
                    violated_variables.append(var)

    violated_variables = sorted(list(set(violated_variables)))
    
    if violated_variables:
        return {
            "passed": False,
            "error_message": f"CAUSAL_VIOLATION: Decision variables {violated_variables} found in feature lists. "
                             f"In optimization problems, these variables cannot be used to train the predictive model "
                             f"as they are unknown at the time of prediction.",
            "violated_variables": violated_variables
        }
        
    return {"passed": True, "error_message": "", "violated_variables": []}


def validate_model_metrics_consistency(
    metrics: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validates consistency between best_model_name and best_model_auc.
    
    UNIVERSAL CHECK for any ML task (classification/regression).
    Ensures the metrics reported match the model actually selected.
    
    Args:
        metrics: The metrics.json dict (usually from data/metrics.json)
        
    Returns:
        {
            "passed": bool,
            "error_message": str,
            "details": {...}
        }
    """
    perf = metrics.get("model_performance", {})
    if not perf:
        return {"passed": True, "error_message": "", "details": {}}

    best_name = perf.get("best_model_name")
    if best_name is None:
        return {"passed": True, "error_message": "", "details": {}}

    metric_name, best_value, baseline_value = _extract_best_vs_baseline_metric(perf)
    baseline_keywords = [
        "logisticregression",
        "linearregression",
        "dummyclassifier",
        "dummyregressor",
        "dummy",
        "naive",
        "nullmodel",
        "baseline",
        "kaplanmeier",
    ]
    is_baseline = any(kw in str(best_name).lower() for kw in baseline_keywords)

    details = {
        "best_model_name": best_name,
        "metric_name": metric_name,
        "best_model_metric": best_value,
        "baseline_metric": baseline_value,
        "is_baseline_selected": is_baseline,
    }

    if best_value is not None and baseline_value is not None:
        tolerance = 0.01
        higher_is_better = metric_higher_is_better(metric_name)
        if is_baseline:
            if abs(best_value - baseline_value) > tolerance:
                return {
                    "passed": False,
                    "error_message": f"Inconsistency: Selected best model is a baseline ({best_name}) but its performance ({best_value}) differs from baseline metric ({baseline_value}) for {metric_name}.",
                    "details": details,
                }
        else:
            worse_than_baseline = (
                best_value < (baseline_value - tolerance)
                if higher_is_better
                else best_value > (baseline_value + tolerance)
            )
            if worse_than_baseline:
                return {
                    "passed": False,
                    "error_message": f"Inconsistency: Selected best model ({best_name}) has performance ({best_value}) significantly worse than baseline metric ({baseline_value}) for {metric_name}.",
                    "details": details,
                }

    return {"passed": True, "error_message": "", "details": details}


def _normalize_metric_token(value: Any) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "", str(value or "").lower())


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _extract_best_vs_baseline_metric(perf: Dict[str, Any]) -> tuple[str, float | None, float | None]:
    primary_metric_name = str(perf.get("primary_metric") or "").strip()
    primary_metric_value = _coerce_float(perf.get("primary_metric_value"))
    if primary_metric_name and primary_metric_value is not None:
        token = str(perf.get("primary_metric_key") or primary_metric_name).strip()
        normalized = _normalize_metric_token(token)
        baseline_candidates = [
            perf.get(f"baseline_{token}"),
            perf.get(f"baseline_{normalized}"),
            perf.get("baseline_primary_metric_value"),
            perf.get(f"dummy_{token}"),
            perf.get(f"dummy_{normalized}"),
            perf.get(f"naive_{token}"),
            perf.get(f"naive_{normalized}"),
        ]
        for candidate in baseline_candidates:
            baseline_value = _coerce_float(candidate)
            if baseline_value is not None:
                return token, primary_metric_value, baseline_value

    for key, value in perf.items():
        if not str(key).startswith("best_model_"):
            continue
        if str(key) == "best_model_name":
            continue
        best_value = _coerce_float(value)
        if best_value is None:
            continue
        suffix = str(key)[len("best_model_") :]
        baseline_candidates = [
            perf.get(f"baseline_{suffix}"),
            perf.get(f"dummy_{suffix}"),
            perf.get(f"naive_{suffix}"),
            perf.get(f"baseline_model_{suffix}"),
        ]
        for candidate in baseline_candidates:
            baseline_value = _coerce_float(candidate)
            if baseline_value is not None:
                return suffix, best_value, baseline_value

    legacy_pairs = (
        ("auc", perf.get("best_model_auc"), perf.get("baseline_auc")),
        ("r2", perf.get("best_model_r2"), perf.get("baseline_r2")),
    )
    for metric_name, best_value, baseline_value in legacy_pairs:
        best_num = _coerce_float(best_value)
        baseline_num = _coerce_float(baseline_value)
        if best_num is not None and baseline_num is not None:
            return metric_name, best_num, baseline_num

    return "unknown_metric", None, None


def validate_metrics_ci_consistency(metrics: Dict[str, Any]) -> List[str]:
    """
    Validate that metric mean lies within ci_lower/ci_upper when present.
    Returns issue strings like: metrics_schema_inconsistent:<metric_name>
    """
    issues: List[str] = []
    if not isinstance(metrics, dict):
        return issues

    def _is_number(value: Any) -> bool:
        try:
            float(value)
            return True
        except Exception:
            return False

    def _scan(obj: Dict[str, Any], prefix: str) -> None:
        if not isinstance(obj, dict):
            return
        if all(key in obj for key in ("mean", "ci_lower", "ci_upper")):
            mean = obj.get("mean")
            lower = obj.get("ci_lower")
            upper = obj.get("ci_upper")
            if not (_is_number(mean) and _is_number(lower) and _is_number(upper)):
                issues.append(f"metrics_schema_inconsistent:{prefix}")
            else:
                mean_f = float(mean)
                lower_f = float(lower)
                upper_f = float(upper)
                if not (lower_f <= mean_f <= upper_f):
                    issues.append(f"metrics_schema_inconsistent:{prefix}")
        for key, value in obj.items():
            if isinstance(value, dict):
                next_prefix = f"{prefix}.{key}" if prefix else str(key)
                _scan(value, next_prefix)

    model_perf = metrics.get("model_performance") if isinstance(metrics.get("model_performance"), dict) else None
    if isinstance(model_perf, dict):
        _scan(model_perf, "model_performance")
    else:
        _scan(metrics, "")

    deduped = []
    seen = set()
    for issue in issues:
        if issue in seen:
            continue
        seen.add(issue)
        deduped.append(issue)
    return deduped
