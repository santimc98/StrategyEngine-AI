from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _select_numeric_columns(df: pd.DataFrame, max_numeric_cols: int, min_rows: int) -> List[str]:
    numeric_cols: List[str] = []
    for col in df.select_dtypes(include=[np.number]).columns:
        non_null = int(df[col].notna().sum())
        if non_null >= min_rows:
            numeric_cols.append(col)
    numeric_cols = sorted(numeric_cols, key=lambda c: int(df[c].notna().sum()), reverse=True)
    return numeric_cols[:max_numeric_cols]


def _looks_like_leakage_name(column_name: str) -> bool:
    token = str(column_name or "").strip().lower()
    if not token:
        return False
    suspicious_parts = ("target", "label", "outcome", "actual", "response", "ground_truth", "groundtruth", "leak")
    return any(part in token for part in suspicious_parts)


def _canonicalize_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "1" if bool(value) else "0"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return ""
        rounded = round(float(value), 12)
        if float(rounded).is_integer():
            return str(int(rounded))
        return f"{rounded:.12g}"
    text = str(value).strip().lower()
    return " ".join(text.split())


def _normalize_text_series(series: pd.Series) -> pd.Series:
    normalized = series.map(_canonicalize_value)
    return normalized.replace("", np.nan)


def _categorical_cardinality_cap(total_rows: int) -> int:
    if total_rows <= 0:
        return 200
    return max(40, min(800, int(np.sqrt(total_rows)) * 4))


def _select_categorical_columns(
    df: pd.DataFrame,
    max_categorical_cols: int,
    min_rows: int,
    cardinality_cap: int,
) -> List[str]:
    candidates: List[Dict[str, Any]] = []
    total_rows = len(df)
    for col in df.columns:
        series = df[col]
        non_null = int(series.notna().sum())
        if non_null < min_rows:
            continue
        unique_count = int(series.nunique(dropna=True))
        is_bool = bool(pd.api.types.is_bool_dtype(series))
        is_numeric = bool(pd.api.types.is_numeric_dtype(series)) and not is_bool
        if is_numeric and unique_count > max(50, min_rows):
            continue
        suspicious_name = _looks_like_leakage_name(col)
        pairwise_eligible = unique_count <= cardinality_cap or suspicious_name
        score = (
            float(non_null / max(total_rows, 1))
            + (3.0 if suspicious_name else 0.0)
            + (1.5 if pairwise_eligible else 0.0)
            + (0.5 if unique_count <= 50 else 0.0)
        )
        candidates.append(
            {
                "column": col,
                "score": score,
                "unique_count": unique_count,
            }
        )
    candidates.sort(
        key=lambda item: (float(item.get("score", 0.0)), -int(item.get("unique_count", 0) or 0), str(item.get("column") or "")),
        reverse=True,
    )
    return [str(item.get("column")) for item in candidates[:max_categorical_cols] if str(item.get("column") or "").strip()]


def _build_risk_flags(df: pd.DataFrame, min_rows: int, frac: float) -> List[Dict[str, object]]:
    flags: List[Dict[str, object]] = []
    total_rows = len(df)
    for col in df.columns:
        series = df[col]
        non_null = int(series.notna().sum())
        if non_null < min_rows:
            continue
        unique_count = int(series.nunique(dropna=True))
        unique_frac = float(unique_count / max(non_null, 1))
        if unique_frac >= frac:
            flags.append(
                {
                    "type": "near_unique_key",
                    "columns": [col],
                    "support": non_null,
                    "support_frac": float(non_null / max(total_rows, 1)),
                    "unique_frac": unique_frac,
                    "unique_count": unique_count,
                }
            )
        if _looks_like_leakage_name(col):
            flags.append(
                {
                    "type": "suspicious_name",
                    "columns": [col],
                    "support": non_null,
                    "support_frac": float(non_null / max(total_rows, 1)),
                }
            )
    flags.sort(
        key=lambda item: (
            1 if str(item.get("type") or "") == "near_unique_key" else 0,
            float(item.get("unique_frac", 0.0) or 0.0),
            float(item.get("support_frac", 0.0) or 0.0),
            ",".join([str(col) for col in (item.get("columns") or [])]),
        ),
        reverse=True,
    )
    return flags


def _identity_or_scale(a: pd.Series, b: pd.Series, tol: float) -> Optional[Dict[str, float]]:
    aligned = pd.concat([a, b], axis=1).dropna()
    if aligned.empty:
        return None
    diff = (aligned.iloc[:, 0] - aligned.iloc[:, 1]).abs()
    if diff.max() <= tol:
        return {"type": "identity", "support": len(aligned)}
    denom = aligned.iloc[:, 1].replace(0, np.nan)
    ratio = aligned.iloc[:, 0] / denom
    ratio = ratio.dropna()
    if ratio.empty:
        return None
    median_ratio = ratio.median()
    if median_ratio == 0:
        return None
    recon = aligned.iloc[:, 1] * median_ratio
    scale_err = (recon - aligned.iloc[:, 0]).abs().max()
    if scale_err <= tol:
        return {"type": "scale", "support": len(recon), "scale": float(median_ratio)}
    return None


def _sum_diff_match(a: pd.Series, b: pd.Series, target: pd.Series, tol: float) -> Optional[Dict[str, float]]:
    aligned = pd.concat([a, b, target], axis=1).dropna()
    if aligned.empty:
        return None
    sum_err = (aligned.iloc[:, 0] + aligned.iloc[:, 1] - aligned.iloc[:, 2]).abs()
    if sum_err.max() <= tol:
        return {"type": "sum", "support": len(sum_err)}
    diff_err = (aligned.iloc[:, 0] - aligned.iloc[:, 1] - aligned.iloc[:, 2]).abs()
    if diff_err.max() <= tol:
        return {"type": "diff", "support": len(diff_err)}
    return None


def _categorical_identity_or_mapping(
    a: pd.Series,
    b: pd.Series,
    *,
    min_rows: int,
    max_cardinality: int,
) -> Optional[Dict[str, float]]:
    aligned = pd.concat([a, b], axis=1).dropna()
    if len(aligned) < min_rows:
        return None
    left = aligned.iloc[:, 0]
    right = aligned.iloc[:, 1]
    if bool((left == right).all()):
        return {"type": "categorical_identity", "support": len(aligned)}
    left_cardinality = int(left.nunique(dropna=True))
    right_cardinality = int(right.nunique(dropna=True))
    if left_cardinality <= 1 or right_cardinality <= 1:
        return None
    if max(left_cardinality, right_cardinality) > max_cardinality:
        return None
    left_mapping = aligned.groupby(aligned.columns[0], sort=False)[aligned.columns[1]].nunique(dropna=True)
    right_mapping = aligned.groupby(aligned.columns[1], sort=False)[aligned.columns[0]].nunique(dropna=True)
    if int(left_mapping.max()) == 1 and int(right_mapping.max()) == 1:
        return {
            "type": "categorical_one_to_one",
            "support": len(aligned),
            "left_cardinality": left_cardinality,
            "right_cardinality": right_cardinality,
        }
    return None


def run_unsupervised_numeric_relation_audit(
    df: pd.DataFrame,
    min_rows: int = 30,
    max_numeric_cols: int = 30,
    max_pairs: int = 1200,
    tol: float = 1e-9,
    frac: float = 0.995,
    max_categorical_cols: int = 12,
) -> Dict[str, object]:
    """
    Detect near-deterministic relations and leakage-prone columns without any
    dataset-specific assumptions.
    """
    findings: List[Dict[str, object]] = []
    total_rows = len(df)
    numeric_cols = _select_numeric_columns(df, max_numeric_cols, min_rows)
    categorical_cap = _categorical_cardinality_cap(total_rows)
    categorical_cols = _select_categorical_columns(
        df,
        max_categorical_cols=max_categorical_cols,
        min_rows=min_rows,
        cardinality_cap=categorical_cap,
    )
    risk_flags = _build_risk_flags(df, min_rows=min_rows, frac=frac)

    n = len(numeric_cols)
    pair_budget = max_pairs
    for i in range(n):
        for j in range(i + 1, n):
            if pair_budget <= 0:
                break
            c1, c2 = numeric_cols[i], numeric_cols[j]
            res = _identity_or_scale(df[c1], df[c2], tol)
            pair_budget -= 1
            if res:
                support_frac = float(res["support"] / max(total_rows, 1))
                if support_frac >= frac:
                    findings.append(
                        {
                            "type": res["type"],
                            "columns": [c1, c2],
                            "support": res["support"],
                            "support_frac": support_frac,
                            "scale": res.get("scale"),
                        }
                    )
        if pair_budget <= 0:
            break

    triple_budget = max_pairs
    for target_col in numeric_cols:
        if triple_budget <= 0:
            break
        for i in range(n):
            if numeric_cols[i] == target_col:
                continue
            if triple_budget <= 0:
                break
            for j in range(i + 1, n):
                if numeric_cols[j] == target_col:
                    continue
                c1, c2 = numeric_cols[i], numeric_cols[j]
                res = _sum_diff_match(df[c1], df[c2], df[target_col], tol)
                triple_budget -= 1
                if res:
                    support_frac = float(res["support"] / max(total_rows, 1))
                    if support_frac >= frac:
                        findings.append(
                            {
                                "type": res["type"],
                                "columns": [c1, c2, target_col],
                                "support": res["support"],
                                "support_frac": support_frac,
                            }
                        )
            if triple_budget <= 0:
                break

    normalized_categoricals = {
        col: _normalize_text_series(df[col])
        for col in categorical_cols
    }
    categorical_pair_budget = max_pairs
    m = len(categorical_cols)
    for i in range(m):
        for j in range(i + 1, m):
            if categorical_pair_budget <= 0:
                break
            c1, c2 = categorical_cols[i], categorical_cols[j]
            res = _categorical_identity_or_mapping(
                normalized_categoricals[c1],
                normalized_categoricals[c2],
                min_rows=min_rows,
                max_cardinality=categorical_cap,
            )
            categorical_pair_budget -= 1
            if res:
                support_frac = float(res["support"] / max(total_rows, 1))
                if support_frac >= frac:
                    findings.append(
                        {
                            "type": res["type"],
                            "columns": [c1, c2],
                            "support": res["support"],
                            "support_frac": support_frac,
                            "left_cardinality": res.get("left_cardinality"),
                            "right_cardinality": res.get("right_cardinality"),
                        }
                    )
        if categorical_pair_budget <= 0:
            break

    scanned_columns = numeric_cols + [col for col in categorical_cols if col not in numeric_cols]
    summary = {
        "relations": findings,
        "risk_flags": risk_flags,
        "scanned_columns": scanned_columns,
        "scanned_numeric_columns": numeric_cols,
        "scanned_categorical_columns": categorical_cols,
        "rows": total_rows,
    }
    return summary


def assert_no_deterministic_target_leakage(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    min_rows: int = 30,
    max_pairs: int = 1200,
    tol: float = 1e-9,
    frac: float = 0.995,
) -> None:
    """
    Raise ValueError if the target is near-deterministically recoverable from
    feature columns via numeric algebra or low-cardinality categorical aliases.
    """
    if target_col not in df.columns:
        return
    target_series = pd.to_numeric(df[target_col], errors="coerce")
    target_valid = target_series.dropna()
    if len(target_valid) >= min_rows:
        features: List[str] = []
        for col in feature_cols:
            if col == target_col or col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors="coerce")
            if int(series.notna().sum()) >= min_rows:
                features.append(col)

        if features:
            total_rows = len(target_series)
            pair_budget = max_pairs
            for feat in features:
                aligned = pd.concat([target_series, pd.to_numeric(df[feat], errors="coerce")], axis=1).dropna()
                if len(aligned) < min_rows:
                    continue
                res = _identity_or_scale(aligned.iloc[:, 0], aligned.iloc[:, 1], tol)
                pair_budget -= 1
                if res:
                    support_frac = float(res["support"] / max(total_rows, 1))
                    if support_frac >= frac:
                        raise ValueError(
                            f"DETERMINISTIC_TARGET_RELATION: target ~ {feat} ({res['type']}), support_frac={support_frac:.3f}"
                        )
                if pair_budget <= 0:
                    return

            triple_budget = max_pairs
            m = len(features)
            for i in range(m):
                if triple_budget <= 0:
                    break
                for j in range(i + 1, m):
                    if triple_budget <= 0:
                        break
                    f1, f2 = features[i], features[j]
                    aligned = pd.concat(
                        [
                            target_series,
                            pd.to_numeric(df[f1], errors="coerce"),
                            pd.to_numeric(df[f2], errors="coerce"),
                        ],
                        axis=1,
                    ).dropna()
                    if len(aligned) < min_rows:
                        triple_budget -= 1
                        continue
                    res = _sum_diff_match(aligned.iloc[:, 1], aligned.iloc[:, 2], aligned.iloc[:, 0], tol)
                    triple_budget -= 1
                    if res:
                        support_frac = float(res["support"] / max(total_rows, 1))
                        if support_frac >= frac:
                            relation = f"{f1} {res['type']} {f2}"
                            raise ValueError(
                                f"DETERMINISTIC_TARGET_RELATION: target ~ {relation}, support_frac={support_frac:.3f}"
                            )

    target_text = _normalize_text_series(df[target_col])
    target_cardinality = int(target_text.nunique(dropna=True))
    categorical_cap = _categorical_cardinality_cap(len(df))
    if target_cardinality < 2 or target_cardinality > categorical_cap:
        return
    pair_budget = max_pairs
    for feat in feature_cols:
        if feat == target_col or feat not in df.columns:
            continue
        feature_text = _normalize_text_series(df[feat])
        res = _categorical_identity_or_mapping(
            target_text,
            feature_text,
            min_rows=min_rows,
            max_cardinality=categorical_cap,
        )
        pair_budget -= 1
        if res:
            support_frac = float(res["support"] / max(len(df), 1))
            if support_frac >= frac:
                raise ValueError(
                    f"DETERMINISTIC_TARGET_RELATION: target ~ {feat} ({res['type']}), support_frac={support_frac:.3f}"
                )
        if pair_budget <= 0:
            return
