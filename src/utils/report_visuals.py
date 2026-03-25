"""
Report visuals generator.

Reads run artifacts (cv_metrics, feature_importance, metric_loop_state,
cleaning_manifest, submission.csv) and generates publication-quality PNG
charts for the executive PDF report.

Only generates charts for which data actually exists — adapts to the
problem type (regression, classification, cleaning-only, etc.).
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# ── Style constants ──────────────────────────────────────────────────────────

_PALETTE = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
_FIGSIZE_WIDE = (9, 4.5)
_FIGSIZE_SQUARE = (6, 5)
_DPI = 150
_TITLE_SIZE = 12
_LABEL_SIZE = 10


def _setup_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#fafafa",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": _LABEL_SIZE,
    })


def _save(fig, path: str):
    fig.savefig(path, dpi=_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ── Individual chart generators ──────────────────────────────────────────────


def _plot_feature_importance(data: Dict[str, Any], out_dir: str) -> Optional[str]:
    """Horizontal bar chart of top 15 features."""
    features = data if isinstance(data, list) else data.get("features", [])
    if not features:
        # Try flat dict format: {"col_name": importance_value, ...}
        if isinstance(data, dict) and all(isinstance(v, (int, float)) for v in data.values()):
            features = [{"feature": k, "importance": abs(v)} for k, v in data.items()]
        else:
            return None
    if not features:
        return None

    # Normalize to list of dicts with feature + importance
    parsed = []
    for item in features:
        if isinstance(item, dict):
            name = item.get("feature") or item.get("name") or item.get("column", "?")
            imp = item.get("importance") or item.get("coefficient") or item.get("weight") or item.get("value", 0)
            try:
                parsed.append((str(name), abs(float(imp))))
            except (TypeError, ValueError):
                continue
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                parsed.append((str(item[0]), abs(float(item[1]))))
            except (TypeError, ValueError):
                continue

    if len(parsed) < 2:
        return None

    parsed.sort(key=lambda x: x[1], reverse=True)
    top = parsed[:15]
    top.reverse()  # lowest at top for horizontal bar

    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
    names = [t[0] for t in top]
    values = [t[1] for t in top]
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(top))]
    ax.barh(names, values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Importancia", fontsize=_LABEL_SIZE)
    ax.set_title("Top Features por Importancia", fontsize=_TITLE_SIZE, fontweight="bold")
    ax.tick_params(axis="y", labelsize=9)

    path = os.path.join(out_dir, "feature_importance.png")
    _save(fig, path)
    return path


def _plot_model_comparison(cv_data: Dict[str, Any], out_dir: str) -> Optional[str]:
    """Grouped bar chart comparing model vs baseline or multiple models."""
    metric_name = cv_data.get("primary_metric") or "metric"
    models = []

    # Format 1: baseline_comparison dict
    comparison = cv_data.get("baseline_comparison") or cv_data.get("model_results") or {}
    if isinstance(comparison, dict) and comparison:
        for model_name, metrics in comparison.items():
            if not isinstance(metrics, dict):
                continue
            for metric_key in ("mean", "cv_mean", "score", "value", metric_name):
                val = metrics.get(metric_key)
                if isinstance(val, (int, float)):
                    models.append((model_name, val))
                    break
            else:
                for v in metrics.values():
                    if isinstance(v, (int, float)):
                        models.append((model_name, v))
                        break

    # Format 2: final_holdout with model_metrics + naive_baseline
    if len(models) < 2:
        holdout = cv_data.get("final_holdout")
        if isinstance(holdout, dict):
            model_m = holdout.get("model_metrics", {})
            baseline_m = holdout.get("naive_temporal_baseline") or holdout.get("baseline_metrics", {})
            if isinstance(model_m, dict) and isinstance(baseline_m, dict):
                for key in (metric_name, "MAE", "RMSE", "accuracy", "f1"):
                    mv = model_m.get(key)
                    bv = baseline_m.get(key)
                    if isinstance(mv, (int, float)) and isinstance(bv, (int, float)):
                        models = [("Modelo", mv), ("Baseline", bv)]
                        break

    if len(models) >= 2:
        fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
        names = [m[0].replace("_", " ").title()[:25] for m in models]
        values = [m[1] for m in models]
        bars = ax.bar(names, values, color=_PALETTE[:len(models)], edgecolor="white", linewidth=0.5)
        is_lower_better = any(k in str(cv_data).lower() for k in ("rmse", "mae", "mse", "error", "loss"))
        best_idx = values.index(min(values)) if is_lower_better else values.index(max(values))
        bars[best_idx].set_edgecolor("#2c3e50")
        bars[best_idx].set_linewidth(2)
        ax.set_ylabel(str(metric_name).upper(), fontsize=_LABEL_SIZE)
        ax.set_title("Modelo vs Baseline", fontsize=_TITLE_SIZE, fontweight="bold")
        plt.xticks(rotation=20, ha="right", fontsize=9)

        path = os.path.join(out_dir, "model_comparison.png")
        _save(fig, path)
        return path
    return None


def _plot_cv_folds(cv_data: Dict[str, Any], out_dir: str) -> Optional[str]:
    """Bar chart showing metric per CV fold with mean line."""
    # Search for fold-level data in multiple formats
    folds = (
        cv_data.get("fold_metrics")
        or cv_data.get("folds")
        or cv_data.get("fold_scores")
    )
    # Common format: cv_data["cv"]["folds"]
    if not folds and isinstance(cv_data.get("cv"), dict):
        folds = cv_data["cv"].get("folds")
    metric_name = cv_data.get("primary_metric") or cv_data.get("metric_name") or "metric"

    if isinstance(folds, list) and len(folds) >= 2:
        values = []
        for fold in folds:
            if isinstance(fold, dict):
                # Try nested model dict: fold["model"]["MAE"]
                model_metrics = fold.get("model")
                if isinstance(model_metrics, dict) and metric_name in model_metrics:
                    val = model_metrics[metric_name]
                    if isinstance(val, (int, float)):
                        values.append(val)
                        continue
                # Try known keys
                for key in (metric_name, "score", "value", "mean"):
                    val = fold.get(key)
                    if isinstance(val, (int, float)):
                        values.append(val)
                        break
                else:
                    # First numeric value
                    for v in fold.values():
                        if isinstance(v, (int, float)):
                            values.append(v)
                            break
            elif isinstance(fold, (int, float)):
                values.append(fold)

        if len(values) >= 2:
            fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
            fold_labels = [f"Fold {i}" for i in range(len(values))]
            bars = ax.bar(fold_labels, values, color=_PALETTE[0], alpha=0.8, edgecolor="white")
            mean_val = np.mean(values)
            ax.axhline(mean_val, color=_PALETTE[1], linestyle="--", linewidth=2, label=f"Media: {mean_val:.4f}")
            ax.set_ylabel(str(metric_name).upper(), fontsize=_LABEL_SIZE)
            ax.set_title(f"Rendimiento por Fold (CV)", fontsize=_TITLE_SIZE, fontweight="bold")
            ax.legend(fontsize=9)

            path = os.path.join(out_dir, "cv_folds.png")
            _save(fig, path)
            return path
    return None


def _plot_optimization_trajectory(loop_data: Dict[str, Any], out_dir: str) -> Optional[str]:
    """Line chart showing metric evolution across optimization rounds."""
    rounds = loop_data.get("rounds") or loop_data.get("history") or loop_data.get("iterations")
    if not isinstance(rounds, list) or len(rounds) < 2:
        return None

    values = []
    labels = []
    for i, r in enumerate(rounds):
        if isinstance(r, dict):
            val = r.get("metric_value") or r.get("value") or r.get("score") or r.get("incumbent_value")
            if isinstance(val, (int, float)):
                values.append(val)
                labels.append(r.get("label") or r.get("hypothesis") or f"Ronda {i}")
        elif isinstance(r, (int, float)):
            values.append(r)
            labels.append(f"Ronda {i}")

    if len(values) < 2:
        return None

    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
    ax.plot(range(len(values)), values, "o-", color=_PALETTE[0], linewidth=2, markersize=8)
    # Highlight best point
    is_lower_better = any(k in str(loop_data).lower() for k in ("rmse", "mae", "mse", "error", "loss"))
    best_idx = values.index(min(values)) if is_lower_better else values.index(max(values))
    ax.plot(best_idx, values[best_idx], "o", color=_PALETTE[2], markersize=14, zorder=5, label=f"Mejor: {values[best_idx]:.4f}")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels([str(l)[:20] for l in labels], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Metrica", fontsize=_LABEL_SIZE)
    ax.set_title("Trayectoria de Optimizacion", fontsize=_TITLE_SIZE, fontweight="bold")
    ax.legend(fontsize=9)

    path = os.path.join(out_dir, "optimization_trajectory.png")
    _save(fig, path)
    return path


def _plot_prediction_distribution(submission_path: str, out_dir: str) -> Optional[str]:
    """Histogram of predictions from submission.csv."""
    try:
        df = pd.read_csv(submission_path)
    except Exception:
        return None

    if len(df.columns) < 2 or len(df) < 10:
        return None

    # The prediction column is typically the last one (or second one)
    pred_col = df.columns[-1]
    if df[pred_col].dtype == "object":
        # Categorical predictions — bar chart of class distribution
        counts = df[pred_col].value_counts()
        if len(counts) < 2 or len(counts) > 30:
            return None
        fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
        counts.plot.bar(ax=ax, color=_PALETTE[:len(counts)], edgecolor="white")
        ax.set_ylabel("Cantidad", fontsize=_LABEL_SIZE)
        ax.set_title(f"Distribucion de Predicciones ({pred_col})", fontsize=_TITLE_SIZE, fontweight="bold")
        plt.xticks(rotation=30, ha="right", fontsize=9)
    else:
        # Numeric predictions — histogram
        fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
        values = df[pred_col].dropna()
        ax.hist(values, bins=min(50, max(10, len(values) // 20)), color=_PALETTE[0], alpha=0.8, edgecolor="white")
        ax.axvline(values.mean(), color=_PALETTE[1], linestyle="--", linewidth=2, label=f"Media: {values.mean():.2f}")
        ax.set_xlabel(pred_col, fontsize=_LABEL_SIZE)
        ax.set_ylabel("Frecuencia", fontsize=_LABEL_SIZE)
        ax.set_title(f"Distribucion de Predicciones ({pred_col})", fontsize=_TITLE_SIZE, fontweight="bold")
        ax.legend(fontsize=9)

    path = os.path.join(out_dir, "prediction_distribution.png")
    _save(fig, path)
    return path


def _plot_data_quality(manifest: Dict[str, Any], out_dir: str) -> Optional[str]:
    """Horizontal bar chart showing null rates per column from cleaning manifest."""
    # Try to extract null info from various manifest formats
    columns_info = manifest.get("column_summary") or manifest.get("columns") or manifest.get("null_counts") or {}

    if isinstance(columns_info, dict) and columns_info:
        items = []
        for col, info in columns_info.items():
            if isinstance(info, dict):
                null_pct = info.get("null_pct") or info.get("null_fraction") or info.get("missing_pct", 0)
            elif isinstance(info, (int, float)):
                null_pct = info
            else:
                continue
            try:
                items.append((str(col), float(null_pct)))
            except (TypeError, ValueError):
                continue

        if not items:
            return None

        # Sort by null rate descending, show top 20
        items.sort(key=lambda x: x[1], reverse=True)
        items = items[:20]
        items.reverse()

        fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
        names = [t[0] for t in items]
        values = [t[1] for t in items]
        colors = ["#e74c3c" if v > 20 else "#f39c12" if v > 5 else "#2ecc71" for v in values]
        ax.barh(names, values, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xlabel("% Valores Nulos", fontsize=_LABEL_SIZE)
        ax.set_title("Calidad de Datos: Valores Faltantes por Columna", fontsize=_TITLE_SIZE, fontweight="bold")
        ax.tick_params(axis="y", labelsize=9)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter())

        path = os.path.join(out_dir, "data_quality.png")
        _save(fig, path)
        return path

    return None


def _generate_csv_preview_table(csv_path: str, max_rows: int = 5) -> Optional[str]:
    """Returns an HTML table with the first N rows of a CSV for embedding in the report."""
    try:
        df = pd.read_csv(csv_path, nrows=max_rows)
    except Exception:
        return None

    if df.empty:
        return None

    # Truncate wide columns
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str[:40]

    html = df.to_html(index=False, classes="exec-table", border=0, na_rep="—")
    return html


# ── Main entry point ─────────────────────────────────────────────────────────


def generate_report_visuals(work_dir: str) -> Dict[str, Any]:
    """
    Scan run artifacts and generate all applicable charts.

    Returns dict with:
        - plots: list of generated PNG paths
        - csv_previews: dict of {artifact_name: html_table}
        - summary: human-readable summary of what was generated
    """
    _setup_style()

    out_dir = os.path.join(work_dir, "artifacts", "plots")
    os.makedirs(out_dir, exist_ok=True)

    plots: List[str] = []
    csv_previews: Dict[str, str] = {}
    summary_parts: List[str] = []

    # Search paths for artifacts
    search_dirs = [
        os.path.join(work_dir, "artifacts", "ml"),
        os.path.join(work_dir, "artifacts", "clean"),
        os.path.join(work_dir, "artifacts", "data"),
        os.path.join(work_dir, "artifacts", "output"),
        os.path.join(work_dir, "data"),
        work_dir,
    ]

    def _find(name: str) -> Optional[str]:
        for d in search_dirs:
            p = os.path.join(d, name)
            if os.path.isfile(p):
                return p
        return None

    # 1. Feature importance
    fi_path = _find("feature_importance.json")
    if fi_path:
        fi_data = _load_json(fi_path)
        if fi_data:
            result = _plot_feature_importance(fi_data, out_dir)
            if result:
                plots.append(result)
                summary_parts.append("feature importance chart")

    # 2. Model comparison
    cv_path = _find("cv_metrics.json")
    if cv_path:
        cv_data = _load_json(cv_path)
        if cv_data:
            result = _plot_model_comparison(cv_data, out_dir)
            if result:
                plots.append(result)
                summary_parts.append("model comparison chart")

            result2 = _plot_cv_folds(cv_data, out_dir)
            if result2:
                plots.append(result2)
                summary_parts.append("CV fold performance chart")

    # 3. Optimization trajectory
    loop_path = _find("metric_loop_state.json")
    if loop_path:
        loop_data = _load_json(loop_path)
        if loop_data:
            result = _plot_optimization_trajectory(loop_data, out_dir)
            if result:
                plots.append(result)
                summary_parts.append("optimization trajectory chart")

    # 4. Prediction distribution
    sub_path = _find("submission.csv")
    if sub_path:
        result = _plot_prediction_distribution(sub_path, out_dir)
        if result:
            plots.append(result)
            summary_parts.append("prediction distribution chart")

    # 5. Data quality
    manifest_path = _find("cleaning_manifest.json")
    if manifest_path:
        manifest_data = _load_json(manifest_path)
        if manifest_data:
            result = _plot_data_quality(manifest_data, out_dir)
            if result:
                plots.append(result)
                summary_parts.append("data quality chart")

    # 6. CSV previews
    for csv_name in ("submission.csv", "dataset_cleaned.csv"):
        csv_p = _find(csv_name)
        if csv_p:
            html = _generate_csv_preview_table(csv_p)
            if html:
                csv_previews[csv_name] = html
                summary_parts.append(f"{csv_name} preview")

    summary = f"Generated {len(plots)} charts and {len(csv_previews)} CSV previews"
    if summary_parts:
        summary += f": {', '.join(summary_parts)}"

    print(f"Report visuals: {summary}")
    return {
        "plots": plots,
        "csv_previews": csv_previews,
        "summary": summary,
    }
