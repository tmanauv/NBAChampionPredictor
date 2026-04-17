"""Comprehensive model evaluation, reporting, and serialization.

Provides functions to evaluate models with multiple metrics, generate
comparison reports, compute confidence intervals via bootstrapping,
and serialize/deserialize trained models.
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    ndcg_score,
    r2_score,
)

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def evaluate_model(model, X_test, y_test, season_labels=None):
    """Evaluate a model with comprehensive metrics.

    Args:
        model: Fitted sklearn-compatible regressor.
        X_test: Test features (array or DataFrame).
        y_test: True target values.
        season_labels: Optional Series of season labels for per-season NDCG.

    Returns:
        dict: {
            'r2': R² score,
            'mae': Mean Absolute Error,
            'mse': Mean Squared Error,
            'rmse': Root Mean Squared Error,
            'ndcg_mean': Mean NDCG@5 across seasons (if season_labels provided),
            'ndcg_per_season': dict of season -> NDCG@5,
        }
    """
    y_pred = model.predict(X_test)

    metrics = {
        "r2": float(r2_score(y_test, y_pred)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "mse": float(mean_squared_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
    }

    if season_labels is not None:
        ndcg_per_season = {}
        for season in season_labels.unique():
            mask = season_labels == season
            y_true_s = y_test[mask].values if hasattr(y_test, "values") else y_test[mask]
            y_pred_s = y_pred[mask]

            if len(y_true_s) > 1:
                score = ndcg_score(
                    y_true=[y_true_s], y_score=[y_pred_s], k=5
                )
                ndcg_per_season[int(season)] = float(score)

        metrics["ndcg_per_season"] = ndcg_per_season
        metrics["ndcg_mean"] = float(np.mean(list(ndcg_per_season.values()))) if ndcg_per_season else 0.0

    return metrics


def compare_models(models_dict, X_test, y_test, season_labels=None):
    """Compare multiple models and return a summary DataFrame.

    Args:
        models_dict: Dict of model_name -> fitted model.
        X_test: Test features.
        y_test: True target values.
        season_labels: Optional season labels for NDCG.

    Returns:
        DataFrame: Comparison table with one row per model.
    """
    rows = []
    for name, model in models_dict.items():
        metrics = evaluate_model(model, X_test, y_test, season_labels)
        row = {"Model": name}
        row.update({k: v for k, v in metrics.items() if k != "ndcg_per_season"})
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("mae", ascending=True).reset_index(drop=True)
    return df


def bootstrap_confidence_interval(
    model, X_test, y_test, n_bootstrap=1000, confidence=0.95, metric_fn=None
):
    """Compute confidence interval for model performance via bootstrapping.

    Args:
        model: Fitted model.
        X_test: Test features.
        y_test: True target values.
        n_bootstrap: Number of bootstrap iterations.
        confidence: Confidence level (default 0.95 for 95% CI).
        metric_fn: Scoring function(y_true, y_pred) -> float.
            Defaults to mean_absolute_error.

    Returns:
        dict: {
            'mean': mean metric across bootstrap samples,
            'ci_lower': lower bound of confidence interval,
            'ci_upper': upper bound of confidence interval,
            'std': standard deviation of bootstrap distribution,
        }
    """
    if metric_fn is None:
        metric_fn = mean_absolute_error

    rng = np.random.RandomState(42)
    n_samples = len(y_test)
    y_pred = model.predict(X_test)

    scores = []
    for _ in range(n_bootstrap):
        indices = rng.randint(0, n_samples, n_samples)
        y_true_boot = np.array(y_test)[indices] if hasattr(y_test, "values") else y_test[indices]
        y_pred_boot = y_pred[indices]
        scores.append(metric_fn(y_true_boot, y_pred_boot))

    scores = np.array(scores)
    alpha = (1 - confidence) / 2

    return {
        "mean": float(scores.mean()),
        "ci_lower": float(np.percentile(scores, 100 * alpha)),
        "ci_upper": float(np.percentile(scores, 100 * (1 - alpha))),
        "std": float(scores.std()),
    }


def save_model(model, name, metadata=None):
    """Serialize a trained model to disk.

    Args:
        model: Fitted sklearn-compatible model.
        name: Model name (used for filename).
        metadata: Optional dict of metadata (params, metrics, etc.).

    Returns:
        Path: Path to the saved model file.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, model_path)
    logger.info("Saved model to %s", model_path)

    if metadata:
        meta_path = MODELS_DIR / f"{name}_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info("Saved metadata to %s", meta_path)

    return model_path


def load_model(name):
    """Load a serialized model from disk.

    Args:
        name: Model name (matching the name used in save_model).

    Returns:
        tuple: (model, metadata_dict or None)

    Raises:
        FileNotFoundError: If the model file doesn't exist.
    """
    model_path = MODELS_DIR / f"{name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"No model found at {model_path}")

    model = joblib.load(model_path)
    logger.info("Loaded model from %s", model_path)

    meta_path = MODELS_DIR / f"{name}_metadata.json"
    metadata = None
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)

    return model, metadata


def generate_report(
    models_dict, X_test, y_test, season_labels=None, output_path=None
):
    """Generate a comprehensive evaluation report.

    Args:
        models_dict: Dict of model_name -> fitted model.
        X_test: Test features.
        y_test: True target values.
        season_labels: Optional season labels for NDCG.
        output_path: Optional path to save the report as markdown.

    Returns:
        str: Markdown-formatted report.
    """
    comparison_df = compare_models(models_dict, X_test, y_test, season_labels)

    lines = [
        "# NBA Champion Predictor — Model Evaluation Report\n",
        "## Model Comparison\n",
        comparison_df.to_markdown(index=False),
        "\n",
    ]

    # Best model details
    best_name = comparison_df.iloc[0]["Model"]
    best_model = models_dict[best_name]
    lines.append(f"## Best Model: {best_name}\n")

    ci = bootstrap_confidence_interval(best_model, X_test, y_test)
    lines.append(f"**MAE 95% CI:** [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
    lines.append(f"  (mean={ci['mean']:.4f}, std={ci['std']:.4f})\n")

    # Per-season NDCG if available
    if season_labels is not None:
        metrics = evaluate_model(best_model, X_test, y_test, season_labels)
        if "ndcg_per_season" in metrics:
            lines.append("## Per-Season NDCG@5\n")
            lines.append("| Season | NDCG@5 |")
            lines.append("|--------|--------|")
            for season, score in sorted(metrics["ndcg_per_season"].items()):
                lines.append(f"| {season} | {score:.4f} |")
            lines.append("")

    report = "\n".join(lines)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)
        logger.info("Report saved to %s", output_path)

    return report
