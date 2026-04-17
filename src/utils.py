"""Shared utility functions for NBA Champion Predictor."""

import pandas as pd
import matplotlib.pyplot as plt


def plot_correlation_matrix(corr_matrix):
    """Plot a correlation matrix heatmap.

    Args:
        corr_matrix: DataFrame correlation matrix.
    """
    f = plt.figure(figsize=(19, 15))
    plt.matshow(corr_matrix, fignum=f.number)
    plt.xticks(
        range(corr_matrix.shape[1]),
        corr_matrix.columns,
        fontsize=14,
        rotation=45,
        ha="left",
    )
    plt.yticks(
        range(corr_matrix.shape[1]),
        corr_matrix.columns,
        fontsize=14,
    )
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title("Correlation Matrix", fontsize=16)
    plt.show()


def plot_feature_importance(feature_importances, feature_names, title="Top Features"):
    """Plot horizontal bar chart of feature importances.

    Args:
        feature_importances: Array of feature importance values.
        feature_names: List of feature names.
        title: Plot title.
    """
    top_features = pd.Series(feature_importances, index=feature_names).sort_values()
    top_features.plot(kind="barh", figsize=(15, 10), title=title)
    plt.show()
