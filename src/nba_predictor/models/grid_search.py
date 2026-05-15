from __future__ import annotations

import sys
from typing import Dict, List

import numpy as np
import pandas as pd

from nba_predictor.models.evaluate import fit_and_evaluate


def run_grid_search(
    clf,
    param_grid: Dict[str, list],
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    df_test: pd.DataFrame,
) -> List[list]:
    """Exhaustive grid search over *param_grid* for *clf*.

    Parameters
    ----------
    clf
        Estimator instance (e.g. ``GradientBoostingRegressor``).
    param_grid : dict[str, list]
        Parameter name → list of values to try.
    train_features, train_labels, test_features, test_labels
        Pre-processed arrays.
    df_test : DataFrame
        Test set with ``season``, ``Team``, ``Champion_Share_Score`` columns
        (used for per-season NDCG).

    Returns
    -------
    list[list]
        Rows of ``[clf_name, r2, mae, mse, rmse, ndcg, param_keys, param_values]``.
    """
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    total = 1
    for vals in param_values:
        total *= len(vals)

    results: List[list] = []

    print(clf.__class__.__name__)

    def _recurse(depth: int, current_params: dict) -> None:
        if depth == len(param_names):
            results.append(
                fit_and_evaluate(
                    clf,
                    current_params,
                    train_features,
                    train_labels,
                    test_features,
                    test_labels,
                    df_test,
                )
            )
            return

        name = param_names[depth]
        for val in param_values[depth]:
            current_params[name] = val
            if depth == 0:
                sys.stdout.write(f"\r{name}: {val}...")
            _recurse(depth + 1, current_params)

    _recurse(0, {})
    print()

    return results


GRID_RESULTS_COLUMNS = [
    "clf",
    "R2",
    "MAE",
    "MSE",
    "RMSE",
    "NDCG",
    "param.keys",
    "param.values",
]


def results_to_dataframe(results: List[list]) -> pd.DataFrame:
    """Convert grid search results list to a sorted DataFrame."""
    df = pd.DataFrame(data=results, columns=GRID_RESULTS_COLUMNS, index=None)
    df.sort_values(by=["NDCG"], ascending=False, inplace=True)
    return df
