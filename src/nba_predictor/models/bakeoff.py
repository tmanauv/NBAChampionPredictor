from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn import svm
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from xgboost import XGBRegressor

from nba_predictor.config import RANDOM_SEED


def default_regressors() -> Dict[str, object]:
    """Return a dict of ``{name: estimator}`` for the model bake-off."""
    return {
        "SVR": svm.SVR(kernel="rbf"),
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(random_state=RANDOM_SEED),
        "XGBoost": XGBRegressor(random_state=RANDOM_SEED),
        "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_SEED),
    }


def run_bakeoff(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    regressors: Dict[str, object] | None = None,
    n_splits: int = 5,
) -> List[Dict[str, object]]:
    """Cross-validate each regressor with ``GroupKFold`` (grouped by season).

    Returns a list of dicts with ``name`` and ``mae`` keys.
    """
    if regressors is None:
        regressors = default_regressors()

    cv = GroupKFold(n_splits=n_splits)
    results: List[Dict[str, object]] = []

    for name, reg in regressors.items():
        scores = cross_val_score(
            reg,
            X_train,
            y_train,
            cv=cv,
            groups=groups,
            scoring="neg_mean_absolute_error",
        )
        mae = float(np.mean(-1 * scores))
        results.append({"name": name, "mae": mae})
        print(f"{name} mean absolute error: {mae:.6f}")

    return results
