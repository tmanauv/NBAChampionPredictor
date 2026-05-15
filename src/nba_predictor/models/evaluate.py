from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    ndcg_score,
    r2_score,
    root_mean_squared_error,
)


def fit_and_evaluate(
    clf,
    param: dict,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    df_test: pd.DataFrame,
) -> list:
    """Fit *clf* with *param*, predict on test, and return a metrics row."""
    clf.set_params(**param)
    clf.fit(train_features, train_labels)

    test_predict = clf.predict(test_features)
    r2 = r2_score(test_labels, test_predict)
    mae = mean_absolute_error(test_labels, test_predict)
    mse = mean_squared_error(test_labels, test_predict)
    rmse = root_mean_squared_error(test_labels, test_predict)

    ndcg = _compute_per_season_ndcg(clf, df_test)

    return [
        clf.__class__.__name__,
        r2,
        mae,
        mse,
        rmse,
        np.mean(ndcg),
        param.keys(),
        param.values(),
    ]


def _compute_per_season_ndcg(
    clf,
    df_test: pd.DataFrame,
    k: int = 5,
) -> List[float]:
    """Compute NDCG@k per season on *df_test*."""
    ndcg_scores: List[float] = []
    for season in df_test["season"].unique():
        df_season = df_test[df_test["season"] == season].copy()
        df_season.sort_values(
            by=["Champion_Share_Score"], ascending=False, inplace=True
        )
        y_true = df_season.pop("Champion_Share_Score")
        df_season = df_season.drop(columns=["Team", "season"])

        features = df_season.to_numpy()
        y_pred = clf.predict(features)
        score = ndcg_score(y_true=[y_true], y_score=[y_pred], k=k)
        ndcg_scores.append(score)

    return ndcg_scores


def predict_per_season(
    model,
    holdout_df: pd.DataFrame,
    header_names: List[str],
    csv_path: str | None = None,
) -> pd.DataFrame:
    """Run predictions per season on *holdout_df*, print results, optionally save CSV.

    Parameters
    ----------
    model
        A fitted sklearn/xgboost estimator.
    holdout_df : DataFrame
        Must contain ``Team``, ``season``, ``Champion_Share_Score``, plus model features.
    header_names : list[str]
        Column names for the feature matrix.
    csv_path : str, optional
        If given, write the combined results to this path.

    Returns
    -------
    DataFrame
        Combined per-season predictions.
    """
    all_results: List[pd.DataFrame] = []
    ndcg_scores: List[float] = []

    for season_n in holdout_df["season"].unique():
        df_n = holdout_df[holdout_df["season"] == season_n].copy()
        names_n = df_n["Team"].values
        df_n.drop(["season", "Team"], axis="columns", inplace=True)
        y_true = df_n.pop("Champion_Share_Score")
        feature_n = df_n.to_numpy()

        prediction = model.predict(feature_n)
        score = ndcg_score(y_true=[y_true], y_score=[prediction], k=5)
        ndcg_scores.append(score)

        result = pd.DataFrame(data=feature_n, index=None, columns=header_names)
        result["season"] = season_n
        result["Teams"] = names_n
        result["Champion_Shares_in_%"] = prediction * 100
        result.sort_values(
            by=["Champion_Shares_in_%"],
            ascending=False,
            ignore_index=True,
            inplace=True,
        )
        all_results.append(result)

        print(season_n)
        print(result[["Teams", "Champion_Shares_in_%"]].head(5))
        print("=" * 77)
        print("=" * 76 + "\n")

    print("NDCG-Score:", np.mean(ndcg_scores))

    combined = pd.concat(all_results, ignore_index=True)
    if csv_path:
        combined.to_csv(csv_path, index=False, encoding="utf-8-sig")

    return combined
