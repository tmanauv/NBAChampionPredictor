from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from nba_predictor.config import RANDOM_SEED
from nba_predictor.models.bakeoff import default_regressors, run_bakeoff
from nba_predictor.models.evaluate import predict_per_season
from nba_predictor.models.grid_search import results_to_dataframe, run_grid_search
from nba_predictor.models.predict import load_model, save_model


def _make_regression_data(n=60, n_features=5, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    y = np.abs(X[:, 0] * 0.5 + 0.3)  # positive target (like Champion_Share_Score)
    groups = np.array([2015 + (i % 5) for i in range(n)])
    return X, y, groups


def _make_test_df(X, y, n_seasons=3):
    """Build a DataFrame matching what evaluate.py expects."""
    n = len(y)
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df["Team"] = [f"T{i}" for i in range(n)]
    df["season"] = [2021 + (i % n_seasons) for i in range(n)]
    df["Champion_Share_Score"] = y
    return df


class TestBakeoff:
    def test_default_regressors_non_empty(self):
        regs = default_regressors()
        assert len(regs) >= 3

    def test_run_bakeoff_returns_results(self):
        X, y, groups = _make_regression_data()
        results = run_bakeoff(X, y, groups, n_splits=3)
        assert len(results) > 0
        assert all("name" in r and "mae" in r for r in results)


class TestGridSearch:
    def test_run_grid_search_returns_rows(self):
        X_train, y_train, _ = _make_regression_data(n=40)
        X_test, y_test, _ = _make_regression_data(n=20, seed=99)
        df_test = _make_test_df(X_test, y_test)

        clf = GradientBoostingRegressor(random_state=RANDOM_SEED)
        results = run_grid_search(
            clf,
            param_grid={"max_depth": [3, 4], "n_estimators": [10]},
            train_features=X_train,
            train_labels=y_train,
            test_features=X_test,
            test_labels=y_test,
            df_test=df_test,
        )
        assert len(results) == 2

    def test_results_to_dataframe_sorts_by_ndcg(self):
        rows = [
            ["GBR", 0.5, 0.1, 0.02, 0.14, 0.7, ["max_depth"], [3]],
            ["GBR", 0.6, 0.09, 0.01, 0.12, 0.8, ["max_depth"], [4]],
        ]
        df = results_to_dataframe(rows)
        assert df.iloc[0]["NDCG"] >= df.iloc[1]["NDCG"]


class TestEvaluate:
    def test_predict_per_season(self, tmp_path):
        rng = np.random.default_rng(42)
        n = 30
        features = rng.standard_normal((n, 3))
        y = np.abs(features[:, 0] * 0.5 + 0.3)

        model = GradientBoostingRegressor(random_state=RANDOM_SEED, n_estimators=10)
        model.fit(features, y)

        holdout = _make_test_df(features, y)

        csv_path = str(tmp_path / "predictions.csv")
        result = predict_per_season(model, holdout, ["f0", "f1", "f2"], csv_path=csv_path)

        assert len(result) == n
        assert "Champion_Shares_in_%" in result.columns
        assert Path(csv_path).exists()


class TestPredict:
    def test_save_and_load_model(self, tmp_path):
        model = GradientBoostingRegressor(random_state=RANDOM_SEED, n_estimators=5)
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1.0, 2.0, 3.0])
        model.fit(X, y)

        path = tmp_path / "model.joblib"
        save_model(model, path)
        assert path.exists()

        loaded = load_model(path)
        np.testing.assert_array_almost_equal(model.predict(X), loaded.predict(X))
