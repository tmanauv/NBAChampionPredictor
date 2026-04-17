"""Model training, evaluation, and prediction for NBA Champion prediction."""

import random
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    ndcg_score,
    r2_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn import svm
from xgboost import XGBRegressor

from src.features import get_feature_splits

RANDOM_SEED = 12345
random.seed(RANDOM_SEED)


def build_preprocessor(X_train):
    """Build a sklearn ColumnTransformer for feature preprocessing.

    Args:
        X_train: Training feature DataFrame.

    Returns:
        ColumnTransformer: Fitted preprocessor.
    """
    numeric_cols, categorical_cols = get_feature_splits(X_train)

    numeric_pipe = Pipeline(
        steps=[
            ("imp_median", SimpleImputer(missing_values=np.nan, strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(sparse_output=False), categorical_cols),
            ("num", numeric_pipe, numeric_cols),
        ]
    )

    return preprocessor


def prepare_train_test(all_seasons_df, train_cutoff=2020):
    """Split data into train and test sets based on season cutoff.

    Args:
        all_seasons_df: DataFrame with all season data.
        train_cutoff: Seasons <= this year go to train, > go to test.

    Returns:
        tuple: (X_train, y_train, X_test, y_test, preprocessor)
    """
    all_seasons_df.sort_values(by=["season"], ascending=False, inplace=True)

    train_df = all_seasons_df[all_seasons_df["season"] <= train_cutoff]
    test_df = all_seasons_df[all_seasons_df["season"] > train_cutoff]

    X_train = train_df.drop(["Team", "season", "Champion_Share_Score"], axis=1)
    y_train = train_df["Champion_Share_Score"]
    X_test = test_df.drop(["Team", "season", "Champion_Share_Score"], axis=1)
    y_test = test_df["Champion_Share_Score"]

    preprocessor = build_preprocessor(X_train)
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    return X_train_transformed, y_train, X_test_transformed, y_test, preprocessor


def cross_validate_models(X_train, y_train):
    """Run cross-validation on multiple regression models.

    Args:
        X_train: Transformed training features.
        y_train: Training labels.

    Returns:
        dict: Model name -> mean absolute error.
    """
    models = {
        "SVR": svm.SVR(kernel="rbf"),
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(),
        "XGBoost": XGBRegressor(),
        "GradientBoosting": GradientBoostingRegressor(),
    }

    results = {}
    for name, model in models.items():
        score = cross_val_score(
            model, X_train, y_train, cv=5, scoring="neg_mean_absolute_error"
        )
        results[name] = np.mean(-1 * score)

    return results


def fit_and_evaluate(clf, param, train_features, train_labels, df_test):
    """Fit a model with given params and evaluate on test data.

    Args:
        clf: Sklearn-compatible regressor.
        param: Dict of hyperparameters to set.
        train_features: Transformed training features.
        train_labels: Training labels.
        df_test: Test DataFrame with features + Team, season, Champion_Share_Score.

    Returns:
        list: [model_name, R2, MAE, MSE, RMSE, NDCG, param_keys, param_values]
    """
    clf.set_params(**param)
    clf.fit(train_features, train_labels)

    # Prepare test data
    test_extra = df_test[["Team", "season", "Champion_Share_Score"]].copy()
    test_features_df = df_test.drop(
        ["Team", "season", "Champion_Share_Score"], axis=1
    )
    test_features = test_features_df.to_numpy()
    test_labels = test_extra["Champion_Share_Score"].values

    test_predict = clf.predict(test_features)
    r2 = r2_score(test_labels, test_predict)
    mae = mean_absolute_error(test_labels, test_predict)
    mse = mean_squared_error(test_labels, test_predict)
    rmse = mean_squared_error(test_labels, test_predict, squared=False)

    ndcg = []
    for season in df_test["season"].unique():
        df_ndcg = df_test[df_test["season"] == season].copy()
        df_ndcg.sort_values(
            by=["Champion_Share_Score"], ascending=False, inplace=True
        )
        test_labels_ndcg = df_ndcg.pop("Champion_Share_Score")
        del df_ndcg["Team"]
        del df_ndcg["season"]

        test_features_ndcg = df_ndcg.to_numpy()
        test_predict_ndcg = clf.predict(test_features_ndcg)

        ndcg_score_ = ndcg_score(
            y_true=[test_labels_ndcg], y_score=[test_predict_ndcg], k=5
        )
        ndcg.append(ndcg_score_)

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


def grid_search_gbr(train_features, train_labels, df_test):
    """Run grid search for GradientBoostingRegressor.

    Args:
        train_features: Transformed training features.
        train_labels: Training labels.
        df_test: Test DataFrame with features + metadata.

    Returns:
        list: List of evaluation results for each parameter combination.
    """
    from tqdm.auto import tqdm

    list_grid = []
    clf = GradientBoostingRegressor(
        criterion="friedman_mse",
        min_samples_split=5,
        min_samples_leaf=5,
        max_features=3,
    )

    for max_depth in tqdm(np.arange(3, 11, 1), desc="GBR grid search"):
        sys.stdout.write(f"\rmax_depth: {max_depth}/10...")
        for n_estimators in [10, 20, 50, 100]:
            for min_samples_split in [5, 7, 9]:
                for min_samples_leaf in np.arange(3, 10, 1):
                    for max_features in np.arange(3, 15, 1):
                        param = {
                            "max_depth": max_depth,
                            "n_estimators": n_estimators,
                            "min_samples_split": min_samples_split,
                            "min_samples_leaf": min_samples_leaf,
                            "max_features": max_features,
                        }
                        list_grid.append(
                            fit_and_evaluate(
                                clf, param, train_features, train_labels, df_test
                            )
                        )

    return list_grid


def grid_search_rf_xgb(train_features, train_labels, df_test):
    """Run grid search for RandomForest and XGBoost.

    Args:
        train_features: Transformed training features.
        train_labels: Training labels.
        df_test: Test DataFrame with features + metadata.

    Returns:
        list: List of evaluation results for each parameter combination.
    """
    list_grid = []

    clf = RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-3)
    for max_depth in np.arange(3, 16, 1):
        sys.stdout.write(f"\rRF max_depth: {max_depth}/15...")
        for n_estimators in np.arange(5, 20, 1):
            param = {"max_depth": max_depth, "n_estimators": n_estimators}
            list_grid.append(
                fit_and_evaluate(
                    clf, param, train_features, train_labels, df_test
                )
            )

    clf = XGBRegressor(random_state=RANDOM_SEED, n_jobs=-3, verbosity=0)
    for max_depth in np.arange(3, 11, 1):
        sys.stdout.write(f"\rXGB max_depth: {max_depth}/10...")
        for n_estimators in [10, 20, 50, 100]:
            for learning_rate in [0.1, 0.2, 0.3]:
                for subsample in np.arange(0.4, 1.01, 0.1):
                    for colsample_bytree in np.arange(0.4, 1.01, 0.1):
                        param = {
                            "max_depth": max_depth,
                            "n_estimators": n_estimators,
                            "learning_rate": learning_rate,
                            "subsample": subsample,
                            "colsample_bytree": colsample_bytree,
                        }
                        list_grid.append(
                            fit_and_evaluate(
                                clf, param, train_features, train_labels, df_test
                            )
                        )

    return list_grid


def predict_champions(model, seasons_df, header_names):
    """Generate champion predictions for given seasons.

    Args:
        model: Fitted sklearn-compatible regressor.
        seasons_df: DataFrame with features + Team, season, Champion_Share_Score.
        header_names: List of feature column names.

    Returns:
        tuple: (predictions DataFrame, mean NDCG score)
    """
    df_n_victory_list = []
    ndcg = []

    for season_n in seasons_df["season"].unique():
        df_n = seasons_df[seasons_df["season"] == season_n].copy()
        names_n = df_n["Team"].values
        df_n.drop(["season", "Team"], axis="columns", inplace=True)
        y_true = df_n.pop("Champion_Share_Score")
        feature_n = df_n.to_numpy()

        prediction = model.predict(feature_n)
        ndcg_score_ = ndcg_score(y_true=[y_true], y_score=[prediction], k=5)
        ndcg.append(ndcg_score_)

        df_n_victory = pd.DataFrame(
            data=feature_n, index=None, columns=header_names
        )
        df_n_victory["season"] = season_n
        df_n_victory["Teams"] = names_n
        df_n_victory["Champion_Shares_in_%"] = prediction * 100
        df_n_victory.sort_values(
            by=["Champion_Shares_in_%"],
            ascending=False,
            ignore_index=True,
            inplace=True,
        )
        df_n_victory_list.append(df_n_victory)

        print(season_n)
        print(df_n_victory[["Teams", "Champion_Shares_in_%"]].head(5))
        print("=" * 77)
        print("=" * 76 + "\n")

    predictions = pd.concat(df_n_victory_list, ignore_index=True)
    mean_ndcg = np.mean(ndcg)
    print("NDCG-Score:", mean_ndcg)

    return predictions, mean_ndcg
