"""Automated hyperparameter tuning using Optuna.

Replaces the manual nested-loop grid search with efficient Bayesian
optimization via Optuna's TPE sampler. Supports time-series-aware
cross-validation to prevent data leakage.
"""

import logging

import optuna
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)

# Suppress Optuna's verbose trial logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_SEED = 12345


def _gbr_objective(trial, X, y, cv):
    """Optuna objective for GradientBoostingRegressor."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", None]
        ),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "random_state": RANDOM_SEED,
    }

    model = GradientBoostingRegressor(**params)
    scores = cross_val_score(
        model, X, y, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1
    )
    return -scores.mean()


def _rf_objective(trial, X, y, cv):
    """Optuna objective for RandomForestRegressor."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2", None]
        ),
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    }

    model = RandomForestRegressor(**params)
    scores = cross_val_score(
        model, X, y, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1
    )
    return -scores.mean()


def _xgb_objective(trial, X, y, cv):
    """Optuna objective for XGBRegressor."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": RANDOM_SEED,
        "verbosity": 0,
        "n_jobs": -1,
    }

    model = XGBRegressor(**params)
    scores = cross_val_score(
        model, X, y, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1
    )
    return -scores.mean()


_OBJECTIVES = {
    "GradientBoosting": _gbr_objective,
    "RandomForest": _rf_objective,
    "XGBoost": _xgb_objective,
}

_MODEL_CLASSES = {
    "GradientBoosting": GradientBoostingRegressor,
    "RandomForest": RandomForestRegressor,
    "XGBoost": XGBRegressor,
}


def tune_model(
    model_name,
    X_train,
    y_train,
    n_trials=100,
    n_cv_splits=5,
    use_timeseries_cv=True,
):
    """Run Optuna hyperparameter optimization for a given model type.

    Args:
        model_name: One of 'GradientBoosting', 'RandomForest', 'XGBoost'.
        X_train: Training features (numpy array or DataFrame).
        y_train: Training labels.
        n_trials: Number of Optuna trials to run.
        n_cv_splits: Number of cross-validation folds.
        use_timeseries_cv: If True, use TimeSeriesSplit (prevents data leakage
            when data is ordered by season). If False, use standard KFold.

    Returns:
        dict: {
            'best_params': dict of optimal hyperparameters,
            'best_score': best mean absolute error achieved,
            'best_model': fitted model with best params,
            'study': the Optuna study object for analysis,
        }
    """
    if model_name not in _OBJECTIVES:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Choose from: {list(_OBJECTIVES.keys())}"
        )

    if use_timeseries_cv:
        cv = TimeSeriesSplit(n_splits=n_cv_splits)
    else:
        cv = n_cv_splits

    objective_fn = _OBJECTIVES[model_name]
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        study_name=f"{model_name}_tuning",
    )

    logger.info(
        "Starting Optuna tuning for %s with %d trials", model_name, n_trials
    )
    study.optimize(
        lambda trial: objective_fn(trial, X_train, y_train, cv),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best_params = study.best_params
    best_params["random_state"] = RANDOM_SEED
    logger.info("Best params for %s: %s (MAE=%.4f)", model_name, best_params, study.best_value)

    # Fit the best model on full training data
    model_cls = _MODEL_CLASSES[model_name]
    if model_name == "XGBoost":
        best_params["verbosity"] = 0
        best_params["n_jobs"] = -1
    elif model_name == "RandomForest":
        best_params["n_jobs"] = -1

    best_model = model_cls(**best_params)
    best_model.fit(X_train, y_train)

    return {
        "best_params": best_params,
        "best_score": study.best_value,
        "best_model": best_model,
        "study": study,
    }


def tune_all_models(X_train, y_train, n_trials=100):
    """Tune all supported model types and return results.

    Args:
        X_train: Training features.
        y_train: Training labels.
        n_trials: Number of Optuna trials per model.

    Returns:
        dict: model_name -> tune_model() result dict.
    """
    results = {}
    for model_name in _OBJECTIVES:
        logger.info("Tuning %s...", model_name)
        results[model_name] = tune_model(
            model_name, X_train, y_train, n_trials=n_trials
        )

    # Log comparison
    for name, result in results.items():
        logger.info("%s: MAE=%.4f", name, result["best_score"])

    best_name = min(results, key=lambda k: results[k]["best_score"])
    logger.info("Best overall model: %s", best_name)

    return results
