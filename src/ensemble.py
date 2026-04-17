"""Ensemble and stacking model implementations for NBA Champion prediction.

Provides VotingRegressor and StackingRegressor wrappers that combine
multiple base models for improved prediction stability and accuracy.
"""

import logging

from lightgbm import LGBMRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
    VotingRegressor,
)
from sklearn.linear_model import RidgeCV
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)

RANDOM_SEED = 12345


def _default_base_estimators():
    """Create default base estimators for ensemble methods.

    Returns:
        list: List of (name, estimator) tuples.
    """
    return [
        (
            "gbr",
            GradientBoostingRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                random_state=RANDOM_SEED,
            ),
        ),
        (
            "rf",
            RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),
        ),
        (
            "xgb",
            XGBRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                random_state=RANDOM_SEED,
                verbosity=0,
                n_jobs=-1,
            ),
        ),
        (
            "lgbm",
            LGBMRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                random_state=RANDOM_SEED,
                verbose=-1,
                n_jobs=-1,
            ),
        ),
    ]


def build_voting_regressor(estimators=None):
    """Build a VotingRegressor ensemble.

    Averages predictions from multiple base models for more stable
    predictions. Each model gets equal weight by default.

    Args:
        estimators: Optional list of (name, estimator) tuples.
            Defaults to GBR + RF + XGB + LightGBM.

    Returns:
        VotingRegressor: Unfitted ensemble model.
    """
    if estimators is None:
        estimators = _default_base_estimators()

    logger.info(
        "Building VotingRegressor with %d estimators: %s",
        len(estimators),
        [name for name, _ in estimators],
    )

    return VotingRegressor(estimators=estimators, n_jobs=-1)


def build_stacking_regressor(estimators=None, final_estimator=None):
    """Build a StackingRegressor ensemble.

    Uses base model predictions as features for a meta-learner (final
    estimator). This can capture complementary strengths of different
    model types.

    Args:
        estimators: Optional list of (name, estimator) tuples for the base layer.
            Defaults to GBR + RF + XGB + LightGBM.
        final_estimator: Optional meta-learner. Defaults to RidgeCV
            (linear model with built-in cross-validation for regularization).

    Returns:
        StackingRegressor: Unfitted stacking ensemble.
    """
    if estimators is None:
        estimators = _default_base_estimators()

    if final_estimator is None:
        final_estimator = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])

    logger.info(
        "Building StackingRegressor with %d base estimators and %s meta-learner",
        len(estimators),
        final_estimator.__class__.__name__,
    )

    return StackingRegressor(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=5,
        n_jobs=-1,
    )


def build_ensemble_from_tuned(tuning_results):
    """Build ensemble models using tuned hyperparameters from Optuna.

    Takes the output of tuning.tune_all_models() and constructs
    ensemble models using each model's optimal hyperparameters.

    Args:
        tuning_results: Dict from tune_all_models(), mapping
            model_name -> {'best_params': {...}, ...}

    Returns:
        dict: {
            'voting': VotingRegressor with tuned estimators,
            'stacking': StackingRegressor with tuned estimators,
        }
    """
    estimators = []

    model_classes = {
        "GradientBoosting": GradientBoostingRegressor,
        "RandomForest": RandomForestRegressor,
        "XGBoost": XGBRegressor,
    }

    for name, cls in model_classes.items():
        if name in tuning_results:
            params = tuning_results[name]["best_params"].copy()
            model = cls(**params)
            short_name = name.lower()[:3]
            estimators.append((short_name, model))
            logger.info("Added tuned %s to ensemble", name)

    # Always add LightGBM with reasonable defaults
    estimators.append(
        (
            "lgbm",
            LGBMRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                random_state=RANDOM_SEED,
                verbose=-1,
                n_jobs=-1,
            ),
        )
    )

    return {
        "voting": build_voting_regressor(estimators=estimators),
        "stacking": build_stacking_regressor(estimators=estimators),
    }
